"""Replay worker: later months played through the live API as if they were production.

For every week of the replayed months, in time order:

1. **Score.** Transactions go to `POST /score/batch` in small batches; the API
   logs predictions and opens review cases.
2. **Labels.** Before each batch, chargebacks whose time has come
   (transaction time + label delay) are sent to `POST /labels`, so the live
   graph learns about fraud exactly as late as a real merchant would.
3. **Drift.** The week is compared with the training months on the champion's
   most important features (Evidently); the HTML report and a summary are saved.
4. **Recall.** For the latest week whose chargebacks have all arrived, recall
   on labelled fraud is measured from the logged decisions.
5. **Trigger.** Retraining starts when the drift share or the recall drop
   crosses its threshold (at most once every few weeks).
6. **Challenger.** A model is retrained on recent labelled data, logged to
   MLflow and registered with the `challenger` alias.
7. **Promote.** Only if it clearly beats the champion on the newest labelled
   days (paired bootstrap); the API then serves it immediately.

Every decision is written to `retraining_events` so the console can show it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import mlflow
import numpy as np
import polars as pl
from mlflow import MlflowClient
from pymongo.database import Database

from fraudguard.config import DataConfig, ModelConfig, MonitoringConfig, Settings
from fraudguard.data import pipeline as data_pipeline
from fraudguard.data import snapshot
from fraudguard.data.load import SECONDS_PER_DAY
from fraudguard.data.schema import AMOUNT, DT, ID, LABEL
from fraudguard.data.split import SPLIT, Split
from fraudguard.decision.policy import Action, decide, realised_cost
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.log import get_logger
from fraudguard.models import champion as champion_mod
from fraudguard.models import tracking
from fraudguard.models.bundle import FraudModel
from fraudguard.monitoring import challenger as chal
from fraudguard.monitoring.drift import DriftSummary, monitored_features, weekly_drift

log = get_logger(__name__)
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY
CHALLENGER_ALIAS = "challenger"
SPLIT_ORDER = [Split.TRAIN.value, Split.VALIDATION.value, Split.TEST.value]
DROP_FOR_API = {LABEL, "ts", "day", SPLIT, "card_uid"}


class ReplayError(RuntimeError):
    pass


@dataclass
class WeekResult:
    week: int
    n_scored: int
    decisions: dict[str, int]
    labels_sent: int
    drift_share: float | None
    recall_week: int | None
    recall: float | None
    retraining: str | None  # outcome, if retraining was triggered


@dataclass
class ReplaySummary:
    weeks: list[WeekResult] = field(default_factory=list)
    baseline_recall: float = 0.0
    final_champion: str = ""
    seconds: float = 0.0


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in row.items()
        if k not in DROP_FOR_API
    }


class ReplayWorker:
    def __init__(
        self,
        client: httpx.Client,
        api_key: str,
        db: Database[dict[str, Any]],
        *,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        mon_cfg: MonitoringConfig,
        settings: Settings,
    ) -> None:
        self.http, self.db = client, db
        self.headers = {"X-API-Key": api_key}
        self.data_cfg, self.model_cfg, self.cfg, self.settings = data_cfg, model_cfg, mon_cfg, settings
        self.delay = mon_cfg.label_delay_days * SECONDS_PER_DAY
        root = data_cfg.paths.snapshot_dir
        self.base = snapshot.load_snapshot(root, data_pipeline.STAGE, "transactions")
        self.features = snapshot.load_snapshot(root, feature_pipeline.STAGE, "features")
        self.tags = {
            "data_digest": snapshot.read_manifest(root, data_pipeline.STAGE).digest(),
            "features_digest": snapshot.read_manifest(root, feature_pipeline.STAGE).digest(),
        }
        self.reports_dir = settings.reports_dir / "drift"
        # The champion was fitted on data before this time (the end of the training split).
        ref = datetime.combine(data_cfg.reference_date, datetime.min.time())
        train_end = datetime.combine(data_cfg.split.train_end, datetime.min.time())
        self.champion_data_end = int((train_end - ref).total_seconds())

    # --- API calls --------------------------------------------------------------------------------
    def _post(self, path: str, body: dict[str, Any]) -> Any:
        r = self.http.post(f"/api/v1{path}", json=body, headers=self.headers, timeout=120)
        if r.status_code >= 400:
            raise ReplayError(f"POST {path} failed with {r.status_code}: {r.text[:300]}")
        return r.json()

    def _health(self) -> dict[str, Any]:
        r = self.http.get("/api/v1/health", timeout=30)
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        return data

    # --- Pieces -----------------------------------------------------------------------------------
    def baseline_recall(self, model: FraudModel) -> float:
        """The champion's recall on validation under its own policy: the level the trigger protects."""
        valid = self.features.filter(pl.col(SPLIT) == Split.VALIDATION.value)
        y, amount = valid[LABEL].to_numpy(), valid[AMOUNT].cast(pl.Float64).to_numpy()
        actions = decide(
            model.predict_proba(valid), amount, (valid[DT] // SECONDS_PER_DAY).to_numpy(), model.policy
        )
        return realised_cost(actions, y, amount, model.policy).fraud_caught

    def week_recall(self, week: int) -> float | None:
        """Share of the week's labelled fraud that was declined or reviewed (from logged decisions)."""
        lo, hi = week * SECONDS_PER_WEEK, (week + 1) * SECONDS_PER_WEEK
        decisions = {
            d["txn_id"]: d["decision"]
            for d in self.db.predictions.find({"dt": {"$gte": lo, "$lt": hi}}, {"txn_id": 1, "decision": 1})
        }
        fraud = [
            d["txn_id"]
            for d in self.db.labels.find({"txn_id": {"$in": list(decisions)}, "label": 1}, {"txn_id": 1})
        ]
        if not fraud:
            return None
        caught: int = sum(decisions[t] != "approve" for t in fraud)
        return caught / len(fraud)

    def drift(
        self, week: int, rows: pl.DataFrame, reference: pl.DataFrame, columns: list[str]
    ) -> DriftSummary:
        summary = weekly_drift(reference, rows, columns, week, self.reports_dir / f"week_{week}.html")
        doc = {
            **summary.as_doc(),
            "week_start_dt": week * SECONDS_PER_WEEK,
            "threshold": self.cfg.drift.share_threshold,
            "updated_at": datetime.now(UTC),
        }
        self.db.drift_reports.replace_one({"week": week}, doc, upsert=True)
        return summary

    def retrain(
        self, now_dt: int, champion: FraudModel, version: str
    ) -> tuple[str, str | None, dict[str, Any] | None, str]:
        """Train, log and register a challenger; promote it through the API if it clearly wins."""
        rcfg = self.cfg.retraining
        w = chal.windows(now_dt, self.cfg.label_delay_days, rcfg)
        if w.evaluation_start < self.champion_data_end:
            # The champion was fitted on those days: comparing there would score it in-sample.
            days = (self.champion_data_end - w.evaluation_start) / SECONDS_PER_DAY
            return (
                "skipped",
                None,
                None,
                (f"not enough labels yet beyond the champion's training data ({days:.0f} more days needed)"),
            )
        try:
            model = chal.train_challenger(
                self.features, champion, w, self.model_cfg.lightgbm, self.data_cfg.seed
            )
            comparison = chal.compare(
                self.features, champion, model, w=w, policy=champion.policy, cfg=rcfg, seed=self.data_cfg.seed
            )
        except chal.NotEnoughDataError as exc:
            return "skipped", None, None, str(exc)

        with tracking.run(
            self.settings.mlflow_tracking_uri,
            self.settings.mlflow_experiment,
            "challenger",
            {**self.tags, "model_type": "lightgbm", "role": "challenger", "champion_version": version},
            {
                "train_start_dt": w.train_start,
                "calibration_start_dt": w.calibration_start,
                "evaluation_start_dt": w.evaluation_start,
                "labelled_end_dt": w.labelled_end,
                "num_boost_round": champion.rounds,
            },
        ) as active:
            mlflow.log_metrics(
                {
                    "eval_pr_auc_champion": comparison.pr_auc_champion.estimate,
                    "eval_pr_auc_challenger": comparison.pr_auc_challenger.estimate,
                    "eval_pr_auc_gain": comparison.pr_auc_gain.estimate,
                    "eval_pr_auc_gain_low": comparison.pr_auc_gain.low,
                    "eval_cost_saved_per_10k": comparison.cost_saved_per_10k.estimate,
                }
            )
            mlflow.log_dict(comparison.as_doc(), "comparison.json")
            sample = self.features.filter((pl.col(DT) >= w.evaluation_start) & (pl.col(DT) < w.labelled_end))
            info = champion_mod.log_bundle(model, sample)
            run_id = active.info.run_id
        client = MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri)
        mv = mlflow.register_model(
            info.model_uri, champion_mod.REGISTERED_NAME, tags={"run_id": run_id, "role": "challenger"}
        )
        client.set_registered_model_alias(champion_mod.REGISTERED_NAME, CHALLENGER_ALIAS, mv.version)
        new_version = str(mv.version)
        if comparison.promote:
            self._post(f"/ops/models/{new_version}/promote", {})
            return "promoted", new_version, comparison.as_doc(), comparison.reason
        return "kept_champion", new_version, comparison.as_doc(), comparison.reason

    # --- Main loop --------------------------------------------------------------------------------
    def _rows_to_replay(self) -> pl.DataFrame:
        splits = SPLIT_ORDER[SPLIT_ORDER.index(self.cfg.replay.start_split) :]
        rows = self.base.filter(pl.col(SPLIT).is_in(splits)).sort(DT, ID)
        if rows.height == 0:
            raise ReplayError(f"no transactions in {splits} to replay")
        first = int(rows[DT][0])
        latest = self._health().get("latest_transaction_dt")
        if latest is not None and latest >= first:
            raise ReplayError(
                f"the API has already seen transactions up to {latest}, but the replay starts at {first}. "
                "Start the API with FRAUDGUARD_SERVE_HISTORY_UNTIL=train on a fresh database "
                "(FRAUDGUARD_MONGO_DB)."
            )
        return rows

    def _triggers(self, drift: DriftSummary, recall: float | None, baseline: float) -> list[str]:
        cfg = self.cfg
        reasons = []
        if drift.drift_share > cfg.drift.share_threshold and drift.n_rows >= cfg.drift.min_rows:
            reasons.append(f"drift share {drift.drift_share:.0%} > {cfg.drift.share_threshold:.0%}")
        if recall is not None and recall < baseline - cfg.retraining.recall_drop:
            reasons.append(
                f"recall {recall:.1%} is more than {cfg.retraining.recall_drop:.0%} below {baseline:.1%}"
            )
        return reasons

    def _score_week(self, rows: pl.DataFrame, feed: LabelFeed) -> tuple[dict[str, int], int]:
        decisions = {a.name.lower(): 0 for a in Action}
        labels_sent = 0
        for batch in rows.iter_slices(self.cfg.replay.batch_size):
            labels_sent += feed.release(int(batch[DT][0]))
            body = {"transactions": [_clean(r) for r in batch.iter_rows(named=True)]}
            for res in self._post("/score/batch", body)["results"]:
                decisions[res["decision"]] += 1
        return decisions, labels_sent

    def run(self, max_weeks: int | None = None, release_remaining_labels: bool = True) -> ReplaySummary:
        start = time.perf_counter()
        rows = self._rows_to_replay()
        first_week = int(rows[DT][0]) // SECONDS_PER_WEEK
        uri = self.settings.mlflow_tracking_uri
        version, _ = champion_mod.champion_version(uri)
        model = champion_mod.load_champion(uri)
        baseline = self.baseline_recall(model)
        columns = monitored_features(model, self.cfg.drift.n_features)
        train = self.features.filter(pl.col(SPLIT) == Split.TRAIN.value)
        reference = train.sample(min(self.cfg.drift.reference_rows, train.height), seed=self.data_cfg.seed)
        log.info(
            "Replaying %s transactions; champion v%s, baseline recall %.1f%%",
            f"{rows.height:,}",
            version,
            baseline * 100,
        )

        feed = LabelFeed(rows, self.delay, self._post)
        summary = ReplaySummary(baseline_recall=round(baseline, 4))
        last_retrain: int | None = None
        by_week = rows.with_columns((pl.col(DT) // SECONDS_PER_WEEK).alias("__week"))
        for part in by_week.partition_by("__week", maintain_order=True)[:max_weeks]:
            week = int(part["__week"][0])
            week_rows = part.drop("__week")
            decisions, labels_sent = self._score_week(week_rows, feed)
            now_dt = int(week_rows[DT][-1])
            drift = self.drift(
                week, self.features.filter(pl.col(ID).is_in(week_rows[ID].implode())), reference, columns
            )
            # The newest week whose chargebacks have all arrived by now.
            recall_week = (now_dt - self.delay) // SECONDS_PER_WEEK - 1
            recall = self.week_recall(recall_week) if recall_week >= first_week else None

            reasons = self._triggers(drift, recall, baseline)
            cooling = last_retrain is not None and week - last_retrain < self.cfg.retraining.min_weeks_between
            outcome = None
            if reasons and not cooling:
                previous = version
                outcome, new_version, comparison, detail = self.retrain(now_dt, model, version)
                if outcome != "skipped":  # a skipped attempt trained nothing, so no cooldown
                    last_retrain = week
                if outcome == "promoted" and new_version:
                    version, model = new_version, champion_mod.load_champion(uri)
                    columns = monitored_features(model, self.cfg.drift.n_features)
                    w = chal.windows(now_dt, self.cfg.label_delay_days, self.cfg.retraining)
                    self.champion_data_end = w.evaluation_start  # the new champion saw data up to here
                self.db.retraining_events.insert_one(
                    {
                        "week": week,
                        "ts": datetime.now(UTC),
                        "trigger": "; ".join(reasons),
                        "outcome": outcome,
                        "detail": detail,
                        "drift_share": drift.drift_share,
                        "recall": recall,
                        "champion_version": previous,  # the champion when the trigger fired
                        "challenger_version": new_version,
                        "comparison": comparison,
                    }
                )
                log.info("Week %d: retraining (%s) -> %s: %s", week, "; ".join(reasons), outcome, detail)
            summary.weeks.append(
                WeekResult(
                    week,
                    week_rows.height,
                    decisions,
                    labels_sent,
                    drift.drift_share,
                    recall_week if recall is not None else None,
                    recall,
                    outcome,
                )
            )
            log.info(
                "Week %d: %d scored, %d labels, drift %.0f%%, recall %s",
                week,
                week_rows.height,
                labels_sent,
                drift.drift_share * 100,
                "-" if recall is None else f"{recall:.1%}",
            )

        if release_remaining_labels:
            # Fast-forward: the chargebacks still outstanding arrive after the replay ends.
            feed.release(2**62)
        summary.final_champion = version
        summary.seconds = round(time.perf_counter() - start, 1)
        return summary


class LabelFeed:
    """Chargebacks for the replayed transactions, released in the order they become known."""

    CHUNK = 5000

    def __init__(self, rows: pl.DataFrame, delay: int, post: Any) -> None:
        due = rows.select(ID, DT, LABEL).with_columns((pl.col(DT) + delay).alias("known_dt")).sort("known_dt")
        self.ids, self.known = due[ID].to_list(), due["known_dt"].to_list()
        self.labels = due[LABEL].to_list()
        self.next = 0
        self.post = post

    def release(self, until: int) -> int:
        """Send every label known by `until`. Returns how many the API stored."""
        end = self.next
        while end < len(self.known) and self.known[end] <= until:
            end += 1
        sent = 0
        for i in range(self.next, end, self.CHUNK):
            body = [
                {"txn_id": self.ids[k], "label": int(self.labels[k]), "known_dt": self.known[k]}
                for k in range(i, min(i + self.CHUNK, end))
            ]
            sent += int(self.post("/labels", {"labels": body})["stored"])
        self.next = end
        return sent
