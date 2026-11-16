"""Ablation: what does each feature group add?

Feature groups are added one at a time (base, + velocity, + behaviour,
+ graph). Each set is retrained on the training months with the champion's
tuned parameters and scored on the validation month, with paired bootstrap
intervals for the gain over the previous step. A last variant drops the
lifetime counters that grow with time. For every set, adversarial validation
measures how easily the test month can be told apart from training, so the
table shows the trade-off between accuracy and drift directly.

The test month is never used to choose anything here; adversarial validation
only looks at its inputs, not its labels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from fraudguard.config import DataConfig, FeatureConfig, ModelConfig, Settings
from fraudguard.data import snapshot
from fraudguard.data.load import SECONDS_PER_DAY
from fraudguard.data.schema import DT, LABEL
from fraudguard.data.split import SPLIT, Split
from fraudguard.evaluation.bootstrap import Interval, Statistics, bootstrap
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.features.registry import group_of
from fraudguard.log import get_logger
from fraudguard.models import champion, lgbm
from fraudguard.models.metrics import recall_precision_at_fpr
from fraudguard.reports import adversarial

log = get_logger(__name__)
STEPS = ("base", "velocity", "behaviour", "graph")


@dataclass
class AblationRow:
    name: str
    n_features: int
    pr_auc: Interval
    recall_at_fpr: Interval
    gain_pr_auc: Interval | None  # paired difference against the previous row
    adversarial_auc: float | None


@dataclass
class AblationReport:
    model_version: str
    rounds: int
    rows: list[AblationRow]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def variants(columns: list[str], drift_prone: list[str]) -> list[tuple[str, list[str]]]:
    out, included = [], []
    for step in STEPS:
        included += [c for c in columns if group_of(c) == step]
        out.append(("base" if step == "base" else f"+ {step}", list(included)))
    dropped = [c for c in columns if c not in set(drift_prone)]
    if len(dropped) < len(columns):
        out.append(("all - lifetime counters", dropped))
    return out


def run_ablation(
    data_cfg: DataConfig, model_cfg: ModelConfig, feat_cfg: FeatureConfig, settings: Settings
) -> AblationReport:
    features = snapshot.load_snapshot(data_cfg.paths.snapshot_dir, feature_pipeline.STAGE, "features")
    version, _ = champion.champion_version(settings.mlflow_tracking_uri)
    best = champion.load_champion(settings.mlflow_tracking_uri)
    if not best.params or not best.rounds:
        raise ValueError(
            f"champion v{version} has no stored parameters; retrain it with `fraudguard train lightgbm`"
        )
    cfg, lcfg, fpr = model_cfg.ablation, model_cfg.lightgbm, model_cfg.evaluation.target_fpr
    train = features.filter(pl.col(SPLIT) == Split.TRAIN.value)
    valid = features.filter(pl.col(SPLIT) == Split.VALIDATION.value)
    y_train, y_valid = train[LABEL].to_numpy(), valid[LABEL].to_numpy()
    day = (valid[DT] // SECONDS_PER_DAY).to_numpy()

    scores: list[np.ndarray] = []
    rows: list[AblationRow] = []
    for name, cols in variants(best.feature_names, cfg.drift_prone):
        cat_cols = lgbm.string_columns(train, cols)
        enc = lgbm.CategoryEncoder.fit(
            train, cat_cols, lcfg.categorical_min_count, lcfg.categorical_max_levels
        )
        booster = lgbm.fit(
            lgbm.to_matrix(train, cols, enc),
            y_train,
            feature_names=cols,
            categorical=[cols.index(c) for c in cat_cols],
            params=best.params,
            rounds=best.rounds,
            seed=data_cfg.seed,
        )
        s = np.asarray(booster.predict(lgbm.to_matrix(valid, cols, enc)))
        # The previous row is the full model when scoring the "all - lifetime counters" variant.
        prev = scores[-1] if scores else None
        scores.append(s)
        intervals = bootstrap(
            _stats(y_valid, s, prev, fpr),
            y_valid.size,
            groups=day,
            n_resamples=cfg.n_resamples,
            confidence=model_cfg.evaluation_final.confidence,
            seed=data_cfg.seed,
        )
        adv = (
            adversarial.run(features, cols, feat_cfg.adversarial, data_cfg.seed).auc_mean
            if cfg.adversarial
            else None
        )
        rows.append(
            AblationRow(
                name=name,
                n_features=len(cols),
                pr_auc=intervals["pr_auc"],
                recall_at_fpr=intervals["recall"],
                gain_pr_auc=intervals.get("gain"),
                adversarial_auc=adv,
            )
        )
        log.info("Ablation %-26s PR-AUC %.4f  adversarial AUC %s", name, intervals["pr_auc"].estimate, adv)
    return AblationReport(version, best.rounds, rows)


def _stats(y: np.ndarray, s: np.ndarray, prev: np.ndarray | None, fpr: float) -> Statistics:
    def stats(idx: np.ndarray) -> dict[str, float]:
        yi = y[idx]
        if not 0 < yi.sum() < yi.size:
            nan = float("nan")
            return {"pr_auc": nan, "recall": nan, **({"gain": nan} if prev is not None else {})}
        ap = float(average_precision_score(yi, s[idx]))
        out = {"pr_auc": ap, "recall": recall_precision_at_fpr(yi, s[idx], fpr)[0]}
        if prev is not None:
            out["gain"] = ap - float(average_precision_score(yi, prev[idx]))
        return out

    return stats


def write_reports(report: AblationReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j, m = out_dir / "ablation.json", out_dir / "ablation.md"
    j.write_text(report.to_json(), encoding="utf-8")
    m.write_text(to_markdown(report), encoding="utf-8")
    return j, m


def to_markdown(r: AblationReport) -> str:
    def ci(iv: Interval, fmt: str) -> str:
        return f"{iv.estimate:{fmt}} [{iv.low:{fmt}}, {iv.high:{fmt}}]"

    lines = [
        "# Feature group ablation (validation month)",
        "",
        f"Each feature set is retrained with champion v{r.model_version}'s tuned parameters "
        f"({r.rounds} rounds). Brackets are bootstrap intervals over days. *Gain* is the paired "
        "PR-AUC difference against the previous row (the last row is compared with the full model). "
        "Adversarial AUC near 0.5 means the test month looks like training; 1.0 means it is "
        "trivially separable.",
        "",
        "| Feature set | Features | PR-AUC | Gain | Recall at 1% FPR | Adversarial AUC |",
        "|---|---:|---:|---:|---:|---:|",
        *(
            f"| {row.name} | {row.n_features} | {ci(row.pr_auc, '.3f')} "
            f"| {ci(row.gain_pr_auc, '+.3f') if row.gain_pr_auc else '-'} | {ci(row.recall_at_fpr, '.1%')} "
            f"| {f'{row.adversarial_auc:.3f}' if row.adversarial_auc is not None else '-'} |"
            for row in r.rows
        ),
        "",
    ]
    return "\n".join(lines)
