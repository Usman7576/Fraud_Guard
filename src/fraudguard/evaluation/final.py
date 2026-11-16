"""Final evaluation on the untouched test month.

Everything used here was fixed before the test month was opened:
- the champion model, its calibrator and its policy come from the registry;
- the rules baseline's review and decline thresholds are chosen on validation.

Reported with cluster-bootstrap confidence intervals (whole days resampled),
including paired differences against the rules baseline, an error analysis of
the fraud the model let through, a global SHAP summary and example reason codes.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from fraudguard.config import DataConfig, ModelConfig, Settings
from fraudguard.data import pipeline as data_pipeline
from fraudguard.data import snapshot
from fraudguard.data.load import SECONDS_PER_DAY
from fraudguard.data.schema import AMOUNT, DT, ID, LABEL
from fraudguard.data.split import SPLIT, Split
from fraudguard.decision import analysis
from fraudguard.decision.policy import Action, CostBreakdown, Policy, decide, realised_cost
from fraudguard.evaluation.bootstrap import Interval, Statistics, bootstrap
from fraudguard.explain import reasons
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.log import get_logger
from fraudguard.models import calibration, champion, tracking
from fraudguard.models.metrics import Metrics, evaluate, recall_precision_at_fpr
from fraudguard.models.rules import RulesBaseline

log = get_logger(__name__)
N_EXAMPLES = 5


@dataclass
class SegmentRow:
    segment: str
    value: str
    n_fraud: int
    fraud_amount: float
    model_caught: float  # share of this segment's fraud declined or reviewed
    rules_caught: float
    model_missed_amount: float


@dataclass
class EvaluationReport:
    model_version: str
    model_run_id: str
    data_digest: str
    features_digest: str
    n_rows: int
    n_fraud: int
    resample_unit: str
    n_resamples: int
    confidence: float
    policy: dict[str, float | int]
    rules_thresholds: dict[str, float | None]
    model_metrics: Metrics
    rules_metrics: Metrics
    ece: float
    reliability: list[calibration.ReliabilityBin]
    costs: dict[str, CostBreakdown]
    intervals: dict[str, Interval]
    error_analysis: list[SegmentRow]
    shap_global: list[tuple[str, float]]
    examples: list[dict[str, object]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def _split(df: pl.DataFrame, which: Split) -> pl.DataFrame:
    return df.filter(pl.col(SPLIT) == which.value)


def _row_costs(actions: np.ndarray, y: np.ndarray, amount: np.ndarray, policy: Policy) -> np.ndarray:
    fraud = y.astype(bool)
    return np.select(
        [(actions == Action.APPROVE) & fraud, (actions == Action.DECLINE) & ~fraud, actions == Action.REVIEW],
        [amount, np.full(amount.shape, policy.false_decline_cost), np.full(amount.shape, policy.review_cost)],
        default=0.0,
    )


def make_statistics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    rules_score: np.ndarray,
    amount: np.ndarray,
    model_actions: np.ndarray,
    rules_actions: np.ndarray,
    policy: Policy,
    target_fpr: float,
) -> Statistics:
    """Every headline number for one set of rows, plus paired differences against the rules."""
    cost_m = _row_costs(model_actions, y, amount, policy)
    cost_r = _row_costs(rules_actions, y, amount, policy)
    fraud_amt = amount * y
    stop_m = (model_actions != Action.APPROVE) * fraud_amt
    stop_r = (rules_actions != Action.APPROVE) * fraud_amt

    def stats(idx: np.ndarray) -> dict[str, float]:
        yi = y[idx]
        n_fraud = int(yi.sum())
        nan = float("nan")
        both = 0 < n_fraud < yi.size
        ap_m = float(average_precision_score(yi, p[idx])) if both else nan
        ap_r = float(average_precision_score(yi, rules_score[idx])) if both else nan
        rec_m = recall_precision_at_fpr(yi, p[idx], target_fpr)[0] if both else nan
        rec_r = recall_precision_at_fpr(yi, rules_score[idx], target_fpr)[0] if both else nan
        per10k_m, per10k_r = cost_m[idx].mean() * 10_000, cost_r[idx].mean() * 10_000
        fraud_total = fraud_amt[idx].sum()
        amt_rec_m = stop_m[idx].sum() / fraud_total if fraud_total else nan
        amt_rec_r = stop_r[idx].sum() / fraud_total if fraud_total else nan
        return {
            "model_pr_auc": ap_m,
            "rules_pr_auc": ap_r,
            "diff_pr_auc": ap_m - ap_r,
            "model_recall_at_fpr": rec_m,
            "rules_recall_at_fpr": rec_r,
            "diff_recall_at_fpr": rec_m - rec_r,
            "model_cost_per_10k": per10k_m,
            "rules_cost_per_10k": per10k_r,
            "cost_saved_per_10k": per10k_r - per10k_m,
            "cost_saving_share": 1 - cost_m[idx].sum() / cost_r[idx].sum() if cost_r[idx].sum() else nan,
            "model_amount_recall": amt_rec_m,
            "rules_amount_recall": amt_rec_r,
            "model_brier": float(np.mean((p[idx] - yi) ** 2)),
        }

    return stats


def band_labels(edges: list[float]) -> list[str]:
    return [
        f"< ${edges[0]:g}",
        *(f"${a:g}-{b:g}" for a, b in itertools.pairwise(edges)),
        f">= ${edges[-1]:g}",
    ]


def amount_band(amount: np.ndarray, edges: list[float]) -> list[str]:
    labels = band_labels(edges)
    return [labels[i] for i in np.searchsorted(edges, amount, side="right")]


def error_analysis(
    test: pl.DataFrame,
    model_actions: np.ndarray,
    rules_actions: np.ndarray,
    bands: list[float],
    top_emails: int,
) -> list[SegmentRow]:
    """Where does the fraud the model approves come from?"""
    amount = test[AMOUNT].cast(pl.Float64).to_numpy()
    fraud = test[LABEL].to_numpy() == 1
    segments: dict[str, list[str]] = {"amount": amount_band(amount, bands)}
    for col in ("ProductCD", "DeviceType", "card_is_new"):
        if col in test.columns:
            segments[col] = [str(v) if v is not None else "missing" for v in test[col].to_list()]
    if "P_emaildomain" in test.columns:
        email = test["P_emaildomain"].cast(pl.String).fill_null("missing")
        # The domains with the most fraud are listed by name; the rest are grouped.
        keep = set(email.filter(pl.Series(fraud)).value_counts(sort=True)["P_emaildomain"].head(top_emails))
        segments["P_emaildomain"] = [v if v in keep else "other" for v in email.to_list()]

    caught_m, caught_r = model_actions != Action.APPROVE, rules_actions != Action.APPROVE
    rows = []
    for segment, values in segments.items():
        vals = np.array(values, dtype=object)
        present = set(vals[fraud].tolist())
        # Amount bands in size order; other segments alphabetically.
        ordered = [b for b in band_labels(bands) if b in present] if segment == "amount" else sorted(present)
        for value in ordered:
            mask = fraud & (vals == value)
            rows.append(
                SegmentRow(
                    segment=segment,
                    value=str(value),
                    n_fraud=int(mask.sum()),
                    fraud_amount=round(float(amount[mask].sum()), 2),
                    model_caught=round(float(caught_m[mask].mean()), 4),
                    rules_caught=round(float(caught_r[mask].mean()), 4),
                    model_missed_amount=round(float(amount[mask & ~caught_m].sum()), 2),
                )
            )
    return rows


def run_evaluation(data_cfg: DataConfig, model_cfg: ModelConfig, settings: Settings) -> EvaluationReport:
    root, fcfg = data_cfg.paths.snapshot_dir, model_cfg.evaluation_final
    features = snapshot.load_snapshot(root, feature_pipeline.STAGE, "features")
    version, run_id = champion.champion_version(settings.mlflow_tracking_uri)
    model = champion.load_champion(settings.mlflow_tracking_uri)
    valid, test = _split(features, Split.VALIDATION), _split(features, Split.TEST)
    if test.height == 0 or test[LABEL].n_unique() < 2:
        raise ValueError("the test month needs both fraud and genuine transactions")

    # Rules thresholds are fixed on validation, like the model's policy.
    rules = RulesBaseline(model_cfg.rules)
    v_amount = valid[AMOUNT].cast(pl.Float64).to_numpy()
    rules_choice = analysis.best_rules_policy(
        rules.score(valid),
        valid[LABEL].to_numpy(),
        v_amount,
        (valid[DT] // SECONDS_PER_DAY).to_numpy(),
        model.policy,
    )

    y = test[LABEL].to_numpy()
    amount = test[AMOUNT].cast(pl.Float64).to_numpy()
    day = (test[DT] // SECONDS_PER_DAY).to_numpy()
    p = model.predict_proba(test)
    rules_score = rules.score(test)
    model_actions = decide(p, amount, day, model.policy)
    rules_actions = analysis.rules_actions(
        rules_score,
        day,
        rules_choice.review_at,
        rules_choice.decline_at,
        model.policy.review_capacity_per_day,
    )

    fpr = model_cfg.evaluation.target_fpr
    stats = make_statistics(
        y,
        p,
        rules_score=rules_score,
        amount=amount,
        model_actions=model_actions,
        rules_actions=rules_actions,
        policy=model.policy,
        target_fpr=fpr,
    )
    log.info("Bootstrapping %d resamples by %s", fcfg.n_resamples, fcfg.resample_unit)
    intervals = bootstrap(
        stats,
        y.size,
        groups=day if fcfg.resample_unit == "day" else None,
        n_resamples=fcfg.n_resamples,
        confidence=fcfg.confidence,
        seed=data_cfg.seed,
    )
    ece, bins = calibration.reliability(p, y, model_cfg.calibration.bin_edges)

    # Explanations: global SHAP on a sample, reason codes for the riskiest declines.
    sample = test.sample(min(model_cfg.explain.global_sample, test.height), seed=data_cfg.seed)
    contrib, _ = model.contributions(sample)
    shap_global = reasons.global_importance(contrib, model.feature_names)
    top = np.argsort(-p)[:N_EXAMPLES]
    top_rows = test[top.tolist()]
    examples = [
        {
            "TransactionID": int(tid),
            "probability": round(float(p[i]), 4),
            "decision": Action(int(model_actions[i])).name.lower(),
            "is_fraud": bool(y[i]),
            "reasons": [r.as_dict() for r in rs],
        }
        for tid, i, rs in zip(
            top_rows[ID].to_list(),
            top,
            reasons.explain(model, top_rows, model_cfg.explain.n_reasons),
            strict=True,
        )
    ]

    report = EvaluationReport(
        model_version=version,
        model_run_id=run_id,
        data_digest=snapshot.read_manifest(root, data_pipeline.STAGE).digest(),
        features_digest=snapshot.read_manifest(root, feature_pipeline.STAGE).digest(),
        n_rows=int(y.size),
        n_fraud=int(y.sum()),
        resample_unit=fcfg.resample_unit,
        n_resamples=fcfg.n_resamples,
        confidence=fcfg.confidence,
        policy=model.policy.as_dict(),
        rules_thresholds={"review_at": rules_choice.review_at, "decline_at": rules_choice.decline_at},
        model_metrics=evaluate(y, p, fpr),
        rules_metrics=evaluate(y, rules_score, fpr, is_probability=False),
        ece=ece,
        reliability=bins,
        costs={
            "model": realised_cost(model_actions, y, amount, model.policy),
            "rules": realised_cost(rules_actions, y, amount, model.policy),
            "approve_all": analysis.approve_all_cost(y, amount, model.policy),
        },
        intervals=intervals,
        error_analysis=error_analysis(
            test, model_actions, rules_actions, fcfg.amount_bands, fcfg.top_email_domains
        ),
        shap_global=shap_global,
        examples=examples,
    )
    _log(report, settings)
    return report


def _log(report: EvaluationReport, settings: Settings) -> None:
    with tracking.run(
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment,
        "evaluation_test",
        {
            "data_digest": report.data_digest,
            "features_digest": report.features_digest,
            "split_evaluated": Split.TEST.value,
            "model_version": report.model_version,
            "model_run_id": report.model_run_id,
        },
        {
            "n_resamples": report.n_resamples,
            "resample_unit": report.resample_unit,
            "confidence": report.confidence,
        },
    ):
        flat = {}
        for name, iv in report.intervals.items():
            flat[f"test_{name}"] = iv.estimate
            flat[f"test_{name}_low"] = iv.low
            flat[f"test_{name}_high"] = iv.high
        mlflow.log_metrics({k: v for k, v in flat.items() if np.isfinite(v)})
        mlflow.log_metric("test_ece", report.ece)
        mlflow.log_dict(json.loads(report.to_json()), "evaluation.json")


def write_reports(report: EvaluationReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j, m = out_dir / "evaluation.json", out_dir / "evaluation.md"
    j.write_text(report.to_json(), encoding="utf-8")
    m.write_text(to_markdown(report), encoding="utf-8")
    return j, m


def _ci(iv: Interval, fmt: str) -> str:
    return f"{iv.estimate:{fmt}} [{iv.low:{fmt}}, {iv.high:{fmt}}]"


def to_markdown(r: EvaluationReport) -> str:
    iv = r.intervals
    pct = f"{r.confidence:.0%}"
    unit = "days" if r.resample_unit == "day" else "rows"
    lines = [
        "# Final evaluation (test month)",
        "",
        f"Champion `fraudguard` v{r.model_version} on {r.n_rows:,} test transactions ({r.n_fraud:,} fraud). "
        f"Brackets are {pct} bootstrap intervals ({r.n_resamples:,} resamples of whole {unit}); "
        "differences are paired.",
        "",
        "| Metric | LightGBM | Rules | Difference |",
        "|---|---:|---:|---:|",
        f"| PR-AUC | {_ci(iv['model_pr_auc'], '.3f')} | {_ci(iv['rules_pr_auc'], '.3f')} "
        f"| {_ci(iv['diff_pr_auc'], '+.3f')} |",
        f"| Recall at {r.model_metrics.target_fpr:.0%} FPR | {_ci(iv['model_recall_at_fpr'], '.1%')} "
        f"| {_ci(iv['rules_recall_at_fpr'], '.1%')} | {_ci(iv['diff_recall_at_fpr'], '+.1%')} |",
        f"| Fraud $ caught | {_ci(iv['model_amount_recall'], '.1%')} "
        f"| {_ci(iv['rules_amount_recall'], '.1%')} | |",
        f"| Cost per 10k | {_ci(iv['model_cost_per_10k'], ',.0f')} | {_ci(iv['rules_cost_per_10k'], ',.0f')} "
        f"| saves {_ci(iv['cost_saved_per_10k'], ',.0f')} |",
        "",
        f"**Cost saving vs rules: {_ci(iv['cost_saving_share'], '.1%')}.** "
        f"Brier {_ci(iv['model_brier'], '.4f')}, expected calibration error {r.ece:.4f}.",
        "",
        "## Error analysis: fraud by segment",
        "",
        "| Segment | Value | Fraud cases | Fraud $ | LightGBM caught | Rules caught | $ missed by LightGBM |",
        "|---|---|---:|---:|---:|---:|---:|",
        *(
            f"| {s.segment} | {s.value} | {s.n_fraud:,} | ${s.fraud_amount:,.0f} | {s.model_caught:.1%} "
            f"| {s.rules_caught:.1%} | ${s.model_missed_amount:,.0f} |"
            for s in r.error_analysis
        ),
        "",
        "## What drives the score (mean |SHAP|, log-odds)",
        "",
        "| Feature | Mean abs contribution |",
        "|---|---:|",
        *(f"| `{name}` | {v:.4f} |" for name, v in r.shap_global[:15]),
        "",
        "## Reason codes for the highest-risk transactions",
        "",
        *(
            f"- **{e['TransactionID']}** p={e['probability']:.2f}, {e['decision']}"
            f"{' (fraud)' if e['is_fraud'] else ' (genuine)'}: "
            + "; ".join(str(x["text"]) for x in e["reasons"])  # type: ignore[attr-defined]
            for e in r.examples
        ),
        "",
    ]
    return "\n".join(lines)
