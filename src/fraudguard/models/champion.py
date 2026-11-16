"""Train the champion: tuned LightGBM, isotonic calibration and a cost-optimal policy.

1. Tune LightGBM with Optuna on time-ordered folds inside the training months.
2. Refit on all training months with the best parameters.
3. Calibrate on the validation month. Calibration quality and every cost figure
   use out-of-fold probabilities (each time block calibrated by the others), so
   the calibrator is never scored on the rows it was fitted to.
4. Pick the decision policy that minimises realised cost on validation, and give
   the rules baseline the same chance with its own best thresholds.
5. Stress-test both under scaled cost assumptions.
6. Log everything to MLflow, register the bundle as `fraudguard`, and point the
   `champion` alias at it when it beats the rules baseline on cost.

The test month is not used; it is held back for the final evaluation.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.pyfunc
import numpy as np
import polars as pl
import sklearn
from mlflow import MlflowClient

from fraudguard.config import DataConfig, LightGBMConfig, ModelConfig, Settings
from fraudguard.data import pipeline as data_pipeline
from fraudguard.data import snapshot
from fraudguard.data.load import SECONDS_PER_DAY
from fraudguard.data.schema import AMOUNT, DT, LABEL
from fraudguard.data.split import SPLIT, Split
from fraudguard.decision import analysis
from fraudguard.decision.policy import decline_threshold
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.features.registry import feature_columns
from fraudguard.log import get_logger
from fraudguard.models import calibration, lgbm, tracking
from fraudguard.models.bundle import FraudModel, pyfunc_input, signature
from fraudguard.models.metrics import Metrics, evaluate
from fraudguard.models.rules import RulesBaseline

log = get_logger(__name__)

REGISTERED_NAME = "fraudguard"
CHAMPION_ALIAS = "champion"
EXAMPLE_AMOUNTS = (25.0, 100.0, 500.0, 2000.0)
# MLflow "models from code": the model is defined by running this file, not by unpickling.
PYFUNC_ENTRY = Path(__file__).with_name("pyfunc_entry.py")


@dataclass
class ChampionReport:
    data_digest: str
    features_digest: str
    run_id: str
    model_version: str | None
    is_champion: bool
    tuning: dict[str, Any]
    validation_raw: Metrics
    validation_calibrated: Metrics
    ece_raw: float
    ece_calibrated: float
    reliability_raw: list[calibration.ReliabilityBin]
    reliability_calibrated: list[calibration.ReliabilityBin]
    model_policy: dict[str, object]
    rules_policy: dict[str, object]
    approve_all: dict[str, float]
    saving_vs_rules: float
    decline_thresholds: dict[str, float]
    sensitivity: list[analysis.SensitivityRow]
    top_features: list[tuple[str, float]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def _split(df: pl.DataFrame, which: Split) -> pl.DataFrame:
    return df.filter(pl.col(SPLIT) == which.value)


def train_champion(
    data_cfg: DataConfig, model_cfg: ModelConfig, settings: Settings, n_trials: int | None = None
) -> ChampionReport:
    lcfg = (
        model_cfg.lightgbm
        if n_trials is None
        else model_cfg.lightgbm.model_copy(update={"n_trials": n_trials})
    )
    root, seed, fpr = data_cfg.paths.snapshot_dir, data_cfg.seed, model_cfg.evaluation.target_fpr
    features = snapshot.load_snapshot(root, feature_pipeline.STAGE, "features")
    tags = {
        "data_digest": snapshot.read_manifest(root, data_pipeline.STAGE).digest(),
        "features_digest": snapshot.read_manifest(root, feature_pipeline.STAGE).digest(),
        "split_evaluated": Split.VALIDATION.value,
        "model_type": "lightgbm",
    }
    train, valid = _split(features, Split.TRAIN), _split(features, Split.VALIDATION)
    y_valid = valid[LABEL].to_numpy()
    columns = lgbm.select_columns(features, feature_columns(features), lcfg)
    booster, encoder, tuning = _tune_and_fit(train, columns, lcfg, seed)

    # Calibration on the validation month.
    raw_valid = np.asarray(booster.predict(lgbm.to_matrix(valid, columns, encoder)))
    ccfg = model_cfg.calibration
    p_oof = calibration.cross_fit(raw_valid, y_valid, ccfg.n_folds)
    calibrator = calibration.Calibrator.fit(raw_valid, y_valid)
    m_raw, m_cal = evaluate(y_valid, raw_valid, fpr), evaluate(y_valid, p_oof, fpr)
    ece_raw, rel_raw = calibration.reliability(raw_valid, y_valid, ccfg.bin_edges)
    ece_cal, rel_cal = calibration.reliability(p_oof, y_valid, ccfg.bin_edges)

    # Decision policy, chosen on out-of-fold probabilities.
    dcfg = model_cfg.decision
    amount = valid[AMOUNT].cast(pl.Float64).to_numpy()
    day = (valid[DT] // SECONDS_PER_DAY).to_numpy()
    base = analysis.base_policy(dcfg)
    model_choice = analysis.best_policy(
        p_oof, y_valid, amount, day, base=base, grid=analysis.multiplier_grid(dcfg)
    )
    rules_score = RulesBaseline(model_cfg.rules).score(valid)
    rules_choice = analysis.best_rules_policy(rules_score, y_valid, amount, day, base)
    approve_all = analysis.approve_all_cost(y_valid, amount, base)
    saving = (
        round(1 - model_choice.cost.total / rules_choice.cost.total, 4) if rules_choice.cost.total else 0.0
    )
    sens = analysis.sensitivity(p_oof, rules_score, y_valid, amount, day, cfg=dcfg)
    policy = model_choice.policy
    thresholds = {f"{a:g}": round(decline_threshold(a, policy), 4) for a in EXAMPLE_AMOUNTS}

    gain = booster.feature_importance("gain")
    top = sorted(zip(columns, gain / max(gain.sum(), 1e-12), strict=True), key=lambda t: -t[1])[:20]
    top_features = [(name, round(float(share), 4)) for name, share in top]
    fraud_model = FraudModel(booster, columns, encoder, calibrator, policy)

    with tracking.run(
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment,
        "lightgbm",
        tags,
        {
            **tuning.best_params,
            "num_boost_round": tuning.best_rounds,
            "n_trials": tuning.n_trials,
            "n_features": len(columns),
            "n_categorical": len(encoder.levels),
            "seed": seed,
            **{f"policy_{k}": v for k, v in policy.as_dict().items()},
        },
    ) as active:
        tracking.log_metrics("valid_raw", m_raw.as_dict())
        tracking.log_metrics("valid_cal", m_cal.as_dict())
        mlflow.log_metrics(
            {
                "tuning_fold_pr_auc": tuning.best_value,
                "valid_ece_raw": ece_raw,
                "valid_ece_cal": ece_cal,
                "valid_cost_per_10k": model_choice.cost.per_10k,
                "valid_rules_cost_per_10k": rules_choice.cost.per_10k,
                "valid_saving_vs_rules": saving,
            }
        )
        mlflow.log_dict({"trials": tuning.trials}, "tuning_trials.json")
        mlflow.log_dict({"top_features": top_features}, "top_features.json")
        mlflow.log_dict({"sensitivity": [asdict(r) for r in sens]}, "sensitivity.json")
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = fraud_model.save(Path(tmp) / "bundle")
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=str(PYFUNC_ENTRY),
                artifacts={"bundle": str(bundle_dir)},
                signature=signature(valid, columns),
                input_example=pyfunc_input(valid.head(5), columns),
                pip_requirements=[
                    f"lightgbm=={lgb.__version__}",
                    f"numpy=={np.__version__}",
                    f"polars=={pl.__version__}",
                    f"scikit-learn=={sklearn.__version__}",
                    f"mlflow=={mlflow.__version__}",
                ],
            )
        run_id = active.info.run_id

    is_champion = model_choice.cost.total < rules_choice.cost.total
    version = _register(info.model_uri, run_id, is_champion)
    log.info(
        "LightGBM: validation PR-AUC %.4f, cost per 10k $%.0f vs rules $%.0f",
        m_raw.pr_auc,
        model_choice.cost.per_10k,
        rules_choice.cost.per_10k,
    )
    return ChampionReport(
        data_digest=tags["data_digest"],
        features_digest=tags["features_digest"],
        run_id=run_id,
        model_version=version,
        is_champion=is_champion,
        tuning={k: v for k, v in asdict(tuning).items() if k != "trials"},
        validation_raw=m_raw,
        validation_calibrated=m_cal,
        ece_raw=ece_raw,
        ece_calibrated=ece_cal,
        reliability_raw=rel_raw,
        reliability_calibrated=rel_cal,
        model_policy=model_choice.as_dict(),
        rules_policy=rules_choice.as_dict(),
        approve_all=approve_all.as_dict(),
        saving_vs_rules=saving,
        decline_thresholds=thresholds,
        sensitivity=sens,
        top_features=top_features,
    )


def _tune_and_fit(
    train: pl.DataFrame, columns: list[str], lcfg: LightGBMConfig, seed: int
) -> tuple[lgb.Booster, lgbm.CategoryEncoder, lgbm.TuningResult]:
    """Optuna search on the training months, then a refit on all of them."""
    cat_cols = lgbm.string_columns(train, columns)
    encoder = lgbm.CategoryEncoder.fit(
        train, cat_cols, lcfg.categorical_min_count, lcfg.categorical_max_levels
    )
    cat_idx = [columns.index(c) for c in cat_cols]
    x, y = lgbm.to_matrix(train, columns, encoder), train[LABEL].to_numpy()
    log.info(
        "Tuning LightGBM: up to %d trials, %s rows x %d features", lcfg.n_trials, f"{len(y):,}", len(columns)
    )
    tuning = lgbm.tune(x, y, cat_idx, lcfg, seed)
    log.info(
        "Best fold PR-AUC %.4f after %d trials (%d pruned)",
        tuning.best_value,
        tuning.n_trials,
        tuning.n_pruned,
    )
    booster = lgbm.fit(
        x,
        y,
        feature_names=columns,
        categorical=cat_idx,
        params=tuning.best_params,
        rounds=tuning.best_rounds,
        seed=seed,
    )
    return booster, encoder, tuning


def _register(model_uri: str, run_id: str, is_champion: bool) -> str | None:
    version = mlflow.register_model(model_uri, REGISTERED_NAME, tags={"run_id": run_id})
    client = MlflowClient()
    if is_champion:
        client.set_registered_model_alias(REGISTERED_NAME, CHAMPION_ALIAS, version.version)
        log.info("Registered %s v%s as %s", REGISTERED_NAME, version.version, CHAMPION_ALIAS)
    else:
        log.warning(
            "%s v%s does not beat the rules baseline on cost; champion alias unchanged",
            REGISTERED_NAME,
            version.version,
        )
    return str(version.version)


def load_champion(tracking_uri: str) -> FraudModel:
    """The bundle behind the `champion` alias (used by the scoring API)."""
    mlflow.set_tracking_uri(tracking_uri)
    pyfunc = mlflow.pyfunc.load_model(f"models:/{REGISTERED_NAME}@{CHAMPION_ALIAS}")
    model: FraudModel = pyfunc.unwrap_python_model().model
    return model


def write_reports(report: ChampionReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "champion.json"
    json_path.write_text(report.to_json(), encoding="utf-8")
    md_path = out_dir / "decision.md"
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, md_path


def to_markdown(r: ChampionReport) -> str:
    mp, rp = r.model_policy, r.rules_policy
    mc, rc = mp["cost"], rp["cost"]
    pol = mp["policy"]
    assert isinstance(mc, dict) and isinstance(rc, dict) and isinstance(pol, dict)

    def row(name: str, c: dict[str, float]) -> str:
        return (
            f"| {name} | ${c['per_10k']:,.0f} | ${c['fraud_loss']:,.0f} | ${c['false_decline_cost']:,.0f} "
            f"| ${c['review_cost']:,.0f} | {c['decline_rate']:.2%} | {c['review_rate']:.2%} "
            f"| {c['fraud_caught']:.1%} | {c['fraud_amount_caught']:.1%} |"
        )

    lines = [
        "# Decision policy (validation month)",
        "",
        f"Costs assumed: false decline ${pol['false_decline_cost']:g}, review ${pol['review_cost']:g}, "
        f"review capacity {pol['review_capacity_per_day']} cases a day. Approved fraud costs its amount.",
        "",
        f"LightGBM PR-AUC {r.validation_raw.pr_auc:.4f}, recall at {r.validation_raw.target_fpr:.0%} FPR "
        f"{r.validation_raw.recall_at_fpr:.1%}. Expected calibration error {r.ece_raw:.4f} raw, "
        f"{r.ece_calibrated:.4f} after isotonic (out-of-fold).",
        "",
        "| Policy | Cost per 10k | Fraud loss | False declines | Reviews | Declined | Reviewed "
        "| Fraud caught | Fraud $ caught |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        row(f"LightGBM (odds x{pol['risk_multiplier']:g})", mc),
        row(f"Rules (review >= {rp['review_at']}, decline >= {rp['decline_at']})", rc),
        row("Approve everything", r.approve_all),
        "",
        f"**Cost saved vs rules: {r.saving_vs_rules:.1%}.** Champion: {'yes' if r.is_champion else 'no'}.",
        "",
        "## Decline threshold by amount",
        "",
        "Declining beats approving when the calibrated fraud probability exceeds:",
        "",
        "| Amount | Decline above |",
        "|---:|---:|",
        *(f"| ${a} | {t:.1%} |" for a, t in r.decline_thresholds.items()),
        "",
        "## Sensitivity to the cost assumptions",
        "",
        "| False decline | Review | LightGBM per 10k | Rules per 10k | Saving | Odds x "
        "| Declined | Reviewed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *(
            f"| ${s.false_decline_cost:g} | ${s.review_cost:g} | ${s.model_per_10k:,.0f} "
            f"| ${s.rules_per_10k:,.0f} | {s.saving_vs_rules:.1%} | {s.model_multiplier:g} "
            f"| {s.model_decline_rate:.2%} "
            f"| {s.model_review_rate:.2%} |"
            for s in r.sensitivity
        ),
        "",
        "## Calibration (out-of-fold, log-spaced bins)",
        "",
        "| Mean predicted | Observed fraud rate | Rows |",
        "|---:|---:|---:|",
        *(f"| {b.mean_predicted:.4f} | {b.fraud_rate:.4f} | {b.count:,} |" for b in r.reliability_calibrated),
        "",
    ]
    return "\n".join(lines)
