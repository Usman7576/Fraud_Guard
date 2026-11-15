"""Train the baselines on the train split and score them on validation.

The test split is not touched here; it is reserved for the final evaluation
with confidence intervals, so model choices cannot overfit it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl

from fraudguard.config import DataConfig, ModelConfig, Settings
from fraudguard.data import pipeline as data_pipeline
from fraudguard.data import snapshot
from fraudguard.data.schema import LABEL
from fraudguard.data.split import SPLIT, Split
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.features.registry import feature_columns
from fraudguard.log import get_logger
from fraudguard.models import logistic, tracking
from fraudguard.models.metrics import Metrics, evaluate
from fraudguard.models.rules import RulesBaseline, RuleStats

log = get_logger(__name__)


@dataclass
class ModelResult:
    name: str
    validation: Metrics
    train: Metrics
    run_id: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class BaselineReport:
    data_digest: str
    features_digest: str
    results: list[ModelResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def _split(df: pl.DataFrame, which: Split) -> pl.DataFrame:
    return df.filter(pl.col(SPLIT) == which.value)


def run_baselines(data_cfg: DataConfig, model_cfg: ModelConfig, settings: Settings) -> BaselineReport:
    root = data_cfg.paths.snapshot_dir
    features = snapshot.load_snapshot(root, feature_pipeline.STAGE, "features")
    tags = {
        "data_digest": snapshot.read_manifest(root, data_pipeline.STAGE).digest(),
        "features_digest": snapshot.read_manifest(root, feature_pipeline.STAGE).digest(),
        "split_evaluated": Split.VALIDATION.value,
    }
    train, valid = _split(features, Split.TRAIN), _split(features, Split.VALIDATION)
    y_train, y_valid = train[LABEL].to_numpy(), valid[LABEL].to_numpy()
    fpr = model_cfg.evaluation.target_fpr
    results: list[ModelResult] = []

    # Rules baseline: nothing to fit, but scored the same way as every model.
    rules = RulesBaseline(model_cfg.rules)
    with tracking.run(
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment,
        "rules_baseline",
        {**tags, "model_type": "rules"},
        {f"rule_{r.name}": f"{r.feature} {r.op} {r.value} (w={r.weight})" for r in model_cfg.rules},
    ) as active:
        m_train = evaluate(y_train, rules.score(train), fpr, is_probability=False)
        m_valid = evaluate(y_valid, rules.score(valid), fpr, is_probability=False)
        tracking.log_metrics("train", m_train.as_dict())
        tracking.log_metrics("valid", m_valid.as_dict())
        stats: list[RuleStats] = rules.rule_stats(valid)
        mlflow.log_dict({"rules": [asdict(s) for s in stats]}, "rule_stats.json")
        results.append(
            ModelResult(
                "rules", m_valid, m_train, active.info.run_id, {"rule_stats": [asdict(s) for s in stats]}
            )
        )
    log.info("Rules baseline: validation PR-AUC %.4f", m_valid.pr_auc)

    # Logistic regression.
    cols = feature_columns(features)
    numeric, categorical = logistic.select_columns(features, cols, model_cfg.logistic)
    pipe = logistic.build_pipeline(numeric, categorical, model_cfg.logistic, data_cfg.seed)
    with tracking.run(
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment,
        "logistic_regression",
        {**tags, "model_type": "logistic_regression"},
        {
            "C": model_cfg.logistic.C,
            "max_iter": model_cfg.logistic.max_iter,
            "n_numeric": len(numeric),
            "n_categorical": len(categorical),
            "seed": data_cfg.seed,
        },
    ) as active:
        x_train = logistic.prepare(train, numeric, categorical)
        x_valid = logistic.prepare(valid, numeric, categorical)
        log.info("Fitting logistic regression on %s rows x %d inputs", f"{train.height:,}", x_train.width)
        pipe.fit(x_train, y_train)
        p_train = pipe.predict_proba(x_train)[:, 1]
        p_valid = pipe.predict_proba(x_valid)[:, 1]
        m_train, m_valid = evaluate(y_train, p_train, fpr), evaluate(y_valid, p_valid, fpr)
        tracking.log_metrics("train", m_train.as_dict())
        tracking.log_metrics("valid", m_valid.as_dict())
        coefs = _top_coefficients(pipe, k=15)
        mlflow.log_dict({"top_coefficients": coefs}, "top_coefficients.json")
        mlflow.sklearn.log_model(pipe, name="model", skops_trusted_types=logistic.TRUSTED_TYPES)
        results.append(
            ModelResult(
                "logistic_regression", m_valid, m_train, active.info.run_id, {"top_coefficients": coefs}
            )
        )
    log.info("Logistic regression: validation PR-AUC %.4f", m_valid.pr_auc)

    return BaselineReport(tags["data_digest"], tags["features_digest"], results)


def _top_coefficients(pipe: object, k: int) -> list[tuple[str, float]]:
    prep = pipe.named_steps["prep"]  # type: ignore[attr-defined]
    model = pipe.named_steps["model"]  # type: ignore[attr-defined]
    names = prep.get_feature_names_out()
    coef = model.coef_[0]
    order = np.argsort(np.abs(coef))[::-1][:k]
    return [(str(names[i]), round(float(coef[i]), 4)) for i in order]


def write_report(report: BaselineReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "baselines.json"
    path.write_text(report.to_json(), encoding="utf-8")
    return path
