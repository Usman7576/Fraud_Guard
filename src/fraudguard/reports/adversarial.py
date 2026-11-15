"""Adversarial validation: how different is the test period from training?

We label training rows 0 and test rows 1 and ask LightGBM to tell them apart.
Cross-validated AUC near 0.5 means the periods look alike; a high AUC means
the inputs have drifted, and the most important features show where. Drift
is expected in fraud data; the point is to measure it, not to hide it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fraudguard.config import AdversarialConfig
from fraudguard.data.split import SPLIT, Split


@dataclass
class AdversarialResult:
    auc_mean: float
    auc_std: float
    fold_aucs: list[float]
    n_train_rows: int
    n_test_rows: int
    n_features: int
    top_features: list[tuple[str, float]]
    verdict: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def verdict(auc: float) -> str:
    if auc < 0.6:
        return "similar: the test period is hard to tell apart from training"
    if auc < 0.75:
        return "moderate drift: some features shift over time"
    return "strong drift: the test period is easy to tell apart from training"


def to_matrix(df: pl.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[int]]:
    """Numeric matrix for LightGBM; string columns become categorical codes, nulls become NaN."""
    exprs, categorical = [], []
    for i, c in enumerate(columns):
        dtype = df.schema[c]
        if dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum):
            exprs.append(pl.col(c).cast(pl.String).cast(pl.Categorical).to_physical().cast(pl.Float64))
            categorical.append(i)
        else:
            exprs.append(pl.col(c).cast(pl.Float64))
    return df.select(exprs).fill_nan(None).to_numpy().astype(np.float64), categorical


def run(features: pl.DataFrame, columns: list[str], cfg: AdversarialConfig, seed: int) -> AdversarialResult:
    data = features.filter(pl.col(SPLIT).is_in([Split.TRAIN.value, Split.TEST.value]))
    if data.height > cfg.max_rows:
        data = data.sample(cfg.max_rows, seed=seed)
    y = (data[SPLIT] == Split.TEST.value).to_numpy().astype(np.int32)
    x, cat_idx = to_matrix(data, columns)

    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "deterministic": True,
        "verbosity": -1,
    }
    folds = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=seed)
    aucs: list[float] = []
    gain = np.zeros(len(columns))
    for tr, va in folds.split(x, y):
        train_set = lgb.Dataset(x[tr], y[tr], categorical_feature=cat_idx, free_raw_data=False)
        valid_set = lgb.Dataset(x[va], y[va], categorical_feature=cat_idx, reference=train_set)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=cfg.num_boost_round,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        aucs.append(float(roc_auc_score(y[va], model.predict(x[va], num_iteration=model.best_iteration))))
        gain += model.feature_importance(importance_type="gain")

    share = gain / gain.sum() if gain.sum() > 0 else gain
    order = np.argsort(share)[::-1][: cfg.top_k]
    auc_mean = float(np.mean(aucs))
    return AdversarialResult(
        auc_mean=round(auc_mean, 4),
        auc_std=round(float(np.std(aucs)), 4),
        fold_aucs=[round(a, 4) for a in aucs],
        n_train_rows=int((y == 0).sum()),
        n_test_rows=int((y == 1).sum()),
        n_features=len(columns),
        top_features=[(columns[i], round(float(share[i]), 4)) for i in order if share[i] > 0],
        verdict=verdict(auc_mean),
    )


def to_markdown(r: AdversarialResult) -> str:
    lines = [
        "# Adversarial validation: train vs test period",
        "",
        f"**AUC {r.auc_mean:.3f} ± {r.auc_std:.3f}** ({len(r.fold_aucs)}-fold CV). Verdict: {r.verdict}.",
        "",
        f"Rows: {r.n_train_rows:,} train vs {r.n_test_rows:,} test. Features: {r.n_features}.",
        "",
        "| Rank | Feature | Share of gain |",
        "|---:|---|---:|",
        *[f"| {i} | `{name}` | {share:.1%} |" for i, (name, share) in enumerate(r.top_features, 1)],
        "",
        "Features near the top drift most over time. Ones that also add little predictive value",
        "are candidates for removal; that check comes with the ablation study.",
    ]
    return "\n".join(lines) + "\n"


def write(r: AdversarialResult, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j, m = out_dir / "adversarial_validation.json", out_dir / "adversarial_validation.md"
    j.write_text(r.to_json(), encoding="utf-8")
    m.write_text(to_markdown(r), encoding="utf-8")
    return j, m
