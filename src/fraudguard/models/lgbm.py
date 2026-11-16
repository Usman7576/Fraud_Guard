"""LightGBM: the main model, tuned with Optuna on time-ordered folds.

Tuning never looks at the validation month. The training months are cut into
`n_folds + 1` contiguous chunks; fold k fits on chunks 0..k and scores chunk
k+1 (an expanding window), so every fold predicts the future from the past,
just like production. Trials are ranked by mean PR-AUC across folds.

String columns become LightGBM categoricals through `CategoryEncoder`, whose
vocabulary is learned on training rows only and saved with the model so the
scoring API encodes exactly the same way.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.metrics import average_precision_score

from fraudguard.config import LightGBMConfig
from fraudguard.log import get_logger

log = get_logger(__name__)

FIXED_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "bagging_freq": 1,
}


@dataclass
class CategoryEncoder:
    """String column -> integer codes. Unseen or rare levels share the "other" code; null stays NaN."""

    levels: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pl.DataFrame, columns: list[str], min_count: int, max_levels: int) -> CategoryEncoder:
        levels = {}
        for col in columns:
            counts = Counter(v for v in df[col].cast(pl.String).to_list() if v is not None)
            common = [v for v, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if n >= min_count]
            levels[col] = common[:max_levels]
        return cls(levels)

    def encode(self, s: pl.Series) -> np.ndarray:
        vocab = self.levels[s.name]
        mapping = {v: float(i) for i, v in enumerate(vocab)}
        other = float(len(vocab))
        return np.array(
            [np.nan if v is None else mapping.get(v, other) for v in s.cast(pl.String).to_list()],
            dtype=np.float32,
        )


def select_columns(df: pl.DataFrame, feature_cols: list[str], cfg: LightGBMConfig) -> list[str]:
    return [c for c in feature_cols if c not in cfg.exclude]


def string_columns(df: pl.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if df.schema[c] in (pl.String, pl.Categorical)]


def to_matrix(df: pl.DataFrame, columns: list[str], encoder: CategoryEncoder) -> np.ndarray:
    """Float32 feature matrix in `columns` order; categoricals encoded, nulls as NaN."""
    out = np.empty((df.height, len(columns)), dtype=np.float32)
    for j, col in enumerate(columns):
        if col in encoder.levels:
            out[:, j] = encoder.encode(df[col])
        else:
            out[:, j] = df[col].cast(pl.Float32).fill_nan(None).to_numpy(allow_copy=True)
    return out


def time_folds(n: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over rows already in time order."""
    chunks = np.array_split(np.arange(n), n_folds + 1)
    return [(np.concatenate(chunks[: k + 1]), chunks[k + 1]) for k in range(n_folds)]


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
    }


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_value: float
    best_rounds: int
    n_trials: int
    n_pruned: int
    seconds: float
    trials: list[dict[str, Any]]


def tune(
    x: np.ndarray, y: np.ndarray, categorical: list[int], cfg: LightGBMConfig, seed: int
) -> TuningResult:
    """Optuna TPE search over time-ordered folds; returns the best params and boosting rounds."""
    # One binned dataset; folds are subsets that share its bin boundaries (much faster than rebinning).
    full = lgb.Dataset(
        x,
        y,
        categorical_feature=categorical,
        free_raw_data=False,
        params={"verbosity": -1, "feature_pre_filter": False, "seed": seed},
    ).construct()
    folds = time_folds(len(y), cfg.n_folds)
    for k, (tr, va) in enumerate(folds):
        if y[tr].sum() == 0 or y[va].sum() == 0:
            raise ValueError(f"tuning fold {k} has no fraud in its train or validation chunk")
    fold_lists = [(tr.tolist(), va.tolist()) for tr, va in folds]

    def objective(trial: optuna.Trial) -> float:
        params = {**FIXED_PARAMS, **suggest_params(trial), "seed": seed}
        scores, rounds = [], []
        for k, ((tr_list, va_list), (_, va)) in enumerate(zip(fold_lists, folds, strict=True)):
            booster = lgb.train(
                params,
                full.subset(tr_list),
                num_boost_round=cfg.max_rounds,
                valid_sets=[full.subset(va_list)],
                callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
            )
            pred = booster.predict(x[va], num_iteration=booster.best_iteration)
            scores.append(float(average_precision_score(y[va], pred)))
            rounds.append(booster.best_iteration or cfg.max_rounds)
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned
        trial.set_user_attr("rounds", rounds)
        trial.set_user_attr("fold_pr_auc", [round(s, 5) for s in scores])
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    start = time.perf_counter()
    study.optimize(
        objective,
        n_trials=cfg.n_trials,
        timeout=cfg.timeout_minutes * 60,
        callbacks=[_progress(cfg.n_trials)],
    )
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = study.best_trial
    return TuningResult(
        best_params=dict(best.params),
        best_value=round(float(best.value or 0.0), 5),
        # Final model trains on all training months, a little more data than any fold.
        best_rounds=max(int(np.mean(best.user_attrs["rounds"])), 10),
        n_trials=len(study.trials),
        n_pruned=len(study.trials) - len(complete),
        seconds=round(time.perf_counter() - start, 1),
        trials=[
            {"number": t.number, "state": t.state.name, "value": t.value, **t.params, **t.user_attrs}
            for t in study.trials
        ],
    )


def _progress(n_trials: int) -> Any:
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        done = len(study.trials)
        if done % 10 == 0 or done == n_trials:
            log.info("Optuna: %d/%d trials, best fold PR-AUC %.4f", done, n_trials, study.best_value)

    return callback


def fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: list[str],
    categorical: list[int],
    params: dict[str, Any],
    rounds: int,
    seed: int,
) -> lgb.Booster:
    data = lgb.Dataset(
        x,
        y,
        feature_name=feature_names,
        categorical_feature=categorical,
        params={"verbosity": -1, "feature_pre_filter": False},
    )
    return lgb.train({**FIXED_PARAMS, **params, "seed": seed}, data, num_boost_round=rounds)
