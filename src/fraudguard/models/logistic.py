"""Logistic regression: how far a simple linear model gets.

Counts and amounts are heavy-tailed, so numeric inputs go through a signed
log transform before scaling; missing values are imputed with the median and
flagged with indicator columns (missingness is itself informative in IEEE-CIS).
Categoricals are one-hot encoded with rare levels grouped together.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraudguard.config import LogisticConfig

# Types MLflow's safe serializer (skops) must be told to trust when loading the pipeline.
TRUSTED_TYPES = ["fraudguard.models.logistic.signed_log1p", "numpy.dtype"]


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def select_columns(
    df: pl.DataFrame, feature_cols: list[str], cfg: LogisticConfig
) -> tuple[list[str], list[str]]:
    usable = [
        c
        for c in feature_cols
        if c not in cfg.exclude and not any(c.startswith(p) for p in cfg.exclude_prefixes)
    ]
    categorical = [c for c in cfg.categorical if c in usable]
    numeric = [c for c in usable if c not in categorical and df.schema[c] != pl.String]
    return numeric, categorical


def prepare(df: pl.DataFrame, numeric: list[str], categorical: list[str]) -> pl.DataFrame:
    """Plain numeric and string columns; nulls (and NaN) become NaN / None for the imputers."""
    return df.select(
        *[pl.col(c).cast(pl.Float64).fill_nan(None) for c in numeric],
        *[pl.col(c).cast(pl.String).fill_null("missing") for c in categorical],
    )


def build_pipeline(numeric: list[str], categorical: list[str], cfg: LogisticConfig, seed: int) -> Pipeline:
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    # Missing categoricals arrive as the explicit level "missing" (see `prepare`).
    categorical_steps = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=cfg.min_category_share
    )
    transformers = [("num", numeric_steps, numeric)]
    if categorical:
        transformers.append(("cat", categorical_steps, categorical))
    return Pipeline(
        [
            ("prep", ColumnTransformer(transformers, sparse_threshold=0.0)),
            ("model", LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, random_state=seed)),
        ]
    )
