"""Behaviour group: how unusual this transaction is for this card.

All statistics use the card's history strictly before the current timestamp:
- amount z-score and ratio against the card's earlier mean and std
- seconds since the card's previous transaction
- whether the card is new (no history)
- how far the hour of day is from the card's usual hour, using a circular mean
  so that 23:00 and 01:00 are two hours apart, not twenty-two
"""

from __future__ import annotations

import math

import polars as pl

from fraudguard.config import FeatureConfig
from fraudguard.data.identity import UID
from fraudguard.data.schema import AMOUNT, DT

TIME = "ts"
COLUMNS = (
    "card_prior_txn",
    "card_is_new",
    "card_secs_since_last",
    "card_amt_mean",
    "card_amt_z",
    "card_amt_ratio",
    "card_hour_dev",
)


def build(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    hist = cfg.history_window

    def prior(expr: pl.Expr, how: str) -> pl.Expr:
        fn = getattr(expr, f"rolling_{how}_by")
        return fn(TIME, window_size=hist, closed="left").over(UID)  # type: ignore[no-any-return]

    angle = (pl.col(TIME).dt.hour() + pl.col(TIME).dt.minute() / 60) * (2 * math.pi / 24)
    helpers = df.with_columns(
        pl.lit(1, dtype=pl.Int32).alias("__one"),
        angle.sin().alias("__sin"),
        angle.cos().alias("__cos"),
        angle.alias("__angle"),
    )

    n = prior(pl.col("__one"), "sum")
    mean = prior(pl.col(AMOUNT), "mean")
    std = prior(pl.col(AMOUNT), "std")
    last_dt = prior(pl.col(DT), "max")
    usual_angle = pl.arctan2(prior(pl.col("__sin"), "mean"), prior(pl.col("__cos"), "mean"))

    stats = helpers.select(
        n.cast(pl.Int32).alias("card_prior_txn"),
        mean.alias("__mean"),
        std.alias("__std"),
        (pl.col(DT) - last_dt).cast(pl.Float32).alias("card_secs_since_last"),
        ((pl.col("__angle") - usual_angle).abs()).alias("__diff"),
        pl.col(AMOUNT),
    )

    enough = (pl.col("card_prior_txn") >= cfg.min_history_for_zscore) & (pl.col("__std") > 0)
    circular = pl.min_horizontal(pl.col("__diff"), 2 * math.pi - pl.col("__diff")) * (24 / (2 * math.pi))
    return stats.select(
        "card_prior_txn",
        (pl.col("card_prior_txn") == 0).alias("card_is_new"),
        "card_secs_since_last",
        pl.col("__mean").cast(pl.Float32).alias("card_amt_mean"),
        pl.when(enough)
        .then((pl.col(AMOUNT) - pl.col("__mean")) / pl.col("__std"))
        .cast(pl.Float32)
        .alias("card_amt_z"),
        (pl.col(AMOUNT) / pl.col("__mean")).cast(pl.Float32).alias("card_amt_ratio"),
        pl.when(pl.col("card_prior_txn") > 0).then(circular).cast(pl.Float32).alias("card_hour_dev"),
    )
