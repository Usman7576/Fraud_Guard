"""Velocity group: how much activity an entity had in the recent past.

For each entity (card, email domain, device) and window (1h, 24h, 7d) we count
transactions and sum amounts over [t - window, t), i.e. strictly before the
current timestamp. Rows sharing the current second are excluded, so the online
scorer can reproduce the value without knowing about concurrent requests.
"""

from __future__ import annotations

import polars as pl

from fraudguard.config import FeatureConfig
from fraudguard.data.schema import AMOUNT

TIME = "ts"
_ONE = "__one"


def column_names(cfg: FeatureConfig) -> list[str]:
    return [
        f"vel_{alias}_{w}_{stat}"
        for alias in cfg.velocity_keys
        for w in cfg.velocity_windows
        for stat in ("cnt", "amt")
    ]


def build(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for alias, key in cfg.velocity_keys.items():
        missing_key = pl.col(key).is_null()
        for w in cfg.velocity_windows:
            cnt = pl.col(_ONE).rolling_sum_by(TIME, window_size=w, closed="left").over(key)
            amt = pl.col(AMOUNT).rolling_sum_by(TIME, window_size=w, closed="left").over(key)
            exprs += [
                pl.when(missing_key).then(None).otherwise(cnt).cast(pl.Int32).alias(f"vel_{alias}_{w}_cnt"),
                pl.when(missing_key).then(None).otherwise(amt).cast(pl.Float32).alias(f"vel_{alias}_{w}_amt"),
            ]
    return df.with_columns(pl.lit(1, dtype=pl.Int32).alias(_ONE)).select(exprs)
