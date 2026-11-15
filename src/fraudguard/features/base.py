"""Base group: raw columns the model may use, plus simple per-row derivations.

Nothing here looks at other rows, so it cannot leak by construction.
"""

from __future__ import annotations

import re

import polars as pl

from fraudguard.config import FeatureConfig
from fraudguard.data.schema import AMOUNT

_RAW_EXACT = (
    AMOUNT,
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "has_identity",
)
# Count (C), timedelta (D), match (M), Vesta (V) and identity (id_) columns.
_RAW_PATTERN = re.compile(r"^(C\d+|D\d+|M\d+|V\d+|id_\d+)$")


def raw_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c in _RAW_EXACT or _RAW_PATTERN.match(c)]


def build(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    amount = pl.col(AMOUNT)
    return df.select(
        *raw_columns(df),
        amount.log1p().cast(pl.Float32).alias("amt_log"),
        # Fractional cents separate foreign-currency conversions from round prices.
        (amount - amount.floor()).round(3).cast(pl.Float32).alias("amt_cents"),
        (pl.col("ts").dt.hour() + pl.col("ts").dt.minute() / 60).cast(pl.Float32).alias("hour"),
        pl.col("ts").dt.weekday().cast(pl.Int8).alias("weekday"),
    )
