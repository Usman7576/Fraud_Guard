"""Time-based train / validation / test split.

Random splits leak the future into training: the model sees later behaviour
of the same cards. Splitting by time mirrors production, where a model trained
on the past scores transactions it has never seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import pairwise

import polars as pl

from fraudguard.config import SplitConfig
from fraudguard.data.schema import LABEL

SPLIT = "split"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SplitSummary:
    name: str
    rows: int
    fraud_rate: float
    start: datetime
    end: datetime


def assign_split(df: pl.DataFrame, cfg: SplitConfig) -> pl.DataFrame:
    if "ts" not in df.columns:
        raise ValueError("assign_split needs the ts column from add_time_columns")
    train_end = datetime.combine(cfg.train_end, datetime.min.time())
    val_end = datetime.combine(cfg.validation_end, datetime.min.time())
    return df.with_columns(
        pl.when(pl.col("ts") < train_end)
        .then(pl.lit(Split.TRAIN.value))
        .when(pl.col("ts") < val_end)
        .then(pl.lit(Split.VALIDATION.value))
        .otherwise(pl.lit(Split.TEST.value))
        .cast(pl.Enum([s.value for s in Split]))
        .alias(SPLIT)
    )


def summarise(df: pl.DataFrame) -> list[SplitSummary]:
    out = []
    for s in Split:
        part = df.filter(pl.col(SPLIT) == s.value)
        if part.height == 0:
            raise ValueError(f"split '{s.value}' is empty; check the split dates in configs/data.yaml")
        out.append(
            SplitSummary(
                name=s.value,
                rows=part.height,
                fraud_rate=float(part[LABEL].mean()),  # type: ignore[arg-type]
                start=part["ts"].min(),  # type: ignore[arg-type]
                end=part["ts"].max(),  # type: ignore[arg-type]
            )
        )
    return out


def check_no_overlap(summaries: list[SplitSummary]) -> None:
    """Every split must end strictly before the next one starts."""
    for earlier, later in pairwise(summaries):
        if earlier.end >= later.start:
            raise AssertionError(f"{earlier.name} ends at {earlier.end}, after {later.name} starts")
