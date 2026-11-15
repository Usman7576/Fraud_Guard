from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from fraudguard.config import SplitConfig
from fraudguard.data import load, split
from fraudguard.data.split import SPLIT, Split


def _with_time(raw: pl.DataFrame) -> pl.DataFrame:
    return load.add_time_columns(raw, date(2017, 11, 30))


def test_splits_are_ordered_and_disjoint(raw_frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    df = split.assign_split(
        _with_time(raw_frames[0]), SplitConfig(train_end="2018-04-01", validation_end="2018-05-01")
    )
    summaries = split.summarise(df)
    split.check_no_overlap(summaries)
    assert [s.name for s in summaries] == ["train", "validation", "test"]
    assert sum(s.rows for s in summaries) == df.height


def test_every_train_row_is_earlier_than_every_test_row(
    raw_frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    df = split.assign_split(
        _with_time(raw_frames[0]), SplitConfig(train_end="2018-04-01", validation_end="2018-05-01")
    )
    train_max = df.filter(pl.col(SPLIT) == Split.TRAIN.value)["TransactionDT"].max()
    test_min = df.filter(pl.col(SPLIT) == Split.TEST.value)["TransactionDT"].min()
    assert train_max < test_min  # type: ignore[operator]


@settings(max_examples=25, deadline=None)
@given(
    train_days=st.integers(min_value=20, max_value=120),
    val_days=st.integers(min_value=5, max_value=40),
)
def test_split_boundaries_hold_for_any_dates(
    raw_frames: tuple[pl.DataFrame, pl.DataFrame], train_days: int, val_days: int
) -> None:
    """Whatever boundaries are chosen, no split may reach into a later one."""
    start = date(2017, 12, 1)
    cfg = SplitConfig(
        train_end=start + timedelta(days=train_days),
        validation_end=start + timedelta(days=train_days + val_days),
    )
    df = split.assign_split(_with_time(raw_frames[0]), cfg)
    split.check_no_overlap(split.summarise(df))
