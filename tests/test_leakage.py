"""The leakage test. If this fails, a feature is using information from the future.

It runs the harness on every registered feature group, then sabotages the
features in common ways to prove the harness would catch each one.
"""

from __future__ import annotations

import polars as pl
import pytest

from fraudguard.data.identity import UID
from fraudguard.data.schema import AMOUNT, LABEL
from fraudguard.features.leakage import find_leaks
from fraudguard.features.registry import GROUPS, build_features
from tests.features_helpers import base_frame, feature_cfg


@pytest.fixture(scope="module")
def base_df(raw_frames: tuple[pl.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    return base_frame(*raw_frames)


@pytest.mark.parametrize("group", list(GROUPS))
def test_feature_group_never_looks_ahead(base_df: pl.DataFrame, group: str) -> None:
    cfg = feature_cfg()
    leaks = find_leaks(base_df, lambda d: GROUPS[group].build(d, cfg))
    assert not leaks, "\n".join(str(leak) for leak in leaks)


def test_full_feature_set_never_looks_ahead(base_df: pl.DataFrame) -> None:
    cfg = feature_cfg()
    leaks = find_leaks(base_df, lambda d: build_features(d, cfg))
    assert not leaks, "\n".join(str(leak) for leak in leaks)


SABOTAGE = {
    "group-wide mean (uses the card's future amounts)": pl.col(AMOUNT).mean().over(UID),
    "forward shift (next transaction's amount)": pl.col(AMOUNT).shift(-1).over(UID),
    "label target encoding over all rows": pl.col(LABEL).mean().over(UID),
    "window that includes later rows": pl.col(AMOUNT)
    .reverse()
    .rolling_sum(window_size=3, min_samples=1)
    .reverse()
    .over(UID),
    "global normalisation": (pl.col(AMOUNT) - pl.col(AMOUNT).mean()) / pl.col(AMOUNT).std(),
}


@pytest.mark.parametrize("name", list(SABOTAGE))
def test_harness_catches_sabotage(base_df: pl.DataFrame, name: str) -> None:
    leaks = find_leaks(base_df, lambda d: d.select(SABOTAGE[name].alias("leaky")))
    assert leaks, f"harness missed: {name}"
    assert leaks[0].column == "leaky"
