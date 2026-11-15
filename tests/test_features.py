from __future__ import annotations

import math

import polars as pl
import pytest

from fraudguard.features import base, behaviour, velocity
from fraudguard.features.registry import META_COLUMNS, build_features, feature_columns
from tests.features_helpers import base_frame, feature_cfg, tiny

HOUR = 3_600


def _by_id(out: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([df.select("TransactionID"), out], how="horizontal").sort("TransactionID")


class TestVelocity:
    def test_counts_and_amounts_use_only_earlier_timestamps(self) -> None:
        df = tiny([(0, "A", 10.0), (600, "A", 20.0), (1_800, "A", 30.0), (1_800, "B", 5.0)])
        out = _by_id(velocity.build(df, feature_cfg()), df)
        assert out["vel_card_1h_cnt"].to_list() == [0, 1, 2, 0]
        assert out["vel_card_1h_amt"].to_list() == [0.0, 10.0, 30.0, 0.0]

    def test_same_second_rows_are_not_counted(self) -> None:
        df = tiny([(100, "A", 1.0), (100, "A", 2.0), (100, "A", 3.0)])
        out = velocity.build(df, feature_cfg())
        assert out["vel_card_1h_cnt"].to_list() == [0, 0, 0]

    def test_window_start_is_inclusive_and_older_rows_drop_out(self) -> None:
        df = tiny([(0, "A", 1.0), (HOUR, "A", 1.0), (HOUR + 1, "A", 1.0)])
        out = _by_id(velocity.build(df, feature_cfg()), df)
        # Row 2 sits exactly 1h after row 1: [t-1h, t) includes it.
        # Row 3 is 1h+1s after row 1, so only row 2 is in its 1h window, but both are in 24h.
        assert out["vel_card_1h_cnt"].to_list() == [0, 1, 1]
        assert out["vel_card_24h_cnt"].to_list() == [0, 1, 2]

    def test_missing_key_gives_null_not_zero(self) -> None:
        df = tiny([(0, "A", 1.0), (10, "A", 1.0)], email=None)
        out = velocity.build(df, feature_cfg())
        assert out["vel_email_1h_cnt"].null_count() == 2
        assert out["vel_device_7d_cnt"].null_count() == 2
        assert out["vel_card_1h_cnt"].null_count() == 0

    def test_column_names_match_config(self) -> None:
        cfg = feature_cfg()
        df = tiny([(0, "A", 1.0)])
        assert velocity.build(df, cfg).columns == velocity.column_names(cfg)
        assert len(velocity.column_names(cfg)) == 3 * 3 * 2


class TestBehaviour:
    def test_history_stats(self) -> None:
        df = tiny([(0, "A", 10.0), (60, "A", 30.0), (120, "A", 50.0), (180, "B", 7.0)])
        out = _by_id(behaviour.build(df, feature_cfg()), df)
        assert out["card_prior_txn"].to_list() == [0, 1, 2, 0]
        assert out["card_is_new"].to_list() == [True, False, False, True]
        assert out["card_secs_since_last"].to_list() == [None, 60.0, 60.0, None]
        assert out["card_amt_mean"].to_list() == [None, 10.0, 20.0, None]
        # Third row: prior amounts 10, 30 -> mean 20, sample std 14.142; z = (50 - 20) / 14.142.
        z = out["card_amt_z"].to_list()
        assert z[0] is None and z[1] is None  # fewer than 2 earlier transactions
        assert z[2] == pytest.approx(30 / math.sqrt(200), rel=1e-5)
        assert out["card_amt_ratio"][2] == pytest.approx(2.5)

    def test_hour_deviation_wraps_around_midnight(self) -> None:
        # Day 40 starts at midnight. Earlier transaction at 23:00, current at 01:00 next day.
        df = tiny([(23 * HOUR, "A", 1.0), (25 * HOUR, "A", 1.0)])
        out = _by_id(behaviour.build(df, feature_cfg()), df)
        assert out["card_hour_dev"][0] is None
        assert out["card_hour_dev"][1] == pytest.approx(2.0, abs=1e-4)


class TestBaseAndRegistry:
    def test_base_derivations(self) -> None:
        df = tiny([(13 * HOUR + 30 * 60, "A", 12.345)])
        out = base.build(df, feature_cfg())
        assert out["amt_cents"][0] == pytest.approx(0.345, abs=1e-3)
        assert out["hour"][0] == pytest.approx(13.5)
        assert out["amt_log"][0] == pytest.approx(math.log1p(12.345), rel=1e-6)

    def test_build_features_keeps_rows_and_meta(self, raw_frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        df = base_frame(*raw_frames)
        out = build_features(df, feature_cfg())
        assert out.height == df.height
        assert out["TransactionID"].equals(df["TransactionID"])
        assert set(META_COLUMNS) <= set(out.columns)
        cols = feature_columns(out)
        assert "isFraud" not in cols and "card_uid" not in cols and "ts" not in cols
        assert {"vel_card_24h_cnt", "card_amt_z", "amt_log", "C1"} <= set(cols)

    def test_unsorted_input_is_rejected(self) -> None:
        df = tiny([(0, "A", 1.0), (10, "A", 1.0)]).reverse()
        with pytest.raises(ValueError, match="sorted"):
            build_features(df, feature_cfg())
