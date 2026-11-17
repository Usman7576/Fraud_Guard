from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from fraudguard.config import DataConfig, load_feature_config
from fraudguard.data import pipeline as data_pipeline
from fraudguard.data import snapshot
from fraudguard.data.schema import LABEL
from fraudguard.data.split import SPLIT, Split
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.features import velocity
from fraudguard.features.online import OnlineFeatureStore, OutOfOrderError, window_seconds
from fraudguard.features.registry import feature_columns
from tests.features_helpers import base_frame, feature_cfg

REL_TOL = 1e-5
REFERENCE = date(2017, 11, 30)


def _same(a: object, b: object) -> bool:
    a_missing = a is None or (isinstance(a, float) and math.isnan(a))
    b_missing = b is None or (isinstance(b, float) and math.isnan(b))
    if a_missing or b_missing:
        return a_missing and b_missing
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=1e-6)
    return a == b


def test_window_parsing() -> None:
    assert window_seconds("1h") == 3_600
    assert window_seconds("7d") == 7 * 86_400
    with pytest.raises(ValueError, match="unsupported"):
        window_seconds("1mo")


def test_velocity_window_edges_match_offline() -> None:
    """[t - window, t): an event exactly one window earlier counts; same-second events do not."""
    t0 = 40 * 86_400
    times = [t0, t0 + 1_800, t0 + 3_600, t0 + 3_600, t0 + 3_601, t0 + 7_300]
    df = base_frame(
        pl.DataFrame(
            {
                "TransactionID": list(range(1, len(times) + 1)),
                "TransactionDT": times,
                "card_uid": ["A"] * len(times),
                "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "isFraud": [0] * len(times),
                "DeviceInfo": ["d"] * len(times),
                "P_emaildomain": ["e"] * len(times),
                "card1": [1] * len(times),
                "addr1": [None] * len(times),
                "D1": [None] * len(times),
            },
            schema_overrides={"addr1": pl.Float64, "D1": pl.Float64},
        )
    )
    cfg = feature_cfg()
    offline = velocity.build(df, cfg)
    store = OnlineFeatureStore(cfg, REFERENCE)
    online = []
    for row in df.iter_rows(named=True):
        online.append(store.features(row))
        store.add(row)
    for col in ("vel_device_1h_cnt", "vel_device_1h_amt", "vel_email_24h_cnt"):
        assert [o[col] for o in online] == pytest.approx(offline[col].to_list()), col


def test_out_of_order_transactions_are_rejected() -> None:
    store = OnlineFeatureStore(feature_cfg(), REFERENCE)
    row = {"TransactionID": 1, "TransactionDT": 1_000, "TransactionAmt": 5.0, "card1": 1}
    store.add(row)
    with pytest.raises(OutOfOrderError):
        store.features({**row, "TransactionDT": 999})


@pytest.mark.slow
def test_online_features_match_offline_snapshot(data_config: DataConfig) -> None:
    """Warm the store with train + validation, replay the test month row by row, compare every feature."""
    feat_cfg = load_feature_config()
    data_pipeline.build_base(data_config)
    feature_pipeline.build(data_config, feat_cfg)
    root = data_config.paths.snapshot_dir
    base = snapshot.load_snapshot(root, data_pipeline.STAGE, "transactions")
    offline = snapshot.load_snapshot(root, feature_pipeline.STAGE, "features")
    engineered = [c for c in feature_columns(offline) if c not in base.columns] + ["card_uid"]

    store = OnlineFeatureStore(feat_cfg, data_config.reference_date)
    store.warm(base.filter(pl.col(SPLIT) != Split.TEST.value))
    test_rows = base.filter(pl.col(SPLIT) == Split.TEST.value)
    expected = offline.filter(pl.col(SPLIT) == Split.TEST.value).select(engineered).rows(named=True)
    assert len(expected) == test_rows.height > 100

    mismatches: list[str] = []
    for row, want in zip(test_rows.iter_rows(named=True), expected, strict=True):
        got = store.features(row)
        mismatches += [
            f"{row['TransactionID']} {c}: online {got[c]!r} vs offline {want[c]!r}"
            for c in engineered
            if not _same(got[c], want[c])
        ]
        store.add(row, row[LABEL] == 1)
    assert not mismatches, f"{len(mismatches)} mismatches, first: {mismatches[:10]}"
