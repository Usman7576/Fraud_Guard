from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fraudguard.config import AdversarialConfig
from fraudguard.data.split import SPLIT
from fraudguard.reports import adversarial
from tests.features_helpers import base_frame


@pytest.fixture(scope="module")
def frame(raw_frames: tuple[pl.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    df = base_frame(*raw_frames)
    rng = np.random.default_rng(0)
    return df.with_columns(
        pl.Series("stable", rng.normal(0, 1, df.height)),
        pl.Series("noise_cat", rng.choice(["a", "b", "c"], df.height)),
        # A feature whose distribution shifts in the test period.
        (pl.Series(rng.normal(0, 1, df.height)) + (pl.col(SPLIT) == "test").cast(pl.Float64) * 3).alias(
            "drifting"
        ),
    )


CFG = AdversarialConfig(n_folds=3, max_rows=10_000, num_boost_round=100, top_k=5)


def test_stable_features_look_similar(frame: pl.DataFrame) -> None:
    r = adversarial.run(frame, ["stable", "noise_cat"], CFG, seed=1)
    assert r.auc_mean < 0.6
    assert r.verdict.startswith("similar")


def test_drift_is_detected_and_attributed(frame: pl.DataFrame) -> None:
    r = adversarial.run(frame, ["stable", "noise_cat", "drifting"], CFG, seed=1)
    assert r.auc_mean > 0.9
    assert r.top_features[0][0] == "drifting"
    assert r.n_test_rows > 0 and r.n_train_rows > r.n_test_rows
    md = adversarial.to_markdown(r)
    assert "drifting" in md and "AUC" in md
