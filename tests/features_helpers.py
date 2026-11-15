from __future__ import annotations

from datetime import date

import polars as pl

from fraudguard.config import FeatureConfig, SplitConfig
from fraudguard.data import identity, load, split

SPLIT_CFG = SplitConfig(train_end="2018-04-01", validation_end="2018-05-01")


def feature_cfg(**overrides: object) -> FeatureConfig:
    base: dict[str, object] = {
        "velocity_windows": ["1h", "24h", "7d"],
        "velocity_keys": {"card": "card_uid", "email": "P_emaildomain", "device": "DeviceInfo"},
        "history_window": "3650d",
        "min_history_for_zscore": 2,
    }
    base.update(overrides)
    return FeatureConfig.model_validate(base)


def base_frame(raw: pl.DataFrame, identity_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Turn a raw-like frame into the base snapshot shape used by feature builders."""
    df = load.join_identity(raw, identity_df) if identity_df is not None else raw
    df = load.add_time_columns(df, date(2017, 11, 30))
    if identity.UID not in df.columns:
        df = identity.add_card_uid(df)
    return split.assign_split(df, SPLIT_CFG)


def tiny(rows: list[tuple[int, str, float]], email: str | None = "gmail.com") -> pl.DataFrame:
    """Rows of (seconds offset, card_uid, amount) on one day; IDs follow input order."""
    return base_frame(
        pl.DataFrame(
            {
                "TransactionID": list(range(1, len(rows) + 1)),
                "TransactionDT": [86_400 * 40 + s for s, _, _ in rows],
                "card_uid": [c for _, c, _ in rows],
                "TransactionAmt": [a for _, _, a in rows],
                "P_emaildomain": [email] * len(rows),
                "DeviceInfo": [None] * len(rows),
                "isFraud": [0] * len(rows),
            },
            schema_overrides={"DeviceInfo": pl.String, "P_emaildomain": pl.String},
        )
    )
