from __future__ import annotations

import polars as pl
import pytest

from fraudguard.data.schema import SchemaError, validate_identity, validate_transactions


def test_valid_synthetic_data_passes(raw_frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    transactions, identity = raw_frames
    report = validate_transactions(transactions)
    validate_identity(identity)
    assert report.rows == transactions.height
    assert 0.01 < report.fraud_rate < 0.15


def test_missing_column_is_rejected(raw_frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    with pytest.raises(SchemaError, match="D1"):
        validate_transactions(raw_frames[0].drop("D1"))


def test_duplicate_ids_are_rejected(raw_frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    df = raw_frames[0]
    with pytest.raises(SchemaError, match="duplicate"):
        validate_transactions(pl.concat([df, df.head(1)]))


def test_non_binary_label_is_rejected(raw_frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    df = raw_frames[0].with_columns(pl.lit(2).alias("isFraud"))
    with pytest.raises(SchemaError, match="0/1"):
        validate_transactions(df)


def test_null_time_is_rejected(raw_frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    df = raw_frames[0].with_columns(pl.lit(None, dtype=pl.Int64).alias("TransactionDT"))
    with pytest.raises(SchemaError, match="TransactionDT"):
        validate_transactions(df)


def test_non_positive_amount_is_rejected(raw_frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    df = raw_frames[0].with_columns(pl.lit(0.0).alias("TransactionAmt"))
    with pytest.raises(SchemaError, match="positive"):
        validate_transactions(df)
