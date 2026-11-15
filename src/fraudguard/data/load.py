"""Load and join the raw IEEE-CIS files into one typed frame."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

from fraudguard.data.schema import AMOUNT, DT, ID, LABEL

SECONDS_PER_DAY = 86_400

# Columns that must keep full precision. Every other float is downcast to
# Float32, which roughly halves memory on the 590K x 430 frame.
_KEEP_PRECISION = {ID, DT, AMOUNT, LABEL}


def read_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the IEEE-CIS files from Kaggle and place them in {path.parent}"
        )
    return pl.read_csv(path, infer_schema_length=50_000)


def downcast(df: pl.DataFrame) -> pl.DataFrame:
    casts = [
        pl.col(name).cast(pl.Float32)
        for name, dtype in df.schema.items()
        if dtype == pl.Float64 and name not in _KEEP_PRECISION
    ]
    return df.with_columns(casts) if casts else df


def join_identity(transactions: pl.DataFrame, identity: pl.DataFrame) -> pl.DataFrame:
    """Left-join identity onto transactions. Most transactions have no identity row."""
    extra = [c for c in identity.columns if c != ID and c in transactions.columns]
    if extra:
        raise ValueError(f"identity and transaction files share non-key columns: {extra}")
    joined = transactions.join(identity, on=ID, how="left")
    return joined.with_columns(pl.col(ID).is_in(identity[ID].implode()).alias("has_identity"))


def add_time_columns(df: pl.DataFrame, reference_date: date) -> pl.DataFrame:
    """Turn the TransactionDT seconds offset into a timestamp and day index."""
    ref = datetime.combine(reference_date, datetime.min.time())
    return df.with_columns(
        (pl.lit(ref) + pl.duration(seconds=pl.col(DT))).alias("ts"),
        (pl.col(DT) // SECONDS_PER_DAY).cast(pl.Int32).alias("day"),
    ).sort(DT, ID)


def load_raw(transactions_path: Path, identity_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    return read_csv(transactions_path), read_csv(identity_path)
