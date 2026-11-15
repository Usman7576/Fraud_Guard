"""Approximate card identity.

IEEE-CIS has no card ID. A widely used approximation combines the card number
proxy (card1), the billing region (addr1) and the account's start day, which
is the transaction day minus D1 (days since the card was first seen). It is an
approximation and is documented as such; `uid_label_consistency` measures it.
"""

from __future__ import annotations

import polars as pl

from fraudguard.data.schema import LABEL

UID = "card_uid"


def add_card_uid(df: pl.DataFrame) -> pl.DataFrame:
    if "day" not in df.columns:
        raise ValueError("add_time_columns must run before add_card_uid")
    # CSV readers may give NaN instead of null for blanks; treat both as missing.
    d1 = pl.col("D1").cast(pl.Float64).fill_nan(None)
    addr1 = pl.col("addr1").cast(pl.Float64).fill_nan(None)
    start_day = (pl.col("day") - d1).floor().cast(pl.Int32)
    return df.with_columns(
        pl.concat_str(
            [
                pl.col("card1").cast(pl.Utf8),
                addr1.cast(pl.Int32).cast(pl.Utf8).fill_null("na"),
                start_day.cast(pl.Utf8).fill_null("na"),
            ],
            separator="_",
        ).alias(UID)
    )


def uid_label_consistency(df: pl.DataFrame, min_rows: int = 2) -> float:
    """Share of multi-transaction uids whose labels are all the same.

    Fraud tends to cluster by card, so a good uid should be label-consistent
    far more often than chance. Used as a quality check, not as a feature.
    """
    grouped = (
        df.group_by(UID)
        .agg(pl.len().alias("n"), pl.col(LABEL).n_unique().alias("labels"))
        .filter(pl.col("n") >= min_rows)
    )
    if grouped.height == 0:
        return float("nan")
    return float((grouped["labels"] == 1).mean())  # type: ignore[arg-type]
