from __future__ import annotations

from datetime import date

import polars as pl

from fraudguard.data import identity, load


def test_card_uid_is_stable_across_a_cards_transactions() -> None:
    # Same card seen on day 10 (D1=5) and day 12 (D1=7): start day 5 both times.
    df = pl.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [10 * 86_400, 12 * 86_400, 12 * 86_400],
            "card1": [1111, 1111, 1111],
            "addr1": [300.0, 300.0, 300.0],
            "D1": [5.0, 7.0, 0.0],
            "isFraud": [0, 0, 1],
        }
    )
    out = identity.add_card_uid(load.add_time_columns(df, date(2017, 11, 30)))
    uids = out.sort("TransactionID")[identity.UID].to_list()
    assert uids[0] == uids[1] == "1111_300_5"
    assert uids[2] == "1111_300_12"  # a different (new) card with the same card1 and addr1


def test_missing_parts_do_not_drop_rows() -> None:
    df = pl.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [86_400],
            "card1": [5],
            "addr1": [None],
            "D1": [None],
            "isFraud": [0],
        },
        schema_overrides={"addr1": pl.Float64, "D1": pl.Float64},
    )
    out = identity.add_card_uid(load.add_time_columns(df, date(2017, 11, 30)))
    assert out[identity.UID].to_list() == ["5_na_na"]


def test_label_consistency_is_high_on_synthetic_cards(raw_frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    df = identity.add_card_uid(load.add_time_columns(raw_frames[0], date(2017, 11, 30)))
    assert identity.uid_label_consistency(df) > 0.5
