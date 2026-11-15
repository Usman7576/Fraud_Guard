from __future__ import annotations

import polars as pl

from fraudguard.features import graph
from tests.features_helpers import base_frame, feature_cfg

DAY = 86_400
START = 40 * DAY


def frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    """rows: dicts with t (seconds after START), card, and optional device, addr, fraud."""
    return base_frame(
        pl.DataFrame(
            {
                "TransactionID": list(range(1, len(rows) + 1)),
                "TransactionDT": [START + int(r["t"]) for r in rows],  # type: ignore[call-overload]
                "card_uid": [r["card"] for r in rows],
                "TransactionAmt": [10.0] * len(rows),
                "isFraud": [int(r.get("fraud", 0)) for r in rows],  # type: ignore[call-overload]
                "DeviceInfo": [r.get("device") for r in rows],
                "P_emaildomain": [r.get("email", "gmail.com") for r in rows],
                "addr1": [r.get("addr") for r in rows],
                "addr2": [87.0 if r.get("addr") is not None else None for r in rows],
            },
            schema_overrides={"DeviceInfo": pl.String, "addr1": pl.Float64, "addr2": pl.Float64},
        )
    )


def build(rows: list[dict[str, object]], **graph_cfg: object) -> pl.DataFrame:
    cfg = feature_cfg(graph={"label_delay_days": 30, "hop_entities": ["device", "address"], **graph_cfg})
    df = frame(rows)
    return pl.concat([df.select("TransactionID"), graph.build(df, cfg)], how="horizontal").sort(
        "TransactionID"
    )


def test_shared_device_counts_only_earlier_cards() -> None:
    out = build(
        [
            {"t": 0, "card": "A", "device": "d1"},
            {"t": 10, "card": "B", "device": "d1"},
            {"t": 20, "card": "C", "device": "d1"},
            {"t": 30, "card": "A", "device": "d2"},
        ]
    )
    assert out["graph_device_n_cards_recent"].to_list() == [0, 1, 2, 0]
    assert out["graph_card_n_devices"].to_list() == [0, 0, 0, 1]


def test_same_timestamp_rows_do_not_see_each_other() -> None:
    out = build([{"t": 5, "card": "A", "device": "d1"}, {"t": 5, "card": "B", "device": "d1"}])
    assert out["graph_device_n_cards_recent"].to_list() == [0, 0]


def test_missing_device_gives_null() -> None:
    out = build([{"t": 0, "card": "A"}, {"t": 1, "card": "A", "device": "d1"}])
    assert out["graph_device_n_cards_recent"].to_list() == [None, 0]
    assert out["graph_card_n_devices"].to_list() == [0, 0]


def test_fraud_label_is_only_used_after_the_delay() -> None:
    rows = [
        {"t": 0, "card": "A", "device": "d1", "fraud": 1},
        {"t": 29 * DAY, "card": "B", "device": "d1"},  # chargeback not in yet
        {"t": 30 * DAY, "card": "C", "device": "d1"},  # exactly 30 days: now known
        {"t": 31 * DAY, "card": "A", "device": "d9"},
    ]
    out = build(rows)
    assert out["graph_device_fraud_cards"].to_list() == [0, 0, 1, 0]
    assert out["graph_card_known_fraud"].to_list() == [0, 0, 0, 1]
    # C sits next to known-fraud A via d1. A's own fraud never counts as a neighbour's.
    assert out["graph_2hop_fraud_cards"].to_list() == [0, 0, 1, 0]

    no_delay = build(rows, label_delay_days=0)
    assert no_delay["graph_device_fraud_cards"].to_list() == [0, 1, 1, 0]


def test_two_hops_reach_through_devices_and_addresses_but_no_further() -> None:
    out = build(
        [
            {"t": 0, "card": "A", "device": "d1", "fraud": 1},
            {"t": 1, "card": "B", "device": "d1", "addr": 300.0},
            {"t": 2, "card": "C", "addr": 300.0},
            {"t": 40 * DAY, "card": "B", "device": "d7"},  # B: A is two hops away via d1
            {"t": 40 * DAY + 1, "card": "C", "device": "d8"},  # C -> addr -> B -> d1 -> A is four hops
        ]
    )
    assert out["graph_2hop_fraud_cards"].to_list()[3:] == [1, 0]
    assert out["graph_2hop_cards"].to_list()[3:] == [2, 1]  # B sees A and C; C sees B


def test_crowded_entities_are_not_expanded() -> None:
    rows: list[dict[str, object]] = [{"t": i, "card": f"c{i}", "device": "shared"} for i in range(5)]
    rows.append({"t": 100, "card": "new", "device": "shared"})
    capped = build(rows, max_neighbours=3)
    uncapped = build(rows, max_neighbours=10)
    assert capped["graph_2hop_cards"][-1] == 0
    assert uncapped["graph_2hop_cards"][-1] == 5
    assert capped["graph_device_n_cards_recent"][-1] == 5  # the entity's own count is still exact


def test_entity_size_is_a_rolling_window_not_all_time() -> None:
    out = build(
        [
            {"t": 0, "card": "A", "device": "d1", "addr": 300.0},
            {"t": 1, "card": "A", "device": "d1", "addr": 300.0},  # same card twice: still one card
            {"t": 10 * DAY, "card": "B", "device": "d1", "addr": 300.0},
            {"t": 35 * DAY, "card": "C", "device": "d1", "addr": 300.0},  # A has aged out; B is 25 days old
        ],
        entity_window_days=30,
    )
    assert out["graph_device_n_cards_recent"].to_list() == [0, 1, 1, 1]
    assert out["graph_address_n_cards_recent"].to_list() == [0, 1, 1, 1]


def test_address_key() -> None:
    df = pl.DataFrame({"addr1": [300.0, None, 120.0], "addr2": [87.0, 87.0, None]})
    assert graph.address_key(df).to_list() == ["300-87", None, "120-na"]
