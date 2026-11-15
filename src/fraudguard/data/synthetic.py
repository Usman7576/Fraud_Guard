"""Synthetic IEEE-CIS-shaped data.

Used by tests and CI (the real dataset cannot be redistributed) and for local
development before the Kaggle files are downloaded. It mimics the columns and
broad structure: cards with histories, shared devices, ~3.5% fraud, a time
span of about six months. It is not a statistical stand-in for the real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from fraudguard.data.load import SECONDS_PER_DAY

DT_START = SECONDS_PER_DAY  # matches the real minimum TransactionDT
SPAN_DAYS = 182
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", "icloud.com"]
DEVICES = ["Windows", "iOS Device", "MacOS", "SM-G960U", "Trident/7.0", "Moto G"]


def _optional_str(rng: np.random.Generator, values: list[str], n: int, p_null: float = 0.2) -> pl.Series:
    """Strings with some nulls, typed explicitly (numpy would otherwise produce object dtype)."""
    picked = rng.choice(values, n).tolist()
    nulls = rng.random(n) < p_null
    return pl.Series([None if z else v for v, z in zip(picked, nulls, strict=True)], dtype=pl.String)


def generate(
    n_transactions: int = 20_000, seed: int = 42, fraud_rate: float = 0.035
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    n_cards = max(n_transactions // 8, 10)

    card1 = rng.integers(1000, 18_000, n_cards)
    addr1 = rng.choice(np.append(rng.integers(100, 540, 40).astype(float), np.nan), n_cards)
    card_start_day = rng.integers(-300, SPAN_DAYS, n_cards)
    card_is_bad = rng.random(n_cards) < fraud_rate * 0.9
    card_email = rng.choice(EMAIL_DOMAINS, n_cards)
    # A small pool of devices shared by bad cards creates entity-graph structure.
    bad_devices = [f"dev-bad-{i}" for i in range(max(n_cards // 200, 2))]

    card_idx = rng.integers(0, n_cards, n_transactions)
    dt = np.sort(rng.integers(DT_START, DT_START + SPAN_DAYS * SECONDS_PER_DAY, n_transactions))
    day = dt // SECONDS_PER_DAY
    # Cards cannot transact before they exist: push start days back if needed.
    start = np.minimum(card_start_day[card_idx], day)

    is_bad = card_is_bad[card_idx]
    # As in IEEE-CIS, once a card is compromised nearly all its transactions are labelled fraud.
    is_fraud = (is_bad & (rng.random(n_transactions) < 0.95)) | (
        rng.random(n_transactions) < fraud_rate * 0.05
    )
    amount = (
        np.round(
            np.where(
                is_fraud, rng.lognormal(5.0, 1.1, n_transactions), rng.lognormal(4.0, 0.9, n_transactions)
            ),
            2,
        )
        + 0.01
    )

    transactions = pl.DataFrame(
        {
            "TransactionID": np.arange(2_987_000, 2_987_000 + n_transactions),
            "isFraud": is_fraud.astype(np.int64),
            "TransactionDT": dt.astype(np.int64),
            "TransactionAmt": amount,
            "ProductCD": rng.choice(
                ["W", "C", "R", "H", "S"], n_transactions, p=[0.7, 0.12, 0.07, 0.06, 0.05]
            ),
            "card1": card1[card_idx],
            "card2": rng.choice([111.0, 321.0, 555.0, np.nan], n_transactions),
            "card3": np.full(n_transactions, 150.0),
            "card4": rng.choice(["visa", "mastercard", "american express", "discover"], n_transactions),
            "card5": rng.choice([226.0, 224.0, 166.0, 102.0], n_transactions),
            "card6": rng.choice(["debit", "credit"], n_transactions, p=[0.75, 0.25]),
            "addr1": addr1[card_idx],
            "addr2": np.where(np.isnan(addr1[card_idx]), np.nan, 87.0),
            "P_emaildomain": card_email[card_idx],
            "R_emaildomain": _optional_str(rng, EMAIL_DOMAINS, n_transactions),
            "D1": (day - start).astype(float),
            "C1": rng.poisson(np.where(is_bad, 4, 1.5)).astype(float),
            "C2": rng.poisson(1.5, n_transactions).astype(float),
            "V1": rng.normal(0, 1, n_transactions),
            "V2": rng.normal(np.where(is_fraud, 0.8, 0.0), 1),
            "M1": _optional_str(rng, ["T", "F"], n_transactions),
        }
    )

    has_identity = rng.random(n_transactions) < np.where(is_bad, 0.6, 0.25)
    ids = transactions["TransactionID"].to_numpy()[has_identity]
    bad_rows = is_bad[has_identity]
    device_info = np.where(
        bad_rows & (rng.random(len(ids)) < 0.7),
        rng.choice(bad_devices, len(ids)),
        rng.choice(DEVICES, len(ids)),
    )
    identity = pl.DataFrame(
        {
            "TransactionID": ids,
            "DeviceType": rng.choice(["desktop", "mobile"], len(ids)),
            "DeviceInfo": device_info,
            "id_01": rng.choice([0.0, -5.0, -10.0], len(ids)),
        }
    )
    return transactions, identity


def write_synthetic(out_dir: Path, n_transactions: int = 20_000, seed: int = 42) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    transactions, identity = generate(n_transactions=n_transactions, seed=seed)
    t_path, i_path = out_dir / "train_transaction.csv", out_dir / "train_identity.csv"
    transactions.write_csv(t_path)
    identity.write_csv(i_path)
    return t_path, i_path
