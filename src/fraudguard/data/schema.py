"""Raw input schema checks.

The IEEE-CIS files have ~430 columns. We check the ones the pipeline relies on,
plus invariants that would silently corrupt a time split if broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

ID = "TransactionID"
LABEL = "isFraud"
DT = "TransactionDT"
AMOUNT = "TransactionAmt"

REQUIRED_TRANSACTION_COLUMNS: tuple[str, ...] = (
    ID,
    LABEL,
    DT,
    AMOUNT,
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "D1",
)

REQUIRED_IDENTITY_COLUMNS: tuple[str, ...] = (ID, "DeviceType", "DeviceInfo")


class SchemaError(ValueError):
    """Raised when raw data breaks an invariant the pipeline depends on."""


@dataclass
class ValidationReport:
    rows: int
    fraud_rate: float
    dt_min: int
    dt_max: int
    warnings: list[str] = field(default_factory=list)


def _missing(df: pl.DataFrame, required: tuple[str, ...]) -> list[str]:
    return [c for c in required if c not in df.columns]


def validate_identity(df: pl.DataFrame) -> None:
    if missing := _missing(df, REQUIRED_IDENTITY_COLUMNS):
        raise SchemaError(f"identity file is missing columns: {missing}")
    if df[ID].n_unique() != df.height:
        raise SchemaError("identity file has duplicate TransactionID values")


def validate_transactions(df: pl.DataFrame) -> ValidationReport:
    if missing := _missing(df, REQUIRED_TRANSACTION_COLUMNS):
        raise SchemaError(f"transaction file is missing columns: {missing}")
    if df.height == 0:
        raise SchemaError("transaction file is empty")
    if df[ID].n_unique() != df.height:
        raise SchemaError("transaction file has duplicate TransactionID values")
    if df[DT].null_count() > 0:
        raise SchemaError("TransactionDT contains nulls; the time split would be undefined")
    if df[LABEL].null_count() > 0:
        raise SchemaError("isFraud contains nulls")

    labels = set(df[LABEL].unique().to_list())
    if not labels <= {0, 1}:
        raise SchemaError(f"isFraud must be 0/1, found {sorted(labels)}")

    amount_min = df[AMOUNT].min()
    if amount_min is None or float(amount_min) <= 0:  # type: ignore[arg-type]
        raise SchemaError("TransactionAmt must be strictly positive")

    dt_min, dt_max = int(df[DT].min()), int(df[DT].max())  # type: ignore[arg-type]
    fraud_rate = float(df[LABEL].mean())  # type: ignore[arg-type]

    report = ValidationReport(rows=df.height, fraud_rate=fraud_rate, dt_min=dt_min, dt_max=dt_max)
    if not 0.005 <= fraud_rate <= 0.2:
        report.warnings.append(f"fraud rate {fraud_rate:.2%} is outside the expected 0.5%-20% range")
    return report
