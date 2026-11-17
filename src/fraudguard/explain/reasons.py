"""Reason codes: the top features pushing a transaction towards fraud, in plain words.

Contributions are exact TreeSHAP values from LightGBM in log-odds. Only
features that *raise* the risk are reasons; a transaction with fewer risky
features gets fewer reasons. The text uses the transaction's original value
(the category name, not its code), for example "7 earlier transactions on
this card in the last 1h".

Masked IEEE-CIS columns (C, D, M, V, id_) have no published meaning, so their
reasons say so honestly instead of inventing one.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from fraudguard.models.bundle import FraudModel

Formatter = Callable[[object], str]
WEEKDAYS = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
ENTITY = {"card": "this card", "email": "this email domain", "device": "this device"}
VELOCITY = re.compile(r"^vel_(\w+?)_(\w+)_(cnt|amt)$")
MASKED = [
    (re.compile(r"^C\d+$"), "count feature"),
    (re.compile(r"^D\d+$"), "time-gap feature"),
    (re.compile(r"^M\d+$"), "match check"),
    (re.compile(r"^V\d+$"), "Vesta risk feature"),
    (re.compile(r"^id_\d+$"), "identity field"),
]


def _num(v: object) -> float:
    return float(v)  # type: ignore[arg-type]


def _duration(seconds: float) -> str:
    for unit, size in (("day", 86_400), ("hour", 3_600), ("minute", 60)):
        if seconds >= size:
            n = seconds / size
            return f"{n:.0f} {unit}{'s' if round(n) != 1 else ''}"
    return f"{seconds:.0f} seconds"


def _plural(n: float, word: str) -> str:
    suffix = "" if round(n) == 1 else ("es" if word.endswith("s") else "s")
    return f"{n:.0f} {word}{suffix}"


def _cents(c: float) -> str:
    """Round prices end in .00 or .x0; three-decimal cents usually come from a currency conversion."""
    if c == 0:
        return "round amount (no cents)"
    if round(c, 2) != round(c, 3):
        return f"amount ends in {f'{c:.3f}'[1:]}, a fraction of a cent (typical of currency conversion)"
    return f"amount ends in .{round(c * 100):02d}"


# feature -> (label used when the value is missing, formatter for a present value)
TEMPLATES: dict[str, tuple[str, Formatter]] = {
    "TransactionAmt": ("amount", lambda v: f"amount of ${_num(v):,.2f}"),
    "amt_log": ("amount", lambda v: f"amount of ${math.expm1(_num(v)):,.2f}"),
    "amt_cents": ("cents", lambda v: _cents(_num(v))),
    "D1": (
        "card age",
        lambda v: f"card first seen {_plural(_num(v), 'day')} ago" if _num(v) else "card first seen today",
    ),
    "hour": ("time of day", lambda v: f"made at {int(_num(v)):02d}:{int(_num(v) % 1 * 60):02d}"),
    "weekday": ("weekday", lambda v: f"made on a {WEEKDAYS.get(int(_num(v)), '?')}"),
    "ProductCD": ("product code", lambda v: f"product code {v}"),
    "card4": ("card network", lambda v: f"{v} card"),
    "card6": ("card type", lambda v: f"{v} card"),
    "addr1": ("billing region", lambda v: f"billing region {_num(v):.0f}"),
    "addr2": ("billing country", lambda v: f"billing country {_num(v):.0f}"),
    "P_emaildomain": ("purchaser email domain", lambda v: f"purchaser email at {v}"),
    "R_emaildomain": ("recipient email domain", lambda v: f"recipient email at {v}"),
    "DeviceType": ("device type", lambda v: f"{v} device"),
    "DeviceInfo": ("device information", lambda v: f"device {v}"),
    "has_identity": ("identity data", lambda v: "identity data present" if v else "no identity data sent"),
    "card_prior_txn": ("card history", lambda v: f"card has {_plural(_num(v), 'earlier transaction')}"),
    "card_is_new": (
        "card history",
        lambda v: "first transaction seen on this card" if v else "card seen before",
    ),
    "card_secs_since_last": (
        "time since the card's last transaction",
        lambda v: f"{_duration(_num(v))} since this card's previous transaction",
    ),
    "card_amt_mean": ("card's usual amount", lambda v: f"card usually spends ${_num(v):,.2f}"),
    "card_amt_z": (
        "comparison with the card's usual amount",
        lambda v: f"amount is {_num(v):+.1f} standard deviations from this card's usual",
    ),
    "card_amt_ratio": (
        "comparison with the card's usual amount",
        lambda v: f"amount is {_num(v):.1f}x this card's average",
    ),
    "card_hour_dev": (
        "card's usual time of day",
        lambda v: f"{_num(v):.1f} hours away from this card's usual time of day",
    ),
    "graph_card_n_devices": ("device history", lambda v: f"card used on {_plural(_num(v), 'device')} before"),
    "graph_card_n_addresses": (
        "address history",
        lambda v: f"card used with {_plural(_num(v), 'billing address')}",
    ),
    "graph_card_n_emails": ("email history", lambda v: f"card used with {_plural(_num(v), 'email domain')}"),
    "graph_device_n_cards_recent": (
        "device information",
        lambda v: f"{_plural(_num(v), 'other card')} used this device in the last 30 days",
    ),
    "graph_address_n_cards_recent": (
        "billing address",
        lambda v: f"{_plural(_num(v), 'other card')} used this billing address in the last 30 days",
    ),
    "graph_card_known_fraud": (
        "card fraud history",
        lambda v: f"card has {_plural(_num(v), 'confirmed fraud chargeback')}",
    ),
    "graph_device_fraud_cards": (
        "device information",
        lambda v: f"{_plural(_num(v), 'card')} with confirmed fraud used this device",
    ),
    "graph_address_fraud_cards": (
        "billing address",
        lambda v: f"{_plural(_num(v), 'card')} with confirmed fraud used this billing address",
    ),
    "graph_2hop_cards": (
        "linked cards",
        lambda v: f"linked to {_plural(_num(v), 'other card')} through shared devices or addresses",
    ),
    "graph_2hop_fraud_cards": (
        "linked cards",
        lambda v: f"linked to {_plural(_num(v), 'confirmed-fraud card')} through shared devices or addresses",
    ),
}


def describe(feature: str, value: object) -> str:
    """Plain-language description of one feature value."""
    missing = value is None or (isinstance(value, float) and math.isnan(value))
    if feature in TEMPLATES:
        label, fmt = TEMPLATES[feature]
        return f"{label} is missing" if missing else fmt(value)
    if m := VELOCITY.match(feature):
        alias, window, kind = m.groups()
        who = ENTITY.get(alias, f"this {alias}")
        if missing:
            return f"no {alias} recorded for a {window} activity check"
        if kind == "cnt":
            n = _num(value)
            what = "no earlier transactions" if n == 0 else _plural(n, "earlier transaction")
            return f"{what} on {who} in the last {window}"
        return f"${_num(value):,.0f} spent on {who} in the last {window}"
    for pattern, kind_label in MASKED:
        if pattern.match(feature):
            shown = (
                "missing" if missing else (f"{_num(value):g}" if isinstance(value, int | float) else value)
            )
            return f"{kind_label} {feature} is {shown} (meaning masked by the data provider)"
    return f"{feature} is {'missing' if missing else value}"


@dataclass(frozen=True)
class Reason:
    feature: str
    value: object
    contribution: float  # log-odds added by this feature
    text: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def reasons_for(
    contrib: np.ndarray, values: dict[str, object], feature_names: list[str], k: int
) -> list[Reason]:
    """Up to `k` risk-raising reasons for one transaction, largest contribution first."""
    order = [j for j in np.argsort(-contrib)[:k] if contrib[j] > 0]
    return [
        Reason(
            feature_names[j],
            values.get(feature_names[j]),
            round(float(contrib[j]), 4),
            describe(feature_names[j], values.get(feature_names[j])),
        )
        for j in order
    ]


def top_reasons(
    contrib: np.ndarray, df: pl.DataFrame, feature_names: list[str], k: int
) -> list[list[Reason]]:
    """Up to `k` risk-raising reasons per row of `df`."""
    return [reasons_for(contrib[i], row, feature_names, k) for i, row in enumerate(df.iter_rows(named=True))]


def explain(model: FraudModel, df: pl.DataFrame, k: int = 3) -> list[list[Reason]]:
    contrib, _ = model.contributions(df)
    return top_reasons(contrib, df, model.feature_names, k)


def global_importance(
    contrib: np.ndarray, feature_names: list[str], top: int = 25
) -> list[tuple[str, float]]:
    """Mean |SHAP| per feature, largest first."""
    mean_abs = np.abs(contrib).mean(axis=0)
    order = np.argsort(-mean_abs)[:top]
    return [(feature_names[j], round(float(mean_abs[j]), 5)) for j in order]
