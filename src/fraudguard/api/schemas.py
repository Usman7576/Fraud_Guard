"""Request and response models (they also generate the OpenAPI docs at /docs)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Masked IEEE-CIS columns may be sent as extra fields: C1.., D1.., M1.., V1.., id_01..
EXTRA_FIELD = re.compile(r"^(C\d+|D\d+|M\d+|V\d+|id_\d+)$")
IDENTITY_FIELDS = ("DeviceType", "DeviceInfo")


class Transaction(BaseModel):
    """One card-not-present payment, in IEEE-CIS column names."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "TransactionID": 3_600_001,
                "TransactionDT": 15_811_200,
                "TransactionAmt": 249.0,
                "ProductCD": "W",
                "card1": 13926,
                "card4": "visa",
                "card6": "debit",
                "addr1": 315.0,
                "addr2": 87.0,
                "P_emaildomain": "gmail.com",
                "D1": 14.0,
                "C1": 1.0,
                "DeviceType": "mobile",
                "DeviceInfo": "iOS Device",
            }
        },
    )

    TransactionID: int = Field(ge=0)
    TransactionDT: int = Field(ge=0, description="Seconds since the dataset's reference date.")
    TransactionAmt: float = Field(gt=0, le=1_000_000)
    ProductCD: str | None = None
    card1: int | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None
    has_identity: bool | None = Field(
        default=None, description="Defaults to whether any device or id_ field is present."
    )

    @model_validator(mode="after")
    def _check_extras(self) -> Transaction:
        for name, value in (self.model_extra or {}).items():
            if not EXTRA_FIELD.match(name):
                raise ValueError(f"unknown field {name!r}")
            if value is not None and not isinstance(value, int | float | str):
                raise ValueError(f"field {name!r} must be a number, a string or null")
        return self

    def row(self) -> dict[str, Any]:
        data = self.model_dump()
        if data["has_identity"] is None:
            data["has_identity"] = any(data.get(f) is not None for f in IDENTITY_FIELDS) or any(
                k.startswith("id_") and v is not None for k, v in data.items()
            )
        return data


class PreviewTransaction(Transaction):
    """A what-if transaction. TransactionID and TransactionDT default to "a new one, now"."""

    TransactionID: int = Field(default=0, ge=0)
    TransactionDT: int | None = Field(default=None, ge=0)  # type: ignore[assignment]


# The demo's card: common issuer and billing values, with a card1 that never occurs in the data, so
# every demo payment is a new card's first and no real card's history is shown.
DEMO_CARD: dict[str, Any] = {
    "card1": 9501,
    "card2": 321.0,
    "card3": 150.0,
    "card5": 226.0,
    "addr1": 330.0,
    "addr2": 87.0,
}


class DemoTransaction(BaseModel):
    """The landing page's live demo: a few plain fields, scored as a new card's payment, now."""

    model_config = ConfigDict(extra="forbid")

    TransactionAmt: float = Field(gt=0, le=10_000)
    ProductCD: Literal["W", "C", "R", "H", "S"] = "W"
    card4: Literal["visa", "mastercard", "american express", "discover"] = "visa"
    card6: Literal["debit", "credit"] = "debit"
    P_emaildomain: (
        Literal["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com", "protonmail.com"]
        | None
    ) = None
    DeviceType: Literal["desktop", "mobile"] | None = None
    card_age_days: int = Field(default=0, ge=0, le=640, description="Days since the card's first use (D1).")

    def row(self) -> dict[str, Any]:
        fields = self.model_dump(exclude={"card_age_days"})
        return PreviewTransaction(**DEMO_CARD, **fields, D1=float(self.card_age_days)).row()


class DemoResponse(BaseModel):
    probability: float
    decision: Literal["approve", "review", "decline"]
    reasons: list[ReasonOut]
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class ReasonOut(BaseModel):
    feature: str
    value: Any
    contribution: float = Field(description="Log-odds added to the fraud score by this feature.")
    text: str


class ScoreResponse(BaseModel):
    transaction_id: int
    probability: float = Field(description="Calibrated probability that the transaction is fraud.")
    decision: Literal["approve", "review", "decline"]
    reasons: list[ReasonOut]
    model_version: str
    latency_ms: float
    case_opened: bool


class PreviewResponse(ScoreResponse):
    transaction_dt: int
    features: dict[str, Any]
    contributions: list[tuple[str, float]]


class BatchResponse(BaseModel):
    results: list[ScoreResponse]


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    role: str
    expires_in: int


class Me(BaseModel):
    username: str
    role: str


class Note(BaseModel):
    author: str
    text: str
    at: datetime


class CaseSummary(BaseModel):
    txn_id: int
    status: Literal["open", "in_review", "closed"]
    probability: float
    amount: float
    decision: str
    reasons: list[ReasonOut]
    assignee: str | None = None
    verdict: Literal["fraud", "genuine"] | None = None
    created_at: datetime
    sla_due: datetime | None = None


class CaseDetail(CaseSummary):
    card_uid: str
    dt: int
    features: dict[str, Any]
    contributions: list[tuple[str, float]]
    notes: list[Note]
    neighbourhood: dict[str, list[dict[str, Any]]]


class CasePage(BaseModel):
    items: list[CaseSummary]
    total: int
    page: int
    page_size: int


class CaseUpdate(BaseModel):
    assignee: str | None = None
    status: Literal["open", "in_review", "closed"] | None = None
    note: str | None = Field(default=None, max_length=5000)
    verdict: Literal["fraud", "genuine"] | None = None


class KpiWeek(BaseModel):
    week: int
    week_start: datetime
    volume: int
    decline_rate: float
    review_rate: float
    labelled_fraud: int
    recall: float | None
    cost: float
    cost_per_10k: float
    p95_latency_ms: float | None


class PolicyModel(BaseModel):
    false_decline_cost: float = Field(ge=0)
    review_cost: float = Field(ge=0)
    review_capacity_per_day: int = Field(ge=0)
    risk_multiplier: float = Field(gt=0)


Scope = Literal["score", "labels", "promote", "demo"]
USERNAME = r"^[a-z0-9][a-z0-9_.-]{2,31}$"


class UserOut(BaseModel):
    username: str
    role: Literal["analyst", "admin"]
    active: bool
    created_at: datetime | None = None


class UserCreate(BaseModel):
    username: str = Field(pattern=USERNAME, description="3-32 characters: a-z, 0-9, _ . -")
    role: Literal["analyst", "admin"] = "analyst"
    password: str = Field(min_length=10, max_length=128)


class UserUpdate(BaseModel):
    role: Literal["analyst", "admin"] | None = None
    active: bool | None = None
    password: str | None = Field(default=None, min_length=10, max_length=128)


class ApiKeyOut(BaseModel):
    id: str
    name: str
    scopes: list[str]
    active: bool
    created_at: datetime | None = None


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_. -]*$")
    scopes: list[Scope] = Field(min_length=1)


class ApiKeyCreated(ApiKeyOut):
    key: str = Field(description="Shown once. Only its hash is stored.")


class AuditEntry(BaseModel):
    id: str
    actor: str
    action: str
    target: str | None = None
    before: Any = None
    after: Any = None
    ts: datetime


class AuditPage(BaseModel):
    entries: list[AuditEntry]
    next_before: datetime | None = Field(description="Pass as `before` for the next (older) page.")


class CostOut(BaseModel):
    n: int
    total: float
    per_10k: float
    fraud_loss: float
    false_decline_cost: float
    review_cost: float
    decline_rate: float
    review_rate: float
    fraud_caught: float
    fraud_amount_caught: float
    false_decline_rate: float


class CurvePointOut(BaseModel):
    false_decline_cost: float
    per_10k: float
    decline_rate: float
    review_rate: float
    fraud_caught: float


class ThresholdOut(BaseModel):
    amount: float
    act_from: float = Field(description="Lowest calibrated probability that is reviewed or declined.")
    decline_from: float = Field(description="Lowest calibrated probability that is declined.")


class SimulateResponse(BaseModel):
    split: Literal["validation"] = "validation"
    model_version: str
    n_rows: int
    n_fraud: int
    proposed: CostOut
    current: CostOut
    curve: list[CurvePointOut]
    thresholds: list[ThresholdOut]


class ModelVersion(BaseModel):
    version: str
    aliases: list[str]
    run_id: str | None
    created_at: datetime | None
    metrics: dict[str, float]
    serving: bool


class Health(BaseModel):
    status: str
    model_version: str
    transactions_in_memory: int
    latest_transaction_dt: int | None


class LabelIn(BaseModel):
    txn_id: int = Field(ge=0)
    label: Literal[0, 1]
    known_dt: int = Field(ge=0, description="TransactionDT-scale time at which the label became known.")


class LabelsRequest(BaseModel):
    labels: list[LabelIn] = Field(min_length=1, max_length=5000)


class LabelsResponse(BaseModel):
    stored: int
    skipped_analyst: int
    n_unknown: int
    unknown: list[int] = Field(description="Up to 20 transaction IDs the API has not scored.")


class DriftWeek(BaseModel):
    week: int
    week_start: datetime | None = None
    week_start_dt: int | None = None
    n_rows: int
    n_features: int
    n_drifted: int
    drift_share: float
    threshold: float | None = Field(default=None, description="Drift share that triggers retraining.")
    top_drifted: list[str] = []
    report_file: str | None = None


class DriftColumn(BaseModel):
    column: str
    method: str
    value: float
    threshold: float
    drifted: bool


class DriftDetail(DriftWeek):
    columns: list[DriftColumn]


class RetrainingEvent(BaseModel):
    week: int
    week_start: datetime | None = None
    ts: datetime
    trigger: str
    outcome: Literal["promoted", "kept_champion", "skipped"]
    detail: str
    drift_share: float | None = None
    recall: float | None = None
    champion_version: str | None = None
    challenger_version: str | None = None
    comparison: dict[str, Any] | None = None
