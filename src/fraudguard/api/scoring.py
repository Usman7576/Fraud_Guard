"""Live scoring: online features -> calibrated probability -> decision -> reasons.

One `Scorer` lives for the life of the API process. Scoring and recording a
transaction run without awaiting anything in between, so on FastAPI's single
event loop each request sees and updates the online state atomically.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from fraudguard.data.load import SECONDS_PER_DAY
from fraudguard.data.schema import AMOUNT, DT, ID
from fraudguard.decision.policy import Action, Policy, decide_one
from fraudguard.explain.reasons import Reason, reasons_for
from fraudguard.features.online import OnlineFeatureStore
from fraudguard.models.bundle import FraudModel

N_CASE_CONTRIBUTIONS = 20


@dataclass
class ScoreResult:
    txn_id: int
    dt: int
    amount: float
    card_uid: str
    probability: float
    decision: Action
    reasons: list[Reason]
    model_version: str
    latency_ms: float
    features: dict[str, Any] = field(default_factory=dict)
    contributions: list[tuple[str, float]] = field(default_factory=list)


class Scorer:
    def __init__(
        self, model: FraudModel, version: str, store: OnlineFeatureStore, policy: Policy, n_reasons: int = 3
    ) -> None:
        self.model, self.version, self.store, self.policy = model, version, store, policy
        self.n_reasons = n_reasons
        self.reviews_by_day: Counter[int] = Counter()

    def swap_model(self, model: FraudModel, version: str) -> None:
        self.model, self.version = model, version

    def score(self, txn: dict[str, Any], *, record: bool = True) -> ScoreResult:
        """Score a transaction. `record=False` is a what-if that leaves online state untouched."""
        start = time.perf_counter()
        model = self.model  # one consistent model even if an admin promotes another mid-request
        feats = self.store.features(txn, peek=not record)
        values = {**txn, **feats}
        x = model.vector([values])
        contrib_full = np.asarray(model.booster.predict(x, pred_contrib=True))[0]
        contrib, base = contrib_full[:-1], contrib_full[-1]
        raw = 1.0 / (1.0 + np.exp(-(contrib.sum() + base)))
        p = float(model.calibrator.predict(np.array([raw]))[0])
        day = int(txn[DT]) // SECONDS_PER_DAY
        action = decide_one(p, float(txn[AMOUNT]), self.policy, self.reviews_by_day[day])
        if record:
            if action == Action.REVIEW:
                self.reviews_by_day[day] += 1
            self.store.add(txn)
        reasons = reasons_for(contrib, values, model.feature_names, self.n_reasons)
        top = np.argsort(-np.abs(contrib))[:N_CASE_CONTRIBUTIONS]
        return ScoreResult(
            txn_id=int(txn[ID]),
            dt=int(txn[DT]),
            amount=float(txn[AMOUNT]),
            card_uid=str(feats["card_uid"]),
            probability=round(p, 6),
            decision=action,
            reasons=reasons,
            model_version=self.version,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            features={c: _plain(values.get(c)) for c in model.feature_names},
            contributions=[(model.feature_names[j], round(float(contrib[j]), 5)) for j in top],
        )


def _plain(v: object) -> object:
    """JSON/BSON-friendly value: NaN becomes None, numpy scalars become Python numbers."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def history_frame(base: pl.DataFrame, splits: list[str]) -> pl.DataFrame:
    """Rows of the base snapshot in the given splits, in time order, for warming the store."""
    return (
        base.filter(pl.col("split").is_in(splits))
        .sort(DT, ID)
        .drop("ts", "day", "split", "card_uid", strict=False)
    )
