"""Cost explorer: what a proposed policy would have cost on the validation month.

The champion scores the validation month once (cached by the API). Each proposal then only
re-runs the decision layer, so moving a slider costs a few milliseconds. The validation month is
the one the policy was tuned on; the test month stays untouched for the final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from fraudguard.config import DataConfig
from fraudguard.data import snapshot
from fraudguard.data.load import SECONDS_PER_DAY
from fraudguard.data.schema import AMOUNT, DT, LABEL
from fraudguard.data.split import SPLIT, Split
from fraudguard.decision.policy import CostBreakdown, Policy, decide, realised_cost
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.models.bundle import FraudModel

THRESHOLD_AMOUNTS = (10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0)
# From $5: at $0 declining everything is free, which says nothing about the policy.
CURVE_FALSE_DECLINE_COSTS = tuple(float(v) for v in range(5, 101, 5))


@dataclass(frozen=True)
class Scored:
    """The validation month as the champion scored it."""

    p: np.ndarray
    y: np.ndarray
    amount: np.ndarray
    day: np.ndarray

    @property
    def n_fraud(self) -> int:
        return int(self.y.sum())


def score_validation(data_cfg: DataConfig, model: FraudModel) -> Scored:
    features = snapshot.load_snapshot(data_cfg.paths.snapshot_dir, feature_pipeline.STAGE, "features")
    valid = features.filter(pl.col(SPLIT) == Split.VALIDATION.value)
    if valid.height == 0:
        raise ValueError("the feature snapshot has no validation month")
    return Scored(
        p=model.predict_proba(valid),
        y=valid[LABEL].to_numpy().astype(bool),
        amount=valid[AMOUNT].cast(pl.Float64).to_numpy(),
        day=(valid[DT] // SECONDS_PER_DAY).to_numpy(),
    )


def simulate(scored: Scored, policy: Policy) -> CostBreakdown:
    return realised_cost(decide(scored.p, scored.amount, scored.day, policy), scored.y, scored.amount, policy)


@dataclass(frozen=True)
class CurvePoint:
    false_decline_cost: float
    per_10k: float
    decline_rate: float
    review_rate: float
    fraud_caught: float


def false_decline_curve(scored: Scored, policy: Policy) -> list[CurvePoint]:
    """Cost per 10,000 payments as the false-decline cost varies, everything else as proposed.

    The cost is counted at each point's own false-decline cost, so the curve shows how sensitive
    the bill is to that one assumption."""
    out = []
    for fd in CURVE_FALSE_DECLINE_COSTS:
        c = simulate(scored, replace(policy, false_decline_cost=fd))
        out.append(CurvePoint(fd, c.per_10k, c.decline_rate, c.review_rate, c.fraud_caught))
    return out


def _unadjust(q: float, multiplier: float) -> float:
    """The calibrated probability whose odds-adjusted value is q (inverse of `adjust_odds`)."""
    if q >= 1.0:
        return 1.0
    q = max(q, 0.0)
    return q / (multiplier * (1.0 - q) + q)


@dataclass(frozen=True)
class Thresholds:
    amount: float
    act_from: float  # lowest probability that is no longer approved (review or decline)
    decline_from: float  # lowest probability that is declined


def implied_thresholds(policy: Policy, amounts: tuple[float, ...] = THRESHOLD_AMOUNTS) -> list[Thresholds]:
    """The probability cut-offs a policy implies at each amount, before the review capacity.

    With adjusted probability q: approving costs q*amount, reviewing costs R, declining costs
    (1-q)*F. Acting beats approving once q > min(R/amount, F/(amount+F)); declining beats both once
    q > max(F/(amount+F), 1-R/F)."""
    r, f, m = policy.review_cost, policy.false_decline_cost, policy.risk_multiplier
    out = []
    for a in amounts:
        to_decline = f / (a + f) if a + f > 0 else 1.0
        act = min(r / a if a > 0 else 1.0, to_decline)
        decline = max(to_decline, 1.0 - r / f if f > 0 else 0.0)
        out.append(Thresholds(a, round(_unadjust(act, m), 6), round(_unadjust(decline, m), 6)))
    return out
