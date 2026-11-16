"""Choosing and stress-testing the decision policy on labelled data.

- `best_policy` sweeps an odds multiplier over the calibrated probabilities. If
  calibration is good, the cheapest multiplier is close to 1.
- `best_rules_policy` gives the rules baseline the same chance: decline and
  review thresholds on its score, chosen to minimise cost on the same data.
- `sensitivity` repeats both under scaled cost assumptions, because the
  false-decline and review costs are guesses, not merchant data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from itertools import product

import numpy as np

from fraudguard.config import DecisionConfig
from fraudguard.decision.policy import (
    Action,
    CostBreakdown,
    Policy,
    apply_review_capacity,
    decide,
    realised_cost,
)

MAX_RULE_THRESHOLDS = 40


def base_policy(cfg: DecisionConfig) -> Policy:
    return Policy(cfg.false_decline_cost, cfg.review_cost, cfg.review_capacity_per_day)


def multiplier_grid(cfg: DecisionConfig) -> np.ndarray:
    g = cfg.risk_multipliers
    return np.unique(np.r_[np.geomspace(g.min, g.max, g.n), 1.0])


@dataclass
class PolicyChoice:
    policy: Policy
    cost: CostBreakdown
    curve: list[tuple[float, float]] = field(default_factory=list)  # (multiplier, cost per 10k)

    def as_dict(self) -> dict[str, object]:
        return {"policy": self.policy.as_dict(), "cost": self.cost.as_dict(), "curve": self.curve}


def best_policy(
    p: np.ndarray, y: np.ndarray, amount: np.ndarray, day: np.ndarray, *, base: Policy, grid: np.ndarray
) -> PolicyChoice:
    """Cheapest odds multiplier; ties go to the one closest to 1 (trust the calibration)."""
    results = []
    for m in grid:
        policy = replace(base, risk_multiplier=round(float(m), 4))
        results.append((policy, realised_cost(decide(p, amount, day, policy), y, amount, policy)))
    best_total = min(c.total for _, c in results)
    policy, cost = min(
        ((pol, c) for pol, c in results if c.total <= best_total + 1e-9),
        key=lambda pc: abs(np.log(pc[0].risk_multiplier)),
    )
    return PolicyChoice(policy, cost, [(pol.risk_multiplier, c.per_10k) for pol, c in results])


@dataclass
class RulesChoice:
    review_at: float | None  # rules score at or above which a case goes to review
    decline_at: float | None
    cost: CostBreakdown

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "cost": self.cost.as_dict()}


def rules_actions(
    score: np.ndarray, day: np.ndarray, review_at: float | None, decline_at: float | None, capacity: int
) -> np.ndarray:
    actions = np.full(score.shape, Action.APPROVE, dtype=int)
    if review_at is not None:
        actions[score >= review_at] = Action.REVIEW
    if decline_at is not None:
        actions[score >= decline_at] = Action.DECLINE
    fallback = np.full(score.shape, Action.APPROVE, dtype=int)
    return apply_review_capacity(actions, score, day, capacity, fallback)


def best_rules_policy(
    score: np.ndarray, y: np.ndarray, amount: np.ndarray, day: np.ndarray, base: Policy
) -> RulesChoice:
    """Best review and decline thresholds for the rules score (None = never)."""
    levels = np.unique(score[score > score.min()]) if score.size else np.array([])
    if levels.size > MAX_RULE_THRESHOLDS:
        levels = np.unique(np.quantile(levels, np.linspace(0, 1, MAX_RULE_THRESHOLDS)))
    candidates: list[float | None] = [None, *(float(v) for v in levels)]
    best: RulesChoice | None = None
    for review_at, decline_at in product(candidates, candidates):
        if review_at is not None and decline_at is not None and review_at >= decline_at:
            continue
        actions = rules_actions(score, day, review_at, decline_at, base.review_capacity_per_day)
        cost = realised_cost(actions, y, amount, base)
        if best is None or cost.total < best.cost.total:
            best = RulesChoice(review_at, decline_at, cost)
    assert best is not None
    return best


def approve_all_cost(y: np.ndarray, amount: np.ndarray, base: Policy) -> CostBreakdown:
    return realised_cost(np.full(y.shape, Action.APPROVE, dtype=int), y, amount, base)


@dataclass
class SensitivityRow:
    false_decline_cost: float
    review_cost: float
    model_multiplier: float
    model_per_10k: float
    rules_per_10k: float
    approve_all_per_10k: float
    saving_vs_rules: float  # share of the rules baseline's cost removed by the model
    model_decline_rate: float
    model_review_rate: float
    model_fraud_caught: float


def sensitivity(
    p: np.ndarray,
    rules_score: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    day: np.ndarray,
    *,
    cfg: DecisionConfig,
) -> list[SensitivityRow]:
    grid, base = multiplier_grid(cfg), base_policy(cfg)
    rows = []
    for f_fd, f_r in product(cfg.sensitivity_factors, cfg.sensitivity_factors):
        scaled = replace(
            base, false_decline_cost=base.false_decline_cost * f_fd, review_cost=base.review_cost * f_r
        )
        model = best_policy(p, y, amount, day, base=scaled, grid=grid)
        rules = best_rules_policy(rules_score, y, amount, day, scaled)
        rows.append(
            SensitivityRow(
                false_decline_cost=scaled.false_decline_cost,
                review_cost=scaled.review_cost,
                model_multiplier=model.policy.risk_multiplier,
                model_per_10k=model.cost.per_10k,
                rules_per_10k=rules.cost.per_10k,
                approve_all_per_10k=approve_all_cost(y, amount, scaled).per_10k,
                saving_vs_rules=round(1 - model.cost.total / rules.cost.total, 4)
                if rules.cost.total
                else 0.0,
                model_decline_rate=model.cost.decline_rate,
                model_review_rate=model.cost.review_rate,
                model_fraud_caught=model.cost.fraud_caught,
            )
        )
    return rows
