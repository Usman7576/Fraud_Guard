"""Cost model and the amount-aware decision policy.

Every transaction gets one of three actions, each with a cost:

- approve: costs the transaction amount if it was fraud (the merchant pays the chargeback);
- decline: costs a fixed `false_decline_cost` if the customer was genuine;
- review:  costs a fixed `review_cost`; the analyst is assumed to get it right.

With a calibrated fraud probability p and amount A, the expected costs are
p*A, (1-p)*C_FD and C_R. The policy picks the cheapest. That rule is
amount-aware by construction: a $2,000 transaction is declined at a much
lower probability than a $20 one, and the review band is wherever the analyst
is cheaper than either mistake.

Analysts have limited time. When more transactions want review on a day than
`review_capacity_per_day`, the ones where review saves the most go first and
the rest fall back to the cheaper of approve and decline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    APPROVE = 0
    REVIEW = 1
    DECLINE = 2


@dataclass(frozen=True)
class Policy:
    false_decline_cost: float
    review_cost: float
    review_capacity_per_day: int
    risk_multiplier: float = 1.0

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def adjust_odds(p: np.ndarray, multiplier: float) -> np.ndarray:
    """Scale the fraud odds by `multiplier`; 1.0 leaves probabilities unchanged."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return multiplier * p / (multiplier * p + (1.0 - p))


def expected_costs(p: np.ndarray, amount: np.ndarray, policy: Policy) -> np.ndarray:
    """(n, 3) expected cost of approve, review and decline for each transaction."""
    q = adjust_odds(p, policy.risk_multiplier)
    amount = np.asarray(amount, dtype=float)
    return np.column_stack(
        [q * amount, np.full(q.shape, policy.review_cost), (1.0 - q) * policy.false_decline_cost]
    )


def apply_review_capacity(
    actions: np.ndarray, priority: np.ndarray, day: np.ndarray, capacity: int, fallback: np.ndarray
) -> np.ndarray:
    """Keep at most `capacity` reviews per day, highest `priority` first; the rest take `fallback`."""
    actions = actions.copy()
    idx = np.flatnonzero(actions == Action.REVIEW)
    if idx.size == 0:
        return actions
    order = idx[np.lexsort((-priority[idx], day[idx]))]  # by day, then priority descending
    days = day[order]
    first = np.r_[0, np.flatnonzero(np.diff(days)) + 1]  # start of each day's block
    rank = np.arange(order.size) - np.repeat(first, np.diff(np.r_[first, order.size]))
    over = order[rank >= capacity]
    actions[over] = fallback[over]
    return actions


def decide(p: np.ndarray, amount: np.ndarray, day: np.ndarray, policy: Policy) -> np.ndarray:
    """Cheapest action per transaction, subject to the daily review capacity."""
    costs = expected_costs(p, amount, policy)
    actions = np.argmin(costs, axis=1)
    no_review = np.where(costs[:, Action.APPROVE] <= costs[:, Action.DECLINE], Action.APPROVE, Action.DECLINE)
    saving = np.minimum(costs[:, Action.APPROVE], costs[:, Action.DECLINE]) - costs[:, Action.REVIEW]
    return apply_review_capacity(actions, saving, np.asarray(day), policy.review_capacity_per_day, no_review)


def decide_one(p: float, amount: float, policy: Policy, reviews_used_today: int) -> Action:
    """Live version of `decide`: reviews are first come, first served until the day's capacity is used.

    Offline evaluation ranks a whole day's review candidates by saving; a live
    system cannot see the rest of the day, so it fills the queue in arrival order."""
    costs = expected_costs(np.array([p]), np.array([amount]), policy)[0]
    best = Action(int(np.argmin(costs)))
    if best == Action.REVIEW and reviews_used_today >= policy.review_capacity_per_day:
        return Action.APPROVE if costs[Action.APPROVE] <= costs[Action.DECLINE] else Action.DECLINE
    return best


def decline_threshold(amount: float, policy: Policy) -> float:
    """Calibrated probability above which declining beats approving, for a given amount."""
    c_fd = policy.false_decline_cost
    q = c_fd / (amount + c_fd) if amount + c_fd > 0 else 1.0
    # Undo the odds multiplier: the policy compares adjusted probabilities.
    m = policy.risk_multiplier
    return float(q / (m * (1 - q) + q))


@dataclass(frozen=True)
class CostBreakdown:
    n: int
    total: float
    per_10k: float
    fraud_loss: float
    false_decline_cost: float
    review_cost: float
    decline_rate: float
    review_rate: float
    fraud_caught: float  # share of fraud transactions declined or reviewed
    fraud_amount_caught: float  # share of fraud dollars stopped
    false_decline_rate: float  # share of genuine customers declined

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def realised_cost(actions: np.ndarray, y: np.ndarray, amount: np.ndarray, policy: Policy) -> CostBreakdown:
    """What the actions actually cost, given the true labels."""
    y = np.asarray(y).astype(bool)
    amount = np.asarray(amount, dtype=float)
    approve, review, decline = (actions == a for a in Action)
    fraud_loss = float(amount[approve & y].sum())
    fd_cost = float((decline & ~y).sum() * policy.false_decline_cost)
    rv_cost = float(review.sum() * policy.review_cost)
    total = fraud_loss + fd_cost + rv_cost
    n = int(y.size)
    stopped = (review | decline) & y
    fraud_amount = float(amount[y].sum())
    return CostBreakdown(
        n=n,
        total=round(total, 2),
        per_10k=round(total / max(n, 1) * 10_000, 2),
        fraud_loss=round(fraud_loss, 2),
        false_decline_cost=round(fd_cost, 2),
        review_cost=round(rv_cost, 2),
        decline_rate=round(float(decline.mean()), 5),
        review_rate=round(float(review.mean()), 5),
        fraud_caught=round(float(stopped.sum() / max(int(y.sum()), 1)), 5),
        fraud_amount_caught=round(float(amount[stopped].sum() / fraud_amount) if fraud_amount else 0.0, 5),
        false_decline_rate=round(float((decline & ~y).sum() / max(int((~y).sum()), 1)), 5),
    )
