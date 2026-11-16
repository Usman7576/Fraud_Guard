from __future__ import annotations

import numpy as np
import pytest

from fraudguard.config import DecisionConfig
from fraudguard.decision import analysis
from fraudguard.decision.policy import (
    Action,
    Policy,
    adjust_odds,
    apply_review_capacity,
    decide,
    decline_threshold,
    realised_cost,
)

POLICY = Policy(false_decline_cost=15.0, review_cost=3.0, review_capacity_per_day=100)


def test_decision_is_amount_aware() -> None:
    p = np.array([0.2, 0.2, 0.2])
    amount = np.array([5.0, 40.0, 2000.0])
    # Expected costs (approve, review, decline) at p = 0.2:
    #   $5:    $1,   $3, $12 -> approve
    #   $40:   $8,   $3, $12 -> review
    #   $2000: $400, $3, $12 -> review (an analyst is cheaper than declining a likely-genuine customer)
    assert decide(p, amount, np.zeros(3), POLICY).tolist() == [Action.APPROVE, Action.REVIEW, Action.REVIEW]
    sure = np.array([0.95])
    assert decide(sure, np.array([2000.0]), np.zeros(1), POLICY).tolist() == [Action.DECLINE]


def test_decline_threshold_matches_the_decision_rule() -> None:
    no_review = Policy(15.0, review_cost=1e9, review_capacity_per_day=0)
    for amount in (25.0, 300.0, 4000.0):
        t = decline_threshold(amount, no_review)
        below, above = decide(np.array([t * 0.99, t * 1.01]), np.full(2, amount), np.zeros(2), no_review)
        assert (below, above) == (Action.APPROVE, Action.DECLINE)


def test_odds_multiplier() -> None:
    p = np.array([0.0, 0.2, 0.5, 1.0])
    assert np.allclose(adjust_odds(p, 1.0), p)
    assert adjust_odds(np.array([0.5]), 3.0)[0] == pytest.approx(0.75)  # odds 1 -> 3
    assert adjust_odds(p, 4.0)[[0, 3]].tolist() == [0.0, 1.0]


def test_review_capacity_keeps_the_highest_priority_each_day() -> None:
    actions = np.full(6, Action.REVIEW)
    priority = np.array([1.0, 5.0, 3.0, 2.0, 9.0, 4.0])
    day = np.array([0, 0, 0, 1, 1, 1])
    fallback = np.full(6, Action.APPROVE)
    out = apply_review_capacity(actions, priority, day, 2, fallback)
    assert out.tolist() == [
        Action.APPROVE,
        Action.REVIEW,
        Action.REVIEW,
        Action.APPROVE,
        Action.REVIEW,
        Action.REVIEW,
    ]
    assert (apply_review_capacity(actions, priority, day, 0, fallback) == Action.APPROVE).all()


def test_capacity_overflow_falls_back_to_cheaper_of_approve_and_decline() -> None:
    p = np.array([0.6, 0.3, 0.3])
    amount = np.array([100.0, 100.0, 200.0])
    # Without review: decline costs $6, $10.5, $10.5. Review ($3) saves $3, $7.5, $7.5.
    out = decide(p, amount, np.zeros(3), Policy(15.0, 3.0, review_capacity_per_day=1))
    assert out.tolist() == [Action.DECLINE, Action.REVIEW, Action.DECLINE]  # tie -> earlier row


def test_realised_cost() -> None:
    actions = np.array([Action.APPROVE, Action.APPROVE, Action.DECLINE, Action.DECLINE, Action.REVIEW])
    y = np.array([1, 0, 1, 0, 1])
    amount = np.array([200.0, 50.0, 80.0, 30.0, 60.0])
    c = realised_cost(actions, y, amount, POLICY)
    assert (c.fraud_loss, c.false_decline_cost, c.review_cost, c.total) == (200.0, 15.0, 3.0, 218.0)
    assert c.per_10k == pytest.approx(218.0 / 5 * 10_000)
    assert c.fraud_caught == pytest.approx(2 / 3, abs=1e-4)
    assert c.fraud_amount_caught == pytest.approx(140 / 340, abs=1e-4)
    assert c.false_decline_rate == 0.5


def _scenario(seed: int = 0, n: int = 4000) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    p = rng.beta(0.3, 8, n)
    y = (rng.random(n) < p).astype(int)
    amount = rng.lognormal(4, 1, n)
    day = np.repeat(np.arange(n // 100), 100)
    return p, y, amount, day


def test_calibrated_probabilities_choose_a_multiplier_near_one() -> None:
    p, y, amount, day = _scenario()
    cfg = DecisionConfig()
    base, grid = analysis.base_policy(cfg), analysis.multiplier_grid(cfg)
    choice = analysis.best_policy(p, y, amount, day, base=base, grid=grid)
    assert 0.5 <= choice.policy.risk_multiplier <= 2.0
    assert len(choice.curve) == grid.size
    # Underconfident scores (true risk is 4x the odds) need a higher multiplier.
    under = analysis.best_policy(adjust_odds(p, 0.25), y, amount, day, base=base, grid=grid)
    assert under.policy.risk_multiplier > 2.0


def test_model_beats_rules_and_approve_all_on_an_informative_score() -> None:
    p, y, amount, day = _scenario()
    rules_score = np.where(p > np.quantile(p, 0.9), 2.0, 0.0) + (amount > 300)
    cfg = DecisionConfig()
    base = analysis.base_policy(cfg)
    model = analysis.best_policy(p, y, amount, day, base=base, grid=analysis.multiplier_grid(cfg))
    rules = analysis.best_rules_policy(rules_score, y, amount, day, base)
    assert model.cost.total < rules.cost.total < analysis.approve_all_cost(y, amount, base).total
    two_factors = cfg.model_copy(update={"sensitivity_factors": [0.5, 2.0]})
    rows = analysis.sensitivity(p, rules_score, y, amount, day, cfg=two_factors)
    assert len(rows) == 4
    assert all(r.saving_vs_rules > 0 for r in rows)


def test_multiplier_grid_always_contains_one() -> None:
    grid = analysis.multiplier_grid(
        DecisionConfig.model_validate({"risk_multipliers": {"min": 0.3, "max": 3, "n": 4}})
    )
    assert 1.0 in grid.tolist()
    with pytest.raises(ValueError, match=r"include 1\.0"):
        DecisionConfig.model_validate({"risk_multipliers": {"min": 2, "max": 3, "n": 4}})
