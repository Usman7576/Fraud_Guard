"""Rules baseline: the hand-written limits a fraud team might start with.

Each rule compares one feature to a limit; the score is the sum of the weights
of the rules that fire. It is the benchmark every model must beat, and its
per-rule hit rates show which rules pull their weight.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from fraudguard.config import RuleConfig
from fraudguard.data.schema import AMOUNT, LABEL

_OPS: dict[str, Callable[[pl.Expr, float], pl.Expr]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Columns that only the rules need."""
    return df.with_columns(
        (pl.col("card_is_new") & (pl.col(AMOUNT) > 300)).cast(pl.Int8).alias("card_amt_new_high")
    )


@dataclass(frozen=True)
class RuleStats:
    name: str
    fired_share: float
    fraud_rate_when_fired: float


class RulesBaseline:
    def __init__(self, rules: list[RuleConfig]) -> None:
        self.rules = rules

    def fired(self, df: pl.DataFrame) -> pl.DataFrame:
        df = add_derived(df)
        missing = [r.feature for r in self.rules if r.feature not in df.columns]
        if missing:
            raise KeyError(f"rules reference unknown features: {missing}")
        return df.select(
            _OPS[r.op](pl.col(r.feature), r.value).fill_null(False).alias(r.name) for r in self.rules
        )

    def score(self, df: pl.DataFrame) -> np.ndarray:
        fired = self.fired(df)
        weights = np.array([r.weight for r in self.rules])
        scores: np.ndarray = fired.to_numpy().astype(float) @ weights
        return scores

    def rule_stats(self, df: pl.DataFrame) -> list[RuleStats]:
        fired = self.fired(df)
        y = df[LABEL].to_numpy()
        out = []
        for r in self.rules:
            mask = fired[r.name].to_numpy()
            out.append(
                RuleStats(
                    name=r.name,
                    fired_share=round(float(mask.mean()), 5),
                    fraud_rate_when_fired=round(float(y[mask].mean()), 5) if mask.any() else 0.0,
                )
            )
        return out
