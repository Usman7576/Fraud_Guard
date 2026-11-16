"""Bootstrap confidence intervals.

Each resample draws whole clusters (days, by default) with replacement, and
every statistic is computed on the same resample. Differences between two
models computed this way are *paired*: both are scored on exactly the same
transactions, which removes the shared noise and gives a much tighter interval
for "is the model better than the baseline?" than comparing two separate CIs.

Percentile intervals. A resample can make a statistic undefined (no fraud
drawn, say); the statistic returns NaN, that draw is skipped for it, and the
number of valid draws is reported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Row indices of one resample -> every statistic of interest, computed on those rows.
Statistics = Callable[[np.ndarray], dict[str, float]]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    n_valid: int

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def as_dict(self) -> dict[str, float | int]:
        return {"estimate": self.estimate, "low": self.low, "high": self.high, "n_valid": self.n_valid}


def resamples(groups: np.ndarray | None, n: int, n_resamples: int, seed: int) -> list[np.ndarray]:
    """Row indices for each resample. `groups` (e.g. day per row) gives a cluster bootstrap."""
    rng = np.random.default_rng(seed)
    if groups is None:
        return [rng.integers(0, n, n) for _ in range(n_resamples)]
    order = np.argsort(groups, kind="stable")
    _, starts = np.unique(groups[order], return_index=True)
    members = np.split(order, starts[1:])
    out = []
    for _ in range(n_resamples):
        picked = rng.integers(0, len(members), len(members))
        out.append(np.concatenate([members[k] for k in picked]))
    return out


def bootstrap(
    stats: Statistics,
    n: int,
    *,
    groups: np.ndarray | None,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Interval]:
    point = stats(np.arange(n))
    draws = [stats(idx) for idx in resamples(groups, n, n_resamples, seed)]
    lo_q, hi_q = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100
    out = {}
    for name, estimate in point.items():
        values = np.array([d[name] for d in draws], dtype=float)
        valid = values[~np.isnan(values)]
        low, high = np.percentile(valid, [lo_q, hi_q]) if valid.size else (np.nan, np.nan)
        out[name] = Interval(
            round(float(estimate), 6), round(float(low), 6), round(float(high), 6), int(valid.size)
        )
    return out
