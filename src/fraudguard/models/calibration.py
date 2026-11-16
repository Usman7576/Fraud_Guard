"""Isotonic calibration: make a score of 0.30 mean "about 30% of these are fraud".

The cost policy multiplies probabilities by amounts, so raw LightGBM scores are
not good enough: they rank well but are not probabilities. Isotonic regression
fits a monotone step function from score to observed fraud rate.

The calibrator is stored as its breakpoints (plain JSON) and applied with
linear interpolation, the same way scikit-learn's IsotonicRegression predicts,
so the scoring API needs nothing but NumPy.

To judge calibration honestly, `cross_fit` calibrates each contiguous time
block of the validation month with a calibrator fitted on the other blocks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class Calibrator:
    x: list[float]
    y: list[float]

    @classmethod
    def fit(cls, score: np.ndarray, label: np.ndarray) -> Calibrator:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        iso.fit(np.asarray(score, dtype=float), np.asarray(label, dtype=float))
        return cls([float(v) for v in iso.X_thresholds_], [float(v) for v in iso.y_thresholds_])

    def predict(self, score: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.interp(np.asarray(score, dtype=float), self.x, self.y)
        return out

    def to_dict(self) -> dict[str, list[float]]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Calibrator:
        return cls([float(v) for v in data["x"]], [float(v) for v in data["y"]])


def cross_fit(score: np.ndarray, label: np.ndarray, n_folds: int) -> np.ndarray:
    """Out-of-fold calibrated probabilities over contiguous blocks (rows in time order)."""
    out = np.empty(score.size, dtype=float)
    for block in np.array_split(np.arange(score.size), n_folds):
        mask = np.ones(score.size, dtype=bool)
        mask[block] = False
        out[block] = Calibrator.fit(score[mask], label[mask]).predict(score[block])
    return out


@dataclass(frozen=True)
class ReliabilityBin:
    mean_predicted: float
    fraud_rate: float
    count: int


def reliability(p: np.ndarray, label: np.ndarray, edges: list[float]) -> tuple[float, list[ReliabilityBin]]:
    """Expected calibration error, plus the bins for a reliability diagram.

    Bin edges are roughly log-spaced: with 3.5% fraud, most rows and most
    decisions sit at small probabilities, which equal-width or equal-count bins
    lump together.
    """
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    bins = []
    ece = 0.0
    for b in range(len(edges) - 1):
        idx = np.flatnonzero(which == b)
        if idx.size == 0:
            continue
        mean_p, rate = float(p[idx].mean()), float(label[idx].mean())
        ece += idx.size * abs(mean_p - rate)
        bins.append(ReliabilityBin(round(mean_p, 5), round(rate, 5), int(idx.size)))
    return round(ece / max(p.size, 1), 6), bins
