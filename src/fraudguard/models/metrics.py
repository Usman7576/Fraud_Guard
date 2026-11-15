"""Evaluation metrics for imbalanced fraud scoring.

ROC-AUC flatters models when only 3.5% of rows are fraud, so the headline
metrics are PR-AUC (average precision) and recall at a fixed false-positive
rate: "if we may bother 1% of genuine customers, what share of fraud do we
catch?" Calibration metrics apply only to models that output probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


@dataclass(frozen=True)
class Metrics:
    n: int
    fraud_rate: float
    pr_auc: float
    roc_auc: float
    recall_at_fpr: float
    precision_at_fpr: float
    threshold_at_fpr: float
    target_fpr: float
    brier: float | None = None
    log_loss: float | None = None

    def as_dict(self) -> dict[str, float]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def threshold_at_fpr(y: np.ndarray, score: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold t such that flagging `score > t` keeps FPR at or below target.

    Conservative with ties: if many genuine rows share a score, none of them are
    flagged unless all of them fit within the budget.
    """
    negatives = np.sort(score[y == 0])[::-1]
    if negatives.size == 0:
        raise ValueError("need at least one genuine transaction to compute FPR")
    budget = int(np.floor(target_fpr * negatives.size))
    return float(negatives[budget]) if budget < negatives.size else float(negatives[-1] - 1)


def recall_precision_at_fpr(
    y: np.ndarray, score: np.ndarray, target_fpr: float
) -> tuple[float, float, float]:
    t = threshold_at_fpr(y, score, target_fpr)
    flagged = score > t
    tp = int((flagged & (y == 1)).sum())
    recall = tp / max(int((y == 1).sum()), 1)
    precision = tp / int(flagged.sum()) if flagged.any() else 0.0
    return recall, precision, t


def evaluate(
    y: np.ndarray, score: np.ndarray, target_fpr: float = 0.01, is_probability: bool = True
) -> Metrics:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if np.unique(y).size < 2:
        raise ValueError("evaluation needs both fraud and genuine transactions")
    recall, precision, t = recall_precision_at_fpr(y, score, target_fpr)
    probs = np.clip(score, 1e-7, 1 - 1e-7) if is_probability else None
    return Metrics(
        n=int(y.size),
        fraud_rate=round(float(y.mean()), 5),
        pr_auc=round(float(average_precision_score(y, score)), 5),
        roc_auc=round(float(roc_auc_score(y, score)), 5),
        recall_at_fpr=round(recall, 5),
        precision_at_fpr=round(precision, 5),
        threshold_at_fpr=round(t, 6),
        target_fpr=target_fpr,
        brier=round(float(brier_score_loss(y, probs)), 6) if probs is not None else None,
        log_loss=round(float(log_loss(y, probs)), 6) if probs is not None else None,
    )
