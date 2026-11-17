"""Champion-challenger: retrain on recent labelled data, promote only on clear evidence.

At simulated time `now`, labels are known for transactions older than
`now - label_delay`. That labelled history is cut by time into three blocks:

    [ ...... train (rolling window) ...... | calibration | evaluation ]  <- labelled up to here

- The challenger is LightGBM with the champion's tuned parameters, fitted on
  the train block (the newest data the champion never saw), calibrated with
  isotonic regression on the calibration block.
- Both models then score the evaluation block, which neither was fitted or
  calibrated on. A paired bootstrap over days measures the PR-AUC gain and the
  cost difference under the live policy.
- The challenger is promoted only if the lower bound of the PR-AUC gain is
  above zero *and* it is not more expensive. Otherwise the champion stays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from fraudguard.config import LightGBMConfig, RetrainingConfig
from fraudguard.data.load import SECONDS_PER_DAY
from fraudguard.data.schema import AMOUNT, DT, LABEL
from fraudguard.decision.policy import Action, Policy, decide
from fraudguard.evaluation.bootstrap import Interval, bootstrap
from fraudguard.models import lgbm
from fraudguard.models.bundle import FraudModel
from fraudguard.models.calibration import Calibrator


class NotEnoughDataError(ValueError):
    """The labelled window is too short, or a block has no fraud."""


@dataclass
class Windows:
    train_start: int
    calibration_start: int
    evaluation_start: int
    labelled_end: int


def windows(now_dt: int, label_delay_days: int, cfg: RetrainingConfig) -> Windows:
    end = now_dt - label_delay_days * SECONDS_PER_DAY
    eval_start = end - cfg.evaluation_days * SECONDS_PER_DAY
    calib_start = eval_start - cfg.calibration_days * SECONDS_PER_DAY
    return Windows(end - cfg.window_days * SECONDS_PER_DAY, calib_start, eval_start, end)


@dataclass
class Comparison:
    n_rows: int
    n_fraud: int
    pr_auc_champion: Interval
    pr_auc_challenger: Interval
    pr_auc_gain: Interval
    cost_saved_per_10k: Interval  # champion cost minus challenger cost: positive is better
    promote: bool
    reason: str

    def as_doc(self) -> dict[str, Any]:
        doc = asdict(self)
        for k in ("pr_auc_champion", "pr_auc_challenger", "pr_auc_gain", "cost_saved_per_10k"):
            doc[k] = getattr(self, k).as_dict()
        return doc


def _block(df: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    return df.filter((pl.col(DT) >= start) & (pl.col(DT) < end))


def _check(name: str, block: pl.DataFrame) -> None:
    n_fraud = int(block[LABEL].sum()) if block.height else 0
    if n_fraud in {0, block.height}:
        raise NotEnoughDataError(
            f"the {name} block has {block.height} rows and {n_fraud} fraud; need both classes"
        )


def train_challenger(
    features: pl.DataFrame, champion: FraudModel, w: Windows, lcfg: LightGBMConfig, seed: int
) -> FraudModel:
    if not champion.params or not champion.rounds:
        raise NotEnoughDataError("the champion has no stored parameters to retrain with")
    train = _block(features, w.train_start, w.calibration_start)
    calib = _block(features, w.calibration_start, w.evaluation_start)
    _check("training", train)
    _check("calibration", calib)
    cols = champion.feature_names
    cat_cols = lgbm.string_columns(train, cols)
    enc = lgbm.CategoryEncoder.fit(train, cat_cols, lcfg.categorical_min_count, lcfg.categorical_max_levels)
    booster = lgbm.fit(
        lgbm.to_matrix(train, cols, enc),
        train[LABEL].to_numpy(),
        feature_names=cols,
        categorical=[cols.index(c) for c in cat_cols],
        params=champion.params,
        rounds=champion.rounds,
        seed=seed,
    )
    raw = np.asarray(booster.predict(lgbm.to_matrix(calib, cols, enc)))
    calibrator = Calibrator.fit(raw, calib[LABEL].to_numpy())
    return FraudModel(booster, cols, enc, calibrator, champion.policy, champion.params, champion.rounds)


def compare(
    features: pl.DataFrame,
    champion: FraudModel,
    challenger: FraudModel,
    *,
    w: Windows,
    policy: Policy,
    cfg: RetrainingConfig,
    seed: int,
) -> Comparison:
    ev = _block(features, w.evaluation_start, w.labelled_end)
    _check("evaluation", ev)
    y = ev[LABEL].to_numpy()
    amount = ev[AMOUNT].cast(pl.Float64).to_numpy()
    day = (ev[DT] // SECONDS_PER_DAY).to_numpy()
    p_old, p_new = champion.predict_proba(ev), challenger.predict_proba(ev)

    def row_cost(p: np.ndarray) -> np.ndarray:
        a = decide(p, amount, day, policy)
        fraud = y == 1
        return np.select(
            [(a == Action.APPROVE) & fraud, (a == Action.DECLINE) & ~fraud, a == Action.REVIEW],
            [
                amount,
                np.full(amount.shape, policy.false_decline_cost),
                np.full(amount.shape, policy.review_cost),
            ],
            default=0.0,
        )

    cost_old, cost_new = row_cost(p_old), row_cost(p_new)

    def stats(idx: np.ndarray) -> dict[str, float]:
        yi = y[idx]
        if not 0 < yi.sum() < yi.size:
            nan = float("nan")
            return {"old": nan, "new": nan, "gain": nan, "saved": nan}
        old = float(average_precision_score(yi, p_old[idx]))
        new = float(average_precision_score(yi, p_new[idx]))
        return {
            "old": old,
            "new": new,
            "gain": new - old,
            "saved": (cost_old[idx].mean() - cost_new[idx].mean()) * 10_000,
        }

    iv = bootstrap(
        stats, y.size, groups=day, n_resamples=cfg.n_resamples, confidence=cfg.confidence, seed=seed
    )
    gain, saved = iv["gain"], iv["saved"]
    if gain.low > 0 and saved.estimate >= 0:
        promote, reason = (
            True,
            f"PR-AUC gain {gain.estimate:+.3f} (lower bound {gain.low:+.3f} > 0) at no extra cost",
        )
    elif gain.low <= 0:
        promote, reason = (
            False,
            f"PR-AUC gain {gain.estimate:+.3f} is not clearly positive (lower bound {gain.low:+.3f})",
        )
    else:
        promote, reason = False, f"more expensive: costs ${-saved.estimate:,.0f} more per 10,000 transactions"
    return Comparison(ev.height, int(y.sum()), iv["old"], iv["new"], gain, saved, promote, reason)
