"""Weekly data drift with Evidently.

Each replayed week is compared with a sample of the training months on the
champion's most important features. Evidently picks a test per column (K-S or
a chi-square / Z-test on small samples, Wasserstein or Jensen-Shannon distance
on large ones). Its HTML report is saved for analysts, and a small summary goes
to MongoDB so the console and the retraining trigger can read it.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fraudguard.models.bundle import FraudModel


@dataclass
class ColumnDrift:
    column: str
    method: str
    value: float
    threshold: float
    drifted: bool


@dataclass
class DriftSummary:
    week: int
    n_rows: int
    n_features: int
    n_drifted: int
    drift_share: float
    columns: list[ColumnDrift] = field(default_factory=list)
    report_file: str | None = None

    def top(self, k: int = 5) -> list[str]:
        return [c.column for c in self.columns if c.drifted][:k]

    def as_doc(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["top_drifted"] = self.top()
        return doc


def monitored_features(model: FraudModel, n: int) -> list[str]:
    """The champion's top-n features by total gain."""
    gain = np.asarray(model.booster.feature_importance("gain"))
    order = np.argsort(-gain)[:n]
    return [model.feature_names[i] for i in order if gain[i] > 0]


def _frame(df: pl.DataFrame, columns: list[str]) -> Any:
    """Pandas frame for Evidently: numbers as float, text as object, booleans as numbers."""
    return df.select(
        pl.col(c).cast(pl.Float64) if df.schema[c] != pl.String else pl.col(c) for c in columns
    ).to_pandas()


def _column_result(metric: dict[str, Any]) -> ColumnDrift | None:
    cfg = metric.get("config", {})
    if not str(cfg.get("type", "")).endswith("ValueDrift"):
        return None
    method, threshold, value = (
        str(cfg.get("method", "")),
        float(cfg.get("threshold", 0.05)),
        metric.get("value"),
    )
    if value is None:
        return None
    value = float(value)
    # p-value tests flag drift below the threshold; distance measures flag it at or above.
    drifted = value < threshold if "p_value" in method else value >= threshold
    return ColumnDrift(str(cfg.get("column")), method, round(value, 6), threshold, drifted)


def weekly_drift(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    columns: list[str],
    week: int,
    html_path: Path | None = None,
) -> DriftSummary:
    from evidently import Report  # noqa: PLC0415  (heavy import, only for monitoring)
    from evidently.presets import DataDriftPreset  # noqa: PLC0415

    # A column that is entirely missing in either sample has nothing to compare.
    usable = [
        c
        for c in columns
        if reference[c].null_count() < reference.height and current[c].null_count() < current.height
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snapshot = Report([DataDriftPreset()]).run(
            current_data=_frame(current, usable), reference_data=_frame(reference, usable)
        )
    results = [r for m in snapshot.dict()["metrics"] if (r := _column_result(m)) is not None]
    results.sort(key=lambda r: (not r.drifted, r.column))
    n_drifted = sum(r.drifted for r in results)
    report_file = None
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
        report_file = html_path.name
    return DriftSummary(
        week=week,
        n_rows=current.height,
        n_features=len(results),
        n_drifted=n_drifted,
        drift_share=round(n_drifted / len(results), 4) if results else 0.0,
        columns=results,
        report_file=report_file,
    )
