"""The deployable model: LightGBM booster, category vocabulary, calibrator and policy.

Everything is stored as plain text (the booster's own text format and JSON),
so loading a model never unpickles arbitrary objects. The bundle is logged to
MLflow as a pyfunc model and registered as `fraudguard`; the scoring API loads
the `champion` alias.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from mlflow.models import ModelSignature
from mlflow.pyfunc.model import PythonModel
from mlflow.types import ColSpec, DataType, Schema, TensorSpec

from fraudguard.decision.policy import Policy
from fraudguard.models.calibration import Calibrator
from fraudguard.models.lgbm import CategoryEncoder, to_matrix

BOOSTER_FILE = "booster.txt"
META_FILE = "bundle.json"


@dataclass
class FraudModel:
    booster: lgb.Booster
    feature_names: list[str]
    encoder: CategoryEncoder
    calibrator: Calibrator
    policy: Policy
    # Tuned LightGBM parameters and rounds, kept so the model can be retrained on other feature sets.
    params: dict[str, Any] = field(default_factory=dict)
    rounds: int = 0

    def matrix(self, df: pl.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise KeyError(f"input is missing model features: {missing[:10]}")
        return to_matrix(df, self.feature_names, self.encoder)

    def vector(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Feature matrix from plain dicts, encoded exactly like `matrix` (used by the live API)."""
        if not hasattr(self, "_codes"):
            self._codes = {
                c: {v: float(i) for i, v in enumerate(lv)} for c, lv in self.encoder.levels.items()
            }
        out = np.full((len(rows), len(self.feature_names)), np.nan, dtype=np.float32)
        for i, row in enumerate(rows):
            for j, col in enumerate(self.feature_names):
                v = row.get(col)
                if v is None:
                    continue
                codes = self._codes.get(col)
                if codes is not None:
                    out[i, j] = codes.get(str(v), float(len(codes)))
                else:
                    out[i, j] = float(v)
        return out

    def raw_score(self, df: pl.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(self.matrix(df)))

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.raw_score(df))

    def contributions(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Exact TreeSHAP values in log-odds, (n, features), plus each row's base value.

        LightGBM computes these natively (`pred_contrib`), so no extra library or
        sampling is needed, and they sum exactly to the raw log-odds score."""
        out = np.asarray(self.booster.predict(self.matrix(df), pred_contrib=True))
        return out[:, :-1], out[:, -1]

    def save(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / BOOSTER_FILE))
        meta = {
            "feature_names": self.feature_names,
            "categorical_levels": self.encoder.levels,
            "calibrator": self.calibrator.to_dict(),
            "policy": self.policy.as_dict(),
            "params": self.params,
            "rounds": self.rounds,
        }
        (path / META_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> FraudModel:
        meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
        return cls(
            booster=lgb.Booster(model_file=str(path / BOOSTER_FILE)),
            feature_names=list(meta["feature_names"]),
            encoder=CategoryEncoder({k: list(v) for k, v in meta["categorical_levels"].items()}),
            calibrator=Calibrator.from_dict(meta["calibrator"]),
            policy=Policy(**meta["policy"]),
            params=dict(meta.get("params", {})),
            rounds=int(meta.get("rounds", 0)),
        )


class FraudGuardPyfunc(PythonModel):
    """MLflow wrapper: returns calibrated fraud probabilities for a pandas frame of features."""

    def load_context(self, context: Any) -> None:
        self.model = FraudModel.load(Path(context.artifacts["bundle"]))

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> np.ndarray:
        return self.model.predict_proba(pl.from_pandas(model_input))


def _is_text(dtype: pl.DataType) -> bool:
    return dtype in (pl.String, pl.Categorical)


def signature(df: pl.DataFrame, columns: list[str]) -> ModelSignature:
    """MLflow signature: every feature optional, numbers as double, text as string.

    Inferring it from a sample would mark columns "required" just because the sample had no
    gaps, and MLflow's integer type cannot hold missing values."""
    inputs = Schema(
        [
            ColSpec(DataType.string if _is_text(df.schema[c]) else DataType.double, c, required=False)
            for c in columns
        ]
    )
    return ModelSignature(inputs=inputs, outputs=Schema([TensorSpec(np.dtype(np.float64), (-1,))]))


def pyfunc_input(df: pl.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Features shaped for the pyfunc signature (numbers as float64)."""
    return df.select(
        pl.col(c) if _is_text(df.schema[c]) else pl.col(c).cast(pl.Float64) for c in columns
    ).to_pandas()
