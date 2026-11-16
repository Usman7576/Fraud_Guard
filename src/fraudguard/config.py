"""Configuration: YAML pipeline configs plus environment settings."""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class Settings(BaseSettings):
    """Runtime settings read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_prefix="FRAUDGUARD_", env_file=".env", extra="ignore")

    env: str = "development"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "fraudguard"
    log_level: str = "INFO"
    mlflow_tracking_uri: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow_experiment: str = "fraudguard"


class Paths(BaseModel):
    raw_dir: Path
    snapshot_dir: Path

    def resolve(self, root: Path) -> Paths:
        return Paths(
            raw_dir=self.raw_dir if self.raw_dir.is_absolute() else root / self.raw_dir,
            snapshot_dir=self.snapshot_dir if self.snapshot_dir.is_absolute() else root / self.snapshot_dir,
        )


class RawFiles(BaseModel):
    transactions: str
    identity: str


class SplitConfig(BaseModel):
    train_end: date
    validation_end: date

    @model_validator(mode="after")
    def _ordered(self) -> SplitConfig:
        if self.train_end >= self.validation_end:
            raise ValueError("split.train_end must be before split.validation_end")
        return self


class DataConfig(BaseModel):
    seed: int = 42
    paths: Paths
    raw_files: RawFiles
    reference_date: date
    split: SplitConfig

    def fingerprint(self) -> str:
        """Stable hash of the config, recorded in snapshot manifests."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class AdversarialConfig(BaseModel):
    n_folds: int = Field(default=5, ge=2)
    max_rows: int = Field(default=200_000, ge=1_000)
    num_boost_round: int = Field(default=300, ge=10)
    top_k: int = Field(default=15, ge=1)


class GraphConfig(BaseModel):
    label_delay_days: int = Field(default=30, ge=0)
    hop_entities: list[Literal["device", "address", "email"]] = Field(default=["device", "address"])
    max_neighbours: int = Field(default=200, ge=1)
    entity_window_days: int = Field(default=30, ge=1)


class FeatureConfig(BaseModel):
    velocity_windows: list[str] = Field(min_length=1)
    velocity_keys: dict[str, str] = Field(min_length=1)
    history_window: str
    min_history_for_zscore: int = Field(default=2, ge=2)
    graph: GraphConfig = GraphConfig()
    adversarial: AdversarialConfig = AdversarialConfig()

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class RuleConfig(BaseModel):
    name: str
    feature: str
    op: Literal[">", ">=", "<", "<=", "=="]
    value: float
    weight: float = Field(default=1.0, gt=0)


class LogisticConfig(BaseModel):
    C: float = Field(default=0.1, gt=0)
    max_iter: int = Field(default=2000, ge=100)
    min_category_share: float = Field(default=0.001, gt=0, lt=1)
    categorical: list[str] = Field(default_factory=list)
    exclude_prefixes: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    target_fpr: float = Field(default=0.01, gt=0, lt=1)


class LightGBMConfig(BaseModel):
    n_trials: int = Field(default=60, ge=1)
    timeout_minutes: float = Field(default=60, gt=0)
    n_folds: int = Field(default=3, ge=1)
    early_stopping_rounds: int = Field(default=100, ge=5)
    max_rounds: int = Field(default=3000, ge=10)
    categorical_min_count: int = Field(default=30, ge=1)
    categorical_max_levels: int = Field(default=200, ge=1)
    exclude: list[str] = Field(default_factory=list)


class CalibrationConfig(BaseModel):
    n_folds: int = Field(default=5, ge=2)
    bin_edges: list[float] = Field(default=[0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0], min_length=2)

    @model_validator(mode="after")
    def _edges(self) -> CalibrationConfig:
        e = self.bin_edges
        if e[0] != 0.0 or e[-1] != 1.0 or any(a >= b for a, b in itertools.pairwise(e)):
            raise ValueError("calibration.bin_edges must rise strictly from 0.0 to 1.0")
        return self


class MultiplierGrid(BaseModel):
    min: float = Field(default=0.125, gt=0)
    max: float = Field(default=8.0, gt=0)
    n: int = Field(default=25, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> MultiplierGrid:
        if self.min > 1.0 or self.max < 1.0:
            raise ValueError("risk_multipliers must include 1.0 (min <= 1 <= max)")
        return self


class DecisionConfig(BaseModel):
    false_decline_cost: float = Field(default=15.0, ge=0)
    review_cost: float = Field(default=3.0, ge=0)
    review_capacity_per_day: int = Field(default=200, ge=0)
    risk_multipliers: MultiplierGrid = MultiplierGrid()
    sensitivity_factors: list[float] = Field(default=[0.5, 1.0, 2.0], min_length=1)


class ModelConfig(BaseModel):
    evaluation: EvaluationConfig = EvaluationConfig()
    rules: list[RuleConfig] = Field(min_length=1)
    logistic: LogisticConfig = LogisticConfig()
    lightgbm: LightGBMConfig = LightGBMConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    decision: DecisionConfig = DecisionConfig()

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_model_config(path: Path | None = None) -> ModelConfig:
    return ModelConfig.model_validate(_read_yaml(path or CONFIG_DIR / "model.yaml"))


def load_feature_config(path: Path | None = None) -> FeatureConfig:
    return FeatureConfig.model_validate(_read_yaml(path or CONFIG_DIR / "features.yaml"))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def load_data_config(path: Path | None = None, root: Path = PROJECT_ROOT) -> DataConfig:
    cfg = DataConfig.model_validate(_read_yaml(path or CONFIG_DIR / "data.yaml"))
    return cfg.model_copy(update={"paths": cfg.paths.resolve(root)})


@lru_cache
def get_settings() -> Settings:
    return Settings()
