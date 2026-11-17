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
from pydantic import BaseModel, Field, SecretStr, model_validator
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

    # API. Without a JWT secret a random one is generated at startup (tokens die with the process).
    jwt_secret: SecretStr | None = None
    access_token_minutes: int = Field(default=15, ge=1)
    refresh_token_days: int = Field(default=7, ge=1)
    cors_origins: list[str] = ["http://localhost:3000"]
    score_rate_limit_per_minute: int = Field(default=6000, ge=1)
    # History loaded into the online feature store at startup: every split up to and including this one.
    serve_history_until: Literal["train", "validation", "test"] = "validation"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Where weekly drift reports (Evidently HTML) are written by the replay worker and served from.
    reports_dir: Path = PROJECT_ROOT / "reports"


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


class FinalEvaluationConfig(BaseModel):
    n_resamples: int = Field(default=1000, ge=10)
    resample_unit: Literal["day", "row"] = "day"
    confidence: float = Field(default=0.95, gt=0.5, lt=1)
    amount_bands: list[float] = Field(default=[50, 100, 250, 500, 1000], min_length=1)
    top_email_domains: int = Field(default=8, ge=1)


class AblationConfig(BaseModel):
    drift_prone: list[str] = Field(default_factory=list)
    n_resamples: int = Field(default=300, ge=10)
    adversarial: bool = True


class ExplainConfig(BaseModel):
    n_reasons: int = Field(default=3, ge=1)
    global_sample: int = Field(default=5000, ge=100)


class ModelConfig(BaseModel):
    evaluation: EvaluationConfig = EvaluationConfig()
    evaluation_final: FinalEvaluationConfig = FinalEvaluationConfig()
    ablation: AblationConfig = AblationConfig()
    explain: ExplainConfig = ExplainConfig()
    rules: list[RuleConfig] = Field(min_length=1)
    logistic: LogisticConfig = LogisticConfig()
    lightgbm: LightGBMConfig = LightGBMConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    decision: DecisionConfig = DecisionConfig()

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ReplayConfig(BaseModel):
    start_split: Literal["validation", "test"] = "validation"
    batch_size: int = Field(default=250, ge=1, le=1000)


class DriftConfig(BaseModel):
    reference_rows: int = Field(default=20_000, ge=100)
    n_features: int = Field(default=30, ge=1)
    share_threshold: float = Field(default=0.3, gt=0, le=1)
    min_rows: int = Field(default=200, ge=1)


class RetrainingConfig(BaseModel):
    recall_drop: float = Field(default=0.05, gt=0, lt=1)
    min_weeks_between: int = Field(default=2, ge=0)
    window_days: int = Field(default=180, ge=14)
    calibration_days: int = Field(default=14, ge=1)
    evaluation_days: int = Field(default=14, ge=1)
    n_resamples: int = Field(default=300, ge=10)
    confidence: float = Field(default=0.95, gt=0.5, lt=1)


class MonitoringConfig(BaseModel):
    replay: ReplayConfig = ReplayConfig()
    label_delay_days: int = Field(default=30, ge=0)
    drift: DriftConfig = DriftConfig()
    retraining: RetrainingConfig = RetrainingConfig()


def load_monitoring_config(path: Path | None = None) -> MonitoringConfig:
    return MonitoringConfig.model_validate(_read_yaml(path or CONFIG_DIR / "monitoring.yaml"))


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
