"""Configuration: YAML pipeline configs plus environment settings."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

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
    label_delay_days: int = Field(ge=0)

    def fingerprint(self) -> str:
        """Stable hash of the config, recorded in snapshot manifests."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class AdversarialConfig(BaseModel):
    n_folds: int = Field(default=5, ge=2)
    max_rows: int = Field(default=200_000, ge=1_000)
    num_boost_round: int = Field(default=300, ge=10)
    top_k: int = Field(default=15, ge=1)


class FeatureConfig(BaseModel):
    velocity_windows: list[str] = Field(min_length=1)
    velocity_keys: dict[str, str] = Field(min_length=1)
    history_window: str
    min_history_for_zscore: int = Field(default=2, ge=2)
    adversarial: AdversarialConfig = AdversarialConfig()

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


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
