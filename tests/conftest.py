from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import yaml

from fraudguard.config import DataConfig, load_data_config
from fraudguard.data import synthetic


@pytest.fixture(scope="session")
def raw_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    return synthetic.generate(n_transactions=5_000, seed=7)


@pytest.fixture
def data_config(tmp_path: Path) -> DataConfig:
    """A config pointing at an isolated temp folder, with synthetic raw files written."""
    cfg_dict = yaml.safe_load((Path(__file__).parents[1] / "configs" / "data.yaml").read_text())
    cfg_dict["paths"] = {"raw_dir": str(tmp_path / "raw"), "snapshot_dir": str(tmp_path / "snapshots")}
    cfg_path = tmp_path / "data.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    synthetic.write_synthetic(tmp_path / "raw", n_transactions=3_000, seed=11)
    return load_data_config(cfg_path, root=tmp_path)
