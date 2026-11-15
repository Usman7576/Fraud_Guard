from __future__ import annotations

import polars as pl
from typer.testing import CliRunner

from fraudguard.cli import app
from fraudguard.config import DataConfig
from fraudguard.data import pipeline, snapshot
from fraudguard.data.split import SPLIT


def test_build_writes_snapshot_then_skips(data_config: DataConfig) -> None:
    first = pipeline.build_base(data_config)
    assert not first.skipped
    df = snapshot.load_snapshot(data_config.paths.snapshot_dir, pipeline.STAGE, "transactions")
    assert {"ts", "day", "card_uid", SPLIT, "has_identity"} <= set(df.columns)
    assert df["TransactionDT"].is_sorted()
    assert df.schema["V1"] == pl.Float32  # downcast applied
    assert df.schema["TransactionAmt"] == pl.Float64  # amount keeps full precision

    second = pipeline.build_base(data_config)
    assert second.skipped
    assert second.manifest.digest() == first.manifest.digest()


def test_build_reruns_when_config_changes(data_config: DataConfig) -> None:
    pipeline.build_base(data_config)
    changed = data_config.model_copy(update={"seed": 123})
    assert not pipeline.build_base(changed).skipped


def test_cli_build_and_info(data_config: DataConfig, tmp_path: object) -> None:
    cfg_path = data_config.paths.raw_dir.parent / "data.yaml"
    runner = CliRunner()
    result = runner.invoke(app, ["data", "build", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "validation" in result.output
    info = runner.invoke(app, ["data", "info", "--config", str(cfg_path)])
    assert info.exit_code == 0, info.output
    assert "transactions" in info.output


def test_cli_features_and_adversarial_report(data_config: DataConfig) -> None:
    cfg_path = data_config.paths.raw_dir.parent / "data.yaml"
    out_dir = data_config.paths.raw_dir.parent / "reports"
    runner = CliRunner()

    missing = runner.invoke(app, ["features", "build", "--config", str(cfg_path)])
    assert missing.exit_code == 1  # no base snapshot yet

    assert runner.invoke(app, ["data", "build", "--config", str(cfg_path)]).exit_code == 0
    built = runner.invoke(app, ["features", "build", "--config", str(cfg_path)])
    assert built.exit_code == 0, built.output
    assert "velocity" in built.output

    features = snapshot.load_snapshot(data_config.paths.snapshot_dir, "features", "features")
    assert features.height == 3_000
    again = runner.invoke(app, ["features", "build", "--config", str(cfg_path)])
    assert "unchanged" in again.output

    report = runner.invoke(app, ["report", "adversarial", "--config", str(cfg_path), "--out", str(out_dir)])
    assert report.exit_code == 0, report.output
    assert (out_dir / "adversarial_validation.json").exists()
    assert (out_dir / "adversarial_validation.md").exists()
