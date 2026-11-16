from __future__ import annotations

import json

import mlflow
import numpy as np
import polars as pl
import pytest

from fraudguard.config import DataConfig, LightGBMConfig, Settings, load_feature_config, load_model_config
from fraudguard.data import pipeline as data_pipeline
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.models import calibration, champion, lgbm
from fraudguard.models.bundle import pyfunc_input


class TestCalibration:
    def test_isotonic_fixes_overconfident_scores(self) -> None:
        rng = np.random.default_rng(0)
        true_p = rng.beta(0.5, 10, 20_000)
        y = (rng.random(true_p.size) < true_p).astype(int)
        score = np.sqrt(true_p)  # same ranking, badly calibrated
        edges = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
        ece_raw, _ = calibration.reliability(score, y, edges)
        oof = calibration.cross_fit(score, y, n_folds=5)
        ece_cal, bins = calibration.reliability(oof, y, edges)
        assert ece_cal < ece_raw / 5
        assert sum(b.count for b in bins) == y.size

    def test_calibrator_round_trips_and_is_monotone(self) -> None:
        rng = np.random.default_rng(1)
        score = rng.random(2_000)
        y = (rng.random(2_000) < score).astype(int)
        cal = calibration.Calibrator.fit(score, y)
        again = calibration.Calibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
        grid = np.linspace(-0.5, 1.5, 101)  # includes out-of-range scores
        assert np.array_equal(cal.predict(grid), again.predict(grid))
        assert np.all(np.diff(cal.predict(grid)) >= 0)
        assert cal.predict(grid).min() >= 0 and cal.predict(grid).max() <= 1


class TestLightGBMHelpers:
    def test_time_folds_are_expanding_and_never_look_back(self) -> None:
        folds = lgbm.time_folds(100, 3)
        assert len(folds) == 3
        for k, (tr, va) in enumerate(folds):
            assert tr.max() < va.min()  # always predict the future
            assert tr.min() == 0 and tr.size == 25 * (k + 1)

    def test_category_encoder(self) -> None:
        df = pl.DataFrame({"c": ["a"] * 5 + ["b"] * 3 + ["rare"] + [None]})
        enc = lgbm.CategoryEncoder.fit(df, ["c"], min_count=2, max_levels=10)
        assert enc.levels == {"c": ["a", "b"]}
        new = pl.Series("c", ["b", "a", "unseen", None, "rare"])
        codes = enc.encode(new)
        assert codes[:3].tolist() == [1.0, 0.0, 2.0] and np.isnan(codes[3]) and codes[4] == 2.0
        assert lgbm.CategoryEncoder.fit(df, ["c"], min_count=1, max_levels=1).levels == {"c": ["a"]}

    def test_to_matrix_keeps_nulls_as_nan(self) -> None:
        df = pl.DataFrame({"x": [1, None, 3], "flag": [True, False, None], "c": ["a", None, "a"]})
        enc = lgbm.CategoryEncoder({"c": ["a"]})
        m = lgbm.to_matrix(df, ["x", "flag", "c"], enc)
        assert m.dtype == np.float32 and m.shape == (3, 3)
        assert np.isnan(m[1, 0]) and np.isnan(m[2, 1]) and np.isnan(m[1, 2])
        assert m[0].tolist() == [1.0, 1.0, 0.0]


@pytest.mark.slow
def test_champion_end_to_end(data_config: DataConfig) -> None:
    data_pipeline.build_base(data_config)
    feature_pipeline.build(data_config, load_feature_config())
    tmp = data_config.paths.raw_dir.parent
    settings = Settings(
        mlflow_tracking_uri=f"sqlite:///{(tmp / 'mlflow.db').as_posix()}", mlflow_experiment="test"
    )
    model_cfg = load_model_config()
    model_cfg = model_cfg.model_copy(
        update={"lightgbm": LightGBMConfig(n_trials=3, n_folds=2, early_stopping_rounds=20, max_rounds=200)}
    )

    report = champion.train_champion(data_config, model_cfg, settings)
    assert report.validation_raw.pr_auc > report.validation_raw.fraud_rate * 3
    assert report.tuning["n_trials"] == 3
    assert report.model_version == "1"
    assert len(report.sensitivity) == len(model_cfg.decision.sensitivity_factors) ** 2
    model_cost, rules_cost = report.model_policy["cost"], report.rules_policy["cost"]
    assert isinstance(model_cost, dict) and isinstance(rules_cost, dict)
    assert report.is_champion == (model_cost["total"] < rules_cost["total"])
    assert model_cost["total"] < report.approve_all["total"]

    json_path, md_path = champion.write_reports(report, tmp / "reports")
    assert json.loads(json_path.read_text())["features_digest"] == report.features_digest
    assert "Sensitivity" in md_path.read_text(encoding="utf-8")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    run = mlflow.get_run(report.run_id)
    assert run.data.tags["features_digest"] == report.features_digest
    assert "valid_cost_per_10k" in run.data.metrics

    # The registered bundle reproduces the in-memory model's probabilities.
    if report.is_champion:
        loaded = champion.load_champion(settings.mlflow_tracking_uri)
        features = pl.read_parquet(data_config.paths.snapshot_dir / "features" / "features.parquet")
        sample = features.filter(pl.col("split") == "validation").head(200)
        p = loaded.predict_proba(sample)
        assert p.shape == (200,) and np.all((p >= 0) & (p <= 1))
        pyfunc = mlflow.pyfunc.load_model(f"models:/{champion.REGISTERED_NAME}@{champion.CHAMPION_ALIAS}")
        assert np.allclose(np.asarray(pyfunc.predict(pyfunc_input(sample, loaded.feature_names))), p)
