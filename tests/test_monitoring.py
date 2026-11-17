from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml
from fastapi.testclient import TestClient
from pymongo import MongoClient

from fraudguard.api.app import create_app
from fraudguard.api.security import hash_password, new_api_key
from fraudguard.config import (
    LightGBMConfig,
    MonitoringConfig,
    RetrainingConfig,
    Settings,
    load_data_config,
    load_feature_config,
    load_model_config,
)
from fraudguard.data import pipeline as data_pipeline
from fraudguard.data import synthetic
from fraudguard.data.load import SECONDS_PER_DAY
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.models import champion
from fraudguard.monitoring import challenger as chal
from fraudguard.monitoring.drift import _column_result, weekly_drift
from fraudguard.monitoring.replay import LabelFeed, ReplayWorker
from tests.test_api import MONGO_UP

MONGO_URI = "mongodb://localhost:27017"


class TestPieces:
    def test_label_feed_releases_in_arrival_order_and_only_once(self) -> None:
        rows = pl.DataFrame(
            {"TransactionID": [1, 2, 3], "TransactionDT": [100, 50, 300], "isFraud": [1, 0, 1]}
        )
        sent: list[list[dict[str, Any]]] = []

        def post(path: str, body: dict[str, Any]) -> dict[str, int]:
            sent.append(body["labels"])
            return {"stored": len(body["labels"])}

        feed = LabelFeed(rows, delay=10, post=post)
        assert feed.release(59) == 0
        assert feed.release(110) == 2  # txn 2 known at 60, txn 1 at 110
        assert [lab["txn_id"] for lab in sent[-1]] == [2, 1]
        assert feed.release(110) == 0  # nothing twice
        assert feed.release(10**9) == 1

    def test_evidently_results_are_read_by_method(self) -> None:
        p_value = {
            "config": {
                "type": "evidently:metric_v2:ValueDrift",
                "column": "a",
                "method": "K-S p_value",
                "threshold": 0.05,
            },
            "value": 0.01,
        }
        distance = {
            "config": {
                "type": "evidently:metric_v2:ValueDrift",
                "column": "b",
                "method": "Wasserstein distance (normed)",
                "threshold": 0.1,
            },
            "value": 0.05,
        }
        other = {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}, "value": {"count": 1}}
        a, b = _column_result(p_value), _column_result(distance)
        assert a is not None and a.drifted  # small p-value: drift
        assert b is not None and not b.drifted  # small distance: no drift
        assert _column_result(other) is None

    def test_weekly_drift_finds_a_shifted_column(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        ref = pl.DataFrame(
            {"x": rng.normal(0, 1, 2000), "y": rng.normal(0, 1, 2000), "z": [None] * 2000},
            schema_overrides={"z": pl.Float64},
        )
        cur = pl.DataFrame({"x": rng.normal(2, 1, 400), "y": rng.normal(0, 1, 400), "z": [1.0] * 400})
        s = weekly_drift(ref, cur, ["x", "y", "z"], week=7, html_path=tmp_path / "w.html")
        assert s.n_features == 2  # all-missing z skipped
        assert s.top() == ["x"] and s.drift_share == 0.5
        assert (tmp_path / "w.html").stat().st_size > 1000

    def test_windows_are_ordered_and_end_one_delay_ago(self) -> None:
        cfg = RetrainingConfig(window_days=60, calibration_days=7, evaluation_days=7)
        w = chal.windows(now_dt=100 * SECONDS_PER_DAY, label_delay_days=30, cfg=cfg)
        assert w.labelled_end == 70 * SECONDS_PER_DAY
        assert w.train_start < w.calibration_start < w.evaluation_start < w.labelled_end
        assert w.labelled_end - w.evaluation_start == 7 * SECONDS_PER_DAY


@pytest.mark.slow
@pytest.mark.skipif(not MONGO_UP, reason="MongoDB is not running on localhost:27017")
def test_replay_end_to_end(tmp_path: Path) -> None:
    """Train a champion, serve history up to the end of training, replay the later months through the API."""
    cfg_dict = yaml.safe_load((Path(__file__).parents[1] / "configs" / "data.yaml").read_text())
    cfg_dict["paths"] = {"raw_dir": str(tmp_path / "raw"), "snapshot_dir": str(tmp_path / "snapshots")}
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg_dict))
    synthetic.write_synthetic(tmp_path / "raw", n_transactions=6_000, seed=3)
    data_cfg = load_data_config(tmp_path / "data.yaml", root=tmp_path)
    feat_cfg = load_feature_config()
    data_pipeline.build_base(data_cfg)
    feature_pipeline.build(data_cfg, feat_cfg)
    db_name = f"fraudguard_test_{uuid.uuid4().hex[:8]}"
    settings = Settings(
        mongo_uri=MONGO_URI,
        mongo_db=db_name,
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        mlflow_experiment="replay-test",
        serve_history_until="train",
        jwt_secret="replay-test-secret-" + uuid.uuid4().hex,  # type: ignore[arg-type]
        reports_dir=tmp_path / "reports",
    )
    model_cfg = load_model_config().model_copy(
        update={"lightgbm": LightGBMConfig(n_trials=2, n_folds=2, early_stopping_rounds=20, max_rounds=200)}
    )
    trained = champion.train_champion(data_cfg, model_cfg, settings)
    if not trained.is_champion:
        champion.MlflowClient(tracking_uri=settings.mlflow_tracking_uri).set_registered_model_alias(
            champion.REGISTERED_NAME, champion.CHAMPION_ALIAS, str(trained.model_version)
        )
    mon_cfg = MonitoringConfig.model_validate(
        {
            "replay": {"start_split": "validation", "batch_size": 400},
            "drift": {"reference_rows": 1500, "n_features": 15, "share_threshold": 0.05, "min_rows": 50},
            "retraining": {
                "window_days": 90,
                "calibration_days": 10,
                "evaluation_days": 10,
                "n_resamples": 40,
                "min_weeks_between": 3,
            },
        }
    )
    sync = MongoClient(MONGO_URI)[db_name]
    score_only, h1 = new_api_key()
    worker_key, h2 = new_api_key()
    sync.api_keys.insert_many(
        [
            {"name": "scorer", "key_hash": h1, "active": True},  # no scopes field: score only
            {"name": "replay", "key_hash": h2, "active": True, "scopes": ["score", "labels", "promote"]},
        ]
    )
    try:
        with TestClient(create_app(settings, data_cfg, feat_cfg, model_cfg)) as api:
            # Scopes: a score-only key cannot feed labels or promote models.
            body = {"labels": [{"txn_id": 1, "label": 1, "known_dt": 1}]}
            assert api.post("/api/v1/labels", json=body, headers={"X-API-Key": score_only}).status_code == 403
            assert (
                api.post("/api/v1/ops/models/1/promote", headers={"X-API-Key": score_only}).status_code == 403
            )

            worker = ReplayWorker(
                api,
                worker_key,
                sync,
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                mon_cfg=mon_cfg,
                settings=settings,
            )
            summary = worker.run()

            replayed = sum(w.n_scored for w in summary.weeks)
            assert replayed == sync.predictions.count_documents({}) > 1000
            assert sync.drift_reports.count_documents({}) == len(summary.weeks)
            # Every replayed fraud eventually gets its chargeback, and the API knew every transaction.
            assert sync.labels.count_documents({"source": "dataset"}) == replayed
            assert any(w.recall is not None for w in summary.weeks), (
                "later weeks should see fully labelled weeks"
            )

            events = list(sync.retraining_events.find())
            assert events, "a 5% drift threshold on synthetic data should trigger retraining"
            assert {e["outcome"] for e in events} <= {"promoted", "kept_champion", "skipped"}
            trained_events = [e for e in events if e["outcome"] != "skipped"]
            if trained_events:
                assert all(
                    e["challenger_version"] and e["comparison"]["pr_auc_gain"]["n_valid"] > 0
                    for e in trained_events
                )
                aliases = (
                    champion.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
                    .get_registered_model(champion.REGISTERED_NAME)
                    .aliases
                )
                assert "challenger" in aliases
            if any(e["outcome"] == "promoted" for e in events):
                assert api.get("/api/v1/health").json()["model_version"] == summary.final_champion
                assert (
                    sync.audit_log.count_documents({"action": "model.promote", "actor": "api_key:replay"})
                    >= 1
                )

            # The console's monitoring endpoints.
            sync.users.insert_one(
                {
                    "username": "mon",
                    "password_hash": hash_password("monitor-pass-1"),
                    "role": "analyst",
                    "active": True,
                }
            )
            token = api.post(
                "/api/v1/auth/login", json={"username": "mon", "password": "monitor-pass-1"}
            ).json()["access_token"]
            auth = {"Authorization": f"Bearer {token}"}
            weeks = api.get("/api/v1/monitoring/drift", headers=auth).json()
            assert len(weeks) == len(summary.weeks) and "columns" not in weeks[0]
            detail = api.get(f"/api/v1/monitoring/drift/{weeks[0]['week']}", headers=auth).json()
            assert detail["columns"]
            report = api.get(f"/api/v1/monitoring/drift/{weeks[0]['week']}/report", headers=auth)
            assert report.status_code == 200 and "sandbox" in report.headers["content-security-policy"]
            assert len(api.get("/api/v1/monitoring/retraining", headers=auth).json()) == len(events)
    finally:
        MongoClient(MONGO_URI).drop_database(db_name)
