"""API tests against a real MongoDB (a throwaway database per run). Skipped when no server is reachable."""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml
from fastapi.testclient import TestClient
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from fraudguard.api.app import create_app
from fraudguard.api.security import hash_api_key, hash_password, new_api_key
from fraudguard.config import (
    LightGBMConfig,
    Settings,
    load_data_config,
    load_feature_config,
    load_model_config,
)
from fraudguard.data import pipeline as data_pipeline
from fraudguard.data import snapshot, synthetic
from fraudguard.data.split import SPLIT, Split
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.models import champion

MONGO_URI = "mongodb://localhost:27017"
API = "/api/v1"
PASSWORDS = {"ana": "analyst-pass-123", "root": "admin-pass-1234"}


def _mongo_up() -> bool:
    try:
        MongoClient(MONGO_URI, serverSelectionTimeoutMS=800).admin.command("ping")
    except PyMongoError:
        return False
    return True


MONGO_UP = _mongo_up()
if not MONGO_UP and os.environ.get("FRAUDGUARD_REQUIRE_MONGO"):
    raise RuntimeError("FRAUDGUARD_REQUIRE_MONGO is set but MongoDB is not reachable on localhost:27017")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not MONGO_UP, reason="MongoDB is not running on localhost:27017"),
]


class Env:
    def __init__(
        self, client: TestClient, settings: Settings, key: str, root: Path, app_args: tuple[Any, ...]
    ) -> None:
        self.client, self.settings, self.key, self.root, self.app_args = client, settings, key, root, app_args

    def token(self, user: str) -> dict[str, str]:
        r = self.client.post(f"{API}/auth/login", json={"username": user, "password": PASSWORDS[user]})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    @property
    def api(self) -> dict[str, str]:
        return {"X-API-Key": self.key}


def payload(row: dict[str, Any]) -> dict[str, Any]:
    """A base-snapshot row as an API request: raw columns only, NaN as null."""
    drop = {"isFraud", "ts", "day", "split", "card_uid"}
    return {
        k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items() if k not in drop
    }


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Env]:
    tmp = tmp_path_factory.mktemp("api")
    cfg_dict = yaml.safe_load((Path(__file__).parents[1] / "configs" / "data.yaml").read_text())
    cfg_dict["paths"] = {"raw_dir": str(tmp / "raw"), "snapshot_dir": str(tmp / "snapshots")}
    (tmp / "data.yaml").write_text(yaml.safe_dump(cfg_dict))
    synthetic.write_synthetic(tmp / "raw", n_transactions=4_000, seed=5)
    data_cfg = load_data_config(tmp / "data.yaml", root=tmp)
    feat_cfg = load_feature_config()
    data_pipeline.build_base(data_cfg)
    feature_pipeline.build(data_cfg, feat_cfg)

    db_name = f"fraudguard_test_{uuid.uuid4().hex[:8]}"
    settings = Settings(
        mongo_uri=MONGO_URI,
        mongo_db=db_name,
        mlflow_tracking_uri=f"sqlite:///{(tmp / 'mlflow.db').as_posix()}",
        mlflow_experiment="api-test",
        serve_history_until="validation",
        jwt_secret="test-secret-" + uuid.uuid4().hex,  # type: ignore[arg-type]
    )
    model_cfg = load_model_config().model_copy(
        update={"lightgbm": LightGBMConfig(n_trials=2, n_folds=2, early_stopping_rounds=20, max_rounds=200)}
    )
    trained = champion.train_champion(data_cfg, model_cfg, settings)
    if not trained.is_champion:
        champion.MlflowClient(tracking_uri=settings.mlflow_tracking_uri).set_registered_model_alias(
            champion.REGISTERED_NAME, champion.CHAMPION_ALIAS, str(trained.model_version)
        )

    sync = MongoClient(MONGO_URI)[db_name]
    for user, role in (("ana", "analyst"), ("root", "admin")):
        sync.users.insert_one(
            {"username": user, "password_hash": hash_password(PASSWORDS[user]), "role": role, "active": True}
        )
    key, key_hash = new_api_key()
    sync.api_keys.insert_one({"name": "tests", "key_hash": key_hash, "active": True})
    assert hash_api_key(key) == key_hash

    args = (settings, data_cfg, feat_cfg, model_cfg)
    with TestClient(create_app(*args)) as client:
        yield Env(client, settings, key, tmp, args)
    MongoClient(MONGO_URI).drop_database(db_name)


def _test_rows(env: Env) -> pl.DataFrame:
    base = snapshot.load_snapshot(env.app_args[1].paths.snapshot_dir, data_pipeline.STAGE, "transactions")
    return base.filter(pl.col(SPLIT) == Split.TEST.value)


def test_health_and_docs(env: Env) -> None:
    r = env.client.get(f"{API}/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["transactions_in_memory"] > 1000
    assert env.client.get("/openapi.json").json()["info"]["title"] == "FraudGuard API"


def test_auth(env: Env) -> None:
    bad = env.client.post(f"{API}/auth/login", json={"username": "ana", "password": "nope"})
    unknown = env.client.post(f"{API}/auth/login", json={"username": "ghost", "password": "nope"})
    assert bad.status_code == unknown.status_code == 401
    assert env.client.get(f"{API}/auth/me").status_code == 401
    tokens = env.client.post(
        f"{API}/auth/login", json={"username": "ana", "password": PASSWORDS["ana"]}
    ).json()
    me = env.client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert me.json() == {"username": "ana", "role": "analyst"}
    # A refresh token is not an access token, and vice versa.
    assert (
        env.client.get(
            f"{API}/auth/me", headers={"Authorization": f"Bearer {tokens['refresh_token']}"}
        ).status_code
        == 401
    )
    assert (
        env.client.post(f"{API}/auth/refresh", json={"refresh_token": tokens["access_token"]}).status_code
        == 401
    )
    assert (
        env.client.post(f"{API}/auth/refresh", json={"refresh_token": tokens["refresh_token"]}).status_code
        == 200
    )


def test_score_validation_and_keys(env: Env) -> None:
    row = payload(_test_rows(env).row(0, named=True))
    assert env.client.post(f"{API}/score", json=row).status_code == 401
    assert env.client.post(f"{API}/score", json=row, headers={"X-API-Key": "fg_wrong"}).status_code == 401
    assert (
        env.client.post(f"{API}/score", json={**row, "TransactionAmt": -5}, headers=env.api).status_code
        == 422
    )
    assert env.client.post(f"{API}/score", json={**row, "evil": 1}, headers=env.api).status_code == 422


def test_replayed_test_month_matches_offline_model(env: Env) -> None:
    """The test month scored through the API gets the same probabilities as the model on offline features."""
    rows = _test_rows(env)
    features = snapshot.load_snapshot(env.app_args[1].paths.snapshot_dir, feature_pipeline.STAGE, "features")
    offline = features.filter(pl.col(SPLIT) == Split.TEST.value)
    model = champion.load_champion(env.settings.mlflow_tracking_uri)
    expected = model.predict_proba(offline)

    got: list[dict[str, Any]] = []
    single = env.client.post(f"{API}/score", json=payload(rows.row(0, named=True)), headers=env.api)
    assert single.status_code == 200, single.text
    got.append(single.json())
    rest = [payload(r) for r in rows.slice(1).iter_rows(named=True)]
    for start in range(0, len(rest), 500):
        r = env.client.post(
            f"{API}/score/batch", json={"transactions": rest[start : start + 500]}, headers=env.api
        )
        assert r.status_code == 200, r.text
        got += r.json()["results"]

    assert [g["transaction_id"] for g in got] == rows["TransactionID"].to_list()
    probs = np.array([g["probability"] for g in got])
    assert np.allclose(probs, expected, atol=1e-5), np.abs(probs - expected).max()
    assert all(g["decision"] in {"approve", "review", "decline"} and len(g["reasons"]) <= 3 for g in got)
    assert all(r["contribution"] > 0 for g in got for r in g["reasons"])

    # Older than what the store has seen: rejected rather than silently mis-featured.
    stale = env.client.post(f"{API}/score", json={**rest[0], "TransactionID": 99_999_999}, headers=env.api)
    assert stale.status_code == 409

    latencies = sorted(g["latency_ms"] for g in got)
    p95 = latencies[int(0.95 * len(latencies))]
    print(f"\nserver-side p95 latency over {len(got)} scores: {p95:.2f} ms")
    assert p95 < 50


def test_cases_verdicts_and_audit(env: Env) -> None:
    analyst = env.token("ana")
    page = env.client.get(f"{API}/cases", params={"status": "all", "page_size": 200}, headers=analyst).json()
    assert page["total"] > 0, "the replay should have opened review cases"
    probs = [c["probability"] for c in page["items"]]
    assert probs == sorted(probs, reverse=True)
    txn = page["items"][0]["txn_id"]

    detail = env.client.get(f"{API}/cases/{txn}", headers=analyst).json()
    assert detail["features"] and detail["contributions"] and "device" in detail["neighbourhood"]
    assert env.client.get(f"{API}/cases/123", headers=analyst).status_code == 404

    r = env.client.patch(f"{API}/cases/{txn}", json={"assignee": "ana", "note": "Checking"}, headers=analyst)
    assert r.json()["status"] == "in_review" and r.json()["notes"][0]["author"] == "ana"
    r = env.client.patch(f"{API}/cases/{txn}", json={"verdict": "fraud"}, headers=analyst)
    assert r.json()["status"] == "closed" and r.json()["verdict"] == "fraud"
    assert (
        env.client.patch(f"{API}/cases/{txn}", json={"verdict": "genuine"}, headers=analyst).status_code
        == 409
    )

    db = MongoClient(MONGO_URI)[env.settings.mongo_db]
    label = db.labels.find_one({"txn_id": txn})
    assert label is not None and label["label"] == 1 and label["source"] == "analyst"
    assert db.audit_log.count_documents({"action": "case.update", "target": f"case:{txn}"}) == 2


def test_kpis(env: Env) -> None:
    weeks = env.client.get(f"{API}/kpi/weekly", headers=env.token("ana")).json()
    db = MongoClient(MONGO_URI)[env.settings.mongo_db]
    assert sum(w["volume"] for w in weeks) == db.predictions.count_documents({})
    assert all(0 <= w["decline_rate"] <= 1 and w["p95_latency_ms"] is not None for w in weeks)
    assert any(w["labelled_fraud"] == 1 and w["recall"] == 1.0 for w in weeks)  # the verdict above


def test_policy_and_models_are_admin_only_and_audited(env: Env) -> None:
    analyst, admin = env.token("ana"), env.token("root")
    policy = env.client.get(f"{API}/policy", headers=analyst).json()
    new = {**policy, "review_cost": 4.5}
    assert env.client.put(f"{API}/policy", json=new, headers=analyst).status_code == 403
    assert env.client.put(f"{API}/policy", json=new, headers=admin).json()["review_cost"] == 4.5
    assert env.client.get(f"{API}/policy", headers=analyst).json()["review_cost"] == 4.5

    assert env.client.get(f"{API}/models", headers=analyst).status_code == 403
    versions = env.client.get(f"{API}/models", headers=admin).json()
    serving = next(v for v in versions if v["serving"])
    promoted = env.client.post(f"{API}/models/{serving['version']}/promote", headers=admin)
    assert promoted.status_code == 200 and "champion" in promoted.json()["aliases"]
    assert env.client.post(f"{API}/models/999/promote", headers=admin).status_code == 404

    db = MongoClient(MONGO_URI)[env.settings.mongo_db]
    assert db.audit_log.count_documents({"action": {"$in": ["policy.update", "model.promote"]}}) == 2


def test_preview_changes_nothing(env: Env) -> None:
    analyst = env.token("ana")
    db = MongoClient(MONGO_URI)[env.settings.mongo_db]
    before, n_predictions = env.client.get(f"{API}/health").json(), db.predictions.count_documents({})
    txn = {
        "TransactionAmt": 1999.0,
        "card1": 4242,
        "addr1": 300.0,
        "P_emaildomain": "anonymous.com",
        "D1": 0.0,
    }
    assert (
        env.client.post(f"{API}/score/preview", json=txn, headers=env.api).status_code == 401
    )  # people only
    r = env.client.post(f"{API}/score/preview", json=txn, headers=analyst)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transaction_dt"] == before["latest_transaction_dt"] and not body["case_opened"]
    assert 0 <= body["probability"] <= 1 and body["features"]["card_is_new"] is True
    again = env.client.post(f"{API}/score/preview", json=txn, headers=analyst).json()
    assert again["probability"] == body["probability"]  # repeatable: the first preview left no trace
    assert env.client.get(f"{API}/health").json() == before
    assert db.predictions.count_documents({}) == n_predictions


def test_demo_needs_its_scope_stores_nothing_and_is_rate_limited(env: Env) -> None:
    db = MongoClient(MONGO_URI)[env.settings.mongo_db]
    key, key_hash = new_api_key()
    db.api_keys.insert_one({"name": "landing", "key_hash": key_hash, "active": True, "scopes": ["demo"]})
    demo = {"X-API-Key": key}
    txn = {
        "TransactionAmt": 849.0,
        "ProductCD": "C",
        "P_emaildomain": "anonymous.com",
        "DeviceType": "mobile",
    }

    assert (
        env.client.post(f"{API}/demo/score", json=txn, headers=env.api).status_code == 403
    )  # score-only key
    assert (
        env.client.post(
            f"{API}/score", json=txn | {"TransactionID": 1, "TransactionDT": 1}, headers=demo
        ).status_code
        == 403
    )
    assert env.client.post(f"{API}/demo/score", json=txn | {"card1": 4242}, headers=demo).status_code == 422
    assert (
        env.client.post(f"{API}/demo/score", json=txn | {"P_emaildomain": "x.com"}, headers=demo).status_code
        == 422
    )

    before, n_predictions = env.client.get(f"{API}/health").json(), db.predictions.count_documents({})
    r = env.client.post(f"{API}/demo/score", json=txn, headers=demo)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"probability", "decision", "reasons", "model_version", "latency_ms"}
    assert 0 <= body["probability"] <= 1 and len(body["reasons"]) == 3
    assert env.client.get(f"{API}/health").json() == before
    assert (
        db.predictions.count_documents({}) == n_predictions and db.cases.count_documents({"txn_id": 0}) == 0
    )

    state = env.client.app.state.fg  # type: ignore[attr-defined]
    limit, state.demo_limiter.per_minute = state.demo_limiter.per_minute, 2
    try:
        codes = [env.client.post(f"{API}/demo/score", json=txn, headers=demo).status_code for _ in range(3)]
    finally:
        state.demo_limiter.per_minute = limit
    assert codes[-1] == 429


def test_restart_restores_state_and_policy(env: Env) -> None:
    before = env.client.get(f"{API}/health").json()
    with TestClient(create_app(*env.app_args)) as again:
        after = again.get(f"{API}/health").json()
        assert after["transactions_in_memory"] == before["transactions_in_memory"]
        assert after["latest_transaction_dt"] == before["latest_transaction_dt"]
        token = again.post(f"{API}/auth/login", json={"username": "ana", "password": PASSWORDS["ana"]}).json()
        policy = again.get(
            f"{API}/policy", headers={"Authorization": f"Bearer {token['access_token']}"}
        ).json()
        assert policy["review_cost"] == 4.5
