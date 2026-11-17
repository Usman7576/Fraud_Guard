"""Throwaway environment for the Playwright end-to-end tests (run by web/e2e/run.mjs).

    python scripts/e2e_env.py setup     # fresh database, two users, a demo key and a score key
    python scripts/e2e_env.py score     # score test-month payments through the running API
    python scripts/e2e_env.py teardown  # drop the database

Passwords come from E2E_ADMIN_PASSWORD and E2E_ANALYST_PASSWORD (generated per run, never stored).
`setup` prints the two keys as JSON on stdout.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from typing import Any

import httpx
import polars as pl
from pymongo import AsyncMongoClient, MongoClient

from fraudguard.api import db
from fraudguard.api.security import hash_password, new_api_key
from fraudguard.config import get_settings, load_data_config
from fraudguard.data import pipeline as data_pipeline
from fraudguard.data import snapshot
from fraudguard.data.split import SPLIT, Split

DB = os.environ.get("FRAUDGUARD_MONGO_DB", "")
DROP = {"isFraud", "ts", "day", "split", "card_uid"}


def _check_db() -> str:
    if not DB.startswith("fraudguard_e2e"):
        sys.exit("FRAUDGUARD_MONGO_DB must start with fraudguard_e2e (this script drops it)")
    return DB


async def _setup() -> dict[str, str]:
    client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(get_settings().mongo_uri)
    try:
        await client.drop_database(DB)
        database = client[DB]
        await db.init_db(database)
        for name, role, env in (
            ("e2e-admin", "admin", "E2E_ADMIN_PASSWORD"),
            ("e2e-analyst", "analyst", "E2E_ANALYST_PASSWORD"),
        ):
            await database.users.insert_one(
                {
                    "username": name,
                    "password_hash": hash_password(os.environ[env]),
                    "role": role,
                    "active": True,
                    "created_at": db.now(),
                }
            )
        keys = {}
        for name, scopes in (("e2e-demo", ["demo"]), ("e2e-score", ["score"])):
            key, key_hash = new_api_key()
            await database.api_keys.insert_one(
                {"name": name, "key_hash": key_hash, "active": True, "scopes": scopes, "created_at": db.now()}
            )
            keys[name] = key
        return keys
    finally:
        await client.close()


def _score(api_url: str, key: str, n: int) -> None:
    base = snapshot.load_snapshot(load_data_config().paths.snapshot_dir, data_pipeline.STAGE, "transactions")
    rows = base.filter(pl.col(SPLIT) == Split.TEST.value).sort("TransactionDT").head(n).to_dicts()
    payload = [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items() if k not in DROP}
        for r in rows
    ]
    r = httpx.post(
        f"{api_url}/api/v1/score/batch",
        json={"transactions": payload},
        headers={"X-API-Key": key},
        timeout=120,
    )
    r.raise_for_status()
    decisions = [x["decision"] for x in r.json()["results"]]
    print(json.dumps({d: decisions.count(d) for d in ("approve", "review", "decline")}))


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    _check_db()
    if command == "setup":
        print(json.dumps(asyncio.run(_setup())))
    elif command == "score":
        _score(
            os.environ["E2E_API_URL"],
            os.environ["E2E_SCORE_KEY"],
            int(os.environ.get("E2E_SCORE_ROWS", "400")),
        )
    elif command == "teardown":
        MongoClient(get_settings().mongo_uri).drop_database(DB)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
