"""MongoDB collections, validators and indexes.

`fraudguard db init` (and the API at startup) runs `init_db`, which is
idempotent: collections are created with a JSON-schema validator, or have it
updated with `collMod`, and indexes are created if missing. Everything can be
browsed in MongoDB Compass under the `fraudguard` database.

The online feature state lives in the API's memory for speed. It is rebuilt at
startup from the feature snapshot plus the `transactions` collection (every
transaction scored live) and analyst fraud verdicts in `labels`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import CollectionInvalid

Doc = dict[str, Any]
DECISIONS = ["approve", "review", "decline"]
CASE_STATUSES = ["open", "in_review", "closed"]


def _schema(required: list[str], properties: dict[str, Doc]) -> Doc:
    schema: Doc = {"bsonType": "object", "properties": properties}
    if required:  # MongoDB rejects an empty "required" list
        schema["required"] = required
    return {"$jsonSchema": schema}


NUM = {"bsonType": ["double", "int", "long", "decimal"]}
STR = {"bsonType": "string"}
DATE = {"bsonType": "date"}
INT = {"bsonType": ["int", "long"]}

COLLECTIONS: dict[str, tuple[Doc, list[IndexModel]]] = {
    "users": (
        _schema(
            ["username", "password_hash", "role", "active"],
            {
                "username": STR,
                "password_hash": STR,
                "role": {"enum": ["analyst", "admin"]},
                "active": {"bsonType": "bool"},
            },
        ),
        [IndexModel([("username", ASCENDING)], unique=True)],
    ),
    "api_keys": (
        _schema(
            ["name", "key_hash", "active"], {"name": STR, "key_hash": STR, "active": {"bsonType": "bool"}}
        ),
        [IndexModel([("key_hash", ASCENDING)], unique=True)],
    ),
    "transactions": (
        _schema(["txn_id", "dt", "row"], {"txn_id": INT, "dt": INT, "row": {"bsonType": "object"}}),
        [IndexModel([("txn_id", ASCENDING)], unique=True), IndexModel([("dt", ASCENDING)])],
    ),
    "predictions": (
        _schema(
            ["txn_id", "dt", "model_version", "probability", "decision", "amount", "latency_ms"],
            {
                "txn_id": INT,
                "dt": INT,
                "model_version": STR,
                "probability": NUM,
                "decision": {"enum": DECISIONS},
                "amount": NUM,
                "latency_ms": NUM,
                "reasons": {"bsonType": "array", "maxItems": 10},
            },
        ),
        [
            IndexModel([("txn_id", ASCENDING)], unique=True),
            IndexModel([("dt", ASCENDING)]),
            IndexModel([("model_version", ASCENDING), ("dt", ASCENDING)]),
        ],
    ),
    "cases": (
        _schema(
            ["txn_id", "status", "probability", "amount", "created_at"],
            {
                "txn_id": INT,
                "status": {"enum": CASE_STATUSES},
                "probability": NUM,
                "amount": NUM,
                "verdict": {"enum": ["fraud", "genuine", None]},
                "notes": {"bsonType": "array"},
                "created_at": DATE,
            },
        ),
        [
            IndexModel([("txn_id", ASCENDING)], unique=True),
            IndexModel([("status", ASCENDING), ("probability", DESCENDING)]),
            IndexModel([("assignee", ASCENDING), ("status", ASCENDING)]),
        ],
    ),
    "labels": (
        _schema(
            ["txn_id", "label", "source", "known_at"],
            {
                "txn_id": INT,
                "label": {"enum": [0, 1]},
                "source": {"enum": ["dataset", "analyst"]},
                "known_at": DATE,
            },
        ),
        [IndexModel([("txn_id", ASCENDING)], unique=True), IndexModel([("known_at", ASCENDING)])],
    ),
    "kpi_weekly": (_schema(["volume"], {"volume": INT}), []),
    "audit_log": (
        _schema(["actor", "action", "ts"], {"actor": STR, "action": STR, "ts": DATE}),
        [IndexModel([("ts", DESCENDING)]), IndexModel([("actor", ASCENDING), ("ts", DESCENDING)])],
    ),
    "policy": (_schema([], {}), []),
}


async def init_db(db: AsyncDatabase[Doc]) -> list[str]:
    """Create or update every collection's validator and indexes. Returns the collection names."""
    existing = set(await db.list_collection_names())
    for name, (validator, indexes) in COLLECTIONS.items():
        if name in existing:
            await db.command({"collMod": name, "validator": validator, "validationLevel": "moderate"})
        else:
            try:
                await db.create_collection(name, validator=validator, validationLevel="moderate")
            except CollectionInvalid:  # created concurrently
                await db.command({"collMod": name, "validator": validator, "validationLevel": "moderate"})
        if indexes:
            await db[name].create_indexes(indexes)
    return list(COLLECTIONS)


def now() -> datetime:
    return datetime.now(UTC)


async def audit(
    db: AsyncDatabase[Doc],
    actor: str,
    action: str,
    target: str,
    *,
    before: object = None,
    after: object = None,
) -> None:
    await db.audit_log.insert_one(
        {"actor": actor, "action": action, "target": target, "before": before, "after": after, "ts": now()}
    )
