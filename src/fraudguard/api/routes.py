"""HTTP routes, grouped by domain. Mounted under /api/v1."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from mlflow import MlflowClient
from pydantic import BaseModel, Field
from pymongo import ReturnDocument
from pymongo.errors import BulkWriteError

from fraudguard.api import db as dbm
from fraudguard.api.app import Admin, Analyst, ApiClient, State
from fraudguard.api.schemas import (
    BatchRequest,
    BatchResponse,
    CaseDetail,
    CasePage,
    CaseSummary,
    CaseUpdate,
    Health,
    KpiWeek,
    LoginRequest,
    Me,
    ModelVersion,
    PolicyModel,
    ReasonOut,
    RefreshRequest,
    ScoreResponse,
    TokenResponse,
    Transaction,
)
from fraudguard.api.scoring import ScoreResult
from fraudguard.api.security import DUMMY_HASH, verify_password
from fraudguard.data.load import SECONDS_PER_DAY
from fraudguard.decision.policy import Action, Policy
from fraudguard.features.online import OutOfOrderError
from fraudguard.log import get_logger
from fraudguard.models import champion

log = get_logger(__name__)
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY
CASE_SLA = timedelta(hours=24)

auth = APIRouter(prefix="/auth", tags=["auth"])
scoring = APIRouter(tags=["scoring"])
cases = APIRouter(prefix="/cases", tags=["cases"])
kpi = APIRouter(prefix="/kpi", tags=["kpi"])
admin = APIRouter(tags=["admin"])
health = APIRouter(tags=["health"])


# Auth -----------------------------------------------------------------------------------------


def _tokens(state: State, username: str, role: str) -> TokenResponse:
    return TokenResponse(
        access_token=state.tokens.issue(username, role, "access"),
        refresh_token=state.tokens.issue(username, role, "refresh"),
        role=role,
        expires_in=int(state.tokens.access.total_seconds()),
    )


@auth.post("/login")
async def login(body: LoginRequest, state: State) -> TokenResponse:
    user = await state.db.users.find_one({"username": body.username, "active": True})
    # Unknown usernames are checked against a dummy hash, so they take as long as wrong passwords.
    ok = verify_password(body.password, user["password_hash"] if user else DUMMY_HASH)
    if user is None or not ok:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong username or password")
    return _tokens(state, user["username"], user["role"])


@auth.post("/refresh")
async def refresh(body: RefreshRequest, state: State) -> TokenResponse:
    try:
        payload = state.tokens.decode(body.refresh_token, "refresh")
    except ValueError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid refresh token: {exc}") from exc
    user = await state.db.users.find_one({"username": payload["sub"], "active": True})
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found or disabled")
    return _tokens(state, user["username"], user["role"])


@auth.get("/me")
async def me(user: Analyst) -> Me:
    return Me(**user)


# Scoring --------------------------------------------------------------------------------------


def _response(r: ScoreResult) -> ScoreResponse:
    return ScoreResponse(
        transaction_id=r.txn_id,
        probability=r.probability,
        decision=r.decision.name.lower(),
        reasons=[ReasonOut(**x.as_dict()) for x in r.reasons],
        model_version=r.model_version,
        latency_ms=r.latency_ms,
        case_opened=r.decision == Action.REVIEW,
    )


async def _persist(state: State, results: list[tuple[ScoreResult, dict[str, Any]]]) -> None:
    """Write predictions, the raw transactions and new cases. Runs after the response is sent."""
    now = dbm.now()
    predictions, transactions, new_cases = [], [], []
    for r, row in results:
        reasons = [x.as_dict() for x in r.reasons]
        predictions.append(
            {
                "txn_id": r.txn_id,
                "dt": r.dt,
                "card_uid": r.card_uid,
                "model_version": r.model_version,
                "probability": r.probability,
                "decision": r.decision.name.lower(),
                "amount": r.amount,
                "reasons": reasons,
                "latency_ms": r.latency_ms,
                "scored_at": now,
            }
        )
        transactions.append({"txn_id": r.txn_id, "dt": r.dt, "row": row})
        if r.decision == Action.REVIEW:
            new_cases.append(
                {
                    "txn_id": r.txn_id,
                    "dt": r.dt,
                    "card_uid": r.card_uid,
                    "status": "open",
                    "probability": r.probability,
                    "amount": r.amount,
                    "decision": "review",
                    "reasons": reasons,
                    "features": r.features,
                    "contributions": r.contributions,
                    "assignee": None,
                    "verdict": None,
                    "notes": [],
                    "created_at": now,
                    "sla_due": now + CASE_SLA,
                    "model_version": r.model_version,
                }
            )
    writes = [
        state.db.predictions.insert_many(predictions, ordered=False),
        state.db.transactions.insert_many(transactions, ordered=False),
        *([state.db.cases.insert_many(new_cases, ordered=False)] if new_cases else []),
    ]
    for outcome in await asyncio.gather(*writes, return_exceptions=True):
        if isinstance(outcome, BulkWriteError):  # e.g. a transaction ID scored twice: keep the first
            log.warning("Some writes were skipped: %s", outcome.details.get("writeErrors", [])[:1])
        elif isinstance(outcome, BaseException):
            log.error("Failed to store scoring results: %r", outcome)


def _score_all(state: State, txns: list[Transaction]) -> list[tuple[ScoreResult, dict[str, Any]]]:
    """Score in time order. The whole request is checked first, so a bad batch changes nothing."""
    ordered = sorted(txns, key=lambda t: (t.TransactionDT, t.TransactionID))
    store = state.scorer.store
    if store.last_dt is not None and ordered[0].TransactionDT < store.last_dt:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"TransactionDT {ordered[0].TransactionDT} is older than the latest seen ({store.last_dt})",
        )
    counts = Counter(t.TransactionID for t in ordered)
    if dupes := sorted(i for i, n in counts.items() if n > 1 or i in store.ids)[:5]:
        raise HTTPException(status.HTTP_409_CONFLICT, f"transaction IDs already scored: {dupes}")
    out = []
    for txn in ordered:
        row = txn.row()
        try:
            out.append((state.scorer.score(row), row))
        except OutOfOrderError as exc:  # pragma: no cover - ruled out above
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return out


@scoring.post("/score")
async def score(txn: Transaction, state: State, client: ApiClient, tasks: BackgroundTasks) -> ScoreResponse:
    """Score one transaction: probability, approve/review/decline, top reasons. Review opens a case."""
    if not state.limiter.allow(client):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
    results = _score_all(state, [txn])
    tasks.add_task(_persist, state, results)
    return _response(results[0][0])


@scoring.post("/score/batch")
async def score_batch(
    body: BatchRequest, state: State, client: ApiClient, tasks: BackgroundTasks
) -> BatchResponse:
    """Score up to 1,000 transactions in time order (used by the replay worker)."""
    if not state.limiter.allow(client, len(body.transactions)):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
    results = _score_all(state, body.transactions)
    tasks.add_task(_persist, state, results)
    return BatchResponse(results=[_response(r) for r, _ in results])


# Cases ----------------------------------------------------------------------------------------


class CaseQuery(BaseModel):
    status: Literal["open", "in_review", "closed", "all"] = "open"
    min_probability: float = Field(default=0.0, ge=0, le=1)
    min_amount: float = Field(default=0.0, ge=0)
    assignee: str | None = None
    sort: Literal["probability", "amount", "created_at", "sla_due"] = "probability"
    order: Literal["asc", "desc"] = "desc"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=200)


@cases.get("")
async def list_cases(state: State, _: Analyst, q: Annotated[CaseQuery, Query()]) -> CasePage:
    """Review queue: filter by status, probability, amount and assignee; sort and paginate."""
    query: dict[str, Any] = {"probability": {"$gte": q.min_probability}, "amount": {"$gte": q.min_amount}}
    if q.status != "all":
        query["status"] = q.status
    if q.assignee:
        query["assignee"] = q.assignee
    total = await state.db.cases.count_documents(query)
    cursor = (
        state.db.cases.find(query, {"features": 0, "contributions": 0, "notes": 0})
        .sort([(q.sort, -1 if q.order == "desc" else 1), ("txn_id", 1)])
        .skip((q.page - 1) * q.page_size)
        .limit(q.page_size)
    )
    items = [CaseSummary(**doc) async for doc in cursor]
    return CasePage(items=items, total=total, page=q.page, page_size=q.page_size)


async def _case_or_404(state: State, txn_id: int) -> dict[str, Any]:
    doc = await state.db.cases.find_one({"txn_id": txn_id})
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no case for transaction {txn_id}")
    return doc


@cases.get("/{txn_id}")
async def get_case(txn_id: int, state: State, _: Analyst) -> CaseDetail:
    doc = await _case_or_404(state, txn_id)
    return CaseDetail(**doc, neighbourhood=state.scorer.store.graph.neighbourhood(doc["card_uid"]))


@cases.patch("/{txn_id}")
async def update_case(txn_id: int, body: CaseUpdate, state: State, user: Analyst) -> CaseDetail:
    """Assign, comment, change status, or record the verdict (which writes a label and closes the case)."""
    before = await _case_or_404(state, txn_id)
    if before["status"] == "closed" and body.verdict is not None and before.get("verdict"):
        raise HTTPException(status.HTTP_409_CONFLICT, "this case already has a verdict")
    now = dbm.now()
    set_: dict[str, Any] = {"updated_at": now}
    push: dict[str, Any] = {}
    if body.assignee is not None:
        set_["assignee"] = body.assignee or None
        set_.setdefault("status", "in_review" if before["status"] == "open" else before["status"])
    if body.status is not None:
        set_["status"] = body.status
    if body.note:
        push["notes"] = {"author": user["username"], "text": body.note, "at": now}
    if body.verdict is not None:
        set_.update(verdict=body.verdict, status="closed", closed_at=now, closed_by=user["username"])
        label = 1 if body.verdict == "fraud" else 0
        known_dt = state.scorer.store.last_dt or before["dt"]
        await state.db.labels.update_one(
            {"txn_id": txn_id},
            {
                "$set": {
                    "label": label,
                    "source": "analyst",
                    "known_at": now,
                    "known_dt": known_dt,
                    "card_uid": before["card_uid"],
                    "analyst": user["username"],
                }
            },
            upsert=True,
        )
        if label:
            state.scorer.store.graph.mark_fraud(before["card_uid"], known_dt)
    update: dict[str, Any] = {"$set": set_}
    if push:
        update["$push"] = push
    after = await state.db.cases.find_one_and_update(
        {"txn_id": txn_id}, update, return_document=ReturnDocument.AFTER
    )
    if after is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no case for transaction {txn_id}")
    changed = {k: before.get(k) for k in ("status", "assignee", "verdict")}
    await dbm.audit(
        state.db,
        user["username"],
        "case.update",
        f"case:{txn_id}",
        before=changed,
        after={k: after.get(k) for k in changed} | ({"note": body.note} if body.note else {}),
    )
    return CaseDetail(**after, neighbourhood=state.scorer.store.graph.neighbourhood(after["card_uid"]))


# KPIs -----------------------------------------------------------------------------------------


def kpi_pipeline(policy: Policy) -> list[dict[str, Any]]:
    """Weekly KPIs from predictions joined to labels; written to `kpi_weekly`."""

    def is_(decision: str) -> dict[str, Any]:
        return {"$eq": ["$decision", decision]}

    fraud, genuine = {"$eq": ["$label", 1]}, {"$eq": ["$label", 0]}
    return [
        {"$lookup": {"from": "labels", "localField": "txn_id", "foreignField": "txn_id", "as": "lab"}},
        {
            "$set": {
                "label": {"$first": "$lab.label"},
                "week": {"$toInt": {"$floor": {"$divide": ["$dt", SECONDS_PER_WEEK]}}},
            }
        },
        {
            "$group": {
                "_id": "$week",
                "volume": {"$sum": 1},
                "declined": {"$sum": {"$cond": [is_("decline"), 1, 0]}},
                "reviewed": {"$sum": {"$cond": [is_("review"), 1, 0]}},
                "labelled_fraud": {"$sum": {"$cond": [fraud, 1, 0]}},
                "caught": {"$sum": {"$cond": [{"$and": [fraud, {"$ne": ["$decision", "approve"]}]}, 1, 0]}},
                "fraud_loss": {"$sum": {"$cond": [{"$and": [fraud, is_("approve")]}, "$amount", 0]}},
                "false_declines": {"$sum": {"$cond": [{"$and": [genuine, is_("decline")]}, 1, 0]}},
                "p95": {"$percentile": {"input": "$latency_ms", "p": [0.95], "method": "approximate"}},
            }
        },
        {
            "$set": {
                "week": "$_id",
                "decline_rate": {"$divide": ["$declined", "$volume"]},
                "review_rate": {"$divide": ["$reviewed", "$volume"]},
                "recall": {
                    "$cond": [
                        {"$gt": ["$labelled_fraud", 0]},
                        {"$divide": ["$caught", "$labelled_fraud"]},
                        None,
                    ]
                },
                "cost": {
                    "$add": [
                        "$fraud_loss",
                        {"$multiply": ["$false_declines", policy.false_decline_cost]},
                        {"$multiply": ["$reviewed", policy.review_cost]},
                    ]
                },
                "p95_latency_ms": {"$first": "$p95"},
            }
        },
        {"$set": {"cost_per_10k": {"$multiply": [{"$divide": ["$cost", "$volume"]}, 10_000]}}},
        {"$unset": "p95"},
        {"$merge": {"into": "kpi_weekly", "whenMatched": "replace"}},
    ]


@kpi.get("/weekly")
async def weekly(state: State, _: Analyst) -> list[KpiWeek]:
    """Volume, decline and review rates, recall on labelled fraud, cost and p95 latency per week."""
    await state.db.predictions.aggregate(kpi_pipeline(state.scorer.policy))
    ref = datetime.combine(state.data_cfg.reference_date, datetime.min.time())
    return [
        KpiWeek(**{**doc, "week_start": ref + timedelta(weeks=doc["week"])})
        async for doc in state.db.kpi_weekly.find().sort("_id", 1)
    ]


# Admin: policy and models -----------------------------------------------------------------------


@admin.get("/policy")
async def get_policy(state: State, _: Analyst) -> PolicyModel:
    return PolicyModel(**state.scorer.policy.as_dict())


@admin.put("/policy")
async def put_policy(body: PolicyModel, state: State, user: Admin) -> PolicyModel:
    """Change cost assumptions or the review capacity. Takes effect immediately; audited."""
    before = state.scorer.policy.as_dict()
    state.scorer.policy = Policy(**body.model_dump())
    await state.db.policy.replace_one(
        {"_id": "current"},
        {**body.model_dump(), "updated_by": user["username"], "updated_at": dbm.now()},
        upsert=True,
    )
    await dbm.audit(
        state.db, user["username"], "policy.update", "policy", before=before, after=body.model_dump()
    )
    return body


def _versions(tracking_uri: str, serving: str) -> list[ModelVersion]:
    client = MlflowClient(tracking_uri=tracking_uri)
    # search_model_versions leaves aliases empty; the registered model holds alias -> version.
    aliases: dict[str, list[str]] = {}
    for alias, v in client.get_registered_model(champion.REGISTERED_NAME).aliases.items():
        aliases.setdefault(str(v), []).append(alias)
    out = []
    for mv in client.search_model_versions(f"name='{champion.REGISTERED_NAME}'"):
        metrics: dict[str, float] = {}
        if mv.run_id:
            run = client.get_run(mv.run_id)
            metrics = {k: v for k, v in run.data.metrics.items() if k.startswith(("valid_", "tuning_"))}
        out.append(
            ModelVersion(
                version=str(mv.version),
                aliases=sorted(aliases.get(str(mv.version), [])),
                run_id=mv.run_id,
                created_at=datetime.fromtimestamp(mv.creation_timestamp / 1000)
                if mv.creation_timestamp
                else None,
                metrics=metrics,
                serving=str(mv.version) == serving,
            )
        )
    return sorted(out, key=lambda m: int(m.version), reverse=True)


@admin.get("/models")
async def models(state: State, _: Admin) -> list[ModelVersion]:
    return await asyncio.to_thread(_versions, state.settings.mlflow_tracking_uri, state.scorer.version)


@admin.post("/models/{version}/promote")
async def promote(version: str, state: State, user: Admin) -> ModelVersion:
    """Point the champion alias at another version (promotion or rollback) and serve it now; audited."""
    uri = state.settings.mlflow_tracking_uri
    client = MlflowClient(tracking_uri=uri)
    try:
        await asyncio.to_thread(client.get_model_version, champion.REGISTERED_NAME, version)
    except Exception as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no model version {version}") from exc
    before = state.scorer.version
    await asyncio.to_thread(
        client.set_registered_model_alias, champion.REGISTERED_NAME, champion.CHAMPION_ALIAS, version
    )
    model = await asyncio.to_thread(champion.load_champion, uri)
    state.scorer.swap_model(model, version)
    await dbm.audit(
        state.db,
        user["username"],
        "model.promote",
        f"model:{champion.REGISTERED_NAME}",
        before={"version": before},
        after={"version": version},
    )
    return next(m for m in await asyncio.to_thread(_versions, uri, version) if m.version == version)


@health.get("/health")
async def get_health(state: State) -> Health:
    await state.client.admin.command("ping")
    return Health(
        status="ok",
        model_version=state.scorer.version,
        transactions_in_memory=state.scorer.store.n_seen,
        latest_transaction_dt=state.scorer.store.last_dt,
    )


ROUTERS = [health, auth, scoring, cases, kpi, admin]
