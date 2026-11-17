"""Application factory, startup and shared dependencies.

Startup:
1. connect to MongoDB and run `init_db` (validators and indexes);
2. load the `fraudguard@champion` model from the MLflow registry;
3. warm the online feature store with the history in the feature snapshot,
   then replay every transaction already scored live (from `transactions`)
   and re-apply analyst fraud verdicts, so a restart loses no state;
4. load the decision policy (an admin's saved policy wins over the model's).
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any

import polars as pl
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from fraudguard import __version__
from fraudguard.api import db as dbm
from fraudguard.api.scoring import Scorer, history_frame
from fraudguard.api.security import Role, TokenError, Tokens, hash_api_key
from fraudguard.config import (
    DataConfig,
    FeatureConfig,
    ModelConfig,
    Settings,
    load_data_config,
    load_feature_config,
    load_model_config,
)
from fraudguard.data import pipeline as data_pipeline
from fraudguard.data import snapshot
from fraudguard.decision.policy import Policy
from fraudguard.features.online import OnlineFeatureStore
from fraudguard.log import get_logger
from fraudguard.models import champion

log = get_logger(__name__)
SPLITS = ["train", "validation", "test"]
API = "/api/v1"
MIN_SECRET_LENGTH = 32


class RateLimiter:
    """Sliding one-minute window per API key, in memory."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, cost: int = 1) -> bool:
        now, q = time.monotonic(), self.hits[key]
        while q and q[0] <= now - 60:
            q.popleft()
        if len(q) + cost > self.per_minute:
            return False
        q.extend([now] * cost)
        return True


@dataclass
class AppState:
    settings: Settings
    data_cfg: DataConfig
    feat_cfg: FeatureConfig
    model_cfg: ModelConfig
    client: AsyncMongoClient[dbm.Doc]
    db: AsyncDatabase[dbm.Doc]
    scorer: Scorer
    tokens: Tokens
    limiter: RateLimiter
    started_at: float = field(default_factory=time.time)


async def _warm(
    state_store: OnlineFeatureStore, data_cfg: DataConfig, until: str, db: AsyncDatabase[dbm.Doc]
) -> None:
    base = snapshot.load_snapshot(data_cfg.paths.snapshot_dir, data_pipeline.STAGE, "transactions")
    splits = SPLITS[: SPLITS.index(until) + 1]
    t0 = time.perf_counter()
    state_store.warm(history_frame(base, splits))
    log.info(
        "Online store warmed with %s historical transactions (%s) in %.1fs",
        f"{state_store.n_seen:,}",
        "+".join(splits),
        time.perf_counter() - t0,
    )
    # Live transactions newer than the history; ones also in the history (same ID) are skipped.
    history_ids = set(base.filter(pl.col("split").is_in(splits))["TransactionID"].to_list())
    live = 0
    async for doc in db.transactions.find({"dt": {"$gte": state_store.last_dt or 0}}).sort(
        [("dt", 1), ("txn_id", 1)]
    ):
        if doc["txn_id"] in history_ids:
            continue
        state_store.add(doc["row"])
        live += 1
    async for lab in db.labels.find({"source": "analyst", "label": 1, "card_uid": {"$exists": True}}):
        state_store.graph.mark_fraud(lab["card_uid"], int(lab.get("known_dt", 0)))
    if live:
        log.info("Replayed %d live transactions from MongoDB", live)


def create_app(
    settings: Settings,
    data_cfg: DataConfig | None = None,
    feat_cfg: FeatureConfig | None = None,
    model_cfg: ModelConfig | None = None,
) -> FastAPI:
    data_cfg = data_cfg or load_data_config()
    feat_cfg = feat_cfg or load_feature_config()
    model_cfg = model_cfg or load_model_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client: AsyncMongoClient[dbm.Doc] = AsyncMongoClient(
            settings.mongo_uri, serverSelectionTimeoutMS=3000, tz_aware=True
        )
        db = client[settings.mongo_db]
        await client.admin.command("ping")
        await dbm.init_db(db)
        version, _ = champion.champion_version(settings.mlflow_tracking_uri)
        model = champion.load_champion(settings.mlflow_tracking_uri)
        store = OnlineFeatureStore(feat_cfg, data_cfg.reference_date)
        await _warm(store, data_cfg, settings.serve_history_until, db)
        saved = await db.policy.find_one({"_id": "current"})
        policy = Policy(**{k: saved[k] for k in model.policy.as_dict()}) if saved else model.policy
        secret = (settings.jwt_secret.get_secret_value() if settings.jwt_secret is not None else "") or None
        if secret is not None and len(secret) < MIN_SECRET_LENGTH:
            raise RuntimeError(f"FRAUDGUARD_JWT_SECRET must be at least {MIN_SECRET_LENGTH} characters")
        if secret is None:
            secret = secrets.token_urlsafe(48)
            log.warning(
                "FRAUDGUARD_JWT_SECRET is not set; using a random secret (tokens end with this process)"
            )
        app.state.fg = AppState(
            settings=settings,
            data_cfg=data_cfg,
            feat_cfg=feat_cfg,
            model_cfg=model_cfg,
            client=client,
            db=db,
            scorer=Scorer(model, version, store, policy, model_cfg.explain.n_reasons),
            tokens=Tokens(secret, settings.access_token_minutes, settings.refresh_token_days),
            limiter=RateLimiter(settings.score_rate_limit_per_minute),
        )
        log.info("FraudGuard API ready: model v%s, policy %s", version, policy.as_dict())
        try:
            yield
        finally:
            await client.close()

    app = FastAPI(
        title="FraudGuard API",
        version=__version__,
        description="Cost-aware fraud scoring with reason codes, review cases and KPIs.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "PUT"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )
    from fraudguard.api import routes  # noqa: PLC0415  (routes import this module's dependencies)

    for router in routes.ROUTERS:
        app.include_router(router, prefix=API)
    return app


# Dependencies -------------------------------------------------------------------------------


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.fg
    return state


State = Annotated[AppState, Depends(get_state)]
_bearer = HTTPBearer(auto_error=False)


async def current_user(
    state: State, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]
) -> dict[str, Any]:
    if creds is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing bearer token", {"WWW-Authenticate": "Bearer"}
        )
    try:
        payload = state.tokens.decode(creds.credentials, "access")
    except TokenError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, f"invalid token: {exc}", {"WWW-Authenticate": "Bearer"}
        ) from exc
    user = await state.db.users.find_one({"username": payload["sub"], "active": True})
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found or disabled")
    return {"username": user["username"], "role": user["role"]}


def require(role: Role) -> Callable[..., Any]:
    async def check(user: Annotated[dict[str, Any], Depends(current_user)]) -> dict[str, Any]:
        if role == Role.ADMIN and user["role"] != Role.ADMIN:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "admin role required")
        return user

    return check


Analyst = Annotated[dict[str, Any], Depends(require(Role.ANALYST))]
Admin = Annotated[dict[str, Any], Depends(require(Role.ADMIN))]


DEFAULT_SCOPES = ["score"]  # keys created before scopes existed can only score


def require_key(scope: str) -> Callable[..., Any]:
    """An active API key holding `scope` (score, labels or promote). Returns the key's name."""

    async def check(state: State, x_api_key: Annotated[str | None, Header()] = None) -> str:
        if not x_api_key:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing X-API-Key header")
        doc = await state.db.api_keys.find_one({"key_hash": hash_api_key(x_api_key), "active": True})
        if doc is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
        if scope not in doc.get("scopes", DEFAULT_SCOPES):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"this API key lacks the {scope!r} scope")
        return str(doc["name"])

    return check


ApiClient = Annotated[str, Depends(require_key("score"))]
LabelFeed = Annotated[str, Depends(require_key("labels"))]
Operator = Annotated[str, Depends(require_key("promote"))]
