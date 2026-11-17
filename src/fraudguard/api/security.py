"""Passwords, tokens and API keys.

- Passwords are hashed with Argon2id; only the hash is stored.
- People get a short-lived JWT access token plus a longer refresh token.
- Machine clients (the replay worker, the landing-page demo) use API keys.
  Only a SHA-256 hash of each key is stored; the key itself is shown once.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

ALGORITHM = "HS256"
API_KEY_PREFIX = "fg_"
_hasher = PasswordHasher()


class Role(StrEnum):
    ANALYST = "analyst"
    ADMIN = "admin"


class TokenError(ValueError):
    pass


def hash_password(password: str) -> str:
    return _hasher.hash(password)


# Checked when a username does not exist, so a failed login takes as long either way.
DUMMY_HASH = _hasher.hash(secrets.token_hex(16))


def verify_password(password: str, hashed: str) -> bool:
    try:
        return _hasher.verify(hashed, password)
    except (VerificationError, InvalidHashError):
        return False


def new_api_key() -> tuple[str, str]:
    """(key shown to the user once, hash to store)."""
    key = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return key, hash_api_key(key)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Tokens:
    def __init__(self, secret: str, access_minutes: int, refresh_days: int) -> None:
        self.secret = secret
        self.access = timedelta(minutes=access_minutes)
        self.refresh = timedelta(days=refresh_days)

    def issue(self, username: str, role: str, kind: str = "access") -> str:
        now = datetime.now(UTC)
        payload = {
            "sub": username,
            "role": role,
            "type": kind,
            "iat": now,
            "exp": now + (self.access if kind == "access" else self.refresh),
            "jti": secrets.token_hex(8),
        }
        return jwt.encode(payload, self.secret, algorithm=ALGORITHM)

    def decode(self, token: str, kind: str = "access") -> dict[str, Any]:
        try:
            payload: dict[str, Any] = jwt.decode(token, self.secret, algorithms=[ALGORITHM])
        except jwt.PyJWTError as exc:
            raise TokenError(str(exc)) from exc
        if payload.get("type") != kind:
            raise TokenError(f"expected a {kind} token")
        return payload
