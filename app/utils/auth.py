"""Incoming request authentication.

The frontend now sends **Keycloak** access tokens (previously plain bearer
strings). This module verifies them against the Keycloak realm — signature
(via the realm JWKS), expiry and issuer — and exposes the caller identity.

Two dependencies are provided:

- ``verify_token`` returns the raw bearer string after verification. It stays
  ``str``-typed so the existing routers keep forwarding the *user* token to
  UrbanDB unchanged.
- ``get_current_user`` returns an :class:`AuthUser` (raw token + ``user_id``).
  The chat flow needs both: the raw token is forwarded to UrbanDB, and the
  ``user_id`` (Keycloak ``sub``) is passed to ChatStorage as ``X-User-Id`` when
  we authenticate to it with the *service* token.

Verification is gated on Keycloak being configured (``KEYCLOAK_URL`` +
``KEYCLOAK_REALM``) and can be disabled for local/dev via ``AUTH_VERIFY=false``
— mirroring the ChatStorage service. When disabled, claims are still read
(unverified) so ``user_id`` is available.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from loguru import logger

http_bearer = HTTPBearer()

_FALSEY = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class AuthUser:
    """Authenticated caller derived from the incoming Keycloak token."""

    token: str
    """Raw bearer string, forwarded downstream (e.g. to UrbanDB)."""

    user_id: str
    """Keycloak ``sub`` claim — the end-user id (``""`` if unavailable)."""

    claims: dict[str, Any] = field(default_factory=dict)


def _env(key: str) -> str | None:
    value = os.getenv(key)
    return value or None


def verification_enabled() -> bool:
    """True when Keycloak is configured and verification isn't switched off."""
    if (os.getenv("AUTH_VERIFY", "true").strip().lower()) in _FALSEY:
        return False
    return bool(_env("KEYCLOAK_URL")) and bool(_env("KEYCLOAK_REALM"))


def _realm_base() -> str:
    return f'{_env("KEYCLOAK_URL").rstrip("/")}/realms/{_env("KEYCLOAK_REALM")}'  # type: ignore[union-attr]


@lru_cache(maxsize=1)
def _jwks_client() -> PyJWKClient:
    # PyJWKClient caches the fetched signing keys and only re-fetches on key
    # rotation, so a single shared instance is enough.
    return PyJWKClient(f"{_realm_base()}/protocol/openid-connect/certs")


def _valid_audiences() -> list[str]:
    raw = os.getenv("AUTH_VALID_AUDIENCES") or ""
    return [a.strip() for a in raw.split(",") if a.strip()]


def _decode_verified(token: str) -> dict[str, Any]:
    """Verify signature/expiry/issuer against Keycloak and return the claims.

    Runs blocking JWKS/urllib work; call via ``asyncio.to_thread``.
    """
    signing_key = _jwks_client().get_signing_key_from_jwt(token).key
    audiences = _valid_audiences()
    return jwt.decode(
        token,
        signing_key,
        algorithms=["RS256"],
        issuer=_realm_base(),
        audience=audiences or None,
        options={"verify_aud": bool(audiences)},
    )


def _decode_unverified(token: str) -> dict[str, Any]:
    """Read claims without verifying — used only when verification is disabled."""
    try:
        return jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_aud": False,
            },
        )
    except jwt.PyJWTError:
        # Not a JWT (e.g. a legacy opaque token in dev): no claims to read.
        return {}


def _get_token_from_header(credentials: HTTPAuthorizationCredentials) -> str:
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing",
        )

    token = credentials.credentials

    if not token:
        raise HTTPException(
            status_code=400,
            detail="Token is missing in the authorization header"
        )

    return token


async def authenticate_token(token: str) -> AuthUser:
    """Verify a raw bearer token string and return the caller identity.

    Shared by the HTTP dependencies below and by non-FastAPI entry points
    (MCP tools, A2A requests) that receive the token outside of Starlette's
    ``HTTPAuthorizationCredentials`` flow.
    """
    if not token:
        raise HTTPException(status_code=400, detail="Token is missing in the authorization header")

    if not verification_enabled():
        claims = _decode_unverified(token)
        return AuthUser(token=token, user_id=str(claims.get("sub") or ""), claims=claims)

    try:
        claims = await asyncio.to_thread(_decode_verified, token)
    except jwt.PyJWTError as exc:
        logger.warning("Rejected token: {}", exc)
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - JWKS fetch / network failure
        logger.error("Token verification failed (Keycloak unreachable?): {}", exc)
        raise HTTPException(status_code=503, detail="Authentication provider unavailable") from exc

    return AuthUser(token=token, user_id=str(claims.get("sub") or ""), claims=claims)


async def _authenticate(credentials: HTTPAuthorizationCredentials) -> AuthUser:
    token = _get_token_from_header(credentials)
    return await authenticate_token(token)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(http_bearer)) -> str:
    """Verify the incoming Keycloak token and return the raw bearer string."""
    return (await _authenticate(credentials)).token


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
) -> AuthUser:
    """Verify the incoming Keycloak token and return the caller identity."""
    return await _authenticate(credentials)
