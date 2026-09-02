"""Protected admin API for persistent runtime environment overrides."""

from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.infrastructure import config_runtime as runtime
from app.settings import (
    Settings,
    _build_settings_cached,
    get_settings,
    settings_env_keys,
)


router = APIRouter(prefix="/admin/config", tags=["admin"])

_SECRET_SETTING_FIELDS = frozenset(
    {
        "admin_api_token",
        "fileserver_access_key",
        "fileserver_secret_key",
        "keycloak_client_secret",
    }
)


class ConfigValueIn(BaseModel):
    value: str = Field(
        ..., max_length=16_384, description="New string value for the env key"
    )
    updated_by: str | None = Field(
        default=None, max_length=128, description="Optional audit label"
    )


def verify_admin(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    settings: Settings = Depends(get_settings),
) -> bool:
    expected = settings.admin_api_token
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin config API is disabled (ADMIN_API_TOKEN not set)",
        )
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token")
    return True


def _masked(key: str, value: str | None) -> str | None:
    if value and runtime.is_secret(key):
        return "***"
    return value


def _key_view(key: str) -> dict[str, Any]:
    if key not in settings_env_keys():
        raise HTTPException(status_code=404, detail=f"Unknown config key '{key}'")
    overrides = {item["key"]: item for item in runtime.list_overrides()}
    override = overrides.get(key)
    return {
        "key": key,
        "effective": _masked(key, os.environ.get(key)),
        "overridden": override is not None,
        "override_value": _masked(key, override["value"] if override else None),
        "overridable": runtime.is_overridable(key),
        "updated_at": override["updated_at"] if override else None,
        "updated_by": override["updated_by"] if override else None,
    }


def _validation_error(key: str, value: str) -> str | None:
    sentinel = object()
    with runtime._lock:
        old = os.environ.get(key, sentinel)
        os.environ[key] = value
        try:
            _build_settings_cached.cache_clear()
            _build_settings_cached()
            return None
        except Exception as exc:  # noqa: BLE001 - returned as a safe 400 detail
            return str(exc)
        finally:
            if old is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
            _build_settings_cached.cache_clear()


@router.get("/settings", dependencies=[Depends(verify_admin)])
def get_resolved_settings() -> dict[str, Any]:
    data = get_settings().model_dump()
    for field in _SECRET_SETTING_FIELDS:
        if data.get(field):
            data[field] = "***"
    return data


@router.get("/overrides", dependencies=[Depends(verify_admin)])
def get_active_overrides() -> dict[str, Any]:
    items = runtime.list_overrides()
    for item in items:
        item["value"] = _masked(item["key"], item["value"])
    return {"count": len(items), "overrides": items}


@router.post("/reload", dependencies=[Depends(verify_admin)])
def reload_overrides() -> dict[str, Any]:
    changed = runtime.apply_overrides(force=True)
    return {"reloaded": True, "changed": changed}


@router.get("/{key}", dependencies=[Depends(verify_admin)])
def get_config_key(key: str) -> dict[str, Any]:
    return _key_view(key)


@router.put("/{key}", dependencies=[Depends(verify_admin)])
def put_config_key(key: str, body: ConfigValueIn) -> dict[str, Any]:
    if not runtime.is_overridable(key):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Key '{key}' is not overridable "
                "(unknown, sensitive, boot-only, or absent from deployed env)."
            ),
        )
    error = _validation_error(key, body.value)
    if error is not None:
        raise HTTPException(
            status_code=400, detail=f"Value rejected for '{key}': {error}"
        )
    runtime.set_override(key, body.value, updated_by=body.updated_by)
    return _key_view(key)


@router.delete("/{key}", dependencies=[Depends(verify_admin)])
def delete_config_key(key: str) -> dict[str, Any]:
    if not runtime.delete_override(key):
        raise HTTPException(status_code=404, detail=f"No override set for '{key}'")
    return _key_view(key)
