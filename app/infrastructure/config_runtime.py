"""Persistent runtime env overrides shared by all application processes.

The deployed environment remains the baseline. Overrides are stored in a small
SQLite database on a Docker volume, synced into each process periodically and
removed cleanly to restore the original deployed value.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.settings import settings_env_keys


_DENY: frozenset[str] = frozenset(
    {
        # Credentials.
        "ADMIN_API_TOKEN",
        "FILESERVER_ACCESS_KEY",
        "FILESERVER_SECRET_KEY",
        "KEYCLOAK_CLIENT_SECRET",
        # Boot-only settings and service-token client configuration.
        "APP_ENV",
        "LOG_FILE",
        "PROMETHEUS_PORT",
        "RUNTIME_CONFIG_PATH",
        "RUNTIME_CONFIG_SYNC_TTL_SECONDS",
        "KEYCLOAK_URL",
        "KEYCLOAK_REALM",
        "KEYCLOAK_CLIENT_ID",
        "KEYCLOAK_SCOPE",
    }
)

_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "ADMIN_API_TOKEN",
        "FILESERVER_ACCESS_KEY",
        "FILESERVER_SECRET_KEY",
        "KEYCLOAK_CLIENT_SECRET",
    }
)

_lock = threading.RLock()
_last_sync = 0.0
_applied: dict[str, str] = {}
_baseline: dict[str, str | None] = {}


def is_secret(key: str) -> bool:
    return key in _SECRET_KEYS


def is_overridable(key: str) -> bool:
    """Only known, deployed, non-sensitive and live-applicable keys are mutable."""
    return key in settings_env_keys() and key not in _DENY and key in os.environ


def _store_path() -> Path:
    return Path(
        os.getenv("RUNTIME_CONFIG_PATH", "runtime_config/overrides.sqlite3")
    ).resolve()


def _connect() -> sqlite3.Connection:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS config_override (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            updated_by TEXT
        )
        """
    )
    return connection


@contextmanager
def _connection():
    """Commit or roll back a store operation and always close its connection."""
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _load_from_store() -> dict[str, str]:
    with _connection() as connection:
        rows = connection.execute(
            "SELECT key, value FROM config_override"
        ).fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def _sync_env(overrides: dict[str, str]) -> bool:
    """Reconcile this process env with stored overrides."""
    changed = False
    known_keys = settings_env_keys()
    for key, value in overrides.items():
        if key in _DENY or key not in known_keys:
            continue
        if _applied.get(key) != value:
            if key not in _baseline:
                _baseline[key] = os.environ.get(key)
            os.environ[key] = value
            _applied[key] = value
            changed = True

    for key in list(_applied):
        if key not in overrides:
            baseline = _baseline.pop(key, None)
            if baseline is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = baseline
            _applied.pop(key, None)
            changed = True
    return changed


def _invalidate_derived_caches() -> None:
    from app.settings import _build_settings_cached

    _build_settings_cached.cache_clear()

    # These objects are derived from env values and otherwise live forever.
    try:
        from app.infrastructure.object_storage import get_object_storage

        get_object_storage.cache_clear()
    except ImportError:
        pass
    try:
        from app.utils.auth import _jwks_client

        _jwks_client.cache_clear()
    except ImportError:
        pass


def apply_overrides(force: bool = False) -> bool:
    """Sync overrides into ``os.environ``; storage failures keep baseline config."""
    global _last_sync
    now = time.monotonic()
    ttl = float(os.getenv("RUNTIME_CONFIG_SYNC_TTL_SECONDS", "5"))
    if not force and now - _last_sync < ttl:
        return False

    with _lock:
        now = time.monotonic()
        if not force and now - _last_sync < ttl:
            return False
        try:
            overrides = _load_from_store()
        except (OSError, sqlite3.Error):
            _last_sync = now
            return False
        changed = _sync_env(overrides)
        _last_sync = now

    if changed:
        _invalidate_derived_caches()
    return changed


def list_overrides() -> list[dict[str, Any]]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT key, value, updated_at, updated_by
            FROM config_override
            ORDER BY key
            """
        ).fetchall()
    return [dict(row) for row in rows]


def set_override(key: str, value: str, updated_by: str | None = None) -> None:
    updated_at = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO config_override (key, value, updated_at, updated_by)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at,
                updated_by = excluded.updated_by
            """,
            (key, value, updated_at, updated_by),
        )
    apply_overrides(force=True)


def delete_override(key: str) -> bool:
    with _connection() as connection:
        cursor = connection.execute(
            "DELETE FROM config_override WHERE key = ?", (key,)
        )
        existed = cursor.rowcount > 0
    apply_overrides(force=True)
    return existed
