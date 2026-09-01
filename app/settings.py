"""Typed application settings resolved from the process environment.

Repository-managed configuration is injected into the container as environment
variables. Runtime overrides are applied on top of that environment by
``app.infrastructure.config_runtime`` before this model is built.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Environment variables used by GenBuilder.

    Aliases intentionally preserve the historic spelling of the deployed env
    keys (``UrbanDB_API``, ``Chat_Model`` and ``ChatStorage_API`` included).
    """

    model_config = ConfigDict(populate_by_name=True)

    log_file: str = Field(default="genbuilder_logs", alias="LOG_FILE")
    urban_db_api: str = Field(default="", alias="UrbanDB_API")
    urban_db_source: str = Field(default="OSM", alias="UrbanDB_SOURCE")
    urban_db_year: int = Field(default=2024, alias="UrbanDB_YEAR")
    prometheus_port: int = Field(
        default=9464, ge=1, le=65535, alias="PROMETHEUS_PORT"
    )

    chat_storage_api: str = Field(default="", alias="ChatStorage_API")
    chat_model: str = Field(default="", alias="Chat_Model")
    llm_api: str = Field(default="", alias="LLM_API")

    keycloak_url: str = Field(default="", alias="KEYCLOAK_URL")
    keycloak_realm: str = Field(default="", alias="KEYCLOAK_REALM")
    keycloak_client_id: str = Field(default="", alias="KEYCLOAK_CLIENT_ID")
    keycloak_client_secret: str = Field(default="", alias="KEYCLOAK_CLIENT_SECRET")
    keycloak_scope: str = Field(default="", alias="KEYCLOAK_SCOPE")

    auth_verify: bool = Field(default=True, alias="AUTH_VERIFY")
    auth_valid_audiences: str = Field(default="", alias="AUTH_VALID_AUDIENCES")

    a2a_public_url: str = Field(
        default="http://localhost:8000", alias="A2A_PUBLIC_URL"
    )
    public_base_url: str = Field(default="", alias="PUBLIC_BASE_URL")

    fileserver_endpoint: str = Field(default="", alias="FILESERVER_ENDPOINT")
    fileserver_access_key: str = Field(default="", alias="FILESERVER_ACCESS_KEY")
    fileserver_secret_key: str = Field(default="", alias="FILESERVER_SECRET_KEY")
    fileserver_bucket_name: str = Field(default="", alias="FILESERVER_BUCKET_NAME")
    fileserver_secure: bool = Field(default=False, alias="FILESERVER_SECURE")
    fileserver_region: str = Field(default="us-east-1", alias="FILESERVER_REGION")
    outputs_dir: str = Field(default="outputs", alias="OUTPUTS_DIR")

    admin_api_token: str = Field(default="", alias="ADMIN_API_TOKEN")
    runtime_config_path: str = Field(
        default="runtime_config/overrides.sqlite3", alias="RUNTIME_CONFIG_PATH"
    )
    runtime_config_sync_ttl_seconds: float = Field(
        default=5.0, ge=0.1, alias="RUNTIME_CONFIG_SYNC_TTL_SECONDS"
    )


def settings_env_keys() -> frozenset[str]:
    """Return the exact env key aliases represented by :class:`Settings`."""
    return frozenset(
        str(field.alias or name) for name, field in Settings.model_fields.items()
    )


@lru_cache(maxsize=1)
def _build_settings_cached() -> Settings:
    values: dict[str, Any] = {}
    for name, field in Settings.model_fields.items():
        env_key = str(field.alias or name)
        if env_key in os.environ:
            values[env_key] = os.environ[env_key]
    return Settings.model_validate(values)


def get_settings() -> Settings:
    """Return effective typed settings after syncing live overrides."""
    from app.infrastructure.config_runtime import apply_overrides

    apply_overrides()
    return _build_settings_cached()
