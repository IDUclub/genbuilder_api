"""Runtime config store and protected admin API regression tests."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def runtime_env(monkeypatch, tmp_path):
    from app.infrastructure import config_runtime as runtime
    from app.settings import _build_settings_cached

    monkeypatch.setenv(
        "RUNTIME_CONFIG_PATH", str(tmp_path / "runtime-overrides.sqlite3")
    )
    monkeypatch.setenv("RUNTIME_CONFIG_SYNC_TTL_SECONDS", "0.1")
    monkeypatch.setenv("ADMIN_API_TOKEN", "local-admin-secret")
    monkeypatch.setenv("LLM_API", "http://original-llm:8000")
    monkeypatch.setenv("UrbanDB_YEAR", "2024")
    runtime._applied.clear()
    runtime._baseline.clear()
    runtime._last_sync = 0.0
    _build_settings_cached.cache_clear()
    yield
    runtime._applied.clear()
    runtime._baseline.clear()
    runtime._last_sync = 0.0
    _build_settings_cached.cache_clear()


def test_sync_applies_and_restores_deployed_value(runtime_env):
    from app.infrastructure import config_runtime as runtime

    runtime.set_override("LLM_API", "http://runtime-llm:8000", "pytest")
    assert os.environ["LLM_API"] == "http://runtime-llm:8000"
    assert runtime.list_overrides()[0]["updated_by"] == "pytest"

    assert runtime.delete_override("LLM_API") is True
    assert os.environ["LLM_API"] == "http://original-llm:8000"


def test_overridable_rules(runtime_env):
    from app.infrastructure import config_runtime as runtime

    assert runtime.is_overridable("LLM_API") is True
    assert runtime.is_overridable("ADMIN_API_TOKEN") is False
    assert runtime.is_overridable("KEYCLOAK_CLIENT_SECRET") is False
    assert runtime.is_overridable("UNKNOWN_CONFIG_KEY") is False


def test_admin_api_auth_masking_and_override_lifecycle(runtime_env):
    from app.main import app

    client = TestClient(app)
    assert client.get("/admin/config/settings").status_code == 401

    headers = {"X-Admin-Token": "local-admin-secret"}
    settings_response = client.get("/admin/config/settings", headers=headers)
    assert settings_response.status_code == 200
    assert settings_response.json()["admin_api_token"] == "***"
    secret_response = client.get("/admin/config/ADMIN_API_TOKEN", headers=headers)
    assert secret_response.status_code == 200
    assert secret_response.json()["effective"] == "***"
    assert client.get("/admin/config/PATH", headers=headers).status_code == 404

    put_response = client.put(
        "/admin/config/LLM_API",
        headers=headers,
        json={"value": "http://runtime-llm:8000", "updated_by": "pytest"},
    )
    assert put_response.status_code == 200
    assert put_response.json()["effective"] == "http://runtime-llm:8000"
    assert put_response.json()["overridden"] is True

    list_response = client.get("/admin/config/overrides", headers=headers)
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1

    delete_response = client.delete("/admin/config/LLM_API", headers=headers)
    assert delete_response.status_code == 200
    assert delete_response.json()["effective"] == "http://original-llm:8000"
    assert delete_response.json()["overridden"] is False


def test_admin_api_rejects_invalid_typed_value(runtime_env):
    from app.main import app

    response = TestClient(app).put(
        "/admin/config/UrbanDB_YEAR",
        headers={"X-Admin-Token": "local-admin-secret"},
        json={"value": "not-a-year"},
    )
    assert response.status_code == 400
    assert "Value rejected" in response.json()["detail"]
