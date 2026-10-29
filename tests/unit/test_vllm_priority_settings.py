"""vLLM priority settings use the same shared matrix as PzzCompareAPI."""

import pytest
from pydantic import ValidationError

from app.settings import Settings, _build_settings_cached, settings_env_keys


def test_priority_defaults_and_runtime_config_keys():
    settings = Settings()

    assert settings.vllm_request_priority == -60
    assert settings.vllm_chat_priority == -100
    assert {"VLLM_REQUEST_PRIORITY", "VLLM_CHAT_PRIORITY"} <= settings_env_keys()


@pytest.mark.parametrize("request_priority,chat_priority", [(-60, -100), (-20, -80), (0, 0)])
def test_priorities_load_from_environment(monkeypatch, request_priority, chat_priority):
    monkeypatch.setenv("VLLM_REQUEST_PRIORITY", str(request_priority))
    monkeypatch.setenv("VLLM_CHAT_PRIORITY", str(chat_priority))
    _build_settings_cached.cache_clear()
    try:
        settings = _build_settings_cached()
        assert settings.vllm_request_priority == request_priority
        assert settings.vllm_chat_priority == chat_priority
    finally:
        _build_settings_cached.cache_clear()


@pytest.mark.parametrize("key", ["VLLM_REQUEST_PRIORITY", "VLLM_CHAT_PRIORITY"])
@pytest.mark.parametrize("value", ["not-an-integer", str(2**63), str(-(2**63) - 1)])
def test_invalid_priority_is_rejected(key, value):
    with pytest.raises(ValidationError):
        Settings.model_validate({key: value})
