import pytest

from app.dependencies import LENINGRAD_OBLAST_TERRITORY_ID, default_services_territory_id

ENV_KEY = "DEFAULT_SERVICES_TERRITORY_ID"


def test_unset_env_defaults_to_leningrad_oblast(monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)

    assert default_services_territory_id() == LENINGRAD_OBLAST_TERRITORY_ID == 1


def test_env_overrides_the_default_region(monkeypatch):
    monkeypatch.setenv(ENV_KEY, " 47 ")

    assert default_services_territory_id() == 47


def test_empty_env_disables_the_default_region(monkeypatch):
    monkeypatch.setenv(ENV_KEY, "")

    assert default_services_territory_id() is None


@pytest.mark.parametrize("raw", ["lo", "0", "-3"])
def test_invalid_env_disables_the_default_region(monkeypatch, raw):
    monkeypatch.setenv(ENV_KEY, raw)

    assert default_services_territory_id() is None
