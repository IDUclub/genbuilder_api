"""Services are placed only by a region's normatives; in the file mode the caller names the region."""
import asyncio

import pandas as pd
import pytest
from fastapi import HTTPException

from app.api.urbandb_api_gateway import NORMATIVE_COLUMNS, UrbanDBAPI
from app.logic.generation import Genbuilder

NORMATIVES = pd.DataFrame(
    [{"service_id": 21, "service_name": "Детский сад", "service_capacity": 61}]
)


class _FakeUrbanApi:
    def __init__(self, normatives=NORMATIVES, exc=None):
        self._normatives = normatives
        self._exc = exc
        self.scenario_calls: list[tuple] = []
        self.normatives_calls: list[tuple] = []

    async def get_territory_by_scenario(self, scenario_id, token):
        self.scenario_calls.append((scenario_id, token))
        return 7

    async def get_normatives_for_territory(self, territory_id, token=None):
        self.normatives_calls.append((territory_id, token))
        if self._exc is not None:
            raise self._exc
        return self._normatives


def _load(urban_api, scenario_id=None, territory_id=None, token=None):
    builder = Genbuilder(
        config=None,
        urban_api=urban_api,
        params_provider=None,
        residential_buildings_generator=None,
        residential_service_generator=None,
        buildings_params_provider=None,
        physical_objects_service=None,
    )
    return asyncio.run(builder._load_service_normatives(scenario_id, territory_id, token))


def _load_with_diagnostics(urban_api, scenario_id=None, territory_id=None, token=None):
    builder = Genbuilder(
        config=None,
        urban_api=urban_api,
        params_provider=None,
        residential_buildings_generator=None,
        residential_service_generator=None,
        buildings_params_provider=None,
        physical_objects_service=None,
    )
    return asyncio.run(
        builder._load_service_normatives_with_diagnostics(
            scenario_id, territory_id, token
        )
    )


def test_file_mode_loads_the_normatives_of_the_given_region_without_a_token():
    urban_api = _FakeUrbanApi()

    normatives = _load(urban_api, territory_id=1)

    assert normatives is NORMATIVES
    assert urban_api.normatives_calls == [(1, None)]
    assert urban_api.scenario_calls == []


def test_without_a_region_services_are_skipped_and_nothing_is_requested():
    urban_api = _FakeUrbanApi()

    assert _load(urban_api) is None
    assert urban_api.normatives_calls == []


def test_the_scenario_region_wins_over_the_given_one():
    urban_api = _FakeUrbanApi()

    _load(urban_api, scenario_id=198, territory_id=1, token="user-token")

    assert urban_api.scenario_calls == [(198, "user-token")]
    assert urban_api.normatives_calls == [(7, "user-token")]


def test_a_region_without_normatives_skips_services():
    urban_api = _FakeUrbanApi(normatives=pd.DataFrame(columns=NORMATIVE_COLUMNS))

    assert _load(urban_api, territory_id=1) is None


def test_a_region_without_normatives_has_an_explicit_diagnostic():
    urban_api = _FakeUrbanApi(normatives=pd.DataFrame(columns=NORMATIVE_COLUMNS))

    normatives, diagnostics = _load_with_diagnostics(urban_api, territory_id=17)

    assert normatives is None
    assert diagnostics == {
        "territory_id_provided": True,
        "territory_id": 17,
        "normatives_found": 0,
        "services_requested": 0,
        "services_placed": 0,
        "services_unplaced": 0,
        "service_buildings_placed": 0,
        "capacity_requested": 0.0,
        "capacity_placed": 0.0,
        "capacity_unplaced": 0.0,
        "status": "normatives_not_found",
        "warning": "для территории 17 нет нормативов",
    }


def test_file_mode_goes_on_without_services_when_urban_api_fails():
    urban_api = _FakeUrbanApi(exc=HTTPException(status_code=503, detail="down"))

    assert _load(urban_api, territory_id=1) is None


def test_an_unknown_region_has_an_explicit_diagnostic():
    urban_api = _FakeUrbanApi(exc=HTTPException(status_code=404, detail="not found"))

    normatives, diagnostics = _load_with_diagnostics(urban_api, territory_id=999)

    assert normatives is None
    assert diagnostics["status"] == "territory_not_found"
    assert diagnostics["warning"] == (
        "не удалось загрузить нормативы для территории 999: not found"
    )


def test_scenario_mode_failure_is_still_fatal():
    urban_api = _FakeUrbanApi(exc=HTTPException(status_code=503, detail="down"))

    with pytest.raises(HTTPException):
        _load(urban_api, scenario_id=198, token="user-token")


class _FakeConfig:
    @staticmethod
    def get(key):
        return "http://urban" if key == "UrbanDB_API" else None


class _FakeHandler:
    def __init__(self, payload):
        self.payload = payload
        self.calls: list[dict] = []

    async def request(self, method, url, session, expect_json=True, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": kwargs.get("headers")})
        return self.payload


def _gateway(payload):
    api = UrbanDBAPI(_FakeConfig())
    api.handler = _FakeHandler(payload)
    return api


def _normative(name, capacity):
    return {
        "service_type": {"id": 21, "name": name},
        "services_capacity_per_1000_normative": capacity,
    }


def test_a_region_without_normatives_gives_an_empty_table_instead_of_a_crash():
    api = _gateway([])

    normatives = asyncio.run(api.get_normatives_for_territory(1))

    assert normatives.empty
    assert list(normatives.columns) == NORMATIVE_COLUMNS


def test_normatives_without_capacity_are_dropped():
    api = _gateway([_normative("Детский сад", 61), _normative("Музей", None)])

    normatives = asyncio.run(api.get_normatives_for_territory(1))

    assert normatives["service_name"].tolist() == ["Детский сад"]


def test_normatives_are_requested_without_authorization_when_there_is_no_token():
    api = _gateway([])

    asyncio.run(api.get_normatives_for_territory(1))

    assert api.handler.calls[0]["headers"] == {}
    assert api.handler.calls[0]["url"].startswith("http://urban/api/v1/territory/1/normatives?")


def test_the_region_of_a_project_is_its_territory():
    api = _gateway({"project_id": 120, "territory": {"id": 1, "name": "Ленинградская область"}})

    region_id = asyncio.run(api.get_region_by_project(120, "user-token"))

    assert region_id == 1
    assert api.handler.calls[0]["url"] == "http://urban/api/v1/projects/120"
    assert api.handler.calls[0]["headers"] == {"Authorization": "Bearer user-token"}


def test_a_project_without_territory_is_not_found():
    api = _gateway({"project_id": 120})

    with pytest.raises(HTTPException) as caught:
        asyncio.run(api.get_region_by_project(120, "user-token"))

    assert caught.value.status_code == 404
