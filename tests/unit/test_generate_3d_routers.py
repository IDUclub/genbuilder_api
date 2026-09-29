"""Router-level coverage for the /generate/3d/* endpoints.

These only pin the HTTP plumbing (query/body parsing, auth wiring, status
code, response shape) by mocking the orchestration functions the router
calls directly. The orchestration logic itself (building generation, facade
style resolution, facade-jobs submission) is covered separately in
test_facade_styles.py and test_facade_jobs_client.py.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import generation_routers
from app.utils import auth

BLOCKS_PAYLOAD = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"zone": "residential"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[30.0, 60.0], [30.1, 60.0], [30.1, 60.1], [30.0, 60.0]]],
            },
        }
    ],
}

FACADE_JOB_RESPONSE = {
    "job_id": "job-1",
    "status_url": "http://facade-jobs/status/job-1",
    "facade_style": "кирпичный",
}


def _client(*, authenticated: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(generation_routers.generation_router)
    if authenticated:
        app.dependency_overrides[auth.get_current_user] = lambda: auth.AuthUser(
            token="user-token", user_id="user-1"
        )
    return TestClient(app)


def test_generate_3d_by_scenario_queues_facade_job(monkeypatch):
    calls: list[dict] = []

    async def _fake(**kwargs):
        calls.append(kwargs)
        return FACADE_JOB_RESPONSE

    monkeypatch.setattr(generation_routers.orchestration, "generate_3d_by_scenario", _fake)

    response = _client().post(
        "/generate/3d/by_scenario",
        params={
            "scenario_id": 198,
            "year": 2024,
            "source": "OSM",
            "functional_zone_types": ["residential", "business"],
            "facade_style": "кирпичный",
        },
    )

    assert response.status_code == 202
    assert response.json() == FACADE_JOB_RESPONSE
    assert len(calls) == 1
    assert calls[0]["scenario_id"] == 198
    assert calls[0]["year"] == 2024
    assert calls[0]["source"] == "OSM"
    assert calls[0]["functional_zone_types"] == ["residential", "business"]
    assert calls[0]["facade_style"] == "кирпичный"
    assert calls[0]["token"] == "user-token"
    assert calls[0]["requested_by"] == "user-1"


def test_generate_3d_by_territory_queues_facade_job_without_auth(monkeypatch):
    calls: list[tuple] = []

    async def _fake(payload, *, requested_by=None, facade_style=None):
        calls.append((payload, requested_by, facade_style))
        return FACADE_JOB_RESPONSE

    monkeypatch.setattr(generation_routers.orchestration, "generate_3d_by_territory", _fake)

    response = _client(authenticated=False).post(
        "/generate/3d/by_territory",
        params={"facade_style": "скандинавский"},
        json={"blocks": BLOCKS_PAYLOAD},
    )

    assert response.status_code == 202
    assert response.json() == FACADE_JOB_RESPONSE
    assert len(calls) == 1
    payload, requested_by, facade_style = calls[0]
    assert payload.blocks.features[0].properties.zone == "residential"
    # The route intentionally forwards no caller identity to this endpoint.
    assert requested_by is None
    assert facade_style == "скандинавский"


def test_generate_3d_by_blocks_queues_facade_job(monkeypatch):
    calls: list[dict] = []

    async def _fake(**kwargs):
        calls.append(kwargs)
        return FACADE_JOB_RESPONSE

    monkeypatch.setattr(generation_routers.orchestration, "generate_3d_by_blocks", _fake)

    body = {
        "zones": [
            {
                "functional_zone_id": 6679027,
                "targets_by_zone": {"residents": {"residential": 1000}},
            }
        ]
    }

    response = _client().post(
        "/generate/3d/by_blocks",
        params={
            "scenario_id": 198,
            "year": 2024,
            "source": "OSM",
            "functional_zone_types": ["residential"],
        },
        json=body,
    )

    assert response.status_code == 202
    assert response.json() == FACADE_JOB_RESPONSE
    assert len(calls) == 1
    assert calls[0]["scenario_id"] == 198
    assert calls[0]["functional_zone_types"] == ["residential"]
    assert calls[0]["token"] == "user-token"
    assert calls[0]["requested_by"] == "user-1"
    assert calls[0]["facade_style"] is None
    assert calls[0]["body"].zones[0].functional_zone_id == 6679027


def test_generate_3d_by_scenario_requires_authentication(monkeypatch):
    async def _fake(**kwargs):
        raise AssertionError("orchestration must not run without authentication")

    monkeypatch.setattr(generation_routers.orchestration, "generate_3d_by_scenario", _fake)

    response = _client(authenticated=False).post(
        "/generate/3d/by_scenario",
        params={
            "scenario_id": 198,
            "year": 2024,
            "source": "OSM",
            "functional_zone_types": ["residential"],
        },
    )

    assert response.status_code in (401, 403)
