import asyncio
import json

import httpx
import pytest

from app.infrastructure.facade_jobs_client import FacadeJobsClient, FacadeJobsError


BUILDINGS = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "id": "42",
            "properties": {"zone": "residential", "floors_count": 7},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[30.0, 60.0], [30.1, 60.0], [30.0, 60.0]]],
            },
        }
    ],
}


def test_submit_job_sends_contract_and_builds_public_status_url():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            202,
            json={
                "job_id": "job/42",
                "status": "queued",
                "status_url": "http://facade-jobs:8000/jobs/job/42",
            },
        )

    async def run() -> dict:
        async with FacadeJobsClient(
            "http://facade-jobs:8000/",
            public_base_url="https://api.example.test/facades/",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.submit_job(BUILDINGS, requested_by="user-7")

    result = asyncio.run(run())

    assert result == {
        "job_id": "job/42",
        "status": "queued",
        "status_url": "https://api.example.test/facades/jobs/job%2F42",
    }
    assert str(requests[0].url) == "http://facade-jobs:8000/jobs"
    assert json.loads(requests[0].content) == {
        "buildings": BUILDINGS,
        "floor_height_m": 3.0,
        "style_by_zone": {},
        "params": {
            "cluster_count": 12,
            "pixels_per_meter": 32,
            "seed": None,
        },
        "requested_by": "user-7",
    }


def test_submit_job_uses_anonymous_quota_for_legacy_territory_route():
    captured_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "anonymous-job"})

    async def run() -> None:
        async with FacadeJobsClient(
            "http://facade-jobs:8000",
            transport=httpx.MockTransport(handler),
        ) as client:
            await client.submit_job(BUILDINGS, requested_by=None)

    asyncio.run(run())
    assert captured_payload["requested_by"] == "anonymous"


@pytest.mark.parametrize("status_code", [400, 409, 422, 503])
def test_submit_job_preserves_downstream_error(status_code):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": "rejected"})

    async def run() -> None:
        async with FacadeJobsClient(
            "http://facade-jobs:8000",
            transport=httpx.MockTransport(handler),
        ) as client:
            await client.submit_job(BUILDINGS, requested_by="user-7")

    with pytest.raises(FacadeJobsError) as exc_info:
        asyncio.run(run())
    assert exc_info.value.status_code == status_code
    assert exc_info.value.body == {"detail": "rejected"}


def test_submit_job_rejects_response_without_job_id():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, json={"status": "queued"})

    async def run() -> None:
        async with FacadeJobsClient(
            "http://facade-jobs:8000",
            transport=httpx.MockTransport(handler),
        ) as client:
            await client.submit_job(BUILDINGS, requested_by="user-7")

    with pytest.raises(FacadeJobsError, match="job_id"):
        asyncio.run(run())


def test_submit_job_maps_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async def run() -> None:
        async with FacadeJobsClient(
            "http://facade-jobs:8000",
            transport=httpx.MockTransport(handler),
        ) as client:
            await client.submit_job(BUILDINGS, requested_by="user-7")

    with pytest.raises(FacadeJobsError) as exc_info:
        asyncio.run(run())
    assert exc_info.value.status_code == 0


def test_client_requires_configuration():
    with pytest.raises(RuntimeError, match="FACADE_JOBS_API"):
        FacadeJobsClient("")
