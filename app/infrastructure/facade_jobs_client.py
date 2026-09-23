"""Async client for the facade generation job service.

``facade-jobs`` owns the long-running GPU queue and object storage.  GenBuilder
only submits the generated building FeatureCollection and returns the job
handle to its caller; it never waits for a GLB in an HTTP request.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


DEFAULT_FACADE_PARAMS: dict[str, Any] = {
    "cluster_count": 12,
    "pixels_per_meter": 32,
    "seed": None,
}


class FacadeJobsError(RuntimeError):
    """A transport, protocol, or non-success response from ``facade-jobs``."""

    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"facade-jobs returned {status_code}: {body!r}")


class FacadeJobsClient:
    """Thin async wrapper over the ``facade-jobs`` REST API."""

    def __init__(
        self,
        base_url: str,
        *,
        public_base_url: str | None = None,
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise RuntimeError("FACADE_JOBS_API is not configured.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._base_url = base_url.rstrip("/")
        self._public_base_url = (public_base_url or base_url).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout_seconds,
            transport=transport,
        )

    async def __aenter__(self) -> "FacadeJobsClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._client.aclose()

    async def submit_job(
        self,
        buildings: dict[str, Any],
        *,
        requested_by: str | None,
        floor_height_m: float = 3.0,
        style_by_zone: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Queue one facade job and return a frontend-safe job handle."""
        payload = {
            "buildings": buildings,
            "floor_height_m": floor_height_m,
            # Empty styles mean "use facade-jobs zone defaults".
            "style_by_zone": style_by_zone or {},
            "params": dict(DEFAULT_FACADE_PARAMS if params is None else params),
            # The legacy by_territory endpoint is anonymous.  Keep those jobs in
            # one explicit quota bucket instead of inventing a user identity.
            "requested_by": requested_by or "anonymous",
        }

        try:
            response = await self._client.post("/jobs", json=payload)
        except httpx.HTTPError as exc:
            raise FacadeJobsError(0, str(exc)) from exc

        if response.status_code >= 400:
            try:
                body: Any = response.json()
            except ValueError:
                body = response.text
            raise FacadeJobsError(response.status_code, body)

        try:
            body = response.json()
        except ValueError as exc:
            raise FacadeJobsError(response.status_code, "response is not JSON") from exc
        if not isinstance(body, dict) or not body.get("job_id"):
            raise FacadeJobsError(
                response.status_code,
                "response does not contain a non-empty job_id",
            )

        job_id = str(body["job_id"])
        # Never expose the internal service hostname returned by a downstream
        # deployment.  The public/published base is controlled by GenBuilder.
        status_url = f"{self._public_base_url}/jobs/{quote(job_id, safe='')}"
        return {
            "job_id": job_id,
            "status": str(body.get("status") or "queued"),
            "status_url": str(status_url),
        }


__all__ = [
    "DEFAULT_FACADE_PARAMS",
    "FacadeJobsClient",
    "FacadeJobsError",
]
