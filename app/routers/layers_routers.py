"""Endpoints serving the geo layers referenced from the chat stream and history.

Two link kinds, deliberately different:

- ``/layers/functional_zones`` re-reads the zones from UrbanDB on every call.
  Nothing is stored, and the caller's own token is forwarded so UrbanDB keeps
  enforcing access to a private scenario. The URL holds scenario coordinates,
  not a capability.
- ``/files/{slot}/{result_id}`` serves our own generation artefacts from object
  storage. Bytes are streamed through the API rather than redirected to a
  presigned URL: object storage sits on a private network the browser cannot
  reach.
"""
from __future__ import annotations

import re
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import StreamingResponse

from app.exceptions.http_exception_wrapper import http_exception
from app.infrastructure.object_storage import ObjectStorage, get_object_storage
from app.logic.functional_zones_service import FunctionalZonesService
from app.logic.geo_layers import FILE_SLOTS, MIME_TYPE, object_key
from app.utils import auth

layers_router = APIRouter()

_RESULT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def get_zones_service() -> FunctionalZonesService:
    """Injected so the router imports cleanly without application wiring."""
    from app.dependencies import zones_service

    return zones_service


@layers_router.get(
    "/layers/functional_zones",
    summary="Functional zones layer for a scenario, as generation sees it",
)
async def functional_zones_layer(
    scenario_id: Annotated[int, Query(..., ge=1, description="Scenario ID", examples=[198])],
    year: Annotated[int, Query(..., description="Data year", examples=[2024])],
    source: Annotated[str, Query(..., description="Data source", examples=["OSM"])],
    functional_zone_types: Annotated[
        Optional[List[str]],
        Query(
            description="Zone types to keep; all zones are returned when omitted.",
            examples=[["residential", "business"]],
        ),
    ] = None,
    token: str = Depends(auth.verify_token),
    zones: FunctionalZonesService = Depends(get_zones_service),
) -> dict:
    return await zones.prepare_zones_layer(
        scenario_id=scenario_id,
        year=year,
        source=source,
        token=token,
        functional_zone_types=functional_zone_types,
    )


@layers_router.get(
    "/files/{slot}/{result_id}",
    summary="Download a stored generation layer",
    response_class=StreamingResponse,
)
def generation_file(
    slot: Annotated[str, Path(description=f"One of: {', '.join(FILE_SLOTS)}")],
    result_id: Annotated[str, Path(description="Generation result id (uuid4 hex)")],
    token: str = Depends(auth.verify_token),
    storage: ObjectStorage = Depends(get_object_storage),
) -> StreamingResponse:
    """Stream one stored layer.

    Declared ``def`` on purpose: the object storage client is synchronous, so
    FastAPI runs this in a worker thread instead of blocking the event loop.
    """
    if slot not in FILE_SLOTS:
        raise http_exception(404, f"Unknown layer slot '{slot}'")
    if not _RESULT_ID_RE.match(result_id):
        raise http_exception(404, "Unknown generation result")

    key = object_key(result_id, slot)
    if not storage.exists(key):
        raise http_exception(
            404,
            "Layer is no longer available: generation results are kept for a "
            "limited time.",
        )

    return StreamingResponse(
        storage.open_stream(key),
        media_type=MIME_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="{slot}.geojson"',
        },
    )
