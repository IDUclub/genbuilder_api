"""SSE endpoint for conversational building generation.

The frontend posts the user's free-text request plus the territory reference
(scenario_id + year + source, or an uploaded ``blocks_file``). The server
extracts the generation targets, and either streams back a ``clarification``
asking for the missing parameters, or runs generation inline and streams
progress → result → summary.

In the project-less mode (``blocks_file``, no scenario) the clarification also
carries the optional existing-buildings question: the frontend answers it with
``buildings_file`` (their footprints are excluded from generation) or with
``skip_existing_buildings=true``.

Response is ``text/event-stream`` (sse-starlette). Each event carries the
envelope ``type`` as the SSE ``event`` field and the rest of the payload as JSON
``data`` — mirroring IDUclub/PzzCompareAPI's ``*/chat/stream`` handlers.
"""
from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.dependencies import (
    CHAT_LA_PER_PERSON,
    build_chat_storage_client,
    build_vllm_chat_client,
    builder,
    chat_llm_configured,
    optional_object_storage,
    public_base_url,
    zones_service,
)
from app.exceptions.http_exception_wrapper import http_exception
from app.logic.chat.generation_chat import stream_generation_chat
from app.utils import auth

generation_chat_router = APIRouter()


def _parse_zone_types(raw: Optional[str]) -> Optional[list[str]]:
    if not raw:
        return None
    zones = [z.strip() for z in raw.split(",") if z.strip()]
    return zones or None


async def _read_geojson_file(
    upload: Optional[UploadFile], label: str
) -> Optional[dict[str, Any]]:
    """Parse an uploaded GeoJSON file into a FeatureCollection dict.

    Only the envelope is checked here; per-feature filtering (blocks: keep only
    residential/business; existing buildings: keep only polygons) happens
    downstream in the orchestrator so dropped-feature warnings are surfaced as
    SSE events.
    """
    if upload is None:
        return None
    raw = await upload.read()
    try:
        geojson = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise http_exception(422, f"Uploaded {label} file is not valid JSON: {exc}")
    if not isinstance(geojson, dict) or geojson.get("type") != "FeatureCollection":
        raise http_exception(
            422, f"Uploaded {label} file must be a GeoJSON FeatureCollection."
        )
    return geojson


def _has_upload(upload: Optional[UploadFile]) -> bool:
    """An empty multipart file field arrives as an UploadFile without a filename."""
    return upload is not None and bool(getattr(upload, "filename", None))


@generation_chat_router.post(
    "/generate/chat/stream",
    summary="Conversational building generation over SSE",
)
async def generate_chat_stream(
    user_query: Annotated[str, Form(min_length=1, description="Free-text request")],
    scenario_id: Annotated[Optional[int], Form(ge=1, description="Scenario ID (omit if uploading a blocks file)")] = None,
    year: Annotated[Optional[int], Form(description="Data year (with scenario_id)")] = None,
    source: Annotated[Optional[str], Form(description="Data source, e.g. OSM (with scenario_id)")] = None,
    blocks_file: Annotated[
        Optional[UploadFile],
        File(description="Optional GeoJSON FeatureCollection of blocks; each feature needs properties.zone"),
    ] = None,
    buildings_file: Annotated[
        Optional[UploadFile],
        File(
            description=(
                "Optional GeoJSON FeatureCollection of existing buildings "
                "(project-less mode): their footprints are excluded from generation"
            )
        ),
    ] = None,
    skip_existing_buildings: Annotated[
        bool,
        Form(
            description=(
                "Set to true when the user declined to upload existing buildings, "
                "so the question is not asked again"
            )
        ),
    ] = False,
    functional_zone_types: Annotated[
        Optional[str],
        Form(description="Optional comma-separated zone filter, e.g. 'residential,business'"),
    ] = None,
    chat_id: Annotated[Optional[str], Form(description="Existing chat id for multi-turn")] = None,
    project_id: Annotated[Optional[int], Form(description="Project id (for chat history)")] = None,
    model: Annotated[Optional[str], Form(description="Override chat model")] = None,
    temperature: Annotated[Optional[float], Form(description="Override sampling temperature")] = None,
    user: auth.AuthUser = Depends(auth.get_current_user),
) -> EventSourceResponse:
    if not chat_llm_configured():
        raise http_exception(
            503,
            "Conversational generation is unavailable: LLM backend is not "
            "configured (set LLM_API and Chat_Model).",
        )

    # Territory comes either from a scenario or from an uploaded blocks file.
    has_file = _has_upload(blocks_file)
    if not has_file and scenario_id is None:
        raise http_exception(
            422,
            "Provide a territory source: either scenario_id (+year, +source) or a blocks_file.",
        )
    if not has_file and (year is None or not source):
        raise http_exception(422, "scenario_id requires both year and source.")

    zone_types = _parse_zone_types(functional_zone_types)
    blocks_geojson = await _read_geojson_file(
        blocks_file if has_file else None, "blocks"
    )
    buildings_geojson = await _read_geojson_file(
        buildings_file if _has_upload(buildings_file) else None, "existing buildings"
    )

    async def event_source():
        async with AsyncExitStack() as stack:
            llm = await stack.enter_async_context(build_vllm_chat_client(temperature))
            storage = build_chat_storage_client()
            if storage is not None:
                await stack.enter_async_context(storage)

            async for event in stream_generation_chat(
                builder=builder,
                llm_client=llm,
                chat_storage_client=storage,
                token=user.token,
                user_id=user.user_id,
                user_query=user_query,
                scenario_id=scenario_id,
                year=year,
                source=source,
                la_per_person=CHAT_LA_PER_PERSON,
                chat_id=chat_id,
                project_id=project_id,
                chat_title=user_query[:256],
                functional_zone_types=zone_types,
                blocks_geojson=blocks_geojson,
                existing_buildings_geojson=buildings_geojson,
                existing_buildings_declined=skip_existing_buildings,
                model=model,
                temperature=temperature,
                zones_service=zones_service,
                object_storage=optional_object_storage(),
                public_base_url=public_base_url(),
            ):
                event_type = event.pop("type", "message")
                yield ServerSentEvent(
                    event=event_type,
                    data=json.dumps(event, ensure_ascii=False),
                )

    return EventSourceResponse(event_source())
