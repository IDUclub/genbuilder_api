"""SSE endpoint for conversational building generation.

The frontend posts the user's free-text request plus the territory reference
(scenario_id + year + source). The server extracts the generation targets, and
either streams back a ``clarification`` asking for the missing mandatory
parameters, or runs generation inline and streams progress → result → summary.

Response is ``text/event-stream`` (sse-starlette). Each event carries the
envelope ``type`` as the SSE ``event`` field and the rest of the payload as JSON
``data`` — mirroring IDUclub/PzzCompareAPI's ``*/chat/stream`` handlers.
"""
from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.dependencies import (
    CHAT_LA_PER_PERSON,
    build_chat_storage_client,
    build_vllm_chat_client,
    builder,
    chat_llm_configured,
    facade_jobs_configured,
    optional_object_storage,
    public_base_url,
    zones_service,
)
from app.exceptions.http_exception_wrapper import http_exception
from app.logic.chat.generation_chat import stream_generation_chat
from app.logic import generation_orchestration as orchestration
from app.utils import auth

generation_chat_router = APIRouter()


def _parse_zone_types(raw: Optional[str]) -> Optional[list[str]]:
    if not raw:
        return None
    zones = [z.strip() for z in raw.split(",") if z.strip()]
    return zones or None


async def _read_blocks_file(blocks_file: Optional[UploadFile]) -> Optional[dict[str, Any]]:
    """Parse an uploaded GeoJSON blocks file into a FeatureCollection dict.

    The zone filtering (keep only residential/business) happens downstream in the
    orchestrator so dropped-feature warnings are surfaced as SSE events.
    """
    if blocks_file is None:
        return None
    raw = await blocks_file.read()
    try:
        geojson = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise http_exception(422, f"Uploaded blocks file is not valid JSON: {exc}")
    if not isinstance(geojson, dict) or geojson.get("type") != "FeatureCollection":
        raise http_exception(422, "Uploaded blocks file must be a GeoJSON FeatureCollection.")
    return geojson


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
    return await _generate_chat_stream_response(
        user_query=user_query,
        scenario_id=scenario_id,
        year=year,
        source=source,
        blocks_file=blocks_file,
        functional_zone_types=functional_zone_types,
        chat_id=chat_id,
        project_id=project_id,
        model=model,
        temperature=temperature,
        user=user,
        queue_facades=False,
    )


@generation_chat_router.post(
    "/generate/chat/stream/3d",
    summary="Conversational building generation with an asynchronous 3D facade job",
)
async def generate_chat_stream_3d(
    user_query: Annotated[str, Form(min_length=1, description="Free-text request")],
    scenario_id: Annotated[Optional[int], Form(ge=1, description="Scenario ID (omit if uploading a blocks file)")] = None,
    year: Annotated[Optional[int], Form(description="Data year (with scenario_id)")] = None,
    source: Annotated[Optional[str], Form(description="Data source, e.g. OSM (with scenario_id)")] = None,
    blocks_file: Annotated[
        Optional[UploadFile],
        File(description="Optional GeoJSON FeatureCollection of blocks; each feature needs properties.zone"),
    ] = None,
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
    return await _generate_chat_stream_response(
        user_query=user_query,
        scenario_id=scenario_id,
        year=year,
        source=source,
        blocks_file=blocks_file,
        functional_zone_types=functional_zone_types,
        chat_id=chat_id,
        project_id=project_id,
        model=model,
        temperature=temperature,
        user=user,
        queue_facades=True,
    )


async def _generate_chat_stream_response(
    *,
    user_query: str,
    scenario_id: int | None,
    year: int | None,
    source: str | None,
    blocks_file: UploadFile | None,
    functional_zone_types: str | None,
    chat_id: str | None,
    project_id: int | None,
    model: str | None,
    temperature: float | None,
    user: auth.AuthUser,
    queue_facades: bool,
) -> EventSourceResponse:
    if not chat_llm_configured():
        raise http_exception(
            503,
            "Conversational generation is unavailable: LLM backend is not "
            "configured (set LLM_API and Chat_Model).",
        )
    if queue_facades and not facade_jobs_configured():
        raise http_exception(
            503,
            "3D facade generation is unavailable: facade-jobs is not "
            "configured (set FACADE_JOBS_API).",
        )

    # Territory comes either from a scenario or from an uploaded blocks file.
    has_file = blocks_file is not None and bool(getattr(blocks_file, "filename", None))
    if not has_file and scenario_id is None:
        raise http_exception(
            422,
            "Provide a territory source: either scenario_id (+year, +source) or a blocks_file.",
        )
    if not has_file and (year is None or not source):
        raise http_exception(422, "scenario_id requires both year and source.")

    zone_types = _parse_zone_types(functional_zone_types)
    blocks_geojson = await _read_blocks_file(blocks_file if has_file else None)

    async def event_source():
        async with AsyncExitStack() as stack:
            llm = await stack.enter_async_context(build_vllm_chat_client(temperature))
            storage = build_chat_storage_client()
            if storage is not None:
                await stack.enter_async_context(storage)

            facade_buildings: dict[str, Any] | None = None
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
                model=model,
                temperature=temperature,
                zones_service=zones_service,
                object_storage=optional_object_storage(),
                public_base_url=public_base_url(),
            ):
                if queue_facades and event.get("type") == "result":
                    content = event.get("content")
                    if isinstance(content, dict):
                        facade_buildings = content

                # Submit after the textual summary and immediately before the
                # terminal event, so ``facade_job`` is the final useful SSE
                # payload while ``done`` remains the stream terminator.
                if (
                    queue_facades
                    and event.get("type") == "done"
                    and facade_buildings is not None
                ):
                    try:
                        job = await orchestration.submit_facade_job(
                            facade_buildings,
                            requested_by=user.user_id,
                        )
                    except HTTPException as exc:
                        yield ServerSentEvent(
                            event="error",
                            data=json.dumps(
                                {
                                    "stage": "facade_job",
                                    "detail": exc.detail,
                                },
                                ensure_ascii=False,
                            ),
                        )
                    else:
                        yield ServerSentEvent(
                            event="facade_job",
                            data=json.dumps(job, ensure_ascii=False),
                        )

                event_type = event.pop("type", "message")
                yield ServerSentEvent(
                    event=event_type,
                    data=json.dumps(event, ensure_ascii=False),
                )

    return EventSourceResponse(event_source())
