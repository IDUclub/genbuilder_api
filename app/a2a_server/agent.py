"""A2A AgentExecutor wrapping GenBuilder's conversational generation flow.

Reuses ``app.logic.chat.generation_chat.stream_generation_chat`` — the same
orchestrator behind the ``/generate/chat/stream`` SSE endpoint — and maps its
event stream onto A2A Task lifecycle calls (``TaskUpdater``). This is the
only A2A skill GenBuilder exposes: a natural-language generation request in,
a Task carrying the generated buildings (as a data artifact) or a
clarification question out. Structured, parameter-exact generation is
already covered by the MCP tools in ``app.mcp_server``.
"""
from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any

from fastapi import HTTPException
from google.protobuf.json_format import MessageToDict
from loguru import logger

from a2a.helpers import get_data_parts, new_data_part, new_text_message, new_text_part, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState

from app.dependencies import (
    CHAT_LA_PER_PERSON,
    build_chat_storage_client,
    build_ollama_chat_client,
    builder,
    chat_llm_configured,
)
from app.logic.chat.generation_chat import stream_generation_chat
from app.utils.auth import authenticate_token


def _extract_bearer(headers: dict[str, str]) -> str | None:
    auth_header = headers.get("authorization") or headers.get("Authorization") or ""
    return auth_header[7:].strip() if auth_header.startswith("Bearer ") else None


def _extract_payload(context: RequestContext) -> dict[str, Any]:
    """Merge structured fields (data Part / message metadata) with the free-text query.

    Accepted structured fields mirror the ``/generate/chat/stream`` form:
    scenario_id, year, source, functional_zone_types, blocks_geojson, chat_id,
    project_id, model, temperature. Any of these may arrive as a JSON data
    Part or as message metadata; metadata is the fallback so a client that
    only knows how to attach plain text + a JSON sidecar still works.
    """
    structured: dict[str, Any] = {}
    for data in get_data_parts(context.message.parts):
        if isinstance(data, dict):
            structured.update(data)
    metadata = MessageToDict(context.message.metadata) if context.message.metadata else {}
    merged: dict[str, Any] = {**metadata, **structured}
    text = context.get_user_input().strip()
    if text:
        merged.setdefault("user_query", text)
    return merged


class GenBuilderAgentExecutor(AgentExecutor):
    """Executes GenBuilder's conversational generation flow as an A2A Task."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        payload = _extract_payload(context)
        user_query = str(payload.get("user_query") or "").strip()
        scenario_id = payload.get("scenario_id")
        blocks_geojson = payload.get("blocks_geojson")

        if not user_query:
            await updater.reject(message=new_text_message("Message must include a text part with the generation request."))
            return
        if scenario_id is None and not blocks_geojson:
            await updater.reject(message=new_text_message(
                "Provide a territory source: either scenario_id (+year +source) "
                "or blocks_geojson."
            ))
            return
        if scenario_id is not None and (payload.get("year") is None or not payload.get("source")):
            await updater.reject(message=new_text_message("scenario_id requires both year and source."))
            return

        if not chat_llm_configured():
            await updater.failed(message=new_text_message(
                "Conversational generation is unavailable: LLM backend is not configured."
            ))
            return

        headers = dict((context.call_context.state or {}).get("headers") or {}) if context.call_context else {}
        token = _extract_bearer(headers)
        user_id = ""
        if token:
            try:
                auth_user = await authenticate_token(token)
                token, user_id = auth_user.token, auth_user.user_id
            except HTTPException as exc:
                await updater.failed(message=new_text_message(f"Authentication failed ({exc.status_code}): {exc.detail}"))
                return

        await updater.start_work()

        terminal = False
        answer_parts: list[str] = []
        try:
            async with AsyncExitStack() as stack:
                ollama = await stack.enter_async_context(build_ollama_chat_client(payload.get("temperature")))
                storage = build_chat_storage_client()
                if storage is not None:
                    await stack.enter_async_context(storage)

                async for event in stream_generation_chat(
                    builder=builder,
                    ollama_client=ollama,
                    chat_storage_client=storage,
                    token=token,
                    user_id=user_id,
                    user_query=user_query,
                    scenario_id=scenario_id,
                    year=payload.get("year"),
                    source=payload.get("source"),
                    la_per_person=CHAT_LA_PER_PERSON,
                    chat_id=payload.get("chat_id"),
                    project_id=payload.get("project_id"),
                    chat_title=user_query[:256],
                    functional_zone_types=payload.get("functional_zone_types"),
                    blocks_geojson=blocks_geojson,
                    generation_parameters=payload.get("generation_parameters"),
                    model=payload.get("model"),
                    temperature=payload.get("temperature"),
                ):
                    event_type = event.get("type")

                    if event_type in ("status", "progress"):
                        content = event.get("content") or ""
                        if content:
                            await updater.update_status(TaskState.TASK_STATE_WORKING, message=new_text_message(content))
                    elif event_type == "clarification":
                        await updater.requires_input(message=new_text_message(event.get("content") or ""))
                        terminal = True
                    elif event_type == "warning":
                        message = event.get("message") or event.get("detail") or ""
                        if message:
                            await updater.update_status(TaskState.TASK_STATE_WORKING, message=new_text_message(message))
                    elif event_type == "token":
                        answer_parts.append(event.get("content") or "")
                    elif event_type == "result":
                        await updater.add_artifact(
                            parts=[new_data_part(event.get("content") or {})],
                            name="generated_buildings",
                        )
                        summary = event.get("summary") or {}
                        await updater.add_artifact(
                            parts=[new_text_part(json.dumps(summary, ensure_ascii=False))],
                            name="generation_summary",
                        )
                    elif event_type == "error":
                        await updater.failed(message=new_text_message(event.get("detail") or "Generation failed."))
                        terminal = True
                    elif event_type == "done":
                        if not terminal:
                            final_text = "".join(answer_parts).strip() or "Генерация застройки завершена."
                            await updater.complete(message=new_text_message(final_text))
                        terminal = True
        except Exception as exc:  # noqa: BLE001 - surface any unexpected failure as a failed task
            logger.exception("A2A generation task failed")
            if not terminal:
                await updater.failed(message=new_text_message(f"Unexpected error: {exc}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel is not supported.")
