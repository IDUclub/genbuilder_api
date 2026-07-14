"""Conversational, streamed building generation (the SSE orchestrator).

Mirrors the gMART-style layer from IDUclub/PzzCompareAPI (``stream_chat_answer``),
adapted to Genbuilder: take the user's free-text request, extract the generation
targets, and either

- ask for the missing mandatory parameters (``clarification`` event), or
- run ``Genbuilder.run`` inline, stream progress, return the generated buildings
  (``result`` event) and a natural-language summary (``token`` events).

The turn is persisted to ChatStorage best-effort: a storage failure is surfaced
as a non-fatal ``warning`` event and never aborts the stream. History is loaded
for an existing ``chat_id`` so multi-turn clarifications accumulate (a follow-up
like "5000 жителей" is combined with the earlier request).

Event envelope (``{"type": ..., ...}``), matching the reference style:

- ``chat_created``  {chat_id, title}        — a new chat was created.
- ``clarification`` {content, missing}       — mandatory params are missing; the
                                               answer is a question, not a result.
- ``status``        {content}                — human-readable progress note.
- ``progress``      {stage, content}         — a pipeline stage marker.
- ``token``         {content}                — a summary-answer content delta.
- ``result``        {content, summary}       — the generated FeatureCollection.
- ``warning``       {stage, detail, message} — non-fatal (e.g. not persisted).
- ``error``         {stage, detail}          — fatal; generation/answer failed.
- ``done``          {chat_id, assistant_message_id} — terminal marker.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from loguru import logger

from app.infrastructure.chat_storage_client import ChatStorageClient, ChatStorageError
from app.infrastructure.ollama_chat_client import OllamaChatClient, OllamaChatError
from app.logic.chat.param_extraction import (
    DEFAULT_FLOOR_GROUP_BY_ZONE,
    GENERATED_ZONES,
    extract_generation_targets,
    validate_targets,
)
from app.schema.dto import BlockFeatureCollection
from app.logic.zone_taxonomy import normalize_zone


def _blocks_from_geojson(
    geojson: dict[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...], int]:
    """Keep only features whose ``properties.zone`` maps to a generated zone.

    The raw ``zone`` name is normalized (granular residential subtypes -> residential,
    mixed_use -> business); features that don't map to residential/business are
    dropped. Kept features keep their raw name — the core normalizes them again
    (and derives the per-block floor group from the subtype). Returns (kept
    features, distinct in-scope canonical zones, dropped count).
    """
    features = geojson.get("features") or []
    kept: list[dict[str, Any]] = []
    zones: list[str] = []
    for feature in features:
        canonical = normalize_zone((feature.get("properties") or {}).get("zone"))
        if canonical in GENERATED_ZONES:
            kept.append(feature)
            if canonical not in zones:
                zones.append(canonical)
    return kept, tuple(zones), len(features) - len(kept)

_SUMMARY_SYSTEM_PROMPT = (
    "Ты — ассистент по генерации застройки. Кратко и по делу опиши на русском "
    "языке результат генерации, опираясь ТОЛЬКО на переданную сводку. Не "
    "выдумывай числа. Упомяни количество зданий, суммарную жилую площадь и "
    "расчётное число жителей, если они есть."
)


def _history_user_text(messages: list[dict[str, Any]], max_messages: int = 10) -> str:
    """Concatenate recent user-turn text from ChatStorage messages.

    ChatStorage returns text either as a top-level ``content`` string or as
    ``parts[*].payload.text``. Only user turns are kept so earlier requests give
    the extractor context for a short follow-up answer.
    """
    texts: list[str] = []
    for message in messages[-max_messages:]:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())
            continue
        for part in message.get("parts") or []:
            if part.get("kind") == "text":
                text = (part.get("payload") or {}).get("text")
                if text:
                    texts.append(str(text).strip())
    return "\n".join(texts)


def _summarize_buildings(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact totals over generated building features, for grounding + the UI."""
    total = len(features)
    living_area = 0.0
    residents = 0
    by_zone: dict[str, int] = {}
    for feature in features:
        props = feature.get("properties") or {}
        living_area += float(props.get("living_area") or 0.0)
        residents += int(props.get("residents_number") or 0)
        zone = props.get("zone")
        if zone:
            by_zone[zone] = by_zone.get(zone, 0) + 1
    return {
        "buildings": total,
        "living_area_total": round(living_area, 1),
        "residents_total": residents,
        "buildings_by_zone": by_zone,
    }


def _merge_result(result: dict | None) -> tuple[dict, dict]:
    """Split a Genbuilder.run result into (merged FeatureCollection, summary)."""
    result = result if isinstance(result, dict) else {}
    generated = result.get("generated_buildings") or {}
    selected = result.get("selected_features") or {}
    gen_features = list(generated.get("features") or [])
    sel_features = list(selected.get("features") or [])
    merged = {
        "type": "FeatureCollection",
        "features": [*gen_features, *sel_features],
    }
    return merged, _summarize_buildings(gen_features)


async def stream_generation_chat(
    *,
    builder: Any,
    ollama_client: OllamaChatClient,
    chat_storage_client: ChatStorageClient | None,
    token: str | None,
    user_id: str | None = None,
    user_query: str,
    scenario_id: int | None,
    year: int | None,
    source: str | None,
    la_per_person: float,
    chat_id: str | None = None,
    project_id: int | str | None = None,
    chat_title: str | None = None,
    functional_zone_types: list[str] | None = None,
    blocks_geojson: dict[str, Any] | None = None,
    generation_parameters: dict[str, Any] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    message_metadata: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    persist = chat_storage_client is not None and bool(user_id)

    # 0. Load prior turns (existing chat) so short follow-ups keep context.
    prior_text = ""
    if persist and chat_id:
        try:
            existing = await chat_storage_client.get_chat(user_id, chat_id)
            prior_text = _history_user_text(existing.get("messages") or [])
        except ChatStorageError as exc:
            logger.warning("chat_storage get_chat (history) failed: {}", exc)
            yield {
                "type": "warning",
                "stage": "load_history",
                "detail": str(exc),
                "message": "Не удалось загрузить историю чата — обрабатываю только "
                "текущее сообщение.",
            }

    # 1. Ensure a chat exists.
    if persist and not chat_id:
        try:
            created = await chat_storage_client.create_chat(
                user_id,
                title=chat_title or user_query[:256],
                scenario_id=scenario_id,
                project_id=project_id,
                metadata=message_metadata,
            )
            chat_id = created.get("chat_id")
            yield {"type": "chat_created", "chat_id": chat_id, "title": created.get("title")}
        except ChatStorageError as exc:
            logger.warning("chat_storage create_chat failed: {}", exc)
            yield {
                "type": "warning",
                "stage": "create_chat",
                "detail": str(exc),
                "message": "Диалог не будет сохранён в историю (проверьте токен).",
            }
            persist = False

    # 2. Persist the user turn.
    if persist and chat_id:
        try:
            await chat_storage_client.add_message(
                user_id, chat_id, role="user", content=user_query, metadata=message_metadata
            )
        except ChatStorageError as exc:
            logger.warning("chat_storage add user message failed: {}", exc)
            yield {
                "type": "warning",
                "stage": "add_user_message",
                "detail": str(exc),
                "message": "Сообщение не сохранено в историю (проверьте токен).",
            }

    # 2.5 Resolve the territory source. A user-uploaded blocks file overrides the
    # scenario; we keep only residential/business features and derive the zones in
    # scope from the file (so a file with just one zone doesn't over-ask).
    blocks: BlockFeatureCollection | None = None
    zones_in_scope: tuple[str, ...] = GENERATED_ZONES
    if blocks_geojson is not None:
        kept, zones_in_scope, dropped = _blocks_from_geojson(blocks_geojson)
        if dropped:
            yield {
                "type": "warning",
                "stage": "load_blocks",
                "detail": f"{dropped} feature(s) dropped",
                "message": f"Отброшено объектов без зоны residential/business: {dropped}.",
            }
        if not kept:
            yield {
                "type": "error",
                "stage": "load_blocks",
                "detail": "no residential/business features in uploaded file",
            }
            yield {"type": "done", "chat_id": chat_id, "assistant_message_id": None}
            return
        try:
            blocks = BlockFeatureCollection.model_validate(
                {"type": "FeatureCollection", "features": kept}
            )
        except Exception as exc:  # noqa: BLE001 - invalid geometry -> surface to client
            logger.warning("invalid blocks file: {}", exc)
            yield {"type": "error", "stage": "load_blocks", "detail": str(exc)}
            yield {"type": "done", "chat_id": chat_id, "assistant_message_id": None}
            return

    # 3. Extract targets from the (accumulated) request text.
    combined_query = f"{prior_text}\n{user_query}".strip() if prior_text else user_query
    extracted = await extract_generation_targets(
        ollama_client,
        user_query=combined_query,
        la_per_person=la_per_person,
        model=model,
    )
    # Zones come from the territory source: both generated zones for a scenario,
    # or the zones actually present in an uploaded blocks file.
    extracted.functional_zone_types = list(zones_in_scope)

    # Pin the policy default floor group per zone unless the user set one
    # explicitly, so the result doesn't depend on the pipeline's fallbacks.
    default_fg = extracted.targets_by_zone.setdefault("default_floor_group", {})
    for zone, floor_group in DEFAULT_FLOOR_GROUP_BY_ZONE.items():
        default_fg.setdefault(zone, floor_group)

    # 4. Missing mandatory params -> ask, persist the question, stop.
    missing = validate_targets(extracted, zones_in_scope)
    if missing:
        content = "Чтобы сгенерировать застройку, уточните:\n" + "\n".join(
            f"— {m.prompt}" for m in missing
        )
        yield {
            "type": "clarification",
            "content": content,
            "missing": [
                {
                    "zone": m.zone,
                    "field": m.field,
                    "control": m.control,
                    "unit": m.unit,
                    "alt_fields": list(m.alt_fields),
                }
                for m in missing
            ],
        }
        assistant_message_id = await _persist_assistant(
            chat_storage_client, persist, user_id, chat_id, content, message_metadata
        )
        yield {"type": "done", "chat_id": chat_id, "assistant_message_id": assistant_message_id}
        return

    # 5. Run generation inline.
    yield {
        "type": "status",
        "content": "Параметры приняты, запускаю генерацию застройки.",
        "targets_by_zone": extracted.targets_by_zone,
        "functional_zone_types": extracted.functional_zone_types,
    }
    yield {"type": "progress", "stage": "generation", "content": "Генерация зданий…"}
    try:
        result = await builder.run(
            targets_by_zone=extracted.targets_by_zone,
            blocks=blocks,
            token=token,
            scenario_id=scenario_id,
            year=year,
            source=source,
            functional_zone_types=extracted.functional_zone_types,
            generation_parameters_override=generation_parameters,
        )
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the client
        logger.exception("generation failed")
        yield {"type": "error", "stage": "generation", "detail": str(exc)}
        yield {"type": "done", "chat_id": chat_id, "assistant_message_id": None}
        return

    merged, summary = _merge_result(result)
    yield {"type": "result", "content": merged, "summary": summary}

    # 6. Stream a natural-language summary grounded on the result (best-effort).
    summary_messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "Сводка генерации (JSON):\n"
            + json.dumps(summary, ensure_ascii=False),
        },
        {"role": "user", "content": user_query},
    ]
    collected: list[str] = []
    try:
        async for delta in ollama_client.stream_chat(
            summary_messages, model=model, temperature=temperature
        ):
            collected.append(delta)
            yield {"type": "token", "content": delta}
    except OllamaChatError as exc:
        logger.warning("summary stream failed: {}", exc)
        yield {
            "type": "warning",
            "stage": "summary",
            "detail": str(exc),
            "message": "Не удалось сформировать текстовое описание результата.",
        }
    answer_text = "".join(collected).strip()

    # 7. Persist the assistant turn.
    assistant_message_id = await _persist_assistant(
        chat_storage_client,
        persist,
        user_id,
        chat_id,
        answer_text or "Генерация застройки завершена.",
        message_metadata,
    )
    yield {"type": "done", "chat_id": chat_id, "assistant_message_id": assistant_message_id}


async def _persist_assistant(
    chat_storage_client: ChatStorageClient | None,
    persist: bool,
    user_id: str | None,
    chat_id: str | None,
    content: str,
    metadata: dict[str, Any] | None,
) -> str | None:
    if not (persist and chat_id and content):
        return None
    try:
        stored = await chat_storage_client.add_message(
            user_id, chat_id, role="assistant", content=content, metadata=metadata
        )
        return stored.get("message_id")
    except ChatStorageError as exc:
        logger.warning("chat_storage add assistant message failed: {}", exc)
        return None
