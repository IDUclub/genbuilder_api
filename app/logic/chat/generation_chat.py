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
- ``clarification`` {content, missing}       — parameters are missing (or the
                                               project-less mode's optional
                                               existing-buildings question is
                                               unanswered); the answer is a
                                               question, not a result.
- ``status``        {content}                — human-readable progress note.
- ``progress``      {stage, content}         — a pipeline stage marker.
- ``token``         {content}                — a summary-answer content delta.
- ``zones``         {content, source}        — the functional zones backdrop,
                                               inline, before generation.
- ``result``        {content, summary}       — the generated FeatureCollection.
- ``file``          {name, title, url, ...}  — a geo-layer link descriptor; the
                                               same payload is persisted to
                                               history as a ``file`` part.
- ``warning``       {stage, detail, message} — non-fatal (e.g. not persisted).
- ``error``         {stage, detail, code?, message?} — fatal; generation/answer failed.
                                              Carries a user-facing ``message``
                                              when the cause is explainable.
- ``done``          {chat_id, assistant_message_id} — terminal marker.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator
from uuid import uuid4

import aiohttp
from fastapi import HTTPException
from loguru import logger

from app.infrastructure.chat_storage_client import ChatStorageClient, ChatStorageError
from app.infrastructure.object_storage import ObjectStorage, ObjectStorageError
from app.infrastructure.vllm_chat_client import VLLMChatClient, VLLMChatError
from app.logic.chat.chat_title import make_chat_title
from app.logic.chat.param_extraction import (
    DEFAULT_FLOOR_GROUP_BY_ZONE,
    GENERATED_ZONES,
    ZONE_LABELS,
    existing_buildings_question,
    extract_generation_targets,
    split_total_residents,
    validate_targets,
)
from app.logic.geo_layers import (
    SLOT_BLOCKS_INPUT,
    SLOT_BUILDINGS,
    SLOT_EXISTING_BUILDINGS,
    SLOT_ZONES,
    build_stored_layer,
    build_zones_layer,
    geo_layer_to_file_part,
    object_key,
)
from app.logic.generation_summary import summarize_buildings
from app.schema.dto import BlockFeatureCollection
from app.logic.chat.zone_detection import ZONE_TYPE_LABELS, detect_zones
from app.logic.zone_taxonomy import normalize_zone


_POLYGONAL = {"Polygon", "MultiPolygon"}


def _quoted_counts(counts: dict[str, int], limit: int = 10) -> str:
    items = list(counts.items())
    text = ", ".join(f"«{value}» — {count}" for value, count in items[:limit])
    return text + (f" и ещё {len(items) - limit}" if len(items) > limit else "")


def _type_counts(counts: dict[str, int]) -> str:
    return ", ".join(
        f"{ZONE_TYPE_LABELS.get(zone, zone)} ({zone}) — {count}"
        for zone, count in sorted(counts.items(), key=lambda kv: -kv[1])
    )


def _load_error(code: str, detail: str, message: str) -> dict[str, Any]:
    return {"type": "error", "stage": "load_blocks", "code": code, "detail": detail, "message": message}


def _load_warning(code: str, detail: str, message: str) -> dict[str, Any]:
    return {"type": "warning", "stage": "load_blocks", "code": code, "detail": detail, "message": message}


_EXPECTED_ZONES_HINT = (
    "Ожидается атрибут zone (или functional_zone_type_name) со значениями вида "
    "residential / «Жилая зона», business / «Общественно-деловая зона», "
    "mixed_use / «Многофункциональная зона»."
)


async def _blocks_from_upload(
    geojson: dict[str, Any],
    llm_client: VLLMChatClient | None,
    model: str | None = None,
) -> tuple[list[dict[str, Any]], tuple[str, ...], list[dict[str, Any]]]:
    """Resolve the uploaded blocks' zone types and keep the generated ones.

    Returns (kept features with a canonical Urban DB ``zone``, distinct in-scope
    generation zones, events to stream). An empty ``kept`` always comes with a
    ``load_blocks`` error carrying a user-facing ``message`` and a ``code``.
    Kept features keep the granular type (``residential_lowrise``) — the core
    normalizes it and derives the per-block floor group from the subtype.
    """
    events: list[dict[str, Any]] = []
    features = geojson.get("features") or []
    if not features:
        events.append(_load_error("blocks_empty", "no features", "В загруженном файле зон нет ни одного объекта."))
        return [], (), events

    detection = await detect_zones(geojson, llm_client, model=model)
    if detection.attribute is None:
        fields = ", ".join(detection.attributes[:15]) or "нет текстовых атрибутов"
        if detection.llm_error:
            reason = (
                "Автоматически определить атрибут не удалось: сервис распознавания "
                "сейчас недоступен."
            )
        else:
            reason = "Ни один атрибут не похож на тип функциональной зоны."
        events.append(
            _load_error(
                "zone_attribute_not_found",
                f"zone attribute not found; attributes: {detection.attributes}",
                f"Не удалось понять, где в файле указан тип функциональной зоны. {reason} "
                f"Атрибуты в файле: {fields}. {_EXPECTED_ZONES_HINT}",
            )
        )
        return [], (), events

    how = " (определён моделью)" if detection.attribute_by_llm else ""
    events.append(
        {"type": "status", "content": f"Тип зоны взят из атрибута «{detection.attribute}»{how}."}
    )
    if detection.llm_mapping:
        pairs = ", ".join(f"«{raw}» → {zone}" for raw, zone in detection.llm_mapping.items())
        events.append(
            {"type": "status", "content": f"Модель распознала нестандартные значения: {pairs}."}
        )
    if detection.unrecognized:
        count = sum(detection.unrecognized.values())
        events.append(
            _load_warning(
                "zone_values_unrecognized",
                f"{count} feature(s) with unrecognized zone type: {list(detection.unrecognized)}",
                f"Не распознан тип зоны у {count} объект(ов), они пропущены: "
                f"{_quoted_counts(detection.unrecognized)}.",
            )
        )
    if detection.missing:
        events.append(
            _load_warning(
                "zone_value_missing",
                f"{detection.missing} feature(s) without a zone value",
                f"У {detection.missing} объект(ов) не заполнен атрибут "
                f"«{detection.attribute}», они пропущены.",
            )
        )

    in_scope = [f for f in detection.features if normalize_zone(f["properties"]["zone"]) in GENERATED_ZONES]
    kept = [f for f in in_scope if (f.get("geometry") or {}).get("type") in _POLYGONAL]
    out_of_scope = {
        zone: count
        for zone, count in detection.type_counts.items()
        if normalize_zone(zone) not in GENERATED_ZONES
    }
    if not in_scope:
        found = _type_counts(detection.type_counts) or "ни одного распознанного типа"
        events.append(
            _load_error(
                "no_generated_zones",
                f"no residential/business features; types: {dict(detection.type_counts)}",
                "В файле нет жилых или общественно-деловых (многофункциональных) зон — "
                f"застройку генерировать негде. Найдено по атрибуту «{detection.attribute}»: "
                f"{found}.",
            )
        )
        return [], (), events
    if out_of_scope:
        events.append(
            _load_warning(
                "zones_out_of_scope",
                f"{sum(out_of_scope.values())} feature(s) outside residential/business",
                "Застройка генерируется только в жилых и общественно-деловых зонах; "
                f"пропущены: {_type_counts(out_of_scope)}.",
            )
        )
    non_polygonal = len(in_scope) - len(kept)
    if not kept:
        events.append(
            _load_error(
                "no_polygons",
                "no Polygon/MultiPolygon residential/business features",
                "Жилые и общественно-деловые зоны в файле заданы не полигонами "
                "(нужны Polygon или MultiPolygon).",
            )
        )
        return [], (), events
    if non_polygonal:
        events.append(
            _load_warning(
                "non_polygon_geometry",
                f"{non_polygonal} non-polygon feature(s) dropped",
                f"Пропущено {non_polygonal} объект(ов) с геометрией не Polygon/MultiPolygon.",
            )
        )
    zones = tuple(dict.fromkeys(normalize_zone(f["properties"]["zone"]) for f in kept))
    return kept, zones, events


def _zones_from_layer(layer: dict[str, Any]) -> tuple[str, ...]:
    """Return the generated-zone types actually present in a zones layer."""
    zones: list[str] = []
    for feature in layer.get("features") or []:
        zone = normalize_zone((feature.get("properties") or {}).get("zone"))
        if zone in GENERATED_ZONES and zone not in zones:
            zones.append(zone)
    return tuple(zones)


def _zone_areas(features: list[dict[str, Any]]) -> dict[str, float]:
    """Area (m²) of each generated zone type in a list of zone features."""
    import geopandas as gpd  # heavy; only needed when a total is split

    try:
        gdf = gpd.GeoDataFrame.from_features(features, crs=4326)
        gdf["zone"] = gdf["zone"].map(normalize_zone)
        gdf = gdf[gdf["zone"].isin(GENERATED_ZONES) & gdf.geometry.notna()]
        if gdf.empty:
            return {}
        areas = gdf.to_crs(gdf.estimate_utm_crs()).geometry.area
        return {str(z): float(a) for z, a in areas.groupby(gdf["zone"]).sum().items()}
    except Exception as exc:  # noqa: BLE001 - fall back to an equal split
        logger.warning("zone areas for the residents split failed: {}", exc)
        return {}


def _existing_buildings_from_geojson(
    geojson: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Keep only the polygonal features of an uploaded existing-buildings file.

    Only a footprint can be cut out of a block, so points and lines are dropped
    (and reported as a warning). Properties are left untouched — the generation
    layer normalizes them into the excluded-object shape. Returns (kept
    features, dropped count).
    """
    features = geojson.get("features") or []
    kept = [
        feature
        for feature in features
        if (feature.get("geometry") or {}).get("type") in {"Polygon", "MultiPolygon"}
    ]
    return kept, len(features) - len(kept)


_SUMMARY_SYSTEM_PROMPT = (
    "Ты — ассистент по генерации застройки. Кратко и по делу опиши на русском "
    "языке результат генерации, опираясь ТОЛЬКО на переданную сводку. Не "
    "выдумывай числа. Упомяни количество зданий, суммарную жилую площадь и "
    "расчётное число жителей, если они есть."
)


_REPORTED_NOTICES_KEY = "reported_notices"


def _reported_notices(messages: list[dict[str, Any]]) -> frozenset[str]:
    """Notices already shown in this chat, recorded on its latest assistant turn."""
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        notices = (message.get("metadata") or {}).get(_REPORTED_NOTICES_KEY)
        if not isinstance(notices, list):
            return frozenset()
        return frozenset(n for n in notices if isinstance(n, str))
    return frozenset()


def _notice_text(event: dict[str, Any]) -> str | None:
    """Text of an informational event; errors are never treated as notices."""
    if event.get("type") not in ("status", "warning"):
        return None
    text = event.get("message") or event.get("content")
    return text if isinstance(text, str) else None


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


def _storage_hint(exc: ChatStorageError) -> str:
    """Only an auth failure is actionable by the user; other errors are ours."""
    if exc.status in (401, 403):
        return " (проверьте токен)"
    return ""


def _services_warning(detail: str, reason: str) -> dict[str, Any]:
    return {
        "type": "warning",
        "stage": "service_normatives",
        "detail": detail,
        "message": f"Сервисы (школы, детские сады и т. п.) не расставлены: {reason}.",
    }


async def _region_for_services(
    territory_id: int | None,
    project_id: int | str | None,
    urban_api: Any | None,
    token: str | None,
    default_territory_id: int | None = None,
) -> tuple[int | None, dict[str, Any] | None]:
    """Region whose normatives place services in the blocks-file mode, plus an event to report.

    An explicit ``territory_id`` wins; otherwise the region of ``project_id`` is looked up.
    Without both, ``default_territory_id`` is used and announced by a status event.
    """
    if territory_id is not None:
        return territory_id, None
    if project_id is None and default_territory_id is not None:
        return default_territory_id, {
            "type": "status",
            "stage": "service_normatives",
            "content": (
                "Регион не указан — сервисы (школы, детские сады и т. п.) расставлены "
                f"по нормативам региона по умолчанию (territory_id={default_territory_id})."
            ),
        }
    if project_id is None or urban_api is None:
        return None, _services_warning(
            "neither territory_id nor project_id is set",
            "не указан регион (territory_id) или проект",
        )
    try:
        return await urban_api.get_region_by_project(project_id, token), None
    except (HTTPException, aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning("project {} region lookup failed: {}", project_id, exc)
        return None, _services_warning(str(exc), "не удалось определить регион проекта")


def _request_metadata(
    *,
    scenario_id: int | None,
    year: int | None,
    source: str | None,
    project_id: int | str | None,
    functional_zone_types: list[str] | None,
    has_blocks_file: bool,
    has_buildings_file: bool,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Territory context stored alongside the chat and every persisted turn."""
    metadata: dict[str, Any] = {
        "blocks_file": has_blocks_file,
        "buildings_file": has_buildings_file,
    }
    optional = {
        "scenario_id": scenario_id,
        "year": year,
        "source": source,
        "project_id": project_id,
        "functional_zone_types": functional_zone_types,
    }
    metadata.update({key: value for key, value in optional.items() if value is not None})
    if extra:
        metadata.update(extra)
    return metadata


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
    return merged, summarize_buildings(gen_features)


async def stream_generation_chat(
    *,
    builder: Any,
    llm_client: VLLMChatClient,
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
    existing_buildings_geojson: dict[str, Any] | None = None,
    existing_buildings_declined: bool = False,
    territory_id: int | None = None,
    default_territory_id: int | None = None,
    generation_parameters: dict[str, Any] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    message_metadata: dict[str, Any] | None = None,
    zones_service: Any | None = None,
    urban_api: Any | None = None,
    object_storage: ObjectStorage | None = None,
    public_base_url: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    persist = chat_storage_client is not None and bool(user_id)
    metadata = _request_metadata(
        scenario_id=scenario_id,
        year=year,
        source=source,
        project_id=project_id,
        functional_zone_types=functional_zone_types,
        has_blocks_file=blocks_geojson is not None,
        has_buildings_file=existing_buildings_geojson is not None,
        extra=message_metadata,
    )

    # 0. Load prior turns (existing chat) so short follow-ups keep context.
    prior_text = ""
    already_reported: frozenset[str] = frozenset()
    if persist and chat_id:
        try:
            existing = await chat_storage_client.get_chat(user_id, chat_id)
            prior_text = _history_user_text(existing.get("messages") or [])
            already_reported = _reported_notices(existing.get("messages") or [])
        except ChatStorageError as exc:
            logger.warning("chat_storage get_chat (history) failed: {}", exc)
            yield {
                "type": "warning",
                "stage": "load_history",
                "detail": str(exc),
                "message": f"Не удалось загрузить историю чата{_storage_hint(exc)} — "
                "обрабатываю только текущее сообщение.",
            }

    # 1. Ensure a chat exists. Its title is written by the LLM: the raw first
    # message makes rows that can't be told apart in the history list.
    if persist and not chat_id:
        title = await make_chat_title(
            llm_client, user_query=user_query, model=model, fallback=chat_title
        )
        try:
            created = await chat_storage_client.create_chat(
                user_id,
                title=title,
                scenario_id=scenario_id,
                project_id=project_id,
                metadata=metadata,
            )
            chat_id = created.get("chat_id")
            yield {"type": "chat_created", "chat_id": chat_id, "title": created.get("title")}
        except ChatStorageError as exc:
            logger.warning("chat_storage create_chat failed: {}", exc)
            yield {
                "type": "warning",
                "stage": "create_chat",
                "detail": str(exc),
                "message": f"Диалог не будет сохранён в историю{_storage_hint(exc)}.",
            }
            persist = False

    # 2. Persist the user turn.
    if persist and chat_id:
        try:
            await chat_storage_client.add_message(
                user_id, chat_id, role="user", content=user_query, metadata=metadata
            )
        except ChatStorageError as exc:
            logger.warning("chat_storage add user message failed: {}", exc)
            yield {
                "type": "warning",
                "stage": "add_user_message",
                "detail": str(exc),
                "message": f"Сообщение не сохранено в историю{_storage_hint(exc)}.",
            }

    # Informational notices repeat on every turn of a clarification dialog
    # because the file and the request are resent; show each text only once
    # per chat and record what was shown on the assistant turn.
    reported: list[str] = []

    def first_report(event: dict[str, Any]) -> bool:
        text = _notice_text(event)
        if text is None:
            return True
        if text not in reported:
            reported.append(text)
        return text not in already_reported

    # 2.5 Resolve the territory source. A user-uploaded blocks file overrides the
    # scenario; derive the zones in scope from the actual territory so a missing
    # business zone never results in a business-demand clarification.
    blocks: BlockFeatureCollection | None = None
    zones_in_scope: tuple[str, ...] = GENERATED_ZONES
    scenario_zones_layer: dict[str, Any] | None = None
    kept: list[dict[str, Any]] = []
    if blocks_geojson is not None:
        kept, zones_in_scope, load_events = await _blocks_from_upload(
            blocks_geojson, llm_client, model
        )
        for event in load_events:
            if first_report(event):
                yield event
        if not kept:
            yield {"type": "done", "chat_id": chat_id, "assistant_message_id": None}
            return
        try:
            blocks = BlockFeatureCollection.model_validate(
                {"type": "FeatureCollection", "features": kept}
            )
        except Exception as exc:  # noqa: BLE001 - invalid geometry -> surface to client
            logger.warning("invalid blocks file: {}", exc)
            yield _load_error(
                "invalid_blocks",
                str(exc),
                "Загруженный файл зон не прошёл проверку: у части объектов некорректная "
                "геометрия или координаты.",
            )
            yield {"type": "done", "chat_id": chat_id, "assistant_message_id": None}
            return

    elif zones_service is not None and scenario_id is not None:
        # Fetch this before asking clarifying questions: the layer is the source
        # of truth for which generated zone types exist in the scenario. Reuse it
        # below for the map backdrop rather than making a second request.
        try:
            scenario_zones_layer = await zones_service.prepare_zones_layer(
                scenario_id=scenario_id,
                year=year,
                source=source,
                token=token,
                functional_zone_types=list(GENERATED_ZONES),
            )
            zones_in_scope = _zones_from_layer(scenario_zones_layer)
        except Exception as exc:  # noqa: BLE001 - preserve generation fallback
            logger.warning("functional zones scope lookup failed: {}", exc)
            yield {
                "type": "warning",
                "stage": "zones",
                "detail": str(exc),
                "message": "Не удалось определить состав функциональных зон сценария.",
            }

    # 2.6 Existing buildings, when the user uploaded them: their footprints are
    # cut out of the blocks before generation, and they come back in the result
    # marked ``is_excluded`` — so nothing is generated on top of what stands.
    existing_buildings: dict[str, Any] | None = None
    if existing_buildings_geojson is not None:
        kept_buildings, dropped_buildings = _existing_buildings_from_geojson(
            existing_buildings_geojson
        )
        if dropped_buildings:
            yield {
                "type": "warning",
                "stage": "load_existing_buildings",
                "detail": f"{dropped_buildings} feature(s) dropped",
                "message": "Отброшено объектов без полигональной геометрии в файле "
                f"существующих зданий: {dropped_buildings}.",
            }
        if kept_buildings:
            existing_buildings = {
                "type": "FeatureCollection",
                "features": kept_buildings,
            }
        else:
            yield {
                "type": "warning",
                "stage": "load_existing_buildings",
                "detail": "no polygonal features in uploaded file",
                "message": "В файле существующих зданий нет полигонов — генерация "
                "пойдёт без исключения существующей застройки.",
            }

    # 3. Extract targets from the (accumulated) request text.
    combined_query = f"{prior_text}\n{user_query}".strip() if prior_text else user_query
    extracted = await extract_generation_targets(
        llm_client,
        user_query=combined_query,
        la_per_person=la_per_person,
        model=model,
    )
    # The extractor is the only way free text becomes targets — when the call
    # itself failed there is nothing to ask for either, so say so instead of
    # asking for the parameters the user has already given.
    if extracted.error:
        yield {
            "type": "error",
            "stage": "param_extraction",
            "detail": extracted.error,
            "message": "Языковая модель недоступна — не удалось разобрать "
            "параметры генерации. Попробуйте позже.",
        }
        yield {"type": "done", "chat_id": chat_id, "assistant_message_id": None}
        return
    # Zones come from the territory source: the generated zones actually present
    # in a scenario or in an uploaded blocks file.
    extracted.functional_zone_types = list(zones_in_scope)

    # "2000 жителей" without a zone is the demand for the whole territory, not
    # for each zone: divide it between the zones by their area.
    zone_areas: dict[str, float] = {}
    if extracted.total_residents and len(zones_in_scope) > 1:
        zone_areas = _zone_areas(
            kept or (scenario_zones_layer or {}).get("features") or []
        )
    split = split_total_residents(extracted, zones_in_scope, zone_areas)
    if len(split) > 1:
        parts = ", ".join(f"{ZONE_LABELS.get(z, z)} — {n}" for z, n in split.items())
        split_event = {
            "type": "status",
            "stage": "param_extraction",
            "content": f"Общий спрос {extracted.total_residents} жителей распределён "
            f"по зонам{' пропорционально площади' if zone_areas else ' поровну'}: {parts}.",
        }
        if first_report(split_event):
            yield split_event

    # Pin the policy default floor group per zone unless the user set one
    # explicitly, so the result doesn't depend on the pipeline's fallbacks.
    default_fg = extracted.targets_by_zone.setdefault("default_floor_group", {})
    for zone, floor_group in DEFAULT_FLOOR_GROUP_BY_ZONE.items():
        default_fg.setdefault(zone, floor_group)

    # 4. Missing mandatory params -> ask, persist the question, stop.
    missing = validate_targets(extracted, zones_in_scope)

    # 4.1 Project-less mode: without a scenario there is nothing to take the
    # existing buildings from, so ask the user once — upload a file, or decline
    # explicitly. Bundled into the same clarification as the missing targets, so
    # everything is answered in a single round.
    if (
        scenario_id is None
        and existing_buildings_geojson is None
        and not existing_buildings_declined
    ):
        missing = [*missing, existing_buildings_question()]

    if missing:
        # An all-optional list means nothing is really incomplete except the
        # unanswered question itself — so don't say the request is.
        lead = (
            "Уточните перед генерацией:"
            if all(m.optional for m in missing)
            else "Чтобы сгенерировать застройку, уточните:"
        )
        content = lead + "\n" + "\n".join(f"— {m.prompt}" for m in missing)
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
                    "optional": m.optional,
                }
                for m in missing
            ],
        }
        assistant_message_id = await _persist_assistant(
            chat_storage_client,
            persist,
            user_id,
            chat_id,
            content,
            {**metadata, _REPORTED_NOTICES_KEY: reported},
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
    # 5.1 Zones backdrop, inline and before generation, so the map can draw the
    # territory while buildings are still being computed. An uploaded file wins
    # over the scenario: the backdrop must match what actually went in.
    file_layers: list[dict[str, Any]] = []
    # One id for every artefact of this run, so all its links resolve together.
    result_id = uuid4().hex
    if blocks_geojson is not None:
        file_zones = {"type": "FeatureCollection", "features": kept}
        yield {"type": "zones", "source": "blocks_file", "content": file_zones}
        # No scenario to query live, so the zones are stored like our own
        # artefacts; otherwise the layer would vanish from the chat history.
        if object_storage is not None:
            async for event in _store_layers(
                object_storage,
                result_id,
                [(SLOT_ZONES, file_zones)],
                public_base_url,
                file_layers,
            ):
                yield event
    elif scenario_zones_layer is not None:
        # The same layer determined ``zones_in_scope`` before clarification.
        yield {"type": "zones", "source": "scenario", "content": scenario_zones_layer}
        descriptor = build_zones_layer(
            scenario_id=scenario_id,
            year=year,
            source=source,
            functional_zone_types=list(extracted.functional_zone_types),
            public_base_url=public_base_url,
        )
        file_layers.append(descriptor)
        yield {"type": "file", **descriptor}
    elif zones_service is not None and scenario_id is not None:
        try:
            zones_layer = await zones_service.prepare_zones_layer(
                scenario_id=scenario_id,
                year=year,
                source=source,
                token=token,
                functional_zone_types=list(extracted.functional_zone_types),
            )
        except Exception as exc:  # noqa: BLE001 - the backdrop is optional, generation is not
            logger.warning("functional zones backdrop failed: {}", exc)
            yield {
                "type": "warning",
                "stage": "zones",
                "detail": str(exc),
                "message": "Не удалось загрузить слой функциональных зон.",
            }
        else:
            yield {"type": "zones", "source": "scenario", "content": zones_layer}
            descriptor = build_zones_layer(
                scenario_id=scenario_id,
                year=year,
                source=source,
                functional_zone_types=list(extracted.functional_zone_types),
                public_base_url=public_base_url,
            )
            file_layers.append(descriptor)
            yield {"type": "file", **descriptor}

    region_id: int | None = None
    if blocks_geojson is not None and scenario_id is None:
        region_id, region_event = await _region_for_services(
            territory_id, project_id, urban_api, token, default_territory_id
        )
        if region_event is not None:
            yield region_event

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
            existing_buildings=existing_buildings,
            territory_id=region_id,
        )
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the client
        logger.exception("generation failed")
        yield {"type": "error", "stage": "generation", "detail": str(exc)}
        yield {"type": "done", "chat_id": chat_id, "assistant_message_id": None}
        return

    merged, summary = _merge_result(result)
    yield {"type": "result", "content": merged, "summary": summary}
    if (
        region_id is not None
        and isinstance(result, dict)
        and result.get("service_normatives_loaded") is False
    ):
        yield _services_warning(
            f"no service normatives loaded for territory {region_id}",
            f"нормативы региона {region_id} не загрузились или пусты",
        )

    # 5.2 Store the artefacts and hand out durable links. Done after ``result``
    # so the client sees the buildings without waiting on the write, and
    # best-effort: a storage failure is a warning, never the end of the stream.
    if object_storage is not None:
        payloads = [(SLOT_BUILDINGS, merged)]
        if blocks_geojson is not None:
            payloads.append((SLOT_BLOCKS_INPUT, blocks_geojson))
        if existing_buildings is not None:
            payloads.append((SLOT_EXISTING_BUILDINGS, existing_buildings))
        async for event in _store_layers(
            object_storage, result_id, payloads, public_base_url, file_layers
        ):
            yield event

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
        async for delta in llm_client.stream_chat(
            summary_messages, model=model, temperature=temperature
        ):
            collected.append(delta)
            yield {"type": "token", "content": delta}
    except VLLMChatError as exc:
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
        {**metadata, _REPORTED_NOTICES_KEY: reported},
        file_parts=[geo_layer_to_file_part(layer) for layer in file_layers],
    )
    yield {"type": "done", "chat_id": chat_id, "assistant_message_id": assistant_message_id}


async def _store_layers(
    object_storage: Any,
    result_id: str,
    payloads: list[tuple[str, Any]],
    public_base_url: str | None,
    file_layers: list[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    """Write each payload to its slot and yield its ``file`` event.

    Best-effort: a failed write becomes a ``store_layer`` warning and the slot
    is skipped. Stored descriptors are appended to ``file_layers`` so they
    reach the chat history.
    """
    for slot, payload in payloads:
        try:
            await asyncio.to_thread(
                object_storage.put_json, payload, object_key(result_id, slot)
            )
        except (ObjectStorageError, OSError) as exc:
            logger.warning("storing layer {} failed: {}", slot, exc)
            yield {
                "type": "warning",
                "stage": "store_layer",
                "detail": str(exc),
                "message": f"Слой «{slot}» не сохранён — ссылка на него не "
                "появится в истории чата.",
            }
            continue
        descriptor = build_stored_layer(
            slot=slot, result_id=result_id, public_base_url=public_base_url
        )
        file_layers.append(descriptor)
        yield {"type": "file", **descriptor}


async def _persist_assistant(
    chat_storage_client: ChatStorageClient | None,
    persist: bool,
    user_id: str | None,
    chat_id: str | None,
    content: str,
    metadata: dict[str, Any] | None,
    file_parts: list[dict[str, Any]] | None = None,
) -> str | None:
    """Persist the assistant turn, carrying layer links when there are any.

    ChatStorage takes either ``content`` or ``parts``, and links only survive as
    ``file`` parts — so the answer text becomes a ``text`` part alongside them.
    """
    if not (persist and chat_id and content):
        return None
    try:
        if file_parts:
            parts = [{"kind": "text", "payload": {"text": content}}]
            parts += [{"kind": "file", "payload": part} for part in file_parts]
            stored = await chat_storage_client.add_message(
                user_id, chat_id, role="assistant", parts=parts, metadata=metadata
            )
        else:
            stored = await chat_storage_client.add_message(
                user_id, chat_id, role="assistant", content=content, metadata=metadata
            )
        return stored.get("message_id")
    except ChatStorageError as exc:
        logger.warning("chat_storage add assistant message failed: {}", exc)
        return None
