"""Extract and validate building-generation parameters from free text.

The agentic chat mode is deliberately narrow: it generates only the two
housing-demand zones — ``residential`` and ``business`` (multifunctional) — and
the target zones are implied by the selected scenario, so the user never picks
zones. The only thing we need from the user is the housing demand per zone.

This module turns the free-text request into the ``targets_by_zone`` structure
that ``Genbuilder.run`` consumes, and reports which mandatory parameters are
still missing so the SSE flow can ask for them (the ``clarification`` event).

Mandatory minimum:
- territory: taken from the endpoint's form fields (scenario_id + year + source);
- per generated zone (residential, business): a housing demand —
  ``residents`` OR ``living_area`` (interchangeable:
  ``living_area = residents * la_per_person``).

Everything else (floors_avg, density_scenario, default_floor_group) is optional
and falls back to service defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from loguru import logger

from app.infrastructure.vllm_chat_client import VLLMChatClient, VLLMChatError

# The agentic chat mode generates only these two zones. Zones are implied by the
# scenario, not chosen by the user; both are housing-demand zones.
GENERATED_ZONES: tuple[str, ...] = ("residential", "business")
KNOWN_ZONES: tuple[str, ...] = GENERATED_ZONES
DENSITY_SCENARIOS: tuple[str, ...] = ("min", "mean", "max")

# Policy default floor group per generated zone (not extracted from user text):
# residential -> medium (5–8 этажей), business -> high (9–16 этажей). Guarantees a
# sensible default independent of the pipeline's internal fallbacks.
DEFAULT_FLOOR_GROUP_BY_ZONE: dict[str, str] = {
    "residential": "medium",
    "business": "high",
}

# Human-readable zone labels for clarification prompts.
_ZONE_LABELS: dict[str, str] = {
    "residential": "жилая",
    "business": "многофункциональная",
}

_EXTRACTION_SYSTEM_PROMPT = (
    "Ты — парсер параметров для генерации застройки. Генерируются только две "
    "зоны: residential (жилая) и business (многофункциональная). Из запроса "
    "пользователя извлеки спрос на жильё по этим зонам. Отвечай строго по схеме. "
    "Заполняй только те значения, которые пользователь назвал явно; всё "
    "остальное оставляй null. Ничего не выдумывай. Числа — без единиц измерения. "
    "residents — число жителей, living_area — жилая площадь в м², "
    "floors_avg — средняя этажность, buildings_count — явно указанное число "
    "зданий, density_scenario — один из: min, mean, max. "
    "Также извлеки стиль фасадов, только если пользователь явно его указал. "
    "facade_style_name_ru — короткое название стиля на русском, "
    "facade_style_prompt — короткое описание этого стиля на английском для "
    "генерации архитектурной текстуры. Если стиль не указан, оставь оба поля null."
)


def build_extraction_schema() -> dict[str, Any]:
    """JSON schema for guided decoding — constrains zones to KNOWN_ZONES."""
    return {
        "type": "object",
        "properties": {
            "facade_style_name_ru": {"type": ["string", "null"]},
            "facade_style_prompt": {"type": ["string", "null"]},
            "zones": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "zone": {"type": "string", "enum": list(KNOWN_ZONES)},
                        "residents": {"type": ["integer", "null"]},
                        "living_area": {"type": ["number", "null"]},
                        "floors_avg": {"type": ["number", "null"]},
                        "buildings_count": {
                            "type": ["integer", "null"],
                            "minimum": 1,
                        },
                        "density_scenario": {
                            "type": ["string", "null"],
                            "enum": [*DENSITY_SCENARIOS, None],
                        },
                    },
                    "required": ["zone"],
                },
            }
        },
        # Guided decoding must emit the nullable style keys.  When optional,
        # some models omit them even after recognizing a style in the text,
        # making an explicit style indistinguishable from no style at all.
        "required": ["zones", "facade_style_name_ru", "facade_style_prompt"],
    }


@dataclass
class Missing:
    """One unmet mandatory requirement, rendered into a clarification prompt.

    ``control``/``unit``/``alt_fields`` let the frontend render the right input.
    ``optional`` marks a question the user may decline (generation is not blocked
    by the answer itself, only by the fact that it hasn't been asked yet);
    ``zone`` is ``None`` for a question that isn't about one particular zone.
    """

    zone: str | None
    field: str
    prompt: str
    control: str = "number"
    unit: str | None = "чел. или м²"
    alt_fields: tuple[str, ...] = ("residents", "living_area")
    optional: bool = False


@dataclass
class ExtractedTargets:
    """Normalized generation targets plus the requested zone list."""

    targets_by_zone: dict[str, dict[str, Any]] = field(default_factory=dict)
    functional_zone_types: list[str] = field(default_factory=list)
    facade_style_name_ru: str | None = None
    facade_style_prompt: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if num > 0 else None


def _text(value: Any, *, max_length: int = 4000) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:max_length] if value else None


def normalize_targets(raw: dict[str, Any], la_per_person: float) -> ExtractedTargets:
    """Turn raw LLM output into ``targets_by_zone`` + the requested zone list.

    Housing demand given as ``living_area`` is converted to ``residents`` (the
    only housing key ``Genbuilder.run`` consumes) via ``la_per_person``.
    """
    residents: dict[str, int] = {}
    floors_avg: dict[str, float] = {}
    buildings_count: dict[str, int] = {}
    density_scenario: dict[str, str] = {}
    requested: list[str] = []

    for item in raw.get("zones") or []:
        zone = str(item.get("zone") or "").strip()
        if zone not in KNOWN_ZONES:
            continue
        if zone not in requested:
            requested.append(zone)

        res = _num(item.get("residents"))
        la = _num(item.get("living_area"))
        if res is None and la is not None:
            res = round(la / la_per_person) if la_per_person > 0 else None
        if res is not None:
            residents[zone] = int(res)

        floors = _num(item.get("floors_avg"))
        if floors is not None:
            floors_avg[zone] = floors

        count = _num(item.get("buildings_count"))
        if count is not None:
            buildings_count[zone] = max(1, int(round(count)))

        dens = item.get("density_scenario")
        if isinstance(dens, str) and dens.strip() in DENSITY_SCENARIOS:
            density_scenario[zone] = dens.strip()

    targets_by_zone: dict[str, dict[str, Any]] = {}
    if residents:
        targets_by_zone["residents"] = residents
    if floors_avg:
        targets_by_zone["floors_avg"] = floors_avg
    if buildings_count:
        targets_by_zone["buildings_count"] = buildings_count
    if density_scenario:
        targets_by_zone["density_scenario"] = density_scenario

    return ExtractedTargets(
        targets_by_zone=targets_by_zone,
        functional_zone_types=requested,
        facade_style_name_ru=_text(raw.get("facade_style_name_ru")),
        facade_style_prompt=_text(raw.get("facade_style_prompt")),
        raw=raw,
    )


async def extract_generation_targets(
    llm_client: VLLMChatClient,
    *,
    user_query: str,
    la_per_person: float,
    model: str | None = None,
) -> ExtractedTargets:
    """Ask the LLM to extract targets from ``user_query`` (structured output).

    On any LLM failure returns empty targets so the caller falls through to
    clarification rather than crashing the stream.
    """
    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    try:
        raw = await llm_client.complete_json(
            messages, schema=build_extraction_schema(), model=model
        )
    except VLLMChatError as exc:
        logger.warning("param extraction failed: {}", exc)
        return ExtractedTargets(raw={"error": str(exc)})
    return normalize_targets(raw, la_per_person)


def validate_targets(
    extracted: ExtractedTargets, zones: Iterable[str] | None = None
) -> list[Missing]:
    """Return the unmet mandatory requirements (empty list == ready to generate).

    The only requirement is a housing demand per in-scope zone — at most one
    clarification per zone. ``zones`` defaults to ``GENERATED_ZONES`` (the
    scenario branch generates both); when the user uploads their own blocks, it
    is the zones actually present in the file, so we don't over-ask.
    """
    zones = tuple(zones) if zones is not None else GENERATED_ZONES
    missing: list[Missing] = []

    residents = extracted.targets_by_zone.get("residents") or {}
    for zone in zones:
        if zone not in GENERATED_ZONES:
            continue
        if not residents.get(zone):
            label = _ZONE_LABELS.get(zone, zone)
            missing.append(
                Missing(
                    zone=zone,
                    field="residents|living_area",
                    prompt=(
                        f"Для зоны «{label}» ({zone}) укажите спрос на жильё — "
                        "число жителей (residents) ИЛИ жилую площадь в м² "
                        "(living_area)."
                    ),
                )
            )

    return missing


def existing_buildings_question() -> Missing:
    """The project-less mode's question about already existing buildings.

    Without a scenario there is nothing to take existing buildings from, so the
    frontend has to ask: upload a GeoJSON file (``buildings_file``) and their
    footprints are cut out of the blocks before generation, or decline
    (``skip_existing_buildings=true``) and generation runs on the bare blocks.
    Optional — but asked once before the first generation, so nothing is built
    on top of what is already standing.
    """
    return Missing(
        zone=None,
        field="existing_buildings",
        prompt=(
            "Хотите загрузить существующие здания? Они будут исключены из "
            "генерации — приложите файл GeoJSON или откажитесь, и застройка "
            "будет сгенерирована по всей площади кварталов."
        ),
        control="file_or_skip",
        unit=None,
        alt_fields=("buildings_file", "skip_existing_buildings"),
        optional=True,
    )
