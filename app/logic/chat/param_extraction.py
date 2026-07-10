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

from app.infrastructure.ollama_chat_client import OllamaChatClient, OllamaChatError

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
    "floors_avg — средняя этажность, density_scenario — один из: min, mean, max."
)


def build_extraction_schema() -> dict[str, Any]:
    """JSON schema for Ollama structured output — constrains zones to KNOWN_ZONES."""
    return {
        "type": "object",
        "properties": {
            "zones": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "zone": {"type": "string", "enum": list(KNOWN_ZONES)},
                        "residents": {"type": ["integer", "null"]},
                        "living_area": {"type": ["number", "null"]},
                        "floors_avg": {"type": ["number", "null"]},
                        "density_scenario": {
                            "type": ["string", "null"],
                            "enum": [*DENSITY_SCENARIOS, None],
                        },
                    },
                    "required": ["zone"],
                },
            }
        },
        "required": ["zones"],
    }


@dataclass
class Missing:
    """One unmet mandatory requirement, rendered into a clarification prompt.

    ``control``/``unit``/``alt_fields`` let the frontend render the right input
    (all current requirements are a single numeric housing-demand field).
    """

    zone: str
    field: str
    prompt: str
    control: str = "number"
    unit: str = "чел. или м²"
    alt_fields: tuple[str, ...] = ("residents", "living_area")


@dataclass
class ExtractedTargets:
    """Normalized generation targets plus the requested zone list."""

    targets_by_zone: dict[str, dict[str, Any]] = field(default_factory=dict)
    functional_zone_types: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if num > 0 else None


def normalize_targets(raw: dict[str, Any], la_per_person: float) -> ExtractedTargets:
    """Turn raw LLM output into ``targets_by_zone`` + the requested zone list.

    Housing demand given as ``living_area`` is converted to ``residents`` (the
    only housing key ``Genbuilder.run`` consumes) via ``la_per_person``.
    """
    residents: dict[str, int] = {}
    floors_avg: dict[str, float] = {}
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

        dens = item.get("density_scenario")
        if isinstance(dens, str) and dens.strip() in DENSITY_SCENARIOS:
            density_scenario[zone] = dens.strip()

    targets_by_zone: dict[str, dict[str, Any]] = {}
    if residents:
        targets_by_zone["residents"] = residents
    if floors_avg:
        targets_by_zone["floors_avg"] = floors_avg
    if density_scenario:
        targets_by_zone["density_scenario"] = density_scenario

    return ExtractedTargets(
        targets_by_zone=targets_by_zone,
        functional_zone_types=requested,
        raw=raw,
    )


async def extract_generation_targets(
    ollama_client: OllamaChatClient,
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
        raw = await ollama_client.complete_json(
            messages, schema=build_extraction_schema(), model=model
        )
    except OllamaChatError as exc:
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
