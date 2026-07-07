"""Extract and validate building-generation parameters from free text.

The conversational endpoint receives the user's request as free text (plus a
few structured form fields — scenario_id, year, source). This module turns that
text into the ``targets_by_zone`` structure that ``Genbuilder.run`` consumes,
and reports which mandatory parameters are still missing so the SSE flow can ask
the user for them (the ``clarification`` event).

Mandatory minimum (see the pipeline diagram):
- territory: taken from the endpoint's form fields (scenario_id + year + source);
- at least one target functional zone;
- per zone, a demand driver:
    * residential-family zones (residential, business, unknown) need a housing
      demand — ``residents`` OR ``living_area`` (they are interchangeable:
      ``living_area = residents * la_per_person``);
    * coverage-family zones (industrial, transport, special) need ``coverage_area``.

Everything else (floors_avg, density_scenario, default_floor_group) is optional
and falls back to service defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.infrastructure.ollama_chat_client import OllamaChatClient, OllamaChatError

# Zone families, mirroring the grouping in Genbuilder.run.
RESIDENTIAL_FAMILY: tuple[str, ...] = ("residential", "business", "unknown")
COVERAGE_FAMILY: tuple[str, ...] = ("industrial", "transport", "special")
KNOWN_ZONES: tuple[str, ...] = RESIDENTIAL_FAMILY + COVERAGE_FAMILY
DENSITY_SCENARIOS: tuple[str, ...] = ("min", "mean", "max")

_EXTRACTION_SYSTEM_PROMPT = (
    "Ты — парсер параметров для генерации застройки. Из запроса пользователя "
    "извлеки цели по функциональным зонам. Отвечай строго по схеме. "
    "Заполняй только те значения, которые пользователь назвал явно; всё "
    "остальное оставляй null. Ничего не выдумывай. Числа — без единиц измерения. "
    "residents — число жителей, living_area — жилая площадь в м², "
    "coverage_area — площадь застройки нежилых зон в м², floors_avg — средняя "
    "этажность, density_scenario — один из: min, mean, max."
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
                        "coverage_area": {"type": ["number", "null"]},
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
    """One unmet mandatory requirement, rendered into a clarification prompt."""

    zone: str
    field: str
    prompt: str


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
    coverage_area: dict[str, float] = {}
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

        cov = _num(item.get("coverage_area"))
        if cov is not None:
            coverage_area[zone] = cov

        floors = _num(item.get("floors_avg"))
        if floors is not None:
            floors_avg[zone] = floors

        dens = item.get("density_scenario")
        if isinstance(dens, str) and dens.strip() in DENSITY_SCENARIOS:
            density_scenario[zone] = dens.strip()

    targets_by_zone: dict[str, dict[str, Any]] = {}
    if residents:
        targets_by_zone["residents"] = residents
    if coverage_area:
        targets_by_zone["coverage_area"] = coverage_area
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


def validate_targets(extracted: ExtractedTargets) -> list[Missing]:
    """Return the unmet mandatory requirements (empty list == ready to generate)."""
    missing: list[Missing] = []

    if not extracted.functional_zone_types:
        missing.append(
            Missing(
                zone="*",
                field="functional_zone_types",
                prompt=(
                    "Какие зоны застраивать? Укажите одну или несколько: "
                    + ", ".join(KNOWN_ZONES)
                    + "."
                ),
            )
        )
        return missing

    residents = extracted.targets_by_zone.get("residents") or {}
    coverage = extracted.targets_by_zone.get("coverage_area") or {}

    for zone in extracted.functional_zone_types:
        if zone in RESIDENTIAL_FAMILY:
            if not residents.get(zone):
                missing.append(
                    Missing(
                        zone=zone,
                        field="residents|living_area",
                        prompt=(
                            f"Для зоны «{zone}» укажите спрос на жильё — "
                            "число жителей (residents) ИЛИ жилую площадь "
                            "в м² (living_area)."
                        ),
                    )
                )
        elif zone in COVERAGE_FAMILY:
            if not coverage.get(zone):
                missing.append(
                    Missing(
                        zone=zone,
                        field="coverage_area",
                        prompt=(
                            f"Для зоны «{zone}» укажите площадь застройки "
                            "coverage_area в м²."
                        ),
                    )
                )

    return missing
