"""Facade style normalization for the ``facade-jobs`` integration.

The public GenBuilder API accepts Russian or English style names.  Facades-3D
works more predictably with short English prompts, while the frontend needs a
Russian name to show to the user.  This module keeps that translation in one
place and leaves an omitted style as an empty override so ``facade-jobs`` can
apply its own per-zone defaults.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


DEFAULT_FACADE_STYLE_NAME_RU = "По умолчанию для функциональной зоны"

# Keep the architectural subject when adding a user-selected visual style.
# These prompts mirror the current zone defaults in facade-jobs.
_ZONE_BASE_PROMPTS: dict[str, str] = {
    "residential": "residential apartment building facade, windows, balconies",
    "business": "office and commercial building facade",
    "industrial": "industrial warehouse and factory building facade",
    "transport": "transport infrastructure building facade",
    "special": "public institution building facade",
    "recreation": "low-rise pavilion facade",
    "agriculture": "farm building facade",
}
_DEFAULT_ZONE_PROMPT = "urban building facade"


@dataclass(frozen=True, slots=True)
class FacadeStyle:
    """A normalized display name and the prompt sent to ``facade-jobs``."""

    name_ru: str
    prompt: str | None
    source: Literal["default", "preset", "free_text"]
    style_id: str | None = None


@dataclass(frozen=True, slots=True)
class FacadeStylePreset:
    """A named style; ``style_id`` addresses its facade library sections."""

    style_id: str
    name_ru: str
    prompt: str
    aliases: tuple[str, ...]


FACADE_STYLE_PRESETS: tuple[FacadeStylePreset, ...] = (
    FacadeStylePreset(
        "contemporary",
        "Современный",
        "contemporary architecture, clean lines, modern facade materials",
        ("современный", "современная", "contemporary", "modern"),
    ),
    FacadeStylePreset(
        "classic",
        "Классический",
        "classical architecture, symmetrical facade, restrained ornament",
        ("классический", "классика", "classic", "classical"),
    ),
    FacadeStylePreset(
        "neoclassic",
        "Неоклассический",
        "neoclassical architecture, symmetrical facade, elegant stone details",
        ("неоклассический", "неоклассика", "neoclassic", "neoclassical"),
    ),
    FacadeStylePreset(
        "art-nouveau",
        "Модерн",
        "Art Nouveau architecture, organic curves, decorative facade details",
        ("модерн", "ар нуво", "ар-нуво", "art nouveau"),
    ),
    FacadeStylePreset(
        "brick",
        "Кирпичный",
        "exposed brick facade, detailed brickwork",
        ("кирпичный", "кирпич", "brick", "brickwork"),
    ),
    FacadeStylePreset(
        "glass",
        "Стеклянный",
        "glass curtain wall facade, reflective glazing",
        ("стеклянный", "стекло", "glass", "glass facade"),
    ),
    FacadeStylePreset(
        "industrial",
        "Индустриальный",
        "industrial architecture, metal wall panels, exposed structure",
        ("индустриальный", "промышленный", "industrial"),
    ),
    FacadeStylePreset(
        "minimalist",
        "Минималистичный",
        "minimalist architecture, simple geometry, restrained material palette",
        ("минималистичный", "минимализм", "minimal", "minimalist"),
    ),
    FacadeStylePreset(
        "scandinavian",
        "Скандинавский",
        "Scandinavian architecture, light wood, pale colors, simple details",
        ("скандинавский", "сканди", "scandinavian", "nordic"),
    ),
    FacadeStylePreset(
        "loft",
        "Лофт",
        "loft style, dark metal, exposed brick, large industrial windows",
        ("лофт", "loft"),
    ),
    FacadeStylePreset(
        "timber",
        "Деревянный",
        "natural timber cladding, warm wood facade",
        ("деревянный", "дерево", "wood", "wooden", "timber"),
    ),
)


def _normalize_name(value: str) -> str:
    normalized = value.strip().lower().replace("ё", "е")
    normalized = re.sub(r"[_\-]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized)


_PRESET_BY_ALIAS: dict[str, FacadeStylePreset] = {
    _normalize_name(alias): preset
    for preset in FACADE_STYLE_PRESETS
    for alias in (preset.style_id, *preset.aliases)
}
PRESETS_BY_ID: dict[str, FacadeStylePreset] = {
    preset.style_id: preset for preset in FACADE_STYLE_PRESETS
}

ZONE_DEFAULT_STYLE_IDS: dict[str, str] = {
    "residential": "contemporary",
    "business": "glass",
    "industrial": "industrial",
    "recreation": "timber",
    "agriculture": "timber",
    "special": "neoclassic",
    "transport": "minimalist",
}
_FALLBACK_ZONE_STYLE_ID = "contemporary"
LIBRARY_BASE_ZONE = "residential"
_DEFAULT_ALIASES = {
    "default",
    "auto",
    "automatic",
    "авто",
    "автоматически",
    "по умолчанию",
    "дефолтный",
}


def resolve_facade_style(
    style: str | None,
    *,
    name_ru: str | None = None,
) -> FacadeStyle:
    """Resolve a public style value into a Russian label and English prompt.

    Unknown values are intentionally accepted as free text.  In the chat flow
    the LLM supplies their English prompt and Russian display name; direct API
    callers can also pass an already prepared English prompt.
    """
    cleaned = (style or "").strip()
    if not cleaned or _normalize_name(cleaned) in _DEFAULT_ALIASES:
        return FacadeStyle(
            name_ru=DEFAULT_FACADE_STYLE_NAME_RU,
            prompt=None,
            source="default",
        )

    preset = _PRESET_BY_ALIAS.get(_normalize_name(cleaned))
    if preset is not None:
        return FacadeStyle(
            name_ru=preset.name_ru,
            prompt=preset.prompt,
            source="preset",
            style_id=preset.style_id,
        )

    return FacadeStyle(
        name_ru=(name_ru or cleaned).strip(),
        prompt=cleaned,
        source="free_text",
    )


def build_style_by_zone(
    buildings: dict[str, Any],
    style: FacadeStyle,
) -> dict[str, dict[str, str]]:
    """Build facade-jobs overrides for the zones present in ``buildings``."""
    if style.prompt is None:
        return {}

    zones = _present_zones(buildings)
    return {
        zone: {
            "prompt": (
                f"{_ZONE_BASE_PROMPTS.get(zone, _DEFAULT_ZONE_PROMPT)}, "
                f"{style.prompt}"
            )[:4000]
        }
        for zone in sorted(zones)
    }


def _present_zones(buildings: dict[str, Any]) -> set[str]:
    zones: set[str] = set()
    for feature in buildings.get("features") or []:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            continue
        zone = str(properties.get("zone") or "unknown").strip() or "unknown"
        zones.add(zone)

    if not zones:
        zones.add("unknown")
    return zones


def library_style_by_zone(
    buildings: dict[str, Any],
    style: FacadeStyle,
) -> dict[str, str] | None:
    """Map each present zone to a facade library ``style_id``.

    An omitted style picks the zone's default preset. Free text has no library
    sections, so it yields ``None``.
    """
    if style.source == "free_text":
        return None
    zones = _present_zones(buildings)
    if style.style_id is not None:
        return {zone: style.style_id for zone in sorted(zones)}
    return {
        zone: ZONE_DEFAULT_STYLE_IDS.get(zone, _FALLBACK_ZONE_STYLE_ID)
        for zone in sorted(zones)
    }


def library_prompt(preset: FacadeStylePreset) -> str:
    """Prompt used to generate a preset's library sections."""
    return f"{_ZONE_BASE_PROMPTS[LIBRARY_BASE_ZONE]}, {preset.prompt}"


FACADE_STYLE_NAMES_RU: tuple[str, ...] = tuple(
    preset.name_ru for preset in FACADE_STYLE_PRESETS
)


__all__ = [
    "DEFAULT_FACADE_STYLE_NAME_RU",
    "FACADE_STYLE_NAMES_RU",
    "FACADE_STYLE_PRESETS",
    "PRESETS_BY_ID",
    "ZONE_DEFAULT_STYLE_IDS",
    "FacadeStyle",
    "FacadeStylePreset",
    "build_style_by_zone",
    "library_prompt",
    "library_style_by_zone",
    "resolve_facade_style",
]
