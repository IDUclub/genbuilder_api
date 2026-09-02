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


@dataclass(frozen=True, slots=True)
class _Preset:
    name_ru: str
    prompt: str
    aliases: tuple[str, ...]


_PRESETS: tuple[_Preset, ...] = (
    _Preset(
        "Современный",
        "contemporary architecture, clean lines, modern facade materials",
        ("современный", "современная", "contemporary", "modern"),
    ),
    _Preset(
        "Классический",
        "classical architecture, symmetrical facade, restrained ornament",
        ("классический", "классика", "classic", "classical"),
    ),
    _Preset(
        "Неоклассический",
        "neoclassical architecture, symmetrical facade, elegant stone details",
        ("неоклассический", "неоклассика", "neoclassic", "neoclassical"),
    ),
    _Preset(
        "Модерн",
        "Art Nouveau architecture, organic curves, decorative facade details",
        ("модерн", "ар нуво", "ар-нуво", "art nouveau"),
    ),
    _Preset(
        "Кирпичный",
        "exposed brick facade, detailed brickwork",
        ("кирпичный", "кирпич", "brick", "brickwork"),
    ),
    _Preset(
        "Стеклянный",
        "glass curtain wall facade, reflective glazing",
        ("стеклянный", "стекло", "glass", "glass facade"),
    ),
    _Preset(
        "Индустриальный",
        "industrial architecture, metal wall panels, exposed structure",
        ("индустриальный", "промышленный", "industrial"),
    ),
    _Preset(
        "Минималистичный",
        "minimalist architecture, simple geometry, restrained material palette",
        ("минималистичный", "минимализм", "minimal", "minimalist"),
    ),
    _Preset(
        "Скандинавский",
        "Scandinavian architecture, light wood, pale colors, simple details",
        ("скандинавский", "сканди", "scandinavian", "nordic"),
    ),
    _Preset(
        "Лофт",
        "loft style, dark metal, exposed brick, large industrial windows",
        ("лофт", "loft"),
    ),
    _Preset(
        "Деревянный",
        "natural timber cladding, warm wood facade",
        ("деревянный", "дерево", "wood", "wooden", "timber"),
    ),
)


def _normalize_name(value: str) -> str:
    normalized = value.strip().lower().replace("ё", "е")
    normalized = re.sub(r"[_\-]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized)


_PRESET_BY_ALIAS: dict[str, _Preset] = {
    _normalize_name(alias): preset for preset in _PRESETS for alias in preset.aliases
}
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

    return {
        zone: {
            "prompt": (
                f"{_ZONE_BASE_PROMPTS.get(zone, _DEFAULT_ZONE_PROMPT)}, "
                f"{style.prompt}"
            )[:4000]
        }
        for zone in sorted(zones)
    }


FACADE_STYLE_NAMES_RU: tuple[str, ...] = tuple(preset.name_ru for preset in _PRESETS)


__all__ = [
    "DEFAULT_FACADE_STYLE_NAME_RU",
    "FACADE_STYLE_NAMES_RU",
    "FacadeStyle",
    "build_style_by_zone",
    "resolve_facade_style",
]
