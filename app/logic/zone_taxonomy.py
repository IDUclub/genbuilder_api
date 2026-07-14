"""Canonical mapping of Urban DB functional zone types to generation zones.

Urban DB exposes granular ``functional_zone_type`` names (e.g.
``residential_individual`` = ИЖС, ``residential_lowrise``, ``mixed_use``). The
generation pipeline groups blocks by a small canonical set (``residential`` /
``business`` / ``industrial`` / ``transport`` / ``special``). This module
normalizes the granular names to that set and, for residential subtypes, derives
the natural floor group (building type) they imply.

The alias map is intentionally *additive*: names it does not list pass through
unchanged (``business``, ``industrial``, ``transport``, ``special``,
``unknown``, ...), so normalizing an already-canonical zone is a no-op and
existing behaviour is preserved.
"""
from __future__ import annotations

# Granular functional_zone_type.name -> canonical generation zone.
_ZONE_ALIASES: dict[str, str] = {
    "residential_individual": "residential",
    "residential_lowrise": "residential",
    "residential_midrise": "residential",
    "residential_multistorey": "residential",
    "mixed_use": "business",
}

# Residential subtype -> floor group (building type family) it implies.
# private = ИЖС, low = малоэтажная, medium = среднеэтажная, high = многоэтажная.
RESIDENTIAL_SUBTYPE_FLOOR_GROUP: dict[str, str] = {
    "residential_individual": "private",
    "residential_lowrise": "low",
    "residential_midrise": "medium",
    "residential_multistorey": "high",
}


def normalize_zone(name: object) -> str:
    """Map a raw functional_zone_type name to a canonical generation zone.

    Unknown / already-canonical names are returned unchanged (lower-cased).
    """
    key = str(name or "").strip().lower()
    return _ZONE_ALIASES.get(key, key)


def subtype_floor_group(name: object) -> str | None:
    """Floor group implied by a residential subtype, or None if not a subtype."""
    return RESIDENTIAL_SUBTYPE_FLOOR_GROUP.get(str(name or "").strip().lower())
