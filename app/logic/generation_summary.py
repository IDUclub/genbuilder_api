"""Compact summaries over generation results.

Shared by the conversational flow (``app.logic.chat.generation_chat``) and the
MCP tools, so an agent gets the same numbers the chat UI grounds its answer
on — without having to walk thousands of GeoJSON features itself.
"""
from __future__ import annotations

from typing import Any, Iterable, Literal, Optional

TargetsSource = Literal["request", "service_defaults"]


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def summarize_buildings(features: list[dict[str, Any]]) -> dict[str, Any]:
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


def combine_targets(targets: Iterable[Optional[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    """Sum the numeric ``residents`` / ``coverage_area`` targets of several runs
    (e.g. one ``targets_by_zone`` per functional zone) into one per-zone total."""
    combined: dict[str, dict[str, float]] = {}
    for targets_by_zone in targets:
        for key in ("residents", "coverage_area"):
            for zone, value in ((targets_by_zone or {}).get(key) or {}).items():
                bucket = combined.setdefault(key, {})
                bucket[zone] = bucket.get(zone, 0.0) + _number(value)
    return combined


def _target_report(
    targets_by_zone: Optional[dict[str, Any]],
    generated: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    achieved_residents: dict[str, float] = {}
    achieved_functional_area: dict[str, float] = {}
    for feature in generated:
        props = feature.get("properties") or {}
        zone = props.get("zone")
        if not zone:
            continue
        achieved_residents[zone] = achieved_residents.get(zone, 0.0) + _number(props.get("residents_number"))
        achieved_functional_area[zone] = achieved_functional_area.get(zone, 0.0) + _number(props.get("functional_area"))

    report: dict[str, dict[str, Any]] = {}
    targets_by_zone = targets_by_zone or {}

    for zone, value in (targets_by_zone.get("residents") or {}).items():
        target = _number(value)
        if target <= 0:
            continue
        achieved = int(round(achieved_residents.get(zone, 0.0)))
        report.setdefault(zone, {}).update(
            {
                "target_residents": int(round(target)),
                "achieved_residents": achieved,
                "residents_deficit": max(int(round(target)) - achieved, 0),
            }
        )

    for zone, value in (targets_by_zone.get("coverage_area") or {}).items():
        target = _number(value)
        if target <= 0:
            continue
        achieved = round(achieved_functional_area.get(zone, 0.0), 1)
        report.setdefault(zone, {}).update(
            {
                "target_functional_area": round(target, 1),
                "achieved_functional_area": achieved,
                "functional_area_deficit": round(max(target - achieved, 0.0), 1),
            }
        )

    return report


def build_generation_summary(
    features: list[dict[str, Any]],
    *,
    targets_by_zone: Optional[dict[str, Any]] = None,
    targets_source: Optional[TargetsSource] = None,
    existing_buildings_preserved: Optional[bool] = None,
) -> dict[str, Any]:
    """Summarize a merged generation FeatureCollection (generated + excluded).

    Generated totals cover only newly generated buildings; features flagged
    ``is_excluded`` (existing buildings / excluded physical objects) are
    counted separately. ``targets`` compares the requested residents /
    non-residential functional area per zone with what was achieved, so a
    caller can see the deficit without recomputing it.
    """
    generated = [f for f in features if not (f.get("properties") or {}).get("is_excluded")]
    excluded = [f for f in features if (f.get("properties") or {}).get("is_excluded")]

    summary = summarize_buildings(generated)

    residents_by_zone: dict[str, int] = {}
    for feature in generated:
        props = feature.get("properties") or {}
        zone = props.get("zone")
        if zone:
            residents_by_zone[zone] = residents_by_zone.get(zone, 0) + int(props.get("residents_number") or 0)
    summary["residents_by_zone"] = residents_by_zone

    summary["excluded_buildings"] = len(excluded)
    summary["existing_living_area"] = round(
        sum(_number((f.get("properties") or {}).get("living_area")) for f in excluded), 1
    )
    summary["existing_residents"] = int(
        round(sum(_number((f.get("properties") or {}).get("residents_number")) for f in excluded))
    )

    summary["targets"] = _target_report(targets_by_zone, generated)
    if targets_source is not None:
        summary["targets_source"] = targets_source
    if existing_buildings_preserved is not None:
        summary["existing_buildings_preserved"] = existing_buildings_preserved
    return summary


__all__ = [
    "TargetsSource",
    "build_generation_summary",
    "combine_targets",
    "summarize_buildings",
]
