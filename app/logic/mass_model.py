"""Mass-model OBJ export for the Facades-3D generation service.

The consumer imposes a strict input contract (see ``src/gen_facades.py`` and
``src/gen_wall.py`` of CTLab-ITMO/Facades-3D):

- OBJ is Y-up in metres, ground at ``y = 0``;
- a wall is a face whose normal has ``|n_y| < 0.01``, and wall splitting only
  accepts 4-vertex faces, so walls must be emitted as quads;
- the face normal decides which way the generated facade looks, so wall quads
  must be wound counter-clockwise as seen from outside the building;
- horizontal faces are kept as flat planes (roofs) and may be n-gons;
- ``o <name>`` lines separate buildings and become node names of the resulting
  GLB, which is how a generated building is mapped back to its feature id.

Every OBJ exported for one territory must share a single ``LocalFrame`` so that
per-zone results can be merged into one scene.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import geopandas as gpd
from loguru import logger
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

DEFAULT_FLOOR_HEIGHT_M = 3.0
UNKNOWN_ZONE = "unknown"

_WALL_SIZE_ROUNDING_M = 0.5


@dataclass(frozen=True)
class MassModelParams:
    floor_height_m: float = DEFAULT_FLOOR_HEIGHT_M
    fallback_floors: int = 1
    min_segment_length_m: float = 0.2


@dataclass(frozen=True)
class LocalFrame:
    """Metric frame shared by every OBJ exported for one territory."""

    crs: str
    origin_x: float
    origin_y: float


@dataclass(frozen=True)
class MassModelStats:
    buildings: int
    walls: int
    roofs: int
    unique_wall_sizes: int
    skipped_features: int
    fallback_height_features: int
    roofs_covering_holes: int


def build_local_frame(feature_collection: Mapping[str, Any]) -> LocalFrame:
    """Pick a UTM zone and a local origin covering the whole feature collection."""
    geometries = [
        shape(feature["geometry"])
        for feature in _features(feature_collection)
        if feature.get("geometry")
    ]
    if not geometries:
        raise ValueError("Feature collection has no geometries to build a local frame from")

    series = gpd.GeoSeries(geometries, crs="EPSG:4326")
    utm_crs = series.estimate_utm_crs()
    min_x, min_y, max_x, max_y = series.to_crs(utm_crs).total_bounds

    return LocalFrame(
        crs=utm_crs.to_string(),
        origin_x=(min_x + max_x) / 2.0,
        origin_y=(min_y + max_y) / 2.0,
    )


def group_features_by_zone(
    feature_collection: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Split a building feature collection into one collection per zone."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for feature in _features(feature_collection):
        properties = feature.get("properties") or {}
        zone = str(properties.get("zone") or UNKNOWN_ZONE).strip() or UNKNOWN_ZONE
        grouped.setdefault(zone, []).append(feature)

    return {
        zone: {"type": "FeatureCollection", "features": features}
        for zone, features in grouped.items()
    }


def buildings_to_obj(
    feature_collection: Mapping[str, Any],
    frame: LocalFrame,
    params: MassModelParams | None = None,
) -> tuple[str, MassModelStats]:
    """Convert generated building footprints into a Facades-3D mass model."""
    params = params or MassModelParams()
    features = _features(feature_collection)
    if not features:
        raise ValueError("Feature collection has no features to export")

    geometries = [
        shape(feature["geometry"]) if feature.get("geometry") else None
        for feature in features
    ]
    projected = _project(geometries, frame)

    lines: list[str] = ["# genbuilder mass model", f"# crs {frame.crs}"]
    vertex_count = 0
    buildings = walls = roofs = 0
    skipped = fallback_height = roofs_covering_holes = 0
    wall_sizes: set[tuple[float, float]] = set()

    for index, (feature, geometry) in enumerate(zip(features, projected)):
        polygons = _polygons(geometry)
        if not polygons:
            skipped += 1
            continue

        floors, is_fallback = _floors(feature, params)
        if is_fallback:
            fallback_height += 1
        height = floors * params.floor_height_m

        building_lines, used_vertices, building_walls, building_roofs, holes = _building_obj(
            name=_building_name(feature, index),
            polygons=polygons,
            height=height,
            frame=frame,
            params=params,
            vertex_offset=vertex_count,
            wall_sizes=wall_sizes,
        )
        if not building_walls:
            skipped += 1
            continue

        lines.extend(building_lines)
        vertex_count += used_vertices
        buildings += 1
        walls += building_walls
        roofs += building_roofs
        roofs_covering_holes += holes

    if not walls:
        raise ValueError(
            "Mass model has no wall faces: every feature was empty, "
            "degenerate or zero-height"
        )

    stats = MassModelStats(
        buildings=buildings,
        walls=walls,
        roofs=roofs,
        unique_wall_sizes=len(wall_sizes),
        skipped_features=skipped,
        fallback_height_features=fallback_height,
        roofs_covering_holes=roofs_covering_holes,
    )
    if skipped:
        logger.warning("Mass model export skipped {} feature(s)", skipped)

    return "\n".join(lines) + "\n", stats


def _features(feature_collection: Mapping[str, Any]) -> list[dict[str, Any]]:
    features = feature_collection.get("features") or []
    if not isinstance(features, list):
        raise ValueError("`features` must be a list")
    return features


def _project(
    geometries: Sequence[BaseGeometry | None],
    frame: LocalFrame,
) -> list[BaseGeometry | None]:
    present = [(i, geom) for i, geom in enumerate(geometries) if geom is not None]
    if not present:
        return list(geometries)

    series = gpd.GeoSeries([geom for _, geom in present], crs="EPSG:4326").to_crs(frame.crs)
    projected: list[BaseGeometry | None] = list(geometries)
    for (index, _), geom in zip(present, series):
        projected[index] = geom
    return projected


def _polygons(geometry: BaseGeometry | None) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    return []


def _floors(feature: Mapping[str, Any], params: MassModelParams) -> tuple[int, bool]:
    properties = feature.get("properties") or {}
    try:
        floors = int(round(float(properties.get("floors_count"))))
    except (TypeError, ValueError):
        floors = 0

    if floors <= 0:
        return params.fallback_floors, True
    return floors, False


def _building_name(feature: Mapping[str, Any], index: int) -> str:
    properties = feature.get("properties") or {}
    physical_object_id = properties.get("physical_object_id")
    if physical_object_id is not None:
        raw_name = f"po_{physical_object_id}"
    elif feature.get("id") is not None:
        raw_name = f"gb_{feature['id']}"
    else:
        raw_name = f"gb_i{index}"
    return "_".join(raw_name.split())


def _building_obj(
    *,
    name: str,
    polygons: Sequence[Polygon],
    height: float,
    frame: LocalFrame,
    params: MassModelParams,
    vertex_offset: int,
    wall_sizes: set[tuple[float, float]],
) -> tuple[list[str], int, int, int, int]:
    vertex_lines: list[str] = []
    face_lines: list[str] = []
    used_vertices = 0
    walls = roofs = holes = 0

    for polygon in polygons:
        oriented = orient(polygon, sign=1.0)
        rings = [(oriented.exterior, True)] + [(ring, False) for ring in oriented.interiors]

        for ring, is_exterior in rings:
            points = _ring_points(ring, params.min_segment_length_m)
            if len(points) < 3:
                continue

            base = vertex_offset + used_vertices + 1
            for x, y in points:
                vertex_lines.append(_vertex(x - frame.origin_x, 0.0, -(y - frame.origin_y)))
                vertex_lines.append(_vertex(x - frame.origin_x, height, -(y - frame.origin_y)))
            used_vertices += 2 * len(points)

            for i in range(len(points)):
                j = (i + 1) % len(points)
                bottom_i, top_i = base + 2 * i, base + 2 * i + 1
                bottom_j, top_j = base + 2 * j, base + 2 * j + 1
                face_lines.append(f"f {bottom_i} {bottom_j} {top_j} {top_i}")
                walls += 1
                wall_sizes.add(_wall_size(points[i], points[j], height))

            if is_exterior:
                roof = " ".join(str(base + 2 * i + 1) for i in range(len(points)))
                face_lines.append(f"f {roof}")
                roofs += 1
                if oriented.interiors:
                    holes += 1

    if not walls:
        return [], 0, 0, 0, 0

    return [f"o {name}", *vertex_lines, *face_lines], used_vertices, walls, roofs, holes


def _ring_points(ring: Any, min_segment_length_m: float) -> list[tuple[float, float]]:
    """Drop the closing point and segments shorter than the minimum length."""
    coords = list(ring.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]

    kept: list[tuple[float, float]] = []
    for point in coords:
        if not kept or _distance(kept[-1], point) >= min_segment_length_m:
            kept.append((float(point[0]), float(point[1])))

    while len(kept) >= 3 and _distance(kept[-1], kept[0]) < min_segment_length_m:
        kept.pop()

    return kept


def _wall_size(
    start: tuple[float, float],
    end: tuple[float, float],
    height: float,
) -> tuple[float, float]:
    width = _distance(start, end)
    return (
        round(width / _WALL_SIZE_ROUNDING_M) * _WALL_SIZE_ROUNDING_M,
        round(height / _WALL_SIZE_ROUNDING_M) * _WALL_SIZE_ROUNDING_M,
    )


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _vertex(x: float, y: float, z: float) -> str:
    return f"v {x:.4f} {y:.4f} {z:.4f}"


__all__ = [
    "DEFAULT_FLOOR_HEIGHT_M",
    "LocalFrame",
    "MassModelParams",
    "MassModelStats",
    "build_local_frame",
    "buildings_to_obj",
    "group_features_by_zone",
]
