import math

import numpy as np
import pytest

from app.logic.mass_model import (
    MassModelParams,
    build_local_frame,
    buildings_to_obj,
    group_features_by_zone,
)

LON, LAT = 31.0, 59.92
METERS_PER_DEGREE_LAT = 111_320.0
METERS_PER_DEGREE_LON = METERS_PER_DEGREE_LAT * math.cos(math.radians(LAT))


def _rect(width_m: float, height_m: float, *, lon_offset_m: float = 0.0) -> dict:
    lon0 = LON + lon_offset_m / METERS_PER_DEGREE_LON
    lon1 = lon0 + width_m / METERS_PER_DEGREE_LON
    lat1 = LAT + height_m / METERS_PER_DEGREE_LAT
    return {
        "type": "Polygon",
        "coordinates": [[[lon0, LAT], [lon1, LAT], [lon1, lat1], [lon0, lat1], [lon0, LAT]]],
    }


def _collection(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


def _feature(geometry: dict | None, *, feature_id="1", zone="residential", floors=5) -> dict:
    properties = {"zone": zone}
    if floors is not None:
        properties["floors_count"] = floors
    return {"type": "Feature", "id": feature_id, "properties": properties, "geometry": geometry}


def _parse_obj(text: str) -> tuple[np.ndarray, dict[str, list[list[int]]]]:
    """Parse OBJ the way Facades-3D does: `v` lines and `o`-grouped `f` lines."""
    vertices: list[tuple[float, float, float]] = []
    groups: dict[str, list[list[int]]] = {}
    current = "building"

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("v "):
            _, x, y, z = line.split()
            vertices.append((float(x), float(y), float(z)))
        elif line.startswith("o "):
            current = line.split(maxsplit=1)[1]
            groups.setdefault(current, [])
        elif line.startswith("f "):
            groups.setdefault(current, []).append(
                [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
            )

    return np.array(vertices), groups


def _normal(points: np.ndarray) -> np.ndarray:
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    norm = np.linalg.norm(normal)
    return normal / norm if norm else normal


def _export(collection: dict, **kwargs):
    frame = build_local_frame(collection)
    return buildings_to_obj(collection, frame, MassModelParams(**kwargs))


def test_rectangle_exports_four_wall_quads_and_one_roof():
    obj, stats = _export(_collection(_feature(_rect(20.0, 30.0))))
    vertices, groups = _parse_obj(obj)

    assert list(groups) == ["gb_1"]
    faces = groups["gb_1"]
    walls = [face for face in faces if abs(_normal(vertices[face])[1]) < 0.01]
    roofs = [face for face in faces if abs(_normal(vertices[face])[1]) >= 0.01]

    assert len(walls) == 4
    assert all(len(face) == 4 for face in walls), "Facades-3D only splits 4-vertex wall faces"
    assert len(roofs) == 1
    assert (stats.buildings, stats.walls, stats.roofs) == (1, 4, 1)


def test_wall_normals_point_away_from_the_building():
    obj, _ = _export(_collection(_feature(_rect(20.0, 30.0))))
    vertices, groups = _parse_obj(obj)
    center = vertices[:, [0, 2]].mean(axis=0)

    for face in groups["gb_1"]:
        points = vertices[face]
        normal = _normal(points)
        if abs(normal[1]) >= 0.01:
            assert normal[1] > 0, "roof normal must point up"
            continue
        outward = points[:, [0, 2]].mean(axis=0) - center
        assert np.dot(normal[[0, 2]], outward) > 0


def test_wall_sizes_match_footprint_and_floor_height():
    obj, _ = _export(_collection(_feature(_rect(20.0, 30.0), floors=5)), floor_height_m=3.0)
    vertices, groups = _parse_obj(obj)

    widths, heights = [], []
    for face in groups["gb_1"]:
        points = vertices[face]
        if abs(_normal(points)[1]) >= 0.01:
            continue
        widths.append(np.linalg.norm(points[1] - points[0]))
        heights.append(np.linalg.norm(points[2] - points[1]))

    assert all(height == pytest.approx(15.0) for height in heights)
    assert sorted(widths)[0] == pytest.approx(20.0, rel=0.02)
    assert sorted(widths)[-1] == pytest.approx(30.0, rel=0.02)


def test_multipolygon_stays_one_building_group():
    geometry = {
        "type": "MultiPolygon",
        "coordinates": [
            _rect(20.0, 30.0)["coordinates"],
            _rect(20.0, 30.0, lon_offset_m=100.0)["coordinates"],
        ],
    }
    obj, stats = _export(_collection(_feature(geometry)))
    _, groups = _parse_obj(obj)

    assert list(groups) == ["gb_1"]
    assert stats.buildings == 1
    assert stats.walls == 8


def test_missing_floors_count_falls_back_to_one_floor():
    obj, stats = _export(_collection(_feature(_rect(20.0, 30.0), floors=None)), floor_height_m=3.0)
    vertices, _ = _parse_obj(obj)

    assert stats.buildings == 1
    assert stats.fallback_height_features == 1
    assert vertices[:, 1].max() == pytest.approx(3.0)


def test_features_without_geometry_are_skipped():
    collection = _collection(
        _feature(_rect(20.0, 30.0), feature_id="1"),
        _feature(None, feature_id="2"),
    )
    obj, stats = _export(collection)
    _, groups = _parse_obj(obj)

    assert list(groups) == ["gb_1"]
    assert stats.skipped_features == 1


def test_existing_physical_objects_keep_their_id_in_the_group_name():
    feature = _feature(_rect(20.0, 30.0))
    feature["properties"]["physical_object_id"] = 2058130
    obj, _ = _export(_collection(feature))
    _, groups = _parse_obj(obj)

    assert list(groups) == ["po_2058130"]


def test_per_zone_exports_share_one_local_frame():
    collection = _collection(
        _feature(_rect(20.0, 30.0), feature_id="1", zone="residential"),
        _feature(_rect(20.0, 30.0, lon_offset_m=200.0), feature_id="2", zone="business"),
    )
    frame = build_local_frame(collection)
    by_zone = group_features_by_zone(collection)

    assert set(by_zone) == {"residential", "business"}

    together, _ = buildings_to_obj(collection, frame)
    residential, _ = buildings_to_obj(by_zone["residential"], frame)
    business, _ = buildings_to_obj(by_zone["business"], frame)

    all_vertices, _ = _parse_obj(together)
    split_vertices = np.vstack([_parse_obj(residential)[0], _parse_obj(business)[0]])
    assert np.allclose(np.sort(all_vertices, axis=0), np.sort(split_vertices, axis=0))


def test_empty_collection_is_rejected():
    with pytest.raises(ValueError):
        _export(_collection())
