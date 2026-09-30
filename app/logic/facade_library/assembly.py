"""Place ready wall templates onto a mass model without GPU inference.

Ported from ``facade-jobs`` (``app/logic/cached_facades.py``); the geometry
must stay identical so a library scene matches a scene assembled there.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import trimesh

DEFAULT_MAX_WALL_ASPECT_RATIO = 1.5
_DEPTH_SCALE_REFERENCE = 32.0
_HORIZONTAL_NORMAL_Y = 0.01
_ROOF_SINK_M = 0.2


class FacadeAssemblyError(RuntimeError):
    """A template or mass model cannot be turned into a scene."""


@dataclass
class BuildingFaces:
    name: str
    walls: list[np.ndarray] = field(default_factory=list)
    roofs: list[np.ndarray] = field(default_factory=list)


def face_normal(points: np.ndarray) -> np.ndarray:
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    length = np.linalg.norm(normal)
    return normal / length if length else normal


def wall_size(points: np.ndarray) -> tuple[float, float]:
    width = np.linalg.norm(points[1] - points[0])
    height = np.linalg.norm(points[2] - points[1])
    if abs(points[1][1] - points[0][1]) >= 0.01:
        width, height = height, width
    return float(width), float(height)


def _parse_obj(
    text: str,
) -> tuple[list[tuple[float, float, float]], dict[str, list[list[int]]]]:
    vertices: list[tuple[float, float, float]] = []
    groups: dict[str, list[list[int]]] = {}
    current_group = "building"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            _, x, y, z = line.split()
            vertices.append((float(x), float(y), float(z)))
        elif line.startswith("o "):
            current_group = line.split(maxsplit=1)[1]
            groups.setdefault(current_group, [])
        elif line.startswith("f "):
            groups.setdefault(current_group, []).append(
                [int(item.split("/")[0]) - 1 for item in line.split()[1:]]
            )
    return vertices, groups


def _vertical_fraction(first: np.ndarray, second: np.ndarray) -> float:
    edge = second - first
    length = float(np.linalg.norm(edge))
    return math.inf if length == 0 else abs(float(edge[1])) / length


def split_wall(points: np.ndarray, max_aspect_ratio: float) -> list[np.ndarray]:
    """Cut a wide wall quad into segments no wider than ``max_aspect_ratio``."""
    if len(points) != 4:
        return [points]
    pair_0 = (
        _vertical_fraction(points[0], points[1])
        + _vertical_fraction(points[2], points[3])
    ) / 2
    pair_1 = (
        _vertical_fraction(points[1], points[2])
        + _vertical_fraction(points[3], points[0])
    ) / 2
    start = 0 if pair_0 <= pair_1 else 1
    a, b, c, d = start, (start + 1) % 4, (start + 2) % 4, (start + 3) % 4
    width_vector = points[b] - points[a]
    opposite_width_vector = points[c] - points[d]
    first_height_vector = points[c] - points[b]
    opposite_height_vector = points[d] - points[a]
    if not np.allclose(width_vector, opposite_width_vector, rtol=1e-3, atol=1e-5):
        return [points]
    if not np.allclose(
        first_height_vector, opposite_height_vector, rtol=1e-3, atol=1e-5
    ):
        return [points]

    width = float(np.linalg.norm(width_vector))
    height = float(np.linalg.norm(first_height_vector))
    if width == 0 or height == 0 or width / height + 1e-9 < max_aspect_ratio:
        return [points]

    count = math.floor(width / height / max_aspect_ratio + 1e-9) + 1
    first = [points[a] + width_vector * (index / count) for index in range(count + 1)]
    opposite = [
        points[d] + opposite_width_vector * (index / count)
        for index in range(count + 1)
    ]
    return [
        np.asarray(
            [first[index], first[index + 1], opposite[index + 1], opposite[index]]
        )
        for index in range(count)
    ]


def building_faces(
    obj_text: str,
    *,
    max_aspect_ratio: float = DEFAULT_MAX_WALL_ASPECT_RATIO,
) -> list[BuildingFaces]:
    """Split an OBJ mass model into per-building wall segments and roofs.

    A ``<name>__roof`` object is folded into building ``<name>``.
    """
    vertices, groups = _parse_obj(obj_text)
    buildings: dict[str, BuildingFaces] = {}
    for group_name, faces in groups.items():
        name = group_name.removesuffix("__roof")
        building = buildings.setdefault(name, BuildingFaces(name=name))
        for face in faces:
            points = np.asarray([vertices[index] for index in face], dtype=float)
            if abs(float(face_normal(points)[1])) >= _HORIZONTAL_NORMAL_Y:
                building.roofs.append(points)
            else:
                building.walls.extend(split_wall(points, max_aspect_ratio))
    return [
        building for building in buildings.values() if building.walls or building.roofs
    ]


def _cross_2d(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    return float(
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _point_in_triangle(
    point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> bool:
    crosses = (_cross_2d(a, b, point), _cross_2d(b, c, point), _cross_2d(c, a, point))
    return (min(crosses) >= -1e-9) or (max(crosses) <= 1e-9)


def _triangulate(points: np.ndarray) -> np.ndarray:
    projected = points[:, [0, 2]]
    area = sum(
        projected[index, 0] * projected[(index + 1) % len(projected), 1]
        - projected[(index + 1) % len(projected), 0] * projected[index, 1]
        for index in range(len(projected))
    )
    orientation = 1.0 if area > 0 else -1.0
    remaining = list(range(len(projected)))
    triangles: list[list[int]] = []
    while len(remaining) > 3:
        for cursor, current in enumerate(remaining):
            previous = remaining[cursor - 1]
            following = remaining[(cursor + 1) % len(remaining)]
            if (
                orientation
                * _cross_2d(
                    projected[previous], projected[current], projected[following]
                )
                <= 1e-9
            ):
                continue
            if any(
                _point_in_triangle(
                    projected[candidate],
                    projected[previous],
                    projected[current],
                    projected[following],
                )
                for candidate in remaining
                if candidate not in {previous, current, following}
            ):
                continue
            triangles.append([previous, current, following])
            remaining.pop(cursor)
            break
        else:
            raise FacadeAssemblyError("could not triangulate roof polygon")
    triangles.append(remaining)
    return np.asarray(triangles, dtype=int)


def roof_mesh(points: np.ndarray) -> trimesh.Trimesh:
    roof = trimesh.Trimesh(vertices=points, faces=_triangulate(points), process=False)
    roof.apply_translation(-face_normal(points) * _ROOF_SINK_M)
    return roof


def load_template_mesh(payload: bytes, label: str) -> trimesh.Trimesh:
    """Load a wall template GLB and centre it on the origin."""
    loaded = trimesh.load(io.BytesIO(payload), file_type="glb", force="scene")
    if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
        raise FacadeAssemblyError(f"template is not a non-empty GLB scene: {label}")
    mesh = loaded.to_mesh()
    if mesh.is_empty:
        raise FacadeAssemblyError(f"template mesh is empty: {label}")
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-center)
    return mesh


def wall_transform(template: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Matrix that stretches a centred template onto one wall quad."""
    width, height = wall_size(points)
    normal = face_normal(points)
    angle = np.arctan2(-normal[0], -normal[2])
    size_x, size_y, _ = template.extents
    if size_x <= 0 or size_y <= 0:
        raise FacadeAssemblyError("template has a zero-size X or Y extent")
    return trimesh.transformations.compose_matrix(
        angles=[0, angle, 0],
        translate=points.mean(axis=0),
        scale=[
            width / size_x,
            height / size_y,
            _DEPTH_SCALE_REFERENCE / max(size_x, size_y),
        ],
    )


@dataclass(frozen=True)
class SceneNode:
    """A glTF node; nodes naming the same ``geometry`` share one stored mesh."""

    name: str
    geometry: str | None = None
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    parent: str | None = None


def export_glb(
    geometries: Mapping[str, trimesh.Trimesh], nodes: Sequence[SceneNode]
) -> bytes:
    """Export a binary glTF scene storing each geometry once.

    Parents must precede their children in ``nodes``.
    """
    scene = trimesh.Scene()
    for name, mesh in geometries.items():
        scene.geometry[name] = mesh
    known_nodes = {scene.graph.base_frame}
    for node in nodes:
        parent = node.parent or scene.graph.base_frame
        if parent not in known_nodes:
            raise FacadeAssemblyError(f"node {node.name} precedes its parent {parent}")
        if node.geometry is not None and node.geometry not in geometries:
            raise FacadeAssemblyError(f"node {node.name} has unknown geometry")
        # trimesh expects the key to be absent, not None, on grouping nodes.
        geometry = {} if node.geometry is None else {"geometry": node.geometry}
        scene.graph.update(
            frame_from=parent, frame_to=node.name, matrix=node.matrix, **geometry
        )
        known_nodes.add(node.name)
    if not scene.graph.nodes_geometry:
        raise FacadeAssemblyError("facade scene has no geometry")
    payload = scene.export(file_type="glb")
    if not isinstance(payload, bytes) or not payload.startswith(b"glTF"):
        raise FacadeAssemblyError("trimesh did not produce a valid binary glTF")
    return payload


__all__ = [
    "DEFAULT_MAX_WALL_ASPECT_RATIO",
    "BuildingFaces",
    "FacadeAssemblyError",
    "SceneNode",
    "building_faces",
    "export_glb",
    "face_normal",
    "load_template_mesh",
    "roof_mesh",
    "split_wall",
    "wall_size",
    "wall_transform",
]
