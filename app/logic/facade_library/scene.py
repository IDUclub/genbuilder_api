"""Build a textured quarter scene from facade library sections."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from app.logic.facade_library.assembly import (
    BuildingFaces,
    SceneNode,
    building_faces,
    export_glb,
    roof_mesh,
    wall_size,
)
from app.logic.facade_library.catalog import (
    FacadeTemplateLibrary,
    FacadeTemplateMiss,
    resolve_section,
)
from app.logic.facade_library.floors import FloorPieces, stack_transforms
from app.logic.facade_library.models import FacadeSection
from app.logic.mass_model import (
    DEFAULT_FLOOR_HEIGHT_M,
    LocalFrame,
    MassModelParams,
    build_local_frame,
    buildings_to_obj,
    group_features_by_zone,
    local_frame_origin_wgs84,
)


class SceneTooLarge(ValueError):
    """The scene has more walls than the library path may assemble in-process."""


class EmptyScene(ValueError):
    """No building produced a wall."""


@dataclass(frozen=True)
class LibraryScene:
    glb: bytes
    origin_lon: float
    origin_lat: float
    buildings: int
    wall_instances: int
    floor_instances: int
    template_count: int
    nearest_substitutions: int


@dataclass(frozen=True)
class _PlacedWall:
    building: str
    points: np.ndarray
    floors: int
    section: FacadeSection


@dataclass(frozen=True)
class _ScenePlan:
    buildings: list[BuildingFaces]
    walls: list[_PlacedWall]
    nearest_substitutions: int


def _zone_faces(
    buildings: Mapping[str, Any],
    frame: LocalFrame,
    floor_height_m: float,
) -> dict[str, list[BuildingFaces]]:
    params = MassModelParams(floor_height_m=floor_height_m)
    faces_by_zone: dict[str, list[BuildingFaces]] = {}
    seen: set[str] = set()
    for zone, collection in group_features_by_zone(buildings).items():
        try:
            obj_text, _ = buildings_to_obj(collection, frame, params)
        except ValueError:
            continue
        faces = building_faces(obj_text)
        for building in faces:
            if building.name in seen:
                building.name = f"{building.name}__{zone}"
            seen.add(building.name)
        faces_by_zone[zone] = faces
    return faces_by_zone


def _pick_section(
    sections: list[FacadeSection],
    *,
    style_id: str,
    width_m: float,
    pixels_per_meter: int,
    max_width_scale: float,
    allow_nearest: bool,
    variant_key: str,
) -> tuple[FacadeSection, bool]:
    try:
        section = resolve_section(
            sections,
            style_id=style_id,
            width_m=width_m,
            pixels_per_meter=pixels_per_meter,
            max_width_scale=max_width_scale,
            allow_nearest=False,
            variant_key=variant_key,
        )
        return section, False
    except FacadeTemplateMiss:
        if not allow_nearest:
            raise
    section = resolve_section(
        sections,
        style_id=style_id,
        width_m=width_m,
        pixels_per_meter=pixels_per_meter,
        max_width_scale=max_width_scale,
        allow_nearest=True,
        variant_key=variant_key,
    )
    return section, True


def _plan(
    faces_by_zone: Mapping[str, list[BuildingFaces]],
    sections: list[FacadeSection],
    *,
    style_by_zone: Mapping[str, str],
    floor_height_m: float,
    pixels_per_meter: int,
    max_width_scale: float,
    allow_nearest: bool,
) -> _ScenePlan:
    walls: list[_PlacedWall] = []
    substitutions = 0
    all_buildings: list[BuildingFaces] = []
    for zone, faces in faces_by_zone.items():
        style_id = style_by_zone[zone]
        all_buildings.extend(faces)
        for building in faces:
            for points in building.walls:
                width, height = wall_size(points)
                section, substituted = _pick_section(
                    sections,
                    style_id=style_id,
                    width_m=width,
                    pixels_per_meter=pixels_per_meter,
                    max_width_scale=max_width_scale,
                    allow_nearest=allow_nearest,
                    variant_key=building.name,
                )
                substitutions += substituted
                floors = max(1, round(height / floor_height_m))
                walls.append(_PlacedWall(building.name, points, floors, section))
    return _ScenePlan(all_buildings, walls, substitutions)


def _wall_nodes(
    name: str, parent: str, wall: _PlacedWall, pieces: FloorPieces, section_name: str
) -> tuple[list[SceneNode], dict[str, trimesh.Trimesh]]:
    """A wall group node with one child per floor, plus the pieces it uses."""
    nodes = [SceneNode(name, parent=parent)]
    used: dict[str, trimesh.Trimesh] = {}
    placed = stack_transforms(pieces, wall.points, wall.floors)
    for index, (kind, matrix) in enumerate(placed):
        geometry = f"{section_name}__{kind}"
        used[geometry] = pieces.meshes[kind]
        nodes.append(
            SceneNode(
                f"{name}__floor_{index}",
                geometry=geometry,
                matrix=matrix,
                parent=name,
            )
        )
    return nodes, used


def _render(plan: _ScenePlan, pieces: Mapping[str, FloorPieces]) -> bytes:
    """Each building holds wall groups whose floor nodes instance shared pieces."""
    section_names = {key: f"section_{index}" for index, key in enumerate(pieces)}
    walls_by_building: dict[str, list[_PlacedWall]] = {}
    for wall in plan.walls:
        walls_by_building.setdefault(wall.building, []).append(wall)

    geometries: dict[str, trimesh.Trimesh] = {}
    nodes: list[SceneNode] = []
    for building in plan.buildings:
        walls = walls_by_building.get(building.name, [])
        if not walls and not building.roofs:
            continue
        nodes.append(SceneNode(building.name))
        for index, wall in enumerate(walls):
            key = wall.section.cache_key
            wall_nodes, used = _wall_nodes(
                f"{building.name}__wall_{index}",
                building.name,
                wall,
                pieces[key],
                section_names[key],
            )
            geometries.update(used)
            nodes.extend(wall_nodes)
        if building.roofs:
            roof_name = f"{building.name}__roof"
            geometries[roof_name] = trimesh.util.concatenate(
                [roof_mesh(points) for points in building.roofs]
            )
            nodes.append(SceneNode(roof_name, geometry=roof_name, parent=building.name))
    return export_glb(geometries, nodes)


async def build_library_scene(
    buildings: Mapping[str, Any],
    *,
    style_by_zone: Mapping[str, str],
    library: FacadeTemplateLibrary,
    pixels_per_meter: int,
    allow_nearest: bool,
    max_walls: int,
    floor_height_m: float = DEFAULT_FLOOR_HEIGHT_M,
) -> LibraryScene:
    """Assemble one GLB for all zones, each textured with its library style.

    Raises :class:`FacadeTemplateMiss` when a wall has no section and
    ``allow_nearest`` is off, :class:`SceneTooLarge` above ``max_walls`` and
    :class:`EmptyScene` when nothing has walls.
    """
    frame = await asyncio.to_thread(build_local_frame, buildings)
    faces_by_zone = await asyncio.to_thread(
        _zone_faces, buildings, frame, floor_height_m
    )
    wall_count = sum(
        len(building.walls) for faces in faces_by_zone.values() for building in faces
    )
    if wall_count == 0:
        raise EmptyScene("generated buildings have no walls")
    if wall_count > max_walls:
        raise SceneTooLarge(f"scene has {wall_count} walls, limit is {max_walls}")

    manifest = await library.manifest()
    plan = _plan(
        faces_by_zone,
        manifest.sections,
        style_by_zone=style_by_zone,
        floor_height_m=floor_height_m,
        pixels_per_meter=pixels_per_meter,
        max_width_scale=library.max_width_scale,
        allow_nearest=allow_nearest,
    )
    unique_sections = list(
        {wall.section.cache_key: wall.section for wall in plan.walls}.values()
    )
    pieces = await library.load_pieces(unique_sections)
    glb = await asyncio.to_thread(_render, plan, pieces)
    origin_lon, origin_lat = local_frame_origin_wgs84(frame)
    return LibraryScene(
        glb=glb,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        buildings=len(plan.buildings),
        wall_instances=len(plan.walls),
        floor_instances=sum(wall.floors for wall in plan.walls),
        template_count=len(unique_sections),
        nearest_substitutions=plan.nearest_substitutions,
    )


__all__ = ["EmptyScene", "LibraryScene", "SceneTooLarge", "build_library_scene"]
