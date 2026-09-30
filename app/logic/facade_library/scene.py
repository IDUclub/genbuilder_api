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
    building_faces,
    export_glb,
    place_wall,
    roof_mesh,
    wall_size,
)
from app.logic.facade_library.catalog import (
    FacadeTemplateLibrary,
    FacadeTemplateMiss,
    resolve_template,
)
from app.logic.facade_library.models import FacadeTemplate
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
    template_count: int
    nearest_substitutions: int


@dataclass(frozen=True)
class _PlacedWall:
    building: str
    points: np.ndarray
    template: FacadeTemplate


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


def _plan(
    faces_by_zone: Mapping[str, list[BuildingFaces]],
    templates: list[FacadeTemplate],
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
                floors = max(1, round(height / floor_height_m))
                try:
                    template = resolve_template(
                        templates,
                        style_id=style_id,
                        floors=floors,
                        width_m=width,
                        height_m=height,
                        pixels_per_meter=pixels_per_meter,
                        max_width_scale=max_width_scale,
                        allow_nearest=False,
                    )
                except FacadeTemplateMiss:
                    if not allow_nearest:
                        raise
                    template = resolve_template(
                        templates,
                        style_id=style_id,
                        floors=floors,
                        width_m=width,
                        height_m=height,
                        pixels_per_meter=pixels_per_meter,
                        max_width_scale=max_width_scale,
                        allow_nearest=True,
                    )
                    substitutions += 1
                walls.append(_PlacedWall(building.name, points, template))
    return _ScenePlan(all_buildings, walls, substitutions)


def _render(plan: _ScenePlan, meshes: Mapping[str, trimesh.Trimesh]) -> bytes:
    walls_by_building: dict[str, list[trimesh.Trimesh]] = {}
    for wall in plan.walls:
        walls_by_building.setdefault(wall.building, []).append(
            place_wall(meshes[wall.template.cache_key], wall.points)
        )
    nodes: list[tuple[str, trimesh.Trimesh]] = []
    for building in plan.buildings:
        parts = walls_by_building.get(building.name)
        if parts:
            nodes.append((building.name, trimesh.util.concatenate(parts)))
        if building.roofs:
            roofs = [roof_mesh(points) for points in building.roofs]
            nodes.append((f"{building.name}__roof", trimesh.util.concatenate(roofs)))
    return export_glb(nodes)


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

    Raises :class:`FacadeTemplateMiss` when a wall has no template and
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
        manifest.templates,
        style_by_zone=style_by_zone,
        floor_height_m=floor_height_m,
        pixels_per_meter=pixels_per_meter,
        max_width_scale=library.max_width_scale,
        allow_nearest=allow_nearest,
    )
    unique_templates = list(
        {wall.template.cache_key: wall.template for wall in plan.walls}.values()
    )
    meshes = await library.load_meshes(unique_templates)
    glb = await asyncio.to_thread(_render, plan, meshes)
    origin_lon, origin_lat = local_frame_origin_wgs84(frame)
    return LibraryScene(
        glb=glb,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        buildings=len(plan.buildings),
        wall_instances=len(plan.walls),
        template_count=len(unique_templates),
        nearest_substitutions=plan.nearest_substitutions,
    )


__all__ = ["EmptyScene", "LibraryScene", "SceneTooLarge", "build_library_scene"]
