"""Shared builders for the facade library tests."""

from __future__ import annotations

import math

import trimesh

from app.infrastructure.object_storage import LocalStorage
from app.logic.facade_library.catalog import FacadeTemplateLibrary
from app.logic.facade_library.models import (
    FacadeLibraryManifest,
    FacadeTemplate,
    floor_group,
)

PREFIX = "facade-library/v1"
LON, LAT = 30.3, 59.93
_METERS_PER_DEGREE_LAT = 111_320.0
_METERS_PER_DEGREE_LON = _METERS_PER_DEGREE_LAT * math.cos(math.radians(LAT))


def wall_glb(width_m: float = 12.0, height_m: float = 9.0) -> bytes:
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(extents=[width_m, height_m, 0.4]),
        node_name="wall",
        geom_name="wall",
    )
    payload = scene.export(file_type="glb")
    assert isinstance(payload, bytes)
    return payload


def make_template(
    style_id: str,
    floors: int,
    width_m: float,
    *,
    pixels_per_meter: int = 32,
    floor_height_m: float = 3.0,
) -> FacadeTemplate:
    group = floor_group(floors)
    height_m = floors * floor_height_m
    size_key = f"w{round(width_m * 100):04d}-h{round(height_m * 100):04d}"
    return FacadeTemplate(
        object_key=f"{PREFIX}/styles/{style_id}/{group}/{size_key}/wall.glb",
        style_id=style_id,
        style_name_ru=style_id,
        style_key="test",
        prompt="test",
        floor_group=group,
        floors=floors,
        floor_height_m=floor_height_m,
        width_m=width_m,
        height_m=height_m,
        pixels_per_meter=pixels_per_meter,
        glb_size_bytes=1,
    )


def seed_library(storage: LocalStorage, templates: list[FacadeTemplate]) -> None:
    for template in templates:
        storage.put_bytes(
            wall_glb(template.width_m, template.height_m),
            template.object_key,
            "model/gltf-binary",
        )
    manifest = FacadeLibraryManifest(templates=templates)
    storage.put_json(manifest.model_dump(mode="json"), f"{PREFIX}/manifest.json")


def make_library(storage: LocalStorage, **overrides: float) -> FacadeTemplateLibrary:
    options = {"manifest_ttl_seconds": 60.0, "max_width_scale": 2.5, **overrides}
    return FacadeTemplateLibrary(storage, prefix=PREFIX, **options)


def rect_feature(
    feature_id: str,
    *,
    zone: str,
    floors: int,
    width_m: float = 18.0,
    depth_m: float = 12.0,
    offset_m: float = 0.0,
) -> dict:
    lon0 = LON + offset_m / _METERS_PER_DEGREE_LON
    lon1 = lon0 + width_m / _METERS_PER_DEGREE_LON
    lat1 = LAT + depth_m / _METERS_PER_DEGREE_LAT
    return {
        "type": "Feature",
        "id": feature_id,
        "properties": {"zone": zone, "floors_count": floors},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[lon0, LAT], [lon1, LAT], [lon1, lat1], [lon0, lat1], [lon0, LAT]]
            ],
        },
    }


def collection(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}
