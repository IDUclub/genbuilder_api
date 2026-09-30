"""Rotatable preview boxes built from library sections."""

from __future__ import annotations

import hashlib

import numpy as np
import trimesh

from app.logic.facade_library.assembly import (
    SceneNode,
    export_glb,
    roof_mesh,
    wall_transform,
)

PREVIEW_WIDTH_M = 12.0
_ETAG_LENGTH = 16


def box_faces(
    *, width_m: float, height_m: float
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return Y-up outward CCW wall quads and the roof polygon of a square box."""
    half = width_m / 2
    corners = [(-half, -half), (-half, half), (half, half), (half, -half)]
    bottom = [np.array([x, 0.0, z]) for x, z in corners]
    top = [np.array([x, height_m, z]) for x, z in corners]
    walls = [
        np.asarray([bottom[index], bottom[following], top[following], top[index]])
        for index, following in ((index, (index + 1) % 4) for index in range(4))
    ]
    return walls, np.asarray(top)


def build_preview_glb(
    template: trimesh.Trimesh,
    *,
    width_m: float,
    height_m: float,
    name: str,
) -> bytes:
    walls, roof = box_faces(width_m=width_m, height_m=height_m)
    section = f"{name}__section"
    roof_name = f"{name}__roof"
    nodes = [
        SceneNode(name),
        *(
            SceneNode(
                f"{name}__wall_{index}",
                geometry=section,
                matrix=wall_transform(template, wall),
                parent=name,
            )
            for index, wall in enumerate(walls)
        ),
        SceneNode(roof_name, geometry=roof_name, parent=name),
    ]
    return export_glb({section: template, roof_name: roof_mesh(roof)}, nodes)


def preview_etag(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:_ETAG_LENGTH]


__all__ = ["PREVIEW_WIDTH_M", "box_faces", "build_preview_glb", "preview_etag"]
