"""Rotatable preview boxes built from library sections."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence

import numpy as np
import trimesh

from app.logic.facade_library.assembly import (
    FacadeAssemblyError,
    SceneNode,
    export_glb,
    roof_mesh,
    wall_transform,
)

PREVIEW_WIDTH_M = 12.0
# Bumped whenever the gallery layout changes, so clients refetch it.
GALLERY_FORMAT_VERSION = 2
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


def gallery_node_name(style_id: str) -> str:
    return f"style_{style_id}"


def build_gallery_glb(previews: Sequence[tuple[str, bytes]]) -> bytes:
    """Stack stored preview GLBs at the origin of one scene.

    Each preview hangs under its own ``style_<id>`` node, all in the same spot,
    so the client shows one style at a time by hiding the other nodes and
    switches styles without another download. Textures and the shared wall
    geometry of every preview are kept as they are.
    """
    if not previews:
        raise FacadeAssemblyError("gallery has no previews")
    geometries: dict[str, trimesh.Trimesh] = {}
    nodes: list[SceneNode] = []
    for style_id, payload in previews:
        root = gallery_node_name(style_id)
        nodes.append(SceneNode(root))
        _append_preview(
            payload, prefix=f"{root}/", root=root, geometries=geometries, nodes=nodes
        )
    return export_glb(geometries, nodes)


def _append_preview(
    payload: bytes,
    *,
    prefix: str,
    root: str,
    geometries: dict[str, trimesh.Trimesh],
    nodes: list[SceneNode],
) -> None:
    scene = trimesh.load(io.BytesIO(payload), file_type="glb", force="scene")
    for name, mesh in scene.geometry.items():
        geometries[prefix + name] = mesh
    graph = scene.graph
    pending = [graph.base_frame]
    while pending:
        parent = pending.pop()
        for child in graph.transforms.children.get(parent, []):
            data = graph.transforms.edge_data[(parent, child)]
            geometry = data.get("geometry")
            nodes.append(
                SceneNode(
                    prefix + child,
                    geometry=None if geometry is None else prefix + geometry,
                    matrix=np.asarray(data.get("matrix", np.eye(4)), dtype=float),
                    parent=root if parent == graph.base_frame else prefix + parent,
                )
            )
            pending.append(child)


def preview_etag(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:_ETAG_LENGTH]


__all__ = [
    "GALLERY_FORMAT_VERSION",
    "PREVIEW_WIDTH_M",
    "box_faces",
    "build_gallery_glb",
    "build_preview_glb",
    "gallery_node_name",
    "preview_etag",
]
