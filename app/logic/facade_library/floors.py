"""Cut a library section into floors and stack them onto walls of any height."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import trimesh

from app.logic.facade_library.assembly import (
    DEPTH_SCALE_REFERENCE,
    FacadeAssemblyError,
    face_normal,
    wall_size,
)

FloorKind = Literal["ground", "typical", "top"]
FLOOR_KINDS: tuple[FloorKind, ...] = ("ground", "typical", "top")
SECTION_HEIGHT_TOLERANCE = 0.01


@dataclass(frozen=True)
class FloorPieces:
    """Ground, typical and top floor meshes cut from one centred section.

    Pieces keep the section's coordinates; ``bottoms`` holds the Y of each cut.
    """

    meshes: dict[FloorKind, trimesh.Trimesh]
    bottoms: dict[FloorKind, float]
    floor_height: float
    section_width: float
    depth_scale: float


def _band(
    mesh: trimesh.Trimesh, low: float | None, high: float | None
) -> trimesh.Trimesh:
    part = mesh
    if low is not None:
        part = part.slice_plane([0.0, low, 0.0], [0.0, 1.0, 0.0])
    if high is not None:
        part = part.slice_plane([0.0, high, 0.0], [0.0, -1.0, 0.0])
    return part


def slice_section(
    mesh: trimesh.Trimesh, section_floors: int, expected_height_m: float
) -> FloorPieces:
    """Cut the bottom, a middle and the top floor on the section's even floor grid.

    The grid is derived from the mesh height, so a mesh taller or shorter than
    ``expected_height_m`` (for example with a parapet) is refused.
    """
    if section_floors < 3:
        raise FacadeAssemblyError("a section needs at least three floors")
    size_x, size_y, _ = mesh.extents
    if size_x <= 0 or size_y <= 0:
        raise FacadeAssemblyError("section has a zero-size X or Y extent")
    if abs(size_y - expected_height_m) > expected_height_m * SECTION_HEIGHT_TOLERANCE:
        raise FacadeAssemblyError(
            f"section is {size_y:.2f} m tall, expected {expected_height_m:.2f} m"
        )
    bottom = float(mesh.bounds[0][1])
    step = float(size_y) / section_floors
    typical_index = (section_floors - 1) // 2
    bottoms: dict[FloorKind, float] = {
        "ground": bottom,
        "typical": bottom + typical_index * step,
        "top": bottom + (section_floors - 1) * step,
    }
    meshes: dict[FloorKind, trimesh.Trimesh] = {
        "ground": _band(mesh, None, bottom + step),
        "typical": _band(mesh, bottoms["typical"], bottoms["typical"] + step),
        "top": _band(mesh, bottoms["top"], None),
    }
    empty = [kind for kind, piece in meshes.items() if len(piece.faces) == 0]
    if empty:
        raise FacadeAssemblyError(f"section has empty floor pieces: {empty}")
    return FloorPieces(
        meshes=meshes,
        bottoms=bottoms,
        floor_height=step,
        section_width=float(size_x),
        depth_scale=DEPTH_SCALE_REFERENCE / max(float(size_x), float(size_y)),
    )


def floor_sequence(floors: int) -> list[FloorKind]:
    if floors < 1:
        raise FacadeAssemblyError("a wall needs at least one floor")
    if floors == 1:
        return ["ground"]
    typical: list[FloorKind] = ["typical"] * (floors - 2)
    return ["ground", *typical, "top"]


def stack_transforms(
    pieces: FloorPieces, points: np.ndarray, floors: int
) -> list[tuple[FloorKind, np.ndarray]]:
    """Matrices placing one floor piece per storey so the stack fills the wall quad."""
    width, height = wall_size(points)
    normal = face_normal(points)
    angle = float(np.arctan2(-normal[0], -normal[2]))
    sequence = floor_sequence(floors)
    storey = height / len(sequence)
    base = points.mean(axis=0)
    base[1] = points[:, 1].min()
    scale = [
        width / pieces.section_width,
        storey / pieces.floor_height,
        pieces.depth_scale,
    ]
    placed: list[tuple[FloorKind, np.ndarray]] = []
    for index, kind in enumerate(sequence):
        place = trimesh.transformations.compose_matrix(
            angles=[0, angle, 0],
            translate=base + np.array([0.0, index * storey, 0.0]),
            scale=scale,
        )
        to_origin = trimesh.transformations.translation_matrix(
            [0.0, -pieces.bottoms[kind], 0.0]
        )
        placed.append((kind, place @ to_origin))
    return placed


__all__ = [
    "FLOOR_KINDS",
    "FloorKind",
    "FloorPieces",
    "floor_sequence",
    "slice_section",
    "stack_transforms",
]
