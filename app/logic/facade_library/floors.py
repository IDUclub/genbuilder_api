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
CUT_SEARCH_SHARE = 0.25
CUT_SEARCH_STEP_M = 0.05
CUT_OFFSET_PENALTY = 0.1
UP = (0.0, 1.0, 0.0)


@dataclass(frozen=True)
class FloorPieces:
    """Ground, typical and top floor meshes cut from one centred section.

    Pieces keep the section's coordinates; ``bottoms`` holds the Y of each
    piece's lower cut and ``heights`` its own height, which differ slightly
    between pieces because cuts avoid balconies and window openings.
    """

    meshes: dict[FloorKind, trimesh.Trimesh]
    bottoms: dict[FloorKind, float]
    heights: dict[FloorKind, float]
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


def _cross_section_lengths(mesh: trimesh.Trimesh, heights: np.ndarray) -> np.ndarray:
    lines, _, _ = trimesh.intersections.mesh_multiplane(
        mesh, (0.0, 0.0, 0.0), UP, heights
    )
    return np.array(
        [float(np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1).sum()) for seg in lines]
    )


def _clean_cut(mesh: trimesh.Trimesh, line: float, step: float, width: float) -> float:
    """Height near a floor grid line where a horizontal cut crosses the least geometry.

    Facades-3D does not keep floors on an even grid, so a cut exactly on the
    line can split a balcony or open a window reveal of the unclosed shell.
    """
    reach = round(step * CUT_SEARCH_SHARE / CUT_SEARCH_STEP_M)
    offsets = np.arange(-reach, reach + 1) * CUT_SEARCH_STEP_M
    lengths = _cross_section_lengths(mesh, line + offsets)
    cost = lengths / width + CUT_OFFSET_PENALTY * np.abs(offsets) / step
    return float(line + offsets[int(np.argmin(cost))])


def slice_section(
    mesh: trimesh.Trimesh, section_floors: int, expected_height_m: float
) -> FloorPieces:
    """Cut the bottom, a middle and the top floor near the section's even floor grid.

    The grid is derived from the mesh height, so a mesh taller or shorter than
    ``expected_height_m`` (for example with a parapet) is refused. Each cut is
    moved within a quarter floor of its grid line to the plainest stretch of wall.
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
    bottom, top = float(mesh.bounds[0][1]), float(mesh.bounds[1][1])
    step = float(size_y) / section_floors
    typical_index = (section_floors - 1) // 2
    grid = {1, typical_index, typical_index + 1, section_floors - 1}
    cuts = {
        index: _clean_cut(mesh, bottom + index * step, step, float(size_x))
        for index in grid
    }
    bottoms: dict[FloorKind, float] = {
        "ground": bottom,
        "typical": cuts[typical_index],
        "top": cuts[section_floors - 1],
    }
    tops: dict[FloorKind, float] = {
        "ground": cuts[1],
        "typical": cuts[typical_index + 1],
        "top": top,
    }
    meshes: dict[FloorKind, trimesh.Trimesh] = {
        "ground": _band(mesh, None, tops["ground"]),
        "typical": _band(mesh, bottoms["typical"], tops["typical"]),
        "top": _band(mesh, bottoms["top"], None),
    }
    empty = [kind for kind, piece in meshes.items() if len(piece.faces) == 0]
    if empty:
        raise FacadeAssemblyError(f"section has empty floor pieces: {empty}")
    return FloorPieces(
        meshes=meshes,
        bottoms=bottoms,
        heights={kind: tops[kind] - bottoms[kind] for kind in FLOOR_KINDS},
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
    """Matrices placing one floor piece per storey so the stack fills the wall quad.

    Pieces are laid one on another at their own heights and share one vertical
    scale, so their proportions are kept.
    """
    width, height = wall_size(points)
    normal = face_normal(points)
    angle = float(np.arctan2(-normal[0], -normal[2]))
    sequence = floor_sequence(floors)
    heights = np.array([pieces.heights[kind] for kind in sequence])
    scale_y = height / float(heights.sum())
    levels = np.concatenate([[0.0], np.cumsum(heights)[:-1]]) * scale_y
    base = points.mean(axis=0)
    base[1] = points[:, 1].min()
    scale = [width / pieces.section_width, scale_y, pieces.depth_scale]
    placed: list[tuple[FloorKind, np.ndarray]] = []
    for level, kind in zip(levels, sequence):
        place = trimesh.transformations.compose_matrix(
            angles=[0, angle, 0],
            translate=base + np.array([0.0, float(level), 0.0]),
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
