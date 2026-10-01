"""Persist library-built scenes so the frontend can fetch them by id."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.infrastructure.object_storage import ObjectStorage
from app.logic.facade_library.scene import LibraryScene

GLB_MIME_TYPE = "model/gltf-binary"
SCENE_URL_PREFIX = "/facade-scenes"
_RESULT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SCENE_KEY_PREFIX = "generated/3d"


def is_scene_id(value: str) -> bool:
    return bool(_RESULT_ID_RE.match(value))


def scene_glb_key(result_id: str) -> str:
    if not is_scene_id(result_id):
        raise ValueError(f"invalid scene id: {result_id!r}")
    return f"{SCENE_KEY_PREFIX}/{result_id}.glb"


def scene_metadata_key(result_id: str) -> str:
    if not is_scene_id(result_id):
        raise ValueError(f"invalid scene id: {result_id!r}")
    return f"{SCENE_KEY_PREFIX}/{result_id}.json"


def scene_url(result_id: str) -> str:
    return f"{SCENE_URL_PREFIX}/{result_id}.glb"


def store_scene(
    storage: ObjectStorage,
    scene: LibraryScene,
    *,
    style_by_zone: Mapping[str, str],
    facade_style: str,
) -> dict[str, Any]:
    """Store the GLB and its metadata; return the public ``ready`` payload."""
    result_id = uuid4().hex
    stats = {
        "buildings": scene.buildings,
        "wall_instances": scene.wall_instances,
        "floor_instances": scene.floor_instances,
        "template_count": scene.template_count,
        "nearest_substitutions": scene.nearest_substitutions,
    }
    origin = {"lon": scene.origin_lon, "lat": scene.origin_lat}
    storage.put_bytes(scene.glb, scene_glb_key(result_id), GLB_MIME_TYPE)
    storage.put_json(
        {
            "result_id": result_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "facade_style": facade_style,
            "style_by_zone": dict(style_by_zone),
            "origin": origin,
            "stats": stats,
            "glb_size_bytes": len(scene.glb),
        },
        scene_metadata_key(result_id),
    )
    return {
        "status": "ready",
        "result_id": result_id,
        "glb_url": scene_url(result_id),
        "origin": origin,
        "facade_style": facade_style,
        "style_by_zone": dict(style_by_zone),
        "source": "library",
        "stats": stats,
    }


__all__ = [
    "GLB_MIME_TYPE",
    "SCENE_KEY_PREFIX",
    "is_scene_id",
    "scene_glb_key",
    "scene_metadata_key",
    "scene_url",
    "store_scene",
]
