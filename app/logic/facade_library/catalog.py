from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Sequence

import trimesh
from pydantic import ValidationError

from app.infrastructure.object_storage import (
    ObjectNotFoundError,
    ObjectStorage,
    ObjectStorageError,
)
from app.logic.facade_library.assembly import load_template_mesh
from app.logic.facade_library.models import (
    FacadeLibraryManifest,
    FacadeTemplate,
    StylePreviewIndex,
    floor_group,
)

_DEFAULT_MESH_CACHE_SIZE = 64


class FacadeTemplateMiss(LookupError):
    """No library template can serve a wall."""


class FacadeLibraryUnavailable(RuntimeError):
    """The library manifest or a template could not be read."""


def resolve_template(
    templates: Sequence[FacadeTemplate],
    *,
    style_id: str,
    floors: int,
    width_m: float,
    height_m: float,
    pixels_per_meter: int,
    max_width_scale: float,
    allow_nearest: bool,
) -> FacadeTemplate:
    """Pick the template for one wall.

    An exact match shares the style, floor group and resolution and stays
    within ``max_width_scale``. With ``allow_nearest`` a miss falls back to the
    same style's template closest in floor count, then in size.
    """
    same_style = [
        template
        for template in templates
        if template.style_id == style_id
        and template.pixels_per_meter == pixels_per_meter
    ]
    group = floor_group(floors)
    exact = [
        template
        for template in same_style
        if template.floor_group == group
        and template.width_scale_for(width_m) <= max_width_scale
    ]
    if exact:
        return min(exact, key=lambda item: item.score(width_m, height_m))
    if allow_nearest and same_style:
        return min(
            same_style,
            key=lambda item: (abs(item.floors - floors), item.score(width_m, height_m)),
        )
    raise FacadeTemplateMiss(
        f"no template for style={style_id}, floors={group}, "
        f"width={width_m:.2f}m, ppm={pixels_per_meter}"
    )


class FacadeTemplateLibrary:
    """Read access to the template library kept in object storage."""

    def __init__(
        self,
        storage: ObjectStorage,
        *,
        prefix: str,
        manifest_ttl_seconds: float,
        max_width_scale: float,
        mesh_cache_size: int = _DEFAULT_MESH_CACHE_SIZE,
    ) -> None:
        self._storage = storage
        self._prefix = prefix.strip("/")
        self._manifest_ttl_seconds = manifest_ttl_seconds
        self.max_width_scale = max_width_scale
        self._mesh_cache_size = mesh_cache_size
        self._manifest: FacadeLibraryManifest | None = None
        self._manifest_loaded_at = 0.0
        self._previews: StylePreviewIndex | None = None
        self._previews_loaded_at = 0.0
        self._lock = asyncio.Lock()
        self._meshes: OrderedDict[str, trimesh.Trimesh] = OrderedDict()

    @property
    def manifest_key(self) -> str:
        return f"{self._prefix}/manifest.json"

    @property
    def preview_index_key(self) -> str:
        return f"{self._prefix}/previews/index.json"

    def preview_key(self, style_id: str, group: str) -> str:
        return f"{self._prefix}/previews/{style_id}/{group}.glb"

    def _fresh(self, loaded_at: float) -> bool:
        return time.monotonic() - loaded_at < self._manifest_ttl_seconds

    async def _read(self, object_key: str) -> bytes:
        try:
            return await asyncio.to_thread(self._storage.get_bytes, object_key)
        except ObjectNotFoundError as exc:
            raise FacadeLibraryUnavailable(
                f"library object is missing: {object_key}"
            ) from exc
        except ObjectStorageError as exc:
            raise FacadeLibraryUnavailable(str(exc)) from exc

    async def manifest(self) -> FacadeLibraryManifest:
        if self._manifest is not None and self._fresh(self._manifest_loaded_at):
            return self._manifest
        async with self._lock:
            if self._manifest is not None and self._fresh(self._manifest_loaded_at):
                return self._manifest
            payload = await self._read(self.manifest_key)
            try:
                manifest = FacadeLibraryManifest.model_validate_json(payload)
            except ValidationError as exc:
                raise FacadeLibraryUnavailable(
                    f"invalid library manifest: {exc}"
                ) from exc
            self._manifest = manifest
            self._manifest_loaded_at = time.monotonic()
            return manifest

    async def preview_index(self) -> StylePreviewIndex:
        if self._previews is not None and self._fresh(self._previews_loaded_at):
            return self._previews
        async with self._lock:
            if self._previews is not None and self._fresh(self._previews_loaded_at):
                return self._previews
            payload = await self._read(self.preview_index_key)
            try:
                index = StylePreviewIndex.model_validate_json(payload)
            except ValidationError as exc:
                raise FacadeLibraryUnavailable(f"invalid preview index: {exc}") from exc
            self._previews = index
            self._previews_loaded_at = time.monotonic()
            return index

    async def read_object(self, object_key: str) -> bytes:
        return await self._read(object_key)

    async def load_meshes(
        self, templates: Sequence[FacadeTemplate]
    ) -> dict[str, trimesh.Trimesh]:
        """Return centred template meshes keyed by ``FacadeTemplate.cache_key``."""
        meshes: dict[str, trimesh.Trimesh] = {}
        for template in templates:
            key = template.cache_key
            if key in meshes:
                continue
            cached = self._meshes.get(key)
            if cached is None:
                payload = await self._read(template.object_key)
                cached = await asyncio.to_thread(load_template_mesh, payload, key)
                self._remember(key, cached)
            else:
                self._meshes.move_to_end(key)
            meshes[key] = cached
        return meshes

    def _remember(self, key: str, mesh: trimesh.Trimesh) -> None:
        self._meshes[key] = mesh
        self._meshes.move_to_end(key)
        while len(self._meshes) > self._mesh_cache_size:
            self._meshes.popitem(last=False)


__all__ = [
    "FacadeLibraryUnavailable",
    "FacadeTemplateLibrary",
    "FacadeTemplateMiss",
    "resolve_template",
]
