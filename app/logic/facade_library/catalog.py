from __future__ import annotations

import asyncio
import time
import zlib
from collections import OrderedDict
from collections.abc import Sequence

from pydantic import ValidationError

from app.infrastructure.object_storage import (
    ObjectNotFoundError,
    ObjectStorage,
    ObjectStorageError,
)
from app.logic.facade_library.assembly import load_template_mesh
from app.logic.facade_library.floors import FloorPieces, slice_section
from app.logic.facade_library.models import (
    FacadeLibraryManifest,
    FacadeSection,
    StylePreviewIndex,
)

_DEFAULT_MESH_CACHE_SIZE = 192


class FacadeTemplateMiss(LookupError):
    """No library template can serve a wall."""


class FacadeLibraryUnavailable(RuntimeError):
    """The library manifest or a template could not be read."""


def _pick_variant(
    variants: Sequence[FacadeSection], variant_key: str, variant_count: int
) -> FacadeSection:
    # crc32 instead of hash(): str hashes are salted per process.
    wanted = zlib.crc32(variant_key.encode("utf-8")) % variant_count
    ordered = sorted(variants, key=lambda item: item.variant)
    for section in ordered:
        if section.variant == wanted:
            return section
    return ordered[wanted % len(ordered)]


def resolve_section(
    sections: Sequence[FacadeSection],
    *,
    style_id: str,
    width_m: float,
    pixels_per_meter: int,
    max_width_scale: float,
    allow_nearest: bool,
    variant_key: str,
) -> FacadeSection:
    """Pick the section for one wall by width; floor count plays no part.

    An exact match shares the style and resolution and stays within
    ``max_width_scale``. With ``allow_nearest`` a miss falls back to the same
    style's section closest in width. ``variant_key`` selects the variant
    number from all the style's variants, so equal keys get the same variant on
    every width that has it.
    """
    same_style = [
        section
        for section in sections
        if section.style_id == style_id and section.pixels_per_meter == pixels_per_meter
    ]
    exact = [
        section
        for section in same_style
        if section.width_scale_for(width_m) <= max_width_scale
    ]
    candidates = exact or (same_style if allow_nearest else [])
    if candidates:
        best = min(candidates, key=lambda item: (item.score(width_m), item.width_m))
        variants = [item for item in candidates if item.width_m == best.width_m]
        variant_count = 1 + max(item.variant for item in same_style)
        return _pick_variant(variants, variant_key, variant_count)
    raise FacadeTemplateMiss(
        f"no section for style={style_id}, width={width_m:.2f}m, "
        f"ppm={pixels_per_meter}"
    )


def _load_pieces(payload: bytes, section: FacadeSection) -> FloorPieces:
    mesh = load_template_mesh(payload, section.cache_key)
    return slice_section(mesh, section.section_floors, section.height_m)


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
        self._pieces: OrderedDict[str, FloorPieces] = OrderedDict()

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

    async def load_pieces(
        self, sections: Sequence[FacadeSection]
    ) -> dict[str, FloorPieces]:
        """Return sliced floor pieces keyed by ``FacadeSection.cache_key``."""
        pieces: dict[str, FloorPieces] = {}
        for section in sections:
            key = section.cache_key
            if key in pieces:
                continue
            cached = self._pieces.get(key)
            if cached is None:
                payload = await self._read(section.object_key)
                cached = await asyncio.to_thread(_load_pieces, payload, section)
                self._remember(key, cached)
            else:
                self._pieces.move_to_end(key)
            pieces[key] = cached
        return pieces

    def _remember(self, key: str, section_pieces: FloorPieces) -> None:
        self._pieces[key] = section_pieces
        self._pieces.move_to_end(key)
        while len(self._pieces) > self._mesh_cache_size:
            self._pieces.popitem(last=False)


__all__ = [
    "FacadeLibraryUnavailable",
    "FacadeTemplateLibrary",
    "FacadeTemplateMiss",
    "resolve_section",
]
