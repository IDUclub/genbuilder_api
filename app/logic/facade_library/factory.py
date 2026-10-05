from __future__ import annotations

from functools import lru_cache

from app.infrastructure.object_storage import get_object_storage
from app.logic.facade_library.catalog import FacadeTemplateLibrary
from app.settings import get_settings


@lru_cache(maxsize=1)
def get_facade_library() -> FacadeTemplateLibrary:
    """Process-wide library so the manifest and template meshes stay cached."""
    settings = get_settings()
    return FacadeTemplateLibrary(
        get_object_storage(),
        prefix=settings.facade_library_prefix,
        manifest_ttl_seconds=settings.facade_library_manifest_ttl_seconds,
        max_width_scale=settings.facade_library_max_width_scale,
    )


__all__ = ["get_facade_library"]
