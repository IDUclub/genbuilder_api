"""Facade style catalogue, style previews and library-built 3D scenes.

Preview GLBs are static library artefacts with no user data, so they are served
without a token and cached by the browser. Scenes are generation results and
need the caller's token, like the geo layers under ``/files``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Path, Query, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from app.exceptions.http_exception_wrapper import http_exception
from app.infrastructure.object_storage import ObjectStorage, get_object_storage
from app.logic.facade_library.catalog import (
    FacadeLibraryUnavailable,
    FacadeTemplateLibrary,
)
from app.logic.facade_library.factory import get_facade_library
from app.logic.facade_library.models import PreviewFloorGroup, StylePreviewIndex
from app.logic.facade_library.results import GLB_MIME_TYPE, is_scene_id, scene_glb_key
from app.logic.facade_styles import FACADE_STYLE_PRESETS, PRESETS_BY_ID
from app.schema.dto import FacadeStyleSummary
from app.utils import auth

facade_styles_router = APIRouter()

_PREVIEW_CACHE_CONTROL = "public, max-age=3600"
_DEFAULT_PREVIEW_GROUP: PreviewFloorGroup = "medium"


def _preview_url(style_id: str) -> str:
    return f"/facade-styles/{style_id}/preview.glb"


async def _preview_index_or_none(
    library: FacadeTemplateLibrary,
) -> StylePreviewIndex | None:
    try:
        return await library.preview_index()
    except FacadeLibraryUnavailable as exc:
        logger.warning("Facade style previews are unavailable: {}", exc)
        return None


@facade_styles_router.get(
    "/facade-styles",
    summary="Facade style presets and their preview GLBs",
    response_model=list[FacadeStyleSummary],
)
async def list_facade_styles(
    library: Annotated[FacadeTemplateLibrary, Depends(get_facade_library)],
) -> list[FacadeStyleSummary]:
    index = await _preview_index_or_none(library)
    groups: dict[str, list[str]] = {}
    for preview in index.previews if index is not None else []:
        groups.setdefault(preview.style_id, []).append(preview.floor_group)
    return [
        FacadeStyleSummary(
            style_id=preset.style_id,
            name_ru=preset.name_ru,
            floor_groups=sorted(groups.get(preset.style_id, [])),
            preview_url=(
                _preview_url(preset.style_id) if preset.style_id in groups else None
            ),
        )
        for preset in FACADE_STYLE_PRESETS
    ]


@facade_styles_router.get(
    "/facade-styles/{style_id}/preview.glb",
    summary="Rotatable GLB box textured with one facade style",
    response_class=Response,
    responses={
        200: {"content": {GLB_MIME_TYPE: {}}},
        304: {"description": "The cached preview is still current"},
    },
)
async def facade_style_preview(
    style_id: Annotated[
        str,
        Path(pattern=r"^[a-z0-9-]{1,64}$", description="Style id", examples=["brick"]),
    ],
    library: Annotated[FacadeTemplateLibrary, Depends(get_facade_library)],
    floor_group: Annotated[
        PreviewFloorGroup, Query(description="Height of the preview box")
    ] = _DEFAULT_PREVIEW_GROUP,
    if_none_match: Annotated[str | None, Header()] = None,
) -> Response:
    if style_id not in PRESETS_BY_ID:
        raise http_exception(404, f"Unknown facade style '{style_id}'")
    try:
        index = await library.preview_index()
    except FacadeLibraryUnavailable as exc:
        logger.warning("Facade style previews are unavailable: {}", exc)
        raise http_exception(503, "Facade style previews are unavailable.") from exc

    preview = next(
        (
            item
            for item in index.previews
            if item.style_id == style_id and item.floor_group == floor_group
        ),
        None,
    )
    if preview is None:
        raise http_exception(404, f"No '{floor_group}' preview for style '{style_id}'")

    etag = f'"{preview.etag}"'
    headers = {"ETag": etag, "Cache-Control": _PREVIEW_CACHE_CONTROL}
    if if_none_match == etag:
        return Response(status_code=304, headers=headers)
    try:
        payload = await library.read_object(preview.object_key)
    except FacadeLibraryUnavailable as exc:
        logger.warning(
            "Facade style preview {} is unreadable: {}", preview.object_key, exc
        )
        raise http_exception(503, "Facade style previews are unavailable.") from exc
    return Response(content=payload, media_type=GLB_MIME_TYPE, headers=headers)


@facade_styles_router.get(
    "/facade-scenes/{result_id}.glb",
    summary="Download a 3D scene assembled from the facade library",
    response_class=StreamingResponse,
    responses={200: {"content": {GLB_MIME_TYPE: {}}}},
)
def facade_scene(
    result_id: Annotated[str, Path(description="Scene id (uuid4 hex)")],
    token: Annotated[str, Depends(auth.verify_token)],
    storage: Annotated[ObjectStorage, Depends(get_object_storage)],
) -> StreamingResponse:
    """Declared ``def``: the storage client is synchronous and runs in a thread."""
    if not is_scene_id(result_id):
        raise http_exception(404, "Unknown 3D scene")
    key = scene_glb_key(result_id)
    if not storage.exists(key):
        raise http_exception(404, "3D scene is no longer available.")
    return StreamingResponse(
        storage.open_stream(key),
        media_type=GLB_MIME_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{result_id}.glb"'},
    )


__all__ = ["facade_styles_router"]
