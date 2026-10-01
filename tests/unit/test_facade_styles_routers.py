import io
import json
import struct

import numpy as np
import pytest
import trimesh
from facade_library_support import make_library
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.infrastructure.object_storage import LocalStorage, get_object_storage
from app.logic.facade_library.factory import get_facade_library
from app.logic.facade_library.models import StylePreview, StylePreviewIndex
from app.logic.facade_library.previews import build_preview_glb
from app.logic.facade_library.results import GLB_MIME_TYPE, scene_glb_key
from app.logic.facade_styles import FACADE_STYLE_PRESETS
from app.routers import facade_styles_routers
from app.utils import auth

PREVIEW_PAYLOAD = b"glTF-preview-payload"
SCENE_ID = "0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def _empty_gallery_cache(monkeypatch):
    monkeypatch.setattr(facade_styles_routers, "_gallery_cache", {})


def _textured_preview(style_id: str, color: tuple[int, int, int]) -> bytes:
    section = trimesh.creation.box(extents=[12.0, 9.0, 0.4])
    uv = np.c_[(section.vertices[:, 0] + 6) / 12, (section.vertices[:, 1] + 4.5) / 9]
    section.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.new("RGB", (8, 8), color)
        ),
    )
    return build_preview_glb(
        section, width_m=12.0, height_m=9.0, name=f"preview_{style_id}"
    )


def _gltf_json(payload: bytes) -> dict:
    length = struct.unpack("<I", payload[12:16])[0]
    return json.loads(payload[20 : 20 + length])


def _seed_previews(
    storage: LocalStorage,
    library,
    previews: dict[str, list[str]],
    payloads: dict[str, bytes] | None = None,
) -> None:
    entries = []
    for style_id, groups in previews.items():
        for group in groups:
            key = library.preview_key(style_id, group)
            payload = (payloads or {}).get(style_id, PREVIEW_PAYLOAD)
            storage.put_bytes(payload, key, GLB_MIME_TYPE)
            entries.append(
                StylePreview(
                    style_id=style_id,
                    style_name_ru=style_id,
                    floor_group=group,
                    object_key=key,
                    etag=f"etag-{style_id}-{group}",
                    size_bytes=len(payload),
                )
            )
    index = StylePreviewIndex(previews=entries)
    storage.put_json(index.model_dump(mode="json"), library.preview_index_key)


def _client(tmp_path, *, previews=None, payloads=None, authenticated=True):
    storage = LocalStorage(str(tmp_path))
    library = make_library(storage)
    if previews is not None:
        _seed_previews(storage, library, previews, payloads)
    app = FastAPI()
    app.include_router(facade_styles_routers.facade_styles_router)
    app.dependency_overrides[get_facade_library] = lambda: library
    app.dependency_overrides[get_object_storage] = lambda: storage
    if authenticated:
        app.dependency_overrides[auth.verify_token] = lambda: "user-token"
    return TestClient(app), storage


def test_list_returns_every_preset_with_preview_links_only_where_previews_exist(
    tmp_path,
):
    client, _ = _client(tmp_path, previews={"brick": ["medium", "low"]})

    response = client.get("/facade-styles")

    assert response.status_code == 200
    body = {item["style_id"]: item for item in response.json()}
    assert list(body) == [preset.style_id for preset in FACADE_STYLE_PRESETS]
    assert body["brick"]["floor_groups"] == ["low", "medium"]
    assert body["brick"]["preview_url"] == "/facade-styles/brick/preview.glb"
    assert body["glass"]["floor_groups"] == []
    assert body["glass"]["preview_url"] is None


def test_list_still_answers_when_the_preview_index_is_missing(tmp_path):
    client, _ = _client(tmp_path)

    response = client.get("/facade-styles")

    assert response.status_code == 200
    assert all(item["preview_url"] is None for item in response.json())


def test_preview_returns_the_glb_with_cache_headers(tmp_path):
    client, _ = _client(tmp_path, previews={"brick": ["medium"]})

    response = client.get("/facade-styles/brick/preview.glb")

    assert response.status_code == 200
    assert response.content == PREVIEW_PAYLOAD
    assert response.headers["content-type"] == GLB_MIME_TYPE
    assert response.headers["etag"] == '"etag-brick-medium"'
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_preview_answers_304_when_the_etag_matches(tmp_path):
    client, _ = _client(tmp_path, previews={"brick": ["medium"]})

    response = client.get(
        "/facade-styles/brick/preview.glb",
        headers={"If-None-Match": '"etag-brick-medium"'},
    )

    assert response.status_code == 304
    assert response.content == b""


def test_preview_of_an_unknown_style_is_404(tmp_path):
    client, _ = _client(tmp_path, previews={"brick": ["medium"]})

    assert client.get("/facade-styles/gothic/preview.glb").status_code == 404


def test_preview_of_a_missing_floor_group_is_404(tmp_path):
    client, _ = _client(tmp_path, previews={"brick": ["medium"]})

    response = client.get(
        "/facade-styles/brick/preview.glb", params={"floor_group": "high"}
    )

    assert response.status_code == 404


def test_preview_rejects_a_malformed_style_id(tmp_path):
    client, _ = _client(tmp_path, previews={"brick": ["medium"]})

    assert client.get("/facade-styles/Brick_1/preview.glb").status_code == 422


def test_preview_is_503_when_the_index_is_missing(tmp_path):
    client, _ = _client(tmp_path)

    assert client.get("/facade-styles/brick/preview.glb").status_code == 503


def _gallery_client(tmp_path):
    colors = {"glass": (0, 0, 200), "brick": (200, 0, 0), "loft": (0, 200, 0)}
    return _client(
        tmp_path,
        previews={style_id: ["medium"] for style_id in colors},
        payloads={
            style_id: _textured_preview(style_id, color)
            for style_id, color in colors.items()
        },
    )


def test_gallery_holds_one_textured_box_per_style_in_preset_order(tmp_path):
    client, _ = _gallery_client(tmp_path)

    response = client.get("/facade-styles/gallery.glb")

    assert response.status_code == 200
    assert response.headers["content-type"] == GLB_MIME_TYPE
    assert response.headers["cache-control"] == "public, max-age=3600"
    gltf = _gltf_json(response.content)
    nodes = gltf["nodes"]
    # trimesh wraps the scene in one "world" node; the style boxes are its children.
    (world,) = (nodes[index] for index in gltf["scenes"][0]["nodes"])
    roots = [nodes[index]["name"] for index in world["children"]]
    order = [preset.style_id for preset in FACADE_STYLE_PRESETS]
    expected = sorted(["glass", "brick", "loft"], key=order.index)
    assert roots == [f"style_{style_id}" for style_id in expected]
    assert len(gltf["images"]) == 3
    # Every style sits at the origin, so the stack is as wide as one box.
    assert all("matrix" not in nodes[index] for index in world["children"])
    scene = trimesh.load(io.BytesIO(response.content), file_type="glb", force="scene")
    assert (scene.extents[[0, 2]] < 2 * 12.0).all()


def test_gallery_answers_304_and_reuses_the_built_payload(tmp_path, monkeypatch):
    client, _ = _gallery_client(tmp_path)
    first = client.get("/facade-styles/gallery.glb")
    builds = []
    monkeypatch.setattr(
        facade_styles_routers,
        "build_gallery_glb",
        lambda previews: builds.append(previews) or b"",
    )

    again = client.get("/facade-styles/gallery.glb")
    cached = client.get(
        "/facade-styles/gallery.glb", headers={"If-None-Match": first.headers["etag"]}
    )

    assert again.content == first.content
    assert again.headers["etag"] == first.headers["etag"]
    assert cached.status_code == 304
    assert builds == []


def test_gallery_of_a_floor_group_without_previews_is_404(tmp_path):
    client, _ = _gallery_client(tmp_path)

    response = client.get("/facade-styles/gallery.glb", params={"floor_group": "high"})

    assert response.status_code == 404


def test_gallery_is_503_when_the_index_is_missing(tmp_path):
    client, _ = _client(tmp_path)

    assert client.get("/facade-styles/gallery.glb").status_code == 503


def test_scene_download_requires_a_token(tmp_path):
    client, _ = _client(tmp_path, authenticated=False)

    response = client.get(f"/facade-scenes/{SCENE_ID}.glb")

    assert response.status_code in {401, 403}


def test_scene_download_of_a_malformed_id_is_404(tmp_path):
    client, _ = _client(tmp_path)

    assert client.get("/facade-scenes/not-a-scene.glb").status_code == 404


def test_scene_download_of_a_missing_scene_is_404(tmp_path):
    client, _ = _client(tmp_path)

    assert client.get(f"/facade-scenes/{SCENE_ID}.glb").status_code == 404


def test_scene_download_streams_the_stored_glb(tmp_path):
    client, storage = _client(tmp_path)
    storage.put_bytes(b"glTF-scene", scene_glb_key(SCENE_ID), GLB_MIME_TYPE)

    response = client.get(f"/facade-scenes/{SCENE_ID}.glb")

    assert response.status_code == 200
    assert response.content == b"glTF-scene"
    assert response.headers["content-type"] == GLB_MIME_TYPE
