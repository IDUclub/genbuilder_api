import asyncio
import io
import json
import struct

import numpy as np
import pytest
import trimesh
from facade_library_support import (
    LAT,
    LON,
    PREFIX,
    collection,
    make_library,
    make_template,
    rect_feature,
    seed_library,
    wall_glb,
)

from app.infrastructure.object_storage import LocalStorage
from app.logic.facade_library.assembly import (
    FacadeAssemblyError,
    SceneNode,
    building_faces,
    export_glb,
    load_template_mesh,
    wall_transform,
)
from app.logic.facade_library.catalog import (
    FacadeLibraryUnavailable,
    FacadeTemplateMiss,
    resolve_template,
)
from app.logic.facade_library.previews import box_faces, build_preview_glb, preview_etag
from app.logic.facade_library.results import (
    is_scene_id,
    scene_glb_key,
    scene_metadata_key,
    store_scene,
)
from app.logic.facade_library.scene import (
    EmptyScene,
    SceneTooLarge,
    build_library_scene,
)
from app.logic.mass_model import build_local_frame, local_frame_origin_wgs84

_RESOLVE = {"height_m": 18.0, "pixels_per_meter": 32, "max_width_scale": 2.5}


def _run(coro):
    return asyncio.run(coro)


def _load_scene(payload: bytes) -> trimesh.Scene:
    loaded = trimesh.load(io.BytesIO(payload), file_type="glb", force="scene")
    assert isinstance(loaded, trimesh.Scene)
    return loaded


def _gltf_json(payload: bytes) -> dict:
    json_length = struct.unpack_from("<I", payload, 12)[0]
    return json.loads(payload[20 : 20 + json_length])


def test_resolve_picks_the_closest_width_within_the_same_floor_group():
    templates = [make_template("brick", 6, 6.0), make_template("brick", 6, 12.0)]

    chosen = resolve_template(
        templates,
        style_id="brick",
        floors=6,
        width_m=11.0,
        allow_nearest=False,
        **_RESOLVE,
    )

    assert chosen.width_m == 12.0


def test_resolve_misses_without_nearest_when_the_floor_group_is_absent():
    templates = [make_template("brick", 3, 12.0)]

    with pytest.raises(FacadeTemplateMiss):
        resolve_template(
            templates,
            style_id="brick",
            floors=12,
            width_m=12.0,
            allow_nearest=False,
            **_RESOLVE,
        )


def test_resolve_nearest_prefers_the_closest_floor_count_of_the_same_style():
    templates = [
        make_template("brick", 3, 12.0),
        make_template("brick", 6, 12.0),
        make_template("glass", 12, 12.0),
    ]

    chosen = resolve_template(
        templates,
        style_id="brick",
        floors=12,
        width_m=12.0,
        allow_nearest=True,
        **_RESOLVE,
    )

    assert (chosen.style_id, chosen.floors) == ("brick", 6)


def test_resolve_nearest_never_borrows_another_style_or_resolution():
    templates = [
        make_template("glass", 6, 12.0),
        make_template("brick", 6, 12.0, pixels_per_meter=64),
    ]

    with pytest.raises(FacadeTemplateMiss):
        resolve_template(
            templates,
            style_id="brick",
            floors=6,
            width_m=12.0,
            allow_nearest=True,
            **_RESOLVE,
        )


def test_resolve_rejects_an_exact_group_beyond_the_width_scale():
    templates = [make_template("brick", 6, 6.0)]

    with pytest.raises(FacadeTemplateMiss):
        resolve_template(
            templates,
            style_id="brick",
            floors=6,
            width_m=24.0,
            allow_nearest=False,
            **_RESOLVE,
        )


def test_library_reports_a_missing_manifest_as_unavailable(tmp_path):
    library = make_library(LocalStorage(str(tmp_path)))

    with pytest.raises(FacadeLibraryUnavailable):
        _run(library.manifest())


def test_library_reports_an_invalid_manifest_as_unavailable(tmp_path):
    storage = LocalStorage(str(tmp_path))
    storage.put_json({"templates": [{"unexpected": True}]}, f"{PREFIX}/manifest.json")

    with pytest.raises(FacadeLibraryUnavailable):
        _run(make_library(storage).manifest())


def test_library_keeps_the_manifest_until_the_ttl_expires(tmp_path):
    storage = LocalStorage(str(tmp_path))
    seed_library(storage, [make_template("brick", 6, 12.0)])
    library = make_library(storage)

    async def scenario():
        first = await library.manifest()
        seed_library(storage, [])
        return first, await library.manifest()

    first, second = _run(scenario())

    assert second is first
    assert len(second.templates) == 1


def test_library_downloads_each_template_mesh_once(tmp_path):
    storage = LocalStorage(str(tmp_path))
    template = make_template("brick", 6, 12.0)
    seed_library(storage, [template])
    reads: list[str] = []
    original = storage.get_bytes

    def counting_get_bytes(key):
        reads.append(key)
        return original(key)

    storage.get_bytes = counting_get_bytes
    library = make_library(storage)

    async def scenario():
        await library.load_meshes([template])
        return await library.load_meshes([template, template])

    meshes = _run(scenario())

    assert reads.count(template.object_key) == 1
    assert list(meshes) == [template.cache_key]


def test_building_faces_separates_walls_and_roofs_and_splits_wide_walls():
    obj_text = (
        "o po_1\n"
        "v 0 0 0\n"
        "v 0 3 0\n"
        "v 30 0 0\n"
        "v 30 3 0\n"
        "v 30 0 -10\n"
        "v 30 3 -10\n"
        "v 0 0 -10\n"
        "v 0 3 -10\n"
        "f 1 3 4 2\n"
        "o po_1__roof\n"
        "f 2 4 6 8\n"
    )

    faces = building_faces(obj_text)

    assert [face.name for face in faces] == ["po_1"]
    assert len(faces[0].roofs) == 1
    assert len(faces[0].walls) > 1


def test_origin_of_the_local_frame_is_inside_the_input_area():
    buildings = collection(rect_feature("1", zone="residential", floors=5))

    lon, lat = local_frame_origin_wgs84(build_local_frame(buildings))

    assert lon == pytest.approx(LON, abs=0.001)
    assert lat == pytest.approx(LAT, abs=0.001)


def _seeded(tmp_path):
    storage = LocalStorage(str(tmp_path))
    seed_library(
        storage,
        [
            make_template("contemporary", 6, 12.0),
            make_template("glass", 6, 12.0),
            make_template("glass", 12, 12.0),
        ],
    )
    return storage, make_library(storage)


def _build(library, buildings, style_by_zone, *, allow_nearest=False, max_walls=1000):
    return _run(
        build_library_scene(
            buildings,
            style_by_zone=style_by_zone,
            library=library,
            pixels_per_meter=32,
            allow_nearest=allow_nearest,
            max_walls=max_walls,
        )
    )


def test_scene_textures_each_zone_with_its_style(tmp_path):
    _, library = _seeded(tmp_path)
    buildings = collection(
        rect_feature("1", zone="residential", floors=6),
        rect_feature("2", zone="business", floors=12, offset_m=60.0),
    )

    scene = _build(
        library, buildings, {"residential": "contemporary", "business": "glass"}
    )

    loaded = _load_scene(scene.glb)
    assert {"gb_1__wall_0", "gb_1__roof", "gb_2__wall_0", "gb_2__roof"} <= set(
        loaded.graph.nodes_geometry
    )
    assert loaded.graph.transforms.parents["gb_1__wall_0"] == "gb_1"
    assert scene.buildings == 2
    assert scene.template_count == 2
    assert scene.nearest_substitutions == 0
    assert scene.wall_instances >= 8
    assert scene.origin_lon == pytest.approx(LON, abs=0.001)
    assert scene.origin_lat == pytest.approx(LAT, abs=0.001)


def test_scene_stores_each_section_once_however_many_walls_use_it(tmp_path):
    _, library = _seeded(tmp_path)
    buildings = collection(
        *(
            rect_feature(
                str(index), zone="residential", floors=6, offset_m=index * 40.0
            )
            for index in range(3)
        )
    )

    scene = _build(library, buildings, {"residential": "contemporary"})

    gltf = _gltf_json(scene.glb)
    section_meshes = [m for m in gltf["meshes"] if m["name"].startswith("section_")]
    wall_nodes = [n for n in gltf["nodes"] if "__wall_" in n["name"]]
    assert len(section_meshes) == scene.template_count == 1
    assert len(wall_nodes) == scene.wall_instances > 3


def test_instanced_wall_matches_a_transformed_copy_of_the_section():
    template = load_template_mesh(wall_glb(12.0, 9.0), "test")
    points = np.array([[0, 0, 0], [8, 0, -6], [8, 9, -6], [0, 9, 0]], dtype=float)
    matrix = wall_transform(template, points)
    expected = template.copy()
    expected.apply_transform(matrix)

    payload = export_glb(
        {"section": template},
        [
            SceneNode("building"),
            SceneNode(
                "building__wall_0", geometry="section", matrix=matrix, parent="building"
            ),
        ],
    )

    assert np.allclose(_load_scene(payload).bounds, expected.bounds, atol=1e-4)


def test_export_refuses_a_child_listed_before_its_parent():
    template = load_template_mesh(wall_glb(12.0, 9.0), "test")

    with pytest.raises(FacadeAssemblyError):
        export_glb(
            {"section": template},
            [SceneNode("wall", geometry="section", parent="building")],
        )


def test_scene_miss_is_raised_when_nearest_is_off(tmp_path):
    _, library = _seeded(tmp_path)
    buildings = collection(rect_feature("1", zone="residential", floors=12))

    with pytest.raises(FacadeTemplateMiss):
        _build(library, buildings, {"residential": "contemporary"})


def test_scene_uses_the_nearest_section_and_counts_substitutions(tmp_path):
    _, library = _seeded(tmp_path)
    buildings = collection(rect_feature("1", zone="residential", floors=12))

    scene = _build(
        library, buildings, {"residential": "contemporary"}, allow_nearest=True
    )

    assert scene.nearest_substitutions == scene.wall_instances > 0


def test_scene_refuses_more_walls_than_the_limit(tmp_path):
    _, library = _seeded(tmp_path)
    buildings = collection(rect_feature("1", zone="residential", floors=6))

    with pytest.raises(SceneTooLarge):
        _build(library, buildings, {"residential": "contemporary"}, max_walls=2)


def test_scene_without_any_walls_is_rejected(tmp_path):
    _, library = _seeded(tmp_path)
    buildings = collection(
        rect_feature("1", zone="residential", floors=6, width_m=0.05, depth_m=0.05)
    )

    with pytest.raises(EmptyScene):
        _build(library, buildings, {"residential": "contemporary"})


def test_store_scene_writes_glb_and_metadata_under_a_fresh_id(tmp_path):
    storage, library = _seeded(tmp_path)
    scene = _build(
        library,
        collection(rect_feature("1", zone="business", floors=6)),
        {"business": "glass"},
    )

    ready = store_scene(
        storage, scene, style_by_zone={"business": "glass"}, facade_style="Стеклянный"
    )

    result_id = ready["result_id"]
    assert is_scene_id(result_id)
    assert ready["status"] == "ready"
    assert ready["glb_url"] == f"/facade-scenes/{result_id}.glb"
    assert storage.get_bytes(scene_glb_key(result_id)) == scene.glb
    assert storage.exists(scene_metadata_key(result_id))


def test_scene_keys_refuse_anything_but_a_hex_id():
    with pytest.raises(ValueError):
        scene_glb_key("../manifest")


def test_preview_box_has_four_outward_walls_and_a_roof():
    walls, roof = box_faces(width_m=12.0, height_m=18.0)

    centre = np.array([0.0, 9.0, 0.0])
    for wall in walls:
        normal = np.cross(wall[1] - wall[0], wall[2] - wall[0])
        assert float(np.dot(normal, wall.mean(axis=0) - centre)) > 0
    assert roof.shape == (4, 3)
    assert np.allclose(roof[:, 1], 18.0)


def test_preview_glb_is_a_box_of_the_requested_height():
    template = load_template_mesh(wall_glb(12.0, 18.0), "test")

    payload = build_preview_glb(
        template, width_m=12.0, height_m=18.0, name="preview_brick"
    )

    loaded = _load_scene(payload)
    assert {"preview_brick__wall_0", "preview_brick__roof"} <= set(
        loaded.graph.nodes_geometry
    )
    assert len(_gltf_json(payload)["meshes"]) == 2
    assert loaded.extents[1] == pytest.approx(18.0, abs=0.5)
    assert len(preview_etag(payload)) == 16
