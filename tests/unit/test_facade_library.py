import asyncio
import io
import json
import struct
from itertools import pairwise

import numpy as np
import pytest
import trimesh
from facade_library_support import (
    LAT,
    LON,
    PREFIX,
    collection,
    make_library,
    make_section,
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
    split_wall,
)
from app.logic.facade_library.catalog import (
    FacadeLibraryUnavailable,
    FacadeTemplateMiss,
    resolve_section,
)
from app.logic.facade_library.floors import (
    FloorKind,
    FloorPieces,
    floor_sequence,
    slice_section,
    stack_transforms,
)
from app.logic.facade_library.models import FacadeLibraryManifest
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

_RESOLVE = {"pixels_per_meter": 32, "max_width_scale": 2.5, "variant_key": "gb_1"}


def _run(coro):
    return asyncio.run(coro)


def _load_scene(payload: bytes) -> trimesh.Scene:
    loaded = trimesh.load(io.BytesIO(payload), file_type="glb", force="scene")
    assert isinstance(loaded, trimesh.Scene)
    return loaded


def _gltf_json(payload: bytes) -> dict:
    json_length = struct.unpack_from("<I", payload, 12)[0]
    return json.loads(payload[20 : 20 + json_length])


SLAB_SHARE = 0.8


def _striped_section(
    floors: int = 8, floor_height: float = 4.0, missing: int | None = None
) -> trimesh.Trimesh:
    """One slab per floor, shifted along X by its floor index.

    Slabs fill the lower 80% of their floor, the top one the upper 80%, so the
    section spans exactly ``floors * floor_height``.
    """
    slab_height = floor_height * SLAB_SHARE
    slabs = []
    for index in range(floors):
        if index == missing:
            continue
        slab = trimesh.creation.box(extents=[1.0, slab_height, 0.2])
        bottom = index * floor_height
        if index == floors - 1:
            bottom += floor_height - slab_height
        slab.apply_translation([index * 2.0, bottom + slab_height / 2, 0.0])
        slabs.append(slab)
    mesh = trimesh.util.concatenate(slabs)
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    return mesh


BALCONY_LOW, BALCONY_HIGH, BALCONY_DEPTH = 27.62, 28.83, 1.2


def _section_with_straddling_balcony() -> trimesh.Trimesh:
    """A 6 x 32 m wall whose top balcony crosses the 28 m floor line, as in Facades-3D output."""
    wall = trimesh.creation.box(extents=[6.0, 32.0, 0.2])
    wall.apply_translation([0.0, 16.0, 0.0])
    balcony = trimesh.creation.box(
        extents=[4.0, BALCONY_HIGH - BALCONY_LOW, BALCONY_DEPTH]
    )
    balcony.apply_translation(
        [0.0, (BALCONY_LOW + BALCONY_HIGH) / 2, 0.1 + BALCONY_DEPTH / 2]
    )
    return trimesh.util.concatenate([wall, balcony])


def _floor_index(piece: trimesh.Trimesh, section: trimesh.Trimesh) -> int:
    return round((piece.centroid[0] - section.bounds[0][0] - 0.5) / 2.0)


def test_resolve_picks_the_closest_width_whatever_the_floor_count():
    sections = [make_section("brick", 6.0), make_section("brick", 12.0)]

    chosen = resolve_section(
        sections, style_id="brick", width_m=11.0, allow_nearest=False, **_RESOLVE
    )

    assert chosen.width_m == 12.0


def test_resolve_rejects_a_section_beyond_the_width_scale():
    sections = [make_section("brick", 6.0)]

    with pytest.raises(FacadeTemplateMiss):
        resolve_section(
            sections, style_id="brick", width_m=24.0, allow_nearest=False, **_RESOLVE
        )


def test_resolve_nearest_takes_the_closest_width_of_the_same_style():
    sections = [
        make_section("brick", 6.0),
        make_section("brick", 9.0),
        make_section("glass", 24.0),
    ]

    chosen = resolve_section(
        sections, style_id="brick", width_m=40.0, allow_nearest=True, **_RESOLVE
    )

    assert (chosen.style_id, chosen.width_m) == ("brick", 9.0)


def test_resolve_nearest_never_borrows_another_style_or_resolution():
    sections = [
        make_section("glass", 12.0),
        make_section("brick", 12.0, pixels_per_meter=64),
    ]

    with pytest.raises(FacadeTemplateMiss):
        resolve_section(
            sections, style_id="brick", width_m=12.0, allow_nearest=True, **_RESOLVE
        )


def _variants(style_id, width_m, count=3):
    return [make_section(style_id, width_m, variant=n) for n in range(count)]


def _resolve_variant(sections, key, width_m=12.0):
    options = {**_RESOLVE, "variant_key": key}
    return resolve_section(
        sections, style_id="brick", width_m=width_m, allow_nearest=False, **options
    ).variant


def test_resolve_gives_the_same_key_the_same_variant():
    sections = _variants("brick", 12.0)

    picks = {_resolve_variant(sections, "gb_42") for _ in range(5)}

    assert len(picks) == 1


def test_resolve_spreads_keys_over_all_variants():
    sections = _variants("brick", 12.0)

    picks = {_resolve_variant(sections, f"gb_{index}") for index in range(30)}

    assert picks == {0, 1, 2}


def test_resolve_picks_the_variant_among_sections_of_the_chosen_width():
    sections = [*_variants("brick", 6.0), *_variants("brick", 12.0)]

    chosen = resolve_section(
        sections, style_id="brick", width_m=11.0, allow_nearest=False, **_RESOLVE
    )

    assert chosen.width_m == 12.0


def test_resolve_uses_the_variants_that_exist():
    sections = [make_section("brick", 12.0, variant=2)]

    picks = {_resolve_variant(sections, f"gb_{index}") for index in range(10)}

    assert picks == {2}


def test_resolve_keeps_the_variant_number_when_a_width_lacks_some_variants():
    sections = [
        *_variants("brick", 6.0),
        make_section("brick", 12.0, variant=0),
        make_section("brick", 12.0, variant=2),
    ]

    for index in range(30):
        key = f"gb_{index}"
        on_narrow = _resolve_variant(sections, key, width_m=6.0)
        on_wide = _resolve_variant(sections, key, width_m=12.0)
        if on_narrow in (0, 2):
            assert on_wide == on_narrow


def test_resolve_breaks_an_equal_width_score_towards_the_narrower_section():
    narrow, wide = make_section("brick", 6.0), make_section("brick", 24.0)

    for order in ([narrow, wide], [wide, narrow]):
        chosen = resolve_section(
            order, style_id="brick", width_m=12.0, allow_nearest=False, **_RESOLVE
        )
        assert chosen.width_m == 6.0


def test_manifest_without_variants_reads_every_section_as_variant_zero():
    payload = FacadeLibraryManifest(sections=[make_section("brick", 12.0)]).model_dump(
        mode="json"
    )
    del payload["sections"][0]["variant"]

    manifest = FacadeLibraryManifest.model_validate(payload)

    assert manifest.sections[0].variant == 0


def test_floor_sequence_puts_typical_floors_between_ground_and_top():
    assert floor_sequence(1) == ["ground"]
    assert floor_sequence(2) == ["ground", "top"]
    assert floor_sequence(5) == ["ground", "typical", "typical", "typical", "top"]
    assert len(floor_sequence(16)) == 16


def test_floor_sequence_refuses_zero_floors():
    with pytest.raises(FacadeAssemblyError):
        floor_sequence(0)


def test_slice_section_cuts_ground_middle_and_top_floors_on_the_grid():
    section = _striped_section()

    pieces = slice_section(section, 8, 32.0)

    assert {
        kind: _floor_index(mesh, section) for kind, mesh in pieces.meshes.items()
    } == {
        "ground": 0,
        "typical": 3,
        "top": 7,
    }
    step = section.extents[1] / 8
    for height in pieces.heights.values():
        assert height == pytest.approx(step, abs=step / 4)
    assert pieces.section_width == pytest.approx(section.extents[0])


def test_slice_section_refuses_fewer_than_three_floors():
    with pytest.raises(FacadeAssemblyError):
        slice_section(_striped_section(), 2, 32.0)


def test_slice_section_refuses_an_empty_floor():
    with pytest.raises(FacadeAssemblyError):
        slice_section(_striped_section(missing=3), 8, 32.0)


def test_slice_section_refuses_a_section_taller_than_its_floors():
    section = _striped_section()
    parapet = trimesh.creation.box(extents=[1.0, 2.0, 0.2])
    parapet.apply_translation([0.0, section.bounds[1][1] + 1.0, 0.0])
    with_parapet = trimesh.util.concatenate([section, parapet])

    with pytest.raises(FacadeAssemblyError, match="expected 32.00 m"):
        slice_section(with_parapet, 8, 32.0)


def test_slice_section_accepts_a_height_within_the_tolerance():
    pieces = slice_section(_striped_section(), 8, 32.2)

    assert pieces.heights["typical"] == pytest.approx(4.0, abs=1.0)


def test_slice_section_keeps_a_straddling_balcony_whole_in_the_top_floor():
    pieces = slice_section(_section_with_straddling_balcony(), 8, 32.0)

    assert pieces.bottoms["top"] < BALCONY_LOW
    assert pieces.meshes["top"].bounds[1][2] == pytest.approx(0.1 + BALCONY_DEPTH)
    assert pieces.meshes["typical"].bounds[1][2] == pytest.approx(0.1)
    assert pieces.heights["top"] == pytest.approx(32.0 - pieces.bottoms["top"])


def test_slice_section_cuts_on_the_grid_where_the_wall_is_plain():
    pieces = slice_section(_section_with_straddling_balcony(), 8, 32.0)

    assert pieces.bottoms["typical"] == pytest.approx(12.0)
    assert pieces.heights["ground"] == pytest.approx(4.0)
    assert pieces.heights["typical"] == pytest.approx(4.0)


def _stacked_bands(
    pieces: FloorPieces, placed: list[tuple[FloorKind, np.ndarray]]
) -> list[tuple[float, float]]:
    bands = []
    for kind, matrix in placed:
        low, high = pieces.meshes[kind].copy().apply_transform(matrix).bounds[:, 1]
        bands.append((float(low), float(high)))
    return bands


def test_stacked_floors_fill_the_wall_quad_exactly():
    pieces = slice_section(_section_with_straddling_balcony(), 8, 32.0)
    points = np.array([[0, 0, 0], [8, 0, -6], [8, 15, -6], [0, 15, 0]], dtype=float)

    placed = stack_transforms(pieces, points, 5)

    assert [kind for kind, _ in placed] == floor_sequence(5)
    bands = _stacked_bands(pieces, placed)
    assert bands[0][0] == pytest.approx(0.0, abs=1e-6)
    for (_, below_top), (above_bottom, _) in pairwise(bands):
        assert above_bottom == pytest.approx(below_top, abs=1e-6)
    assert bands[-1][1] == pytest.approx(15.0, abs=1e-6)


def test_stacked_floors_keep_the_proportions_of_uneven_pieces():
    pieces = slice_section(_section_with_straddling_balcony(), 8, 32.0)
    points = np.array([[0, 0, 0], [8, 0, 0], [8, 9, 0], [0, 9, 0]], dtype=float)

    bands = _stacked_bands(pieces, stack_transforms(pieces, points, 3))

    ground, typical, top = (high - low for low, high in bands)
    assert ground == pytest.approx(typical, abs=1e-6)
    assert top / ground == pytest.approx(
        pieces.heights["top"] / pieces.heights["ground"]
    )


def test_library_reports_a_missing_manifest_as_unavailable(tmp_path):
    library = make_library(LocalStorage(str(tmp_path)))

    with pytest.raises(FacadeLibraryUnavailable):
        _run(library.manifest())


def test_library_reports_an_invalid_manifest_as_unavailable(tmp_path):
    storage = LocalStorage(str(tmp_path))
    storage.put_json({"sections": [{"unexpected": True}]}, f"{PREFIX}/manifest.json")

    with pytest.raises(FacadeLibraryUnavailable):
        _run(make_library(storage).manifest())


def test_library_refuses_a_version_1_manifest(tmp_path):
    storage = LocalStorage(str(tmp_path))
    storage.put_json({"version": 1, "templates": []}, f"{PREFIX}/manifest.json")

    with pytest.raises(FacadeLibraryUnavailable):
        _run(make_library(storage).manifest())


def test_library_keeps_the_manifest_until_the_ttl_expires(tmp_path):
    storage = LocalStorage(str(tmp_path))
    seed_library(storage, [make_section("brick", 12.0)])
    library = make_library(storage)

    async def scenario():
        first = await library.manifest()
        seed_library(storage, [])
        return first, await library.manifest()

    first, second = _run(scenario())

    assert second is first
    assert len(second.sections) == 1


def test_library_downloads_and_slices_each_section_once(tmp_path):
    storage = LocalStorage(str(tmp_path))
    section = make_section("brick", 12.0)
    seed_library(storage, [section])
    reads: list[str] = []
    original = storage.get_bytes

    def counting_get_bytes(key):
        reads.append(key)
        return original(key)

    storage.get_bytes = counting_get_bytes
    library = make_library(storage)

    async def scenario():
        first = await library.load_pieces([section])
        return first, await library.load_pieces([section, section])

    first, second = _run(scenario())

    assert reads.count(section.object_key) == 1
    assert list(second) == [section.cache_key]
    assert second[section.cache_key] is first[section.cache_key]
    assert set(second[section.cache_key].meshes) == {"ground", "typical", "top"}


def test_library_refuses_a_section_whose_glb_height_differs_from_the_manifest(
    tmp_path,
):
    storage = LocalStorage(str(tmp_path))
    section = make_section("brick", 12.0)
    seed_library(storage, [section])
    storage.put_bytes(wall_glb(12.0, 20.0), section.object_key, "model/gltf-binary")

    with pytest.raises(FacadeAssemblyError):
        _run(make_library(storage).load_pieces([section]))


def test_building_faces_separates_walls_and_roofs_and_splits_wide_walls():
    obj_text = (
        "o po_1\n"
        "v 0 0 0\n"
        "v 0 3 0\n"
        "v 50 0 0\n"
        "v 50 3 0\n"
        "v 50 0 -10\n"
        "v 50 3 -10\n"
        "v 0 0 -10\n"
        "v 0 3 -10\n"
        "f 1 3 4 2\n"
        "o po_1__roof\n"
        "f 2 4 6 8\n"
    )

    faces = building_faces(obj_text)

    assert [face.name for face in faces] == ["po_1"]
    assert len(faces[0].roofs) == 1
    assert len(faces[0].walls) == 3


@pytest.mark.parametrize(
    ("width", "height", "segments"),
    [(24.0, 3.0, 1), (24.0, 60.0, 1), (25.0, 3.0, 2), (60.0, 30.0, 3)],
)
def test_split_wall_depends_on_width_only(width, height, segments):
    points = np.array(
        [[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]], dtype=float
    )

    parts = split_wall(points, 24.0)

    assert len(parts) == segments
    assert sum(float(np.linalg.norm(p[1] - p[0])) for p in parts) == pytest.approx(
        width
    )


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
            make_section("contemporary", 12.0),
            make_section("glass", 12.0),
            make_section("glass", 24.0),
            make_section("minimalist", 6.0),
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
    assert {"gb_1__wall_0__floor_0", "gb_1__roof", "gb_2__wall_0__floor_11"} <= set(
        loaded.graph.nodes_geometry
    )
    parents = loaded.graph.transforms.parents
    assert parents["gb_1__wall_0__floor_0"] == "gb_1__wall_0"
    assert parents["gb_1__wall_0"] == "gb_1"
    assert scene.buildings == 2
    assert scene.template_count == 3
    assert scene.nearest_substitutions == 0
    assert scene.wall_instances == 8
    assert scene.floor_instances == 4 * 6 + 4 * 12
    assert scene.origin_lon == pytest.approx(LON, abs=0.001)
    assert scene.origin_lat == pytest.approx(LAT, abs=0.001)


def test_scene_stores_three_floor_pieces_per_section_for_any_heights(tmp_path):
    _, library = _seeded(tmp_path)
    buildings = collection(
        *(
            rect_feature(
                str(floors), zone="residential", floors=floors, offset_m=floors * 40.0
            )
            for floors in (3, 9, 16)
        )
    )

    scene = _build(library, buildings, {"residential": "contemporary"})

    gltf = _gltf_json(scene.glb)
    piece_meshes = sorted(
        m["name"] for m in gltf["meshes"] if m["name"].startswith("section_")
    )
    floor_nodes = [n for n in gltf["nodes"] if "__floor_" in n["name"]]
    assert piece_meshes == [
        "section_0__ground",
        "section_0__top",
        "section_0__typical",
    ]
    assert len(floor_nodes) == scene.floor_instances == 4 * (3 + 9 + 16)


def _many_buildings(count):
    return collection(
        *(
            rect_feature(
                str(index),
                zone="residential",
                floors=5,
                width_m=12.0,
                offset_m=index * 40.0,
            )
            for index in range(count)
        )
    )


def test_all_walls_of_a_building_share_one_variant(tmp_path):
    storage = LocalStorage(str(tmp_path))
    seed_library(storage, _variants("contemporary", 12.0))

    scene = _build(
        make_library(storage), _many_buildings(1), {"residential": "contemporary"}
    )

    assert scene.wall_instances == 4
    assert scene.template_count == 1


def test_different_buildings_get_different_variants(tmp_path):
    storage = LocalStorage(str(tmp_path))
    seed_library(storage, _variants("contemporary", 12.0))

    scene = _build(
        make_library(storage), _many_buildings(12), {"residential": "contemporary"}
    )

    assert scene.template_count == 3


def test_one_storey_building_uses_only_the_ground_piece(tmp_path):
    _, library = _seeded(tmp_path)
    buildings = collection(rect_feature("1", zone="residential", floors=1))

    scene = _build(library, buildings, {"residential": "contemporary"})

    gltf = _gltf_json(scene.glb)
    assert [m["name"] for m in gltf["meshes"] if m["name"].startswith("section_")] == [
        "section_0__ground"
    ]
    assert scene.floor_instances == scene.wall_instances == 4


def test_scene_is_as_tall_as_the_buildings(tmp_path):
    _, library = _seeded(tmp_path)
    buildings = collection(rect_feature("1", zone="business", floors=9))

    scene = _build(library, buildings, {"business": "glass"})

    assert _load_scene(scene.glb).extents[1] == pytest.approx(27.0, abs=0.5)


def test_floor_node_matches_a_transformed_copy_of_its_piece():
    pieces = slice_section(load_template_mesh(wall_glb(12.0, 32.0), "test"), 8, 32.0)
    points = np.array([[0, 0, 0], [8, 0, -6], [8, 9, -6], [0, 9, 0]], dtype=float)
    kind, matrix = stack_transforms(pieces, points, 3)[1]
    expected = pieces.meshes[kind].copy()
    expected.apply_transform(matrix)

    payload = export_glb(
        {"piece": pieces.meshes[kind]},
        [
            SceneNode("building"),
            SceneNode(
                "building__wall_0__floor_1",
                geometry="piece",
                matrix=matrix,
                parent="building",
            ),
        ],
    )

    assert np.allclose(_load_scene(payload).bounds, expected.bounds, atol=1e-4)


def test_export_makes_every_material_non_metallic():
    template = load_template_mesh(wall_glb(12.0, 9.0), "test")
    roof = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 0, 1]], faces=[[0, 1, 2]], process=False
    )

    payload = export_glb(
        {"section": template, "roof": roof},
        [SceneNode("wall", geometry="section"), SceneNode("roof", geometry="roof")],
    )

    gltf = _gltf_json(payload)
    primitives = [p for mesh in gltf["meshes"] for p in mesh["primitives"]]
    assert all("material" in primitive for primitive in primitives)
    assert all(
        material["pbrMetallicRoughness"]["metallicFactor"] == 0.0
        for material in gltf["materials"]
    )
    assert struct.unpack_from("<I", payload, 8)[0] == len(payload)
    assert len(_load_scene(payload).geometry) == 2


def test_export_refuses_a_child_listed_before_its_parent():
    template = load_template_mesh(wall_glb(12.0, 9.0), "test")

    with pytest.raises(FacadeAssemblyError):
        export_glb(
            {"section": template},
            [SceneNode("wall", geometry="section", parent="building")],
        )


def test_scene_miss_is_raised_when_nearest_is_off(tmp_path):
    _, library = _seeded(tmp_path)
    buildings = collection(rect_feature("1", zone="residential", floors=6))

    with pytest.raises(FacadeTemplateMiss):
        _build(library, buildings, {"residential": "minimalist"})


def test_scene_uses_the_nearest_section_and_counts_substitutions(tmp_path):
    _, library = _seeded(tmp_path)
    buildings = collection(rect_feature("1", zone="residential", floors=6))

    scene = _build(
        library, buildings, {"residential": "minimalist"}, allow_nearest=True
    )

    assert scene.nearest_substitutions == 2
    assert scene.wall_instances == 4


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
    assert ready["stats"]["floor_instances"] == scene.floor_instances
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


def test_preview_glb_stacks_the_requested_floors():
    pieces = slice_section(load_template_mesh(wall_glb(12.0, 32.0), "test"), 8, 32.0)

    payload = build_preview_glb(
        pieces, width_m=12.0, floors=6, floor_height_m=3.0, name="preview_brick"
    )

    loaded = _load_scene(payload)
    assert {"preview_brick__wall_0__floor_5", "preview_brick__roof"} <= set(
        loaded.graph.nodes_geometry
    )
    assert len(_gltf_json(payload)["meshes"]) == 4
    assert loaded.extents[1] == pytest.approx(18.0, abs=0.5)
    assert len(preview_etag(payload)) == 16
