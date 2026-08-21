from urllib.parse import parse_qs, urlparse

import pytest

from app.common.urls import durable_url
from app.logic.geo_layers import (
    FILE_SLOTS,
    SLOT_BLOCKS_INPUT,
    SLOT_BUILDINGS,
    build_stored_layer,
    build_zones_layer,
    geo_layer_to_file_part,
    object_key,
)

PUBLIC = "http://10.32.1.46:8200"
RESULT_ID = "a1b2c3d4e5f6"


def test_durable_url_prefers_the_configured_public_base():
    assert (
        durable_url("/files/buildings/x", PUBLIC, "http://internal:8000/")
        == "http://10.32.1.46:8200/files/buildings/x"
    )


def test_durable_url_falls_back_to_the_request_base():
    assert (
        durable_url("/files/buildings/x", None, "http://internal:8000/")
        == "http://internal:8000/files/buildings/x"
    )


def test_durable_url_is_relative_without_any_base():
    assert durable_url("/files/buildings/x", None) == "/files/buildings/x"


def test_durable_url_does_not_double_the_slash():
    assert durable_url("/a", "http://host:8200/") == "http://host:8200/a"


def test_object_key_is_derived_from_result_id_and_slot():
    assert object_key(RESULT_ID, SLOT_BUILDINGS) == f"{RESULT_ID}/buildings.geojson"
    assert (
        object_key(RESULT_ID, SLOT_BLOCKS_INPUT) == f"{RESULT_ID}/blocks_input.geojson"
    )


def test_object_key_rejects_an_unknown_slot():
    with pytest.raises(ValueError):
        object_key(RESULT_ID, "../etc/passwd")


def test_object_key_requires_a_result_id():
    with pytest.raises(ValueError):
        object_key("", SLOT_BUILDINGS)


def test_file_slots_are_the_declared_whitelist():
    assert FILE_SLOTS == (SLOT_BUILDINGS, SLOT_BLOCKS_INPUT)


def test_build_stored_layer_points_at_the_file_endpoint():
    layer = build_stored_layer(
        slot=SLOT_BUILDINGS, result_id=RESULT_ID, public_base_url=PUBLIC
    )

    assert layer["url"] == f"{PUBLIC}/files/buildings/{RESULT_ID}"
    assert layer["role"] == "result"
    assert layer["download_url"] is None
    assert layer["mime_type"] == "application/geo+json"


def test_build_stored_layer_marks_the_uploaded_blocks_as_input():
    layer = build_stored_layer(
        slot=SLOT_BLOCKS_INPUT, result_id=RESULT_ID, public_base_url=PUBLIC
    )

    assert layer["role"] == "input"
    assert layer["filename"] == "blocks_input.geojson"


def test_build_zones_layer_encodes_scenario_coordinates():
    layer = build_zones_layer(
        scenario_id=198,
        year=2024,
        source="OSM",
        functional_zone_types=["residential", "business"],
        public_base_url=PUBLIC,
    )

    parsed = urlparse(layer["url"])
    assert parsed.path == "/layers/functional_zones"
    assert parse_qs(parsed.query) == {
        "scenario_id": ["198"],
        "year": ["2024"],
        "source": ["OSM"],
        "functional_zone_types": ["residential", "business"],
    }


def test_build_zones_layer_never_offers_a_direct_download():
    layer = build_zones_layer(
        scenario_id=198,
        year=2024,
        source="OSM",
        functional_zone_types=["residential"],
        public_base_url=PUBLIC,
    )

    assert layer["download_url"] is None
    assert layer["role"] == "input"


def test_file_part_keeps_the_durable_url():
    layer = build_stored_layer(
        slot=SLOT_BUILDINGS, result_id=RESULT_ID, public_base_url=PUBLIC
    )

    part = geo_layer_to_file_part(layer)

    assert part["url"] == layer["url"]
    assert part["name"] == "buildings"
    assert part["source_service"] == "genbuilder"


def test_file_part_never_persists_an_ephemeral_download_url():
    """The invariant the whole design rests on: history holds durable links only."""
    layer = build_stored_layer(
        slot=SLOT_BUILDINGS, result_id=RESULT_ID, public_base_url=PUBLIC
    )
    layer["download_url"] = "http://10.32.1.42:9000/genbuilder/x?X-Amz-Signature=deadbeef"

    part = geo_layer_to_file_part(layer)

    assert "download_url" not in part
    assert "X-Amz-Signature" not in repr(part)
