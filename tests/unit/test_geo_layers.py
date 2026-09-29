from urllib.parse import parse_qs, urlparse

import pytest

from app.logic.geo_layers import (
    FILE_SLOTS,
    SLOT_BLOCKS_INPUT,
    SLOT_BUILDINGS,
    SLOT_EXISTING_BUILDINGS,
    SLOT_ZONES,
    build_stored_layer,
    build_zones_layer,
    geo_layer_to_file_part,
    object_key,
)

RESULT_ID = "a1b2c3d4e5f6"


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
    assert FILE_SLOTS == (
        SLOT_BUILDINGS,
        SLOT_BLOCKS_INPUT,
        SLOT_EXISTING_BUILDINGS,
        SLOT_ZONES,
    )


def test_stored_zones_share_the_name_of_the_live_zones_layer():
    """Project-less mode stores the zones itself; the map still sees one zones layer."""
    layer = build_stored_layer(slot=SLOT_ZONES, result_id=RESULT_ID)

    assert layer["name"] == "functional_zones"
    assert layer["title"] == "Функциональные зоны"
    assert layer["role"] == "input"
    assert layer["url"] == f"/files/zones/{RESULT_ID}"
    assert object_key(RESULT_ID, SLOT_ZONES) == f"{RESULT_ID}/zones.geojson"


def test_build_stored_layer_points_at_the_file_endpoint():
    layer = build_stored_layer(slot=SLOT_BUILDINGS, result_id=RESULT_ID)

    assert layer["url"] == f"/files/buildings/{RESULT_ID}"
    assert layer["role"] == "result"
    assert layer["download_url"] is None
    assert layer["mime_type"] == "application/geo+json"


def test_build_stored_layer_marks_the_uploaded_blocks_as_input():
    layer = build_stored_layer(slot=SLOT_BLOCKS_INPUT, result_id=RESULT_ID)

    assert layer["role"] == "input"
    assert layer["filename"] == "blocks_input.geojson"


def test_build_zones_layer_encodes_scenario_coordinates():
    layer = build_zones_layer(
        scenario_id=198,
        year=2024,
        source="OSM",
        functional_zone_types=["residential", "business"],
    )

    parsed = urlparse(layer["url"])
    assert not parsed.scheme and not parsed.netloc
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
    )

    assert layer["download_url"] is None
    assert layer["role"] == "input"


def test_file_part_keeps_the_durable_url():
    layer = build_stored_layer(slot=SLOT_BUILDINGS, result_id=RESULT_ID)

    part = geo_layer_to_file_part(layer)

    assert part["url"] == layer["url"]
    assert part["name"] == "buildings"
    assert part["source_service"] == "genbuilder"


def test_file_part_never_persists_an_ephemeral_download_url():
    """The invariant the whole design rests on: history holds durable links only."""
    layer = build_stored_layer(slot=SLOT_BUILDINGS, result_id=RESULT_ID)
    layer["download_url"] = "http://10.32.1.42:9000/genbuilder/x?X-Amz-Signature=deadbeef"

    part = geo_layer_to_file_part(layer)

    assert "download_url" not in part
    assert "X-Amz-Signature" not in repr(part)
