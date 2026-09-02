import geopandas as gpd
import pytest
from pydantic import ValidationError

from app.schema.dto import BlockFeatureCollection, TerritoryRequest

POLYGON = {
    "type": "Polygon",
    "coordinates": [[[31.0, 59.91], [31.01, 59.91], [31.01, 59.92], [31.0, 59.92], [31.0, 59.91]]],
}

MULTIPOLYGON = {
    "type": "MultiPolygon",
    "coordinates": [
        [[[31.0, 59.91], [31.01, 59.91], [31.01, 59.92], [31.0, 59.92], [31.0, 59.91]]],
        [[[31.02, 59.91], [31.03, 59.91], [31.03, 59.92], [31.02, 59.92], [31.02, 59.91]]],
    ],
}


def _collection(*geometries, zone="residential"):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"block_id": i, "zone": zone}, "geometry": geometry}
            for i, geometry in enumerate(geometries, 1)
        ],
    }


@pytest.mark.parametrize(
    "geometry",
    [POLYGON, MULTIPOLYGON],
    ids=["polygon", "multipolygon"],
)
def test_territory_request_accepts_polygonal_geometry(geometry):
    payload = TerritoryRequest.model_validate({"blocks": _collection(geometry)})

    assert payload.blocks.features[0].geometry.type == geometry["type"]


def test_territory_request_accepts_mixed_geometry_types():
    payload = TerritoryRequest.model_validate({"blocks": _collection(POLYGON, MULTIPOLYGON)})

    assert [f.geometry.type for f in payload.blocks.features] == ["Polygon", "MultiPolygon"]


def test_territory_request_rejects_missing_geometry():
    blocks = _collection(POLYGON)
    blocks["features"][0]["geometry"] = None

    with pytest.raises(ValidationError, match="Polygon or MultiPolygon geometry"):
        TerritoryRequest.model_validate({"blocks": blocks})


def test_territory_request_rejects_non_polygonal_geometry():
    point = {"type": "Point", "coordinates": [31.0, 59.91]}

    with pytest.raises(ValidationError):
        TerritoryRequest.model_validate({"blocks": _collection(point)})


def test_multipolygon_blocks_survive_geodataframe_conversion():
    payload = TerritoryRequest.model_validate({"blocks": _collection(MULTIPOLYGON)})

    gdf = gpd.GeoDataFrame.from_features(payload.blocks.model_dump()["features"])

    assert list(gdf.geom_type) == ["MultiPolygon"]
    assert list(gdf["zone"]) == ["residential"]
    assert gdf.geometry.iloc[0].is_valid


def test_blocks_file_and_territory_request_accept_the_same_collection():
    """Chat mode (blocks_file) and /generate/by_territory must not diverge."""
    blocks = _collection(POLYGON, MULTIPOLYGON)

    BlockFeatureCollection.model_validate(blocks)
    TerritoryRequest.model_validate({"blocks": blocks})


BUILDING = {
    "type": "Polygon",
    "coordinates": [[[31.001, 59.911], [31.002, 59.911], [31.002, 59.912], [31.001, 59.911]]],
}


def _buildings(*geometries, **properties):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": dict(properties), "geometry": geometry}
            for geometry in geometries
        ],
    }


def test_existing_buildings_are_optional():
    payload = TerritoryRequest.model_validate({"blocks": _collection(POLYGON)})

    assert payload.existing_buildings is None


def test_existing_buildings_accept_free_form_properties():
    payload = TerritoryRequest.model_validate(
        {
            "blocks": _collection(POLYGON),
            "existing_buildings": _buildings(BUILDING, floors_count=5, whatever="kept"),
        }
    )

    properties = payload.existing_buildings.features[0].properties
    assert properties == {"floors_count": 5, "whatever": "kept"}


def test_existing_buildings_accept_a_bare_footprint():
    """Only the geometry is needed to exclude a building from generation."""
    collection = _buildings(BUILDING)
    collection["features"][0]["properties"] = None

    payload = TerritoryRequest.model_validate(
        {"blocks": _collection(POLYGON), "existing_buildings": collection}
    )

    assert payload.existing_buildings.features[0].geometry.type == "Polygon"


def test_existing_buildings_reject_non_polygonal_geometry():
    point = {"type": "Point", "coordinates": [31.0, 59.91]}

    with pytest.raises(ValidationError):
        TerritoryRequest.model_validate(
            {"blocks": _collection(POLYGON), "existing_buildings": _buildings(point)}
        )
