"""Existing buildings uploaded in the project-less mode.

Two things have to hold: their footprints are cut out of the blocks, and they
come back shaped like any other excluded object (``is_excluded``), so the
frontend renders them with the same legend as scenario-mode exclusions.
"""
import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from app.logic.physical_objects_service import PhysicalObjectsService


@pytest.fixture
def service():
    return PhysicalObjectsService()


def _feature(geometry, **properties):
    return {"type": "Feature", "properties": properties, "geometry": geometry}


SQUARE = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]],
}


def test_uploaded_buildings_carry_the_excluded_object_contract(service):
    fc = {
        "type": "FeatureCollection",
        "features": [
            _feature(SQUARE, floors_count=5, living_area=3200, building_area=8000)
        ],
    }

    props = service.normalize_uploaded_features(fc)[0]["properties"]

    assert props["is_excluded"] is True
    assert props["floors_count"] == 5.0
    assert props["living_area"] == 3200.0
    assert props["building_area"] == 8000.0


def test_properties_are_optional(service):
    """A bare footprint is valid input — only the geometry is needed to exclude."""
    fc = {"type": "FeatureCollection", "features": [_feature(SQUARE)]}

    props = service.normalize_uploaded_features(fc)[0]["properties"]

    assert props["is_excluded"] is True
    assert props["living_area"] == 0.0
    assert props["physical_object_id"] is None
    assert props["service"] == []


def test_non_polygonal_features_are_skipped(service):
    fc = {
        "type": "FeatureCollection",
        "features": [
            _feature(SQUARE),
            _feature({"type": "Point", "coordinates": [1.0, 1.0]}),
            _feature({"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]}),
        ],
    }

    assert len(service.normalize_uploaded_features(fc)) == 1


def test_empty_input_is_not_an_error(service):
    assert service.normalize_uploaded_features(None) == []
    assert service.normalize_uploaded_features({"type": "FeatureCollection"}) == []


def test_footprints_are_cut_out_of_the_blocks(service):
    blocks = gpd.GeoDataFrame(
        {"zone": ["residential"]}, geometry=[box(0, 0, 100, 100)], crs="EPSG:32636"
    )
    buildings = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:32636"
    )

    out = service.exclude(blocks=blocks, physical_objects=buildings)

    assert out.geometry.iloc[0].area == pytest.approx(100 * 100 - 10 * 10)


def test_a_buffer_widens_the_cut(service):
    blocks = gpd.GeoDataFrame(
        {"zone": ["residential"]}, geometry=[box(0, 0, 100, 100)], crs="EPSG:32636"
    )
    buildings = gpd.GeoDataFrame(
        geometry=[box(40, 40, 50, 50)], crs="EPSG:32636"
    )

    plain = service.exclude(blocks=blocks, physical_objects=buildings)
    buffered = service.exclude(blocks=blocks, physical_objects=buildings, buffer_m=5.0)

    assert buffered.geometry.iloc[0].area < plain.geometry.iloc[0].area
