import geopandas as gpd
from shapely.geometry import box

from app.logic.block_generator import BlockGenerator
from app.logic.chat.param_extraction import build_extraction_schema, normalize_targets


def test_chat_extraction_keeps_explicit_building_count():
    zone_properties = build_extraction_schema()["properties"]["zones"]["items"][
        "properties"
    ]
    assert zone_properties["buildings_count"]["minimum"] == 1

    extracted = normalize_targets(
        {
            "zones": [
                {
                    "zone": "residential",
                    "residents": 80,
                    "floors_avg": 5,
                    "buildings_count": 1,
                }
            ]
        },
        la_per_person=18.0,
    )

    assert extracted.targets_by_zone["buildings_count"] == {"residential": 1}


def test_explicit_count_limits_blocks_and_generated_buildings():
    blocks = gpd.GeoDataFrame(
        {"name": ["small", "large", "medium"]},
        geometry=[box(0, 0, 10, 10), box(0, 0, 30, 30), box(0, 0, 20, 20)],
        crs="EPSG:3857",
    )

    limited = BlockGenerator._limit_candidate_blocks(blocks, 1)

    assert limited["name"].tolist() == ["large"]

    buildings = gpd.GeoDataFrame(
        {"living_area": [900.0, 1420.0, 2400.0]},
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1)],
        crs="EPSG:3857",
    )

    capped = BlockGenerator._cap_generated_buildings(buildings, 1, 1440.0)

    assert capped["living_area"].tolist() == [1420.0]
