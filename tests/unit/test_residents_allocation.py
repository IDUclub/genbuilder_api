import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from app.logic.block_generator import BlockGenerator
from app.logic.chat.generation_chat import _zone_areas
from app.logic.chat.param_extraction import normalize_targets, split_total_residents

LA_PER_PERSON = 18.0


def _blocks(*widths):
    """Square-ish blocks side by side, 100 m deep, in a metric CRS."""
    geoms, x = [], 0.0
    for w in widths:
        geoms.append(box(x, 0, x + w, 100))
        x += w + 10
    return gpd.GeoDataFrame(geometry=geoms, crs=32637)


def test_whole_zone_is_rounded_once_not_per_block():
    # 36 000 m² at 1 344 m² per building: 26.8 -> 27 buildings for the zone,
    # while ceil() per block would give one extra building in almost every block.
    blocks = _blocks(*[37] * 20)
    usable = pd.Series(1344.0, index=blocks.index)

    counts = BlockGenerator._split_buildings_by_area(blocks, 36_000, usable)

    assert counts.sum() == 27
    assert counts.max() - counts.min() <= 1
    targets = BlockGenerator._targets_from_counts(counts, usable)
    assert targets.sum() <= 27 * 1344 and targets.sum() > 36_000


def test_buildings_follow_block_area():
    blocks = _blocks(300, 100, 100)
    usable = pd.Series(1000.0, index=blocks.index)

    counts = BlockGenerator._split_buildings_by_area(blocks, 5_000, usable)

    assert counts.tolist() == [3, 1, 1]


def test_small_target_leaves_blocks_empty():
    blocks = _blocks(100, 100, 100, 100)
    usable = pd.Series(1000.0, index=blocks.index)

    counts = BlockGenerator._split_buildings_by_area(blocks, 1_500, usable)

    assert counts.sum() == 2
    assert (counts == 0).sum() == 2


def test_blocks_of_unknown_type_get_nothing():
    blocks = _blocks(100, 100)
    usable = pd.Series([0.0, 1000.0], index=blocks.index)

    counts = BlockGenerator._split_buildings_by_area(blocks, 2_500, usable)

    assert counts.tolist() == [0, 3]


def test_largest_remainder_keeps_the_total():
    need = pd.Series([0.4, 0.4, 0.2])
    areas = pd.Series([1.0, 2.0, 3.0])

    counts = BlockGenerator._largest_remainder(need, 1, areas)

    # Equal remainders: the larger block wins.
    assert counts.tolist() == [0, 1, 0]


def test_zone_less_total_is_not_copied_into_every_zone():
    raw = {
        "zones": [
            {"zone": "residential", "residents": 2000},
            {"zone": "business", "residents": 2000},
        ],
        "total_residents": 2000,
    }

    extracted = normalize_targets(raw, LA_PER_PERSON)

    assert extracted.total_residents == 2000
    assert "residents" not in extracted.targets_by_zone


def test_total_is_split_by_zone_area():
    extracted = normalize_targets({"zones": [], "total_residents": 2000}, LA_PER_PERSON)

    split = split_total_residents(
        extracted, ("residential", "business"), {"residential": 3.0, "business": 1.0}
    )

    assert split == {"residential": 1500, "business": 500}
    assert extracted.targets_by_zone["residents"] == split


def test_total_is_split_equally_without_areas():
    extracted = normalize_targets({"zones": [], "total_residents": 1001}, LA_PER_PERSON)

    split = split_total_residents(extracted, ("residential", "business"))

    assert sorted(split.values()) == [500, 501] and sum(split.values()) == 1001


def test_total_living_area_is_converted():
    extracted = normalize_targets({"zones": [], "total_living_area": 36_000}, LA_PER_PERSON)

    assert split_total_residents(extracted, ("residential",)) == {"residential": 2000}


def test_explicit_zone_keeps_its_value_and_takes_from_the_total():
    raw = {"zones": [{"zone": "residential", "residents": 1500}], "total_residents": 2000}
    extracted = normalize_targets(raw, LA_PER_PERSON)

    split = split_total_residents(extracted, ("residential", "business"), {"residential": 1.0})

    assert split == {"business": 500}
    assert extracted.targets_by_zone["residents"] == {"residential": 1500, "business": 500}


@pytest.mark.parametrize("raw", [{"zones": []}, {"zones": [], "total_residents": None}])
def test_no_total_changes_nothing(raw):
    extracted = normalize_targets(raw, LA_PER_PERSON)

    assert split_total_residents(extracted, ("residential", "business")) == {}
    assert extracted.targets_by_zone == {}


def test_zone_areas_group_subtypes():
    sq = lambda x0, s: {  # noqa: E731
        "type": "Polygon",
        "coordinates": [[[x0, 55], [x0 + s, 55], [x0 + s, 55 + s], [x0, 55 + s], [x0, 55]]],
    }
    features = [
        {"type": "Feature", "properties": {"zone": "residential_lowrise"}, "geometry": sq(37.0, 0.01)},
        {"type": "Feature", "properties": {"zone": "residential"}, "geometry": sq(37.1, 0.01)},
        {"type": "Feature", "properties": {"zone": "mixed_use"}, "geometry": sq(37.2, 0.01)},
    ]

    areas = _zone_areas(features)

    assert set(areas) == {"residential", "business"}
    assert areas["residential"] == pytest.approx(2 * areas["business"], rel=0.01)
