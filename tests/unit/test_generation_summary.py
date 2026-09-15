"""``build_generation_summary`` gives MCP callers totals and target deficits
without walking the FeatureCollection themselves."""
from app.logic.generation_summary import build_generation_summary, combine_targets


def _feature(**props):
    return {"type": "Feature", "properties": props, "geometry": None}


FEATURES = [
    _feature(zone="residential", residents_number=400, living_area=8000.0),
    _feature(zone="residential", residents_number=300, living_area=6000.0),
    _feature(zone="business", functional_area=6000.0),
    _feature(is_excluded=True, residents_number=120, living_area=2400.0),
]


def test_summary_splits_generated_and_excluded():
    summary = build_generation_summary(FEATURES)

    assert summary["buildings"] == 3
    assert summary["residents_total"] == 700
    assert summary["living_area_total"] == 14000.0
    assert summary["buildings_by_zone"] == {"residential": 2, "business": 1}
    assert summary["residents_by_zone"] == {"residential": 700, "business": 0}
    assert summary["excluded_buildings"] == 1
    assert summary["existing_residents"] == 120
    assert summary["existing_living_area"] == 2400.0
    assert summary["targets"] == {}
    assert "targets_source" not in summary
    assert "existing_buildings_preserved" not in summary


def test_summary_reports_target_deficits():
    summary = build_generation_summary(
        FEATURES,
        targets_by_zone={
            "residents": {"residential": 1000, "business": 0},
            "coverage_area": {"business": 5000},
        },
        targets_source="request",
        existing_buildings_preserved=True,
    )

    assert summary["targets"] == {
        "residential": {"target_residents": 1000, "achieved_residents": 700, "residents_deficit": 300},
        "business": {
            "target_functional_area": 5000.0,
            "achieved_functional_area": 6000.0,
            "functional_area_deficit": 0.0,
        },
    }
    assert summary["targets_source"] == "request"
    assert summary["existing_buildings_preserved"] is True


def test_combine_targets_sums_per_zone():
    combined = combine_targets(
        [
            {"residents": {"residential": 1000}, "coverage_area": {"business": 500}},
            {"residents": {"residential": 250}},
            None,
        ]
    )

    assert combined == {"residents": {"residential": 1250.0}, "coverage_area": {"business": 500.0}}
