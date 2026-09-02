from app.logic.building_params import BuildingType
from app.schema.building_properties import (
    BUILDING_PROPERTIES_SCHEMA,
    BuildingPropertiesSchema,
)

GENERATED_BUILDING_PROPERTIES = {
    "floors_count",
    "living_area",
    "building_area",
    "service",
    "broke_restriction_zone",
    "building_type",
    "zone",
    "residents_number",
}

EXCLUDED_ONLY_PROPERTIES = {"is_excluded", "physical_object_id"}


def test_schema_covers_every_property_the_api_returns():
    assert set(BUILDING_PROPERTIES_SCHEMA.properties) == (
        GENERATED_BUILDING_PROPERTIES | EXCLUDED_ONLY_PROPERTIES
    )


def test_excluded_only_flag_marks_exactly_the_excluded_object_properties():
    flagged = {
        name
        for name, label in BUILDING_PROPERTIES_SCHEMA.properties.items()
        if label.excluded_only
    }

    assert flagged == EXCLUDED_ONLY_PROPERTIES


def test_every_building_type_has_a_label():
    labels = BUILDING_PROPERTIES_SCHEMA.values["building_type"]

    assert set(labels) == {building_type.value for building_type in BuildingType}


def test_enum_properties_have_a_value_dictionary():
    enum_properties = {
        name
        for name, label in BUILDING_PROPERTIES_SCHEMA.properties.items()
        if label.kind == "enum"
    }

    assert enum_properties == set(BUILDING_PROPERTIES_SCHEMA.values)


def test_only_measurable_properties_carry_a_unit():
    for name, label in BUILDING_PROPERTIES_SCHEMA.properties.items():
        if label.unit is not None:
            assert label.kind in {"number", "integer"}, name


def test_schema_round_trips_through_json():
    dumped = BUILDING_PROPERTIES_SCHEMA.model_dump()

    assert BuildingPropertiesSchema.model_validate(dumped) == BUILDING_PROPERTIES_SCHEMA
