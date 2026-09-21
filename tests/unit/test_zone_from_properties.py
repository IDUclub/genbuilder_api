import asyncio

import pytest

from app.infrastructure.vllm_chat_client import VLLMChatError
from app.logic.chat.generation_chat import _blocks_from_upload
from app.logic.chat.zone_detection import ZONE_TYPES, detect_zones, resolve_zone_value
from app.logic.zone_taxonomy import zone_from_properties
from app.schema.dto import BlockFeatureCollection

GEOMETRY = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}


def _fc(*props):
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": p, "geometry": GEOMETRY} for p in props],
    }


class _LLM:
    """Scripted LLM: answers by schema shape, records every call."""

    def __init__(self, attribute=None, mapping=None, exc=None):
        self.attribute = attribute
        self.mapping = mapping or {}
        self.exc = exc
        self.calls = []

    async def complete_json(self, messages, *, schema, model=None, temperature=0.0):
        self.calls.append(schema)
        if self.exc is not None:
            raise self.exc
        if "attribute" in schema["properties"]:
            return {"attribute": self.attribute}
        return {"mapping": [{"value": v, "zone": z} for v, z in self.mapping.items()]}


def _run(coro):
    return asyncio.run(coro)


def test_zone_wins_over_export_fields():
    assert zone_from_properties({"zone": "business", "functional_zone_type_name": "residential"}) == "business"


def test_prostor_export_fields_are_used_as_fallback():
    assert zone_from_properties({"functional_zone_type_name": "mixed_use"}) == "mixed_use"
    assert zone_from_properties({"functional_zone_type": {"name": "residential"}}) == "residential"
    assert zone_from_properties({"zone": None, "functional_zone_type_name": "residential"}) == "residential"
    assert zone_from_properties({"zone_name": "Жилая зона"}) is None
    assert zone_from_properties(None) is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("residential", "residential"),
        ("Residential_LowRise", "residential_lowrise"),
        ("residental", "residential"),
        ("bussiness", "business"),
        ("mixed-use", "mixed_use"),
        ("Жилая зона", "residential"),
        ("Зона застройки индивидуальными жилыми домами", "residential_individual"),
        ("Зона многоэтажной жилой застройки", "residential_multistorey"),
        ("Общественно-деловая зона", "business"),
        ("Многофункциональная зона", "mixed_use"),
        ("Рекреационная зона", "recreation"),
        ("Зона специального назначения", "special"),
    ],
)
def test_known_spellings_resolve_to_urban_db_types(raw, expected):
    assert resolve_zone_value(raw) == expected


@pytest.mark.parametrize(
    "raw", ["qwerty", "resi", "Зона жилая и деловая", "нежилая", "15", 15, None, "", "Квартал 5"]
)
def test_garbage_and_ambiguous_values_are_never_guessed(raw):
    assert resolve_zone_value(raw) is None


def test_prostor_export_is_resolved_by_rules_without_llm():
    llm = _LLM()
    geojson = _fc(
        {"functional_zone_id": 1, "zone_code": 1, "functional_zone_type_name": "residential"},
        {"functional_zone_id": 2, "zone_code": 15, "functional_zone_type_name": "mixed_use"},
        {"functional_zone_id": 3, "zone_code": 3, "functional_zone_type_name": "special"},
    )

    kept, zones, events = _run(_blocks_from_upload(geojson, llm))

    assert llm.calls == []
    assert zones == ("residential", "business")
    assert [f["properties"]["zone"] for f in kept] == ["residential", "mixed_use"]
    assert [e.get("code") for e in events if e["type"] == "warning"] == ["zones_out_of_scope"]
    blocks = BlockFeatureCollection.model_validate({"type": "FeatureCollection", "features": kept})
    assert [f.properties.zone for f in blocks.features] == ["residential", "mixed_use"]


def test_llm_picks_attribute_the_rules_cannot_and_maps_its_values():
    llm = _LLM(attribute="vid", mapping={"Ж1": "residential", "ОД": "business"})
    geojson = _fc({"id": "a1", "vid": "Ж1"}, {"id": "a2", "vid": "ОД"})

    kept, zones, events = _run(_blocks_from_upload(geojson, llm))

    attribute_schema = llm.calls[0]["properties"]["attribute"]
    assert set(attribute_schema["enum"]) == {"id", "vid", None}
    assert zones == ("residential", "business")
    assert [f["properties"]["zone"] for f in kept] == ["residential", "business"]
    statuses = [e["content"] for e in events if e["type"] == "status"]
    assert "«vid» (определён моделью)" in statuses[0]
    assert "«Ж1» → residential" in statuses[1]


def test_unknown_attribute_resolved_by_rules_skips_llm():
    llm = _LLM(attribute="vid")
    geojson = _fc({"id": "a1", "vid": "Жилая зона"}, {"id": "a2", "vid": "Рекреационная зона"})
    geojson["features"].append(
        {"type": "Feature", "properties": {"id": "a3", "vid": "Общественно-деловая"}, "geometry": GEOMETRY}
    )

    detection = _run(detect_zones(geojson, llm))

    # Rules already cover "vid" (>=50%), so the LLM is not even asked.
    assert detection.attribute == "vid" and not detection.attribute_by_llm
    assert llm.calls == []


def test_llm_attribute_answer_outside_candidates_is_rejected():
    llm = _LLM(attribute="made_up_field")
    geojson = _fc({"code": "Ж-1", "name": "Квартал 1"}, {"code": "ОД-2", "name": "Квартал 2"})

    kept, _, events = _run(_blocks_from_upload(geojson, llm))

    assert kept == []
    error = [e for e in events if e["type"] == "error"][0]
    assert error["code"] == "zone_attribute_not_found"
    assert "code, name" in error["message"]


def test_llm_can_only_map_misspelled_values_to_known_types():
    llm = _LLM(mapping={"Жлая зна": "residential", "абырвалг": "none"})
    geojson = _fc(
        {"zone": "residential"},
        {"zone": "residential"},
        {"zone": "Жлая зна"},
        {"zone": "абырвалг"},
    )

    kept, zones, events = _run(_blocks_from_upload(geojson, llm))

    value_schema = llm.calls[0]["properties"]["mapping"]["items"]["properties"]
    assert value_schema["zone"]["enum"] == [*ZONE_TYPES, "none"]
    assert set(value_schema["value"]["enum"]) == {"Жлая зна", "абырвалг"}
    assert [f["properties"]["zone"] for f in kept] == ["residential"] * 3
    assert zones == ("residential",)
    warning = [e for e in events if e.get("code") == "zone_values_unrecognized"][0]
    assert "«абырвалг» — 1" in warning["message"]


def test_llm_answer_outside_enum_is_dropped():
    llm = _LLM(mapping={"Жлая зна": "super_residential"})
    detection = _run(detect_zones(_fc({"zone": "residential"}, {"zone": "Жлая зна"}), llm))

    assert detection.mapping == {"residential": "residential"}
    assert detection.unrecognized == {"Жлая зна": 1}


def test_llm_outage_keeps_rule_results_and_reports_unrecognized():
    llm = _LLM(exc=VLLMChatError(503, "vllm down"))
    detection = _run(detect_zones(_fc({"zone": "residential"}, {"zone": "Жлая зна"}), llm))

    assert [f["properties"]["zone"] for f in detection.features] == ["residential"]
    assert detection.unrecognized == {"Жлая зна": 1}
    assert "vllm down" in detection.llm_error


def test_llm_outage_without_zone_attribute_is_explained():
    llm = _LLM(exc=VLLMChatError(503, "vllm down"))

    kept, _, events = _run(_blocks_from_upload(_fc({"kind": "X"}, {"kind": "Y"}), llm))

    error = [e for e in events if e["type"] == "error"][0]
    assert kept == [] and error["code"] == "zone_attribute_not_found"
    assert "недоступен" in error["message"]


def test_file_without_generated_zones_lists_what_was_found():
    kept, _, events = _run(
        _blocks_from_upload(_fc({"zone": "recreation"}, {"zone": "Зона специального назначения"}), _LLM())
    )

    error = [e for e in events if e["type"] == "error"][0]
    assert kept == [] and error["code"] == "no_generated_zones"
    assert "рекреационная (recreation) — 1" in error["message"]
    assert "специального назначения (special) — 1" in error["message"]


def test_empty_file_is_explained():
    _, _, events = _run(_blocks_from_upload({"type": "FeatureCollection", "features": []}, _LLM()))

    assert events[0]["code"] == "blocks_empty" and events[0]["message"]
