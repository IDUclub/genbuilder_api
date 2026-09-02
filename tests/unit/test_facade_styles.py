import asyncio

from app.logic import generation_orchestration as orchestration
from app.logic.chat.param_extraction import build_extraction_schema, normalize_targets
from app.logic.facade_styles import (
    DEFAULT_FACADE_STYLE_NAME_RU,
    build_style_by_zone,
    resolve_facade_style,
)


BUILDINGS = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"zone": "business"}},
        {"type": "Feature", "properties": {"zone": "residential"}},
        {"type": "Feature", "properties": {"zone": "residential"}},
    ],
}


def test_omitted_style_uses_facade_jobs_zone_defaults():
    style = resolve_facade_style(None)

    assert style.name_ru == DEFAULT_FACADE_STYLE_NAME_RU
    assert style.prompt is None
    assert build_style_by_zone(BUILDINGS, style) == {}


def test_russian_and_english_aliases_resolve_to_same_preset():
    russian = resolve_facade_style("кирпичный")
    english = resolve_facade_style("BRICK")

    assert russian == english
    assert russian.name_ru == "Кирпичный"
    assert russian.source == "preset"


def test_style_override_is_built_only_for_present_zones():
    style = resolve_facade_style("стеклянный")

    overrides = build_style_by_zone(BUILDINGS, style)

    assert set(overrides) == {"business", "residential"}
    assert "office and commercial" in overrides["business"]["prompt"]
    assert "glass curtain wall" in overrides["business"]["prompt"]
    assert "residential apartment" in overrides["residential"]["prompt"]


def test_free_text_keeps_translated_chat_prompt_and_russian_label():
    style = resolve_facade_style(
        "warm sandstone facade with arched windows",
        name_ru="Тёплый песчаник с арочными окнами",
    )

    assert style.name_ru == "Тёплый песчаник с арочными окнами"
    assert style.prompt == "warm sandstone facade with arched windows"
    assert style.source == "free_text"


def test_chat_extraction_schema_and_normalization_include_facade_style():
    properties = build_extraction_schema()["properties"]
    assert "facade_style_name_ru" in properties
    assert "facade_style_prompt" in properties

    extracted = normalize_targets(
        {
            "zones": [],
            "facade_style_name_ru": " Кирпичный ",
            "facade_style_prompt": " exposed brick facade ",
        },
        la_per_person=30.0,
    )
    assert extracted.facade_style_name_ru == "Кирпичный"
    assert extracted.facade_style_prompt == "exposed brick facade"


def test_submit_facade_job_passes_resolved_styles_and_returns_russian_name(
    monkeypatch,
):
    captured: dict = {}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return None

        async def submit_job(self, buildings, **kwargs):
            captured.update(kwargs)
            return {"job_id": "job-1", "status_url": "http://facades/jobs/job-1"}

    monkeypatch.setattr(orchestration, "facade_jobs_configured", lambda: True)
    monkeypatch.setattr(orchestration, "build_facade_jobs_client", FakeClient)

    result = asyncio.run(
        orchestration.submit_facade_job(
            BUILDINGS,
            requested_by="user-1",
            facade_style="brick",
        )
    )

    assert result["facade_style"] == "Кирпичный"
    assert set(captured["style_by_zone"]) == {"business", "residential"}
    assert captured["requested_by"] == "user-1"
