import asyncio

import pytest
from facade_library_support import (
    collection,
    make_library,
    make_template,
    rect_feature,
    seed_library,
)
from fastapi import HTTPException

from app.infrastructure.object_storage import LocalStorage
from app.logic import generation_orchestration as orchestration
from app.logic.facade_library.results import scene_glb_key
from app.settings import Settings

QUEUED = {
    "status": "queued",
    "job_id": "job-1",
    "status_url": "http://facade-jobs/jobs/job-1",
    "facade_style": "Кирпичный",
}


class _Harness:
    def __init__(
        self, monkeypatch, tmp_path, *, facade_jobs: bool, seeded: bool = True
    ):
        self.storage = LocalStorage(str(tmp_path))
        if seeded:
            seed_library(
                self.storage,
                [make_template("brick", 6, 12.0), make_template("glass", 6, 12.0)],
            )
        self.submitted: list[dict] = []
        library = make_library(self.storage)

        async def fake_submit(buildings, **kwargs):
            self.submitted.append(kwargs)
            return QUEUED

        monkeypatch.setattr(orchestration, "get_facade_library", lambda: library)
        monkeypatch.setattr(orchestration, "get_object_storage", lambda: self.storage)
        monkeypatch.setattr(
            orchestration, "facade_jobs_configured", lambda: facade_jobs
        )
        monkeypatch.setattr(orchestration, "submit_facade_job", fake_submit)
        monkeypatch.setattr(orchestration, "get_settings", lambda: Settings())

    def produce(self, buildings, *, facade_style, facade_source):
        return asyncio.run(
            orchestration.produce_facade_scene(
                buildings,
                requested_by="user-1",
                facade_style=facade_style,
                facade_source=facade_source,
            )
        )


def _six_floors(zone="residential"):
    return collection(rect_feature("1", zone=zone, floors=6))


def _twelve_floors():
    return collection(rect_feature("1", zone="residential", floors=12))


def test_library_source_returns_a_stored_ready_scene(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, facade_jobs=False)

    result = harness.produce(
        _six_floors(), facade_style="кирпичный", facade_source="library"
    )

    assert result["status"] == "ready"
    assert result["facade_style"] == "Кирпичный"
    assert result["style_by_zone"] == {"residential": "brick"}
    assert harness.storage.get_bytes(scene_glb_key(result["result_id"])).startswith(
        b"glTF"
    )
    assert harness.submitted == []


def test_omitted_style_uses_the_zone_default_preset(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, facade_jobs=False)

    result = harness.produce(
        _six_floors("business"), facade_style=None, facade_source="library"
    )

    assert result["style_by_zone"] == {"business": "glass"}


def test_library_source_borrows_the_nearest_section_on_a_miss(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, facade_jobs=True)

    result = harness.produce(
        _twelve_floors(), facade_style="кирпичный", facade_source="library"
    )

    assert result["status"] == "ready"
    assert result["stats"]["nearest_substitutions"] > 0
    assert harness.submitted == []


def test_library_then_gpu_queues_the_whole_request_on_a_miss(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, facade_jobs=True)

    result = harness.produce(
        _twelve_floors(), facade_style="кирпичный", facade_source="library_then_gpu"
    )

    assert result == QUEUED
    assert harness.submitted[0]["facade_style"] == "кирпичный"


def test_library_then_gpu_answers_from_the_library_when_everything_is_cached(
    monkeypatch, tmp_path
):
    harness = _Harness(monkeypatch, tmp_path, facade_jobs=True)

    result = harness.produce(
        _six_floors(), facade_style="кирпичный", facade_source="library_then_gpu"
    )

    assert result["status"] == "ready"
    assert harness.submitted == []


def test_gpu_source_always_queues(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, facade_jobs=True)

    result = harness.produce(
        _six_floors(), facade_style="кирпичный", facade_source="gpu"
    )

    assert result == QUEUED


def test_free_text_style_queues_when_facade_jobs_is_configured(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, facade_jobs=True)

    result = harness.produce(
        _six_floors(), facade_style="warm sandstone facade", facade_source="library"
    )

    assert result == QUEUED


def test_free_text_style_is_rejected_without_facade_jobs(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, facade_jobs=False)

    with pytest.raises(HTTPException) as excinfo:
        harness.produce(
            _six_floors(), facade_style="warm sandstone facade", facade_source="library"
        )

    assert excinfo.value.status_code == 422


def test_unavailable_library_is_503_in_library_mode(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, facade_jobs=True, seeded=False)

    with pytest.raises(HTTPException) as excinfo:
        harness.produce(
            _six_floors(), facade_style="кирпичный", facade_source="library"
        )

    assert excinfo.value.status_code == 503
    assert harness.submitted == []


def test_unavailable_library_falls_back_to_gpu_in_library_then_gpu_mode(
    monkeypatch, tmp_path
):
    harness = _Harness(monkeypatch, tmp_path, facade_jobs=True, seeded=False)

    result = harness.produce(
        _six_floors(), facade_style="кирпичный", facade_source="library_then_gpu"
    )

    assert result == QUEUED


def test_library_mode_does_not_require_facade_jobs_up_front(monkeypatch):
    monkeypatch.setattr(orchestration, "facade_jobs_configured", lambda: False)

    orchestration._require_facade_backend("library")
    with pytest.raises(HTTPException) as excinfo:
        orchestration._require_facade_backend("gpu")

    assert excinfo.value.status_code == 503


def test_omitted_facade_source_uses_the_server_default(monkeypatch):
    monkeypatch.setattr(
        orchestration,
        "get_settings",
        lambda: Settings(FACADE_SOURCE_DEFAULT="library_then_gpu"),
    )

    assert orchestration._effective_facade_source(None) == "library_then_gpu"
    assert orchestration._effective_facade_source("gpu") == "gpu"
