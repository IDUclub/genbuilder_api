import importlib.util
import json
from pathlib import Path

import pytest
from facade_library_support import PREFIX, make_template, seed_library

from app.infrastructure.object_storage import LocalStorage

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_facade_library.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("build_facade_library", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _index(root: Path) -> dict:
    return json.loads(
        (root / PREFIX / "previews" / "index.json").read_text(encoding="utf-8")
    )


def test_previews_are_built_only_for_floor_groups_that_have_sections(script, tmp_path):
    seed_library(LocalStorage(str(tmp_path)), [make_template("brick", 6, 12.0)])

    script.main(
        ["previews", "--styles", "brick", "glass", "--local-root", str(tmp_path)]
    )

    previews = _index(tmp_path)["previews"]
    assert [(item["style_id"], item["floor_group"]) for item in previews] == [
        ("brick", "medium")
    ]
    assert (tmp_path / PREFIX / "previews" / "brick" / "medium.glb").read_bytes()[
        :4
    ] == b"glTF"


def test_rebuilding_one_style_keeps_the_previews_of_the_others(script, tmp_path):
    seed_library(
        LocalStorage(str(tmp_path)),
        [make_template("brick", 6, 12.0), make_template("glass", 6, 12.0)],
    )
    script.main(["previews", "--local-root", str(tmp_path)])

    script.main(["previews", "--styles", "glass", "--local-root", str(tmp_path)])

    assert {item["style_id"] for item in _index(tmp_path)["previews"]} == {
        "brick",
        "glass",
    }


def test_dry_run_prewarm_writes_nothing(script, tmp_path, capsys):
    script.main(
        [
            "prewarm",
            "--dry-run",
            "--styles",
            "brick",
            "--widths",
            "12",
            "--local-root",
            str(tmp_path),
        ]
    )

    assert "would generate brick/medium/12m" in capsys.readouterr().out
    assert not any(tmp_path.iterdir())


def test_check_probes_both_library_and_scene_folders(script, tmp_path, capsys):
    script.main(["check", "--local-root", str(tmp_path)])

    output = capsys.readouterr().out
    assert f"under {PREFIX}/" in output
    assert "under generated/3d/" in output


def test_check_keeps_probing_when_delete_is_denied(script, tmp_path, capsys):
    storage = LocalStorage(str(tmp_path))

    def denied_delete(key):
        raise script.ObjectStorageError(f"Object storage failed to delete {key}")

    storage.delete = denied_delete

    script.run_check(storage, PREFIX)

    output = capsys.readouterr().out
    assert f"ok: write/read under {PREFIX}/" in output
    assert "ok: write/read under generated/3d/" in output
    assert output.count("warning: cannot delete") == 2


def test_library_keys_match_the_facade_jobs_layout(script):
    assert script.size_key(12.0, 18.0) == "w1200-h1800"
    assert script.template_seed("art-nouveau", 0, 0) == script.SEED_BASE
