import importlib.util
import json
from pathlib import Path

import pytest
from facade_library_support import PREFIX, make_section, seed_library

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


def test_every_floor_group_preview_is_stacked_from_the_style_section(script, tmp_path):
    seed_library(LocalStorage(str(tmp_path)), [make_section("brick", 12.0)])

    script.main(
        ["previews", "--styles", "brick", "glass", "--local-root", str(tmp_path)]
    )

    previews = _index(tmp_path)["previews"]
    assert [(item["style_id"], item["floor_group"]) for item in previews] == [
        ("brick", "low"),
        ("brick", "medium"),
        ("brick", "high"),
    ]
    assert (tmp_path / PREFIX / "previews" / "brick" / "high.glb").read_bytes()[
        :4
    ] == b"glTF"


def test_rebuilding_one_style_keeps_the_previews_of_the_others(script, tmp_path):
    seed_library(
        LocalStorage(str(tmp_path)),
        [make_section("brick", 12.0), make_section("glass", 12.0)],
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

    output = capsys.readouterr().out
    assert "would generate brick/12m/v0 (12x32 m)" in output
    assert "would generate brick/12m/v2 (12x32 m)" in output
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


def test_section_keys_and_seeds_are_stable(script):
    assert script.size_key(12.0, 32.0) == "w1200-h3200"
    assert script.section_seed("art-nouveau", 0, 0) == script.SEED_BASE
    assert script.section_seed("brick", 2, 0) == script.SEED_BASE + 102
    assert script.section_seed("brick", 2, 1) == script.SEED_BASE + 112


def test_variant_zero_keeps_the_original_key(script):
    assert (
        script.section_object_key(PREFIX, "brick", 9.0, 32.0, 0)
        == f"{PREFIX}/styles/brick/w0900-h3200/wall.glb"
    )
    assert (
        script.section_object_key(PREFIX, "brick", 9.0, 32.0, 2)
        == f"{PREFIX}/styles/brick/w0900-h3200/v2/wall.glb"
    )


def test_planned_section_orders_eight_four_metre_floors(script):
    args = script.parse_args(["prewarm", "--local-root", "unused"])

    section = script._planned_section(
        args, style_id="brick", width_m=9.0, variant=1, seed=1
    )

    assert (section.section_floors, section.height_m) == (8, 32.0)
    assert section.variant == 1
    assert section.object_key == f"{PREFIX}/styles/brick/w0900-h3200/v1/wall.glb"


def test_prewarm_skips_variants_already_in_the_manifest(script, tmp_path, capsys):
    seed_library(LocalStorage(str(tmp_path)), [make_section("brick", 12.0)])

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

    output = capsys.readouterr().out
    assert "skip brick/12m/v0: already present" in output
    assert "would generate brick/12m/v1 (12x32 m)" in output
    assert "would generate brick/12m/v2 (12x32 m)" in output


@pytest.mark.parametrize("variants", ["0", "10"])
def test_variant_count_out_of_range_is_refused(script, variants):
    with pytest.raises(SystemExit):
        script.parse_args(["prewarm", "--variants", variants])


def test_previews_are_built_from_variant_zero(script):
    args = script.parse_args(["previews", "--local-root", "unused"])
    sections = [make_section("brick", 12.0, variant=n) for n in (2, 1, 0)]

    chosen = script.pick_preview_section(sections, style_id="brick", args=args)

    assert chosen is not None and chosen.variant == 0


def test_sections_shorter_than_three_floors_are_refused(script):
    with pytest.raises(SystemExit):
        script.parse_args(["prewarm", "--section-floors", "2"])
