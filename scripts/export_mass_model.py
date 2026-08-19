"""Export generated buildings to Facades-3D mass models and optionally benchmark them.

Usage::

    python scripts/export_mass_model.py --input buildings.geojson --outdir out
    python scripts/export_mass_model.py --input buildings.geojson --outdir out \
        --send-to http://facades-host:8000 --cluster-count 12

Every zone is exported in the same local metric frame, so the resulting GLB
scenes can be merged without realignment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx

from app.logic.mass_model import (
    DEFAULT_FLOOR_HEIGHT_M,
    LocalFrame,
    MassModelParams,
    MassModelStats,
    build_local_frame,
    buildings_to_obj,
    group_features_by_zone,
)

ZONE_PROMPTS = {
    "residential": "modern residential building, apartment windows, balconies",
    "business": "modern office building, glass curtain wall, commercial architecture",
    "industrial": "industrial warehouse, metal wall panels, factory building",
    "transport": "transport infrastructure building, concrete panels",
    "special": "public institution building, brick walls, plain windows",
    "recreation": "low-rise pavilion, wood and glass",
    "agriculture": "farm building, metal siding",
}
DEFAULT_PROMPT = "modern style architecture, urban house"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="GeoJSON FeatureCollection of buildings")
    parser.add_argument("--outdir", type=Path, default=Path("mass_models"), help="Output directory")
    parser.add_argument("--floor-height", type=float, default=DEFAULT_FLOOR_HEIGHT_M, help="Floor height in metres")
    parser.add_argument("--single", action="store_true", help="Export one OBJ instead of one per zone")
    parser.add_argument("--send-to", type=str, default=None, help="Facades-3D base URL, e.g. http://host:8000")
    parser.add_argument("--cluster-count", type=int, default=12, help="Representative wall models per request")
    parser.add_argument("--pixels-per-meter", type=int, default=32, help="Facade image resolution")
    parser.add_argument("--timeout", type=float, default=7200.0, help="Generation request timeout in seconds")
    return parser.parse_args()


def load_feature_collection(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise SystemExit(f"{path} is not a GeoJSON FeatureCollection")
    return payload


def export(
    collections: dict[str, dict[str, Any]],
    frame: LocalFrame,
    params: MassModelParams,
    outdir: Path,
) -> list[tuple[str, Path, MassModelStats]]:
    outdir.mkdir(parents=True, exist_ok=True)
    exported: list[tuple[str, Path, MassModelStats]] = []

    for zone, collection in sorted(collections.items()):
        try:
            obj_text, stats = buildings_to_obj(collection, frame, params)
        except ValueError as error:
            print(f"[{zone}] skipped: {error}")
            continue

        path = outdir / f"{zone}.obj"
        path.write_text(obj_text, encoding="utf-8")
        exported.append((zone, path, stats))

    return exported


def report(exported: list[tuple[str, Path, MassModelStats]]) -> None:
    print(f"\n{'zone':<14}{'buildings':>10}{'walls':>8}{'sizes':>7}{'skipped':>9}{'MB':>7}")
    for zone, path, stats in exported:
        size_mb = path.stat().st_size / 1024 / 1024
        print(
            f"{zone:<14}{stats.buildings:>10}{stats.walls:>8}"
            f"{stats.unique_wall_sizes:>7}{stats.skipped_features:>9}{size_mb:>7.2f}"
        )
        if stats.fallback_height_features:
            print(f"  {stats.fallback_height_features} feature(s) had no floors_count, used fallback height")
        if stats.roofs_covering_holes:
            print(f"  {stats.roofs_covering_holes} roof(s) cover an inner courtyard (OBJ faces have no holes)")
    print(
        "\nunique wall sizes is the upper bound for a useful --cluster-count: "
        "clustering beyond it generates duplicates."
    )


def generate(
    base_url: str,
    zone: str,
    obj_path: Path,
    cluster_count: int,
    pixels_per_meter: int,
    timeout: float,
) -> None:
    prompt = ZONE_PROMPTS.get(zone, DEFAULT_PROMPT)
    output_path = obj_path.with_suffix(".glb")

    print(f"\n[{zone}] POST {base_url}/generate  cluster_count={cluster_count} prompt={prompt!r}")
    started = time.perf_counter()
    with obj_path.open("rb") as model:
        response = httpx.post(
            f"{base_url.rstrip('/')}/generate",
            files={"input_model": (obj_path.name, model, "model/obj")},
            data={
                "pixels_per_meter": str(pixels_per_meter),
                "cluster_count": str(cluster_count),
                "prompt": prompt,
                "output_filename": output_path.name,
            },
            timeout=timeout,
        )
    elapsed = time.perf_counter() - started

    if response.status_code != 200:
        print(f"[{zone}] failed in {elapsed:.1f}s: HTTP {response.status_code} {response.text[:500]}")
        return

    output_path.write_bytes(response.content)
    size_mb = len(response.content) / 1024 / 1024
    print(f"[{zone}] done in {elapsed:.1f}s -> {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    args = parse_args()
    feature_collection = load_feature_collection(args.input)
    frame = build_local_frame(feature_collection)
    params = MassModelParams(floor_height_m=args.floor_height)

    collections = (
        {"all": feature_collection}
        if args.single
        else group_features_by_zone(feature_collection)
    )

    print(f"frame: {frame.crs} origin=({frame.origin_x:.1f}, {frame.origin_y:.1f})")
    exported = export(collections, frame, params, args.outdir)
    if not exported:
        raise SystemExit("Nothing was exported")
    report(exported)

    if args.send_to:
        for zone, path, _ in exported:
            generate(args.send_to, zone, path, args.cluster_count, args.pixels_per_meter, args.timeout)


if __name__ == "__main__":
    main()
