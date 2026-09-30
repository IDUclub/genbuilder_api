"""Fill the facade library in MinIO and build the style preview boxes.

Usage (on a machine that reaches both MinIO and the GPU host, e.g. over VPN)::

    python scripts/build_facade_library.py check
    python scripts/build_facade_library.py prewarm --styles brick glass
    python scripts/build_facade_library.py previews
    python scripts/build_facade_library.py all --dry-run

Storage comes from the same FILESERVER_* variables as the service. The local
HTTP(S)_PROXY is ignored for Facades-3D requests because it breaks the tunnel
to the GPU hosts.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import struct
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx

from app.infrastructure.object_storage import (
    LocalStorage,
    MinioStorage,
    ObjectNotFoundError,
    ObjectStorage,
    ObjectStorageError,
    get_object_storage,
)
from app.logic.facade_library.catalog import (
    FacadeLibraryUnavailable,
    FacadeTemplateLibrary,
    FacadeTemplateMiss,
    resolve_template,
)
from app.logic.facade_library.models import (
    PREVIEW_FLOOR_GROUPS,
    REPRESENTATIVE_FLOORS,
    FacadeLibraryManifest,
    FacadeTemplate,
    PreviewFloorGroup,
    StylePreview,
    StylePreviewIndex,
    canonical_style_key,
)
from app.logic.facade_library.previews import (
    PREVIEW_WIDTH_M,
    build_preview_glb,
    preview_etag,
)
from app.logic.facade_library.results import GLB_MIME_TYPE, SCENE_KEY_PREFIX
from app.logic.facade_styles import PRESETS_BY_ID, library_prompt

STYLE_IDS = sorted(PRESETS_BY_ID)
DEFAULT_WIDTHS = (6.0, 9.0, 12.0, 18.0, 24.0)
GENERATION_TIMEOUT_S = 2700
BUSY_RETRIES = 5
MAX_BACKOFF_S = 60
SEED_BASE = 8400


def canonical_wall_obj(width_m: float, height_m: float) -> bytes:
    # The -Z face normal makes Facades-3D's placement rotation equal to zero.
    return (
        "o facade_template\n"
        "v 0.0000 0.0000 0.0000\n"
        f"v 0.0000 {height_m:.4f} 0.0000\n"
        f"v {width_m:.4f} {height_m:.4f} 0.0000\n"
        f"v {width_m:.4f} 0.0000 0.0000\n"
        "f 1 2 3 4\n"
    ).encode()


def validate_glb(payload: bytes) -> None:
    if len(payload) < 12 or payload[:4] != b"glTF":
        raise RuntimeError("Facades-3D response is not a GLB")
    declared_length = struct.unpack_from("<I", payload, 8)[0]
    if declared_length != len(payload):
        raise RuntimeError(
            f"truncated GLB: header declares {declared_length}, received {len(payload)}"
        )


def template_seed(style_id: str, group_index: int, width_index: int) -> int:
    return SEED_BASE + STYLE_IDS.index(style_id) * 100 + group_index * 10 + width_index


def size_key(width_m: float, height_m: float) -> str:
    return f"w{round(width_m * 100):04d}-h{round(height_m * 100):04d}"


def load_manifest(storage: ObjectStorage, key: str) -> FacadeLibraryManifest:
    try:
        return FacadeLibraryManifest.model_validate_json(storage.get_bytes(key))
    except ObjectNotFoundError:
        return FacadeLibraryManifest()


def open_storage(local_root: str | None) -> ObjectStorage:
    if local_root is not None:
        return LocalStorage(local_root)
    storage = get_object_storage()
    if not isinstance(storage, MinioStorage):
        raise SystemExit(
            "FILESERVER_* variables are not set; refusing to write the library to local disk "
            "(pass --local-root to do that on purpose)"
        )
    return storage


def generate_wall(
    client: httpx.Client,
    *,
    prompt: str,
    width_m: float,
    height_m: float,
    pixels_per_meter: int,
    seed: int,
) -> bytes:
    data = {
        "prompt": prompt,
        "cluster_count": "1",
        "pixels_per_meter": str(pixels_per_meter),
        "seed": str(seed),
        "max_wall_aspect_ratio": "100",
    }
    files = {
        "input_model": (
            "facade-template.obj",
            canonical_wall_obj(width_m, height_m),
            "text/plain",
        )
    }
    for attempt in range(BUSY_RETRIES):
        response = client.post("/generate", data=data, files=files)
        if response.status_code < 400:
            validate_glb(response.content)
            return response.content
        if response.status_code != 503 or attempt == BUSY_RETRIES - 1:
            raise RuntimeError(
                f"Facades-3D returned {response.status_code}: {response.text[:1000]}"
            )
        time.sleep(min(5 * 2**attempt, MAX_BACKOFF_S))
    raise RuntimeError("Facades-3D stayed busy")


def run_check(storage: ObjectStorage, prefix: str) -> None:
    """Probe write and read access; delete is optional because nothing requires it."""
    for folder in (prefix, SCENE_KEY_PREFIX):
        key = f"{folder}/.write-probe"
        storage.put_bytes(b"probe", key, "application/octet-stream")
        if storage.get_bytes(key) != b"probe":
            raise RuntimeError(f"read-back mismatch for {key}")
        print(f"ok: write/read under {folder}/")
        try:
            storage.delete(key)
        except ObjectStorageError as exc:
            print(f"warning: cannot delete {key}, the probe object stays: {exc}")


def _identity(template: FacadeTemplate) -> tuple[str, str, float, int]:
    return (
        template.style_id,
        template.floor_group,
        template.width_m,
        template.pixels_per_meter,
    )


def run_prewarm(storage: ObjectStorage, args: argparse.Namespace) -> None:
    manifest_key = f"{args.prefix}/manifest.json"
    manifest = load_manifest(storage, manifest_key)
    existing = {_identity(item) for item in manifest.templates}
    with httpx.Client(
        base_url=args.facades_url.rstrip("/"),
        timeout=httpx.Timeout(GENERATION_TIMEOUT_S),
        trust_env=False,
    ) as api:
        for style_id in args.styles:
            for group_index, group in enumerate(args.floor_groups):
                for width_index, width_m in enumerate(args.widths):
                    template = _planned_template(
                        args,
                        style_id=style_id,
                        group=group,
                        width_m=width_m,
                        seed=template_seed(style_id, group_index, width_index),
                    )
                    label = f"{style_id}/{group}/{width_m:g}m"
                    if _identity(template) in existing and not args.force:
                        print(f"skip {label}: already present")
                        continue
                    if args.dry_run:
                        print(
                            f"would generate {label} ({width_m:g}x{template.height_m:g} m)"
                        )
                        continue
                    manifest = _generate_template(
                        api,
                        storage,
                        manifest,
                        manifest_key=manifest_key,
                        template=template,
                    )
                    existing.add(_identity(template))
    print(f"manifest: {manifest_key} ({len(manifest.templates)} templates)")


def _planned_template(
    args: argparse.Namespace,
    *,
    style_id: str,
    group: PreviewFloorGroup,
    width_m: float,
    seed: int,
) -> FacadeTemplate:
    preset = PRESETS_BY_ID[style_id]
    prompt = library_prompt(preset)
    floors = REPRESENTATIVE_FLOORS[group]
    height_m = floors * args.floor_height
    return FacadeTemplate(
        object_key=(
            f"{args.prefix}/styles/{style_id}/{group}/{size_key(width_m, height_m)}/wall.glb"
        ),
        style_id=style_id,
        style_name_ru=preset.name_ru,
        style_key=canonical_style_key(prompt),
        prompt=prompt,
        floor_group=group,
        floors=floors,
        floor_height_m=args.floor_height,
        width_m=width_m,
        height_m=height_m,
        pixels_per_meter=args.pixels_per_meter,
        seed=seed,
        glb_size_bytes=1,
    )


def _generate_template(
    api: httpx.Client,
    storage: ObjectStorage,
    manifest: FacadeLibraryManifest,
    *,
    manifest_key: str,
    template: FacadeTemplate,
) -> FacadeLibraryManifest:
    started = time.monotonic()
    print(f"generate {template.object_key}")
    payload = generate_wall(
        api,
        prompt=template.prompt,
        width_m=template.width_m,
        height_m=template.height_m,
        pixels_per_meter=template.pixels_per_meter,
        seed=template.seed if template.seed is not None else SEED_BASE,
    )
    storage.put_bytes(payload, template.object_key, GLB_MIME_TYPE)
    now = datetime.now(timezone.utc)
    stored = template.model_copy(
        update={"glb_size_bytes": len(payload), "generated_at": now}
    )
    updated = FacadeLibraryManifest(
        templates=[
            *(
                item
                for item in manifest.templates
                if _identity(item) != _identity(stored)
            ),
            stored,
        ],
        updated_at=now,
    )
    storage.put_json(updated.model_dump(mode="json"), manifest_key)
    elapsed = time.monotonic() - started
    print(f"uploaded {template.object_key} ({len(payload)} bytes, {elapsed:.1f}s)")
    return updated


def pick_preview_template(
    templates: Sequence[FacadeTemplate],
    *,
    style_id: str,
    group: PreviewFloorGroup,
    args: argparse.Namespace,
) -> FacadeTemplate | None:
    """Only a section of the same floor group, so a preview never shows a stretched texture."""
    floors = REPRESENTATIVE_FLOORS[group]
    try:
        return resolve_template(
            templates,
            style_id=style_id,
            floors=floors,
            width_m=PREVIEW_WIDTH_M,
            height_m=floors * args.floor_height,
            pixels_per_meter=args.pixels_per_meter,
            max_width_scale=args.max_width_scale,
            allow_nearest=False,
        )
    except FacadeTemplateMiss:
        return None


async def run_previews(storage: ObjectStorage, args: argparse.Namespace) -> None:
    library = FacadeTemplateLibrary(
        storage,
        prefix=args.prefix,
        manifest_ttl_seconds=0.0,
        max_width_scale=args.max_width_scale,
    )
    try:
        manifest = await library.manifest()
    except FacadeLibraryUnavailable as exc:
        raise SystemExit(f"facade library manifest is unavailable: {exc}") from exc

    previews: list[StylePreview] = []
    for style_id in args.styles:
        for group in PREVIEW_FLOOR_GROUPS:
            template = pick_preview_template(
                manifest.templates, style_id=style_id, group=group, args=args
            )
            if template is None:
                print(
                    f"skip preview {style_id}/{group}: no section in this floor group"
                )
                continue
            if args.dry_run:
                print(
                    f"would build preview {style_id}/{group} from {template.object_key}"
                )
                continue
            previews.append(await _upload_preview(storage, library, template, group))
    if args.dry_run:
        return
    index = StylePreviewIndex(previews=_merge_previews(storage, library, previews))
    storage.put_json(index.model_dump(mode="json"), library.preview_index_key)
    print(
        f"preview index: {library.preview_index_key} ({len(index.previews)} previews)"
    )


async def _upload_preview(
    storage: ObjectStorage,
    library: FacadeTemplateLibrary,
    template: FacadeTemplate,
    group: PreviewFloorGroup,
) -> StylePreview:
    mesh = (await library.load_meshes([template]))[template.cache_key]
    payload = build_preview_glb(
        mesh,
        width_m=PREVIEW_WIDTH_M,
        height_m=template.height_m,
        name=f"preview_{template.style_id}",
    )
    key = library.preview_key(template.style_id, group)
    storage.put_bytes(payload, key, GLB_MIME_TYPE)
    print(f"uploaded {key} ({len(payload)} bytes)")
    return StylePreview(
        style_id=template.style_id,
        style_name_ru=PRESETS_BY_ID[template.style_id].name_ru,
        floor_group=group,
        object_key=key,
        etag=preview_etag(payload),
        size_bytes=len(payload),
    )


def _merge_previews(
    storage: ObjectStorage, library: FacadeTemplateLibrary, fresh: list[StylePreview]
) -> list[StylePreview]:
    try:
        current = StylePreviewIndex.model_validate_json(
            storage.get_bytes(library.preview_index_key)
        ).previews
    except ObjectNotFoundError:
        current = []
    replaced = {(item.style_id, item.floor_group) for item in fresh}
    kept = [
        item for item in current if (item.style_id, item.floor_group) not in replaced
    ]
    return [*kept, *fresh]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill the facade library and its previews."
    )
    parser.add_argument("command", choices=["check", "prewarm", "previews", "all"])
    parser.add_argument(
        "--prefix", default=os.getenv("FACADE_LIBRARY_PREFIX", "facade-library/v1")
    )
    parser.add_argument(
        "--facades-url", default=os.getenv("FACADES_3D_API", "http://a6k4.dgx:8030")
    )
    parser.add_argument("--styles", nargs="+", choices=STYLE_IDS, default=STYLE_IDS)
    parser.add_argument(
        "--floor-groups",
        nargs="+",
        choices=PREVIEW_FLOOR_GROUPS,
        default=list(PREVIEW_FLOOR_GROUPS),
    )
    parser.add_argument("--widths", nargs="+", type=float, default=list(DEFAULT_WIDTHS))
    parser.add_argument("--floor-height", type=float, default=3.0)
    parser.add_argument(
        "--pixels-per-meter",
        type=int,
        default=int(os.getenv("FACADE_LIBRARY_PPM", "32")),
    )
    parser.add_argument(
        "--max-width-scale",
        type=float,
        default=float(os.getenv("FACADE_LIBRARY_MAX_WIDTH_SCALE", "2.5")),
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate existing sections"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List the work, write nothing"
    )
    parser.add_argument(
        "--local-root", default=None, help="Use a local directory instead of MinIO"
    )
    args = parser.parse_args(argv)
    args.prefix = args.prefix.strip("/")
    if any(width <= 0 for width in args.widths):
        parser.error("all widths must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    storage = open_storage(args.local_root)
    if args.command in {"check", "all"} and not args.dry_run:
        run_check(storage, args.prefix)
    if args.command in {"prewarm", "all"}:
        run_prewarm(storage, args)
    if args.command in {"previews", "all"}:
        asyncio.run(run_previews(storage, args))


if __name__ == "__main__":
    main()
