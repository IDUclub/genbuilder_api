"""Patch two defects in a deployed Facades-3D image (CTLab-ITMO/Facades-3D@main).

1. ``gen_wall.py`` reads the model-loading flag as ``os.getenv("SKIP_MODEL_LOAD")``
   but ``release_generation_memory`` dereferences a bare ``SKIP_MODEL_LOAD`` that
   is never defined.  The resulting ``NameError`` runs in the ``finally`` block of
   every request, so even a successful generation is reported as a failure and
   CUDA memory is never released.
2. The IP-Adapter is loaded unconditionally, but ``ip_adapter_image`` is only
   passed when a style reference is present.  A UNet with ``encoder_hid_proj``
   still requires ``added_cond_kwargs["image_embeds"]``, so a request without a
   style reference dies inside diffusers with
   ``TypeError: argument of type 'NoneType' is not iterable``.  Passing a blank
   image keeps the contribution at zero because the adapter scale is already 0.

The script is idempotent: re-running it reports "already patched" and changes
nothing.  Run it inside the container, then restart the container.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FLAG_ANCHOR = "MODEL_CONFIG = DEFAULT_MODEL_CONFIG\n"
FLAG_PATCH = 'MODEL_CONFIG = DEFAULT_MODEL_CONFIG\nSKIP_MODEL_LOAD = bool(os.getenv("SKIP_MODEL_LOAD"))\n'

ADAPTER_ANCHOR = """        if use_style_reference:
            pipeline_kwargs["ip_adapter_image"] = style_ref
"""
ADAPTER_PATCH = """        pipeline_kwargs["ip_adapter_image"] = (
            style_ref if use_style_reference else Image.new("RGB", (224, 224), "white")
        )
"""


def apply(source: str) -> tuple[str, list[str]]:
    applied: list[str] = []

    if "SKIP_MODEL_LOAD = bool(" in source:
        applied.append("SKIP_MODEL_LOAD: already patched")
    elif FLAG_ANCHOR in source:
        source = source.replace(FLAG_ANCHOR, FLAG_PATCH, 1)
        applied.append("SKIP_MODEL_LOAD: defined")
    else:
        raise SystemExit(f"anchor not found: {FLAG_ANCHOR!r}")

    if ADAPTER_PATCH in source:
        applied.append("ip_adapter_image: already patched")
    elif ADAPTER_ANCHOR in source:
        source = source.replace(ADAPTER_ANCHOR, ADAPTER_PATCH, 1)
        applied.append("ip_adapter_image: always passed")
    else:
        raise SystemExit(f"anchor not found: {ADAPTER_ANCHOR!r}")

    return source, applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to gen_wall.py inside the container")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    original = args.path.read_text(encoding="utf-8")
    patched, applied = apply(original)

    for line in applied:
        print(line)

    if patched == original:
        print("nothing to write")
        return
    if args.dry_run:
        print("dry run: not written")
        return

    backup = args.path.with_suffix(args.path.suffix + ".orig")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"backup: {backup}")
    args.path.write_text(patched, encoding="utf-8")
    print(f"written: {args.path}")


if __name__ == "__main__":
    sys.exit(main())
