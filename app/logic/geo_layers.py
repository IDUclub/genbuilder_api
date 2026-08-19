"""Geo-layer link descriptors for the chat stream and the chat history.

A descriptor tells the frontend where a layer can be fetched, without carrying
the layer itself. Two kinds exist, and the difference is deliberate:

- **stored layers** (``buildings``, ``blocks_input``) are our own artefacts,
  written to object storage and served back by ``/files/{slot}/{result_id}``;
- the **functional zones layer** belongs to UrbanDB, so its descriptor is a live
  query against ``/layers/functional_zones`` — nothing is copied, and UrbanDB
  keeps enforcing access to a private scenario on every fetch.

``download_url`` is always ``None``: object storage sits on a private network,
so bytes are streamed through the API rather than handed out as presigned URLs.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from app.common.urls import durable_url

SOURCE_SERVICE = "genbuilder"
MIME_TYPE = "application/geo+json"

SLOT_BUILDINGS = "buildings"
SLOT_BLOCKS_INPUT = "blocks_input"

ZONES_LAYER_NAME = "functional_zones"
ZONES_LAYER_PATH = "/layers/functional_zones"

_SLOT_SPECS: dict[str, tuple[str, str, str]] = {
    SLOT_BUILDINGS: ("buildings", "Сгенерированная застройка", "result"),
    SLOT_BLOCKS_INPUT: ("blocks_input", "Загруженные кварталы", "input"),
}

FILE_SLOTS: tuple[str, ...] = tuple(_SLOT_SPECS)


def object_key(result_id: str, slot: str) -> str:
    """Storage key for one slot of one generation result.

    Derived from the result id alone, which is what lets the file endpoint
    resolve a link without any database lookup.
    """
    if slot not in _SLOT_SPECS:
        raise ValueError(f"Unknown geo-layer slot: {slot!r}")
    if not result_id:
        raise ValueError("result_id is required")
    return f"{result_id}/{slot}.geojson"


def build_stored_layer(
    *,
    slot: str,
    result_id: str,
    public_base_url: str | None = None,
    request_base_url: str | None = None,
) -> dict[str, Any]:
    """Descriptor for a layer served from object storage."""
    name, title, role = _SLOT_SPECS[slot]
    return {
        "name": name,
        "title": title,
        "role": role,
        "url": durable_url(
            f"/files/{slot}/{result_id}", public_base_url, request_base_url
        ),
        "download_url": None,
        "filename": f"{slot}.geojson",
        "mime_type": MIME_TYPE,
        "source_service": SOURCE_SERVICE,
    }


def build_zones_layer(
    *,
    scenario_id: int,
    year: int,
    source: str,
    functional_zone_types: list[str] | tuple[str, ...],
    public_base_url: str | None = None,
    request_base_url: str | None = None,
) -> dict[str, Any]:
    """Descriptor for the functional zones layer, as a live query.

    The link stays valid indefinitely because it holds scenario coordinates
    rather than a stored object, and it is not a capability: reading it still
    requires the caller's own token.
    """
    query = urlencode(
        [
            ("scenario_id", scenario_id),
            ("year", year),
            ("source", source),
            *(("functional_zone_types", zone) for zone in functional_zone_types),
        ]
    )
    return {
        "name": ZONES_LAYER_NAME,
        "title": "Функциональные зоны",
        "role": "input",
        "url": durable_url(
            f"{ZONES_LAYER_PATH}?{query}", public_base_url, request_base_url
        ),
        "download_url": None,
        "filename": "functional_zones.geojson",
        "mime_type": MIME_TYPE,
        "source_service": SOURCE_SERVICE,
    }


def geo_layer_to_file_part(layer: dict[str, Any]) -> dict[str, Any]:
    """ChatStorage ``file`` part payload from a layer descriptor.

    Stores only the durable ``url``. ``download_url`` is dropped on principle:
    it is ephemeral by contract and must never reach permanent chat history.
    """
    return {
        "url": layer["url"],
        "name": layer.get("name"),
        "title": layer.get("title"),
        "filename": layer.get("filename"),
        "mime_type": layer.get("mime_type"),
        "source_service": layer.get("source_service"),
    }
