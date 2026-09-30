"""Manifest schema of the facade template library.

The manifest is shared with ``facade-jobs``, which reads the same objects, so
the template model mirrors its schema field for field and forbids extras.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PreviewFloorGroup = Literal["low", "medium", "high"]

PREVIEW_FLOOR_GROUPS: tuple[PreviewFloorGroup, ...] = ("low", "medium", "high")
REPRESENTATIVE_FLOORS: dict[PreviewFloorGroup, int] = {
    "low": 3,
    "medium": 6,
    "high": 12,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_style_key(prompt: str, negative_prompt: str | None = None) -> str:
    """Return the prompt identity ``facade-jobs`` uses to match templates."""
    payload = json.dumps(
        {
            "prompt": " ".join(prompt.strip().lower().split()),
            "negative_prompt": " ".join(
                (negative_prompt or "").strip().lower().split()
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def floor_group(floors: int) -> str:
    if floors == 1:
        return "single"
    if floors <= 4:
        return "low"
    if floors <= 8:
        return "medium"
    if floors <= 16:
        return "high"
    return f"extra-high-{floors}"


class FacadeTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_key: str
    metadata_key: str | None = None
    style_id: str
    style_name_ru: str
    style_key: str
    prompt: str
    negative_prompt: str | None = None
    floor_group: str
    floors: int = Field(gt=0)
    floor_height_m: float = Field(gt=0)
    width_m: float = Field(gt=0)
    height_m: float = Field(gt=0)
    pixels_per_meter: int = Field(gt=0)
    seed: int | None = None
    glb_size_bytes: int = Field(ge=1)
    generated_at: datetime = Field(default_factory=_utc_now)

    @property
    def cache_key(self) -> str:
        return f"{self.object_key}@{self.generated_at.isoformat()}"

    def width_scale_for(self, width_m: float) -> float:
        return max(width_m / self.width_m, self.width_m / width_m)

    def score(self, width_m: float, height_m: float) -> float:
        return abs(math.log(width_m / self.width_m)) + 0.25 * abs(
            math.log(height_m / self.height_m)
        )


class FacadeLibraryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    updated_at: datetime = Field(default_factory=_utc_now)
    templates: list[FacadeTemplate] = Field(default_factory=list)


class StylePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style_id: str
    style_name_ru: str
    floor_group: PreviewFloorGroup
    object_key: str
    etag: str
    size_bytes: int = Field(ge=1)
    generated_at: datetime = Field(default_factory=_utc_now)


class StylePreviewIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    updated_at: datetime = Field(default_factory=_utc_now)
    previews: list[StylePreview] = Field(default_factory=list)


__all__ = [
    "PREVIEW_FLOOR_GROUPS",
    "REPRESENTATIVE_FLOORS",
    "FacadeLibraryManifest",
    "FacadeTemplate",
    "PreviewFloorGroup",
    "StylePreview",
    "StylePreviewIndex",
    "canonical_style_key",
    "floor_group",
]
