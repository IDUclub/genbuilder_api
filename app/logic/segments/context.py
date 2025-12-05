from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple, Optional, List

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class BlockSegmentsContext:
    block_id: Any
    row: pd.Series
    mode: str
    target_col: str
    target_val: float
    la_ratio: float
    plot_area_base: float
    far_initial: float
    rects_block: gpd.GeoDataFrame
    seg_fronts: List[float] = field(default_factory=list)
    seg_depths: List[float] = field(default_factory=list)

    @property
    def has_rects(self) -> bool:
        return not self.rects_block.empty

    @property
    def has_positive_target(self) -> bool:
        return self.target_val > 0.0

    @property
    def is_trivial_zero_case(self) -> bool:
        return (not self.has_rects) or (not self.has_positive_target)


class BlockSegmentsContextBuilder:
    """
    Builds BlockSegmentsContext for a single block by normalizing the target,
    deriving initial FAR (if needed), and computing segment fronts and depths
    from rectangular segments.
    """

    @staticmethod
    def build(
        row: pd.Series,
        rects_block: gpd.GeoDataFrame,
        *,
        mode: str,
        target_col: str,
        la_ratio: float,
        plot_area_base: float,
        base_dims: Optional[Tuple[float, float, float]] = None,
    ) -> BlockSegmentsContext:
        block_id = row.name

        raw_target = row.get(target_col, 0.0)
        try:
            target_val = float(raw_target)
        except (TypeError, ValueError):
            target_val = 0.0
        raw_far_initial = row.get("far_initial", np.nan)
        if pd.isna(raw_far_initial):
            if base_dims is not None and plot_area_base > 0:
                L0, W0, H0 = base_dims
                far_initial = (float(L0) * float(W0) * float(H0)) / float(
                    plot_area_base
                )
            else:
                far_initial = np.nan
        else:
            try:
                far_initial = float(raw_far_initial)
            except (TypeError, ValueError):
                far_initial = np.nan

        seg_fronts, seg_depths = BlockSegmentsContextBuilder._compute_fronts_and_depths(
            rects_block
        )

        logger.debug(
            f"[BlockSegmentsContext.build] block_id={block_id}, mode={mode}, "
            f"target_col={target_col}, raw_target={raw_target}, target_val={target_val}, "
            f"la_ratio={la_ratio}, plot_area_base={plot_area_base}, "
            f"far_initial={far_initial}, "
            f"segments_count={len(rects_block)}, "
            f"seg_fronts={seg_fronts}, seg_depths={seg_depths}"
        )

        return BlockSegmentsContext(
            block_id=block_id,
            row=row,
            mode=mode,
            target_col=target_col,
            target_val=target_val,
            la_ratio=float(la_ratio),
            plot_area_base=float(plot_area_base),
            far_initial=float(far_initial) if not np.isnan(far_initial) else np.nan,
            rects_block=rects_block,
            seg_fronts=seg_fronts,
            seg_depths=seg_depths,
        )

    @staticmethod
    def _compute_fronts_and_depths(
        rects_block: gpd.GeoDataFrame,
    ) -> Tuple[List[float], List[float]]:
        seg_fronts: List[float] = []
        seg_depths: List[float] = []

        if rects_block.empty:
            return seg_fronts, seg_depths

        missing = [c for c in ("width", "height") if c not in rects_block.columns]
        if missing:
            raise ValueError(
                f"rects_block is missing required columns {missing} "
                f"for BlockSegmentsContextBuilder._compute_fronts_and_depths()"
            )

        for _, seg in rects_block.iterrows():
            w = float(seg["width"])
            h = float(seg["height"])
            seg_fronts.append(max(w, h))
            seg_depths.append(min(w, h))

        return seg_fronts, seg_depths
