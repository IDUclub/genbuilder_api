from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from loguru import logger

from app.logic.segments.context import BlockSegmentsContext
from app.logic.building_params import BuildingParams


@dataclass
class SegmentCapacityResult:
    capacity_by_front_idx: List[int]
    max_capacity: int


class SegmentCapacityCalculator:
    """
    Computes how many building plots can fit along a block’s segments for each
    candidate frontage F in building_params.plot_side, returning per-F capacities
    and the maximum achievable capacity across all frontages.
    """

    def compute(
        self,
        ctx: BlockSegmentsContext,
        building_params: BuildingParams,
    ) -> SegmentCapacityResult:
        if not ctx.seg_fronts or not len(building_params.plot_side):
            logger.debug(
                "[SegmentCapacityCalculator.compute] block_id={}: "
                "no seg_fronts or empty plot_side -> zero capacities".format(
                    ctx.block_id
                )
            )
            return SegmentCapacityResult(capacity_by_front_idx=[], max_capacity=0)

        capacity_by_front_idx: List[int] = []

        logger.debug(
            "[SegmentCapacityCalculator.compute] block_id={}: "
            "start capacity calculation, segments_count={}, F_range={}".format(
                ctx.block_id,
                len(ctx.seg_fronts),
                list(building_params.plot_side),
            )
        )

        for F in building_params.plot_side:
            F_val = float(F)
            if F_val <= 0:
                capacity_by_front_idx.append(0)
                continue

            cap_total = 0
            for front_len in ctx.seg_fronts:
                cap_total += int(front_len // F_val)

            capacity_by_front_idx.append(cap_total)

        max_cap = max(capacity_by_front_idx) if capacity_by_front_idx else 0

        logger.debug(
            "[SegmentCapacityCalculator.compute] block_id={}: "
            "capacity_by_front_idx={}, max_capacity={}".format(
                ctx.block_id, capacity_by_front_idx, max_cap
            )
        )

        return SegmentCapacityResult(
            capacity_by_front_idx=capacity_by_front_idx,
            max_capacity=int(max_cap),
        )
