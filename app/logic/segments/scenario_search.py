from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

from loguru import logger

from app.logic.segments.context import BlockSegmentsContext
from app.logic.segments.capacity_calculator import SegmentCapacityResult
from app.logic.building_params import BuildingParams
from app.common.building_math import (
    usable_per_building,
    far_from_dims,
    building_need,
)


@dataclass
class ChosenScenario:
    L: float
    W: float
    H: float
    F: float

    building_need: int
    building_capacity: int
    buildings_count: int

    living_per_building: float
    total_usable_area: float
    far_final: float


@dataclass
class ScenarioSearchResult:
    best_success: Optional[ChosenScenario]
    fallback: Optional[ChosenScenario]


class BlockScenarioSearch:
    """
    Explores discrete (L, W, H, F) building scenarios for a single block using
    segment capacities, selecting a best-success option that meets building_need
    and a fallback option that maximizes total usable area when full coverage
    is impossible.
    """

    def search(
        self,
        ctx: BlockSegmentsContext,
        building_params: BuildingParams,
        capacity_result: SegmentCapacityResult,
        base_indices: Tuple[int, int, int, int],
    ) -> ScenarioSearchResult:
        capacity_by_front_idx = capacity_result.capacity_by_front_idx
        if not capacity_by_front_idx:
            logger.debug(
                "[BlockScenarioSearch.search] block_id={}: "
                "capacity_by_front_idx is empty -> no candidates".format(
                    ctx.block_id
                )
            )
            return ScenarioSearchResult(best_success=None, fallback=None)

        base_L_idx, base_W_idx, base_H_idx, base_F_idx = base_indices

        base_L = float(building_params.building_length_range[base_L_idx])
        base_W = float(building_params.building_width_range[base_W_idx])
        base_H = float(building_params.building_height[base_H_idx])
        base_F = float(building_params.plot_side[base_F_idx])

        best_success: Optional[ChosenScenario] = None
        best_success_score: Optional[Tuple[float, float, float, float, float]] = None

        fallback: Optional[ChosenScenario] = None
        fallback_total_la: float = -1.0

        plot_area_base = float(ctx.plot_area_base)
        target_val = float(ctx.target_val)
        far_initial = float(ctx.far_initial) if ctx.far_initial == ctx.far_initial else 0.0
        la_ratio = float(ctx.la_ratio)

        def consider_candidate(
            L_idx: int,
            W_idx: int,
            H_idx: int,
            F_idx: int,
            only_long_front: bool,
        ) -> None:
            nonlocal best_success, best_success_score, fallback, fallback_total_la
            if F_idx < 0 or F_idx >= len(building_params.plot_side):
                return
            if F_idx >= len(capacity_by_front_idx):
                return

            L = float(building_params.building_length_range[L_idx])
            W = float(building_params.building_width_range[W_idx])
            H = float(building_params.building_height[H_idx])
            F = float(building_params.plot_side[F_idx])

            usable_one = usable_per_building(L, W, H, la_ratio)
            if usable_one <= 0:
                return
            if F > 0:
                plot_depth = plot_area_base / F
            else:
                plot_depth = float("inf")

            is_long_side_along_front = F >= plot_depth

            building_need_val = building_need(target_val, usable_one)
            capacity_total = int(capacity_by_front_idx[F_idx])
            if building_need_val > 0:
                buildings_count_fallback = min(capacity_total, building_need_val)
            else:
                buildings_count_fallback = capacity_total
            total_la_fallback = buildings_count_fallback * usable_one

            if total_la_fallback > fallback_total_la and capacity_total > 0:
                far_final_fb = far_from_dims(L, W, H, plot_area_base)
                fallback_total_la = float(total_la_fallback)
                fallback = ChosenScenario(
                    L=L,
                    W=W,
                    H=H,
                    F=F,
                    building_need=int(building_need_val),
                    building_capacity=int(capacity_total),
                    buildings_count=int(buildings_count_fallback),
                    living_per_building=float(usable_one),
                    total_usable_area=float(total_la_fallback),
                    far_final=float(far_final_fb),
                )
                logger.debug(
                    "[BlockScenarioSearch.search] block_id={}: "
                    "fallback updated -> L={}, W={}, H={}, F={}, "
                    "capacity_total={}, total_la_fallback={}".format(
                        ctx.block_id,
                        L,
                        W,
                        H,
                        F,
                        capacity_total,
                        total_la_fallback,
                    )
                )
            if building_need_val <= 0 or capacity_total < building_need_val:
                return

            if only_long_front and not is_long_side_along_front:
                return

            buildings_count = int(building_need_val)
            total_la = buildings_count * usable_one
            la_diff_abs = abs(total_la - target_val)

            far_final = far_from_dims(L, W, H, plot_area_base)
            far_diff_abs = abs(far_final - far_initial)

            front_reduction = base_F - F
            floor_delta = H - base_H
            area_diff_abs = abs((L * W) - (base_L * base_W))

            score = (
                la_diff_abs,
                far_diff_abs,
                front_reduction,
                floor_delta,
                area_diff_abs,
            )

            if (best_success_score is None) or (score < best_success_score):
                best_success_score = score
                best_success = ChosenScenario(
                    L=L,
                    W=W,
                    H=H,
                    F=F,
                    building_need=int(building_need_val),
                    building_capacity=int(capacity_total),
                    buildings_count=int(buildings_count),
                    living_per_building=float(usable_one),
                    total_usable_area=float(total_la),
                    far_final=float(far_final),
                )
                logger.debug(
                    "[BlockScenarioSearch.search] block_id={}: "
                    "best_success updated -> L={}, W={}, H={}, F={}, "
                    "building_need={}, capacity_total={}, total_la={}, score={}".format(
                        ctx.block_id,
                        L,
                        W,
                        H,
                        F,
                        building_need_val,
                        capacity_total,
                        total_la,
                        score,
                    )
                )

        logger.debug(
            "[BlockScenarioSearch.search] block_id={}: "
            "searching candidates: "
            "len(L_range)={}, len(W_range)={}, len(H_range)={}, len(F_range)={}".format(
                ctx.block_id,
                len(building_params.building_length_range),
                len(building_params.building_width_range),
                len(building_params.building_height),
                len(building_params.plot_side),
            )
        )
        for L_idx in range(base_L_idx, len(building_params.building_length_range)):
            for W_idx in range(base_W_idx, len(building_params.building_width_range)):
                consider_candidate(
                    L_idx=L_idx,
                    W_idx=W_idx,
                    H_idx=base_H_idx,
                    F_idx=base_F_idx,
                    only_long_front=True,
                )
        if best_success is None:
            logger.debug(
                "[BlockScenarioSearch.search] block_id={}: "
                "no best_success after L/W scan, trying H-range with fixed F".format(
                    ctx.block_id
                )
            )
            for H_idx in range(base_H_idx, len(building_params.building_height)):
                for L_idx in range(
                    base_L_idx, len(building_params.building_length_range)
                ):
                    for W_idx in range(
                        base_W_idx, len(building_params.building_width_range)
                    ):
                        consider_candidate(
                            L_idx=L_idx,
                            W_idx=W_idx,
                            H_idx=H_idx,
                            F_idx=base_F_idx,
                            only_long_front=True,
                        )
        if best_success is None:
            logger.debug(
                "[BlockScenarioSearch.search] block_id={}: "
                "no best_success after H-range, trying smaller F with only_long_front=True".format(
                    ctx.block_id
                )
            )
            for F_idx in range(base_F_idx, 0, -1):
                for H_idx in range(base_H_idx, len(building_params.building_height)):
                    for L_idx in range(
                        base_L_idx, len(building_params.building_length_range)
                    ):
                        for W_idx in range(
                            base_W_idx, len(building_params.building_width_range)
                        ):
                            consider_candidate(
                                L_idx=L_idx,
                                W_idx=W_idx,
                                H_idx=H_idx,
                                F_idx=F_idx,
                                only_long_front=True,
                            )
        if best_success is None:
            logger.debug(
                "[BlockScenarioSearch.search] block_id={}: "
                "no best_success yet, allowing short-front (only_long_front=False)".format(
                    ctx.block_id
                )
            )
            for F_idx in range(base_F_idx, 0, -1):
                for H_idx in range(base_H_idx, len(building_params.building_height)):
                    for L_idx in range(
                        base_L_idx, len(building_params.building_length_range)
                    ):
                        for W_idx in range(
                            base_W_idx, len(building_params.building_width_range)
                        ):
                            consider_candidate(
                                L_idx=L_idx,
                                W_idx=W_idx,
                                H_idx=H_idx,
                                F_idx=F_idx,
                                only_long_front=False,
                            )

        return ScenarioSearchResult(
            best_success=best_success,
            fallback=fallback,
        )
