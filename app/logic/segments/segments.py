from __future__ import annotations

from loguru import logger
import geopandas as gpd
import numpy as np

from app.logic.building_capacity_optimizer import CapacityOptimizer
from app.logic.building_params import (
    BuildingGenParams,
    BuildingParamsProvider,
)

from app.logic.segments.solver import BlockSolver


class SegmentsAllocator:
    """
    Allocates building capacity across block rectangles by solving a per-block
    (L, W, H, plot_front) scenario, updating blocks with final capacity/FAR and
    segments with plot_front, plot_depth, and plots_capacity for each mode.
    """

    def __init__(
        self,
        capacity_optimizer: CapacityOptimizer,
        building_params_provider: BuildingParamsProvider,
        block_solver: BlockSolver
    ):
        self.capacity_optimizer = capacity_optimizer
        self._building_params = building_params_provider
        self.block_solver = block_solver

    @property
    def building_generation_parameters(self) -> BuildingGenParams:
        return self._building_params.current()

    def update_blocks_with_segments(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        rects_gdf: gpd.GeoDataFrame,
        far: str,
        *,
        mode: str = "residential",
        target_col: str = "la_target",
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

        blocks = blocks_gdf.copy()
        rects = rects_gdf.copy()

        logger.debug(
            f"[update_blocks] start: mode={mode}, target_col={target_col}, "
            f"blocks_count={len(blocks)}, rects_count={len(rects)}, far={far}"
        )

        if rects.empty:
            logger.debug("[update_blocks] rects is EMPTY, nothing to update")
            return blocks, rects
        for col, default in [
            ("plots_capacity", 0),
            ("plot_front", np.nan),
            ("plot_depth", np.nan),
        ]:
            if col not in rects.columns:
                rects[col] = default

        try:
            unique_src = rects["src_index"].unique()
        except Exception:
            unique_src = "N/A"

        logger.debug(
            f"[update_blocks] mode={mode}: rects.src_index.unique={unique_src}"
        )
        for idx, row in blocks.iterrows():
            fg = row.get("floors_group")
            target_val = row.get(target_col)
            logger.debug(
                f"[update_blocks] block_idx={idx}, mode={mode}: "
                f"floors_group={fg}, target_col={target_col}, target_val={target_val}"
            )

            block_rects = rects[rects["src_index"] == idx]
            logger.debug(
                f"[update_blocks] block_idx={idx}, mode={mode}: "
                f"matched_rects_count={len(block_rects)}"
            )
            if block_rects.empty:
                logger.debug(
                    f"[update_blocks] block_idx={idx}, mode={mode}: NO rects for this block, "
                    f"set building_capacity=0, buildings_count=0"
                )
                blocks.at[idx, "building_capacity"] = 0
                blocks.at[idx, "buildings_count"] = 0
                continue
            building_type = None
            if "building_type" in row.index:
                building_type = row.get("building_type")
                logger.debug(
                    f"[update_blocks] block_idx={idx}, mode={mode}: "
                    f"using existing building_type from blocks -> {building_type}"
                )

            if building_type is None:
                blocks.at[idx, "building_capacity"] = 0
                blocks.at[idx, "buildings_count"] = 0
                continue
            try:
                building_params = self.building_generation_parameters.params_by_type[
                    building_type
                ]
            except KeyError:
                logger.error(
                    f"[update_blocks] block_idx={idx}, mode={mode}: building_type={building_type} "
                    f"not in params_by_type keys="
                    f"{list(self.building_generation_parameters.params_by_type.keys())} "
                    f"-> set building_capacity=0, buildings_count=0"
                )
                blocks.at[idx, "building_capacity"] = 0
                blocks.at[idx, "buildings_count"] = 0
                continue

            logger.debug(
                f"[update_blocks] block_idx={idx}, mode={mode}: calling BlockSolver.solve_block_with_segments "
                f"with {len(block_rects)} rects, building_type={building_type}"
            )
            block_res, block_rects_res = self.block_solver.solve_block_with_segments(
                row=row,
                rects_block=block_rects,
                far=far,
                building_params=building_params,
                la_ratio=building_params.la_coef,
                mode=mode,
                target_col=target_col,
            )

            logger.debug(
                f"[update_blocks] block_idx={idx}, mode={mode}: BlockSolver returned -> "
                f"building_capacity={block_res.get('building_capacity')}, "
                f"buildings_count={block_res.get('buildings_count')}, "
                f"plot_front={block_res.get('plot_front')}, "
                f"plot_depth={block_res.get('plot_depth')}"
            )
            for key, value in block_res.items():
                blocks.at[idx, key] = value
            rects.loc[
                block_rects_res.index, ["plots_capacity", "plot_front", "plot_depth"]
            ] = block_rects_res[["plots_capacity", "plot_front", "plot_depth"]].values

            rects.loc[block_rects_res.index, "src_index"] = idx
            rects.loc[block_rects_res.index, target_col] = row.get(target_col, np.nan)

            rects.loc[block_rects_res.index, "far_initial"] = block_res.get(
                "far_initial",
                row.get("far_initial", np.nan),
            )
            rects.loc[block_rects_res.index, "building_length"] = block_res.get(
                "building_length",
                row.get("building_length", np.nan),
            )
            rects.loc[block_rects_res.index, "building_width"] = block_res.get(
                "building_width",
                row.get("building_width", np.nan),
            )
            rects.loc[block_rects_res.index, "floors_count"] = block_res.get(
                "floors_count",
                row.get("floors_count", np.nan),
            )
            rects.loc[block_rects_res.index, "floors_group"] = row.get(
                "floors_group", np.nan
            )

            logger.debug(
                f"[update_blocks] block_idx={idx}, mode={mode}: rects updated for indices "
                f"{list(block_rects_res.index)}"
            )

        logger.debug(f"[update_blocks] DONE, mode={mode}")
        return blocks, rects
