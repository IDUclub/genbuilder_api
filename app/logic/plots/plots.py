from __future__ import annotations

import geopandas as gpd
import pandas as pd
from loguru import logger

from app.logic.building_capacity_optimizer import CapacityOptimizer
from app.logic.building_params import (
    BuildingGenParams,
    BuildingParamsProvider,
)
from app.logic.generation_params import (
    GenParams,
    ParamsProvider,
)

from app.logic.plots.plot_slicer import PlotSegmentSlicer
from app.logic.plots.plot_merge import PlotMerger
from app.logic.plots.plot_tuner import PlotTuner


class PlotsGenerator:
    """
    Orchestrates the plots pipeline: slices segments into plots, iteratively
    merges undersized plots, and retunes per-plot building parameters for the
    chosen generation mode and target column.
    """

    def __init__(
        self,
        params_provider: ParamsProvider,
        building_params_provider: BuildingParamsProvider,
        capacity_optimizer: CapacityOptimizer,
        slicer: PlotSegmentSlicer,
        merger: PlotMerger,
        tuner: PlotTuner,
    ):
        self._params = params_provider
        self._building_params = building_params_provider
        self.capacity_optimizer = capacity_optimizer
        self.slicer = slicer
        self.merger = merger
        self.tuner = tuner

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()

    @property
    def building_generation_parameters(self) -> BuildingGenParams:
        return self._building_params.current()

    def generate_plots(
        self,
        segments: gpd.GeoDataFrame,
        mode: str,
        target_col: str,
    ) -> gpd.GeoDataFrame:

        logger.debug(
            f"[generate_plots] start: mode={mode}, target_col={target_col}, "
            f"segments_count={len(segments)}"
        )

        segments = segments.dropna(subset=["plot_depth"]).copy()
        crs = segments.crs
        parts_list: list[gpd.GeoDataFrame] = []
        for _, r in segments.iterrows():
            p = self.slicer._slice_segment_with_plots(r, crs=crs)
            if p is not None and not p.empty:
                parts_list.append(p)

        if not parts_list:
            logger.debug(
                f"[generate_plots] no plots produced from segments "
                f"(mode={mode}, target_col={target_col})"
            )
            return gpd.GeoDataFrame(
                columns=list(segments.columns),
                geometry="geometry",
                crs=crs,
            )

        result = gpd.GeoDataFrame(
            pd.concat(parts_list, ignore_index=True),
            geometry="geometry",
            crs=crs,
        )

        result = self.merger.merge_small_plots_iterative(result, area_factor=0.5)
        result = result[result.geometry.notna() & ~result.geometry.is_empty]
        result = result[result.geom_type.isin(["Polygon", "MultiPolygon"])]

        result["plot_area"] = result.geometry.area
        result = result[result["plot_area"] > 0]
        result = self.tuner._recalc_buildings_for_plots(
            result,
            mode=mode,
            target_col=target_col,
        )

        logger.debug(
            f"[generate_plots] done: mode={mode}, target_col={target_col}, "
            f"plots_count={len(result)}"
        )

        return result