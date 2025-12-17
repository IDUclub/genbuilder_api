from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional, Hashable

import math
import random
import asyncio

import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon, box, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.affinity import rotate, translate
from shapely.errors import GEOSException
from shapely.validation import make_valid

from app.logic.generation_params import GenParams, ParamsProvider

from app.common.geo_utils import longest_edge_angle_mrr
from app.common.geo_utils import safe_make_valid
from app.common.geo_utils import ensure_crs


class ServiceGenerator:
    """
    Generates non-residential service buildings for residential blocks by
    converting built living area into per-block service capacity targets,
    sampling service sites in block free space, and placing template-based
    buildings (from OSM-derived projects) into those sites.
    """

    def __init__(self, params_provider: ParamsProvider):
        self._params = params_provider

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()

    def compute_service_limits_for_blocks(
        self,
        blocks: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        service_normatives: pd.DataFrame,
    ) -> Dict[Any, Dict[str, float]]:
        la_per_block = buildings.groupby("src_index", dropna=False)["living_area"].sum()

        blocks = blocks.copy()
        blocks["actual_la"] = blocks["src_index"].map(la_per_block).fillna(0.0)

        all_limits: Dict[Any, Dict[str, float]] = {}

        for _, row in blocks.iterrows():
            zid = row["src_index"]
            actual_la = float(row["actual_la"])

            people = actual_la / self.generation_parameters.la_per_person

            limits: Dict[str, float] = {}

            for _, srv in service_normatives.iterrows():
                service_name = srv["service_name"]
                cap_per_1000 = float(srv["service_capacity"])

                target_cap = round(cap_per_1000 * (people / 1000.0), 0)
                limits[service_name] = target_cap

            all_limits[zid] = limits

        return all_limits

    def load_service_projects(self) -> gpd.GeoDataFrame:
        projects_gdf = gpd.read_file(self.generation_parameters.service_projects_file)
        projects_gdf = projects_gdf.to_crs("EPSG:4326")

        expected_cols = [
            "service",
            "type_id",
            "capacity",
            "floors_count",
            "plot_length_min",
            "plot_length_max",
            "plot_width_min",
            "plot_width_max",
            "address",
            "osm_type",
            "osm_id",
            "osm_url",
            "geometry",
        ]
        missing = [c for c in expected_cols if c not in projects_gdf.columns]
        if missing:
            raise ValueError(f"Columns are missing: {missing}.")

        projects_gdf = projects_gdf[expected_cols]
        return projects_gdf

    @staticmethod
    def _get_block_free_area(
        block_geom: BaseGeometry,
        plots_gdf: gpd.GeoDataFrame,
    ) -> BaseGeometry:
        if block_geom is None or block_geom.is_empty:
            return block_geom

        block_geom_valid = safe_make_valid(block_geom)
        if block_geom_valid is None or block_geom_valid.is_empty:
            return block_geom
        block_geom = block_geom_valid

        if plots_gdf.empty:
            return block_geom

        cleaned_geoms: List[BaseGeometry] = []
        for g in plots_gdf.geometry:
            if g is None or g.is_empty:
                continue

            g_valid = safe_make_valid(g)
            if g_valid is None or g_valid.is_empty:
                continue

            cleaned_geoms.append(g_valid)

        if not cleaned_geoms:
            return block_geom

        try:
            plots_union = unary_union(cleaned_geoms)
        except GEOSException:
            union_geom = cleaned_geoms[0]
            for g in cleaned_geoms[1:]:
                try:
                    union_geom = union_geom.union(g)
                except GEOSException:
                    continue
            plots_union = union_geom

        try:
            free_area = block_geom.difference(plots_union)
        except GEOSException:
            block_fixed = make_valid(block_geom)
            plots_fixed = make_valid(plots_union)
            free_area = block_fixed.difference(plots_fixed)

        return free_area

    @staticmethod
    def _compute_main_axis_angle(poly: BaseGeometry) -> float:
        angle = longest_edge_angle_mrr(poly, degrees=True)
        if angle < -90.0:
            angle += 180.0
        elif angle >= 90.0:
            angle -= 180.0

        return angle

    def _sample_rect_in_polygon(
        self,
        poly: BaseGeometry,
        length_range: Tuple[float, float],
        width_range: Tuple[float, float],
        max_attempts: int = 200,
        preferred_angle: Optional[float] = None,
        existing_centroids: Optional[List[Point]] = None,
        min_dist_between_centers: Optional[float] = None,
    ) -> Optional[Polygon]:
        if poly.is_empty:
            return None

        minx, miny, maxx, maxy = poly.bounds
        if minx == maxx or miny == maxy:
            return None

        len_min, len_max = length_range
        wid_min, wid_max = width_range

        if preferred_angle is not None:
            base = preferred_angle
            angle_candidates = [
                base,
                base + 5.0,
                base - 5.0,
                base + 15.0,
                base - 15.0,
                base + 90.0,
                base - 90.0,
            ]
        else:
            angle_candidates = [0.0, 90.0, 45.0, -45.0, 30.0, -30.0]

        for _ in range(max_attempts):
            length = random.uniform(len_min, len_max)
            width = random.uniform(wid_min, wid_max)

            rect = box(-length / 2.0, -width / 2.0, length / 2.0, width / 2.0)

            angle = random.choice(angle_candidates)
            rect_rot = rotate(rect, angle, origin=(0, 0), use_radians=False)

            cx = random.uniform(minx, maxx)
            cy = random.uniform(miny, maxy)

            rect_shifted = translate(rect_rot, xoff=cx, yoff=cy)

            if not rect_shifted.within(poly):
                continue

            if (
                existing_centroids
                and min_dist_between_centers
                and min_dist_between_centers > 0.0
            ):
                center = rect_shifted.centroid
                too_close = any(
                    center.distance(c) < min_dist_between_centers
                    for c in existing_centroids
                )
                if too_close:
                    continue

            return rect_shifted

        return None

    @staticmethod
    def _normalize_building_geometry(geom: BaseGeometry) -> BaseGeometry:
        c = geom.centroid
        return translate(geom, xoff=-c.x, yoff=-c.y)

    def _place_building_in_plot(
        self, building_template: BaseGeometry, plot_geom: BaseGeometry
    ) -> Optional[BaseGeometry]:
        allowed_area = plot_geom.buffer(-self.generation_parameters.INNER_BORDER)
        if allowed_area.is_empty:
            return None

        minx, miny, maxx, maxy = allowed_area.bounds
        if minx == maxx or miny == maxy:
            return None

        for _ in range(self.generation_parameters.max_service_attempts):
            cx = random.uniform(minx, maxx)
            cy = random.uniform(miny, maxy)

            if not allowed_area.contains(box(cx, cy, cx, cy)):
                continue

            b_shifted = translate(building_template, xoff=cx, yoff=cy)

            if b_shifted.within(allowed_area):
                return b_shifted

        return None

    def _select_project_for_remaining(
        self,
        service_projects: gpd.GeoDataFrame,
        remaining_capacity: float,
    ) -> List[pd.Series]:
        if service_projects.empty:
            return []

        df = service_projects.copy()
        df["capacity_diff"] = (df["capacity"] - remaining_capacity).abs()
        df_sorted = df.sort_values("capacity_diff")

        return list(df_sorted.itertuples(index=False))

    def place_service_buildings(
        self,
        blocks: gpd.GeoDataFrame,
        plots_gdf: gpd.GeoDataFrame,
        all_limits: Dict[Hashable, Dict[str, float]],
        projects_gdf: gpd.GeoDataFrame,
        blocks_crs: int | str = 32636,
    ) -> gpd.GeoDataFrame:
        projects_local = ensure_crs(projects_gdf, blocks_crs)
        normalized_buildings: Dict[Any, BaseGeometry] = {}
        for row in projects_local.itertuples():
            type_id = getattr(row, "type_id")
            geom = getattr(row, "geometry")
            normalized_buildings[type_id] = self._normalize_building_geometry(geom)

        service_buildings_rows: List[Dict[str, Any]] = []

        for block_row in blocks.itertuples():
            block_id = getattr(block_row, "src_index")
            block_geom = getattr(block_row, "geometry")

            block_angle = self._compute_main_axis_angle(block_geom)

            minx_b, miny_b, maxx_b, maxy_b = block_geom.bounds
            span_min = min(maxx_b - minx_b, maxy_b - miny_b)

            min_spacing = 0.0
            if (
                self.generation_parameters.max_sites_per_service_per_block > 1
                and span_min > 0
            ):
                min_spacing = 0.25 * span_min

            placed_plot_centers: List[Point] = []

            plots_block = plots_gdf[plots_gdf["src_index"] == block_id]

            free_area = self._get_block_free_area(block_geom, plots_block)
            if free_area.is_empty:
                continue

            block_limits = all_limits.get(block_id, {})
            if not block_limits:
                continue

            for service_name, target_capacity in block_limits.items():
                if target_capacity <= 0:
                    continue

                service_projects = projects_local[
                    projects_local["service"] == service_name
                ]
                if service_projects.empty:
                    continue

                placed_capacity = 0.0
                sites_count = 0

                while (
                    placed_capacity < target_capacity
                    and sites_count
                    < self.generation_parameters.max_sites_per_service_per_block
                    and not free_area.is_empty
                ):
                    remaining_capacity = target_capacity - placed_capacity

                    project_candidates = self._select_project_for_remaining(
                        service_projects,
                        remaining_capacity,
                    )
                    if not project_candidates:
                        break

                    placed_in_iteration = False

                    for project_row in project_candidates:
                        type_id = getattr(project_row, "type_id")
                        capacity = float(getattr(project_row, "capacity"))
                        floors = getattr(project_row, "floors_count")
                        plot_length_min = float(getattr(project_row, "plot_length_min"))
                        plot_length_max = float(getattr(project_row, "plot_length_max"))
                        plot_width_min = float(getattr(project_row, "plot_width_min"))
                        plot_width_max = float(getattr(project_row, "plot_width_max"))
                        osm_url = getattr(project_row, "osm_url", None)
                        address = getattr(project_row, "address", None)

                        building_template_norm = normalized_buildings.get(type_id)
                        if building_template_norm is None:
                            continue

                        plot_geom = self._sample_rect_in_polygon(
                            free_area,
                            length_range=(plot_length_min, plot_length_max),
                            width_range=(plot_width_min, plot_width_max),
                            max_attempts=self.generation_parameters.max_service_attempts,
                            preferred_angle=block_angle,
                            existing_centroids=placed_plot_centers,
                            min_dist_between_centers=min_spacing,
                        )

                        if plot_geom is None:
                            continue

                        building_oriented_main = rotate(
                            building_template_norm,
                            block_angle,
                            origin=(0, 0),
                            use_radians=False,
                        )

                        building_geom = self._place_building_in_plot(
                            building_template=building_oriented_main,
                            plot_geom=plot_geom,
                        )

                        if building_geom is None:
                            building_oriented_orth = rotate(
                                building_template_norm,
                                block_angle + 90.0,
                                origin=(0, 0),
                                use_radians=False,
                            )
                            building_geom = self._place_building_in_plot(
                                building_template=building_oriented_orth,
                                plot_geom=plot_geom,
                            )

                        if building_geom is None:
                            continue
                        footprint_area = building_geom.area
                        try:
                            floors_val = float(floors) if floors is not None else 0.0
                        except (TypeError, ValueError):
                            floors_val = 0.0

                        building_area = (
                            footprint_area * floors_val if floors_val > 0 else 0.0
                        )

                        row_out: Dict[str, Any] = {
                            "src_index": block_id,
                            "service": service_name,
                            "project_id": type_id,
                            "capacity": capacity,
                            "floors_count": floors,
                            "living_area": 0.0,              
                            "functional_area": building_area,
                            "osm_url": osm_url,
                            "address": address,
                            "geometry": building_geom,
                        }
                        service_buildings_rows.append(row_out)

                        placed_capacity += capacity
                        sites_count += 1
                        placed_in_iteration = True

                        free_area = free_area.difference(plot_geom)
                        placed_plot_centers.append(plot_geom.centroid)

                        break

                    if not placed_in_iteration:
                        break

        if not service_buildings_rows:
            return gpd.GeoDataFrame(
                columns=[
                    "service",
                    "capacity",
                    "floors_count",
                    "living_area",
                    "functional_area",
                    "geometry",
                ],
                geometry="geometry",
                crs=blocks_crs,
            )

        service_buildings_gdf = gpd.GeoDataFrame(
            service_buildings_rows,
            geometry="geometry",
            crs=blocks_crs,
        )

        return service_buildings_gdf

    async def generate_services(
        self,
        blocks: gpd.GeoDataFrame,
        plots: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        service_normatives: pd.DataFrame,
        crs: int | str,
    ) -> gpd.GeoDataFrame:
        blocks = blocks.reset_index()
        blocks.rename(columns={"index": "src_index"}, inplace=True)
        projects_gdf = await asyncio.to_thread(self.load_service_projects)
        all_limits = await asyncio.to_thread(
            self.compute_service_limits_for_blocks,
            blocks,
            buildings,
            service_normatives,
        )
        services_buildings_gdf = await asyncio.to_thread(
            self.place_service_buildings,
            blocks,
            plots,
            all_limits,
            projects_gdf,
            crs,
        )
        return services_buildings_gdf
