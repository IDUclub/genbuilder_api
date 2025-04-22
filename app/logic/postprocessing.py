import math
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate
from loguru import logger

class Postprocessing:
    async def remove_near_collinear_points(self, poly, angle_threshold=5):
        """
        Удаляет вершины, если угол между соседними сегментами меньше angle_threshold (в градусах).
        Поддерживает Polygon и MultiPolygon.
        """
        if poly.geom_type == "MultiPolygon":
            return MultiPolygon([
                await self.remove_near_collinear_points(p, angle_threshold)
                for p in poly.geoms
            ])
        coords = list(poly.exterior.coords)[:-1]
        if len(coords) < 3:
            return poly
        new_coords = []
        for i in range(len(coords)):
            p_prev, p_curr, p_next = coords[i-1], coords[i], coords[(i+1)%len(coords)]
            v1 = (p_curr[0]-p_prev[0], p_curr[1]-p_prev[1])
            v2 = (p_next[0]-p_curr[0], p_next[1]-p_curr[1])
            ang1 = math.degrees(math.atan2(v1[1], v1[0]))
            ang2 = math.degrees(math.atan2(v2[1], v2[0]))
            diff = abs(ang2-ang1) % 360
            if diff > 180: diff = 360 - diff
            if diff >= angle_threshold:
                new_coords.append(p_curr)
        if new_coords and new_coords[0] != new_coords[-1]:
            new_coords.append(new_coords[0])
        if len(new_coords) < 3:
            return poly
        return Polygon(new_coords)

    async def snap_to_grid_rotated(self, poly, grid_size, angle_deg):
        """
        Поворачивает на -angle_deg и привязывает вершины к сетке, затем поворачивает обратно.
        Поддерживает Polygon и MultiPolygon.
        """
        if poly.geom_type == "MultiPolygon":
            return MultiPolygon([
                await self.snap_to_grid_rotated(p, grid_size, angle_deg)
                for p in poly.geoms
            ])
        poly_rot = rotate(poly, -angle_deg, origin='centroid', use_radians=False)
        snapped = [(round(x/grid_size)*grid_size, round(y/grid_size)*grid_size)
                   for x,y in poly_rot.exterior.coords]
        poly_snapped = Polygon(snapped)
        return rotate(poly_snapped, angle_deg, origin='centroid', use_radians=False)

    async def fix_angles(self, poly, grid_size=1.0, allowed_angles=[0,45,90], error_metric='area', straighten_threshold=5):
        """
        Для каждого angle из allowed_angles делает snap_to_grid_rotated и выбирает лучший по метрике.
        Поддерживает Polygon и MultiPolygon.
        """
        if poly.geom_type == "MultiPolygon":
            return MultiPolygon([
                await self.fix_angles(p, grid_size, allowed_angles, error_metric, straighten_threshold)
                for p in poly.geoms
            ])
        orig_area = poly.area
        best, best_err = None, float('inf')
        for ang in allowed_angles:
            cand = await self.snap_to_grid_rotated(poly, grid_size, ang)
            err = abs(cand.area - orig_area) if error_metric=='area' else cand.hausdorff_distance(poly)
            if err < best_err:
                best, best_err = cand, err
        result = await self.remove_near_collinear_points(best, angle_threshold=straighten_threshold)
        return result

    async def remove_narrow_corridors(self, poly, corridor_threshold=7.0):
        """
        Удаляет узкие «проезды» менее corridor_threshold через буфер.
        Поддерживает Polygon и MultiPolygon.
        """
        if poly.geom_type == "MultiPolygon":
            return MultiPolygon([
                await self.remove_narrow_corridors(p, corridor_threshold)
                for p in poly.geoms
            ])
        d = corridor_threshold/2
        return poly.buffer(d).buffer(-d)

    async def regularize_buildings_gdf_custom(self, gdf, simplify_tol=0.5, grid_size=1.0, allowed_angles=[0,45,90],
                                              corridor_threshold=7.0, straighten_threshold=5):
        """
        Применяет simplify, straighten, remove corridors, fix angles.
        Сохраняет все атрибуты и CRS.
        """
        logger.info("regularize buildings, count={}", len(gdf))
        gdf_copy = gdf.copy()
        regs = []
        for geom in gdf_copy.geometry:
            g = await self.remove_near_collinear_points(geom.simplify(simplify_tol, True), straighten_threshold)
            g = await self.remove_narrow_corridors(g, corridor_threshold)
            g = await self.fix_angles(g, grid_size, allowed_angles, 'area', straighten_threshold)
            regs.append(g)
        gdf_copy['regularized_geometry'] = regs
        return gdf_copy

    async def merge_close_polygons(self, geos, distance_threshold=2.0):
        buf = geos.buffer(distance_threshold/2)
        merged = buf.unary_union.buffer(-distance_threshold/2)
        geoms = [merged] if isinstance(merged, Polygon) else list(merged.geoms)
        return gpd.GeoSeries(geoms, crs=geos.crs)

    def compute_min_width(self, poly):
        if poly.geom_type=='MultiPolygon':
            w = [self.compute_min_width(p) for p in poly.geoms if not p.is_empty]
            return min(w) if w else float('inf')
        mrr = poly.minimum_rotated_rectangle
        pts = list(mrr.exterior.coords)[:-1]
        d = [math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1)]
        return min(d) if d else 0

    async def filter_by_min_width(self, geos, min_width=7.0):
        geoms = [g for g in geos if self.compute_min_width(g)>=min_width]
        return gpd.GeoSeries(geoms, crs=geos.crs)

    async def process_buildings(self, buildings_gdf):
        """
        Основной метод: объединение, фильтр, регуляризация.
        Реально reproj входной и выходной слоёв.
        """
        orig_crs = buildings_gdf.crs
        attrs = buildings_gdf.drop(columns='geometry')
        merged = await self.merge_close_polygons(
            gpd.GeoSeries(buildings_gdf.geometry, crs=buildings_gdf.crs)
        )
        filtered = await self.filter_by_min_width(merged)
        gdf_pre = gpd.GeoDataFrame(attrs, geometry=filtered, crs=buildings_gdf.crs)
        gdf_reg = await self.regularize_buildings_gdf_custom(
            gdf_pre,
            simplify_tol=1.5,
            grid_size=1.0,
            allowed_angles=[35,45,90],
            corridor_threshold=7.0,
            straighten_threshold=5
        )
        gdf_final = gdf_reg.set_geometry('regularized_geometry', drop=True)
        gdf_final = gdf_final.to_crs(orig_crs)
        logger.info("process_buildings completed count={}", len(gdf_final))
        return gdf_final

postprocessing = Postprocessing()
