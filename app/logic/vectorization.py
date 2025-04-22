import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
from loguru import logger
from PIL import Image

class Vectorizer:
    async def inverse_transform_polygon(self, poly, minx, miny, maxx, maxy, width, height):
        """
        Преобразуем полигон из координат изображения (width x height)
        обратно в координаты исходного bbox.
        """
        if isinstance(poly, MultiPolygon):
            parts = [
                await self.inverse_transform_polygon(p, minx, miny, maxx, maxy, width, height)
                for p in poly.geoms
            ]
            return MultiPolygon(parts)

        range_x = maxx - minx
        range_y = maxy - miny
        new_coords = [
            (
                x * (range_x / width) + minx,
                maxy - y * (range_y / height)
            )
            for x, y in poly.exterior.coords
        ]
        return Polygon(new_coords)

    async def vectorize_buildings(
        self,
        input_zones: gpd.GeoDataFrame,
        generated_image: Image.Image,
        min_area: float = 100,
        sat_thresh: int = 30,
        dark_thresh: int = 60,
        dominance_ratio: float = 0.3
    ):
        """
        1) Находит контуры любых объектов не‑белого/серого по HSV.
        2) Для каждого полигона опционально присваивает класс,
           если один из трёх цветовых масок доминирует по площади.
        """
        minx, miny, maxx, maxy = input_zones.total_bounds
        bbox = (minx, miny, maxx, maxy)

        img_bgr = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v_eq = clahe.apply(v)
        hsv_eq = cv2.merge([h, s, v_eq])

        _, mask_dark  = cv2.threshold(v_eq, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        _, mask_col   = cv2.threshold(s,       sat_thresh, 255, cv2.THRESH_BINARY)
        mask_all      = cv2.bitwise_or(mask_dark, mask_col)

        brown_lower = np.array([10,  80,  40]);  brown_upper = np.array([30, 255, 200])
        black_lower = np.array([  0,   0,   0]);  black_upper = np.array([180, 255,  50])
        green_lower = np.array([35,  50,  50]);  green_upper = np.array([85, 255, 255])

        mask_brown = cv2.inRange(hsv_eq, brown_lower, brown_upper)
        mask_black = cv2.inRange(hsv_eq, black_lower, black_upper)
        mask_green = cv2.inRange(hsv_eq, green_lower, green_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel, iterations=2)
        height, width = mask_all.shape

        contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info("Found {} total contours", len(contours))

        geoms, classes = [], []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            pts = cnt.reshape(-1,2)
            if len(pts) < 3:
                continue
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            poly = await self.inverse_transform_polygon(poly, *bbox, width, height)
            geoms.append(poly)

            mask_poly = np.zeros(mask_all.shape, dtype=np.uint8)
            cv2.drawContours(mask_poly, [cnt], -1, 255, cv2.FILLED)

            cnt_b = cv2.countNonZero(cv2.bitwise_and(mask_brown, mask_poly))
            cnt_k = cv2.countNonZero(cv2.bitwise_and(mask_black, mask_poly))
            cnt_g = cv2.countNonZero(cv2.bitwise_and(mask_green, mask_poly))

            cls = None
            if cnt_b > dominance_ratio * area and cnt_b > cnt_k and cnt_b > cnt_g:
                cls = "Facility"
            elif cnt_k > dominance_ratio * area and cnt_k > cnt_b and cnt_k > cnt_g:
                cls = "Building"
            elif cnt_g > dominance_ratio * area and cnt_g > cnt_b and cnt_g > cnt_k:
                cls = "Park"

            classes.append(cls)

        logger.info(
            "Returning {} polygons, {} of them classified",
            len(geoms), sum(1 for c in classes if c is not None)
        )

        gdf_out = gpd.GeoDataFrame(
            {"class": classes, "geometry": geoms},
            crs=input_zones.crs or "EPSG:32636"
        )
        return gdf_out

vectorizer = Vectorizer()
