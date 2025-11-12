from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from app.logic.postprocessing.generation_params import GenParams, ParamsProvider


class ShapesLibrary:
    """
    Библиотека форм сервисных зданий и утилиты генерации вариантов.

    Ответственности:
      • Описание «ядер» (offsets) форм сервисов по типам (сад, школа, поликлиника)
      • Повороты/зеркала, нормализация в (0,0), построение всех уникальных вариантов
      • Подбор прямоугольников для площадок по требуемой площади (в клетках)
      • Расчёт min количества клеток площадки с учётом внутреннего отступа
      • Подбор и сортировка вариантов по соотношению сторон ядра (core fit)
      • Нормативы площадок/вместимости по типу сервиса и имени паттерна

    Параметры берутся из GenParams (cell_size_m, service_site_rules,
    randomize_service_forms, inner_margin_cells и др.).
    """
    def __init__(self, params_provider: ParamsProvider):
        self._params = params_provider

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()

    def pattern_library(self) -> Dict[str, List[Tuple[str, List[Tuple[int, int]], bool]]]:
        service_parameters = self.generation_parameters.service_patterns
        if not service_parameters:
            raise ValueError("GenParams.service_patterns пуст — нет доступных форм сервисов.")

        by_svc: Dict[str, List[Tuple[str, List[Tuple[int, int]], bool]]] = {}
        for (svc, pattern), rec in service_parameters.items():
            offsets_raw = rec["offsets"]
            offsets = [(int(dr), int(dc)) for dr, dc in offsets_raw]
            allow = bool(rec.get("allow_rotations", True))
            by_svc.setdefault(svc, []).append((pattern, offsets, allow))
        return by_svc

    @staticmethod
    def transform_offsets(
        offsets: List[Tuple[int, int]], rot_k: int, mirror: bool
    ) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for (dr, dc) in offsets:
            r, c = dr, dc
            for _ in range(rot_k % 4):
                r, c = c, -r
            if mirror:
                c = -c
            out.append((r, c))
        minr = min(r for r, _ in out)
        minc = min(c for _, c in out)
        return [(r - minr, c - minc) for (r, c) in out]

    @staticmethod
    def shape_variants(
        offsets: List[Tuple[int, int]], allow_rotations: bool
    ) -> List[List[Tuple[int, int]]]:
        variants = set()
        rots = [0, 1, 2, 3] if allow_rotations else [0]
        mirrors = [False, True] if allow_rotations else [False]
        for k in rots:
            for m in mirrors:
                var = tuple(sorted(ShapesLibrary.transform_offsets(offsets, k, m)))
                variants.add(var)
        return [list(v) for v in variants]

    @staticmethod
    def shape_length(var: List[Tuple[int, int]]) -> int:
        rs = [dr for dr, _ in var]
        cs = [dc for _, dc in var]
        return max(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    def site_cells_required(self, area_m2: float) -> int:
        cs = float(self.generation_parameters.cell_size_m)
        return int(math.ceil(max(area_m2, 1.0) / (cs * cs)))

    @staticmethod
    def rect_variants_for_cells(
        ncells: int,
        *,
        max_variants: int = 12,
        ar_min: float = 0.33,
        ar_max: float = 3.0,
    ) -> List[Tuple[str, List[Tuple[int, int]]]]:
        base = int(round(math.sqrt(max(1, ncells))))
        pairs: List[Tuple[int, int, int]] = []
        span = max(1, base) + 12
        for r in range(1, span + 1):
            c = int(math.ceil(ncells / r))
            ar = max(r, c) / max(1.0, min(r, c))
            if ar_min <= ar <= ar_max:
                pairs.append((r, c, r * c - ncells))
        pairs.sort(key=lambda t: (t[2], abs(t[0] - t[1])))
        pairs = pairs[:max_variants]
        variants: List[Tuple[str, List[Tuple[int, int]]]] = []
        for r, c, _ in pairs:
            offs = [(dr, dc) for dr in range(r) for dc in range(c)]
            variants.append((f"RECT_{r}x{c}", offs))
        return variants

    def territory_shape_variants(self, area_m2: float) -> List[Tuple[str, List[Tuple[int, int]]]]:
        ncells = self.site_cells_required(area_m2)
        return self.rect_variants_for_cells(ncells, max_variants=12)

    def service_site_spec(self, svc: str, pattern: str | None) -> Tuple[float, int, int]:
        rule = self.generation_parameters.service_site_rules.get((svc, pattern))
        site_area_m2 = float(rule.get("site_area_m2", 0.0))
        capacity     = int(rule.get("capacity", 0))

        pat_meta = self.generation_parameters.service_patterns.get((svc, pattern))
        floors = pat_meta.get("floors")
        return site_area_m2, capacity, int(floors)
    
    def min_site_cells_for_service_with_margin(
        self,
        svc: str,
        shape_variants_by_svc: Dict[str, List[Tuple[str, List[List[Tuple[int, int]]]]]],
        *,
        inner_margin_cells: int | None = None,
    ) -> int:
        if inner_margin_cells is None:
            inner_margin_cells = int(self.generation_parameters.inner_margin_cells)
        best: int | None = None
        for _pat, vars_ in shape_variants_by_svc.get(svc, []):
            for var in vars_:
                vr = [dr for dr, _ in var]
                vc = [dc for _, dc in var]
                h = (max(vr) - min(vr) + 1) + 2 * inner_margin_cells
                w = (max(vc) - min(vc) + 1) + 2 * inner_margin_cells
                cells_needed = h * w
                best = cells_needed if best is None else min(best, cells_needed)
        return 0 if best is None else int(best)
    
    @staticmethod
    def sort_variants_by_core_fit(
        variants: List[List[Tuple[int, int]]], core_h: int, core_w: int
    ) -> List[List[Tuple[int, int]]]:
        if core_h <= 0 or core_w <= 0:
            return []
        core_ar = core_w / core_h if core_h > 0 else 1.0

        def dims(var: List[Tuple[int, int]]):
            vr = [dr for (dr, dc) in var]
            vc = [dc for (dr, dc) in var]
            h = max(vr) - min(vr) + 1
            w = max(vc) - min(vc) + 1
            return h, w

        scored: List[Tuple[int, float, int, List[Tuple[int, int]]]] = []
        for var in variants:
            h, w = dims(var)
            var_ar = w / h if h > 0 else 1.0
            same_orient = int(not ((core_w >= core_h) ^ (w >= h)))
            ar_diff = abs(math.log(max(var_ar, 1e-6) / max(core_ar, 1e-6)))
            scored.append((0 if same_orient else 1, ar_diff, h * w, var))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        return [t[3] for t in scored]

    def build_shape_variants_from_library(
        self, *, rng: random.Random | None = None
    ) -> Dict[str, List[Tuple[str, List[List[Tuple[int, int]]]]]]:
        lib = self.pattern_library()
        if rng is None:
            rng = random.Random(self.generation_parameters.service_random_seed)
        shape_variants: Dict[str, List[Tuple[str, List[List[Tuple[int, int]]]]]] = {}
        for svc, shapes in lib.items():
            items = list(shapes)
            if self.generation_parameters.randomize_service_forms:
                rng.shuffle(items)
            buf: List[Tuple[str, List[List[Tuple[int, int]]]]] = []
            for name, offsets, allow_rot in items:
                vars_ = self.shape_variants(offsets, allow_rot)
                if self.generation_parameters.randomize_service_forms:
                    rng.shuffle(vars_)
                buf.append((name, vars_))
            shape_variants[svc] = buf
        return shape_variants
