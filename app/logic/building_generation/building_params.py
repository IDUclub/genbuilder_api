from __future__ import annotations

import contextvars
import contextlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Iterator


class BuildingType(str, Enum):
    """Типы жилой застройки по этажности/формату."""
    IZH = "private"              
    MKD_2_4 = "low"     
    MKD_5_8 = "medium"    
    MKD_9_16 = "high"    
    HIGHRISE = "extreme"    

@dataclass(frozen=True)
class BuildingParams:

    building_length_range: List[int]  
    building_width_range: List[int]    
    building_height: List[int]         


    plot_side: List[int]
    plot_area_min: float
    plot_area_max: float


    la_coef: float


PARAMS_BY_TYPE: Dict[BuildingType, BuildingParams] = {



    BuildingType.IZH: BuildingParams(
        building_length_range=list(range(8, 16)),
        building_width_range=list(range(8, 13)),
        building_height=list(range(1, 4)),
        plot_side=list(range(20, 41)),
        plot_area_min=450.0,
        plot_area_max=900.0,
        la_coef=0.65,
    ),




    BuildingType.MKD_2_4: BuildingParams(
        building_length_range=list(range(30, 81)),
        building_width_range=list(range(10, 20)),
        building_height=list(range(2, 5)),
        plot_side=list(range(36, 87)),
        plot_area_min=1500.0,
        plot_area_max=4000.0,
        la_coef=0.55,
    ),




    BuildingType.MKD_5_8: BuildingParams(
        building_length_range=list(range(40, 121)),
        building_width_range=list(range(12, 19)),
        building_height=list(range(5, 9)),
        plot_side=list(range(48, 121)),
        plot_area_min=2000.0,
        plot_area_max=6000.0,
        la_coef=0.50,
    ),




    BuildingType.MKD_9_16: BuildingParams(
        building_length_range=list(range(40, 121)),
        building_width_range=list(range(14, 23)),
        building_height=list(range(9, 17)),
        plot_side=list(range(60, 141)),
        plot_area_min=2500.0,
        plot_area_max=8000.0,
        la_coef=0.45,
    ),




    BuildingType.HIGHRISE: BuildingParams(
        building_length_range=list(range(30, 71)),
        building_width_range=list(range(18, 31)),
        building_height=list(range(17, 31)),
        plot_side=list(range(60, 151)),
        plot_area_min=3000.0,
        plot_area_max=10000.0,
        la_coef=0.40,
    ),
}


@dataclass(frozen=True)
class BuildingGenParams:
    """
    Аналог GenParams, но для параметров типов застройки.

    params_by_type:
      ключ  - BuildingType
      значение - BuildingParams
    """
    params_by_type: Dict[BuildingType, BuildingParams] = field(default_factory=dict)

    def patched(self, patch: Dict[str, Any]) -> "BuildingGenParams":
        """
        Возвращает новый BuildingGenParams поверх текущего,
        с учётом рекурсивного deep-merge патча.

        Ожидаемый формат patch примерно такой:
        {
            "params_by_type": {
                "MKD_5_8": {
                    "plot_area_min": 1800.0,
                    "plot_area_max": 5500.0,
                },
                "IZH": {
                    "la_coef": 0.6
                }
            }
        }
        """
        def deep_merge(a: Any, b: Any) -> Any:
            if isinstance(a, dict) and isinstance(b, dict):
                c = dict(a)
                for k, v in b.items():
                    c[k] = deep_merge(c.get(k), v)
                return c
            return b if b is not None else a


        base_dict = asdict(self)
        merged = deep_merge(base_dict, patch)


        raw_params_by_type = merged.get("params_by_type", {})

        def _normalize_bt(key: Any) -> BuildingType:
            if isinstance(key, BuildingType):
                return key
            if isinstance(key, str):

                try:
                    return BuildingType(key)
                except ValueError:
                    return BuildingType[key]
            raise KeyError(f"Unknown BuildingType key: {key!r}")

        new_mapping: Dict[BuildingType, BuildingParams] = {}
        for bt_key, params_dict in raw_params_by_type.items():
            bt = _normalize_bt(bt_key)

            new_mapping[bt] = BuildingParams(**params_dict)

        return BuildingGenParams(params_by_type=new_mapping)


class BuildingParamsProvider:
    """
    Аналог ParamsProvider для конфигурации типов застройки.

    Хранит BuildingGenParams в contextvars, чтобы можно было
    локально переопределять параметры внутри контекстного менеджера.
    """
    def __init__(self, base: BuildingGenParams):
        self._var: contextvars.ContextVar[BuildingGenParams] = contextvars.ContextVar(
            "building_gen_params",
            default=base,
        )

    def current(self) -> BuildingGenParams:
        return self._var.get()

    def get(self, building_type: BuildingType) -> BuildingParams:
        return self.current().params_by_type[building_type]

    @contextlib.contextmanager
    def override(self, new_params: BuildingGenParams) -> Iterator[None]:
        token = self._var.set(new_params)
        try:
            yield
        finally:
            self._var.reset(token)
# TODO: remove later
BASE_BUILDING_GEN_PARAMS = BuildingGenParams(
    params_by_type=PARAMS_BY_TYPE,
)

BUILDING_PARAMS_PROVIDER = BuildingParamsProvider(
    base=BASE_BUILDING_GEN_PARAMS,
)


def get_params(building_type: BuildingType) -> BuildingParams:
    """
    Backward-совместимая обёртка, аналог старой функции get_params.
    Теперь берёт данные из глобального BUILDING_PARAMS_PROVIDER.
    """
    return BUILDING_PARAMS_PROVIDER.get(building_type)