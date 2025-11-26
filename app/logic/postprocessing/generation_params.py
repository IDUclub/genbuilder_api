import contextvars
import contextlib

from dataclasses import field
from typing import Dict, List, Tuple, Any, Iterator
from pydantic import BaseModel, ConfigDict, Field

from app.logic.postprocessing.service_types import ServiceType

class GenParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_run: int = 8
    ''' max_run - maximal length of living building'''
    neigh_empty_thr: int = 3
    '''neigh_empty_thr - criterion for square to be qualified as outer (e.g. 3 neighbor squares are not buildings)'''
    cell_size_m: float = 15.0
    '''cell_size_m - size of square in meters'''
    edge_share_frac: float = 0.2
    '''edge_share_frac - criterion for qualifing as direct neighbor'''
    merge_predicate: str = "intersects"
    '''merge_predicate - predicate for squares merge strategy'''
    merge_fix_eps: float = 0.0
    '''merge_fix_eps - fix for bad geometry'''
    living_area_normative: int = 18
    '''number of living area meters per person'''
    created_services: List[ServiceType] = [
        ServiceType.SCHOOL,
        ServiceType.KINDERGARTEN,
        ServiceType.POLYCLINIC,
    ]
    '''list of services supported by algorithm'''
    randomize_service_forms: bool = True
    '''randomize_service_forms - flag for randomization of service building forms (for better diversity)'''
    service_random_seed: int = 42
    '''service_random_seed - seed for randomization'''
    gap_to_houses_cheb: int = 2
    '''gap_to_houses_cheb - minimal distance between service territory and buildings'''
    gap_between_sites_cheb: int = 2
    '''gap_between_sites_cheb - minimal distance between two service territories'''
    same_type_site_gap_cheb: int = 10
    '''same_type_site_gap_cheb - minimal distance between service territories of same type to improve even distribution of services in block'''
    inner_margin_cells: int = 1
    '''inner_margin_cells - distance between border of service territory and service building'''

    service_patterns: Dict[Tuple[ServiceType, str], Dict[str, Any]] = Field(
        default_factory=lambda: {
            (ServiceType.KINDERGARTEN, "H7"): {
                "offsets": [(-1, -1), (0, -1), (1, -1), (0, 0), (-1, 1), (0, 1), (1, 1)],
                "allow_rotations": True,
                "floors": 2,
            },
            (ServiceType.KINDERGARTEN, "W5"): {
                "offsets": [(0, 0), (1, 1), (0, 2), (1, 3), (0, 4)],
                "allow_rotations": True,
                "floors": 2,
            },
            (ServiceType.KINDERGARTEN, "LINE3"): {
                "offsets": [(0, 0), (0, 1), (0, 2)],
                "allow_rotations": True,
                "floors": 2,
            },

            (ServiceType.POLYCLINIC, "RECT_2x4"): {
                "offsets": [(r, c) for r in range(2) for c in range(4)],
                "allow_rotations": True,
                "floors": 4,
            },

            (ServiceType.SCHOOL, "H_5x4"): {
                "offsets": ([(r, 0) for r in range(5)]
                            + [(r, 3) for r in range(5)]
                            + [(2, c) for c in range(4)]),
                "allow_rotations": True,
                "floors": 3,
            },
            (ServiceType.SCHOOL, "RING_5x5_WITH_COURTYARD"): {
                "offsets": [
                    (r, c)
                    for r in range(5)
                    for c in range(5)
                    if (r in {0, 4} or c in {0, 4}) and not (r in {0, 4} and c in {0, 4})
                ],
                "allow_rotations": False,
                "floors": 3,
            },
            (ServiceType.SCHOOL, "RECT_5x2_WITH_OPEN_3"): {
                "offsets": ([(1, c) for c in range(5)] + [(0, 0), (0, 4)]),
                "allow_rotations": True,
                "floors": 3,
            },
        },
    )
    '''service_patterns - geometry for services'''

    service_site_rules: Dict[Tuple[ServiceType, str], Dict[str, float | int]] = field(
        default_factory=lambda: {
            (ServiceType.SCHOOL, "RECT_5x2_WITH_OPEN_3"): {
                "capacity": 600,
                "site_area_m2": 33000.0,
            },
            (ServiceType.SCHOOL, "H_5x4"): {
                "capacity": 800,
                "site_area_m2": 36000.0,
            },
            (ServiceType.SCHOOL, "RING_5x5_WITH_COURTYARD"): {
                "capacity": 1100,
                "site_area_m2": 39600.0,
            },
            (ServiceType.KINDERGARTEN, "LINE3"): {
                "capacity": 60,
                "site_area_m2": 2640.0,
            },
            (ServiceType.KINDERGARTEN, "W5"): {
                "capacity": 100,
                "site_area_m2": 4400.0,
            },
            (ServiceType.KINDERGARTEN, "H7"): {
                "capacity": 150,
                "site_area_m2": 5700.0,
            },
            (ServiceType.POLYCLINIC, "RECT_2x4"): {
                "capacity": 300,
                "site_area_m2": 3000.0,
            },
        }
    )
    '''service_site_rules - mapping of capacity and area of territory for each type of service building'''

    svc_order: List[ServiceType] = field(
        default_factory=lambda: [
            ServiceType.SCHOOL,
            ServiceType.KINDERGARTEN,
            ServiceType.POLYCLINIC,
        ]
    )
    '''svc_order - priority for service generation (between types)'''
    zone_id_col: str = "zone_id"
    '''zone_id_col - name of zone id column'''
    zone_name_col: str = "zone"
    '''zone_name_col - name of zone type column'''
    verbose: bool = True
    '''verbose - if True, prints stats for generation results'''
    INNER_BORDER: int = 3
    MAX_COVERAGE: float = 0.9
    rectangle_finder_step: int = 5
    minimal_rectangle_side: int = 40
    jobs_number: int = 5
    la_per_person: int = 18
    max_service_attempts: int = 200
    max_sites_per_service_per_block: int = 10

    def patched(self, patch: Dict[str, Any]) -> "GenParams":
        def deep_merge(a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                c = dict(a)
                for k, v in b.items():
                    c[k] = deep_merge(c.get(k), v)
                return c
            return b if b is not None else a

        data = self.model_dump()
        merged = deep_merge(data, patch)
        return self.__class__.model_validate(merged)
    

class ParamsProvider:
    def __init__(self, base: GenParams):
        self._var: contextvars.ContextVar[GenParams] = contextvars.ContextVar("gen_params", default=base)

    def current(self) -> GenParams:
        return self._var.get()

    @contextlib.contextmanager
    def override(self, new_params: GenParams) -> Iterator[None]:
        token = self._var.set(new_params)
        try:
            yield
        finally:
            self._var.reset(token)