from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class GenParams:
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

    max_services_per_zone: Dict[str, int] = field(
        default_factory=lambda: {"school": 3, "kindergarten": 5, "polyclinics": 1}
    )
    '''max_services_per_zone - number of service building per zone'''
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

    service_site_rules: Dict[Tuple[str, str], Dict[str, float | int]] = field(
        default_factory=lambda: {
            ("school", "RECT_5x2_WITH_OPEN_3"): {"capacity": 600, "site_area_m2": 33000.0},
            ("school", "H_5x4"): {"capacity": 800, "site_area_m2": 36000.0},
            ("school", "RING_5x5_WITH_COURTYARD"): {"capacity": 1100, "site_area_m2": 39600.0},
            ("kindergarten", "LINE3"): {"capacity": 60, "site_area_m2": 2640.0},
            ("kindergarten", "W5"): {"capacity": 100, "site_area_m2": 4400.0},
            ("kindergarten", "H7"): {"capacity": 150, "site_area_m2": 5700.0},
            ("polyclinics", "RECT_2x4"): {"capacity": 300, "site_area_m2": 3000.0},
        }
    )
    '''service_site_rules - mapping of capacity and area of territory for each type of service building'''

    svc_order: List[str] = field(default_factory=lambda: ["school", "kindergarten", "polyclinics"])
    '''svc_order - priority for service generation (between types)'''
    zone_id_col: str = "zone_id"
    '''zone_id_col - name of zone id column'''
    zone_name_col: str = "zone"
    '''zone_name_col - name of zone type column'''
    verbose: bool = True
    '''verbose - if True, prints stats for generation results'''