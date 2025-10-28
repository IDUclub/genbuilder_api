from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class GenParams:
    '''
    max_run - maximal length of living building
    neigh_empty_thr - criterion for square to be qualified as outer (e.g. 3 neighbor squares are not buildings)
    cell_size_m - size of square in meters
    edge_share_frac - criterion for qualifing as direct neighbor
    merge_predicate - predicate for squares merge strategy
    merge_fix_eps - fix for bad geometry
    max_services_per_zone - number of service building per zone
    randomize_service_forms - flag for randomization of service building forms (for better diversity)
    service_random_seed - seed for randomization
    gap_to_houses_cheb - minimal distance between service territory and buildings
    gap_between_sites_cheb - minimal distance between two service territories
    same_type_site_gap_cheb - minimal distance between service territories of same type to improve even distribution of services in block
    inner_margin_cells - distance between border of service territory and service building
    service_site_rules - mapping of capacity and area of territory for each type of service building
    svc_order - priority for service generation (between types)
    zone_id_col - name of zone id column
    zone_name_col - name of zone type column
    verbose - if True, prints stats for generation results
    '''
    max_run: int = 8
    neigh_empty_thr: int = 3
    cell_size_m: float = 15.0
    edge_share_frac: float = 0.2
    merge_predicate: str = "intersects"
    merge_fix_eps: float = 0.0

    max_services_per_zone: Dict[str, int] = field(
        default_factory=lambda: {"school": 3, "kindergarten": 5, "polyclinics": 1}
    )
    randomize_service_forms: bool = True
    service_random_seed: int = 42
    gap_to_houses_cheb: int = 2
    gap_between_sites_cheb: int = 2
    same_type_site_gap_cheb: int = 10
    inner_margin_cells: int = 1

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

    svc_order: List[str] = field(default_factory=lambda: ["school", "kindergarten", "polyclinics"])
    zone_id_col: str = "zone_id"
    zone_name_col: str = "zone"
    verbose: bool = True