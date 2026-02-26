from copy import deepcopy
from typing import Any


DEFAULT_BLOCK_GENERATION_PARAMETERS: dict[str, Any] = {
    "rectangle_finder_step": 5,
}

#default_generation parameters targeted to maximize resident numbers
DEFAULT_BLOCK_TARGETS_BY_ZONE: dict[str, Any] = {
    "coverage_area": {
        "business": 10000,
        "industrial": 20000,
        "special": 10000,
        "transport": 10000,
        "unknown": 10000,
    },
    "default_floor_group": {
        "business": "extreme",
        "residential": "extreme",
        "unknown": "extreme",
    },
    "density_scenario": {
        "business": "min",
        "residential": "min",
        "unknown": "min",
    },
    "floors_avg": {
        "business": 19,
        "industrial": 19,
        "special": 19,
        "transport": 19,
        "unknown": 19,
    },
    "residents": {
        "business": 50000,
        "residential": 50000,
        "unknown": 50000,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge dictionaries. Override has priority."""
    if not override:
        return base

    result = deepcopy(base)

    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v

    return result
