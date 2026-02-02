from pathlib import Path
import contextvars
import contextlib

from typing import Dict, Any, Iterator
from pydantic import BaseModel, ConfigDict


class GenParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    INNER_BORDER: int = 3
    """width of inner plot border restricted for construction"""
    MAX_COVERAGE: float = 0.9
    """control parameter for building generation"""
    rectangle_finder_step: int = 5
    """parameter for maximum rectangle inscription, lower - higher quality and lower speed"""
    minimal_rectangle_side: int = 40
    """minimal side length for rectangle segments"""
    jobs_number: int = 5
    """jobs number for MIR"""
    la_per_person: int = 18
    """normative square meters for person, used in service deman calculation"""
    max_service_attempts: int = 200
    """limit for attempts for service placement"""
    max_sites_per_service_per_block: int = 10
    """limit for number of service of one type in block"""
    physical_objects_exclusion_buffer_m: float = 5.0
    """Buffer (in meters) around physical objects that will be excluded from generation."""
    physical_objects_exclusion_dynamic: bool = True
    """If True, buffer is chosen per physical object based on its geometry size and building_params."""
    physical_objects_exclusion_min_buffer_m: float = 3.0
    """Lower bound for dynamic buffer (in meters)."""
    physical_objects_exclusion_max_buffer_m: float = 60.0
    """Upper bound for dynamic buffer (in meters)."""
    service_projects_file: str = str( Path(__file__).resolve().parent / "service_projects.geojson")
    """path to service projects file with geometry and plot/building parameters"""

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
        self._var: contextvars.ContextVar[GenParams] = contextvars.ContextVar(
            "gen_params", default=base
        )

    def current(self) -> GenParams:
        return self._var.get()

    @contextlib.contextmanager
    def override(self, new_params: GenParams) -> Iterator[None]:
        token = self._var.set(new_params)
        try:
            yield
        finally:
            self._var.reset(token)
