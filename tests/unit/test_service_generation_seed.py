"""Service placement is the only random step of generation — a fixed seed must
make it repeatable."""
import random

from shapely.geometry import box

from app.logic.generation_params import GenParams, ParamsProvider
from app.logic.service_generation import ServiceGenerator


def _sample(generator, rng):
    return generator._sample_rect_in_polygon(box(0, 0, 200, 150), (20.0, 40.0), (15.0, 30.0), rng=rng)


def test_same_seed_gives_same_rectangle():
    generator = ServiceGenerator(ParamsProvider(GenParams()))

    first = _sample(generator, random.Random(42))
    second = _sample(generator, random.Random(42))

    assert first is not None
    assert first.equals_exact(second, 0)


def test_seed_is_a_generation_parameter():
    assert GenParams().seed is None
    assert GenParams().patched({"seed": 42}).seed == 42
