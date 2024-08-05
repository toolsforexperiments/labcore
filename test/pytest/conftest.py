import pytest

from labcore.measurement.record import record_as
from labcore.measurement.sweep import sweep_parameter

@pytest.fixture
def short_sweep():
    sweep = sweep_parameter('x', range(10), record_as(lambda x: x*2, 'y'))
    return sweep


@pytest.fixture()
def long_sweep():
    sweep = sweep_parameter('a', range(10000), record_as(lambda a: a*2, 'b'), record_as(lambda a: a**2, 'c'))
    return sweep

