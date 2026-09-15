from __future__ import annotations

import pytest

from rflab.dut import DUT
from rflab.duts import amplifier_x
from rflab.simulation import AmplifierModel, SimulatedBench, simulated_bench


@pytest.fixture
def model() -> AmplifierModel:
    return AmplifierModel()


@pytest.fixture
def bench(model: AmplifierModel) -> SimulatedBench:
    return simulated_bench(model)


@pytest.fixture
def dut() -> DUT:
    return amplifier_x.DUT
