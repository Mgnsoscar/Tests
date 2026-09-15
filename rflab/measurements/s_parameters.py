"""Reflection (S11, S22) on the vector network analyzer.

One trace per requested S-parameter is created on channel 1 of the VNA, the
band is swept once, and the complex data is stored as magnitude (dB) and
phase (degrees) per frequency, **raw** at the VNA ports. With a path on a
port, :func:`rflab.analysis.at_dut` adds twice the path loss to the reflection
magnitude (the wave passes the path going in and coming back).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from labkit.units import Quantity, quantity as Q

from ..bench import BenchLike
from ..dut import DUT, Channel
from ._base import Result, base_fields, describe_instruments, register_result

__all__ = ["SParameterSettings", "SParameterResult", "measure"]

_PHASE_SUFFIX = " phase"


@dataclass(frozen=True)
class SParameterSettings:
    """How the reflection sweep is taken."""

    points: int = 201
    if_bandwidth: Quantity = Q(1, "kHz")
    power: Quantity = Q(-10, "dBm")
    #: Sweep averaging count (1 = none).
    average_count: int = 1
    parameters: tuple[str, ...] = ("S11", "S22")


@register_result
@dataclass(kw_only=True)
class SParameterResult(Result):
    """Magnitude (dB) and phase (deg) of each S-parameter per frequency."""

    measurement: ClassVar[str] = "S-Parameters"

    frequency: Quantity
    magnitudes: dict[str, Quantity]
    phases: dict[str, Quantity]

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(self.magnitudes)

    def columns(self) -> dict[str, Any]:
        cols: dict[str, Any] = {"Frequency": self.frequency}
        for name in self.parameters:
            cols[name] = self.magnitudes[name]
            cols[name + _PHASE_SUFFIX] = self.phases[name]
        return cols

    @classmethod
    def from_columns(cls, meta: dict[str, Any], columns: dict[str, Any]) -> "SParameterResult":
        names = [k for k in columns if k != "Frequency" and not k.endswith(_PHASE_SUFFIX)]
        return cls(
            **meta,
            frequency=columns["Frequency"],
            magnitudes={n: columns[n] for n in names},
            phases={n: columns[n + _PHASE_SUFFIX] for n in names},
        )


def measure(
    bench: BenchLike,
    dut: DUT,
    channel: Channel,
    settings: SParameterSettings = SParameterSettings(),
) -> SParameterResult:
    """Sweep the band and read each requested S-parameter as complex data."""
    vna = bench.vna
    vna.trace.delete_all()
    vna.frequency.set_start(channel.f_start)
    vna.frequency.set_stop(channel.f_stop)
    vna.sweep.set_points(settings.points)
    vna.bandwidth.set_if_bandwidth(settings.if_bandwidth)
    vna.power.set_power(settings.power)
    if settings.average_count > 1:
        vna.average.set_state(True)
        vna.average.set_count(settings.average_count)
    else:
        vna.average.set_state(False)
    vna.display.set_window_state(1, True)

    frequency: Quantity | None = None
    magnitudes: dict[str, Quantity] = {}
    phases: dict[str, Quantity] = {}
    for index, parameter in enumerate(settings.parameters, start=1):
        name = f"Trc{index}"
        vna.trace.create(name, parameter)  # type: ignore[arg-type]
        vna.display.feed_trace(1, index, name)
        vna.trace.select(name)
        stimulus, values = vna.measure()
        frequency = stimulus
        magnitudes[parameter] = Q(20 * np.log10(np.abs(values)), "dB")
        phases[parameter] = Q(np.degrees(np.angle(values)), "deg")
    vna.power.set_output(False)

    assert frequency is not None, "at least one S-parameter must be requested"
    return SParameterResult(
        **base_fields(dut, channel, describe_instruments(vna), settings),
        frequency=frequency,
        magnitudes=magnitudes,
        phases=phases,
    )
