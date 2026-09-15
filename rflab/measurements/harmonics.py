"""Harmonic levels with the analyzer's harmonic-distortion measurement.

Generator A drives the DUT with a CW tone at each test frequency; the
analyzer's harmonics function measures the fundamental and the first
`harmonics` harmonics. Levels are stored **raw** at the analyzer port, one row
per harmonic; :func:`rflab.analysis.at_dut` adds the output path loss
**at each harmonic's own frequency** — the case that makes interpolated loss
tables essential, since the components were rarely characterized at 3·f0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from labkit.units import Quantity, quantity as Q

from ..bench import BenchLike
from ..dut import DUT, Channel
from ._base import Result, base_fields, describe_instruments, register_result, settle

__all__ = ["HarmonicsSettings", "HarmonicsResult", "measure"]


@dataclass(frozen=True)
class HarmonicsSettings:
    """How the harmonic measurement is taken."""

    n_frequencies: int = 1
    #: Generator level for the fundamental.
    level: Quantity = Q(-10, "dBm")
    #: Number of harmonics to measure, including the fundamental (1–26).
    harmonics: int = 5
    ref_level: Quantity = Q(20, "dBm")
    settle: Quantity = Q(100, "ms")


@register_result
@dataclass(kw_only=True)
class HarmonicsResult(Result):
    """One row per (fundamental, harmonic order)."""

    measurement: ClassVar[str] = "Harmonics"

    fundamental: Quantity
    order: np.ndarray
    frequency: Quantity
    level: Quantity

    def columns(self) -> dict[str, Any]:
        return {
            "Fundamental": self.fundamental,
            "Harmonic": self.order,
            "Frequency": self.frequency,
            "Level": self.level,
        }

    @classmethod
    def from_columns(cls, meta: dict[str, Any], columns: dict[str, Any]) -> "HarmonicsResult":
        return cls(
            **meta,
            fundamental=columns["Fundamental"],
            order=np.asarray(columns["Harmonic"], dtype=np.int64),
            frequency=columns["Frequency"],
            level=columns["Level"],
        )


def measure(
    bench: BenchLike,
    dut: DUT,
    channel: Channel,
    settings: HarmonicsSettings = HarmonicsSettings(),
) -> HarmonicsResult:
    """Measure the fundamental and harmonics at each test frequency."""
    gen, sa = bench.gen_a, bench.fsv
    freqs = channel.frequencies(settings.n_frequencies)

    sa.amplitude.set_ref_level(settings.ref_level)
    sa.sweep.set_continuous(False)
    sa.measurement.enable_harmonics(True)
    sa.measurement.set_harmonic_count(settings.harmonics)

    f0_rows: list[float] = []
    order_rows: list[int] = []
    f_rows: list[float] = []
    level_rows: list[float] = []
    try:
        for f0 in freqs:
            f0_hz = float(f0.to("Hz").magnitude)
            gen.set_cw(f0, settings.level)
            settle(settings.settle)
            sa.frequency.set_center(f0)
            sa.measurement.preset_harmonics()
            sa.trigger(wait_for_completion=True)
            levels = sa.measurement.get_harmonics()
            for n, level in enumerate(levels[: settings.harmonics], start=1):
                f0_rows.append(f0_hz)
                order_rows.append(n)
                f_rows.append(n * f0_hz)
                level_rows.append(float(level))
    finally:
        gen.power.set_rf_enabled(False)
        sa.measurement.enable_harmonics(False)

    return HarmonicsResult(
        **base_fields(dut, channel, describe_instruments(gen, sa), settings),
        fundamental=Q(np.array(f0_rows), "Hz"),
        order=np.array(order_rows, dtype=np.int64),
        frequency=Q(np.array(f_rows), "Hz"),
        level=Q(np.array(level_rows), "dBm"),
    )
