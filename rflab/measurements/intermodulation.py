"""Two-tone intermodulation: tone and IM3 product levels.

Generators A and B drive the DUT with two equal tones `tone_spacing` apart,
centred on each test frequency; the analyzer reads the two tones and the two
third-order products (2·f1 − f2 and 2·f2 − f1) with a marker, and its own
third-order-intercept function is read as a cross-check. Levels are stored
**raw** at the analyzer port; :mod:`rflab.analysis.intermodulation` moves them
to the DUT output and computes OIP3 / IIP3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from labkit.units import Quantity, quantity as Q

from ..bench import BenchLike
from ..dut import DUT, Channel
from ._base import Result, base_fields, describe_instruments, register_result, settle

__all__ = ["IntermodulationSettings", "IntermodulationResult", "measure", "TONES"]

#: The four spectral lines read at each centre frequency, in row order.
TONES = ("f1", "f2", "im3_low", "im3_high")


@dataclass(frozen=True)
class IntermodulationSettings:
    """How the two-tone measurement is taken."""

    n_frequencies: int = 1
    #: Level of each tone at the generators.
    level: Quantity = Q(-10, "dBm")
    tone_spacing: Quantity = Q(1, "MHz")
    span: Quantity = Q(10, "MHz")
    rbw: Quantity = Q(10, "kHz")
    ref_level: Quantity = Q(20, "dBm")
    settle: Quantity = Q(100, "ms")


@register_result
@dataclass(kw_only=True)
class IntermodulationResult(Result):
    """One row per (centre frequency, spectral line)."""

    measurement: ClassVar[str] = "Intermodulation"

    center: Quantity
    tone: np.ndarray
    frequency: Quantity
    level: Quantity
    #: The analyzer's own TOI result per row (repeated for the centre's rows).
    toi_instrument: Quantity

    def columns(self) -> dict[str, Any]:
        return {
            "Center": self.center,
            "Tone": self.tone,
            "Frequency": self.frequency,
            "Level": self.level,
            "TOI instrument": self.toi_instrument,
        }

    @classmethod
    def from_columns(cls, meta: dict[str, Any], columns: dict[str, Any]) -> "IntermodulationResult":
        return cls(
            **meta,
            center=columns["Center"],
            tone=np.asarray(columns["Tone"], dtype=object),
            frequency=columns["Frequency"],
            level=columns["Level"],
            toi_instrument=columns["TOI instrument"],
        )


def measure(
    bench: BenchLike,
    dut: DUT,
    channel: Channel,
    settings: IntermodulationSettings = IntermodulationSettings(),
) -> IntermodulationResult:
    """Drive two tones and read tones and IM3 products at each centre frequency."""
    gen_a, gen_b, sa = bench.gen_a, bench.gen_b, bench.fsv
    freqs = channel.frequencies(settings.n_frequencies)
    half = settings.tone_spacing / 2

    sa.frequency.set_span(settings.span)
    sa.bandwidth.set_rbw(settings.rbw)
    sa.amplitude.set_ref_level(settings.ref_level)
    sa.sweep.set_continuous(False)
    marker = sa.marker(1, enable=True)

    center_rows: list[float] = []
    tone_rows: list[str] = []
    f_rows: list[float] = []
    level_rows: list[float] = []
    toi_rows: list[float] = []
    try:
        for f0 in freqs:
            f1, f2 = f0 - half, f0 + half
            lines = {"f1": f1, "f2": f2, "im3_low": 2 * f1 - f2, "im3_high": 2 * f2 - f1}
            gen_a.set_cw(f1, settings.level)
            gen_b.frequency.set_frequency(f2)
            gen_b.output.set_level(settings.level)
            gen_b.output.set_rf_enabled(True)
            settle(settings.settle)

            sa.frequency.set_center(f0)
            sa.measurement.enable_toi(True)
            sa.trigger(wait_for_completion=True)
            toi = sa.measurement.get_toi()
            for tone in TONES:
                f = lines[tone]
                marker.set_x(f)
                level = marker.get_y()
                center_rows.append(float(f0.to("Hz").magnitude))
                tone_rows.append(tone)
                f_rows.append(float(f.to("Hz").magnitude))
                level_rows.append(float(level.to("dBm").magnitude))
                toi_rows.append(float(toi))
    finally:
        gen_a.power.set_rf_enabled(False)
        gen_b.output.set_rf_enabled(False)
        sa.measurement.enable_toi(False)

    return IntermodulationResult(
        **base_fields(dut, channel, describe_instruments(gen_a, gen_b, sa), settings),
        center=Q(np.array(center_rows), "Hz"),
        tone=np.array(tone_rows, dtype=object),
        frequency=Q(np.array(f_rows), "Hz"),
        level=Q(np.array(level_rows), "dBm"),
        toi_instrument=Q(np.array(toi_rows), "dBm"),
    )
