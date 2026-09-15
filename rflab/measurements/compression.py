"""Gain compression: output power versus input power at one or more frequencies.

Generator A drives the DUT input with a CW tone stepped from `p_start` to
`p_stop`; the spectrum analyzer, in a narrow span around the tone, reads the
peak level after each step. Both the generator setting (``p_in``) and the
analyzer reading (``p_out``) are stored **raw**, at the instrument reference
planes; :func:`rflab.analysis.at_dut` moves them to the DUT ports using the
channel's recorded input and output paths, and
:mod:`rflab.analysis.compression` finds the 1 dB compression point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from labkit.units import Quantity, quantity as Q

from ..bench import BenchLike
from ..dut import DUT, Channel
from ._base import Result, base_fields, describe_instruments, register_result, settle

__all__ = ["CompressionSettings", "CompressionResult", "measure", "levels"]


@dataclass(frozen=True)
class CompressionSettings:
    """How a compression sweep is taken."""

    #: Number of test frequencies across the channel (1 = the band centre only).
    n_frequencies: int = 1
    p_start: Quantity = Q(-30, "dBm")
    p_stop: Quantity = Q(0, "dBm")
    p_step: Quantity = Q(1, "dB")
    #: Analyzer span around the tone, and resolution bandwidth.
    span: Quantity = Q(1, "MHz")
    rbw: Quantity = Q(10, "kHz")
    #: Analyzer reference level (must exceed the largest expected output).
    ref_level: Quantity = Q(20, "dBm")
    #: Pause after each generator step before the analyzer sweeps.
    settle: Quantity = Q(50, "ms")


def levels(settings: CompressionSettings) -> Quantity:
    """The generator levels of the sweep, `p_start` to `p_stop` in `p_step`."""
    start = settings.p_start.to("dBm").magnitude
    stop = settings.p_stop.to("dBm").magnitude
    step = settings.p_step.to("dB").magnitude
    return Q(np.arange(start, stop + step / 2, step), "dBm")


@register_result
@dataclass(kw_only=True)
class CompressionResult(Result):
    """Raw compression data: one row per (frequency, input level)."""

    measurement: ClassVar[str] = "Compression"

    frequency: Quantity
    p_in: Quantity
    p_out: Quantity

    def columns(self) -> dict[str, Any]:
        return {"Frequency": self.frequency, "P_in": self.p_in, "P_out": self.p_out}

    @classmethod
    def from_columns(cls, meta: dict[str, Any], columns: dict[str, Any]) -> "CompressionResult":
        return cls(**meta, frequency=columns["Frequency"], p_in=columns["P_in"], p_out=columns["P_out"])


def measure(
    bench: BenchLike,
    dut: DUT,
    channel: Channel,
    settings: CompressionSettings = CompressionSettings(),
) -> CompressionResult:
    """Sweep generator A's level and read the DUT output on the analyzer."""
    gen, sa = bench.gen_a, bench.fsv
    freqs = channel.frequencies(settings.n_frequencies)
    p_levels = levels(settings)

    sa.frequency.set_span(settings.span)
    sa.bandwidth.set_rbw(settings.rbw)
    sa.amplitude.set_ref_level(settings.ref_level)
    sa.sweep.set_continuous(False)
    marker = sa.marker(1, enable=True)

    f_rows: list[float] = []
    p_in_rows: list[float] = []
    p_out_rows: list[float] = []
    try:
        for f in freqs:
            sa.frequency.set_center(f)
            gen.set_cw(f, p_levels[0])
            for p in p_levels:
                gen.power.set_level(p)
                settle(settings.settle)
                sa.trigger(wait_for_completion=True)
                marker.peak_search()
                p_out = marker.get_y()
                f_rows.append(float(f.to("Hz").magnitude))
                p_in_rows.append(float(p.to("dBm").magnitude))
                p_out_rows.append(float(p_out.to("dBm").magnitude))
    finally:
        gen.power.set_rf_enabled(False)

    return CompressionResult(
        **base_fields(dut, channel, describe_instruments(gen, sa), settings),
        frequency=Q(np.array(f_rows), "Hz"),
        p_in=Q(np.array(p_in_rows), "dBm"),
        p_out=Q(np.array(p_out_rows), "dBm"),
    )
