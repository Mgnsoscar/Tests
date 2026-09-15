"""Harmonics analysis: each harmonic relative to its fundamental, and the plot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from labkit.plotting import GridMajor, Legend, LinePlot, Title, XLabel, YLabel, plot
from labkit.units import Quantity, quantity as Q

from ..measurements.harmonics import HarmonicsResult
from .reference_plane import at_dut

__all__ = ["HarmonicLevel", "summarize", "plot_result"]


@dataclass(frozen=True)
class HarmonicLevel:
    fundamental: Quantity
    order: int
    frequency: Quantity
    level: Quantity
    #: Level relative to the fundamental (dBc, negative below it).
    relative: Quantity


def summarize(result: HarmonicsResult) -> list[HarmonicLevel]:
    """Every harmonic's absolute level and its level relative to the fundamental, at the DUT plane."""
    dut = at_dut(result)
    f0 = np.asarray(dut.fundamental.to("Hz").magnitude, dtype=np.float64)
    level = np.asarray(dut.level.to("dBm").magnitude, dtype=np.float64)
    order = np.asarray(dut.order, dtype=np.int64)
    freq = np.asarray(dut.frequency.to("Hz").magnitude, dtype=np.float64)
    out: list[HarmonicLevel] = []
    for i in range(len(order)):
        same = (f0 == f0[i]) & (order == 1)
        fundamental_level = float(level[same][0]) if same.any() else float("nan")
        out.append(
            HarmonicLevel(
                Q(f0[i], "Hz"), int(order[i]), Q(freq[i], "Hz"),
                Q(float(level[i]), "dBm"), Q(float(level[i]) - fundamental_level, "dB"),
            )
        )
    return out


def plot_result(
    result: HarmonicsResult,
    show: bool = False,
    save_folder: Optional[str] = None,
    filename: Optional[str] = None,
) -> Any:
    """Harmonic level versus order, one line per fundamental, at the DUT plane."""
    dut = at_dut(result)
    f0 = np.asarray(dut.fundamental.to("Hz").magnitude, dtype=np.float64)
    lines: list[Any] = []
    for value in np.unique(f0):
        rows = np.flatnonzero(f0 == value)
        lines.append(
            LinePlot(dut.order[rows].astype(float), dut.level[rows], label=f"{Q(value, 'Hz').to('MHz'):~.4g}")
        )
    return plot(
        *lines,
        Title(f"{dut.dut} {dut.version} {dut.channel_label} — harmonics"),
        XLabel("Harmonic order", show_unit=False),
        YLabel("Level at DUT output"),
        GridMajor(),
        Legend(),
        show=show, save_folder=save_folder, filename=filename,
    )
