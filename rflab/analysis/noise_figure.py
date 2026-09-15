"""Noise-figure analysis: band statistics and the NF / gain plot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from labkit.plotting import GridMajor, Legend, LinePlot, Title, XLabel, YLabel, plot
from labkit.units import Quantity, quantity as Q

from ..measurements.noise_figure import NoiseFigureResult
from .reference_plane import at_dut

__all__ = ["NoiseFigureSummary", "summarize", "plot_result"]


@dataclass(frozen=True)
class NoiseFigureSummary:
    nf_mean: Quantity
    nf_max: Quantity
    gain_mean: Quantity
    gain_min: Quantity


def summarize(result: NoiseFigureResult) -> NoiseFigureSummary:
    """Mean and worst-case noise figure and gain across the band (in dB)."""
    dut = at_dut(result)
    nf = np.asarray(dut.noise_figure.magnitude, dtype=np.float64)
    g = np.asarray(dut.gain.magnitude, dtype=np.float64)
    return NoiseFigureSummary(
        Q(float(np.nanmean(nf)), "dB"), Q(float(np.nanmax(nf)), "dB"),
        Q(float(np.nanmean(g)), "dB"), Q(float(np.nanmin(g)), "dB"),
    )


def plot_result(
    result: NoiseFigureResult,
    show: bool = False,
    save_folder: Optional[str] = None,
    filename: Optional[str] = None,
) -> Any:
    """Noise figure (left axis) and gain (right axis) versus frequency."""
    dut = at_dut(result)
    return plot(
        LinePlot(dut.frequency, dut.noise_figure, label="Noise figure"),
        LinePlot(dut.frequency, dut.gain, label="Gain", y_axis="right"),
        Title(f"{dut.dut} {dut.version} {dut.channel_label} — noise figure"),
        XLabel("Frequency"),
        YLabel("Noise figure"),
        YLabel("Gain", y_axis="right"),
        GridMajor(),
        Legend(),
        show=show, save_folder=save_folder, filename=filename,
    )
