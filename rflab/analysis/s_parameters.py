"""S-parameter analysis: worst-case return loss in the band and the magnitude plot."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from labkit.plotting import GridMajor, Legend, LinePlot, Title, XLabel, YLabel, plot
from labkit.units import Quantity, quantity as Q

from ..measurements.s_parameters import SParameterResult
from .reference_plane import at_dut

__all__ = ["worst_case", "plot_result"]


def worst_case(result: SParameterResult) -> dict[str, Quantity]:
    """The highest (worst) magnitude of each reflection parameter, in dB, at the DUT plane."""
    dut = at_dut(result)
    return {
        name: Q(float(np.nanmax(np.asarray(m.magnitude, dtype=np.float64))), "dB")
        for name, m in dut.magnitudes.items()
    }


def plot_result(
    result: SParameterResult,
    show: bool = False,
    save_folder: Optional[str] = None,
    filename: Optional[str] = None,
) -> Any:
    """Magnitude of every measured S-parameter versus frequency, at the DUT plane."""
    dut = at_dut(result)
    lines = [LinePlot(dut.frequency, m, label=name) for name, m in dut.magnitudes.items()]
    return plot(
        *lines,
        Title(f"{dut.title} — S-parameters"),
        XLabel("Frequency"),
        YLabel("Magnitude"),
        GridMajor(),
        Legend(),
        show=show, save_folder=save_folder, filename=filename,
    )
