"""S-parameter analysis: worst-case return loss inside the channel band, and the plot."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from labkit.plotting import GridMajor, Legend, LinePlot, Marker, Title, XLabel, YLabel, plot
from labkit.units import Quantity, quantity as Q

from ..measurements.s_parameters import SParameterResult
from .reference_plane import at_dut

__all__ = ["worst_case", "plot_result"]


def _in_band(result: SParameterResult) -> np.ndarray:
    """Mask of the sweep points inside the channel band (all points if the band is unknown)."""
    f = np.asarray(result.frequency.to("Hz").magnitude, dtype=np.float64)
    band = result.band
    if band is None:
        return np.ones(f.shape, dtype=bool)
    lo, hi = band[0].to("Hz").magnitude, band[1].to("Hz").magnitude
    mask: np.ndarray = (f >= lo) & (f <= hi)
    return mask if mask.any() else np.ones(f.shape, dtype=bool)


def worst_case(result: SParameterResult) -> dict[str, Quantity]:
    """The highest (worst) magnitude of each reflection parameter inside the channel band, at the DUT plane."""
    dut = at_dut(result)
    mask = _in_band(dut)
    return {
        name: Q(float(np.nanmax(np.asarray(m.magnitude, dtype=np.float64)[mask])), "dB")
        for name, m in dut.magnitudes.items()
    }


def plot_result(
    result: SParameterResult,
    show: bool = False,
    save_folder: Optional[str] = None,
    filename: Optional[str] = None,
) -> Any:
    """Magnitude of every measured S-parameter versus frequency at the DUT plane, band edges marked."""
    dut = at_dut(result)
    objects: list[Any] = [LinePlot(dut.frequency, m, label=name) for name, m in dut.magnitudes.items()]
    band = dut.band
    if band is not None:
        worst = worst_case(result)
        for name, level in worst.items():
            f = np.asarray(dut.frequency.to("Hz").magnitude, dtype=np.float64)
            m = np.asarray(dut.magnitudes[name].magnitude, dtype=np.float64)
            mask = _in_band(dut)
            i = int(np.flatnonzero(mask)[np.nanargmax(m[mask])])
            objects.append(Marker(dut.frequency[i], dut.magnitudes[name][i], label=f"worst {name} in band"))
    objects += [
        Title(f"{dut.title} — S-parameters"),
        XLabel("Frequency"),
        YLabel("Magnitude"),
        GridMajor(),
        Legend(),
    ]
    return plot(*objects, show=show, save_folder=save_folder, filename=filename)
