"""Analysis layer: reference-plane correction, derived numbers, plots.

Everything here works on a :class:`~rflab.measurements.Result` — fresh from a
measurement or loaded from disk with :class:`rflab.store.ResultStore` — and
never touches an instrument. Start with :func:`at_dut`, which moves a raw result
to the DUT ports using the recorded signal paths; each measurement's module
then summarizes and plots the DUT-plane data.
"""

from __future__ import annotations

from typing import Any, Callable

from . import compression, harmonics, intermodulation, noise_figure, s_parameters
from .reference_plane import at_dut

__all__ = [
    "at_dut",
    "compression",
    "noise_figure",
    "s_parameters",
    "harmonics",
    "intermodulation",
    "SUMMARIZERS",
    "PLOTTERS",
]

#: ``measurement name -> summarize(result)``.
SUMMARIZERS: dict[str, Callable[[Any], Any]] = {
    "Compression": compression.summarize,
    "Noise Figure": noise_figure.summarize,
    "S-Parameters": s_parameters.worst_case,
    "Harmonics": harmonics.summarize,
    "Intermodulation": intermodulation.summarize,
}

#: ``measurement name -> plot_result(result, show=, save_folder=, filename=)``.
PLOTTERS: dict[str, Callable[..., Any]] = {
    "Compression": compression.plot_result,
    "Noise Figure": noise_figure.plot_result,
    "S-Parameters": s_parameters.plot_result,
    "Harmonics": harmonics.plot_result,
    "Intermodulation": intermodulation.plot_result,
}
