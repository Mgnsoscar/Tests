"""Acquisition layer: one module per measurement, each with ``measure()``.

Every ``measure(bench, dut, channel, settings)`` talks to the instruments and
returns a :class:`~rflab.measurements.Result` holding raw data and the context
needed to interpret it. Nothing here writes files or plots.
"""

from __future__ import annotations

from . import compression, harmonics, intermodulation, noise_figure, s_parameters
from ._base import RESULT_TYPES, Result
from .compression import CompressionResult, CompressionSettings
from .harmonics import HarmonicsResult, HarmonicsSettings
from .intermodulation import IntermodulationResult, IntermodulationSettings
from .noise_figure import NoiseFigureResult, NoiseFigureSettings
from .s_parameters import SParameterResult, SParameterSettings

__all__ = [
    "Result",
    "RESULT_TYPES",
    "compression",
    "noise_figure",
    "s_parameters",
    "harmonics",
    "intermodulation",
    "CompressionResult",
    "CompressionSettings",
    "NoiseFigureResult",
    "NoiseFigureSettings",
    "SParameterResult",
    "SParameterSettings",
    "HarmonicsResult",
    "HarmonicsSettings",
    "IntermodulationResult",
    "IntermodulationSettings",
]
