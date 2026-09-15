"""Compression analysis: small-signal gain and the 1 dB compression point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from labkit.plotting import GridMajor, Legend, LinePlot, Marker, Title, XLabel, YLabel, plot
from labkit.units import Quantity, quantity as Q

from ..measurements.compression import CompressionResult
from .reference_plane import at_dut

__all__ = ["CompressionSummary", "gain", "summarize", "plot_result"]


@dataclass(frozen=True)
class CompressionSummary:
    """The numbers for one test frequency (``None`` if the sweep never compressed)."""

    frequency: Quantity
    gain_small_signal: Quantity
    p_in_1db: Optional[Quantity]
    p_out_1db: Optional[Quantity]


def gain(result: CompressionResult) -> Quantity:
    """Gain per row, ``p_out - p_in``, as a dB array."""
    return result.p_out - result.p_in  # type: ignore[no-any-return]


def _groups(result: CompressionResult) -> list[tuple[float, np.ndarray]]:
    """``[(frequency_hz, row_indices)]`` for each distinct test frequency."""
    f = np.asarray(result.frequency.to("Hz").magnitude)
    values, inverse = np.unique(f, return_inverse=True)
    return [(float(v), np.flatnonzero(inverse == i)) for i, v in enumerate(values)]


def summarize(result: CompressionResult, n_linear: int = 3) -> list[CompressionSummary]:
    """Small-signal gain (mean of the first `n_linear` points) and P1dB per frequency.

    The result is moved to the DUT plane first, so the numbers are the DUT's.
    """
    dut = at_dut(result)
    g = np.asarray(gain(dut).magnitude, dtype=np.float64)
    p_in = np.asarray(dut.p_in.to("dBm").magnitude, dtype=np.float64)
    summaries: list[CompressionSummary] = []
    for f_hz, rows in _groups(dut):
        order = rows[np.argsort(p_in[rows])]
        gi, pi = g[order], p_in[order]
        g_ss = float(np.mean(gi[:n_linear]))
        target = g_ss - 1.0
        compressed = np.flatnonzero(gi <= target)
        p_in_1db: Optional[Quantity] = None
        p_out_1db: Optional[Quantity] = None
        if compressed.size and compressed[0] > 0:
            i = int(compressed[0])
            # gain falls between points i-1 and i: interpolate the input level at exactly -1 dB
            x = float(np.interp(target, [gi[i], gi[i - 1]], [pi[i], pi[i - 1]]))
            p_in_1db = Q(x, "dBm")
            p_out_1db = Q(x + target, "dBm")
        summaries.append(
            CompressionSummary(Q(f_hz, "Hz"), Q(g_ss, "dB"), p_in_1db, p_out_1db)
        )
    return summaries


def plot_result(
    result: CompressionResult,
    show: bool = False,
    save_folder: Optional[str] = None,
    filename: Optional[str] = None,
) -> Any:
    """Output power versus input power at the DUT plane, with the P1dB point marked."""
    dut = at_dut(result)
    objects: list[Any] = []
    for summary, (_, rows) in zip(summarize(result), _groups(dut)):
        label = f"{summary.frequency.to('MHz'):~.4g}"
        objects.append(LinePlot(dut.p_in[rows], dut.p_out[rows], label=label))
        if summary.p_in_1db is not None and summary.p_out_1db is not None:
            objects.append(Marker(summary.p_in_1db, summary.p_out_1db, label=f"P1dB {label}"))
    objects += [
        Title(f"{dut.title} — compression"),
        XLabel("Input power at DUT"),
        YLabel("Output power at DUT"),
        GridMajor(),
        Legend(),
    ]
    return plot(*objects, show=show, save_folder=save_folder, filename=filename)
