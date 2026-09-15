"""Compression analysis: small-signal gain and the 1 dB compression point, robustly.

The small-signal gain is **not** taken from the first few points. Points too
close to the noise floor (recorded with the result) are dropped; the gain of
the remaining points is smoothed with a moving median so one bad reading
cannot set the reference; the reference is the plateau of that smoothed
curve — every point within `plateau_tolerance` of its maximum — and the
small-signal gain is the median of the raw gains over the plateau. The 1 dB
point is the first crossing of ``gain <= small-signal gain − 1 dB`` after the
plateau, interpolated between the two neighbouring points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from labkit.plotting import GridMajor, Legend, LinePlot, Marker, Title, XLabel, YLabel, plot
from labkit.units import Quantity, quantity as Q

from ..measurements.compression import CompressionResult
from .reference_plane import at_dut

__all__ = ["CompressionSummary", "gain", "summarize", "plot_result"]

_MEDIAN_WINDOW = 3


@dataclass(frozen=True)
class CompressionSummary:
    """The numbers for one test frequency (``None`` if the sweep never compressed)."""

    frequency: Quantity
    gain_small_signal: Quantity
    p_in_1db: Optional[Quantity]
    p_out_1db: Optional[Quantity]
    #: Points that were far enough above the noise floor to be used.
    points_used: int
    #: Points dropped for being too close to the noise floor.
    points_dropped: int


def gain(result: CompressionResult) -> Quantity:
    """Gain per row, ``p_out - p_in``, as a dB array."""
    return result.p_out - result.p_in  # type: ignore[no-any-return]


def _groups(result: CompressionResult) -> list[tuple[float, np.ndarray]]:
    """``[(frequency_hz, row_indices)]`` for each distinct test frequency."""
    f = np.asarray(result.frequency.to("Hz").magnitude, dtype=np.float64)
    values, inverse = np.unique(f, return_inverse=True)
    return [(float(v), np.flatnonzero(inverse == i)) for i, v in enumerate(values)]


def _moving_median(values: np.ndarray, window: int = _MEDIAN_WINDOW) -> np.ndarray:
    """Causal moving median: element i is the median of the last `window` values up to i.

    The first ``window - 1`` elements use the first full window, so a wild
    first reading is outvoted rather than standing alone.
    """
    if len(values) <= window:
        return np.full(len(values), np.median(values))
    return np.array(
        [np.median(values[max(0, min(i, len(values) - 1) - window + 1): max(i, window - 1) + 1])
         for i in range(len(values))]
    )


def summarize(
    result: CompressionResult,
    plateau_tolerance: Quantity = Q(0.2, "dB"),
    noise_margin: Optional[Quantity] = None,
) -> list[CompressionSummary]:
    """Small-signal gain (plateau of the smoothed gain) and P1dB per frequency, at the DUT plane."""
    dut = at_dut(result)
    valid_all = dut.valid(noise_margin)
    g_all = np.asarray(gain(dut).magnitude, dtype=np.float64)
    p_in_all = np.asarray(dut.p_in.to("dBm").magnitude, dtype=np.float64)
    tolerance = float(plateau_tolerance.to("dB").magnitude)

    summaries: list[CompressionSummary] = []
    for f_hz, rows in _groups(dut):
        rows = rows[np.argsort(p_in_all[rows])]
        used = rows[valid_all[rows]]
        dropped = int(len(rows) - len(used))
        if len(used) == 0:
            summaries.append(CompressionSummary(Q(f_hz, "Hz"), Q(float("nan"), "dB"), None, None, 0, dropped))
            continue
        g, p_in = g_all[used], p_in_all[used]
        smoothed = _moving_median(g)
        plateau = np.flatnonzero(smoothed >= smoothed.max() - tolerance)
        g_ss = float(np.median(g[plateau]))
        target = g_ss - 1.0
        crossing = [i for i in range(int(plateau[-1]) + 1, len(g)) if g[i] <= target]
        p_in_1db: Optional[Quantity] = None
        p_out_1db: Optional[Quantity] = None
        if crossing:
            i = crossing[0]
            # gain falls between points i-1 and i: interpolate the input level at exactly -1 dB
            x = float(np.interp(target, [g[i], g[i - 1]], [p_in[i], p_in[i - 1]]))
            p_in_1db = Q(x, "dBm")
            p_out_1db = Q(x + target, "dBm")
        summaries.append(
            CompressionSummary(Q(f_hz, "Hz"), Q(g_ss, "dB"), p_in_1db, p_out_1db, int(len(used)), dropped)
        )
    return summaries


def plot_result(
    result: CompressionResult,
    show: bool = False,
    save_folder: Optional[str] = None,
    filename: Optional[str] = None,
) -> Any:
    """Output power versus input power at the DUT plane, with the P1dB point marked.

    Points dropped for being too close to the noise floor are drawn as grey crosses.
    """
    dut = at_dut(result)
    valid = dut.valid()
    objects: list[Any] = []
    for summary, (_, rows) in zip(summarize(result), _groups(dut)):
        label = f"{summary.frequency.to('MHz'):~.4g}"
        used, dropped = rows[valid[rows]], rows[~valid[rows]]
        if len(used):
            objects.append(LinePlot(dut.p_in[used], dut.p_out[used], label=label))
        for i in dropped:
            objects.append(Marker(dut.p_in[i], dut.p_out[i], style="x", color="gray"))
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
