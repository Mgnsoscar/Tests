"""Intermodulation analysis: OIP3 / IIP3 per centre frequency, and the plot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from labkit.plotting import GridMajor, Legend, LinePlot, Marker, Title, XLabel, YLabel, plot
from labkit.units import Quantity, quantity as Q

from ..measurements.intermodulation import IntermodulationResult
from .reference_plane import at_dut

__all__ = ["IntermodulationSummary", "summarize", "plot_result"]


@dataclass(frozen=True)
class IntermodulationSummary:
    center: Quantity
    #: Mean output level of the two tones at the DUT output.
    tone_level: Quantity
    #: The larger of the two IM3 products at the DUT output.
    im3_level: Quantity
    #: Tone-to-IM3 ratio (dBc).
    im3_ratio: Quantity
    oip3: Quantity
    iip3: Quantity
    #: The analyzer's own third-order intercept reading (raw, instrument plane).
    toi_instrument: Quantity


def summarize(result: IntermodulationResult) -> list[IntermodulationSummary]:
    """OIP3 and IIP3 per centre frequency, from the DUT-plane tone and IM3 levels.

    ``OIP3 = P_tone + (P_tone − P_IM3) / 2`` (in dBm/dB), and
    ``IIP3 = OIP3 − gain`` with the gain taken as the DUT-plane tone output
    over the DUT-plane tone input (generator level minus the input path loss).
    """
    dut = at_dut(result)
    center = np.asarray(dut.center.to("Hz").magnitude, dtype=np.float64)
    tone = np.asarray(dut.tone, dtype=object)
    level = np.asarray(dut.level.to("dBm").magnitude, dtype=np.float64)
    freq = np.asarray(dut.frequency.to("Hz").magnitude, dtype=np.float64)
    toi = np.asarray(dut.toi_instrument.to("dBm").magnitude, dtype=np.float64)
    drive = result.settings.get("level")
    summaries: list[IntermodulationSummary] = []
    for value in np.unique(center):
        rows = np.flatnonzero(center == value)
        by_tone = {str(tone[i]): i for i in rows}
        p_tones = np.array([level[by_tone["f1"]], level[by_tone["f2"]]])
        p_im3 = max(level[by_tone["im3_low"]], level[by_tone["im3_high"]])
        p_tone = float(10 * np.log10(np.mean(10 ** (p_tones / 10))))  # linear-domain mean
        ratio = p_tone - p_im3
        oip3 = p_tone + ratio / 2
        if drive is not None and hasattr(drive, "to"):
            f_tones = Q(np.array([freq[by_tone["f1"]], freq[by_tone["f2"]]]), "Hz")
            p_in = dut.path("in").after(Q(np.full(2, drive.to("dBm").magnitude), "dBm"), f_tones)
            gain_db = p_tone - float(10 * np.log10(np.mean(10 ** (np.asarray(p_in.magnitude) / 10))))
            iip3 = oip3 - gain_db
        else:
            iip3 = float("nan")
        summaries.append(
            IntermodulationSummary(
                Q(float(value), "Hz"), Q(p_tone, "dBm"), Q(float(p_im3), "dBm"),
                Q(float(ratio), "dB"), Q(float(oip3), "dBm"), Q(float(iip3), "dBm"),
                Q(float(toi[rows[0]]), "dBm"),
            )
        )
    return summaries


def plot_result(
    result: IntermodulationResult,
    show: bool = False,
    save_folder: Optional[str] = None,
    filename: Optional[str] = None,
) -> Any:
    """The four spectral lines at each centre frequency, at the DUT output."""
    dut = at_dut(result)
    center = np.asarray(dut.center.to("Hz").magnitude, dtype=np.float64)
    objects: list[Any] = []
    for value in np.unique(center):
        rows = np.flatnonzero(center == value)
        order = rows[np.argsort(np.asarray(dut.frequency.magnitude)[rows])]
        label = f"{Q(value, 'Hz').to('MHz'):~.4g}"
        objects.append(LinePlot(dut.frequency[order], dut.level[order], label=label, style=":"))
        for i in order:
            objects.append(Marker(dut.frequency[i], dut.level[i]))
    objects += [
        Title(f"{dut.title} — two-tone IMD"),
        XLabel("Frequency"),
        YLabel("Level at DUT output"),
        GridMajor(),
        Legend(),
    ]
    return plot(*objects, show=show, save_folder=save_folder, filename=filename)
