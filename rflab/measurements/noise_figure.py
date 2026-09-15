"""Noise figure and gain with the analyzer's K30 application.

Unlike the other measurements, the path losses are entered **on the
instrument** before measuring — they change the noise figure the application
computes — so the result is already at the DUT reference plane. The input and
output paths are evaluated (with interpolation) at the measurement frequencies
and loaded as K30 loss tables; a port with no path gets a 0 dB spot loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from labkit.units import Quantity, quantity as Q

from ..bench import BenchLike
from ..dut import DUT, Channel
from ._base import Result, base_fields, describe_instruments, register_result

__all__ = ["NoiseFigureSettings", "NoiseFigureResult", "measure"]


@dataclass(frozen=True)
class NoiseFigureSettings:
    """How a noise-figure sweep is taken."""

    points: int = 51
    #: Excess noise ratio of the noise source, in dB (constant across the band).
    enr_db: float = 15.2
    #: Enable the second-stage (calibration) correction. Calibrate first.
    second_stage_correction: bool = True
    #: Name of the K30 measurement channel on the analyzer.
    channel_name: str = "Noise"


@register_result
@dataclass(kw_only=True)
class NoiseFigureResult(Result):
    """Noise figure and gain per frequency, already at the DUT reference plane."""

    measurement: ClassVar[str] = "Noise Figure"

    frequency: Quantity
    noise_figure: Quantity
    gain: Quantity

    def columns(self) -> dict[str, Any]:
        return {"Frequency": self.frequency, "Noise figure": self.noise_figure, "Gain": self.gain}

    @classmethod
    def from_columns(cls, meta: dict[str, Any], columns: dict[str, Any]) -> "NoiseFigureResult":
        return cls(
            **meta,
            frequency=columns["Frequency"],
            noise_figure=columns["Noise figure"],
            gain=columns["Gain"],
        )


def measure(
    bench: BenchLike,
    dut: DUT,
    channel: Channel,
    settings: NoiseFigureSettings = NoiseFigureSettings(),
) -> NoiseFigureResult:
    """Run the K30 noise-figure sweep across the channel with path losses applied."""
    sa = bench.fsv
    nf = sa.noise_figure
    freqs = channel.frequencies(settings.points)

    existing = {name for _, name in sa.list_channels()}
    if settings.channel_name in existing:
        nf.select(settings.channel_name)
    else:
        nf.create(settings.channel_name)

    nf.set_enr_mode("CONSTANT")
    nf.set_enr(settings.enr_db)
    nf.set_start(channel.f_start)
    nf.set_stop(channel.f_stop)
    nf.set_points(settings.points)
    nf.use_frequency_list()

    for port, loss_port in (("in", "INPUT"), ("out", "OUTPUT")):
        path = channel.path(port)
        if len(path):
            nf.set_loss_from_path(loss_port, path, freqs, name=f"LabKit {port}")  # type: ignore[arg-type]
        else:
            nf.set_loss(loss_port, Q(0, "dB"))  # type: ignore[arg-type]

    nf.set_second_stage_correction(settings.second_stage_correction)
    sa.trigger(wait_for_completion=True)
    nf_db = nf.get_noise_figure()
    gain_db = nf.get_gain()

    return NoiseFigureResult(
        **base_fields(dut, channel, describe_instruments(sa), settings),
        reference_plane="dut",
        frequency=freqs.to("Hz"),
        noise_figure=Q(np.array(nf_db, dtype=np.float64), "dB"),
        gain=Q(np.array(gain_db, dtype=np.float64), "dB"),
    )
