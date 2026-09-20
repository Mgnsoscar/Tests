"""S-parameters (S11, S21, S22) on the vector network analyzer.

The sweep is configured with :func:`configure` — the frequency range (a fixed
range covering every channel, or the channel's own band), points, IF
bandwidth, source power and averaging — and **then** the calibration is dealt
with (:mod:`rflab.calibration`), because a calibration is only valid for the
sweep it was made with. :func:`measure` finally creates one trace per
requested S-parameter on channel 1 of the VNA, sweeps once (averaged), and
stores the complex data as magnitude (dB) and phase (degrees) per frequency,
**raw** at the VNA ports, together with the correction state and the
calibration description. With a path on a port,
:func:`rflab.analysis.at_dut` adds twice the path loss to a reflection
magnitude (the wave passes the path going in and coming back) and the input
plus the output path loss to a transmission magnitude (S21 is the gain).

One result holds one DUT configuration (its ``state``: attenuation and
bypass). Measure every configuration of interest, then
:mod:`rflab.analysis.channel_report` compares them: main and max gain, the
filter cutoff, the passband variation, and the S11 averaged over all
configurations.

By default each channel sweeps only the region that matters: its band plus a
`margin` on each side (25 MHz, enough to reach a cutoff 20 MHz outside the
band), 401 points, 1 kHz IF bandwidth, 8 sweeps averaged, and −20 dBm source
power so an active DUT's input is measured well below compression. A fixed
`frequency_range` covering every channel can be given instead, so that one
calibration serves all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional, cast

import numpy as np

from labkit.instruments import ZNLE18
from labkit.units import Quantity, quantity as Q

from ..bench import BenchLike
from ..dut import DUT, Channel
from ._base import Result, base_fields, describe_instruments, register_result

__all__ = ["SParameterSettings", "SParameterResult", "sweep_range", "configure", "measure"]

_PHASE_SUFFIX = " phase"


@dataclass(frozen=True)
class SParameterSettings:
    """How the reflection sweep is taken."""

    #: Sweep range; ``None`` sweeps the channel's own band plus `margin` on
    #: each side. A fixed range that covers every channel lets one
    #: calibration serve all of them.
    frequency_range: Optional[tuple[Quantity, Quantity]] = None
    #: How far beyond the band edges the sweep reaches when `frequency_range` is ``None``.
    margin: Quantity = Q(25, "MHz")
    points: int = 401
    if_bandwidth: Quantity = Q(1, "kHz")
    #: Source power; keep an active DUT well out of compression.
    power: Quantity = Q(-20, "dBm")
    #: Sweeps averaged (1 = none).
    average_count: int = 8
    #: S21 is the gain the channel report checks; S11 and S22 the matches.
    parameters: tuple[str, ...] = ("S11", "S21", "S22")


def sweep_range(channel: Channel, settings: SParameterSettings) -> tuple[Quantity, Quantity]:
    """The ``(start, stop)`` the sweep covers for this channel."""
    if settings.frequency_range is not None:
        return settings.frequency_range
    return channel.f_start - settings.margin, channel.f_stop + settings.margin


def configure(vna: ZNLE18, channel: Channel, settings: SParameterSettings) -> None:
    """Set up the sweep (range, points, IF bandwidth, power, averaging). Do this before calibrating."""
    start, stop = sweep_range(channel, settings)
    vna.trace.delete_all()
    vna.frequency.set_start(start)
    vna.frequency.set_stop(stop)
    vna.sweep.set_type("LINEAR")
    vna.sweep.set_points(settings.points)
    vna.bandwidth.set_if_bandwidth(settings.if_bandwidth)
    vna.power.set_power(settings.power)
    if settings.average_count > 1:
        vna.average.set_state(True)
        vna.average.set_count(settings.average_count)
        vna.sweep.set_count(settings.average_count)
    else:
        vna.average.set_state(False)
        vna.sweep.set_count(1)
    vna.display.set_window_state(1, True)
    vna.power.set_output(True)


@register_result
@dataclass(kw_only=True)
class SParameterResult(Result):
    """Magnitude (dB) and phase (deg) of each S-parameter per frequency."""

    measurement: ClassVar[str] = "S-Parameters"

    frequency: Quantity
    magnitudes: dict[str, Quantity]
    phases: dict[str, Quantity]

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(self.magnitudes)

    @property
    def band(self) -> Optional[tuple[Quantity, Quantity]]:
        """The channel's band as recorded with the result (``None`` if unknown)."""
        start, stop = self.settings.get("band_start"), self.settings.get("band_stop")
        if hasattr(start, "to") and hasattr(stop, "to"):
            return cast(Quantity, start), cast(Quantity, stop)
        return None

    def columns(self) -> dict[str, Any]:
        cols: dict[str, Any] = {"Frequency": self.frequency}
        for name in self.parameters:
            cols[name] = self.magnitudes[name]
            cols[name + _PHASE_SUFFIX] = self.phases[name]
        return cols

    @classmethod
    def from_columns(cls, meta: dict[str, Any], columns: dict[str, Any]) -> "SParameterResult":
        names = [k for k in columns if k != "Frequency" and not k.endswith(_PHASE_SUFFIX)]
        return cls(
            **meta,
            frequency=columns["Frequency"],
            magnitudes={n: columns[n] for n in names},
            phases={n: columns[n + _PHASE_SUFFIX] for n in names},
        )


def measure(
    bench: BenchLike,
    dut: DUT,
    channel: Channel,
    settings: SParameterSettings = SParameterSettings(),
    state: Optional[Mapping[str, Any]] = None,
    configure_sweep: bool = True,
    calibration: str = "not recorded",
) -> SParameterResult:
    """Sweep and read each requested S-parameter as complex data.

    Pass ``configure_sweep=False`` when :func:`configure` was already called
    (and a calibration made or loaded on top of it); `calibration` is the
    description returned by the calibration dialog, stored with the result.
    """
    vna = bench.vna
    if configure_sweep:
        configure(vna, channel, settings)
    else:
        vna.trace.delete_all()
        vna.power.set_output(True)

    frequency: Quantity | None = None
    magnitudes: dict[str, Quantity] = {}
    phases: dict[str, Quantity] = {}
    for index, parameter in enumerate(settings.parameters, start=1):
        name = f"Trc{index}"
        vna.trace.create(name, parameter)  # type: ignore[arg-type]
        vna.display.feed_trace(1, index, name)
        vna.trace.select(name)
        if settings.average_count > 1:
            vna.average.clear()
        stimulus, values = vna.measure()
        frequency = stimulus
        magnitudes[parameter] = Q(20 * np.log10(np.abs(values)), "dB")
        phases[parameter] = Q(np.degrees(np.angle(values)), "deg")
    correction_enabled = vna.calibration.is_correction_enabled()
    correction_date = vna.calibration.get_correction_date() if correction_enabled else ""
    vna.power.set_output(False)
    vna.check_errors("Network analyzer after the S-parameter sweep")

    assert frequency is not None, "at least one S-parameter must be requested"
    fields = base_fields(dut, channel, describe_instruments(vna), settings, state)
    start, stop = sweep_range(channel, settings)
    fields["settings"].update(
        {
            "frequency_range": f"{start:~} to {stop:~}",
            "band_start": channel.f_start,
            "band_stop": channel.f_stop,
            "calibration": calibration,
            "correction": "on" if correction_enabled else "off",
            "correction_date": correction_date,
        }
    )
    return SParameterResult(**fields, frequency=frequency, magnitudes=magnitudes, phases=phases)
