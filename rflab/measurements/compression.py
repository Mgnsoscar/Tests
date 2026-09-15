"""Gain compression: output power versus input power, measured carefully.

The Keysight MXG (generator A) drives the DUT input with a CW tone stepped up
from `p_start`; the R&S FSV (spectrum analyzer) reads the tone level after each
step. Three things make the numbers trustworthy:

**The generator's step attenuator is held.** With automatic attenuation the
MXG switches its mechanical step attenuator as the level crosses attenuator
bands, and each switch puts a small bump in the measured curve. The sweep
therefore runs with attenuator hold (``POW:ATT:AUTO OFF``): the attenuator is
fixed once, at the smallest value that still reaches `p_start` (the ALC can
level from about −15 dBm up to the maximum output with 0 dB of attenuation,
per the data sheet), and only the ALC moves the level. The generator's error
queue is read after each step, so an unleveled or out-of-range setting raises
instead of being recorded as data.

**The analyzer reads accurately.** RMS detector, trace averaging over
`averages` sweeps, a narrow span and RBW around the tone, and a reference
level that is kept about 10 dB above the signal — re-set and re-measured
whenever the signal drifts out of that window — so the input attenuator and
the display range stay right as the output rises. The analyzer's error queue
is checked at the end.

**The noise floor is known.** Before the sweep, with the generator's RF off,
the analyzer's noise floor at the test frequency is measured and stored with
the result; the analysis ignores any point closer than `noise_margin` to it
when it derives the gain and the compression point.

The sweep **stops itself** once the DUT is into compression. The reference
gain is not "the first few points" — a bad reading there would poison the
whole result — but the highest value of a moving median of the gain seen so
far (an amplifier's gain only falls with drive); stepping ends once the
moving-median gain has fallen `stop_compression` below it, after at least
`min_points` valid points. `p_stop` remains a hard ceiling.

Levels are stored **raw** at the instrument reference planes;
:func:`rflab.analysis.at_dut` moves them to the DUT ports, and
:mod:`rflab.analysis.compression` derives small-signal gain and P1dB with the
same robust rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional

import numpy as np

from labkit.instruments import FSV3007, N5183A
from labkit.instruments.drivers.rohde_schwarz.marker import Marker
from labkit.units import Quantity, quantity as Q

from ..bench import BenchLike
from ..dut import DUT, Channel
from ._base import Result, base_fields, describe_instruments, register_result, settle

__all__ = [
    "CompressionSettings",
    "CompressionResult",
    "measure",
    "levels",
    "generator_attenuation",
    "moving_median",
    "MXG_HOLD_RANGE_MIN",
]

#: Lowest level the MXG's ALC can set with the step attenuator held at 0 dB
#: (N5183A data sheet, "Amplitude ... with step attenuator hold range in 0 dB").
MXG_HOLD_RANGE_MIN = Q(-15, "dBm")
#: The MXG step attenuator moves in 5 dB steps (Option 1E1).
_ATTENUATOR_STEP_DB = 5.0


@dataclass(frozen=True)
class CompressionSettings:
    """How a compression sweep is taken."""

    #: Number of test frequencies across the channel (1 = the band centre only).
    n_frequencies: int = 1
    p_start: Quantity = Q(-30, "dBm")
    #: Hard ceiling for the generator level; the sweep normally stops before it.
    p_stop: Quantity = Q(0, "dBm")
    p_step: Quantity = Q(1, "dB")
    #: Stop stepping once the moving-median gain has fallen this far below its peak.
    stop_compression: Quantity = Q(2, "dB")
    #: Never stop before this many valid points have been taken.
    min_points: int = 5
    #: Hold the generator's step attenuator for the whole sweep (no bumps).
    attenuator_hold: bool = True
    #: Attenuator value to hold; ``None`` picks the smallest that reaches `p_start`.
    generator_attenuation: Optional[Quantity] = None
    #: Analyzer span around the tone, and resolution bandwidth.
    span: Quantity = Q(1, "MHz")
    rbw: Quantity = Q(10, "kHz")
    #: Sweeps averaged per reading (RMS detector, average trace mode).
    averages: int = 5
    #: Initial analyzer reference level; tracked to ~10 dB above the signal after that.
    ref_level: Quantity = Q(0, "dBm")
    #: Points closer than this to the measured noise floor are not trusted.
    noise_margin: Quantity = Q(15, "dB")
    #: Pause after each generator step before the analyzer sweeps.
    settle: Quantity = Q(50, "ms")


def levels(settings: CompressionSettings) -> Quantity:
    """The generator levels the sweep may visit, `p_start` to `p_stop` in `p_step`."""
    start = settings.p_start.to("dBm").magnitude
    stop = settings.p_stop.to("dBm").magnitude
    step = settings.p_step.to("dB").magnitude
    return Q(np.arange(start, stop + step / 2, step), "dBm")


def generator_attenuation(settings: CompressionSettings) -> Quantity:
    """The step-attenuator value to hold: `generator_attenuation`, or the smallest
    multiple of 5 dB that brings `p_start` inside the ALC's hold range."""
    if settings.generator_attenuation is not None:
        return settings.generator_attenuation
    shortfall = MXG_HOLD_RANGE_MIN.to("dBm").magnitude - settings.p_start.to("dBm").magnitude
    steps = max(0, math.ceil(shortfall / _ATTENUATOR_STEP_DB - 1e-9))
    return Q(steps * _ATTENUATOR_STEP_DB, "dB")


#: Points in the moving median used for the online gain reference.
MEDIAN_WINDOW = 3


def moving_median(values: list[float], window: int = MEDIAN_WINDOW) -> float:
    """The median of the last `window` values (fewer if not yet available)."""
    return float(np.median(values[-window:]))


@register_result
@dataclass(kw_only=True)
class CompressionResult(Result):
    """Raw compression data: one row per (frequency, input level)."""

    measurement: ClassVar[str] = "Compression"

    frequency: Quantity
    p_in: Quantity
    p_out: Quantity
    #: The analyzer's noise floor at each row's frequency (generator off).
    noise_floor: Quantity

    def columns(self) -> dict[str, Any]:
        return {
            "Frequency": self.frequency,
            "P_in": self.p_in,
            "P_out": self.p_out,
            "Noise floor": self.noise_floor,
        }

    @classmethod
    def from_columns(cls, meta: dict[str, Any], columns: dict[str, Any]) -> "CompressionResult":
        return cls(
            **meta,
            frequency=columns["Frequency"],
            p_in=columns["P_in"],
            p_out=columns["P_out"],
            noise_floor=columns["Noise floor"],
        )

    def valid(self, noise_margin: Optional[Quantity] = None) -> np.ndarray:
        """Boolean mask of rows whose output is at least `noise_margin` above the noise floor.

        `noise_margin` defaults to the margin the measurement was taken with.
        """
        margin = noise_margin if noise_margin is not None else self.settings.get("noise_margin", Q(15, "dB"))
        margin_db = float(margin.to("dB").magnitude) if hasattr(margin, "to") else float(margin)
        p_out = np.asarray(self.p_out.to("dBm").magnitude, dtype=np.float64)
        floor = np.asarray(self.noise_floor.to("dBm").magnitude, dtype=np.float64)
        return p_out >= floor + margin_db


# --- the analyzer side --------------------------------------------------------

_HEADROOM_DB = 10.0        # reference level target above the signal
_HEADROOM_MIN_DB = 3.0     # re-set when the signal gets closer than this ...
_HEADROOM_MAX_DB = 20.0    # ... or farther than this


def _configure_analyzer(sa: FSV3007, settings: CompressionSettings) -> Marker:
    sa.frequency.set_span(settings.span)
    sa.bandwidth.set_rbw(settings.rbw)
    sa.bandwidth.set_vbw_auto(True)
    sa.amplitude.set_unit("DBM")
    sa.amplitude.set_attenuation_auto(True)
    sa.amplitude.set_preamp(False)
    sa.amplitude.set_ref_level(settings.ref_level)
    sa.trace.set_detector("RMS")
    sa.trace.set_mode("AVERAGE" if settings.averages > 1 else "WRITE")
    sa.sweep.set_count(max(1, settings.averages))
    sa.sweep.set_time_auto(True)
    sa.sweep.set_continuous(False)
    return sa.marker(1, enable=True)


def _read_peak(sa: FSV3007, marker: Marker) -> float:
    """One averaged single sweep, then the peak level in dBm."""
    sa.trigger(wait_for_completion=True)
    marker.peak_search()
    return float(marker.get_y().to("dBm").magnitude)


def _read_level(sa: FSV3007, marker: Marker, ref_level: float) -> tuple[float, float]:
    """Read the tone level, keeping the reference level ~10 dB above it.

    Returns ``(level_dbm, ref_level_dbm)``; when the reference level had to
    change the reading is repeated at the new setting.
    """
    level = _read_peak(sa, marker)
    if not (level + _HEADROOM_MIN_DB <= ref_level <= level + _HEADROOM_MAX_DB):
        ref_level = float(math.ceil(level + _HEADROOM_DB))
        sa.amplitude.set_ref_level(Q(ref_level, "dBm"))
        level = _read_peak(sa, marker)
    return level, ref_level


# --- the generator side -------------------------------------------------------

def _configure_generator(gen: N5183A, settings: CompressionSettings) -> Quantity:
    gen.frequency.set_mode("CW")
    gen.power.set_mode("FIXED")
    gen.power.set_modulation_enabled(False)
    gen.power.set_alc_enabled(True)
    attenuation = generator_attenuation(settings)
    if settings.attenuator_hold:
        gen.power.set_attenuation_auto(False)
        gen.power.set_attenuation(attenuation)
    else:
        gen.power.set_attenuation_auto(True)
    return attenuation


# --- the measurement ----------------------------------------------------------

def measure(
    bench: BenchLike,
    dut: DUT,
    channel: Channel,
    settings: CompressionSettings = CompressionSettings(),
    state: Optional[Mapping[str, Any]] = None,
) -> CompressionResult:
    """Step the MXG's level up until the DUT compresses, reading the output on the FSV."""
    gen, sa = bench.gen_a, bench.fsv
    freqs = channel.frequencies(settings.n_frequencies)
    p_levels = levels(settings)
    stop_after = float(settings.stop_compression.to("dB").magnitude)
    margin_db = float(settings.noise_margin.to("dB").magnitude)

    marker = _configure_analyzer(sa, settings)
    attenuation = _configure_generator(gen, settings)

    f_rows: list[float] = []
    p_in_rows: list[float] = []
    p_out_rows: list[float] = []
    floor_rows: list[float] = []
    try:
        for f in freqs:
            f_hz = float(f.to("Hz").magnitude)
            sa.frequency.set_center(f)

            # Noise floor at this frequency, generator off.
            gen.power.set_rf_enabled(False)
            gen.frequency.set_frequency(f)
            settle(settings.settle)
            ref_level = float(settings.ref_level.to("dBm").magnitude)
            sa.amplitude.set_ref_level(Q(ref_level, "dBm"))
            noise_floor = _read_peak(sa, marker)

            gen.power.set_level(p_levels[0])
            gen.power.set_rf_enabled(True)
            gen.check_errors(f"Generator at {p_levels[0]:~} with attenuator {attenuation:~}")

            gains: list[float] = []      # gain of the valid points, in order
            reference = -math.inf        # highest moving-median gain seen so far
            for p in p_levels:
                gen.power.set_level(p)
                gen.check_errors(f"Generator at {p:~} with attenuator {attenuation:~}")
                settle(settings.settle)
                p_out, ref_level = _read_level(sa, marker, ref_level)
                p_in = float(p.to("dBm").magnitude)
                f_rows.append(f_hz)
                p_in_rows.append(p_in)
                p_out_rows.append(p_out)
                floor_rows.append(noise_floor)

                if p_out < noise_floor + margin_db:
                    continue  # too close to the noise floor to judge the gain
                gains.append(p_out - p_in)
                if len(gains) < MEDIAN_WINDOW:
                    continue  # a median needs a full window before it can outvote a bad point
                smoothed = moving_median(gains)
                reference = max(reference, smoothed)
                if len(gains) >= settings.min_points and smoothed <= reference - stop_after:
                    break  # into compression: no need to drive the DUT harder
    finally:
        gen.power.set_rf_enabled(False)
        gen.power.set_attenuation_auto(True)
    sa.check_errors("Spectrum analyzer after the compression sweep")

    fields = base_fields(dut, channel, describe_instruments(gen, sa), settings, state)
    fields["settings"]["generator_attenuation"] = attenuation
    return CompressionResult(
        **fields,
        frequency=Q(np.array(f_rows), "Hz"),
        p_in=Q(np.array(p_in_rows), "dBm"),
        p_out=Q(np.array(p_out_rows), "dBm"),
        noise_floor=Q(np.array(floor_rows), "dBm"),
    )
