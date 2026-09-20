"""The per-channel requirements report, from the S-parameter results of every configuration.

Each S-parameter result holds one DUT configuration (its ``state``:
attenuation and bypass). The report takes all of them for one channel, keeps
the latest result per configuration (:func:`latest_per_state`), moves them to
the DUT plane and answers the specification's questions:

**S21 (gain)** — for every configuration, :func:`evaluate_gain` reads the
*nominal* gain (S21 at the band centre) and checks

- the **passband variation**: the largest deviation of S21 from the nominal
  gain inside the band must stay below Z (``requirements.passband_variation``);
- the **filter cutoff**: at Y (``cutoff_offset``) below ``f_start`` and above
  ``f_stop`` the gain must be at least X (``cutoff_rejection``) below the
  nominal gain.

Two configurations are singled out: the **main gain** (the highest
attenuation with the bypass on) and the **max gain** (no attenuation, bypass
off).

**S11 / S22 (match)** — :func:`average_reflection` averages the reflection
of all configurations as power ratios (``10·log10(mean(10^(S/10)))``), which
is what "the average S11" means for a level in dB, and finds its worst value
inside the band.

:func:`report` builds the whole :class:`ChannelReport`; :meth:`ChannelReport.text`
prints it, :func:`plot_gain` and :func:`plot_reflection` draw it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from labkit.plotting import (
    ColorName,
    GridMajor,
    Legend,
    LinePlot,
    Marker,
    Panel,
    Title,
    XLabel,
    XLimits,
    YLabel,
    YLimits,
    plot,
)
from labkit.units import Quantity, is_quantity, quantity as Q

from ..dut import Channel, ChannelRequirements
from ..measurements.s_parameters import SParameterResult
from .reference_plane import at_dut

__all__ = [
    "GainEvaluation",
    "ReflectionAverage",
    "ChannelReport",
    "latest_per_state",
    "evaluate_gain",
    "average_reflection",
    "report",
    "plot_gain",
    "plot_reflection",
]

_MAIN_COLOR: ColorName = "tab:red"
_MAX_COLOR: ColorName = "tab:blue"
_AVERAGE_COLOR: ColorName = "black"
_BAND_COLOR: ColorName = "gray"
_OTHER_COLORS: tuple[ColorName, ...] = ("tab:orange", "tab:green", "tab:purple", "brown", "pink", "cyan", "magenta")


# --- helpers -------------------------------------------------------------------

def _hz(value: Quantity) -> float:
    return float(value.to("Hz").magnitude)


def _mhz_axis(f_hz: Any) -> Quantity:
    """Frequencies for the x axis, in MHz (a Hz axis gets a ``1e8`` exponent)."""
    return Q(np.asarray(f_hz, dtype=np.float64) / 1e6, "MHz")


def _db(values: Quantity) -> np.ndarray:
    return np.asarray(values.magnitude, dtype=np.float64)


def _mhz(f_hz: float) -> str:
    return f"{Q(f_hz, 'Hz').to('MHz'):~.4g}"


def _verdict(ok: Optional[bool]) -> str:
    return "PASS" if ok else "FAIL" if ok is False else "not checked"


def _attenuation_db(result: SParameterResult) -> Optional[float]:
    value = result.state.get("attenuation")
    return float(value.to("dB").magnitude) if is_quantity(value) and value is not None else None


def _bypass(result: SParameterResult) -> Optional[bool]:
    value = result.state.get("bypass")
    return value if isinstance(value, bool) else None


def _state_key(result: SParameterResult) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, str(v)) for k, v in result.state.items()))


def _config_label(result: SParameterResult) -> str:
    return result.state_label or "(no state recorded)"


def latest_per_state(results: Iterable[SParameterResult]) -> list[SParameterResult]:
    """The newest result of each DUT configuration, ordered by attenuation, then bypass off → on."""
    newest: dict[tuple[tuple[str, str], ...], SParameterResult] = {}
    for result in results:
        key = _state_key(result)
        if key not in newest or result.timestamp > newest[key].timestamp:
            newest[key] = result

    def order(result: SParameterResult) -> tuple[float, int, str]:
        attenuation = _attenuation_db(result)
        bypass = _bypass(result)
        return (attenuation if attenuation is not None else -1.0, int(bool(bypass)), result.state_label)

    return sorted(newest.values(), key=order)


def _band_mask(f_hz: np.ndarray, channel: Channel) -> np.ndarray:
    mask: np.ndarray = (f_hz >= _hz(channel.f_start)) & (f_hz <= _hz(channel.f_stop))
    return mask


def _value_at(f_hz: np.ndarray, values: np.ndarray, at_hz: float) -> Optional[float]:
    """`values` interpolated at `at_hz`, or ``None`` when the sweep does not reach it."""
    if at_hz < f_hz[0] or at_hz > f_hz[-1]:
        return None
    return float(np.interp(at_hz, f_hz, values))


# --- S21: gain, passband variation, cutoff ---------------------------------------

@dataclass(frozen=True)
class GainEvaluation:
    """One configuration's S21 against the channel's requirements (levels at the DUT plane)."""

    #: The result, at the DUT plane.
    result: SParameterResult
    attenuation: Optional[Quantity]
    bypass: Optional[bool]
    #: The nominal gain: S21 at the band centre.
    gain_nominal: Quantity
    #: Lowest and highest gain inside the band.
    gain_min: Quantity
    gain_max: Quantity
    #: The largest |S21 − nominal| inside the band.
    variation: Quantity
    #: How far below the nominal gain S21 is at ``f_start − Y`` / ``f_stop + Y``
    #: (``None`` when the sweep does not reach that frequency, or without requirements).
    rejection_below: Optional[Quantity]
    rejection_above: Optional[Quantity]
    requirements: Optional[ChannelRequirements]

    @property
    def label(self) -> str:
        """The configuration, e.g. ``"attenuation 6 dB bypass off"``."""
        return _config_label(self.result)

    @property
    def variation_ok(self) -> Optional[bool]:
        if self.requirements is None:
            return None
        return bool(self.variation < self.requirements.passband_variation)

    @property
    def rejection_ok(self) -> Optional[bool]:
        if self.requirements is None or self.rejection_below is None or self.rejection_above is None:
            return None
        limit = self.requirements.cutoff_rejection
        return bool(self.rejection_below >= limit and self.rejection_above >= limit)

    @property
    def passed(self) -> Optional[bool]:
        """Both checks passed; ``None`` if either could not be made."""
        checks = (self.variation_ok, self.rejection_ok)
        if any(c is None for c in checks):
            return None
        return all(checks)

    def describe(self) -> str:
        """One line with the numbers and verdicts."""
        text = (
            f"gain {self.gain_nominal:~.2f} at centre, {self.gain_min:~.2f} to {self.gain_max:~.2f} in band, "
            f"variation {self.variation:~.2f}"
        )
        if self.requirements is not None:
            r = self.requirements
            text += f" (< {r.passband_variation:~}: {_verdict(self.variation_ok)})"
            if self.rejection_below is None or self.rejection_above is None:
                text += f"; cutoff at ±{r.cutoff_offset:~}: outside the sweep, not checked"
            else:
                text += (
                    f"; cutoff {self.rejection_below:~.1f} below / {self.rejection_above:~.1f} above the band "
                    f"(≥ {r.cutoff_rejection:~}: {_verdict(self.rejection_ok)})"
                )
        return text


def evaluate_gain(channel: Channel, result: SParameterResult) -> GainEvaluation:
    """Nominal gain, passband variation and cutoff rejection of one result's S21."""
    if "S21" not in result.magnitudes:
        raise ValueError(f"{result.title}: no S21 in the result (has {', '.join(result.parameters)}).")
    dut = at_dut(result)
    f = np.asarray(dut.frequency.to("Hz").magnitude, dtype=np.float64)
    s21 = _db(dut.magnitudes["S21"])
    order = np.argsort(f)
    f, s21 = f[order], s21[order]

    nominal = _value_at(f, s21, _hz(channel.f_center))
    if nominal is None:
        raise ValueError(f"{result.title}: the sweep does not cover the band centre {channel.f_center:~}.")
    mask = _band_mask(f, channel)
    if not mask.any():
        raise ValueError(f"{result.title}: no sweep point inside the band {channel.f_start:~} to {channel.f_stop:~}.")
    in_band = s21[mask]
    gain_min, gain_max = float(in_band.min()), float(in_band.max())
    variation = max(gain_max - nominal, nominal - gain_min)

    requirements = channel.requirements
    below = above = None
    if requirements is not None:
        offset = _hz(requirements.cutoff_offset)
        low = _value_at(f, s21, _hz(channel.f_start) - offset)
        high = _value_at(f, s21, _hz(channel.f_stop) + offset)
        below = Q(nominal - low, "dB") if low is not None else None
        above = Q(nominal - high, "dB") if high is not None else None

    attenuation = result.state.get("attenuation")
    return GainEvaluation(
        result=dut,
        attenuation=attenuation if is_quantity(attenuation) else None,
        bypass=_bypass(result),
        gain_nominal=Q(nominal, "dB"),
        gain_min=Q(gain_min, "dB"),
        gain_max=Q(gain_max, "dB"),
        variation=Q(variation, "dB"),
        rejection_below=below,
        rejection_above=above,
        requirements=requirements,
    )


def _pick(evaluations: Sequence[GainEvaluation], bypass: bool, highest_attenuation: bool) -> Optional[GainEvaluation]:
    """The configuration with the bypass in the given position (or unknown) and the
    highest/lowest attenuation; ``None`` if there is no such configuration."""
    candidates = [e for e in evaluations if e.bypass is None or e.bypass == bypass]
    if not candidates:
        return None

    def attenuation(e: GainEvaluation) -> float:
        return float(e.attenuation.to("dB").magnitude) if e.attenuation is not None else 0.0

    return (max if highest_attenuation else min)(candidates, key=attenuation)


# --- S11 / S22: the average over the configurations ------------------------------

@dataclass(frozen=True)
class ReflectionAverage:
    """One reflection parameter of every configuration, and their average (DUT plane)."""

    parameter: str
    frequency: Quantity
    #: The power-average of all configurations, in dB.
    average: Quantity
    #: ``{configuration label: magnitude}`` on the same frequency grid.
    traces: dict[str, Quantity]
    #: The worst (highest) value of the average inside the band, and where.
    worst_in_band: Quantity
    worst_frequency: Quantity
    #: The single worst point of any configuration inside the band: ``(label, level, frequency)``.
    worst_individual: tuple[str, Quantity, Quantity]

    def describe(self) -> str:
        label, level, f = self.worst_individual
        return (
            f"{self.parameter} average of {len(self.traces)} configurations: worst in band "
            f"{self.worst_in_band:~.1f} at {self.worst_frequency.to('MHz'):~.4g} "
            f"(worst single configuration: {level:~.1f} at {f.to('MHz'):~.4g}, {label})"
        )


def average_reflection(
    channel: Channel, results: Sequence[SParameterResult], parameter: str = "S11"
) -> ReflectionAverage:
    """Average `parameter` of all `results` as power ratios, on the first result's frequency grid."""
    dut_results = [at_dut(r) for r in results if parameter in r.magnitudes]
    if not dut_results:
        raise ValueError(f"no result with {parameter} for {channel.label}.")
    grid = np.asarray(dut_results[0].frequency.to("Hz").magnitude, dtype=np.float64)
    grid = np.sort(grid)

    traces: dict[str, np.ndarray] = {}
    for r in dut_results:
        f = np.asarray(r.frequency.to("Hz").magnitude, dtype=np.float64)
        order = np.argsort(f)
        traces[_config_label(r)] = np.interp(grid, f[order], _db(r.magnitudes[parameter])[order])
    stack = np.vstack(list(traces.values()))
    average = 10 * np.log10(np.mean(10 ** (stack / 10), axis=0))

    mask = _band_mask(grid, channel)
    if not mask.any():
        mask = np.ones(grid.shape, dtype=bool)
    i_avg = int(np.flatnonzero(mask)[np.argmax(average[mask])])
    row, col = np.unravel_index(np.argmax(np.where(mask, stack, -np.inf)), stack.shape)
    label = list(traces)[int(row)]
    return ReflectionAverage(
        parameter=parameter,
        frequency=Q(grid, "Hz"),
        average=Q(average, "dB"),
        traces={k: Q(v, "dB") for k, v in traces.items()},
        worst_in_band=Q(float(average[i_avg]), "dB"),
        worst_frequency=Q(float(grid[i_avg]), "Hz"),
        worst_individual=(label, Q(float(stack[row, col]), "dB"), Q(float(grid[col]), "Hz")),
    )


# --- the report -------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelReport:
    """Everything the requirements ask about one channel."""

    dut: str
    channel: Channel
    #: One evaluation per configuration (latest result each), in attenuation order.
    gains: list[GainEvaluation]
    #: Highest attenuation with the bypass on.
    main: Optional[GainEvaluation]
    #: Lowest attenuation with the bypass off.
    max_gain: Optional[GainEvaluation]
    #: ``{"S11": ..., "S22": ...}`` for the reflection parameters that were measured.
    reflections: dict[str, ReflectionAverage]

    @property
    def title(self) -> str:
        return f"{self.dut} {self.channel.label}"

    @property
    def passed(self) -> Optional[bool]:
        """Every configuration passed both S21 checks (``None`` if any could not be checked)."""
        verdicts = [e.passed for e in self.gains]
        if not verdicts or any(v is None for v in verdicts):
            return None
        return all(verdicts)

    def text(self) -> str:
        """The report as lines of text."""
        ch = self.channel
        lines = [
            f"{self.title} — band {ch.f_start:~} to {ch.f_stop:~}, centre {ch.f_center:~.6g}",
            f"Requirements: {ch.requirements.describe() if ch.requirements else 'none defined for this channel'}",
            f"Configurations (latest result of each): {len(self.gains)}",
        ]
        for e in self.gains:
            when = e.result.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"  {e.label} [{when}]: {e.describe()}")
        lines.append(_gain_line("Main gain (highest attenuation, bypass on)", self.main))
        lines.append(_gain_line("Max gain (no attenuation, bypass off)", self.max_gain))
        for avg in self.reflections.values():
            lines.append(avg.describe())
        if ch.requirements is not None:
            failed = [e.label for e in self.gains if e.passed is False]
            unchecked = [e.label for e in self.gains if e.passed is None]
            verdict = _verdict(self.passed)
            if failed:
                verdict += " — failed: " + "; ".join(failed)
            if unchecked:
                verdict += " — not fully checked: " + "; ".join(unchecked)
            lines.append(f"S21 requirements: {verdict}")
        return "\n".join(lines)


def _gain_line(what: str, e: Optional[GainEvaluation]) -> str:
    if e is None:
        return f"{what}: no such configuration measured"
    return f"{what}: {e.gain_nominal:~.2f} at the band centre — {e.label}"


def report(channel: Channel, results: Iterable[SParameterResult], dut: Optional[str] = None) -> ChannelReport:
    """The channel report from every S-parameter result of `channel` (latest per configuration)."""
    latest = latest_per_state(r for r in results if r.channel == channel.number)
    if not latest:
        raise ValueError(f"no S-parameter results for {channel.label}.")
    gains = [evaluate_gain(channel, r) for r in latest if "S21" in r.magnitudes]
    reflections = {
        p: average_reflection(channel, latest, p)
        for p in ("S11", "S22")
        if any(p in r.magnitudes for r in latest)
    }
    return ChannelReport(
        dut=dut or f"{latest[0].dut} {latest[0].version}",
        channel=channel,
        gains=gains,
        main=_pick(gains, bypass=True, highest_attenuation=True),
        max_gain=_pick(gains, bypass=False, highest_attenuation=False),
        reflections=reflections,
    )


# --- plots --------------------------------------------------------------------------

def _vertical(f_hz: float, lo: float, hi: float, style: Any, label: Optional[str] = None) -> LinePlot:
    return LinePlot(_mhz_axis([f_hz, f_hz]), Q([lo, hi], "dB"), label=label, color=_BAND_COLOR, style=style, width=1.0)


def _band_lines(channel: Channel, lo: float, hi: float) -> list[Any]:
    """Dashed band edges, dotted cutoff frequencies."""
    objects: list[Any] = [
        _vertical(_hz(channel.f_start), lo, hi, "--", label=f"band {channel.f_start:~} to {channel.f_stop:~}"),
        _vertical(_hz(channel.f_stop), lo, hi, "--"),
    ]
    if channel.requirements is not None:
        offset = _hz(channel.requirements.cutoff_offset)
        objects += [
            _vertical(_hz(channel.f_start) - offset, lo, hi, ":", label=f"cutoff ±{channel.requirements.cutoff_offset:~}"),
            _vertical(_hz(channel.f_stop) + offset, lo, hi, ":"),
        ]
    return objects


def _role(report_: ChannelReport, e: GainEvaluation) -> tuple[str, Optional[ColorName], float, float]:
    """``(prefix, color, width, alpha)`` — main and max stand out, the rest are thin."""
    if e is report_.main and e is report_.max_gain:
        return "main = max gain: ", _MAIN_COLOR, 2.5, 1.0
    if e is report_.main:
        return "main gain: ", _MAIN_COLOR, 2.5, 1.0
    if e is report_.max_gain:
        return "max gain: ", _MAX_COLOR, 2.5, 1.0
    return "", None, 1.0, 0.7


def plot_gain(
    report_: ChannelReport,
    show: bool = False,
    save_folder: Optional[str] = None,
    filename: Optional[str] = None,
) -> Any:
    """Two panels: S21 of every configuration with the cutoff points marked, and the
    passband gain relative to each configuration's centre gain against the ±Z limit."""
    ch = report_.channel
    r = ch.requirements
    others = iter(_OTHER_COLORS)
    all_s21 = np.concatenate([_db(e.result.magnitudes["S21"]) for e in report_.gains])
    lo, hi = float(np.nanmin(all_s21)) - 3.0, float(np.nanmax(all_s21)) + 3.0

    top: list[Any] = [Panel(0, 0)]
    bottom: list[Any] = [Panel(1, 0)]
    largest_variation = 0.0
    for e in report_.gains:
        prefix, color, width, alpha = _role(report_, e)
        color = color or next(others, None)
        f = _mhz_axis(e.result.frequency.to("Hz").magnitude)
        s21 = e.result.magnitudes["S21"]
        label = f"{prefix}{e.label}: {e.gain_nominal:~.2f} at {ch.f_center:~.6g}"
        top.append(LinePlot(f, s21, label=label, color=color, width=width, alpha=alpha))
        if r is not None and e.rejection_below is not None and e.rejection_above is not None and prefix:
            offset = _hz(r.cutoff_offset)
            below = float(e.gain_nominal.magnitude - e.rejection_below.magnitude)
            above = float(e.gain_nominal.magnitude - e.rejection_above.magnitude)
            top.append(Marker(_mhz_axis(_hz(ch.f_start) - offset), Q(below, "dB"), color=color, style="v", size=9.0))
            top.append(
                Marker(
                    _mhz_axis(_hz(ch.f_stop) + offset), Q(above, "dB"), color=color, style="v", size=9.0,
                    label=(
                        f"{prefix.strip(': ')} cutoff: {e.rejection_below:~.1f} / {e.rejection_above:~.1f} "
                        f"below centre (≥ {r.cutoff_rejection:~}: {_verdict(e.rejection_ok)})"
                    ),
                )
            )
        variation_label = f"{prefix}{e.label}: ±{e.variation:~.2f}"
        if r is not None:
            variation_label += f" (< {r.passband_variation:~}: {_verdict(e.variation_ok)})"
        largest_variation = max(largest_variation, float(e.variation.magnitude))
        deviation = Q(_db(s21) - float(e.gain_nominal.magnitude), "dB")
        bottom.append(
            LinePlot(
                f, deviation, label=variation_label, color=color, width=width, alpha=alpha,
                min_x=ch.f_start.to("MHz"), max_x=ch.f_stop.to("MHz"),
            )
        )
    top += _band_lines(ch, lo, hi)
    top += [
        Title(f"{report_.title} — S21 at the DUT plane, {len(report_.gains)} configurations"),
        XLabel("Frequency"),
        YLabel("S21"),
        GridMajor(),
        Legend(location="best"),
    ]
    # y range: room for the ±Z limit lines and a legend below the curves
    span = 1.3 * largest_variation
    if r is not None:
        z = float(r.passband_variation.to("dB").magnitude)
        span = max(span, 2.5 * z)
        edges = _mhz_axis([_hz(ch.f_start), _hz(ch.f_stop)])
        bottom.append(LinePlot(edges, Q([z, z], "dB"), label=f"limit ±{r.passband_variation:~}", color="red", style="--"))
        bottom.append(LinePlot(edges, Q([-z, -z], "dB"), color="red", style="--"))
    bottom += [
        Title("Passband gain relative to the gain at the band centre"),
        XLabel("Frequency"),
        YLabel("Gain − centre gain"),
        XLimits(ch.f_start.to("MHz"), ch.f_stop.to("MHz")),
        YLimits(Q(-span, "dB"), Q(span, "dB")),
        GridMajor(),
        Legend(location="lower center"),
    ]
    return plot(*top, *bottom, show=show, save_folder=save_folder, filename=filename)


def plot_reflection(
    report_: ChannelReport,
    parameter: str = "S11",
    show: bool = False,
    save_folder: Optional[str] = None,
    filename: Optional[str] = None,
) -> Any:
    """Every configuration's reflection as thin lines, their average bold on top, worst point marked."""
    ch = report_.channel
    avg = report_.reflections[parameter]
    others = iter(_OTHER_COLORS)
    f = _mhz_axis(avg.frequency.to("Hz").magnitude)
    objects: list[Any] = []
    for label, trace in avg.traces.items():
        objects.append(LinePlot(f, trace, label=label, color=next(others, None), width=1.0, alpha=0.6))
    all_values = np.concatenate([_db(t) for t in avg.traces.values()])
    lo, hi = float(np.nanmin(all_values)) - 2.0, float(np.nanmax(all_values)) + 2.0
    objects += _band_lines(ch, lo, hi)
    objects.append(
        LinePlot(
            f, avg.average, color=_AVERAGE_COLOR, width=3.0,
            label=f"average of {len(avg.traces)} configurations (power average)",
        )
    )
    objects.append(
        Marker(
            avg.worst_frequency.to("MHz"), avg.worst_in_band, color=_AVERAGE_COLOR, style="D", size=8.0,
            label=f"worst average in band: {avg.worst_in_band:~.1f} at {avg.worst_frequency.to('MHz'):~.4g}",
        )
    )
    objects += [
        Title(f"{report_.title} — {parameter} at the DUT plane, all configurations and their average"),
        XLabel("Frequency"),
        YLabel(parameter),
        GridMajor(),
        Legend(location="best"),
    ]
    return plot(*objects, show=show, save_folder=save_folder, filename=filename)
