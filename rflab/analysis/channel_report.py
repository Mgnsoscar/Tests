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
    DEFAULT_THEME,
    FigureTitle,
    GridMajor,
    HLine,
    Layout,
    LinePlot,
    Marker,
    Panel,
    Span,
    Swatch,
    Text,
    Theme,
    Title,
    VLine,
    XLabel,
    XLimits,
    XTicks,
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

#: Line widths: the main and max gain configurations (and the average) stand out, the rest are thin.
_BOLD, _THIN = 2.6, 1.5
#: Type sizes (points). The pages are read on a screen across a desk, so nothing is small.
_TITLE, _SUBTITLE, _PANEL, _HEADING, _TEXT, _HEADER, _CHIP, _NOTE = 20.0, 13.5, 13.0, 14.0, 12.5, 11.0, 11.0, 12.0


# --- helpers -------------------------------------------------------------------

def _hz(value: Quantity) -> float:
    return float(value.to("Hz").magnitude)


def _mhz(value: Quantity) -> float:
    return float(value.to("MHz").magnitude)


def _db(values: Quantity) -> np.ndarray:
    return np.asarray(values.magnitude, dtype=np.float64)


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
#
# The layout: the plots on the left carry only the data and the reference
# geometry (passband shaded, band edges and cutoff frequencies as hairlines,
# the cutoff points as dots); every number and verdict lives in a text column
# on the right, one row per configuration, so nothing has to be read off the
# plot. The title line states the answers. Configuration i keeps the theme's
# series colour i on every page.

_COLUMN_LAYOUT = Layout(
    width_ratios=[3.1, 1.35], left=0.065, right=0.985, top=0.86, bottom=0.075, hspace=0.32, wspace=0.05,
)


def _short(label: str) -> str:
    """``"attenuation 12 dB bypass on"`` → ``"att 12 dB, bypass on"`` for the column."""
    return label.replace("attenuation ", "att ").replace(" bypass", ", bypass")


def _palette(report_: ChannelReport, theme: Theme) -> dict[str, str]:
    """``{configuration label: colour}`` — fixed per configuration across the pages."""
    labels = [e.label for e in report_.gains]
    for avg in report_.reflections.values():
        labels += [label for label in avg.traces if label not in labels]
    return {label: theme.series_color(i) for i, label in enumerate(labels)}


def _emphasis(report_: ChannelReport, e: GainEvaluation) -> Optional[str]:
    if e is report_.main and e is report_.max_gain:
        return "main = max gain"
    if e is report_.main:
        return "main gain"
    if e is report_.max_gain:
        return "max gain"
    return None


def _band_geometry(ch: Channel, theme: Theme) -> list[Any]:
    """The passband as a wash with hairline edges, and the cutoff frequencies dotted."""
    objects: list[Any] = [
        Span(ch.f_start.to("MHz"), ch.f_stop.to("MHz")),
        VLine(ch.f_start.to("MHz"), color=theme.axis, width=0.8),
        VLine(ch.f_stop.to("MHz"), color=theme.axis, width=0.8),
    ]
    if ch.requirements is not None:
        y = ch.requirements.cutoff_offset
        objects += [
            VLine((ch.f_start - y).to("MHz"), color=theme.axis, width=0.8, style=":"),
            VLine((ch.f_stop + y).to("MHz"), color=theme.axis, width=0.8, style=":"),
        ]
    return objects


def _frequency_ticks(ch: Channel, sweep: tuple[float, float]) -> XTicks:
    """Ticks at the cutoff frequencies (labelled so), the band edges, the centre and the quarter points."""
    f0, f1, fc = _mhz(ch.f_start), _mhz(ch.f_stop), _mhz(ch.f_center)
    positions = [f0, (f0 + fc) / 2, fc, (fc + f1) / 2, f1]
    labels = [f"{v:g}" for v in positions]
    if ch.requirements is not None:
        y = _mhz(ch.requirements.cutoff_offset)
        positions = [f0 - y, *positions, f1 + y]
        labels = [f"{f0 - y:g}\ncutoff", *labels, f"{f1 + y:g}\ncutoff"]
    lo, hi = sweep
    keep = [(pos, lab) for pos, lab in zip(positions, labels) if lo - 1e-9 <= pos <= hi + 1e-9]
    return XTicks([pos for pos, _ in keep], [lab for _, lab in keep])


def _sweep(frequencies: Iterable[Quantity]) -> tuple[float, float]:
    """The union of the results' sweeps, in MHz."""
    los, his = [], []
    for f in frequencies:
        m = np.asarray(f.to("MHz").magnitude, dtype=np.float64)
        los.append(float(m.min()))
        his.append(float(m.max()))
    return min(los), max(his)


def _verdict_style(ok: Optional[bool], theme: Theme) -> tuple[str, str]:
    if ok is None:
        return "not checked", theme.muted
    return ("✓ PASS", theme.good) if ok else ("✗ FAIL", theme.critical)


def _gain_column(report_: ChannelReport, colors: dict[str, str], theme: Theme) -> list[Any]:
    """The results column of the S21 page: one row per configuration, then the requirements."""
    ch, r = report_.channel, report_.channel.requirements
    objects: list[Any] = [Panel(0, 1, row_span=2, axes=False)]
    y = 0.98
    objects.append(Text("Configurations", 0.0, y, weight="bold", size=_HEADING))
    y -= 0.055
    columns = [("gain at\ncentre", 0.44), ("variation\nin band", 0.68), ("cutoff below\n/ above", 1.0)]
    for header, x in columns:
        objects.append(Text(header, x, y, color=theme.muted, size=_HEADER, h_align="right", line_spacing=1.1))
    y -= 0.075
    # the rows share the space above the requirements block (about half the column)
    row = min(0.125, 0.52 / max(len(report_.gains), 1))
    for e in report_.gains:
        tag = _emphasis(report_, e)
        width = _BOLD if tag else _THIN
        objects.append(Swatch(0.0, y - 0.013, colors[e.label], width=width))
        name = _short(e.label) + (f"  ({tag})" if tag else "")
        objects.append(Text(name, 0.085, y, size=_TEXT, weight="bold" if tag else None))
        y -= 0.045
        objects.append(Text(f"{e.gain_nominal.magnitude:+.1f} dB", 0.44, y, color=theme.ink_secondary, size=_TEXT, h_align="right"))
        objects.append(Text(f"±{e.variation.magnitude:.2f}", 0.68, y, color=theme.ink_secondary, size=_TEXT, h_align="right"))
        if e.rejection_below is not None and e.rejection_above is not None:
            cutoff = f"{e.rejection_below.magnitude:.1f} / {e.rejection_above.magnitude:.1f}"
        else:
            cutoff = "—" if r is None else "not in sweep"
        objects.append(Text(cutoff, 1.0, y, color=theme.ink_secondary, size=_TEXT, h_align="right"))
        if r is not None:
            for ok, x in ((e.variation_ok, 0.68), (e.rejection_ok, 1.0)):
                text, color = _verdict_style(ok, theme)
                objects.append(Text(text, x, y - 0.036, color=color, size=_CHIP, weight="bold", h_align="right"))
        y -= row - 0.045
    y -= 0.02
    objects.append(Text("Requirements", 0.0, y, weight="bold", size=_HEADING))
    y -= 0.05
    if r is None:
        objects.append(Text("None defined for this channel.", 0.0, y, color=theme.ink_secondary, size=_NOTE))
        y -= 0.05
    else:
        f0, f1 = _mhz(ch.f_start), _mhz(ch.f_stop)
        yc = _mhz(r.cutoff_offset)
        lines = [
            f"Cutoff: S21 ≥ {r.cutoff_rejection:~} below the centre gain\n"
            f"   at {r.cutoff_offset:~} outside the band ({f0 - yc:g} and {f1 + yc:g} MHz)",
            f"Passband: within ±{r.passband_variation:~} of the centre gain\n   between {f0:g} and {f1:g} MHz",
        ]
        for line in lines:
            objects.append(Text(line, 0.0, y, color=theme.ink_secondary, size=_NOTE, line_spacing=1.25))
            y -= 0.085
    objects.append(Text(
        "Main gain: highest attenuation, bypass on\nMax gain: no attenuation, bypass off",
        0.0, y, color=theme.ink_secondary, size=_NOTE, line_spacing=1.25,
    ))
    y -= 0.085
    if r is not None:
        text, color = _verdict_style(report_.passed, theme)
        objects.append(Text(f"All configurations: {text}", 0.0, y - 0.01, color=color, size=_HEADING, weight="bold"))
    return objects


def plot_gain(
    report_: ChannelReport,
    show: bool = False,
    save_folder: Optional[str] = None,
    filename: Optional[str] = None,
    theme: Theme = DEFAULT_THEME,
) -> Any:
    """The S21 page: gain over the measured range, the passband variation against
    its limit, and a results column with every configuration's numbers and verdicts."""
    ch, r = report_.channel, report_.channel.requirements
    colors = _palette(report_, theme)
    f0, f1, fc = ch.f_start.to("MHz"), ch.f_stop.to("MHz"), ch.f_center.to("MHz")
    sweep = _sweep(e.result.frequency for e in report_.gains)
    room = 0.12 * (sweep[1] - sweep[0])   # space for the direct labels at the right end

    main, max_gain = report_.main, report_.max_gain
    subtitle = []
    if main is not None:
        subtitle.append(f"Main gain {main.gain_nominal.magnitude:.1f} dB")
    if max_gain is not None:
        subtitle.append(f"Max gain {max_gain.gain_nominal.magnitude:.1f} dB")
    subtitle.append(f"band {_mhz(f0):g}–{_mhz(f1):g} MHz, centre {_mhz(fc):g} MHz")
    subtitle.append("DUT plane, latest result per configuration")

    # thin configurations first, the emphasised ones on top
    ordered = sorted(report_.gains, key=lambda e: _emphasis(report_, e) is not None)

    top: list[Any] = [Panel(0, 0), *_band_geometry(ch, theme)]
    bottom: list[Any] = [Panel(1, 0), *_band_geometry(ch, theme)]
    largest_variation = 0.0
    for e in ordered:
        tag = _emphasis(report_, e)
        color = colors[e.label]
        width, alpha = (_BOLD, 1.0) if tag else (_THIN, 0.75)
        f = e.result.frequency.to("MHz")
        s21 = e.result.magnitudes["S21"]
        top.append(LinePlot(f, s21, color=color, width=width, alpha=alpha))
        if e.rejection_below is not None and e.rejection_above is not None and r is not None:
            yc = r.cutoff_offset
            for x, rejection in ((ch.f_start - yc, e.rejection_below), (ch.f_stop + yc, e.rejection_above)):
                level = Q(float(e.gain_nominal.magnitude - rejection.magnitude), "dB")
                top.append(Marker(x.to("MHz"), level, color=color, size=8 if tag else 6,
                                  edge_color=theme.surface, edge_width=1.4))
        largest_variation = max(largest_variation, float(e.variation.magnitude))
        deviation = Q(_db(s21) - float(e.gain_nominal.magnitude), "dB")
        bottom.append(LinePlot(f, deviation, color=color, width=width, alpha=alpha, min_x=f0, max_x=f1))

    # the cutoff requirement, drawn once for the main-gain configuration: a short limit dash
    top_title = "Gain over the measured range · passband shaded · dots = cutoff points"
    if r is not None and main is not None and main.rejection_above is not None:
        yc, x_lim = r.cutoff_offset, r.cutoff_rejection
        level = Q(float(main.gain_nominal.magnitude - x_lim.magnitude), "dB")
        dash = Q(3, "MHz")
        for x in (ch.f_start - yc, ch.f_stop + yc):
            top.append(HLine(level, color=theme.ink_secondary, style="--", width=1.2,
                             from_x=(x - dash).to("MHz"), to_x=(x + dash).to("MHz")))
        top_title += f" · dashes = the main gain's limit ({x_lim:~} below its centre gain)"
    top += [
        _frequency_ticks(ch, sweep),
        XLimits(Q(sweep[0] - 0.02 * (sweep[1] - sweep[0]), "MHz"), Q(sweep[1] + room, "MHz")),
        Title(top_title, align="left", size=_PANEL, color=theme.ink_secondary),
        YLabel("S21"),
        GridMajor(axis="y"),
    ]

    span = 1.3 * largest_variation
    bottom_title = "Passband variation relative to the gain at the band centre"
    if r is not None:
        z = float(r.passband_variation.to("dB").magnitude)
        span = max(span, 1.5 * z)
        bottom += [
            HLine(Q(z, "dB"), color=theme.ink_secondary, style="--", width=1.2),
            HLine(Q(-z, "dB"), color=theme.ink_secondary, style="--", width=1.2),
        ]
        bottom_title += f" · dashes = limit ±{r.passband_variation:~}"
    bottom += [
        XLimits(f0 - Q(1, "MHz"), f1 + Q(1, "MHz")),
        YLimits(Q(-span, "dB"), Q(span, "dB")),
        XTicks([_mhz(f0), (_mhz(f0) + _mhz(fc)) / 2, _mhz(fc), (_mhz(fc) + _mhz(f1)) / 2, _mhz(f1)]),
        Title(bottom_title, align="left", size=_PANEL, color=theme.ink_secondary),
        XLabel("Frequency"),
        YLabel("gain − centre gain"),
        GridMajor(axis="y"),
    ]

    layout = Layout(**{**_COLUMN_LAYOUT.__dict__, "height_ratios": [2.1, 1.0]})
    return plot(
        layout,
        FigureTitle(f"{report_.title} — S21 (gain), {len(report_.gains)} configurations",
                    subtitle="  ·  ".join(subtitle), size=_TITLE, subtitle_size=_SUBTITLE),
        *top, *bottom, *_gain_column(report_, colors, theme),
        figsize=(15, 10), show=show, save_folder=save_folder, filename=filename, theme=theme,
    )


def plot_reflection(
    report_: ChannelReport,
    parameter: str = "S11",
    show: bool = False,
    save_folder: Optional[str] = None,
    filename: Optional[str] = None,
    theme: Theme = DEFAULT_THEME,
) -> Any:
    """The S11 (or S22) page: every configuration thin, the power average bold on top,
    the worst in-band point of the average marked, and a column with each configuration's worst."""
    ch = report_.channel
    avg = report_.reflections[parameter]
    colors = _palette(report_, theme)
    f = avg.frequency.to("MHz")
    f_hz = np.asarray(avg.frequency.to("Hz").magnitude, dtype=np.float64)
    in_band = _band_mask(f_hz, ch)
    sweep = _sweep([avg.frequency])
    worst_label, worst_level, worst_f = avg.worst_individual

    # y range: the data plus a margin, but never tighter than 10 dB — a flat
    # match must read as flat, not as a noise band filling the panel
    all_values = np.concatenate([_db(tr) for tr in avg.traces.values()])
    lo, hi = float(np.nanmin(all_values)) - 1.0, float(np.nanmax(all_values)) + 1.0
    if hi - lo < 10.0:
        mid = (lo + hi) / 2
        lo, hi = mid - 5.0, mid + 5.0
    lo, hi = float(np.floor(lo)), float(np.ceil(hi))

    objects: list[Any] = [Panel(0, 0), *_band_geometry(ch, theme)]
    for label, trace in avg.traces.items():
        objects.append(LinePlot(f, trace, color=colors[label], width=_THIN, alpha=0.6))
    objects += [
        YLimits(Q(lo, "dB"), Q(hi, "dB")),
        LinePlot(f, avg.average, color=theme.ink, width=_BOLD),
        Marker(avg.worst_frequency.to("MHz"), avg.worst_in_band, color=theme.ink, size=10, style="D",
               edge_color=theme.surface, edge_width=1.5),
        _frequency_ticks(ch, sweep),
        XLimits(Q(sweep[0] - 0.02 * (sweep[1] - sweep[0]), "MHz"), Q(sweep[1] + 0.02 * (sweep[1] - sweep[0]), "MHz")),
        Title("Every configuration thin · the power average bold · passband shaded · "
              "◆ = worst of the average in band", align="left", size=_PANEL, color=theme.ink_secondary),
        XLabel("Frequency"),
        YLabel(parameter),
        GridMajor(axis="y"),
    ]

    column: list[Any] = [Panel(0, 1, axes=False)]
    y = 0.98
    column.append(Text("Configurations", 0.0, y, weight="bold", size=_HEADING))
    y -= 0.06
    column.append(Text(f"worst {parameter}\nin band", 1.0, y, color=theme.muted, size=_HEADER, h_align="right", line_spacing=1.1))
    y -= 0.085
    row = min(0.075, 0.45 / (len(avg.traces) + 1))
    for label, trace in avg.traces.items():
        worst = float(np.max(_db(trace)[in_band])) if in_band.any() else float(np.max(_db(trace)))
        column.append(Swatch(0.0, y - 0.014, colors[label], width=_THIN, alpha=0.8))
        column.append(Text(_short(label), 0.085, y, size=_TEXT))
        column.append(Text(f"{worst:.1f} dB", 1.0, y, color=theme.ink_secondary, size=_TEXT, h_align="right"))
        y -= row
    column.append(Swatch(0.0, y - 0.014, theme.ink, width=_BOLD))
    column.append(Text("average (power average)", 0.085, y, size=_TEXT, weight="bold"))
    column.append(Text(f"{avg.worst_in_band.magnitude:.1f} dB", 1.0, y, size=_TEXT, weight="bold", h_align="right"))
    y -= row + 0.02
    column.append(Text(
        f"◆ worst of the average in band: {avg.worst_in_band.magnitude:.1f} dB\n"
        f"   at {_mhz(avg.worst_frequency):g} MHz",
        0.085, y, color=theme.ink_secondary, size=_NOTE, line_spacing=1.25,
    ))
    y -= 0.13
    column.append(Text("How the average is taken", 0.0, y, weight="bold", size=_HEADING))
    y -= 0.06
    column.append(Text(
        f"|{parameter}|² of every configuration averaged\nat each frequency, then back to dB —\n"
        "the mean reflected power, not the\nmean of the dB values.",
        0.0, y, color=theme.ink_secondary, size=_NOTE, line_spacing=1.3,
    ))

    subtitle = (
        f"Average {parameter} in band: worst {avg.worst_in_band.magnitude:.1f} dB at {_mhz(avg.worst_frequency):g} MHz"
        f"  ·  single worst configuration {worst_level.magnitude:.1f} dB ({worst_label})  ·  DUT plane"
    )
    layout = Layout(**{**_COLUMN_LAYOUT.__dict__, "top": 0.83, "bottom": 0.10})
    return plot(
        layout,
        FigureTitle(f"{report_.title} — {parameter}, average over {len(avg.traces)} configurations",
                    subtitle=subtitle, size=_TITLE, subtitle_size=_SUBTITLE),
        *objects, *column,
        figsize=(15, 7.5), show=show, save_folder=save_folder, filename=filename, theme=theme,
    )
