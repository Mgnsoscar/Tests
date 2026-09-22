"""A simulated bench: real drivers, scripted instruments, a modelled amplifier.

Every driver is wired to a LabKit ``MockBackend`` whose responses come from a
small behavioural amplifier model, so the whole project — measurements, the
store, analysis, plots — runs end to end with no hardware:

    from rflab.simulation import simulated_bench
    bench = simulated_bench()
    result = compression.measure(bench, DUT, DUT.channel(1))

The model is deliberately simple (flat gain with a soft saturation, fixed
noise figure, fixed IM3 intercept, harmonics at a fixed dBc per order, fixed
return loss) and does **not** include the bench's path losses: it behaves as if
the instruments were connected straight to the DUT. That makes the simulated
raw data "instrument-plane" numbers with zero path loss, which is fine for
exercising the code but means ``at_dut`` will *add* the configured path
losses to it — expected, and visible in the tests.

With a `passband` the model becomes a filtered amplifier: the gain is flat
(with a little ripple) inside the band and rolls off outside it like a
Butterworth band-pass, which is what the channel report's cutoff and
passband-variation checks need to see. The DUT's switch settings —
`attenuation_db` and `bypass` — reduce the gain the way the real device's
would; :meth:`SimulatedBench.set_dut` copies them from a channel and a run
script's ``state`` so a ``--simulate`` run behaves like the configured DUT.
``set_dut`` also connects the channel's signal paths: from then on the
simulated instruments see the DUT *through* those paths (tones arrive at the
DUT minus the input loss, output levels and S-parameters reach the analyzer
minus the output loss), so the raw data really is instrument-plane data and
``at_dut`` recovers the model's values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional

import numpy as np

from labkit.instruments import FSV3007, N5183A, RTO64, TGR6000, ZNLE18
from labkit.instruments.mock import MockBackend, mock_instrument
from labkit.signal_path import SignalPath
from labkit.units import quantity

from .dut import Channel

__all__ = ["AmplifierModel", "SimulatedBench", "simulated_bench"]


@dataclass
class AmplifierModel:
    """The behavioural model behind the simulated instruments (levels in dBm/dB)."""

    gain_db: float = 20.0
    #: Output saturation level; P1dB (output) lands about 1 dB below it.
    p_sat_dbm: float = 12.0
    noise_figure_db: float = 3.0
    oip3_dbm: float = 22.0
    #: Each harmonic order n > 1 sits at ``harmonic_dbc * (n - 1)`` below the fundamental.
    harmonic_dbc: float = -25.0
    s11_db: float = -15.0
    s22_db: float = -12.0
    noise_floor_dbm: float = -100.0

    # -- the filtered amplifier (only with a passband) ------------------------
    #: ``(f_start_hz, f_stop_hz)`` of the passband; ``None`` = flat gain at every frequency.
    passband_hz: Optional[tuple[float, float]] = None
    #: Order of the Butterworth-like band-pass roll-off outside the band.
    filter_order: int = 6
    #: How far outside each band edge the response is 3 dB down.
    edge_margin_hz: float = 5e6
    #: Peak-to-peak gain ripple inside the band.
    ripple_db: float = 0.4
    #: RMS measurement noise added to every VNA reading (0 = exact, as the tests need).
    noise_db: float = 0.0

    # -- the DUT's switch settings (set by hand on the real device) ----------
    #: The DUT's own attenuator setting; it takes this much off the gain.
    attenuation_db: float = 0.0
    #: Bypass on: the `bypass_stage_gain_db` stage is switched out of the chain.
    bypass: bool = False
    bypass_stage_gain_db: float = 10.0

    # -- gain -----------------------------------------------------------------
    def _relative(self, f_hz: float) -> float:
        """Signed distance from the band centre in units of the 3 dB half-width (0 inside)."""
        assert self.passband_hz is not None
        f_start, f_stop = self.passband_hz
        centre = (f_start + f_stop) / 2
        half = (f_stop - f_start) / 2 + self.edge_margin_hz
        return (f_hz - centre) / half

    def s21_db(self, f_hz: float) -> float:
        """The small-signal gain at `f_hz` in the current configuration (state and filter)."""
        gain = self.gain_db - self.attenuation_db - (self.bypass_stage_gain_db if self.bypass else 0.0)
        if self.passband_hz is None:
            return gain
        x = self._relative(f_hz)
        rolloff = -10 * np.log10(1 + x ** (2 * self.filter_order))
        ripple = -(self.ripple_db / 2) * (1 - np.cos(2 * np.pi * 1.5 * x))  # 0 at the centre, dips in between
        return float(gain + rolloff + ripple)

    def s11_db_at(self, f_hz: float) -> float:
        """S11 in the current configuration: a slightly worse match with the bypass on and a
        gentle slope across the band (flat `s11_db` without a passband)."""
        value = self.s11_db + (2.0 if self.bypass else 0.0) - 0.1 * self.attenuation_db
        if self.passband_hz is not None:
            value += 1.5 * self._relative(f_hz)
        return float(value)

    def output_level(self, p_in_dbm: float, f_hz: Optional[float] = None) -> float:
        """Soft-limited output level: linear gain that rolls into `p_sat_dbm`."""
        gain = self.gain_db if f_hz is None else self.s21_db(f_hz)
        linear_out = p_in_dbm + gain
        return float(self.p_sat_dbm - 10 * np.log10(1 + 10 ** ((self.p_sat_dbm - linear_out) / 10)))

    def im3_level(self, p_tone_out_dbm: float) -> float:
        return p_tone_out_dbm - 2 * (self.oip3_dbm - p_tone_out_dbm)

    def p1db_output(self) -> float:
        """The output level at exactly 1 dB compression, from :meth:`output_level`'s curve."""
        # compression c(d) = 10 log10(1 + 10^(d/10)) - d with d = p_sat - (p_in + gain); solve c = 1
        d = -10 * np.log10(10 ** 0.1 - 1)
        return float(self.p_sat_dbm - 10 * np.log10(1 + 10 ** (d / 10)))


def _last(writes: list[str], pattern: str) -> Optional[str]:
    """The captured group of the last write matching `pattern`, or ``None``."""
    regex = re.compile(pattern)
    for command in reversed(writes):
        match = regex.match(command)
        if match:
            return match.group(1)
    return None


def _last_float(writes: list[str], pattern: str, default: float) -> float:
    text = _last(writes, pattern)
    return float(text) if text is not None else default


def _num(value: float) -> str:
    """A plain decimal for a simulated response (never numpy's ``np.float64(...)`` repr)."""
    return repr(float(value))


class SimulatedBench:
    """Drivers on mock backends answering from an :class:`AmplifierModel`."""

    fsv: FSV3007
    vna: ZNLE18
    gen_a: N5183A
    gen_b: TGR6000
    scope: RTO64

    def __init__(self, model: Optional[AmplifierModel] = None) -> None:
        self.model = model or AmplifierModel()
        #: The signal paths between the instruments and the DUT ports (none until ``set_dut``).
        self.paths: dict[str, SignalPath] = {}
        self._rng = np.random.default_rng(seed=7)
        #: The simulated contact: when it is open, as ``(opens, closes)`` seconds into every
        #: single run of the scope. A 2 µs dropout, a 100 µs one (long enough for the scope
        #: to trigger on both of its edges) and a bounce (open 1 µs, closed 2 µs, open 1 µs).
        self.contact_openings: list[tuple[float, float]] = [
            (0.5, 0.5 + 2e-6), (1.0, 1.0 + 100e-6), (1.5, 1.5 + 1e-6), (1.5 + 3e-6, 1.5 + 4e-6),
        ]
        #: The simulated scope cannot trigger for this long after a record ends (re-arm time).
        self.blind_seconds = 1e-6
        #: The simulated scope's clock at the start of its first run; each run starts a minute later.
        self.scope_epoch = datetime(2026, 9, 21, 8, 0, 0)
        self.scope, self.scope_backend = mock_instrument(RTO64, name="Oscilloscope", responses=self._scope)
        self.gen_a, self.gen_a_backend = mock_instrument(
            N5183A, name="Signal generator A", responses={"*IDN?": "Agilent Technologies, N5183A, SIM, 1.0"}
        )
        self.gen_b, self.gen_b_backend = mock_instrument(
            TGR6000, name="Signal generator B", responses={"*IDN?": "THURLBY THANDAR,TGR6000,SIM,1.0"}
        )
        self.fsv, self.fsv_backend = mock_instrument(FSV3007, name="Spectrum analyzer", responses=self._fsv)
        self.vna, self.vna_backend = mock_instrument(ZNLE18, name="Network analyzer", responses=self._vna)

    def set_dut(self, channel: Channel, state: Optional[Mapping[str, Any]] = None) -> None:
        """Put the modelled DUT into `channel`'s band and the run script's `state`.

        The real DUT is set by hand; this is the simulated equivalent. The
        passband is the channel's band, ``state["attenuation"]`` (a dB
        quantity) and ``state["bypass"]`` (a bool) set the switches, and the
        channel's signal paths are "connected" between the instruments and
        the DUT.
        """
        self.paths = dict(channel.ports)
        self.model.passband_hz = (
            float(channel.f_start.to("Hz").magnitude),
            float(channel.f_stop.to("Hz").magnitude),
        )
        state = state or {}
        attenuation = state.get("attenuation")
        self.model.attenuation_db = float(attenuation.to("dB").magnitude) if attenuation is not None else 0.0
        self.model.bypass = bool(state.get("bypass", False))

    # -- generator state (from what the drivers wrote) -----------------------
    def _tone_a(self) -> tuple[float, float, bool]:
        w = self.gen_a_backend.writes
        f = _last_float(w, r"^FREQ:CW (\S+)$", 1e9)
        p = _last_float(w, r"^POW (\S+?)DBM$", -100.0)
        on = _last(w, r"^OUTP (ON|OFF)$") == "ON"
        return f, p, on

    def _tone_b(self) -> tuple[float, float, bool]:
        w = self.gen_b_backend.writes
        f = _last_float(w, r"^FREQ (\S+)$", 1.0) * 1e6
        p = _last_float(w, r"^DBMLEV (\S+)$", -100.0)
        on = _last(w, r"^(RFON|RFOFF)$") == "RFON"
        return f, p, on

    def _tones(self) -> list[tuple[float, float]]:
        """``[(frequency_hz, input_level_dbm)]`` of the tones currently on."""
        return [(f, p) for f, p, on in (self._tone_a(), self._tone_b()) if on]

    # -- the paths between instruments and DUT --------------------------------
    def _loss(self, port: str, f_hz: float) -> float:
        """The connected path's loss on `port` at `f_hz` in dB (0 with no path)."""
        path = self.paths.get(port)
        if path is None or len(path) == 0:
            return 0.0
        return float(path.loss_at(quantity(f_hz, "Hz"), extrapolate=True).magnitude)

    def _tones_at_dut(self) -> list[tuple[float, float]]:
        """The tones as they arrive at the DUT input: generator level minus the input path."""
        return [(f, p - self._loss("in", f)) for f, p in self._tones()]

    # -- spectrum analyzer ---------------------------------------------------
    def _spectrum_level(self, f_hz: float, tolerance_hz: float = 1e3) -> float:
        """The level at the analyzer at `f_hz`: a tone, an IM3 product, or the noise floor."""
        tones = self._tones_at_dut()
        for f, p in tones:
            if abs(f - f_hz) <= tolerance_hz:
                return self.model.output_level(p, f) - self._loss("out", f)
        if len(tones) == 2:
            (f1, p1), (f2, p2) = tones
            p_out = self.model.output_level(min(p1, p2), (f1 + f2) / 2)
            for f_im3 in (2 * f1 - f2, 2 * f2 - f1):
                if abs(f_im3 - f_hz) <= tolerance_hz:
                    return self.model.im3_level(p_out) - self._loss("out", f_im3)
        return self.model.noise_floor_dbm

    def _fsv(self, query: str) -> str:
        w = self.fsv_backend.writes
        if query == "*IDN?":
            return "Rohde&Schwarz,FSV3007,SIM,1.0"
        if query == "*OPC?":
            return "1"
        if query == "INST:LIST?":
            return ""
        if query == "CALC:MARK1:Y?":
            if _last(w, r"^(CALC:MARK1:MAX)$") is not None and (
                w[::-1].index("CALC:MARK1:MAX") < _index_from_end(w, r"^CALC:MARK1:X ")
            ):
                tones = self._tones()
                f = tones[0][0] if tones else 0.0
            else:
                f = _last_float(w, r"^CALC:MARK1:X (\S+)$", 0.0)
            return _num(self._spectrum_level(f))
        if query == "CALC:MARK:FUNC:HARM:LIST?":
            n = int(_last_float(w, r"^CALC:MARK:FUNC:HARM:NHAR (\d+)$", 1))
            tones = self._tones_at_dut()
            if not tones:
                return ",".join([_num(self.model.noise_floor_dbm)] * n)
            f0, p_in = tones[0]
            p_out = self.model.output_level(p_in, f0)
            levels = [p_out + self.model.harmonic_dbc * (k - 1) - self._loss("out", k * f0) for k in range(1, n + 1)]
            return ",".join(_num(v) for v in levels)
        if query == "CALC:MARK:FUNC:TOI:RES?":
            tones = self._tones()
            f = tones[0][0] if tones else 1e9
            return _num(self.model.oip3_dbm - self._loss("out", f))
        if query.startswith("TRAC1:DATA? TRACE1,"):
            points = int(_last_float(w, r"^FREQ:POIN (\d+)$", 1))
            kind = query.rsplit(",", 1)[1]
            if kind == "NOIS":
                return ",".join([_num(self.model.noise_figure_db)] * points)
            start = _last_float(w, r"^FREQ:STAR (\S+)$", 1e9)
            stop = _last_float(w, r"^FREQ:STOP (\S+)$", start)
            grid = np.linspace(start, stop, points) if points > 1 else np.array([(start + stop) / 2])
            return ",".join(_num(self.model.s21_db(f)) for f in grid)
        return ""

    # -- network analyzer ----------------------------------------------------
    def _vna(self, query: str) -> str:
        w = self.vna_backend.writes
        if query == "*IDN?":
            return "Rohde&Schwarz,ZNLE18,SIM,1.0"
        if query == "*OPC?":
            return "1"
        if query.endswith("CORR:STAT?"):
            # correction is on once a cal was loaded or explicitly enabled
            return "1" if _last(w, r"^(MMEM:LOAD:CORR .*|SENS1:CORR:STAT ON)$") else "0"
        if query.endswith("CORR:DATE?"):
            return "'2026-09-15 10:02:11'"
        if query.endswith("CORR:SST?"):
            return "'CAL OK'"
        start = _last_float(w, r"^.*FREQ:STAR (\S+)$", 1e9)
        stop = _last_float(w, r"^.*FREQ:STOP (\S+)$", 2e9)
        points = int(_last_float(w, r"^.*SWE:POIN (\d+)$", 2))
        if query.endswith("DATA:STIM?"):
            return ",".join(_num(v) for v in np.linspace(start, stop, points))
        if query.endswith("DATA? SDAT"):
            selected = _last(w, r"^CALC1:PAR:SEL '(\w+)'$") or "Trc1"
            parameter = _last(w, rf"^CALC1:PAR:SDEF '{selected}','(S\d\d)'$") or "S11"
            grid = np.linspace(start, stop, points)
            if parameter == "S11":
                db = [self.model.s11_db_at(f) - 2 * self._loss("in", f) for f in grid]
            elif parameter == "S22":
                db = [self.model.s22_db - 2 * self._loss("out", f) for f in grid]
            elif parameter in ("S21", "S12"):
                db = [self.model.s21_db(f) - self._loss("in", f) - self._loss("out", f) for f in grid]
            else:
                db = [-30.0] * points
            if self.model.noise_db > 0:
                noise = self._rng.normal(0.0, self.model.noise_db, len(db))
                db = [v + n for v, n in zip(db, noise)]
            return ",".join(f"{_num(10 ** (v / 20))},0.0" for v in db)
        return ""


    # -- oscilloscope ----------------------------------------------------------
    def _scope_run(self) -> tuple[list[float], Optional[datetime]]:
        """The trigger times (seconds into the run) of the last single run, and the run's start on the scope clock.

        Behaves like the instrument: an edge triggers only when its slope is
        selected and the scope is armed — not during the record after a
        trigger and not in the blind time after it; ``TRIG1:FORC`` after
        ``RUNS`` adds a forced acquisition at the start; the run stops at the
        fast-segmentation maximum.
        """
        w = self.scope_backend.writes
        runs = [i for i, command in enumerate(w) if command == "RUNS"]
        if not runs:
            return [], None
        forced = "TRIG1:FORC" in w[runs[-1]:]
        window = _last_float(w, r"^TIM:RANG (\S+)$", 5e-5)
        reference = _last_float(w, r"^TIM:REF (\S+)$", 20.0) / 100
        post = window * (1 - reference)
        max_segments = int(_last_float(w, r"^ACQ:SEGM:MAX (\S+)$", 1e6))
        slope = _last(w, r"^TRIG1:EDGE:SLOP (\S+)$") or "POS"
        crossings = sorted(
            [(a, True) for a, _ in self.contact_openings] + [(b, False) for _, b in self.contact_openings]
        )
        triggers: list[float] = []
        armed_at = 0.0
        if forced:
            triggers.append(0.0)
            armed_at = post + self.blind_seconds
        for t, falling in crossings:
            if (slope == "NEG" and not falling) or (slope == "POS" and falling):
                continue
            if t < armed_at or len(triggers) >= max_segments:
                continue
            triggers.append(t)
            armed_at = t + post + self.blind_seconds
        return triggers, self.scope_epoch + timedelta(seconds=60 * (len(runs) - 1))

    def _contact_record(self, trigger: float, t: np.ndarray, peak_detect: bool) -> np.ndarray:
        """The contact voltage around `trigger`: 1 V closed, 0 V open; min/max per sample interval for peak detect."""
        at = trigger + t
        step = float(t[1] - t[0]) if len(t) > 1 else 0.0
        if not peak_detect:
            v = np.ones(len(t))
            for a, b in self.contact_openings:
                v[(at >= a) & (at < b)] = 0.0
            return v
        low = np.ones(len(t))
        high = np.ones(len(t))
        for a, b in self.contact_openings:
            low[(at < b) & (at + step > a)] = 0.0          # the sample interval touches the opening
            high[(at >= a) & (at + step <= b)] = 0.0       # the sample interval lies inside it
        return np.column_stack([low, high]).ravel()

    def _scope(self, query: str) -> str:
        """The RTO64 watching the simulated contact of :attr:`contact_openings`."""
        w = self.scope_backend.writes
        if query == "*IDN?":
            return "Rohde&Schwarz,RTO,SIM,1.0"
        if query == "*OPC?":
            return "1"
        if query == "SYST:ERR?":
            return '0,"No error"'
        now = datetime.now()
        if query == "SYST:DATE?":
            return f"{now.year},{now.month},{now.day}"
        if query == "SYST:TIME?":
            return f"{now.hour},{now.minute},{now.second}"
        triggers, run_start = self._scope_run()
        count = len(triggers)
        if query == "ACQ:AVA?":
            return str(count)
        index = int(_last_float(w, r"^CHAN\d:WAV1:HIST:CURR (\S+)$", 0.0))
        if query.endswith("HIST:CURR?"):
            return str(index)
        if not triggers or run_start is None:
            return ""
        trigger = triggers[count - 1 + index]
        stamp = run_start + timedelta(seconds=trigger)
        if query.endswith("HIST:TSD?"):
            return f"'{stamp.date().isoformat()}'"
        if query.endswith("HIST:TSAB?"):
            return f"'{stamp.strftime('%H:%M:%S.%f')}000'"
        if query.endswith("HIST:TSR?"):
            return _num(trigger - triggers[-1])
        window = _last_float(w, r"^TIM:RANG (\S+)$", 5e-5)
        reference = _last_float(w, r"^TIM:REF (\S+)$", 20.0) / 100
        rate = _last_float(w, r"^ACQ:SRAT (\S+)$", 2e8)
        peak_detect = (_last(w, r"^CHAN\d:WAV1:ARIT (\S+)$") or "OFF") == "PDET"
        points = int(round(window * rate)) + 1
        if query.endswith("DATA:HEAD?"):
            return f"{_num(-window * reference)},{_num(window * (1 - reference))},{points},{2 if peak_detect else 1}"
        if query.endswith("DATA?"):
            t = np.linspace(-window * reference, window * (1 - reference), points)
            return ",".join(_num(x) for x in self._contact_record(trigger, t, peak_detect))
        return ""


def _index_from_end(writes: list[str], pattern: str) -> int:
    """Distance from the end of `writes` to the last write matching `pattern` (large if none)."""
    regex = re.compile(pattern)
    for i, command in enumerate(reversed(writes)):
        if regex.match(command):
            return i
    return len(writes) + 1


def simulated_bench(model: Optional[AmplifierModel] = None) -> SimulatedBench:
    """A fresh simulated bench (new mock instruments every call)."""
    return SimulatedBench(model)
