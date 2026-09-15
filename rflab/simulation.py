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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from labkit.instruments import FSV3007, N5183A, TGR6000, ZNLE18
from labkit.instruments.mock import MockBackend, mock_instrument

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

    def output_level(self, p_in_dbm: float) -> float:
        """Soft-limited output level: linear gain that rolls into `p_sat_dbm`."""
        linear_out = p_in_dbm + self.gain_db
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

    def __init__(self, model: Optional[AmplifierModel] = None) -> None:
        self.model = model or AmplifierModel()
        self.gen_a, self.gen_a_backend = mock_instrument(
            N5183A, name="Signal generator A", responses={"*IDN?": "Agilent Technologies, N5183A, SIM, 1.0"}
        )
        self.gen_b, self.gen_b_backend = mock_instrument(
            TGR6000, name="Signal generator B", responses={"*IDN?": "THURLBY THANDAR,TGR6000,SIM,1.0"}
        )
        self.fsv, self.fsv_backend = mock_instrument(FSV3007, name="Spectrum analyzer", responses=self._fsv)
        self.vna, self.vna_backend = mock_instrument(ZNLE18, name="Network analyzer", responses=self._vna)

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

    # -- spectrum analyzer ---------------------------------------------------
    def _spectrum_level(self, f_hz: float, tolerance_hz: float = 1e3) -> float:
        """The output level at `f_hz`: a tone, an IM3 product, or the noise floor."""
        tones = self._tones()
        for f, p in tones:
            if abs(f - f_hz) <= tolerance_hz:
                return self.model.output_level(p)
        if len(tones) == 2:
            (f1, p1), (f2, p2) = tones
            p_out = self.model.output_level(min(p1, p2))
            for f_im3 in (2 * f1 - f2, 2 * f2 - f1):
                if abs(f_im3 - f_hz) <= tolerance_hz:
                    return self.model.im3_level(p_out)
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
            tones = self._tones()
            p_out = self.model.output_level(tones[0][1]) if tones else self.model.noise_floor_dbm
            levels = [p_out + self.model.harmonic_dbc * (k - 1) if tones else p_out for k in range(1, n + 1)]
            return ",".join(_num(v) for v in levels)
        if query == "CALC:MARK:FUNC:TOI:RES?":
            return _num(self.model.oip3_dbm)
        if query.startswith("TRAC1:DATA? TRACE1,"):
            points = int(_last_float(w, r"^FREQ:POIN (\d+)$", 1))
            kind = query.rsplit(",", 1)[1]
            value = self.model.noise_figure_db if kind == "NOIS" else self.model.gain_db
            return ",".join([_num(value)] * points)
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
            db = {"S11": self.model.s11_db, "S22": self.model.s22_db}.get(parameter, -30.0)
            magnitude = _num(10 ** (db / 20))
            return ",".join(f"{magnitude},0.0" for _ in range(points))
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
