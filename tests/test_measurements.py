"""Every measurement against the simulated bench: SCPI it sends, data it returns."""

from __future__ import annotations

import numpy as np
import pytest

from labkit.units import quantity as Q

from rflab.dut import DUT
from rflab.measurements import (
    CompressionSettings,
    HarmonicsSettings,
    IntermodulationSettings,
    NoiseFigureSettings,
    SParameterSettings,
    compression,
    harmonics,
    intermodulation,
    noise_figure,
    s_parameters,
)
from rflab.simulation import AmplifierModel, SimulatedBench

NO_SETTLE = Q(0, "s")


def test_compression_sweeps_levels_and_records_context(bench: SimulatedBench, dut: DUT, model: AmplifierModel) -> None:
    settings = CompressionSettings(p_start=Q(-30, "dBm"), p_stop=Q(-10, "dBm"), p_step=Q(10, "dB"), settle=NO_SETTLE)
    result = compression.measure(bench, dut, dut.channel(1), settings)

    assert result.measurement == "Compression"
    assert result.dut == "Amplifier X" and result.version == "v1.0" and result.channel == 1
    assert result.reference_plane == "instrument"
    assert set(result.paths) == {"in", "out"}
    assert result.instruments["Spectrum analyzer"].startswith("Rohde&Schwarz,FSV3007")
    assert result.settings["p_step"] == Q(10, "dB")

    np.testing.assert_allclose(result.p_in.magnitude, [-30.0, -20.0, -10.0])
    expected = [model.output_level(p) for p in (-30.0, -20.0, -10.0)]
    np.testing.assert_allclose(result.p_out.magnitude, expected)
    np.testing.assert_allclose(result.frequency.to("MHz").magnitude, [600, 600, 600])

    np.testing.assert_allclose(result.noise_floor.magnitude, [model.noise_floor_dbm] * 3)
    assert result.settings["generator_attenuation"] == Q(15, "dB")  # -30 dBm start needs 15 dB held

    w = bench.fsv_backend.writes
    assert "FREQ:CENT 600000000.0" in w and "INIT:CONT OFF" in w and "CALC:MARK1:MAX" in w
    assert "DET1 RMS" in w and "DISP:TRAC1:MODE AVER" in w and "SWE:COUN 5" in w
    assert any(c.startswith("DISP:TRAC:Y:SCAL:RLEV ") for c in w)  # reference level tracked

    g = bench.gen_a_backend.writes
    assert "POW:ALC ON" in g and "POW:ATT:AUTO OFF" in g and "POW:ATT 15.0DB" in g
    assert g.index("POW:ATT:AUTO OFF") < g.index("OUTP ON")  # attenuator held before RF on
    assert g[-2:] == ["OUTP OFF", "POW:ATT:AUTO ON"]  # RF left off, attenuator released


def test_noise_figure_loads_path_losses_on_the_instrument(bench: SimulatedBench, dut: DUT, model: AmplifierModel) -> None:
    result = noise_figure.measure(bench, dut, dut.channel(1), NoiseFigureSettings(points=3))
    assert result.reference_plane == "dut"
    np.testing.assert_allclose(result.noise_figure.magnitude, [model.noise_figure_db] * 3)
    np.testing.assert_allclose(result.gain.magnitude, [model.gain_db] * 3)

    w = bench.fsv_backend.writes
    assert "INST:CRE NOISe,'Noise'" in w
    assert "CORR:LOSS:INP:TABL:SEL 'LabKit in'" in w and "CORR:LOSS:INP:MODE TABL" in w
    assert "CORR:LOSS:OUTP:TABL:SEL 'LabKit out'" in w
    table = next(c for c in w if c.startswith("CORR:LOSS:INP:TABL 5"))
    # 3 points over 580-620 MHz: cable (interpolated) + 10.2 dB pad at each
    pairs = [float(x) for x in table.split(" ", 1)[1].split(",")]
    assert pairs[0] == 580e6 and pairs[4] == 620e6
    assert pairs[1] == pytest.approx(0.3008 + 10.2, abs=1e-6)  # 580 MHz: 0.28 + 0.13*0.16


def test_s_parameters_reads_reflections_and_gain(bench: SimulatedBench, dut: DUT, model: AmplifierModel) -> None:
    result = s_parameters.measure(bench, dut, dut.channel(1), SParameterSettings(points=5, average_count=4))
    assert result.parameters == ("S11", "S21", "S22")
    assert result.frequency.shape == (5,)
    np.testing.assert_allclose(result.magnitudes["S11"].magnitude, [model.s11_db] * 5, atol=1e-9)
    np.testing.assert_allclose(result.magnitudes["S21"].magnitude, [model.gain_db] * 5, atol=1e-9)
    np.testing.assert_allclose(result.magnitudes["S22"].magnitude, [model.s22_db] * 5, atol=1e-9)
    w = bench.vna_backend.writes
    assert "CALC1:PAR:SDEF 'Trc1','S11'" in w and "CALC1:PAR:SDEF 'Trc2','S21'" in w
    assert "CALC1:PAR:SDEF 'Trc3','S22'" in w
    assert w[-1] == "OUTP OFF"


def test_harmonics_one_row_per_order(bench: SimulatedBench, dut: DUT, model: AmplifierModel) -> None:
    result = harmonics.measure(bench, dut, dut.channel(2), HarmonicsSettings(harmonics=3, settle=NO_SETTLE))
    np.testing.assert_array_equal(result.order, [1, 2, 3])
    np.testing.assert_allclose(result.frequency.to("MHz").magnitude, [700, 1400, 2100])
    p1 = model.output_level(-10.0)
    np.testing.assert_allclose(result.level.magnitude, [p1, p1 - 25.0, p1 - 50.0])
    assert "CALC:MARK:FUNC:HARM:NHAR 3" in bench.fsv_backend.writes
    assert bench.fsv_backend.writes[-1] == "CALC:MARK:FUNC:HARM OFF"


def test_intermodulation_reads_tones_and_products(bench: SimulatedBench, dut: DUT, model: AmplifierModel) -> None:
    settings = IntermodulationSettings(level=Q(-20, "dBm"), tone_spacing=Q(2, "MHz"), settle=NO_SETTLE)
    result = intermodulation.measure(bench, dut, dut.channel(1), settings)
    assert list(result.tone) == ["f1", "f2", "im3_low", "im3_high"]
    np.testing.assert_allclose(result.frequency.to("MHz").magnitude, [599.0, 601.0, 597.0, 603.0])
    p_out = model.output_level(-20.0)
    np.testing.assert_allclose(result.level.magnitude, [p_out, p_out, model.im3_level(p_out), model.im3_level(p_out)])
    np.testing.assert_allclose(result.toi_instrument.magnitude, [model.oip3_dbm] * 4)
    assert bench.gen_b_backend.writes[-1] == "RFOFF"
    assert bench.gen_a_backend.writes[-1] == "OUTP OFF"
