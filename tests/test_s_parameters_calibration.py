"""The S11/S22 workflow: sweep configured first, calibration dialog, results record the correction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from labkit.units import quantity as Q

from rflab import analysis
from rflab.calibration import cal_file_name, calibration_dialog, describe_correction
from rflab.dut import DUT
from rflab.measurements import SParameterSettings, s_parameters
from rflab.simulation import AmplifierModel, SimulatedBench


def _dialog(bench: SimulatedBench, answers: list[str]) -> tuple[str, list[str]]:
    said: list[str] = []
    it = iter(answers)
    outcome = calibration_dialog(bench.vna, "amp 550-750MHz", ask=lambda _: next(it), say=said.append)
    return outcome, said


# --- sweep configuration -----------------------------------------------------

def test_configure_sets_the_range_before_any_calibration(bench: SimulatedBench, dut: DUT) -> None:
    s_parameters.configure(bench.vna, dut.channel(1), SParameterSettings())   # default: band ± 25 MHz
    w = bench.vna_backend.writes
    assert any(c.endswith("FREQ:STAR 555000000.0") for c in w)
    assert any(c.endswith("FREQ:STOP 645000000.0") for c in w)
    assert any(c.endswith("SWE:POIN 401") for c in w)
    assert "SENS1:BAND:RES 1000.0" in w
    assert any("AVER:COUN 8" in c for c in w)
    assert "OUTP ON" in w
    assert not any("MMEM:" in c for c in w)  # configuring never touches the calibration


def test_sweep_range_is_the_band_plus_margin_or_the_fixed_range(dut: DUT) -> None:
    start, stop = s_parameters.sweep_range(dut.channel(1), SParameterSettings(margin=Q(0, "MHz")))
    assert start == Q(580, "MHz") and stop == Q(620, "MHz")
    start, stop = s_parameters.sweep_range(dut.channel(2), SParameterSettings(margin=Q(10, "MHz")))
    assert start == Q(670, "MHz") and stop == Q(730, "MHz")
    fixed = SParameterSettings(frequency_range=(Q(550, "MHz"), Q(750, "MHz")))
    assert s_parameters.sweep_range(dut.channel(1), fixed) == (Q(550, "MHz"), Q(750, "MHz"))


# --- calibration dialog ---------------------------------------------------------

def test_cal_file_name() -> None:
    assert cal_file_name("amp 550-750MHz") == "amp 550-750MHz.cal"
    assert cal_file_name(" amp.CAL ") == "amp.CAL"


def test_dialog_keep(bench: SimulatedBench) -> None:
    outcome, said = _dialog(bench, ["k"])
    assert outcome.startswith("kept (")
    assert not any("MMEM:" in c for c in bench.vna_backend.writes)
    assert any("[c] calibrate now" in line for line in said)


def test_dialog_load_default_and_named(bench: SimulatedBench) -> None:
    outcome, _ = _dialog(bench, ["l", ""])
    assert outcome == "loaded amp 550-750MHz.cal"
    w = bench.vna_backend.writes
    assert "MMEM:LOAD:CORR 1,'amp 550-750MHz.cal'" in w
    assert "SENS1:CORR:STAT ON" in w
    assert describe_correction(bench.vna) == "correction on, calibrated 2026-09-15 10:02:11"

    outcome, _ = _dialog(bench, ["L", "other"])
    assert outcome == "loaded other.cal"
    assert "MMEM:LOAD:CORR 1,'other.cal'" in bench.vna_backend.writes


def test_dialog_calibrate_then_save(bench: SimulatedBench) -> None:
    outcome, said = _dialog(bench, ["c", "", ""])   # c, Enter when done, Enter = save under default
    assert outcome == "manual, saved as amp 550-750MHz.cal"
    w = bench.vna_backend.writes
    assert "MMEM:STOR:CORR 1,'amp 550-750MHz.cal'" in w
    assert any("Perform the calibration on the instrument now" in line for line in said)


def test_dialog_calibrate_save_under_other_name_or_not_at_all(bench: SimulatedBench) -> None:
    outcome, _ = _dialog(bench, ["c", "", "monday"])
    assert outcome == "manual, saved as monday.cal"
    assert "MMEM:STOR:CORR 1,'monday.cal'" in bench.vna_backend.writes

    before = len(bench.vna_backend.writes)
    outcome, _ = _dialog(bench, ["c", "", "n"])
    assert outcome == "manual, not saved"
    assert not any("MMEM:STOR" in c for c in bench.vna_backend.writes[before:])


def test_dialog_repeats_until_a_valid_choice(bench: SimulatedBench) -> None:
    outcome, said = _dialog(bench, ["x", "", "k"])
    assert outcome.startswith("kept")
    assert sum("please answer" in line for line in said) == 2


# --- measurement records the calibration ----------------------------------------

def test_measure_after_dialog_records_correction_and_band(bench: SimulatedBench, dut: DUT, model: AmplifierModel) -> None:
    ch = dut.channel(1)
    fixed = SParameterSettings(frequency_range=(Q(550, "MHz"), Q(750, "MHz")), points=5)
    s_parameters.configure(bench.vna, ch, fixed)
    outcome, _ = _dialog(bench, ["l", ""])
    result = s_parameters.measure(
        bench, dut, ch, fixed, {"attenuation": Q(0, "dB")},
        configure_sweep=False, calibration=outcome,
    )
    assert result.settings["calibration"] == "loaded amp 550-750MHz.cal"
    assert result.settings["correction"] == "on"
    assert result.settings["correction_date"] == "2026-09-15 10:02:11"
    assert result.settings["frequency_range"] == "550 MHz to 750 MHz"
    assert result.band == (Q(580, "MHz"), Q(620, "MHz"))
    np.testing.assert_allclose(result.frequency.to("MHz").magnitude, [550, 600, 650, 700, 750])
    np.testing.assert_allclose(result.magnitudes["S11"].magnitude, [model.s11_db] * 5, atol=1e-9)
    # the second configure was skipped: only one FREQ:STAR write in total
    assert sum(c.endswith("FREQ:STAR 550000000.0") for c in bench.vna_backend.writes) == 1


def test_worst_case_is_limited_to_the_channel_band(bench: SimulatedBench, dut: DUT, model: AmplifierModel) -> None:
    ch = dut.channel(2)
    settings = SParameterSettings(frequency_range=(Q(650, "MHz"), Q(750, "MHz")), points=11)
    result = s_parameters.measure(bench, dut, ch, settings)
    # make the out-of-band points terrible: worst_case must ignore them
    f = result.frequency.to("MHz").magnitude
    s11 = np.array(result.magnitudes["S11"].magnitude)
    s11[(f < 680) | (f > 720)] = 0.0
    tampered = result.replace(magnitudes={**result.magnitudes, "S11": Q(s11, "dB")})
    worst = analysis.s_parameters.worst_case(tampered)
    in_band_loss = 2 * ch.path("in").loss_at(Q(700, "MHz")).magnitude
    assert worst["S11"].magnitude == pytest.approx(model.s11_db + in_band_loss, abs=0.2)


def test_plot_marks_worst_in_band(tmp_path: Any, bench: SimulatedBench, dut: DUT) -> None:
    import matplotlib

    matplotlib.use("Agg")
    result = s_parameters.measure(bench, dut, dut.channel(1), SParameterSettings(points=9))
    analysis.s_parameters.plot_result(result, save_folder=str(tmp_path), filename="s")
    assert (tmp_path / "s.png").exists()
