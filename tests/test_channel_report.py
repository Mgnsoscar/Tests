"""The channel requirements report: min/max gain, cutoff, passband variation, S11 average."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest

from labkit.units import quantity as Q

from rflab.analysis import channel_report
from rflab.dut import Channel, ChannelRequirements, DUT
from rflab.measurements import SParameterSettings, s_parameters
from rflab.simulation import AmplifierModel, SimulatedBench
from rflab.store import ResultStore

matplotlib.use("Agg")

CONFIGS: list[dict[str, Any]] = [
    {"attenuation": Q(0, "dB"), "bypass": False},
    {"attenuation": Q(6, "dB"), "bypass": False},
    {"attenuation": Q(12, "dB"), "bypass": False},
    {"attenuation": Q(12, "dB"), "bypass": True},
]


def _measure_all(bench: SimulatedBench, dut: DUT, channel: Channel, points: int = 201) -> list[Any]:
    results = []
    for state in CONFIGS:
        bench.set_dut(channel, state)
        results.append(s_parameters.measure(bench, dut, channel, SParameterSettings(points=points), state))
    return results


# --- the bypass state -------------------------------------------------------------

def test_bypass_state_in_names_header_and_round_trip(tmp_path: Path, bench: SimulatedBench, dut: DUT) -> None:
    ch = dut.channel(1)
    result = s_parameters.measure(bench, dut, ch, SParameterSettings(points=3), {"attenuation": Q(6, "dB"), "bypass": True})
    assert result.state_label == "attenuation 6 dB bypass on"
    store = ResultStore(tmp_path)
    path = Path(store.save(result))
    assert path.name.endswith(" (B) S-Parameters Ch1 attenuation 6 dB bypass on.csv")
    assert "# State bypass: on\n" in path.read_text(encoding="utf-8")
    loaded = store.load(path)
    assert loaded.state == {"attenuation": Q(6, "dB"), "bypass": True}
    assert loaded.state["bypass"] is True


# --- the simulated DUT behind its paths ---------------------------------------------

def test_simulated_dut_seen_through_its_paths(bench: SimulatedBench, dut: DUT, model: AmplifierModel) -> None:
    """With set_dut, the raw data is instrument-plane data and at_dut recovers the model."""
    ch = dut.channel(1)
    bench.set_dut(ch, {"attenuation": Q(6, "dB"), "bypass": False})
    result = s_parameters.measure(bench, dut, ch, SParameterSettings(points=201))
    from rflab.analysis import at_dut

    corrected = at_dut(result)
    f = corrected.frequency
    centre = int(np.argmin(np.abs(f.to("Hz").magnitude - ch.f_center.to("Hz").magnitude)))
    assert corrected.magnitudes["S21"].magnitude[centre] == pytest.approx(model.gain_db - 6.0, abs=1e-6)
    assert corrected.magnitudes["S11"].magnitude[centre] == pytest.approx(model.s11_db - 0.6, abs=1e-6)
    # raw S21 is lower by both path losses
    both = (ch.path("in").loss_at(f) + ch.path("out").loss_at(f)).magnitude
    np.testing.assert_allclose(result.magnitudes["S21"].magnitude, corrected.magnitudes["S21"].magnitude - both)


# --- evaluation --------------------------------------------------------------------------

def test_latest_per_state_keeps_the_newest_of_each_configuration(bench: SimulatedBench, dut: DUT) -> None:
    ch = dut.channel(1)
    results = _measure_all(bench, dut, ch, points=21)
    older = results[0].replace(timestamp=results[0].timestamp - timedelta(days=1))
    newer = results[0].replace(timestamp=results[0].timestamp + timedelta(days=1))
    latest = channel_report.latest_per_state([older, *results, newer])
    assert len(latest) == 4
    assert latest[0].timestamp == newer.timestamp
    assert [r.state_label for r in latest] == [
        "attenuation 0 dB bypass off",
        "attenuation 6 dB bypass off",
        "attenuation 12 dB bypass off",
        "attenuation 12 dB bypass on",
    ]


def test_report_answers_the_requirements(bench: SimulatedBench, dut: DUT, model: AmplifierModel) -> None:
    ch = dut.channel(1)
    rep = channel_report.report(ch, _measure_all(bench, dut, ch), dut=dut.label)

    assert rep.min_gain is not None and rep.max_gain is not None
    assert rep.min_gain.label == "attenuation 12 dB bypass on"
    assert rep.min_gain.gain_nominal.magnitude == pytest.approx(model.gain_db - 12 - model.bypass_stage_gain_db, abs=1e-6)
    assert rep.max_gain.label == "attenuation 0 dB bypass off"
    assert rep.max_gain.gain_nominal.magnitude == pytest.approx(model.gain_db, abs=1e-6)

    for e in rep.gains:
        assert e.variation.magnitude == pytest.approx(model.ripple_db, abs=0.05)   # the model's ripple, dips only
        assert e.variation_ok is True
        assert e.rejection_below is not None and e.rejection_above is not None
        assert e.rejection_below.magnitude == pytest.approx(24.9, abs=0.2)         # 6th-order roll-off at 1.6 half-widths
        assert e.rejection_ok is True and e.passed is True
    assert rep.passed is True

    s11 = rep.reflections["S11"]
    assert len(s11.traces) == 4
    # a power average of the four traces, point by point
    stack = np.vstack([t.magnitude for t in s11.traces.values()])
    np.testing.assert_allclose(s11.average.magnitude, 10 * np.log10(np.mean(10 ** (stack / 10), axis=0)))
    assert s11.average.magnitude.max() > s11.average.magnitude.min()
    # worst of the average inside the band is at the top band edge (the simulated match degrades upwards)
    assert s11.worst_frequency.to("MHz").magnitude == pytest.approx(620.0, abs=0.5)
    assert s11.worst_individual[0] == "attenuation 12 dB bypass on"

    text = rep.text()
    assert "Min gain (highest attenuation, bypass on): -2.00 dB" in text
    assert "Max gain (no attenuation, bypass off): 20.00 dB" in text
    assert "S21 requirements: PASS" in text
    assert "S11 average of 4 configurations" in text


def test_report_flags_failures_and_missing_configurations(bench: SimulatedBench, dut: DUT) -> None:
    ch = dut.channel(1)
    strict = Channel(
        ch.number, ch.f_start, ch.f_stop, ch.ports,
        requirements=ChannelRequirements(Q(40, "dB"), Q(20, "MHz"), Q(0.2, "dB")),
    )
    # only bypass-off configurations measured: no "min gain" configuration exists
    results = _measure_all(bench, dut, strict)[:3]
    rep = channel_report.report(strict, results)
    assert rep.min_gain is None and rep.max_gain is not None
    assert all(e.variation_ok is False and e.rejection_ok is False for e in rep.gains)
    assert rep.passed is False
    text = rep.text()
    assert "Min gain (highest attenuation, bypass on): no such configuration measured" in text
    assert "S21 requirements: FAIL — failed:" in text


def test_report_without_requirements_still_reports_gains(bench: SimulatedBench, dut: DUT) -> None:
    ch = dut.channel(2)
    bare = Channel(ch.number, ch.f_start, ch.f_stop, ch.ports)
    rep = channel_report.report(bare, _measure_all(bench, dut, bare, points=41))
    assert rep.passed is None
    assert all(e.rejection_below is None and e.passed is None for e in rep.gains)
    assert "none defined" in rep.text()


def test_cutoff_outside_the_sweep_is_not_checked(bench: SimulatedBench, dut: DUT) -> None:
    ch = dut.channel(1)
    bench.set_dut(ch, CONFIGS[0])
    narrow = SParameterSettings(frequency_range=(Q(570, "MHz"), Q(630, "MHz")), points=61)
    result = s_parameters.measure(bench, dut, ch, narrow, CONFIGS[0])
    e = channel_report.evaluate_gain(ch, result)
    assert e.rejection_below is None and e.rejection_ok is None and e.passed is None
    assert "outside the sweep" in e.describe()


def test_report_needs_s21_and_the_band(bench: SimulatedBench, dut: DUT) -> None:
    ch = dut.channel(1)
    only_reflections = s_parameters.measure(bench, dut, ch, SParameterSettings(points=5, parameters=("S11",)))
    with pytest.raises(ValueError, match="no S21"):
        channel_report.evaluate_gain(ch, only_reflections)
    with pytest.raises(ValueError, match="no S-parameter results"):
        channel_report.report(ch, [])


# --- plots -------------------------------------------------------------------------------------

def test_plots_are_written(tmp_path: Path, bench: SimulatedBench, dut: DUT) -> None:
    ch = dut.channel(1)
    rep = channel_report.report(ch, _measure_all(bench, dut, ch, points=101))
    channel_report.plot_gain(rep, save_folder=str(tmp_path), filename="s21")
    channel_report.plot_reflection(rep, "S11", save_folder=str(tmp_path), filename="s11")
    channel_report.plot_reflection(rep, "S22", save_folder=str(tmp_path), filename="s22")
    assert {p.name for p in tmp_path.glob("*.png")} == {"s21.png", "s11.png", "s22.png"}
