"""Save, load, move to the DUT plane, summarize, plot — for every measurement."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest

from labkit.units import quantity as Q

from rflab import analysis
from rflab.analysis import at_dut
from rflab.dut import DUT
from rflab.measurements import (
    CompressionResult,
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
from rflab.store import ResultStore

matplotlib.use("Agg")
NO_SETTLE = Q(0, "s")


def _all_results(bench: SimulatedBench, dut: DUT) -> list[Any]:
    ch = dut.channel(1)
    return [
        compression.measure(bench, dut, ch, CompressionSettings(p_step=Q(2, "dB"), settle=NO_SETTLE)),
        noise_figure.measure(bench, dut, ch, NoiseFigureSettings(points=5)),
        s_parameters.measure(bench, dut, ch, SParameterSettings(points=5)),
        harmonics.measure(bench, dut, ch, HarmonicsSettings(harmonics=3, settle=NO_SETTLE)),
        intermodulation.measure(bench, dut, ch, IntermodulationSettings(settle=NO_SETTLE)),
    ]


# --- naming -----------------------------------------------------------------

def test_folder_and_file_names_follow_the_lab_convention(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    result = CompressionResult(
        dut="Amplifier X", version="v1.0", channel=1, timestamp=datetime(2026, 9, 15, 10, 30),
        frequency=Q([2.3e9], "Hz"), p_in=Q([-10.0], "dBm"), p_out=Q([9.0], "dBm"),
        noise_floor=Q([-100.0], "dBm"),
    )
    first = Path(store.save(result))
    second = Path(store.save(result))
    assert first.parent.parent.name == "(B) Amplifier X v1.0"
    assert first.parent.name == "(B) Compression"
    assert first.name == "2026-09-15 (B) Compression Ch1.csv"
    assert second.name == "2026-09-15 (B) Compression Ch1 (2).csv"
    folder, figure = store.figure_location(result)
    assert figure == "2026-09-15 (B) Compression Ch1"
    assert store.find(dut="Amplifier X v1.0", measurement="Compression", channel=1) == [first, second]
    assert store.find(channel=2) == []


# --- round trip ---------------------------------------------------------------

def test_every_result_round_trips_through_the_store(tmp_path: Path, bench: SimulatedBench, dut: DUT) -> None:
    store = ResultStore(tmp_path)
    for original in _all_results(bench, dut):
        loaded = store.load(store.save(original))
        assert type(loaded) is type(original)
        assert loaded.dut == original.dut and loaded.channel == original.channel
        assert loaded.timestamp.replace(microsecond=0) == original.timestamp.replace(microsecond=0)
        assert loaded.reference_plane == original.reference_plane
        assert loaded.instruments == original.instruments
        for port, path in original.paths.items():
            assert loaded.paths[port].describe() == path.describe()
        for key, value in original.settings.items():
            if hasattr(value, "magnitude"):
                assert loaded.settings[key] == value, key
            else:
                assert str(loaded.settings[key]) == str(value) or loaded.settings[key] == value, key
        for label, column in original.columns().items():
            got = loaded.columns()[label]
            if hasattr(column, "magnitude"):
                assert str(got.units) == str(column.units), label
                np.testing.assert_allclose(got.magnitude, column.magnitude, err_msg=label)
            elif column.dtype == object:
                assert list(got) == list(column)
            else:
                np.testing.assert_allclose(np.asarray(got, dtype=np.float64), column)


def test_header_records_paths_and_settings(tmp_path: Path, bench: SimulatedBench, dut: DUT) -> None:
    store = ResultStore(tmp_path)
    path = store.save(compression.measure(bench, dut, dut.channel(1), CompressionSettings(settle=NO_SETTLE)))
    text = Path(path).read_text(encoding="utf-8")
    assert "# Path in: SMA cable A (2026-09-01), 10 dB pad SN1234 (2026-08-20)\n" in text
    assert "# Path out: 20 dB coupler SN0042 (2026-08-20), SMA cable B (2026-09-01)\n" in text
    assert "# Reference plane: instrument\n" in text
    assert "# Setting rbw [kHz]: 10\n" in text
    assert "# Instrument Spectrum analyzer: Rohde&Schwarz,FSV3007,SIM,1.0\n" in text
    assert "Frequency [Hz],P_in [dBm],P_out [dBm],Noise floor [dBm]\n" in text
    assert "# Setting generator_attenuation [dB]: 15.0\n" in text


# --- reference plane ------------------------------------------------------------

def test_at_dut_applies_path_losses_at_each_rows_frequency(bench: SimulatedBench, dut: DUT) -> None:
    ch = dut.channel(1)
    raw = compression.measure(bench, dut, ch, CompressionSettings(p_step=Q(10, "dB"), settle=NO_SETTLE))
    corrected = at_dut(raw)
    assert corrected.reference_plane == "dut" and raw.reference_plane == "instrument"
    f = raw.frequency
    loss_in = ch.path("in").loss_at(f).magnitude
    loss_out = ch.path("out").loss_at(f).magnitude
    np.testing.assert_allclose(corrected.p_in.magnitude, raw.p_in.magnitude - loss_in)
    np.testing.assert_allclose(corrected.p_out.magnitude, raw.p_out.magnitude + loss_out)
    np.testing.assert_allclose(corrected.noise_floor.magnitude, raw.noise_floor.magnitude + loss_out)
    np.testing.assert_array_equal(corrected.valid(), raw.valid())  # validity is plane-independent
    assert at_dut(corrected) is corrected  # idempotent

    harm = harmonics.measure(bench, dut, ch, HarmonicsSettings(harmonics=3, settle=NO_SETTLE))
    # the 3rd harmonic of 600 MHz is 1.8 GHz: loss interpolated there, not at f0
    loss_3rd = ch.path("out").loss_at(Q(1.8, "GHz")).magnitude
    assert at_dut(harm).level.magnitude[2] == pytest.approx(harm.level.magnitude[2] + loss_3rd)

    refl = s_parameters.measure(bench, dut, ch, SParameterSettings(points=3))
    twice = 2 * ch.path("in").loss_at(refl.frequency).magnitude
    np.testing.assert_allclose(at_dut(refl).magnitudes["S11"].magnitude, refl.magnitudes["S11"].magnitude + twice)
    both = (ch.path("in").loss_at(refl.frequency) + ch.path("out").loss_at(refl.frequency)).magnitude
    np.testing.assert_allclose(at_dut(refl).magnitudes["S21"].magnitude, refl.magnitudes["S21"].magnitude + both)

    nf = noise_figure.measure(bench, dut, ch, NoiseFigureSettings(points=3))
    assert at_dut(nf) is nf  # already at the DUT plane


# --- summaries ------------------------------------------------------------------

def test_compression_summary_finds_p1db(bench: SimulatedBench, dut: DUT, model: AmplifierModel) -> None:
    ch = dut.channel(2)  # direct-ish paths: cable in, pad + cable out
    settings = CompressionSettings(p_start=Q(-40, "dBm"), p_stop=Q(0, "dBm"), p_step=Q(0.5, "dB"), settle=NO_SETTLE)
    (summary,) = analysis.compression.summarize(compression.measure(bench, dut, ch, settings))
    f = ch.f_center
    expected_gain = model.gain_db + ch.path("in").loss_at(f).magnitude + ch.path("out").loss_at(f).magnitude
    assert summary.gain_small_signal.magnitude == pytest.approx(expected_gain, abs=0.1)
    assert summary.points_dropped == 0 and summary.points_used > 0
    assert summary.p_in_1db is not None and summary.p_out_1db is not None
    # the model's output P1dB, plus the output path loss we add back at the DUT plane
    assert summary.p_out_1db.magnitude == pytest.approx(
        model.p1db_output() + ch.path("out").loss_at(f).magnitude, abs=0.3
    )
    assert summary.p_in_1db.magnitude == pytest.approx(
        summary.p_out_1db.magnitude - (summary.gain_small_signal.magnitude - 1.0), abs=1e-9
    )


def test_other_summaries(bench: SimulatedBench, dut: DUT, model: AmplifierModel) -> None:
    ch = dut.channel(1)
    nf = analysis.noise_figure.summarize(noise_figure.measure(bench, dut, ch, NoiseFigureSettings(points=3)))
    assert nf.nf_mean == Q(model.noise_figure_db, "dB") and nf.gain_min == Q(model.gain_db, "dB")

    worst = analysis.s_parameters.worst_case(s_parameters.measure(bench, dut, ch, SParameterSettings(points=3)))
    assert worst["S11"].magnitude > model.s11_db  # path loss added back makes it worse

    levels = analysis.harmonics.summarize(harmonics.measure(bench, dut, ch, HarmonicsSettings(harmonics=2, settle=NO_SETTLE)))
    assert levels[0].relative == Q(0, "dB")
    assert levels[1].order == 2 and levels[1].relative.magnitude < 0

    (imd,) = analysis.intermodulation.summarize(
        intermodulation.measure(bench, dut, ch, IntermodulationSettings(level=Q(-20, "dBm"), settle=NO_SETTLE))
    )
    f = ch.f_center
    assert imd.oip3.magnitude == pytest.approx(model.oip3_dbm + ch.path("out").loss_at(f).magnitude, abs=0.05)
    assert imd.iip3.magnitude < imd.oip3.magnitude


# --- plots ------------------------------------------------------------------------

def test_every_result_plots_to_its_result_folder(tmp_path: Path, bench: SimulatedBench, dut: DUT) -> None:
    store = ResultStore(tmp_path)
    for result in _all_results(bench, dut):
        folder, filename = store.figure_location(result)
        analysis.PLOTTERS[result.measurement](result, save_folder=folder, filename=filename)
        assert (Path(folder) / (filename + ".png")).exists()
        analysis.SUMMARIZERS[result.measurement](result)
