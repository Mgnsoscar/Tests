"""DUT state in results and file names, the self-stopping compression sweep, and
component characterization with fixture de-embedding."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from labkit.signal_path import Component, LossTable, SignalPath
from labkit.units import quantity as Q

from rflab.components.characterize import CharacterizationSettings, characterize, save
from rflab.dut import DUT
from rflab.measurements import CompressionSettings, compression, harmonics, HarmonicsSettings
from rflab.simulation import AmplifierModel, SimulatedBench
from rflab.store import ResultStore

NO_SETTLE = Q(0, "s")


# --- DUT state ---------------------------------------------------------------

def test_state_is_recorded_named_and_reloaded(tmp_path: Path, bench: SimulatedBench, dut: DUT) -> None:
    store = ResultStore(tmp_path)
    result = harmonics.measure(
        bench, dut, dut.channel(1), HarmonicsSettings(harmonics=2, settle=NO_SETTLE),
        state={"attenuation": Q(6, "dB")},
    )
    assert result.state == {"attenuation": Q(6, "dB")}
    assert result.state_label == "attenuation 6 dB"
    assert result.title == "Amplifier X v1.0 Ch1 attenuation 6 dB"

    path = Path(store.save(result))
    assert path.name.endswith(" (B) Harmonics Ch1 attenuation 6 dB.csv")
    assert "# State attenuation [dB]: 6\n" in path.read_text(encoding="utf-8")

    loaded = store.load(path)
    assert loaded.state == {"attenuation": Q(6, "dB")}

    # a different setting on the same day gets its own file, not a run counter
    other = harmonics.measure(
        bench, dut, dut.channel(1), HarmonicsSettings(harmonics=2, settle=NO_SETTLE),
        state={"attenuation": Q(12, "dB")},
    )
    assert Path(store.save(other)).name.endswith("Ch1 attenuation 12 dB.csv")


def test_no_state_keeps_plain_names(tmp_path: Path, bench: SimulatedBench, dut: DUT) -> None:
    result = harmonics.measure(bench, dut, dut.channel(1), HarmonicsSettings(harmonics=1, settle=NO_SETTLE))
    assert result.state == {} and result.state_label == "" and result.title == "Amplifier X v1.0 Ch1"
    assert Path(ResultStore(tmp_path).save(result)).name.endswith(" (B) Harmonics Ch1.csv")


# --- compression stops itself ------------------------------------------------

def test_compression_stops_once_compressed(bench: SimulatedBench, dut: DUT, model: AmplifierModel) -> None:
    settings = CompressionSettings(
        p_start=Q(-40, "dBm"), p_stop=Q(20, "dBm"), p_step=Q(1, "dB"),
        n_linear=3, stop_compression=Q(2, "dB"), settle=NO_SETTLE,
    )
    result = compression.measure(bench, dut, dut.channel(1), settings)
    p_in = result.p_in.magnitude
    gain = result.p_out.magnitude - p_in
    assert len(p_in) < len(compression.levels(settings))  # did not run to p_stop
    small_signal = np.mean(gain[:3])
    assert gain[-1] <= small_signal - 2.0          # stopped once 2 dB compressed ...
    assert gain[-2] > small_signal - 2.0           # ... and not a step later
    assert p_in[-1] < 20.0
    # the 1 dB point is inside the data that was taken
    assert np.any(gain <= small_signal - 1.0) and np.any(gain > small_signal - 1.0)


def test_compression_runs_to_p_stop_when_never_compressed(bench: SimulatedBench, dut: DUT) -> None:
    settings = CompressionSettings(p_start=Q(-60, "dBm"), p_stop=Q(-50, "dBm"), p_step=Q(2, "dB"), settle=NO_SETTLE)
    result = compression.measure(bench, dut, dut.channel(1), settings)
    assert len(result.p_in) == len(compression.levels(settings))


# --- characterization with fixture de-embedding -------------------------------

def test_characterize_de_embeds_fixture_and_saves_a_loadable_table(tmp_path: Path, bench: SimulatedBench) -> None:
    fixture = SignalPath(
        Component("adapter A", Q(0.1, "dB"), date(2026, 8, 20)),
        Component("adapter B", LossTable(Q([0.1, 20.0], "GHz"), Q([0.1, 0.3], "dB")), date(2026, 8, 20)),
    )
    settings = CharacterizationSettings(start=Q(1, "GHz"), stop=Q(3, "GHz"), points=3)
    result = characterize(bench.vna, "Test cable", settings, fixture, when=date(2026, 9, 15))

    np.testing.assert_allclose(result.frequency.to("GHz").magnitude, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result.loss_raw.magnitude, [30.0] * 3, atol=1e-9)   # simulator: S21 = -30 dB
    expected = 30.0 - (0.1 + fixture.components[1].loss_at(result.frequency).magnitude)
    np.testing.assert_allclose(result.loss.magnitude, expected, atol=1e-9)
    assert "CALC1:PAR:SDEF 'Trc1','S21'" in bench.vna_backend.writes

    path = save(result, folder=tmp_path)
    assert path.name == "test_cable_2026-09-15.csv"
    text = path.read_text(encoding="utf-8")
    assert "# Component: Test cable\n" in text
    assert "# Fixture (de-embedded): adapter A (2026-08-20), adapter B (2026-08-20)\n" in text
    assert "Frequency [MHz],Loss [dB],Loss raw [dB]\n" in text

    table = LossTable.from_csv(str(path))   # picks the de-embedded "Loss" column
    assert table.loss_at(Q(2, "GHz")).magnitude == pytest.approx(expected[1])
    assert table.loss_at(Q(2.5, "GHz")).magnitude == pytest.approx((expected[1] + expected[2]) / 2)


def test_characterize_without_fixture_keeps_raw(bench: SimulatedBench) -> None:
    result = characterize(bench.vna, "Bare", CharacterizationSettings(points=2), when=date(2026, 9, 15))
    np.testing.assert_allclose(result.loss.magnitude, result.loss_raw.magnitude)
    assert result.fixture.describe() == "direct"
