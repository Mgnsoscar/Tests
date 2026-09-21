"""The contact monitor: scope setup, dropout duration, interval readout, the logging loop."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from labkit.units import quantity as Q

from rflab.monitoring.continuity import (
    ContinuityLog,
    ContinuitySettings,
    DropoutEvent,
    configure,
    dropout_duration,
    monitor,
    read_interval,
)
from rflab.simulation import SimulatedBench

SETTINGS = ContinuitySettings(interval=Q(2, "s"), waveforms_per_interval=2, segments=100)


def test_configure_sets_50_ohm_trigger_and_segmentation(bench: SimulatedBench) -> None:
    configure(bench.scope, SETTINGS)
    w = bench.scope_backend.writes
    assert "SYST:DISP:UPD OFF" in w
    assert "CHAN1:COUP DC" in w                      # the 50 Ω input, so the node collapses in ns
    assert "CHAN1:SCAL 0.2" in w and "CHAN1:OFFS 0.5" in w
    assert "TIM:RANG 5e-05" in w and "TIM:REF 20.0" in w
    assert "ACQ:SRAT 200000000.0" in w
    assert "ACQ:COUN 100" in w and "ACQ:SEGM:STAT ON" in w and "ACQ:SEGM:MAX 100" in w
    assert "TRIG1:SOUR CHAN1" in w and "TRIG1:EDGE:SLOP NEG" in w and "TRIG1:LEV1 0.5" in w
    assert "TRIG1:MODE NORM" in w
    assert "TRIG:HOLD:MODE TIME" in w and "TRIG:HOLD:TIME 1e-05" in w
    assert "CHAN1:WAV1:HIST:STAT ON" in w


def test_dropout_duration_and_minimum() -> None:
    t = Q(np.linspace(0, 9e-6, 10), "s")            # 1 µs per point
    v = Q([1, 1, 0, 0, 0.9, 0, 1, 1, 1, 1], "V")     # open at 2, 3 and 5 µs: a bounce
    duration, minimum = dropout_duration(t, v, Q(0.5, "V"))
    assert duration.to("us").magnitude == pytest.approx(4.0)   # first to last opening, plus one sample
    assert minimum == Q(0, "V")
    duration, _ = dropout_duration(t, Q([1] * 10, "V"), Q(0.5, "V"))
    assert duration.magnitude == 0.0
    # an envelope record: the minimum per interval counts
    env = Q(np.column_stack([[1, 0.2, 1], [1, 1, 1]]), "V")
    duration, minimum = dropout_duration(Q([0, 1e-6, 2e-6], "s"), env, Q(0.5, "V"))
    assert duration.to("us").magnitude == pytest.approx(1.0) and minimum == Q(0.2, "V")


def test_read_interval_returns_every_stored_dropout_oldest_first(bench: SimulatedBench) -> None:
    scope = bench.scope
    configure(scope, SETTINGS)
    scope.run_single(wait_for_completion=False)
    scope.stop()
    events, memory_full = read_interval(scope, SETTINGS, interval=7)
    assert memory_full is False
    assert [e.number for e in events] == [1, 2, 3]
    assert [e.relative for e in events] == pytest.approx([-0.8, -0.4, 0.0])
    assert all(e.interval == 7 for e in events)
    # the record was read for the first two events only, and the k-th dropout lasts k × 2 µs
    assert events[0].duration is not None and events[0].duration.to("us").magnitude == pytest.approx(2.0, abs=0.06)
    assert events[1].duration is not None and events[1].duration.to("us").magnitude == pytest.approx(4.0, abs=0.06)
    assert events[2].duration is None
    assert events[0].minimum == Q(0, "V")
    # the history was walked from the oldest to the newest acquisition
    selects = [c for c in bench.scope_backend.writes if c.startswith("CHAN1:WAV1:HIST:CURR ")]
    assert selects[:3] == ["CHAN1:WAV1:HIST:CURR -2", "CHAN1:WAV1:HIST:CURR -1", "CHAN1:WAV1:HIST:CURR 0"]


def test_memory_full_is_flagged(bench: SimulatedBench) -> None:
    bench.dropouts_per_run = 100
    scope = bench.scope
    scope.run_single(wait_for_completion=False)
    scope.stop()
    _, memory_full = read_interval(scope, ContinuitySettings(segments=100, waveforms_per_interval=0), 1)
    assert memory_full is True


def test_monitor_loop_writes_events_and_intervals_flushed(tmp_path: Path, bench: SimulatedBench) -> None:
    scope = bench.scope
    configure(scope, SETTINGS)
    log = ContinuityLog(tmp_path, "run").open({"DUT": "Antenna X", "Test": "vibration"})
    now = [datetime(2026, 9, 21, 8, 0, 0)]

    def clock() -> datetime:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += timedelta(seconds=seconds)

    lines: list[str] = []
    total = monitor(scope, SETTINGS, log, intervals=2, clock=clock, sleep=sleep, report=lines.append)
    assert total == 6
    assert lines == ["interval 1: 3 dropout(s), 3 in total", "interval 2: 3 dropout(s), 6 in total"]

    events = (tmp_path / "run events.csv").read_text(encoding="utf-8").splitlines()
    assert events[0] == "# DUT: Antenna X" and events[1] == "# Test: vibration"
    assert events[2].startswith("Interval,Event,Scope date,Scope time,Relative [s],Duration [us],Minimum [V]")
    rows = [line.split(",") for line in events[3:]]
    assert len(rows) == 6
    assert [r[0] for r in rows] == ["1", "1", "1", "2", "2", "2"]
    assert rows[0][5] != "" and rows[2][5] == ""             # duration only for the first two per interval
    intervals = (tmp_path / "run intervals.csv").read_text(encoding="utf-8").splitlines()
    assert intervals[-1].startswith("2,2026-09-21T08:00:02.000,2026-09-21T08:00:04.000,3,")
    # the scope was run, stopped, then run again
    w = bench.scope_backend.writes
    assert w.count("RUNS") == 2 and w.count("STOP") == 2
    log.close()


def test_monitor_reads_out_the_interval_on_keyboard_interrupt(tmp_path: Path, bench: SimulatedBench) -> None:
    scope = bench.scope
    log = ContinuityLog(tmp_path, "run").open({})

    def sleep(seconds: float) -> None:
        raise KeyboardInterrupt

    total = monitor(scope, SETTINGS, log, sleep=sleep, report=lambda _: None)
    assert total == 3
    assert bench.scope_backend.writes.count("STOP") == 1
    rows = (tmp_path / "run events.csv").read_text(encoding="utf-8").splitlines()
    assert len([r for r in rows if r.startswith("1,")]) == 3
    log.close()


def test_log_appends_to_existing_files_without_repeating_headers(tmp_path: Path) -> None:
    log = ContinuityLog(tmp_path, "run").open({"DUT": "A"})
    log.write_event(DropoutEvent(1, 1, "2026-09-21", "08:00:00.000", 0.0))
    log.close()
    log = ContinuityLog(tmp_path, "run").open({"DUT": "A"})
    log.write_event(DropoutEvent(2, 1, "2026-09-21", "08:01:00.000", 0.0, Q(2.5, "us"), Q(0.01, "V")))
    log.close()
    text = (tmp_path / "run events.csv").read_text(encoding="utf-8")
    assert text.count("# DUT: A") == 1 and text.count("Interval,Event") == 1
    assert text.splitlines()[-1] == "2,1,2026-09-21,08:01:00.000,0.000000000,2.500,0.0100"
