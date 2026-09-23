"""The contact monitor: scope setup, record analysis, the timeline, dropout pairing, the logging loop."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from labkit.units import quantity as Q

from rflab.monitoring.continuity import (
    Acquisition,
    ContinuityLog,
    ContinuitySettings,
    DropoutRecord,
    DropoutTracker,
    RecordAnalysis,
    _absolute_seconds,
    _time_of_day,
    analyse_record,
    configure,
    interval_crossings,
    monitor,
    read_interval,
)
from rflab.simulation import SimulatedBench

SETTINGS = ContinuitySettings(interval=Q(2, "s"), segments=100)


def acq(
    interval: int, number: int, relative: float, starts_open: bool, ends_open: bool,
    crossings: tuple[tuple[float, bool], ...] = (), edge: str = "none",
    time: str = "08:00:00.000000000", date: str = "2026-09-21",
) -> Acquisition:
    """A synthetic acquisition with a 50 µs record, 10 µs of it before the trigger."""
    analysis = RecordAnalysis(starts_open, ends_open, crossings, edge, 0.0 if (starts_open or crossings) else 1.0, 1.0)
    return Acquisition(interval, number, date, time, relative, relative - 1e-5, relative + 4e-5, analysis)


def run_once(bench: SimulatedBench) -> None:
    scope = bench.scope
    scope.run_single(wait_for_completion=False)
    scope.trigger.force()
    scope.stop()


# -- the scope -------------------------------------------------------------------


def test_configure_sets_either_edge_peak_detect_and_segmentation(bench: SimulatedBench) -> None:
    setup = configure(bench.scope, SETTINGS)
    assert setup.capacity == 100 and setup.granted_segments == 100
    assert setup.record_length == 2501 and setup.sample_rate.to("MHz").magnitude == 50
    assert setup.describe() == "record 2501 points at 50 MSa/s; the scope holds 100 acquisitions per interval"
    w = bench.scope_backend.writes
    assert "SYST:DISP:UPD OFF" in w
    assert "CHAN1:COUP DC" in w                      # the 50 Ω input, so the node collapses in ns
    assert "CHAN1:SCAL 0.2" in w and "CHAN1:OFFS 0.5" in w
    assert "CHAN1:WAV1:ARIT OFF" in w and "CHAN1:WAV1:TYPE PDET" in w   # peak-detect decimation: no crossing escapes
    assert "TIM:RANG 5e-05" in w and "TIM:REF 20.0" in w
    assert "ACQ:SRAT 50000000.0" in w
    assert "ACQ:COUN 100" in w and "ACQ:SEGM:STAT ON" in w and "ACQ:SEGM:MAX 100" in w
    assert "TRIG1:SOUR CHAN1" in w and "TRIG1:EDGE:SLOP EITH" in w and "TRIG1:LEV1 0.5" in w
    assert "TRIG1:MODE NORM" in w and "TRIG:HOLD:MODE OFF" in w
    assert "CHAN1:WAV1:HIST:STAT ON" in w


def test_read_interval_returns_every_acquisition_oldest_first_with_the_forced_one(bench: SimulatedBench) -> None:
    scope = bench.scope
    configure(scope, SETTINGS)
    run_once(bench)
    acquisitions, memory_full = read_interval(scope, SETTINGS, interval=7)
    assert memory_full is False
    # forced at the start, the 2 µs dropout, both edges of the 100 µs one, the bounce
    assert [a.trigger for a in acquisitions] == ["forced", "falling", "falling", "rising", "falling"]
    assert [a.number for a in acquisitions] == [1, 2, 3, 4, 5]
    assert all(a.interval == 7 for a in acquisitions)
    assert [a.relative for a in acquisitions] == pytest.approx([-1.5, -1.0, -0.5, -0.5 + 1e-4, 0.0], abs=1e-9)
    assert acquisitions[1].analysis.starts_open is False and acquisitions[1].analysis.ends_open is False
    assert acquisitions[2].analysis.ends_open is True                # still open at the end of its record
    assert acquisitions[3].analysis.starts_open is True              # the rising edge's record starts open
    assert len(acquisitions[4].analysis.crossings) == 4              # the bounce: F R F R
    assert acquisitions[1].analysis.minimum == 0.0 and acquisitions[1].analysis.maximum == 1.0
    # the history was entered again after the stop and walked from the oldest to the newest acquisition,
    # waiting for the instrument after every selection
    w = bench.scope_backend.writes
    assert w.index("CHAN1:WAV1:HIST:STAT ON", w.index("STOP")) > w.index("STOP")
    selects = [c for c in w if c.startswith("CHAN1:WAV1:HIST:CURR ")]
    assert selects == [f"CHAN1:WAV1:HIST:CURR {i}" for i in range(-4, 1)]
    assert bench.scope_backend.queries.count("*OPC?") >= 6


def test_read_interval_refuses_records_that_did_not_follow_the_selection(bench: SimulatedBench) -> None:
    scope = bench.scope
    configure(scope, SETTINGS)
    run_once(bench)
    backend = bench.scope_backend
    original_write = backend.write

    def ignore_selection(command: str) -> None:      # an instrument on which HIST:CURR does nothing
        if not command.startswith("CHAN1:WAV1:HIST:CURR "):
            original_write(command)

    backend.write = ignore_selection  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="same timestamp"):
        read_interval(scope, SETTINGS, 1)


def test_memory_full_is_flagged(bench: SimulatedBench) -> None:
    bench.contact_openings = [(0.001 * k, 0.001 * k + 2e-6) for k in range(1, 101)]
    configure(bench.scope, ContinuitySettings(segments=50))
    run_once(bench)
    acquisitions, memory_full = read_interval(bench.scope, ContinuitySettings(segments=50), 1)
    assert memory_full is True and len(acquisitions) == 50
    # the instrument clips the series to its memory: configure reports that, and the flag follows it
    bench.scope_backend.writes.clear()
    bench.scope_segment_capacity = 40
    setup = configure(bench.scope, ContinuitySettings(segments=50))
    assert setup.capacity == 40 and "50 requested" in setup.describe()
    run_once(bench)
    acquisitions, memory_full = read_interval(bench.scope, ContinuitySettings(segments=50), 1, setup.capacity)
    assert memory_full is True and len(acquisitions) == 40


def test_configure_treats_a_nonsense_segment_readback_as_unknown(bench: SimulatedBench) -> None:
    bench.scope_segment_capacity = 1                    # the instrument answers 1: below the command's own range
    setup = configure(bench.scope, ContinuitySettings(segments=150_000))
    assert setup.granted_segments is None and setup.capacity == 150_000
    assert setup.segments_answer == "1" and "capacity unknown" in setup.describe()
    assert "*OPC?" in bench.scope_backend.queries      # the settings were allowed to apply before the readback


# -- one record ------------------------------------------------------------------


def test_analyse_record_finds_crossings_and_the_trigger_edge() -> None:
    t = Q(np.linspace(-2e-6, 7e-6, 10), "s")                 # 1 µs per point, trigger at index 2
    v = Q([1, 1, 0, 0, 0.9, 0, 1, 1, 1, 1], "V")              # open at 0, 1 and 3 µs: a bounce
    r = analyse_record(t, v, Q(0.5, "V"))
    assert r.starts_open is False and r.ends_open is False
    assert [(round(x * 1e6), f) for x, f in r.crossings] == [(0, True), (2, False), (3, True), (4, False)]
    assert r.trigger_edge == "falling" and r.minimum == 0.0 and r.maximum == 1.0
    assert analyse_record(t, Q([1] * 10, "V"), Q(0.5, "V")).crossings == ()
    # a peak-detect record: the minimum per sample interval decides, and a rising edge at the trigger
    env = Q(np.column_stack([[0, 0, 0.2, 1, 1, 1], [1, 1, 1, 1, 1, 1]]), "V")
    r = analyse_record(Q(np.linspace(-2e-6, 3e-6, 6), "s"), env, Q(0.5, "V"))
    assert r.starts_open is True and r.ends_open is False
    assert [(round(x * 1e6), f) for x, f in r.crossings] == [(1, False)]
    assert r.trigger_edge == "rising" and r.minimum == 0.0


def test_analyse_record_hysteresis_ignores_noise_at_the_threshold() -> None:
    # a slow ramp through the threshold with noise on it: the supply switching off, not a contact break
    t = Q(np.linspace(-2e-6, 9e-6, 12), "s")
    v = Q([0.9, 0.9, 0.52, 0.48, 0.51, 0.47, 0.53, 0.45, 0.3, 0.1, 0.0, 0.0], "V")
    without = analyse_record(t, v, Q(0.5, "V"))
    assert len(without.crossings) == 5                                     # every wiggle across 0.5 V
    with_h = analyse_record(t, v, Q(0.5, "V"), hysteresis=Q(100, "mV"))
    assert [(round(x * 1e6), f) for x, f in with_h.crossings] == [(6, True)]  # opens once, below 0.4 V
    assert with_h.starts_open is False and with_h.ends_open is True
    # a contact that closes again must come back above threshold + hysteresis
    v = Q([1.0, 0.0, 0.55, 0.55, 0.7, 0.7, 0.0, 0.0, 0.7, 0.7, 0.7, 0.7], "V")
    r = analyse_record(t, v, Q(0.5, "V"), hysteresis=Q(100, "mV"))
    assert [(round(x * 1e6), f) for x, f in r.crossings] == [(-1, True), (2, False), (4, True), (6, False)]
    # a record that starts inside the band takes its first state from the threshold alone, and says it is unsure
    r = analyse_record(t, Q([0.45] + [0.45] * 11, "V"), Q(0.5, "V"), hysteresis=Q(100, "mV"))
    assert r.starts_open is True and r.crossings == ()
    assert r.start_uncertain is True and r.end_uncertain is True


def test_analyse_record_lets_the_tail_level_overrule_a_noise_crossing() -> None:
    # a ramp down whose chatter ends on a rising edge: the tail is clearly low, so the record ends open
    # and the trailing noise crossing is dropped, leaving a falling crossing as the last one
    t = Q(np.linspace(0, 19e-6, 20), "s")
    v = Q([0.9] * 8 + [0.3, 0.7, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], "V")
    r = analyse_record(t, v, Q(0.5, "V"))                                   # no hysteresis: every wiggle counts
    assert r.ends_open is True and r.end_uncertain is False
    assert r.crossings[-1][1] is True and len(r.crossings) == 5
    # the same with a wiggle inside the tail: its mean level (0.4 V) still says open
    v = Q([0.9] * 8 + [0.3] * 8 + [0.3, 0.3, 0.7, 0.3], "V")
    r = analyse_record(t, v, Q(0.5, "V"))
    assert r.ends_open is True and r.crossings[-1][1] is True and len(r.crossings) == 3
    # a head whose first sample is a noise dip but whose level is high starts closed: the rising
    # crossing that would leave an open start is dropped
    v = Q([0.3, 0.7, 0.7, 0.7] + [0.9] * 16, "V")
    r = analyse_record(t, v, Q(0.5, "V"))
    assert r.starts_open is False and r.start_uncertain is False and r.crossings == ()
    # a tail inside the hysteresis band decides nothing and is flagged
    v = Q([0.9] * 10 + [0.45] * 10, "V")
    r = analyse_record(t, v, Q(0.5, "V"), hysteresis=Q(100, "mV"))
    assert r.end_uncertain is True and r.ends_open is False and r.crossings == ()


# -- the timeline ----------------------------------------------------------------


def test_interval_crossings_takes_overlapping_crossings_once_and_infers_blind_time_edges() -> None:
    a1 = acq(1, 1, -1.0, False, True, ((0.0, True),), "falling")                     # opens, still open at +40 µs
    a2 = acq(1, 2, -1.0 + 45e-6, True, False, ((0.0, False),), "rising")            # closes at +45 µs; its record
    #   starts at +35 µs and overlaps a1's — a2 repeats nothing, but the crossing at +45 µs is new
    a3 = acq(1, 3, -0.5, True, False, ((0.0, False),), "rising")                    # a rising edge from an open
    #   contact — but a2 ended closed and the scope was armed: the opening fell in the blind time
    line = interval_crossings([a1, a2, a3])
    assert [(round((c.time + 1.0) * 1e6), c.falling, c.inferred) for c in line[:2]] == [
        (0, True, False), (45, False, False),
    ]
    assert line[2].inferred is True and line[2].falling is True
    assert line[2].time == pytest.approx(a2.record_end)
    assert (line[3].time, line[3].falling, line[3].inferred) == (-0.5, False, False)


# -- pairing ---------------------------------------------------------------------


def test_tracker_pairs_short_long_and_bouncing_dropouts(bench: SimulatedBench) -> None:
    configure(bench.scope, SETTINGS)
    run_once(bench)
    acquisitions, _ = read_interval(bench.scope, SETTINGS, 1)
    tracker = DropoutTracker(SETTINGS)
    dropouts = tracker.feed(1, acquisitions, memory_full=False)
    assert tracker.pending is None and tracker.count == 3
    assert [d.number for d in dropouts] == [1, 2, 3]
    assert [d.certainty for d in dropouts] == ["exact", "exact", "exact"]
    assert [d.openings for d in dropouts] == [1, 1, 2]
    us = [d.duration * 1e6 for d in dropouts]
    assert us[0] == pytest.approx(2.0, abs=0.05)        # inside one record, ±one sample interval
    assert us[1] == pytest.approx(100.0, abs=0.05)      # from the falling to the rising trigger timestamp
    assert us[2] == pytest.approx(4.0, abs=0.05)        # first opening to last closing of the bounce
    assert [d.minimum for d in dropouts] == [0.0, 0.0, 0.0]
    assert all(d.note == "" for d in dropouts)
    # the start is on the scope clock: the acquisition's timestamp plus its offset in the record
    assert dropouts[0].scope_date == "2026-09-21" and dropouts[0].scope_time == "08:00:00.500000"
    assert dropouts[1].scope_time == "08:00:01.000000" and dropouts[2].scope_time == "08:00:01.500000"
    assert dropouts[0].relative == pytest.approx(-1.0, abs=1e-7)


def test_tracker_merges_only_openings_closer_than_merge_within(bench: SimulatedBench) -> None:
    bench.contact_openings = [(0.5, 0.5 + 1e-6), (0.5 + 20e-6, 0.5 + 21e-6)]     # 19 µs apart, in one record
    configure(bench.scope, SETTINGS)
    run_once(bench)
    acquisitions, _ = read_interval(bench.scope, SETTINGS, 1)
    assert len(acquisitions) == 2                                                 # forced + one trigger
    two = DropoutTracker(SETTINGS).feed(1, acquisitions, False)
    assert [d.openings for d in two] == [1, 1]
    one = DropoutTracker(ContinuitySettings(merge_within=Q(25, "us"))).feed(1, acquisitions, False)
    assert [d.openings for d in one] == [2] and one[0].duration * 1e6 == pytest.approx(21.0, abs=0.05)


def test_tracker_carries_an_open_contact_across_the_readout_gap() -> None:
    tracker = DropoutTracker(SETTINGS)
    first = [acq(1, 1, -1.0, False, False), acq(1, 2, 0.0, False, True, ((0.0, True),), "falling", time="08:00:59.000")]
    assert tracker.feed(1, first, False) == []
    assert tracker.pending is not None and tracker.count == 1
    # the next interval: the forced acquisition finds the contact still open, it closes 3 s after it opened
    second = [
        acq(2, 1, -1.0, True, True, time="08:01:01.000"),
        acq(2, 2, 0.0, True, False, ((0.0, False),), "rising", time="08:01:02.000"),
    ]
    (d,) = tracker.feed(2, second, False)
    assert tracker.pending is None
    assert d.number == 1 and d.interval == 1 and d.scope_time == "08:00:59.000000"
    assert d.certainty == "approx" and d.duration == pytest.approx(3.0, abs=1e-6)
    assert "spans the stop between intervals 1 and 2" in d.note and "absolute timestamps" in d.note


def test_tracker_reports_a_closing_it_never_saw_as_at_least() -> None:
    tracker = DropoutTracker(SETTINGS)
    first = [acq(1, 1, -1.0, False, False), acq(1, 2, 0.0, False, True, ((0.0, True),), "falling")]
    tracker.feed(1, first, False)
    # the next interval starts with the contact closed: it closed while the scope was stopped
    (d,) = tracker.feed(2, [acq(2, 1, -1.0, False, False)], False)
    assert d.certainty == "at least" and d.duration == pytest.approx(4e-5)          # open to the record's end
    assert "closed while the scope was stopped before interval 2 (the readout)" in d.note
    # and one still open when the monitor stops
    tracker.feed(3, [acq(3, 1, -0.5, False, False), acq(3, 2, 0.0, False, True, ((0.0, True),), "falling")], True)
    (d,) = tracker.finish()
    assert d.number == 2 and d.certainty == "at least" and d.note == "still open when the monitor stopped"


def test_tracker_notes_a_contact_already_open_at_the_start() -> None:
    tracker = DropoutTracker(SETTINGS)
    (d,) = tracker.feed(1, [acq(1, 1, -1.0, True, True), acq(1, 2, 0.0, True, False, ((0.0, False),), "rising")], False)
    assert d.certainty == "at least" and d.duration == pytest.approx(1.0 + 1e-5)
    assert d.note == "already open at the start of interval 1; only the open time seen is counted"


def test_tracker_marks_edges_that_fell_into_the_blind_time() -> None:
    tracker = DropoutTracker(SETTINGS)
    a1 = acq(1, 1, -1.0, False, True, ((0.0, True),), "falling")             # opens, still open at +40 µs
    a2 = acq(1, 2, -0.5, False, True, ((0.0, True),), "falling")             # opens again from closed: the closing
    #   between them was never triggered, so it happened right after a1's record
    a3 = acq(1, 3, 0.0, True, False, ((0.0, False),), "rising")
    d1, d2 = tracker.feed(1, [a1, a2, a3], False)
    assert d1.certainty == "approx" and d1.duration == pytest.approx(4e-5)
    assert d1.note.startswith("closed between two records with no trigger in between, 0.5 s apart")
    assert d2.certainty == "exact" and d2.duration == pytest.approx(0.5)


def test_a_later_record_that_finds_the_contact_open_dates_the_opening_between_records() -> None:
    # the supply ramps down: the trigger fires at the threshold, the record ends still "closed" (inside the
    # hysteresis band), nothing triggers afterwards, and a record 10 s later finds the contact open
    tracker = DropoutTracker(SETTINGS)
    a1 = acq(1, 1, -20.0, False, False)                                       # the forced one at the interval start
    a2 = acq(1, 2, -15.0, False, False, ((0.0, True), (2e-6, False)), "none")  # the ramp: noise, then in the band
    a3 = acq(1, 3, -5.0, True, True)                                           # a glitch trigger while open
    a4 = acq(1, 4, 0.0, True, False, ((0.0, False),), "rising")                # the supply comes back
    assert a2.trigger == "glitch" and a3.trigger == "forced"
    noise, d = tracker.feed(1, [a1, a2, a3, a4], False)
    assert noise.certainty == "exact" and noise.duration == pytest.approx(2e-6)   # the noise pair on its own
    assert d.certainty == "approx" and d.relative == pytest.approx(a2.record_end)
    assert d.duration == pytest.approx(0.0 - a2.record_end)
    assert d.note.startswith("opened between two records with no trigger in between, 10 s apart")
    assert "too slow to trigger" in d.note


# -- the supply channel ---------------------------------------------------------

SUPPLY = ContinuitySettings(interval=Q(3, "s"), segments=100, supply_channel=2)


def test_configure_sets_up_the_supply_channel_on_the_high_impedance_input(bench: SimulatedBench) -> None:
    configure(bench.scope, SUPPLY)
    w = bench.scope_backend.writes
    assert "CHAN2:STAT ON" in w and "CHAN2:COUP DCL" in w              # 1 MΩ: must not load the antenna body node
    assert "CHAN2:SCAL 0.5" in w and "CHAN2:OFFS 1.0" in w
    assert "CHAN2:WAV1:TYPE PDET" in w and "CHAN2:WAV1:HIST:STAT ON" in w
    assert "TRIG1:SOUR CHAN1" in w                                     # the coax still triggers
    with pytest.raises(ValueError):
        configure(bench.scope, ContinuitySettings(supply_channel=1))


def test_supply_channel_tells_antenna_dropouts_from_supply_cable_dropouts(bench: SimulatedBench) -> None:
    bench.supply_openings = [(2.0, 2.0 + 50e-6)]                        # the supply cable breaks for 50 µs
    scope = bench.scope
    configure(scope, SUPPLY)
    run_once(bench)
    acquisitions, _ = read_interval(scope, SUPPLY, 1)
    # the coax sees the supply break as a dropout like any other (both its edges trigger, it outlasts the
    # record); the body wire tells it apart
    assert [a.trigger for a in acquisitions] == [
        "forced", "falling", "falling", "rising", "falling", "falling", "rising",
    ]
    supply = [(a.supply.minimum, a.supply.maximum) for a in acquisitions if a.supply is not None]
    assert len(supply) == 7
    assert supply[0] == (1.0, 1.0)
    assert supply[1] == (1.0, 2.0) and supply[2] == (1.0, 2.0)          # antenna open: the body rises to 2 V
    assert supply[5] == (0.0, 1.0) and supply[6] == (0.0, 1.0)          # supply gone: the body drops to 0 V
    tracker = DropoutTracker(SUPPLY)
    dropouts = tracker.feed(1, acquisitions, False)
    assert [d.cause for d in dropouts] == ["antenna", "antenna", "antenna", "supply"]
    assert [d.supply_minimum for d in dropouts] == [1.0, 1.0, 1.0, 0.0]
    assert dropouts[3].duration * 1e6 == pytest.approx(50.0, abs=0.05)
    assert (tracker.count, tracker.antenna_count, tracker.supply_count) == (4, 3, 1)
    # without a supply channel nothing is attributed
    bare = [Acquisition(a.interval, a.number, a.scope_date, a.scope_time, a.relative, a.record_start,
                        a.record_end, a.analysis) for a in acquisitions]
    plain = DropoutTracker(SETTINGS).feed(1, bare, False)
    assert all(d.cause == "" and np.isnan(d.supply_minimum) for d in plain)


def test_monitor_reports_and_logs_the_cause(tmp_path: Path, bench: SimulatedBench) -> None:
    bench.supply_openings = [(2.0, 2.0 + 50e-6)]
    scope = bench.scope
    configure(scope, SUPPLY)
    log = ContinuityLog(tmp_path, "run").open({})
    now = [datetime(2026, 9, 21, 8, 0, 0)]

    def clock() -> datetime:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += timedelta(seconds=seconds)

    lines: list[str] = []
    monitor(scope, SUPPLY, log, intervals=1, clock=clock, sleep=sleep, report=lines.append)
    assert lines[-1] == "interval 1: 7 acquisition(s), 4 dropout(s) final, 4 in total (3 antenna, 1 supply cable)"
    header, *rows = (tmp_path / "run dropouts.csv").read_text(encoding="utf-8").splitlines()
    assert header.endswith("Minimum [V],Cause,Supply min [V],Note")
    assert [r.split(",")[9:11] for r in rows] == [
        ["antenna", "1.0000"], ["antenna", "1.0000"], ["antenna", "1.0000"], ["supply", "0.0000"],
    ]
    header, *rows = (tmp_path / "run acquisitions.csv").read_text(encoding="utf-8").splitlines()
    assert header.endswith("Maximum [V],Supply min [V],Supply max [V]")
    assert rows[1].split(",")[-2:] == ["1.0000", "2.0000"] and rows[-1].split(",")[-2:] == ["0.0000", "1.0000"]
    log.close()


# -- the loop and the files ----------------------------------------------------


def test_monitor_loop_writes_the_three_files_flushed(tmp_path: Path, bench: SimulatedBench) -> None:
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
    assert lines == [
        "interval 1: 5 acquisition(s), 3 dropout(s) final, 3 in total",
        "interval 2: 5 acquisition(s), 3 dropout(s) final, 6 in total",
    ]
    acquisitions = (tmp_path / "run acquisitions.csv").read_text(encoding="utf-8").splitlines()
    assert acquisitions[:2] == ["# DUT: Antenna X", "# Test: vibration"]
    assert acquisitions[2].startswith("Interval,Acquisition,Scope date,Scope time,Relative [s],Trigger,Starts,Ends,")
    rows = [line.split(",") for line in acquisitions[3:]]
    assert len(rows) == 10 and [r[0] for r in rows] == ["1"] * 5 + ["2"] * 5
    assert [r[5] for r in rows[:5]] == ["forced", "falling", "falling", "rising", "falling"]
    assert rows[2][8].startswith("F+0.000") and rows[3][8].startswith("R+0.000")
    assert [r[6] for r in rows[:5]] == ["closed", "closed", "closed", "open", "closed"]
    assert [r[7] for r in rows[:5]] == ["closed", "closed", "open", "closed", "closed"]
    dropouts = (tmp_path / "run dropouts.csv").read_text(encoding="utf-8").splitlines()
    assert dropouts[2].startswith("Dropout,Interval,Scope date,Scope time,Relative [s],Duration [us],Duration is,")
    rows = [line.split(",") for line in dropouts[3:]]
    assert [r[0] for r in rows] == ["1", "2", "3", "4", "5", "6"]
    assert [r[6] for r in rows] == ["exact"] * 6 and rows[1][5] == "100.000"
    assert rows[3][3] == "08:01:00.500000"                                # the second run starts a minute later
    intervals = (tmp_path / "run intervals.csv").read_text(encoding="utf-8").splitlines()
    assert intervals[-1].startswith("2,2026-09-21T08:00:02.200,2026-09-21T08:00:04.400,5,")
    w = bench.scope_backend.writes
    assert w.count("RUNS") == 2 and w.count("TRIG1:FORC") == 2 and w.count("STOP") == 2
    log.close()


def test_monitor_reads_out_early_when_the_memory_is_nearly_full(tmp_path: Path, bench: SimulatedBench) -> None:
    bench.contact_openings = [(0.001 * k, 0.001 * k + 2e-6) for k in range(1, 96)]   # 95 edges + the forced one
    scope = bench.scope
    capacity = configure(scope, SETTINGS).capacity                                    # 100 segments: 90 is "nearly full"
    log = ContinuityLog(tmp_path, "run").open({})
    now = [datetime(2026, 9, 21, 8, 0, 0)]

    def clock() -> datetime:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += timedelta(seconds=seconds)

    lines: list[str] = []
    monitor(scope, SETTINGS, log, intervals=1, clock=clock, sleep=sleep, report=lines.append, capacity=capacity)
    assert lines[0] == "interval 1: the scope holds 96 acquisitions, reading out early"
    assert lines[1].startswith("interval 1: 96 acquisition(s), 95 dropout(s) final")
    intervals = (tmp_path / "run intervals.csv").read_text(encoding="utf-8").splitlines()
    assert intervals[-1].startswith("1,2026-09-21T08:00:00.000,2026-09-21T08:00:01.200,96,0.000,no")  # not the 2 s
    assert bench.scope_backend.queries.count("ACQ:AVA?") == 2                          # the poll, then the readout
    log.close()


def test_monitor_reads_out_the_interval_on_keyboard_interrupt(tmp_path: Path, bench: SimulatedBench) -> None:
    scope = bench.scope
    configure(scope, SETTINGS)
    log = ContinuityLog(tmp_path, "run").open({})
    calls = [0]

    def sleep(seconds: float) -> None:
        calls[0] += 1
        if calls[0] > 1:            # let the scope arm and the forced acquisition happen, then Ctrl-C
            raise KeyboardInterrupt

    total = monitor(scope, SETTINGS, log, sleep=sleep, report=lambda _: None)
    assert total == 3
    assert bench.scope_backend.writes.count("STOP") == 1
    rows = (tmp_path / "run dropouts.csv").read_text(encoding="utf-8").splitlines()
    assert len([r for r in rows if r.startswith(("1,", "2,", "3,"))]) == 3
    log.close()


def test_log_appends_to_existing_files_without_repeating_headers(tmp_path: Path) -> None:
    log = ContinuityLog(tmp_path, "run").open({"DUT": "A"})
    log.write_dropout(DropoutRecord(1, 1, "2026-09-21", "08:00:00.000000", 0.0, 2.5e-6, "exact", 1, 0.01, ""))
    log.close()
    log = ContinuityLog(tmp_path, "run").open({"DUT": "A"})
    log.write_dropout(DropoutRecord(2, 2, "2026-09-21", "08:01:00.000000", -0.5, 4e-5, "at least", 2, 0.0, "x"))
    log.close()
    text = (tmp_path / "run dropouts.csv").read_text(encoding="utf-8")
    assert text.count("# DUT: A") == 1 and text.count("Dropout,Interval") == 1
    assert text.splitlines()[-2:] == [
        "1,1,2026-09-21,08:00:00.000000,0.000000000,2.500,exact,1,0.0100,,,",
        "2,2,2026-09-21,08:01:00.000000,-0.500000000,40.000,at least,2,0.0000,,,x",
    ]


def test_scope_clock_strings_parse_leniently() -> None:
    assert _time_of_day("08:00:59.123456789", 0.5) == "08:00:59.623457"
    assert _time_of_day("8:00:59", -1.0) == "08:00:58.000000"
    assert _time_of_day("nonsense", 0.0) is None
    iso = acq(1, 1, 0.0, False, False, date="2026-09-21", time="08:00:00.000")
    dotted = acq(1, 1, 0.0, False, False, date="21.09.2026", time="08:00:00")
    csv_date = acq(1, 1, 0.0, False, False, date="2026,9,21", time="08:00:00")
    assert _absolute_seconds(iso) == _absolute_seconds(dotted) == _absolute_seconds(csv_date)
    assert _absolute_seconds(acq(1, 1, 0.0, False, False, date="?", time="08:00:00")) is None
