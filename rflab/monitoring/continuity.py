"""Electrical-contact monitoring on the oscilloscope during environmental tests.

The circuit
-----------
A lab supply at 2 V drives 20 mA through a 50 Ω resistor soldered to the
antenna body, through the contact under test, down the coax into the scope's
50 Ω input. With the contact closed the scope sees 1 V; when the contact opens,
the scope side collapses to 0 V within nanoseconds. Every dropout is a falling
edge.

How the scope catches them
--------------------------
The scope triggers on that edge in NORMAL mode, so it acquires only when a
dropout happens, and fast segmentation lets it re-arm in microseconds and keep
thousands of acquisitions in memory, each with its own timestamp. A holdoff
merges the bounce of one opening into one event.

How nothing is lost
-------------------
:func:`monitor` runs in intervals: start a single run, wait `interval`, stop,
read the number of stored acquisitions and the timestamp of each from the
history, append them to the events file and **flush**, restart. The file is
complete at any moment the test is interrupted. Each interval is logged too,
with its start and stop on the PC clock, the event count and the readout dead
time — so a gap in coverage is written down instead of passing as a clean
minute. For the first `waveforms_per_interval` events of an interval the
captured record is read as well and the dropout duration is measured from it.

Timestamps come from the scope's clock; the offset to the PC clock is measured
once at the start and written into the log header.
"""

from __future__ import annotations

import csv
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TextIO

import numpy as np

from labkit.instruments import RTO64
from labkit.units import Quantity, quantity as Q

__all__ = [
    "ContinuitySettings",
    "DropoutEvent",
    "IntervalRecord",
    "ContinuityLog",
    "configure",
    "read_interval",
    "dropout_duration",
    "monitor",
]

Slope = Literal["NEGATIVE", "POSITIVE"]


@dataclass(frozen=True)
class ContinuitySettings:
    """How the scope watches the contact."""

    #: Scope channel the coax is connected to.
    channel: int = 1
    #: Voltage with the contact closed (2 V through 50 Ω + the 50 Ω input = 1 V).
    closed_level: Quantity = Q(1.0, "V")
    #: Trigger level and edge: a falling edge through 0.5 V is a dropout.
    trigger_level: Quantity = Q(0.5, "V")
    slope: Slope = "NEGATIVE"
    #: One bouncing contact counts as one event: no new trigger within this time.
    holdoff: Quantity = Q(10, "us")
    #: The record captured around each dropout (pre-trigger 20 %), and its sample rate.
    window: Quantity = Q(50, "us")
    sample_rate: Quantity = Q(200, "MHz")
    #: Acquisitions kept in memory per interval; a run stops early when full.
    segments: int = 10_000
    #: How long the scope runs before the captured events are read out and saved.
    interval: Quantity = Q(60, "s")
    #: For this many events per interval the record is read too, to measure the dropout duration.
    waveforms_per_interval: int = 20
    #: Vertical scale, so 0 V and 1 V are both well inside the screen.
    scale: Quantity = Q(200, "mV")

    @property
    def pre_trigger(self) -> Quantity:
        return self.window * 0.2  # type: ignore[no-any-return]


@dataclass(frozen=True)
class DropoutEvent:
    """One loss of contact."""

    interval: int
    #: Position in the interval, oldest first (1-based).
    number: int
    #: The scope clock: date and time of day of the acquisition.
    scope_date: str
    scope_time: str
    #: Seconds relative to the newest acquisition of the interval (zero or negative).
    relative: float
    #: Time the voltage stayed below the trigger level inside the captured record, if the record was read.
    duration: Optional[Quantity] = None
    #: Lowest voltage in the record, if read.
    minimum: Optional[Quantity] = None


@dataclass(frozen=True)
class IntervalRecord:
    """One monitoring interval as seen from the PC."""

    interval: int
    started: datetime
    stopped: datetime
    events: int
    #: Seconds the scope was stopped for the readout before the next interval started.
    dead_time: float
    #: The scope stopped by itself because the segment memory was full.
    memory_full: bool


@dataclass
class ContinuityLog:
    """Two CSV files, flushed after every row: the events and the intervals."""

    folder: Path
    name: str
    _events: Optional[TextIO] = field(default=None, repr=False)
    _intervals: Optional[TextIO] = field(default=None, repr=False)
    _event_writer: Any = field(default=None, repr=False)
    _interval_writer: Any = field(default=None, repr=False)

    @property
    def events_path(self) -> Path:
        return self.folder / f"{self.name} events.csv"

    @property
    def intervals_path(self) -> Path:
        return self.folder / f"{self.name} intervals.csv"

    def open(self, header: dict[str, str]) -> "ContinuityLog":
        """Create the files (appending if they exist) and write the header lines."""
        self.folder.mkdir(parents=True, exist_ok=True)
        new_events = not self.events_path.exists()
        new_intervals = not self.intervals_path.exists()
        self._events = self.events_path.open("a", newline="", encoding="utf-8")
        self._intervals = self.intervals_path.open("a", newline="", encoding="utf-8")
        for handle, new in ((self._events, new_events), (self._intervals, new_intervals)):
            if new:
                for key, value in header.items():
                    handle.write(f"# {key}: {value}\n")
        self._event_writer = csv.writer(self._events)
        self._interval_writer = csv.writer(self._intervals)
        if new_events:
            self._event_writer.writerow([
                "Interval", "Event", "Scope date", "Scope time", "Relative [s]", "Duration [us]", "Minimum [V]",
            ])
        if new_intervals:
            self._interval_writer.writerow([
                "Interval", "Started (PC)", "Stopped (PC)", "Events", "Readout dead time [s]", "Memory full",
            ])
        self._flush()
        return self

    def write_event(self, event: DropoutEvent) -> None:
        assert self._event_writer is not None, "open() first"
        duration = "" if event.duration is None else f"{event.duration.to('us').magnitude:.3f}"
        minimum = "" if event.minimum is None else f"{event.minimum.to('V').magnitude:.4f}"
        self._event_writer.writerow([
            event.interval, event.number, event.scope_date, event.scope_time, f"{event.relative:.9f}", duration, minimum,
        ])
        self._flush()

    def write_interval(self, record: IntervalRecord) -> None:
        assert self._interval_writer is not None, "open() first"
        self._interval_writer.writerow([
            record.interval, record.started.isoformat(timespec="milliseconds"),
            record.stopped.isoformat(timespec="milliseconds"), record.events, f"{record.dead_time:.3f}",
            "yes" if record.memory_full else "no",
        ])
        self._flush()

    def _flush(self) -> None:
        for handle in (self._events, self._intervals):
            if handle is not None:
                handle.flush()

    def close(self) -> None:
        for handle in (self._events, self._intervals):
            if handle is not None:
                handle.close()
        self._events = self._intervals = None


def configure(scope: RTO64, settings: ContinuitySettings) -> None:
    """Set the scope up for the monitor: channel, timebase, trigger, fast segmentation."""
    ch = scope.channel(settings.channel)
    scope.system.set_display_update(False)
    ch.enable(True)
    ch.set_coupling("DC")                        # the 50 Ω input: the node collapses in ns when the contact opens
    ch.set_bandwidth_limit("FULL")
    ch.set_scale(settings.scale)
    ch.set_offset(settings.closed_level / 2)     # centre the 0 V – closed_level swing on screen
    ch.set_arithmetics("OFF")
    scope.timebase.set_range(settings.window)
    scope.timebase.set_reference(20)
    scope.timebase.set_position(Q(0, "s"))
    scope.acquisition.set_points_mode("RESOLUTION")
    scope.acquisition.set_sample_rate(settings.sample_rate)
    scope.acquisition.set_count(settings.segments)
    scope.acquisition.set_fast_segmentation(True, settings.segments)
    scope.trigger.edge(f"CH{settings.channel}", settings.trigger_level, slope=settings.slope, mode="NORMAL")
    scope.trigger.set_holdoff("TIME", time=settings.holdoff)
    scope.history.enable(settings.channel, True)


def dropout_duration(time: Quantity, voltage: Quantity, threshold: Quantity) -> tuple[Quantity, Quantity]:
    """``(duration below threshold, minimum voltage)`` of one captured record.

    The duration is the time between the first and the last sample below the
    threshold, so a bouncing contact counts from the first opening to the
    last. A record that is still below the threshold at its end reports the
    time to the end of the record: the dropout outlasted the window.
    """
    t = np.asarray(time.to("s").magnitude, dtype=np.float64)
    v = np.asarray(voltage.to("V").magnitude, dtype=np.float64)
    if v.ndim == 2:  # envelope / peak-detect record: use the minimum per interval
        v = v.min(axis=1)
    below = np.flatnonzero(v < threshold.to("V").magnitude)
    minimum = Q(float(v.min()), "V")
    if len(below) == 0:
        return Q(0.0, "s"), minimum
    step = float(t[1] - t[0]) if len(t) > 1 else 0.0
    return Q(float(t[below[-1]] - t[below[0]] + step), "s"), minimum


def read_interval(
    scope: RTO64, settings: ContinuitySettings, interval: int
) -> tuple[list[DropoutEvent], bool]:
    """Read every acquisition the stopped scope holds: ``(events oldest first, memory was full)``."""
    count = scope.history.available()
    memory_full = count >= settings.segments
    events: list[DropoutEvent] = []
    stamps = scope.history.timestamps(settings.channel, count)
    for number, stamp in enumerate(stamps, start=1):
        duration = minimum = None
        if number <= settings.waveforms_per_interval:
            scope.history.select(settings.channel, stamp.index)
            t, v = scope.waveform.get_data(settings.channel)
            duration, minimum = dropout_duration(t, v, settings.trigger_level)
        events.append(DropoutEvent(interval, number, stamp.date, stamp.time, stamp.relative, duration, minimum))
    return events, memory_full


def monitor(
    scope: RTO64,
    settings: ContinuitySettings,
    log: ContinuityLog,
    stop: Callable[[], bool] = lambda: False,
    intervals: Optional[int] = None,
    clock: Callable[[], datetime] = datetime.now,
    sleep: Callable[[float], None] = _time.sleep,
    report: Callable[[str], None] = print,
) -> int:
    """Run the monitor until `stop` returns true (or `intervals` are done); returns the event total.

    Every interval: single run, wait, stop, read the history, write every
    event and the interval record, run again. A keyboard interrupt (Ctrl-C)
    ends the wait early: the current interval is still stopped, read out and
    written before the function returns, so nothing captured is lost.
    """
    total = 0
    number = 0
    seconds = float(settings.interval.to("s").magnitude)
    readout_ended: Optional[datetime] = None
    interrupted = False
    while not interrupted and not stop() and (intervals is None or number < intervals):
        number += 1
        scope.run_single(wait_for_completion=False)
        started = clock()
        dead_time = 0.0 if readout_ended is None else (started - readout_ended).total_seconds()
        waited = 0.0
        try:
            while waited < seconds and not stop():
                step = min(1.0, seconds - waited)
                sleep(step)
                waited += step
        except KeyboardInterrupt:
            interrupted = True
            report("interrupted: reading out the current interval ...")
        scope.stop()
        stopped = clock()
        events, memory_full = read_interval(scope, settings, number)
        for event in events:
            log.write_event(event)
        log.write_interval(IntervalRecord(number, started, stopped, len(events), dead_time, memory_full))
        readout_ended = clock()
        total += len(events)
        note = "  MEMORY FULL: the scope stopped early, events after that were not captured" if memory_full else ""
        report(f"interval {number}: {len(events)} dropout(s), {total} in total{note}")
    return total
