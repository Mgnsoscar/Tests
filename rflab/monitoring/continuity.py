"""Electrical-contact monitoring on the oscilloscope during environmental tests.

The circuit
-----------
A lab supply at 2 V drives 20 mA through a 50 Ω resistor soldered to the
antenna body, through the contact under test, down the coax into the scope's
50 Ω input. With the contact closed the scope sees 1 V; when the contact opens,
the scope side collapses to 0 V within nanoseconds. Contact open = voltage
below the threshold.

How the scope catches every edge
--------------------------------
The scope triggers on **either** edge through the threshold in NORMAL mode, so
it acquires only when the contact opens or closes, and fast segmentation lets
it re-arm within microseconds and keep thousands of acquisitions in memory,
each with its own timestamp. Every acquisition holds a record around its
trigger (20 % before, 80 % after) in **peak-detect** mode: each sample interval
keeps its minimum and maximum, so any crossing the trigger saw is in the
record too, however brief.

How the edges are paired
------------------------
Nothing is assumed from the order of triggers. Each record is analysed for
its threshold crossings (the time and direction of every one, plus the level
at its start and end); the crossings of an interval are stitched into one
timeline on the scope's timestamps, overlaps between neighbouring records
removed. A dropout is the stretch from a falling crossing to the rising
crossing that closes it, and openings closer together than `merge_within`
count as one bouncing dropout. Where the scope could not see — its blind time
after a record, the readout gap between intervals, a stop on full memory — the
level before and after the blind spot is compared and any crossing that must
have happened there is inferred and marked, so a duration is reported as
*exact*, *approx* (an edge fell into the sub-microsecond blind time after a
record, or the dropout spans a stop and was measured on the scope's absolute
clock) or *at least*
(the closing was never seen). At the start of every interval one acquisition
is forced, so the contact state at that moment is on record even when nothing
else triggers.

How nothing is lost
-------------------
:func:`monitor` runs in intervals: start a single run, wait `interval`, stop,
read every stored acquisition, write and **flush** three files, restart. An
interval also ends early when the scope's memory is nearly full, so a
chattering contact costs a readout gap, not lost edges:

* the *acquisitions* file — one row per trigger with its timestamp, the edge
  that triggered, the crossings in its record and its extreme voltages: the raw
  evidence, enough to redo the pairing by hand;
* the *dropouts* file — one row per dropout with its start on the scope clock,
  its duration and how certain that duration is;
* the *intervals* file — each interval's start and stop on the PC clock, its
  acquisition count, the readout dead time and whether the memory filled.

Timestamps come from the scope's clock; the offset to the PC clock is measured
once at the start and written into the file headers.
"""

from __future__ import annotations

import csv
import re
import time as _time
from dataclasses import dataclass, field
from datetime import date as _date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TextIO

import numpy as np

from labkit.instruments import RTO64
from labkit.units import Quantity, quantity as Q

__all__ = [
    "ContinuitySettings",
    "RecordAnalysis",
    "Acquisition",
    "Crossing",
    "DropoutRecord",
    "IntervalRecord",
    "ContinuityLog",
    "DropoutTracker",
    "configure",
    "analyse_record",
    "read_interval",
    "interval_crossings",
    "monitor",
]

Certainty = Literal["exact", "approx", "at least"]

#: Seconds to let the scope arm after ``RUNSingle`` before the first acquisition is forced.
_ARM_DELAY = 0.2


@dataclass(frozen=True)
class ContinuitySettings:
    """How the scope watches the contact."""

    #: Scope channel the coax is connected to.
    channel: int = 1
    #: Voltage with the contact closed (2 V through 50 Ω + the 50 Ω input = 1 V).
    closed_level: Quantity = Q(1.0, "V")
    #: The trigger level, and the level below which the contact counts as open.
    threshold: Quantity = Q(0.5, "V")
    #: The record captured around each edge (20 % before it), and its sample rate.
    #: Peak detect keeps the min and max of every sample interval, so the rate
    #: only sets the time resolution of the crossings, not what is caught.
    window: Quantity = Q(50, "us")
    sample_rate: Quantity = Q(50, "MHz")
    #: Acquisitions kept in memory per interval; a run stops early when full.
    segments: int = 10_000
    #: How long the scope runs before the captured acquisitions are read out and saved.
    interval: Quantity = Q(60, "s")
    #: Read out early when the scope already holds this fraction of its segment capacity,
    #: so a chattering contact rolls into a new interval instead of filling the memory.
    readout_when_full: float = 0.9
    #: Openings closer together than this are one bouncing dropout.
    merge_within: Quantity = Q(10, "us")
    #: Vertical scale, so 0 V and 1 V are both well inside the screen.
    scale: Quantity = Q(200, "mV")

    @property
    def pre_trigger(self) -> Quantity:
        return self.window * 0.2  # type: ignore[no-any-return]

    @property
    def post_trigger(self) -> Quantity:
        return self.window * 0.8  # type: ignore[no-any-return]


# -- one record ----------------------------------------------------------------


@dataclass(frozen=True)
class RecordAnalysis:
    """What one captured record says about the contact."""

    #: Contact state at the first and the last sample.
    starts_open: bool
    ends_open: bool
    #: Every threshold crossing: ``(time in the record in seconds, falling)``.
    crossings: tuple[tuple[float, bool], ...]
    #: The edge at the trigger point: ``"falling"``, ``"rising"`` or ``"none"`` (a forced
    #: acquisition, or a glitch shorter than the guard around the trigger point).
    trigger_edge: str
    minimum: float
    maximum: float


def analyse_record(time: Quantity, voltage: Quantity, threshold: Quantity) -> RecordAnalysis:
    """Find the threshold crossings of a record; peak-detect records (min/max per sample) are supported.

    A sample interval counts as open when its minimum is below the threshold,
    so a crossing is placed at the first sample interval of the new state.
    """
    t = np.asarray(time.to("s").magnitude, dtype=np.float64)
    v = np.asarray(voltage.to("V").magnitude, dtype=np.float64)
    low, high = (v.min(axis=1), v.max(axis=1)) if v.ndim == 2 else (v, v)
    is_open = low < float(threshold.to("V").magnitude)
    changes = np.flatnonzero(is_open[1:] != is_open[:-1]) + 1
    crossings = tuple((float(t[i]), bool(is_open[i])) for i in changes)
    at_trigger = int(np.argmin(np.abs(t)))
    guard = 3
    before = bool(is_open[max(at_trigger - guard, 0)])
    after = bool(is_open[min(at_trigger + guard, len(t) - 1)])
    edge = "falling" if (not before and after) else "rising" if (before and not after) else "none"
    return RecordAnalysis(bool(is_open[0]), bool(is_open[-1]), crossings, edge, float(low.min()), float(high.max()))


@dataclass(frozen=True)
class Acquisition:
    """One stored acquisition of an interval, with its record analysed."""

    interval: int
    #: Position in the interval, oldest first (1-based).
    number: int
    #: The scope clock: date and time of day of the trigger.
    scope_date: str
    scope_time: str
    #: Trigger time in seconds relative to the newest acquisition of the interval (zero or negative).
    relative: float
    #: The record's span in that same frame.
    record_start: float
    record_end: float
    analysis: RecordAnalysis

    @property
    def trigger(self) -> str:
        """``forced`` for the state snapshot at the interval start, else the edge seen at the trigger."""
        if self.number == 1 and self.analysis.trigger_edge == "none":
            return "forced"
        return self.analysis.trigger_edge


# -- the timeline of one interval ----------------------------------------------


@dataclass(frozen=True)
class Crossing:
    """One threshold crossing on the interval's timeline."""

    #: Seconds relative to the newest acquisition of the interval.
    time: float
    falling: bool
    #: Not seen in a record: deduced from the level after a blind spot of the scope.
    inferred: bool = False


def interval_crossings(acquisitions: list[Acquisition]) -> list[Crossing]:
    """Stitch the crossings of an interval's records into one timeline, oldest first.

    Where two records overlap, a crossing is taken once. Where they do not,
    the level at the end of one and at the start of the next must agree: the
    scope was armed in between and any crossing would have triggered — unless
    it fell into the scope's blind time right after the record. A disagreement
    is such a crossing; it is placed at the end of the earlier record and
    marked inferred.
    """
    timeline: list[Crossing] = []
    covered_until: Optional[float] = None
    open_at_end = False
    for a in acquisitions:
        if covered_until is not None and a.record_start > covered_until and a.analysis.starts_open != open_at_end:
            timeline.append(Crossing(covered_until, falling=a.analysis.starts_open, inferred=True))
        for t, falling in a.analysis.crossings:
            at = a.relative + t
            if covered_until is None or at > covered_until:
                timeline.append(Crossing(at, falling))
        if covered_until is None or a.record_end > covered_until:
            covered_until = a.record_end
            open_at_end = a.analysis.ends_open
    return timeline


# -- dropouts ------------------------------------------------------------------


@dataclass(frozen=True)
class DropoutRecord:
    """One loss of contact, as written to the dropouts file."""

    number: int
    #: The interval the opening was seen in.
    interval: int
    #: The scope clock at the opening (the acquisition's timestamp plus the offset in its record).
    scope_date: str
    scope_time: str
    #: The opening in seconds relative to the newest acquisition of its interval.
    relative: float
    #: Open time in seconds; for ``at least`` the time the contact was seen open.
    duration: float
    certainty: Certainty
    #: Openings merged into this dropout (a bouncing contact).
    openings: int
    minimum: float
    note: str


@dataclass(frozen=True)
class IntervalRecord:
    """One monitoring interval as seen from the PC."""

    interval: int
    started: datetime
    stopped: datetime
    acquisitions: int
    #: Seconds from the previous interval's stop to this one's start: the scope was stopped for the readout.
    dead_time: float
    #: The scope stopped by itself because the segment memory was full.
    memory_full: bool


@dataclass
class _Dropout:
    """A dropout being assembled. Times are in the frame of the interval named next to them."""

    number: int
    interval: int
    start: float
    start_acquisition: Acquisition
    start_inferred: bool
    #: The opening was never seen: the contact was already open when the interval's first record began.
    start_unknown: bool = False
    openings: int = 1
    end: Optional[float] = None
    end_interval: Optional[int] = None
    end_inferred: bool = False
    #: The latest time the contact was seen open: in the start's own frame, and in any frame.
    open_until_own: float = 0.0
    open_until: float = 0.0
    open_until_interval: int = 0
    minimum: float = float("inf")
    notes: list[str] = field(default_factory=list)


class DropoutTracker:
    """Turns the intervals' crossings into dropouts; keeps one open across intervals."""

    def __init__(self, settings: ContinuitySettings) -> None:
        self.merge_within = float(settings.merge_within.to("s").magnitude)
        #: Dropouts opened so far.
        self.count = 0
        self._pending: Optional[_Dropout] = None
        self._memory_full_before = False
        #: Per interval: the newest acquisition's trigger time on the scope's absolute clock, if it parsed.
        self._frames: dict[int, Optional[float]] = {}

    @property
    def pending(self) -> Optional[_Dropout]:
        """The dropout open at the end of the last interval fed, if any."""
        return self._pending

    def feed(self, interval: int, acquisitions: list[Acquisition], memory_full: bool) -> list[DropoutRecord]:
        """Digest one interval; returns the dropouts that became final, in order."""
        pending = self._pending
        if not acquisitions:
            if pending is not None:
                pending.notes.append(f"interval {interval} held no acquisition, contact state unknown")
            self._memory_full_before = memory_full
            return []
        self._frames[interval] = _absolute_seconds(acquisitions[-1])
        first = acquisitions[0]
        done: list[_Dropout] = []

        # -- the stop between the previous interval and this one --------------
        if pending is not None and not first.analysis.starts_open:
            why = "the scope had stopped on full memory" if self._memory_full_before else "the readout"
            pending.notes.append(f"closed while the scope was stopped before interval {interval} ({why})")
            done.append(pending)
            pending = None
        elif pending is not None:
            pending.notes.append(f"spans the stop between intervals {pending.interval} and {interval}")
        elif first.analysis.starts_open:
            pending = self._open(interval, first.record_start, acquisitions, inferred=False)
            pending.start_unknown = True
            pending.notes.append(f"already open at the start of interval {interval}; only the open time seen is counted")

        # -- this interval's crossings ----------------------------------------
        settling: Optional[_Dropout] = None       # closed less than merge_within ago: may reopen
        is_open = pending is not None
        for c in interval_crossings(acquisitions):
            if c.falling:
                if is_open:
                    continue
                is_open = True
                if settling is not None and c.time - float(settling.end or 0.0) < self.merge_within:
                    settling.openings += 1
                    settling.end = settling.end_interval = None
                    settling.end_inferred = False
                    pending, settling = settling, None
                else:
                    if settling is not None:
                        done.append(settling)
                    settling = None
                    pending = self._open(interval, c.time, acquisitions, inferred=c.inferred)
            elif is_open and pending is not None:
                is_open = False
                pending.end, pending.end_interval, pending.end_inferred = c.time, interval, c.inferred
                self._seen_open(pending, interval, c.time)
                settling, pending = pending, None
        if settling is not None:
            done.append(settling)
        if pending is not None:
            self._seen_open(pending, interval, max(a.record_end for a in acquisitions))

        for d in done + ([pending] if pending is not None else []):
            self._track_minimum(d, interval, acquisitions)
        self._pending = pending
        self._memory_full_before = memory_full
        return [self._finish(d) for d in done]

    def finish(self) -> list[DropoutRecord]:
        """The dropout still open when the monitor stops, if any."""
        if self._pending is None:
            return []
        pending, self._pending = self._pending, None
        pending.notes.append("still open when the monitor stopped")
        return [self._finish(pending)]

    # -- helpers -----------------------------------------------------------
    def _open(self, interval: int, time: float, acquisitions: list[Acquisition], inferred: bool) -> _Dropout:
        self.count += 1
        d = _Dropout(self.count, interval, time, _acquisition_at(acquisitions, time), inferred)
        self._seen_open(d, interval, time)
        if inferred:
            d.notes.append("opened in the scope's blind time after a record; start is that record's end")
        return d

    @staticmethod
    def _seen_open(d: _Dropout, interval: int, time: float) -> None:
        d.open_until, d.open_until_interval = time, interval
        if interval == d.interval:
            d.open_until_own = time

    @staticmethod
    def _track_minimum(d: _Dropout, interval: int, acquisitions: list[Acquisition]) -> None:
        """Fold in the lowest voltage of this interval's records that overlap the dropout."""
        span_start = d.start if d.interval == interval else -np.inf
        span_end = d.end if (d.end is not None and d.end_interval == interval) else np.inf
        for a in acquisitions:
            if a.record_end >= span_start and a.record_start <= span_end:
                d.minimum = min(d.minimum, a.analysis.minimum)

    def _finish(self, d: _Dropout) -> DropoutRecord:
        duration, certainty = self._duration(d)
        if d.end_inferred:
            d.notes.append("closed in the scope's blind time after a record; end is that record's end")
        acq = d.start_acquisition
        offset = d.start - acq.relative
        scope_time = _time_of_day(acq.scope_time, offset)
        if scope_time is None:
            scope_time = acq.scope_time
            d.notes.append(f"opening at {offset * 1e6:+.3f} us from that acquisition's timestamp")
        minimum = float("nan") if np.isinf(d.minimum) else d.minimum
        return DropoutRecord(
            d.number, d.interval, acq.scope_date, scope_time, d.start, duration, certainty,
            d.openings, minimum, "; ".join(d.notes),
        )

    def _absolute(self, interval: int, time: float) -> Optional[float]:
        frame = self._frames.get(interval)
        return None if frame is None else frame + time

    def _duration(self, d: _Dropout) -> tuple[float, Certainty]:
        if d.end is not None and d.end_interval == d.interval:
            if d.start_unknown:
                return d.end - d.start, "at least"
            return d.end - d.start, "approx" if (d.start_inferred or d.end_inferred) else "exact"
        if d.end is not None and d.end_interval is not None:
            start, end = self._absolute(d.interval, d.start), self._absolute(d.end_interval, d.end)
            if start is not None and end is not None:
                d.notes.append("duration across the stop from the scope's absolute timestamps")
                return end - start, "approx"
            d.notes.append("scope timestamps could not be parsed; only the open time seen before the stop is counted")
            return d.open_until_own - d.start, "at least"
        if d.open_until_interval == d.interval:
            return d.open_until - d.start, "at least"
        start, end = self._absolute(d.interval, d.start), self._absolute(d.open_until_interval, d.open_until)
        if start is not None and end is not None:
            return end - start, "at least"
        return d.open_until_own - d.start, "at least"


def _acquisition_at(acquisitions: list[Acquisition], time: float) -> Acquisition:
    """The acquisition whose record holds `time`, else the nearest one."""
    for a in acquisitions:
        if a.record_start <= time <= a.record_end:
            return a
    return min(acquisitions, key=lambda a: abs(a.relative - time))


# -- scope clock strings --------------------------------------------------------

_TIME = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d+))?")
_DATES = (
    (re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})"), (1, 2, 3)),
    (re.compile(r"(\d{4}),(\d{1,2}),(\d{1,2})"), (1, 2, 3)),
    (re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})"), (3, 2, 1)),
)


def _seconds_of_day(text: str) -> Optional[float]:
    m = _TIME.search(text)
    if m is None:
        return None
    fraction = float("0." + m.group(4)) if m.group(4) else 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + fraction


def _parse_date(text: str) -> Optional[_date]:
    for regex, (y, mo, d) in _DATES:
        m = regex.search(text)
        if m is not None:
            try:
                return _date(int(m.group(y)), int(m.group(mo)), int(m.group(d)))
            except ValueError:
                return None
    return None


def _absolute_seconds(a: Acquisition) -> Optional[float]:
    """The acquisition's trigger time as seconds on the scope's clock, if its strings parse."""
    day = _parse_date(a.scope_date)
    seconds = _seconds_of_day(a.scope_time)
    if day is None or seconds is None:
        return None
    return day.toordinal() * 86400.0 + seconds


def _time_of_day(acquisition_time: str, offset: float) -> Optional[str]:
    """The acquisition's time of day plus `offset` seconds, to the microsecond."""
    seconds = _seconds_of_day(acquisition_time)
    if seconds is None:
        return None
    total = (seconds + offset) % 86400
    return (datetime(2000, 1, 1) + timedelta(seconds=total)).strftime("%H:%M:%S.%f")


# -- files ---------------------------------------------------------------------

_COLUMNS = {
    "acquisitions": [
        "Interval", "Acquisition", "Scope date", "Scope time", "Relative [s]", "Trigger",
        "Starts", "Ends", "Crossings [us from trigger]", "Minimum [V]", "Maximum [V]",
    ],
    "dropouts": [
        "Dropout", "Interval", "Scope date", "Scope time", "Relative [s]", "Duration [us]",
        "Duration is", "Openings", "Minimum [V]", "Note",
    ],
    "intervals": [
        "Interval", "Started (PC)", "Stopped (PC)", "Acquisitions", "Readout dead time [s]", "Memory full",
    ],
}


@dataclass
class ContinuityLog:
    """Three CSV files, flushed after every row: acquisitions, dropouts, intervals."""

    folder: Path
    name: str
    _handles: dict[str, TextIO] = field(default_factory=dict, repr=False)
    _writers: dict[str, Any] = field(default_factory=dict, repr=False)

    def path(self, kind: str) -> Path:
        return self.folder / f"{self.name} {kind}.csv"

    @property
    def acquisitions_path(self) -> Path:
        return self.path("acquisitions")

    @property
    def dropouts_path(self) -> Path:
        return self.path("dropouts")

    @property
    def intervals_path(self) -> Path:
        return self.path("intervals")

    def open(self, header: dict[str, str]) -> "ContinuityLog":
        """Create the files (appending if they exist) and write the header lines."""
        self.folder.mkdir(parents=True, exist_ok=True)
        for kind, columns in _COLUMNS.items():
            path = self.path(kind)
            new = not path.exists()
            handle = path.open("a", newline="", encoding="utf-8")
            self._handles[kind] = handle
            self._writers[kind] = csv.writer(handle)
            if new:
                for key, value in header.items():
                    handle.write(f"# {key}: {value}\n")
                self._writers[kind].writerow(columns)
            handle.flush()
        return self

    def write_acquisition(self, a: Acquisition) -> None:
        r = a.analysis
        crossings = " ".join(f"{'F' if falling else 'R'}{t * 1e6:+.3f}" for t, falling in r.crossings)
        self._row("acquisitions", [
            a.interval, a.number, a.scope_date, a.scope_time, f"{a.relative:.9f}", a.trigger,
            "open" if r.starts_open else "closed", "open" if r.ends_open else "closed", crossings,
            f"{r.minimum:.4f}", f"{r.maximum:.4f}",
        ])

    def write_dropout(self, d: DropoutRecord) -> None:
        self._row("dropouts", [
            d.number, d.interval, d.scope_date, d.scope_time, f"{d.relative:.9f}", f"{d.duration * 1e6:.3f}",
            d.certainty, d.openings, "" if np.isnan(d.minimum) else f"{d.minimum:.4f}", d.note,
        ])

    def write_interval(self, record: IntervalRecord) -> None:
        self._row("intervals", [
            record.interval, record.started.isoformat(timespec="milliseconds"),
            record.stopped.isoformat(timespec="milliseconds"), record.acquisitions, f"{record.dead_time:.3f}",
            "yes" if record.memory_full else "no",
        ])

    def _row(self, kind: str, values: list[Any]) -> None:
        assert kind in self._writers, "open() first"
        self._writers[kind].writerow(values)
        self._handles[kind].flush()

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._writers.clear()


# -- the scope -----------------------------------------------------------------


def configure(scope: RTO64, settings: ContinuitySettings) -> int:
    """Set the scope up for the monitor: channel, timebase, either-edge trigger, fast segmentation.

    Returns the number of acquisitions the scope will actually hold per
    interval: `settings.segments`, or less if the instrument clipped it to
    its memory at this record length. Pass it to :func:`monitor`.
    """
    ch = scope.channel(settings.channel)
    scope.system.set_display_update(False)
    ch.enable(True)
    ch.set_coupling("DC")                        # the 50 Ω input: the node collapses in ns when the contact opens
    ch.set_bandwidth_limit("FULL")
    ch.set_scale(settings.scale)
    ch.set_offset(settings.closed_level / 2)     # centre the 0 V – closed_level swing on screen
    ch.set_arithmetics("OFF")                    # no averaging or envelope across acquisitions
    ch.set_decimation("PEAK_DETECT")             # min and max per sample interval: no crossing escapes the record
    scope.timebase.set_range(settings.window)
    scope.timebase.set_reference(20)
    scope.timebase.set_position(Q(0, "s"))
    scope.acquisition.set_points_mode("RESOLUTION")
    scope.acquisition.set_sample_rate(settings.sample_rate)
    scope.acquisition.set_count(settings.segments)
    scope.acquisition.set_fast_segmentation(True, settings.segments)
    scope.trigger.edge(f"CH{settings.channel}", settings.threshold, slope="EITHER", mode="NORMAL")
    scope.trigger.set_holdoff("OFF")             # the record after each trigger already covers a bounce
    scope.history.enable(settings.channel, True)
    return min(settings.segments, scope.acquisition.get_max_segments())


def read_interval(
    scope: RTO64, settings: ContinuitySettings, interval: int, capacity: Optional[int] = None
) -> tuple[list[Acquisition], bool]:
    """Read every acquisition the stopped scope holds, oldest first: ``(acquisitions, memory was full)``.

    `capacity` is what :func:`configure` returned; without it the memory is
    taken as full at `settings.segments` acquisitions.
    """
    count = scope.history.available()
    memory_full = count >= (capacity or settings.segments)
    acquisitions: list[Acquisition] = []
    ch = settings.channel
    for number, index in enumerate(range(-(count - 1), 1) if count else [], start=1):
        scope.history.select(ch, index)
        stamp = scope.history.timestamp(ch)
        t, v = scope.waveform.get_data(ch)
        x = np.asarray(t.to("s").magnitude, dtype=np.float64)
        acquisitions.append(Acquisition(
            interval, number, stamp.date, stamp.time, stamp.relative,
            stamp.relative + float(x[0]), stamp.relative + float(x[-1]),
            analyse_record(t, v, settings.threshold),
        ))
    return acquisitions, memory_full


def monitor(
    scope: RTO64,
    settings: ContinuitySettings,
    log: ContinuityLog,
    stop: Callable[[], bool] = lambda: False,
    intervals: Optional[int] = None,
    clock: Callable[[], datetime] = datetime.now,
    sleep: Callable[[float], None] = _time.sleep,
    report: Callable[[str], None] = print,
    capacity: Optional[int] = None,
) -> int:
    """Run the monitor until `stop` returns true (or `intervals` are done); returns the dropout total.

    `capacity` is what :func:`configure` returned: the acquisitions the scope
    holds per interval, used to recognise a run that stopped on full memory.

    Every interval: single run, one forced acquisition (the contact state at
    the start), wait, stop, read the history, write every acquisition, every
    dropout that became final and the interval record, run again. During the
    wait the scope's acquisition count is polled once a second, and the
    interval ends early once the memory is `readout_when_full` full, so a
    burst of edges rolls into a new interval instead of stopping the scope.
    A keyboard interrupt (Ctrl-C) ends the wait early too: the current
    interval is still stopped, read out and written before the function
    returns, so nothing captured is lost.
    """
    tracker = DropoutTracker(settings)
    number = 0
    seconds = float(settings.interval.to("s").magnitude)
    nearly_full = int(np.ceil(settings.readout_when_full * (capacity or settings.segments)))
    previous_stop: Optional[datetime] = None
    interrupted = False
    while not interrupted and not stop() and (intervals is None or number < intervals):
        number += 1
        scope.run_single(wait_for_completion=False)
        started = clock()
        dead_time = 0.0 if previous_stop is None else (started - previous_stop).total_seconds()
        waited = 0.0
        try:
            sleep(_ARM_DELAY)
            scope.trigger.force()
            while waited < seconds and not stop():
                step = min(1.0, seconds - waited)
                sleep(step)
                waited += step
                held = scope.history.available()
                if held >= nearly_full:
                    report(f"interval {number}: the scope holds {held} acquisitions, reading out early")
                    break
        except KeyboardInterrupt:
            interrupted = True
            report("interrupted: reading out the current interval ...")
        scope.stop()
        stopped = clock()
        acquisitions, memory_full = read_interval(scope, settings, number, capacity)
        for a in acquisitions:
            log.write_acquisition(a)
        dropouts = tracker.feed(number, acquisitions, memory_full)
        for d in dropouts:
            log.write_dropout(d)
        log.write_interval(IntervalRecord(number, started, stopped, len(acquisitions), dead_time, memory_full))
        previous_stop = stopped
        note = "  MEMORY FULL: the scope stopped early, edges after that were not captured" if memory_full else ""
        state = "; contact open at the end" if tracker.pending is not None else ""
        report(
            f"interval {number}: {len(acquisitions)} acquisition(s), {len(dropouts)} dropout(s) final, "
            f"{tracker.count} in total{state}{note}"
        )
    for d in tracker.finish():
        log.write_dropout(d)
    return tracker.count
