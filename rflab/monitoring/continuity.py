"""Electrical-contact monitoring on the oscilloscope during environmental tests.

The circuit
-----------
A lab supply (5 V, 50 mA limit) feeds the antenna body through a series
resistor (about 580 Ω); the current goes through the contact under test and
down the antenna's coax into scope channel 1. A second coax, cut open at one
end, connects the antenna body itself to scope channel 2. Measured with
everything connected:

===================================  ==========  ==========
situation                            channel 1   channel 2
===================================  ==========  ==========
contact closed, current flowing      2.2 V       1.1 V
antenna's own wiring open            0 V         2.2 V
supply cable to the body open        0 V         0 V
===================================  ==========  ==========

Contact open = channel 1 below the threshold (1.1 V, half the closed level).
When the contact opens the coax side collapses within nanoseconds.

Which side opened
-----------------
A dropout on the coax can be the antenna's own wiring or the supply cable to
the body. Channel 2 tells them apart: when the antenna opens the body rises
to 2.2 V (the supply is there, the antenna no longer draws current); when the
supply cable breaks it falls to 0 V. Every dropout is attributed from that
channel's records: *antenna* when the body stayed above `supply_threshold`,
*supply* when it fell below it.

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
    "SupplyAnalysis",
    "analyse_supply",
    "Acquisition",
    "Crossing",
    "DropoutRecord",
    "IntervalRecord",
    "ContinuityLog",
    "DropoutTracker",
    "ScopeSetup",
    "Levels",
    "configure",
    "read_levels",
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
    #: Voltage on the coax channel with the contact closed (measured: 2.2 V).
    closed_level: Quantity = Q(2.2, "V")
    #: The trigger level, and the level below which the contact counts as open.
    threshold: Quantity = Q(1.1, "V")
    #: Hysteresis for the record analysis: the contact counts as open once the
    #: voltage is below threshold − hysteresis and as closed again only above
    #: threshold + hysteresis, so noise on a signal near the threshold is not
    #: a train of crossings.
    hysteresis: Quantity = Q(300, "mV")
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
    #: Vertical scale, so 0 V and the closed level are both well inside the screen.
    scale: Quantity = Q(500, "mV")
    #: A second scope channel wired to the antenna body itself (the supply side of the
    #: contact), on the 1 MΩ input so it does not load the node, or ``None``. With current
    #: flowing it sits at `supply_normal`; when the antenna's own wiring opens it rises to
    #: `supply_open` (nothing draws current any more); when the supply cable breaks it drops
    #: to 0 V. So it says which side of the contact a dropout is on.
    supply_channel: Optional[int] = None
    #: Below this on the supply channel during a dropout, the supply was gone: the cause is
    #: the supply cable, not the antenna.
    supply_threshold: Quantity = Q(0.5, "V")
    #: The supply channel's level with current flowing, and with the antenna open (measured).
    supply_normal: Quantity = Q(1.1, "V")
    supply_open: Quantity = Q(2.2, "V")
    #: The supply channel's scale, so 0 V and `supply_open` are both on screen.
    supply_scale: Quantity = Q(500, "mV")

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

    #: Contact state at the start and the end of the record, decided by the level of its
    #: head and tail (the first and last few percent of the samples), not by the nearest crossing.
    starts_open: bool
    ends_open: bool
    #: Every threshold crossing: ``(time in the record in seconds, falling)``, consistent with the
    #: start and end states (noise crossings that contradict them are dropped).
    crossings: tuple[tuple[float, bool], ...]
    #: The edge at the trigger point: ``"falling"``, ``"rising"`` or ``"none"`` (a forced
    #: acquisition, or a glitch shorter than the guard around the trigger point).
    trigger_edge: str
    minimum: float
    maximum: float
    #: The head or tail level sat inside the hysteresis band, so that state is the last
    #: decisive sample's, not the level's: treat it as unsure.
    start_uncertain: bool = False
    end_uncertain: bool = False


#: The share of a record's samples that make its head and its tail.
_EDGE_SHARE = 0.05


def analyse_record(
    time: Quantity, voltage: Quantity, threshold: Quantity, hysteresis: Quantity = Q(0, "V")
) -> RecordAnalysis:
    """Find the threshold crossings of a record; peak-detect records (min/max per sample) are supported.

    A sample interval counts as open when its minimum is below the threshold,
    so a crossing is placed at the first sample interval of the new state.
    With a hysteresis the contact opens below ``threshold − hysteresis`` and
    closes again only once the interval's minimum is above ``threshold +
    hysteresis``; in between, the state stays what it was.

    The state at the start and at the end of the record is decided by the
    mean level of its head and tail, not by the first or last crossing: a
    signal ramping through the threshold with noise on it can end its
    chatter on the wrong edge, but its tail is still clearly on one side.
    Crossings at either end that contradict those states are dropped. A
    head or tail inside the hysteresis band decides nothing; the sample
    states are kept and the record is marked uncertain there.
    """
    t = np.asarray(time.to("s").magnitude, dtype=np.float64)
    v = np.asarray(voltage.to("V").magnitude, dtype=np.float64)
    low, high = (v.min(axis=1), v.max(axis=1)) if v.ndim == 2 else (v, v)
    thr = float(threshold.to("V").magnitude)
    h = float(hysteresis.to("V").magnitude)
    is_open = _open_states(low, thr, h)
    changes = np.flatnonzero(is_open[1:] != is_open[:-1]) + 1
    crossings = [(float(t[i]), bool(is_open[i])) for i in changes]

    edge_samples = max(4, int(len(t) * _EDGE_SHARE))
    starts_open, start_uncertain = _level_state(low[:edge_samples], thr, h, bool(is_open[0]))
    ends_open, end_uncertain = _level_state(low[-edge_samples:], thr, h, bool(is_open[-1]))
    while crossings and crossings[0][1] != (not starts_open):    # a first crossing must leave the start state
        crossings.pop(0)
    while crossings and crossings[-1][1] != ends_open:           # the last crossing must lead into the end state
        crossings.pop()

    at_trigger = int(np.argmin(np.abs(t)))
    guard = 3
    before = bool(is_open[max(at_trigger - guard, 0)])
    after = bool(is_open[min(at_trigger + guard, len(t) - 1)])
    edge = "falling" if (not before and after) else "rising" if (before and not after) else "none"
    return RecordAnalysis(
        starts_open, ends_open, tuple(crossings), edge, float(low.min()), float(high.max()),
        start_uncertain, end_uncertain,
    )


def _level_state(samples: np.ndarray, threshold: float, hysteresis: float, fallback: bool) -> tuple[bool, bool]:
    """``(open, uncertain)`` from the mean of the per-sample minima in a record's head or tail."""
    level = float(samples.mean())
    if level < threshold - hysteresis:
        return True, False
    if level > threshold + hysteresis:
        return False, False
    return fallback, True


def _open_states(low: np.ndarray, threshold: float, hysteresis: float) -> np.ndarray:
    """Open/closed per sample from the per-sample minimum, with hysteresis around the threshold."""
    if hysteresis <= 0:
        return np.asarray(low < threshold)
    decisive = np.where(low < threshold - hysteresis, 1, np.where(low > threshold + hysteresis, 0, -1))
    if decisive[0] == -1:
        decisive[0] = 1 if low[0] < threshold else 0        # the first sample decides by the threshold alone
    last_decisive = np.maximum.accumulate(np.where(decisive >= 0, np.arange(len(low)), 0))
    return np.asarray(decisive[last_decisive] == 1)


@dataclass(frozen=True)
class SupplyAnalysis:
    """What the supply channel's record of the same acquisition shows: its extreme voltages."""

    minimum: float
    maximum: float


def analyse_supply(voltage: Quantity) -> SupplyAnalysis:
    v = np.asarray(voltage.to("V").magnitude, dtype=np.float64)
    return SupplyAnalysis(float(v.min()), float(v.max()))


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
    #: The supply channel's record, when one is monitored.
    supply: Optional[SupplyAnalysis] = None

    @property
    def trigger(self) -> str:
        """The edge seen at the trigger point; ``forced`` for the state snapshot at the interval start (no
        crossing near the trigger point), ``glitch`` for a trigger whose record shows only a crossing too
        short to change the state."""
        if self.analysis.trigger_edge != "none":
            return self.analysis.trigger_edge
        near = [t for t, _ in self.analysis.crossings if abs(t) < 1e-6]
        return "glitch" if near else "forced"


# -- the timeline of one interval ----------------------------------------------


@dataclass(frozen=True)
class Crossing:
    """One threshold crossing on the interval's timeline."""

    #: Seconds relative to the newest acquisition of the interval.
    time: float
    falling: bool
    #: Not seen in a record: deduced from the level at the start of the next record.
    inferred: bool = False
    #: For an inferred crossing, the seconds between the two records it happened between.
    window: float = 0.0


def interval_crossings(acquisitions: list[Acquisition]) -> list[Crossing]:
    """Stitch the crossings of an interval's records into one timeline, oldest first.

    Where two records overlap, a crossing is taken once. Where they do not,
    the level at the end of one and at the start of the next must agree: the
    scope was armed in between and any crossing would have triggered — unless
    it fell into the scope's blind time right after the record, or the
    voltage drifted across without an edge sharp enough to trigger. A
    disagreement is such a crossing; it is placed at the end of the earlier
    record, marked inferred, with the gap between the records as its window.
    """
    timeline: list[Crossing] = []
    covered_until: Optional[float] = None
    open_at_end = False
    for a in acquisitions:
        if covered_until is not None and a.record_start > covered_until and a.analysis.starts_open != open_at_end:
            timeline.append(Crossing(
                covered_until, falling=a.analysis.starts_open, inferred=True, window=a.record_start - covered_until,
            ))
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
    #: ``antenna`` (the supply stayed on the antenna body: the antenna's own wiring opened),
    #: ``supply`` (the supply was gone too: the supply cable), ``unknown`` (no supply record
    #: overlapped the dropout) or ``""`` when no supply channel is monitored.
    cause: str = ""
    #: The lowest supply-channel voltage in the records overlapping the dropout.
    supply_minimum: float = float("nan")


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
    start_window: float = 0.0
    #: The opening was never seen: the contact was already open when the interval's first record began.
    start_unknown: bool = False
    openings: int = 1
    end: Optional[float] = None
    end_interval: Optional[int] = None
    end_inferred: bool = False
    end_window: float = 0.0
    #: The latest time the contact was seen open: in the start's own frame, and in any frame.
    open_until_own: float = 0.0
    open_until: float = 0.0
    open_until_interval: int = 0
    minimum: float = float("inf")
    supply_minimum: float = float("inf")
    notes: list[str] = field(default_factory=list)


class DropoutTracker:
    """Turns the intervals' crossings into dropouts; keeps one open across intervals."""

    def __init__(self, settings: ContinuitySettings) -> None:
        self.merge_within = float(settings.merge_within.to("s").magnitude)
        self.supply_monitored = settings.supply_channel is not None
        self.supply_threshold = float(settings.supply_threshold.to("V").magnitude)
        #: Dropouts opened so far, and of the final ones, how many were the antenna's and the supply cable's.
        self.count = 0
        self.antenna_count = 0
        self.supply_count = 0
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
                    pending = self._open(interval, c.time, acquisitions, inferred=c.inferred, window=c.window)
            elif is_open and pending is not None:
                is_open = False
                pending.end, pending.end_interval, pending.end_inferred = c.time, interval, c.inferred
                pending.end_window = c.window
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
    def _open(
        self, interval: int, time: float, acquisitions: list[Acquisition], inferred: bool, window: float = 0.0
    ) -> _Dropout:
        self.count += 1
        d = _Dropout(self.count, interval, time, _acquisition_at(acquisitions, time), inferred, window)
        self._seen_open(d, interval, time)
        if inferred:
            d.notes.append(_inferred_note("opened", window))
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
                if a.supply is not None:
                    d.supply_minimum = min(d.supply_minimum, a.supply.minimum)

    def _cause(self, d: _Dropout) -> str:
        if not self.supply_monitored:
            return ""
        if np.isinf(d.supply_minimum):
            return "unknown"
        if d.supply_minimum < self.supply_threshold:
            self.supply_count += 1
            return "supply"
        self.antenna_count += 1
        return "antenna"

    def _finish(self, d: _Dropout) -> DropoutRecord:
        duration, certainty = self._duration(d)
        cause = self._cause(d)
        if d.end_inferred:
            d.notes.append(_inferred_note("closed", d.end_window))
        acq = d.start_acquisition
        offset = d.start - acq.relative
        scope_time = _time_of_day(acq.scope_time, offset)
        if scope_time is None:
            scope_time = acq.scope_time
            d.notes.append(f"opening at {offset * 1e6:+.3f} us from that acquisition's timestamp")
        minimum = float("nan") if np.isinf(d.minimum) else d.minimum
        supply_minimum = float("nan") if np.isinf(d.supply_minimum) else d.supply_minimum
        return DropoutRecord(
            d.number, d.interval, acq.scope_date, scope_time, d.start, duration, certainty,
            d.openings, minimum, "; ".join(d.notes), cause, supply_minimum,
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


def _inferred_note(what: str, window: float) -> str:
    """The note for a crossing no record shows: it happened in the gap between two records."""
    if window < 1e-3:
        cause = "the scope's blind time after the record"
    else:
        cause = "a change too slow to trigger, or an edge the trigger missed"
    return (f"{what} between two records with no trigger in between, {window:.3g} s apart ({cause}); "
            f"time taken as the earlier record's end")


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
        "Supply min [V]", "Supply max [V]",
    ],
    "dropouts": [
        "Dropout", "Interval", "Scope date", "Scope time", "Relative [s]", "Duration [us]",
        "Duration is", "Openings", "Minimum [V]", "Cause", "Supply min [V]", "Note",
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
        starts = ("open" if r.starts_open else "closed") + ("?" if r.start_uncertain else "")
        ends = ("open" if r.ends_open else "closed") + ("?" if r.end_uncertain else "")
        supply = ["", ""] if a.supply is None else [f"{a.supply.minimum:.4f}", f"{a.supply.maximum:.4f}"]
        self._row("acquisitions", [
            a.interval, a.number, a.scope_date, a.scope_time, f"{a.relative:.9f}", a.trigger,
            starts, ends, crossings, f"{r.minimum:.4f}", f"{r.maximum:.4f}", *supply,
        ])

    def write_dropout(self, d: DropoutRecord) -> None:
        self._row("dropouts", [
            d.number, d.interval, d.scope_date, d.scope_time, f"{d.relative:.9f}", f"{d.duration * 1e6:.3f}",
            d.certainty, d.openings, "" if np.isnan(d.minimum) else f"{d.minimum:.4f}",
            d.cause, "" if np.isnan(d.supply_minimum) else f"{d.supply_minimum:.4f}", d.note,
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


@dataclass(frozen=True)
class ScopeSetup:
    """What the scope reports after :func:`configure`: the settings as the instrument took them."""

    requested_segments: int
    #: ``ACQ:SEGM:MAX?`` after the setup, or ``None`` when the answer was not a
    #: valid series size (the command's range starts at 2), so the capacity is unknown.
    granted_segments: Optional[int]
    #: The raw answer, for the log.
    segments_answer: str
    record_length: int
    sample_rate: Quantity

    @property
    def capacity(self) -> int:
        """Acquisitions per interval to plan with: what was granted, else what was requested."""
        return self.granted_segments if self.granted_segments is not None else self.requested_segments

    def describe(self) -> str:
        rate = self.sample_rate.to("MHz").magnitude
        record = f"record {self.record_length} points at {rate:g} MSa/s"
        if self.granted_segments is None:
            return (f"{record}; the scope answered {self.segments_answer!r} to ACQ:SEGM:MAX?, which is not a "
                    f"series size — capacity unknown, planning with the requested {self.requested_segments}")
        if self.granted_segments < self.requested_segments:
            return (f"{record}; the scope holds {self.granted_segments} acquisitions per interval "
                    f"({self.requested_segments} requested; its memory holds no more at this record length)")
        return f"{record}; the scope holds {self.granted_segments} acquisitions per interval"


def configure(scope: RTO64, settings: ContinuitySettings) -> ScopeSetup:
    """Set the scope up for the monitor: channel, timebase, either-edge trigger, fast segmentation.

    Waits for the instrument to apply the settings, then reads back the
    segment capacity, record length and sample rate; pass ``.capacity`` to
    :func:`monitor`.
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
    if settings.supply_channel is not None:
        if settings.supply_channel == settings.channel:
            raise ValueError("The supply channel must differ from the contact channel.")
        sc = scope.channel(settings.supply_channel)
        sc.enable(True)
        sc.set_coupling("DC_1M")                 # high impedance: must not load the antenna body node
        sc.set_bandwidth_limit("FULL")
        sc.set_scale(settings.supply_scale)
        sc.set_offset(settings.supply_open / 2)  # centre the 0 V – supply_open swing on screen
        sc.set_arithmetics("OFF")
        sc.set_decimation("PEAK_DETECT")
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
    if settings.supply_channel is not None:
        scope.history.enable(settings.supply_channel, True)
    scope.wait_for_instrument()                  # the settings are asynchronous: let them apply before reading back
    answer = scope.query("ACQ:SEGM:MAX?").strip()
    try:
        granted: Optional[int] = int(round(float(answer)))
    except ValueError:
        granted = None
    if granted is not None and granted < 2:
        granted = None
    return ScopeSetup(
        settings.segments,
        None if granted is None else min(settings.segments, granted),
        answer,
        scope.acquisition.get_record_length(),
        scope.acquisition.get_sample_rate(),
    )


@dataclass(frozen=True)
class Levels:
    """The channels' mean voltages right now, from one forced acquisition: the wiring check."""

    contact: float
    supply: Optional[float]

    def check(self, settings: ContinuitySettings) -> list[str]:
        """Plain-language warnings when the levels do not match the settings; empty when all is well."""
        warnings: list[str] = []
        closed = float(settings.closed_level.to("V").magnitude)
        threshold = float(settings.threshold.to("V").magnitude)
        if self.contact < threshold:
            warnings.append(
                f"channel {settings.channel} (the coax) reads {self.contact:.2f} V: the contact is open, "
                f"or the supply is off, or the trigger level {threshold:g} V does not fit this circuit"
            )
        elif abs(self.contact - closed) > 0.25 * closed:
            warnings.append(
                f"channel {settings.channel} (the coax) reads {self.contact:.2f} V, but {closed:g} V is expected "
                f"with the contact closed: check the supply voltage, the series resistor and the 50 Ω input"
            )
        if self.supply is not None and settings.supply_channel is not None:
            normal = float(settings.supply_normal.to("V").magnitude)
            supply_threshold = float(settings.supply_threshold.to("V").magnitude)
            if self.supply < supply_threshold:
                warnings.append(
                    f"channel {settings.supply_channel} (the antenna body) reads {self.supply:.2f} V: no supply on the "
                    "body, or the body wire is not connected — every dropout would be blamed on the supply cable"
                )
            elif abs(self.supply - normal) > 0.25 * normal:
                warnings.append(
                    f"channel {settings.supply_channel} (the antenna body) reads {self.supply:.2f} V, but {normal:g} V "
                    "is expected with current flowing: check the wire and that the channel is the 1 MΩ input"
                )
        return warnings


def read_levels(scope: RTO64, settings: ContinuitySettings, sleep: Callable[[float], None] = _time.sleep) -> Levels:
    """One forced acquisition now, and the mean level of each channel in it.

    Run after :func:`configure`, before the test: with the contact closed the
    coax channel must read `closed_level` and the supply channel
    `supply_normal`, or the wiring or the settings are wrong.
    """
    scope.run_single(wait_for_completion=False)
    sleep(_ARM_DELAY)
    scope.trigger.force()
    sleep(_ARM_DELAY)                            # let the forced acquisition complete; never *OPC? while a
    scope.stop()                                 # single run of thousands of acquisitions is in progress
    scope.wait_for_instrument()
    ch, sc = settings.channel, settings.supply_channel
    oldest = -(max(scope.history.available(), 1) - 1)     # the forced acquisition is the first of the run
    scope.history.enable(ch, True)
    scope.history.select(ch, oldest)
    contact = float(np.mean(scope.waveform.get_data(ch)[1].to("V").magnitude))
    supply = None
    if sc is not None:
        scope.history.enable(sc, True)
        scope.history.select(sc, oldest)
        supply = float(np.mean(scope.waveform.get_data(sc)[1].to("V").magnitude))
    return Levels(contact, supply)


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
    sc = settings.supply_channel
    if count:
        scope.history.enable(ch, True)           # a run leaves the history; enter it again for the readout
        if sc is not None:
            scope.history.enable(sc, True)
    for number, index in enumerate(range(-(count - 1), 1) if count else [], start=1):
        scope.history.select(ch, index)          # waits until the selection has taken effect
        stamp = scope.history.timestamp(ch)
        t, v = scope.waveform.get_data(ch)
        supply = None
        if sc is not None:
            scope.history.select(sc, index)      # the same acquisition on the supply channel
            supply = analyse_supply(scope.waveform.get_data(sc)[1])
        x = np.asarray(t.to("s").magnitude, dtype=np.float64)
        acquisitions.append(Acquisition(
            interval, number, stamp.date, stamp.time, stamp.relative,
            stamp.relative + float(x[0]), stamp.relative + float(x[-1]),
            analyse_record(t, v, settings.threshold, settings.hysteresis), supply,
        ))
    stamps = {(a.scope_date, a.scope_time, a.relative) for a in acquisitions}
    if len(acquisitions) > 1 and len(stamps) == 1:
        raise RuntimeError(
            f"The scope holds {count} acquisitions but every one read back carries the same timestamp "
            f"{acquisitions[0].scope_date} {acquisitions[0].scope_time}: selecting acquisitions in the "
            "history had no effect, so the records are not the stored ones. Nothing was logged for this interval."
        )
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
        scope.wait_for_instrument()              # STOP is asynchronous too: the count is right once it has taken effect
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
        causes = ""
        if tracker.supply_monitored:
            causes = f" ({tracker.antenna_count} antenna, {tracker.supply_count} supply cable)"
        report(
            f"interval {number}: {len(acquisitions)} acquisition(s), {len(dropouts)} dropout(s) final, "
            f"{tracker.count} in total{causes}{state}{note}"
        )
    for d in tracker.finish():
        log.write_dropout(d)
    return tracker.count
