"""Watch a DUT's electrical contact on the oscilloscope for the length of an environmental test.

Circuit: lab supply 2 V (current limit 50 mA) -> 50 Ω resistor soldered to the
antenna body -> the contact under test -> coax -> scope channel, 50 Ω input.
Supply negative to the DUT ground. Closed contact = 1 V at the scope, open = 0 V.

Edit the block below, then:

    python scripts/monitor_continuity.py            # runs until Ctrl-C
    python scripts/monitor_continuity.py --simulate # three short intervals on the simulated scope

The scope triggers on both edges; every acquisition is read out at the end
of its interval and its record analysed for the crossings it holds. Three
files are written, flushed after every row:

    ... acquisitions.csv   one row per trigger: timestamp, edge, crossings, min/max (the raw evidence)
    ... dropouts.csv       one row per dropout: start on the scope clock, duration, how certain it is
    ... intervals.csv      each interval's start/stop on the PC clock, dead time, memory-full flag

Before the chamber closes: run it, pull the connector once by hand, tap the
cable, and check the dropouts file shows those with sensible times.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _cli import _parser, make_bench, results_root  # noqa: E402
from labkit.units import quantity as Q  # noqa: E402

from rflab.monitoring.continuity import ContinuityLog, ContinuitySettings, configure, monitor  # noqa: E402

# ── Edit before running ──────────────────────────────────────────────────────
DUT = "Antenna X SN001"              # goes into the folder and file names
TEST = "vibration"                   # goes into the file names: which environmental test this is
SETTINGS = ContinuitySettings(
    channel=1,                       # scope channel the coax is on
    closed_level=Q(1.0, "V"),        # 2 V through 50 Ω + the 50 Ω input
    threshold=Q(0.5, "V"),           # trigger level; below it the contact counts as open
    window=Q(50, "us"),              # record kept around each edge, 20 % before it
    sample_rate=Q(50, "MHz"),        # 20 ns per point in that record (peak detect: nothing shorter is missed)
    segments=10_000,                 # acquisitions the scope can hold per interval before it stops early
    interval=Q(60, "s"),             # how often the acquisitions are read out and saved
    merge_within=Q(10, "us"),        # openings closer together than this are one bouncing dropout
)
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = _parser(__doc__)
    args = p.parse_args()
    b = make_bench(args)
    scope = b.scope
    settings = SETTINGS
    intervals = None
    if args.simulate:
        settings = ContinuitySettings(**{**SETTINGS.__dict__, "interval": Q(1, "s")})
        intervals = 3

    started = datetime.now()
    folder = Path(results_root(args)) / f"(B) {DUT}" / "(B) Continuity"
    name = f"{started.date().isoformat()} (B) Continuity {TEST}"
    log = ContinuityLog(folder, name)

    configure(scope, settings)
    scope.check_errors("Oscilloscope after configuration")
    scope_clock = scope.system.get_datetime()
    offset = (scope_clock - datetime.now()).total_seconds()
    log.open({
        "DUT": DUT,
        "Test": TEST,
        "Started (PC)": started.isoformat(timespec="seconds"),
        "Scope clock": scope_clock.isoformat(timespec="seconds"),
        "Scope clock minus PC clock [s]": f"{offset:.1f}",
        "Instrument": scope.get_id(),
        "Channel": str(settings.channel),
        "Trigger": f"either edge through {settings.threshold:~}; contact open below that level",
        "Record": f"{settings.window:~} at {settings.sample_rate:~}, peak detect, {settings.segments} segments",
        "Interval": f"{settings.interval:~}",
        "Bounce": f"openings closer than {settings.merge_within:~} are one dropout",
    })
    print(f"logging to {log.dropouts_path}")
    print(f"scope clock is {offset:+.1f} s from the PC clock; timestamps in the files are scope time")
    print("monitoring — Ctrl-C stops after reading out the current interval")
    try:
        total = monitor(scope, settings, log, intervals=intervals)
    finally:
        log.close()
        scope.system.set_display_update(True)
    print(f"done: {total} dropout(s) written to {log.dropouts_path}")


if __name__ == "__main__":
    main()
