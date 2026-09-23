"""Watch a DUT's electrical contact on the oscilloscope for the length of an environmental test.

Circuit: lab supply 5 V (current limit 50 mA) -> ~580 Ω resistor -> antenna
body -> the contact under test -> the antenna's coax -> scope channel 1. A
second coax, cut open at one end, from the antenna body -> scope channel 2.
Measured: contact closed, channel 1 at 2.2 V and channel 2 at 1.1 V; antenna
open, channel 1 at 0 V and channel 2 at 2.2 V; supply cable open, both at 0 V.
Every dropout is attributed "antenna" or "supply" from channel 2.

At start the script takes one acquisition and prints both levels against
these expectations, so a wiring mistake shows before the chamber closes.

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

from rflab.monitoring.continuity import (  # noqa: E402
    ContinuityLog, ContinuitySettings, configure, monitor, read_levels,
)

# ── Edit before running ──────────────────────────────────────────────────────
DUT = "Antenna X SN001"              # goes into the folder and file names
TEST = "vibration"                   # goes into the file names: which environmental test this is
SETTINGS = ContinuitySettings(
    channel=1,                       # scope channel the coax is on
    closed_level=Q(2.2, "V"),        # measured on channel 1 with the contact closed
    threshold=Q(1.1, "V"),           # trigger level; below it the contact counts as open ...
    hysteresis=Q(300, "mV"),         # ... open below 0.8 V, closed again only above 1.4 V (noise is not a crossing)
    window=Q(50, "us"),              # record kept around each edge, 20 % before it
    sample_rate=Q(50, "MHz"),        # 20 ns per point in that record (peak detect: nothing shorter is missed)
    segments=10_000,                 # acquisitions the scope can hold per interval before it stops early
    interval=Q(60, "s"),             # how often the acquisitions are read out and saved ...
    readout_when_full=0.9,           # ... or sooner, once the scope holds this fraction of its segments
    merge_within=Q(10, "us"),        # openings closer together than this are one bouncing dropout
    supply_channel=2,                # the coax from the antenna body (1 MΩ input); None to monitor the coax alone
    supply_threshold=Q(0.5, "V"),    # below this on that channel during a dropout: the supply cable, not the antenna
    supply_normal=Q(1.1, "V"),       # measured on channel 2 with current flowing ...
    supply_open=Q(2.2, "V"),         # ... and with the antenna open (nothing draws current)
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

    setup = configure(scope, settings)
    scope.check_errors("Oscilloscope after configuration")
    capacity = setup.capacity
    levels = read_levels(scope, settings)
    scope.check_errors("Oscilloscope after the wiring check")
    level_text = f"channel {settings.channel} (coax) {levels.contact:.2f} V, expected {settings.closed_level:~} closed"
    if levels.supply is not None:
        level_text += (f"; channel {settings.supply_channel} (antenna body) {levels.supply:.2f} V, "
                       f"expected {settings.supply_normal:~} with current flowing")
    warnings = levels.check(settings)
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
        "Supply channel": (
            f"{settings.supply_channel} on the antenna body (1 MOhm): a dropout is the supply cable's when it "
            f"drops below {settings.supply_threshold:~}, else the antenna's"
            if settings.supply_channel is not None else "none"
        ),
        "Trigger": f"either edge through {settings.threshold:~}; contact open below that level, "
                   f"hysteresis {settings.hysteresis:~} in the record analysis",
        "Record": f"{settings.window:~} at {settings.sample_rate:~}, peak detect",
        "Scope": setup.describe(),
        "Interval": f"{settings.interval:~}",
        "Bounce": f"openings closer than {settings.merge_within:~} are one dropout",
        "Levels at start": level_text + ("" if not warnings else "; WARNING: " + "; ".join(warnings)),
    })
    print(f"logging to {log.dropouts_path}")
    print(f"scope clock is {offset:+.1f} s from the PC clock; timestamps in the files are scope time")
    print(setup.describe())
    if setup.granted_segments is None:
        print("  NOTE: check the scope's fast-segmentation dialog for the maximum it allows at this record length")
    print(f"levels now: {level_text}")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    print("monitoring — Ctrl-C stops after reading out the current interval")
    try:
        total = monitor(scope, settings, log, intervals=intervals, capacity=capacity)
    finally:
        log.close()
        scope.system.set_display_update(True)
    print(f"done: {total} dropout(s) written to {log.dropouts_path}")


if __name__ == "__main__":
    main()
