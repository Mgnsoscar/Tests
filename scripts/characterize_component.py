"""Characterize a path component on the VNA and add it to the component library data.

    python scripts/characterize_component.py "SMA cable A" --start 100MHz --stop 12GHz --points 401

Measures S21 of the component connected between VNA ports 1 and 2 and writes
``rflab/components/data/<slug>_<date>.csv`` with ``Frequency [MHz]`` and
``Loss [dB]`` columns (loss = −|S21| in dB). Then add or update the entry in
``rflab/components/__init__.py`` to point at the new file — keep the old file
and its date, so results measured with the old characterization still resolve.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labkit.io.csv import Column, Value, write  # noqa: E402
from labkit.units import quantity as Q  # noqa: E402

from rflab.bench import bench  # noqa: E402
from rflab.components import DATA  # noqa: E402


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="component name, e.g. 'SMA cable A'")
    p.add_argument("--start", default="100MHz")
    p.add_argument("--stop", default="12GHz")
    p.add_argument("--points", type=int, default=401)
    p.add_argument("--power", default="-10dBm")
    p.add_argument("--if-bandwidth", default="1kHz")
    p.add_argument("--simulate", action="store_true", help="use the simulated bench")
    args = p.parse_args()

    if args.simulate:
        from rflab.simulation import simulated_bench

        vna = simulated_bench().vna
    else:
        vna = bench().vna

    vna.trace.delete_all()
    vna.frequency.set_start(Q(args.start))
    vna.frequency.set_stop(Q(args.stop))
    vna.sweep.set_points(args.points)
    vna.bandwidth.set_if_bandwidth(Q(args.if_bandwidth))
    vna.power.set_power(Q(args.power))
    vna.trace.create("Trc1", "S21")
    vna.display.set_window_state(1, True)
    vna.display.feed_trace(1, 1, "Trc1")
    vna.trace.select("Trc1")
    frequency, s21 = vna.measure()
    vna.power.set_output(False)
    loss = Q(-20 * np.log10(np.abs(s21)), "dB")

    today = date.today()
    filename = f"{slug(args.name)}_{today.isoformat()}"
    path = write(
        Value("Component", args.name, unit=None),
        Value("Characterized", today.isoformat(), unit=None),
        Value("Instrument", vna.get_id().strip(), unit=None),
        Column("Frequency", frequency.to("MHz")),
        Column("Loss", loss),
        filename=filename,
        folder=str(DATA),
    )
    print(f"wrote {path}")
    print(
        f"now register it in rflab/components/__init__.py:\n"
        f'    tabulated("{args.name}", "{filename}.csv", date({today.year}, {today.month}, {today.day}))'
    )


if __name__ == "__main__":
    main()
