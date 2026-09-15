"""Shared command-line plumbing for the run_* scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from a checkout without installing

from rflab import duts  # noqa: E402
from rflab.analysis import PLOTTERS, SUMMARIZERS  # noqa: E402
from rflab.bench import BenchLike, bench  # noqa: E402
from rflab.dut import DUT, Channel  # noqa: E402
from rflab.measurements._base import Result  # noqa: E402
from rflab.store import ResultStore  # noqa: E402


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dut", default="amplifier_x", help="DUT module in rflab.duts (default: amplifier_x)")
    p.add_argument("--channel", type=int, nargs="+", default=None, help="channel number(s); default: all")
    p.add_argument("--root", default="results", help="results folder (default: results)")
    p.add_argument("--simulate", action="store_true", help="use the simulated bench instead of the lab")
    p.add_argument("--plot", action="store_true", help="also analyse and save a figure next to the data")
    p.add_argument("--show", action="store_true", help="show the figure interactively (implies --plot)")
    return p


def run(
    description: str,
    measure: Callable[[BenchLike, DUT, Channel], Result],
) -> None:
    """Parse arguments, measure every requested channel, save, optionally analyse."""
    args = parser(description).parse_args()
    dut = duts.load(args.dut)
    channels = dut.channels if args.channel is None else [dut.channel(n) for n in args.channel]
    store = ResultStore(args.root)

    if args.simulate:
        from rflab.simulation import simulated_bench

        b: BenchLike = simulated_bench()
    else:
        b = bench()

    for channel in channels:
        print(f"{dut.label} {channel.label}: measuring ...", flush=True)
        result = measure(b, dut, channel)
        path = store.save(result)
        print(f"  saved {path}")
        if args.plot or args.show:
            report(result, store, show=args.show)


def report(result: Result, store: ResultStore, show: bool = False) -> None:
    """Print the summary and save the figure for a result."""
    summary: Any = SUMMARIZERS[result.measurement](result)
    print(f"  summary ({result.measurement}, DUT plane):")
    items = summary if isinstance(summary, list) else [summary]
    for item in items:
        print(f"    {item}")
    folder, filename = store.figure_location(result)
    PLOTTERS[result.measurement](result, show=show, save_folder=folder, filename=filename)
    print(f"  figure {Path(folder) / (filename + '.png')}")
