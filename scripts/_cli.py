"""Shared plumbing for the run_* scripts.

Each run script states *what to measure* in an "edit before running" block at
its top (DUT, channels, the DUT's own attenuation setting, measurement
settings) and hands it to :func:`run`, which handles the bench, the results
store and the optional analysis. The only command-line flags are about *how*
to run: ``--simulate``, ``--plot``, ``--show``, ``--root``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from a checkout without installing

from rflab import duts  # noqa: E402
from rflab.analysis import PLOTTERS, SUMMARIZERS  # noqa: E402
from rflab.bench import BenchLike, bench  # noqa: E402
from rflab.measurements._base import Result  # noqa: E402
from rflab.store import ResultStore  # noqa: E402


def _parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--simulate", action="store_true", help="use the simulated bench instead of the lab")
    p.add_argument("--plot", action="store_true", help="also analyse and save a figure next to the data")
    p.add_argument("--show", action="store_true", help="show the figure interactively (implies --plot)")
    p.add_argument("--root", default="results", help="results folder (default: results)")
    return p


def run(
    description: str,
    measure: Callable[..., Result],
    *,
    dut: str,
    channels: Optional[Sequence[int]],
    settings: Any,
    state: Mapping[str, Any],
) -> None:
    """Measure every requested channel with the hard-coded configuration, save, optionally analyse.

    Parameters
    ----------
    measure:
        The measurement's ``measure(bench, dut, channel, settings, state)``.
    dut:
        Module name in ``rflab.duts``.
    channels:
        Channel numbers, or ``None`` for every channel of the DUT.
    settings:
        The measurement's settings dataclass.
    state:
        The DUT's own settings during the run (e.g. ``{"attenuation": 6 dB}``);
        recorded in the result header and file name.
    """
    args = _parser(description).parse_args()
    device = duts.load(dut)
    selected = device.channels if channels is None else [device.channel(n) for n in channels]
    store = ResultStore(args.root)

    if args.simulate:
        from rflab.simulation import simulated_bench

        b: BenchLike = simulated_bench()
    else:
        b = bench()

    state_text = ", ".join(f"{k} {v:~}" if hasattr(v, "to") else f"{k} {v}" for k, v in state.items())
    for channel in selected:
        print(f"{device.label} {channel.label}" + (f" ({state_text})" if state_text else "") + ": measuring ...", flush=True)
        result = measure(b, device, channel, settings, state)
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
