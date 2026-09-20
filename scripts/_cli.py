"""Shared plumbing for the run_* scripts.

Each run script states *what to measure* in an "edit before running" block at
its top (DUT, channels, the DUT's own attenuation and bypass settings,
measurement settings) and hands it to :func:`run`, which handles the bench,
the results store and the optional analysis. The only command-line flags are
about *how* to run: ``--simulate``, ``--plot``, ``--show``, ``--root``.

``--simulate`` writes to ``results-simulated/`` instead of ``results/`` (unless
``--root`` says otherwise), so simulated data never mixes with the lab's.
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
from rflab.dut import Channel  # noqa: E402
from rflab.measurements._base import Result, format_state_value  # noqa: E402
from rflab.store import ResultStore  # noqa: E402

SIMULATED_ROOT = "results-simulated"


def _parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--simulate", action="store_true", help="use the simulated bench instead of the lab")
    p.add_argument("--plot", action="store_true", help="also analyse and save a figure next to the data")
    p.add_argument("--show", action="store_true", help="show the figure interactively (implies --plot)")
    p.add_argument("--root", default=None, help=f"results folder (default: results, or {SIMULATED_ROOT} with --simulate)")
    return p


def results_root(args: argparse.Namespace) -> str:
    """The results folder for these arguments."""
    if args.root is not None:
        return str(args.root)
    return SIMULATED_ROOT if getattr(args, "simulate", False) else "results"


def make_bench(args: argparse.Namespace) -> BenchLike:
    """The lab bench, or the simulated one with ``--simulate``."""
    if args.simulate:
        from rflab.simulation import simulated_bench

        return simulated_bench()
    return bench()


def set_dut_state(b: BenchLike, channel: Channel, state: Mapping[str, Any]) -> None:
    """On the simulated bench, put the modelled DUT into this channel and state (no-op in the lab)."""
    setter = getattr(b, "set_dut", None)
    if setter is not None:
        setter(channel, state)


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
        The DUT's own settings during the run (e.g. ``{"attenuation": 6 dB,
        "bypass": False}``); recorded in the result header and file name.
    """
    args = _parser(description).parse_args()
    device = duts.load(dut)
    selected = device.channels if channels is None else [device.channel(n) for n in channels]
    store = ResultStore(results_root(args))
    b = make_bench(args)

    state_text = describe_state(state)
    for channel in selected:
        set_dut_state(b, channel, state)
        print(f"{device.label} {channel.label}" + (f" ({state_text})" if state_text else "") + ": measuring ...", flush=True)
        result = measure(b, device, channel, settings, state)
        path = store.save(result)
        print(f"  saved {path}")
        if args.plot or args.show:
            report(result, store, show=args.show)


def describe_state(state: Mapping[str, Any]) -> str:
    """``"attenuation 6 dB, bypass off"`` for the console."""
    return ", ".join(f"{k} {format_state_value(v)}" for k, v in state.items())


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
