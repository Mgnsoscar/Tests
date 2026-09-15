"""Characterize a path component on the VNA and add it to the component library data.

Connect the component between VNA ports 1 and 2 — through whatever adapters
it needs — edit the block below to say what it is and which adapters are in
the fixture, then:

    python scripts/characterize_component.py [--simulate]

The fixture's loss is de-embedded from the measurement. The script writes
``rflab/components/data/<slug>_<date>.csv`` and prints the library line to add
to ``rflab/components/__init__.py``. Keep old files and entries: results
record components as ``"name (date)"`` and resolve to the characterization
that was current when they were measured.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labkit.signal_path import SignalPath  # noqa: E402
from labkit.units import quantity as Q  # noqa: E402

from rflab.bench import bench  # noqa: E402
from rflab.components import SMA_FF_ADAPTER_A, SMA_FF_ADAPTER_B  # noqa: E402
from rflab.components.characterize import CharacterizationSettings, characterize, save  # noqa: E402

# ── Edit before running ──────────────────────────────────────────────────────
NAME = "SMA cable A"                                  # the component being characterized
FIXTURE = SignalPath(SMA_FF_ADAPTER_A, SMA_FF_ADAPTER_B)  # adapters etc. in the measurement, not part of NAME
SETTINGS = CharacterizationSettings(
    start=Q(100, "MHz"),
    stop=Q(12, "GHz"),
    points=401,
    power=Q(-10, "dBm"),
    if_bandwidth=Q(1, "kHz"),
)
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simulate", action="store_true", help="use the simulated bench")
    args = p.parse_args()

    if args.simulate:
        from rflab.simulation import simulated_bench

        vna = simulated_bench().vna
    else:
        vna = bench().vna

    result = characterize(vna, NAME, SETTINGS, FIXTURE)
    path = save(result)
    print(f"wrote {path}")
    print(f"fixture de-embedded: {FIXTURE.describe()}")
    print(
        "now register it in rflab/components/__init__.py:\n"
        f'    tabulated("{NAME}", "{path.name}", date({result.when.year}, {result.when.month}, {result.when.day}))'
    )


if __name__ == "__main__":
    main()
