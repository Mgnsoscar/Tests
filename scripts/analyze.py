"""Post-process saved results without touching an instrument.

    python scripts/analyze.py results/"(B) Amplifier X v1.0"/"(B) Compression"/*.csv
    python scripts/analyze.py --all --measurement Compression        # everything of one kind
    python scripts/analyze.py <file.csv> --show                        # interactive figure

Each file is loaded back into its result object, moved to the DUT reference
plane with the recorded signal paths, summarized, and plotted next to the data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _cli import report  # noqa: E402
from rflab.store import ResultStore  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="result CSV files")
    p.add_argument("--all", action="store_true", help="every result under --root")
    p.add_argument("--root", default="results")
    p.add_argument("--dut", default=None, help="filter --all by DUT label, e.g. 'Amplifier X v1.0'")
    p.add_argument("--measurement", default=None, help="filter --all by measurement, e.g. Compression")
    p.add_argument("--channel", type=int, default=None, help="filter --all by channel")
    p.add_argument("--show", action="store_true", help="show figures interactively")
    args = p.parse_args()

    store = ResultStore(args.root)
    files = [Path(f) for f in args.files]
    if args.all:
        files += store.find(dut=args.dut, measurement=args.measurement, channel=args.channel)
    if not files:
        p.error("give result files, or --all")

    for path in files:
        print(path)
        result = store.load(path)
        report(result, store, show=args.show)


if __name__ == "__main__":
    main()
