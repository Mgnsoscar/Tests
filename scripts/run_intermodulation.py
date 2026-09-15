"""Measure two-tone IMD: ``python scripts/run_intermodulation.py --channel 1 [--plot]``."""

from __future__ import annotations

from _cli import run
from rflab.measurements import intermodulation

if __name__ == "__main__":
    run("Two-tone intermodulation (OIP3) on the spectrum analyzer.", intermodulation.measure)
