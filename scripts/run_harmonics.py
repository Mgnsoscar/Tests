"""Measure harmonics: ``python scripts/run_harmonics.py --channel 1 [--plot]``."""

from __future__ import annotations

from _cli import run
from rflab.measurements import harmonics

if __name__ == "__main__":
    run("Harmonic levels on the spectrum analyzer.", harmonics.measure)
