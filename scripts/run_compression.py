"""Measure gain compression: ``python scripts/run_compression.py --channel 1 [--plot]``."""

from __future__ import annotations

from _cli import run
from rflab.measurements import compression

if __name__ == "__main__":
    run("Gain compression (P1dB) on the spectrum analyzer.", compression.measure)
