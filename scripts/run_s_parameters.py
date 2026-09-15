"""Measure S11 / S22: ``python scripts/run_s_parameters.py --channel 1 [--plot]``."""

from __future__ import annotations

from _cli import run
from rflab.measurements import s_parameters

if __name__ == "__main__":
    run("Reflection (S11, S22) on the network analyzer.", s_parameters.measure)
