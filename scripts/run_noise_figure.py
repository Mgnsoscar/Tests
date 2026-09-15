"""Measure noise figure: ``python scripts/run_noise_figure.py --channel 1 [--plot]``.

Calibrate the K30 application with the noise source before running.
"""

from __future__ import annotations

from _cli import run
from rflab.measurements import noise_figure

if __name__ == "__main__":
    run("Noise figure and gain with the FSV K30 application.", noise_figure.measure)
