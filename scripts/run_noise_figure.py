"""Noise figure and gain with the FSV K30 application.

Calibrate the K30 application with the noise source first. Edit the block
below, set the DUT by hand to match, then:

    python scripts/run_noise_figure.py [--plot] [--simulate]
"""

from __future__ import annotations

from _cli import run
from labkit.units import quantity as Q
from rflab.measurements.noise_figure import NoiseFigureSettings, measure

# ── Edit before running ──────────────────────────────────────────────────────
DUT = "amplifier_x"                 # module in rflab/duts/
CHANNELS = [1]                      # channel numbers, or None for all channels
DUT_ATTENUATION = Q(0, "dB")        # the DUT's own attenuation setting (set it by hand)
DUT_BYPASS = False                  # the DUT's bypass switch (set it by hand)
SETTINGS = NoiseFigureSettings(
    points=51,
    enr_db=15.2,                    # noise source ENR, dB
    second_stage_correction=True,
)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        __doc__, measure,
        dut=DUT, channels=CHANNELS, settings=SETTINGS, state={"attenuation": DUT_ATTENUATION, "bypass": DUT_BYPASS},
    )
