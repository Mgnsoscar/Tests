"""Harmonic levels on the spectrum analyzer.

Edit the block below, set the DUT by hand to match, then:

    python scripts/run_harmonics.py [--plot] [--simulate]
"""

from __future__ import annotations

from _cli import run
from labkit.units import quantity as Q
from rflab.measurements.harmonics import HarmonicsSettings, measure

# ── Edit before running ──────────────────────────────────────────────────────
DUT = "amplifier_x"                 # module in rflab/duts/
CHANNELS = [1]                      # channel numbers, or None for all channels
DUT_ATTENUATION = Q(0, "dB")        # the DUT's own attenuation setting (set it by hand)
DUT_BYPASS = False                  # the DUT's bypass switch (set it by hand)
SETTINGS = HarmonicsSettings(
    level=Q(-10, "dBm"),            # generator level for the fundamental
    harmonics=5,                    # orders measured, including the fundamental
    n_frequencies=1,                # 1 = band centre only
)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        __doc__, measure,
        dut=DUT, channels=CHANNELS, settings=SETTINGS, state={"attenuation": DUT_ATTENUATION, "bypass": DUT_BYPASS},
    )
