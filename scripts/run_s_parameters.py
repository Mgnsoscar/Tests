"""Reflection (S11, S22) on the network analyzer.

Edit the block below, set the DUT by hand to match, then:

    python scripts/run_s_parameters.py [--plot] [--simulate]
"""

from __future__ import annotations

from _cli import run
from labkit.units import quantity as Q
from rflab.measurements.s_parameters import SParameterSettings, measure

# ── Edit before running ──────────────────────────────────────────────────────
DUT = "amplifier_x"                 # module in rflab/duts/
CHANNELS = None                     # channel numbers, or None for all channels
DUT_ATTENUATION = Q(0, "dB")        # the DUT's own attenuation setting (set it by hand)
SETTINGS = SParameterSettings(
    points=201,
    if_bandwidth=Q(1, "kHz"),
    power=Q(-10, "dBm"),
    average_count=1,
    parameters=("S11", "S22"),
)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        __doc__, measure,
        dut=DUT, channels=CHANNELS, settings=SETTINGS, state={"attenuation": DUT_ATTENUATION},
    )
