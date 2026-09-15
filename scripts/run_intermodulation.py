"""Two-tone intermodulation (OIP3) on the spectrum analyzer.

Edit the block below, set the DUT by hand to match, then:

    python scripts/run_intermodulation.py [--plot] [--simulate]
"""

from __future__ import annotations

from _cli import run
from labkit.units import quantity as Q
from rflab.measurements.intermodulation import IntermodulationSettings, measure

# ── Edit before running ──────────────────────────────────────────────────────
DUT = "amplifier_x"                 # module in rflab/duts/
CHANNELS = [1]                      # channel numbers, or None for all channels
DUT_ATTENUATION = Q(0, "dB")        # the DUT's own attenuation setting (set it by hand)
SETTINGS = IntermodulationSettings(
    level=Q(-10, "dBm"),            # level of each tone at the generators
    tone_spacing=Q(1, "MHz"),
    n_frequencies=1,                # 1 = band centre only
)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        __doc__, measure,
        dut=DUT, channels=CHANNELS, settings=SETTINGS, state={"attenuation": DUT_ATTENUATION},
    )
