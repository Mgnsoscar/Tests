"""Gain compression (P1dB) on the spectrum analyzer.

Edit the block below, set the DUT by hand to match, then:

    python scripts/run_compression.py [--plot] [--simulate]
"""

from __future__ import annotations

from _cli import run
from labkit.units import quantity as Q
from rflab.measurements.compression import CompressionSettings, measure

# ── Edit before running ──────────────────────────────────────────────────────
DUT = "amplifier_x"                 # module in rflab/duts/
CHANNELS = [1]                      # channel numbers, or None for all channels
DUT_ATTENUATION = Q(0, "dB")        # the DUT's own attenuation setting (set it by hand)
DUT_BYPASS = False                  # the DUT's bypass switch (set it by hand)
SETTINGS = CompressionSettings(
    p_start=Q(-30, "dBm"),          # first generator level
    p_stop=Q(0, "dBm"),             # hard ceiling; the sweep stops earlier once compressed
    p_step=Q(1, "dB"),
    stop_compression=Q(2, "dB"),    # stop when gain has dropped this far (past the 1 dB point)
    n_frequencies=1,                # 1 = band centre only
    attenuator_hold=True,           # MXG step attenuator fixed for the whole sweep (no bumps)
    generator_attenuation=None,     # None = smallest 5 dB step that reaches p_start; or e.g. Q(15, "dB")
    averages=5,                     # FSV sweeps averaged per reading (RMS detector)
    noise_margin=Q(15, "dB"),       # readings closer than this to the noise floor are not used
)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        __doc__, measure,
        dut=DUT, channels=CHANNELS, settings=SETTINGS, state={"attenuation": DUT_ATTENUATION, "bypass": DUT_BYPASS},
    )
