"""Amplifier X — an example DUT definition. Replace with the real device.

Each channel lists its band and the signal path on each port, using the
components from :mod:`rflab.components`. Paths differ per channel when the
bench is re-cabled between channels; list exactly what is connected. The
requirements (X, Y, Z of the specification) are checked by
``scripts/report_channel.py`` against the measured S21.
"""

from __future__ import annotations

from labkit.signal_path import SignalPath
from labkit.units import quantity as Q

from ..components import COUPLER_20DB, PAD_10DB, SMA_CABLE_A, SMA_CABLE_B
from ..dut import DUT as _DUT, Channel, ChannelRequirements

# Placeholder specification: X = 20 dB rejection at Y = 20 MHz outside the
# band, Z = 1 dB gain variation inside it.
REQUIREMENTS = ChannelRequirements(
    cutoff_rejection=Q(20, "dB"),
    cutoff_offset=Q(20, "MHz"),
    passband_variation=Q(1, "dB"),
)

DUT = _DUT(
    name="Amplifier X",
    version="v1.0",
    channels=(
        Channel(
            1, Q(580, "MHz"), Q(620, "MHz"),
            ports={
                "in": SignalPath(SMA_CABLE_A, PAD_10DB),      # generator -> cable -> pad -> DUT in
                "out": SignalPath(COUPLER_20DB, SMA_CABLE_B),  # DUT out -> coupler -> cable -> analyzer
            },
            requirements=REQUIREMENTS,
        ),
        Channel(
            2, Q(680, "MHz"), Q(720, "MHz"),
            ports={
                "in": SignalPath(SMA_CABLE_A),
                "out": SignalPath(PAD_10DB, SMA_CABLE_B),
            },
            requirements=REQUIREMENTS,
        ),
    ),
)
