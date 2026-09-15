"""Amplifier X — an example DUT definition. Replace with the real device.

Each channel lists its band and the signal path on each port, using the
components from :mod:`rflab.components`. Paths differ per channel when the
bench is re-cabled between channels; list exactly what is connected.
"""

from __future__ import annotations

from labkit.signal_path import SignalPath
from labkit.units import quantity as Q

from ..components import COUPLER_20DB, PAD_10DB, SMA_CABLE_A, SMA_CABLE_B
from ..dut import DUT as _DUT, Channel

DUT = _DUT(
    name="Amplifier X",
    version="v1.0",
    channels=(
        Channel(
            1, Q(2.2, "GHz"), Q(2.4, "GHz"),
            ports={
                "in": SignalPath(SMA_CABLE_A, PAD_10DB),      # generator -> cable -> pad -> DUT in
                "out": SignalPath(COUPLER_20DB, SMA_CABLE_B),  # DUT out -> coupler -> cable -> analyzer
            },
        ),
        Channel(
            2, Q(3.4, "GHz"), Q(3.6, "GHz"),
            ports={
                "in": SignalPath(SMA_CABLE_A),
                "out": SignalPath(PAD_10DB, SMA_CABLE_B),
            },
        ),
    ),
)
