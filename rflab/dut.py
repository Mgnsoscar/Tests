"""Devices under test and their channels.

A :class:`DUT` is a named, versioned device with one or more :class:`Channel`
definitions. A channel is a frequency band plus the signal paths connected to
each of its ports — ``"in"`` and ``"out"`` for an amplifier, plus ``"lo"`` for a
mixer, or whatever the device has. Measurements read the band from the channel
and record the ports' paths in every result, so a result can later be moved to
the DUT reference plane (see :func:`rflab.analysis.at_dut`).

Concrete devices are defined in :mod:`rflab.duts`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from labkit.signal_path import SignalPath
from labkit.units import Quantity, ensure_frequency, quantity

__all__ = ["Channel", "DUT"]


@dataclass(frozen=True)
class Channel:
    """One channel of a DUT: its band and the signal path on each port.

    Parameters
    ----------
    number:
        The channel number, used in file names (``Ch1``).
    f_start, f_stop:
        The band edges.
    ports:
        ``{"in": SignalPath(...), "out": SignalPath(...)}`` — the components
        between each instrument and the DUT port. A port that is not listed is
        treated as a direct (lossless) connection.
    f_lo:
        The LO frequency, for a mixer channel.
    """

    number: int
    f_start: Quantity
    f_stop: Quantity
    ports: Mapping[str, SignalPath] = field(default_factory=dict)
    f_lo: Optional[Quantity] = None

    def __post_init__(self) -> None:
        ensure_frequency(self.f_start)
        ensure_frequency(self.f_stop)
        if self.f_lo is not None:
            ensure_frequency(self.f_lo)
        if not self.f_start < self.f_stop:
            raise ValueError(f"Channel {self.number}: f_start must be below f_stop.")

    @property
    def label(self) -> str:
        """``"Ch1"`` — the channel's short name for files and plots."""
        return f"Ch{self.number}"

    @property
    def f_center(self) -> Quantity:
        """The band centre."""
        return (self.f_start + self.f_stop) / 2  # type: ignore[no-any-return]

    @property
    def span(self) -> Quantity:
        """The band width."""
        return self.f_stop - self.f_start  # type: ignore[no-any-return]

    def frequencies(self, points: int) -> Quantity:
        """`points` equally spaced frequencies across the band, as a Hz array.

        Values are rounded to the nearest millihertz so unit conversions do
        not leave binary-float dust (``2300000000.0000005``) in commands and
        files.
        """
        if points < 1:
            raise ValueError("points must be at least 1.")
        start = self.f_start.to("Hz").magnitude
        stop = self.f_stop.to("Hz").magnitude
        grid = np.array([(start + stop) / 2]) if points == 1 else np.linspace(start, stop, points)
        return quantity(np.round(grid, 3), "Hz")

    def path(self, port: str) -> SignalPath:
        """The signal path on `port`; a direct connection if none was declared."""
        return self.ports.get(port, SignalPath())


@dataclass(frozen=True)
class DUT:
    """A device under test: name, version, and its channels."""

    name: str
    version: str
    channels: tuple[Channel, ...]

    @property
    def label(self) -> str:
        """``"Amplifier X v1.0"`` — used for the results folder."""
        return f"{self.name} {self.version}"

    def channel(self, number: int) -> Channel:
        """The channel with this number."""
        for channel in self.channels:
            if channel.number == number:
                return channel
        raise KeyError(
            f"{self.label} has no channel {number}; channels are "
            f"{[c.number for c in self.channels]}."
        )
