"""What every measurement result carries, and helpers the measurements share.

A :class:`Result` is the contract between the three layers: acquisition builds
one, the store writes and reads one, analysis consumes one. Besides the raw
data columns a subclass adds, it records everything needed to interpret the
data later — the DUT and channel, the signal paths that were connected on each
port, the instruments' identification strings, the settings used, and which
reference plane the data is at (``"instrument"`` for raw readings,
``"dut"`` once path losses are accounted for).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from time import sleep
from typing import Any, ClassVar, Mapping, Optional, TypeVar

from labkit.instruments import BaseInstrument
from labkit.signal_path import SignalPath
from labkit.units import Quantity, is_quantity

from ..dut import DUT, Channel

__all__ = [
    "Result",
    "RESULT_TYPES",
    "register_result",
    "describe_instruments",
    "settings_dict",
    "settle",
    "result_metadata",
    "format_state_value",
]

_R = TypeVar("_R", bound="Result")

#: Every result class by its ``measurement`` name, for loading from disk.
RESULT_TYPES: dict[str, type["Result"]] = {}


def register_result(cls: type[_R]) -> type[_R]:
    """Class decorator: make a result type loadable by its ``measurement`` name."""
    RESULT_TYPES[cls.measurement] = cls
    return cls


@dataclass(kw_only=True)
class Result:
    """Base of every measurement result. Subclasses add the data columns."""

    #: The measurement's name — used in folder and file names ("Compression").
    measurement: ClassVar[str] = "Result"

    dut: str
    version: str
    channel: int
    timestamp: datetime = field(default_factory=datetime.now)
    #: The signal path connected on each DUT port when the data was taken.
    paths: dict[str, SignalPath] = field(default_factory=dict)
    #: ``{instrument name: *IDN? string}``.
    instruments: dict[str, str] = field(default_factory=dict)
    #: The settings the measurement was taken with (scalars and quantities).
    settings: dict[str, Any] = field(default_factory=dict)
    #: The DUT's own state during the measurement, e.g. ``{"attenuation": 6 dB}``.
    state: dict[str, Any] = field(default_factory=dict)
    #: ``"instrument"`` (raw readings) or ``"dut"`` (path losses accounted for).
    reference_plane: str = "instrument"

    def columns(self) -> dict[str, Any]:
        """The data columns, in the order they are written: ``{label: array}``."""
        raise NotImplementedError

    @classmethod
    def from_columns(cls: type[_R], meta: dict[str, Any], columns: dict[str, Any]) -> _R:
        """Rebuild a result from metadata and the columns read from disk."""
        raise NotImplementedError

    def replace(self: _R, **changes: Any) -> _R:
        """A copy with some fields replaced."""
        return dataclasses.replace(self, **changes)

    def path(self, port: str) -> SignalPath:
        """The path on `port`, or a direct connection if none was recorded."""
        return self.paths.get(port, SignalPath())

    @property
    def channel_label(self) -> str:
        return f"Ch{self.channel}"

    @property
    def state_label(self) -> str:
        """``"attenuation 6 dB"`` — the DUT state for file names and titles (``""`` if none)."""
        return " ".join(f"{key} {_format_state(value)}" for key, value in self.state.items())

    @property
    def title(self) -> str:
        """``"Amplifier X v1.0 Ch1 attenuation 6 dB"`` — for plot titles."""
        parts = [self.dut, self.version, self.channel_label]
        if self.state:
            parts.append(self.state_label)
        return " ".join(parts)


def format_state_value(value: Any) -> str:
    """One state value for names and headers: ``6 dB``, ``on``/``off`` for a switch."""
    if is_quantity(value):
        return f"{value:~g}"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


_format_state = format_state_value


def result_metadata(result: Result) -> dict[str, Any]:
    """The base fields of a result as keyword arguments for ``from_columns``."""
    return {
        "dut": result.dut,
        "version": result.version,
        "channel": result.channel,
        "timestamp": result.timestamp,
        "paths": dict(result.paths),
        "instruments": dict(result.instruments),
        "settings": dict(result.settings),
        "state": dict(result.state),
        "reference_plane": result.reference_plane,
    }


def describe_instruments(*instruments: BaseInstrument) -> dict[str, str]:
    """``{name: *IDN?}`` for the instruments a measurement used."""
    return {inst._name: inst.get_id().strip() for inst in instruments}


def settings_dict(settings: Any) -> dict[str, Any]:
    """A settings dataclass as a plain dict of scalars/quantities (for the file header)."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(settings):
        value = getattr(settings, f.name)
        if isinstance(value, (tuple, list)):
            value = ", ".join(str(v) for v in value)
        out[f.name] = value
    return out


def settle(duration: Quantity) -> None:
    """Pause for a settling time given as a duration quantity."""
    seconds = float(duration.to("s").magnitude)
    if seconds > 0:
        sleep(seconds)


def _identity(dut: DUT, channel: Channel) -> dict[str, Any]:
    return {"dut": dut.name, "version": dut.version, "channel": channel.number, "paths": dict(channel.ports)}


def base_fields(
    dut: DUT,
    channel: Channel,
    instruments: dict[str, str],
    settings: Any,
    state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """The base result fields for a measurement of `channel` on `dut` in `state`."""
    fields = _identity(dut, channel)
    fields["instruments"] = instruments
    fields["settings"] = settings_dict(settings)
    fields["state"] = dict(state or {})
    return fields


def as_quantity(value: Any) -> bool:
    """``True`` if `value` is a LabKit quantity (re-exported for the store)."""
    return is_quantity(value)
