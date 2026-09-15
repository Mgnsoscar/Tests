"""Saving results and loading them back — with the lab's naming rules.

Every folder name starts with ``"(B) "`` and every file name with
``"{date} (B) "``::

    results/
      (B) Amplifier X v1.0/
        (B) Compression/
          2026-09-15 (B) Compression Ch1 attenuation 0 dB.csv
          2026-09-15 (B) Compression Ch1 attenuation 6 dB.csv
          2026-09-15 (B) Compression Ch1 attenuation 6 dB (2).csv   <- same day, second run
          2026-09-15 (B) Compression Ch1 attenuation 6 dB.png       <- figures next to the data

The CSV header records everything the result object carries (DUT, channel,
timestamp, reference plane, the signal path on each port, instrument
identities, the DUT state, settings), so :meth:`ResultStore.load` rebuilds the very same
result object — ready for :func:`rflab.analysis.at_dut` and the rest of the
analysis layer, without touching an instrument.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from labkit.io.csv import Column, Value, read, write
from labkit.units import is_quantity

from . import components
from .measurements._base import RESULT_TYPES, Result

__all__ = ["ResultStore"]

_PATH_PREFIX = "Path "
_INSTRUMENT_PREFIX = "Instrument "
_SETTING_PREFIX = "Setting "
_STATE_PREFIX = "State "
_COUNTER = re.compile(r"^(?P<base>.*?)(?: \((?P<n>\d+)\))?$")


def _sort_key(path: Path) -> tuple[str, str, int]:
    """Order files by folder, then name, then their ``(2)``, ``(3)`` run counter."""
    match = _COUNTER.match(path.stem)
    assert match is not None
    return (str(path.parent), match.group("base"), int(match.group("n") or 1))


class ResultStore:
    """Writes results to, and reads them from, a results tree.

    Parameters
    ----------
    root:
        The results folder (created on first save).
    prefix:
        The lab's folder/file prefix. Folders are ``"{prefix}{name}"``, files
        ``"{date} {prefix}{measurement} Ch{n}"``.
    """

    def __init__(self, root: str | Path = "results", prefix: str = "(B) ") -> None:
        self.root = Path(root)
        self.prefix = prefix

    # -- naming --------------------------------------------------------------
    def folder(self, result: Result) -> Path:
        """``root/(B) <DUT> <version>/(B) <Measurement>`` for this result."""
        return self.root / f"{self.prefix}{result.dut} {result.version}" / f"{self.prefix}{result.measurement}"

    def basename(self, result: Result, suffix: str = "") -> str:
        """``"2026-09-15 (B) Compression Ch1 attenuation 6 dB"`` (plus `suffix`), without extension.

        The DUT state (its attenuation setting) is part of the name, so runs at
        different settings never collide.
        """
        day = result.timestamp.date().isoformat()
        state = f" {result.state_label}" if result.state else ""
        return f"{day} {self.prefix}{result.measurement} {result.channel_label}{state}{suffix}"

    def _unique(self, folder: Path, basename: str, extension: str) -> str:
        """`basename`, or ``"basename (2)"`` etc., so an existing file is not overwritten."""
        candidate = basename
        counter = 2
        while (folder / f"{candidate}{extension}").exists():
            candidate = f"{basename} ({counter})"
            counter += 1
        return candidate

    def figure_location(self, result: Result, suffix: str = "") -> tuple[str, str]:
        """``(folder, filename)`` for a figure belonging to `result`, for ``labkit.plotting.plot``."""
        folder = self.folder(result)
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder), self._unique(folder, self.basename(result, suffix), ".png")

    # -- save ----------------------------------------------------------------
    def save(self, result: Result) -> str:
        """Write `result` as a CSV and return its path."""
        folder = self.folder(result)
        folder.mkdir(parents=True, exist_ok=True)
        filename = self._unique(folder, self.basename(result), ".csv")

        values = [
            Value("Measurement", result.measurement, unit=None),
            Value("DUT", result.dut, unit=None),
            Value("Version", result.version, unit=None),
            Value("Channel", result.channel, unit=None),
            Value("Timestamp", result.timestamp.isoformat(timespec="seconds"), unit=None),
            Value("Reference plane", result.reference_plane, unit=None),
        ]
        for port, path in result.paths.items():
            values.append(Value(f"{_PATH_PREFIX}{port}", path.describe(), unit=None))
        for name, idn in result.instruments.items():
            values.append(Value(f"{_INSTRUMENT_PREFIX}{name}", idn, unit=None))
        for key, value in result.state.items():
            values.append(Value(f"{_STATE_PREFIX}{key}", value, unit="" if is_quantity(value) else None))
        for key, value in result.settings.items():
            values.append(Value(f"{_SETTING_PREFIX}{key}", value, unit="" if is_quantity(value) else None))
        columns = [Column(label, data) for label, data in result.columns().items()]

        return write(*values, *columns, filename=filename, folder=str(folder))

    # -- load ----------------------------------------------------------------
    def load(self, path: str | Path) -> Result:
        """Read a saved CSV back into the result type it was written from."""
        table = read(str(path))
        measurement = str(table.value("Measurement"))
        try:
            cls = RESULT_TYPES[measurement]
        except KeyError:
            raise ValueError(f"{path}: unknown measurement '{measurement}'.") from None

        paths = {
            key[len(_PATH_PREFIX):]: components.resolve_path(str(text))
            for key, text in table.values.items()
            if key.startswith(_PATH_PREFIX)
        }
        instruments = {
            key[len(_INSTRUMENT_PREFIX):]: str(idn)
            for key, idn in table.values.items()
            if key.startswith(_INSTRUMENT_PREFIX)
        }
        settings = {
            key[len(_SETTING_PREFIX):]: value
            for key, value in table.values.items()
            if key.startswith(_SETTING_PREFIX)
        }
        state = {
            key[len(_STATE_PREFIX):]: value
            for key, value in table.values.items()
            if key.startswith(_STATE_PREFIX)
        }
        meta: dict[str, Any] = {
            "dut": str(table.value("DUT")),
            "version": str(table.value("Version")),
            "channel": int(table.value("Channel")),
            "timestamp": datetime.fromisoformat(str(table.value("Timestamp"))),
            "reference_plane": str(table.value("Reference plane")),
            "paths": paths,
            "instruments": instruments,
            "settings": settings,
            "state": state,
        }
        return cls.from_columns(meta, table.columns)

    # -- browse --------------------------------------------------------------
    def find(
        self,
        dut: Optional[str] = None,
        measurement: Optional[str] = None,
        channel: Optional[int] = None,
    ) -> list[Path]:
        """Saved CSVs under `root`, optionally filtered by DUT label, measurement and channel."""
        found: list[Path] = []
        for path in sorted(self.root.rglob("*.csv"), key=_sort_key):
            if dut is not None and path.parent.parent.name != f"{self.prefix}{dut}":
                continue
            if measurement is not None and path.parent.name != f"{self.prefix}{measurement}":
                continue
            if channel is not None and f" Ch{channel}" not in path.stem:
                continue
            found.append(path)
        return found
