"""Characterize a path component on the VNA, de-embedding the fixture used to connect it.

A cable with two male ends cannot be connected straight to the VNA's male
test-port cables; a female-to-female adapter bridges each end. Those adapters
are themselves components with a loss, so :func:`characterize` takes a
`fixture` — a :class:`~labkit.signal_path.SignalPath` of the parts that were
in the measurement but are **not** part of the component being characterized
— and subtracts their loss (interpolated at each measured frequency) from the
measured insertion loss. Both the de-embedded and the raw loss are saved, and
the fixture is recorded in the file header, so the characterization is
traceable and can be redone if a fixture part is re-characterized.

Getting the fixture's own loss: measure the two adapters back-to-back (with a
known-good thru), record that as ``"SMA F-F adapter pair"`` and register each
adapter with half of it — or include the adapter in the VNA calibration and use
a zero-loss fixture here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

from labkit.instruments import ZNLE18
from labkit.io.csv import Column, Value, write
from labkit.signal_path import SignalPath
from labkit.units import Quantity, quantity as Q

from . import DATA

__all__ = ["CharacterizationSettings", "Characterization", "characterize", "save", "slug"]


@dataclass(frozen=True)
class CharacterizationSettings:
    """The VNA sweep used to characterize a component (S21 between ports 1 and 2)."""

    start: Quantity = Q(100, "MHz")
    stop: Quantity = Q(12, "GHz")
    points: int = 401
    power: Quantity = Q(-10, "dBm")
    if_bandwidth: Quantity = Q(1, "kHz")


@dataclass(frozen=True)
class Characterization:
    """A measured component: de-embedded loss, the raw loss, and how it was measured."""

    name: str
    frequency: Quantity
    loss: Quantity
    loss_raw: Quantity
    fixture: SignalPath
    instrument: str
    settings: CharacterizationSettings
    when: date


def characterize(
    vna: ZNLE18,
    name: str,
    settings: CharacterizationSettings = CharacterizationSettings(),
    fixture: SignalPath = SignalPath(),
    when: Optional[date] = None,
) -> Characterization:
    """Measure S21 of the component between VNA ports 1 and 2 and de-embed `fixture`."""
    vna.trace.delete_all()
    vna.frequency.set_start(settings.start)
    vna.frequency.set_stop(settings.stop)
    vna.sweep.set_points(settings.points)
    vna.bandwidth.set_if_bandwidth(settings.if_bandwidth)
    vna.power.set_power(settings.power)
    vna.trace.create("Trc1", "S21")
    vna.display.set_window_state(1, True)
    vna.display.feed_trace(1, 1, "Trc1")
    vna.trace.select("Trc1")
    frequency, s21 = vna.measure()
    vna.power.set_output(False)

    loss_raw = Q(-20 * np.log10(np.abs(s21)), "dB")
    loss = loss_raw - fixture.loss_at(frequency)
    return Characterization(
        name=name,
        frequency=frequency,
        loss=loss,
        loss_raw=loss_raw,
        fixture=fixture,
        instrument=vna.get_id().strip(),
        settings=settings,
        when=when or date.today(),
    )


def slug(name: str) -> str:
    """``"SMA cable A"`` -> ``"sma_cable_a"`` for file names."""
    import re

    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save(result: Characterization, folder: Path = DATA) -> Path:
    """Write ``<folder>/<slug>_<date>.csv`` for the component library and return its path.

    The de-embedded ``Loss`` column comes first so ``LossTable.from_csv`` picks
    it; ``Loss raw`` keeps what the VNA measured with the fixture in place.
    """
    filename = f"{slug(result.name)}_{result.when.isoformat()}"
    path = write(
        Value("Component", result.name, unit=None),
        Value("Characterized", result.when.isoformat(), unit=None),
        Value("Instrument", result.instrument, unit=None),
        Value("Fixture (de-embedded)", result.fixture.describe(), unit=None),
        Value("Source power", result.settings.power),
        Value("IF bandwidth", result.settings.if_bandwidth),
        Column("Frequency", result.frequency.to("MHz")),
        Column("Loss", result.loss),
        Column("Loss raw", result.loss_raw),
        filename=filename,
        folder=str(folder),
    )
    return Path(path)
