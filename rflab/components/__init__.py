"""The library of characterized path components.

Every cable, attenuator and coupler that can sit between an instrument and a
DUT port is defined here once, with its loss:

- a **flat** loss for a pad whose attenuation is effectively constant, or
- a **table** from a CSV in ``rflab/components/data/``, measured on the VNA
  with ``scripts/characterize_component.py`` and interpolated by LabKit to
  whatever frequency a measurement uses.

Re-characterizing a component means adding a **new** data file with the new
date and pointing the library entry at it; old files stay, so a result saved
with the old characterization still resolves to it (results record each
component as ``"name (date)"``).

Results are written with :meth:`SignalPath.describe` strings and read back
with :func:`resolve_path`, which looks the components up here.
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

from labkit.signal_path import Component, LossTable, SignalPath
from labkit.units import Quantity, quantity as Q

__all__ = [
    "DATA",
    "LIBRARY",
    "register",
    "tabulated",
    "flat",
    "lookup",
    "resolve_path",
    "SMA_CABLE_A",
    "SMA_CABLE_B",
    "COUPLER_20DB",
    "PAD_10DB",
]

#: Where the measured loss tables live.
DATA = Path(__file__).parent / "data"

#: Every known component, keyed by its ``describe()`` string (``"name (date)"``).
LIBRARY: dict[str, Component] = {}


def register(component: Component) -> Component:
    """Add a component to the library (and return it, for assignment)."""
    LIBRARY[component.describe()] = component
    return component


def tabulated(name: str, filename: str, characterized: date) -> Component:
    """A component whose loss table is ``data/<filename>``."""
    return register(Component(name, LossTable.from_csv(str(DATA / filename)), characterized))


def flat(name: str, loss: Quantity, characterized: date) -> Component:
    """A component with a constant loss."""
    return register(Component(name, loss, characterized))


def lookup(description: str) -> Component:
    """Find a component by its ``"name (date)"`` description.

    Falls back to the newest entry with the same name (with a warning) when
    the exact characterization date is no longer in the library.
    """
    description = description.strip()
    if description in LIBRARY:
        return LIBRARY[description]
    name = description.rsplit(" (", 1)[0] if description.endswith(")") else description
    candidates = [c for c in LIBRARY.values() if c.name == name]
    if not candidates:
        raise KeyError(f"No component '{description}' in the library.")
    newest = max(candidates, key=lambda c: c.characterized or date.min)
    warnings.warn(
        f"Component '{description}' not found; using '{newest.describe()}' instead.",
        stacklevel=2,
    )
    return newest


def resolve_path(text: str) -> SignalPath:
    """Rebuild a :class:`SignalPath` from a ``describe()`` string (``"direct"`` = no loss)."""
    text = text.strip()
    if text == "" or text == "direct":
        return SignalPath()
    return SignalPath(*(lookup(part) for part in text.split(", ")))


# --- the components on the bench --------------------------------------------
# Replace these with the lab's real parts; keep one entry per characterization.

SMA_CABLE_A = tabulated("SMA cable A", "sma_cable_a_2026-09-01.csv", date(2026, 9, 1))
SMA_CABLE_B = tabulated("SMA cable B", "sma_cable_b_2026-09-01.csv", date(2026, 9, 1))
COUPLER_20DB = tabulated("20 dB coupler SN0042", "coupler_20db_2026-08-20.csv", date(2026, 8, 20))
PAD_10DB = flat("10 dB pad SN1234", Q(10.2, "dB"), date(2026, 8, 20))
