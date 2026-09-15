"""Device definitions — one module per DUT, each exporting ``DUT``.

``scripts/run_*.py --dut amplifier_x`` imports ``rflab.duts.amplifier_x`` and
uses its ``DUT``. Add a device by copying :mod:`rflab.duts.amplifier_x`.
"""

from __future__ import annotations

import importlib

from ..dut import DUT

__all__ = ["load"]


def load(name: str) -> DUT:
    """Import ``rflab.duts.<name>`` and return its ``DUT``."""
    module = importlib.import_module(f"{__name__}.{name}")
    dut = getattr(module, "DUT", None)
    if not isinstance(dut, DUT):
        raise ImportError(f"rflab.duts.{name} does not define a DUT.")
    return dut
