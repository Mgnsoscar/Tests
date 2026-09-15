"""The lab bench: every instrument and its address, defined once.

Edit the addresses below to match the lab. Get the bench from anywhere with
:func:`bench`; it connects on the first call and returns the same connected
instruments on every later call (see ``TestEnvironment.instance`` in LabKit),
so importing this module from several scripts and helpers never reconnects.

Which instrument does what:

- ``fsv``   — R&S FSV3007 spectrum analyzer: compression, harmonics,
  intermodulation, and the noise-figure application (K30).
- ``vna``   — R&S ZNLE18 vector network analyzer: S11 / S22.
- ``gen_a`` — Keysight N5183A MXG: the main (and first IMD) tone.
- ``gen_b`` — Aim-TTi TGR6000: the second IMD tone.
"""

from __future__ import annotations

from typing import Protocol

from labkit.instruments import FSV3007, N5183A, TGR6000, ZNLE18, TestEnvironment

__all__ = ["LabBench", "BenchLike", "bench"]


class LabBench(TestEnvironment):
    """The instruments on the bench and their IP addresses."""

    fsv: FSV3007
    vna: ZNLE18
    gen_a: N5183A
    gen_b: TGR6000

    def _configure_instruments(self) -> None:
        self.fsv = FSV3007(self, "Spectrum analyzer", "192.168.0.10")
        self.vna = ZNLE18(self, "Network analyzer", "192.168.0.11")
        self.gen_a = N5183A(self, "Signal generator A", "192.168.0.12")
        self.gen_b = TGR6000(self, "Signal generator B", "192.168.0.13")


class BenchLike(Protocol):
    """What a measurement needs from a bench — satisfied by :class:`LabBench`
    and by the simulated bench used in tests (:mod:`rflab.simulation`)."""

    fsv: FSV3007
    vna: ZNLE18
    gen_a: N5183A
    gen_b: TGR6000


def bench(dummy: bool = False) -> LabBench:
    """The shared, connected bench (or the SCPI-printing dummy bench)."""
    return LabBench.instance(use_dummy_instruments=dummy)
