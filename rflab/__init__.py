"""rflab — the lab's measurement project, built on LabKit.

Three layers that never mix:

1. **Acquire** (:mod:`rflab.measurements`) — talks to the instruments on the
   :mod:`rflab.bench` and returns a result object holding raw quantity arrays
   plus everything needed to interpret them later: DUT and channel, the
   signal-path components that were connected, instrument identities, and
   the settings used. No files, no plots.
2. **Store** (:mod:`rflab.store`) — writes a result to a CSV named by the lab's
   conventions (``"(B) "`` folders, ``"{date} (B) "`` files) and reads it back
   into the same result object.
3. **Analyse** (:mod:`rflab.analysis`) — moves a result to the DUT reference
   plane (:func:`rflab.analysis.at_dut`), derives numbers (P1dB, OIP3, noise
   figure, return loss, harmonic levels) and plots. Works on a fresh result or
   one loaded from disk, identically.

The bench (:mod:`rflab.bench`), DUT channels (:mod:`rflab.duts`) and the
component library (:mod:`rflab.components`) describe *what is on the table*;
the scripts in ``scripts/`` glue the layers together.
"""

from __future__ import annotations

__version__ = "0.1.0"
