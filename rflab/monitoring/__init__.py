"""Long-running monitors: tests that watch a DUT for events over hours rather
than taking one measurement.

- :mod:`.continuity` — electrical-contact monitoring on the oscilloscope:
  every dropout timestamped and written to disk as it happens.
"""

from __future__ import annotations

from . import continuity

__all__ = ["continuity"]
