"""Move a result from the instrument connectors to the DUT ports.

:func:`at_dut` returns a copy of a result with every level expressed at the
DUT reference plane, using the signal paths the result recorded for each port
and the loss of each path **at each row's own frequency** (harmonics at
3·f0, IM3 products off-centre, and so on — interpolated from the components'
characterization tables). A result already at the DUT plane is returned as is.

What each measurement needs:

- **Compression, harmonics, intermodulation** — input levels lose the input
  path (``in.after``), output levels gain back the output path (``out.before``).
- **S-parameters** — a reflection passes its port's path twice, so the
  magnitude gains back **2×** the path loss; a transmission (S21, S12) passes
  the input path and the output path once each, so its magnitude gains back
  both losses. Phase would need the electrical length and is left as measured.
- **Noise figure** — the K30 application already accounted for the losses
  (``reference_plane == "dut"``). Should a raw K30 result ever be loaded, the
  passive input loss L (at room temperature) is subtracted from the noise
  figure and both losses are added back to the gain, the usual first-order
  correction.
"""

from __future__ import annotations

from functools import singledispatch
from typing import TypeVar, cast

from labkit.units import Quantity

from ..measurements._base import Result
from ..measurements.compression import CompressionResult
from ..measurements.harmonics import HarmonicsResult
from ..measurements.intermodulation import IntermodulationResult
from ..measurements.noise_figure import NoiseFigureResult
from ..measurements.s_parameters import SParameterResult

__all__ = ["at_dut"]

_R = TypeVar("_R", bound=Result)


def at_dut(result: _R, extrapolate: bool = False) -> _R:
    """A copy of `result` at the DUT reference plane (see the module docs).

    `extrapolate` lets a path loss table clamp to its end values outside the
    characterized range instead of raising — use deliberately.
    """
    return cast(_R, _at_dut(result, extrapolate))


@singledispatch
def _at_dut(result: Result, extrapolate: bool = False) -> Result:
    raise NotImplementedError(f"at_dut is not implemented for {type(result).__name__}.")


@_at_dut.register
def _compression(result: CompressionResult, extrapolate: bool = False) -> CompressionResult:
    if result.reference_plane == "dut":
        return result
    out = result.path("out")
    return result.replace(
        p_in=result.path("in").after(result.p_in, result.frequency, extrapolate),
        p_out=out.before(result.p_out, result.frequency, extrapolate),
        # the noise floor is an analyzer-port level too: move it with the data
        noise_floor=out.before(result.noise_floor, result.frequency, extrapolate),
        reference_plane="dut",
    )


@_at_dut.register
def _harmonics(result: HarmonicsResult, extrapolate: bool = False) -> HarmonicsResult:
    if result.reference_plane == "dut":
        return result
    return result.replace(
        level=result.path("out").before(result.level, result.frequency, extrapolate),
        reference_plane="dut",
    )


@_at_dut.register
def _intermodulation(result: IntermodulationResult, extrapolate: bool = False) -> IntermodulationResult:
    if result.reference_plane == "dut":
        return result
    return result.replace(
        level=result.path("out").before(result.level, result.frequency, extrapolate),
        reference_plane="dut",
    )


@_at_dut.register
def _s_parameters(result: SParameterResult, extrapolate: bool = False) -> SParameterResult:
    if result.reference_plane == "dut":
        return result
    loss_in = result.path("in").loss_at(result.frequency, extrapolate)
    loss_out = result.path("out").loss_at(result.frequency, extrapolate)
    magnitudes: dict[str, Quantity] = {}
    for name, magnitude in result.magnitudes.items():
        if name == "S11":
            magnitudes[name] = magnitude + loss_in + loss_in
        elif name == "S22":
            magnitudes[name] = magnitude + loss_out + loss_out
        elif name in ("S21", "S12"):
            magnitudes[name] = magnitude + loss_in + loss_out
        else:  # not a two-port parameter of this channel: leave as measured
            magnitudes[name] = magnitude
    return result.replace(magnitudes=magnitudes, reference_plane="dut")


@_at_dut.register
def _noise_figure(result: NoiseFigureResult, extrapolate: bool = False) -> NoiseFigureResult:
    if result.reference_plane == "dut":
        return result
    loss_in = result.path("in").loss_at(result.frequency, extrapolate)
    loss_out = result.path("out").loss_at(result.frequency, extrapolate)
    return result.replace(
        noise_figure=result.noise_figure - loss_in,
        gain=result.gain + loss_in + loss_out,
        reference_plane="dut",
    )
