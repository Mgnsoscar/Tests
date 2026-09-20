"""DUT channels, ports and the component library."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from labkit.signal_path import SignalPath
from labkit.units import quantity as Q

from rflab import components
from rflab.dut import DUT, Channel
from rflab.duts import load


def test_channel_band_helpers() -> None:
    ch = Channel(1, Q(2.2, "GHz"), Q(2.4, "GHz"))
    assert ch.label == "Ch1"
    assert ch.f_center.to("GHz").magnitude == pytest.approx(2.3)
    assert ch.span.to("MHz").magnitude == pytest.approx(200.0)
    np.testing.assert_allclose(ch.frequencies(1).to("GHz").magnitude, [2.3])
    np.testing.assert_allclose(ch.frequencies(3).to("GHz").magnitude, [2.2, 2.3, 2.4])
    assert ch.path("in").describe() == "direct"  # undeclared port = lossless
    with pytest.raises(ValueError):
        Channel(1, Q(2.4, "GHz"), Q(2.2, "GHz"))


def test_dut_lookup() -> None:
    dut = load("amplifier_x")
    assert dut.label == "Amplifier X v1.0"
    assert dut.channel(2).number == 2
    with pytest.raises(KeyError):
        dut.channel(9)
    with pytest.raises(ImportError):
        load("no_such_device")


def test_channel_ports_differ_per_channel() -> None:
    dut: DUT = load("amplifier_x")
    ch1_in = dut.channel(1).path("in").loss_at(dut.channel(1).f_center)
    ch2_in = dut.channel(2).path("in").loss_at(dut.channel(2).f_center)
    assert ch1_in > ch2_in  # channel 1 has the pad in its input path


def test_library_lookup_and_path_resolution() -> None:
    exact = components.lookup("SMA cable A (2026-09-01)")
    assert exact is components.SMA_CABLE_A
    with pytest.warns(UserWarning):
        by_name = components.lookup("SMA cable A (2020-01-01)")
    assert by_name is components.SMA_CABLE_A
    with pytest.raises(KeyError):
        components.lookup("no such part (2026-01-01)")

    path = SignalPath(components.SMA_CABLE_A, components.PAD_10DB)
    rebuilt = components.resolve_path(path.describe())
    assert [c.name for c in rebuilt] == ["SMA cable A", "10 dB pad SN1234"]
    assert rebuilt.loss_at(Q(2, "GHz")) == path.loss_at(Q(2, "GHz"))
    assert len(components.resolve_path("direct")) == 0


def test_tables_interpolate_between_characterized_points() -> None:
    cable = components.SMA_CABLE_A
    assert cable.characterized == date(2026, 9, 1)
    # 2 GHz -> 0.60 dB, 3 GHz -> 0.75 dB: 2.5 GHz interpolates to 0.675 dB
    assert cable.loss_at(Q(2.5, "GHz")).magnitude == pytest.approx(0.675)
