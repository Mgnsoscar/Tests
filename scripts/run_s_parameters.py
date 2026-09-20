"""S-parameters (S11, S21, S22) on the ZNLE, with the calibration handled first.

Edit the block below, set the DUT by hand to match, then:

    python scripts/run_s_parameters.py [--plot] [--simulate] [--skip-calibration]

The script configures the sweep, then asks whether to calibrate now on the
instrument (and save that calibration to its cal pool), load a saved
calibration, or keep the current correction — and only then measures. With a
fixed FREQUENCY_RANGE covering every channel, one calibration serves all of
them; with ``frequency_range=None`` each channel sweeps its own band and is
asked about calibration separately.

Run it once per DUT configuration (attenuation, bypass), then
``scripts/report_channel.py`` checks the channel's requirements across all
of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _cli import _parser, describe_state, make_bench, report, results_root, set_dut_state  # noqa: E402
from labkit.units import quantity as Q  # noqa: E402

from rflab import duts  # noqa: E402
from rflab.calibration import calibration_dialog  # noqa: E402
from rflab.measurements.s_parameters import SParameterSettings, configure, measure, sweep_range  # noqa: E402
from rflab.store import ResultStore  # noqa: E402

# ── Edit before running ──────────────────────────────────────────────────────
DUT = "amplifier_x"                 # module in rflab/duts/
CHANNELS = None                     # channel numbers, or None for all channels
DUT_ATTENUATION = Q(0, "dB")        # the DUT's own attenuation setting (set it by hand)
DUT_BYPASS = False                  # the DUT's bypass switch (set it by hand)
SETTINGS = SParameterSettings(
    frequency_range=(Q(550, "MHz"), Q(750, "MHz")),   # one sweep (and one calibration) for all channels
    points=401,                     # 0.5 MHz per point over 200 MHz
    if_bandwidth=Q(1, "kHz"),
    power=Q(-20, "dBm"),            # keep an active DUT's input well out of compression
    average_count=8,
    parameters=("S11", "S21", "S22"),   # S21 = gain (the requirements), S11/S22 = match
)
CALIBRATION_NAME = "amplifier_x 550-750MHz"   # cal-pool name offered when saving or loading
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = _parser(__doc__)
    p.add_argument("--skip-calibration", action="store_true", help="no calibration dialog; keep the current correction")
    args = p.parse_args()

    device = duts.load(DUT)
    channels = device.channels if CHANNELS is None else [device.channel(n) for n in CHANNELS]
    store = ResultStore(results_root(args))
    b = make_bench(args)
    state = {"attenuation": DUT_ATTENUATION, "bypass": DUT_BYPASS}

    # Channels sharing a sweep range share one configuration and one calibration.
    done_ranges: dict[tuple[float, float], str] = {}
    for channel in channels:
        set_dut_state(b, channel, state)
        start, stop = sweep_range(channel, SETTINGS)
        key = (float(start.to("Hz").magnitude), float(stop.to("Hz").magnitude))
        if key not in done_ranges:
            configure(b.vna, channel, SETTINGS)
            print(
                f"Sweep configured: {start:~} to {stop:~}, {SETTINGS.points} points, "
                f"IF BW {SETTINGS.if_bandwidth:~}, {SETTINGS.power:~}, {SETTINGS.average_count} averages"
            )
            if args.skip_calibration or args.simulate:
                done_ranges[key] = "kept (no dialog)"
            else:
                done_ranges[key] = calibration_dialog(b.vna, CALIBRATION_NAME)
        print(f"{device.label} {channel.label} ({describe_state(state)}): measuring ...", flush=True)
        result = measure(b, device, channel, SETTINGS, state, configure_sweep=False, calibration=done_ranges[key])
        path = store.save(result)
        print(f"  saved {path}")
        if args.plot or args.show:
            report(result, store, show=args.show)


if __name__ == "__main__":
    main()
