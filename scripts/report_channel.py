"""Check each channel's requirements across all its S-parameter results.

    python scripts/report_channel.py [--show] [--root results]
    python scripts/report_channel.py --simulate        # measures 4 configurations first, then reports

For every channel in the edit block the script gathers the saved S-parameter
results (one per DUT configuration: attenuation and bypass), keeps the latest
result of each configuration, and answers the requirements:

- **S21**: main gain (highest attenuation, bypass on) and max gain (no
  attenuation, bypass off); the filter cutoff — at least X dB below the
  centre gain Y MHz outside the band; the passband variation — less than
  Z dB from the centre gain inside the band. X, Y, Z come from the channel's
  ``requirements`` in its DUT definition (or REQUIREMENTS below).
- **S11** (and S22 when measured): the average over all configurations,
  drawn on top of the individual traces, with the worst in-band value.

The text report and the figures are saved next to the data, in the
S-Parameters folder of the DUT.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _cli import _parser, make_bench, results_root, set_dut_state  # noqa: E402
from labkit.units import quantity as Q  # noqa: E402

from rflab import duts  # noqa: E402
from rflab.analysis import channel_report  # noqa: E402
from rflab.dut import Channel, ChannelRequirements  # noqa: E402
from rflab.measurements.s_parameters import SParameterResult, SParameterSettings, measure  # noqa: E402
from rflab.store import ResultStore  # noqa: E402

# ── Edit before running ──────────────────────────────────────────────────────
DUT = "amplifier_x"                 # module in rflab/duts/
CHANNELS = None                     # channel numbers, or None for all channels
REQUIREMENTS: Optional[ChannelRequirements] = None   # None = the channel's own; or override here, e.g.
# REQUIREMENTS = ChannelRequirements(cutoff_rejection=Q(20, "dB"), cutoff_offset=Q(20, "MHz"), passband_variation=Q(1, "dB"))

# With --simulate, these configurations are measured on the simulated bench first.
SIMULATED_CONFIGURATIONS = [
    {"attenuation": Q(0, "dB"), "bypass": False},
    {"attenuation": Q(6, "dB"), "bypass": False},
    {"attenuation": Q(12, "dB"), "bypass": False},
    {"attenuation": Q(12, "dB"), "bypass": True},
]
# ─────────────────────────────────────────────────────────────────────────────


def _with_requirements(channel: Channel) -> Channel:
    if REQUIREMENTS is None:
        return channel
    return Channel(channel.number, channel.f_start, channel.f_stop, channel.ports, channel.f_lo, REQUIREMENTS)


def main() -> None:
    p = _parser(__doc__)
    args = p.parse_args()
    device = duts.load(DUT)
    channels = device.channels if CHANNELS is None else [device.channel(n) for n in CHANNELS]
    store = ResultStore(results_root(args))

    if args.simulate:
        b = make_bench(args)
        settings = SParameterSettings(points=201)
        for channel in channels:
            for state in SIMULATED_CONFIGURATIONS:
                set_dut_state(b, channel, state)
                store.save(measure(b, device, channel, settings, state))
        print(f"simulated {len(SIMULATED_CONFIGURATIONS)} configurations per channel into {store.root}")

    for channel in channels:
        channel = _with_requirements(channel)
        files = store.find(dut=device.label, measurement=SParameterResult.measurement, channel=channel.number)
        results = [r for r in (store.load(f) for f in files) if isinstance(r, SParameterResult)]
        if not results:
            print(f"{device.label} {channel.label}: no S-parameter results under {store.root}")
            continue
        rep = channel_report.report(channel, results, dut=device.label)
        print()
        print(rep.text())

        newest = max((e.result for e in rep.gains), key=lambda r: r.timestamp, default=results[-1])
        text_path = store.companion_path(newest, " report", ".txt")
        text_path.write_text(rep.text() + "\n", encoding="utf-8")
        print(f"  report {text_path}")
        if rep.gains:
            folder, name = store.figure_location(newest, " report S21")
            channel_report.plot_gain(rep, show=args.show, save_folder=folder, filename=name)
            print(f"  figure {Path(folder) / (name + '.png')}")
        for parameter in rep.reflections:
            folder, name = store.figure_location(newest, f" report {parameter}")
            channel_report.plot_reflection(rep, parameter, show=args.show, save_folder=folder, filename=name)
            print(f"  figure {Path(folder) / (name + '.png')}")


if __name__ == "__main__":
    main()
