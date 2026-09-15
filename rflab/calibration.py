"""The ZNLE calibration step: calibrate by hand, save it, or load a saved one.

A VNA calibration belongs to a particular sweep — frequency range, number of
points, IF bandwidth, source power — so the sweep is configured **first** and
the calibration comes after. :func:`calibration_dialog` then offers three
routes and returns a one-line description that is recorded with every result
measured under that calibration:

- **calibrate now**: the guided calibration is done on the instrument's own
  screen (connect open, short, match, through as it asks); when it is finished
  the script continues, and offers to store the fresh calibration in the
  instrument's cal pool under a name, so it can be loaded next time;
- **load**: apply a calibration saved earlier in the cal pool;
- **keep**: continue with whatever correction the channel currently has.

The dialog is plain ``input``/``print`` by default; tests script it through
the `ask`/`say` parameters.
"""

from __future__ import annotations

from typing import Callable

from labkit.instruments import ZNLE18

__all__ = ["calibration_dialog", "cal_file_name", "describe_correction"]

Ask = Callable[[str], str]
Say = Callable[[str], None]


def cal_file_name(name: str) -> str:
    """``"Amplifier X 550-750MHz"`` -> ``"Amplifier X 550-750MHz.cal"`` (the cal-pool file name)."""
    name = name.strip()
    return name if name.lower().endswith(".cal") else f"{name}.cal"


def describe_correction(vna: ZNLE18, channel: int = 1) -> str:
    """``"correction on, calibrated 2026-09-15 10:02:11"`` for the channel's current state."""
    if not vna.calibration.is_correction_enabled(channel):
        return "correction off"
    date = vna.calibration.get_correction_date(channel)
    return f"correction on, calibrated {date}" if date else "correction on"


def calibration_dialog(
    vna: ZNLE18,
    default_name: str,
    channel: int = 1,
    ask: Ask = input,
    say: Say = print,
) -> str:
    """Offer to calibrate, load or keep the calibration; return a description of what was done.

    Must be called **after** the sweep is configured. `default_name` is the
    cal-pool name proposed for saving or loading (``.cal`` is added).
    """
    say("")
    say("Calibration for the configured sweep")
    say(f"  currently: {describe_correction(vna, channel)}")
    say("  [c] calibrate now on the ZNLE (guided calibration on its screen), then continue here")
    say("  [l] load a saved calibration from the cal pool")
    say("  [k] keep the current correction as it is")
    while True:
        choice = ask("Choice [c/l/k]: ").strip().lower()[:1]
        if choice in ("c", "l", "k"):
            break
        say("  please answer c, l or k")

    if choice == "k":
        return f"kept ({describe_correction(vna, channel)})"

    if choice == "l":
        answer = ask(f"Cal pool file to load [{cal_file_name(default_name)}]: ").strip()
        file = cal_file_name(answer or default_name)
        vna.calibration.load(channel, file)
        vna.calibration.set_correction_enabled(True, channel)
        say(f"  loaded '{file}': {describe_correction(vna, channel)}")
        return f"loaded {file}"

    say("")
    say("Perform the calibration on the instrument now (Cal > Start Cal, follow the")
    say("standards it asks for, then Apply). The sweep settings are already in place.")
    ask("Press Enter here when the calibration is finished... ")
    if not vna.calibration.is_correction_enabled(channel):
        vna.calibration.set_correction_enabled(True, channel)
    say(f"  {describe_correction(vna, channel)}")
    answer = ask(f"Save it to the cal pool as [{cal_file_name(default_name)}] (Enter = yes, n = no, or a name): ").strip()
    if answer.lower() in ("n", "no"):
        return "manual, not saved"
    file = cal_file_name(answer if answer and answer.lower() not in ("y", "yes") else default_name)
    vna.calibration.save(channel, file)
    say(f"  saved as '{file}'")
    return f"manual, saved as {file}"
