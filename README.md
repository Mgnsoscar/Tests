# rflab — lab measurements on LabKit

The measurement project for the bench: what is on the table, how each test is
run, where results go, and how they are analysed afterwards — with the
signal-path components between instruments and DUT accounted for.

## Layout

```
rflab/
  bench.py            the instruments and their addresses (one shared instance)
  dut.py              Channel / DUT model
  duts/               one module per device: amplifier_x.py, ...
  components/         the characterized cables, pads, couplers, adapters (+ data/ CSVs, characterize.py)
  measurements/       acquisition: compression, noise_figure, s_parameters, harmonics, intermodulation
  store.py            "(B) " folders, "{date} (B) " files, CSV save + load
  analysis/           at_dut() reference-plane correction, summaries, plots
  monitoring/         long-running monitors: continuity.py (contact dropouts on the scope)
  simulation.py       a simulated bench so everything runs without hardware
scripts/
  run_<measurement>.py   edit the block at the top, then: measure -> save [-> analyse]
  analyze.py             load -> at_dut -> summarize -> plot, for saved results
  report_channel.py      the per-channel requirements report from all S-parameter results
  monitor_continuity.py  contact monitoring on the oscilloscope during environmental tests
  characterize_component.py   measure a component on the VNA (fixture de-embedded) into components/data/
tests/                pytest suite against the simulated bench
results/              the data (not committed)
```

## Workflow

```bash
pip install -e ".[dev]"                          # pulls LabKit from GitHub
```

Each run script has an **"Edit before running"** block at its top: the DUT
module, the channels, the DUT's own attenuation and bypass settings (which
you set on the device by hand, and the script records), and the measurement
settings. Edit it, then run:

```bash
python scripts/run_compression.py --plot
python scripts/run_noise_figure.py
python scripts/run_s_parameters.py
python scripts/run_harmonics.py
python scripts/run_intermodulation.py

python scripts/analyze.py --all --measurement Compression    # post-process later, no instruments
python scripts/analyze.py "results/(B) Amplifier X v1.0/(B) Harmonics/2026-09-15 (B) Harmonics Ch1 attenuation 0 dB.csv" --show
```

Add `--simulate` to any script to exercise it against the simulated bench
(its data goes to `results-simulated/`, never into `results/`).

Results land in
`results/(B) <DUT> <version>/(B) <Measurement>/{date} (B) <Measurement> Ch<n> attenuation <x> dB bypass <on|off>.csv`,
with a header that records the DUT, channel, timestamp, reference plane, the
signal path on every port (each component with its characterization date), the
instruments' `*IDN?` strings, the DUT state and the settings. Figures are saved
next to the data with the same name. A second run with the same name on the
same day gets a `(2)` suffix rather than overwriting.

The S-parameter script measures S11, S21 and S22 over each channel's band plus
a 25 MHz margin on either side (401 points, 1 kHz IF bandwidth, 8 averages,
−20 dBm) — only the region the requirements are about — and handles the
calibration before measuring: it configures the sweep, then asks whether to
**calibrate now** on the ZNLE's screen (and afterwards save the calibration to
the instrument's cal pool under `CALIBRATION_NAME` plus the sweep range),
**load** a saved calibration, or **keep** the current correction. What was
done, and the instrument's correction state and date, are recorded with every
result. `--skip-calibration` skips the dialog. Set a fixed `frequency_range`
covering every channel instead if one calibration should serve them all.

## Contact monitoring during environmental tests

`scripts/monitor_continuity.py` watches a DUT's electrical contact on the
RTO64 for as long as a temperature, shock or vibration test runs. The circuit:
a lab supply at 2 V (50 mA limit) through a 50 Ω resistor soldered to the
antenna body, through the contact under test, down the coax into the scope's
50 Ω input, supply negative to the DUT ground. Closed contact = 1 V at the
scope, open = 0 V within nanoseconds. The scope triggers on **either** edge
through 0.5 V in NORMAL mode with fast segmentation, so every opening and
every closing is captured with its own timestamp, and each record is taken in
peak-detect mode, so any crossing the trigger saw is in the record too.
Every minute the script stops the scope, reads every stored acquisition,
analyses its record for the threshold crossings it holds, stitches the
crossings of the interval into one timeline and pairs each falling crossing
with the rising one that closes it — nothing is inferred from the order of
triggers alone, and openings closer than 10 µs count as one bouncing
dropout. One acquisition is forced at the start of every interval so the
contact state at that moment is on record. Three files are written under
`results/(B) <DUT>/(B) Continuity/{date} (B) Continuity <test> ...`, flushed
after every row: `acquisitions.csv` (one row per trigger: scope timestamp,
the edge, the crossings in its record, min/max voltage — the raw evidence),
`dropouts.csv` (one row per dropout: start on the scope clock, duration in
µs, whether that duration is *exact*, *approx* — an edge fell into the scope's
few-µs blind time after a record, or the dropout spans a readout stop and
was measured on the scope's absolute clock — or *at least* — the closing was
never seen — the number of merged openings and a note), and `intervals.csv`
(each interval's start and stop on the PC clock, the acquisition count, the
readout dead time and whether the scope's memory filled). Ctrl-C stops after
reading out the current interval. Before the chamber closes, pull the
connector once by hand and check the dropouts file shows it.

## Channel requirements report

Run `run_s_parameters.py` once per DUT configuration (each attenuation
setting, bypass off and on), then

```bash
python scripts/report_channel.py            # every channel of the DUT in the edit block
python scripts/report_channel.py --simulate # demo: measures 4 configurations on the simulated bench first
```

For each channel it takes the latest result of every configuration, moves
them to the DUT plane, and answers the specification:

- **S21** — the **min gain** (highest attenuation, bypass on) and the **max
  gain** (no attenuation, bypass off), each read at the band centre; the
  **filter cutoff** — S21 at least X dB below the centre gain Y MHz below
  `f_start` and above `f_stop`; the **passband variation** — every in-band
  point within Z dB of the centre gain. X, Y and Z are the channel's
  `requirements` in its DUT definition (`ChannelRequirements`), and every
  configuration is checked. The S21 page shows every configuration over the
  measured range with min and max gain bold, the passband shaded, the cutoff
  frequencies as labelled ticks and the cutoff points as dots; a second panel
  shows the passband gain relative to the centre gain against the ±Z limit.
  Every number and verdict sits in a results column beside the plots, one row
  per configuration, so nothing has to be read off the curves.
- **S11** (and S22) — every configuration as a thin trace and their
  **average** (a power average of the dB values) bold on top, the worst
  in-band value of the average marked, and each configuration's worst in-band
  value in the column.

The text report and the figures are saved next to the S-parameter data as
`... report.txt`, `... report S21.png`, `... report S11.png`, `... report S22.png`.

The compression measurement is built for trustworthy numbers: the MXG's step
attenuator is held for the whole sweep (fixed at the smallest value that reaches
the start level, so only the ALC moves the level and no attenuator switch puts a
bump in the curve); the FSV reads with an RMS detector, trace averaging, and a
reference level tracked to about 10 dB above the signal; the analyzer's noise
floor is measured with the generator off and saved with the data; and both
instruments' error queues are checked. The sweep stops itself once the
moving-median gain has fallen `stop_compression` (2 dB by default) below its
peak, so the DUT is never driven harder than needed. The analysis derives the
small-signal gain from the plateau of the smoothed gain curve — never from
"the first few points" — and ignores readings closer than `noise_margin` to
the noise floor.

## Components and reference planes

Raw data is saved **at the instrument connectors**. `rflab.analysis.at_dut`
moves a result to the DUT ports using the paths recorded in it, evaluating each
component's loss table (interpolated) at each row's own frequency — so a 3rd
harmonic at 1.8 GHz is corrected with the loss at 1.8 GHz, an S11 gets back
twice the input-path loss, and an S21 the input plus the output path loss. Noise figure is the exception: the K30 application
needs the losses to compute NF, so they are loaded onto the analyzer as
frequency tables from the channel's paths before measuring, and that result is
saved already at the DUT plane.

To add or re-characterize a component, connect it between the VNA ports,
edit the block at the top of `scripts/characterize_component.py` (the
component's name, the **fixture** — the female-to-female adapters or other
parts that had to be in the measurement but are not part of the component —
and the sweep) and run it:

```bash
python scripts/characterize_component.py
```

The fixture's loss is de-embedded from the measurement; the file keeps both
the de-embedded and the raw loss and names the fixture in its header. Then
register the new file in `rflab/components/__init__.py`. Keep the old file and
entry: results record components as `"name (date)"`, and old results resolve
to the characterization that was current when they were measured.

## Adding a DUT

Copy `rflab/duts/amplifier_x.py`, set the bands, the signal path on each
port and the requirements per channel, and name the module in the scripts'
edit blocks.
