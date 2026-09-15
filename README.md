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
  simulation.py       a simulated bench so everything runs without hardware
scripts/
  run_<measurement>.py   edit the block at the top, then: measure -> save [-> analyse]
  analyze.py             load -> at_dut -> summarize -> plot, for saved results
  characterize_component.py   measure a component on the VNA (fixture de-embedded) into components/data/
tests/                pytest suite against the simulated bench
results/              the data (not committed)
```

## Workflow

```bash
pip install -e ".[dev]"                          # pulls LabKit from GitHub
```

Each run script has an **"Edit before running"** block at its top: the DUT
module, the channels, the DUT's own attenuation setting (which you set on the
device by hand, and the script records), and the measurement settings.
Edit it, then run:

```bash
python scripts/run_compression.py --plot
python scripts/run_noise_figure.py
python scripts/run_s_parameters.py
python scripts/run_harmonics.py
python scripts/run_intermodulation.py

python scripts/analyze.py --all --measurement Compression    # post-process later, no instruments
python scripts/analyze.py "results/(B) Amplifier X v1.0/(B) Harmonics/2026-09-15 (B) Harmonics Ch1 attenuation 0 dB.csv" --show
```

Add `--simulate` to any script to exercise it against the simulated bench.

Results land in
`results/(B) <DUT> <version>/(B) <Measurement>/{date} (B) <Measurement> Ch<n> attenuation <x> dB.csv`,
with a header that records the DUT, channel, timestamp, reference plane, the
signal path on every port (each component with its characterization date), the
instruments' `*IDN?` strings, the DUT state and the settings. Figures are saved
next to the data with the same name. A second run with the same name on the
same day gets a `(2)` suffix rather than overwriting.

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
harmonic at 6.9 GHz is corrected with the loss at 6.9 GHz, and an S11 gets back
twice the input-path loss. Noise figure is the exception: the K30 application
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

Copy `rflab/duts/amplifier_x.py`, set the bands and the signal path on each
port per channel, and run the scripts with `--dut <module name>`.
