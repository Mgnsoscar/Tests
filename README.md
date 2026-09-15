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
  components/         the characterized cables, pads, couplers (+ data/ CSVs)
  measurements/       acquisition: compression, noise_figure, s_parameters, harmonics, intermodulation
  store.py            "(B) " folders, "{date} (B) " files, CSV save + load
  analysis/           at_dut() reference-plane correction, summaries, plots
  simulation.py       a simulated bench so everything runs without hardware
scripts/
  run_<measurement>.py   measure -> save [-> analyse]
  analyze.py             load -> at_dut -> summarize -> plot, for saved results
  characterize_component.py   measure a component on the VNA into components/data/
tests/                pytest suite against the simulated bench
results/              the data (not committed)
```

## Workflow

```bash
pip install -e ".[dev]"                          # pulls LabKit from GitHub

python scripts/run_compression.py --dut amplifier_x --channel 1 --plot
python scripts/run_noise_figure.py --channel 1 2
python scripts/run_s_parameters.py
python scripts/run_harmonics.py --channel 1
python scripts/run_intermodulation.py --channel 1

python scripts/analyze.py --all --measurement Compression    # post-process later, no instruments
python scripts/analyze.py "results/(B) Amplifier X v1.0/(B) Harmonics/2026-09-15 (B) Harmonics Ch1.csv" --show
```

Add `--simulate` to any run script to exercise it against the simulated bench.

Results land in `results/(B) <DUT> <version>/(B) <Measurement>/{date} (B) <Measurement> Ch<n>.csv`,
with a header that records the DUT, channel, timestamp, reference plane, the
signal path on every port (each component with its characterization date), the
instruments' `*IDN?` strings and the settings. Figures are saved next to the
data with the same name.

## Components and reference planes

Raw data is saved **at the instrument connectors**. `rflab.analysis.at_dut`
moves a result to the DUT ports using the paths recorded in it, evaluating each
component's loss table (interpolated) at each row's own frequency — so a 3rd
harmonic at 6.9 GHz is corrected with the loss at 6.9 GHz, and an S11 gets back
twice the input-path loss. Noise figure is the exception: the K30 application
needs the losses to compute NF, so they are loaded onto the analyzer as
frequency tables from the channel's paths before measuring, and that result is
saved already at the DUT plane.

To add or re-characterize a component:

```bash
python scripts/characterize_component.py "SMA cable A" --start 100MHz --stop 12GHz
```

then register the new file in `rflab/components/__init__.py`. Keep the old
file and entry: results record components as `"name (date)"`, and old results
resolve to the characterization that was current when they were measured.

## Adding a DUT

Copy `rflab/duts/amplifier_x.py`, set the bands and the signal path on each
port per channel, and run the scripts with `--dut <module name>`.
