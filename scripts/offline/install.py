"""Install LabKit and rflab from this offline bundle, no internet needed.

Copy the bundle folder onto the computer's disk first (the packages are
installed in place, so the folder has to stay), then, in the folder:

    python install.py                 install into this Python
    python install.py --venv .venv    create a virtual environment here and install into it

Standard library only; runs with the Python the bundle was made for.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WHEELHOUSE = HERE / "wheelhouse"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--venv", default=None, help="create this virtual environment and install into it")
    args = p.parse_args()

    info = json.loads((HERE / "bundle.json").read_text(encoding="utf-8"))
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    if running != info["python"]:
        sys.exit(
            f"this bundle was made for Python {info['python']} ({info['platform']}); "
            f"you are running {platform.python_version()}. Start the installer with the matching Python."
        )
    if not WHEELHOUSE.is_dir():
        sys.exit(f"no wheelhouse folder next to {Path(__file__).name}; is this the bundle folder?")

    python = sys.executable
    if args.venv:
        venv = Path(args.venv).resolve()
        print(f"creating virtual environment {venv}")
        run([sys.executable, "-m", "venv", str(venv)])
        python = str(venv / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python"))

    offline = [python, "-m", "pip", "install", "--no-index", "--find-links", str(WHEELHOUSE)]
    print("installing the build backend")
    run(offline + ["hatchling", "editables"])
    print("installing LabKit with plotting, instruments and the test tools")
    run(offline + ["--no-build-isolation", "-e", f"{HERE / 'LabKit'}[dev]"])
    print("installing the pure-Python VISA backend")
    run(offline + ["pyvisa-py"])
    print("installing rflab")
    run(offline + ["--no-build-isolation", "--no-deps", "-e", str(HERE / "rflab")])
    run(offline + ["pytest", "mypy"])

    print("checking the install")
    run([python, "-c", CHECK])
    print()
    print(f"done. LabKit and rflab are installed from {HERE} — keep this folder where it is.")
    if args.venv:
        activate = (Path(args.venv) / "Scripts" / "activate") if sys.platform == "win32" else f"source {args.venv}/bin/activate"
        print(f"activate the environment first in every new terminal:  {activate}")
    print(f"try:  cd {HERE / 'rflab'}  then  python scripts/monitor_continuity.py --simulate")


CHECK = """
import labkit, rflab, pyvisa, numpy, matplotlib, pint
from labkit.instruments import RTO64
print("  labkit", labkit.__file__)
print("  rflab ", rflab.__file__)
print("  numpy", numpy.__version__, "matplotlib", matplotlib.__version__, "pint", pint.__version__, "pyvisa", pyvisa.__version__)
rm = pyvisa.ResourceManager()
print("  VISA backend:", rm.visalib)
"""


def run(command: list[str]) -> None:
    result = subprocess.run(command)
    if result.returncode != 0:
        sys.exit(f"failed: {' '.join(command)}")


if __name__ == "__main__":
    main()
