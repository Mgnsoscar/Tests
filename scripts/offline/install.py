"""Install LabKit from this offline bundle and set up the rflab project folder, no internet needed.

In the bundle folder, with the Python the bundle was made for:

    python install.py --venv C:\\labkit-env --project C:\\rflab

--venv     creates a virtual environment there and installs LabKit (with plotting,
           instruments, pyvisa-py, pytest and mypy) into it; without it the
           packages go into the Python running this script
--project  copies the rflab project folder there; without it the project stays
           in the bundle folder

Nothing about rflab is installed into the environment: the project folder is
self-contained and any environment with LabKit can run it. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WHEELHOUSE = HERE / "wheelhouse"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--venv", default=None, help="create this virtual environment and install into it")
    p.add_argument("--project", default=None, help="copy the rflab project folder to this path")
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
        python = str(venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))

    print("installing LabKit with plotting, instruments, pyvisa-py and the test tools")
    run([python, "-m", "pip", "install", "--no-index", "--find-links", str(WHEELHOUSE),
         "labkit[all]", "pyvisa-py", "pytest", "mypy"])

    project = HERE / "rflab"
    if args.project:
        project = Path(args.project).resolve()
        if project.exists() and any(project.iterdir()):
            sys.exit(f"{project} exists and is not empty; choose a new folder or delete it first")
        print(f"copying the rflab project to {project}")
        shutil.copytree(HERE / "rflab", project, dirs_exist_ok=True)

    print("checking the install")
    run([python, "-c", CHECK], cwd=project)
    print()
    print("done.")
    if args.venv:
        activate = f"{args.venv}\\Scripts\\activate" if sys.platform == "win32" else f"source {args.venv}/bin/activate"
        print(f"in every new terminal:  {activate}")
    print(f"then:  cd {project}  and  python scripts/monitor_continuity.py --simulate")


CHECK = """
import labkit, pyvisa, numpy, matplotlib, pint
import rflab
from labkit.instruments import RTO64
print("  labkit  ", labkit.__file__)
print("  project ", rflab.__file__)
print("  numpy", numpy.__version__, "matplotlib", matplotlib.__version__, "pint", pint.__version__, "pyvisa", pyvisa.__version__)
print("  VISA backend:", pyvisa.ResourceManager().visalib)
"""


def run(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"failed: {' '.join(command)}")


if __name__ == "__main__":
    main()
