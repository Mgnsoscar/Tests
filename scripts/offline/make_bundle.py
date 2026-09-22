"""Build a USB-stick bundle that installs LabKit and rflab on a computer without internet.

Run this on a computer *with* internet, from the rflab checkout, with the
LabKit checkout next to it (or point at it with ``--labkit``):

    python scripts/offline/make_bundle.py --out /path/to/usb/rflab-bundle
    python scripts/offline/make_bundle.py --out D:\\rflab-bundle --platform win_amd64 --python 3.11

The bundle holds the LabKit wheel and every dependency as a wheel built for
the lab computer's platform and Python version (``--platform`` and
``--python``; the defaults are 64-bit Windows and Python 3.11), a copy of
both repositories, the installer ``install.py`` and a README. On the lab
computer ``python install.py --venv ... --project ...`` creates a virtual
environment with LabKit inside it and puts the rflab project folder where
you want it — see the README it writes.

The wheels are downloaded with pip's cross-platform options, so this works
from a Linux or macOS machine for a Windows lab computer; ``--platform
current`` bundles for the machine you are on.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
RFLAB_ROOT = HERE.parent.parent

#: Needed beyond what the two pyproject files declare: the pure-Python VISA
#: backend, so no NI-VISA is required for the Ethernet instruments.
EXTRA_REQUIREMENTS = ["pyvisa-py>=0.7"]

#: What is copied out of each repository: everything git tracks, or, without
#: git, everything but these.
IGNORED = shutil.ignore_patterns(
    ".git", "__pycache__", "*.pyc", ".mypy_cache", ".pytest_cache", ".venv", "venv", "site",
    "results", "results-simulated", "*.egg-info", "build", "dist",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="the bundle folder to create (on the USB stick, or anywhere)")
    p.add_argument("--labkit", default=None, help="the LabKit checkout (default: ../LabKit next to this project)")
    p.add_argument("--platform", default="win_amd64",
                   help="pip platform tag of the lab computer: win_amd64 (default), manylinux_2_17_x86_64, "
                        "macosx_11_0_arm64, ... or 'current' for this machine")
    p.add_argument("--python", default="3.11", help="the lab computer's Python major.minor (default 3.11)")
    args = p.parse_args()

    labkit = Path(args.labkit).resolve() if args.labkit else RFLAB_ROOT.parent / "LabKit"
    if not (labkit / "pyproject.toml").exists():
        sys.exit(f"LabKit checkout not found at {labkit}; pass --labkit")
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty; choose a new folder")
    out.mkdir(parents=True, exist_ok=True)

    print(f"copying LabKit from {labkit}")
    copy_repo(labkit, out / "LabKit")
    print(f"copying rflab from {RFLAB_ROOT}")
    copy_repo(RFLAB_ROOT, out / "rflab")

    print("building the LabKit wheel")
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(out / "wheelhouse"), str(labkit)], check=True)
    requirements = collect_requirements(labkit, RFLAB_ROOT) + EXTRA_REQUIREMENTS
    (out / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    print(f"downloading {len(requirements)} requirements for {args.platform}, Python {args.python} ...")
    download_wheels(requirements, out / "wheelhouse", args.platform, args.python)

    shutil.copy2(HERE / "install.py", out / "install.py")
    (out / "bundle.json").write_text(json.dumps({
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "python": args.python,
        "platform": args.platform,
        "labkit_commit": git_commit(labkit),
        "rflab_commit": git_commit(RFLAB_ROOT),
    }, indent=2), encoding="utf-8")
    (out / "README.txt").write_text(README, encoding="utf-8")

    wheels = sorted(out.glob("wheelhouse/*"))
    size = sum(w.stat().st_size for w in wheels) / 1e6
    print(f"bundle ready at {out}: {len(wheels)} wheels, {size:.0f} MB")
    print("on the lab computer:  python install.py --venv <env folder> --project <project folder>")


def copy_repo(source: Path, target: Path) -> None:
    """Copy the files git tracks (plus uncommitted edits to them); without git, copy everything sensible."""
    tracked = git_tracked_files(source)
    if tracked is None:
        shutil.copytree(source, target, ignore=IGNORED)
        return
    for relative in tracked:
        path = source / relative
        if path.is_file():
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / relative)


def git_tracked_files(repo: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in result.stdout.splitlines() if line]


def git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_requirements(labkit: Path, rflab: Path) -> list[str]:
    """The third-party requirements of both projects, all extras included (self-references dropped)."""
    requirements: list[str] = []
    for root, name in ((labkit, "labkit"), (rflab, "rflab")):
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        candidates = list(project.get("dependencies", []))
        for extra, items in project.get("optional-dependencies", {}).items():
            if extra != "docs":
                candidates.extend(items)
        for item in candidates:
            base = item.split("[")[0].split(" ")[0].split(">")[0].split("=")[0].strip().lower()
            if base in ("labkit", "rflab"):
                continue                      # our own packages: installed from the bundle's copies
            if item not in requirements:
                requirements.append(item)
    return requirements


def download_wheels(requirements: list[str], wheelhouse: Path, platform: str, python: str) -> None:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pip", "download", "--dest", str(wheelhouse), "--only-binary=:all:"]
    if platform != "current":
        abi = "cp" + python.replace(".", "")
        command += ["--platform", platform, "--python-version", python, "--implementation", "cp", "--abi", abi]
    subprocess.run(command + requirements, check=True)
    sources = list(wheelhouse.glob("*.tar.gz")) + list(wheelhouse.glob("*.zip"))
    if sources:
        sys.exit(f"no binary wheel for {', '.join(s.name for s in sources)}; the lab computer could not build them")


README = """rflab offline bundle
====================

Contents
  wheelhouse/        LabKit and every dependency as wheels for the lab computer (bundle.json: platform, Python)
  rflab/             the measurement project: rflab package, scripts, tests
  LabKit/            the LabKit source, for reading and its docs (the install uses the wheel)
  install.py         the installer
  requirements.txt   what the wheelhouse holds

Install on the lab computer
  1. Check the Python version matches bundle.json (python --version).
  2. Open a terminal in this folder and run, with folders of your choosing:

        python install.py --venv C:\\labkit-env --project C:\\rflab

     --venv     creates a virtual environment there with LabKit, its plotting and
                instrument extras, pyvisa-py, pytest and mypy inside it
     --project  copies the rflab project folder there (scripts, DUTs, tests; results go under it)

     Without --venv the packages go into the Python you ran the installer with.
     Without --project the project stays in this bundle folder.

  3. Use it: activate the environment, go to the project, run the scripts.

        C:\\labkit-env\\Scripts\\activate
        cd C:\\rflab
        python scripts\\monitor_continuity.py --simulate      no instruments needed
        python -m pytest

Other projects
  Any folder can use LabKit the same way: activate the same environment (or
  choose its interpreter in your editor) and import labkit. Nothing about rflab
  is installed in the environment; the project folder is self-contained.
  To make another environment later, run install.py again with another --venv;
  the bundle keeps working offline.

Instruments
  pyvisa-py is a pure-Python VISA backend that talks to the Ethernet instruments
  without NI-VISA. If NI-VISA or R&S VISA is installed, PyVISA prefers it.
"""


if __name__ == "__main__":
    main()
