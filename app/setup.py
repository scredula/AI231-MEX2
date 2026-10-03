#!/usr/bin/env python3
"""
Bootstrap setup for the "Hey Mason" demo app.

Running:

    python setup.py

will:

  1. create a virtual environment at  ./venv   (next to this file)
  2. upgrade pip and install everything in  requirements.txt
  3. (re)generate demo assets if they are missing  ->  assets/jpgs, assets/audio
  4. print exactly how to launch the app

Options:
    python setup.py --recreate     # delete and rebuild ./venv
    python setup.py --no-assets    # skip asset generation
    python setup.py --selftest     # run the headless self-test afterwards

On Windows:   py setup.py        then run  run.bat
On Linux/Mac: python3 setup.py   then run  ./run.sh
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_DIR = HERE / "venv"
REQUIREMENTS = HERE / "requirements.txt"


def venv_python(venv_dir: Path) -> Path:
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(cmd: list[str]) -> int:
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.call(cmd)


def ensure_venv(recreate: bool) -> Path:
    py = venv_python(VENV_DIR)
    if recreate and VENV_DIR.exists():
        print(f"Removing existing venv: {VENV_DIR}")
        shutil.rmtree(VENV_DIR)
    if not py.exists():
        print(f"Creating virtual environment at {VENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(str(VENV_DIR))
    else:
        print(f"Reusing existing virtual environment: {VENV_DIR}")
    return py


def main() -> int:
    ap = argparse.ArgumentParser(description="Set up the Hey Mason demo app")
    ap.add_argument("--recreate", action="store_true", help="delete and rebuild ./venv")
    ap.add_argument("--no-assets", action="store_true", help="skip asset generation")
    ap.add_argument("--selftest", action="store_true", help="run the headless self-test after install")
    args = ap.parse_args()

    if not REQUIREMENTS.exists():
        print(f"ERROR: {REQUIREMENTS} not found next to setup.py")
        return 1

    if os.environ.get("VIRTUAL_ENV"):
        print("Note: a virtual environment is currently active; setup still targets ./venv.")

    py = ensure_venv(args.recreate)

    print("Upgrading pip / wheel / setuptools …")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])

    print("Installing requirements …")
    if run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)]) != 0:
        print("ERROR: dependency installation failed.")
        return 1

    if not args.no_assets:
        have_assets = (HERE / "assets" / "jpgs").exists() and (HERE / "assets" / "audio").exists()
        have_music = any((HERE / "music").glob("*.wav")) or any((HERE / "music").glob("*.mp3"))
        if have_assets and have_music:
            print("Demo assets + music already present.")
        else:
            print("Generating demo assets (JPGs / weather.mp3 / demo music) …")
            run([str(py), str(HERE / "generate_assets.py")])

    if args.selftest:
        print("Running self-test …")
        run([str(py), str(HERE / "selftest.py")])

    # ---- done ----
    print("\n" + "=" * 64)
    print("Setup complete!")
    print("=" * 64)
    if platform.system() == "Windows":
        print("Launch the demo with:")
        print(r"    venv\Scripts\python app.py")
        print(r"    venv\Scripts\python app.py --model DS-CNN   (or TC-ResNet / MatchboxNet / VGG)")
    else:
        print("Launch the demo with:")
        print("    ./venv/bin/python app.py")
        print("    ./venv/bin/python app.py --model DS-CNN    (or TC-ResNet / MatchboxNet / VGG)")
    print("\nList available models with:  python app.py --list-models")
    print("Add songs to app/music/ to test the music commands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
