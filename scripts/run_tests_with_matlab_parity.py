"""Run the unittest suite with MATLAB parity tests enabled."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


MATLAB = Path(r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe")


def main() -> int:
    flam_ref = Path(os.environ.get("FLAM_REFERENCE", Path(tempfile.gettempdir()) / "FLAM-ref"))
    chunkie_ref = Path(os.environ.get("CHUNKIE_REFERENCE", r"C:\Users\haiya\git\chunkie"))
    if not MATLAB.exists():
        print(f"MATLAB executable not found: {MATLAB}", file=sys.stderr)
        return 2
    if not flam_ref.exists():
        print(f"FLAM reference checkout not found: {flam_ref}", file=sys.stderr)
        return 2
    if not chunkie_ref.exists():
        print(f"ChunkIE reference checkout not found: {chunkie_ref}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["PYFLAM_RUN_MATLAB_PARITY"] = "1"
    env["FLAM_REFERENCE"] = str(flam_ref)
    env["CHUNKIE_REFERENCE"] = str(chunkie_ref)
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
