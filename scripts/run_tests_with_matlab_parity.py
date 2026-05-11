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
    if not MATLAB.exists():
        print(f"MATLAB executable not found: {MATLAB}", file=sys.stderr)
        return 2
    if not flam_ref.exists():
        print(f"FLAM reference checkout not found: {flam_ref}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["PYFLAM_RUN_MATLAB_PARITY"] = "1"
    env["FLAM_REFERENCE"] = str(flam_ref)
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
