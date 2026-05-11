"""Run the unittest suite with MATLAB parity tests enabled."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


MATLAB = Path(os.environ.get("MATLAB", r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"))
FLAM_MARKERS = (
    Path("rskelf") / "rskelf.m",
    Path("rskel") / "rskel.m",
    Path("ifmm") / "ifmm.m",
    Path("mf") / "mf2.m",
    Path("hifie") / "hifie2.m",
    Path("hifde") / "hifde2.m",
)


def _complete_flam_reference(path: Path) -> bool:
    return path.exists() and all((path / marker).exists() for marker in FLAM_MARKERS)


def _default_flam_reference() -> Path:
    env_ref = os.environ.get("FLAM_REFERENCE")
    if env_ref:
        return Path(env_ref)
    candidates = [
        Path(tempfile.gettempdir()) / "flam-reference",
        Path(tempfile.gettempdir()) / "FLAM-ref",
        Path.home() / "git" / "FLAM",
    ]
    for path in candidates:
        if _complete_flam_reference(path):
            return path
    return candidates[0]


def main() -> int:
    flam_ref = _default_flam_reference()
    chunkie_ref = Path(os.environ.get("CHUNKIE_REFERENCE", Path.home() / "git" / "chunkie"))
    if not MATLAB.exists():
        print(f"MATLAB executable not found: {MATLAB}", file=sys.stderr)
        return 2
    if not _complete_flam_reference(flam_ref):
        missing = [str(flam_ref / marker) for marker in FLAM_MARKERS if not (flam_ref / marker).exists()]
        print("FLAM reference checkout is missing required entry points:", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
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
