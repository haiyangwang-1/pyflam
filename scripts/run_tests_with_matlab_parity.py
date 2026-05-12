"""Run the unittest suite with MATLAB parity tests enabled."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from matlab_parity_utils import (  # noqa: E402
    MATLAB,
    default_chunkie_reference,
    default_flam_reference,
    require_flam_reference,
    require_paths,
    require_pinned_reference,
)


def main() -> int:
    flam_ref = default_flam_reference()
    chunkie_ref = default_chunkie_reference()
    try:
        require_paths(MATLAB, flam_ref, chunkie_ref, label="MATLAB parity runner")
        require_flam_reference(flam_ref, label="MATLAB parity runner")
        require_pinned_reference(chunkie_ref, "chunkie", label="MATLAB parity runner")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["PYFLAM_RUN_MATLAB_PARITY"] = "1"
    env["FLAM_REFERENCE"] = str(flam_ref)
    env["CHUNKIE_REFERENCE"] = str(chunkie_ref)
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
