"""Shared helpers for MATLAB/FLAM parity tests."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io


MATLAB = Path(r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe")


def matlab_path(path: Path) -> str:
    """Return a MATLAB-safe path literal body."""

    return str(path).replace("\\", "/").replace("'", "''")


def require_paths(*paths: Path, label: str = "MATLAB parity") -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"{label} requires: " + ", ".join(missing))


def run_matlab_script(script: Path, timeout: int = 120) -> None:
    subprocess.run(
        [str(MATLAB), "-batch", f"run('{script.as_posix()}')"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def run_matlab_export(name: str, script_body: str, timeout: int = 120):
    """Run a MATLAB script body that saves results to ``__OUT__``."""

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        out = tmpdir / f"{name}.mat"
        script = tmpdir / f"run_{name}.m"
        script.write_text(script_body.replace("__OUT__", matlab_path(out)))
        run_matlab_script(script, timeout)
        return scipy.io.loadmat(out)


def relerr(a, b) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def logdet_mod_error(a, b) -> float:
    diff = np.asarray(a).item() - np.asarray(b).item()
    if abs(diff.imag):
        diff = diff - 2j * np.pi * np.round(diff.imag / (2 * np.pi))
    return float(abs(diff))
