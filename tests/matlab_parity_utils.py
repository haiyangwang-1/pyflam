"""Shared helpers for MATLAB/FLAM parity tests."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np
import scipy.io


REPO_ROOT = Path(__file__).resolve().parents[1]
MATLAB = Path(os.environ.get("MATLAB", r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"))
FLAM_MARKERS = (
    Path("rskelf") / "rskelf.m",
    Path("rskel") / "rskel.m",
    Path("ifmm") / "ifmm.m",
    Path("mf") / "mf2.m",
    Path("hifie") / "hifie2.m",
    Path("hifde") / "hifde2.m",
)


def matlab_path(path: Path) -> str:
    """Return a MATLAB-safe path literal body."""

    return str(path).replace("\\", "/").replace("'", "''")


def require_paths(*paths: Path, label: str = "MATLAB parity") -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"{label} requires: " + ", ".join(missing))


def require_flam_reference(path: Path, label: str = "MATLAB parity") -> None:
    require_paths(path, *(path / marker for marker in FLAM_MARKERS), label=f"{label} FLAM reference")
    require_pinned_reference(path, "flam", label=label)


def load_reference_dependencies(pyproject: Path | None = None) -> dict[str, dict]:
    """Load pinned external MATLAB parity reference metadata from pyproject."""

    pyproject = REPO_ROOT / "pyproject.toml" if pyproject is None else pyproject
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    return data.get("tool", {}).get("pyflam", {}).get("test-reference-dependencies", {})


def require_pinned_reference(path: Path, key: str, label: str = "MATLAB parity") -> None:
    """Require a local reference checkout to match the version pinned by PyFLAM."""

    deps = load_reference_dependencies()
    if key not in deps:
        raise RuntimeError(f"{label} has no pinned reference dependency named {key!r}")
    dep = deps[key]
    name = dep.get("name", key)
    expected_commit = dep.get("commit")
    require_paths(path, label=f"{label} {name} pinned reference")
    head = _git_stdout(path, "rev-parse", "HEAD", label=label, name=name).strip()
    if head != expected_commit:
        raise RuntimeError(
            f"{label} requires pinned {name} commit {expected_commit}, but {path} is at {head}. "
            f"Run `git -C {path} checkout {expected_commit}`."
        )

    if dep.get("required_clean"):
        status = _git_stdout(path, "status", "--porcelain", label=label, name=name)
        if status.strip():
            raise RuntimeError(f"{label} requires pinned {name} checkout {path} to be clean.")


def _git_stdout(path: Path, *args: str, label: str, name: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{label} requires pinned {name} git checkout at {path}: {detail}")
    return result.stdout


def default_flam_reference() -> Path:
    """Return a FLAM checkout containing the public entry-point m-files."""

    env_ref = os.environ.get("FLAM_REFERENCE")
    if env_ref:
        return Path(env_ref)

    candidates = [
        Path(tempfile.gettempdir()) / "flam-reference",
        Path(tempfile.gettempdir()) / "FLAM-ref",
        Path.home() / "git" / "FLAM",
    ]
    for path in candidates:
        if all((path / marker).exists() for marker in FLAM_MARKERS):
            return path
    return candidates[0]


def run_matlab_script(script: Path, timeout: int = 120) -> None:
    attempts = max(1, int(os.environ.get("PYFLAM_MATLAB_RETRIES", "2")) + 1)
    cmd = [str(MATLAB), "-batch", f"run('{script.as_posix()}')"]
    result = None
    for attempt in range(attempts):
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return
        if attempt + 1 < attempts:
            time.sleep(1.0)
    assert result is not None
    raise RuntimeError(
        f"MATLAB script failed with exit code {result.returncode} after {attempts} attempts: {script}\n"
        f"{result.stdout}"
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


def factor_metadata_code(factor_name: str = "F", metadata_name: str = "factor_meta") -> str:
    """Return MATLAB code that stores common FLAM factor metadata in a struct."""

    return f"""
{metadata_name} = struct();
if isfield({factor_name},'nlvl'), {metadata_name}.nlvl = {factor_name}.nlvl; else, {metadata_name}.nlvl = -1; end
if isfield({factor_name},'lvp'), {metadata_name}.lvp = {factor_name}.lvp; else, {metadata_name}.lvp = []; end
if isfield({factor_name},'lvpd'), {metadata_name}.lvpd = {factor_name}.lvpd; else, {metadata_name}.lvpd = []; end
if isfield({factor_name},'lvpu'), {metadata_name}.lvpu = {factor_name}.lvpu; else, {metadata_name}.lvpu = []; end
if isfield({factor_name},'lvpb'), {metadata_name}.lvpb = {factor_name}.lvpb; else, {metadata_name}.lvpb = []; end
if isfield({factor_name},'factors'), {metadata_name}.nfactors = length({factor_name}.factors); else, {metadata_name}.nfactors = 0; end
if isfield({factor_name},'D'), {metadata_name}.nd = length({factor_name}.D); else, {metadata_name}.nd = 0; end
if isfield({factor_name},'U'), {metadata_name}.nu = length({factor_name}.U); else, {metadata_name}.nu = 0; end
if isfield({factor_name},'B'), {metadata_name}.nb = length({factor_name}.B); else, {metadata_name}.nb = 0; end
if isfield({factor_name},'Si'), {metadata_name}.nsi = length({factor_name}.Si); else, {metadata_name}.nsi = 0; end
if isfield({factor_name},'S'), {metadata_name}.s_nnz = nnz({factor_name}.S); else, {metadata_name}.s_nnz = 0; end
"""


def load_factor_metadata(data, metadata_name: str = "factor_meta") -> dict[str, np.ndarray | int]:
    """Convert a MATLAB metadata struct loaded by SciPy into a simple dict."""

    raw = np.asarray(data[metadata_name]).reshape(-1)[0]
    out = {}
    for field in raw.dtype.names or ():
        value = raw[field]
        arr = np.asarray(value)
        if arr.size == 1:
            scalar = arr.reshape(-1)[0]
            out[field] = int(scalar) if np.issubdtype(arr.dtype, np.integer) else scalar.item()
        else:
            out[field] = arr.reshape(-1)
    return out


def relerr(a, b) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def logdet_mod_error(a, b) -> float:
    diff = np.asarray(a).item() - np.asarray(b).item()
    if abs(diff.imag):
        diff = diff - 2j * np.pi * np.round(diff.imag / (2 * np.pi))
    return float(abs(diff))
