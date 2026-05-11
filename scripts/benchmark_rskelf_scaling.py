"""ChunkIE/PyFLAM rskelf scaling benchmark.

For each requested size, ChunkIE assembles a dense boundary integral operator.
MATLAB FLAM and PyFLAM then factor the same dense matrix. Build is timed once;
matvec, solve, and logdet are repeated and reported as mean +/- std.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import time
from typing import Callable

import numpy as np
import scipy.io

from pyflam import rskelf, rskelf_logdet, rskelf_mv, rskelf_sv
from benchmark_rskelf_parity import DEFAULT_CHUNKIE, DEFAULT_MATLAB, _matlab_path, logdet_errors, relerr


def parse_sizes(text: str) -> list[int]:
    sizes = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not sizes:
        raise ValueError("at least one size is required")
    return sizes


def stats(values: list[float] | np.ndarray) -> dict[str, float | list[float]]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "samples": [float(x) for x in arr],
    }


def time_repeated(fn: Callable[[], object], repeats: int, warmups: int) -> tuple[object, list[float]]:
    result = None
    for _ in range(warmups):
        result = fn()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = fn()
            times.append(time.perf_counter() - t0)
    finally:
        if gc_was_enabled:
            gc.enable()
    return result, times


def write_matlab_driver(
    path: Path,
    chunkie: Path,
    out: Path,
    n: int,
    k: int,
    occ: int,
    tol: float,
    nrhs: int,
    repeats: int,
    warmups: int,
) -> None:
    if n % k:
        raise ValueError(f"size {n} is not divisible by k={k}; choose a multiple of k")
    nch = n // k
    path.write_text(
        textwrap.dedent(
            f"""
            cd('{_matlab_path(chunkie)}');
            startup(struct('testfmm', false));
            rng(1234);

            n_target = {n};
            k = {k};
            nch = {nch};
            occ = {occ};
            tol = {tol:.17g};
            nrhs = {nrhs};
            repeats = {repeats};
            warmups = {warmups};

            pref = []; pref.k = k;
            chnkr = chunkerfuncuni(@(t) starfish(t,5,0.15), nch, [], pref);
            kernd = -2*kernel('helm','d',10);

            tm_assemble = tic;
            Ad = eye(chnkr.npt) + chunkermat(chnkr, kernd);
            assemble_time = toc(tm_assemble);

            xflam = real(chnkr.r(:,:));
            n = size(Ad,1);
            X = reshape(sin((1:(nrhs*n))/37), n, nrhs);
            b = X(:,1);
            A = @(i,j) Ad(i,j);

            tm_build = tic;
            F = rskelf(A, xflam, occ, tol, [], struct('symm','n'));
            matlab_build_time = toc(tm_build);

            for rr = 1:warmups
                rskelf_mv(F, X);
                rskelf_sv(F, X);
                rskelf_logdet(F);
            end

            matlab_mv_times = zeros(repeats,1);
            matlab_sv_times = zeros(repeats,1);
            matlab_logdet_times = zeros(repeats,1);
            for rr = 1:repeats
                tm = tic; Ymv = rskelf_mv(F, X); matlab_mv_times(rr) = toc(tm);
                tm = tic; Ysv = rskelf_sv(F, X); matlab_sv_times(rr) = toc(tm);
                tm = tic; ld = rskelf_logdet(F); matlab_logdet_times(rr) = toc(tm);
            end

            dense_resid = norm(Ad*Ysv(:,1) - b) / norm(b);
            save('{_matlab_path(out)}', 'n', 'k', 'nch', 'occ', 'tol', 'nrhs', ...
                 'xflam', 'Ad', 'X', 'Ymv', 'Ysv', 'ld', 'dense_resid', ...
                 'assemble_time', 'matlab_build_time', 'matlab_mv_times', ...
                 'matlab_sv_times', 'matlab_logdet_times', '-v7');
            exit;
            """
        )
    )


def run_matlab(matlab: Path, script: Path, timeout: int) -> str:
    result = subprocess.run(
        [str(matlab), "-batch", f"run('{script.as_posix()}')"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def estimate_dense_gib(n: int, complex_matrix: bool = True) -> float:
    bytes_per_entry = 16 if complex_matrix else 8
    return n * n * bytes_per_entry / (1024**3)


def run_size(args: argparse.Namespace, n: int) -> dict:
    if estimate_dense_gib(n) > args.max_dense_gib and not args.force:
        return {
            "n": n,
            "skipped": True,
            "reason": f"estimated dense matrix {estimate_dense_gib(n):.2f} GiB exceeds --max-dense-gib={args.max_dense_gib}",
        }

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        mat_out = tmpdir / f"chunkie_rskelf_scaling_{n}.mat"
        driver = tmpdir / f"run_chunkie_rskelf_scaling_{n}.m"
        write_matlab_driver(driver, args.chunkie, mat_out, n, args.k, args.occ, args.tol, args.nrhs, args.repeats, args.warmups)
        matlab_stdout = run_matlab(args.matlab, driver, args.timeout)
        data = scipy.io.loadmat(mat_out)

    A = np.asarray(data["Ad"])
    x = np.asarray(data["xflam"])
    X = np.asarray(data["X"])
    occ = int(data["occ"].item())
    tol = float(data["tol"].item())

    t0 = time.perf_counter()
    F = rskelf(A, x, occ, tol, opts={"symm": "n"})
    python_build_time = time.perf_counter() - t0

    Ymv_py, python_mv_times = time_repeated(lambda: rskelf_mv(F, X), args.repeats, args.warmups)
    Ysv_py, python_sv_times = time_repeated(lambda: rskelf_sv(F, X), args.repeats, args.warmups)
    ld_py, python_logdet_times = time_repeated(lambda: rskelf_logdet(F), args.repeats, args.warmups)

    b = X[:, [0]]
    errors = {
        "mv_vs_matlab": relerr(Ymv_py, data["Ymv"]),
        "sv_vs_matlab": relerr(Ysv_py, data["Ysv"]),
        "mv_vs_dense": relerr(Ymv_py, A @ X),
        "sv_residual": relerr(A @ Ysv_py[:, [0]], b),
        "matlab_sv_residual": float(np.asarray(data["dense_resid"]).item()),
    }
    errors.update(logdet_errors(ld_py, data["ld"]))

    return {
        "n": int(data["n"].item()),
        "k": int(data["k"].item()),
        "nch": int(data["nch"].item()),
        "occ": occ,
        "tol": tol,
        "nrhs": int(data["nrhs"].item()),
        "repeats": args.repeats,
        "warmups": args.warmups,
        "estimated_dense_gib": estimate_dense_gib(int(data["n"].item())),
        "errors": errors,
        "timings_seconds": {
            "chunkie_dense_assemble": float(np.asarray(data["assemble_time"]).item()),
            "matlab_build": float(np.asarray(data["matlab_build_time"]).item()),
            "matlab_mv": stats(np.asarray(data["matlab_mv_times"]).reshape(-1)),
            "matlab_sv": stats(np.asarray(data["matlab_sv_times"]).reshape(-1)),
            "matlab_logdet": stats(np.asarray(data["matlab_logdet_times"]).reshape(-1)),
            "python_build": python_build_time,
            "python_mv": stats(python_mv_times),
            "python_sv": stats(python_sv_times),
            "python_logdet": stats(python_logdet_times),
        },
        "python_factor": {
            "nlvl": F.nlvl,
            "nfactors": len(F.factors),
            "remaining_skeletons": int(F.Si.size if F.Si is not None else 0),
        },
        "matlab_stdout_tail": matlab_stdout.strip().splitlines()[-8:],
    }


def flatten_result(row: dict) -> dict:
    flat = {
        "n": row.get("n"),
        "skipped": row.get("skipped", False),
        "reason": row.get("reason", ""),
        "k": row.get("k"),
        "nch": row.get("nch"),
        "occ": row.get("occ"),
        "tol": row.get("tol"),
        "nrhs": row.get("nrhs"),
        "repeats": row.get("repeats"),
        "warmups": row.get("warmups"),
        "estimated_dense_gib": row.get("estimated_dense_gib"),
    }
    if row.get("skipped"):
        return flat
    for name, value in row["errors"].items():
        flat[f"err_{name}"] = value
    timings = row["timings_seconds"]
    for name, value in timings.items():
        if isinstance(value, dict):
            flat[f"time_{name}_mean"] = value["mean"]
            flat[f"time_{name}_std"] = value["std"]
        else:
            flat[f"time_{name}"] = value
    for name, value in row["python_factor"].items():
        flat[f"factor_{name}"] = value
    return flat


def write_csv(path: Path, rows: list[dict]) -> None:
    flat_rows = [flatten_result(row) for row in rows]
    fieldnames: list[str] = []
    for row in flat_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1024,2048,4096,8192,10000")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--occ", type=int, default=128)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--nrhs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--matlab", type=Path, default=DEFAULT_MATLAB)
    parser.add_argument("--chunkie", type=Path, default=Path(os.environ.get("CHUNKIE_REFERENCE", DEFAULT_CHUNKIE)))
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/rskelf_scaling.json"))
    parser.add_argument("--csv", type=Path, default=Path("benchmark_results/rskelf_scaling.csv"))
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--max-dense-gib", type=float, default=4.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.matlab.exists():
        raise FileNotFoundError(f"MATLAB executable not found: {args.matlab}")
    if not args.chunkie.exists():
        raise FileNotFoundError(f"ChunkIE reference not found: {args.chunkie}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for n in parse_sizes(args.sizes):
        print(f"\n=== rskelf scaling n={n} ===", flush=True)
        row = run_size(args, n)
        rows.append(row)
        args.out.write_text(json.dumps({"results": rows}, indent=2))
        write_csv(args.csv, rows)
        print(json.dumps(flatten_result(row), indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
