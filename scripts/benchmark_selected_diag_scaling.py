"""Benchmark selected diagonal extraction against explicit identity RHS solves.

The correctness-first diagonal path used earlier was equivalent to
``diag(solve(F, eye(N)))`` for inverse diagonals and ``diag(apply(F, eye(N)))``
for matrix diagonals.  This benchmark keeps that path as a small-size baseline
and compares it with the selected-inversion diagonal routines now used by
``rskelf``, ``mf``, ``hifie``, and ``hifde``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import time
import tracemalloc
from typing import Callable

import numpy as np
import scipy.sparse as sp

from pyflam import (
    hifde2,
    hifde_diag,
    hifde_mv,
    hifde_sv,
    hifie2,
    hifie_diag,
    hifie_mv,
    hifie_sv,
    mf2,
    mf_diag,
    mf_mv,
    mf_sv,
    rskelf,
    rskelf_diag,
    rskelf_mv,
    rskelf_sv,
)


def parse_ints(text: str) -> list[int]:
    vals = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not vals:
        raise ValueError("at least one integer size is required")
    return vals


def stats(values: list[float]) -> dict[str, float | list[float]]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "samples": [float(x) for x in arr],
    }


def time_repeated(fn: Callable[[], np.ndarray], repeats: int, warmups: int) -> tuple[np.ndarray, list[float]]:
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
    return np.asarray(result), times


def traced_peak(fn: Callable[[], np.ndarray]) -> tuple[np.ndarray, int]:
    gc.collect()
    tracemalloc.start()
    try:
        result = np.asarray(fn())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, int(peak)


def relerr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def dense_kernel_case(n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)
    x = np.vstack((t, 0.2 * np.sin(2 * np.pi * t)))
    dist = np.abs(t[:, None] - t[None, :])
    A = np.exp(-dist / 0.12) + 2.5 * np.eye(n)
    return A, x


def spd_grid2(n: int) -> sp.csc_matrix:
    nd = n - 1
    A = sp.lil_matrix((nd * nd, nd * nd))
    for j in range(nd):
        for i in range(nd):
            idx = i + nd * j
            A[idx, idx] = 4.0
            if i > 0:
                A[idx, idx - 1] = -1.0
            if i + 1 < nd:
                A[idx, idx + 1] = -1.0
            if j > 0:
                A[idx, idx - nd] = -1.0
            if j + 1 < nd:
                A[idx, idx + nd] = -1.0
    return A.tocsc()


def factor_workspace_bytes(factors, dtype: np.dtype) -> int:
    itemsize = np.dtype(dtype).itemsize
    max_block = 0
    for f in factors:
        rd = np.asarray(getattr(f, "rd", []))
        sk = np.asarray(getattr(f, "sk", []))
        max_block = max(max_block, int(rd.size + sk.size))
    return int(max_block * max_block * itemsize)


def factor_dtype(F, fallback: np.dtype) -> np.dtype:
    factors = getattr(F, "factors", [])
    if factors:
        L = getattr(factors[0], "L", None)
        if L is not None and np.asarray(L).size:
            return np.asarray(L).dtype
    return np.dtype(fallback)


def run_operation(
    family: str,
    size_label: str,
    size_value: int,
    F,
    diag_fn,
    mv_fn,
    sv_fn,
    dtype: np.dtype,
    mode: str,
    repeats: int,
    warmups: int,
    factor_seconds: float,
) -> dict:
    n = int(F.N)
    dtype = np.dtype(dtype)
    dinv = mode == "invdiag"

    def selected():
        return diag_fn(F, dinv)

    def identity_baseline():
        eye = np.eye(n, dtype=dtype)
        Y = sv_fn(F, eye) if dinv else mv_fn(F, eye)
        return np.diag(Y)

    selected_out, selected_times = time_repeated(selected, repeats, warmups)
    baseline_out, baseline_times = time_repeated(identity_baseline, repeats, warmups)
    selected_peak_out, selected_peak = traced_peak(selected)
    baseline_peak_out, baseline_peak = traced_peak(identity_baseline)

    if relerr(selected_out, baseline_out) > 1e-8:
        raise AssertionError(f"{family} {mode} selected diagonal disagrees with identity RHS baseline")
    if relerr(selected_peak_out, baseline_peak_out) > 1e-8:
        raise AssertionError(f"{family} {mode} traced run disagrees with identity RHS baseline")

    selected_stat = stats(selected_times)
    baseline_stat = stats(baseline_times)
    selected_mean = float(selected_stat["mean"])
    baseline_mean = float(baseline_stat["mean"])
    return {
        "family": family,
        "mode": mode,
        "size_label": size_label,
        "size_value": size_value,
        "n": n,
        "nfactors": len(getattr(F, "factors", [])),
        "nlvl": int(getattr(F, "nlvl", 0)),
        "factor_seconds": factor_seconds,
        "identity_rhs_bytes": int(n * n * dtype.itemsize),
        "largest_factor_block_bytes": factor_workspace_bytes(getattr(F, "factors", []), dtype),
        "selected_peak_tracemalloc_bytes": selected_peak,
        "identity_peak_tracemalloc_bytes": baseline_peak,
        "selected_seconds": selected_stat,
        "identity_rhs_seconds": baseline_stat,
        "speedup_identity_over_selected": baseline_mean / max(selected_mean, 1e-300),
        "relative_error_vs_identity_rhs": relerr(selected_out, baseline_out),
    }


def build_rskelf(n: int, occ: int, tol: float):
    A, x = dense_kernel_case(n)
    t0 = time.perf_counter()
    F = rskelf(A, x, occ, tol, opts={"symm": "p"})
    return F, np.dtype(A.dtype), time.perf_counter() - t0


def build_hifie(n: int, occ: int, tol: float):
    A, x = dense_kernel_case(n)
    t0 = time.perf_counter()
    F = hifie2(A, x, occ, tol, opts={"symm": "p"})
    return F, np.dtype(A.dtype), time.perf_counter() - t0


def build_mf(grid_n: int, occ: int):
    A = spd_grid2(grid_n)
    t0 = time.perf_counter()
    F = mf2(A, grid_n, occ, opts={"symm": "p"})
    return F, np.dtype(A.dtype), time.perf_counter() - t0


def build_hifde(grid_n: int, occ: int, tol: float):
    A = spd_grid2(grid_n)
    t0 = time.perf_counter()
    F = hifde2(A, grid_n, occ, tol, opts={"symm": "p", "skip": 1})
    return F, np.dtype(A.dtype), time.perf_counter() - t0


def flatten_row(row: dict) -> dict:
    out = {
        key: value
        for key, value in row.items()
        if key not in {"selected_seconds", "identity_rhs_seconds"}
    }
    for prefix in ("selected_seconds", "identity_rhs_seconds"):
        stat = row[prefix]
        out[f"{prefix}_mean"] = stat["mean"]
        out[f"{prefix}_std"] = stat["std"]
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    flat_rows = [flatten_row(row) for row in rows]
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
    parser.add_argument("--dense-sizes", default="64,128,256")
    parser.add_argument("--grid-sizes", default="9,13,17", help="regular grid n values; matrix size is (n - 1)^2")
    parser.add_argument("--occ", type=int, default=32)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/selected_diag_scaling.json"))
    parser.add_argument("--csv", type=Path, default=Path("benchmark_results/selected_diag_scaling.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for n in parse_ints(args.dense_sizes):
        for family, builder, diag_fn, mv_fn, sv_fn in (
            ("rskelf", build_rskelf, rskelf_diag, rskelf_mv, rskelf_sv),
            ("hifie", build_hifie, hifie_diag, hifie_mv, hifie_sv),
        ):
            print(f"=== {family} selected diagonal n={n} ===", flush=True)
            F, dtype, factor_seconds = builder(n, args.occ, args.tol)
            dtype = factor_dtype(F, dtype)
            for mode in ("diag", "invdiag"):
                rows.append(
                    run_operation(
                        family,
                        "n",
                        n,
                        F,
                        diag_fn,
                        mv_fn,
                        sv_fn,
                        dtype,
                        mode,
                        args.repeats,
                        args.warmups,
                        factor_seconds,
                    )
                )
                print(json.dumps(flatten_row(rows[-1]), indent=2), flush=True)
            args.out.write_text(json.dumps({"results": rows}, indent=2))
            write_csv(args.csv, rows)

    for grid_n in parse_ints(args.grid_sizes):
        for family, builder, diag_fn, mv_fn, sv_fn in (
            ("mf", build_mf, mf_diag, mf_mv, mf_sv),
            ("hifde", build_hifde, hifde_diag, hifde_mv, hifde_sv),
        ):
            print(f"=== {family} selected diagonal grid_n={grid_n} ===", flush=True)
            if family == "mf":
                F, dtype, factor_seconds = builder(grid_n, args.occ)
            else:
                F, dtype, factor_seconds = builder(grid_n, args.occ, args.tol)
            dtype = factor_dtype(F, dtype)
            for mode in ("diag", "invdiag"):
                rows.append(
                    run_operation(
                        family,
                        "grid_n",
                        grid_n,
                        F,
                        diag_fn,
                        mv_fn,
                        sv_fn,
                        dtype,
                        mode,
                        args.repeats,
                        args.warmups,
                        factor_seconds,
                    )
                )
                print(json.dumps(flatten_row(rows[-1]), indent=2), flush=True)
            args.out.write_text(json.dumps({"results": rows}, indent=2))
            write_csv(args.csv, rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
