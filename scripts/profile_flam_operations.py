"""Profile FLAM operation families on representative Python cases."""

from __future__ import annotations

import argparse
import cProfile
import csv
import json
from pathlib import Path
import pstats
import time
from typing import Callable

import numpy as np
import scipy.sparse as sp

from pyflam import (
    hifde2,
    hifde_diag,
    hifde_logdet,
    hifde_mv,
    hifde_sv,
    hifie2,
    hifie_diag,
    hifie_logdet,
    hifie_mv,
    hifie_sv,
    mf2,
    mf_diag,
    mf_logdet,
    mf_mv,
    mf_sv,
    ifmm,
    ifmm_mv,
    rskel,
    rskel_mv,
    rskelf,
    rskelf_diag,
    rskelf_logdet,
    rskelf_mv,
    rskelf_sv,
)


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


def top_functions(profile: cProfile.Profile, count: int) -> list[dict]:
    stats = pstats.Stats(profile).sort_stats("cumtime")
    rows = []
    for (filename, line, func), (cc, nc, tottime, cumtime, _callers) in stats.stats.items():
        rows.append(
            {
                "file": str(Path(filename)),
                "line": int(line),
                "function": func,
                "primitive_calls": int(cc),
                "total_calls": int(nc),
                "total_seconds": float(tottime),
                "cumulative_seconds": float(cumtime),
            }
        )
    rows.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
    return rows[:count]


def profile_call(fn: Callable[[], object], top_n: int) -> tuple[object, dict]:
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    result = profiler.runcall(fn)
    elapsed = time.perf_counter() - t0
    return result, {"elapsed_seconds": elapsed, "top_functions": top_functions(profiler, top_n)}


def case_meta(F) -> dict:
    nfactors = len(getattr(F, "factors", []))
    if nfactors == 0:
        nfactors = len(getattr(F, "U", [])) + len(getattr(F, "D", [])) + len(getattr(F, "B", []))
    return {
        "n": int(F.N),
        "nlvl": int(getattr(F, "nlvl", 0)),
        "nfactors": nfactors,
        "symm": str(getattr(F, "symm", "")),
    }


def run_dense_case(family: str, n: int, occ: int, tol: float, top_n: int) -> list[dict]:
    A, x = dense_kernel_case(n)
    if family == "rskelf":
        build = lambda: rskelf(A, x, occ, tol, opts={"symm": "p"})
        mv_fn, sv_fn, logdet_fn, diag_fn = rskelf_mv, rskelf_sv, rskelf_logdet, rskelf_diag
    elif family == "hifie":
        build = lambda: hifie2(A, x, occ, tol, opts={"symm": "p"})
        mv_fn, sv_fn, logdet_fn, diag_fn = hifie_mv, hifie_sv, hifie_logdet, hifie_diag
    else:
        raise ValueError(f"unknown dense family: {family}")

    F, build_profile = profile_call(build, top_n)
    X = np.arange(F.N * 3, dtype=float).reshape(F.N, 3) / (F.N * 3 + 1.0)
    operations = [
        ("build", build_profile, None),
        ("apply", None, lambda: mv_fn(F, X)),
        ("solve", None, lambda: sv_fn(F, X)),
        ("logdet", None, lambda: logdet_fn(F)),
        ("selected_diag", None, lambda: diag_fn(F, True)),
    ]
    return profile_operations(family, "dense_n", n, F, operations, top_n)


def run_compression_case(family: str, n: int, occ: int, tol: float, top_n: int) -> list[dict]:
    A, x = dense_kernel_case(n)
    if family == "rskel":
        build = lambda: rskel(A, x, x, occ, tol, opts={"symm": "p"})
        mv_fn = rskel_mv
    elif family == "ifmm":
        build = lambda: ifmm(A, x, x, occ, tol, opts={"store": "a", "near": 1, "symm": "p"})
        mv_fn = ifmm_mv
    else:
        raise ValueError(f"unknown compression family: {family}")

    F, build_profile = profile_call(build, top_n)
    X = np.arange(F.N * 3, dtype=float).reshape(F.N, 3) / (F.N * 3 + 1.0)
    operations = [
        ("build", build_profile, None),
        ("apply", None, lambda: mv_fn(F, X)),
    ]
    return profile_operations(family, "dense_n", n, F, operations, top_n)


def run_sparse_case(family: str, grid_n: int, occ: int, tol: float, top_n: int) -> list[dict]:
    A = spd_grid2(grid_n)
    if family == "mf":
        build = lambda: mf2(A, grid_n, occ, opts={"symm": "p"})
        mv_fn, sv_fn, logdet_fn, diag_fn = mf_mv, mf_sv, mf_logdet, mf_diag
    elif family == "hifde":
        build = lambda: hifde2(A, grid_n, occ, tol, opts={"symm": "p", "skip": 1})
        mv_fn, sv_fn, logdet_fn, diag_fn = hifde_mv, hifde_sv, hifde_logdet, hifde_diag
    else:
        raise ValueError(f"unknown sparse family: {family}")

    F, build_profile = profile_call(build, top_n)
    X = np.arange(F.N * 3, dtype=float).reshape(F.N, 3) / (F.N * 3 + 1.0)
    operations = [
        ("build", build_profile, None),
        ("apply", None, lambda: mv_fn(F, X)),
        ("solve", None, lambda: sv_fn(F, X)),
        ("logdet", None, lambda: logdet_fn(F)),
        ("selected_diag", None, lambda: diag_fn(F, True)),
    ]
    return profile_operations(family, "grid_n", grid_n, F, operations, top_n)


def profile_operations(
    family: str,
    size_label: str,
    size_value: int,
    F,
    operations: list[tuple[str, dict | None, Callable[[], object] | None]],
    top_n: int,
) -> list[dict]:
    rows = []
    meta = case_meta(F)
    for operation, existing_profile, fn in operations:
        print(f"=== profile {family} {operation} {size_label}={size_value} ===", flush=True)
        profile_data = existing_profile
        if profile_data is None:
            assert fn is not None
            _, profile_data = profile_call(fn, top_n)
        rows.append(
            {
                "family": family,
                "operation": operation,
                "size_label": size_label,
                "size_value": int(size_value),
                **meta,
                **profile_data,
            }
        )
        print(json.dumps(flatten_row(rows[-1]), indent=2), flush=True)
    return rows


def flatten_row(row: dict) -> dict:
    top = {}
    for candidate in row["top_functions"]:
        if candidate.get("function") != "<lambda>" or Path(candidate.get("file", "")).name != "profile_flam_operations.py":
            top = candidate
            break
    return {
        "family": row["family"],
        "operation": row["operation"],
        "size_label": row["size_label"],
        "size_value": row["size_value"],
        "n": row["n"],
        "nlvl": row["nlvl"],
        "nfactors": row["nfactors"],
        "elapsed_seconds": row["elapsed_seconds"],
        "top_function": top.get("function", ""),
        "top_file": top.get("file", ""),
        "top_line": top.get("line", ""),
        "top_cumulative_seconds": top.get("cumulative_seconds", ""),
    }


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
    parser.add_argument("--dense-n", type=int, default=256)
    parser.add_argument("--grid-n", type=int, default=17)
    parser.add_argument("--occ", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/flam_operation_profiles.json"))
    parser.add_argument("--csv", type=Path, default=Path("benchmark_results/flam_operation_profiles.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for family in ("rskel", "ifmm", "rskelf", "hifie"):
        if family in {"rskel", "ifmm"}:
            rows.extend(run_compression_case(family, args.dense_n, args.occ, args.tol, args.top))
        else:
            rows.extend(run_dense_case(family, args.dense_n, args.occ, args.tol, args.top))
        args.out.write_text(json.dumps({"results": rows}, indent=2))
        write_csv(args.csv, rows)
    for family in ("mf", "hifde"):
        rows.extend(run_sparse_case(family, args.grid_n, args.occ, args.tol, args.top))
        args.out.write_text(json.dumps({"results": rows}, indent=2))
        write_csv(args.csv, rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
