"""ChunkIE-generated rskelf parity benchmark.

ChunkIE is responsible for geometry, kernels, quadrature, and dense matrix
assembly. The benchmark then runs MATLAB FLAM ``rskelf`` and PyFLAM ``rskelf``
on the exact same generated dense matrix and compares apply/solve/logdet
outputs and timings.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import time

import numpy as np
import scipy.io

from pyflam import rskelf, rskelf_logdet, rskelf_mv, rskelf_sv


DEFAULT_MATLAB = Path(os.environ.get("MATLAB", r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"))
DEFAULT_CHUNKIE = Path(tempfile.gettempdir()) / "chunkie-reference"


def _matlab_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _case_matlab(case: str, nch: int, k: int) -> str:
    if case == "single":
        return textwrap.dedent(
            f"""
            pref = []; pref.k = {k};
            chnkr = chunkerfuncuni(@(t) starfish(t,5,0.15), {nch}, [], pref);
            kernd = -2*kernel('helm','d',10);
            tm_assemble = tic;
            Ad = eye(chnkr.npt) + chunkermat(chnkr, kernd);
            assemble_time = toc(tm_assemble);
            xflam = real(chnkr.r(:,:));
            case_name = 'single';
            """
        )
    if case == "flamopdims":
        return textwrap.dedent(
            """
            zk1 = 10; zk2 = 15;
            cparams = [];
            cparams.eps = 1.0e-10;
            cparams.nover = 0;
            cparams.maxchunklen = 4.0/max(real([zk1, zk2]));
            pref = [];
            pref.k = 10;
            chnkr1 = chunkerfunc(@(t) starfish(t,5,0.15, [], [], 1.15), cparams, pref);
            chnkr2 = chunkerfunc(@(t) starfish(t,10,0.03, [], [], 0.9), cparams, pref);
            chnkrs(2,1) = chunker();
            chnkrs(1,1) = chnkr1;
            chnkrs(2,1) = chnkr2;

            skdiff = kernel('helmdiff', 's', [zk1, zk2]);
            skpdiff = kernel('helmdiff', 'sp', [zk1, zk2]);
            dkdiff = kernel('helmdiff', 'd', [zk1, zk2]);
            dkpdiff = kernel('helmdiff', 'dp', [zk1, zk2]);
            K = kernel([dkdiff, -1*skdiff; dkpdiff, -1*skpdiff]);
            dk2 = kernel('helm', 'd', zk2);
            sk2 = kernel('helm', 's', zk2);
            ck2 = kernel('helm', 'c', zk2, [2, -2*1j*zk2]);
            ck2p = kernel('helm', 'cp', zk2, [2, -2*1j*zk2]);
            K2 = kernel([dk2, -1*sk2]);
            K3 = -1*kernel([ck2; ck2p]);
            Kmat(2,2) = kernel();
            Kmat(1,1) = K;
            Kmat(2,1) = K2;
            Kmat(2,2) = ck2;
            Kmat(1,2) = K3;

            opts_loc = [];
            opts_loc.adaptive_correction = true;
            tm_assemble = tic;
            Ad = chunkermat(chnkrs, Kmat, opts_loc);
            Ad = Ad + eye(size(Ad,1));
            assemble_time = toc(tm_assemble);

            n1 = chnkr1.npt;
            n2 = chnkr2.npt;
            xflam = zeros(2,2*n1+n2);
            x1 = real(chnkr1.r(:,:));
            x2 = real(chnkr2.r(:,:));
            xflam(:,1:2:2*n1) = x1;
            xflam(:,2:2:2*n1) = x1;
            xflam(:,2*n1+1:end) = x2;
            case_name = 'flamopdims';
            """
        )
    raise ValueError(f"unknown case: {case}")


def write_matlab_driver(path: Path, chunkie: Path, out: Path, case: str, nch: int, k: int, occ: int, tol: float) -> None:
    path.write_text(
        textwrap.dedent(
            f"""
            cd('{_matlab_path(chunkie)}');
            startup(struct('testfmm', false));
            rng(1234);
            {_case_matlab(case, nch, k)}
            n = size(Ad,1);
            occ = {occ};
            tol = {tol:.17g};
            X = reshape(sin((1:(3*n))/37), n, 3);
            b = X(:,1);
            A = @(i,j) Ad(i,j);

            tm_build = tic;
            F = rskelf(A, xflam, occ, tol, [], struct('symm','n'));
            matlab_build_time = toc(tm_build);

            tm_mv = tic;
            Ymv = rskelf_mv(F, X);
            matlab_mv_time = toc(tm_mv);

            tm_sv = tic;
            Ysv = rskelf_sv(F, X);
            matlab_sv_time = toc(tm_sv);

            tm_logdet = tic;
            ld = rskelf_logdet(F);
            matlab_logdet_time = toc(tm_logdet);

            dense_resid = norm(Ad*Ysv(:,1) - b) / norm(b);
            save('{_matlab_path(out)}', 'case_name', 'n', 'occ', 'tol', 'xflam', ...
                 'Ad', 'X', 'Ymv', 'Ysv', 'ld', 'dense_resid', 'assemble_time', ...
                 'matlab_build_time', 'matlab_mv_time', 'matlab_sv_time', ...
                 'matlab_logdet_time', '-v7');
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


def relerr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def logdet_errors(py_ld, matlab_ld) -> dict[str, float]:
    py_ld = np.asarray(py_ld).item()
    matlab_ld = np.asarray(matlab_ld).item()
    raw = py_ld - matlab_ld
    if abs(raw.imag) > 0:
        k = np.round(raw.imag / (2 * np.pi))
        adjusted = raw - 2j * np.pi * k
    else:
        adjusted = raw
    return {
        "logdet_abs": float(abs(raw)),
        "logdet_mod_2pi_i_abs": float(abs(adjusted)),
        "det_from_logdet_rel": float(abs(np.exp(py_ld - matlab_ld) - 1.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["single", "flamopdims"], default="single")
    parser.add_argument("--nch", type=int, default=64, help="single-case chunks; with k=16 gives 1024 unknowns")
    parser.add_argument("--k", type=int, default=16, help="single-case Legendre nodes per chunk")
    parser.add_argument("--occ", type=int, default=128)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--matlab", type=Path, default=DEFAULT_MATLAB)
    parser.add_argument("--chunkie", type=Path, default=Path(os.environ.get("CHUNKIE_REFERENCE", DEFAULT_CHUNKIE)))
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/rskelf_chunkie_single_1024.json"))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    if not args.matlab.exists():
        raise FileNotFoundError(f"MATLAB executable not found: {args.matlab}")
    if not args.chunkie.exists():
        raise FileNotFoundError(f"ChunkIE reference not found: {args.chunkie}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        mat_out = tmpdir / "chunkie_rskelf_parity.mat"
        driver = tmpdir / "run_chunkie_rskelf_parity.m"
        write_matlab_driver(driver, args.chunkie, mat_out, args.case, args.nch, args.k, args.occ, args.tol)
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

    t0 = time.perf_counter()
    Ymv_py = rskelf_mv(F, X)
    python_mv_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    Ysv_py = rskelf_sv(F, X)
    python_sv_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    ld_py = rskelf_logdet(F)
    python_logdet_time = time.perf_counter() - t0

    b = X[:, [0]]
    errors = {
        "mv_vs_matlab": relerr(Ymv_py, data["Ymv"]),
        "sv_vs_matlab": relerr(Ysv_py, data["Ysv"]),
        "mv_vs_dense": relerr(Ymv_py, A @ X),
        "sv_residual": relerr(A @ Ysv_py[:, [0]], b),
        "matlab_sv_residual": float(np.asarray(data["dense_resid"]).item()),
    }
    errors.update(logdet_errors(ld_py, data["ld"]))

    report = {
        "case": args.case,
        "n": int(data["n"].item()),
        "occ": occ,
        "tol": tol,
        "errors": errors,
        "timings_seconds": {
            "chunkie_dense_assemble": float(np.asarray(data["assemble_time"]).item()),
            "matlab_build": float(np.asarray(data["matlab_build_time"]).item()),
            "matlab_mv": float(np.asarray(data["matlab_mv_time"]).item()),
            "matlab_sv": float(np.asarray(data["matlab_sv_time"]).item()),
            "matlab_logdet": float(np.asarray(data["matlab_logdet_time"]).item()),
            "python_build": python_build_time,
            "python_mv": python_mv_time,
            "python_sv": python_sv_time,
            "python_logdet": python_logdet_time,
        },
        "python_factor": {
            "nlvl": F.nlvl,
            "nfactors": len(F.factors),
            "remaining_skeletons": int(F.Si.size if F.Si is not None else 0),
        },
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if matlab_stdout.strip():
        print("\n--- MATLAB stdout ---")
        print(matlab_stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
