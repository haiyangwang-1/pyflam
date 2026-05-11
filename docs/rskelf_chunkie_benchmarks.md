# ChunkIE rskelf Parity Benchmarks

These benchmarks use ChunkIE to generate the dense operator matrix, then run
MATLAB FLAM `rskelf` and PyFLAM `rskelf` on that same matrix.

## Commands

```powershell
uv run python scripts\benchmark_rskelf_parity.py --case single --nch 64 --k 16 --occ 128 --tol 1e-10 --out benchmark_results\rskelf_chunkie_single_1024.json
uv run python scripts\benchmark_rskelf_parity.py --case flamopdims --occ 128 --tol 1e-10 --out benchmark_results\rskelf_chunkie_flamopdims.json --timeout 2400
```

## Results

| Case | Matrix size | Py vs MATLAB matvec | Py vs MATLAB solve | Py solve residual | MATLAB solve residual |
|---|---:|---:|---:|---:|---:|
| `single` | 1024 | `2.31e-15` | `4.14e-15` | `1.17e-10` | `1.17e-10` |
| `flamopdims` | 1960 | `4.97e-15` | `6.47e-14` | `5.20e-09` | `5.20e-09` |

For complex matrices, raw `logdet` can differ by multiples of `2*pi*i`.
The benchmark therefore records both raw log difference and determinant
equivalence via `exp(logdet)`.

| Case | MATLAB build | Python build | MATLAB solve | Python solve | logdet modulo branch |
|---|---:|---:|---:|---:|---:|
| `single` | `0.366 s` | `1.743 s` | `0.017 s` | `0.006 s` | `5.69e-14` |
| `flamopdims` | `1.972 s` | `3.395 s` | `0.029 s` | `0.016 s` | `4.26e-14` |

## Dense Placeholder to Compact Factor Timing Change

| Case | Python build before | Python build now | Python logdet before | Python logdet now |
|---|---:|---:|---:|---:|
| `single` | `2.721 s` | `1.743 s` | `2.979 s` | `0.00083 s` |
| `flamopdims` | `4.795 s` | `3.395 s` | `1.693 s` | `0.00124 s` |

The compact Python matvec/solve are currently Python-loop implementations, so
for these 1k-2k matrices they are slower than the previous dense BLAS-backed
placeholder (`single`: solve `0.0023 s` before, `0.0042 s` now;
`flamopdims`: solve `0.0068 s` before, `0.0161 s` now). They now use the FLAM
factor sweeps, though, so this is the right baseline for optimizing the faithful
implementation.

The implementation uses NumPy for pure array operations, Cholesky, norms, tiny
determinants, and permutation handling, while retaining SciPy for pivoted QR,
triangular solves, sparse matrices, and LDL. The local LU path avoids forming a
full permutation matrix and uses vector pivots.

## Interpretation

The current PyFLAM public outputs are correct against MATLAB FLAM at the target
`1e-10` to `1e-8` level on ChunkIE-generated operators. The unsymmetric
`rskelf` path now uses compact FLAM factor sweeps for `rskelf_mv`,
`rskelf_sv`, and `rskelf_logdet`; remaining optimization work should focus on
reducing Python-loop overhead and broadening the compact path to symmetric,
Hermitian, positive-definite, and partial factorizations.
