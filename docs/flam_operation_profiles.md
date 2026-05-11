# FLAM Operation Profiles

This profile separates build, apply, solve, logdet, and selected-inversion
diagonal extraction on representative dense-kernel and sparse-grid cases.

## Command

```powershell
uv run python scripts\profile_flam_operations.py --dense-n 256 --grid-n 17 --occ 8 --top 12 --out benchmark_results\flam_operation_profiles.json --csv benchmark_results\flam_operation_profiles.csv
```

Outputs:

- `benchmark_results/flam_operation_profiles.json`
- `benchmark_results/flam_operation_profiles.csv`

The raw output files are local ignored artifacts and are not stored in git.
Keep only curated benchmark summaries in `docs/`.

## Timings

| Family | Operation | N | Factors | Elapsed |
|---|---|---:|---:|---:|
| `rskel` | `build` | 256 | 380 | `0.13821 s` |
| `rskel` | `apply` | 256 | 380 | `0.00712 s` |
| `ifmm` | `build` | 256 | 453 | `0.26545 s` |
| `ifmm` | `apply` | 256 | 453 | `0.01090 s` |
| `rskelf` | `build` | 256 | 79 | `0.15492 s` |
| `rskelf` | `apply` | 256 | 79 | `0.01108 s` |
| `rskelf` | `solve` | 256 | 79 | `0.03439 s` |
| `rskelf` | `logdet` | 256 | 79 | `0.00168 s` |
| `rskelf` | `selected_diag` | 256 | 79 | `0.13638 s` |
| `hifie` | `build` | 256 | 83 | `0.46515 s` |
| `hifie` | `apply` | 256 | 83 | `0.00506 s` |
| `hifie` | `solve` | 256 | 83 | `0.02848 s` |
| `hifie` | `logdet` | 256 | 83 | `0.00240 s` |
| `hifie` | `selected_diag` | 256 | 83 | `0.41761 s` |
| `mf` | `build` | 256 | 13 | `0.02579 s` |
| `mf` | `apply` | 256 | 13 | `0.00157 s` |
| `mf` | `solve` | 256 | 13 | `0.00813 s` |
| `mf` | `logdet` | 256 | 13 | `0.00136 s` |
| `mf` | `selected_diag` | 256 | 13 | `0.04651 s` |
| `hifde` | `build` | 256 | 15 | `0.03263 s` |
| `hifde` | `apply` | 256 | 15 | `0.00174 s` |
| `hifde` | `solve` | 256 | 15 | `0.00664 s` |
| `hifde` | `logdet` | 256 | 15 | `0.00050 s` |
| `hifde` | `selected_diag` | 256 | 15 | `0.08380 s` |

## Hotspots

- `rskel`, `ifmm`, and `rskelf` build: `id` in `core.py`, SciPy finite-check
  wrappers, QR, and small triangular solves.
- `rskel` apply: `_rskel_mv_compact`, with minor dtype discovery overhead.
- `ifmm` apply: `_apply_interaction`, with minor dtype discovery overhead.
- `rskelf` and `hifde` solve: compact Cholesky solve sweeps, dominated by many
  small triangular solves.
- `rskelf`, `hifie`, `mf`, and `hifde` selected diagonal extraction: sparse
  keep-pattern indexing and local block unfolding.
- `hifie` build: sparse block extraction inside `_hifie_base`.
- `hifde` build: `_factor_sparse_block` and sparse block extraction.

## Optimization Record

The first low-risk optimization from this profile removed repeated factor dtype
discovery inside selected-inversion block unfolding. The dtype is now computed
once per diagonal extraction and passed into each local unfold block for
`rskelf`, `hifie`, `mf`, and `hifde`.

The same pass removed a duplicate factor dtype scan from `rskel_mv` by passing
the already-computed dtype into the compact apply sweep.

Larger solve optimizations were intentionally deferred. The profile shows many
small triangular solves, but batching them would require changing sweep
structure and factor grouping, which is harder to audit than this pass should
be.

No JIT dependency was added. The hot paths mix sparse indexing, BLAS calls, and
small SciPy triangular solves, so a readable local `numba` change is not
available without a larger rewrite.
