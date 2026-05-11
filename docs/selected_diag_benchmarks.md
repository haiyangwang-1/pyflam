# Selected Diagonal Benchmarks

This benchmark compares the selected-inversion diagonal routines against the
old correctness-first baseline:

- `diag(apply(F, eye(N)))` for `diag(A)`.
- `diag(solve(F, eye(N)))` for `diag(inv(A))`.

The baseline is retained only inside the benchmark script so the selected
paths can be checked for accuracy and measured against the full identity RHS
allocation.

## Command

```powershell
uv run python scripts\benchmark_selected_diag_scaling.py --dense-sizes 64,128,256 --grid-sizes 9,13,17 --occ 8 --repeats 3 --warmups 1 --out benchmark_results\selected_diag_scaling.json --csv benchmark_results\selected_diag_scaling.csv
```

Outputs:

- `benchmark_results/selected_diag_scaling.json`
- `benchmark_results/selected_diag_scaling.csv`

## Inverse Diagonal Results

| Family | N | Identity RHS | Largest local block | Selected peak | Identity peak | Selected time | Identity time | Rel. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rskelf` | 64 | `0.031 MiB` | `0.5 KiB` | `50.8 KiB` | `81.0 KiB` | `0.01585 s` | `0.00748 s` | `1.6e-16` |
| `rskelf` | 128 | `0.125 MiB` | `0.5 KiB` | `77.4 KiB` | `294.1 KiB` | `0.02697 s` | `0.01601 s` | `1.4e-16` |
| `rskelf` | 256 | `0.500 MiB` | `0.5 KiB` | `125.1 KiB` | `1080.5 KiB` | `0.05465 s` | `0.03042 s` | `1.5e-16` |
| `hifie` | 64 | `0.031 MiB` | `0.5 KiB` | `73.8 KiB` | `84.5 KiB` | `0.02451 s` | `0.00714 s` | `1.3e-16` |
| `hifie` | 128 | `0.125 MiB` | `0.5 KiB` | `122.4 KiB` | `289.9 KiB` | `0.04940 s` | `0.01554 s` | `1.4e-16` |
| `hifie` | 256 | `0.500 MiB` | `0.5 KiB` | `207.0 KiB` | `1081.3 KiB` | `0.10083 s` | `0.03176 s` | `1.5e-16` |
| `mf` | 64 | `0.031 MiB` | `10.1 KiB` | `62.0 KiB` | `134.6 KiB` | `0.00566 s` | `0.00094 s` | `7.1e-17` |
| `mf` | 144 | `0.158 MiB` | `32.0 KiB` | `178.5 KiB` | `652.5 KiB` | `0.00508 s` | `0.00230 s` | `7.4e-17` |
| `mf` | 256 | `0.500 MiB` | `18.8 KiB` | `174.8 KiB` | `1723.1 KiB` | `0.01432 s` | `0.00419 s` | `9.8e-17` |
| `hifde` | 64 | `0.031 MiB` | `10.1 KiB` | `65.5 KiB` | `105.7 KiB` | `0.00631 s` | `0.00228 s` | `1.1e-16` |
| `hifde` | 144 | `0.158 MiB` | `32.0 KiB` | `183.4 KiB` | `493.8 KiB` | `0.00652 s` | `0.00242 s` | `1.5e-16` |
| `hifde` | 256 | `0.500 MiB` | `18.8 KiB` | `273.6 KiB` | `1203.8 KiB` | `0.02414 s` | `0.00824 s` | `2.5e-16` |

## Notes

- The selected paths match the explicit identity-RHS baseline to roundoff.
- The identity-RHS allocation grows like `N^2`; the selected paths operate on
  local factor blocks and sparse keep patterns instead of materializing a full
  dense identity RHS.
- At these small benchmark sizes, the selected Python implementations are
  still slower than the identity baseline because the latter runs through
  dense BLAS/SciPy kernels on a small number of large RHS columns. This is a
  performance optimization target, not a correctness issue.
- `tracemalloc` does not capture every native BLAS allocator, so the benchmark
  also reports the exact identity RHS bytes and the largest local factor block
  estimate. The raw JSON/CSV include both `diag(A)` and `diag(inv(A))`.
