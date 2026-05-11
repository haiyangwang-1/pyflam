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
| `rskelf` | 64 | `0.031 MiB` | `0.5 KiB` | `50.6 KiB` | `83.4 KiB` | `0.01702 s` | `0.00956 s` | `1.6e-16` |
| `hifie` | 64 | `0.031 MiB` | `0.5 KiB` | `73.1 KiB` | `81.4 KiB` | `0.03791 s` | `0.01186 s` | `1.3e-16` |
| `rskelf` | 128 | `0.125 MiB` | `0.5 KiB` | `74.2 KiB` | `293.9 KiB` | `0.03794 s` | `0.02659 s` | `1.4e-16` |
| `hifie` | 128 | `0.125 MiB` | `0.5 KiB` | `122.9 KiB` | `294.8 KiB` | `0.05185 s` | `0.02170 s` | `1.4e-16` |
| `rskelf` | 256 | `0.500 MiB` | `0.5 KiB` | `127.0 KiB` | `1079.7 KiB` | `0.05283 s` | `0.03508 s` | `1.5e-16` |
| `hifie` | 256 | `0.500 MiB` | `0.5 KiB` | `210.1 KiB` | `1078.8 KiB` | `0.13313 s` | `0.03971 s` | `1.5e-16` |
| `mf` | 64 | `0.031 MiB` | `10.1 KiB` | `62.0 KiB` | `134.6 KiB` | `0.00692 s` | `0.00178 s` | `7.1e-17` |
| `hifde` | 64 | `0.031 MiB` | `10.1 KiB` | `65.6 KiB` | `105.9 KiB` | `0.01164 s` | `0.00271 s` | `1.1e-16` |
| `mf` | 144 | `0.158 MiB` | `32.0 KiB` | `178.6 KiB` | `652.5 KiB` | `0.00878 s` | `0.00239 s` | `7.4e-17` |
| `hifde` | 144 | `0.158 MiB` | `32.0 KiB` | `183.4 KiB` | `493.8 KiB` | `0.00811 s` | `0.00243 s` | `1.5e-16` |
| `mf` | 256 | `0.500 MiB` | `18.8 KiB` | `175.4 KiB` | `1723.1 KiB` | `0.01515 s` | `0.00400 s` | `9.8e-17` |
| `hifde` | 256 | `0.500 MiB` | `18.8 KiB` | `273.1 KiB` | `1204.1 KiB` | `0.02629 s` | `0.00981 s` | `2.5e-16` |

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
