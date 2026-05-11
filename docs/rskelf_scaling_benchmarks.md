# rskelf Scaling Benchmarks

This benchmark uses ChunkIE to assemble dense single-boundary Helmholtz
double-layer operators, then compares MATLAB FLAM `rskelf` and PyFLAM
`rskelf` on the same dense matrix.

## Command

```powershell
uv run python scripts\benchmark_rskelf_scaling.py
```

Actual command used:

```powershell
uv run python scripts\benchmark_rskelf_scaling.py --sizes 1024,2048,4096,8192,10000 --repeats 10 --warmups 1 --out benchmark_results\rskelf_scaling.json --csv benchmark_results\rskelf_scaling.csv --timeout 7200
```

Outputs:

- `benchmark_results/rskelf_scaling.json`
- `benchmark_results/rskelf_scaling.csv`

## Accuracy

| N | mv vs MATLAB | solve vs MATLAB | mv vs dense | Py solve residual | MATLAB solve residual | logdet error |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | `2.31e-15` | `4.14e-15` | `1.71e-11` | `1.17e-10` | `1.17e-10` | `5.69e-14` |
| 2048 | `3.22e-15` | `6.94e-15` | `1.88e-11` | `2.27e-11` | `2.27e-11` | `1.14e-13` |
| 4096 | `5.44e-13` | `5.54e-13` | `2.74e-11` | `3.06e-11` | `3.06e-11` | `9.10e-13` |
| 8192 | `9.08e-15` | `1.10e-14` | `2.61e-11` | `2.43e-11` | `2.43e-11` | `5.69e-13` |
| 10000 | `8.07e-13` | `8.52e-13` | `2.77e-11` | `2.82e-11` | `2.82e-11` | `1.04e-12` |

## Timings

Build and dense assembly are timed once. Matvec, solve, and logdet are reported
as mean +/- sample std over 10 repeats after one warmup.

| N | ChunkIE dense assemble | MATLAB build | Python build |
|---:|---:|---:|---:|
| 1024 | `0.877 s` | `0.358 s` | `0.798 s` |
| 2048 | `1.774 s` | `0.828 s` | `2.258 s` |
| 4096 | `5.903 s` | `2.371 s` | `4.799 s` |
| 8192 | `23.526 s` | `8.478 s` | `13.373 s` |
| 10000 | `34.699 s` | `9.880 s` | `19.140 s` |

| N | MATLAB matvec | Python matvec | MATLAB solve | Python solve | MATLAB logdet | Python logdet |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | `0.00208 +/- 0.00102` | `0.00152 +/- 0.00017` | `0.00221 +/- 0.00096` | `0.00345 +/- 0.00017` | `0.00065 +/- 0.00042` | `0.00043 +/- 0.00001` |
| 2048 | `0.00442 +/- 0.00120` | `0.00361 +/- 0.00018` | `0.00460 +/- 0.00106` | `0.00924 +/- 0.00023` | `0.00103 +/- 0.00042` | `0.00097 +/- 0.00002` |
| 4096 | `0.00840 +/- 0.00110` | `0.00766 +/- 0.00041` | `0.00965 +/- 0.00134` | `0.02458 +/- 0.00144` | `0.00176 +/- 0.00039` | `0.00230 +/- 0.00012` |
| 8192 | `0.01940 +/- 0.00169` | `0.01425 +/- 0.00047` | `0.02177 +/- 0.00356` | `0.04638 +/- 0.00879` | `0.00314 +/- 0.00044` | `0.00542 +/- 0.00174` |
| 10000 | `0.02193 +/- 0.00447` | `0.01625 +/- 0.00032` | `0.02319 +/- 0.00279` | `0.04458 +/- 0.00391` | `0.00329 +/- 0.00043` | `0.00477 +/- 0.00020` |

## Python / MATLAB Runtime Ratios

Values below are `Python time / MATLAB time`; below `1.0x` means Python was
faster in this run.

| N | Build | Matvec | Solve | Logdet |
|---:|---:|---:|---:|---:|
| 1024 | `2.23x` | `0.73x` | `1.56x` | `0.66x` |
| 2048 | `2.73x` | `0.82x` | `2.01x` | `0.94x` |
| 4096 | `2.02x` | `0.91x` | `2.55x` | `1.31x` |
| 8192 | `1.58x` | `0.73x` | `2.13x` | `1.72x` |
| 10000 | `1.94x` | `0.74x` | `1.92x` | `1.45x` |

## Notes

- Accuracy stays stable across the 1k-10k range and tracks MATLAB FLAM closely.
- Dense assembly dominates total cost at the high end; the 10k matrix is about
  `1.49 GiB` as a complex dense array before extra copies.
- Python matvec is faster than MATLAB in this run; Python solve is still slower,
  likely due to repeated SciPy triangular-solve overhead inside Python factor
  loops.
- Python build is within a small constant factor of MATLAB build but remains
  slower, so build optimization should focus on ID/LU/factor-loop hotspots.
