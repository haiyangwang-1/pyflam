# Implementation Matrix

| Family | Construction default | Apply/solve | Diagonal extraction | MATLAB parity | Fallback/debug path |
|---|---|---|---|---|---|
| `core` | Direct Python utilities | N/A | N/A | Local tests plus downstream parity | None |
| `rskel` | Compact recursive skeletonization `D/U` blocks | Compact `rskel_mv` for normal, transpose, and adjoint modes | `rskel_xsp` sparse expansion | Symmetric, Hermitian, positive, rectangular complex, proxy, and upstream representative cases | None |
| `ifmm` | Compact IFMM `B/U` blocks and direct interaction discovery | Compact `ifmm_mv` for all store/near/symmetry modes | N/A | Store modes, near modes, symmetry, proxy, rectangular complex, upstream representative cases | Missing interactions can be supplied through callback/matrix `A` |
| `rskelf` | Compact recursive skeletonization elimination factors; `rskelf_structured` keeps tensor/block channels visible to proxy IDs | Compact `rskelf_mv`, `rskelf_sv`, Cholesky helpers, partial-factor helpers, and logdet | Selected inversion for complete factors; compact identity-RHS fallback for partial factors | Symmetry modes, complex, proxy, stop functions, transpose/adjoint solves, diagonal/spdiag modes; structured channel proxy tests | Partial factors use compact sweeps where selected inversion is not defined |
| `mf` | Hierarchical multifrontal `mf2`, `mf3`, and `mfx` factors | Hierarchical `mf_mv`, `mf_sv`, Cholesky helpers, and logdet | Selected inversion and sparse active-block `mf_spdiag` | Grid, point-cloud, complex, Hermitian, positive, singular/near-singular cases | `opts={"debug_dense": True}` uses the direct sparse backend |
| `hifie` | Hierarchical HIFIE builders with dimensional-reduction helpers | Compact rskelf-style `hifie_mv`, `hifie_sv`, Cholesky helpers, and logdet | Selected inversion and HIFIE sparse active-block `hifie_spdiag` | Compression callbacks, entry points, covariance/proxy representative cases | `opts={"debug_rskelf": True}` routes through plain `rskelf` |
| `hifde` | Hierarchical sparse elimination plus skeletonization for regular and point-cloud entry points | Compact rskelf-style `hifde_mv`, `hifde_sv`, Cholesky helpers, and logdet | Selected inversion and HIFDE sparse active-block `hifde_spdiag` | Grid and entry-point representative cases | None beyond lower-level complete/partial factor behavior |

## Performance Status

- Compact apply/solve kernels are faithful Python/NumPy/SciPy sweeps.
- Profiling results are recorded in `docs/flam_operation_profiles.md` and
  local ignored `benchmark_results/flam_operation_profiles.*` artifacts.
- The current low-risk optimization pass removes repeated dtype discovery from
  selected-inversion block unfolding and `rskel_mv`.
- Larger batching/JIT changes are deferred unless they can be made without
  obscuring the factor-sweep algorithms.
