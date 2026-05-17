# FLAM Dependency Map

This map tracks the current Python port against the upstream MATLAB FLAM
families used by the public API.

## Core Layer

- `chktrans`, `chksymm`, `detperm`, `hypoct`, `hypoct_perm`, `id`, `ismemb`,
  `logdet_ldl`, `snorm`, and sparse helpers are implemented directly in
  Python.
- Matrix access uses structured callbacks through `submatrix`; callbacks
  receive 0-based NumPy index arrays.
- Pivoted QR, triangular solves, sparse matrices, sparse LU, LDL, and Cholesky
  use SciPy/NumPy primitives.

## Compression And Integral Kernels

- `rskel` depends on `hypoct`, `hypoct_perm`, `id`, and sparse/dense block
  access. Its `D/U` block layout and `rskel_mv` compact sweeps are ported.
- `ifmm` depends on `hypoct`, `hypoct_perm`, `id`, direct-interaction discovery,
  and proxy callbacks. All `store` modes and compact `ifmm_mv` paths are
  implemented.
- `rskelf` depends on `hypoct`, `id`, sparse update helpers, local LU/Cholesky
  blocks, and mode-specific apply/solve/logdet/selected-inversion sweeps.
  Complete callback factors avoid full dense matrix retention.
- `rskelf_structured` adapts block/tensor operators to `rskelf` while preserving
  row space, column space, output component, and input component metadata during
  proxy sampling. The factor still stores ordinary flat FLAM indices for
  apply/solve.

## Sparse And HIF Kernels

- `mf2`, `mf3`, and `mfx` build hierarchical multifrontal factors by default.
  Dense/SciPy sparse direct factorization is an explicit debug path.
- `hifie2`, `hifie2x`, `hifie3`, and `hifie3x` build hierarchical integral
  equation factors with dimensional-reduction helpers and compact rskelf-style
  operation sweeps.
- `hifde2`, `hifde3`, `hifde2x`, and `hifde3x` build hierarchical differential
  equation factors using sparse local elimination plus skeletonization and
  compact rskelf-style operation sweeps.
- `rskelf_diag`, `mf_diag`, `hifie_diag`, and `hifde_diag` use selected
  inversion for complete hierarchical factors. Sparse-style `spdiag` wrappers
  use FLAM-style active-block propagation.

## Current Port Order

1. Core utilities and sparse helper layer.
2. `rskel`, `ifmm`, and `rskelf` construction plus compact apply/solve/logdet.
3. MATLAB/ChunkIE parity harness with callback/proxy/quadrature-correction
   coverage.
4. Hierarchical `mf`, `hifie`, and `hifde` construction and operation wrappers.
5. Selected-inversion diagonal extraction.
6. Profiling, low-risk dtype/allocation cleanup, and documentation.

## Explicit Fallbacks

- `mf2`/`mf3`/`mfx` accept `opts={"debug_dense": True}` to use the direct sparse
  backend for debugging.
- `hifie*` accepts `opts={"debug_rskelf": True}` to route through the plain
  `rskelf` builder for debugging.
- `rskelf_diag` and related wrappers fall back to compact identity-RHS sweeps
  for partial factors because selected inversion is only defined for complete
  factors.
