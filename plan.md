# PyFLAM Full FLAM Implementation Plan

This plan is intended for agents to execute sequentially. Use `uv` for the
project Python environment. MATLAB parity is mandatory: do not skip MATLAB,
FLAM, or ChunkIE parity tests when those dependencies are present. If a parity
test cannot run, fail loudly and explain the missing dependency or reference
bug instead of silently skipping.

## Execution Rules

- [ ] Before editing, check `git status --short` and do not overwrite unrelated
      user changes.
- [ ] Prefer small, verifiable commits that each move one checklist section
      forward.
- [ ] For every algorithmic change, add or extend parity tests against MATLAB
      FLAM, plus local residual/dense-reference tests where practical.
- [ ] Keep callback semantics Pythonic and documented: callbacks receive
      0-based NumPy index arrays.
- [ ] When using ChunkIE parity, export enough data from MATLAB to reconstruct
      the matrix in Python from geometry, weights, quadrature corrections, and
      kernel/proxy callbacks. Do not reduce these tests to dense-matrix-only
      comparisons.
- [ ] Run the full suite with MATLAB/ChunkIE parity enabled before considering
      a phase complete:
      `uv run python -m unittest discover -s tests -v`.

## 1. Lock The Parity Harness

- [x] Keep MATLAB parity tests non-skipping by default. Missing MATLAB, FLAM, or
      ChunkIE should produce a clear failure.
- [x] Consolidate repeated MATLAB driver code in `tests/test_matlab_parity.py`
      into helper functions for writing scripts, running MATLAB, and loading
      `.mat` files.
- [x] Add helpers for logdet branch comparison modulo `2*pi*i`.
- [x] Add helpers for exporting FLAM factor metadata needed for structural
      comparisons: level pointers, factor counts, skeleton/redundant sizes,
      sparse correction nnz, and proxy-call evidence.
- [x] Preserve the existing two ChunkIE-style `rskelf` parity tests:
      Laplace Dirichlet starfish and Helmholtz Dirichlet starfish.
- [x] Add at least two more ChunkIE-derived tests from
      `chunkie/devtools/test`, prioritizing cases that use near/self quadrature
      correction and proxy compression.

## 2. Expand ChunkIE-Based `rskelf` Parity

- [x] Inspect `chunkie/devtools/test/flamutilitiesTest.m`,
      `chunkermatTest.m`, `chunkermat_helm2dTest.m`,
      `chunkermat_l2scaleTest.m`, and adaptive-correction tests.
- [x] For each selected ChunkIE case, export from MATLAB:
      discretization points, derivatives, normals, weights, operator
      dimensions, sparse near/self correction matrix, proxy points/tangents/
      weights, right-hand sides, MATLAB FLAM outputs, and dense system matrix
      only as a reference.
- [x] Implement the matching Python kernel callback rather than calling back to
      the exported dense matrix.
- [x] Add sparse quadrature correction overwrite logic to the Python callback.
- [x] Add Python proxy callback matching `chnk.flam.proxyfun`.
- [x] Verify callback matrix reconstruction against ChunkIE dense matrix.
- [x] Verify PyFLAM `rskelf_mv`, `rskelf_sv`, solve residual, and logdet against
      MATLAB FLAM and the ChunkIE reference matrix.
- [x] Benchmark representative sizes and record performance versus MATLAB FLAM
      and ChunkIE dense assembly.

## 3. Finish `rskelf` Fidelity

- [x] Add MATLAB parity for `rskelf` with `symm='s'`.
- [x] Add MATLAB parity for `rskelf` with `symm='h'`.
- [x] Add MATLAB parity for `rskelf` with `symm='p'`.
- [x] Add complex matrix parity for each supported symmetry mode.
- [x] Add proxy-callback parity for each supported symmetry mode where MATLAB
      FLAM supports it.
- [x] Add partial-factorization parity for scalar `stop` and callable/nontrivial
      stop functions.
- [x] Verify compact `rskelf_mv` for `trans='n'`, `'t'`, and `'c'`.
- [x] Verify compact `rskelf_sv` for `trans='n'`, `'t'`, and `'c'`.
- [x] Verify `rskelf_cholmv` and `rskelf_cholsv` for positive-definite factors.
- [x] Verify `rskelf_logdet` for real, complex, symmetric, Hermitian, positive,
      and partial factorizations.
- [x] Ensure callback-based factors do not require eager dense materialization.
- [x] Port or validate all upstream mode helper equivalents under
      `FLAM/rskelf/mv`, `FLAM/rskelf/sv`, and `FLAM/rskelf/spdiag`.

## 4. Finish `rskel` Fidelity

- [x] Add MATLAB parity for unsymmetric `rskel` with callback matrix access.
- [x] Add MATLAB parity for `symm='s'`.
- [x] Add MATLAB parity for `symm='h'`.
- [x] Add MATLAB parity for `symm='p'` and confirm MATLAB semantics map to the
      Python factor mode.
- [x] Add complex rectangular cases.
- [x] Add proxy-callback parity for row and column proxy paths.
- [x] Add full `rskel_xsp` parity for symmetric/Hermitian/positive cases.
- [x] Verify `rskel_mv` for `trans='n'`, `'t'`, and `'c'`.
- [x] Ensure callback-based factors avoid eager dense materialization.
- [x] Cover representative upstream tests under `FLAM/rskel/test`.

## 5. Finish `ifmm` Fidelity

- [x] Add MATLAB parity for every `store` mode: `'n'`, `'s'`, `'r'`, and `'a'`.
- [x] Add MATLAB parity for `near=0` and `near=1`.
- [x] Add MATLAB parity for `symm='n'`, `'s'`, `'h'`, and `'p'` where supported.
- [x] Add proxy callback parity.
- [x] Add rectangular complex matrix parity.
- [x] Verify `ifmm_mv` for `trans='n'`, `'t'`, and `'c'`.
- [x] Verify behavior when missing interactions must be supplied through `A`.
- [x] Cover representative upstream tests under `FLAM/ifmm/test`.

## 6. Complete Stress Coverage

- [x] Cover empty/singleton trees.
- [x] Cover repeated/degenerate points that cannot be split.
- [x] Cover high-dimensional tree child-code overflow.
- [x] Add degenerate-point end-to-end tests for `rskelf`, `rskel`, and `ifmm`.
- [x] Add rank-cap versus tolerance-mode tests for `id`.
- [x] Add `id` tests with fixed columns, complex inputs, empty matrices, and
      rank-deficient matrices.
- [x] Add nontrivial callable `stop` tests for partial `rskelf`.
- [x] Cover complex sparse LU logdet and transpose/adjoint solves in `mf`.
- [x] Add sparse singular/near-singular failure-mode tests where MATLAB FLAM has
      defined behavior.

## 7. Implement True FLAM `mf`

- [x] Study upstream `FLAM/mf/mf2.m`, `mf3.m`, `mfx.m`, `mv/*`, `sv/*`, and
      `spdiag/*`.
- [x] Replace correctness-first dense/SciPy sparse LU backend with FLAM
      hierarchical multifrontal construction for `mf2`.
- [x] Implement true `mf3`.
- [ ] Implement true point-cloud `mfx`.
- [ ] Port factor block layout and level pointers to match MATLAB FLAM.
- [ ] Port `mf_mv_nn`, `mf_mv_nc`, `mf_mv_h`, and `mf_mv_p` behavior.
- [ ] Port `mf_sv_nn`, `mf_sv_nc`, `mf_sv_h`, and `mf_sv_p` behavior.
- [ ] Port `mf_cholmv`, `mf_cholsv`, and `mf_logdet` to use hierarchical factors.
- [ ] Add MATLAB parity for `mf2`, `mf3`, and `mfx`.
- [ ] Add parity for real, complex, symmetric, Hermitian, positive-definite,
      transpose solve, and adjoint solve cases.
- [ ] Keep dense/SciPy sparse fallback only as an explicit debug fallback, not
      the default implementation path.

## 8. Implement True HIFIE

- [ ] Study upstream `FLAM/hifie/base/hifie2_base.m`,
      `hifie3_base.m`, `hifie_id.m`, and `hifie_idx.m`.
- [ ] Define Python HIFIE factor dataclasses rather than aliasing `RSkelFFactor`.
- [ ] Port the internal dimensional-reduction helper family.
- [ ] Implement true `hifie2`.
- [ ] Implement true `hifie2x`.
- [ ] Implement true `hifie3`.
- [ ] Implement true `hifie3x`.
- [ ] Port `hifie_mv`, `hifie_sv`, `hifie_logdet`, `hifie_cholmv`, and
      `hifie_cholsv`.
- [ ] Port `hifie_diag` and `hifie_spdiag` selected-inversion paths.
- [ ] Add MATLAB parity from representative upstream tests under
      `FLAM/hifie/test`.
- [ ] Ensure public behavior remains compatible with current wrappers while
      removing the `rskelf` routing as the default implementation.

## 9. Implement True HIFDE

- [ ] Study upstream `FLAM/hifde/hifde2.m`, `hifde3.m`, `hifde2x.m`,
      `hifde3x.m`, `mv/*`, `sv/*`, and `spdiag/*`.
- [ ] Build HIFDE on top of the true hierarchical `mf` implementation.
- [ ] Define Python HIFDE factor dataclasses rather than aliasing `MFFactor`.
- [ ] Implement true `hifde2`.
- [ ] Implement true `hifde3`.
- [ ] Implement true `hifde2x`.
- [ ] Implement true `hifde3x`.
- [ ] Port `hifde_mv`, `hifde_sv`, `hifde_logdet`, `hifde_cholmv`, and
      `hifde_cholsv`.
- [ ] Port `hifde_diag` and `hifde_spdiag` selected-inversion paths.
- [ ] Add MATLAB parity from representative upstream tests under
      `FLAM/hifde/test`.
- [ ] Ensure public behavior remains compatible with current wrappers while
      removing the `mf` routing as the default implementation.

## 10. Implement Fast Selected Inversion

- [ ] Port `rskelf/spdiag/*` selected-inversion algorithms.
- [ ] Replace correctness-first `rskelf_diag` identity-RHS extraction with the
      selected-inversion implementation.
- [ ] Port `mf/spdiag/*` selected-inversion algorithms.
- [ ] Replace correctness-first `mf_diag` identity-RHS extraction.
- [ ] Port `hifie/spdiag/*` selected-inversion algorithms.
- [ ] Replace correctness-first `hifie_diag`.
- [ ] Port `hifde/spdiag/*` selected-inversion algorithms.
- [ ] Replace correctness-first `hifde_diag`.
- [ ] Add MATLAB parity for diagonal and inverse diagonal extraction across
      unsymmetric, symmetric, Hermitian, and positive-definite modes.
- [ ] Add memory/performance tests showing diagonal extraction no longer scales
      like solving against a full identity matrix.

## 11. Performance Optimization

- [ ] Profile build, apply, solve, logdet, and selected inversion separately on
      representative dense-kernel and sparse-grid cases.
- [ ] Identify Python-loop hotspots in compact `rskelf`, `rskel`, `ifmm`, `mf`,
      `hifie`, and `hifde` sweeps.
- [ ] Batch small triangular solves where it does not obscure the algorithm.
- [ ] Reduce repeated indexing, allocation, and dtype conversion in sweep loops.
- [ ] Consider `numba` or another JIT only if it can be enabled locally with
      small, readable changes and no major rewrite.
- [ ] Abort or defer any optimization that requires a large, hard-to-audit code
      change.
- [ ] Keep accuracy parity fixed while optimizing.
- [ ] Record benchmark results in `docs/` and `benchmark_results/`.

## 12. Documentation Cleanup

- [ ] Update `README.md` to distinguish implemented faithful FLAM kernels from
      any explicit fallback/debug paths.
- [ ] Update `docs/flam_dependency_map.md`; it still describes older dense-core
      phases and should match the current implementation.
- [ ] Document how to run MATLAB parity and ChunkIE parity with required
      environment variables.
- [ ] Document callback conventions, including 0-based indexing and sparse
      correction overwrite semantics.
- [ ] Document known numerical differences such as complex logdet branch
      handling.
- [ ] Add a current implementation matrix for `rskelf`, `rskel`, `ifmm`, `mf`,
      `hifie`, and `hifde`.
- [ ] Add a release checklist requiring full non-skipped MATLAB/ChunkIE parity.

## Final Acceptance Criteria

- [ ] Public APIs match MATLAB FLAM signatures and semantics, with documented
      Python indexing conventions.
- [ ] All representative upstream FLAM test families have Python parity tests.
- [ ] At least four ChunkIE-style tests reconstruct operators in Python from
      geometry, quadrature corrections, kernels, and proxies.
- [ ] `rskelf`, `rskel`, and `ifmm` operate from compact factors without dense
      matrix retention for callback factors.
- [ ] `mf`, `hifie`, and `hifde` use true FLAM hierarchical fast algorithms by
      default.
- [ ] Diagonal extraction uses selected inversion, not identity-RHS solves.
- [ ] Full test command passes with MATLAB/FLAM/ChunkIE parity enabled and no
      parity skips.
- [ ] Documentation accurately reflects the implementation.
