# ChunkIE rskelf Parity Inventory

Inspected upstream files under `chunkie/devtools/test`:

The ChunkIE reference used by these tests is pinned in `pyproject.toml`:
commit `87cc6ea7828c0ef8bdc921171415b7918eb078f0` plus the tracked patch in
`tests/reference_patches/chunkie-87cc6ea7828c0ef8bdc921171415b7918eb078f0-tracked-dirty.patch`.
The untracked local `devtools/test/untitled.m` file was not used.

- `flamutilitiesTest.m`: direct FLAM utility coverage for `kernbyindex`,
  sparse near/self correction, `proxy_square_pts`, `proxyfun`, `rskelf`, and
  `rskel`.
- `chunkermatTest.m`: Laplace Dirichlet starfish dense solve and boundary
  evaluation baseline.
- `chunkermat_helm2dTest.m`: Helmholtz Dirichlet starfish dense solve baseline.
- `chunkermat_l2scaleTest.m`: l2-scaled Helmholtz transmission-style block
  assembly; the PyFLAM parity subset uses the same l2 weight scaling rule in a
  scalar Laplace case.
- `chunkermat_quadadapTest.m` and
  `chunkermat_quadadap_closetotouchingTest.m`: adaptive near-neighbor
  correction baselines. Current PyFLAM rskelf parity uses ChunkIE's sparse
  `chunkermat(..., nonsmoothonly=true, quad='ggq')` correction export, which is
  the same overwrite mechanism consumed by `chnk.flam.kernbyindex`.

Implemented Python parity cases:

- `tests/test_matlab_parity.py`: Laplace Dirichlet starfish and Helmholtz
  Dirichlet starfish, matching the original requested ChunkIE-style rskelf
  coverage.
- `tests/test_chunkie_rskelf_parity.py`: l2-scaled Laplace Dirichlet starfish
  and Helmholtz combined-layer starfish.

Each implemented case exports the discretization points, first and second
derivatives, normals, quadrature weights, sparse near/self correction matrix,
operator dimensions through the scalar ChunkIE FLAM path, proxy geometry and
weights, right-hand sides, MATLAB FLAM outputs, logdet, and the dense ChunkIE
matrix as a reference only. The Python side reconstructs matrix blocks with a
kernel callback, overwrites entries from the sparse correction matrix, and
supplies a proxy callback modeled on `chnk.flam.proxyfun`.

The benchmark record for representative ChunkIE-derived rskelf cases is in
`docs/rskelf_chunkie_benchmarks.md`, with raw outputs under
`benchmark_results/`.
