# FLAM Dependency Map

This map was derived from the upstream MATLAB reference at commit `2c9361e`.
It is intended to guide the Python port from the bottom up.

## Upstream Function Counts

- `core`: 14 functions
- `compat`: 4 functions
- `ifmm`: 2 library functions plus examples/tests
- `rskel`: 3 library functions plus examples/tests
- `rskelf`: 18 library functions plus examples/tests and mode-specific helpers
- `mf`, `hifie`, `hifde`: out of scope for the dense-core pass

## Dense-Core Dependency Closure

The target APIs depend on the following upstream routines:

- `hypoct`: no FLAM dependencies
- `hypoct_perm`: no FLAM dependencies
- `id`: no FLAM dependencies beyond MATLAB QR primitives
- `rskel`: `chksymm`, `hypoct`, `hypoct_perm`, `id`
- `rskel_mv`: `chktrans`
- `rskel_xsp`: no FLAM dependencies
- `ifmm`: `chksymm`, `hypoct`, `hypoct_perm`, `id`, `ismemb`
- `ifmm_mv`: `chktrans`
- `rskelf`: `chksymm`, `hypoct`, `id`, `isoctave`
- `rskelf_mv`: `chktrans`, mode helpers under `rskelf/mv`
- `rskelf_sv`: `chktrans`, mode helpers under `rskelf/sv`
- `rskelf_logdet`: `detperm`, `logdet_ldl`
- `rskelf_partial_*`: `chktrans`, `rskelf_mv_*`, `rskelf_sv_*`

## Port Order

1. Core utilities: `chktrans`, `chksymm`, `hypoct`, `hypoct_perm`, `id`, `snorm`.
2. Sparse utility layer: `spget`, `spgetv`, `spaddv`, `sppush2`, `sppush3`, `spsymm`, `spsymm2`, `detperm`, `ismemb`, `logdet_ldl`.
3. Compression/factor construction: `rskel`, `ifmm`, `rskelf`.
4. Apply/solve/logdet routines: `rskel_mv`, `ifmm_mv`, `rskelf_mv`, `rskelf_sv`, `rskelf_logdet`.
5. Optional selected/partial helpers once full hierarchical factors replace dense fallback storage.

## Current Implementation Status

- Implemented directly: core tree, permutation, ID, norm estimate, sparse utilities, public API dataclasses.
- Implemented as bottom-up FLAM-style construction: `rskel` `D/U` blocks and `rskelf` skeleton/elimination factor blocks.
- Implemented as foundational IFMM construction: tree/direct interaction discovery and self/direct `B` block storage.
- Still using exact dense retained matrices for public apply/solve/logdet correctness while hierarchical sweep ports are underway.
- Not yet implemented as full hierarchical sweeps: `rskel_mv`, `ifmm_mv`, and mode-specific `rskelf_{mv,sv}_*` routines operating only from compact factors.
