PyFLAM
======

PyFLAM is a minimal NumPy/SciPy Python port of the public API from
[FLAM](https://github.com/klho/FLAM), centered on `hypoct`, `id`, `rskel`,
`rskelf`, `ifmm`, `mf`, `hifie`, `hifde`, and their apply/solve helpers.

The API intentionally follows the MATLAB function signatures, with one Python
adaptation: matrix and proxy callbacks receive 0-based NumPy index arrays.

The `rskelf`/`rskel` core contains compact factor application paths and avoids
eagerly materializing callback matrices. Some broader solver-family entry
points are currently correctness-first wrappers around dense or sparse
NumPy/SciPy factorizations; their public names and operation semantics are in
place for future faithful fast kernels.

This project is GPL-3.0-or-later because it is a faithful port of GPLv3 FLAM.
