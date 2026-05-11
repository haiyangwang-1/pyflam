PyFLAM
======

PyFLAM is a NumPy/SciPy Python port of the public API from
[FLAM](https://github.com/klho/FLAM), centered on `hypoct`, `id`, `rskel`,
`rskelf`, `ifmm`, `mf`, `hifie`, `hifde`, and their apply/solve/diagonal
helpers.

The API intentionally follows the MATLAB function signatures, with one Python
adaptation: matrix and proxy callbacks receive 0-based NumPy index arrays.

Current default implementations use compact FLAM-style factors for `rskel`,
`ifmm`, `rskelf`, true hierarchical multifrontal factors for `mf2`/`mf3`/`mfx`,
and hierarchical HIFIE/HIFDE factor builders for the public `hifie*` and
`hifde*` entry points. Callback-based `rskel`, `ifmm`, and `rskelf` factors do
not eagerly materialize full matrices.

Dense or sparse direct factorizations remain only as explicit debug/fallback
paths, such as `opts={"debug_dense": True}` for `mf` and
`opts={"debug_rskelf": True}` for `hifie`. Diagonal extraction for complete
hierarchical factors uses selected-inversion paths rather than solving against
a full identity RHS.

Use `uv` for the package environment:

```powershell
uv run python -m unittest discover -s tests -v
```

Full MATLAB/FLAM/ChunkIE parity requires local reference checkouts:

```powershell
$env:FLAM_REFERENCE='C:\Users\haiya\git\FLAM'
$env:CHUNKIE_REFERENCE='C:\Users\haiya\git\chunkie'
uv run python scripts\run_tests_with_matlab_parity.py
```

See `docs/parity_callbacks_release.md` for parity, callback, logdet, and
release-checklist details, and `docs/implementation_matrix.md` for the current
implementation matrix.

This project is GPL-3.0-or-later because it is a faithful port of GPLv3 FLAM.
