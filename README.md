PyFLAM
======

PyFLAM is Haiya's NumPy/SciPy re-implementation of the public API from Ken
Ho's [FLAM](https://github.com/klho/FLAM), centered on `hypoct`, `id`,
`rskel`, `rskelf`, `ifmm`, `mf`, `hifie`, `hifde`, and their
apply/solve/diagonal helpers.

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

Full MATLAB/FLAM/ChunkIE parity requires MATLAB plus local reference checkouts.
The exact FLAM and ChunkIE reference commits used by the parity suite are
pinned in `pyproject.toml` under
`[tool.pyflam.test-reference-dependencies]`. The test harness validates those
clean commits before running MATLAB, including the upstream ChunkIE checkout
used by the current ChunkIE-style parity fixtures.

```powershell
$env:FLAM_REFERENCE='<path-to-FLAM-checkout>'
$env:CHUNKIE_REFERENCE='<path-to-ChunkIE-checkout>'
uv run python scripts\run_tests_with_matlab_parity.py
```

See `docs/parity_callbacks_release.md` for parity, callback, logdet, and
release-checklist details, and `docs/implementation_matrix.md` for the current
implementation matrix.

PyFLAM is GPL-3.0-or-later because it re-implements GPLv3 FLAM. See `NOTICE`
for upstream attribution to FLAM and ChunkIE.
