PyFLAM
======

PyFLAM is Haiyang's NumPy/SciPy re-implementation of the public API from Ken
Ho's [FLAM](https://github.com/klho/FLAM), centered on `hypoct`, `id`,
`rskel`, `rskelf`, `ifmm`, `mf`, `hifie`, `hifde`, and their
apply/solve/diagonal helpers.

The API intentionally follows the MATLAB function signatures where practical,
with Pythonic adaptations for data and indexing: matrix and proxy callbacks
receive 0-based NumPy index arrays, and Python result objects use descriptive
fields such as `HypOctTree.widths` instead of MATLAB's single-letter struct
field names.

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

Use `uv` for the package environment. The local unit layer does not require
MATLAB or external reference repositories:

```powershell
uv run python scripts\run_local_tests.py
uvx ruff check .
```

Full MATLAB/FLAM/ChunkIE parity requires MATLAB plus the pinned reference
checkouts in `pyproject.toml` under
`[tool.pyflam.test-reference-dependencies]`. FLAM and ChunkIE are tracked as
test-only Git submodules under `tests/references/`:

```powershell
git submodule update --init --recursive
uv run python scripts\run_tests_with_matlab_parity.py
```

`FLAM_REFERENCE` and `CHUNKIE_REFERENCE` can still override the submodule paths
when needed. The parity harness validates both reference commits before running
MATLAB.

Start with `docs/quickstart.md` for examples, `docs/development.md` for CI and
test workflows, `docs/parity_callbacks_release.md` for parity/callback/logdet
details, and `docs/implementation_matrix.md` for the current implementation
matrix.

PyFLAM is GPL-3.0-or-later because it re-implements GPLv3 FLAM. See `NOTICE`
for upstream attribution to FLAM and ChunkIE.
