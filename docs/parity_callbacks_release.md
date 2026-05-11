# Parity, Callbacks, And Release Checks

## MATLAB And ChunkIE Parity

Use `uv` and set both reference paths before running parity. Missing MATLAB,
FLAM, or ChunkIE should be treated as a failed environment setup, not as a
silent skip.

```powershell
$env:FLAM_REFERENCE='<path-to-FLAM-checkout>'
$env:CHUNKIE_REFERENCE='<path-to-ChunkIE-checkout>'
uv run python scripts\run_tests_with_matlab_parity.py
```

The harness validates that `FLAM_REFERENCE` contains the public entry-point
files used by the parity suite, including `rskelf/rskelf.m`, `rskel/rskel.m`,
`ifmm/ifmm.m`, `mf/mf2.m`, `hifie/hifie2.m`, and `hifde/hifde2.m`.
This avoids opaque MATLAB errors from incomplete helper checkouts.

The direct full-suite command is:

```powershell
$env:FLAM_REFERENCE='<path-to-FLAM-checkout>'
$env:CHUNKIE_REFERENCE='<path-to-ChunkIE-checkout>'
uv run python -m unittest discover -s tests -v
```

Representative targeted parity commands:

```powershell
$env:FLAM_REFERENCE='<path-to-FLAM-checkout>'
uv run python -m unittest discover -s tests -p test_rskelf_option_parity.py -v -k diag
uv run python -m unittest discover -s tests -p test_rskel_option_parity.py -v
uv run python -m unittest discover -s tests -p test_ifmm_option_parity.py -v

$env:CHUNKIE_REFERENCE='<path-to-ChunkIE-checkout>'
uv run python -m unittest discover -s tests -p test_matlab_parity.py -v
uv run python -m unittest discover -s tests -p test_chunkie_rskelf_parity.py -v
```

Do not launch multiple MATLAB parity jobs in parallel. MATLAB startup, path
setup, and license/resource handling are more reliable serially.
The shared MATLAB runner retries failed MATLAB subprocesses twice by default;
set `PYFLAM_MATLAB_RETRIES=0` to disable retries when debugging a persistent
reference failure.

## Callback Conventions

- Python matrix callbacks receive 0-based NumPy integer arrays.
- A matrix callback should return a dense block with shape
  `(len(row_indices), len(col_indices))`.
- Proxy callbacks receive 0-based skeleton/self indices and should return the
  proxy interaction matrix plus any replacement neighbor indices using the same
  0-based convention.
- ChunkIE-style parity reconstructs operators from geometry, weights, sparse
  quadrature corrections, kernels, and proxy callbacks. The exported dense
  ChunkIE matrix is a reference only.
- Sparse near/self quadrature correction overwrites the matching kernel entries
  in Python callback blocks. This mirrors ChunkIE's `kernbyindex` behavior:
  corrected sparse entries take precedence over the analytic far-kernel value.

## Logdet Branch Handling

Complex `logdet` values can differ from MATLAB by integer multiples of
`2*pi*i`. Parity tests compare the raw difference where useful and also compare
the branch-adjusted difference through helpers such as `logdet_mod_error`.
When validating complex factors manually, determinant equivalence through
`exp(py_logdet - matlab_logdet)` is usually the more stable check.

## Release Checklist

- Run `uv run python -m unittest discover -s tests -v` with
  `FLAM_REFERENCE` and `CHUNKIE_REFERENCE` set.
- Confirm MATLAB/FLAM/ChunkIE parity tests ran rather than being excluded by a
  missing local dependency.
- Run the ChunkIE rskelf parity tests that reconstruct operators from kernels,
  quadrature corrections, and proxy callbacks.
- Run the selected-diagonal and operation-profile benchmarks when changing
  diagonal extraction or compact sweep performance.
- Update `docs/implementation_matrix.md` when any API moves between default,
  fallback, or unsupported status.
- Record known upstream MATLAB reference failures as explicit notes in tests or
  docs; do not silently skip them.
- Keep generated benchmark outputs under `benchmark_results/` local and
  ignored; commit only curated benchmark summaries in `docs/`.
