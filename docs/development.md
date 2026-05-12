# Development

Use `uv` for every Python environment command in this repository.

## Local Checks

Run the local unit layer when MATLAB or reference repositories are not needed:

```powershell
uv run python scripts\run_local_tests.py
uvx ruff check .
uv build
uv pip check
```

`scripts/run_local_tests.py` intentionally excludes MATLAB/FLAM/ChunkIE parity
tests. It is the fast check used by the Python CI job.

## MATLAB Parity

The full parity suite compares PyFLAM against pinned FLAM and ChunkIE commits.
ChunkIE is a submodule at `tests/references/chunkie`; initialize it before the
first parity run:

```powershell
git submodule update --init --recursive
```

FLAM is not vendored. Clone the pinned FLAM reference and point the harness at
it:

```powershell
git clone https://github.com/klho/FLAM.git tests\references\flam
git -C tests\references\flam checkout b928b2b1b4e0c3a00558bcdc7e3147fe83e720c4

$env:FLAM_REFERENCE = "$PWD\tests\references\flam"
uv run python scripts\run_tests_with_matlab_parity.py
```

Set `CHUNKIE_REFERENCE` only when you need to use a checkout other than the
repo submodule. The harness validates that both references are clean and pinned
before it launches MATLAB.

## CI Layout

`.github/workflows/ci.yml` runs on pushes, pull requests, and manual dispatch:

- `python`: Python 3.11 and 3.12, local tests, Ruff, package build, dependency
  metadata check.
- `matlab-parity`: one Python version, MATLAB setup, pinned FLAM clone, ChunkIE
  submodule checkout, full parity runner.

Do not run multiple MATLAB parity jobs in parallel from the same machine unless
you know the local MATLAB license and startup behavior are stable under that
load.

## Naming Style

Use descriptive Python names in new code. Avoid carrying over MATLAB-only names
such as `I`, `J`, `V`, or `l` unless they appear inside MATLAB source strings
used for parity exports. Public callback behavior remains positional, so local
callback implementations can name the FLAM box-width argument `box_size`.
