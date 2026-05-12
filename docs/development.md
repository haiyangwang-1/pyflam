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
Both references are test-only submodules under `tests/references/`; initialize
them before the first parity run:

```powershell
git submodule update --init --recursive
```

```powershell
uv run python scripts\run_tests_with_matlab_parity.py
```

Set `FLAM_REFERENCE` or `CHUNKIE_REFERENCE` only when you need to use a checkout
other than the repo submodules. The harness validates that both references are
clean and pinned before it launches MATLAB.

## CI Layout

`.github/workflows/ci.yml` runs on pushes, pull requests, and manual dispatch:

- `python`: Python 3.11 and 3.12, local tests, Ruff, package build, dependency
  metadata check.
- `matlab-parity`: one Python version, MATLAB setup, pinned FLAM clone, ChunkIE
  submodule checkout, full parity runner.

The hosted MATLAB parity job sets `MW_BATCH_LICENSING_ONLINE=true` before
launching the Python harness, downloads MathWorks' `run-matlab-command`
launcher, and sets `PYFLAM_MATLAB_LAUNCHER=run-matlab-command`. The harness
starts many MATLAB subprocesses directly, so hosted CI uses the same command
launcher family as MathWorks' `run-command` action on public GitHub-hosted
runners.

Do not run multiple MATLAB parity jobs in parallel from the same machine unless
you know the local MATLAB license and startup behavior are stable under that
load.

## Naming Style

Use descriptive Python names in new code. Avoid carrying over MATLAB-only names
such as `I`, `J`, `V`, or `l` unless they appear inside MATLAB source strings
used for parity exports. Public callback behavior remains positional, so local
callback implementations can name the FLAM box-width argument `box_size`.
