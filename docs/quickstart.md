# Quickstart

PyFLAM exposes FLAM-style hierarchical matrix routines through NumPy and
SciPy objects. Inputs are Python arrays, and callback indices are 0-based
NumPy integer arrays.

## Install And Import

From the repository root:

```powershell
uv sync --locked
uv run python -c "import pyflam; print(len(pyflam.__all__))"
```

Typical imports:

```python
import numpy as np

from pyflam import rskelf, rskelf_mv, rskelf_sv, rskelf_logdet
```

## Dense Kernel Factorization

`rskelf` builds a compact factorization from either a dense matrix or a matrix
callback. The factor does not retain a dense matrix when the input is a
callback.

```python
import numpy as np

from pyflam import rskelf, rskelf_mv, rskelf_sv, rskelf_logdet

n = 128
t = np.linspace(0.0, 1.0, n)
x = np.vstack((t, 0.2 * np.sin(2 * np.pi * t)))


def kernel(rows, cols):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    dist = np.linalg.norm(x[:, rows, None] - x[:, None, cols], axis=0)
    return np.exp(-dist / 0.15) + 2.0 * (rows[:, None] == cols[None, :])


factor = rskelf(kernel, x, occ=16, rank_or_tol=1e-10, opts={"symm": "p"})
rhs = np.ones((n, 2))

applied = rskelf_mv(factor, rhs)
solved = rskelf_sv(factor, rhs)
logdet = rskelf_logdet(factor)
```

## Compression-Only Apply

`rskel` and `ifmm` build compact apply-only factors. Use `ifmm` when direct and
far interactions are useful for repeated matrix-vector products.

```python
from pyflam import ifmm, ifmm_mv

factor = ifmm(kernel, x, x, occ=16, rank_or_tol=1e-10, opts={"store": "a", "near": 1, "symm": "p"})
y = ifmm_mv(factor, np.ones(n), A=kernel)
```

## Sparse Grid Operators

For sparse grid matrices, use `mf2`/`mf3`/`mfx` or the HIFDE wrappers. The
regular-grid entry points accept the grid size `n`; the matrix dimension is
`(n - 1) ** dim`.

```python
import scipy.sparse as sp

from pyflam import mf2, mf_sv

n = 17
nd = n - 1
main = 4.0 * np.ones(nd * nd)
A = sp.diags(main, format="lil")
for row in range(nd * nd):
    if row % nd:
        A[row, row - 1] = -1.0
    if row % nd != nd - 1:
        A[row, row + 1] = -1.0
    if row >= nd:
        A[row, row - nd] = -1.0
    if row + nd < nd * nd:
        A[row, row + nd] = -1.0
A = A.tocsc()

factor = mf2(A, n, occ=8, opts={"symm": "p"})
solution = mf_sv(factor, np.ones(A.shape[0]))
```

## Callback Rules

- Matrix callbacks receive `(rows, cols)` as 0-based integer arrays and return
  a dense block of shape `(len(rows), len(cols))`.
- Proxy callbacks are called positionally. Use descriptive Python parameter
  names such as `box_size` and `center`; the value corresponds to FLAM's
  box-width argument.
- Sparse near/self quadrature corrections should overwrite analytic kernel
  entries when reconstructing ChunkIE-style operators.
