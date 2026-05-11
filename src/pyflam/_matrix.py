"""Matrix access helpers."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.sparse as sp


def submatrix(A: Any, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    if callable(A):
        out = A(i, j)
    elif sp.issparse(A):
        out = A[np.ix_(i, j)]
    else:
        out = np.asarray(A)[np.ix_(i, j)]
    return np.asarray(out.toarray() if sp.issparse(out) else out)


def materialize(A: Any, m: int, n: int) -> np.ndarray:
    if callable(A):
        return submatrix(A, np.arange(m, dtype=np.int64), np.arange(n, dtype=np.int64))
    if sp.issparse(A):
        return np.asarray(A.toarray())
    arr = np.asarray(A)
    if arr.shape != (m, n):
        raise ValueError(f"matrix has shape {arr.shape}, expected {(m, n)}")
    return arr


def apply_transpose(A: np.ndarray, X: np.ndarray, trans: str) -> np.ndarray:
    if trans == "n":
        return A @ X
    if trans == "t":
        return A.T @ X
    return A.conj().T @ X


def infer_callback_shape(A: Callable[[np.ndarray, np.ndarray], np.ndarray], m: int, n: int) -> tuple[int, int]:
    return materialize(A, m, n).shape
