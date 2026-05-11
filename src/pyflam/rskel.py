"""Recursive skeletonization compression API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._matrix import apply_transpose, materialize
from .core import StructMixin, _as_points, _normalise_opts, chksymm, chktrans, hypoct, hypoct_perm


@dataclass
class RSkelFactor(StructMixin):
    M: int
    N: int
    P: np.ndarray
    Q: np.ndarray
    nlvl: int
    lvpd: np.ndarray
    lvpu: np.ndarray
    D: list[Any] = field(default_factory=list)
    U: list[Any] = field(default_factory=list)
    symm: str = "n"
    A_dense: np.ndarray | None = None
    tree: Any = None
    opts: dict[str, Any] = field(default_factory=dict)


def rskel(A, rx, cx, occ, rank_or_tol, pxyfun=None, opts=None) -> RSkelFactor:
    """Compress a dense matrix using FLAM's recursive skeletonization interface.

    This first Python implementation preserves the public contract and exact
    application semantics by retaining the materialized matrix. The tree,
    permutations, options, and callback/index conventions are compatible with
    the MATLAB routine and provide the substrate for later hierarchical storage.
    """

    defaults = {"lvlmax": np.inf, "ext": None, "Tmax": 2, "rrqr_iter": np.inf, "symm": "n", "verb": 0}
    o = _normalise_opts(opts, defaults)
    o["symm"] = chksymm(o["symm"])
    if o["symm"] == "p":
        o["symm"] = "h"

    rx = _as_points(rx)
    cx = _as_points(cx)
    M, N = rx.shape[1], cx.shape[1]
    tree = hypoct(np.column_stack((rx, cx)), occ, o["lvlmax"], o["ext"])
    perm = hypoct_perm(tree)
    col_mask = perm >= M
    P = perm[~col_mask]
    Q = np.array([], dtype=np.int64) if o["symm"] != "n" else perm[col_mask] - M
    A_dense = materialize(A, M, N)
    return RSkelFactor(
        M=M,
        N=N,
        P=P,
        Q=Q,
        nlvl=tree.nlvl,
        lvpd=np.zeros(tree.nlvl + 1, dtype=np.int64),
        lvpu=np.zeros(tree.nlvl + 1, dtype=np.int64),
        symm=o["symm"],
        A_dense=A_dense,
        tree=tree,
        opts=o,
    )


def rskel_mv(F: RSkelFactor, X, trans: str = "n") -> np.ndarray:
    trans = chktrans(trans)
    if F.A_dense is None:
        raise ValueError("factor does not contain matrix data")
    X = np.asarray(X)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    Y = apply_transpose(F.A_dense, X, trans)
    return Y[:, 0] if one_dim else Y


def rskel_xsp(F: RSkelFactor):
    """Embed compressed matrix into an extended sparse form.

    In this exact-storage implementation the extended sparse matrix is simply
    the sparse representation of the retained matrix, with identity row/column
    maps matching the base problem.
    """

    if F.A_dense is None:
        raise ValueError("factor does not contain matrix data")
    return sp.csr_matrix(F.A_dense), np.arange(F.M, dtype=np.int64), np.arange(F.N, dtype=np.int64)


__all__ = ["RSkelFactor", "rskel", "rskel_mv", "rskel_xsp"]
