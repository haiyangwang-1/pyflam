"""Interpolative fast multipole method API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._matrix import apply_transpose, materialize
from .core import StructMixin, _as_points, _normalise_opts, chksymm, chktrans, hypoct, hypoct_perm


@dataclass
class IFMMFactor(StructMixin):
    M: int
    N: int
    P: np.ndarray
    Q: np.ndarray
    nlvl: int
    lvpb: np.ndarray
    lvpu: np.ndarray
    B: list[Any] = field(default_factory=list)
    U: list[Any] = field(default_factory=list)
    store: str = "n"
    symm: str = "n"
    A_dense: np.ndarray | None = None
    tree: Any = None
    opts: dict[str, Any] = field(default_factory=dict)


def ifmm(A, rx, cx, occ, rank_or_tol, pxyfun=None, opts=None) -> IFMMFactor:
    defaults = {
        "lvlmax": np.inf,
        "ext": None,
        "Tmax": 2,
        "rrqr_iter": np.inf,
        "near": 0,
        "store": "n",
        "symm": "n",
        "verb": 0,
    }
    o = _normalise_opts(opts, defaults)
    o["store"] = str(o["store"]).lower()[0]
    if o["store"] not in {"n", "s", "r", "a"}:
        raise ValueError("storage parameter must be one of 'N', 'S', 'R', or 'A'")
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
    return IFMMFactor(
        M=M,
        N=N,
        P=P,
        Q=Q,
        nlvl=tree.nlvl + 1,
        lvpb=np.zeros(tree.nlvl + 3, dtype=np.int64),
        lvpu=np.zeros(tree.nlvl + 2, dtype=np.int64),
        store=o["store"],
        symm=o["symm"],
        A_dense=A_dense,
        tree=tree,
        opts=o,
    )


def ifmm_mv(F: IFMMFactor, X, A=None, trans: str = "n") -> np.ndarray:
    trans = chktrans(trans)
    mat = F.A_dense
    if mat is None:
        if A is None:
            raise ValueError("A must be supplied when interactions are not stored")
        mat = materialize(A, F.M, F.N)
    X = np.asarray(X)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    Y = apply_transpose(mat, X, trans)
    return Y[:, 0] if one_dim else Y


__all__ = ["IFMMFactor", "ifmm", "ifmm_mv"]
