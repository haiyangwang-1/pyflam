"""Recursive skeletonization factorization API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as la

from ._matrix import apply_transpose, materialize
from .core import StructMixin, _as_points, _normalise_opts, chksymm, chktrans, hypoct


@dataclass
class RSkelFFactor(StructMixin):
    N: int
    nlvl: int
    lvp: np.ndarray
    factors: list[Any] = field(default_factory=list)
    symm: str = "n"
    A_dense: np.ndarray | None = None
    lu: tuple[np.ndarray, np.ndarray] | None = None
    chol: np.ndarray | None = None
    tree: Any = None
    opts: dict[str, Any] = field(default_factory=dict)
    Si: np.ndarray | None = None
    S: np.ndarray | None = None


def rskelf(A, x, occ, rank_or_tol, pxyfun=None, opts=None) -> RSkelFFactor:
    defaults = {
        "lvlmax": np.inf,
        "ext": None,
        "Tmax": 2,
        "rrqr_iter": np.inf,
        "symm": "n",
        "stop": np.inf,
        "verb": 0,
    }
    o = _normalise_opts(opts, defaults)
    o["symm"] = chksymm(o["symm"])
    x = _as_points(x)
    N = x.shape[1]
    tree = hypoct(x, occ, o["lvlmax"], o["ext"])
    A_dense = materialize(A, N, N)
    lu = None
    chol = None
    if o["symm"] == "p":
        chol = la.cholesky(A_dense, lower=True)
    else:
        lu = la.lu_factor(A_dense)
    return RSkelFFactor(
        N=N,
        nlvl=tree.nlvl,
        lvp=np.zeros(tree.nlvl + 1, dtype=np.int64),
        symm=o["symm"],
        A_dense=A_dense,
        lu=lu,
        chol=chol,
        tree=tree,
        opts=o,
        Si=np.arange(N, dtype=np.int64),
        S=np.zeros((N, N), dtype=A_dense.dtype),
    )


def rskelf_mv(F: RSkelFFactor, X, trans: str = "n") -> np.ndarray:
    trans = chktrans(trans)
    if F.A_dense is None:
        raise ValueError("factor does not contain matrix data")
    X = np.asarray(X)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    Y = apply_transpose(F.A_dense, X, trans)
    return Y[:, 0] if one_dim else Y


def rskelf_sv(F: RSkelFFactor, X, trans: str = "n") -> np.ndarray:
    trans = chktrans(trans)
    X = np.asarray(X)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    if F.chol is not None:
        if trans == "n":
            Y = la.cho_solve((F.chol, True), X)
        elif trans == "t":
            Y = la.solve(F.A_dense.T, X, assume_a="sym")
        else:
            Y = la.solve(F.A_dense.conj().T, X, assume_a="her")
    elif F.lu is not None:
        if trans == "n":
            Y = la.lu_solve(F.lu, X)
        elif trans == "t":
            Y = la.lu_solve(F.lu, X, trans=1)
        else:
            Y = la.lu_solve(F.lu, X, trans=2)
    else:
        raise ValueError("factor does not contain solve data")
    return Y[:, 0] if one_dim else Y


def rskelf_logdet(F: RSkelFFactor):
    if F.A_dense is None:
        raise ValueError("factor does not contain matrix data")
    sign, ld = np.linalg.slogdet(F.A_dense)
    if np.iscomplexobj(F.A_dense):
        return np.log(np.linalg.det(F.A_dense))
    if sign <= 0:
        return np.log(sign) + ld
    return ld


def rskelf_partial_info(F: RSkelFFactor):
    if F.Si is None:
        return np.array([], dtype=np.int64), np.zeros((0, 0))
    return F.Si.copy(), np.array(F.S, copy=True)


def rskelf_partial_mv(F: RSkelFFactor, X, mvfun=None, trans: str = "n") -> np.ndarray:
    return rskelf_mv(F, X, trans)


def rskelf_partial_sv(F: RSkelFFactor, X, svfun=None, trans: str = "n") -> np.ndarray:
    return rskelf_sv(F, X, trans)


__all__ = [
    "RSkelFFactor",
    "rskelf",
    "rskelf_logdet",
    "rskelf_mv",
    "rskelf_partial_info",
    "rskelf_partial_mv",
    "rskelf_partial_sv",
    "rskelf_sv",
]
