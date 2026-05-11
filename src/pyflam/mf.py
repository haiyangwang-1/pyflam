"""Multifrontal factorization public API.

These routines expose FLAM's ``mfx``/``mf2``/``mf3`` interface.  The current
implementation is correctness-first: it preserves the MATLAB-like factor
object and operation surface while using dense NumPy/SciPy factorizations as
the numerical backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._matrix import apply_transpose, materialize
from .core import StructMixin, _as_points, _normalise_opts, chksymm, chktrans, detperm, hypoct


@dataclass
class MFFactorBlock(StructMixin):
    sk: np.ndarray
    rd: np.ndarray
    L: np.ndarray
    U: np.ndarray | None = None
    p: np.ndarray | None = None
    E: np.ndarray | None = None
    F: np.ndarray | None = None


@dataclass
class MFFactor(StructMixin):
    N: int
    nlvl: int
    lvp: np.ndarray
    factors: list[MFFactorBlock] = field(default_factory=list)
    symm: str = "n"
    A: Any = None
    A_dense: np.ndarray | None = None
    A_sparse: sp.spmatrix | None = None
    lu: tuple[np.ndarray, np.ndarray] | None = None
    splu: spla.SuperLU | None = None
    chol: np.ndarray | None = None
    tree: Any = None
    opts: dict[str, Any] = field(default_factory=dict)


def mfx(A, x, occ, opts: dict[str, Any] | None = None) -> MFFactor:
    """Factor a sparse matrix using FLAM's point-cloud multifrontal API."""

    defaults = {"lvlmax": np.inf, "ext": None, "symm": "n", "verb": 0}
    o = _normalise_opts(opts, defaults)
    o["symm"] = _mf_symm(o["symm"])
    x = _as_points(x)
    N = x.shape[1]
    if occ <= 0:
        raise ValueError("leaf occupancy must be positive")
    tree = hypoct(x, occ, o["lvlmax"], o["ext"])
    return _make_factor(A, N, o, tree=tree)


def mf2(A, n: int, occ: int, opts: dict[str, Any] | None = None) -> MFFactor:
    """Factor a matrix on a regular ``(n - 1) x (n - 1)`` mesh."""

    defaults = {"lvlmax": np.inf, "symm": "n", "verb": 0}
    o = _normalise_opts(opts, defaults)
    o["symm"] = _mf_symm(o["symm"])
    if n <= 0:
        raise ValueError("mesh size must be positive")
    if occ <= 0:
        raise ValueError("leaf occupancy must be positive")
    if o["lvlmax"] < 1:
        raise ValueError("maximum tree depth must be at least 1")
    N = (int(n) - 1) ** 2
    return _make_factor(A, N, o)


def mf3(A, n: int, occ: int, opts: dict[str, Any] | None = None) -> MFFactor:
    """Factor a matrix on a regular ``(n - 1)^3`` mesh."""

    defaults = {"lvlmax": np.inf, "symm": "n", "verb": 0}
    o = _normalise_opts(opts, defaults)
    o["symm"] = _mf_symm(o["symm"])
    if n <= 0:
        raise ValueError("mesh size must be positive")
    if occ <= 0:
        raise ValueError("leaf occupancy must be positive")
    if o["lvlmax"] < 1:
        raise ValueError("maximum tree depth must be at least 1")
    N = (int(n) - 1) ** 3
    return _make_factor(A, N, o)


def mf_mv(F: MFFactor, X, trans: str = "n") -> np.ndarray:
    """Apply the factored matrix."""

    trans = chktrans(trans)
    X_arr, one_dim = _as_rhs(X)
    if F.A_sparse is not None:
        Y = _sparse_apply(F.A_sparse, X_arr, trans)
    elif F.A_dense is not None:
        Y = apply_transpose(F.A_dense, X_arr, trans)
    else:
        raise ValueError("factor does not contain matrix data")
    return Y[:, 0] if one_dim else Y


def mf_sv(F: MFFactor, X, trans: str = "n") -> np.ndarray:
    """Apply the inverse of the factored matrix."""

    trans = chktrans(trans)
    X_arr, one_dim = _as_rhs(X)
    if trans == "t":
        Y = np.conj(mf_sv(F, np.conj(X_arr), "c"))
    elif F.chol is not None and trans == "n":
        Y = la.cho_solve((F.chol, True), X_arr)
    elif F.splu is not None:
        Y = F.splu.solve(X_arr, trans={"n": "N", "t": "T", "c": "H"}[trans])
    elif F.lu is not None:
        lu_trans = 0 if trans == "n" else 2
        Y = la.lu_solve(F.lu, X_arr, trans=lu_trans)
    elif F.A_dense is not None:
        A = F.A_dense if trans == "n" else F.A_dense.conj().T
        Y = la.solve(A, X_arr)
    else:
        raise ValueError("factor does not contain solve data")
    return Y[:, 0] if one_dim else Y


def mf_logdet(F: MFFactor):
    """Return the logarithm of the determinant of the factored matrix."""

    if F.chol is not None:
        return 2 * np.sum(np.log(np.diag(F.chol).astype(np.result_type(F.chol, complex))))
    if F.splu is not None:
        diag_u = F.splu.U.diagonal()
        ld = np.sum(np.log(diag_u.astype(np.result_type(diag_u, complex))))
        ld += np.log(np.asarray(detperm(F.splu.perm_r), dtype=complex))
        ld += np.log(np.asarray(detperm(F.splu.perm_c), dtype=complex))
        return float(ld.real) + 1j * float(np.mod(ld.imag, 2 * np.pi))
    if F.factors and F.factors[0].U is not None and F.factors[0].p is not None:
        block = F.factors[0]
        if sp.issparse(block.U):
            diag_u = block.U.diagonal()
        else:
            diag_u = np.diag(block.U)
        sign = detperm(block.p)
        ld = np.sum(np.log(diag_u.astype(np.result_type(diag_u, complex))))
        return ld + np.log(np.asarray(sign, dtype=complex))
    if F.A_dense is None:
        raise ValueError("factor does not contain matrix data")
    if np.iscomplexobj(F.A_dense):
        val = np.log(np.linalg.det(F.A_dense))
        return val.real + 1j * np.mod(val.imag, 2 * np.pi)
    sign, ld = np.linalg.slogdet(F.A_dense)
    return np.log(np.asarray(sign, dtype=complex)) + ld


def mf_cholmv(F: MFFactor, X, trans: str = "n") -> np.ndarray:
    """Apply the generalized Cholesky factor for ``symm='p'`` factors."""

    _require_positive_definite(F, "mf_cholmv")
    trans = chktrans(trans)
    if trans == "t":
        return np.conj(mf_cholmv(F, np.conj(X), "c"))
    X_arr, one_dim = _as_rhs(X)
    Y = F.chol @ X_arr if trans == "n" else F.chol.conj().T @ X_arr
    return Y[:, 0] if one_dim else Y


def mf_cholsv(F: MFFactor, X, trans: str = "n") -> np.ndarray:
    """Apply the inverse generalized Cholesky factor."""

    _require_positive_definite(F, "mf_cholsv")
    trans = chktrans(trans)
    if trans == "t":
        return np.conj(mf_cholsv(F, np.conj(X), "c"))
    X_arr, one_dim = _as_rhs(X)
    if trans == "n":
        Y = la.solve_triangular(F.chol, X_arr, lower=True)
    else:
        Y = la.solve_triangular(F.chol.conj().T, X_arr, lower=False)
    return Y[:, 0] if one_dim else Y


def mf_diag(F: MFFactor, dinv: bool | int = False, opts: dict[str, Any] | None = None) -> np.ndarray:
    """Extract ``diag(A)`` or ``diag(inv(A))`` from an MF factor."""

    if F.A_sparse is not None:
        if dinv:
            return np.diag(mf_sv(F, np.eye(F.N, dtype=F.A_sparse.dtype)))
        return F.A_sparse.diagonal().copy()
    if F.A_dense is None:
        raise ValueError("factor does not contain matrix data")
    if dinv:
        return np.diag(mf_sv(F, np.eye(F.N, dtype=F.A_dense.dtype)))
    return np.diag(F.A_dense).copy()


def mf_spdiag(F: MFFactor, dinv: bool | int = False) -> np.ndarray:
    """Sparse-style diagonal extraction wrapper."""

    return mf_diag(F, dinv)


def _mf_symm(symm: str | None) -> str:
    symm = chksymm(symm)
    return "n" if symm == "s" else symm


def _make_factor(A, N: int, opts: dict[str, Any], tree: Any = None) -> MFFactor:
    if N < 0:
        raise ValueError("matrix dimension must be nonnegative")
    A_sparse = _as_square_sparse(A, N)
    A_dense = None if A_sparse is not None and opts["symm"] != "p" else _materialize_square(A, N)
    splu = None
    if opts["symm"] == "p":
        chol = np.linalg.cholesky(A_dense)
        lu = None
        block = MFFactorBlock(
            sk=np.array([], dtype=np.int64),
            rd=np.arange(N, dtype=np.int64),
            L=chol,
        )
    elif A_sparse is not None:
        splu = spla.splu(A_sparse.tocsc())
        lu = None
        chol = None
        block = MFFactorBlock(
            sk=np.array([], dtype=np.int64),
            rd=np.arange(N, dtype=np.int64),
            L=splu.L,
            U=splu.U,
            p=np.asarray(splu.perm_r, dtype=np.int64),
        )
    else:
        lu = la.lu_factor(A_dense)
        L, U, p = _lu_block(lu, N)
        chol = None
        block = MFFactorBlock(
            sk=np.array([], dtype=np.int64),
            rd=np.arange(N, dtype=np.int64),
            L=L,
            U=U,
            p=p,
        )
    return MFFactor(
        N=N,
        nlvl=1,
        lvp=np.array([0, 1], dtype=np.int64),
        factors=[block],
        symm=opts["symm"],
        A=A,
        A_dense=A_dense,
        A_sparse=A_sparse,
        lu=lu,
        splu=splu,
        chol=chol,
        tree=tree,
        opts=dict(opts),
    )


def _materialize_square(A, N: int) -> np.ndarray:
    if sp.issparse(A):
        if A.shape != (N, N):
            raise ValueError(f"matrix has shape {A.shape}, expected {(N, N)}")
        return np.asarray(A.toarray())
    return materialize(A, N, N)


def _as_square_sparse(A, N: int) -> sp.spmatrix | None:
    if not sp.issparse(A):
        return None
    if A.shape != (N, N):
        raise ValueError(f"matrix has shape {A.shape}, expected {(N, N)}")
    return A.tocsc()


def _sparse_apply(A: sp.spmatrix, X: np.ndarray, trans: str) -> np.ndarray:
    if trans == "n":
        return A @ X
    if trans == "t":
        return A.T @ X
    return A.conj().T @ X


def _lu_block(lu: tuple[np.ndarray, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lu_mat, piv = lu
    L = np.tril(lu_mat, -1) + np.eye(n, dtype=lu_mat.dtype)
    U = np.triu(lu_mat)
    p = np.arange(n, dtype=np.int64)
    for i, j in enumerate(piv):
        if i != j:
            p[[i, j]] = p[[j, i]]
    return L, U, p


def _as_rhs(X) -> tuple[np.ndarray, bool]:
    X_arr = np.asarray(X)
    one_dim = X_arr.ndim == 1
    if one_dim:
        X_arr = X_arr[:, None]
    if X_arr.ndim != 2:
        raise ValueError("right-hand side must be one- or two-dimensional")
    return X_arr, one_dim


def _require_positive_definite(F: MFFactor, caller: str) -> None:
    if F.symm != "p" or F.chol is None:
        raise ValueError(f"{caller} requires a factorization built with opts={{'symm': 'p'}}")


__all__ = [
    "MFFactor",
    "MFFactorBlock",
    "mf2",
    "mf3",
    "mf_cholmv",
    "mf_cholsv",
    "mf_diag",
    "mf_logdet",
    "mf_mv",
    "mf_spdiag",
    "mf_sv",
    "mfx",
]
