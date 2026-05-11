"""Multifrontal factorization public API.

These routines expose FLAM's ``mfx``/``mf2``/``mf3`` interface.  The current
implementation is correctness-first: it preserves the MATLAB-like factor
object and operation surface while using dense NumPy/SciPy factorizations as
the numerical backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._matrix import apply_transpose, materialize
from .core import StructMixin, _as_points, _normalise_opts, chksymm, chktrans, detperm, hypoct, logdet_ldl


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
    hierarchical: bool = False
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
    return _mfx_hierarchical(A, N, o, tree)


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
    return _mf2_hierarchical(A, int(n), int(occ), o)


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
    return _mf3_hierarchical(A, int(n), int(occ), o)


def mf_mv(F: MFFactor, X, trans: str = "n") -> np.ndarray:
    """Apply the factored matrix."""

    trans = chktrans(trans)
    X_arr, one_dim = _as_rhs(X)
    if F.hierarchical:
        if trans == "t":
            Y = np.conj(_mf_mv_hierarchical(F, np.conj(X_arr), "c"))
        else:
            Y = _mf_mv_hierarchical(F, X_arr, trans)
    elif F.A_sparse is not None:
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
    if F.hierarchical:
        if trans == "t":
            Y = np.conj(_mf_sv_hierarchical(F, np.conj(X_arr), "c"))
        else:
            Y = _mf_sv_hierarchical(F, X_arr, trans)
    elif trans == "t":
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

    if F.hierarchical:
        return _mf_factor_logdet(F)
    if F.chol is not None:
        return 2 * np.sum(np.log(np.diag(F.chol).astype(np.result_type(F.chol, complex))))
    if F.splu is not None:
        diag_u = F.splu.U.diagonal()
        with np.errstate(divide="ignore", invalid="ignore"):
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
        with np.errstate(divide="ignore", invalid="ignore"):
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
    if F.hierarchical:
        Y = _mf_cholmv_hierarchical(F, X_arr, trans)
    else:
        Y = F.chol @ X_arr if trans == "n" else F.chol.conj().T @ X_arr
    return Y[:, 0] if one_dim else Y


def mf_cholsv(F: MFFactor, X, trans: str = "n") -> np.ndarray:
    """Apply the inverse generalized Cholesky factor."""

    _require_positive_definite(F, "mf_cholsv")
    trans = chktrans(trans)
    if trans == "t":
        return np.conj(mf_cholsv(F, np.conj(X), "c"))
    X_arr, one_dim = _as_rhs(X)
    if F.hierarchical:
        Y = _mf_cholsv_hierarchical(F, X_arr, trans)
    elif trans == "n":
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


def _copy_for_factor(F: MFFactor, X: np.ndarray) -> np.ndarray:
    return np.array(X, dtype=np.result_type(X, _factor_dtype(F), 1.0), copy=True)


def _factor_dtype(F: MFFactor):
    dtype = np.dtype(float)
    if F.A_sparse is not None:
        dtype = np.result_type(dtype, F.A_sparse.dtype)
    if F.A_dense is not None:
        dtype = np.result_type(dtype, F.A_dense.dtype)
    for f in F.factors:
        for value in (f.L, f.U, f.E, f.F):
            if value is not None:
                dtype = np.result_type(dtype, value)
    return dtype


def _mf_mv_hierarchical(F: MFFactor, X: np.ndarray, trans: str) -> np.ndarray:
    if F.symm == "n":
        return _mf_mv_nn(F, X) if trans == "n" else _mf_mv_nc(F, X)
    if F.symm == "h":
        return _mf_mv_h(F, X)
    if F.symm == "p":
        return _mf_mv_p(F, X)
    raise ValueError(f"unsupported symmetry mode {F.symm!r}")


def _mf_sv_hierarchical(F: MFFactor, X: np.ndarray, trans: str) -> np.ndarray:
    if F.symm == "n":
        return _mf_sv_nn(F, X) if trans == "n" else _mf_sv_nc(F, X)
    if F.symm == "h":
        return _mf_sv_h(F, X)
    if F.symm == "p":
        return _mf_sv_p(F, X)
    raise ValueError(f"unsupported symmetry mode {F.symm!r}")


def _mf_cholmv_hierarchical(F: MFFactor, X: np.ndarray, trans: str) -> np.ndarray:
    Y = _copy_for_factor(F, X)
    if trans == "n":
        for f in reversed(F.factors):
            Y[f.sk, :] = Y[f.sk, :] + f.E @ Y[f.rd, :]
            Y[f.rd, :] = f.L @ Y[f.rd, :]
    else:
        for f in F.factors:
            Y[f.rd, :] = f.L.conj().T @ Y[f.rd, :]
            Y[f.rd, :] = Y[f.rd, :] + f.E.conj().T @ Y[f.sk, :]
    return Y


def _mf_cholsv_hierarchical(F: MFFactor, X: np.ndarray, trans: str) -> np.ndarray:
    Y = _copy_for_factor(F, X)
    if trans == "n":
        for f in F.factors:
            Y[f.rd, :] = _triangular_solve(f.L, Y[f.rd, :], lower=True)
            Y[f.sk, :] = Y[f.sk, :] - f.E @ Y[f.rd, :]
    else:
        for f in reversed(F.factors):
            Y[f.rd, :] = Y[f.rd, :] - f.E.conj().T @ Y[f.sk, :]
            Y[f.rd, :] = _triangular_solve(f.L.conj().T, Y[f.rd, :], lower=False)
    return Y


def _mf_mv_nn(F: MFFactor, X: np.ndarray) -> np.ndarray:
    Y = _copy_for_factor(F, X)
    for f in F.factors:
        Y[f.rd, :] = f.U @ Y[f.rd, :]
        Y[f.rd, :] = Y[f.rd, :] + f.F @ Y[f.sk, :]
    for f in reversed(F.factors):
        Y[f.sk, :] = Y[f.sk, :] + f.E @ Y[f.rd, :]
        Y[f.rd[f.p], :] = f.L @ Y[f.rd, :]
    return Y


def _mf_mv_nc(F: MFFactor, X: np.ndarray) -> np.ndarray:
    Y = _copy_for_factor(F, X)
    for f in F.factors:
        Y[f.rd, :] = f.L.conj().T @ Y[f.rd[f.p], :]
        Y[f.rd, :] = Y[f.rd, :] + f.E.conj().T @ Y[f.sk, :]
    for f in reversed(F.factors):
        Y[f.sk, :] = Y[f.sk, :] + f.F.conj().T @ Y[f.rd, :]
        Y[f.rd, :] = f.U.conj().T @ Y[f.rd, :]
    return Y


def _mf_mv_h(F: MFFactor, X: np.ndarray) -> np.ndarray:
    Y = _copy_for_factor(F, X)
    for f in F.factors:
        Y[f.rd, :] = f.L.conj().T @ Y[f.rd, :]
        Y[f.rd, :] = Y[f.rd, :] + f.E.conj().T @ Y[f.sk, :]
        Y[f.rd, :] = f.U @ Y[f.rd, :]
    for f in reversed(F.factors):
        Y[f.sk, :] = Y[f.sk, :] + f.E @ Y[f.rd, :]
        Y[f.rd, :] = f.L @ Y[f.rd, :]
    return Y


def _mf_mv_p(F: MFFactor, X: np.ndarray) -> np.ndarray:
    Y = _copy_for_factor(F, X)
    for f in F.factors:
        Y[f.rd, :] = f.L.conj().T @ Y[f.rd, :]
        Y[f.rd, :] = Y[f.rd, :] + f.E.conj().T @ Y[f.sk, :]
    for f in reversed(F.factors):
        Y[f.sk, :] = Y[f.sk, :] + f.E @ Y[f.rd, :]
        Y[f.rd, :] = f.L @ Y[f.rd, :]
    return Y


def _mf_sv_nn(F: MFFactor, X: np.ndarray) -> np.ndarray:
    Y = _copy_for_factor(F, X)
    for f in F.factors:
        Y[f.rd, :] = _triangular_solve(f.L, Y[f.rd[f.p], :], lower=True, unit_diagonal=True)
        Y[f.sk, :] = Y[f.sk, :] - f.E @ Y[f.rd, :]
    for f in reversed(F.factors):
        Y[f.rd, :] = Y[f.rd, :] - f.F @ Y[f.sk, :]
        Y[f.rd, :] = _triangular_solve(f.U, Y[f.rd, :], lower=False)
    return Y


def _mf_sv_nc(F: MFFactor, X: np.ndarray) -> np.ndarray:
    Y = _copy_for_factor(F, X)
    for f in F.factors:
        Y[f.rd, :] = _triangular_solve(f.U.conj().T, Y[f.rd, :], lower=True)
        Y[f.sk, :] = Y[f.sk, :] - f.F.conj().T @ Y[f.rd, :]
    for f in reversed(F.factors):
        Y[f.rd, :] = Y[f.rd, :] - f.E.conj().T @ Y[f.sk, :]
        Y[f.rd[f.p], :] = _triangular_solve(f.L.conj().T, Y[f.rd, :], lower=False, unit_diagonal=True)
    return Y


def _mf_sv_h(F: MFFactor, X: np.ndarray) -> np.ndarray:
    Y = _copy_for_factor(F, X)
    for f in F.factors:
        Y[f.rd, :] = _triangular_solve(f.L, Y[f.rd, :], lower=True, unit_diagonal=True)
        Y[f.sk, :] = Y[f.sk, :] - f.E @ Y[f.rd, :]
        Y[f.rd, :] = la.solve(f.U, Y[f.rd, :], check_finite=False)
    for f in reversed(F.factors):
        Y[f.rd, :] = Y[f.rd, :] - f.E.conj().T @ Y[f.sk, :]
        Y[f.rd, :] = _triangular_solve(f.L.conj().T, Y[f.rd, :], lower=False, unit_diagonal=True)
    return Y


def _mf_sv_p(F: MFFactor, X: np.ndarray) -> np.ndarray:
    return _mf_cholsv_hierarchical(F, _mf_cholsv_hierarchical(F, X, "n"), "c")


def _mf_factor_logdet(F: MFFactor):
    ld = 0.0 + 0.0j
    for f in F.factors:
        if F.symm == "p":
            with np.errstate(divide="ignore", invalid="ignore"):
                ld += 2 * np.sum(np.log(np.diag(f.L).astype(np.result_type(f.L, complex))))
        elif F.symm == "h":
            ld += logdet_ldl(f.U)
        else:
            sign = detperm(f.p)
            with np.errstate(divide="ignore", invalid="ignore"):
                ld += np.sum(np.log(np.diag(f.U).astype(np.result_type(f.U, complex))))
            ld += np.log(np.asarray(sign, dtype=complex))
    return float(ld.real) + 1j * float(np.mod(ld.imag, 2 * np.pi))


def _mf_symm(symm: str | None) -> str:
    symm = chksymm(symm)
    return "n" if symm == "s" else symm


def _mf2_hierarchical(A, n: int, occ: int, opts: dict[str, Any]) -> MFFactor:
    nd = n - 1
    N = nd**2
    A0 = _as_square_sparse_matrix(A, N)
    A_work = A0.copy().tocsc()
    nlvl = min(float(opts["lvlmax"]), np.ceil(max(0.0, np.log2(n / occ))) + 1)
    nlvl = int(nlvl)

    factors: list[MFFactorBlock] = []
    lvp = [0]
    grid = np.arange(N, dtype=np.int64).reshape((nd, nd), order="F")
    rem = np.ones(N, dtype=bool)

    w = n
    for _ in range(nlvl):
        w = int(np.ceil(w / 2))

    for _lvl in range(nlvl, 0, -1):
        w *= 2
        nb = int(np.ceil(n / w))
        upd_i: list[np.ndarray] = []
        upd_j: list[np.ndarray] = []
        upd_v: list[np.ndarray] = []

        for i in range(1, nb + 1):
            ia = (i - 1) * w
            ib = i * w
            is_ = np.arange(max(1, ia) - 1, min(nd, ib), dtype=np.int64)
            for j in range(1, nb + 1):
                ja = (j - 1) * w
                jb = j * w
                js = np.arange(max(1, ja) - 1, min(nd, jb), dtype=np.int64)
                if is_.size == 0 or js.size == 0:
                    continue

                cell = grid[np.ix_(is_, js)].ravel(order="F")
                slf = cell[rem[cell]]
                if slf.size == 0:
                    continue

                jj = slf // nd + 1
                ii = slf - nd * (jj - 1) + 1
                interior = (ii != ia) & (ii != ib) & (jj != ja) & (jj != jb)
                sk_pos = np.flatnonzero(~interior)
                rd_pos = np.flatnonzero(interior)
                if rd_pos.size == 0:
                    continue
                rem[slf[rd_pos]] = False

                K = _spget_dense(A_work, slf, slf)
                Krr = K[np.ix_(rd_pos, rd_pos)]
                Ksr = K[np.ix_(sk_pos, rd_pos)]
                Krs = K[np.ix_(rd_pos, sk_pos)]

                Ufac = None
                p = None
                G = None
                if opts["symm"] == "p":
                    L = np.linalg.cholesky(Krr)
                    E = _triangular_solve(L, Ksr.T, lower=True).T
                    X = -E @ E.conj().T
                elif opts["symm"] == "h":
                    Lraw, D, perm = la.ldl(Krr, lower=True, hermitian=True, check_finite=False)
                    perm = np.asarray(perm, dtype=np.int64)
                    rd_pos = rd_pos[perm]
                    Ksr = K[np.ix_(sk_pos, rd_pos)]
                    L = Lraw[perm, :]
                    Ufac = D
                    E = _triangular_solve(L, Ksr.conj().T, lower=True, unit_diagonal=True).conj().T
                    E = la.solve(D, E.T, check_finite=False).T
                    X = -E @ (D @ E.conj().T)
                else:
                    lu = _lu_factor_allow_singular(Krr)
                    L, Ufac, p = _lu_block(lu, rd_pos.size)
                    E = _triangular_solve(Ufac.T, Ksr.T, lower=True).T
                    G = _triangular_solve(L, Krs[p, :], lower=True, unit_diagonal=True)
                    X = -E @ G

                sk = slf[sk_pos]
                rd = slf[rd_pos]
                if sk.size:
                    rows, cols = np.meshgrid(sk, sk, indexing="ij")
                    upd_i.append(rows.ravel())
                    upd_j.append(cols.ravel())
                    upd_v.append(X.ravel())

                factors.append(MFFactorBlock(sk=sk, rd=rd, L=L, U=Ufac, p=p, E=E, F=G))

        lvp.append(len(factors))
        coo = A_work.tocoo()
        keep = rem[coo.row] & rem[coo.col]
        if keep.any():
            upd_i.append(coo.row[keep])
            upd_j.append(coo.col[keep])
            upd_v.append(coo.data[keep])
        if upd_i:
            rows = np.concatenate(upd_i)
            cols = np.concatenate(upd_j)
            vals = np.concatenate(upd_v)
            A_work = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))
        else:
            A_work = sp.csc_matrix((N, N), dtype=A0.dtype)

    return MFFactor(
        N=N,
        nlvl=nlvl,
        lvp=np.asarray(lvp, dtype=np.int64),
        factors=factors,
        symm=opts["symm"],
        A=A,
        A_sparse=A0,
        hierarchical=True,
        opts=dict(opts),
    )


def _mf3_hierarchical(A, n: int, occ: int, opts: dict[str, Any]) -> MFFactor:
    nd = n - 1
    N = nd**3
    A0 = _as_square_sparse_matrix(A, N)
    A_work = A0.copy().tocsc()
    nlvl = min(float(opts["lvlmax"]), np.ceil(max(0.0, np.log2(n / occ))) + 1)
    nlvl = int(nlvl)

    factors: list[MFFactorBlock] = []
    lvp = [0]
    grid = np.arange(N, dtype=np.int64).reshape((nd, nd, nd), order="F")
    rem = np.ones(N, dtype=bool)

    w = n
    for _ in range(nlvl):
        w = int(np.ceil(w / 2))

    for _lvl in range(nlvl, 0, -1):
        w *= 2
        nb = int(np.ceil(n / w))
        upd_i: list[np.ndarray] = []
        upd_j: list[np.ndarray] = []
        upd_v: list[np.ndarray] = []

        for i in range(1, nb + 1):
            ia = (i - 1) * w
            ib = i * w
            is_ = np.arange(max(1, ia) - 1, min(nd, ib), dtype=np.int64)
            for j in range(1, nb + 1):
                ja = (j - 1) * w
                jb = j * w
                js = np.arange(max(1, ja) - 1, min(nd, jb), dtype=np.int64)
                for k in range(1, nb + 1):
                    ka = (k - 1) * w
                    kb = k * w
                    ks = np.arange(max(1, ka) - 1, min(nd, kb), dtype=np.int64)
                    if is_.size == 0 or js.size == 0 or ks.size == 0:
                        continue

                    cell = grid[np.ix_(is_, js, ks)].ravel(order="F")
                    slf = cell[rem[cell]]
                    if slf.size == 0:
                        continue

                    kk = slf // (nd**2) + 1
                    idx = slf - (nd**2) * (kk - 1)
                    jj = idx // nd + 1
                    ii = idx - nd * (jj - 1) + 1
                    interior = (ii != ia) & (ii != ib) & (jj != ja) & (jj != jb) & (kk != ka) & (kk != kb)
                    sk_pos = np.flatnonzero(~interior)
                    rd_pos = np.flatnonzero(interior)
                    if rd_pos.size == 0:
                        continue
                    rem[slf[rd_pos]] = False

                    K = _spget_dense(A_work, slf, slf)
                    Krr = K[np.ix_(rd_pos, rd_pos)]
                    Ksr = K[np.ix_(sk_pos, rd_pos)]
                    Krs = K[np.ix_(rd_pos, sk_pos)]

                    Ufac = None
                    p = None
                    G = None
                    if opts["symm"] == "p":
                        L = np.linalg.cholesky(Krr)
                        E = _triangular_solve(L, Ksr.T, lower=True).T
                        X = -E @ E.conj().T
                    elif opts["symm"] == "h":
                        Lraw, D, perm = la.ldl(Krr, lower=True, hermitian=True, check_finite=False)
                        perm = np.asarray(perm, dtype=np.int64)
                        rd_pos = rd_pos[perm]
                        Ksr = K[np.ix_(sk_pos, rd_pos)]
                        L = Lraw[perm, :]
                        Ufac = D
                        E = _triangular_solve(L, Ksr.conj().T, lower=True, unit_diagonal=True).conj().T
                        E = la.solve(D, E.T, check_finite=False).T
                        X = -E @ (D @ E.conj().T)
                    else:
                        lu = _lu_factor_allow_singular(Krr)
                        L, Ufac, p = _lu_block(lu, rd_pos.size)
                        E = _triangular_solve(Ufac.T, Ksr.T, lower=True).T
                        G = _triangular_solve(L, Krs[p, :], lower=True, unit_diagonal=True)
                        X = -E @ G

                    sk = slf[sk_pos]
                    rd = slf[rd_pos]
                    if sk.size:
                        rows, cols = np.meshgrid(sk, sk, indexing="ij")
                        upd_i.append(rows.ravel())
                        upd_j.append(cols.ravel())
                        upd_v.append(X.ravel())

                    factors.append(MFFactorBlock(sk=sk, rd=rd, L=L, U=Ufac, p=p, E=E, F=G))

        lvp.append(len(factors))
        coo = A_work.tocoo()
        keep = rem[coo.row] & rem[coo.col]
        if keep.any():
            upd_i.append(coo.row[keep])
            upd_j.append(coo.col[keep])
            upd_v.append(coo.data[keep])
        if upd_i:
            rows = np.concatenate(upd_i)
            cols = np.concatenate(upd_j)
            vals = np.concatenate(upd_v)
            A_work = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))
        else:
            A_work = sp.csc_matrix((N, N), dtype=A0.dtype)

    return MFFactor(
        N=N,
        nlvl=nlvl,
        lvp=np.asarray(lvp, dtype=np.int64),
        factors=factors,
        symm=opts["symm"],
        A=A,
        A_sparse=A0,
        hierarchical=True,
        opts=dict(opts),
    )


def _mfx_hierarchical(A, N: int, opts: dict[str, Any], tree: Any) -> MFFactor:
    A0 = _as_square_sparse_matrix(A, N)
    A_work = A0.copy().tocsc()
    factors: list[MFFactorBlock] = []
    lvp = [0]
    rem = np.ones(N, dtype=bool)

    for lvl in range(tree.nlvl - 1, -1, -1):
        upd_i: list[np.ndarray] = []
        upd_j: list[np.ndarray] = []
        upd_v: list[np.ndarray] = []
        A_trans = A_work.T.tocsc() if opts["symm"] == "n" else None

        for node_idx in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
            node = tree.nodes[node_idx]
            child_xi = _concat_unique([tree.nodes[ch].xi for ch in node.chld])
            if child_xi.size:
                node.xi = np.unique(np.concatenate((node.xi, child_xi)))

        for node_idx in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
            node = tree.nodes[node_idx]
            slf = np.asarray(node.xi, dtype=np.int64).copy()
            if slf.size == 0:
                continue
            sslf = np.sort(slf)

            I_ext, J_ext = _external_column_interactions(A_work, slf, sslf)
            if opts["symm"] == "n":
                Ic, Jc = _external_column_interactions(A_trans, slf, sslf)
                if J_ext.size or Jc.size:
                    I_ext = np.concatenate((I_ext, Ic))
                    J_ext = np.concatenate((J_ext, Jc))
                    order = np.argsort(J_ext, kind="stable")
                    I_ext = I_ext[order]
                    J_ext = J_ext[order]
            sk_pos = np.unique(J_ext)

            nbr = [idx for idx in node.nbor if idx < node_idx]
            if nbr:
                nbr_xi = _concat_unique([tree.nodes[idx].xi for idx in nbr])
                if nbr_xi.size and sk_pos.size:
                    keep = np.ones(sk_pos.size, dtype=bool)
                    nbrsk_parts = []
                    for pos_idx, col in enumerate(sk_pos):
                        segment = J_ext == col
                        if segment.any() and np.all(np.isin(I_ext[segment], nbr_xi)):
                            keep[pos_idx] = False
                            nbrsk_parts.append(I_ext[segment])
                    nbrsk = _concat_unique(nbrsk_parts)
                    sk_pos = np.concatenate((sk_pos[keep], slf.size + np.arange(nbrsk.size, dtype=np.int64)))
                    slf = np.concatenate((slf, nbrsk))

            if sk_pos.size:
                node.xi = slf[sk_pos]
            else:
                node.xi = np.array([], dtype=np.int64)
            rd_pos = np.setdiff1d(np.arange(slf.size, dtype=np.int64), np.sort(sk_pos), assume_unique=False)
            if rd_pos.size == 0:
                continue
            rem[slf[rd_pos]] = False

            K = _spget_dense(A_work, slf, slf)
            Krr = K[np.ix_(rd_pos, rd_pos)]
            Ksr = K[np.ix_(sk_pos, rd_pos)]
            Krs = K[np.ix_(rd_pos, sk_pos)]

            Ufac = None
            p = None
            G = None
            if opts["symm"] == "p":
                L = np.linalg.cholesky(Krr)
                E = _triangular_solve(L, Ksr.T, lower=True).T
                X = -E @ E.conj().T
            elif opts["symm"] == "h":
                Lraw, D, perm = la.ldl(Krr, lower=True, hermitian=True, check_finite=False)
                perm = np.asarray(perm, dtype=np.int64)
                rd_pos = rd_pos[perm]
                Ksr = K[np.ix_(sk_pos, rd_pos)]
                L = Lraw[perm, :]
                Ufac = D
                E = _triangular_solve(L, Ksr.conj().T, lower=True, unit_diagonal=True).conj().T
                E = la.solve(D, E.T, check_finite=False).T
                X = -E @ (D @ E.conj().T)
            else:
                lu = _lu_factor_allow_singular(Krr)
                L, Ufac, p = _lu_block(lu, rd_pos.size)
                E = _triangular_solve(Ufac.T, Ksr.T, lower=True).T
                G = _triangular_solve(L, Krs[p, :], lower=True, unit_diagonal=True)
                X = -E @ G

            sk = slf[sk_pos]
            rd = slf[rd_pos]
            if sk.size:
                rows, cols = np.meshgrid(sk, sk, indexing="ij")
                upd_i.append(rows.ravel())
                upd_j.append(cols.ravel())
                upd_v.append(X.ravel())
            factors.append(MFFactorBlock(sk=sk, rd=rd, L=L, U=Ufac, p=p, E=E, F=G))

        lvp.append(len(factors))
        coo = A_work.tocoo()
        keep = rem[coo.row] & rem[coo.col]
        if keep.any():
            upd_i.append(coo.row[keep])
            upd_j.append(coo.col[keep])
            upd_v.append(coo.data[keep])
        if upd_i:
            rows = np.concatenate(upd_i)
            cols = np.concatenate(upd_j)
            vals = np.concatenate(upd_v)
            A_work = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))
        else:
            A_work = sp.csc_matrix((N, N), dtype=A0.dtype)

    return MFFactor(
        N=N,
        nlvl=tree.nlvl,
        lvp=np.asarray(lvp, dtype=np.int64),
        factors=factors,
        symm=opts["symm"],
        A=A,
        A_sparse=A0,
        hierarchical=True,
        tree=tree,
        opts=dict(opts),
    )


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
        try:
            splu = spla.splu(A_sparse.tocsc())
            lu = None
            L = splu.L
            U = splu.U
            p = np.asarray(splu.perm_r, dtype=np.int64)
        except RuntimeError as exc:
            if "singular" not in str(exc).lower():
                raise
            A_dense = np.asarray(A_sparse.toarray())
            lu = _lu_factor_allow_singular(A_dense)
            L, U, p = _lu_block(lu, N)
            splu = None
        chol = None
        block = MFFactorBlock(
            sk=np.array([], dtype=np.int64),
            rd=np.arange(N, dtype=np.int64),
            L=L,
            U=U,
            p=p,
        )
    else:
        lu = _lu_factor_allow_singular(A_dense)
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


def _as_square_sparse_matrix(A, N: int) -> sp.csc_matrix:
    if sp.issparse(A):
        if A.shape != (N, N):
            raise ValueError(f"matrix has shape {A.shape}, expected {(N, N)}")
        return A.tocsc()
    return sp.csc_matrix(_materialize_square(A, N))


def _spget_dense(A: sp.spmatrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if rows.size == 0 or cols.size == 0:
        return np.zeros((rows.size, cols.size), dtype=A.dtype)
    return np.asarray(A[np.ix_(rows, cols)].toarray())


def _external_column_interactions(
    A: sp.spmatrix,
    slf: np.ndarray,
    sorted_slf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sub = A[:, slf].tocoo()
    if sub.nnz == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    external = ~np.isin(sub.row, sorted_slf, assume_unique=False)
    return sub.row[external].astype(np.int64), sub.col[external].astype(np.int64)


def _concat_unique(parts: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(part, dtype=np.int64).reshape(-1) for part in parts if np.asarray(part).size]
    if not arrays:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(arrays))


def _triangular_solve(
    A: np.ndarray,
    B: np.ndarray,
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> np.ndarray:
    A = np.asarray(A)
    X = np.array(B, dtype=np.result_type(A, B, 1.0), copy=True)
    n = A.shape[0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if lower:
            for i in range(n):
                if i:
                    X[i, :] = X[i, :] - A[i, :i] @ X[:i, :]
                if not unit_diagonal:
                    X[i, :] = X[i, :] / A[i, i]
        else:
            for i in range(n - 1, -1, -1):
                if i + 1 < n:
                    X[i, :] = X[i, :] - A[i, i + 1 :] @ X[i + 1 :, :]
                if not unit_diagonal:
                    X[i, :] = X[i, :] / A[i, i]
    return X


def _lu_factor_allow_singular(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", la.LinAlgWarning)
        return la.lu_factor(A)


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
    if F.symm != "p" or (F.chol is None and not F.hierarchical):
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
