"""Recursive skeletonization factorization API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as la

from ._matrix import apply_transpose, materialize, submatrix
from .core import StructMixin, _as_points, _normalise_opts, chksymm, chktrans, detperm, hypoct, id, logdet_ldl


@dataclass
class RSkelFFactorBlock(StructMixin):
    sk: np.ndarray
    rd: np.ndarray
    T: np.ndarray
    L: np.ndarray
    U: np.ndarray | None = None
    p: np.ndarray | None = None
    E: np.ndarray | None = None
    F: np.ndarray | None = None


@dataclass
class RSkelFFactor(StructMixin):
    N: int
    nlvl: int
    lvp: np.ndarray
    factors: list[RSkelFFactorBlock] = field(default_factory=list)
    symm: str = "n"
    A_dense: np.ndarray | None = None
    lu: tuple[np.ndarray, np.ndarray] | None = None
    chol: np.ndarray | None = None
    tree: Any = None
    opts: dict[str, Any] = field(default_factory=dict)
    Si: np.ndarray | None = None
    S: np.ndarray | None = None


def _stop_fun(stop):
    if stop is None:
        return lambda lvl, l: False
    if callable(stop):
        return stop
    if np.isinf(stop):
        return lambda lvl, l: False
    return lambda lvl, l: lvl >= stop


def _concat_indices(parts: list[np.ndarray]) -> np.ndarray:
    parts = [np.asarray(p, dtype=np.int64).reshape(-1) for p in parts if np.asarray(p).size]
    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


def _lu_vector(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MATLAB-style ``L, U, p`` with ``A[p, :] = L @ U``."""

    lu, piv = la.lu_factor(A, check_finite=False)
    n = A.shape[0]
    L = np.tril(lu, -1) + np.eye(n, dtype=lu.dtype)
    U = np.triu(lu)
    p = np.arange(n, dtype=np.int64)
    for i, j in enumerate(piv):
        if i != j:
            p[[i, j]] = p[[j, i]]
    return L, U, p


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

    rem = np.ones(N, dtype=bool)
    modified: list[np.ndarray | None] = [None for _ in tree.nodes]
    modified_idx: list[np.ndarray | None] = [None for _ in tree.nodes]
    factors: list[RSkelFFactorBlock] = []
    lvp = [0]
    stop = _stop_fun(o["stop"])

    for lvl in range(tree.nlvl - 1, -1, -1):
        node_size = tree.l[:, lvl]
        if stop(tree.nlvl - 1 - lvl, node_size):
            break

        for node_idx in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
            node = tree.nodes[node_idx]
            child_xi = _concat_indices([tree.nodes[ch].xi for ch in node.chld])
            if child_xi.size:
                node.xi = np.concatenate((node.xi, child_xi)) if node.xi.size else child_xi

        for node_idx in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
            node = tree.nodes[node_idx]
            slf = np.asarray(node.xi, dtype=np.int64)
            if slf.size == 0:
                modified[node_idx] = np.zeros((0, 0), dtype=A_dense.dtype)
                modified_idx[node_idx] = slf
                continue

            nbr = _concat_indices([tree.nodes[j].xi for j in node.nbor])
            nslf = slf.size
            M = np.zeros((nslf, nslf), dtype=A_dense.dtype)
            if lvl < tree.nlvl - 1 and node.chld:
                pos = {int(v): k for k, v in enumerate(slf)}
                for ch in node.chld:
                    child_M = modified[ch]
                    child_idx = modified_idx[ch]
                    if child_M is None or child_idx is None or child_idx.size == 0:
                        continue
                    loc = np.array([pos[int(v)] for v in child_idx], dtype=np.int64)
                    M[np.ix_(loc, loc)] = child_M
                    modified[ch] = None
                    modified_idx[ch] = None

            Kpxy = np.zeros((0, nslf), dtype=A_dense.dtype)
            if lvl + 1 > 2:
                if pxyfun is None:
                    nbr = np.setdiff1d(np.flatnonzero(rem), slf, assume_unique=False)
                else:
                    Kpxy, nbr = pxyfun(x, slf, nbr, node_size, node.ctr)
                    Kpxy = np.asarray(Kpxy)
                    nbr = np.asarray(nbr, dtype=np.int64)

            K = submatrix(A, nbr, slf) if nbr.size else np.zeros((0, nslf), dtype=A_dense.dtype)
            if o["symm"] == "n":
                K2 = submatrix(A, slf, nbr).conj().T if nbr.size else np.zeros((0, nslf), dtype=A_dense.dtype)
                K = np.vstack((K, K2))
            if Kpxy.size:
                K = np.vstack((K, Kpxy))

            sk, rd, T = id(K, rank_or_tol, o["Tmax"], o["rrqr_iter"])
            if rd.size == 0:
                modified[node_idx] = M
                modified_idx[node_idx] = slf
                continue

            Kself = submatrix(A, slf, slf) + M
            if o["symm"] == "s":
                Kself[rd, :] = Kself[rd, :] - T.T @ Kself[sk, :]
            else:
                Kself[rd, :] = Kself[rd, :] - T.conj().T @ Kself[sk, :]
            Kself[:, rd] = Kself[:, rd] - Kself[:, sk] @ T

            Krr = Kself[np.ix_(rd, rd)]
            Ksr = Kself[np.ix_(sk, rd)]
            Krs = Kself[np.ix_(rd, sk)]

            Ufac = None
            p = None
            E = None
            G = None
            if o["symm"] == "p":
                L = np.linalg.cholesky(Krr)
                E = la.solve_triangular(L, Ksr.conj().T, lower=True, check_finite=False).conj().T
                schur = E @ E.conj().T
            elif o["symm"] == "h":
                Lraw, D, perm = la.ldl(Krr, lower=True, hermitian=True, check_finite=False)
                perm = np.asarray(perm, dtype=np.int64)
                rd = rd[perm]
                T = T[:, perm]
                Ksr = Kself[np.ix_(sk, rd)]
                Krs = Kself[np.ix_(rd, sk)]
                L = Lraw[perm, :]
                Ufac = D
                E = la.solve_triangular(L, Ksr.conj().T, lower=True, unit_diagonal=True, check_finite=False).conj().T
                E = la.solve(D, E.T, check_finite=False).T
                schur = E @ (D @ E.conj().T)
            else:
                L, Umat, p = _lu_vector(Krr)
                Ufac = Umat
                E = la.solve_triangular(Umat.T, Ksr.T, lower=True, check_finite=False).T
                G = la.solve_triangular(L, Krs[p, :], lower=True, unit_diagonal=True, check_finite=False)
                schur = E @ G

            Mnext = M[np.ix_(sk, sk)] - schur
            modified[node_idx] = Mnext
            modified_idx[node_idx] = slf[sk]

            factors.append(
                RSkelFFactorBlock(
                    sk=slf[sk],
                    rd=slf[rd],
                    T=T,
                    L=L,
                    U=Ufac,
                    p=p,
                    E=E,
                    F=G,
                )
            )
            node.xi = slf[sk]
            rem[slf[rd]] = False

        lvp.append(len(factors))

    remaining = np.flatnonzero(rem)
    S = np.zeros((remaining.size, remaining.size), dtype=A_dense.dtype)
    if remaining.size:
        pos = {int(v): k for k, v in enumerate(remaining)}
        for M, idx in zip(modified, modified_idx):
            if M is None or idx is None or idx.size == 0:
                continue
            loc = np.array([pos[int(v)] for v in idx if int(v) in pos], dtype=np.int64)
            if loc.size == idx.size:
                S[np.ix_(loc, loc)] += M

    lu = None
    chol = None
    if o["symm"] == "p":
        chol = np.linalg.cholesky(A_dense)
    elif o["symm"] != "n" or remaining.size:
        lu = la.lu_factor(A_dense)
    return RSkelFFactor(
        N=N,
        nlvl=len(lvp) - 1,
        lvp=np.asarray(lvp, dtype=np.int64),
        factors=factors,
        symm=o["symm"],
        A_dense=A_dense,
        lu=lu,
        chol=chol,
        tree=tree,
        opts=o,
        Si=remaining,
        S=S,
    )


def rskelf_mv(F: RSkelFFactor, X, trans: str = "n") -> np.ndarray:
    trans = chktrans(trans)
    if trans == "t":
        return np.conj(rskelf_mv(F, np.conj(X), "c"))
    X = _prepare_rhs(F, X)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    if _has_complete_compact_factor(F):
        if F.symm == "n":
            Y = _rskelf_mv_nn(F, X, 3) if trans == "n" else _rskelf_mv_nc(F, X, 3)
        elif F.symm == "s":
            Y = _rskelf_mv_sn(F, X, 3) if trans == "n" else _rskelf_mv_sc(F, X, 3)
        elif F.symm == "h":
            Y = _rskelf_mv_h(F, X, 3)
        else:
            Y = _rskelf_mv_p(F, X, 3)
    else:
        if F.A_dense is None:
            raise ValueError("factor does not contain matrix data")
        Y = apply_transpose(F.A_dense, X, trans)
    return Y[:, 0] if one_dim else Y


def rskelf_sv(F: RSkelFFactor, X, trans: str = "n") -> np.ndarray:
    trans = chktrans(trans)
    if trans == "t":
        return np.conj(rskelf_sv(F, np.conj(X), "c"))
    X = _prepare_rhs(F, X)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    if _has_complete_compact_factor(F):
        if F.symm == "n":
            Y = _rskelf_sv_nn(F, X, 3) if trans == "n" else _rskelf_sv_nc(F, X, 3)
        elif F.symm == "s":
            Y = _rskelf_sv_sn(F, X, 3) if trans == "n" else _rskelf_sv_sc(F, X, 3)
        elif F.symm == "h":
            Y = _rskelf_sv_h(F, X, 3)
        else:
            Y = _rskelf_sv_p(F, X, 3)
    elif F.chol is not None:
        if trans == "n":
            Y = la.cho_solve((F.chol, True), X)
        elif trans == "t":
            Y = la.solve(F.A_dense.T, X, assume_a="sym")
        else:
            Y = la.solve(F.A_dense.conj().T, X, assume_a="her")
    elif F.lu is not None:
        if trans == "n":
            Y = la.lu_solve(F.lu, X, check_finite=False)
        elif trans == "t":
            Y = la.lu_solve(F.lu, X, trans=1, check_finite=False)
        else:
            Y = la.lu_solve(F.lu, X, trans=2, check_finite=False)
    else:
        raise ValueError("factor does not contain solve data")
    return Y[:, 0] if one_dim else Y


def rskelf_logdet(F: RSkelFFactor):
    ld = 0.0 + 0.0j
    for f in F.factors:
        if F.symm == "p":
            ld += 2 * np.sum(np.log(np.diag(f.L).astype(np.result_type(f.L, complex))))
        elif F.symm == "h":
            ld += logdet_ldl(f.U)
        else:
            if f.U is None or f.p is None:
                continue
            sign = detperm(f.p)
            ld += np.sum(np.log(np.diag(f.U).astype(np.result_type(f.U, complex))))
            ld += np.log(np.asarray(sign, dtype=complex))
    return float(ld.real) + 1j * float(np.mod(ld.imag, 2 * np.pi))


def rskelf_cholmv(F: RSkelFFactor, X, trans: str = "n") -> np.ndarray:
    """Apply the generalized Cholesky factor for a positive-definite factorization."""

    _require_positive_definite(F, "rskelf_cholmv")
    trans = chktrans(trans)
    if trans == "t":
        return np.conj(rskelf_cholmv(F, np.conj(X), "c"))
    X = _prepare_rhs(F, X)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    if _has_complete_compact_factor(F):
        Y = _rskelf_cholmv_p(F, X, trans)
    elif trans == "n":
        Y = F.chol @ X
    else:
        Y = F.chol.conj().T @ X
    return Y[:, 0] if one_dim else Y


def rskelf_cholsv(F: RSkelFFactor, X, trans: str = "n") -> np.ndarray:
    """Apply the inverse generalized Cholesky factor for a positive-definite factorization."""

    _require_positive_definite(F, "rskelf_cholsv")
    trans = chktrans(trans)
    if trans == "t":
        return np.conj(rskelf_cholsv(F, np.conj(X), "c"))
    X = _prepare_rhs(F, X)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    if _has_complete_compact_factor(F):
        Y = _rskelf_cholsv_p(F, X, trans)
    elif trans == "n":
        Y = la.solve_triangular(F.chol, X, lower=True)
    else:
        Y = la.solve_triangular(F.chol.conj().T, X, lower=False)
    return Y[:, 0] if one_dim else Y


def rskelf_diag(F: RSkelFFactor, dinv: bool | int = False, opts: dict[str, Any] | None = None) -> np.ndarray:
    """Extract ``diag(F)`` or ``diag(inv(F))``.

    This provides the FLAM public interface with dense exact semantics. Compact
    selected-inversion can be layered underneath this API later.
    """

    if F.A_dense is None:
        raise ValueError("factor does not contain matrix data")
    if dinv:
        eye = np.eye(F.N, dtype=F.A_dense.dtype)
        return np.diag(rskelf_sv(F, eye))
    return np.diag(F.A_dense).copy()


def rskelf_spdiag(F: RSkelFFactor, dinv: bool | int = False) -> np.ndarray:
    """Extract a diagonal using the sparse-apply style FLAM API."""

    return rskelf_diag(F, dinv)


def _require_positive_definite(F: RSkelFFactor, caller: str) -> None:
    if F.symm != "p" or F.chol is None:
        raise ValueError(f"{caller} requires a factorization built with opts={{'symm': 'p'}}")


def _has_complete_compact_factor(F: RSkelFFactor) -> bool:
    return F.Si is not None and F.Si.size == 0


def _factor_dtype(F: RSkelFFactor, X) -> np.dtype:
    dtype = np.asarray(X).dtype
    for f in F.factors:
        dtype = np.result_type(dtype, f.T, f.L)
        if f.U is not None:
            dtype = np.result_type(dtype, f.U)
        if f.E is not None:
            dtype = np.result_type(dtype, f.E)
        if f.F is not None:
            dtype = np.result_type(dtype, f.F)
    return np.dtype(dtype)


def _prepare_rhs(F: RSkelFFactor, X) -> np.ndarray:
    return np.array(X, dtype=_factor_dtype(F, X), copy=True)


def _rskelf_mv_nn(F: RSkelFFactor, X: np.ndarray, mode: int = 3) -> np.ndarray:
    if mode & 1:
        for f in F.factors:
            sk, rd = f.sk, f.rd
            X[sk, :] = X[sk, :] + f.T @ X[rd, :]
            X[rd, :] = f.U @ X[rd, :]
            X[rd, :] = X[rd, :] + f.F @ X[sk, :]
    if mode & 2:
        for f in reversed(F.factors):
            sk, rd = f.sk, f.rd
            X[sk, :] = X[sk, :] + f.E @ X[rd, :]
            X[rd[f.p], :] = f.L @ X[rd, :]
            X[rd, :] = X[rd, :] + f.T.conj().T @ X[sk, :]
    return X


def _rskelf_mv_nc(F: RSkelFFactor, X: np.ndarray, mode: int = 3) -> np.ndarray:
    if mode & 1:
        for f in F.factors:
            sk, rd = f.sk, f.rd
            X[sk, :] = X[sk, :] + f.T @ X[rd, :]
            X[rd, :] = f.L.conj().T @ X[rd[f.p], :]
            X[rd, :] = X[rd, :] + f.E.conj().T @ X[sk, :]
    if mode & 2:
        for f in reversed(F.factors):
            sk, rd = f.sk, f.rd
            X[sk, :] = X[sk, :] + f.F.conj().T @ X[rd, :]
            X[rd, :] = f.U.conj().T @ X[rd, :]
            X[rd, :] = X[rd, :] + f.T.conj().T @ X[sk, :]
    return X


def _rskelf_sv_nn(F: RSkelFFactor, X: np.ndarray, mode: int = 3) -> np.ndarray:
    if mode & 1:
        for f in F.factors:
            sk, rd = f.sk, f.rd
            X[rd, :] = X[rd, :] - f.T.conj().T @ X[sk, :]
            X[rd, :] = la.solve_triangular(
                f.L,
                X[rd[f.p], :],
                lower=True,
                unit_diagonal=True,
                check_finite=False,
                overwrite_b=True,
            )
            X[sk, :] = X[sk, :] - f.E @ X[rd, :]
    if mode & 2:
        for f in reversed(F.factors):
            sk, rd = f.sk, f.rd
            X[rd, :] = X[rd, :] - f.F @ X[sk, :]
            X[rd, :] = la.solve_triangular(f.U, X[rd, :], lower=False, check_finite=False, overwrite_b=True)
            X[sk, :] = X[sk, :] - f.T @ X[rd, :]
    return X


def _rskelf_sv_nc(F: RSkelFFactor, X: np.ndarray, mode: int = 3) -> np.ndarray:
    if mode & 1:
        for f in F.factors:
            sk, rd = f.sk, f.rd
            X[rd, :] = X[rd, :] - f.T.conj().T @ X[sk, :]
            X[rd, :] = la.solve_triangular(
                f.U.conj().T,
                X[rd, :],
                lower=True,
                check_finite=False,
                overwrite_b=True,
            )
            X[sk, :] = X[sk, :] - f.F.conj().T @ X[rd, :]
    if mode & 2:
        for f in reversed(F.factors):
            sk, rd = f.sk, f.rd
            X[rd, :] = X[rd, :] - f.E.conj().T @ X[sk, :]
            X[rd[f.p], :] = la.solve_triangular(
                f.L.conj().T,
                X[rd, :],
                lower=False,
                unit_diagonal=True,
                check_finite=False,
                overwrite_b=True,
            )
            X[sk, :] = X[sk, :] - f.T @ X[rd, :]
    return X

def _rskelf_mv_sn(F: RSkelFFactor, X: np.ndarray, mode: int = 3) -> np.ndarray:
    if mode & 1:
        for f in F.factors:
            sk, rd = f.sk, f.rd
            X[sk, :] = X[sk, :] + f.T @ X[rd, :]
            X[rd, :] = f.U @ X[rd, :]
            X[rd, :] = X[rd, :] + f.F @ X[sk, :]
    if mode & 2:
        for f in reversed(F.factors):
            sk, rd = f.sk, f.rd
            X[sk, :] = X[sk, :] + f.E @ X[rd, :]
            X[rd[f.p], :] = f.L @ X[rd, :]
            X[rd, :] = X[rd, :] + f.T.T @ X[sk, :]
    return X


def _rskelf_mv_sc(F: RSkelFFactor, X: np.ndarray, mode: int = 3) -> np.ndarray:
    if mode & 1:
        for f in F.factors:
            sk, rd = f.sk, f.rd
            X[sk, :] = X[sk, :] + np.conj(f.T) @ X[rd, :]
            X[rd, :] = f.L.conj().T @ X[rd[f.p], :]
            X[rd, :] = X[rd, :] + f.E.conj().T @ X[sk, :]
    if mode & 2:
        for f in reversed(F.factors):
            sk, rd = f.sk, f.rd
            X[sk, :] = X[sk, :] + f.F.conj().T @ X[rd, :]
            X[rd, :] = f.U.conj().T @ X[rd, :]
            X[rd, :] = X[rd, :] + f.T.conj().T @ X[sk, :]
    return X


def _rskelf_sv_sn(F: RSkelFFactor, X: np.ndarray, mode: int = 3) -> np.ndarray:
    if mode & 1:
        for f in F.factors:
            sk, rd = f.sk, f.rd
            X[rd, :] = X[rd, :] - f.T.T @ X[sk, :]
            X[rd, :] = la.solve_triangular(
                f.L,
                X[rd[f.p], :],
                lower=True,
                unit_diagonal=True,
                check_finite=False,
                overwrite_b=True,
            )
            X[sk, :] = X[sk, :] - f.E @ X[rd, :]
    if mode & 2:
        for f in reversed(F.factors):
            sk, rd = f.sk, f.rd
            X[rd, :] = X[rd, :] - f.F @ X[sk, :]
            X[rd, :] = la.solve_triangular(f.U, X[rd, :], lower=False, check_finite=False, overwrite_b=True)
            X[sk, :] = X[sk, :] - f.T @ X[rd, :]
    return X


def _rskelf_sv_sc(F: RSkelFFactor, X: np.ndarray, mode: int = 3) -> np.ndarray:
    if mode & 1:
        for f in F.factors:
            sk, rd = f.sk, f.rd
            X[rd, :] = X[rd, :] - f.T.conj().T @ X[sk, :]
            X[rd, :] = la.solve_triangular(
                f.U.conj().T,
                X[rd, :],
                lower=True,
                check_finite=False,
                overwrite_b=True,
            )
            X[sk, :] = X[sk, :] - f.F.conj().T @ X[rd, :]
    if mode & 2:
        for f in reversed(F.factors):
            sk, rd = f.sk, f.rd
            X[rd, :] = X[rd, :] - f.E.conj().T @ X[sk, :]
            X[rd[f.p], :] = la.solve_triangular(
                f.L.conj().T,
                X[rd, :],
                lower=False,
                unit_diagonal=True,
                check_finite=False,
                overwrite_b=True,
            )
            X[sk, :] = X[sk, :] - np.conj(f.T) @ X[rd, :]
    return X


def _rskelf_mv_h(F: RSkelFFactor, X: np.ndarray, mode: int = 3) -> np.ndarray:
    if mode & 1:
        for f in F.factors:
            sk, rd = f.sk, f.rd
            X[sk, :] = X[sk, :] + f.T @ X[rd, :]
            X[rd, :] = f.L.conj().T @ X[rd, :]
            X[rd, :] = X[rd, :] + f.E.conj().T @ X[sk, :]
            X[rd, :] = f.U @ X[rd, :]
    if mode & 2:
        for f in reversed(F.factors):
            sk, rd = f.sk, f.rd
            X[sk, :] = X[sk, :] + f.E @ X[rd, :]
            X[rd, :] = f.L @ X[rd, :]
            X[rd, :] = X[rd, :] + f.T.conj().T @ X[sk, :]
    return X


def _rskelf_sv_h(F: RSkelFFactor, X: np.ndarray, mode: int = 3) -> np.ndarray:
    if mode & 1:
        for f in F.factors:
            sk, rd = f.sk, f.rd
            X[rd, :] = X[rd, :] - f.T.conj().T @ X[sk, :]
            X[rd, :] = la.solve_triangular(f.L, X[rd, :], lower=True, unit_diagonal=True, check_finite=False)
            X[sk, :] = X[sk, :] - f.E @ X[rd, :]
            X[rd, :] = la.solve(f.U, X[rd, :], check_finite=False)
    if mode & 2:
        for f in reversed(F.factors):
            sk, rd = f.sk, f.rd
            X[rd, :] = X[rd, :] - f.E.conj().T @ X[sk, :]
            X[rd, :] = la.solve_triangular(
                f.L.conj().T,
                X[rd, :],
                lower=False,
                unit_diagonal=True,
                check_finite=False,
            )
            X[sk, :] = X[sk, :] - f.T @ X[rd, :]
    return X


def _rskelf_mv_p(F: RSkelFFactor, X: np.ndarray, mode: int = 3) -> np.ndarray:
    if mode & 1:
        X = _rskelf_cholmv_p(F, X, "c")
    if mode & 2:
        X = _rskelf_cholmv_p(F, X, "n")
    return X


def _rskelf_sv_p(F: RSkelFFactor, X: np.ndarray, mode: int = 3) -> np.ndarray:
    if mode & 1:
        X = _rskelf_cholsv_p(F, X, "n")
    if mode & 2:
        X = _rskelf_cholsv_p(F, X, "c")
    return X


def _rskelf_cholmv_p(F: RSkelFFactor, X: np.ndarray, trans: str) -> np.ndarray:
    if trans == "n":
        for f in reversed(F.factors):
            sk, rd = f.sk, f.rd
            X[sk, :] = X[sk, :] + f.E @ X[rd, :]
            X[rd, :] = f.L @ X[rd, :]
            X[rd, :] = X[rd, :] + f.T.conj().T @ X[sk, :]
    else:
        for f in F.factors:
            sk, rd = f.sk, f.rd
            X[sk, :] = X[sk, :] + f.T @ X[rd, :]
            X[rd, :] = f.L.conj().T @ X[rd, :]
            X[rd, :] = X[rd, :] + f.E.conj().T @ X[sk, :]
    return X


def _rskelf_cholsv_p(F: RSkelFFactor, X: np.ndarray, trans: str) -> np.ndarray:
    if trans == "n":
        for f in F.factors:
            sk, rd = f.sk, f.rd
            X[rd, :] = X[rd, :] - f.T.conj().T @ X[sk, :]
            X[rd, :] = la.solve_triangular(f.L, X[rd, :], lower=True, check_finite=False, overwrite_b=True)
            X[sk, :] = X[sk, :] - f.E @ X[rd, :]
    else:
        for f in reversed(F.factors):
            sk, rd = f.sk, f.rd
            X[rd, :] = X[rd, :] - f.E.conj().T @ X[sk, :]
            X[rd, :] = la.solve_triangular(
                f.L.conj().T,
                X[rd, :],
                lower=False,
                check_finite=False,
                overwrite_b=True,
            )
            X[sk, :] = X[sk, :] - f.T @ X[rd, :]
    return X


def rskelf_partial_info(F: RSkelFFactor):
    if F.Si is None:
        return np.array([], dtype=np.int64), np.zeros((0, 0))
    return F.Si.copy(), np.array(F.S, copy=True)


def rskelf_partial_mv(F: RSkelFFactor, X, mvfun=None, trans: str = "n") -> np.ndarray:
    trans = chktrans(trans)
    if trans == "t":
        return np.conj(rskelf_partial_mv(F, np.conj(X), mvfun, "c"))
    X = _prepare_rhs(F, X)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    X = _rskelf_apply_mode(F, X, trans, mode=1, solve=False)
    if F.Si is not None and F.Si.size:
        fun = mvfun if mvfun is not None else _default_partial_mvfun(F)
        X[F.Si, :] = fun(X[F.Si, :], trans)
    X = _rskelf_apply_mode(F, X, trans, mode=2, solve=False)
    return X[:, 0] if one_dim else X


def rskelf_partial_sv(F: RSkelFFactor, X, svfun=None, trans: str = "n") -> np.ndarray:
    trans = chktrans(trans)
    if trans == "t":
        return np.conj(rskelf_partial_sv(F, np.conj(X), svfun, "c"))
    X = _prepare_rhs(F, X)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    X = _rskelf_apply_mode(F, X, trans, mode=1, solve=True)
    if F.Si is not None and F.Si.size:
        fun = svfun if svfun is not None else _default_partial_svfun(F)
        X[F.Si, :] = fun(X[F.Si, :], trans)
    X = _rskelf_apply_mode(F, X, trans, mode=2, solve=True)
    return X[:, 0] if one_dim else X


def _rskelf_apply_mode(F: RSkelFFactor, X: np.ndarray, trans: str, mode: int, solve: bool) -> np.ndarray:
    if F.symm == "n":
        if solve:
            return _rskelf_sv_nn(F, X, mode) if trans == "n" else _rskelf_sv_nc(F, X, mode)
        return _rskelf_mv_nn(F, X, mode) if trans == "n" else _rskelf_mv_nc(F, X, mode)
    if F.symm == "s":
        if solve:
            return _rskelf_sv_sn(F, X, mode) if trans == "n" else _rskelf_sv_sc(F, X, mode)
        return _rskelf_mv_sn(F, X, mode) if trans == "n" else _rskelf_mv_sc(F, X, mode)
    if F.symm == "h":
        return _rskelf_sv_h(F, X, mode) if solve else _rskelf_mv_h(F, X, mode)
    return _rskelf_sv_p(F, X, mode) if solve else _rskelf_mv_p(F, X, mode)


def _default_partial_matrix(F: RSkelFFactor) -> np.ndarray:
    if F.Si is None or F.A_dense is None or F.S is None:
        raise ValueError("partial factor does not contain skeleton matrix data")
    return F.A_dense[np.ix_(F.Si, F.Si)] + F.S


def _default_partial_mvfun(F: RSkelFFactor):
    A = _default_partial_matrix(F)

    def mvfun(X, trans="n"):
        return apply_transpose(A, X, chktrans(trans))

    return mvfun


def _default_partial_svfun(F: RSkelFFactor):
    A = _default_partial_matrix(F)
    lu = la.lu_factor(A, check_finite=False)

    def svfun(X, trans="n"):
        trans = chktrans(trans)
        if trans == "n":
            return la.lu_solve(lu, X, check_finite=False)
        if trans == "t":
            return la.lu_solve(lu, X, trans=1, check_finite=False)
        return la.lu_solve(lu, X, trans=2, check_finite=False)

    return svfun


__all__ = [
    "RSkelFFactorBlock",
    "RSkelFFactor",
    "rskelf",
    "rskelf_cholmv",
    "rskelf_cholsv",
    "rskelf_diag",
    "rskelf_logdet",
    "rskelf_mv",
    "rskelf_partial_info",
    "rskelf_partial_mv",
    "rskelf_partial_sv",
    "rskelf_spdiag",
    "rskelf_sv",
]
