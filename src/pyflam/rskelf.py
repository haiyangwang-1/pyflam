"""Recursive skeletonization factorization API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from ._matrix import apply_transpose, materialize, submatrix
from .core import (
    StructMixin,
    _as_points,
    _normalise_opts,
    chksymm,
    chktrans,
    detperm,
    hypoct,
    id,
    logdet_ldl,
    spsymm,
    spsymm2,
)


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
    A: Any = None
    A_dense: np.ndarray | None = None
    lu: tuple[np.ndarray, np.ndarray] | None = None
    chol: np.ndarray | None = None
    tree: Any = None
    opts: dict[str, Any] = field(default_factory=dict)
    Si: np.ndarray | None = None
    S: Any = None


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
    A_dense = None if callable(A) else materialize(A, N, N)
    A_dtype = _matrix_dtype(A, N, A_dense)

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
                modified[node_idx] = np.zeros((0, 0), dtype=A_dtype)
                modified_idx[node_idx] = slf
                continue

            nbr = _concat_indices([tree.nodes[j].xi for j in node.nbor])
            nslf = slf.size
            M = np.zeros((nslf, nslf), dtype=A_dtype)
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

            Kpxy = np.zeros((0, nslf), dtype=A_dtype)
            if lvl + 1 > 2:
                if pxyfun is None:
                    nbr = np.setdiff1d(np.flatnonzero(rem), slf, assume_unique=False)
                else:
                    Kpxy, nbr = pxyfun(x, slf, nbr, node_size, node.ctr)
                    Kpxy = np.asarray(Kpxy)
                    nbr = np.asarray(nbr, dtype=np.int64)

            K = submatrix(A, nbr, slf) if nbr.size else np.zeros((0, nslf), dtype=A_dtype)
            if o["symm"] == "n":
                K2 = submatrix(A, slf, nbr).conj().T if nbr.size else np.zeros((0, nslf), dtype=A_dtype)
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
    S = np.zeros((remaining.size, remaining.size), dtype=A_dtype)
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
    if o["symm"] == "p" and A_dense is not None:
        chol = np.linalg.cholesky(A_dense)
    elif A_dense is not None and (o["symm"] != "n" or remaining.size):
        lu = la.lu_factor(A_dense)
    return RSkelFFactor(
        N=N,
        nlvl=len(lvp) - 1,
        lvp=np.asarray(lvp, dtype=np.int64),
        factors=factors,
        symm=o["symm"],
        A=A,
        A_dense=A_dense,
        lu=lu,
        chol=chol,
        tree=tree,
        opts=o,
        Si=remaining,
        S=sp.csr_matrix(S),
    )


def rskelf_mv(F: RSkelFFactor, X, trans: str = "n") -> np.ndarray:
    trans = chktrans(trans)
    if trans == "t":
        return np.conj(rskelf_mv(F, np.conj(X), "c"))
    X = _prepare_rhs(F, X)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    if _has_compact_factor(F):
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
    if _has_compact_factor(F):
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
    if _has_compact_factor(F):
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
    if _has_compact_factor(F):
        Y = _rskelf_cholsv_p(F, X, trans)
    elif trans == "n":
        Y = la.solve_triangular(F.chol, X, lower=True)
    else:
        Y = la.solve_triangular(F.chol.conj().T, X, lower=False)
    return Y[:, 0] if one_dim else Y


def rskelf_diag(F: RSkelFFactor, dinv: bool | int = False, opts: dict[str, Any] | None = None) -> np.ndarray:
    """Extract ``diag(F)`` or ``diag(inv(F))``.

    Complete compact factors use FLAM's matrix-unfolding selected-inversion
    algorithm. Partial factors and dense debug factors fall back to the exact
    public semantics.
    """

    if _has_complete_compact_factor(F):
        return _rskelf_diag_unfold(F, dinv, external=False)
    if _has_compact_factor(F):
        eye = np.eye(F.N, dtype=_factor_dtype(F, np.array(0.0)))
        return np.diag(rskelf_sv(F, eye) if dinv else rskelf_mv(F, eye))
    if F.A_dense is None:
        raise ValueError("factor does not contain matrix data")
    if dinv:
        eye = np.eye(F.N, dtype=F.A_dense.dtype)
        return np.diag(rskelf_sv(F, eye))
    return np.diag(F.A_dense).copy()


def rskelf_spdiag(F: RSkelFFactor, dinv: bool | int = False) -> np.ndarray:
    """Extract a diagonal using the sparse-apply style FLAM API."""

    if _has_complete_compact_factor(F):
        return _rskelf_spdiag_sparse(F, dinv)
    return rskelf_diag(F, dinv)


def _rskelf_spdiag_sparse(F: RSkelFFactor, dinv: bool | int = False) -> np.ndarray:
    spinfo_i, spinfo_t = _rskelf_spdiag_info(F)
    return _rskelf_spdiag_sparse_from_info(F, dinv, spinfo_i, spinfo_t)


def _rskelf_spdiag_sparse_from_info(
    F: RSkelFFactor,
    dinv: bool | int,
    spinfo_i: np.ndarray,
    spinfo_t,
) -> np.ndarray:
    D = np.zeros(F.N, dtype=_factor_dtype(F, np.array(0.0)))
    P = -np.ones(F.N, dtype=np.int64)
    for row_idx in range(spinfo_i.size - 1, -1, -1):
        factor_ids = np.asarray(spinfo_t[row_idx], dtype=np.int64)
        factor_ids = factor_ids[factor_ids >= 0]
        if factor_ids.size == 0:
            continue
        active = _concat_indices(
            [F.factors[int(j)].sk for j in factor_ids] + [F.factors[int(j)].rd for j in factor_ids]
        )
        active = np.unique(active)
        if active.size == 0:
            continue
        P[active] = np.arange(active.size, dtype=np.int64)

        leaf = F.factors[int(spinfo_i[row_idx])]
        slf = np.concatenate((leaf.sk, leaf.rd))
        if slf.size == 0:
            continue
        local_factors = [_localize_rskelf_block(F.factors[int(j)], P) for j in factor_ids]
        local = RSkelFFactor(
            N=active.size,
            nlvl=1,
            lvp=np.array([0, len(local_factors)], dtype=np.int64),
            factors=local_factors,
            symm=F.symm,
            Si=np.array([], dtype=np.int64),
            S=sp.csr_matrix((0, 0), dtype=D.dtype),
        )
        Y = np.zeros((active.size, slf.size), dtype=D.dtype)
        Y[P[slf], :] = np.eye(slf.size, dtype=D.dtype)
        Y = rskelf_sv(local, Y) if dinv else rskelf_mv(local, Y)
        D[slf] = np.diag(Y[P[slf], :])
    return D


def _rskelf_spdiag_info(F: RSkelFFactor) -> tuple[np.ndarray, np.ndarray]:
    n = int(F.lvp[-1])
    t = -np.ones((n, F.nlvl), dtype=np.int64)
    x = -np.ones(F.N, dtype=np.int64)
    for lvl in range(F.nlvl):
        for factor_idx in range(int(F.lvp[lvl]), int(F.lvp[lvl + 1])):
            f = F.factors[factor_idx]
            slf = np.concatenate((f.sk, f.rd))
            t[factor_idx, lvl] = factor_idx
            if lvl > 0 and slf.size:
                child = np.unique(x[slf])
                child = child[child >= 0]
                t[child, lvl] = factor_idx
            if slf.size:
                x[slf] = factor_idx

    for factor_idx in range(n - 1, -1, -1):
        cols = np.flatnonzero(t[factor_idx] >= 0)
        if cols.size:
            lvl = int(cols[-1])
            parent = int(t[factor_idx, lvl])
            if lvl + 1 < F.nlvl:
                t[factor_idx, lvl + 1 :] = t[parent, lvl + 1 :]
        f = F.factors[factor_idx]
        slf = np.concatenate((f.sk, f.rd))
        if slf.size:
            x[slf] = factor_idx
    leaves = np.unique(x[x >= 0])
    return leaves.astype(np.int64), t[leaves]


def _localize_rskelf_block(f: RSkelFFactorBlock, P: np.ndarray) -> RSkelFFactorBlock:
    return RSkelFFactorBlock(
        sk=P[f.sk],
        rd=P[f.rd],
        T=f.T,
        L=f.L,
        U=f.U,
        p=f.p,
        E=f.E,
        F=f.F,
    )


def _rskelf_diag_unfold(F: RSkelFFactor, dinv: bool | int = False, *, external: bool = False) -> np.ndarray:
    N = F.N
    dtype = _factor_dtype(F, np.array(0.0))
    if N == 0:
        return np.array([], dtype=dtype)

    keep = _diag_keep_patterns(F, external)
    M = sp.csc_matrix((N, N), dtype=dtype)
    for lvl in range(F.nlvl - 1, -1, -1):
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        coo = M.tocoo()
        if coo.nnz:
            rows.append(coo.row)
            cols.append(coo.col)
            vals.append(coo.data)

        keep_lvl = keep[lvl].tocsc()
        keep_trans = keep_lvl.T.tocsc() if external else None
        for factor_idx in range(int(F.lvp[lvl]), int(F.lvp[lvl + 1])):
            f = F.factors[factor_idx]
            rd = np.asarray(f.rd, dtype=np.int64)
            sk = np.asarray(f.sk, dtype=np.int64)
            if external:
                ex = _diag_external_indices(keep_lvl, keep_trans, rd, sk)
                rse = np.concatenate((rd, sk, ex))
                X = _diag_unfold_block(F, f, M, rd, sk, ex, bool(dinv))
            else:
                rse = np.concatenate((rd, sk))
                X = _diag_unfold_block(F, f, M, rd, sk, None, bool(dinv))
            if rse.size == 0:
                continue
            local_keep = np.asarray(keep_lvl[np.ix_(rse, rse)].toarray(), dtype=bool)
            mask = local_keep & (X != 0)
            if np.any(mask):
                rr, cc = np.meshgrid(rse, rse, indexing="ij")
                rows.append(rr[mask].astype(np.int64, copy=False))
                cols.append(cc[mask].astype(np.int64, copy=False))
                vals.append(X[mask])

        if rows:
            M = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
            M = M.multiply(keep_lvl)
        else:
            M = sp.csc_matrix((N, N), dtype=dtype)
    return np.asarray(M.diagonal())


def _diag_keep_patterns(F: RSkelFFactor, external: bool) -> list[sp.csc_matrix]:
    N = F.N
    keep: list[sp.csc_matrix] = [sp.eye(N, dtype=bool, format="csc")]
    rem = np.ones(N, dtype=bool)
    for lvl in range(F.nlvl - 1):
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        cur = keep[lvl].tocoo()
        if cur.nnz:
            idx = rem[cur.row] & rem[cur.col]
            if np.any(idx):
                rows.append(cur.row[idx])
                cols.append(cur.col[idx])

        for factor_idx in range(int(F.lvp[lvl]), int(F.lvp[lvl + 1])):
            rd = np.asarray(F.factors[factor_idx].rd, dtype=np.int64)
            if rd.size:
                rem[rd] = False

        keep_lvl = keep[lvl].tocsc()
        keep_trans = keep_lvl.T.tocsc() if external else None
        for factor_idx in range(int(F.lvp[lvl]), int(F.lvp[lvl + 1])):
            f = F.factors[factor_idx]
            sk = np.asarray(f.sk, dtype=np.int64)
            if sk.size:
                rr, cc = np.meshgrid(sk, sk, indexing="ij")
                rows.append(rr.ravel())
                cols.append(cc.ravel())
            if external and lvl > 0 and sk.size:
                rd = np.asarray(f.rd, dtype=np.int64)
                ex = _diag_external_indices(keep_lvl, keep_trans, rd, sk, rem=rem)
                if ex.size:
                    rr, cc = np.meshgrid(ex, sk, indexing="ij")
                    rows.append(np.concatenate((rr.ravel(), cc.ravel())))
                    cols.append(np.concatenate((cc.ravel(), rr.ravel())))

        if rows:
            row = np.concatenate(rows).astype(np.int64, copy=False)
            col = np.concatenate(cols).astype(np.int64, copy=False)
            if F.symm != "n":
                idx = row >= col
                row = row[idx]
                col = col[idx]
            data = np.ones(row.size, dtype=bool)
            keep.append(sp.csc_matrix((data, (row, col)), shape=(N, N), dtype=bool))
        else:
            keep.append(sp.csc_matrix((N, N), dtype=bool))
    return keep


def _diag_external_indices(
    keep: sp.csc_matrix,
    keep_trans: sp.csc_matrix | None,
    rd: np.ndarray,
    sk: np.ndarray,
    rem: np.ndarray | None = None,
) -> np.ndarray:
    if sk.size == 0:
        return np.array([], dtype=np.int64)
    row_idx = keep[:, sk].tocoo().row
    if keep_trans is not None:
        row_idx = np.concatenate((row_idx, keep_trans[:, sk].tocoo().row))
    if rem is not None and row_idx.size:
        row_idx = row_idx[rem[row_idx]]
    if row_idx.size == 0:
        return np.array([], dtype=np.int64)
    slf = np.sort(np.concatenate((rd, sk)))
    return np.setdiff1d(np.unique(row_idx.astype(np.int64)), slf, assume_unique=False)


def _diag_unfold_block(
    F: RSkelFFactor,
    f: RSkelFFactorBlock,
    M: sp.csc_matrix,
    rd: np.ndarray,
    sk: np.ndarray,
    ex: np.ndarray | None,
    dinv: bool,
) -> np.ndarray:
    nrd = rd.size
    nsk = sk.size
    nex = 0 if ex is None else ex.size
    dtype = _factor_dtype(F, np.array(0.0))
    X = np.zeros((nrd + nsk + nex, nrd + nsk + nex), dtype=dtype)
    ird = np.arange(nrd)
    isk = nrd + np.arange(nsk)

    if F.symm == "h":
        X[np.ix_(ird, ird)] = la.inv(f.U) if dinv else f.U
    else:
        X[np.ix_(ird, ird)] = np.eye(nrd, dtype=dtype)

    if ex is None:
        Xsk = np.asarray(M[np.ix_(sk, sk)].toarray())
        Xsk = spsymm(Xsk, F.symm)
        X[np.ix_(isk, isk)] = Xsk
    else:
        iex = nrd + nsk + np.arange(nex)
        se = np.concatenate((sk, ex))
        Xse = np.asarray(M[np.ix_(se, se)].toarray())
        Xse[:nsk, :nsk] = spsymm(Xse[:nsk, :nsk], F.symm)
        ise = np.concatenate((isk, iex))
        X[np.ix_(ise, ise)] = Xse
        Aex, Bex = spsymm2(X[np.ix_(iex, isk)], X[np.ix_(isk, iex)], F.symm)
        X[np.ix_(iex, isk)] = Aex
        X[np.ix_(isk, iex)] = Bex

    T = f.T
    L = f.L
    p = f.p
    E = f.E
    if F.symm in ("n", "s"):
        U = f.U
        G = f.F
    else:
        U = f.L.conj().T
        G = f.E.conj().T

    if dinv:
        X[:, ird] = _right_solve_triangular(X[:, ird] - X[:, isk] @ E, L, lower=True, unit_diagonal=(F.symm != "p"))
        rhs = X[ird, :] - G @ X[isk, :]
        if F.symm == "h":
            X[ird, :] = la.solve(U, rhs, check_finite=False)
        else:
            X[ird, :] = _triangular_solve_allow_singular(U, rhs, lower=False)
        if p is not None:
            tmp = X[:, ird].copy()
            X[:, ird[p]] = tmp
            if F.symm == "h":
                tmp = X[ird, :].copy()
                X[ird[p], :] = tmp
        if F.symm == "s":
            X[:, isk] = X[:, isk] - X[:, ird] @ T.T
        else:
            X[:, isk] = X[:, isk] - X[:, ird] @ T.conj().T
        X[isk, :] = X[isk, :] - T @ X[ird, :]
    else:
        X[:, isk] = X[:, isk] + X[:, ird] @ G
        X[isk, :] = X[isk, :] + E @ X[ird, :]
        X[:, ird] = X[:, ird] @ U
        X[ird, :] = L @ X[ird, :]
        if p is not None:
            tmp = X[:, ird].copy()
            if F.symm == "h":
                X[:, ird[p]] = tmp
            tmp = X[ird, :].copy()
            X[ird[p], :] = tmp
        X[:, ird] = X[:, ird] + X[:, isk] @ T
        if F.symm == "s":
            X[ird, :] = X[ird, :] + T.T @ X[isk, :]
        else:
            X[ird, :] = X[ird, :] + T.conj().T @ X[isk, :]

    if ex is None:
        X[np.ix_(isk, isk)] = X[np.ix_(isk, isk)] - Xsk
    else:
        X[np.ix_(ise, ise)] = X[np.ix_(ise, ise)] - Xse
    return X


def _right_solve_triangular(B: np.ndarray, A: np.ndarray, *, lower: bool, unit_diagonal: bool = False) -> np.ndarray:
    return la.solve_triangular(
        A.T,
        B.T,
        lower=not lower,
        unit_diagonal=unit_diagonal,
        check_finite=False,
    ).T


def _triangular_solve_allow_singular(A: np.ndarray, B: np.ndarray, *, lower: bool) -> np.ndarray:
    X = np.array(B, dtype=np.result_type(A, B, 1.0), copy=True)
    n = A.shape[0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if lower:
            for i in range(n):
                if i:
                    X[i, :] = X[i, :] - A[i, :i] @ X[:i, :]
                X[i, :] = X[i, :] / A[i, i]
        else:
            for i in range(n - 1, -1, -1):
                if i + 1 < n:
                    X[i, :] = X[i, :] - A[i, i + 1 :] @ X[i + 1 :, :]
                X[i, :] = X[i, :] / A[i, i]
    return X


def _require_positive_definite(F: RSkelFFactor, caller: str) -> None:
    if F.symm != "p" or (F.chol is None and not _has_compact_factor(F)):
        raise ValueError(f"{caller} requires a factorization built with opts={{'symm': 'p'}}")


def _has_compact_factor(F: RSkelFFactor) -> bool:
    return F.Si is not None


def _has_complete_compact_factor(F: RSkelFFactor) -> bool:
    return F.Si is not None and F.Si.size == 0


def _factor_dtype(F: RSkelFFactor, X) -> np.dtype:
    dtype = np.asarray(X).dtype
    if F.A_dense is not None:
        dtype = np.result_type(dtype, F.A_dense)
    for f in F.factors:
        dtype = np.result_type(dtype, f.T, f.L)
        if f.U is not None:
            dtype = np.result_type(dtype, f.U)
        if f.E is not None:
            dtype = np.result_type(dtype, f.E)
        if f.F is not None:
            dtype = np.result_type(dtype, f.F)
    return np.dtype(dtype)


def _matrix_dtype(A, n: int, A_dense: np.ndarray | None) -> np.dtype:
    if A_dense is not None:
        return A_dense.dtype
    if n == 0:
        return np.dtype(float)
    sample = submatrix(A, np.array([0], dtype=np.int64), np.array([0], dtype=np.int64))
    return np.asarray(sample).dtype


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
        return np.array([], dtype=np.int64), sp.csr_matrix((0, 0))
    return F.Si.copy(), F.S.copy() if sp.issparse(F.S) else np.array(F.S, copy=True)


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
    if F.Si is None or F.S is None:
        raise ValueError("partial factor does not contain skeleton matrix data")
    if F.A_dense is not None:
        A_skel = F.A_dense[np.ix_(F.Si, F.Si)]
    elif F.A is not None:
        A_skel = submatrix(F.A, F.Si, F.Si)
    else:
        raise ValueError("partial factor does not contain skeleton matrix data")
    S = F.S.toarray() if sp.issparse(F.S) else F.S
    return A_skel + S


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
