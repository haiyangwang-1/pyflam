"""Recursive skeletonization compression API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._matrix import materialize, submatrix
from .core import StructMixin, _as_points, _normalise_opts, chksymm, chktrans, hypoct, hypoct_perm, id


@dataclass
class RSkelDBlock(StructMixin):
    i: np.ndarray
    j: np.ndarray
    D: np.ndarray


@dataclass
class RSkelUBlock(StructMixin):
    rsk: np.ndarray
    rrd: np.ndarray
    csk: np.ndarray
    crd: np.ndarray
    rT: np.ndarray
    cT: np.ndarray


@dataclass
class RSkelFactor(StructMixin):
    M: int
    N: int
    P: np.ndarray
    Q: np.ndarray
    nlvl: int
    lvpd: np.ndarray
    lvpu: np.ndarray
    D: list[RSkelDBlock] = field(default_factory=list)
    U: list[RSkelUBlock] = field(default_factory=list)
    symm: str = "n"
    A_dense: np.ndarray | None = None
    tree: Any = None
    opts: dict[str, Any] = field(default_factory=dict)


def rskel(A, rx, cx, occ, rank_or_tol, pxyfun=None, opts=None) -> RSkelFactor:
    """Compress a dense matrix using FLAM's recursive skeletonization interface.

    The tree, permutations, options, and callback/index conventions are
    compatible with the MATLAB routine. Matrix callbacks are evaluated only for
    requested subblocks; dense/sparse array inputs are retained for dtype and
    compatibility metadata.
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
    A_dense = None if callable(A) else materialize(A, M, N)
    A_dtype = _matrix_dtype(A, M, N, A_dense)
    for node in tree.nodes:
        xi = np.asarray(node.xi, dtype=np.int64)
        row_mask = xi < M
        node.rxi = xi[row_mask]
        node.cxi = xi[~row_mask] - M
        node.xi = np.array([], dtype=np.int64)

    D_blocks: list[RSkelDBlock] = []
    U_blocks: list[RSkelUBlock] = []
    lvpd = [0]
    lvpu = [0]
    rrem = np.ones(M, dtype=bool)
    crem = np.ones(N, dtype=bool)

    for lvl in range(tree.nlvl - 1, -1, -1):
        node_size = tree.l[:, lvl]
        for node_idx in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
            node = tree.nodes[node_idx]
            child_r = [tree.nodes[ch].rxi for ch in node.chld if getattr(tree.nodes[ch], "rxi").size]
            child_c = [tree.nodes[ch].cxi for ch in node.chld if getattr(tree.nodes[ch], "cxi").size]
            if child_r:
                node.rxi = np.concatenate((node.rxi, *child_r)) if node.rxi.size else np.concatenate(child_r)
            if child_c:
                node.cxi = np.concatenate((node.cxi, *child_c)) if node.cxi.size else np.concatenate(child_c)

        for node_idx in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
            node = tree.nodes[node_idx]
            rslf = np.asarray(node.rxi, dtype=np.int64)
            cslf = np.asarray(node.cxi, dtype=np.int64)
            rnbr = [tree.nodes[j].rxi for j in node.nbor if getattr(tree.nodes[j], "rxi").size]
            cnbr = [tree.nodes[j].cxi for j in node.nbor if getattr(tree.nodes[j], "cxi").size]
            rnbr = np.concatenate(rnbr) if rnbr else np.array([], dtype=np.int64)
            cnbr = np.concatenate(cnbr) if cnbr else np.array([], dtype=np.int64)

            if not node.chld:
                if rslf.size and cslf.size:
                    D_blocks.append(RSkelDBlock(rslf, cslf, submatrix(A, rslf, cslf)))
            else:
                for ch in node.chld:
                    other = [j for j in node.chld if j != ch]
                    if not other:
                        continue
                    rxi_parts = [tree.nodes[j].rxi for j in other if getattr(tree.nodes[j], "rxi").size]
                    rxi = np.concatenate(rxi_parts) if rxi_parts else np.array([], dtype=np.int64)
                    cxi = np.asarray(tree.nodes[ch].cxi, dtype=np.int64)
                    if rxi.size and cxi.size:
                        D_blocks.append(RSkelDBlock(rxi, cxi, submatrix(A, rxi, cxi)))

            Kpxy = np.zeros((rslf.size, 0), dtype=A_dtype)
            if lvl + 1 > 2 and rslf.size:
                if pxyfun is None:
                    cnbr = np.setdiff1d(np.flatnonzero(crem), cslf, assume_unique=False)
                else:
                    Kpxy, cnbr = pxyfun("r", rx, cx, rslf, cnbr, node_size, node.ctr)
                    Kpxy = np.asarray(Kpxy)
                    cnbr = np.asarray(cnbr, dtype=np.int64)
            K = submatrix(A, rslf, cnbr).conj().T if (rslf.size and cnbr.size) else np.zeros((0, rslf.size))
            if Kpxy.size:
                K = np.vstack((K, Kpxy.conj().T))
            rsk, rrd, rT = id(K, rank_or_tol, o["Tmax"], o["rrqr_iter"]) if rslf.size else (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.zeros((0, 0), dtype=A_dtype),
            )

            if o["symm"] == "n":
                Kpxy = np.zeros((0, cslf.size), dtype=A_dtype)
                if lvl + 1 > 2 and cslf.size:
                    if pxyfun is None:
                        rnbr = np.setdiff1d(np.flatnonzero(rrem), rslf, assume_unique=False)
                    else:
                        Kpxy, rnbr = pxyfun("c", rx, cx, cslf, rnbr, node_size, node.ctr)
                        Kpxy = np.asarray(Kpxy)
                        rnbr = np.asarray(rnbr, dtype=np.int64)
                K = submatrix(A, rnbr, cslf) if (rnbr.size and cslf.size) else np.zeros((0, cslf.size))
                if Kpxy.size:
                    K = np.vstack((K, Kpxy))
                csk, crd, cT = id(K, rank_or_tol, o["Tmax"], o["rrqr_iter"]) if cslf.size else (
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.int64),
                    np.zeros((0, 0), dtype=A_dtype),
                )
            else:
                csk = crd = np.array([], dtype=np.int64)
                cT = np.zeros((0, 0))

            if rrd.size == 0 and crd.size == 0:
                continue

            U_blocks.append(
                RSkelUBlock(
                    rsk=rslf[rsk],
                    rrd=rslf[rrd],
                    csk=cslf[csk] if o["symm"] == "n" else np.array([], dtype=np.int64),
                    crd=cslf[crd] if o["symm"] == "n" else np.array([], dtype=np.int64),
                    rT=rT,
                    cT=cT,
                )
            )
            node.rxi = rslf[rsk]
            rrem[rslf[rrd]] = False
            if o["symm"] == "n":
                node.cxi = cslf[csk]
                crem[cslf[crd]] = False
            else:
                node.cxi = node.rxi
                crem = rrem.copy()

        lvpd.append(len(D_blocks))
        lvpu.append(len(U_blocks))

    return RSkelFactor(
        M=M,
        N=N,
        P=P,
        Q=Q,
        nlvl=tree.nlvl,
        lvpd=np.asarray(lvpd, dtype=np.int64),
        lvpu=np.asarray(lvpu, dtype=np.int64),
        D=D_blocks,
        U=U_blocks,
        symm=o["symm"],
        A_dense=A_dense,
        tree=tree,
        opts=o,
    )


def rskel_mv(F: RSkelFactor, X, trans: str = "n") -> np.ndarray:
    trans = chktrans(trans)
    if trans == "t":
        return np.conj(rskel_mv(F, np.conj(X), "c"))
    dtype = _factor_dtype(F, X)
    X = np.asarray(X, dtype=dtype)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    Y = _rskel_mv_compact(F, X, trans, dtype)
    return Y[:, 0] if one_dim else Y


def _rskel_mv_compact(F: RSkelFactor, X: np.ndarray, trans: str, dtype: np.dtype | None = None) -> np.ndarray:
    nlvl = F.nlvl
    p = F.Q if F.symm == "n" and trans == "c" else F.P
    q = F.Q if F.symm == "n" and trans == "n" else F.P
    np_ = p.size
    nq = q.size
    prem = np.ones(np_, dtype=bool)
    qrem = np.ones(nq, dtype=bool)
    qmap = np.zeros((nq, 2), dtype=np.int64)
    dtype = np.dtype(dtype) if dtype is not None else _factor_dtype(F, X)
    Z: list[np.ndarray] = [np.empty((0, X.shape[1]), dtype=dtype)]
    Z.extend(np.empty((0, X.shape[1]), dtype=dtype) for _ in range(nlvl - 1))
    Ylevels: list[np.ndarray] = [np.empty((0, X.shape[1]), dtype=dtype) for _ in range(nlvl + 1)]

    qmap[q, 0] = np.arange(nq)
    pf = 0
    Z[0] = np.array(X[q, :], dtype=dtype, copy=True)
    for lvl in range(nlvl - 1):
        for i in range(F.lvpu[lvl], F.lvpu[lvl + 1]):
            f = F.U[i]
            if F.symm == "n" and trans == "n":
                qrem[f.crd] = False
            else:
                qrem[f.rrd] = False
        p1 = pf
        p2 = 1 - pf
        pf = p2
        q_active = qrem[q]
        qmap[q[q_active], p2] = np.arange(np.count_nonzero(q_active))
        Z[lvl + 1] = np.empty((np.count_nonzero(qrem), X.shape[1]), dtype=Z[0].dtype)
        Z[lvl + 1][qmap[q[q_active], p2], :] = Z[lvl][qmap[q[q_active], p1], :]

        for i in range(F.lvpu[lvl], F.lvpu[lvl + 1]):
            f = F.U[i]
            if F.symm == "n" and trans == "n":
                rd = qmap[f.crd, p1]
                sk = qmap[f.csk, p2]
            else:
                rd = qmap[f.rrd, p1]
                sk = qmap[f.rsk, p2]
            T = _rskel_upward_T(F, f, trans)
            Z[lvl + 1][sk, :] = Z[lvl + 1][sk, :] + T @ Z[lvl][rd, :]

    prem[:] = False
    pmap = np.zeros((np_, 2), dtype=np.int64)
    qout = np.zeros(nq, dtype=np.int64)
    pf = 0
    Ylevels[nlvl] = np.zeros((0, X.shape[1]), dtype=Z[0].dtype)
    for lvl in range(nlvl - 1, -1, -1):
        r_mask = prem[p]
        r = p[r_mask]
        for i in range(F.lvpu[lvl], F.lvpu[lvl + 1]):
            f = F.U[i]
            if F.symm == "n" and trans == "c":
                prem[f.crd] = True
            else:
                prem[f.rrd] = True
            if F.symm == "n" and trans == "n":
                qrem[f.crd] = True
            else:
                qrem[f.rrd] = True
        p1 = pf
        p2 = 1 - pf
        pf = p2
        p_active = prem[p]
        np_active = np.count_nonzero(p_active)
        pmap[p[p_active], p1] = np.arange(np_active)
        qout[q[qrem[q]]] = np.arange(np.count_nonzero(qrem))
        Ylevels[lvl] = np.zeros((np_active, X.shape[1]), dtype=Z[0].dtype)
        if r.size:
            Ylevels[lvl][pmap[r, p1], :] = Ylevels[lvl + 1][pmap[r, p2], :]

        for i in range(F.lvpu[lvl], F.lvpu[lvl + 1]):
            f = F.U[i]
            if F.symm == "n" and trans == "c":
                rd = pmap[f.crd, p1]
                sk1 = pmap[f.csk, p1]
                sk2 = pmap[f.csk, p2]
            else:
                rd = pmap[f.rrd, p1]
                sk1 = pmap[f.rsk, p1]
                sk2 = pmap[f.rsk, p2]
            T = _rskel_downward_T(F, f, trans)
            Ylevels[lvl][rd, :] = T.conj().T @ Ylevels[lvl + 1][sk2, :]
            Ylevels[lvl][sk1, :] = Ylevels[lvl + 1][sk2, :]

        for i in range(F.lvpd[lvl], F.lvpd[lvl + 1]):
            f = F.D[i]
            if trans == "n":
                j = pmap[f.i, p1]
                k = qout[f.j]
                D = f.D
            else:
                j = pmap[f.j, p1]
                k = qout[f.i]
                D = f.D.conj().T
            Ylevels[lvl][j, :] = Ylevels[lvl][j, :] + D @ Z[lvl][k, :]

    return Ylevels[0][pmap[np.arange(np_), p1], :]


def _factor_dtype(F: RSkelFactor, X) -> np.dtype:
    dtype = np.asarray(X).dtype
    if F.A_dense is not None:
        dtype = np.result_type(dtype, F.A_dense)
    for f in F.D:
        dtype = np.result_type(dtype, f.D)
    for f in F.U:
        dtype = np.result_type(dtype, f.rT, f.cT)
    return np.dtype(dtype)


def _matrix_dtype(A, m: int, n: int, A_dense: np.ndarray | None) -> np.dtype:
    if A_dense is not None:
        return A_dense.dtype
    if m == 0 or n == 0:
        return np.dtype(float)
    sample = submatrix(A, np.array([0], dtype=np.int64), np.array([0], dtype=np.int64))
    return np.asarray(sample).dtype


def _rskel_upward_T(F: RSkelFactor, f: RSkelUBlock, trans: str) -> np.ndarray:
    if F.symm == "n":
        return f.cT if trans == "n" else f.rT
    if F.symm == "s":
        return np.conj(f.rT) if trans == "n" else f.rT
    return f.rT


def _rskel_downward_T(F: RSkelFactor, f: RSkelUBlock, trans: str) -> np.ndarray:
    if F.symm == "n":
        return f.rT if trans == "n" else f.cT
    if F.symm == "s":
        return f.rT if trans == "n" else np.conj(f.rT)
    return f.rT


def rskel_xsp(F: RSkelFactor):
    """Return FLAM's extended sparse embedding of an ``rskel`` factor."""

    nlvl = F.nlvl
    p = F.P
    q = F.Q if F.symm == "n" else F.P
    P = np.zeros((F.M, 2), dtype=np.int64)
    Q = np.zeros((F.N, 2), dtype=np.int64)
    P[p, 0] = np.arange(F.M, dtype=np.int64)
    Q[q, 0] = np.arange(F.N, dtype=np.int64)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rrem = np.ones(F.M, dtype=bool)
    crem = np.ones(F.N, dtype=bool)
    pf = 0
    M = 0
    N = 0

    for lvl in range(nlvl):
        rn = int(np.count_nonzero(rrem))
        cn = int(np.count_nonzero(crem))
        for f in F.U[F.lvpu[lvl] : F.lvpu[lvl + 1]]:
            rrem[f.rrd] = False
            if F.symm == "n":
                crem[f.crd] = False
            else:
                crem[f.rrd] = False

        rk = int(np.count_nonzero(rrem))
        ck = int(np.count_nonzero(crem))
        p1 = pf
        p2 = 1 - pf
        pf = p2
        P[p[rrem[p]], p2] = np.arange(rk, dtype=np.int64)
        Q[q[crem[q]], p2] = np.arange(ck, dtype=np.int64)

        for f in F.D[F.lvpd[lvl] : F.lvpd[lvl + 1]]:
            rr, cc = np.meshgrid(f.i, f.j, indexing="ij")
            _push_sparse_block(rows, cols, vals, M + P[rr.ravel(), p1], N + Q[cc.ravel(), p1], f.D.ravel())

        if lvl == nlvl - 1:
            M += rn
            N += cn
            break

        if F.symm == "n":
            idx = np.flatnonzero(rrem)
            _push_sparse_block(rows, cols, vals, M + P[idx, p1], N + cn + P[idx, p2], np.ones(rk))
        idx = np.flatnonzero(crem)
        _push_sparse_block(rows, cols, vals, M + rn + Q[idx, p2], N + Q[idx, p1], np.ones(ck))

        for f in F.U[F.lvpu[lvl] : F.lvpu[lvl + 1]]:
            rrd = f.rrd
            rsk = f.rsk
            rT = f.rT.conj().T
            if F.symm == "n":
                crd = f.crd
                csk = f.csk
                cT = f.cT
            elif F.symm == "s":
                crd = f.rrd
                csk = f.rsk
                cT = rT.T
            else:
                crd = f.rrd
                csk = f.rsk
                cT = rT.conj().T

            if F.symm == "n":
                rr, cc = np.meshgrid(rrd, rsk, indexing="ij")
                _push_sparse_block(
                    rows,
                    cols,
                    vals,
                    M + P[rr.ravel(), p1],
                    N + cn + P[cc.ravel(), p2],
                    rT.ravel(),
                )

            rr, cc = np.meshgrid(csk, crd, indexing="ij")
            _push_sparse_block(
                rows,
                cols,
                vals,
                M + rn + Q[rr.ravel(), p2],
                N + Q[cc.ravel(), p1],
                cT.ravel(),
            )

        M += rn
        N += cn
        if F.symm == "n":
            _push_sparse_block(rows, cols, vals, M + np.arange(ck), N + rk + np.arange(ck), -np.ones(ck))
        _push_sparse_block(rows, cols, vals, M + ck + np.arange(rk), N + np.arange(rk), -np.ones(rk))

        M += ck
        N += rk

    if rows:
        I = np.concatenate(rows)
        J = np.concatenate(cols)
        V = np.concatenate(vals)
    else:
        I = np.array([], dtype=np.int64)
        J = np.array([], dtype=np.int64)
        V = np.array([], dtype=float)
    if F.symm != "n":
        mask = I >= J
        I = I[mask]
        J = J[mask]
        V = V[mask]
    return sp.csr_matrix((V, (I, J)), shape=(M, N)), p.copy(), q.copy()


def _push_sparse_block(
    rows: list[np.ndarray],
    cols: list[np.ndarray],
    vals: list[np.ndarray],
    i: np.ndarray,
    j: np.ndarray,
    v: np.ndarray,
) -> None:
    if i.size == 0:
        return
    rows.append(np.asarray(i, dtype=np.int64).reshape(-1))
    cols.append(np.asarray(j, dtype=np.int64).reshape(-1))
    vals.append(np.asarray(v).reshape(-1))


__all__ = ["RSkelDBlock", "RSkelFactor", "RSkelUBlock", "rskel", "rskel_mv", "rskel_xsp"]
