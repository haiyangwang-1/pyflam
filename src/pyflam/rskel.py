"""Recursive skeletonization compression API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._matrix import apply_transpose, materialize, submatrix
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

            Kpxy = np.zeros((rslf.size, 0), dtype=A_dense.dtype)
            if lvl + 1 > 2 and rslf.size:
                if pxyfun is None:
                    cnbr = np.setdiff1d(np.flatnonzero(crem), cslf, assume_unique=False)
                else:
                    Kpxy, cnbr = pxyfun("r", rx, cx, rslf, cnbr, node_size, node.ctr)
                    Kpxy = np.asarray(Kpxy)
                    cnbr = np.asarray(cnbr, dtype=np.int64)
            K = submatrix(A, rslf, cnbr).T if (rslf.size and cnbr.size) else np.zeros((0, rslf.size))
            if Kpxy.size:
                K = np.vstack((K, Kpxy.T))
            rsk, rrd, rT = id(K, rank_or_tol, o["Tmax"], o["rrqr_iter"]) if rslf.size else (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.zeros((0, 0)),
            )

            if o["symm"] == "n":
                Kpxy = np.zeros((0, cslf.size))
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
                    np.zeros((0, 0)),
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
    X = np.asarray(X)
    one_dim = X.ndim == 1
    if one_dim:
        X = X[:, None]
    if F.symm == "n":
        Y = _rskel_mv_n(F, X, trans)
    else:
        if F.A_dense is None:
            raise ValueError("factor does not contain matrix data")
        Y = apply_transpose(F.A_dense, X, trans)
    return Y[:, 0] if one_dim else Y


def _rskel_mv_n(F: RSkelFactor, X: np.ndarray, trans: str) -> np.ndarray:
    nlvl = F.nlvl
    p = F.Q if trans == "c" else F.P
    q = F.Q if trans == "n" else F.P
    np_ = p.size
    nq = q.size
    prem = np.ones(np_, dtype=bool)
    qrem = np.ones(nq, dtype=bool)
    qmap = np.zeros((nq, 2), dtype=np.int64)
    Z: list[np.ndarray] = [np.empty((0, X.shape[1]), dtype=np.result_type(X, F.A_dense))]
    Z.extend(np.empty((0, X.shape[1]), dtype=np.result_type(X, F.A_dense)) for _ in range(nlvl - 1))
    Ylevels: list[np.ndarray] = [np.empty((0, X.shape[1]), dtype=np.result_type(X, F.A_dense)) for _ in range(nlvl + 1)]

    qmap[q, 0] = np.arange(nq)
    pf = 0
    Z[0] = np.array(X[q, :], dtype=np.result_type(X, F.A_dense), copy=True)
    for lvl in range(nlvl - 1):
        for i in range(F.lvpu[lvl], F.lvpu[lvl + 1]):
            f = F.U[i]
            if trans == "n":
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
            if trans == "n":
                rd = qmap[f.crd, p1]
                sk = qmap[f.csk, p2]
                T = f.cT
            else:
                rd = qmap[f.rrd, p1]
                sk = qmap[f.rsk, p2]
                T = f.rT
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
            if trans == "c":
                prem[f.crd] = True
            else:
                prem[f.rrd] = True
            if trans == "n":
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
            if trans == "c":
                rd = pmap[f.crd, p1]
                sk1 = pmap[f.csk, p1]
                sk2 = pmap[f.csk, p2]
                T = f.cT
            else:
                rd = pmap[f.rrd, p1]
                sk1 = pmap[f.rsk, p1]
                sk2 = pmap[f.rsk, p2]
                T = f.rT
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


def rskel_xsp(F: RSkelFactor):
    """Embed compressed matrix into an extended sparse form.

    In this exact-storage implementation the extended sparse matrix is simply
    the sparse representation of the retained matrix, with identity row/column
    maps matching the base problem.
    """

    if F.A_dense is None:
        raise ValueError("factor does not contain matrix data")
    return sp.csr_matrix(F.A_dense), np.arange(F.M, dtype=np.int64), np.arange(F.N, dtype=np.int64)


__all__ = ["RSkelDBlock", "RSkelFactor", "RSkelUBlock", "rskel", "rskel_mv", "rskel_xsp"]
