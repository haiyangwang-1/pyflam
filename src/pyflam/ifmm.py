"""Interpolative fast multipole method API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._matrix import apply_transpose, materialize, submatrix
from .core import StructMixin, _as_points, _normalise_opts, chksymm, chktrans, hypoct, hypoct_perm, id


@dataclass
class IFMMBBlock(StructMixin):
    is_: np.ndarray
    js: np.ndarray
    ie: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    je: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    D: np.ndarray | None = None
    Bo: np.ndarray | None = None
    Bi: np.ndarray | None = None

    @property
    def isx(self) -> np.ndarray:
        return self.is_

    def __getitem__(self, key: str) -> Any:
        if key == "is":
            return self.is_
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "is":
            self.is_ = value
        else:
            super().__setitem__(key, value)


@dataclass
class IFMMUBlock(StructMixin):
    rsk: np.ndarray
    rrd: np.ndarray
    csk: np.ndarray
    crd: np.ndarray
    rT: np.ndarray
    cT: np.ndarray


@dataclass
class IFMMFactor(StructMixin):
    M: int
    N: int
    P: np.ndarray
    Q: np.ndarray
    nlvl: int
    lvpb: np.ndarray
    lvpu: np.ndarray
    B: list[IFMMBBlock] = field(default_factory=list)
    U: list[IFMMUBlock] = field(default_factory=list)
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
    for node in tree.nodes:
        xi = np.asarray(node.xi, dtype=np.int64)
        row_mask = xi < M
        node.rxi = xi[row_mask]
        node.cxi = xi[~row_mask] - M
        node.xi = np.array([], dtype=np.int64)
        node.dir = []

    for lvl in range(tree.nlvl):
        for i in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
            node = tree.nodes[i]
            if node.rxi.size == 0 and node.cxi.size == 0:
                continue
            for j in node.nbor:
                other = tree.nodes[j]
                if other.rxi.size == 0 and other.cxi.size == 0:
                    continue
                node.dir.append(j)
                if j < tree.lvp[lvl]:
                    tree.nodes[j].dir.append(i)

    B_blocks: list[IFMMBBlock] = []
    U_blocks: list[IFMMUBlock] = []
    lvpb = [0]
    lvpu = [0]
    rrem = np.ones(M, dtype=bool)
    crem = np.ones(N, dtype=bool)

    for lvl in range(tree.nlvl):
        node_size = tree.l[:, lvl]
        for i in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
            node = tree.nodes[i]
            rslf = np.asarray(node.rxi, dtype=np.int64)
            cslf = np.asarray(node.cxi, dtype=np.int64)
            if rslf.size == 0 or cslf.size == 0:
                continue
            D = submatrix(A, rslf, cslf) if o["store"] != "n" else None
            B_blocks.append(IFMMBBlock(is_=rslf, js=cslf, D=D))

            if not o["near"]:
                continue

            dir_nodes = [tree.nodes[j] for j in node.dir]
            rdir = np.concatenate([n.rxi for n in dir_nodes if n.rxi.size]) if dir_nodes else np.array([], dtype=np.int64)
            cdir = np.concatenate([n.cxi for n in dir_nodes if n.cxi.size]) if dir_nodes else np.array([], dtype=np.int64)

            Kpxy = np.zeros((rslf.size, 0), dtype=A_dense.dtype)
            if pxyfun is None:
                cnbr = np.setdiff1d(np.flatnonzero(crem), cslf, assume_unique=False)
            else:
                Kpxy, cnbr = pxyfun("r", rx, cx, rslf, cdir, node_size, node.ctr)
                Kpxy = np.asarray(Kpxy)
                cnbr = np.asarray(cnbr, dtype=np.int64)
            K = submatrix(A, rslf, cnbr).T if cnbr.size else np.zeros((0, rslf.size), dtype=A_dense.dtype)
            if Kpxy.size:
                K = np.vstack((K, Kpxy.T))
            rsk, rrd, rT = id(K, rank_or_tol, o["Tmax"], o["rrqr_iter"]) if rslf.size else (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.zeros((0, 0)),
            )

            if o["symm"] == "n":
                Kpxy = np.zeros((0, cslf.size), dtype=A_dense.dtype)
                if pxyfun is None:
                    rnbr = np.setdiff1d(np.flatnonzero(rrem), rslf, assume_unique=False)
                else:
                    Kpxy, rnbr = pxyfun("c", rx, cx, cslf, rdir, node_size, node.ctr)
                    Kpxy = np.asarray(Kpxy)
                    rnbr = np.asarray(rnbr, dtype=np.int64)
                K = submatrix(A, rnbr, cslf) if rnbr.size else np.zeros((0, cslf.size), dtype=A_dense.dtype)
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

            if rrd.size or crd.size:
                U_blocks.append(
                    IFMMUBlock(
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

        lvpb.append(len(B_blocks))
        lvpu.append(len(U_blocks))

    return IFMMFactor(
        M=M,
        N=N,
        P=P,
        Q=Q,
        nlvl=tree.nlvl + 1,
        lvpb=np.asarray(lvpb + [len(B_blocks)], dtype=np.int64),
        lvpu=np.asarray(lvpu, dtype=np.int64),
        B=B_blocks,
        U=U_blocks,
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


__all__ = ["IFMMBBlock", "IFMMFactor", "IFMMUBlock", "ifmm", "ifmm_mv"]
