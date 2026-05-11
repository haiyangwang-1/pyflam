"""Interpolative fast multipole method API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._matrix import submatrix
from .core import StructMixin, _as_points, _normalise_opts, chksymm, chktrans, hypoct, hypoct_perm, id


@dataclass
class IFMMBBlock(StructMixin):
    is_: np.ndarray
    js: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
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
    P = perm[~col_mask].astype(np.int64)
    Q = (perm[col_mask] - M).astype(np.int64) if o["symm"] == "n" else np.array([], dtype=np.int64)

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
            if _node_empty(node):
                continue
            for j in node.nbor:
                other = tree.nodes[j]
                if _node_empty(other):
                    continue
                node.dir.append(j)
                if j < tree.lvp[lvl]:
                    other.dir.append(i)

    nlvl = tree.nlvl + 1
    B_blocks: list[IFMMBBlock] = []
    U_blocks: list[IFMMUBlock] = []
    lvpb = np.zeros(nlvl + 2, dtype=np.int64)
    lvpu = np.zeros(nlvl + 1, dtype=np.int64)
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
            B_blocks.append(IFMMBBlock(is_=rslf, js=cslf if o["symm"] == "n" else np.array([], dtype=np.int64), D=D))

            if not o["near"]:
                continue

            dir_nodes = [tree.nodes[j] for j in node.dir]
            rdir = _concat_attr(dir_nodes, "rxi")
            cdir = _concat_attr(dir_nodes, "cxi")
            if (rslf.size == 0 or cdir.size == 0) and (rdir.size == 0 or cslf.size == 0):
                continue

            Kpxy, cnbr = _proxy_with_neighbors(pxyfun, "r", rx, cx, rslf, cdir, node_size, node.ctr)
            if cnbr is None:
                cnbr = np.setdiff1d(np.flatnonzero(crem), cslf, assume_unique=False)
            K = _hstack_blocks(_eval_block(A, rslf, cnbr), Kpxy).conj().T
            rsk, rrd, rT = id(K, rank_or_tol, o["Tmax"], o["rrqr_iter"])

            if o["symm"] == "n":
                Kpxy, rnbr = _proxy_with_neighbors(pxyfun, "c", rx, cx, cslf, rdir, node_size, node.ctr)
                if rnbr is None:
                    rnbr = np.setdiff1d(np.flatnonzero(rrem), rslf, assume_unique=False)
                K = _vstack_blocks(_eval_block(A, rnbr, cslf), Kpxy)
                csk, crd, cT = id(K, rank_or_tol, o["Tmax"], o["rrqr_iter"])
            else:
                csk = crd = np.array([], dtype=np.int64)
                cT = np.zeros((0, 0))

            if rrd.size == 0 and crd.size == 0:
                continue

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
            else:
                node.cxi = node.rxi
                crem[cslf[rrd]] = False

    lvpb[1] = len(B_blocks)
    lvpu[1] = len(U_blocks)

    for lvl in range(tree.nlvl):
        for i in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
            node = tree.nodes[i]
            rslf = np.asarray(node.rxi, dtype=np.int64)
            cslf = np.asarray(node.cxi, dtype=np.int64)
            dir_ids = [j for j in node.dir if j > i]
            dir_nodes = [tree.nodes[j] for j in dir_ids]
            rdir = _concat_attr(dir_nodes, "rxi")
            cdir = _concat_attr(dir_nodes, "cxi")
            if (rslf.size == 0 or cdir.size == 0) and (rdir.size == 0 or cslf.size == 0):
                continue

            Bo = Bi = None
            if o["store"] in {"r", "a"}:
                Bo = _eval_block(A, rdir, cslf)
                if o["symm"] == "n":
                    Bi = _eval_block(A, rslf, cdir)
            B_blocks.append(
                IFMMBBlock(
                    is_=rslf,
                    ie=rdir,
                    js=cslf if o["symm"] == "n" else np.array([], dtype=np.int64),
                    je=cdir if o["symm"] == "n" else np.array([], dtype=np.int64),
                    Bo=Bo,
                    Bi=Bi,
                )
            )
    lvpb[2] = len(B_blocks)

    offset = 1
    for lvl in range(tree.nlvl - 1, -1, -1):
        offset += 1
        node_size = tree.l[:, lvl]

        for i in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
            node = tree.nodes[i]
            child_nodes = [tree.nodes[ch] for ch in node.chld]
            node.rxi = _concat_arrays([node.rxi, _concat_attr(child_nodes, "rxi")])
            node.cxi = _concat_arrays([node.cxi, _concat_attr(child_nodes, "cxi")])

        for i in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
            node = tree.nodes[i]
            rslf = np.asarray(node.rxi, dtype=np.int64)
            cslf = np.asarray(node.cxi, dtype=np.int64)
            rnbr = _concat_attr([tree.nodes[j] for j in node.nbor], "rxi")
            cnbr = _concat_attr([tree.nodes[j] for j in node.nbor], "cxi")

            if lvl > 1:
                if pxyfun is None:
                    cfar = np.setdiff1d(np.flatnonzero(crem), _concat_arrays([cslf, cnbr]), assume_unique=False)
                    Krow = _eval_block(A, rslf, cfar)
                else:
                    Krow = _proxy_matrix(pxyfun, "r", rx, cx, rslf, cnbr, node_size, node.ctr)
            else:
                Krow = np.zeros((rslf.size, 0))
            rsk, rrd, rT = id(np.asarray(Krow).conj().T, rank_or_tol, o["Tmax"], o["rrqr_iter"])

            if o["symm"] == "n":
                if lvl > 1:
                    if pxyfun is None:
                        rfar = np.setdiff1d(np.flatnonzero(rrem), _concat_arrays([rslf, rnbr]), assume_unique=False)
                        Kcol = _eval_block(A, rfar, cslf)
                    else:
                        Kcol = _proxy_matrix(pxyfun, "c", rx, cx, cslf, rnbr, node_size, node.ctr)
                else:
                    Kcol = np.zeros((0, cslf.size))
                csk, crd, cT = id(Kcol, rank_or_tol, o["Tmax"], o["rrqr_iter"])
            else:
                csk = crd = np.array([], dtype=np.int64)
                cT = np.zeros((0, 0))

            if rrd.size == 0 and crd.size == 0:
                continue

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
            else:
                node.cxi = node.rxi
                crem[cslf[rrd]] = False

        lvpu[offset] = len(U_blocks)

        if lvl > 1:
            for i in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
                node = tree.nodes[i]
                rslf = np.asarray(node.rxi, dtype=np.int64)
                cslf = np.asarray(node.cxi, dtype=np.int64)
                ilst = _interaction_list(tree, i, lvl)
                int_nodes = [tree.nodes[j] for j in ilst]
                rint = _concat_attr(int_nodes, "rxi")
                cint = _concat_attr(int_nodes, "cxi")
                if (rslf.size == 0 or cint.size == 0) and (cslf.size == 0 or rint.size == 0):
                    continue

                Bo = Bi = None
                if o["store"] == "a":
                    Bo = _eval_block(A, rint, cslf)
                    if o["symm"] == "n":
                        Bi = _eval_block(A, rslf, cint)
                B_blocks.append(
                    IFMMBBlock(
                        is_=rslf,
                        ie=rint,
                        js=cslf if o["symm"] == "n" else np.array([], dtype=np.int64),
                        je=cint if o["symm"] == "n" else np.array([], dtype=np.int64),
                        Bo=Bo,
                        Bi=Bi,
                    )
                )
        lvpb[offset + 1] = len(B_blocks)

    lvpb[-1] = len(B_blocks)
    lvpu[-1] = len(U_blocks)
    return IFMMFactor(
        M=M,
        N=N,
        P=P,
        Q=Q,
        nlvl=nlvl,
        lvpb=lvpb,
        lvpu=lvpu,
        B=B_blocks,
        U=U_blocks,
        store=o["store"],
        symm=o["symm"],
        A_dense=None,
        tree=tree,
        opts=o,
    )


def ifmm_mv(F: IFMMFactor, X, A=None, trans: str = "n") -> np.ndarray:
    trans = chktrans(trans)
    if trans == "t":
        return np.conj(ifmm_mv(F, np.conj(X), A, "c"))

    X_arr = np.asarray(X)
    one_dim = X_arr.ndim == 1
    if one_dim:
        X_arr = X_arr[:, None]
    if X_arr.ndim != 2:
        raise ValueError("input must be one- or two-dimensional")
    X_arr = np.asarray(X_arr, dtype=_factor_dtype(F, X_arr, A))

    left_order = F.Q if F.symm == "n" and trans == "c" else F.P
    right_order = F.Q if F.symm == "n" and trans == "n" else F.P
    left_dim = F.N if F.symm == "n" and trans == "c" else F.M
    right_dim = F.N if F.symm == "n" and trans == "n" else F.M
    if X_arr.shape[0] != right_dim:
        raise ValueError(f"input has {X_arr.shape[0]} rows, expected {right_dim}")

    qrem = np.ones(right_dim, dtype=bool)
    right_map = np.full(right_dim, -1, dtype=np.int64)
    right_map[right_order] = np.arange(right_order.size, dtype=np.int64)
    Z_levels = [X_arr[right_order, :].copy()]

    for lvl in range(F.nlvl - 1):
        for f in F.U[F.lvpu[lvl] : F.lvpu[lvl + 1]]:
            if F.symm == "n" and trans == "n":
                qrem[f.crd] = False
            else:
                qrem[f.rrd] = False

        next_order = right_order[qrem[right_order]]
        next_map = np.full(right_dim, -1, dtype=np.int64)
        next_map[next_order] = np.arange(next_order.size, dtype=np.int64)
        Z_next = np.zeros((next_order.size, X_arr.shape[1]), dtype=X_arr.dtype)
        if next_order.size:
            Z_next[next_map[next_order], :] = Z_levels[-1][right_map[next_order], :]

        for f in F.U[F.lvpu[lvl] : F.lvpu[lvl + 1]]:
            if F.symm == "n" and trans == "n":
                rd = right_map[f.crd]
                sk = next_map[f.csk]
                T = f.cT
            else:
                rd = right_map[f.rrd]
                sk = next_map[f.rsk]
                T = _upward_T(F, f, trans)
            if rd.size and sk.size:
                Z_next[sk, :] = Z_next[sk, :] + T @ Z_levels[-1][rd, :]

        right_map = next_map
        Z_levels.append(Z_next)

    prem = np.zeros(left_dim, dtype=bool)
    qrem = qrem.copy()
    left_map_next = np.full(left_dim, -1, dtype=np.int64)
    Y_next = np.zeros((0, X_arr.shape[1]), dtype=Z_levels[-1].dtype)
    Y_curr = Y_next
    left_map = left_map_next

    for lvl in range(F.nlvl - 1, -1, -1):
        old_left = left_order[prem[left_order]]
        for f in F.U[F.lvpu[lvl] : F.lvpu[lvl + 1]]:
            if F.symm == "n" and trans == "c":
                prem[f.crd] = True
            else:
                prem[f.rrd] = True
            if F.symm == "n" and trans == "n":
                qrem[f.crd] = True
            else:
                qrem[f.rrd] = True

        current_left = left_order[prem[left_order]]
        left_map = np.full(left_dim, -1, dtype=np.int64)
        left_map[current_left] = np.arange(current_left.size, dtype=np.int64)
        current_right = right_order[qrem[right_order]]
        Qmap = np.full(right_dim, -1, dtype=np.int64)
        Qmap[current_right] = np.arange(current_right.size, dtype=np.int64)
        Y_curr = np.zeros((current_left.size, X_arr.shape[1]), dtype=Z_levels[lvl].dtype)
        if old_left.size:
            Y_curr[left_map[old_left], :] = Y_next[left_map_next[old_left], :]

        for f in F.U[F.lvpu[lvl] : F.lvpu[lvl + 1]]:
            if F.symm == "n" and trans == "c":
                rd = left_map[f.crd]
                sk1 = left_map[f.csk]
                sk2 = left_map_next[f.csk]
                T = f.cT
            else:
                rd = left_map[f.rrd]
                sk1 = left_map[f.rsk]
                sk2 = left_map_next[f.rsk]
                T = _downward_T(F, f, trans)
            if rd.size and sk2.size:
                Y_curr[rd, :] = T.conj().T @ Y_next[sk2, :]
                Y_curr[sk1, :] = Y_next[sk2, :]

        for f in F.B[F.lvpb[lvl] : F.lvpb[lvl + 1]]:
            _apply_interaction(F, f, Y_curr, Z_levels[lvl], left_map, Qmap, A, lvl, trans)

        Y_next = Y_curr
        left_map_next = left_map

    out = Y_curr[left_map[np.arange(left_dim, dtype=np.int64)], :]
    return out[:, 0] if one_dim else out


def _apply_interaction(
    F: IFMMFactor,
    f: IFMMBBlock,
    Y: np.ndarray,
    Z: np.ndarray,
    left_map: np.ndarray,
    right_map: np.ndarray,
    A,
    lvl: int,
    trans: str,
) -> None:
    is_ = f.is_
    ie = f.ie
    if F.symm == "n":
        js = f.js
        je = f.je
    else:
        js = is_
        je = ie

    if lvl == 0:
        D = f.D if f.D is not None else _required_block(A, is_, js)
    else:
        near = lvl == 1 and F.store == "r"
        Bo = f.Bo if (F.store == "a" or near) and f.Bo is not None else _required_block(A, ie, js)
        if F.symm == "n":
            Bi = f.Bi if (F.store == "a" or near) and f.Bi is not None else _required_block(A, is_, je)
        elif F.symm == "s":
            Bi = Bo.T
        else:
            Bi = Bo.conj().T

    if trans == "n":
        if lvl == 0:
            Y[left_map[is_], :] = Y[left_map[is_], :] + D @ Z[right_map[js], :]
        else:
            if ie.size and js.size:
                Y[left_map[ie], :] = Y[left_map[ie], :] + Bo @ Z[right_map[js], :]
            if is_.size and je.size:
                Y[left_map[is_], :] = Y[left_map[is_], :] + Bi @ Z[right_map[je], :]
    else:
        if lvl == 0:
            Y[left_map[js], :] = Y[left_map[js], :] + D.conj().T @ Z[right_map[is_], :]
        else:
            BoT = Bo.conj().T
            BiT = Bi.conj().T
            if js.size and ie.size:
                Y[left_map[js], :] = Y[left_map[js], :] + BoT @ Z[right_map[ie], :]
            if je.size and is_.size:
                Y[left_map[je], :] = Y[left_map[je], :] + BiT @ Z[right_map[is_], :]


def _upward_T(F: IFMMFactor, f: IFMMUBlock, trans: str) -> np.ndarray:
    if F.symm == "n":
        return f.rT
    if F.symm == "s":
        return np.conj(f.rT) if trans == "n" else f.rT
    return f.rT


def _downward_T(F: IFMMFactor, f: IFMMUBlock, trans: str) -> np.ndarray:
    if F.symm == "n":
        return f.rT if trans == "n" else f.cT
    if F.symm == "s":
        return f.rT if trans == "n" else np.conj(f.rT)
    return f.rT


def _factor_dtype(F: IFMMFactor, X: np.ndarray, A) -> np.dtype:
    dtype = np.asarray(X).dtype
    if A is not None:
        if callable(A):
            if F.M and F.N:
                dtype = np.result_type(
                    dtype,
                    _eval_block(A, np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)),
                )
        else:
            dtype = np.result_type(dtype, A.dtype if hasattr(A, "dtype") else np.asarray(A).dtype)
    for f in F.U:
        dtype = np.result_type(dtype, f.rT, f.cT)
    for f in F.B:
        for block in (f.D, f.Bo, f.Bi):
            if block is not None:
                dtype = np.result_type(dtype, block)
    return np.dtype(dtype)


def _required_block(A, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    if A is None:
        raise ValueError("A must be supplied when required interactions are not stored")
    return _eval_block(A, I, J)


def _eval_block(A, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    I = np.asarray(I, dtype=np.int64)
    J = np.asarray(J, dtype=np.int64)
    if I.size == 0 or J.size == 0:
        return np.zeros((I.size, J.size))
    return submatrix(A, I, J)


def _proxy_with_neighbors(pxyfun, rc: str, rx, cx, slf, nbr, l, ctr) -> tuple[np.ndarray, np.ndarray | None]:
    if pxyfun is None:
        return np.zeros((slf.size, 0)) if rc == "r" else np.zeros((0, slf.size)), None
    out = pxyfun(rc, rx, cx, slf, nbr, l, ctr)
    if isinstance(out, tuple):
        K, nbr_out = out
        return np.asarray(K), np.asarray(nbr_out, dtype=np.int64)
    return np.asarray(out), None


def _proxy_matrix(pxyfun, rc: str, rx, cx, slf, nbr, l, ctr) -> np.ndarray:
    out = pxyfun(rc, rx, cx, slf, nbr, l, ctr)
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out)


def _hstack_blocks(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0:
        return B
    if B.size == 0:
        return A
    return np.hstack((A, B))


def _vstack_blocks(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0:
        return B
    if B.size == 0:
        return A
    return np.vstack((A, B))


def _concat_attr(nodes: list[Any], attr: str) -> np.ndarray:
    return _concat_arrays([getattr(node, attr) for node in nodes])


def _concat_arrays(parts: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(part, dtype=np.int64).reshape(-1) for part in parts if np.asarray(part).size]
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.int64)


def _node_empty(node) -> bool:
    return np.asarray(node.rxi).size == 0 and np.asarray(node.cxi).size == 0


def _interaction_list(tree, i: int, lvl: int) -> list[int]:
    parent = tree.nodes[i].prnt
    if parent is None:
        return []
    ilst: list[int] = []
    for j in tree.nodes[parent].nbor:
        if not _node_empty(tree.nodes[j]):
            ilst.append(j)
        if lvl > 0 and j >= tree.lvp[lvl - 1]:
            ilst.extend(tree.nodes[j].chld)
    if not ilst:
        return []
    nbor = set(tree.nodes[i].nbor)
    out = []
    for j in sorted(ilst):
        if j in nbor:
            continue
        if j < tree.lvp[lvl] or (j >= tree.lvp[lvl] and j > i):
            out.append(j)
    return out


__all__ = ["IFMMBBlock", "IFMMFactor", "IFMMUBlock", "ifmm", "ifmm_mv"]
