"""Hierarchical interpolative factorization for integral equations.

The public HIFIE routines share the same apply/solve/logdet surface as
``rskelf`` while building HIFIE-specific dimensional-reduction factors by
default.  A plain ``rskelf`` route remains available only as an explicit debug
option.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from ._matrix import submatrix
from .core import StructMixin, _as_points, _normalise_opts, chksymm, hypoct, id, snorm
from .rskelf import (
    RSkelFFactor,
    RSkelFFactorBlock,
    _rskelf_diag_unfold,
    _rskelf_spdiag_sparse_from_info,
    rskelf,
    rskelf_cholmv,
    rskelf_cholsv,
    rskelf_diag,
    rskelf_logdet,
    rskelf_mv,
    rskelf_sv,
)


@dataclass
class HIFIEFactor(StructMixin):
    backend: RSkelFFactor
    variant: str
    N: int
    nlvl: int
    lvp: np.ndarray
    factors: list[Any]
    symm: str
    opts: dict[str, Any]

    @classmethod
    def from_backend(cls, backend: RSkelFFactor, variant: str) -> "HIFIEFactor":
        opts = dict(backend.opts)
        opts["hifie_variant"] = variant
        return cls(
            backend=backend,
            variant=variant,
            N=backend.N,
            nlvl=backend.nlvl,
            lvp=backend.lvp,
            factors=backend.factors,
            symm=backend.symm,
            opts=opts,
        )

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "backend"), name)


def hifie_id(K, K1, K2, rank_or_tol, Tmax=2, rrqr_iter=np.inf):
    """Compression callback used by FLAM's first-kind HIFIE routines."""

    return id(K, rank_or_tol, Tmax, rrqr_iter)


def hifie_idx(K, K1, K2, rank_or_tol, Tmax=2, rrqr_iter=np.inf):
    """Compression callback used by FLAM's second-kind HIFIE routines."""

    K = np.asarray(K)
    K1 = np.asarray(K1)
    K2 = np.asarray(K2)
    if K.ndim != 2:
        raise ValueError("K must be two-dimensional")
    n = K.shape[1]

    ratio = 1.0
    if rank_or_tol < 1 and np.count_nonzero(K2):
        nrm1 = snorm(n, lambda x: K1 @ x, lambda x: K1.conj().T @ x)[0]
        nrm2 = snorm(n, lambda x: K2 @ x, lambda x: K2.conj().T @ x)[0]
        ratio = min(1.0, nrm1 / nrm2) if nrm2 else 1.0

    K2_pattern = K2 != 0
    if K2_pattern.size:
        K2_pattern = K2_pattern[np.any(K2_pattern, axis=1), :]
    s = np.sum(K2_pattern, axis=0).astype(np.int64) if K2_pattern.size else np.zeros(n, dtype=np.int64)
    if np.sum(s) == 0:
        groups = [np.arange(n, dtype=np.int64)]
    else:
        C = K2_pattern.astype(np.int64).T @ K2_pattern.astype(np.int64)
        smax = np.maximum(s[:, None], s[None, :])
        processed = np.zeros(n, dtype=bool)
        groups = []
        for k in range(n):
            if processed[k]:
                continue
            idx = np.flatnonzero((C[:, k] == smax[:, k]) & ~processed)
            if idx.size == 0:
                continue
            groups.append(idx.astype(np.int64))
            processed[idx] = True

    sk_parts: list[np.ndarray] = []
    rd_parts: list[np.ndarray] = []
    T_parts: list[np.ndarray] = []
    for group in groups:
        sk_, rd_, T_ = id(K[:, group], ratio * rank_or_tol, Tmax, rrqr_iter)
        sk_parts.append(group[sk_])
        rd_parts.append(group[rd_])
        T_parts.append(T_)

    sk = np.concatenate(sk_parts) if sk_parts else np.array([], dtype=np.int64)
    rd = np.concatenate(rd_parts) if rd_parts else np.array([], dtype=np.int64)
    T = np.zeros((sk.size, rd.size), dtype=np.result_type(K, float))
    r0 = c0 = 0
    for T_ in T_parts:
        r1 = r0 + T_.shape[0]
        c1 = c0 + T_.shape[1]
        T[r0:r1, c0:c1] = T_
        r0 = r1
        c0 = c1
    return sk, rd, T


def hifie2(A, x, occ, rank_or_tol, pxyfun=None, opts: dict[str, Any] | None = None) -> HIFIEFactor:
    """Factor a 2D integral-equation matrix using the HIFIE API."""

    return _hifie(A, x, occ, rank_or_tol, pxyfun, opts, variant="hifie2")


def hifie2x(A, x, occ, rank_or_tol, pxyfun=None, opts: dict[str, Any] | None = None) -> HIFIEFactor:
    """Second-kind 2D HIFIE entry point."""

    return _hifie(A, x, occ, rank_or_tol, pxyfun, opts, variant="hifie2x")


def hifie3(A, x, occ, rank_or_tol, pxyfun=None, opts: dict[str, Any] | None = None) -> HIFIEFactor:
    """Factor a 3D integral-equation matrix using the HIFIE API."""

    return _hifie(A, x, occ, rank_or_tol, pxyfun, opts, variant="hifie3")


def hifie3x(A, x, occ, rank_or_tol, pxyfun=None, opts: dict[str, Any] | None = None) -> HIFIEFactor:
    """Second-kind 3D HIFIE entry point."""

    return _hifie(A, x, occ, rank_or_tol, pxyfun, opts, variant="hifie3x")


def hifie_mv(F: HIFIEFactor, X, trans: str = "n"):
    return rskelf_mv(F.backend, X, trans)


def hifie_sv(F: HIFIEFactor, X, trans: str = "n"):
    return rskelf_sv(F.backend, X, trans)


def hifie_logdet(F: HIFIEFactor):
    return rskelf_logdet(F.backend)


def hifie_cholmv(F: HIFIEFactor, X, trans: str = "n"):
    return rskelf_cholmv(F.backend, X, trans)


def hifie_cholsv(F: HIFIEFactor, X, trans: str = "n"):
    return rskelf_cholsv(F.backend, X, trans)


def hifie_diag(F: HIFIEFactor, dinv: bool | int = False, opts: dict[str, Any] | None = None):
    if F.backend.Si is not None and F.backend.Si.size == 0:
        return _rskelf_diag_unfold(F.backend, dinv, external=True)
    return rskelf_diag(F.backend, dinv, opts)


def hifie_spdiag(F: HIFIEFactor, dinv: bool | int = False):
    if F.backend.Si is not None and F.backend.Si.size == 0:
        return _rskelf_spdiag_sparse_from_info(F.backend, dinv, *_hifie_spdiag_info(F.backend))
    return hifie_diag(F, dinv)


def _hifie(A, x, occ, rank_or_tol, pxyfun, opts, variant: str) -> HIFIEFactor:
    defaults = {
        "lvlmax": np.inf,
        "ext": None,
        "Tmax": 2,
        "rrqr_iter": np.inf,
        "skip": 0,
        "symm": "n",
        "verb": 0,
        "debug_rskelf": False,
    }
    o = _normalise_opts(opts, defaults)
    if o.get("debug_rskelf"):
        return HIFIEFactor.from_backend(rskelf(A, x, occ, rank_or_tol, pxyfun=pxyfun, opts=o), variant)

    idfun = hifie_idx if variant.endswith("x") else hifie_id
    dim = 3 if variant.startswith("hifie3") else 2
    return HIFIEFactor.from_backend(_hifie_base(A, x, occ, rank_or_tol, idfun, pxyfun, o, dim), variant)


@dataclass
class _DimBlock:
    ctr: np.ndarray
    xi: np.ndarray
    prnt: np.ndarray
    pnbr: list[int]
    nbr: list[int]


def _hifie_base(A, x, occ, rank_or_tol, idfun, pxyfun, opts: dict[str, Any], dim: int) -> RSkelFFactor:
    opts = dict(opts)
    opts["symm"] = chksymm(opts["symm"])
    x = _as_points(x)
    N = x.shape[1]
    tree = hypoct(x, occ, opts["lvlmax"], opts["ext"])
    skip = _skip_fun(opts["skip"])
    factors: list[RSkelFFactorBlock] = []
    lvp = [0]
    rem = np.ones(N, dtype=bool)
    M = sp.csc_matrix((N, N), dtype=_matrix_dtype(A, N))

    for lvl in range(tree.nlvl - 1, -1, -1):
        level_l = tree.l[:, lvl]
        start, end = int(tree.lvp[lvl]), int(tree.lvp[lvl + 1])

        for node_idx in range(start, end):
            child_xi = _concat_arrays([tree.nodes[ch].xi for ch in tree.nodes[node_idx].chld])
            if child_xi.size:
                tree.nodes[node_idx].xi = np.concatenate((tree.nodes[node_idx].xi, child_xi))

        for stage_dim in range(dim, 0, -1):
            if stage_dim < dim:
                if lvl == 0:
                    break
                if skip(tree.nlvl - lvl - 1, level_l):
                    continue
                blocks = _dimensional_blocks(tree, x, rem, lvl, level_l, stage_dim, dim)
                for node_idx in range(start, end):
                    tree.nodes[node_idx].xi = np.array([], dtype=np.int64)
            else:
                blocks = [
                    _DimBlock(
                        ctr=tree.nodes[node_idx].ctr,
                        xi=np.asarray(tree.nodes[node_idx].xi, dtype=np.int64),
                        prnt=np.array([], dtype=np.int64),
                        pnbr=[],
                        nbr=list(tree.nodes[node_idx].nbor),
                    )
                    for node_idx in range(start, end)
                ]

            upd_i: list[np.ndarray] = []
            upd_j: list[np.ndarray] = []
            upd_v: list[np.ndarray] = []

            for local_idx, blk in enumerate(blocks):
                slf = np.asarray(blk.xi, dtype=np.int64)
                if slf.size == 0:
                    continue
                if stage_dim == dim:
                    node_idx = start + local_idx
                    nbr = _concat_arrays([tree.nodes[j].xi for j in tree.nodes[node_idx].nbor])
                else:
                    nbr = _concat_arrays(
                        [tree.nodes[j].xi for j in blk.pnbr] + [blocks[j].xi for j in blk.nbr]
                    )

                Kpxy = np.zeros((0, slf.size), dtype=M.dtype)
                if lvl > 1:
                    if pxyfun is None:
                        nbr = np.setdiff1d(np.flatnonzero(rem), slf, assume_unique=False)
                    else:
                        Kpxy, nbr = pxyfun(x, slf, nbr, level_l, blk.ctr)
                        Kpxy = np.asarray(Kpxy)
                        nbr = np.asarray(nbr, dtype=np.int64)

                nbr_mod = np.unique(M[:, slf].nonzero()[0])
                nbr_mod = nbr_mod[~np.isin(nbr_mod, np.sort(slf), assume_unique=False)]
                nbr = np.unique(np.concatenate((np.asarray(nbr, dtype=np.int64).reshape(-1), nbr_mod)))

                K1 = submatrix(A, nbr, slf) if nbr.size else np.zeros((0, slf.size), dtype=M.dtype)
                if opts["symm"] == "n":
                    K1r = submatrix(A, slf, nbr).conj().T if nbr.size else np.zeros((0, slf.size), dtype=M.dtype)
                    K1 = np.vstack((K1, K1r))
                K2 = _spget_dense(M, nbr, slf)
                if opts["symm"] == "n":
                    K2r = _spget_dense(M, slf, nbr).conj().T
                    K2 = np.vstack((K2, K2r))
                K = np.vstack((K1 + K2, Kpxy)) if Kpxy.size else K1 + K2
                sk, rd, T = idfun(K, K1, K2, rank_or_tol, opts["Tmax"], opts["rrqr_iter"])
                sk = np.asarray(sk, dtype=np.int64)
                rd = np.asarray(rd, dtype=np.int64)
                T = np.asarray(T)

                if stage_dim == dim:
                    tree.nodes[start + local_idx].xi = slf[sk]
                else:
                    for pos in sk:
                        tree.nodes[int(blk.prnt[pos])].xi = np.append(tree.nodes[int(blk.prnt[pos])].xi, slf[pos])

                if rd.size == 0:
                    continue
                rem[slf[rd]] = False

                Kself = submatrix(A, slf, slf) + _spget_dense(M, slf, slf)
                if opts["symm"] == "s":
                    Kself[rd, :] = Kself[rd, :] - T.T @ Kself[sk, :]
                else:
                    Kself[rd, :] = Kself[rd, :] - T.conj().T @ Kself[sk, :]
                Kself[:, rd] = Kself[:, rd] - Kself[:, sk] @ T

                L, Ufac, p, E, G, Xschur, rd = _local_factor(Kself, sk, rd, T, opts["symm"])
                sk_idx = slf[sk]
                rd_idx = slf[rd]
                if sk_idx.size:
                    rows, cols = np.meshgrid(sk_idx, sk_idx, indexing="ij")
                    upd_i.append(rows.ravel())
                    upd_j.append(cols.ravel())
                    upd_v.append(Xschur.ravel())
                factors.append(RSkelFFactorBlock(sk=sk_idx, rd=rd_idx, T=T, L=L, U=Ufac, p=p, E=E, F=G))

            lvp.append(len(factors))
            M = _updated_sparse_workspace(M, rem, upd_i, upd_j, upd_v)

    return RSkelFFactor(
        N=N,
        nlvl=len(lvp) - 1,
        lvp=np.asarray(lvp, dtype=np.int64),
        factors=factors,
        symm=opts["symm"],
        A=A,
        A_dense=None,
        tree=tree,
        opts=opts,
        Si=np.array([], dtype=np.int64),
        S=sp.csr_matrix((0, 0), dtype=M.dtype),
    )


def _hifie_spdiag_info(F: RSkelFFactor) -> tuple[np.ndarray, list[list[int]]]:
    n = int(F.lvp[-1])
    sp_t: list[set[int]] = [set() for _ in range(n)]
    x = -np.ones(F.N, dtype=np.int64)
    for lvl in range(F.nlvl):
        for factor_idx in range(int(F.lvp[lvl]), int(F.lvp[lvl + 1])):
            f = F.factors[factor_idx]
            slf = np.concatenate((f.sk, f.rd))
            if lvl > 0 and slf.size:
                child = np.unique(x[slf])
                for j in child[child >= 0]:
                    sp_t[int(j)].add(factor_idx)
            if slf.size:
                x[slf] = factor_idx

    for factor_idx in range(n - 1, -1, -1):
        inherited = set()
        for parent in sp_t[factor_idx]:
            inherited.update(sp_t[parent])
        sp_t[factor_idx] = {factor_idx} | inherited
        f = F.factors[factor_idx]
        slf = np.concatenate((f.sk, f.rd))
        if slf.size:
            x[slf] = factor_idx

    leaves = np.unique(x[x >= 0])
    return leaves.astype(np.int64), [sorted(sp_t[int(i)]) for i in leaves]


def _local_factor(Kself: np.ndarray, sk: np.ndarray, rd: np.ndarray, T: np.ndarray, symm: str):
    Krr = Kself[np.ix_(rd, rd)]
    Ksr = Kself[np.ix_(sk, rd)]
    Krs = Kself[np.ix_(rd, sk)]
    if symm == "p":
        L = np.linalg.cholesky(Krr)
        E = la.solve_triangular(L, Ksr.conj().T, lower=True, check_finite=False).conj().T
        return L, None, None, E, None, -E @ E.conj().T, rd
    if symm == "h":
        Lraw, D, perm = la.ldl(Krr, lower=True, hermitian=True, check_finite=False)
        perm = np.asarray(perm, dtype=np.int64)
        rd = rd[perm]
        if T.size:
            T[:, :] = T[:, perm]
        Ksr = Kself[np.ix_(sk, rd)]
        L = Lraw[perm, :]
        E = la.solve_triangular(L, Ksr.conj().T, lower=True, unit_diagonal=True, check_finite=False).conj().T
        E = la.solve(D, E.T, check_finite=False).T
        return L, D, None, E, None, -E @ (D @ E.conj().T), rd

    L, Ufac, p = _lu_vector(Krr)
    E = la.solve_triangular(Ufac.T, Ksr.T, lower=True, check_finite=False).T
    G = la.solve_triangular(L, Krs[p, :], lower=True, unit_diagonal=True, check_finite=False)
    return L, Ufac, p, E, G, -E @ G, rd


def _dimensional_blocks(tree, x: np.ndarray, rem: np.ndarray, lvl: int, level_l: np.ndarray, stage_dim: int, dim: int):
    start, end = int(tree.lvp[lvl]), int(tree.lvp[lvl + 1])
    centers, box2ctr = _candidate_centers(tree, lvl, level_l, stage_dim, dim)
    if centers.shape[1] == 0:
        return []
    centers, box2ctr = _shared_centers(tree, lvl, level_l, centers, box2ctr, stage_dim)
    centers, box2ctr = _add_unassigned_box_centers(tree, lvl, centers, box2ctr)
    nb = centers.shape[1]
    blocks = [
        _DimBlock(ctr=centers[:, i], xi=np.array([], dtype=np.int64), prnt=np.array([], dtype=np.int64), pnbr=[], nbr=[])
        for i in range(nb)
    ]
    if nb == 0:
        return blocks

    point_to_center = -np.ones(x.shape[1], dtype=np.int64)
    for box, node_idx in enumerate(range(start, end)):
        xi = np.asarray(tree.nodes[node_idx].xi, dtype=np.int64)
        ctr_idx = box2ctr[box]
        if xi.size == 0 or ctr_idx.size == 0:
            continue
        diff = x[:, xi][None, :, :] - centers[:, ctr_idx].T[:, :, None]
        dist = np.linalg.norm(diff, axis=1)
        point_to_center[xi] = ctr_idx[np.argmin(dist, axis=0)]

    for box, node_idx in enumerate(range(start, end)):
        xi = np.asarray(tree.nodes[node_idx].xi, dtype=np.int64)
        if xi.size == 0:
            continue
        order = np.argsort(point_to_center[xi], kind="stable")
        xi = xi[order]
        for ctr_idx in box2ctr[box]:
            assigned = xi[point_to_center[xi] == ctr_idx]
            if assigned.size:
                blocks[ctr_idx].xi = np.concatenate((blocks[ctr_idx].xi, assigned))
                blocks[ctr_idx].prnt = np.concatenate(
                    (blocks[ctr_idx].prnt, np.full(assigned.size, node_idx, dtype=np.int64))
                )

    counts = np.bincount(point_to_center[rem & (point_to_center >= 0)], minlength=nb)
    keep = counts > 0
    old_to_new = -np.ones(nb, dtype=np.int64)
    old_to_new[keep] = np.arange(np.count_nonzero(keep), dtype=np.int64)
    centers = centers[:, keep]
    blocks = [blocks[i] for i in np.flatnonzero(keep)]
    for i, block in enumerate(blocks):
        block.ctr = centers[:, i]
    box2ctr = [old_to_new[idx[keep[idx]]] for idx in box2ctr]

    proc = np.zeros(len(blocks), dtype=bool)
    for box, node_idx in enumerate(range(start, end)):
        slf = box2ctr[box]
        nbr_nodes = list(tree.nodes[node_idx].nbor)
        pnbr = [j for j in nbr_nodes if j < start]
        for ctr_idx in slf:
            blocks[ctr_idx].pnbr.extend(pnbr)
        same_level = [j - start for j in nbr_nodes if start <= j < end]
        nbr_centers = _unique_concat([slf] + [box2ctr[j] for j in same_level])
        if slf.size == 0 or nbr_centers.size == 0:
            continue
        diff = centers[:, nbr_centers][:, :, None] - centers[:, slf][:, None, :]
        delta = np.abs(np.round(diff / level_l[:, None, None]))
        near = np.max(delta, axis=0) <= 1
        for local_pos, ctr_idx in enumerate(slf):
            if proc[ctr_idx]:
                continue
            cand = nbr_centers[near[:, local_pos]]
            blocks[ctr_idx].nbr = [int(j) for j in cand if int(j) != int(ctr_idx)]
            proc[ctr_idx] = True
    return blocks


def _candidate_centers(tree, lvl: int, level_l: np.ndarray, stage_dim: int, dim: int):
    start, end = int(tree.lvp[lvl]), int(tree.lvp[lvl + 1])
    if dim == 2:
        offsets = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]], dtype=float)
    elif stage_dim == 2:
        offsets = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    else:
        offsets = np.array(
            [
                [0, -1, -1],
                [0, -1, 1],
                [0, 1, -1],
                [0, 1, 1],
                [-1, 0, -1],
                [-1, 0, 1],
                [1, 0, -1],
                [1, 0, 1],
                [-1, -1, 0],
                [-1, 1, 0],
                [1, -1, 0],
                [1, 1, 0],
            ],
            dtype=float,
        )
    centers = []
    box2ctr = []
    for node_idx in range(start, end):
        base = len(centers)
        ctrs = tree.nodes[node_idx].ctr[:, None] + 0.5 * level_l[:, None] * offsets.T
        centers.extend([ctrs[:, i] for i in range(ctrs.shape[1])])
        box2ctr.append(np.arange(base, base + ctrs.shape[1], dtype=np.int64))
    return np.column_stack(centers) if centers else np.empty((dim, 0)), box2ctr


def _shared_centers(tree, lvl: int, level_l: np.ndarray, centers: np.ndarray, box2ctr: list[np.ndarray], stage_dim: int):
    disp = np.round(2 * (centers - tree.nodes[0].ctr[:, None]) / level_l[:, None]).T
    unique_rows, first, inverse, counts = np.unique(disp, axis=0, return_index=True, return_inverse=True, return_counts=True)
    shared = np.flatnonzero(counts > 1)
    if centers.shape[0] == 3 and stage_dim == 1 and shared.size:
        parent_box = np.concatenate([np.full(idx.size, box, dtype=np.int64) for box, idx in enumerate(box2ctr)])
        order = np.argsort(inverse, kind="stable")
        sorted_inv = inverse[order]
        sorted_parent = parent_box[order]
        keep = []
        start = int(tree.lvp[lvl])
        for center_id in shared:
            parents = sorted_parent[sorted_inv == center_id]
            if parents.size < 2:
                continue
            ctrs = np.column_stack([tree.nodes[start + p].ctr for p in parents])
            dist = np.round(np.sum(np.abs((ctrs[:, :, None] - ctrs[:, None, :]) / level_l[:, None, None]), axis=0))
            if np.any(dist == 2):
                keep.append(center_id)
        shared = np.asarray(keep, dtype=np.int64)
    centers = centers[:, first[shared]] if shared.size else np.empty((centers.shape[0], 0), dtype=centers.dtype)
    old_to_new = -np.ones(unique_rows.shape[0], dtype=np.int64)
    old_to_new[shared] = np.arange(shared.size, dtype=np.int64)
    return centers, [old_to_new[inverse[idx]][old_to_new[inverse[idx]] >= 0] for idx in box2ctr]


def _add_unassigned_box_centers(tree, lvl: int, centers: np.ndarray, box2ctr: list[np.ndarray]):
    start, end = int(tree.lvp[lvl]), int(tree.lvp[lvl + 1])
    extra = []
    for box, node_idx in enumerate(range(start, end)):
        if box2ctr[box].size or tree.nodes[node_idx].xi.size == 0:
            continue
        box2ctr[box] = np.array([centers.shape[1] + len(extra)], dtype=np.int64)
        extra.append(tree.nodes[node_idx].ctr)
    if extra:
        centers = np.column_stack((centers, np.column_stack(extra))) if centers.size else np.column_stack(extra)
    return centers, box2ctr


def _updated_sparse_workspace(M, rem, upd_i, upd_j, upd_v):
    coo = M.tocoo()
    rows = []
    cols = []
    vals = []
    keep = rem[coo.row] & rem[coo.col]
    if keep.any():
        rows.append(coo.row[keep])
        cols.append(coo.col[keep])
        vals.append(coo.data[keep])
    rows.extend(upd_i)
    cols.extend(upd_j)
    vals.extend(upd_v)
    if not rows:
        return sp.csc_matrix(M.shape, dtype=M.dtype)
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=M.shape)


def _lu_vector(A: np.ndarray):
    lu, piv = la.lu_factor(A, check_finite=False)
    n = A.shape[0]
    L = np.tril(lu, -1) + np.eye(n, dtype=lu.dtype)
    U = np.triu(lu)
    p = np.arange(n, dtype=np.int64)
    for i, j in enumerate(piv):
        if i != j:
            p[[i, j]] = p[[j, i]]
    return L, U, p


def _spget_dense(A: sp.spmatrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if rows.size == 0 or cols.size == 0:
        return np.zeros((rows.size, cols.size), dtype=A.dtype)
    return np.asarray(A[np.ix_(rows, cols)].toarray())


def _matrix_dtype(A, n: int):
    if n == 0:
        return np.dtype(float)
    return np.asarray(submatrix(A, np.array([0], dtype=np.int64), np.array([0], dtype=np.int64))).dtype


def _skip_fun(skip):
    if callable(skip):
        return skip
    return lambda lvl, l: lvl < skip


def _concat_arrays(parts: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(part, dtype=np.int64).reshape(-1) for part in parts if np.asarray(part).size]
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.int64)


def _unique_concat(parts: list[np.ndarray]) -> np.ndarray:
    values = _concat_arrays(parts)
    return np.unique(values) if values.size else values


__all__ = [
    "HIFIEFactor",
    "hifie_id",
    "hifie_idx",
    "hifie2",
    "hifie2x",
    "hifie3",
    "hifie3x",
    "hifie_cholmv",
    "hifie_cholsv",
    "hifie_diag",
    "hifie_logdet",
    "hifie_mv",
    "hifie_spdiag",
    "hifie_sv",
]
