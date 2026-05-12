"""Hierarchical interpolative factorization for differential equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .core import StructMixin, _as_points, _normalise_opts, chksymm, hypoct, id
from .hifie import _dimensional_blocks
from .mf import (
    _as_square_sparse_matrix,
    _concat_unique,
    _external_column_interactions,
    _lu_block,
    _lu_factor_allow_singular,
    _spget_dense,
    _triangular_solve,
)
from .rskelf import (
    RSkelFFactor,
    RSkelFFactorBlock,
    _rskelf_diag_unfold,
    _rskelf_spdiag_sparse_from_info,
    rskelf_cholmv,
    rskelf_cholsv,
    rskelf_diag,
    rskelf_logdet,
    rskelf_mv,
    rskelf_sv,
)


@dataclass
class HIFDEFactor(StructMixin):
    backend: RSkelFFactor
    variant: str
    N: int
    nlvl: int
    lvp: np.ndarray
    factors: list[Any]
    symm: str
    opts: dict[str, Any]

    @classmethod
    def from_backend(cls, backend: RSkelFFactor, variant: str, rank_or_tol, opts: dict[str, Any]) -> "HIFDEFactor":
        factor_opts = dict(opts)
        factor_opts["hifde_variant"] = variant
        factor_opts["rank_or_tol"] = rank_or_tol
        backend.opts = dict(factor_opts)
        return cls(
            backend=backend,
            variant=variant,
            N=backend.N,
            nlvl=backend.nlvl,
            lvp=backend.lvp,
            factors=backend.factors,
            symm=backend.symm,
            opts=factor_opts,
        )

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "backend"), name)


@dataclass
class _HIFDEBlock:
    slf: np.ndarray
    sk: np.ndarray
    rd: np.ndarray
    T: np.ndarray


def hifde2(A, n: int, occ: int, rank_or_tol, opts: dict[str, Any] | None = None) -> HIFDEFactor:
    """Factor a sparse matrix on a regular 2D mesh using the HIFDE API."""

    o = _hifde_opts(opts, ext=False, point_dim=2)
    backend = _hifde_regular(A, int(n), int(occ), rank_or_tol, o, dim=2)
    return HIFDEFactor.from_backend(backend, "hifde2", rank_or_tol, o)


def hifde3(A, n: int, occ: int, rank_or_tol, opts: dict[str, Any] | None = None) -> HIFDEFactor:
    """Factor a sparse matrix on a regular 3D mesh using the HIFDE API."""

    o = _hifde_opts(opts, ext=False, point_dim=3)
    backend = _hifde_regular(A, int(n), int(occ), rank_or_tol, o, dim=3)
    return HIFDEFactor.from_backend(backend, "hifde3", rank_or_tol, o)


def hifde2x(A, x, occ: int, rank_or_tol, opts: dict[str, Any] | None = None) -> HIFDEFactor:
    """Point-cloud 2D HIFDE entry point."""

    o = _hifde_opts(opts, ext=True, point_dim=2)
    backend = _hifde_point(A, x, int(occ), rank_or_tol, o, dim=2)
    return HIFDEFactor.from_backend(backend, "hifde2x", rank_or_tol, o)


def hifde3x(A, x, occ: int, rank_or_tol, opts: dict[str, Any] | None = None) -> HIFDEFactor:
    """Point-cloud 3D HIFDE entry point."""

    o = _hifde_opts(opts, ext=True, point_dim=3)
    backend = _hifde_point(A, x, int(occ), rank_or_tol, o, dim=3)
    return HIFDEFactor.from_backend(backend, "hifde3x", rank_or_tol, o)


def hifde_mv(F: HIFDEFactor, X, trans: str = "n"):
    """Apply a HIFDE factor or its transpose/adjoint."""

    return rskelf_mv(F.backend, X, trans)


def hifde_sv(F: HIFDEFactor, X, trans: str = "n"):
    """Solve with a HIFDE factor or its transpose/adjoint."""

    return rskelf_sv(F.backend, X, trans)


def hifde_logdet(F: HIFDEFactor):
    """Return the log determinant represented by a HIFDE factor."""

    return rskelf_logdet(F.backend)


def hifde_cholmv(F: HIFDEFactor, X, trans: str = "n"):
    """Apply the generalized Cholesky factor for positive-definite HIFDE."""

    return rskelf_cholmv(F.backend, X, trans)


def hifde_cholsv(F: HIFDEFactor, X, trans: str = "n"):
    """Apply the inverse generalized Cholesky factor for positive-definite HIFDE."""

    return rskelf_cholsv(F.backend, X, trans)


def hifde_diag(F: HIFDEFactor, dinv: bool | int = False, opts: dict[str, Any] | None = None):
    """Extract ``diag(F)`` or ``diag(inv(F))`` from a HIFDE factor."""

    if F.backend.Si is not None and F.backend.Si.size == 0:
        return _rskelf_diag_unfold(F.backend, dinv, external=True)
    return rskelf_diag(F.backend, dinv, opts)


def hifde_spdiag(F: HIFDEFactor, dinv: bool | int = False):
    """Extract a HIFDE diagonal through the sparse selected-inversion path."""

    if F.backend.Si is not None and F.backend.Si.size == 0:
        return _rskelf_spdiag_sparse_from_info(F.backend, dinv, *_hifde_spdiag_info(F.backend))
    return hifde_diag(F, dinv)


def _hifde_opts(opts: dict[str, Any] | None, ext: bool, point_dim: int) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "lvlmax": np.inf,
        "Tmax": 2,
        "rrqr_iter": np.inf,
        "skip": 0,
        "symm": "n",
        "verb": 0,
    }
    if ext:
        defaults["ext"] = None
    out = _normalise_opts(opts, defaults)
    out["symm"] = chksymm(out["symm"])
    return out


def _hifde_regular(A, n: int, occ: int, rank_or_tol, opts: dict[str, Any], dim: int) -> RSkelFFactor:
    if n <= 0:
        raise ValueError("mesh size must be positive")
    if occ <= 0:
        raise ValueError("leaf occupancy must be positive")
    if opts["lvlmax"] < 1:
        raise ValueError("maximum tree depth must be at least 1")

    nd = n - 1
    N = nd**dim
    A0 = _as_square_sparse_matrix(A, N)
    A_work = A0.copy().tocsc()
    nlvl_tree = int(min(float(opts["lvlmax"]), np.ceil(max(0.0, np.log2(n / occ))) + 1))
    grid = np.arange(N, dtype=np.int64).reshape((nd,) * dim, order="F")
    rem = np.ones(N, dtype=bool)
    factors: list[RSkelFFactorBlock] = []
    lvp = [0]
    skip = _skip_fun(opts["skip"], None)

    w = n
    for _ in range(nlvl_tree):
        w = int(np.ceil(w / 2))

    for level in range(nlvl_tree, 0, -1):
        w *= 2
        nb = int(np.ceil(n / w))
        stages = (2, 1) if dim == 2 else (3, 2)
        for stage_dim in stages:
            if stage_dim < dim:
                if level == 1:
                    break
                if skip(nlvl_tree - level, None, stage_dim):
                    continue

            blocks = (
                _regular_cell_blocks(grid, rem, n, w, nb, dim, A_work.dtype)
                if stage_dim == dim
                else _regular_separator_blocks(A_work, grid, rem, n, w, nb, dim, stage_dim, rank_or_tol, opts)
            )
            upd_i: list[np.ndarray] = []
            upd_j: list[np.ndarray] = []
            upd_v: list[np.ndarray] = []
            for block in blocks:
                if block.rd.size == 0:
                    continue
                factor, rows, cols, vals = _factor_sparse_block(A_work, block, opts["symm"])
                factors.append(factor)
                if rows.size:
                    upd_i.append(rows)
                    upd_j.append(cols)
                    upd_v.append(vals)
            lvp.append(len(factors))
            A_work = _updated_sparse(A_work, rem, upd_i, upd_j, upd_v)

    return _backend_factor(A, A0, A_work.dtype, N, lvp, factors, opts)


def _hifde_point(A, x, occ: int, rank_or_tol, opts: dict[str, Any], dim: int) -> RSkelFFactor:
    if occ <= 0:
        raise ValueError("leaf occupancy must be positive")
    x = _as_points(x)
    if x.shape[0] != dim:
        raise ValueError(f"expected {dim}-D point coordinates")
    N = x.shape[1]
    A0 = _as_square_sparse_matrix(A, N)
    A_work = A0.copy().tocsc()
    tree = hypoct(x, occ, opts["lvlmax"], opts["ext"])
    rem = np.ones(N, dtype=bool)
    factors: list[RSkelFFactorBlock] = []
    lvp = [0]
    skip = _skip_fun(opts["skip"], dim)

    for lvl in range(tree.nlvl - 1, -1, -1):
        start, end = int(tree.lvp[lvl]), int(tree.lvp[lvl + 1])
        level_widths = tree.widths[:, lvl]

        for node_idx in range(start, end):
            child_xi = _concat_unique([tree.nodes[ch].xi for ch in tree.nodes[node_idx].chld])
            if child_xi.size:
                tree.nodes[node_idx].xi = np.unique(np.concatenate((tree.nodes[node_idx].xi, child_xi)))

        for stage_dim in range(dim, 0, -1):
            if stage_dim < dim:
                if lvl == 0:
                    break
                if skip(tree.nlvl - lvl - 1, level_widths, stage_dim):
                    continue
                dim_blocks = _dimensional_blocks(tree, x, rem, lvl, level_widths, stage_dim, dim)
                for node_idx in range(start, end):
                    tree.nodes[node_idx].xi = np.array([], dtype=np.int64)
                blocks = _point_separator_blocks(A_work, tree, dim_blocks, rem, rank_or_tol, opts)
            else:
                blocks = _point_cell_blocks(A_work, tree, start, end, rem, opts)

            upd_i: list[np.ndarray] = []
            upd_j: list[np.ndarray] = []
            upd_v: list[np.ndarray] = []
            for block in blocks:
                if block.rd.size == 0:
                    continue
                factor, rows, cols, vals = _factor_sparse_block(A_work, block, opts["symm"])
                factors.append(factor)
                if rows.size:
                    upd_i.append(rows)
                    upd_j.append(cols)
                    upd_v.append(vals)
            lvp.append(len(factors))
            A_work = _updated_sparse(A_work, rem, upd_i, upd_j, upd_v)

    return _backend_factor(A, A0, A_work.dtype, N, lvp, factors, opts, tree=tree)


def _regular_cell_blocks(
    grid: np.ndarray,
    rem: np.ndarray,
    n: int,
    w: int,
    nb: int,
    dim: int,
    dtype,
) -> list[_HIFDEBlock]:
    nd = n - 1
    blocks: list[_HIFDEBlock] = []
    if dim == 2:
        for i in range(1, nb + 1):
            ia, ib = (i - 1) * w, i * w
            is_ = np.arange(max(1, ia) - 1, min(nd, ib), dtype=np.int64)
            for j in range(1, nb + 1):
                ja, jb = (j - 1) * w, j * w
                js = np.arange(max(1, ja) - 1, min(nd, jb), dtype=np.int64)
                if is_.size == 0 or js.size == 0:
                    continue
                slf = grid[np.ix_(is_, js)].ravel(order="F")
                slf = slf[rem[slf]]
                if slf.size == 0:
                    continue
                jj = slf // nd + 1
                ii = slf - nd * (jj - 1) + 1
                interior = (ii != ia) & (ii != ib) & (jj != ja) & (jj != jb)
                blocks.extend(_make_cell_block(slf, interior, rem, dtype))
    else:
        for i in range(1, nb + 1):
            ia, ib = (i - 1) * w, i * w
            is_ = np.arange(max(1, ia) - 1, min(nd, ib), dtype=np.int64)
            for j in range(1, nb + 1):
                ja, jb = (j - 1) * w, j * w
                js = np.arange(max(1, ja) - 1, min(nd, jb), dtype=np.int64)
                for k in range(1, nb + 1):
                    ka, kb = (k - 1) * w, k * w
                    ks = np.arange(max(1, ka) - 1, min(nd, kb), dtype=np.int64)
                    if is_.size == 0 or js.size == 0 or ks.size == 0:
                        continue
                    slf = grid[np.ix_(is_, js, ks)].ravel(order="F")
                    slf = slf[rem[slf]]
                    if slf.size == 0:
                        continue
                    kk = slf // (nd**2) + 1
                    idx = slf - (nd**2) * (kk - 1)
                    jj = idx // nd + 1
                    ii = idx - nd * (jj - 1) + 1
                    interior = (ii != ia) & (ii != ib) & (jj != ja) & (jj != jb) & (kk != ka) & (kk != kb)
                    blocks.extend(_make_cell_block(slf, interior, rem, dtype))
    return blocks


def _make_cell_block(slf: np.ndarray, interior: np.ndarray, rem: np.ndarray, dtype) -> list[_HIFDEBlock]:
    sk = np.flatnonzero(~interior)
    rd = np.flatnonzero(interior)
    if rd.size == 0:
        return []
    rem[slf[rd]] = False
    T = np.zeros((sk.size, rd.size), dtype=dtype)
    return [_HIFDEBlock(slf=slf, sk=sk, rd=rd, T=T)]


def _regular_separator_blocks(A, grid, rem, n: int, w: int, nb: int, dim: int, stage_dim: int, rank_or_tol, opts):
    blocks: list[_HIFDEBlock] = []
    if dim == 2 and stage_dim == 1:
        for i in range(1, 2 * nb):
            for j in range(1, 2 * nb):
                mi, mj = i % 2 == 0, j % 2 == 0
                if int(mi) + int(mj) != 1:
                    continue
                ib, jb = i // 2, j // 2
                if mi:
                    is1, in1 = np.array([ib * w]), ib * w + np.arange(-w, w + 1)
                    js1, jn1 = jb * w + np.arange(1, w), jb * w + np.arange(0, w + 1)
                else:
                    is1, in1 = ib * w + np.arange(1, w), ib * w + np.arange(0, w + 1)
                    js1, jn1 = np.array([jb * w]), jb * w + np.arange(-w, w + 1)
                is1, in1 = _inside(is1, n), _inside(in1, n)
                js1, jn1 = _inside(js1, n), _inside(jn1, n)
                if is1.size == 0 or js1.size == 0:
                    continue
                slf = grid[np.ix_(is1 - 1, js1 - 1)].ravel(order="F")
                slf = slf[rem[slf]]
                if slf.size == 0:
                    continue
                nbr = grid[np.ix_(in1 - 1, jn1 - 1)].ravel(order="F")
                nbr = nbr[rem[nbr]]
                blocks.extend(_compress_separator(A, slf, nbr, rem, rank_or_tol, opts))
    elif dim == 3 and stage_dim == 2:
        for i in range(1, 2 * nb):
            for j in range(1, 2 * nb):
                for k in range(1, 2 * nb):
                    mi, mj, mk = i % 2 == 0, j % 2 == 0, k % 2 == 0
                    if int(mi) + int(mj) + int(mk) != 1:
                        continue
                    ib, jb, kb = i // 2, j // 2, k // 2
                    if mi:
                        is1, in1 = np.array([ib * w]), ib * w + np.arange(-w, w + 1)
                        js1, jn1 = jb * w + np.arange(1, w), jb * w + np.arange(0, w + 1)
                        ks1, kn1 = kb * w + np.arange(1, w), kb * w + np.arange(0, w + 1)
                    elif mj:
                        is1, in1 = ib * w + np.arange(1, w), ib * w + np.arange(0, w + 1)
                        js1, jn1 = np.array([jb * w]), jb * w + np.arange(-w, w + 1)
                        ks1, kn1 = kb * w + np.arange(1, w), kb * w + np.arange(0, w + 1)
                    else:
                        is1, in1 = ib * w + np.arange(1, w), ib * w + np.arange(0, w + 1)
                        js1, jn1 = jb * w + np.arange(1, w), jb * w + np.arange(0, w + 1)
                        ks1, kn1 = np.array([kb * w]), kb * w + np.arange(-w, w + 1)
                    is1, in1 = _inside(is1, n), _inside(in1, n)
                    js1, jn1 = _inside(js1, n), _inside(jn1, n)
                    ks1, kn1 = _inside(ks1, n), _inside(kn1, n)
                    if is1.size == 0 or js1.size == 0 or ks1.size == 0:
                        continue
                    slf = grid[np.ix_(is1 - 1, js1 - 1, ks1 - 1)].ravel(order="F")
                    slf = slf[rem[slf]]
                    if slf.size == 0:
                        continue
                    nbr = grid[np.ix_(in1 - 1, jn1 - 1, kn1 - 1)].ravel(order="F")
                    nbr = nbr[rem[nbr]]
                    blocks.extend(_compress_separator(A, slf, nbr, rem, rank_or_tol, opts))
    return blocks


def _inside(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    return values[(values > 0) & (values < n)]


def _compress_separator(A, slf: np.ndarray, nbr: np.ndarray, rem: np.ndarray, rank_or_tol, opts) -> list[_HIFDEBlock]:
    nbr = np.setdiff1d(np.unique(nbr), np.sort(slf), assume_unique=False)
    K = _spget_dense(A, nbr, slf)
    if opts["symm"] == "n":
        K = np.vstack((K, _spget_dense(A, slf, nbr).conj().T))
    sk, rd, T = id(K, rank_or_tol, opts["Tmax"], opts["rrqr_iter"])
    sk = np.asarray(sk, dtype=np.int64)
    rd = np.asarray(rd, dtype=np.int64)
    T = np.asarray(T)
    if rd.size == 0:
        return []
    rem[slf[rd]] = False
    return [_HIFDEBlock(slf=slf, sk=sk, rd=rd, T=T)]


def _point_cell_blocks(A, tree, start: int, end: int, rem: np.ndarray, opts: dict[str, Any]) -> list[_HIFDEBlock]:
    blocks: list[_HIFDEBlock] = []
    A_trans = A.T.tocsc() if opts["symm"] == "n" else None
    for node_idx in range(start, end):
        node = tree.nodes[node_idx]
        slf = np.asarray(node.xi, dtype=np.int64).copy()
        if slf.size == 0:
            continue
        sslf = np.sort(slf)
        I_ext, J_ext = _external_column_interactions(A, slf, sslf)
        if opts["symm"] == "n":
            Ic, Jc = _external_column_interactions(A_trans, slf, sslf)
            if J_ext.size or Jc.size:
                I_ext = np.concatenate((I_ext, Ic))
                J_ext = np.concatenate((J_ext, Jc))
                order = np.argsort(J_ext, kind="stable")
                I_ext = I_ext[order]
                J_ext = J_ext[order]
        sk = np.unique(J_ext)

        nbr = [idx for idx in node.nbor if idx < node_idx]
        if nbr:
            nbr_xi = _concat_unique([tree.nodes[idx].xi for idx in nbr])
            if nbr_xi.size and sk.size:
                keep = np.ones(sk.size, dtype=bool)
                nbrsk_parts = []
                for pos_idx, col in enumerate(sk):
                    segment = J_ext == col
                    if segment.any() and np.all(np.isin(I_ext[segment], nbr_xi)):
                        keep[pos_idx] = False
                        nbrsk_parts.append(I_ext[segment])
                nbrsk = _concat_unique(nbrsk_parts)
                sk = np.concatenate((sk[keep], slf.size + np.arange(nbrsk.size, dtype=np.int64)))
                slf = np.concatenate((slf, nbrsk))

        node.xi = slf[sk] if sk.size else np.array([], dtype=np.int64)
        rd = np.setdiff1d(np.arange(slf.size, dtype=np.int64), np.sort(sk), assume_unique=False)
        if rd.size == 0:
            continue
        rem[slf[rd]] = False
        T = np.zeros((sk.size, rd.size), dtype=A.dtype)
        blocks.append(_HIFDEBlock(slf=slf, sk=sk, rd=rd, T=T))
    return blocks


def _hifde_spdiag_info(F: RSkelFFactor) -> tuple[np.ndarray, list[list[int]]]:
    n = int(F.lvp[-1])
    sp_t: list[set[int]] = [set() for _ in range(n)]
    nbor: list[set[int]] = [set() for _ in range(n)]
    prnt: list[set[int]] = [set() for _ in range(n)]
    x: list[set[int]] = [set() for _ in range(F.N)]
    rem = np.ones(F.N, dtype=bool)

    for lvl in range(F.nlvl):
        x_prev = [set(v) for v in x]
        x = [set() for _ in range(F.N)]
        for factor_idx in range(int(F.lvp[lvl]), int(F.lvp[lvl + 1])):
            f = F.factors[factor_idx]
            slf = np.concatenate((f.sk, f.rd))
            if lvl == 0:
                nbr = set().union(*(x[int(j)] for j in slf)) if slf.size else set()
                for j in nbr:
                    nbor[factor_idx].add(j)
                    nbor[j].add(factor_idx)
            else:
                chld = set().union(*(x_prev[int(j)] for j in slf)) if slf.size else set()
                for j in chld:
                    prnt[j].add(factor_idx)
            for j in slf:
                x[int(j)].add(factor_idx)
            if f.rd.size:
                rem[f.rd] = False
        if lvl > 0:
            for idx in np.flatnonzero(rem):
                if not x[int(idx)]:
                    x[int(idx)] = set(x_prev[int(idx)])

    leaf_for_index = -np.ones(F.N, dtype=np.int64)
    for factor_idx in range(n - 1, -1, -1):
        inherited = set()
        for parent in prnt[factor_idx]:
            inherited.update(sp_t[parent])
        sp_t[factor_idx] = {factor_idx} | nbor[factor_idx] | inherited
        f = F.factors[factor_idx]
        slf = np.concatenate((f.sk, f.rd))
        if slf.size:
            leaf_for_index[slf] = factor_idx

    leaves = np.unique(leaf_for_index[leaf_for_index >= 0])
    return leaves.astype(np.int64), [sorted(sp_t[int(i)]) for i in leaves]


def _point_separator_blocks(A, tree, dim_blocks, rem: np.ndarray, rank_or_tol, opts) -> list[_HIFDEBlock]:
    out: list[_HIFDEBlock] = []
    A_trans = A.T.tocsc() if opts["symm"] == "n" else None
    for block in dim_blocks:
        slf, prnt = _unique_with_parent(block.xi, block.prnt)
        if slf.size == 0:
            continue
        rows = _external_sparse_rows(A, slf)
        if opts["symm"] == "n":
            rows = np.unique(np.concatenate((rows, _external_sparse_rows(A_trans, slf))))
        nbr = np.setdiff1d(rows, np.sort(slf), assume_unique=False)
        K = _spget_dense(A, nbr, slf)
        if opts["symm"] == "n":
            K = np.vstack((K, _spget_dense(A, slf, nbr).conj().T))
        sk, rd, T = id(K, rank_or_tol, opts["Tmax"], opts["rrqr_iter"])
        sk = np.asarray(sk, dtype=np.int64)
        rd = np.asarray(rd, dtype=np.int64)
        T = np.asarray(T)
        for pos in sk:
            tree.nodes[int(prnt[pos])].xi = np.append(tree.nodes[int(prnt[pos])].xi, slf[pos])
        if rd.size == 0:
            continue
        rem[slf[rd]] = False
        out.append(_HIFDEBlock(slf=slf, sk=sk, rd=rd, T=T))
    return out


def _unique_with_parent(xi: np.ndarray, prnt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xi = np.asarray(xi, dtype=np.int64)
    prnt = np.asarray(prnt, dtype=np.int64)
    if xi.size == 0:
        return xi, prnt
    _, first = np.unique(xi, return_index=True)
    first = np.sort(first)
    return xi[first], prnt[first]


def _external_sparse_rows(A, slf: np.ndarray) -> np.ndarray:
    sub = A[:, slf].tocoo()
    return np.unique(sub.row.astype(np.int64)) if sub.nnz else np.array([], dtype=np.int64)


def _factor_sparse_block(A, block: _HIFDEBlock, symm: str):
    slf, sk, rd, T = block.slf, block.sk, block.rd, block.T
    K = _spget_dense(A, slf, slf)
    if T.size:
        if symm == "s":
            K[rd, :] = K[rd, :] - T.T @ K[sk, :]
        else:
            K[rd, :] = K[rd, :] - T.conj().T @ K[sk, :]
        K[:, rd] = K[:, rd] - K[:, sk] @ T

    Krr = K[np.ix_(rd, rd)]
    Ksr = K[np.ix_(sk, rd)]
    Krs = K[np.ix_(rd, sk)]
    Ufac = None
    p = None
    G = None
    if symm == "p":
        L = np.linalg.cholesky(Krr)
        E = _triangular_solve(L, Ksr.conj().T, lower=True).conj().T
        X = -E @ E.conj().T
    elif symm == "h":
        Lraw, D, perm = la.ldl(Krr, lower=True, hermitian=True, check_finite=False)
        perm = np.asarray(perm, dtype=np.int64)
        rd = rd[perm]
        if T.size:
            T = T[:, perm]
        Ksr = K[np.ix_(sk, rd)]
        L = Lraw[perm, :]
        Ufac = D
        E = _triangular_solve(L, Ksr.conj().T, lower=True, unit_diagonal=True).conj().T
        E = la.solve(D, E.T, check_finite=False).T
        X = -E @ (D @ E.conj().T)
    else:
        lu = _lu_factor_allow_singular(Krr)
        L, Ufac, p = _lu_block(lu, rd.size)
        E = _triangular_solve(Ufac.T, Ksr.T, lower=True).T
        G = _triangular_solve(L, Krs[p, :], lower=True, unit_diagonal=True)
        X = -E @ G

    sk_idx = slf[sk]
    rd_idx = slf[rd]
    rows = cols = vals = np.array([], dtype=np.int64)
    if sk_idx.size:
        rr, cc = np.meshgrid(sk_idx, sk_idx, indexing="ij")
        rows = rr.ravel()
        cols = cc.ravel()
        vals = X.ravel()
    factor = RSkelFFactorBlock(sk=sk_idx, rd=rd_idx, T=T, L=L, U=Ufac, p=p, E=E, F=G)
    return factor, rows, cols, vals


def _updated_sparse(A, rem: np.ndarray, upd_i: list[np.ndarray], upd_j: list[np.ndarray], upd_v: list[np.ndarray]):
    coo = A.tocoo()
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    keep = rem[coo.row] & rem[coo.col]
    if keep.any():
        rows.append(coo.row[keep])
        cols.append(coo.col[keep])
        vals.append(coo.data[keep])
    rows.extend(upd_i)
    cols.extend(upd_j)
    vals.extend(upd_v)
    if not rows:
        return sp.csc_matrix(A.shape, dtype=A.dtype)
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=A.shape)


def _backend_factor(A, A0, dtype, N: int, lvp: list[int], factors, opts, tree=None) -> RSkelFFactor:
    return RSkelFFactor(
        N=N,
        nlvl=len(lvp) - 1,
        lvp=np.asarray(lvp, dtype=np.int64),
        factors=factors,
        symm=opts["symm"],
        A=A0 if sp.issparse(A0) else A,
        A_dense=None,
        tree=tree,
        opts=dict(opts),
        Si=np.array([], dtype=np.int64),
        S=sp.csr_matrix((0, 0), dtype=dtype),
    )


def _skip_fun(skip, point_dim: int | None):
    if callable(skip):
        if point_dim == 3:
            return skip
        return lambda level, box_size, axis: skip(level, box_size) if box_size is not None else skip(level)
    values = np.asarray(skip).reshape(-1)
    if point_dim == 3:
        if values.size == 1:
            values = np.repeat(values, 2)
        return lambda level, box_size, axis: level < values[int(axis) - 1]
    threshold = values[0] if values.size else 0
    return lambda level, box_size, axis: level < threshold


__all__ = [
    "HIFDEFactor",
    "hifde2",
    "hifde2x",
    "hifde3",
    "hifde3x",
    "hifde_cholmv",
    "hifde_cholsv",
    "hifde_diag",
    "hifde_logdet",
    "hifde_mv",
    "hifde_spdiag",
    "hifde_sv",
]
