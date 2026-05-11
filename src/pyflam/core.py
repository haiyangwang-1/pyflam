"""Shared helpers for PyFLAM.

The public API follows FLAM's MATLAB routines, while arrays and callback
indices are deliberately Python/NumPy 0-based.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Iterable
import warnings

import numpy as np
from numpy.typing import ArrayLike
import scipy.linalg as la
import scipy.sparse as sp


class StructMixin:
    """Small MATLAB-struct-like dataclass mixin."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def keys(self) -> list[str]:
        return [f.name for f in fields(self)]

    def asdict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.keys()}


@dataclass
class HypOctNode(StructMixin):
    ctr: np.ndarray
    xi: np.ndarray
    prnt: int | None = None
    chld: list[int] | None = None
    nbor: list[int] | None = None

    def __post_init__(self) -> None:
        if self.chld is None:
            self.chld = []
        if self.nbor is None:
            self.nbor = []
        self.xi = np.asarray(self.xi, dtype=np.int64)
        self.ctr = np.asarray(self.ctr)


@dataclass
class HypOctTree(StructMixin):
    nlvl: int
    lvp: np.ndarray
    l: np.ndarray
    nodes: list[HypOctNode]


def _as_points(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("point array must have shape (d, n)")
    return arr


def _normalise_opts(opts: dict[str, Any] | None, defaults: dict[str, Any]) -> dict[str, Any]:
    out = dict(defaults)
    if opts:
        for key, value in opts.items():
            match = next((name for name in out if name.lower() == key.lower()), key)
            out[match] = value
    return out


def chksymm(symm: str | None) -> str:
    if symm is None or symm == "":
        return "n"
    val = str(symm).lower()[0]
    if val not in {"n", "s", "h", "p"}:
        raise ValueError("symmetry parameter must be one of 'N', 'S', 'H', or 'P'")
    return val


def chktrans(trans: str | None) -> str:
    if trans is None or trans == "":
        return "n"
    val = str(trans).lower()[0]
    if val not in {"n", "t", "c"}:
        raise ValueError("transpose parameter must be one of 'N', 'T', or 'C'")
    return val


def hypoct(x: ArrayLike, occ: int, lvlmax: float = np.inf, ext: ArrayLike | None = None) -> HypOctTree:
    """Build an adaptive hyperoctree over columns of ``x``.

    This is a direct Python port of FLAM's tree construction semantics with
    0-based node and point indices.
    """

    if occ < 0:
        raise ValueError("leaf occupancy must be nonnegative")
    if lvlmax < 1:
        raise ValueError("maximum tree depth must be at least 1")

    x = _as_points(x)
    d, n = x.shape
    if ext is None:
        ext_arr = np.column_stack((np.min(x, axis=1), np.max(x, axis=1))) if n else np.zeros((d, 2))
    else:
        ext_arr = np.asarray(ext, dtype=x.dtype)
        if ext_arr.shape != (d, 2):
            raise ValueError("ext must have shape (d, 2)")

    l = ext_arr[:, 1] - ext_arr[:, 0]
    ctr = 0.5 * (ext_arr[:, 0] + ext_arr[:, 1])
    nodes: list[HypOctNode] = [HypOctNode(ctr=ctr, xi=np.arange(n, dtype=np.int64))]
    lvp = [0, 1]
    l_by_level = [l.copy()]
    nlvl = 1

    while nlvl < lvlmax:
        nbox_before = len(nodes)
        current_l = l_by_level[-1].copy()
        max_l = np.max(current_l) if current_l.size else 0
        ldiv = current_l >= max_l / np.sqrt(2) if max_l > 0 else np.zeros(d, dtype=bool)
        next_l = current_l.copy()
        next_l[ldiv] *= 0.5

        for prnt in range(lvp[nlvl - 1], lvp[nlvl]):
            xi = nodes[prnt].xi
            if xi.size <= occ:
                continue
            pctr = nodes[prnt].ctr
            side = (ldiv[:, None] & (x[:, xi] > pctr[:, None])).astype(np.int64)
            child_code = np.sum(side * (2 ** np.arange(d, dtype=np.int64))[:, None], axis=0)
            for code in np.unique(child_code):
                bit = np.array([(int(code) >> k) & 1 for k in range(d)], dtype=current_l.dtype)
                child_ctr = pctr + next_l * ldiv * (bit - 0.5)
                child_xi = xi[child_code == code]
                child_idx = len(nodes)
                nodes.append(HypOctNode(ctr=child_ctr, xi=child_xi, prnt=prnt))
                nodes[prnt].chld.append(child_idx)
            nodes[prnt].xi = np.array([], dtype=np.int64)

        if len(nodes) == nbox_before:
            break
        nlvl += 1
        lvp.append(len(nodes))
        l_by_level.append(next_l)

    lvp_arr = np.asarray(lvp, dtype=np.int64)
    l_arr = np.column_stack(l_by_level) if l_by_level else np.empty((d, 0))
    tree = HypOctTree(nlvl=nlvl, lvp=lvp_arr, l=l_arr, nodes=nodes)

    for lvl in range(1, nlvl):
        level_l = tree.l[:, lvl]
        for i in range(tree.lvp[lvl], tree.lvp[lvl + 1]):
            node = tree.nodes[i]
            if node.prnt is None:
                continue
            prnt = tree.nodes[node.prnt]
            node.nbor = [j for j in prnt.chld if j != i]

            for j in prnt.nbor:
                other = tree.nodes[j]
                if other.xi.size:
                    jlvl = int(np.searchsorted(tree.lvp, j, side="right") - 1)
                    jl = tree.l[:, jlvl]
                    dist = np.round((np.abs(node.ctr - other.ctr) - 0.5 * (level_l + jl)) / np.maximum(level_l, 1e-300))
                    if np.max(dist) <= 0:
                        node.nbor.append(j)

            candidates: list[int] = []
            for j in prnt.nbor:
                candidates.extend(tree.nodes[j].chld)
            if candidates:
                ctrs = np.column_stack([tree.nodes[j].ctr for j in candidates])
                dist = np.round(np.abs(node.ctr[:, None] - ctrs) / np.maximum(level_l[:, None], 1e-300))
                node.nbor.extend([candidates[k] for k in np.flatnonzero(np.max(dist, axis=0) <= 1)])

    return tree


def hypoct_perm(t: HypOctTree) -> np.ndarray:
    """Return FLAM's natural preorder tree permutation."""

    total = sum(node.xi.size for node in t.nodes)
    perm = np.empty(total, dtype=np.int64)
    n = 0
    stack = [0]
    while stack:
        i = stack.pop()
        xi = t.nodes[i].xi
        if xi.size:
            perm[n : n + xi.size] = xi
            n += xi.size
        stack.extend(t.nodes[i].chld)
    return perm[:n]


def id(
    A: ArrayLike,
    rank_or_tol: float,
    Tmax: float = 2,
    rrqr_iter: float = np.inf,
    fixed: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Interpolative decomposition of matrix columns.

    The implementation uses SciPy's pivoted QR for the primary decomposition.
    It preserves FLAM's return contract but does not perform Gu-Eisenstat
    refinement iterations; ``rrqr_iter`` is accepted for API compatibility.
    """

    if rank_or_tol < 0:
        raise ValueError("rank or tolerance must be nonnegative")
    if Tmax < 1:
        raise ValueError("interpolation matrix entry bound must be >= 1")
    if rrqr_iter < 0:
        raise ValueError("maximum RRQR iterations must be nonnegative")

    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")
    m, n = A.shape
    fixed_arr = np.asarray(list(fixed) if fixed is not None else [], dtype=np.int64)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.zeros((0, 0), dtype=A.dtype)
    if m == 0:
        return np.array([], dtype=np.int64), np.arange(n, dtype=np.int64), np.zeros((0, n), dtype=A.dtype)

    tol = rank_or_tol % 1
    kmax = int(np.floor(rank_or_tol))
    if kmax == 0 or kmax > n:
        kmax = n

    if fixed_arr.size:
        fixed_arr = np.unique(fixed_arr)
        free_mask = np.ones(n, dtype=bool)
        free_mask[fixed_arr] = False
        free = np.flatnonzero(free_mask)
        if fixed_arr.size >= kmax:
            sk = fixed_arr[:kmax]
            rd = np.setdiff1d(np.arange(n), sk, assume_unique=False)
            T = la.lstsq(A[:, sk], A[:, rd])[0] if sk.size and rd.size else np.zeros((sk.size, rd.size), dtype=A.dtype)
            return sk, rd, T
        Q, _ = la.qr(A[:, fixed_arr], mode="economic")
        residual = A[:, free] - Q @ (Q.conj().T @ A[:, free])
        sk_free, rd_free, T_free = id(residual, max(kmax - fixed_arr.size, tol), Tmax, rrqr_iter)  # type: ignore[misc]
        sk = np.concatenate((fixed_arr, free[sk_free]))
        rd = free[rd_free]
        if rd.size:
            T = la.lstsq(A[:, sk], A[:, rd])[0]
        else:
            T = np.zeros((sk.size, 0), dtype=A.dtype)
        return sk, rd, T

    _, R, piv = la.qr(A, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R)) if R.ndim == 2 else np.asarray([abs(R[0])])
    scale = diag[0] if diag.size else 0.0
    k_tol = int(np.count_nonzero(diag > tol * scale))
    k = min(k_tol, kmax)
    if k == n:
        return piv[:k].astype(np.int64), np.array([], dtype=np.int64), np.zeros((k, 0), dtype=A.dtype)
    if k == 0:
        return np.array([], dtype=np.int64), piv.astype(np.int64), np.zeros((0, n), dtype=A.dtype)

    T = la.solve_triangular(R[:k, :k], R[:k, k:], lower=False)
    sk = piv[:k].astype(np.int64)
    rd = piv[k:].astype(np.int64)
    return sk, rd, T


def snorm(
    n: int,
    mv: Callable[[np.ndarray], np.ndarray],
    mva: Callable[[np.ndarray], np.ndarray] | None = None,
    tol: float = 1e-6,
    herm: bool = False,
    niter_max: int = 100,
) -> tuple[float, int]:
    """Estimate spectral norm by a power iteration."""

    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 1))
    x /= np.linalg.norm(x)
    s_old = 0.0
    for it in range(1, niter_max + 1):
        y = mv(x)
        z = mv(y) if herm else (mva(y) if mva is not None else mv(y))
        norm_z = np.linalg.norm(z)
        if norm_z == 0:
            return 0.0, it
        x = z / norm_z
        s = float(np.sqrt(abs(np.vdot(x, z))))
        if it > 1 and abs(s - s_old) <= tol * max(s, 1.0):
            return s, it
        s_old = s
    warnings.warn("maximum power iterations reached", RuntimeWarning)
    return s_old, niter_max


def spget(A: ArrayLike, I: ArrayLike, J: ArrayLike) -> np.ndarray:
    """Return ``full(A[I, J])`` for sparse or dense matrices."""

    I = np.asarray(I, dtype=np.int64)
    J = np.asarray(J, dtype=np.int64)
    if sp.issparse(A):
        return np.asarray(A[:, J][I, :].toarray())
    return np.asarray(A)[np.ix_(I, J)]


def spgetv(A: list[Any], I: ArrayLike, J: ArrayLike) -> np.ndarray:
    """Sparse column-list access equivalent to ``M[I, J]``."""

    I = np.asarray(I, dtype=np.int64)
    J = np.asarray(J, dtype=np.int64)
    if not J.size:
        return np.zeros((I.size, 0))
    cols = []
    for j in J:
        col = A[int(j)]
        if sp.issparse(col):
            cols.append(np.asarray(col[I].toarray()).reshape(-1))
        else:
            cols.append(np.asarray(col)[I].reshape(-1))
    return np.column_stack(cols)


def spaddv(A: list[Any], I: ArrayLike, J: ArrayLike, V: ArrayLike) -> list[Any]:
    """Add ``V`` into sparse column-list storage at rows ``I`` and columns ``J``."""

    I = np.asarray(I, dtype=np.int64)
    J = np.asarray(J, dtype=np.int64)
    V = np.asarray(V)
    if V.shape != (I.size, J.size):
        raise ValueError("V must have shape (len(I), len(J))")
    for col_pos, j in enumerate(J):
        jj = int(j)
        col = A[jj].tolil(copy=True) if sp.issparse(A[jj]) else sp.lil_matrix(np.asarray(A[jj]).reshape(-1, 1))
        for row_pos, i in enumerate(I):
            col[int(i), 0] = col[int(i), 0] + V[row_pos, col_pos]
        A[jj] = col.tocsc()
    return A


def sppush2(I: ArrayLike, J: ArrayLike, nz: int, i: ArrayLike, j: ArrayLike) -> tuple[np.ndarray, np.ndarray, int]:
    I = np.asarray(I)
    J = np.asarray(J)
    i = np.asarray(i).reshape(-1)
    j = np.asarray(j).reshape(-1)
    if I.size != J.size or i.size != j.size:
        raise ValueError("arrays I and J must have the same size, as must i and j")
    nznew = nz + i.size
    if I.size < nznew:
        new_size = max(1, I.size)
        while new_size < nznew:
            new_size *= 2
        I = np.pad(I, (0, new_size - I.size))
        J = np.pad(J, (0, new_size - J.size))
    I[nz:nznew] = i
    J[nz:nznew] = j
    return I, J, nznew


def sppush3(
    I: ArrayLike,
    J: ArrayLike,
    V: ArrayLike,
    nz: int,
    i: ArrayLike,
    j: ArrayLike,
    v: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    I = np.asarray(I)
    J = np.asarray(J)
    V = np.asarray(V)
    i = np.asarray(i).reshape(-1)
    j = np.asarray(j).reshape(-1)
    v = np.asarray(v).reshape(-1)
    if I.size != J.size or I.size != V.size or i.size != j.size or i.size != v.size:
        raise ValueError("arrays I, J, and V must have compatible sizes")
    nznew = nz + i.size
    if I.size < nznew:
        new_size = max(1, I.size)
        while new_size < nznew:
            new_size *= 2
        I = np.pad(I, (0, new_size - I.size))
        J = np.pad(J, (0, new_size - J.size))
        V = np.pad(V, (0, new_size - V.size))
    I[nz:nznew] = i
    J[nz:nznew] = j
    V[nz:nznew] = v
    return I, J, V, nznew


def spsymm(A: ArrayLike, symm: str) -> Any:
    """Recover a full sparse/dense symmetric matrix from compact storage."""

    symm = chksymm(symm)
    if symm == "n":
        return A
    if sp.issparse(A):
        D = sp.diags(A.diagonal(), format=A.format)
        return A + (A.T if symm == "s" else A.conj().T) - D
    arr = np.asarray(A)
    D = np.diag(np.diag(arr))
    return arr + (arr.T if symm == "s" else arr.conj().T) - D


def spsymm2(A: ArrayLike, B: ArrayLike, symm: str) -> tuple[Any, Any]:
    """Symmetrize paired off-diagonal sparse/dense blocks."""

    symm = chksymm(symm)
    if symm == "n":
        return A, B
    Bt = B.T if symm == "s" else B.conj().T
    A2 = A + Bt
    B2 = A2.T if symm == "s" else A2.conj().T
    return A2, B2


def detperm(p: ArrayLike) -> int:
    """Return the sign of a 0-based permutation vector."""

    p = np.asarray(p, dtype=np.int64).reshape(-1)
    if np.sort(p).tolist() != list(range(p.size)):
        raise ValueError("p must be a 0-based permutation")
    seen = np.zeros(p.size, dtype=bool)
    cycles = 0
    for start in range(p.size):
        if seen[start]:
            continue
        cycles += 1
        j = start
        while not seen[j]:
            seen[j] = True
            j = p[j]
    return -1 if (p.size - cycles) % 2 else 1


def ismemb(A: ArrayLike, S: ArrayLike) -> np.ndarray:
    """Return boolean membership of ``A`` in sorted or unsorted set ``S``."""

    return np.isin(A, S)


def logdet_ldl(D: ArrayLike) -> complex:
    """Compute log determinant of a block diagonal LDL ``D`` factor."""

    arr = np.asarray(D)
    if arr.ndim == 1:
        return np.sum(np.log(arr))
    nonzero_per_col = np.count_nonzero(arr, axis=0)
    one_by_one = nonzero_per_col == 1
    diag = np.diag(arr)
    ld = np.sum(np.log(diag[one_by_one]))
    idx = np.flatnonzero(~one_by_one)
    for k in range(0, idx.size, 2):
        block_idx = idx[k : k + 2]
        if block_idx.size == 2:
            block = arr[np.ix_(block_idx, block_idx)]
            ld = ld + np.log(block[0, 0] * block[1, 1] - block[0, 1] * block[1, 0])
    return ld


__all__ = [
    "HypOctNode",
    "HypOctTree",
    "StructMixin",
    "_normalise_opts",
    "chktrans",
    "chksymm",
    "hypoct",
    "hypoct_perm",
    "id",
    "detperm",
    "ismemb",
    "logdet_ldl",
    "snorm",
    "spaddv",
    "spget",
    "spgetv",
    "sppush2",
    "sppush3",
    "spsymm",
    "spsymm2",
]
