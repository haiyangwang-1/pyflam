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
    widths: np.ndarray
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
    """Normalize a FLAM symmetry option to ``n``, ``s``, ``h``, or ``p``."""

    if symm is None or symm == "":
        return "n"
    val = str(symm).lower()[0]
    if val not in {"n", "s", "h", "p"}:
        raise ValueError("symmetry parameter must be one of 'N', 'S', 'H', or 'P'")
    return val


def chktrans(trans: str | None) -> str:
    """Normalize a transpose option to ``n``, ``t``, or ``c``."""

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

    root_widths = ext_arr[:, 1] - ext_arr[:, 0]
    ctr = 0.5 * (ext_arr[:, 0] + ext_arr[:, 1])
    nodes: list[HypOctNode] = [HypOctNode(ctr=ctr, xi=np.arange(n, dtype=np.int64))]
    lvp = [0, 1]
    widths_by_level = [root_widths.copy()]
    nlvl = 1

    while nlvl < lvlmax:
        nbox_before = len(nodes)
        current_widths = widths_by_level[-1].copy()
        max_width = np.max(current_widths) if current_widths.size else 0
        split_dims = current_widths >= max_width / np.sqrt(2) if max_width > 0 else np.zeros(d, dtype=bool)
        next_widths = current_widths.copy()
        next_widths[split_dims] *= 0.5

        for prnt in range(lvp[nlvl - 1], lvp[nlvl]):
            xi = nodes[prnt].xi
            if xi.size <= occ:
                continue
            if _unique_point_count(x[:, xi]) <= 1:
                continue
            pctr = nodes[prnt].ctr
            side = (split_dims[:, None] & (x[:, xi] > pctr[:, None])).astype(np.int64)
            for bit, mask in _child_partitions(side):
                child_ctr = pctr + next_widths * split_dims * (bit - 0.5)
                child_xi = xi[mask]
                child_idx = len(nodes)
                nodes.append(HypOctNode(ctr=child_ctr, xi=child_xi, prnt=prnt))
                nodes[prnt].chld.append(child_idx)
            nodes[prnt].xi = np.array([], dtype=np.int64)

        if len(nodes) == nbox_before:
            break
        nlvl += 1
        lvp.append(len(nodes))
        widths_by_level.append(next_widths)

    lvp_arr = np.asarray(lvp, dtype=np.int64)
    widths_arr = np.column_stack(widths_by_level) if widths_by_level else np.empty((d, 0))
    tree = HypOctTree(nlvl=nlvl, lvp=lvp_arr, widths=widths_arr, nodes=nodes)

    for lvl in range(1, nlvl):
        level_widths = tree.widths[:, lvl]
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
                    other_widths = tree.widths[:, jlvl]
                    dist = np.round(
                        (np.abs(node.ctr - other.ctr) - 0.5 * (level_widths + other_widths))
                        / np.maximum(level_widths, 1e-300)
                    )
                    if np.max(dist) <= 0:
                        node.nbor.append(j)

            candidates: list[int] = []
            for j in prnt.nbor:
                candidates.extend(tree.nodes[j].chld)
            if candidates:
                ctrs = np.column_stack([tree.nodes[j].ctr for j in candidates])
                dist = np.round(np.abs(node.ctr[:, None] - ctrs) / np.maximum(level_widths[:, None], 1e-300))
                node.nbor.extend([candidates[k] for k in np.flatnonzero(np.max(dist, axis=0) <= 1)])

    return tree


def _unique_point_count(x: np.ndarray) -> int:
    if x.size == 0:
        return 0
    return np.unique(x, axis=1).shape[1]


def _child_partitions(side: np.ndarray):
    d, n = side.shape
    if d <= 63:
        child_code = np.sum(side * (2 ** np.arange(d, dtype=np.int64))[:, None], axis=0)
        for code in np.unique(child_code):
            bit = np.array([(int(code) >> k) & 1 for k in range(d)], dtype=side.dtype)
            yield bit, child_code == code
        return

    codes = []
    for j in range(n):
        code = 0
        for k in np.flatnonzero(side[:, j]):
            code |= 1 << int(k)
        codes.append(code)
    for code in sorted(set(codes)):
        bit = np.array([(code >> k) & 1 for k in range(d)], dtype=side.dtype)
        yield bit, np.fromiter((value == code for value in codes), dtype=bool, count=n)


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
    return_niter: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Interpolative decomposition of matrix columns.

    This follows FLAM's pivoted QR and RRQR-refinement control flow, including
    fixed-column preprocessing.  The returned indices are 0-based.
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
        out = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.zeros((0, 0), dtype=A.dtype))
        return (*out, 0) if return_niter else out
    if m == 0:
        out = (np.array([], dtype=np.int64), np.arange(n, dtype=np.int64), np.zeros((0, n), dtype=A.dtype))
        return (*out, 0) if return_niter else out

    tol = rank_or_tol % 1
    kmax = int(np.floor(rank_or_tol))
    if kmax == 0 or kmax > n:
        kmax = n
    niter = 0

    if fixed_arr.size:
        if np.any(fixed_arr < 0) or np.any(fixed_arr >= n):
            raise IndexError("fixed column index out of range")
        if np.unique(fixed_arr).size != fixed_arr.size:
            raise ValueError("fixed column indices must be unique")
        free_mask = np.ones(n, dtype=bool)
        free_mask[fixed_arr] = False
        free = np.flatnonzero(free_mask)
        if free.size == 0:
            out = (fixed_arr.copy(), np.array([], dtype=np.int64), np.zeros((fixed_arr.size, 0), dtype=A.dtype))
            return (*out, niter) if return_niter else out

        Afix = A[:, fixed_arr]
        cmax = float(np.sqrt(np.max(np.sum(np.abs(Afix) ** 2, axis=0)))) if Afix.size else 0.0
        Q, R1 = la.qr(Afix, mode="economic", check_finite=False)
        Afree = A[:, free]
        R2 = Q.conj().T @ Afree
        Awork = Afree - Q @ R2
        kmax = max(kmax - fixed_arr.size, 0)
    else:
        free = np.arange(n, dtype=np.int64)
        cmax = 0.0
        R1 = np.zeros((0, 0), dtype=A.dtype)
        R2 = np.zeros((0, n), dtype=A.dtype)
        Awork = A

    mw, nw = Awork.shape
    if mw > 8 * nw:
        _, Awork = la.qr(Awork, mode="economic", check_finite=False)

    _, R, piv = la.qr(Awork, mode="economic", pivoting=True, check_finite=False)
    R = np.asarray(R)
    if R.size:
        cmax = max(cmax, float(abs(R.flat[0])))
    atol = cmax * tol
    diagR = np.diag(R) if R.ndim == 2 else np.asarray([R.flat[0]])
    k_prec = int(np.count_nonzero(np.abs(diagR) > atol))
    R = R[:k_prec, :].copy()
    k = min(k_prec, kmax)
    if k > 0 and k < nw:
        R[:k, k:] = la.solve_triangular(R[:k, :k], R[:k, k:], lower=False, check_finite=False)

    if np.isfinite(Tmax) and rrqr_iter > 0 and k > 0 and k < nw:
        R, piv, k, niter = _rrqr_refine(R, piv.astype(np.int64), k, nw, atol, Tmax, rrqr_iter)

    sk = piv[:k].astype(np.int64)
    rd = piv[k:].astype(np.int64)
    T = R[:k, k:].copy()

    if fixed_arr.size:
        if rd.size:
            top = la.solve_triangular(
                R1,
                R2[:, rd] - (R2[:, sk] @ T if sk.size else 0),
                lower=False,
                check_finite=False,
            )
        else:
            top = np.zeros((fixed_arr.size, 0), dtype=A.dtype)
        T = np.vstack((top, T))
        sk = np.concatenate((fixed_arr, free[sk]))
        rd = free[rd]

    out = (sk, rd, T)
    return (*out, niter) if return_niter else out


def _rrqr_refine(
    R: np.ndarray,
    piv: np.ndarray,
    k: int,
    n: int,
    atol: float,
    Tmax: float,
    rrqr_iter: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    f2 = Tmax**2
    c2 = _residual_col_norms(R, k, n)
    r2 = _inverse_row_norms(R[:k, :k])
    niter = 0
    conv = False
    eye_dtype = np.result_type(R, complex if np.iscomplexobj(R) else float)

    while niter < rrqr_iter and k > 0 and k < n:
        tmp = np.abs(R[:k, k:]) ** 2 + r2[:, None] * c2[None, :]
        idx = int(np.argmax(tmp, axis=None))
        t2 = float(tmp.reshape(-1)[idx])
        if t2 <= f2:
            conv = True
            break
        niter += 1
        i, j = np.unravel_index(idx, tmp.shape)

        if i < k - 1:
            _swap(piv, i, k - 1)
            u_qr = R[:k, k - 1] - R[:k, i]
            v_qr = np.zeros(k, dtype=R.dtype)
            v_qr[i] = 1
            v_qr[k - 1] = -1
            _, R11 = la.qr_update(
                np.eye(k, dtype=eye_dtype),
                R[:k, :k],
                u_qr,
                v_qr,
                check_finite=False,
            )
            R[:k, :k] = R11
            R[[i, k - 1], k:] = R[[k - 1, i], k:]
            _swap(r2, i, k - 1)

        if j > 0:
            _swap(piv, k, k + j)
            R[:, [k, k + j]] = R[:, [k + j, k]]
            _swap(c2, 0, j)

        if k > 1:
            v = la.solve_triangular(R[: k - 1, : k - 1], R[: k - 1, k - 1], check_finite=False)
            r2[: k - 1] = r2[: k - 1] - np.abs(v / R[k - 1, k - 1]) ** 2
        r2[k - 1] = 0

        if R.shape[0] == k:
            u = R[:, k].copy()
            R[:, k] = R[:, :k] @ R[:, k]
            R[: k - 1, k - 1] = 0
            R[k - 1, k - 1] = 1
        else:
            hh = R[k:, k].copy()
            phase = np.exp(1j * np.angle(R[k, k])) if np.iscomplexobj(R) else (1.0 if R[k, k] >= 0 else -1.0)
            hh[0] = hh[0] + np.sqrt(max(c2[0], 0.0)) * phase
            nhh = np.linalg.norm(hh)
            if nhh:
                hh = hh / nhh
                R[k:, k + 1 :] = R[k:, k + 1 :] - 2 * np.outer(hh, hh.conj() @ R[k:, k + 1 :])
            R[k + 1 :, k] = 0
            R[k, k] = np.sqrt(max(c2[0], 0.0))

            R[:k, k] = R[:k, :k] @ R[:k, k]
            R[k - 1, k + 1 :] = R[k - 1, k - 1] * R[k - 1, k + 1 :]
            r = R[:k, k - 1].copy()
            s = R[k - 1, k + 1 :].copy()

            c2 = c2 - np.abs(R[k, k:]) ** 2
            G = _givens(R[k - 1, k], R[k, k])
            R[k - 1 : k + 1, k - 1 :] = G @ R[k - 1 : k + 1, k - 1 :]
            c2 = c2 + np.abs(R[k, k:]) ** 2
            c2[0] = np.abs(R[k, k - 1]) ** 2

            tmp_diag = R[k - 1, k - 1]
            R[k - 1, k - 1] = r[k - 1]
            u = la.solve_triangular(R[:k, :k], R[:k, k], check_finite=False)
            R[k - 1, k - 1] = tmp_diag

            if k > 1:
                v = la.solve_triangular(R[: k - 1, : k - 1], r[: k - 1], check_finite=False)
                R[k - 1, k - 1] = R[k - 1, k - 1] / r[k - 1]
                R[: k - 1, k - 1] = v * (1 - R[k - 1, k - 1])
                R[: k - 1, k + 1 :] = R[: k - 1, k + 1 :] + (v / r[k - 1])[:, None] * (
                    s - R[k - 1, k + 1 :]
                )
            else:
                R[k - 1, k - 1] = R[k - 1, k - 1] / r[k - 1]
            R[k - 1, k + 1 :] = R[k - 1, k + 1 :] / r[k - 1]

        _swap(piv, k - 1, k)
        R[:, [k - 1, k]] = R[:, [k, k - 1]]
        if k > 1:
            v = la.solve_triangular(R[: k - 1, : k - 1], R[: k - 1, k - 1], check_finite=False)
            r2[: k - 1] = r2[: k - 1] + np.abs(v / R[k - 1, k - 1]) ** 2
        r2[k - 1] = 1 / np.abs(R[k - 1, k - 1]) ** 2

        u[k - 1] = u[k - 1] - 1
        R[:k, k:] = R[:k, k:] - np.outer(u, R[k - 1, k:]) / (1 + u[k - 1])

        rvals = np.full_like(r2, np.inf, dtype=float)
        nz = r2 > 0
        rvals[nz] = 1 / np.sqrt(r2[nz])
        drop = int(np.argmin(rvals))
        if rvals[drop] > atol:
            continue

        if drop < k - 1:
            _swap(piv, drop, k - 1)
            u_qr = R[:k, k - 1] - R[:k, drop]
            v_qr = np.zeros(k, dtype=R.dtype)
            v_qr[drop] = 1
            v_qr[k - 1] = -1
            _, R11 = la.qr_update(
                np.eye(k, dtype=eye_dtype),
                R[:k, :k],
                u_qr,
                v_qr,
                check_finite=False,
            )
            R[:k, :k] = R11
            R[[drop, k - 1], k:] = R[[k - 1, drop], k:]
            _swap(r2, drop, k - 1)

        c2 = np.zeros(n - k + 1, dtype=float)
        if k > 1:
            v = la.solve_triangular(R[: k - 1, : k - 1], R[: k - 1, k - 1], check_finite=False)
            r2 = r2[: k - 1] - np.abs(v / R[k - 1, k - 1]) ** 2
        else:
            r2 = np.zeros(0, dtype=float)
        k -= 1
        if k == 0:
            R = R[:0, :]
            break
        r = la.solve_triangular(R[:k, :k], R[:k, k], check_finite=False)
        tail = R[:k, k + 1 :] + r[:, None] * R[k, k + 1 :]
        R[:k, k:] = np.column_stack((r, tail))
        R = R[:k, :]

    if not conv and k > 0 and k < n:
        warnings.warn("maximum RRQR iterations reached", RuntimeWarning)
    return R, piv, k, niter


def _residual_col_norms(R: np.ndarray, k: int, n: int) -> np.ndarray:
    if R.shape[0] <= k:
        return np.zeros(n - k, dtype=float)
    return np.sum(np.abs(R[k:, k:]) ** 2, axis=0)


def _inverse_row_norms(R: np.ndarray) -> np.ndarray:
    if R.size == 0:
        return np.zeros(0, dtype=float)
    invR = la.solve_triangular(R, np.eye(R.shape[0], dtype=R.dtype), check_finite=False)
    return np.sum(np.abs(invR) ** 2, axis=1)


def _givens(a, b) -> np.ndarray:
    if b == 0:
        c = 1.0
        s = 0.0
    elif a == 0:
        c = 0.0
        s = np.conj(b) / abs(b)
    else:
        scale = abs(a) + abs(b)
        norm = scale * np.sqrt((abs(a) / scale) ** 2 + (abs(b) / scale) ** 2)
        alpha = a / abs(a)
        c = abs(a) / norm
        s = alpha * np.conj(b) / norm
    return np.array([[c, s], [-np.conj(s), c]], dtype=np.result_type(a, b, complex)) if np.iscomplexobj([a, b]) else np.array([[c, s], [-s, c]])


def _swap(arr: np.ndarray, i: int, j: int) -> None:
    if i != j:
        arr[[i, j]] = arr[[j, i]]


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


def spget(A: ArrayLike, rows: ArrayLike, cols: ArrayLike) -> np.ndarray:
    """Return ``full(A[rows, cols])`` for sparse or dense matrices."""

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if sp.issparse(A):
        return np.asarray(A[:, cols][rows, :].toarray())
    return np.asarray(A)[np.ix_(rows, cols)]


def spgetv(A: list[Any], rows: ArrayLike, cols: ArrayLike) -> np.ndarray:
    """Sparse column-list access equivalent to ``M[rows, cols]``."""

    rows = np.asarray(rows, dtype=np.int64)
    col_indices = np.asarray(cols, dtype=np.int64)
    if not col_indices.size:
        return np.zeros((rows.size, 0))
    out_cols = []
    for col_idx in col_indices:
        col = A[int(col_idx)]
        if sp.issparse(col):
            out_cols.append(np.asarray(col[rows].toarray()).reshape(-1))
        else:
            out_cols.append(np.asarray(col)[rows].reshape(-1))
    return np.column_stack(out_cols)


def spaddv(A: list[Any], rows: ArrayLike, cols: ArrayLike, values: ArrayLike) -> list[Any]:
    """Add ``values`` into sparse column-list storage at selected rows and columns."""

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    values = np.asarray(values)
    if values.shape != (rows.size, cols.size):
        raise ValueError("values must have shape (len(rows), len(cols))")
    for col_pos, col_idx in enumerate(cols):
        column = A[int(col_idx)]
        column = column.tolil(copy=True) if sp.issparse(column) else sp.lil_matrix(np.asarray(column).reshape(-1, 1))
        for row_pos, row_idx in enumerate(rows):
            column[int(row_idx), 0] = column[int(row_idx), 0] + values[row_pos, col_pos]
        A[int(col_idx)] = column.tocsc()
    return A


def sppush2(
    row_buffer: ArrayLike,
    col_buffer: ArrayLike,
    nz: int,
    rows: ArrayLike,
    cols: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Append row/column index pairs to expandable sparse COO buffers."""

    row_buffer = np.asarray(row_buffer)
    col_buffer = np.asarray(col_buffer)
    rows = np.asarray(rows).reshape(-1)
    cols = np.asarray(cols).reshape(-1)
    if row_buffer.size != col_buffer.size or rows.size != cols.size:
        raise ValueError("row and column buffers must have compatible sizes")
    nznew = nz + rows.size
    if row_buffer.size < nznew:
        new_size = max(1, row_buffer.size)
        while new_size < nznew:
            new_size *= 2
        row_buffer = np.pad(row_buffer, (0, new_size - row_buffer.size))
        col_buffer = np.pad(col_buffer, (0, new_size - col_buffer.size))
    row_buffer[nz:nznew] = rows
    col_buffer[nz:nznew] = cols
    return row_buffer, col_buffer, nznew


def sppush3(
    row_buffer: ArrayLike,
    col_buffer: ArrayLike,
    value_buffer: ArrayLike,
    nz: int,
    rows: ArrayLike,
    cols: ArrayLike,
    values: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Append row/column/value triples to expandable sparse COO buffers."""

    row_buffer = np.asarray(row_buffer)
    col_buffer = np.asarray(col_buffer)
    value_buffer = np.asarray(value_buffer)
    rows = np.asarray(rows).reshape(-1)
    cols = np.asarray(cols).reshape(-1)
    values = np.asarray(values).reshape(-1)
    if (
        row_buffer.size != col_buffer.size
        or row_buffer.size != value_buffer.size
        or rows.size != cols.size
        or rows.size != values.size
    ):
        raise ValueError("row, column, and value buffers must have compatible sizes")
    nznew = nz + rows.size
    if row_buffer.size < nznew:
        new_size = max(1, row_buffer.size)
        while new_size < nznew:
            new_size *= 2
        row_buffer = np.pad(row_buffer, (0, new_size - row_buffer.size))
        col_buffer = np.pad(col_buffer, (0, new_size - col_buffer.size))
        value_buffer = np.pad(value_buffer, (0, new_size - value_buffer.size))
    row_buffer[nz:nznew] = rows
    col_buffer[nz:nznew] = cols
    value_buffer[nz:nznew] = values
    return row_buffer, col_buffer, value_buffer, nznew


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
