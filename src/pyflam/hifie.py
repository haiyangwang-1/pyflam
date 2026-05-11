"""Hierarchical interpolative factorization for integral equations.

The public HIFIE routines share the same apply/solve/logdet surface as
``rskelf``.  This module currently routes through the dense recursive
skeletonization backend, giving exact operation semantics for the MATLAB-facing
API while preserving the HIFIE entry points for future dimensional-reduction
kernels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import id, snorm
from .rskelf import (
    RSkelFFactor as HIFIEFactor,
    rskelf,
    rskelf_cholmv,
    rskelf_cholsv,
    rskelf_diag,
    rskelf_logdet,
    rskelf_mv,
    rskelf_spdiag,
    rskelf_sv,
)


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
    return rskelf_mv(F, X, trans)


def hifie_sv(F: HIFIEFactor, X, trans: str = "n"):
    return rskelf_sv(F, X, trans)


def hifie_logdet(F: HIFIEFactor):
    return rskelf_logdet(F)


def hifie_cholmv(F: HIFIEFactor, X, trans: str = "n"):
    return rskelf_cholmv(F, X, trans)


def hifie_cholsv(F: HIFIEFactor, X, trans: str = "n"):
    return rskelf_cholsv(F, X, trans)


def hifie_diag(F: HIFIEFactor, dinv: bool | int = False, opts: dict[str, Any] | None = None):
    return rskelf_diag(F, dinv, opts)


def hifie_spdiag(F: HIFIEFactor, dinv: bool | int = False):
    return rskelf_spdiag(F, dinv)


def _hifie(A, x, occ, rank_or_tol, pxyfun, opts, variant: str) -> HIFIEFactor:
    F = rskelf(A, x, occ, rank_or_tol, pxyfun=pxyfun, opts=opts)
    F.opts = dict(F.opts)
    F.opts["hifie_variant"] = variant
    return F


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
