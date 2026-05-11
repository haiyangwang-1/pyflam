"""Hierarchical interpolative factorization for integral equations.

The public HIFIE routines share the same apply/solve/logdet surface as
``rskelf``.  This module currently routes through the dense recursive
skeletonization backend, giving exact operation semantics for the MATLAB-facing
API while preserving the HIFIE entry points for future dimensional-reduction
kernels.
"""

from __future__ import annotations

from typing import Any

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
