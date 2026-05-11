"""Hierarchical interpolative factorization for differential equations."""

from __future__ import annotations

from typing import Any

from .core import _normalise_opts
from .mf import (
    MFFactor as HIFDEFactor,
    mf2,
    mf3,
    mf_cholmv,
    mf_cholsv,
    mf_diag,
    mf_logdet,
    mf_mv,
    mf_spdiag,
    mf_sv,
    mfx,
)


def hifde2(A, n: int, occ: int, rank_or_tol, opts: dict[str, Any] | None = None) -> HIFDEFactor:
    """Factor a sparse matrix on a regular 2D mesh using the HIFDE API."""

    o = _hifde_opts(opts, ext=False)
    F = mf2(A, n, occ, _mf_opts(o, ext=False))
    return _tag(F, "hifde2", rank_or_tol, o)


def hifde3(A, n: int, occ: int, rank_or_tol, opts: dict[str, Any] | None = None) -> HIFDEFactor:
    """Factor a sparse matrix on a regular 3D mesh using the HIFDE API."""

    o = _hifde_opts(opts, ext=False)
    F = mf3(A, n, occ, _mf_opts(o, ext=False))
    return _tag(F, "hifde3", rank_or_tol, o)


def hifde2x(A, x, occ: int, rank_or_tol, opts: dict[str, Any] | None = None) -> HIFDEFactor:
    """Point-cloud 2D HIFDE entry point."""

    o = _hifde_opts(opts, ext=True)
    F = mfx(A, x, occ, _mf_opts(o, ext=True))
    return _tag(F, "hifde2x", rank_or_tol, o)


def hifde3x(A, x, occ: int, rank_or_tol, opts: dict[str, Any] | None = None) -> HIFDEFactor:
    """Point-cloud 3D HIFDE entry point."""

    o = _hifde_opts(opts, ext=True)
    F = mfx(A, x, occ, _mf_opts(o, ext=True))
    return _tag(F, "hifde3x", rank_or_tol, o)


def hifde_mv(F: HIFDEFactor, X, trans: str = "n"):
    return mf_mv(F, X, trans)


def hifde_sv(F: HIFDEFactor, X, trans: str = "n"):
    return mf_sv(F, X, trans)


def hifde_logdet(F: HIFDEFactor):
    return mf_logdet(F)


def hifde_cholmv(F: HIFDEFactor, X, trans: str = "n"):
    return mf_cholmv(F, X, trans)


def hifde_cholsv(F: HIFDEFactor, X, trans: str = "n"):
    return mf_cholsv(F, X, trans)


def hifde_diag(F: HIFDEFactor, dinv: bool | int = False, opts: dict[str, Any] | None = None):
    return mf_diag(F, dinv, opts)


def hifde_spdiag(F: HIFDEFactor, dinv: bool | int = False):
    return mf_spdiag(F, dinv)


def _hifde_opts(opts: dict[str, Any] | None, ext: bool) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "lvlmax": float("inf"),
        "Tmax": 2,
        "rrqr_iter": float("inf"),
        "skip": 0,
        "symm": "n",
        "verb": 0,
    }
    if ext:
        defaults["ext"] = None
    return _normalise_opts(opts, defaults)


def _mf_opts(opts: dict[str, Any], ext: bool) -> dict[str, Any]:
    out = {"lvlmax": opts["lvlmax"], "symm": opts["symm"], "verb": opts["verb"]}
    if ext:
        out["ext"] = opts["ext"]
    return out


def _tag(F: HIFDEFactor, variant: str, rank_or_tol, opts: dict[str, Any]) -> HIFDEFactor:
    F.opts = dict(opts)
    F.opts["hifde_variant"] = variant
    F.opts["rank_or_tol"] = rank_or_tol
    return F


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
