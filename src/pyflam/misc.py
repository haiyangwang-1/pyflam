"""Miscellaneous FLAM helper routines."""

from __future__ import annotations

import warnings

import numpy as np


def gausspdf(x, mu=0, sigma=1):
    """Normal probability density function."""

    x = np.asarray(x)
    sigma = np.asarray(sigma)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def lsedc(lsfun, A, B, C, D, tau, tol: float = 1e-12, niter_max: int = 8):
    """Equality-constrained least squares via deferred correction."""

    if tol < 0:
        raise ValueError("tolerance must be nonnegative")
    if niter_max < 0:
        raise ValueError("maximum number of iterations must be nonnegative")

    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)

    x = lsfun(np.vstack((tau * D, B)))
    w = D - C @ x
    if np.linalg.norm(w) <= tol:
        return x, w, 0

    r = B - A @ x
    lam = tau * w
    for niter in range(1, int(niter_max) + 1):
        dx = lsfun(np.vstack((tau * w + lam, r)))
        x = x + dx
        w = w - C @ dx
        if np.linalg.norm(w) <= tol:
            return x, w, niter
        r = r - A @ dx
        lam = lam + tau * w

    warnings.warn("maximum deferred-correction iterations reached", RuntimeWarning)
    return x, w, -1


__all__ = ["gausspdf", "lsedc"]
