"""Quadrature helper routines."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la


def gqgw(alpha, beta, mu):
    """Gaussian quadrature by the Golub-Welsch algorithm."""

    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    if alpha.ndim != 1:
        alpha = alpha.reshape(-1)
    if beta.ndim != 1:
        beta = beta.reshape(-1)
    if beta.size != max(alpha.size - 1, 0):
        raise ValueError("beta must have length len(alpha) - 1")
    x, V = la.eigh_tridiagonal(alpha, beta)
    w = mu * V[0, :] ** 2
    return x, w


def glegquad(n: int, a=-1, b=1):
    """Gauss-Legendre quadrature nodes and weights."""

    if n < 1:
        raise ValueError("quadrature order must be at least 1")
    alpha = np.zeros(int(n))
    k = np.arange(1, int(n), dtype=float)
    beta = 0.5 / np.sqrt(1 - (2 * k) ** -2)
    x, w = gqgw(alpha, beta, 2.0)
    x = 0.5 * ((b - a) * x + a + b)
    w = 0.5 * (b - a) * w
    return x, w


def quad_sqtri3(x, w, v):
    """Map a unit-square quadrature rule to a triangle in 3D."""

    x = np.array(x, copy=True)
    w = np.array(w, copy=True).reshape(-1)
    v = np.asarray(v)
    if x.shape[0] != 2:
        raise ValueError("square quadrature nodes must have shape (2, n)")
    if v.shape != (3, 3):
        raise ValueError("triangle vertices must have shape (3, 3)")
    if w.size != x.shape[1]:
        raise ValueError("weights must have one entry per node")

    x[1, :] = x[0, :] * x[1, :]
    w = w * x[0, :]

    A = np.column_stack((v[:, 1] - v[:, 0], v[:, 2] - v[:, 1]))
    x = A @ x + v[:, [0]]
    _, R = np.linalg.qr(A, mode="reduced")
    w = w * abs(np.linalg.det(R))
    return x, w


__all__ = ["glegquad", "gqgw", "quad_sqtri3"]
