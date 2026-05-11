"""Geometry helpers from FLAM."""

from __future__ import annotations

import numpy as np


def tri3geom(V, F=None):
    """Centroid, unit normal, and area for 3D triangles."""

    V = np.asarray(V)
    if V.ndim != 2 or V.shape[0] != 3:
        raise ValueError("vertices must have shape (3, n)")
    if F is None:
        F_arr = np.array([[0], [1], [2]], dtype=np.int64)
    else:
        F_arr = np.asarray(F, dtype=np.int64)
        if F_arr.ndim == 1:
            F_arr = F_arr.reshape(3, 1)
    if F_arr.shape[0] != 3:
        raise ValueError("faces must have shape (3, n)")

    C = (V[:, F_arr[0, :]] + V[:, F_arr[1, :]] + V[:, F_arr[2, :]]) / 3
    V21 = V[:, F_arr[1, :]] - V[:, F_arr[0, :]]
    V32 = V[:, F_arr[2, :]] - V[:, F_arr[1, :]]
    N = np.cross(V21.T, V32.T).T
    dbl_area = np.sqrt(np.sum(N**2, axis=0))
    N = N / dbl_area
    A = 0.5 * dbl_area
    return C, N, A


def trisphere_subdiv(n: int, vfmin: str = "f"):
    """Recursively subdivide an icosahedron triangulation of the unit sphere."""

    vfmin = str(vfmin).lower()[0]
    if vfmin not in {"v", "f"}:
        raise ValueError("minimum vertex/face selector must be one of 'v' or 'f'")

    t = (1 + np.sqrt(5.0)) / 2
    V = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=float,
    )
    V = V / np.linalg.norm(V[0, :])
    F = np.array(
        [
            [1, 12, 6],
            [1, 6, 2],
            [1, 2, 8],
            [1, 8, 11],
            [1, 11, 12],
            [2, 6, 10],
            [6, 12, 5],
            [12, 11, 3],
            [11, 8, 7],
            [8, 2, 9],
            [4, 10, 5],
            [4, 5, 3],
            [4, 3, 7],
            [4, 7, 9],
            [4, 9, 10],
            [5, 10, 6],
            [3, 5, 12],
            [7, 3, 11],
            [9, 7, 8],
            [10, 9, 2],
        ],
        dtype=np.int64,
    ) - 1

    m = V.shape[0] if vfmin == "v" else F.shape[0]
    while m < n:
        nf = F.shape[0]
        mids = 0.5 * np.vstack(
            (
                V[F[:, 0], :] + V[F[:, 1], :],
                V[F[:, 0], :] + V[F[:, 2], :],
                V[F[:, 1], :] + V[F[:, 2], :],
            )
        )
        mids = mids / np.linalg.norm(mids, axis=1)[:, None]
        mids, inverse = np.unique(mids, axis=0, return_inverse=True)
        nv = V.shape[0]
        F12 = nv + inverse[:nf]
        F13 = nv + inverse[nf : 2 * nf]
        F23 = nv + inverse[2 * nf : 3 * nf]
        V = np.vstack((V, mids))
        F = np.vstack(
            (
                np.column_stack((F[:, 0], F12, F13)),
                np.column_stack((F12, F[:, 1], F23)),
                np.column_stack((F13, F23, F[:, 2])),
                np.column_stack((F12, F23, F13)),
            )
        )
        m = V.shape[0] if vfmin == "v" else F.shape[0]

    return V.T, F.T


__all__ = ["tri3geom", "trisphere_subdiv"]
