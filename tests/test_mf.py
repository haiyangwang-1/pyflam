import unittest

import numpy as np
import scipy.sparse as sp

from pyflam import (
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


def _spd_tridiag(n):
    return sp.diags(
        [-np.ones(n - 1), 3.0 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csc",
    )


def _spd_grid2(n):
    nd = n - 1
    A = sp.lil_matrix((nd * nd, nd * nd))
    for j in range(nd):
        for i in range(nd):
            k = i + nd * j
            A[k, k] = 4.0
            if i > 0:
                A[k, k - 1] = -1.0
            if i + 1 < nd:
                A[k, k + 1] = -1.0
            if j > 0:
                A[k, k - nd] = -1.0
            if j + 1 < nd:
                A[k, k + nd] = -1.0
    return A.tocsc()


def _spd_grid3(n):
    nd = n - 1
    A = sp.lil_matrix((nd**3, nd**3))
    for k in range(nd):
        for j in range(nd):
            for i in range(nd):
                idx = i + nd * j + nd * nd * k
                A[idx, idx] = 6.0
                if i > 0:
                    A[idx, idx - 1] = -1.0
                if i + 1 < nd:
                    A[idx, idx + 1] = -1.0
                if j > 0:
                    A[idx, idx - nd] = -1.0
                if j + 1 < nd:
                    A[idx, idx + nd] = -1.0
                if k > 0:
                    A[idx, idx - nd * nd] = -1.0
                if k + 1 < nd:
                    A[idx, idx + nd * nd] = -1.0
    return A.tocsc()


class MultifontalTests(unittest.TestCase):
    def test_mfx_mv_sv_logdet_match_dense(self):
        A_sparse = _spd_tridiag(5)
        A = A_sparse.toarray()
        x = np.linspace(0.0, 1.0, A.shape[0]).reshape(1, -1)
        X = np.arange(10.0).reshape(5, 2) / 11.0
        F = mfx(A_sparse, x, occ=2)

        self.assertEqual(F.N, A.shape[0])
        self.assertEqual(F.lvp[-1], len(F.factors))
        self.assertTrue(F.hierarchical)
        self.assertIsNone(F.A_dense)
        self.assertIsNotNone(F.A_sparse)
        self.assertIsNone(F.splu)
        np.testing.assert_allclose(mf_mv(F, X), A @ X, atol=1e-12)
        np.testing.assert_allclose(mf_mv(F, X, "t"), A.T @ X, atol=1e-12)
        np.testing.assert_allclose(mf_sv(F, X), np.linalg.solve(A, X), atol=1e-12)
        np.testing.assert_allclose(mf_sv(F, X, "t"), np.linalg.solve(A.T, X), atol=1e-12)
        self.assertAlmostEqual(mf_logdet(F), np.linalg.slogdet(A)[1])
        np.testing.assert_allclose(mf_diag(F), np.diag(A))
        np.testing.assert_allclose(mf_diag(F, True), np.diag(np.linalg.inv(A)))
        np.testing.assert_allclose(mf_spdiag(F), np.diag(A))
        np.testing.assert_allclose(mf_spdiag(F, True), np.diag(np.linalg.inv(A)))

    def test_mfx_complex_sparse_transpose_solves_and_logdet(self):
        A = np.diag(np.array([3.0 + 1.0j, 4.0 - 0.5j, 5.0 + 0.75j, 2.5 + 1.5j]))
        A += np.diag(np.array([1.0 - 2.0j, 1.0j, 2.0]), 1)
        A += np.diag(np.array([0.5 + 0.25j, -1.0 + 0.2j, 0.5 - 0.5j]), -1)
        x = np.linspace(0.0, 1.0, A.shape[0]).reshape(1, -1)
        X = (np.arange(8.0).reshape(4, 2) + 1j * np.arange(8.0, 16.0).reshape(4, 2)) / 17.0
        F = mfx(sp.csc_matrix(A), x, occ=2)
        ref_logdet = np.log(np.linalg.det(A))
        ref_logdet = ref_logdet.real + 1j * np.mod(ref_logdet.imag, 2 * np.pi)

        np.testing.assert_allclose(mf_mv(F, X), A @ X)
        np.testing.assert_allclose(mf_mv(F, X, "t"), A.T @ X)
        np.testing.assert_allclose(mf_mv(F, X, "c"), A.conj().T @ X)
        np.testing.assert_allclose(mf_sv(F, X), np.linalg.solve(A, X))
        np.testing.assert_allclose(mf_sv(F, X, "t"), np.linalg.solve(A.T, X))
        np.testing.assert_allclose(mf_sv(F, X, "c"), np.linalg.solve(A.conj().T, X))
        np.testing.assert_allclose(mf_logdet(F), ref_logdet)

    def test_mf2_positive_definite_cholesky_helpers(self):
        A_sparse = _spd_grid2(4)
        A = A_sparse.toarray()
        X = np.arange(18.0).reshape(9, 2) / 19.0
        F = mf2(A_sparse, n=4, occ=2, opts={"symm": "p"})

        np.testing.assert_allclose(mf_mv(F, X), A @ X, atol=1e-12)
        np.testing.assert_allclose(mf_sv(F, X), np.linalg.solve(A, X), atol=1e-12)
        np.testing.assert_allclose(mf_cholmv(F, mf_cholmv(F, X, "c")), A @ X, atol=1e-12)
        np.testing.assert_allclose(mf_cholsv(F, mf_cholmv(F, X)), X, atol=1e-12)
        np.testing.assert_allclose(mf_cholsv(F, mf_cholsv(F, X), "c"), np.linalg.solve(A, X), atol=1e-12)
        self.assertAlmostEqual(float(np.real(mf_logdet(F))), np.linalg.slogdet(A)[1])

    def test_mf3_dimension_and_solve(self):
        A_sparse = _spd_grid3(3)
        A = A_sparse.toarray()
        x = np.ones(8)
        F = mf3(A_sparse, n=3, occ=2)

        self.assertTrue(F.hierarchical)
        np.testing.assert_allclose(mf_mv(F, x), A @ x, atol=1e-12)
        np.testing.assert_allclose(mf_sv(F, x), np.linalg.solve(A, x), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
