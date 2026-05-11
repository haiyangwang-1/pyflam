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


class MultifontalTests(unittest.TestCase):
    def test_mfx_mv_sv_logdet_match_dense(self):
        A = np.array(
            [
                [4.0, 1.0, 0.0, 0.0, 0.2],
                [2.0, 5.0, 1.0, 0.0, 0.0],
                [0.0, 1.5, 6.0, 1.0, 0.0],
                [0.1, 0.0, 1.0, 4.5, 1.0],
                [0.0, 0.2, 0.0, 1.0, 5.0],
            ]
        )
        x = np.linspace(0.0, 1.0, A.shape[0]).reshape(1, -1)
        X = np.arange(10.0).reshape(5, 2) / 11.0
        F = mfx(sp.csc_matrix(A), x, occ=2)

        self.assertEqual(F.N, A.shape[0])
        self.assertEqual(F.lvp[-1], len(F.factors))
        self.assertIsNone(F.A_dense)
        self.assertIsNotNone(F.A_sparse)
        self.assertIsNotNone(F.splu)
        np.testing.assert_allclose(mf_mv(F, X), A @ X)
        np.testing.assert_allclose(mf_mv(F, X, "t"), A.T @ X)
        np.testing.assert_allclose(mf_sv(F, X), np.linalg.solve(A, X))
        np.testing.assert_allclose(mf_sv(F, X, "t"), np.linalg.solve(A.T, X))
        self.assertAlmostEqual(mf_logdet(F), np.linalg.slogdet(A)[1])
        np.testing.assert_allclose(mf_diag(F), np.diag(A))
        np.testing.assert_allclose(mf_diag(F, True), np.diag(np.linalg.inv(A)))
        np.testing.assert_allclose(mf_spdiag(F), np.diag(A))
        np.testing.assert_allclose(mf_spdiag(F, True), np.diag(np.linalg.inv(A)))

    def test_mf2_positive_definite_cholesky_helpers(self):
        A_sparse = _spd_tridiag(9)
        A = A_sparse.toarray()
        X = np.arange(18.0).reshape(9, 2) / 19.0
        F = mf2(A_sparse, n=4, occ=2, opts={"symm": "p"})

        np.testing.assert_allclose(mf_mv(F, X), A @ X)
        np.testing.assert_allclose(mf_sv(F, X), np.linalg.solve(A, X))
        np.testing.assert_allclose(mf_cholmv(F, X), F.chol @ X)
        np.testing.assert_allclose(mf_cholmv(F, X, "c"), F.chol.conj().T @ X)
        np.testing.assert_allclose(mf_cholsv(F, X), np.linalg.solve(F.chol, X))
        np.testing.assert_allclose(mf_cholsv(F, X, "c"), np.linalg.solve(F.chol.conj().T, X))
        self.assertAlmostEqual(float(np.real(mf_logdet(F))), np.linalg.slogdet(A)[1])

    def test_mf3_dimension_and_solve(self):
        A_sparse = _spd_tridiag(8)
        A = A_sparse.toarray()
        x = np.ones(8)
        F = mf3(A_sparse, n=3, occ=2)

        np.testing.assert_allclose(mf_mv(F, x), A @ x)
        np.testing.assert_allclose(mf_sv(F, x), np.linalg.solve(A, x))


if __name__ == "__main__":
    unittest.main()
