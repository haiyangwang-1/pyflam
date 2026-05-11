import unittest

import numpy as np
import scipy.sparse as sp

from pyflam import (
    hifde2,
    hifde2x,
    hifde3,
    hifde3x,
    hifde_cholmv,
    hifde_cholsv,
    hifde_diag,
    hifde_logdet,
    hifde_mv,
    hifde_spdiag,
    hifde_sv,
)


def _spd_tridiag(n):
    return sp.diags(
        [-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csc",
    )


class HIFDETests(unittest.TestCase):
    def test_hifde2_operations_match_sparse_matrix(self):
        A_sparse = _spd_tridiag(9)
        A = A_sparse.toarray()
        X = np.arange(18.0).reshape(9, 2) / 19.0
        F = hifde2(A_sparse, n=4, occ=2, rank_or_tol=1e-10, opts={"skip": 1})

        self.assertEqual(F.opts["hifde_variant"], "hifde2")
        self.assertEqual(F.opts["rank_or_tol"], 1e-10)
        self.assertEqual(F.opts["skip"], 1)
        np.testing.assert_allclose(hifde_mv(F, X), A @ X)
        np.testing.assert_allclose(hifde_sv(F, X), np.linalg.solve(A, X))
        self.assertAlmostEqual(hifde_logdet(F), np.linalg.slogdet(A)[1])
        np.testing.assert_allclose(hifde_diag(F), np.diag(A))
        np.testing.assert_allclose(hifde_spdiag(F, True), np.diag(np.linalg.inv(A)))

    def test_hifde2x_and_hifde3x_entry_points(self):
        A_sparse = _spd_tridiag(6)
        A = A_sparse.toarray()
        x2 = np.vstack((np.linspace(0.0, 1.0, 6), np.linspace(1.0, 2.0, 6)))
        x3 = np.vstack((x2, np.linspace(2.0, 3.0, 6)))
        X = np.arange(12.0).reshape(6, 2) / 13.0

        F2x = hifde2x(A_sparse, x2, occ=2, rank_or_tol=1e-8)
        F3x = hifde3x(A_sparse, x3, occ=2, rank_or_tol=1e-8)

        self.assertEqual(F2x.opts["hifde_variant"], "hifde2x")
        self.assertEqual(F3x.opts["hifde_variant"], "hifde3x")
        np.testing.assert_allclose(hifde_mv(F2x, X, "t"), A.T @ X)
        np.testing.assert_allclose(hifde_sv(F3x, X, "t"), np.linalg.solve(A.T, X))

    def test_hifde3_positive_definite_cholesky_helpers(self):
        A_sparse = _spd_tridiag(8)
        A = A_sparse.toarray()
        X = np.arange(16.0).reshape(8, 2) / 17.0
        F = hifde3(A_sparse, n=3, occ=2, rank_or_tol=1e-10, opts={"symm": "p"})

        np.testing.assert_allclose(hifde_mv(F, X), A @ X)
        np.testing.assert_allclose(hifde_sv(F, X), np.linalg.solve(A, X))
        np.testing.assert_allclose(hifde_cholmv(F, X), F.chol @ X)
        np.testing.assert_allclose(hifde_cholsv(F, X), np.linalg.solve(F.chol, X))
        np.testing.assert_allclose(hifde_cholmv(F, X, "c"), F.chol.conj().T @ X)
        np.testing.assert_allclose(hifde_cholsv(F, X, "c"), np.linalg.solve(F.chol.conj().T, X))


if __name__ == "__main__":
    unittest.main()
