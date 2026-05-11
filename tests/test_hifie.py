import unittest

import numpy as np

from pyflam import (
    hifie2,
    hifie2x,
    hifie3,
    hifie_cholmv,
    hifie_cholsv,
    hifie_diag,
    hifie_logdet,
    hifie_mv,
    hifie_spdiag,
    hifie_sv,
)


class HIFIETests(unittest.TestCase):
    def setUp(self):
        t = np.linspace(0.0, 1.0, 10)
        self.x2 = np.vstack((t, t**2))
        self.x3 = np.vstack((t, t**2, t**3))
        dist = np.abs(t[:, None] - t[None, :])
        self.A = 1.0 / (1.0 + dist) + 3.0 * np.eye(t.size)
        self.X = np.arange(20.0).reshape(10, 2) / 21.0

    def test_hifie2_operations_match_dense(self):
        F = hifie2(self.A, self.x2, occ=3, rank_or_tol=1e-10)

        self.assertEqual(F.opts["hifie_variant"], "hifie2")
        np.testing.assert_allclose(hifie_mv(F, self.X), self.A @ self.X)
        np.testing.assert_allclose(hifie_sv(F, self.X), np.linalg.solve(self.A, self.X))
        self.assertAlmostEqual(hifie_logdet(F), np.linalg.slogdet(self.A)[1])
        np.testing.assert_allclose(hifie_diag(F), np.diag(self.A))
        np.testing.assert_allclose(hifie_spdiag(F, True), np.diag(np.linalg.inv(self.A)))

    def test_hifie2x_and_hifie3_entry_points(self):
        F2x = hifie2x(self.A, self.x2, occ=3, rank_or_tol=1e-10)
        F3 = hifie3(self.A, self.x3, occ=3, rank_or_tol=1e-10)

        self.assertEqual(F2x.opts["hifie_variant"], "hifie2x")
        self.assertEqual(F3.opts["hifie_variant"], "hifie3")
        np.testing.assert_allclose(hifie_mv(F2x, self.X, "t"), self.A.T @ self.X)
        np.testing.assert_allclose(hifie_sv(F3, self.X, "t"), np.linalg.solve(self.A.T, self.X))

    def test_hifie_positive_definite_cholesky_helpers(self):
        F = hifie2(self.A, self.x2, occ=3, rank_or_tol=1e-10, opts={"symm": "p"})

        np.testing.assert_allclose(hifie_cholmv(F, self.X), F.chol @ self.X)
        np.testing.assert_allclose(hifie_cholmv(F, self.X, "c"), F.chol.conj().T @ self.X)
        np.testing.assert_allclose(hifie_cholsv(F, self.X), np.linalg.solve(F.chol, self.X))
        np.testing.assert_allclose(hifie_cholsv(F, self.X, "c"), np.linalg.solve(F.chol.conj().T, self.X))


if __name__ == "__main__":
    unittest.main()
