import unittest

import numpy as np

from pyflam import hypoct, hypoct_perm, id, snorm


class CoreTests(unittest.TestCase):
    def test_hypoct_contains_each_point_once(self):
        x = np.array([[0.0, 0.1, 0.8, 0.9], [0.0, 0.2, 0.7, 1.0]])
        tree = hypoct(x, occ=1)
        perm = hypoct_perm(tree)
        self.assertGreaterEqual(tree.nlvl, 2)
        np.testing.assert_array_equal(np.sort(perm), np.arange(x.shape[1]))
        leaves = [node for node in tree.nodes if len(node.chld) == 0]
        self.assertTrue(all(node.xi.size <= 1 for node in leaves))

    def test_id_reconstructs_low_rank_columns(self):
        rng = np.random.default_rng(1)
        U = rng.standard_normal((8, 2))
        V = rng.standard_normal((2, 6))
        A = U @ V
        sk, rd, T = id(A, 1e-12)
        self.assertLessEqual(sk.size, 2)
        if rd.size:
            np.testing.assert_allclose(A[:, rd], A[:, sk] @ T, atol=1e-10)

    def test_id_honors_fixed_columns_and_reports_iterations(self):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((8, 6))

        sk, rd, T, niter = id(A, 3, fixed=[4, 1], return_niter=True)

        np.testing.assert_array_equal(sk[:2], [4, 1])
        self.assertEqual(sk.size, 3)
        self.assertGreaterEqual(niter, 0)
        self.assertEqual(T.shape, (sk.size, rd.size))

    def test_id_rrqr_refinement_bounds_interpolation(self):
        rng = np.random.default_rng(0)
        A = None
        for _ in range(3):
            A = rng.standard_normal((8, 10))

        sk, rd, T, niter = id(A, 4, Tmax=1.01, rrqr_iter=50, return_niter=True)

        self.assertEqual(sk.size, 4)
        self.assertGreater(niter, 0)
        self.assertLessEqual(np.max(np.abs(T)), 1.01 + 1e-12)
        self.assertEqual(T.shape, (sk.size, rd.size))

    def test_snorm_matches_diagonal_norm(self):
        A = np.diag([1.0, -3.0, 2.0])
        s, _ = snorm(3, lambda x: A @ x, lambda x: A.T @ x, tol=1e-8)
        self.assertAlmostEqual(s, 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
