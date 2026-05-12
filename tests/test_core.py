import unittest

import numpy as np

from pyflam import hypoct, hypoct_perm, id, snorm


class CoreTests(unittest.TestCase):
    def test_hypoct_empty_and_singleton_trees(self):
        empty = np.empty((2, 0))
        tree = hypoct(empty, occ=1)
        self.assertEqual(tree.nlvl, 1)
        self.assertEqual(len(tree.nodes), 1)
        np.testing.assert_array_equal(hypoct_perm(tree), np.array([], dtype=np.int64))

        single = np.array([[0.25], [0.75]])
        tree = hypoct(single, occ=0)
        self.assertEqual(tree.nlvl, 1)
        self.assertEqual(len(tree.nodes), 1)
        np.testing.assert_array_equal(hypoct_perm(tree), [0])

    def test_hypoct_contains_each_point_once(self):
        x = np.array([[0.0, 0.1, 0.8, 0.9], [0.0, 0.2, 0.7, 1.0]])
        tree = hypoct(x, occ=1)
        perm = hypoct_perm(tree)
        self.assertGreaterEqual(tree.nlvl, 2)
        np.testing.assert_array_equal(np.sort(perm), np.arange(x.shape[1]))
        leaves = [node for node in tree.nodes if len(node.chld) == 0]
        self.assertTrue(all(node.xi.size <= 1 for node in leaves))

    def test_hypoct_repeated_points_do_not_refine_forever(self):
        x = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]])

        tree = hypoct(x, occ=1)

        self.assertEqual(tree.nlvl, 1)
        self.assertEqual(len(tree.nodes), 1)
        np.testing.assert_array_equal(hypoct_perm(tree), np.arange(x.shape[1]))

    def test_hypoct_high_dimension_child_codes_do_not_overflow(self):
        x = np.zeros((70, 4))
        x[64, :] = [0.0, 1.0, 0.0, 1.0]
        x[65, :] = [0.0, 0.0, 1.0, 1.0]

        tree = hypoct(x, occ=1)

        self.assertEqual(tree.nlvl, 2)
        leaves = [node for node in tree.nodes if len(node.chld) == 0]
        self.assertTrue(all(node.xi.size == 1 for node in leaves))
        np.testing.assert_array_equal(np.sort(hypoct_perm(tree)), np.arange(x.shape[1]))

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
        U = rng.standard_normal((8, 3))
        V = rng.standard_normal((3, 6))
        A = U @ V

        sk, rd, T, niter = id(A, 3, fixed=[4, 1], return_niter=True)

        np.testing.assert_array_equal(sk[:2], [4, 1])
        self.assertEqual(sk.size, 3)
        self.assertGreaterEqual(niter, 0)
        self.assertEqual(T.shape, (sk.size, rd.size))
        np.testing.assert_allclose(A[:, rd], A[:, sk] @ T, atol=1e-10)

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
        residual = np.linalg.norm(A[:, rd] - A[:, sk] @ T)
        singular_values = np.linalg.svd(A, compute_uv=False)
        best_rank4_residual = np.linalg.norm(singular_values[4:])
        self.assertLessEqual(residual, 1.5 * best_rank4_residual)

    def test_id_rank_cap_and_tolerance_modes(self):
        A = np.diag([10.0, 1.0, 1e-4, 0.0])

        sk, rd, T = id(A, 2)
        self.assertEqual(sk.size, 2)
        self.assertEqual(T.shape, (2, 2))
        capped_residual = np.linalg.norm(A[:, rd] - A[:, sk] @ T)
        best_rank2_residual = np.linalg.norm(np.linalg.svd(A, compute_uv=False)[2:])
        np.testing.assert_allclose(capped_residual, best_rank2_residual, rtol=1e-12, atol=1e-12)

        sk, rd, T = id(A, 1e-3)
        self.assertEqual(sk.size, 2)
        if rd.size:
            np.testing.assert_allclose(A[:, rd[:1]], A[:, sk] @ T[:, :1], atol=2e-4)

        sk, rd, T = id(A, 1e-6)
        self.assertEqual(sk.size, 3)
        if rd.size:
            np.testing.assert_allclose(A[:, rd], A[:, sk] @ T, atol=1e-10)

    def test_id_complex_empty_fixed_and_rank_deficient_inputs(self):
        sk, rd, T = id(np.empty((3, 0)), 1e-12)
        np.testing.assert_array_equal(sk, [])
        np.testing.assert_array_equal(rd, [])
        self.assertEqual(T.shape, (0, 0))

        sk, rd, T = id(np.empty((0, 4), dtype=complex), 1e-12)
        np.testing.assert_array_equal(sk, [])
        np.testing.assert_array_equal(rd, np.arange(4))
        self.assertEqual(T.shape, (0, 4))
        self.assertTrue(np.iscomplexobj(T))

        u = np.array([[1.0 + 1.0j], [2.0 - 0.5j], [-1.0j]])
        v = np.array([[1.0, 2.0j, -1.0, 0.5j]])
        A = u @ v
        sk, rd, T = id(A, 1e-12, fixed=[2])
        np.testing.assert_array_equal(sk[:1], [2])
        self.assertEqual(sk.size, 1)
        np.testing.assert_allclose(A[:, rd], A[:, sk] @ T, atol=1e-10)
        self.assertTrue(np.iscomplexobj(T))

    def test_snorm_matches_diagonal_norm(self):
        A = np.diag([1.0, -3.0, 2.0])
        s, _ = snorm(3, lambda x: A @ x, lambda x: A.T @ x, tol=1e-8)
        self.assertAlmostEqual(s, 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
