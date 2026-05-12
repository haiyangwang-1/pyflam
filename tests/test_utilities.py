import unittest

import numpy as np

from pyflam import gausspdf, glegquad, gqgw, lsedc, quad_sqtri3


class UtilityTests(unittest.TestCase):
    def test_gausspdf_matches_standard_normal_value(self):
        self.assertAlmostEqual(float(gausspdf(0.0)), 1.0 / np.sqrt(2 * np.pi))
        np.testing.assert_allclose(gausspdf(np.array([0.0]), mu=1.0, sigma=2.0), [0.17603266])

    def test_glegquad_integrates_polynomial(self):
        x, w = glegquad(3, 0.0, 1.0)

        np.testing.assert_allclose(np.sum(w * x**4), 0.2)
        np.testing.assert_allclose(np.sum(w), 1.0)

    def test_gqgw_matches_legendre_rule_on_minus_one_one(self):
        alpha = np.zeros(2)
        beta = np.array([1.0 / np.sqrt(3.0)])
        x, w = gqgw(alpha, beta, 2.0)

        np.testing.assert_allclose(x, [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        np.testing.assert_allclose(w, [1.0, 1.0])

    def test_quadrature_helpers_validate_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            glegquad(0)

        with self.assertRaisesRegex(ValueError, "beta"):
            gqgw(np.zeros(3), np.ones(3), 2.0)

        vertices = np.zeros((3, 3))
        with self.assertRaisesRegex(ValueError, "shape"):
            quad_sqtri3(np.zeros((3, 1)), np.ones(1), vertices)

        with self.assertRaisesRegex(ValueError, "vertices"):
            quad_sqtri3(np.zeros((2, 1)), np.ones(1), np.zeros((2, 3)))

        with self.assertRaisesRegex(ValueError, "weights"):
            quad_sqtri3(np.zeros((2, 2)), np.ones(1), vertices)

    def test_quad_sqtri3_weight_sum_is_triangle_area(self):
        x0 = np.array([[0.5], [0.5]])
        w0 = np.array([1.0])
        v = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        x, w = quad_sqtri3(x0, w0, v)

        self.assertEqual(x.shape, (3, 1))
        np.testing.assert_allclose(x[:, 0], [0.25, 0.25, 0.0])
        np.testing.assert_allclose(np.sum(w), 0.5)

        q, qw = glegquad(2, 0.0, 1.0)
        u, vq = np.meshgrid(q, q, indexing="ij")
        xq = np.vstack((u.ravel(), vq.ravel()))
        wq = np.outer(qw, qw).ravel()
        tri_x, tri_w = quad_sqtri3(xq, wq, v)
        np.testing.assert_allclose(np.sum(tri_w), 0.5)
        np.testing.assert_allclose(np.sum(tri_w * tri_x[0, :]), 1.0 / 6.0)
        np.testing.assert_allclose(np.sum(tri_w * tri_x[1, :]), 1.0 / 6.0)

    def test_lsedc_validates_controls(self):
        A = np.eye(1)
        B = np.ones((1, 1))
        C = np.ones((1, 1))
        D = np.ones((1, 1))

        def lsfun(rhs):
            return rhs[-1:, :]

        with self.assertRaisesRegex(ValueError, "tolerance"):
            lsedc(lsfun, A, B, C, D, tau=1.0, tol=-1.0)

        with self.assertRaisesRegex(ValueError, "iterations"):
            lsedc(lsfun, A, B, C, D, tau=1.0, niter_max=-1)

    def test_lsedc_enforces_constraints(self):
        A = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        B = np.array([[1.0], [2.0], [2.0]])
        C = np.array([[1.0, 1.0]])
        D = np.array([[1.0]])
        tau = 100.0
        Atau = np.vstack((tau * C, A))

        def lsfun(rhs):
            return np.linalg.lstsq(Atau, rhs, rcond=None)[0]

        x, cres, niter = lsedc(lsfun, A, B, C, D, tau)

        np.testing.assert_allclose(C @ x, D, atol=1e-12)
        np.testing.assert_allclose(cres, D - C @ x, atol=1e-14)
        self.assertGreaterEqual(niter, 0)

        kkt = np.block([[A.T @ A, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
        rhs = np.vstack((A.T @ B, D))
        expected = np.linalg.solve(kkt, rhs)[: A.shape[1], :]
        np.testing.assert_allclose(x, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
