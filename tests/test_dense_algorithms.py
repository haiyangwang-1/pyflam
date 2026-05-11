import unittest

import numpy as np
import scipy.sparse as sp

from pyflam import (
    ifmm,
    ifmm_mv,
    rskel,
    rskel_mv,
    rskel_xsp,
    rskelf,
    rskelf_cholmv,
    rskelf_cholsv,
    rskelf_diag,
    rskelf_logdet,
    rskelf_mv,
    rskelf_partial_info,
    rskelf_partial_mv,
    rskelf_partial_sv,
    rskelf_spdiag,
    rskelf_sv,
)


def kernel_matrix(x, y):
    return 1.0 / (1.0 + np.abs(x[:, None] - y[None, :]))


class DenseAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0.0, 1.0, 8).reshape(1, -1)
        self.A = kernel_matrix(self.x.ravel(), self.x.ravel()) + 2.0 * np.eye(8)
        self.X = np.arange(16.0).reshape(8, 2) / 17.0

        def Afun(i, j):
            return self.A[np.ix_(i, j)]

        self.Afun = Afun

    def test_rskel_mv_matches_dense(self):
        F = rskel(self.Afun, self.x, self.x, occ=3, rank_or_tol=1e-10)
        self.assertGreater(len(F.D), 0)
        self.assertGreater(len(F.U), 0)
        self.assertEqual(F.lvpd[-1], len(F.D))
        self.assertEqual(F.lvpu[-1], len(F.U))
        np.testing.assert_allclose(rskel_mv(F, self.X), self.A @ self.X)
        np.testing.assert_allclose(rskel_mv(F, self.X, "c"), self.A.conj().T @ self.X)
        Xsp, p, q = rskel_xsp(F)
        self.assertGreater(Xsp.shape[0], self.A.shape[0])
        self.assertGreater(Xsp.shape[1], self.A.shape[1])
        self.assertEqual(Xsp.nnz, 160)
        np.testing.assert_array_equal(p, F.P)
        np.testing.assert_array_equal(q, F.Q)

    def test_rskel_symmetric_mv_paths_match_dense(self):
        x = np.linspace(0.0, 1.0, 20).reshape(1, -1)
        A = kernel_matrix(x.ravel(), x.ravel()) + 3.0 * np.eye(20)
        X = np.random.default_rng(7).standard_normal((20, 3))

        for symm in ("s", "h", "p"):
            with self.subTest(symm=symm):
                F = rskel(A, x, x, occ=3, rank_or_tol=1e-10, opts={"symm": symm})
                self.assertGreater(len(F.U), 0)
                np.testing.assert_allclose(rskel_mv(F, X), A @ X, rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(rskel_mv(F, X, "c"), A.conj().T @ X, rtol=1e-9, atol=1e-9)

    def test_rskel_callback_is_not_eagerly_materialized(self):
        x = np.linspace(0.0, 1.0, 20).reshape(1, -1)
        A = kernel_matrix(x.ravel(), x.ravel()) + 3.0 * np.eye(20)
        X = np.random.default_rng(8).standard_normal((20, 2))
        calls = []

        def Afun(i, j):
            calls.append((i.size, j.size))
            if i.size == A.shape[0] and j.size == A.shape[1]:
                raise AssertionError("callback was eagerly materialized")
            return A[np.ix_(i, j)]

        F = rskel(Afun, x, x, occ=2, rank_or_tol=1e-10)

        self.assertIsNone(F.A_dense)
        self.assertGreater(len(calls), 0)
        np.testing.assert_allclose(rskel_mv(F, X), A @ X, rtol=1e-9, atol=1e-9)

    def test_ifmm_mv_matches_dense(self):
        F = ifmm(self.Afun, self.x, self.x, occ=3, rank_or_tol=1e-10, opts={"store": "a", "near": 1})
        self.assertGreater(len(F.B), 0)
        self.assertEqual(F.lvpb[-1], len(F.B))
        self.assertEqual(F.lvpu[-1], len(F.U))
        np.testing.assert_allclose(ifmm_mv(F, self.X), self.A @ self.X)
        np.testing.assert_allclose(ifmm_mv(F, self.X, trans="t"), self.A.T @ self.X)

    def test_ifmm_mv_generates_missing_interactions_from_A(self):
        F = ifmm(self.Afun, self.x, self.x, occ=3, rank_or_tol=1e-10, opts={"store": "n", "near": 0})

        np.testing.assert_allclose(ifmm_mv(F, self.X, self.Afun), self.A @ self.X)
        with self.assertRaises(ValueError):
            ifmm_mv(F, self.X)

    def test_ifmm_mv_rectangular_adjoint(self):
        rx = np.linspace(0.0, 1.0, 7).reshape(1, -1)
        cx = np.linspace(0.05, 0.95, 5).reshape(1, -1)
        A = kernel_matrix(rx.ravel(), cx.ravel())
        X = np.arange(10.0).reshape(5, 2) / 11.0
        Y = np.arange(14.0).reshape(7, 2) / 15.0

        def Afun(i, j):
            return A[np.ix_(i, j)]

        F = ifmm(Afun, rx, cx, occ=3, rank_or_tol=1e-10, opts={"store": "a", "near": 1})

        np.testing.assert_allclose(ifmm_mv(F, X), A @ X)
        np.testing.assert_allclose(ifmm_mv(F, Y, trans="c"), A.conj().T @ Y)

    def test_ifmm_store_near_and_symmetry_modes_match_dense(self):
        x = np.linspace(0.0, 1.0, 16).reshape(1, -1)
        base = kernel_matrix(x.ravel(), x.ravel()) + 2.0 * np.eye(16)
        rng = np.random.default_rng(10)

        cases = {
            "s": base,
            "h": base.astype(complex) + 0.03j * (x.T - x),
        }
        for symm, A in cases.items():
            X = rng.standard_normal((16, 3))
            if np.iscomplexobj(A):
                X = X + 1j * rng.standard_normal((16, 3))
            for store in ("n", "s", "r", "a"):
                for near in (0, 1):
                    with self.subTest(symm=symm, store=store, near=near):
                        F = ifmm(A, x, x, occ=2, rank_or_tol=1e-10, opts={"symm": symm, "store": store, "near": near})
                        np.testing.assert_allclose(ifmm_mv(F, X, A), A @ X, rtol=1e-9, atol=1e-9)
                        np.testing.assert_allclose(ifmm_mv(F, X, A, trans="c"), A.conj().T @ X, rtol=1e-9, atol=1e-9)

    def test_ifmm_rectangular_complex_proxy_callback(self):
        rx = np.linspace(0.0, 1.0, 20).reshape(1, -1)
        cx = np.linspace(0.05, 0.95, 15).reshape(1, -1)
        A = kernel_matrix(rx.ravel(), cx.ravel()).astype(complex)
        A += 0.1j * np.cos(rx.ravel()[:, None] + 2.0 * cx.ravel()[None, :])
        rng = np.random.default_rng(11)
        X = rng.standard_normal((15, 2)) + 1j * rng.standard_normal((15, 2))
        Z = rng.standard_normal((20, 2)) + 1j * rng.standard_normal((20, 2))
        calls = []

        def pxyfun(rc, rxp, cxp, slf, nbr, l, ctr):
            calls.append(rc)
            if rc == "r":
                far = np.setdiff1d(np.arange(cxp.shape[1]), nbr, assume_unique=False)
                return A[np.ix_(slf, far)], nbr
            far = np.setdiff1d(np.arange(rxp.shape[1]), nbr, assume_unique=False)
            return A[np.ix_(far, slf)], nbr

        F = ifmm(A, rx, cx, occ=2, rank_or_tol=1e-10, pxyfun=pxyfun, opts={"store": "a", "near": 1})

        self.assertIn("r", calls)
        self.assertIn("c", calls)
        np.testing.assert_allclose(ifmm_mv(F, X), A @ X, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(ifmm_mv(F, Z, trans="c"), A.conj().T @ Z, rtol=1e-9, atol=1e-9)

    def test_ifmm_mv_promotes_complex_stored_blocks(self):
        A = self.A.astype(complex)
        A[0, 1] += 0.25j
        A[3, 2] -= 0.5j

        def Afun(i, j):
            return A[np.ix_(i, j)]

        F = ifmm(Afun, self.x, self.x, occ=3, rank_or_tol=1e-10, opts={"store": "a", "near": 1})

        np.testing.assert_allclose(ifmm_mv(F, self.X), A @ self.X)
        np.testing.assert_allclose(ifmm_mv(F, self.X, trans="c"), A.conj().T @ self.X)

    def test_ifmm_mv_promotes_complex_callback_blocks(self):
        A = self.A.astype(complex)
        A[0, 1] += 0.25j
        A[3, 2] -= 0.5j

        def Afun(i, j):
            return A[np.ix_(i, j)]

        F = ifmm(Afun, self.x, self.x, occ=3, rank_or_tol=1e-10, opts={"store": "n", "near": 0})

        np.testing.assert_allclose(ifmm_mv(F, self.X, Afun), A @ self.X)
        np.testing.assert_allclose(ifmm_mv(F, self.X, Afun, trans="c"), A.conj().T @ self.X)

    def test_rskelf_mv_sv_logdet_match_dense(self):
        F = rskelf(self.Afun, self.x, occ=3, rank_or_tol=1e-10)
        self.assertGreater(len(F.factors), 0)
        self.assertEqual(F.lvp[-1], len(F.factors))
        for factor in F.factors:
            self.assertGreater(factor.rd.size, 0)
            self.assertEqual(factor.T.shape[1], factor.rd.size)
        np.testing.assert_allclose(rskelf_mv(F, self.X), self.A @ self.X)
        np.testing.assert_allclose(rskelf_mv(F, self.X, "c"), self.A.conj().T @ self.X)
        np.testing.assert_allclose(rskelf_sv(F, self.X), np.linalg.solve(self.A, self.X))
        np.testing.assert_allclose(rskelf_sv(F, self.X, "c"), np.linalg.solve(self.A.conj().T, self.X))
        self.assertAlmostEqual(rskelf_logdet(F), np.linalg.slogdet(self.A)[1])
        np.testing.assert_allclose(rskelf_diag(F), np.diag(self.A))
        np.testing.assert_allclose(rskelf_diag(F, True), np.diag(np.linalg.inv(self.A)))
        np.testing.assert_allclose(rskelf_spdiag(F), np.diag(self.A))
        np.testing.assert_allclose(rskelf_spdiag(F, True), np.diag(np.linalg.inv(self.A)))

    def test_rskelf_positive_definite(self):
        F = rskelf(self.A, self.x, occ=3, rank_or_tol=1e-10, opts={"symm": "p"})
        np.testing.assert_allclose(rskelf_sv(F, self.X), np.linalg.solve(self.A, self.X))
        C = F.chol
        np.testing.assert_allclose(rskelf_cholmv(F, self.X), C @ self.X)
        np.testing.assert_allclose(rskelf_cholmv(F, self.X, "c"), C.conj().T @ self.X)
        np.testing.assert_allclose(rskelf_cholsv(F, self.X), np.linalg.solve(C, self.X))
        np.testing.assert_allclose(rskelf_cholsv(F, self.X, "c"), np.linalg.solve(C.conj().T, self.X))

    def test_rskelf_symmetric_compact_paths_match_dense(self):
        x = np.linspace(0.0, 1.0, 20).reshape(1, -1)
        A = kernel_matrix(x.ravel(), x.ravel()) + 3.0 * np.eye(20)
        X = np.random.default_rng(4).standard_normal((20, 3))

        for symm in ("s", "h", "p"):
            with self.subTest(symm=symm):
                F = rskelf(A, x, occ=2, rank_or_tol=1e-10, opts={"symm": symm})
                self.assertGreater(len(F.factors), 1)
                self.assertEqual(F.Si.size, 0)
                np.testing.assert_allclose(rskelf_mv(F, X), A @ X, rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(rskelf_mv(F, X, "c"), A.conj().T @ X, rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(rskelf_sv(F, X), np.linalg.solve(A, X), rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(rskelf_sv(F, X, "c"), np.linalg.solve(A.conj().T, X), rtol=1e-9, atol=1e-9)
                self.assertAlmostEqual(rskelf_logdet(F), np.linalg.slogdet(A)[1])

    def test_rskelf_generalized_cholesky_round_trips(self):
        x = np.linspace(0.0, 1.0, 20).reshape(1, -1)
        A = kernel_matrix(x.ravel(), x.ravel()) + 3.0 * np.eye(20)
        X = np.random.default_rng(5).standard_normal((20, 2))
        F = rskelf(A, x, occ=2, rank_or_tol=1e-10, opts={"symm": "p"})

        CctX = rskelf_cholmv(F, rskelf_cholmv(F, X, "c"))
        np.testing.assert_allclose(CctX, A @ X, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_cholsv(F, rskelf_cholmv(F, X)), X, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(rskelf_cholsv(F, rskelf_cholmv(F, X, "c"), "c"), X, rtol=1e-10, atol=1e-10)

    def test_rskelf_callback_is_not_eagerly_materialized(self):
        x = np.linspace(0.0, 1.0, 20).reshape(1, -1)
        A = kernel_matrix(x.ravel(), x.ravel()) + 3.0 * np.eye(20)
        X = np.random.default_rng(9).standard_normal((20, 2))
        calls = []

        def Afun(i, j):
            calls.append((i.size, j.size))
            if i.size == A.shape[0] and j.size == A.shape[1]:
                raise AssertionError("callback was eagerly materialized")
            return A[np.ix_(i, j)]

        F = rskelf(Afun, x, occ=2, rank_or_tol=1e-10)

        self.assertIsNone(F.A_dense)
        self.assertGreater(len(calls), 0)
        np.testing.assert_allclose(rskelf_mv(F, X), A @ X, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_sv(F, X), np.linalg.solve(A, X), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_diag(F), np.diag(A), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_diag(F, True), np.diag(np.linalg.inv(A)), rtol=1e-9, atol=1e-9)

    def test_rskelf_partial_mv_sv_use_skeleton_callback(self):
        x = np.linspace(0.0, 1.0, 20).reshape(1, -1)
        A = kernel_matrix(x.ravel(), x.ravel()) + 3.0 * np.eye(20)
        X = np.random.default_rng(6).standard_normal((20, 3))
        F = rskelf(A, x, occ=2, rank_or_tol=1e-10, opts={"stop": 4})
        sk, S = rskelf_partial_info(F)
        self.assertTrue(sp.issparse(S))
        Ask = A[np.ix_(sk, sk)] + S.toarray()

        def mvfun(Y, trans="n"):
            return Ask @ Y if trans == "n" else Ask.conj().T @ Y

        def svfun(Y, trans="n"):
            return np.linalg.solve(Ask if trans == "n" else Ask.conj().T, Y)

        self.assertGreater(len(F.factors), 0)
        self.assertGreater(sk.size, 0)
        self.assertLess(sk.size, A.shape[0])
        np.testing.assert_allclose(rskelf_partial_mv(F, X, mvfun), A @ X, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_partial_sv(F, X, svfun), np.linalg.solve(A, X), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_partial_mv(F, X), A @ X, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_partial_sv(F, X), np.linalg.solve(A, X), rtol=1e-9, atol=1e-9)

    def test_rskelf_partial_diag_uses_compact_factor(self):
        x = np.linspace(0.0, 1.0, 18).reshape(1, -1)
        A = kernel_matrix(x.ravel(), x.ravel()) + 2.0 * np.eye(18)
        F = rskelf(A, x, occ=3, rank_or_tol=1e-10, opts={"stop": 3})
        eye = np.eye(A.shape[0])

        self.assertGreater(len(F.factors), 0)
        self.assertGreater(F.Si.size, 0)
        np.testing.assert_allclose(rskelf_diag(F), np.diag(rskelf_mv(F, eye)), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_diag(F, True), np.diag(rskelf_sv(F, eye)), rtol=1e-9, atol=1e-9)
        self.assertGreater(np.linalg.norm(rskelf_diag(F) - np.diag(A)), 1.0)


if __name__ == "__main__":
    unittest.main()
