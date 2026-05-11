import unittest

import numpy as np
import scipy.sparse as sp

from pyflam import (
    detperm,
    ismemb,
    logdet_ldl,
    spaddv,
    spget,
    spgetv,
    sppush2,
    sppush3,
    spsymm,
    spsymm2,
)


class SparseCoreTests(unittest.TestCase):
    def test_spget_and_column_storage_helpers(self):
        A = sp.csr_matrix(np.arange(16.0).reshape(4, 4))
        I = np.array([0, 2])
        J = np.array([1, 3])
        np.testing.assert_allclose(spget(A, I, J), A.toarray()[np.ix_(I, J)])

        cols = [sp.csc_matrix(A[:, j]) for j in range(A.shape[1])]
        np.testing.assert_allclose(spgetv(cols, I, J), A.toarray()[np.ix_(I, J)])
        cols = spaddv(cols, I, J, np.ones((2, 2)))
        expected = A.toarray()
        expected[np.ix_(I, J)] += 1
        np.testing.assert_allclose(spgetv(cols, np.arange(4), np.arange(4)), expected)

    def test_sparse_push_helpers_expand_capacity(self):
        I, J, nz = sppush2(np.zeros(1, dtype=int), np.zeros(1, dtype=int), 0, [1, 2], [3, 4])
        self.assertEqual(nz, 2)
        np.testing.assert_array_equal(I[:nz], [1, 2])
        np.testing.assert_array_equal(J[:nz], [3, 4])

        I, J, V, nz = sppush3(np.zeros(1, dtype=int), np.zeros(1, dtype=int), np.zeros(1), 0, [1, 2], [3, 4], [5.0, 6.0])
        self.assertEqual(nz, 2)
        np.testing.assert_array_equal(I[:nz], [1, 2])
        np.testing.assert_array_equal(J[:nz], [3, 4])
        np.testing.assert_allclose(V[:nz], [5.0, 6.0])

    def test_symmetry_helpers(self):
        A = np.array([[1.0, 2.0], [0.0, 3.0]])
        np.testing.assert_allclose(spsymm(A, "s"), [[1.0, 2.0], [2.0, 3.0]])
        B = np.array([[0.0, 4.0], [0.0, 0.0]])
        C, D = spsymm2(A, B, "s")
        np.testing.assert_allclose(D, C.T)

    def test_detperm_ismemb_and_logdet_ldl(self):
        self.assertEqual(detperm([1, 0, 2]), -1)
        self.assertEqual(detperm([2, 0, 1]), 1)
        np.testing.assert_array_equal(ismemb([1, 3, 5], [1, 2, 5]), [True, False, True])
        D = np.diag([2.0, 3.0, 5.0])
        self.assertAlmostEqual(logdet_ldl(D), np.log(30.0))


if __name__ == "__main__":
    unittest.main()
