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
        rows = np.array([0, 2])
        col_indices = np.array([1, 3])
        np.testing.assert_allclose(spget(A, rows, col_indices), A.toarray()[np.ix_(rows, col_indices)])

        cols = [sp.csc_matrix(A[:, j]) for j in range(A.shape[1])]
        np.testing.assert_allclose(spgetv(cols, rows, col_indices), A.toarray()[np.ix_(rows, col_indices)])
        cols = spaddv(cols, rows, col_indices, np.ones((2, 2)))
        expected = A.toarray()
        expected[np.ix_(rows, col_indices)] += 1
        np.testing.assert_allclose(spgetv(cols, np.arange(4), np.arange(4)), expected)

    def test_sparse_push_helpers_expand_capacity(self):
        row_buffer, col_buffer, nz = sppush2(np.zeros(1, dtype=int), np.zeros(1, dtype=int), 0, [1, 2], [3, 4])
        self.assertEqual(nz, 2)
        np.testing.assert_array_equal(row_buffer[:nz], [1, 2])
        np.testing.assert_array_equal(col_buffer[:nz], [3, 4])

        row_buffer, col_buffer, value_buffer, nz = sppush3(
            np.zeros(1, dtype=int),
            np.zeros(1, dtype=int),
            np.zeros(1),
            0,
            [1, 2],
            [3, 4],
            [5.0, 6.0],
        )
        self.assertEqual(nz, 2)
        np.testing.assert_array_equal(row_buffer[:nz], [1, 2])
        np.testing.assert_array_equal(col_buffer[:nz], [3, 4])
        np.testing.assert_allclose(value_buffer[:nz], [5.0, 6.0])

    def test_symmetry_helpers(self):
        A = np.array([[1.0, 2.0], [0.0, 3.0]])
        np.testing.assert_allclose(spsymm(A, "s"), [[1.0, 2.0], [2.0, 3.0]])
        Ah = np.array([[1.0, 2.0 + 1.0j], [0.0, 3.0]])
        expected_h = np.array([[1.0, 2.0 + 1.0j], [2.0 - 1.0j, 3.0]])
        np.testing.assert_allclose(spsymm(Ah, "h"), expected_h)
        np.testing.assert_allclose(spsymm(sp.csr_matrix(Ah), "h").toarray(), expected_h)

        B = np.array([[0.0, 4.0], [0.0, 0.0]])
        C, D = spsymm2(A, B, "s")
        np.testing.assert_allclose(C, [[1.0, 2.0], [4.0, 3.0]])
        np.testing.assert_allclose(D, [[1.0, 4.0], [2.0, 3.0]])
        np.testing.assert_allclose(D, C.T)

        Bh = np.array([[0.0, 4.0 - 2.0j], [0.0, 0.0]])
        Ch, Dh = spsymm2(Ah, Bh, "h")
        np.testing.assert_allclose(Ch, [[1.0, 2.0 + 1.0j], [4.0 + 2.0j, 3.0]])
        np.testing.assert_allclose(Dh, Ch.conj().T)

    def test_detperm_ismemb_and_logdet_ldl(self):
        self.assertEqual(detperm([1, 0, 2]), -1)
        self.assertEqual(detperm([2, 0, 1]), 1)
        np.testing.assert_array_equal(ismemb([1, 3, 5], [1, 2, 5]), [True, False, True])
        D = np.diag([2.0, 3.0, 5.0])
        self.assertAlmostEqual(logdet_ldl(D), np.log(30.0))

        block_D = np.array(
            [
                [4.0, 0.0, 0.0],
                [0.0, 2.0, 1.0],
                [0.0, 1.0, 3.0],
            ]
        )
        np.testing.assert_allclose(logdet_ldl(block_D), np.log(20.0))


if __name__ == "__main__":
    unittest.main()
