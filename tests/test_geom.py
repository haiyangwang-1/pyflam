import unittest

import numpy as np

from pyflam import tri3geom, trisphere_subdiv


class GeometryTests(unittest.TestCase):
    def test_tri3geom_single_triangle(self):
        V = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        C, N, A = tri3geom(V)

        np.testing.assert_allclose(C[:, 0], [1.0 / 3.0, 1.0 / 3.0, 0.0])
        np.testing.assert_allclose(N[:, 0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(A, [0.5])

    def test_tri3geom_multiple_faces_use_zero_based_indices(self):
        V = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        F = np.array([[0, 0], [1, 2], [2, 3]])
        C, N, A = tri3geom(V, F)

        self.assertEqual(C.shape, (3, 2))
        self.assertEqual(N.shape, (3, 2))
        np.testing.assert_allclose(C, [[1.0 / 3.0, 0.0], [1.0 / 3.0, 1.0 / 3.0], [0.0, 1.0 / 3.0]])
        np.testing.assert_allclose(N, [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(A, [0.5, 0.5])

    def test_trisphere_subdiv_base_and_refined_sizes(self):
        V, F = trisphere_subdiv(20, "f")

        self.assertEqual(V.shape, (3, 12))
        self.assertEqual(F.shape, (3, 20))
        self.assertGreaterEqual(np.min(F), 0)
        self.assertLess(np.max(F), V.shape[1])
        np.testing.assert_allclose(np.linalg.norm(V, axis=0), 1.0)

        Vr, Fr = trisphere_subdiv(21, "f")
        self.assertEqual(Vr.shape, (3, 42))
        self.assertEqual(Fr.shape[1], 80)
        self.assertGreaterEqual(np.min(Fr), 0)
        self.assertLess(np.max(Fr), Vr.shape[1])
        np.testing.assert_allclose(np.linalg.norm(Vr, axis=0), 1.0)
        self.assertTrue(np.all([np.unique(face).size == 3 for face in Fr.T]))
        face_vertices = Vr[:, Fr]
        doubled_area = np.linalg.norm(
            np.cross(
                (face_vertices[:, 1, :] - face_vertices[:, 0, :]).T,
                (face_vertices[:, 2, :] - face_vertices[:, 1, :]).T,
            ),
            axis=1,
        )
        self.assertGreater(np.min(doubled_area), 0.0)


if __name__ == "__main__":
    unittest.main()
