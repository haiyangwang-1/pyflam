import tempfile
import unittest
from pathlib import Path

import numpy as np

from matlab_parity_utils import logdet_mod_error, matlab_path, relerr, require_paths


class MatlabParityUtilsTests(unittest.TestCase):
    def test_matlab_path_escapes_windows_paths_and_quotes(self):
        path = Path(r"C:\tmp\can't\break.mat")

        self.assertEqual(matlab_path(path), "C:/tmp/can''t/break.mat")

    def test_logdet_mod_error_ignores_branch_offset(self):
        ref = 3.0 + 0.25j
        got = ref + 4j * np.pi

        self.assertLess(logdet_mod_error(got, ref), 1e-12)

    def test_relerr_uses_absolute_scale_for_zero_reference(self):
        self.assertEqual(relerr(np.zeros((2, 2)), np.zeros((2, 2))), 0.0)

    def test_require_paths_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            with self.assertRaisesRegex(RuntimeError, "test parity requires"):
                require_paths(missing, label="test parity")


if __name__ == "__main__":
    unittest.main()
