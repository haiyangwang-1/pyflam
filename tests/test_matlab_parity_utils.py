import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from matlab_parity_utils import (
    FLAM_MARKERS,
    REPO_ROOT,
    default_chunkie_reference,
    default_flam_reference,
    factor_metadata_code,
    load_reference_dependencies,
    load_factor_metadata,
    logdet_mod_error,
    matlab_path,
    relerr,
    require_flam_reference,
    require_paths,
    require_pinned_reference,
)


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

    def test_require_flam_reference_reports_missing_entry_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "rskelf"):
                require_flam_reference(ref, label="test parity")

    def test_default_flam_reference_prefers_complete_checkout(self):
        ref = default_flam_reference()

        if ref.exists():
            self.assertTrue(all((ref / marker).exists() for marker in FLAM_MARKERS))

    def test_default_chunkie_reference_prefers_repo_submodule(self):
        ref = default_chunkie_reference()
        repo_ref = REPO_ROOT / "tests" / "references" / "chunkie"

        if "CHUNKIE_REFERENCE" not in os.environ and repo_ref.exists():
            self.assertEqual(ref, repo_ref)

    def test_reference_dependency_pins_are_loaded(self):
        deps = load_reference_dependencies()

        self.assertEqual(deps["flam"]["commit"], "b928b2b1b4e0c3a00558bcdc7e3147fe83e720c4")
        self.assertEqual(deps["chunkie"]["commit"], "af34cc41c81114e693b515066e4d308067bf7e63")
        self.assertTrue(deps["chunkie"]["required_clean"])
        self.assertNotIn("tracked_dirty_patch", deps["chunkie"])

    def test_require_pinned_reference_reports_non_git_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "pinned FLAM git checkout"):
                require_pinned_reference(Path(tmp), "flam", label="test parity")

    def test_factor_metadata_code_covers_common_factor_fields(self):
        code = factor_metadata_code("F", "meta")

        for field in ("nlvl", "lvp", "lvpd", "lvpu", "lvpb", "nfactors", "nd", "nu", "nb", "nsi", "s_nnz"):
            self.assertIn(f"meta.{field}", code)

    def test_load_factor_metadata_simplifies_matlab_struct(self):
        meta = np.array(
            [
                (
                    np.array([[2]]),
                    np.array([[0, 3, 5]]),
                    np.array([[7]]),
                )
            ],
            dtype=[("nlvl", "O"), ("lvp", "O"), ("nfactors", "O")],
        )

        out = load_factor_metadata({"factor_meta": meta})

        self.assertEqual(out["nlvl"], 2)
        np.testing.assert_array_equal(out["lvp"], [0, 3, 5])
        self.assertEqual(out["nfactors"], 7)


if __name__ == "__main__":
    unittest.main()
