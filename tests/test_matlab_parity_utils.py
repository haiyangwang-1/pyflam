import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

import matlab_parity_utils as parity_utils
from matlab_parity_utils import (
    FLAM_MARKERS,
    factor_metadata_code,
    load_reference_dependencies,
    load_factor_metadata,
    logdet_mod_error,
    matlab_script_command,
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

    def test_matlab_script_command_defaults_to_matlab_batch_flag(self):
        old_launcher = os.environ.pop("PYFLAM_MATLAB_LAUNCHER", None)
        try:
            cmd = matlab_script_command(Path("/tmp/run_case.m"))
        finally:
            if old_launcher is not None:
                os.environ["PYFLAM_MATLAB_LAUNCHER"] = old_launcher

        self.assertEqual(cmd[1:], ["-batch", "run('/tmp/run_case.m')"])

    def test_matlab_script_command_supports_command_launcher(self):
        old_launcher = os.environ.get("PYFLAM_MATLAB_LAUNCHER")
        os.environ["PYFLAM_MATLAB_LAUNCHER"] = "run-matlab-command"
        try:
            cmd = matlab_script_command(Path("/tmp/run_case.m"))
        finally:
            if old_launcher is None:
                os.environ.pop("PYFLAM_MATLAB_LAUNCHER", None)
            else:
                os.environ["PYFLAM_MATLAB_LAUNCHER"] = old_launcher

        self.assertEqual(cmd[1:], ["run('/tmp/run_case.m')"])

    def test_logdet_mod_error_ignores_branch_offset(self):
        ref = 3.0 + 0.25j
        got = ref + 4j * np.pi

        self.assertLess(logdet_mod_error(got, ref), 1e-12)

    def test_relerr_uses_absolute_scale_for_zero_reference(self):
        self.assertEqual(relerr(np.zeros((2, 2)), np.zeros((2, 2))), 0.0)
        np.testing.assert_allclose(relerr(np.array([2.0]), np.array([0.0])), 2e300, rtol=1e-15)
        self.assertEqual(relerr(np.array([2.0, 4.0]), np.array([1.0, 2.0])), 1.0)

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
        def populate_flam_markers(root):
            for marker in FLAM_MARKERS:
                path = root / marker
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            env_ref = tmpdir / "env-flam"
            repo_root = tmpdir / "repo"
            temp_root = tmpdir / "temp"
            repo_ref = repo_root / "tests" / "references" / "flam"
            temp_ref = temp_root / "flam-reference"
            populate_flam_markers(repo_ref)
            populate_flam_markers(temp_ref)

            with (
                mock.patch.dict(os.environ, {"FLAM_REFERENCE": str(env_ref)}),
                mock.patch.object(parity_utils, "REPO_ROOT", repo_root),
                mock.patch.object(parity_utils.tempfile, "gettempdir", return_value=str(temp_root)),
            ):
                self.assertEqual(parity_utils.default_flam_reference(), env_ref)

            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(parity_utils, "REPO_ROOT", repo_root),
                mock.patch.object(parity_utils.tempfile, "gettempdir", return_value=str(temp_root)),
                mock.patch.object(parity_utils.Path, "home", return_value=tmpdir / "home"),
            ):
                self.assertEqual(parity_utils.default_flam_reference(), repo_ref)

            for marker in FLAM_MARKERS:
                (repo_ref / marker).unlink()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(parity_utils, "REPO_ROOT", repo_root),
                mock.patch.object(parity_utils.tempfile, "gettempdir", return_value=str(temp_root)),
                mock.patch.object(parity_utils.Path, "home", return_value=tmpdir / "home"),
            ):
                self.assertEqual(parity_utils.default_flam_reference(), temp_ref)

    def test_default_chunkie_reference_prefers_repo_submodule(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            env_ref = tmpdir / "env-chunkie"
            repo_root = tmpdir / "repo"
            temp_root = tmpdir / "temp"
            repo_ref = repo_root / "tests" / "references" / "chunkie"
            temp_ref = temp_root / "chunkie-reference"
            repo_ref.mkdir(parents=True)
            temp_ref.mkdir(parents=True)

            with (
                mock.patch.dict(os.environ, {"CHUNKIE_REFERENCE": str(env_ref)}),
                mock.patch.object(parity_utils, "REPO_ROOT", repo_root),
                mock.patch.object(parity_utils.tempfile, "gettempdir", return_value=str(temp_root)),
            ):
                self.assertEqual(parity_utils.default_chunkie_reference(), env_ref)

            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(parity_utils, "REPO_ROOT", repo_root),
                mock.patch.object(parity_utils.tempfile, "gettempdir", return_value=str(temp_root)),
                mock.patch.object(parity_utils.Path, "home", return_value=tmpdir / "home"),
            ):
                self.assertEqual(parity_utils.default_chunkie_reference(), repo_ref)

            repo_ref.rmdir()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(parity_utils, "REPO_ROOT", repo_root),
                mock.patch.object(parity_utils.tempfile, "gettempdir", return_value=str(temp_root)),
                mock.patch.object(parity_utils.Path, "home", return_value=tmpdir / "home"),
            ):
                self.assertEqual(parity_utils.default_chunkie_reference(), temp_ref)

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

        expected_lines = {
            "nlvl": "if isfield(F,'nlvl'), meta.nlvl = F.nlvl; else, meta.nlvl = -1; end",
            "lvp": "if isfield(F,'lvp'), meta.lvp = F.lvp; else, meta.lvp = []; end",
            "lvpd": "if isfield(F,'lvpd'), meta.lvpd = F.lvpd; else, meta.lvpd = []; end",
            "lvpu": "if isfield(F,'lvpu'), meta.lvpu = F.lvpu; else, meta.lvpu = []; end",
            "lvpb": "if isfield(F,'lvpb'), meta.lvpb = F.lvpb; else, meta.lvpb = []; end",
            "nfactors": "if isfield(F,'factors'), meta.nfactors = length(F.factors); else, meta.nfactors = 0; end",
            "nd": "if isfield(F,'D'), meta.nd = length(F.D); else, meta.nd = 0; end",
            "nu": "if isfield(F,'U'), meta.nu = length(F.U); else, meta.nu = 0; end",
            "nb": "if isfield(F,'B'), meta.nb = length(F.B); else, meta.nb = 0; end",
            "nsi": "if isfield(F,'Si'), meta.nsi = length(F.Si); else, meta.nsi = 0; end",
            "s_nnz": "if isfield(F,'S'), meta.s_nnz = nnz(F.S); else, meta.s_nnz = 0; end",
        }
        for line in expected_lines.values():
            self.assertIn(line, code)

    def test_load_factor_metadata_simplifies_matlab_struct(self):
        meta = np.array(
            [
                (
                    np.array([[2]]),
                    np.array([[0, 3, 5]]),
                    np.array([[0, 1]]),
                    np.array([[0, 2]]),
                    np.array([[0, 4]]),
                    np.array([[7]]),
                    np.array([[8]]),
                    np.array([[9]]),
                    np.array([[10]]),
                    np.array([[11]]),
                    np.array([[12]]),
                )
            ],
            dtype=[
                ("nlvl", "O"),
                ("lvp", "O"),
                ("lvpd", "O"),
                ("lvpu", "O"),
                ("lvpb", "O"),
                ("nfactors", "O"),
                ("nd", "O"),
                ("nu", "O"),
                ("nb", "O"),
                ("nsi", "O"),
                ("s_nnz", "O"),
            ],
        )

        out = load_factor_metadata({"factor_meta": meta})

        self.assertEqual(out["nlvl"], 2)
        np.testing.assert_array_equal(out["lvp"], [0, 3, 5])
        np.testing.assert_array_equal(out["lvpd"], [0, 1])
        np.testing.assert_array_equal(out["lvpu"], [0, 2])
        np.testing.assert_array_equal(out["lvpb"], [0, 4])
        self.assertEqual(out["nfactors"], 7)
        self.assertEqual(out["nd"], 8)
        self.assertEqual(out["nu"], 9)
        self.assertEqual(out["nb"], 10)
        self.assertEqual(out["nsi"], 11)
        self.assertEqual(out["s_nnz"], 12)


if __name__ == "__main__":
    unittest.main()
