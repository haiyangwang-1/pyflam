import os
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from matlab_parity_utils import MATLAB, logdet_mod_error, require_paths, run_matlab_export
from pyflam import rskelf, rskelf_logdet, rskelf_mv, rskelf_sv


_DEFAULT_FLAM_REF = Path(tempfile.gettempdir()) / "flam-reference"
if not _DEFAULT_FLAM_REF.exists():
    _DEFAULT_FLAM_REF = Path(tempfile.gettempdir()) / "FLAM-ref"
FLAM_REF = Path(os.environ.get("FLAM_REFERENCE", _DEFAULT_FLAM_REF))


class RSkelfOptionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_paths(MATLAB, FLAM_REF, label="rskelf option parity")

    def test_symmetric_mode_matches_matlab(self):
        data = run_matlab_export(
            "rskelf_symm_s",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                n = 20;
                x = linspace(0,1,n);
                Ad = 1./(1 + abs(reshape(x,[],1) - reshape(x,1,[]))) + 3*eye(n);
                X = reshape(sin((1:(3*n))/17), n, 3);
                F = rskelf(Ad,x,2,1e-10,[],struct('symm','s'));
                Ymv = rskelf_mv(F,X);
                Ysv = rskelf_sv(F,X);
                ld = rskelf_logdet(F);
                nfactors = length(F.factors);
                save('__OUT__','Ad','X','Ymv','Ysv','ld','nfactors');
                exit;
                """
            ),
        )

        x = np.linspace(0.0, 1.0, 20).reshape(1, -1)
        F = rskelf(data["Ad"], x, occ=2, rank_or_tol=1e-10, opts={"symm": "s"})

        self.assertEqual(len(F.factors), int(data["nfactors"].ravel()[0]))
        np.testing.assert_allclose(rskelf_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        self.assertLess(logdet_mod_error(rskelf_logdet(F), data["ld"].item()), 1e-9)

    def test_hermitian_mode_matches_matlab(self):
        data = run_matlab_export(
            "rskelf_symm_h",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                n = 20;
                x = linspace(0,1,n);
                dx = reshape(x,[],1) - reshape(x,1,[]);
                Ad = 1./(1 + abs(dx)) + 3*eye(n);
                X = reshape(sin((1:(3*n))/17), n, 3) ...
                    + 1i*reshape(cos((1:(3*n))/19), n, 3);
                F = rskelf(Ad,x,2,1e-10,[],struct('symm','h'));
                Ymv = rskelf_mv(F,X);
                Ysv = rskelf_sv(F,X);
                ld = rskelf_logdet(F);
                nfactors = length(F.factors);
                save('__OUT__','Ad','X','Ymv','Ysv','ld','nfactors');
                exit;
                """
            ),
        )

        x = np.linspace(0.0, 1.0, 20).reshape(1, -1)
        F = rskelf(data["Ad"], x, occ=2, rank_or_tol=1e-10, opts={"symm": "h"})

        self.assertEqual(len(F.factors), int(data["nfactors"].ravel()[0]))
        np.testing.assert_allclose(rskelf_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        self.assertLess(logdet_mod_error(rskelf_logdet(F), data["ld"].item()), 1e-9)


if __name__ == "__main__":
    unittest.main()
