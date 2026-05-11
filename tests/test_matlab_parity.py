import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import scipy.io

from pyflam import rskelf, rskelf_mv, rskelf_sv


MATLAB = Path(r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe")
FLAM_REF = Path(os.environ.get("FLAM_REFERENCE", Path(tempfile.gettempdir()) / "flam-reference"))


@unittest.skipUnless(
    os.environ.get("PYFLAM_RUN_MATLAB_PARITY") == "1" and MATLAB.exists() and FLAM_REF.exists(),
    "set PYFLAM_RUN_MATLAB_PARITY=1 with MATLAB and FLAM reference available",
)
class MatlabParityTests(unittest.TestCase):
    def test_rskelf_small_apply_and_solve(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "parity.mat"
            script = Path(tmp) / "run_parity.m"
            script.write_text(
                textwrap.dedent(
                    f"""
                    addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                    n = 8;
                    x = linspace(0,1,n);
                    A = @(i,j) 1./(1 + abs(reshape(x(i),[],1) - reshape(x(j),1,[]))) + 2*(i(:)==j(:)');
                    X = reshape((0:15)/17,8,2);
                    F = rskelf(A,x,3,1e-10,[],struct('symm','n'));
                    Ymv = rskelf_mv(F,X);
                    Ysv = rskelf_sv(F,X);
                    Ad = A(1:n,1:n);
                    save('{str(out).replace("'", "''")}','Ad','X','Ymv','Ysv');
                    exit;
                    """
                )
            )
            subprocess.run(
                [str(MATLAB), "-batch", f"run('{script.as_posix()}')"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
            data = scipy.io.loadmat(out)

        A = data["Ad"]
        X = data["X"]
        x = np.linspace(0.0, 1.0, 8).reshape(1, -1)
        F = rskelf(A, x, 3, 1e-10)
        np.testing.assert_allclose(rskelf_mv(F, X), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_sv(F, X), data["Ysv"], rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
