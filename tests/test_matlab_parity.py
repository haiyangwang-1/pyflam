import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import scipy.io

from pyflam import id, ifmm, ifmm_mv, rskelf, rskelf_logdet, rskelf_mv, rskelf_sv


MATLAB = Path(r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe")
_DEFAULT_FLAM_REF = Path(tempfile.gettempdir()) / "flam-reference"
if not _DEFAULT_FLAM_REF.exists():
    _DEFAULT_FLAM_REF = Path(tempfile.gettempdir()) / "FLAM-ref"
FLAM_REF = Path(os.environ.get("FLAM_REFERENCE", _DEFAULT_FLAM_REF))


@unittest.skipUnless(
    os.environ.get("PYFLAM_RUN_MATLAB_PARITY") == "1" and MATLAB.exists() and FLAM_REF.exists(),
    "set PYFLAM_RUN_MATLAB_PARITY=1 with MATLAB and FLAM reference available",
)
class MatlabParityTests(unittest.TestCase):
    def test_id_fixed_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "id_parity.mat"
            script = Path(tmp) / "run_id_parity.m"
            script.write_text(
                textwrap.dedent(
                    f"""
                    addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                    A = [1 2 3 4 5 6;
                         0 1 1 2 3 5;
                         2 1 0 1 0 2;
                         3 5 8 13 21 34;
                         1 -1 2 -2 3 -3] / 17;
                    [sk,rd,T,niter] = id(A,3+1e-12,2,Inf,[2 5]);
                    save('{str(out).replace("'", "''")}','A','sk','rd','T','niter');
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

        sk, rd, T, niter = id(data["A"], 3 + 1e-12, 2, np.inf, fixed=[1, 4], return_niter=True)
        np.testing.assert_array_equal(sk, data["sk"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(rd, data["rd"].ravel().astype(np.int64) - 1)
        np.testing.assert_allclose(T, data["T"], rtol=1e-12, atol=1e-12)
        self.assertEqual(niter, int(data["niter"].ravel()[0]))

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

    def test_rskelf_partial_logdet(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "partial_logdet.mat"
            script = Path(tmp) / "run_partial_logdet.m"
            script.write_text(
                textwrap.dedent(
                    f"""
                    addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                    n = 18;
                    x = linspace(0,1,n);
                    A = @(i,j) 1./(1 + abs(reshape(x(i),[],1) - reshape(x(j),1,[]))) + 2*(i(:)==j(:)');
                    F = rskelf(A,x,3,1e-10,[],struct('symm','n','stop',1));
                    ld = rskelf_logdet(F);
                    Ad = A(1:n,1:n);
                    save('{str(out).replace("'", "''")}','Ad','ld');
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

        x = np.linspace(0.0, 1.0, 18).reshape(1, -1)
        F = rskelf(data["Ad"], x, 3, 1e-10, opts={"symm": "n", "stop": 1})
        self.assertGreater(F.Si.size, 0)
        np.testing.assert_allclose(rskelf_logdet(F), data["ld"].ravel()[0], rtol=1e-9, atol=1e-9)

    def test_ifmm_small_apply_and_adjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ifmm_parity.mat"
            script = Path(tmp) / "run_ifmm_parity.m"
            script.write_text(
                textwrap.dedent(
                    f"""
                    addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                    rx = linspace(0,1,9);
                    cx = linspace(0.05,0.95,7);
                    A = @(i,j) 1./(1 + abs(reshape(rx(i),[],1) - reshape(cx(j),1,[])));
                    X = reshape((0:20)/23,7,3);
                    Z = reshape((0:26)/29,9,3);
                    opts = struct('store','a','near',1,'symm','n');
                    F = ifmm(A,rx,cx,3,1e-10,[],opts);
                    Ymv = ifmm_mv(F,X);
                    Yadj = ifmm_mv(F,Z,[],'c');
                    P = F.P;
                    Q = F.Q;
                    lvpb = F.lvpb;
                    lvpu = F.lvpu;
                    nb = length(F.B);
                    nu = length(F.U);
                    Ad = A(1:length(rx),1:length(cx));
                    save('{str(out).replace("'", "''")}','Ad','X','Z','Ymv','Yadj','P','Q','lvpb','lvpu','nb','nu');
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
        Z = data["Z"]
        rx = np.linspace(0.0, 1.0, 9).reshape(1, -1)
        cx = np.linspace(0.05, 0.95, 7).reshape(1, -1)
        F = ifmm(A, rx, cx, 3, 1e-10, opts={"store": "a", "near": 1})

        np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.Q, data["Q"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.lvpb, data["lvpb"].ravel().astype(np.int64))
        np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
        self.assertEqual(len(F.B), int(data["nb"].ravel()[0]))
        self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
        np.testing.assert_allclose(ifmm_mv(F, X), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(ifmm_mv(F, Z, trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
