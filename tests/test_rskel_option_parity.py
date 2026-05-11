import os
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from matlab_parity_utils import MATLAB, require_paths, run_matlab_export
from pyflam import rskel, rskel_mv


_DEFAULT_FLAM_REF = Path(tempfile.gettempdir()) / "flam-reference"
if not _DEFAULT_FLAM_REF.exists():
    _DEFAULT_FLAM_REF = Path(tempfile.gettempdir()) / "FLAM-ref"
FLAM_REF = Path(os.environ.get("FLAM_REFERENCE", _DEFAULT_FLAM_REF))


def _rect_kernel(rx, cx):
    rx = np.asarray(rx).reshape(-1, 1)
    cx = np.asarray(cx).reshape(1, -1)
    return 1.0 / (1.0 + np.abs(rx - cx)) + 0.02 * (rx + 2.0 * cx)


class RSkelOptionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_paths(MATLAB, FLAM_REF, label="rskel option parity")

    def test_unsymmetric_callback_matrix_access_matches_matlab(self):
        data = run_matlab_export(
            "rskel_callback_n",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                m = 12;
                n = 10;
                rx = linspace(0,1,m);
                cx = linspace(0.05,0.95,n);
                A = @(i,j) 1./(1 + abs(reshape(rx(i),[],1) - reshape(cx(j),1,[]))) ...
                    + 0.02*(reshape(rx(i),[],1) + 2*reshape(cx(j),1,[]));
                X = reshape((0:(2*n-1))/23,n,2);
                Z = reshape((0:(2*m-1))/29,m,2);
                F = rskel(A,rx,cx,3,1e-10,[],struct('symm','n'));
                Ymv = rskel_mv(F,X);
                Yadj = rskel_mv(F,Z,'c');
                P = F.P;
                Q = F.Q;
                lvpd = F.lvpd;
                lvpu = F.lvpu;
                nd = length(F.D);
                nu = length(F.U);
                save('__OUT__','X','Z','Ymv','Yadj','P','Q','lvpd','lvpu','nd','nu');
                exit;
                """
            ),
        )

        rx = np.linspace(0.0, 1.0, 12)
        cx = np.linspace(0.05, 0.95, 10)
        calls = []

        def Afun(i, j):
            i = np.asarray(i, dtype=np.int64).reshape(-1)
            j = np.asarray(j, dtype=np.int64).reshape(-1)
            calls.append((i.size, j.size))
            return _rect_kernel(rx[i], cx[j])

        F = rskel(Afun, rx.reshape(1, -1), cx.reshape(1, -1), occ=3, rank_or_tol=1e-10)

        self.assertIsNone(F.A_dense)
        self.assertGreater(len(calls), 0)
        np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.Q, data["Q"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.lvpd, data["lvpd"].ravel().astype(np.int64))
        np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
        self.assertEqual(len(F.D), int(data["nd"].ravel()[0]))
        self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
        np.testing.assert_allclose(rskel_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskel_mv(F, data["Z"], trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)

    def test_symmetric_mode_matches_matlab(self):
        data = run_matlab_export(
            "rskel_symm_s",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                n = 14;
                x = linspace(0,1,n);
                row = reshape(x,[],1);
                col = reshape(x,1,[]);
                dx = row - col;
                Ad = 1./(1 + abs(dx)) + 2*eye(n) + 0.02i*(row + col);
                X = reshape(sin((1:(2*n))/17), n, 2) ...
                    + 1i*reshape(cos((1:(2*n))/19), n, 2);
                Z = reshape(cos((1:(2*n))/23), n, 2) ...
                    + 1i*reshape(sin((1:(2*n))/29), n, 2);
                F = rskel(Ad,x,x,3,1e-10,[],struct('symm','s'));
                Ymv = rskel_mv(F,X);
                Yadj = rskel_mv(F,Z,'c');
                P = F.P;
                lvpd = F.lvpd;
                lvpu = F.lvpu;
                nd = length(F.D);
                nu = length(F.U);
                save('__OUT__','Ad','X','Z','Ymv','Yadj','P','lvpd','lvpu','nd','nu');
                exit;
                """
            ),
        )

        x = np.linspace(0.0, 1.0, 14).reshape(1, -1)
        F = rskel(data["Ad"], x, x, occ=3, rank_or_tol=1e-10, opts={"symm": "s"})

        self.assertEqual(F.symm, "s")
        self.assertEqual(F.Q.size, 0)
        np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.lvpd, data["lvpd"].ravel().astype(np.int64))
        np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
        self.assertEqual(len(F.D), int(data["nd"].ravel()[0]))
        self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
        np.testing.assert_allclose(rskel_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskel_mv(F, data["Z"], trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
