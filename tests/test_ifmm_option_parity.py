import os
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from matlab_parity_utils import MATLAB, require_paths, run_matlab_export
from pyflam import ifmm, ifmm_mv


_DEFAULT_FLAM_REF = Path(tempfile.gettempdir()) / "flam-reference"
if not _DEFAULT_FLAM_REF.exists():
    _DEFAULT_FLAM_REF = Path(tempfile.gettempdir()) / "FLAM-ref"
FLAM_REF = Path(os.environ.get("FLAM_REFERENCE", _DEFAULT_FLAM_REF))


class IFMMOptionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_paths(MATLAB, FLAM_REF, label="ifmm option parity")

    def test_store_modes_match_matlab(self):
        for store in ("n", "s", "r", "a"):
            with self.subTest(store=store):
                data = run_matlab_export(
                    f"ifmm_store_{store}",
                    textwrap.dedent(
                        f"""
                        addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                        m = 11;
                        n = 9;
                        rx = linspace(0,1,m);
                        cx = linspace(0.04,0.94,n);
                        A = @(i,j) 1./(1 + abs(reshape(rx(i),[],1) - reshape(cx(j),1,[]))) ...
                            + 0.02*(reshape(rx(i),[],1) + 2*reshape(cx(j),1,[]));
                        Ad = A(1:m,1:n);
                        X = reshape((0:(2*n-1))/23,n,2);
                        Z = reshape((0:(2*m-1))/29,m,2);
                        F = ifmm(Ad,rx,cx,3,1e-10,[],struct('store','{store}','near',1,'symm','n'));
                        Ymv = ifmm_mv(F,X,Ad);
                        Yadj = ifmm_mv(F,Z,Ad,'c');
                        P = F.P;
                        Q = F.Q;
                        lvpb = F.lvpb;
                        lvpu = F.lvpu;
                        nb = length(F.B);
                        nu = length(F.U);
                        save('__OUT__','Ad','X','Z','Ymv','Yadj','P','Q','lvpb','lvpu','nb','nu');
                        exit;
                        """
                    ),
                )

                rx = np.linspace(0.0, 1.0, 11).reshape(1, -1)
                cx = np.linspace(0.04, 0.94, 9).reshape(1, -1)
                F = ifmm(data["Ad"], rx, cx, occ=3, rank_or_tol=1e-10, opts={"store": store, "near": 1, "symm": "n"})

                self.assertEqual(F.store, store)
                np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
                np.testing.assert_array_equal(F.Q, data["Q"].ravel().astype(np.int64) - 1)
                np.testing.assert_array_equal(F.lvpb, data["lvpb"].ravel().astype(np.int64))
                np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
                self.assertEqual(len(F.B), int(data["nb"].ravel()[0]))
                self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
                np.testing.assert_allclose(ifmm_mv(F, data["X"], data["Ad"]), data["Ymv"], rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(ifmm_mv(F, data["Z"], data["Ad"], trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)

    def test_near_modes_match_matlab(self):
        for near in (0, 1):
            with self.subTest(near=near):
                data = run_matlab_export(
                    f"ifmm_near_{near}",
                    textwrap.dedent(
                        f"""
                        addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                        m = 11;
                        n = 9;
                        rx = linspace(0,1,m);
                        cx = linspace(0.04,0.94,n);
                        Ad = 1./(1 + abs(reshape(rx,[],1) - reshape(cx,1,[]))) ...
                            + 0.02*(reshape(rx,[],1) + 2*reshape(cx,1,[]));
                        X = reshape((0:(2*n-1))/23,n,2);
                        Z = reshape((0:(2*m-1))/29,m,2);
                        F = ifmm(Ad,rx,cx,3,1e-10,[],struct('store','a','near',{near},'symm','n'));
                        Ymv = ifmm_mv(F,X);
                        Yadj = ifmm_mv(F,Z,[],'c');
                        P = F.P;
                        Q = F.Q;
                        lvpb = F.lvpb;
                        lvpu = F.lvpu;
                        nb = length(F.B);
                        nu = length(F.U);
                        save('__OUT__','Ad','X','Z','Ymv','Yadj','P','Q','lvpb','lvpu','nb','nu');
                        exit;
                        """
                    ),
                )

                rx = np.linspace(0.0, 1.0, 11).reshape(1, -1)
                cx = np.linspace(0.04, 0.94, 9).reshape(1, -1)
                F = ifmm(data["Ad"], rx, cx, occ=3, rank_or_tol=1e-10, opts={"store": "a", "near": near, "symm": "n"})

                self.assertEqual(F.opts["near"], near)
                np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
                np.testing.assert_array_equal(F.Q, data["Q"].ravel().astype(np.int64) - 1)
                np.testing.assert_array_equal(F.lvpb, data["lvpb"].ravel().astype(np.int64))
                np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
                self.assertEqual(len(F.B), int(data["nb"].ravel()[0]))
                self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
                np.testing.assert_allclose(ifmm_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(ifmm_mv(F, data["Z"], trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)

    def test_symmetry_modes_match_matlab(self):
        for symm in ("n", "s", "h", "p"):
            with self.subTest(symm=symm):
                data = run_matlab_export(
                    f"ifmm_symm_{symm}",
                    textwrap.dedent(
                        f"""
                        addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                        n = 11;
                        x = linspace(0,1,n);
                        row = reshape(x,[],1);
                        col = reshape(x,1,[]);
                        dx = row - col;
                        if '{symm}' == 'n'
                          Ad = 1./(1 + abs(dx)) + 0.02*(row + 2*col);
                        else
                          Ad = 1./(1 + abs(dx)) + 0.02*(row + col);
                        end
                        X = reshape((0:(2*n-1))/23,n,2);
                        Z = reshape((0:(2*n-1))/29,n,2);
                        F = ifmm(Ad,x,x,3,1e-10,[],struct('store','a','near',1,'symm','{symm}'));
                        Ymv = ifmm_mv(F,X);
                        Yadj = ifmm_mv(F,Z,[],'c');
                        P = F.P;
                        Q = F.Q;
                        lvpb = F.lvpb;
                        lvpu = F.lvpu;
                        nb = length(F.B);
                        nu = length(F.U);
                        mapped_to_h = strcmp(F.symm,'h');
                        save('__OUT__','Ad','X','Z','Ymv','Yadj','P','Q','lvpb','lvpu','nb','nu','mapped_to_h');
                        exit;
                        """
                    ),
                )

                x = np.linspace(0.0, 1.0, 11).reshape(1, -1)
                F = ifmm(data["Ad"], x, x, occ=3, rank_or_tol=1e-10, opts={"store": "a", "near": 1, "symm": symm})

                if symm == "p":
                    self.assertEqual(F.symm, "h")
                    self.assertEqual(int(data["mapped_to_h"].ravel()[0]), 1)
                else:
                    self.assertEqual(F.symm, symm)
                np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
                if symm == "n":
                    np.testing.assert_array_equal(F.Q, data["Q"].ravel().astype(np.int64) - 1)
                else:
                    self.assertEqual(F.Q.size, 0)
                    self.assertEqual(data["Q"].size, 0)
                np.testing.assert_array_equal(F.lvpb, data["lvpb"].ravel().astype(np.int64))
                np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
                self.assertEqual(len(F.B), int(data["nb"].ravel()[0]))
                self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
                np.testing.assert_allclose(ifmm_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(ifmm_mv(F, data["Z"], trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
