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


if __name__ == "__main__":
    unittest.main()
