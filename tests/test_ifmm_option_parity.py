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


def _rect_kernel(rx, cx):
    rx = np.asarray(rx).reshape(-1, 1)
    cx = np.asarray(cx).reshape(1, -1)
    return 1.0 / (1.0 + np.abs(rx - cx)) + 0.02 * (rx + 2.0 * cx)


def _rect_proxy_callbacks(rx, cx):
    calls = {"matrix": 0, "row_proxy": 0, "col_proxy": 0}

    def Afun(i, j):
        i = np.asarray(i, dtype=np.int64).reshape(-1)
        j = np.asarray(j, dtype=np.int64).reshape(-1)
        calls["matrix"] += 1
        return _rect_kernel(rx[i], cx[j])

    def pxyfun(rc, rx_arg, cx_arg, slf, nbr, l, ctr):
        slf = np.asarray(slf, dtype=np.int64).reshape(-1)
        nbr = np.asarray(nbr, dtype=np.int64).reshape(-1)
        width = float(np.asarray(l).reshape(-1)[0])
        center = float(np.asarray(ctr).reshape(-1)[0])
        proxy = center + width * np.array([-1.75, -1.25, 1.25, 1.75])
        if rc == "r":
            calls["row_proxy"] += 1
            nbr = nbr[np.abs(cx[nbr] - center) <= 1.25 * width]
            return _rect_kernel(rx[slf], proxy), nbr
        calls["col_proxy"] += 1
        nbr = nbr[np.abs(rx[nbr] - center) <= 1.25 * width]
        return _rect_kernel(proxy, cx[slf]), nbr

    return Afun, pxyfun, calls


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

    def test_proxy_callback_paths_match_matlab(self):
        data = run_matlab_export(
            "ifmm_proxy_paths",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                global PYFLAM_IFMM_ROW_PROXY_CALLS PYFLAM_IFMM_COL_PROXY_CALLS;
                PYFLAM_IFMM_ROW_PROXY_CALLS = 0;
                PYFLAM_IFMM_COL_PROXY_CALLS = 0;
                m = 14;
                n = 12;
                rx = linspace(0,1,m);
                cx = linspace(0.04,0.94,n);
                A = @(i,j) rect_kernel_(reshape(rx(i),[],1),reshape(cx(j),1,[]));
                pxyfun = @(rc,rx,cx,slf,nbr,l,ctr) pxyfun_(rc,rx,cx,slf,nbr,l,ctr);
                X = reshape((0:(2*n-1))/23,n,2);
                Z = reshape((0:(2*m-1))/29,m,2);
                F = ifmm(A,rx,cx,3,1e-10,pxyfun,struct('store','a','near',1,'symm','n'));
                Ymv = ifmm_mv(F,X,A);
                Yadj = ifmm_mv(F,Z,A,'c');
                P = F.P;
                Q = F.Q;
                lvpb = F.lvpb;
                lvpu = F.lvpu;
                nb = length(F.B);
                nu = length(F.U);
                row_proxy_calls = PYFLAM_IFMM_ROW_PROXY_CALLS;
                col_proxy_calls = PYFLAM_IFMM_COL_PROXY_CALLS;
                save('__OUT__','X','Z','Ymv','Yadj','P','Q','lvpb','lvpu','nb','nu', ...
                     'row_proxy_calls','col_proxy_calls');
                exit;

                function K = rect_kernel_(rx,cx)
                  K = 1./(1 + abs(rx - cx)) + 0.02*(rx + 2*cx);
                end

                function [Kpxy,nbr] = pxyfun_(rc,rx,cx,slf,nbr,l,ctr)
                  global PYFLAM_IFMM_ROW_PROXY_CALLS PYFLAM_IFMM_COL_PROXY_CALLS;
                  proxy = ctr + l*[-1.75 -1.25 1.25 1.75];
                  if rc == 'r'
                    PYFLAM_IFMM_ROW_PROXY_CALLS = PYFLAM_IFMM_ROW_PROXY_CALLS + 1;
                    keep = abs(cx(nbr) - ctr) <= 1.25*l;
                    nbr = nbr(keep);
                    Kpxy = rect_kernel_(reshape(rx(slf),[],1),reshape(proxy,1,[]));
                  else
                    PYFLAM_IFMM_COL_PROXY_CALLS = PYFLAM_IFMM_COL_PROXY_CALLS + 1;
                    keep = abs(rx(nbr) - ctr) <= 1.25*l;
                    nbr = nbr(keep);
                    Kpxy = rect_kernel_(reshape(proxy,[],1),reshape(cx(slf),1,[]));
                  end
                end
                """
            ),
        )

        rx = np.linspace(0.0, 1.0, 14)
        cx = np.linspace(0.04, 0.94, 12)
        Afun, pxyfun, calls = _rect_proxy_callbacks(rx, cx)
        F = ifmm(Afun, rx.reshape(1, -1), cx.reshape(1, -1), occ=3, rank_or_tol=1e-10, pxyfun=pxyfun, opts={"store": "a", "near": 1, "symm": "n"})

        self.assertGreater(calls["matrix"], 0)
        self.assertGreater(calls["row_proxy"], 0)
        self.assertGreater(calls["col_proxy"], 0)
        self.assertGreater(int(data["row_proxy_calls"].ravel()[0]), 0)
        self.assertGreater(int(data["col_proxy_calls"].ravel()[0]), 0)
        np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.Q, data["Q"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.lvpb, data["lvpb"].ravel().astype(np.int64))
        np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
        self.assertEqual(len(F.B), int(data["nb"].ravel()[0]))
        self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
        np.testing.assert_allclose(ifmm_mv(F, data["X"], Afun), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(ifmm_mv(F, data["Z"], Afun, trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
