import os
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from matlab_parity_utils import MATLAB, require_paths, run_matlab_export
from pyflam import rskel, rskel_mv, rskel_xsp


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


def _line_kernel(x, y):
    return np.abs(np.asarray(x).reshape(-1, 1) - np.asarray(y).reshape(1, -1))


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

    def test_hermitian_mode_matches_matlab(self):
        data = run_matlab_export(
            "rskel_symm_h",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                n = 14;
                x = linspace(0,1,n);
                dx = reshape(x,[],1) - reshape(x,1,[]);
                Ad = 1./(1 + abs(dx)) + 2*eye(n) + 0.02i*dx;
                X = reshape(sin((1:(2*n))/17), n, 2) ...
                    + 1i*reshape(cos((1:(2*n))/19), n, 2);
                Z = reshape(cos((1:(2*n))/23), n, 2) ...
                    + 1i*reshape(sin((1:(2*n))/29), n, 2);
                F = rskel(Ad,x,x,3,1e-10,[],struct('symm','h'));
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
        F = rskel(data["Ad"], x, x, occ=3, rank_or_tol=1e-10, opts={"symm": "h"})

        self.assertEqual(F.symm, "h")
        self.assertEqual(F.Q.size, 0)
        np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.lvpd, data["lvpd"].ravel().astype(np.int64))
        np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
        self.assertEqual(len(F.D), int(data["nd"].ravel()[0]))
        self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
        np.testing.assert_allclose(rskel_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskel_mv(F, data["Z"], trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)

    def test_positive_definite_mode_maps_to_hermitian_and_matches_matlab(self):
        data = run_matlab_export(
            "rskel_symm_p",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                n = 14;
                x = linspace(0,1,n);
                dx = reshape(x,[],1) - reshape(x,1,[]);
                Ad = 1./(1 + abs(dx)) + 2*eye(n) + 0.02i*dx;
                X = reshape(sin((1:(2*n))/17), n, 2) ...
                    + 1i*reshape(cos((1:(2*n))/19), n, 2);
                Z = reshape(cos((1:(2*n))/23), n, 2) ...
                    + 1i*reshape(sin((1:(2*n))/29), n, 2);
                F = rskel(Ad,x,x,3,1e-10,[],struct('symm','p'));
                Ymv = rskel_mv(F,X);
                Yadj = rskel_mv(F,Z,'c');
                P = F.P;
                lvpd = F.lvpd;
                lvpu = F.lvpu;
                nd = length(F.D);
                nu = length(F.U);
                mapped_to_h = strcmp(F.symm,'h');
                save('__OUT__','Ad','X','Z','Ymv','Yadj','P','lvpd','lvpu','nd','nu','mapped_to_h');
                exit;
                """
            ),
        )

        x = np.linspace(0.0, 1.0, 14).reshape(1, -1)
        F = rskel(data["Ad"], x, x, occ=3, rank_or_tol=1e-10, opts={"symm": "p"})

        self.assertEqual(F.symm, "h")
        self.assertEqual(int(data["mapped_to_h"].ravel()[0]), 1)
        self.assertEqual(F.Q.size, 0)
        np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.lvpd, data["lvpd"].ravel().astype(np.int64))
        np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
        self.assertEqual(len(F.D), int(data["nd"].ravel()[0]))
        self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
        np.testing.assert_allclose(rskel_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskel_mv(F, data["Z"], trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)

    def test_complex_rectangular_matches_matlab(self):
        data = run_matlab_export(
            "rskel_complex_rect",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                m = 13;
                n = 9;
                rx = linspace(0,1,m);
                cx = linspace(0.04,0.94,n);
                row = reshape(rx,[],1);
                col = reshape(cx,1,[]);
                dx = row - col;
                Ad = 1./(1 + abs(dx)) + 0.03i*(row + 2*col);
                X = reshape(sin((1:(2*n))/17), n, 2) ...
                    + 1i*reshape(cos((1:(2*n))/19), n, 2);
                Z = reshape(cos((1:(2*m))/23), m, 2) ...
                    + 1i*reshape(sin((1:(2*m))/29), m, 2);
                F = rskel(Ad,rx,cx,3,1e-10,[],struct('symm','n'));
                Ymv = rskel_mv(F,X);
                Yadj = rskel_mv(F,Z,'c');
                P = F.P;
                Q = F.Q;
                lvpd = F.lvpd;
                lvpu = F.lvpu;
                nd = length(F.D);
                nu = length(F.U);
                save('__OUT__','Ad','X','Z','Ymv','Yadj','P','Q','lvpd','lvpu','nd','nu');
                exit;
                """
            ),
        )

        rx = np.linspace(0.0, 1.0, 13).reshape(1, -1)
        cx = np.linspace(0.04, 0.94, 9).reshape(1, -1)
        F = rskel(data["Ad"], rx, cx, occ=3, rank_or_tol=1e-10, opts={"symm": "n"})

        self.assertEqual(F.symm, "n")
        np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.Q, data["Q"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.lvpd, data["lvpd"].ravel().astype(np.int64))
        np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
        self.assertEqual(len(F.D), int(data["nd"].ravel()[0]))
        self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
        np.testing.assert_allclose(rskel_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskel_mv(F, data["Z"], trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)

    def test_proxy_row_and_column_paths_match_matlab(self):
        data = run_matlab_export(
            "rskel_proxy_paths",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                global PYFLAM_RSKEL_ROW_PROXY_CALLS PYFLAM_RSKEL_COL_PROXY_CALLS;
                PYFLAM_RSKEL_ROW_PROXY_CALLS = 0;
                PYFLAM_RSKEL_COL_PROXY_CALLS = 0;
                m = 18;
                n = 15;
                rx = linspace(0,1,m);
                cx = linspace(0.04,0.94,n);
                A = @(i,j) rect_kernel_(reshape(rx(i),[],1),reshape(cx(j),1,[]));
                pxyfun = @(rc,rx,cx,slf,nbr,l,ctr) pxyfun_(rc,rx,cx,slf,nbr,l,ctr);
                X = reshape((0:(2*n-1))/23,n,2);
                Z = reshape((0:(2*m-1))/29,m,2);
                F = rskel(A,rx,cx,3,1e-10,pxyfun,struct('symm','n'));
                Ymv = rskel_mv(F,X);
                Yadj = rskel_mv(F,Z,'c');
                P = F.P;
                Q = F.Q;
                lvpd = F.lvpd;
                lvpu = F.lvpu;
                nd = length(F.D);
                nu = length(F.U);
                row_proxy_calls = PYFLAM_RSKEL_ROW_PROXY_CALLS;
                col_proxy_calls = PYFLAM_RSKEL_COL_PROXY_CALLS;
                save('__OUT__','X','Z','Ymv','Yadj','P','Q','lvpd','lvpu','nd','nu', ...
                     'row_proxy_calls','col_proxy_calls');
                exit;

                function K = rect_kernel_(rx,cx)
                  K = 1./(1 + abs(rx - cx)) + 0.02*(rx + 2*cx);
                end

                function [Kpxy,nbr] = pxyfun_(rc,rx,cx,slf,nbr,l,ctr)
                  global PYFLAM_RSKEL_ROW_PROXY_CALLS PYFLAM_RSKEL_COL_PROXY_CALLS;
                  proxy = ctr + l*[-1.75 -1.25 1.25 1.75];
                  if rc == 'r'
                    PYFLAM_RSKEL_ROW_PROXY_CALLS = PYFLAM_RSKEL_ROW_PROXY_CALLS + 1;
                    keep = abs(cx(nbr) - ctr) <= 1.25*l;
                    nbr = nbr(keep);
                    Kpxy = rect_kernel_(reshape(rx(slf),[],1),reshape(proxy,1,[]));
                  else
                    PYFLAM_RSKEL_COL_PROXY_CALLS = PYFLAM_RSKEL_COL_PROXY_CALLS + 1;
                    keep = abs(rx(nbr) - ctr) <= 1.25*l;
                    nbr = nbr(keep);
                    Kpxy = rect_kernel_(reshape(proxy,[],1),reshape(cx(slf),1,[]));
                  end
                end
                """
            ),
        )

        rx = np.linspace(0.0, 1.0, 18)
        cx = np.linspace(0.04, 0.94, 15)
        Afun, pxyfun, calls = _rect_proxy_callbacks(rx, cx)
        F = rskel(Afun, rx.reshape(1, -1), cx.reshape(1, -1), occ=3, rank_or_tol=1e-10, pxyfun=pxyfun)

        self.assertIsNone(F.A_dense)
        self.assertGreater(calls["matrix"], 0)
        self.assertGreater(calls["row_proxy"], 0)
        self.assertGreater(calls["col_proxy"], 0)
        self.assertGreater(int(data["row_proxy_calls"].ravel()[0]), 0)
        self.assertGreater(int(data["col_proxy_calls"].ravel()[0]), 0)
        np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.Q, data["Q"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.lvpd, data["lvpd"].ravel().astype(np.int64))
        np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
        self.assertEqual(len(F.D), int(data["nd"].ravel()[0]))
        self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
        np.testing.assert_allclose(rskel_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskel_mv(F, data["Z"], trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)

    def test_xsp_symmetric_hermitian_positive_modes_match_matlab(self):
        for symm in ("s", "h", "p"):
            with self.subTest(symm=symm):
                data = run_matlab_export(
                    f"rskel_xsp_{symm}",
                    textwrap.dedent(
                        f"""
                        addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                        n = 14;
                        x = linspace(0,1,n);
                        row = reshape(x,[],1);
                        col = reshape(x,1,[]);
                        dx = row - col;
                        if '{symm}' == 's'
                          Ad = 1./(1 + abs(dx)) + 2*eye(n) + 0.02i*(row + col);
                        else
                          Ad = 1./(1 + abs(dx)) + 2*eye(n) + 0.02i*dx;
                        end
                        F = rskel(Ad,x,x,3,1e-10,[],struct('symm','{symm}'));
                        [S,p,q] = rskel_xsp(F);
                        mapped_to_h = strcmp(F.symm,'h');
                        save('__OUT__','Ad','S','p','q','mapped_to_h');
                        exit;
                        """
                    ),
                )

                x = np.linspace(0.0, 1.0, 14).reshape(1, -1)
                F = rskel(data["Ad"], x, x, occ=3, rank_or_tol=1e-10, opts={"symm": symm})
                S, p, q = rskel_xsp(F)

                if symm == "p":
                    self.assertEqual(F.symm, "h")
                    self.assertEqual(int(data["mapped_to_h"].ravel()[0]), 1)
                else:
                    self.assertEqual(F.symm, symm)
                self.assertEqual(S.shape, data["S"].shape)
                self.assertEqual(S.nnz, data["S"].nnz)
                np.testing.assert_allclose((S - data["S"]).toarray(), 0, rtol=5e-8, atol=5e-8)
                np.testing.assert_array_equal(p, data["p"].ravel().astype(np.int64) - 1)
                np.testing.assert_array_equal(q, data["q"].ravel().astype(np.int64) - 1)

    def test_mv_transpose_modes_match_matlab(self):
        data = run_matlab_export(
            "rskel_mv_trans",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                m = 13;
                n = 9;
                rx = linspace(0,1,m);
                cx = linspace(0.04,0.94,n);
                row = reshape(rx,[],1);
                col = reshape(cx,1,[]);
                dx = row - col;
                Ad = 1./(1 + abs(dx)) + 0.03i*(row + 2*col);
                Xn = reshape(sin((1:(2*n))/17), n, 2) ...
                    + 1i*reshape(cos((1:(2*n))/19), n, 2);
                Xm = reshape(cos((1:(2*m))/23), m, 2) ...
                    + 1i*reshape(sin((1:(2*m))/29), m, 2);
                F = rskel(Ad,rx,cx,3,1e-10,[],struct('symm','n'));
                Yn = rskel_mv(F,Xn,'n');
                Yt = rskel_mv(F,Xm,'t');
                Yc = rskel_mv(F,Xm,'c');
                save('__OUT__','Ad','Xn','Xm','Yn','Yt','Yc');
                exit;
                """
            ),
        )

        rx = np.linspace(0.0, 1.0, 13).reshape(1, -1)
        cx = np.linspace(0.04, 0.94, 9).reshape(1, -1)
        F = rskel(data["Ad"], rx, cx, occ=3, rank_or_tol=1e-10, opts={"symm": "n"})

        np.testing.assert_allclose(rskel_mv(F, data["Xn"], trans="n"), data["Yn"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskel_mv(F, data["Xm"], trans="t"), data["Yt"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskel_mv(F, data["Xm"], trans="c"), data["Yc"], rtol=1e-9, atol=1e-9)

    def test_upstream_mv_line_proxy_case_matches_matlab(self):
        data = run_matlab_export(
            "rskel_upstream_mv_line",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                rng(7,'twister');
                n = 24;
                occ = 4;
                p = 4;
                x = rand(1,n);
                proxy = linspace(1.5,2.5,p);
                proxy = [-proxy proxy];
                Afun = @(i,j) abs(reshape(x(i),[],1) - reshape(x(j),1,[]));
                pxyfun = @(rc,rx,cx,slf,nbr,l,ctr) pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy);
                X = reshape((0:(2*n-1))/31,n,2);
                Z = reshape((0:(2*n-1))/37,n,2);
                F = rskel(Afun,x,x,occ,1e-10,pxyfun,struct('symm','s'));
                Ymv = rskel_mv(F,X,'n');
                Yadj = rskel_mv(F,Z,'c');
                P = F.P;
                lvpd = F.lvpd;
                lvpu = F.lvpu;
                nd = length(F.D);
                nu = length(F.U);
                save('__OUT__','x','X','Z','Ymv','Yadj','P','lvpd','lvpu','nd','nu');
                exit;

                function [Kpxy,nbr] = pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy)
                  pxy = proxy.*l + ctr;
                  if rc == 'r'
                    Kpxy = abs(reshape(rx(slf),[],1) - reshape(pxy,1,[]));
                    dr = cx(nbr) - ctr;
                  else
                    Kpxy = abs(reshape(pxy,[],1) - reshape(cx(slf),1,[]));
                    dr = rx(nbr) - ctr;
                  end
                  nbr = nbr(abs(dr)/l < 1.5);
                end
                """
            ),
        )

        x = data["x"].reshape(-1)
        proxy = np.concatenate((-np.linspace(1.5, 2.5, 4), np.linspace(1.5, 2.5, 4)))
        calls = {"matrix": 0, "proxy": 0}

        def Afun(i, j):
            calls["matrix"] += 1
            i = np.asarray(i, dtype=np.int64).reshape(-1)
            j = np.asarray(j, dtype=np.int64).reshape(-1)
            return _line_kernel(x[i], x[j])

        def pxyfun(rc, rx, cx, slf, nbr, l, ctr):
            calls["proxy"] += 1
            slf = np.asarray(slf, dtype=np.int64).reshape(-1)
            nbr = np.asarray(nbr, dtype=np.int64).reshape(-1)
            width = float(np.asarray(l).reshape(-1)[0])
            center = float(np.asarray(ctr).reshape(-1)[0])
            pxy = proxy * width + center
            if rc == "r":
                out = _line_kernel(x[slf], pxy)
                dr = x[nbr] - center
            else:
                out = _line_kernel(pxy, x[slf])
                dr = x[nbr] - center
            return out, nbr[np.abs(dr) / width < 1.5]

        F = rskel(Afun, x.reshape(1, -1), x.reshape(1, -1), occ=4, rank_or_tol=1e-10, pxyfun=pxyfun, opts={"symm": "s"})

        self.assertIsNone(F.A_dense)
        self.assertGreater(calls["matrix"], 0)
        self.assertGreater(calls["proxy"], 0)
        np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.lvpd, data["lvpd"].ravel().astype(np.int64))
        np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
        self.assertEqual(len(F.D), int(data["nd"].ravel()[0]))
        self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
        np.testing.assert_allclose(rskel_mv(F, data["X"], trans="n"), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskel_mv(F, data["Z"], trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
