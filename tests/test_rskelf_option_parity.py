import os
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from matlab_parity_utils import MATLAB, logdet_mod_error, require_paths, run_matlab_export
from pyflam import (
    rskelf,
    rskelf_cholmv,
    rskelf_cholsv,
    rskelf_logdet,
    rskelf_mv,
    rskelf_partial_info,
    rskelf_sv,
)


_DEFAULT_FLAM_REF = Path(tempfile.gettempdir()) / "flam-reference"
if not _DEFAULT_FLAM_REF.exists():
    _DEFAULT_FLAM_REF = Path(tempfile.gettempdir()) / "FLAM-ref"
FLAM_REF = Path(os.environ.get("FLAM_REFERENCE", _DEFAULT_FLAM_REF))


def _proxy_kernel(target, source, symm):
    target = np.asarray(target).reshape(-1, 1)
    source = np.asarray(source).reshape(1, -1)
    out = 1.0 / (1.0 + np.abs(target - source))
    if symm == "n":
        out = out + 0.05 * (target + 2.0 * source)
    elif symm == "s":
        out = out + 0.05 * (target + source)
    return out


def _proxy_case_callbacks(coords, symm):
    calls = {"matrix": 0, "proxy": 0}

    def Afun(i, j):
        calls["matrix"] += 1
        i = np.asarray(i, dtype=np.int64).reshape(-1)
        j = np.asarray(j, dtype=np.int64).reshape(-1)
        out = _proxy_kernel(coords[i], coords[j], symm)
        return out + 3.0 * (i[:, None] == j[None, :])

    def pxyfun(x, slf, nbr, l, ctr):
        calls["proxy"] += 1
        slf = np.asarray(slf, dtype=np.int64).reshape(-1)
        nbr = np.asarray(nbr, dtype=np.int64).reshape(-1)
        width = float(np.asarray(l).reshape(-1)[0])
        center = float(np.asarray(ctr).reshape(-1)[0])
        proxy = center + width * np.array([-1.75, -1.25, 1.25, 1.75])
        nbr = nbr[np.abs(coords[nbr] - center) <= 1.25 * width]
        out = _proxy_kernel(proxy, coords[slf], symm)
        if symm == "n":
            reverse = _proxy_kernel(coords[slf], proxy, symm)
            out = np.vstack((out, reverse.T))
        return out, nbr

    return Afun, pxyfun, calls


class RSkelfOptionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_paths(MATLAB, FLAM_REF, label="rskelf option parity")

    def assert_factor_matches_matlab(self, data, symm, n, *, chol=False):
        x = np.linspace(0.0, 1.0, n).reshape(1, -1)
        F = rskelf(data["Ad"], x, occ=2, rank_or_tol=1e-10, opts={"symm": symm})

        self.assertEqual(len(F.factors), int(data["nfactors"].ravel()[0]))
        np.testing.assert_allclose(rskelf_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        if chol:
            np.testing.assert_allclose(rskelf_cholmv(F, data["X"]), data["Ycholmv"], rtol=5e-8, atol=5e-8)
            np.testing.assert_allclose(rskelf_cholsv(F, data["X"]), data["Ycholsv"], rtol=5e-8, atol=5e-8)
        self.assertLess(logdet_mod_error(rskelf_logdet(F), data["ld"].item()), 1e-9)

    def assert_proxy_factor_matches_matlab(self, data, symm, n, *, chol=False):
        coords = np.linspace(0.0, 1.0, n)
        x = coords.reshape(1, -1)
        Afun, pxyfun, calls = _proxy_case_callbacks(coords, symm)
        F = rskelf(Afun, x, occ=2, rank_or_tol=1e-10, pxyfun=pxyfun, opts={"symm": symm})

        self.assertIsNone(F.A_dense)
        self.assertGreater(calls["matrix"], 0)
        self.assertGreater(calls["proxy"], 0)
        self.assertGreater(int(data["proxy_calls"].ravel()[0]), 0)
        self.assertEqual(len(F.factors), int(data["nfactors"].ravel()[0]))
        np.testing.assert_allclose(rskelf_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        if chol:
            np.testing.assert_allclose(rskelf_cholmv(F, data["X"]), data["Ycholmv"], rtol=5e-8, atol=5e-8)
            np.testing.assert_allclose(rskelf_cholsv(F, data["X"]), data["Ycholsv"], rtol=5e-8, atol=5e-8)
        self.assertLess(logdet_mod_error(rskelf_logdet(F), data["ld"].item()), 1e-9)

    def export_proxy_case(self, symm, *, chol=False):
        chol_lines = (
            """
                Ycholmv = rskelf_cholmv(F,X);
                Ycholsv = rskelf_cholsv(F,X);
            """
            if chol
            else """
                Ycholmv = [];
                Ycholsv = [];
            """
        )
        return run_matlab_export(
            f"rskelf_proxy_{symm}",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                global PYFLAM_PROXY_CALLS;
                PYFLAM_PROXY_CALLS = 0;
                n = 24;
                x = linspace(0,1,n);
                symm = '{symm}';
                Afun = @(i,j) Afun_(i,j,x,symm);
                pxyfun = @(x,slf,nbr,l,ctr) pxyfun_(x,slf,nbr,l,ctr,symm);
                X = reshape(sin((1:(2*n))/17), n, 2);
                F = rskelf(Afun,x,2,1e-10,pxyfun,struct('symm',symm));
                Ymv = rskelf_mv(F,X);
                Ysv = rskelf_sv(F,X);
                {textwrap.indent(textwrap.dedent(chol_lines).strip(), "                ")}
                ld = rskelf_logdet(F);
                nfactors = length(F.factors);
                proxy_calls = PYFLAM_PROXY_CALLS;
                save('__OUT__','X','Ymv','Ysv','Ycholmv','Ycholsv','ld','nfactors','proxy_calls');
                exit;

                function K = Afun_(i,j,x,symm)
                  ii = reshape(i,[],1);
                  jj = reshape(j,1,[]);
                  xi = reshape(x(i),[],1);
                  xj = reshape(x(j),1,[]);
                  K = proxy_kernel_(xi,xj,symm);
                  K = K + 3*double(ii == jj);
                end

                function [Kpxy,nbr] = pxyfun_(x,slf,nbr,l,ctr,symm)
                  global PYFLAM_PROXY_CALLS;
                  PYFLAM_PROXY_CALLS = PYFLAM_PROXY_CALLS + 1;
                  proxy = ctr + l*[-1.75 -1.25 1.25 1.75];
                  keep = abs(x(nbr) - ctr) <= 1.25*l;
                  nbr = nbr(keep);
                  Kpxy = proxy_kernel_(reshape(proxy,[],1),reshape(x(slf),1,[]),symm);
                  if symm == 'n'
                    reverse = proxy_kernel_(reshape(x(slf),[],1),reshape(proxy,1,[]),symm);
                    Kpxy = [Kpxy; reverse'];
                  end
                end

                function K = proxy_kernel_(target,source,symm)
                  K = 1./(1 + abs(target - source));
                  if symm == 'n'
                    K = K + 0.05*(target + 2*source);
                  elseif symm == 's'
                    K = K + 0.05*(target + source);
                  end
                end
                """
            ),
        )

    def test_proxy_unsymmetric_mode_matches_matlab(self):
        data = self.export_proxy_case("n")
        self.assert_proxy_factor_matches_matlab(data, "n", 24)

    def test_proxy_symmetric_mode_matches_matlab(self):
        data = self.export_proxy_case("s")
        self.assert_proxy_factor_matches_matlab(data, "s", 24)

    def test_proxy_hermitian_mode_matches_matlab(self):
        data = self.export_proxy_case("h")
        self.assert_proxy_factor_matches_matlab(data, "h", 24)

    def test_proxy_positive_definite_mode_matches_matlab(self):
        data = self.export_proxy_case("p")
        self.assert_proxy_factor_matches_matlab(data, "p", 24)

    def test_callable_stop_matches_matlab_partial_factorization(self):
        data = run_matlab_export(
            "rskelf_callable_stop",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                n = 24;
                x = linspace(0,1,n);
                A = @(i,j) 1./(1 + abs(reshape(x(i),[],1) - reshape(x(j),1,[]))) ...
                    + 2*double(reshape(i,[],1) == reshape(j,1,[]));
                X = reshape((0:(2*n-1))/37,n,2);
                stopfun = @(lvl,l)(lvl >= 2 && l <= 0.26);
                F = rskelf(A,x,2,1e-10,[],struct('symm','n','stop',stopfun));
                Ymv = rskelf_mv(F,X);
                Ysv = rskelf_sv(F,X);
                ld = rskelf_logdet(F);
                if isfield(F,'Si'), sk = F.Si; else, sk = []; end
                if isfield(F,'S'), S = F.S; else, S = sparse(0,0); end
                nfactors = length(F.factors);
                Ad = A(1:n,1:n);
                save('__OUT__','Ad','X','Ymv','Ysv','ld','sk','S','nfactors');
                exit;
                """
            ),
        )

        x = np.linspace(0.0, 1.0, 24).reshape(1, -1)
        stop_calls = []

        def stopfun(lvl, width):
            stop_calls.append((lvl, float(np.asarray(width).reshape(-1)[0])))
            return lvl >= 2 and float(np.asarray(width).reshape(-1)[0]) <= 0.26

        F = rskelf(data["Ad"], x, occ=2, rank_or_tol=1e-10, opts={"symm": "n", "stop": stopfun})
        sk, S = rskelf_partial_info(F)

        self.assertGreater(len(stop_calls), 0)
        self.assertGreater(F.Si.size, 0)
        self.assertEqual(len(F.factors), int(data["nfactors"].ravel()[0]))
        np.testing.assert_array_equal(sk, data["sk"].ravel().astype(np.int64) - 1)
        self.assertEqual(S.shape, data["S"].shape)
        self.assertEqual(S.nnz, data["S"].nnz)
        np.testing.assert_allclose((S - data["S"]).toarray(), 0, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(rskelf_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        self.assertLess(logdet_mod_error(rskelf_logdet(F), data["ld"].item()), 1e-9)

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

    def test_positive_definite_mode_matches_matlab(self):
        data = run_matlab_export(
            "rskelf_symm_p",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                n = 20;
                x = linspace(0,1,n);
                dx = reshape(x,[],1) - reshape(x,1,[]);
                Ad = 1./(1 + abs(dx)) + 3*eye(n);
                X = reshape(sin((1:(2*n))/17), n, 2);
                F = rskelf(Ad,x,2,1e-10,[],struct('symm','p'));
                Ymv = rskelf_mv(F,X);
                Ysv = rskelf_sv(F,X);
                Ycholmv = rskelf_cholmv(F,X);
                Ycholsv = rskelf_cholsv(F,X);
                ld = rskelf_logdet(F);
                nfactors = length(F.factors);
                save('__OUT__','Ad','X','Ymv','Ysv','Ycholmv','Ycholsv','ld','nfactors');
                exit;
                """
            ),
        )

        x = np.linspace(0.0, 1.0, 20).reshape(1, -1)
        F = rskelf(data["Ad"], x, occ=2, rank_or_tol=1e-10, opts={"symm": "p"})

        self.assertEqual(len(F.factors), int(data["nfactors"].ravel()[0]))
        np.testing.assert_allclose(rskelf_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_cholmv(F, data["X"]), data["Ycholmv"], rtol=5e-8, atol=5e-8)
        np.testing.assert_allclose(rskelf_cholsv(F, data["X"]), data["Ycholsv"], rtol=5e-8, atol=5e-8)
        self.assertLess(logdet_mod_error(rskelf_logdet(F), data["ld"].item()), 1e-9)

    def test_complex_unsymmetric_mode_matches_matlab(self):
        data = run_matlab_export(
            "rskelf_complex_n",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                n = 20;
                x = linspace(0,1,n);
                row = reshape(x,[],1);
                col = reshape(x,1,[]);
                dx = row - col;
                Ad = 1./(1 + abs(dx)) + 3*eye(n) + 0.03i*(row + 2*col);
                X = reshape(sin((1:(2*n))/17), n, 2) ...
                    + 1i*reshape(cos((1:(2*n))/19), n, 2);
                F = rskelf(Ad,x,2,1e-10,[],struct('symm','n'));
                Ymv = rskelf_mv(F,X);
                Ysv = rskelf_sv(F,X);
                ld = rskelf_logdet(F);
                nfactors = length(F.factors);
                save('__OUT__','Ad','X','Ymv','Ysv','ld','nfactors');
                exit;
                """
            ),
        )

        self.assert_factor_matches_matlab(data, "n", 20)

    def test_complex_symmetric_mode_matches_matlab(self):
        data = run_matlab_export(
            "rskelf_complex_s",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                n = 20;
                x = linspace(0,1,n);
                row = reshape(x,[],1);
                col = reshape(x,1,[]);
                dx = row - col;
                Ad = 1./(1 + abs(dx)) + 3*eye(n) + 0.03i*(row + col);
                X = reshape(sin((1:(2*n))/17), n, 2) ...
                    + 1i*reshape(cos((1:(2*n))/19), n, 2);
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

        self.assert_factor_matches_matlab(data, "s", 20)

    def test_complex_hermitian_mode_matches_matlab(self):
        data = run_matlab_export(
            "rskelf_complex_h",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                n = 8;
                x = linspace(0,1,n);
                dx = reshape(x,[],1) - reshape(x,1,[]);
                Ad = 1./(1 + abs(dx)) + 3*eye(n) + 0.03i*dx;
                X = reshape(sin((1:(2*n))/17), n, 2) ...
                    + 1i*reshape(cos((1:(2*n))/19), n, 2);
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

        self.assert_factor_matches_matlab(data, "h", 8)

    def test_complex_positive_definite_mode_matches_matlab(self):
        data = run_matlab_export(
            "rskelf_complex_p",
            textwrap.dedent(
                f"""
                addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                n = 8;
                x = linspace(0,1,n);
                dx = reshape(x,[],1) - reshape(x,1,[]);
                Ad = 1./(1 + abs(dx)) + 3*eye(n) + 0.03i*dx;
                X = reshape(sin((1:(2*n))/17), n, 2) ...
                    + 1i*reshape(cos((1:(2*n))/19), n, 2);
                F = rskelf(Ad,x,2,1e-10,[],struct('symm','p'));
                Ymv = rskelf_mv(F,X);
                Ysv = rskelf_sv(F,X);
                Ycholmv = rskelf_cholmv(F,X);
                Ycholsv = rskelf_cholsv(F,X);
                ld = rskelf_logdet(F);
                nfactors = length(F.factors);
                save('__OUT__','Ad','X','Ymv','Ysv','Ycholmv','Ycholsv','ld','nfactors');
                exit;
                """
            ),
        )

        self.assert_factor_matches_matlab(data, "p", 8, chol=True)


if __name__ == "__main__":
    unittest.main()
