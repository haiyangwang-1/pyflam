import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import scipy.io

from pyflam import (
    hypoct,
    hypoct_perm,
    id,
    ifmm,
    ifmm_mv,
    rskel,
    rskel_mv,
    rskel_xsp,
    rskelf,
    rskelf_logdet,
    rskelf_mv,
    rskelf_sv,
)


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
    def test_hypoct_layout_and_permutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hypoct_parity.mat"
            script = Path(tmp) / "run_hypoct_parity.m"
            script.write_text(
                textwrap.dedent(
                    f"""
                    addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                    x = [0.1 0.2 0.9 0.8 0.45 0.55 0.12 0.88;
                         0.1 0.85 0.2 0.75 0.52 0.48 0.9 0.05];
                    t = hypoct(x,2,Inf,[]);
                    p = hypoct_perm(t);
                    leaf_counts = zeros(1,length(t.nodes));
                    nbor_counts = zeros(1,length(t.nodes));
                    chld_counts = zeros(1,length(t.nodes));
                    for k = 1:length(t.nodes)
                      leaf_counts(k) = length(t.nodes(k).xi);
                      nbor_counts(k) = length(t.nodes(k).nbor);
                      chld_counts(k) = length(t.nodes(k).chld);
                    end
                    nlvl = t.nlvl;
                    lvp = t.lvp;
                    l = t.l;
                    save('{str(out).replace("'", "''")}','x','p','nlvl','lvp','l','leaf_counts','nbor_counts','chld_counts');
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

        tree = hypoct(data["x"], 2)
        np.testing.assert_array_equal(hypoct_perm(tree), data["p"].ravel().astype(np.int64) - 1)
        self.assertEqual(tree.nlvl, int(data["nlvl"].ravel()[0]))
        np.testing.assert_array_equal(tree.lvp, data["lvp"].ravel().astype(np.int64))
        np.testing.assert_allclose(tree.l, data["l"])
        np.testing.assert_array_equal([node.xi.size for node in tree.nodes], data["leaf_counts"].ravel())
        np.testing.assert_array_equal([len(node.nbor) for node in tree.nodes], data["nbor_counts"].ravel())
        np.testing.assert_array_equal([len(node.chld) for node in tree.nodes], data["chld_counts"].ravel())

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

    def test_rskel_apply_and_extended_sparse(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rskel_parity.mat"
            script = Path(tmp) / "run_rskel_parity.m"
            script.write_text(
                textwrap.dedent(
                    f"""
                    addpath(genpath('{str(FLAM_REF).replace("'", "''")}'));
                    n = 9;
                    rx = linspace(0,1,n);
                    cx = linspace(0.05,0.95,n-1);
                    A = @(i,j) 1./(1 + abs(reshape(rx(i),[],1) - reshape(cx(j),1,[])));
                    X = reshape((0:15)/17,n-1,2);
                    Z = reshape((0:17)/19,n,2);
                    F = rskel(A,rx,cx,3,1e-10,[],struct('symm','n'));
                    Ymv = rskel_mv(F,X);
                    Yadj = rskel_mv(F,Z,'c');
                    [S,p,q] = rskel_xsp(F);
                    P = F.P;
                    Q = F.Q;
                    lvpd = F.lvpd;
                    lvpu = F.lvpu;
                    nd = length(F.D);
                    nu = length(F.U);
                    Ad = A(1:n,1:n-1);
                    save('{str(out).replace("'", "''")}','Ad','X','Z','Ymv','Yadj','P','Q','lvpd','lvpu','nd','nu','S','p','q');
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

        rx = np.linspace(0.0, 1.0, 9).reshape(1, -1)
        cx = np.linspace(0.05, 0.95, 8).reshape(1, -1)
        F = rskel(data["Ad"], rx, cx, 3, 1e-10)
        S, p, q = rskel_xsp(F)

        np.testing.assert_array_equal(F.P, data["P"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.Q, data["Q"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(F.lvpd, data["lvpd"].ravel().astype(np.int64))
        np.testing.assert_array_equal(F.lvpu, data["lvpu"].ravel().astype(np.int64))
        self.assertEqual(len(F.D), int(data["nd"].ravel()[0]))
        self.assertEqual(len(F.U), int(data["nu"].ravel()[0]))
        np.testing.assert_allclose(rskel_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskel_mv(F, data["Z"], trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)
        self.assertEqual(S.shape, data["S"].shape)
        self.assertEqual(S.nnz, data["S"].nnz)
        np.testing.assert_allclose((S - data["S"]).toarray(), 0, rtol=1e-9, atol=1e-9)
        np.testing.assert_array_equal(p, data["p"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(q, data["q"].ravel().astype(np.int64) - 1)

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
