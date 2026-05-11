import os
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.special

from matlab_parity_utils import (
    MATLAB,
    logdet_mod_error as _logdet_mod_error,
    matlab_path,
    relerr as _relerr,
    require_paths,
    run_matlab_export,
)
from pyflam import (
    hypoct,
    hypoct_perm,
    hifde2,
    hifde_diag,
    hifde_logdet,
    hifde_mv,
    hifde_sv,
    hifie_id,
    hifie_idx,
    id,
    ifmm,
    ifmm_mv,
    mf2,
    mf_diag,
    mf_logdet,
    mf_mv,
    mf_sv,
    rskel,
    rskel_mv,
    rskel_xsp,
    rskelf,
    rskelf_logdet,
    rskelf_mv,
    rskelf_partial_info,
    rskelf_sv,
)


_DEFAULT_FLAM_REF = Path(tempfile.gettempdir()) / "flam-reference"
if not _DEFAULT_FLAM_REF.exists():
    _DEFAULT_FLAM_REF = Path(tempfile.gettempdir()) / "FLAM-ref"
FLAM_REF = Path(os.environ.get("FLAM_REFERENCE", _DEFAULT_FLAM_REF))
CHUNKIE_REF = Path(os.environ.get("CHUNKIE_REFERENCE", Path(r"C:\Users\haiya\git\chunkie")))


def _run_flam_export(name: str, body: str, timeout: int = 120):
    return run_matlab_export(
        name,
        textwrap.dedent(
            f"""
            addpath(genpath('{matlab_path(FLAM_REF)}'));
            {body}
            """
        ),
        timeout=timeout,
    )


class MatlabParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_paths(MATLAB, FLAM_REF, label="MATLAB parity tests")

    def test_hypoct_layout_and_permutation(self):
        data = _run_flam_export(
            "hypoct_parity",
            """
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
                    save('__OUT__','x','p','nlvl','lvp','l','leaf_counts','nbor_counts','chld_counts');
                    exit;
            """,
        )

        tree = hypoct(data["x"], 2)
        np.testing.assert_array_equal(hypoct_perm(tree), data["p"].ravel().astype(np.int64) - 1)
        self.assertEqual(tree.nlvl, int(data["nlvl"].ravel()[0]))
        np.testing.assert_array_equal(tree.lvp, data["lvp"].ravel().astype(np.int64))
        np.testing.assert_allclose(tree.l, data["l"])
        np.testing.assert_array_equal([node.xi.size for node in tree.nodes], data["leaf_counts"].ravel())
        np.testing.assert_array_equal([len(node.nbor) for node in tree.nodes], data["nbor_counts"].ravel())
        np.testing.assert_array_equal([len(node.chld) for node in tree.nodes], data["chld_counts"].ravel())

    def test_id_fixed_columns(self):
        data = _run_flam_export(
            "id_parity",
            """
                    A = [1 2 3 4 5 6;
                         0 1 1 2 3 5;
                         2 1 0 1 0 2;
                         3 5 8 13 21 34;
                         1 -1 2 -2 3 -3] / 17;
                    [sk,rd,T,niter] = id(A,3+1e-12,2,Inf,[2 5]);
                    save('__OUT__','A','sk','rd','T','niter');
                    exit;
            """,
        )

        sk, rd, T, niter = id(data["A"], 3 + 1e-12, 2, np.inf, fixed=[1, 4], return_niter=True)
        np.testing.assert_array_equal(sk, data["sk"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(rd, data["rd"].ravel().astype(np.int64) - 1)
        np.testing.assert_allclose(T, data["T"], rtol=1e-12, atol=1e-12)
        self.assertEqual(niter, int(data["niter"].ravel()[0]))

    def test_hifie_compression_callbacks(self):
        data = _run_flam_export(
            "hifie_callbacks",
            """
                    K = [1 2 1 0 3 1;
                         0 1 2 1 1 0;
                         2 0 1 3 0 1;
                         1 1 0 2 2 1] / 11;
                    K1 = K + 0.1*eye(4,6);
                    K2 = [1 0 1 0 0 0;
                          0 1 0 1 0 0;
                          1 0 1 0 0 0;
                          0 1 0 1 0 0];
                    [sk1,rd1,T1] = hifie_id(K,K1,K2,3,2,Inf);
                    [sk2,rd2,T2] = hifie_idx(K,K1,K2,3,2,Inf);
                    save('__OUT__','K','K1','K2','sk1','rd1','T1','sk2','rd2','T2');
                    exit;
            """,
        )

        sk, rd, T = hifie_id(data["K"], data["K1"], data["K2"], 3, 2, np.inf)
        np.testing.assert_array_equal(sk, data["sk1"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(rd, data["rd1"].ravel().astype(np.int64) - 1)
        np.testing.assert_allclose(T, data["T1"], rtol=1e-12, atol=1e-12)

        sk, rd, T = hifie_idx(data["K"], data["K1"], data["K2"], 3, 2, np.inf)
        np.testing.assert_array_equal(sk, data["sk2"].ravel().astype(np.int64) - 1)
        np.testing.assert_array_equal(rd, data["rd2"].ravel().astype(np.int64) - 1)
        np.testing.assert_allclose(T, data["T2"], rtol=1e-12, atol=1e-12)

    def test_rskelf_small_apply_and_solve(self):
        data = _run_flam_export(
            "parity",
            """
                    n = 8;
                    x = linspace(0,1,n);
                    A = @(i,j) 1./(1 + abs(reshape(x(i),[],1) - reshape(x(j),1,[]))) + 2*(i(:)==j(:)');
                    X = reshape((0:15)/17,8,2);
                    F = rskelf(A,x,3,1e-10,[],struct('symm','n'));
                    Ymv = rskelf_mv(F,X);
                    Ysv = rskelf_sv(F,X);
                    Ad = A(1:n,1:n);
                    save('__OUT__','Ad','X','Ymv','Ysv');
                    exit;
            """,
        )

        A = data["Ad"]
        X = data["X"]
        x = np.linspace(0.0, 1.0, 8).reshape(1, -1)
        F = rskelf(A, x, 3, 1e-10)
        np.testing.assert_allclose(rskelf_mv(F, X), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_sv(F, X), data["Ysv"], rtol=1e-9, atol=1e-9)

    def test_rskelf_partial_logdet(self):
        data = _run_flam_export(
            "partial_logdet",
            """
                    n = 18;
                    x = linspace(0,1,n);
                    A = @(i,j) 1./(1 + abs(reshape(x(i),[],1) - reshape(x(j),1,[]))) + 2*(i(:)==j(:)');
                    F = rskelf(A,x,3,1e-10,[],struct('symm','n','stop',3));
                    ld = rskelf_logdet(F);
                    Ad = A(1:n,1:n);
                    save('__OUT__','Ad','ld');
                    exit;
            """,
        )

        x = np.linspace(0.0, 1.0, 18).reshape(1, -1)
        F = rskelf(data["Ad"], x, 3, 1e-10, opts={"symm": "n", "stop": 3})
        self.assertGreater(F.Si.size, 0)
        np.testing.assert_allclose(rskelf_logdet(F), data["ld"].ravel()[0], rtol=1e-9, atol=1e-9)

    def test_rskelf_partial_apply_and_solve(self):
        data = _run_flam_export(
            "partial_apply",
            """
                    n = 18;
                    x = linspace(0,1,n);
                    A = @(i,j) 1./(1 + abs(reshape(x(i),[],1) - reshape(x(j),1,[]))) + 2*(i(:)==j(:)');
                    X = reshape((0:35)/37,n,2);
                    F = rskelf(A,x,3,1e-10,[],struct('symm','n','stop',3));
                    Ymv = rskelf_mv(F,X);
                    Ysv = rskelf_sv(F,X);
                    Ad = A(1:n,1:n);
                    save('__OUT__','Ad','X','Ymv','Ysv');
                    exit;
            """,
        )

        x = np.linspace(0.0, 1.0, 18).reshape(1, -1)
        F = rskelf(data["Ad"], x, 3, 1e-10, opts={"symm": "n", "stop": 3})
        self.assertGreater(F.Si.size, 0)
        np.testing.assert_allclose(rskelf_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rskelf_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)

    def test_rskelf_partial_info(self):
        data = _run_flam_export(
            "partial_info",
            """
                    n = 18;
                    x = linspace(0,1,n);
                    A = @(i,j) 1./(1 + abs(reshape(x(i),[],1) - reshape(x(j),1,[]))) + 2*(i(:)==j(:)');
                    F = rskelf(A,x,3,1e-10,[],struct('symm','n','stop',3));
                    % The FLAM reference helper currently has a case typo
                    % (F.si instead of F.Si), so read the fields directly.
                    if isfield(F,'Si'), sk = F.Si; else, sk = []; end
                    if isfield(F,'S'), S = F.S; else, S = sparse(0,0); end
                    Ad = A(1:n,1:n);
                    save('__OUT__','Ad','sk','S');
                    exit;
            """,
        )

        x = np.linspace(0.0, 1.0, 18).reshape(1, -1)
        F = rskelf(data["Ad"], x, 3, 1e-10, opts={"symm": "n", "stop": 3})
        sk, S = rskelf_partial_info(F)
        np.testing.assert_array_equal(sk, data["sk"].ravel().astype(np.int64) - 1)
        self.assertEqual(S.shape, data["S"].shape)
        self.assertEqual(S.nnz, data["S"].nnz)
        np.testing.assert_allclose((S - data["S"]).toarray(), 0, rtol=1e-8, atol=1e-8)

    def test_rskel_apply_and_extended_sparse(self):
        data = _run_flam_export(
            "rskel_parity",
            """
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
                    save('__OUT__','Ad','X','Z','Ymv','Yadj','P','Q','lvpd','lvpu','nd','nu','S','p','q');
                    exit;
            """,
        )

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
        data = _run_flam_export(
            "ifmm_parity",
            """
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
                    save('__OUT__','Ad','X','Z','Ymv','Yadj','P','Q','lvpb','lvpu','nb','nu');
                    exit;
            """,
        )

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

    def test_mf2_grid_operator(self):
        data = _run_flam_export(
            "mf2_parity",
            """
                    n = 4;
                    N = (n-1)^2;
                    e = ones(N,1);
                    A = spdiags([-e 4*e -e],[-3 0 3],N,N);
                    X = reshape((0:(2*N-1))/(2*N+1),N,2);
                    F = mf2(A,n,2,struct('symm','n'));
                    Ymv = mf_mv(F,X);
                    Ysv = mf_sv(F,X);
                    ld = mf_logdet(F);
                    D = mf_diag(F);
                    Di = mf_diag(F,1);
                    save('__OUT__','A','X','Ymv','Ysv','ld','D','Di');
                    exit;
            """,
        )

        F = mf2(data["A"], n=4, occ=2)
        np.testing.assert_allclose(mf_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_logdet(F), data["ld"].ravel()[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_diag(F), data["D"].ravel(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_diag(F, True), data["Di"].ravel(), rtol=1e-9, atol=1e-9)

    def test_mf2_sparse_singular_and_near_singular_modes(self):
        data = _run_flam_export(
            "mf2_sparse_singular_modes",
            """
                    A0 = sparse(0);
                    X0 = 1;
                    warn_state = warning('off','all');
                    F0 = mf2(A0,2,2,struct('symm','n'));
                    Ymv0 = mf_mv(F0,X0);
                    Ysv0 = mf_sv(F0,X0);
                    ld0 = mf_logdet(F0);
                    D0 = mf_diag(F0);
                    Di0 = mf_diag(F0,1);

                    tiny = 1e-14;
                    A1 = spdiags([1;2;tiny;4],0,4,4);
                    X1 = reshape((1:8)/11,4,2);
                    F1 = mf2(A1,3,2,struct('symm','n'));
                    Ymv1 = mf_mv(F1,X1);
                    Ysv1 = mf_sv(F1,X1);
                    ld1 = mf_logdet(F1);
                    D1 = mf_diag(F1);
                    Di1 = mf_diag(F1,1);
                    warning(warn_state);
                    save('__OUT__','A1','X0','Ymv0','Ysv0','ld0','D0','Di0', ...
                         'X1','Ymv1','Ysv1','ld1','D1','Di1');
                    exit;
            """,
        )

        F0 = mf2(sp.csc_matrix((1, 1), dtype=float), n=2, occ=2)
        np.testing.assert_allclose(mf_mv(F0, data["X0"]), data["Ymv0"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(mf_sv(F0, data["X0"]), data["Ysv0"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(mf_logdet(F0), data["ld0"].ravel()[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(mf_diag(F0), data["D0"].ravel(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(mf_diag(F0, True), data["Di0"].ravel(), rtol=1e-12, atol=1e-12)

        F1 = mf2(data["A1"], n=3, occ=2)
        np.testing.assert_allclose(mf_mv(F1, data["X1"]), data["Ymv1"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(mf_sv(F1, data["X1"]), data["Ysv1"], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(mf_logdet(F1), data["ld1"].ravel()[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(mf_diag(F1), data["D1"].ravel(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(mf_diag(F1, True), data["Di1"].ravel(), rtol=1e-8, atol=1e-8)

    def test_hifde2_grid_operator(self):
        data = _run_flam_export(
            "hifde2_parity",
            """
                    n = 4;
                    N = (n-1)^2;
                    e = ones(N,1);
                    A = spdiags([-e 4*e -e],[-3 0 3],N,N);
                    X = reshape((0:(2*N-1))/(2*N+1),N,2);
                    F = hifde2(A,n,2,1e-10,struct('symm','n'));
                    Ymv = hifde_mv(F,X);
                    Ysv = hifde_sv(F,X);
                    ld = hifde_logdet(F);
                    D = hifde_diag(F);
                    Di = hifde_diag(F,1);
                    save('__OUT__','A','X','Ymv','Ysv','ld','D','Di');
                    exit;
            """,
        )

        F = hifde2(data["A"], n=4, occ=2, rank_or_tol=1e-10)
        np.testing.assert_allclose(hifde_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_logdet(F), data["ld"].ravel()[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_diag(F), data["D"].ravel(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_diag(F, True), data["Di"].ravel(), rtol=1e-9, atol=1e-9)


class ChunkIEStyleRSkelfParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_paths(MATLAB, FLAM_REF, CHUNKIE_REF, label="ChunkIE parity tests")

    def test_laplace_dirichlet_starfish_rskelf_callback(self):
        self._run_chunkie_rskelf_case("laplace_d")

    def test_helmholtz_dirichlet_starfish_rskelf_callback(self):
        self._run_chunkie_rskelf_case("helmholtz_d")

    def _run_chunkie_rskelf_case(self, kernel_kind: str):
        data = run_matlab_export(f"chunkie_{kernel_kind}", _chunkie_rskelf_driver_body(kernel_kind), timeout=240)

        op = _ChunkIERSkelfOperator(data, kernel_kind)
        n = data["sys"].shape[0]
        idx = np.arange(n, dtype=np.int64)
        self.assertEqual(np.asarray(data["opdims"]).size, 2)
        self.assertEqual(data["r"].shape, data["dd"].shape)
        self.assertEqual(data["r"].shape, data["d2"].shape)
        self.assertEqual(data["r"].shape, data["nn"].shape)
        dense_from_callback = op(idx, idx)
        np.testing.assert_allclose(_relerr(dense_from_callback, data["sys"]), 0.0, atol=1e-12)
        self.assertGreater(op.spmat.nnz, n)

        F = rskelf(
            op,
            data["xflam"],
            int(data["occ"].item()),
            float(data["tol"].item()),
            pxyfun=op.proxy_callback(data["pr"], data["ptau"], data["pw"]),
        )
        Ymv = rskelf_mv(F, data["X"])
        Ysv = rskelf_sv(F, data["X"])

        self.assertIsNone(F.A_dense)
        self.assertGreater(len(F.factors), 0)
        self.assertGreater(op.proxy_calls, 0)
        np.testing.assert_allclose(_relerr(Ymv, data["Ymv"]), 0.0, atol=1e-10)
        np.testing.assert_allclose(_relerr(Ysv, data["Ysv"]), 0.0, atol=1e-10)
        np.testing.assert_allclose(_relerr(Ymv, data["sys"] @ data["X"]), 0.0, atol=1e-9)
        np.testing.assert_allclose(_relerr(data["sys"] @ Ysv, data["X"]), 0.0, atol=1e-9)
        self.assertLess(_logdet_mod_error(rskelf_logdet(F), data["ld"].item()), 1e-9)


def _chunkie_rskelf_driver_body(kernel_kind: str) -> str:
    if kernel_kind == "laplace_d":
        kernel_setup = "zk = 0; fkern = @(s,t) chnk.lap2d.kern(s,t,'D');"
        rhs_setup = "X = reshape(sin((1:(3*chnkr.npt))/37), chnkr.npt, 3);"
    elif kernel_kind == "helmholtz_d":
        kernel_setup = "zk = 1.1; fkern = @(s,t) chnk.helm2d.kern(zk,s,t,'D');"
        rhs_setup = "X = reshape(exp(1i*(1:(3*chnkr.npt))/41), chnkr.npt, 3);"
    else:
        raise ValueError(f"unknown ChunkIE kernel case: {kernel_kind}")

    return textwrap.dedent(
        f"""
            cd('{matlab_path(CHUNKIE_REF)}');
            addpath('./chunkie');
            addpath(genpath('{matlab_path(FLAM_REF)}'));
            rng(8675309);

            cparams = [];
            cparams.ifclosed = 1;
            cparams.nover = 0;
            pref = [];
            pref.k = 12;
            chnkr = chunkerfuncuni(@(t) starfish(t,3,0.25), 12, cparams, pref);
            {kernel_setup}

            dval = -0.5;
            qopts = [];
            qopts.nonsmoothonly = true;
            qopts.quad = 'ggq';
            qopts.type = 'log';
            qopts.eps = 1e-10;
            spmat = chunkermat(chnkr, fkern, qopts) + dval*speye(chnkr.npt);
            sys = chunkermat(chnkr, fkern) + dval*eye(chnkr.npt);

            r = chnkr.r(:,:);
            dd = chnkr.d(:,:);
            d2 = chnkr.d2(:,:);
            nn = chnkr.n(:,:);
            wts = weights(chnkr);
            wts = wts(:);
            xflam = r;
            occ = 12;
            tol = 1e-10;
            opdims = ones([2,1,1]);
            {rhs_setup}

            matfun = @(i,j) chnk.flam.kernbyindex(i,j,chnkr,wts,fkern,opdims,spmat);
            [pr,ptau,pw,pin] = chnk.flam.proxy_square_pts(64);
            pxyfun = @(x,slf,nbr,l,ctr) chnk.flam.proxyfun(slf,nbr,l,ctr,chnkr,wts, ...
                fkern,opdims,pr,ptau,pw,pin,true);
            F = rskelf(matfun, xflam, occ, tol, pxyfun);
            Ymv = rskelf_mv(F, X);
            Ysv = rskelf_sv(F, X);
            ld = rskelf_logdet(F);

            save('__OUT__','sys','spmat','r','dd','d2','nn','wts','xflam', ...
                 'occ','tol','opdims','X','Ymv','Ysv','ld','pr','ptau','pw','zk','-v7');
            exit;
            """
    )


class _ChunkIERSkelfOperator:
    def __init__(self, data, kernel_kind: str):
        self.r = np.asarray(data["r"])
        self.n = np.asarray(data["nn"])
        self.wts = np.asarray(data["wts"]).reshape(-1)
        self.spmat = data["spmat"].tocsc()
        self.kernel_kind = kernel_kind
        self.zk = float(np.asarray(data["zk"]).reshape(-1)[0])
        self.proxy_calls = 0

    def __call__(self, i, j):
        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)
        mat = self._kernel(self.r[:, j], self.n[:, j], self.r[:, i])
        mat = mat * self.wts[j][None, :]
        corrections = self.spmat[np.ix_(i, j)].tocoo()
        if corrections.nnz:
            mat[corrections.row, corrections.col] = corrections.data
        return mat

    def proxy_callback(self, pr, ptau, pw):
        pr = np.asarray(pr)
        ptau = np.asarray(ptau)
        pw = np.asarray(pw).reshape(-1)

        def pxyfun(x, slf, nbr, l, ctr):
            self.proxy_calls += 1
            slf = np.asarray(slf, dtype=np.int64)
            nbr = np.asarray(nbr, dtype=np.int64)
            lmax = float(np.max(l))
            ctr = np.asarray(ctr).reshape(2, 1)
            pxy = pr * lmax + ctr
            pweights = lmax * pw
            pnorm = np.vstack((-ptau[1, :], ptau[0, :]))
            pnorm = pnorm / np.linalg.norm(pnorm, axis=0)
            if nbr.size:
                inside = np.max(np.abs((self.r[:, nbr] - ctr) / lmax), axis=0) < 1.5
                nbr = nbr[inside]
            src_to_proxy = self._kernel(self.r[:, slf], self.n[:, slf], pxy) * self.wts[slf][None, :]
            proxy_to_src = self._kernel(pxy, pnorm, self.r[:, slf]) * pweights[None, :]
            return np.vstack((src_to_proxy, proxy_to_src.T)), nbr

        return pxyfun

    def _kernel(self, src_r, src_n, targ_r):
        rx = targ_r[0, :, None] - src_r[0, None, :]
        ry = targ_r[1, :, None] - src_r[1, None, :]
        r2 = rx * rx + ry * ry
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.kernel_kind == "laplace_d":
                return (rx * src_n[0, None, :] + ry * src_n[1, None, :]) / (2 * np.pi * r2)
            r = np.sqrt(r2)
            h1 = scipy.special.hankel1(1, self.zk * r)
            grad_x = -0.25j * self.zk * h1 * rx / r
            grad_y = -0.25j * self.zk * h1 * ry / r
            return -(grad_x * src_n[0, None, :] + grad_y * src_n[1, None, :])


if __name__ == "__main__":
    unittest.main()
