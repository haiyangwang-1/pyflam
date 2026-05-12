import os
import textwrap
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.special

from matlab_parity_utils import (
    MATLAB,
    default_flam_reference,
    logdet_mod_error as _logdet_mod_error,
    matlab_path,
    relerr as _relerr,
    require_flam_reference,
    require_paths,
    require_pinned_reference,
    run_matlab_export,
)
from pyflam import (
    hypoct,
    hypoct_perm,
    hifde2,
    hifde2x,
    hifde3,
    hifde3x,
    hifde_cholmv,
    hifde_cholsv,
    hifde_diag,
    hifde_logdet,
    hifde_mv,
    hifde_spdiag,
    hifde_sv,
    hifie2,
    hifie2x,
    hifie3,
    hifie3x,
    hifie_cholmv,
    hifie_cholsv,
    hifie_diag,
    hifie_id,
    hifie_idx,
    hifie_logdet,
    hifie_mv,
    hifie_spdiag,
    hifie_sv,
    id,
    ifmm,
    ifmm_mv,
    mf2,
    mf3,
    mf_cholmv,
    mf_cholsv,
    mf_diag,
    mf_logdet,
    mf_mv,
    mf_spdiag,
    mf_sv,
    mfx,
    rskel,
    rskel_mv,
    rskel_xsp,
    rskelf,
    rskelf_logdet,
    rskelf_mv,
    rskelf_partial_info,
    rskelf_sv,
)


FLAM_REF = default_flam_reference()
CHUNKIE_REF = Path(os.environ.get("CHUNKIE_REFERENCE", Path.home() / "git" / "chunkie"))


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
        require_flam_reference(FLAM_REF, label="MATLAB parity tests")

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

    def test_hifie_entry_points_match_matlab(self):
        data = _run_flam_export(
            "hifie_entry_points",
            """
                    [gx,gy] = meshgrid(linspace(0,1,4),linspace(0,1,4));
                    x2 = [gx(:).'; gy(:).'];
                    n2 = size(x2,2);
                    D2 = sqrt((x2(1,:).' - x2(1,:)).^2 + (x2(2,:).' - x2(2,:)).^2);
                    A2 = 1./(1 + D2) + 3*eye(n2);
                    X2 = reshape((0:(2*n2-1))/(2*n2+1),n2,2);

                    F2 = hifie2(A2,x2,4,1e-10,[],struct('symm','n'));
                    Y2 = hifie_mv(F2,X2);
                    Z2 = hifie_sv(F2,X2);
                    ld2 = hifie_logdet(F2);
                    D2 = hifie_diag(F2);
                    Di2 = hifie_diag(F2,1);
                    SD2 = hifie_spdiag(F2);
                    SDi2 = hifie_spdiag(F2,1);
                    lvp2 = F2.lvp; nf2 = length(F2.factors);

                    F2x = hifie2x(A2,x2,4,1e-10,[],struct('symm','n'));
                    Y2x = hifie_mv(F2x,X2,'t');
                    Z2x = hifie_sv(F2x,X2,'t');
                    ld2x = hifie_logdet(F2x);

                    [gx,gy,gz] = ndgrid(linspace(0,1,3),linspace(0,1,3),linspace(0,1,3));
                    x3 = [gx(:).'; gy(:).'; gz(:).'];
                    n3 = size(x3,2);
                    D3 = sqrt((x3(1,:).' - x3(1,:)).^2 + (x3(2,:).' - x3(2,:)).^2 + ...
                              (x3(3,:).' - x3(3,:)).^2);
                    A3 = 1./(1 + D3) + 3*eye(n3);
                    X3 = reshape((0:(2*n3-1))/(2*n3+1),n3,2);
                    opts3 = struct('symm','n','skip',99);

                    F3 = hifie3(A3,x3,4,1e-10,[],opts3);
                    Y3 = hifie_mv(F3,X3);
                    Z3 = hifie_sv(F3,X3);
                    ld3 = hifie_logdet(F3);

                    F3x = hifie3x(A3,x3,4,1e-10,[],opts3);
                    Y3x = hifie_mv(F3x,X3,'t');
                    Z3x = hifie_sv(F3x,X3,'t');
                    ld3x = hifie_logdet(F3x);

                    save('__OUT__','A2','A3','x2','x3','X2','X3','Y2','Z2','ld2','D2','Di2','SD2','SDi2','lvp2','nf2', ...
                         'Y2x','Z2x','ld2x','Y3','Z3','ld3','Y3x','Z3x','ld3x');
                    exit;
            """,
            timeout=180,
        )

        F2 = hifie2(data["A2"], data["x2"], occ=4, rank_or_tol=1e-10)
        self.assertIsNone(F2.backend.A_dense)
        np.testing.assert_array_equal(F2.lvp, data["lvp2"].ravel().astype(np.int64))
        self.assertEqual(len(F2.factors), int(data["nf2"].ravel()[0]))
        np.testing.assert_allclose(hifie_mv(F2, data["X2"]), data["Y2"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifie_sv(F2, data["X2"]), data["Z2"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifie_logdet(F2), data["ld2"].ravel()[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifie_diag(F2), data["D2"].ravel(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifie_diag(F2, True), data["Di2"].ravel(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifie_spdiag(F2), data["SD2"].ravel(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifie_spdiag(F2, True), data["SDi2"].ravel(), rtol=1e-9, atol=1e-9)

        F2x = hifie2x(data["A2"], data["x2"], occ=4, rank_or_tol=1e-10)
        np.testing.assert_allclose(hifie_mv(F2x, data["X2"], trans="t"), data["Y2x"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifie_sv(F2x, data["X2"], trans="t"), data["Z2x"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifie_logdet(F2x), data["ld2x"].ravel()[0], rtol=1e-9, atol=1e-9)

        F3 = hifie3(data["A3"], data["x3"], occ=4, rank_or_tol=1e-10, opts={"skip": 99})
        np.testing.assert_allclose(hifie_mv(F3, data["X3"]), data["Y3"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifie_sv(F3, data["X3"]), data["Z3"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifie_logdet(F3), data["ld3"].ravel()[0], rtol=1e-9, atol=1e-9)

        F3x = hifie3x(data["A3"], data["x3"], occ=4, rank_or_tol=1e-10, opts={"skip": 99})
        np.testing.assert_allclose(hifie_mv(F3x, data["X3"], trans="t"), data["Y3x"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifie_sv(F3x, data["X3"], trans="t"), data["Z3x"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifie_logdet(F3x), data["ld3x"].ravel()[0], rtol=1e-9, atol=1e-9)

    def test_hifie_covariance_proxy_matches_matlab(self):
        data = _run_flam_export(
            "hifie_covariance_proxy",
            """
                    n = 8; occ = 4; p = 4; rank_or_tol = 1e-8;
                    Tmax = 2; skip = 1; symm = 'p'; noise = 1e-1; scale = 3;
                    [x1,x2] = ndgrid((1:n)/n); x = [x1(:) x2(:)]';
                    N = size(x,2);
                    theta = (1:p)*2*pi/p; proxy_ = [cos(theta); sin(theta)];
                    proxy_ = proxy_./max(abs(proxy_));
                    R = 3/scale;
                    proxy = reshape(proxy_(:)*linspace(0,R,p),2,p,p);
                    shift = 1.5*proxy_;

                    Afun = @(i,j)Afun_cov(i,j,x,noise,scale);
                    pxyfun = @(x,slf,nbr,l,ctr)pxyfun_cov(x,slf,nbr,l,ctr,proxy,shift,scale);
                    opts = struct('Tmax',Tmax,'skip',skip,'symm',symm);
                    F = hifie2(Afun,x,occ,rank_or_tol,pxyfun,opts);
                    A = Afun(1:N,1:N);
                    X = reshape((0:(2*N-1))/(2*N+1),N,2);
                    Y = hifie_mv(F,X);
                    Z = hifie_sv(F,X);
                    C = hifie_cholmv(F,hifie_cholmv(F,X,'c'));
                    W = hifie_cholsv(F,hifie_cholsv(F,X),'c');
                    ld = hifie_logdet(F);
                    lvp = F.lvp; nf = length(F.factors);
                    save('__OUT__','A','x','X','Y','Z','C','W','ld','lvp','nf', ...
                         'proxy','shift','noise','scale');
                    exit;

                    function K = Kfun_cov(x,y,scale)
                      dx = x(1,:)' - y(1,:);
                      dy = x(2,:)' - y(2,:);
                      dr = scale*sqrt(dx.^2 + dy.^2);
                      K = exp(-0.5*dr.^2);
                    end

                    function A = Afun_cov(i,j,x,noise,scale)
                      A = Kfun_cov(x(:,i),x(:,j),scale);
                      [I,J] = ndgrid(i,j);
                      A(I == J) = A(I == J) + noise^2;
                    end

                    function [Kpxy,nbr] = pxyfun_cov(x,slf,nbr,l,ctr,proxy,shift,scale)
                      pxy = proxy + shift.*l + ctr;
                      Kpxy = Kfun_cov(pxy,x(:,slf),scale);
                      nbr = nbr(max(abs(x(:,nbr) - ctr)./l) < 1.5);
                    end
            """,
            timeout=180,
        )

        x = data["x"]
        proxy = data["proxy"]
        shift = data["shift"]
        noise = float(data["noise"].ravel()[0])
        scale = float(data["scale"].ravel()[0])

        def cov_kernel(src, dst):
            diff = src[:, :, None] - dst[:, None, :]
            dr = scale * np.linalg.norm(diff, axis=0)
            return np.exp(-0.5 * dr**2)

        def afun(i, j):
            i = np.asarray(i, dtype=np.int64)
            j = np.asarray(j, dtype=np.int64)
            out = cov_kernel(x[:, i], x[:, j])
            same = i[:, None] == j[None, :]
            out[same] += noise**2
            return out

        def pxyfun(all_x, slf, nbr, l, ctr):
            slf = np.asarray(slf, dtype=np.int64)
            nbr = np.asarray(nbr, dtype=np.int64)
            l = np.asarray(l).reshape(-1)
            ctr = np.asarray(ctr).reshape(-1)
            pxy = proxy + (shift * l[:, None])[:, :, None] + ctr[:, None, None]
            pxy = pxy.reshape((x.shape[0], -1), order="F")
            if nbr.size:
                keep = np.max(np.abs((all_x[:, nbr] - ctr[:, None]) / l[:, None]), axis=0) < 1.5
                nbr = nbr[keep]
            return cov_kernel(pxy, all_x[:, slf]), nbr

        F = hifie2(afun, x, occ=4, rank_or_tol=1e-8, pxyfun=pxyfun, opts={"symm": "p", "skip": 1})
        self.assertIsNone(F.backend.A_dense)
        np.testing.assert_array_equal(F.lvp, data["lvp"].ravel().astype(np.int64))
        self.assertEqual(len(F.factors), int(data["nf"].ravel()[0]))
        np.testing.assert_allclose(afun(np.arange(x.shape[1]), np.arange(x.shape[1])), data["A"])
        np.testing.assert_allclose(hifie_mv(F, data["X"]), data["Y"], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(hifie_sv(F, data["X"]), data["Z"], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(hifie_cholmv(F, hifie_cholmv(F, data["X"], "c")), data["C"], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(hifie_cholsv(F, hifie_cholsv(F, data["X"]), "c"), data["W"], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(hifie_logdet(F), data["ld"].ravel()[0], rtol=1e-8, atol=1e-8)

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
                    SD = mf_spdiag(F);
                    SDi = mf_spdiag(F,1);
                    nlvl = F.nlvl;
                    lvp = F.lvp;
                    nf = length(F.factors);
                    sk_counts = zeros(1,nf);
                    rd_counts = zeros(1,nf);
                    for k = 1:nf
                      sk_counts(k) = length(F.factors(k).sk);
                      rd_counts(k) = length(F.factors(k).rd);
                    end
                    save('__OUT__','A','X','Ymv','Ysv','ld','D','Di','SD','SDi', ...
                         'nlvl','lvp','nf','sk_counts','rd_counts');
                    exit;
            """,
        )

        F = mf2(data["A"], n=4, occ=2)
        self.assertTrue(F.hierarchical)
        self.assertEqual(F.nlvl, int(data["nlvl"].ravel()[0]))
        np.testing.assert_array_equal(F.lvp, data["lvp"].ravel().astype(np.int64))
        self.assertEqual(len(F.factors), int(data["nf"].ravel()[0]))
        np.testing.assert_array_equal([f.sk.size for f in F.factors], data["sk_counts"].ravel().astype(np.int64))
        np.testing.assert_array_equal([f.rd.size for f in F.factors], data["rd_counts"].ravel().astype(np.int64))
        np.testing.assert_allclose(mf_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_logdet(F), data["ld"].ravel()[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_diag(F), data["D"].ravel(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_diag(F, True), data["Di"].ravel(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_spdiag(F), data["SD"].ravel(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_spdiag(F, True), data["SDi"].ravel(), rtol=1e-9, atol=1e-9)

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

    def test_mf3_grid_operator(self):
        data = _run_flam_export(
            "mf3_parity",
            """
                    n = 3;
                    nd = n - 1;
                    N = nd^3;
                    A = sparse(N,N);
                    for kk = 1:nd
                      for jj = 1:nd
                        for ii = 1:nd
                          idx = ii + nd*(jj-1) + nd^2*(kk-1);
                          A(idx,idx) = 6;
                          if ii > 1,  A(idx,idx-1) = -1; end
                          if ii < nd, A(idx,idx+1) = -1; end
                          if jj > 1,  A(idx,idx-nd) = -1; end
                          if jj < nd, A(idx,idx+nd) = -1; end
                          if kk > 1,  A(idx,idx-nd^2) = -1; end
                          if kk < nd, A(idx,idx+nd^2) = -1; end
                        end
                      end
                    end
                    X = reshape((0:(2*N-1))/(2*N+1),N,2);
                    F = mf3(A,n,2,struct('symm','n'));
                    Ymv = mf_mv(F,X);
                    Ysv = mf_sv(F,X);
                    ld = mf_logdet(F);
                    nlvl = F.nlvl;
                    lvp = F.lvp;
                    nf = length(F.factors);
                    sk_counts = zeros(1,nf);
                    rd_counts = zeros(1,nf);
                    for k = 1:nf
                      sk_counts(k) = length(F.factors(k).sk);
                      rd_counts(k) = length(F.factors(k).rd);
                    end
                    save('__OUT__','A','X','Ymv','Ysv','ld','nlvl','lvp','nf','sk_counts','rd_counts');
                    exit;
            """,
        )

        F = mf3(data["A"], n=3, occ=2)
        self.assertTrue(F.hierarchical)
        self.assertEqual(F.nlvl, int(data["nlvl"].ravel()[0]))
        np.testing.assert_array_equal(F.lvp, data["lvp"].ravel().astype(np.int64))
        self.assertEqual(len(F.factors), int(data["nf"].ravel()[0]))
        np.testing.assert_array_equal([f.sk.size for f in F.factors], data["sk_counts"].ravel().astype(np.int64))
        np.testing.assert_array_equal([f.rd.size for f in F.factors], data["rd_counts"].ravel().astype(np.int64))
        np.testing.assert_allclose(mf_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_logdet(F), data["ld"].ravel()[0], rtol=1e-9, atol=1e-9)

    def test_mfx_line_operator(self):
        data = _run_flam_export(
            "mfx_line_parity",
            """
                    n = 5;
                    e = ones(n,1);
                    A = spdiags([-e 3*e -e],[-1 0 1],n,n);
                    x = linspace(0,1,n);
                    X = reshape((0:(2*n-1))/(2*n+1),n,2);
                    F = mfx(A,x,2,struct('symm','n'));
                    Ymv = mf_mv(F,X);
                    Yadj = mf_mv(F,X,'c');
                    Ysv = mf_sv(F,X);
                    Ysad = mf_sv(F,X,'c');
                    ld = mf_logdet(F);
                    nlvl = F.nlvl;
                    lvp = F.lvp;
                    nf = length(F.factors);
                    sk_counts = zeros(1,nf);
                    rd_counts = zeros(1,nf);
                    for k = 1:nf
                      sk_counts(k) = length(F.factors(k).sk);
                      rd_counts(k) = length(F.factors(k).rd);
                    end
                    save('__OUT__','A','x','X','Ymv','Yadj','Ysv','Ysad','ld', ...
                         'nlvl','lvp','nf','sk_counts','rd_counts');
                    exit;
            """,
        )

        F = mfx(data["A"], data["x"], occ=2)
        self.assertTrue(F.hierarchical)
        self.assertEqual(F.nlvl, int(data["nlvl"].ravel()[0]))
        np.testing.assert_array_equal(F.lvp, data["lvp"].ravel().astype(np.int64))
        self.assertEqual(len(F.factors), int(data["nf"].ravel()[0]))
        np.testing.assert_array_equal([f.sk.size for f in F.factors], data["sk_counts"].ravel().astype(np.int64))
        np.testing.assert_array_equal([f.rd.size for f in F.factors], data["rd_counts"].ravel().astype(np.int64))
        np.testing.assert_allclose(mf_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_mv(F, data["X"], trans="c"), data["Yadj"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_sv(F, data["X"], trans="c"), data["Ysad"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_logdet(F), data["ld"].ravel()[0], rtol=1e-9, atol=1e-9)

    def test_mf2_hermitian_and_positive_modes(self):
        data = _run_flam_export(
            "mf2_symmetry_modes",
            """
                    n = 4;
                    nd = n - 1;
                    N = nd^2;
                    A = sparse(N,N);
                    for jj = 1:nd
                      for ii = 1:nd
                        idx = ii + nd*(jj-1);
                        A(idx,idx) = 4;
                        if ii > 1,  A(idx,idx-1) = -1; end
                        if ii < nd, A(idx,idx+1) = -1; end
                        if jj > 1,  A(idx,idx-nd) = -1; end
                        if jj < nd, A(idx,idx+nd) = -1; end
                      end
                    end
                    X = reshape((0:(2*N-1))/(2*N+1),N,2);

                    Fh = mf2(A,n,2,struct('symm','h'));
                    Yhmv = mf_mv(Fh,X);
                    Yhmvc = mf_mv(Fh,X,'c');
                    Yhsv = mf_sv(Fh,X);
                    Yhsvc = mf_sv(Fh,X,'c');
                    ldh = mf_logdet(Fh);

                    Fp = mf2(A,n,2,struct('symm','p'));
                    Ypmv = mf_mv(Fp,X);
                    Ypsv = mf_sv(Fp,X);
                    Cpmv = mf_cholmv(Fp,X);
                    Cpmvc = mf_cholmv(Fp,X,'c');
                    Cpsv = mf_cholsv(Fp,X);
                    Cpsvc = mf_cholsv(Fp,X,'c');
                    ldp = mf_logdet(Fp);

                    save('__OUT__','A','X','Yhmv','Yhmvc','Yhsv','Yhsvc','ldh', ...
                         'Ypmv','Ypsv','Cpmv','Cpmvc','Cpsv','Cpsvc','ldp');
                    exit;
            """,
        )

        Fh = mf2(data["A"], n=4, occ=2, opts={"symm": "h"})
        np.testing.assert_allclose(mf_mv(Fh, data["X"]), data["Yhmv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_mv(Fh, data["X"], trans="c"), data["Yhmvc"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_sv(Fh, data["X"]), data["Yhsv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_sv(Fh, data["X"], trans="c"), data["Yhsvc"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_logdet(Fh), data["ldh"].ravel()[0], rtol=1e-9, atol=1e-9)

        Fp = mf2(data["A"], n=4, occ=2, opts={"symm": "p"})
        np.testing.assert_allclose(mf_mv(Fp, data["X"]), data["Ypmv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_sv(Fp, data["X"]), data["Ypsv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_cholmv(Fp, data["X"]), data["Cpmv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_cholmv(Fp, data["X"], trans="c"), data["Cpmvc"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_cholsv(Fp, data["X"]), data["Cpsv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_cholsv(Fp, data["X"], trans="c"), data["Cpsvc"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_logdet(Fp), data["ldp"].ravel()[0], rtol=1e-9, atol=1e-9)

    def test_mfx_complex_and_symmetric_modes(self):
        data = _run_flam_export(
            "mfx_complex_symmetry",
            """
                    n = 4;
                    x = linspace(0,1,n);
                    A = diag([3+1i, 4-0.5i, 5+0.75i, 2.5+1.5i]);
                    A = A + diag([1-2i, 1i, 2],1);
                    A = A + diag([0.5+0.25i, -1+0.2i, 0.5-0.5i],-1);
                    A = sparse(A);
                    X = reshape((1:8)/17 + 1i*(9:16)/19,n,2);
                    F = mfx(A,x,2,struct('symm','n'));
                    Ymv = mf_mv(F,X);
                    Ymvt = mf_mv(F,X,'t');
                    Ymvc = mf_mv(F,X,'c');
                    Ysv = mf_sv(F,X);
                    Ysvt = mf_sv(F,X,'t');
                    Ysvc = mf_sv(F,X,'c');
                    ld = mf_logdet(F);

                    As = spdiags([-ones(n,1) 3*ones(n,1) -ones(n,1)],[-1 0 1],n,n);
                    Xs = reshape((0:(2*n-1))/(2*n+1),n,2);
                    Fs = mfx(As,x,2,struct('symm','s'));
                    Ysmv = mf_mv(Fs,Xs);
                    Yssv = mf_sv(Fs,Xs);
                    lds = mf_logdet(Fs);

                    save('__OUT__','A','x','X','Ymv','Ymvt','Ymvc','Ysv','Ysvt','Ysvc','ld', ...
                         'As','Xs','Ysmv','Yssv','lds');
                    exit;
            """,
        )

        F = mfx(data["A"], data["x"], occ=2)
        np.testing.assert_allclose(mf_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_mv(F, data["X"], trans="t"), data["Ymvt"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_mv(F, data["X"], trans="c"), data["Ymvc"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_sv(F, data["X"], trans="t"), data["Ysvt"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_sv(F, data["X"], trans="c"), data["Ysvc"], rtol=1e-9, atol=1e-9)
        self.assertLess(_logdet_mod_error(mf_logdet(F), data["ld"].ravel()[0]), 1e-9)

        Fs = mfx(data["As"], data["x"], occ=2, opts={"symm": "s"})
        self.assertEqual(Fs.symm, "n")
        np.testing.assert_allclose(mf_mv(Fs, data["Xs"]), data["Ysmv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_sv(Fs, data["Xs"]), data["Yssv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mf_logdet(Fs), data["lds"].ravel()[0], rtol=1e-9, atol=1e-9)

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
                    SD = hifde_spdiag(F);
                    SDi = hifde_spdiag(F,1);
                    save('__OUT__','A','X','Ymv','Ysv','ld','D','Di','SD','SDi');
                    exit;
            """,
        )

        F = hifde2(data["A"], n=4, occ=2, rank_or_tol=1e-10)
        np.testing.assert_allclose(hifde_mv(F, data["X"]), data["Ymv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_sv(F, data["X"]), data["Ysv"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_logdet(F), data["ld"].ravel()[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_diag(F), data["D"].ravel(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_diag(F, True), data["Di"].ravel(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_spdiag(F), data["SD"].ravel(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_spdiag(F, True), data["SDi"].ravel(), rtol=1e-9, atol=1e-9)

    def test_hifde_entry_points_match_matlab(self):
        data = _run_flam_export(
            "hifde_entry_points",
            """
                    n2 = 4; nd2 = n2 - 1; N2 = nd2^2;
                    A2 = sparse(N2,N2);
                    for j = 1:nd2, for i = 1:nd2
                      idx = i + nd2*(j - 1);
                      A2(idx,idx) = 4;
                      if i > 1,   A2(idx,idx - 1) = -1; end
                      if i < nd2, A2(idx,idx + 1) = -1; end
                      if j > 1,   A2(idx,idx - nd2) = -1; end
                      if j < nd2, A2(idx,idx + nd2) = -1; end
                    end, end
                    [gx,gy] = ndgrid((1:nd2)/n2); x2 = [gx(:).'; gy(:).'];
                    X2 = reshape((0:(2*N2-1))/(2*N2+1),N2,2);
                    % R2026a errors in the upstream point-cloud histc path for
                    % these tiny grids, so this entry-point check skips the
                    % extra point-cloud reductions while regular hifde2/hifde3
                    % parity covers the dimensional skeletonization stages.
                    optsx = struct('symm','p','skip',99);
                    F2x = hifde2x(A2,x2,2,1e-10,optsx);
                    Y2x = hifde_mv(F2x,X2);
                    Z2x = hifde_sv(F2x,X2);
                    C2x = hifde_cholmv(F2x,hifde_cholmv(F2x,X2,'c'));
                    W2x = hifde_cholsv(F2x,hifde_cholsv(F2x,X2),'c');
                    ld2x = hifde_logdet(F2x);
                    lvp2x = F2x.lvp; nf2x = length(F2x.factors);

                    n3 = 4; nd3 = n3 - 1; N3 = nd3^3;
                    A3 = sparse(N3,N3);
                    for k = 1:nd3, for j = 1:nd3, for i = 1:nd3
                      idx = i + nd3*(j - 1) + nd3^2*(k - 1);
                      A3(idx,idx) = 6;
                      if i > 1,   A3(idx,idx - 1) = -1; end
                      if i < nd3, A3(idx,idx + 1) = -1; end
                      if j > 1,   A3(idx,idx - nd3) = -1; end
                      if j < nd3, A3(idx,idx + nd3) = -1; end
                      if k > 1,   A3(idx,idx - nd3^2) = -1; end
                      if k < nd3, A3(idx,idx + nd3^2) = -1; end
                    end, end, end
                    [gx,gy,gz] = ndgrid((1:nd3)/n3,(1:nd3)/n3,(1:nd3)/n3);
                    x3 = [gx(:).'; gy(:).'; gz(:).'];
                    X3 = reshape((0:(2*N3-1))/(2*N3+1),N3,2);

                    F3 = hifde3(A3,n3,2,1e-10,struct('symm','p'));
                    Y3 = hifde_mv(F3,X3);
                    Z3 = hifde_sv(F3,X3);
                    ld3 = hifde_logdet(F3);
                    lvp3 = F3.lvp; nf3 = length(F3.factors);

                    F3x = hifde3x(A3,x3,2,1e-10,optsx);
                    Y3x = hifde_mv(F3x,X3);
                    Z3x = hifde_sv(F3x,X3);
                    ld3x = hifde_logdet(F3x);
                    lvp3x = F3x.lvp; nf3x = length(F3x.factors);

                    save('__OUT__','A2','A3','x2','x3','X2','X3','Y2x','Z2x','C2x','W2x','ld2x', ...
                         'lvp2x','nf2x','Y3','Z3','ld3','lvp3','nf3','Y3x','Z3x','ld3x','lvp3x','nf3x');
                    exit;
            """,
            timeout=180,
        )

        F2x = hifde2x(data["A2"], data["x2"], occ=2, rank_or_tol=1e-10, opts={"symm": "p", "skip": 99})
        self.assertIsNone(F2x.backend.A_dense)
        np.testing.assert_array_equal(F2x.lvp, data["lvp2x"].ravel().astype(np.int64))
        self.assertEqual(len(F2x.factors), int(data["nf2x"].ravel()[0]))
        np.testing.assert_allclose(hifde_mv(F2x, data["X2"]), data["Y2x"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_sv(F2x, data["X2"]), data["Z2x"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_cholmv(F2x, hifde_cholmv(F2x, data["X2"], "c")), data["C2x"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_cholsv(F2x, hifde_cholsv(F2x, data["X2"]), "c"), data["W2x"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_logdet(F2x), data["ld2x"].ravel()[0], rtol=1e-9, atol=1e-9)

        F3 = hifde3(data["A3"], n=4, occ=2, rank_or_tol=1e-10, opts={"symm": "p"})
        np.testing.assert_array_equal(F3.lvp, data["lvp3"].ravel().astype(np.int64))
        self.assertEqual(len(F3.factors), int(data["nf3"].ravel()[0]))
        np.testing.assert_allclose(hifde_mv(F3, data["X3"]), data["Y3"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_sv(F3, data["X3"]), data["Z3"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_logdet(F3), data["ld3"].ravel()[0], rtol=1e-9, atol=1e-9)

        F3x = hifde3x(data["A3"], data["x3"], occ=2, rank_or_tol=1e-10, opts={"symm": "p", "skip": 99})
        np.testing.assert_array_equal(F3x.lvp, data["lvp3x"].ravel().astype(np.int64))
        self.assertEqual(len(F3x.factors), int(data["nf3x"].ravel()[0]))
        np.testing.assert_allclose(hifde_mv(F3x, data["X3"]), data["Y3x"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_sv(F3x, data["X3"]), data["Z3x"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hifde_logdet(F3x), data["ld3x"].ravel()[0], rtol=1e-9, atol=1e-9)


class ChunkIEStyleRSkelfParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_paths(MATLAB, FLAM_REF, CHUNKIE_REF, label="ChunkIE parity tests")
        require_flam_reference(FLAM_REF, label="ChunkIE parity tests")
        require_pinned_reference(CHUNKIE_REF, "chunkie", label="ChunkIE parity tests")

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

            matfun = @(i,j) chnk.flam.kernbyindex(i,j,chnkr,fkern,opdims,spmat);
            [pr,ptau,pw,pin] = chnk.flam.proxy_square_pts(64);
            pxyfun = @(x,slf,nbr,l,ctr) chnk.flam.proxyfun(slf,nbr,l,ctr,chnkr, ...
                fkern,opdims,pr,ptau,pw,pin,true,false);
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
