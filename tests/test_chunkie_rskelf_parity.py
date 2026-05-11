import textwrap
import unittest
from pathlib import Path

import numpy as np
import scipy.special

from matlab_parity_utils import MATLAB, logdet_mod_error, matlab_path, relerr, require_paths, run_matlab_export
from pyflam import rskelf, rskelf_logdet, rskelf_mv, rskelf_sv


FLAM_REF = Path(r"C:\Users\haiya\git\FLAM")
CHUNKIE_REF = Path(r"C:\Users\haiya\git\chunkie")


class ChunkIEMoreRSkelfParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_paths(MATLAB, FLAM_REF, CHUNKIE_REF, label="ChunkIE rskelf parity")

    def test_laplace_dirichlet_l2scaled_starfish(self):
        self._run_case("laplace_d_l2scale")

    def test_helmholtz_combined_layer_starfish(self):
        self._run_case("helmholtz_c")

    def _run_case(self, case: str):
        data = run_matlab_export(f"chunkie_{case}", _chunkie_driver(case), timeout=240)
        op = _ChunkIEOperator(data, case)
        n = data["sys"].shape[0]
        idx = np.arange(n, dtype=np.int64)

        np.testing.assert_allclose(relerr(op(idx, idx), data["sys"]), 0.0, atol=1e-12)
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
        np.testing.assert_allclose(relerr(Ymv, data["Ymv"]), 0.0, atol=1e-10)
        np.testing.assert_allclose(relerr(Ysv, data["Ysv"]), 0.0, atol=1e-10)
        np.testing.assert_allclose(relerr(Ymv, data["sys"] @ data["X"]), 0.0, atol=1e-9)
        np.testing.assert_allclose(relerr(data["sys"] @ Ysv, data["X"]), 0.0, atol=1e-9)
        self.assertLess(logdet_mod_error(rskelf_logdet(F), data["ld"].item()), 1e-9)


def _chunkie_driver(case: str) -> str:
    l2scale = "false"
    rhs = "X = reshape(sin((1:(3*chnkr.npt))/31), chnkr.npt, 3);"
    if case == "laplace_d_l2scale":
        kernel_setup = "zk = 0; coef = [0, 0]; fkern = @(s,t) chnk.lap2d.kern(s,t,'D');"
        l2scale = "true"
    elif case == "helmholtz_c":
        kernel_setup = "zk = 1.3; coef = [1, -1i*zk]; fkern = @(s,t) chnk.helm2d.kern(zk,s,t,'c',coef);"
        rhs = "X = reshape(exp(1i*(1:(3*chnkr.npt))/43), chnkr.npt, 3);"
    else:
        raise ValueError(f"unknown ChunkIE case: {case}")

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
        pref.k = 10;
        chnkr = chunkerfuncuni(@(t) starfish(t,3,0.25), 10, cparams, pref);
        {kernel_setup}

        dval = -0.5;
        qopts = [];
        qopts.nonsmoothonly = true;
        qopts.quad = 'ggq';
        qopts.type = 'log';
        qopts.eps = 1e-10;
        qopts.l2scale = {l2scale};
        spmat = chunkermat(chnkr, fkern, qopts) + dval*speye(chnkr.npt);

        fullopts = [];
        fullopts.l2scale = {l2scale};
        sys = chunkermat(chnkr, fkern, fullopts) + dval*eye(chnkr.npt);

        r = chnkr.r(:,:);
        nn = chnkr.n(:,:);
        wts = weights(chnkr);
        wts = wts(:);
        xflam = r;
        occ = 10;
        tol = 1e-10;
        {rhs}

        matfun = @(i,j) chnk.flam.kernbyindex(i,j,chnkr,wts,fkern,ones([2,1,1]),spmat,{l2scale});
        [pr,ptau,pw,pin] = chnk.flam.proxy_square_pts(64);
        pxyfun = @(x,slf,nbr,l,ctr) chnk.flam.proxyfun(slf,nbr,l,ctr,chnkr,wts, ...
            fkern,ones([2,1,1]),pr,ptau,pw,pin,true,{l2scale});
        F = rskelf(matfun, xflam, occ, tol, pxyfun);
        Ymv = rskelf_mv(F, X);
        Ysv = rskelf_sv(F, X);
        ld = rskelf_logdet(F);

        save('__OUT__','sys','spmat','r','nn','wts','xflam','occ','tol','X','Ymv','Ysv', ...
             'ld','pr','ptau','pw','zk','coef','-v7');
        exit;
        """
    )


class _ChunkIEOperator:
    def __init__(self, data, case: str):
        self.r = np.asarray(data["r"])
        self.n = np.asarray(data["nn"])
        self.wts = np.asarray(data["wts"]).reshape(-1)
        self.spmat = data["spmat"].tocsc()
        self.case = case
        self.zk = float(np.asarray(data["zk"]).reshape(-1)[0])
        self.coef = np.asarray(data["coef"]).reshape(-1)
        self.proxy_calls = 0

    @property
    def l2scale(self) -> bool:
        return self.case.endswith("_l2scale")

    def __call__(self, i, j):
        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)
        mat = self._weighted_kernel(self.r[:, j], self.n[:, j], self.r[:, i], self.wts[j], self.wts[i])
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
            src_to_proxy = self._weighted_kernel(self.r[:, slf], self.n[:, slf], pxy, self.wts[slf], pweights)
            proxy_to_src = self._weighted_kernel(pxy, pnorm, self.r[:, slf], pweights, self.wts[slf])
            return np.vstack((src_to_proxy, proxy_to_src.T)), nbr

        return pxyfun

    def _weighted_kernel(self, src_r, src_n, targ_r, src_w, targ_w):
        mat = self._kernel(src_r, src_n, targ_r)
        if self.l2scale:
            return np.sqrt(targ_w)[:, None] * mat * np.sqrt(src_w)[None, :]
        return mat * src_w[None, :]

    def _kernel(self, src_r, src_n, targ_r):
        rx = targ_r[0, :, None] - src_r[0, None, :]
        ry = targ_r[1, :, None] - src_r[1, None, :]
        r2 = rx * rx + ry * ry
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.case.startswith("laplace_d"):
                return (rx * src_n[0, None, :] + ry * src_n[1, None, :]) / (2 * np.pi * r2)
            r = np.sqrt(r2)
            h0 = scipy.special.hankel1(0, self.zk * r)
            h1 = scipy.special.hankel1(1, self.zk * r)
            single = 0.25j * h0
            grad_x = -0.25j * self.zk * h1 * rx / r
            grad_y = -0.25j * self.zk * h1 * ry / r
            double = -(grad_x * src_n[0, None, :] + grad_y * src_n[1, None, :])
            return self.coef[0] * double + self.coef[1] * single


if __name__ == "__main__":
    unittest.main()
