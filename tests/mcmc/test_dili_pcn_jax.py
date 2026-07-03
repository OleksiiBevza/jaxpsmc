# ruff: noqa: E402
import chex
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.mcmc.dili_pcn_jax import (
    _dili_li_prior_proposal,
    _standard_normal_log_reference,
    dili_pcn_jax,
)
from jaxpsmc.sampler.mutate_jax import mutate
from jaxpsmc.scaler_jax import init_bounds_config_jax, masks_jax


class IdBij:
    def transform_and_log_det(self, u, condition=None):
        return u, jnp.zeros((), dtype=u.dtype)

    def inverse_and_log_det(self, theta, condition=None):
        return theta, jnp.zeros((), dtype=theta.dtype)


class AffBij:
    def __init__(self, scale, shift):
        self.scale = scale
        self.shift = shift

    def transform_and_log_det(self, u, condition=None):
        scale = jnp.asarray(self.scale, dtype=u.dtype)
        shift = jnp.asarray(self.shift, dtype=u.dtype)
        theta = scale * u + shift
        logdet = u.shape[-1] * jnp.log(jnp.abs(scale))
        return theta, logdet.astype(u.dtype)

    def inverse_and_log_det(self, theta, condition=None):
        scale = jnp.asarray(self.scale, dtype=theta.dtype)
        shift = jnp.asarray(self.shift, dtype=theta.dtype)
        u = (theta - shift) / scale
        logdet = -theta.shape[-1] * jnp.log(jnp.abs(scale))
        return u, logdet.astype(theta.dtype)


class Flow:
    def __init__(self, bijection=None):
        self.bijection = IdBij() if bijection is None else bijection


class DiliPcnTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(13)
        self.flow = Flow()

    def cfg(self, dim):
        cfg = init_bounds_config_jax(dim, scale=False)
        msk = masks_jax(cfg["low"], cfg["high"])
        return cfg, msk

    def data(self, N=5, D=3, B=1, dtype=jnp.float64):
        base = jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.5, -0.5, 0.25],
                [1.0, 1.0, -0.25],
                [-0.5, 0.2, 0.75],
                [0.25, -1.0, 0.5],
            ],
            dtype=dtype,
        )
        u = base[:N, :D]
        x = u
        logdetj = jnp.zeros((N,), dtype=dtype)
        logdetj_flow = jnp.zeros((N,), dtype=dtype)
        logp = -0.5 * jnp.sum(x * x, axis=1)
        logl = -0.5 * jnp.sum((x - 0.25) ** 2, axis=1)
        blobs = jnp.zeros((N, B), dtype=dtype)
        beta = jnp.asarray(0.5, dtype=dtype)

        return {
            "u": u,
            "x": x,
            "logdetj": logdetj,
            "logdetj_flow": logdetj_flow,
            "logl": logl,
            "logp": logp,
            "blobs": blobs,
            "beta": beta,
        }

    def cur(self, N=5, D=3, B=1, dtype=jnp.float64):
        out = self.data(N=N, D=D, B=B, dtype=dtype)
        out["calls"] = jnp.asarray(5, dtype=jnp.int32)
        out["proposal_scale"] = jnp.asarray(0.2, dtype=dtype)
        return out

    def geom(self, dim, rank=2, dtype=jnp.float64):
        center = jnp.linspace(
            jnp.asarray(0.1, dtype=dtype),
            jnp.asarray(0.1 * dim, dtype=dtype),
            dim,
            dtype=dtype,
        )
        basis = jnp.eye(dim, rank, dtype=dtype)
        post_var = jnp.linspace(
            jnp.asarray(0.5, dtype=dtype),
            jnp.asarray(1.5, dtype=dtype),
            rank,
            dtype=dtype,
        )
        cov_ref = jnp.eye(dim, dtype=dtype)
        return center, basis, post_var, cov_ref

    def like(self, x):
        ll = -0.5 * jnp.sum((x - 0.25) ** 2)
        blob = jnp.asarray([jnp.sum(x)], dtype=x.dtype)
        return ll, blob

    def like0(self, x):
        ll = -0.5 * jnp.sum((x - 0.25) ** 2)
        blob = jnp.zeros((0,), dtype=x.dtype)
        return ll, blob

    def approx(self, x):
        return -0.25 * jnp.sum((x - 0.25) ** 2)

    def prior(self, x):
        return -0.5 * jnp.sum(x * x)

    def _run(self, key, data=None, **kwargs):
        if data is None:
            data = self.data()

        dim = data["u"].shape[1]
        cfg, msk = self.cfg(dim)

        rank = min(2, dim)
        center, basis, post_var, cov_ref = self.geom(
            dim,
            rank=rank,
            dtype=data["u"].dtype,
        )

        return dili_pcn_jax(
            key,
            u=data["u"],
            x=data["x"],
            logdetj=data["logdetj"],
            logl=data["logl"],
            logp=data["logp"],
            logdetj_flow=data["logdetj_flow"],
            blobs=data["blobs"],
            beta=data["beta"],
            loglike_fn=kwargs.pop("loglike_fn", self.like),
            loglike_approx_fn=kwargs.pop("loglike_approx_fn", self.approx),
            logprior_fn=kwargs.pop("logprior_fn", self.prior),
            flow=kwargs.pop("flow", self.flow),
            scaler_cfg=cfg,
            scaler_masks=msk,
            dili_center=kwargs.pop("dili_center", center),
            dili_basis=kwargs.pop("dili_basis", basis),
            dili_post_var=kwargs.pop("dili_post_var", post_var),
            dili_cov_ref=kwargs.pop("dili_cov_ref", cov_ref),
            n_max=kwargs.pop("n_max", 4),
            n_steps=kwargs.pop("n_steps", 2),
            proposal_scale=kwargs.pop(
                "proposal_scale", jnp.asarray(0.2, dtype=data["u"].dtype)
            ),
            dili_lis_scale=kwargs.pop("dili_lis_scale", 1.0),
            dili_cs_scale=kwargs.pop("dili_cs_scale", 0.5),
            use_delayed_acceptance=kwargs.pop(
                "use_delayed_acceptance", jnp.asarray(False)
            ),
            da_c_const=kwargs.pop("da_c_const", jnp.asarray(0.01)),
            da_d_const=kwargs.pop("da_d_const", jnp.asarray(2.0)),
            condition=kwargs.pop("condition", None),
        )

    def check_kernel(self, out):
        expected = {
            "key",
            "u",
            "x",
            "logdetj",
            "logdetj_flow",
            "logl",
            "logp",
            "blobs",
            "efficiency",
            "accept",
            "steps",
            "calls",
            "proposal_scale",
        }
        assert set(out.keys()) == expected

    def check_mutate(self, out):
        expected = {
            "u",
            "x",
            "logdetj",
            "logl",
            "logp",
            "logdetj_flow",
            "blobs",
            "beta",
            "calls",
            "proposal_scale",
            "efficiency",
            "steps",
            "accept",
        }
        assert set(out.keys()) == expected

    def test_ref(self):
        theta = jnp.asarray(
            [
                [1.0, -1.0],
                [3.0, 1.0],
            ],
            dtype=jnp.float64,
        )
        center = jnp.asarray([1.0, -1.0], dtype=jnp.float64)

        out = _standard_normal_log_reference(theta, center)

        expected = jnp.asarray([0.0, -4.0], dtype=jnp.float64)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_prop_shape(self):
        theta = jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, -1.0, 0.5],
            ],
            dtype=jnp.float64,
        )
        center = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float64)
        basis = jnp.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        post_var = jnp.asarray([0.5, 2.0], dtype=jnp.float64)

        out = _dili_li_prior_proposal(
            self.key,
            theta,
            center,
            basis,
            post_var,
            jnp.asarray(0.2, dtype=jnp.float64),
            dili_lis_scale=1.0,
            dili_cs_scale=0.5,
        )

        assert out.shape == theta.shape
        assert out.dtype == theta.dtype
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_prop_zero(self):
        theta = jnp.asarray(
            [
                [0.0, 1.0, -1.0],
                [2.0, -1.0, 0.5],
            ],
            dtype=jnp.float64,
        )
        center = jnp.asarray([0.5, -0.5, 0.25], dtype=jnp.float64)
        basis = jnp.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        post_var = jnp.asarray([1.0, 2.0], dtype=jnp.float64)

        out = _dili_li_prior_proposal(
            self.key,
            theta,
            center,
            basis,
            post_var,
            jnp.asarray(0.0, dtype=jnp.float64),
            dili_lis_scale=1.0,
            dili_cs_scale=1.0,
        )

        assert out.shape == theta.shape
        assert out.dtype == theta.dtype
        assert bool(jnp.all(jnp.isfinite(out)))
        np.testing.assert_allclose(out, theta, rtol=1e-2, atol=1e-2)

    def test_prop_lis(self):
        theta = jnp.asarray(
            [
                [0.0, 1.0, -1.0],
                [2.0, -1.0, 0.5],
            ],
            dtype=jnp.float64,
        )
        center = jnp.asarray([0.5, -0.5, 0.25], dtype=jnp.float64)
        basis = jnp.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        post_var = jnp.asarray([1.0, 2.0], dtype=jnp.float64)

        out = _dili_li_prior_proposal(
            self.key,
            theta,
            center,
            basis,
            post_var,
            jnp.asarray(0.25, dtype=jnp.float64),
            dili_lis_scale=1.0,
            dili_cs_scale=0.0,
        )

        assert out.shape == theta.shape
        assert bool(jnp.all(jnp.isfinite(out)))
        np.testing.assert_allclose(out[:, 2], theta[:, 2], rtol=1e-2, atol=1e-2)
        assert not np.allclose(np.asarray(out[:, :2]), np.asarray(theta[:, :2]))

    @chex.all_variants(with_pmap=False)
    def test_noop(self):
        data = self.data()

        out = self.variant(lambda key: self._run(key, data=data, n_max=0))(self.key)

        self.check_kernel(out)
        np.testing.assert_allclose(out["u"], data["u"])
        np.testing.assert_allclose(out["x"], data["x"])
        np.testing.assert_allclose(out["logdetj"], data["logdetj"])
        np.testing.assert_allclose(out["logdetj_flow"], data["logdetj_flow"])
        np.testing.assert_allclose(out["logl"], data["logl"])
        np.testing.assert_allclose(out["logp"], data["logp"])
        np.testing.assert_allclose(out["blobs"], data["blobs"])
        np.testing.assert_array_equal(
            jax.random.key_data(out["key"]),
            jax.random.key_data(self.key),
        )
        np.testing.assert_allclose(out["accept"], 0.0)
        np.testing.assert_array_equal(out["steps"], jnp.asarray(0, dtype=jnp.int32))
        np.testing.assert_array_equal(out["calls"], jnp.asarray(0, dtype=jnp.int32))
        np.testing.assert_allclose(out["proposal_scale"], 0.2)

    @chex.all_variants(with_pmap=False)
    def test_cap(self):
        data = self.data()

        out = self.variant(
            lambda key: self._run(
                key,
                data=data,
                n_max=0,
                proposal_scale=jnp.asarray(2.0, dtype=data["u"].dtype),
            )
        )(self.key)

        np.testing.assert_allclose(out["proposal_scale"], 0.99, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            out["efficiency"], out["proposal_scale"], rtol=1e-6, atol=1e-6
        )

    @chex.all_variants(with_pmap=False)
    def test_shapes(self):
        data = self.data(N=5, D=3, B=1)

        out = self.variant(lambda key: self._run(key, data=data, n_max=3))(self.key)

        self.check_kernel(out)
        assert out["u"].shape == data["u"].shape
        assert out["x"].shape == data["x"].shape
        assert out["logdetj"].shape == data["logdetj"].shape
        assert out["logdetj_flow"].shape == data["logdetj_flow"].shape
        assert out["logl"].shape == data["logl"].shape
        assert out["logp"].shape == data["logp"].shape
        assert out["blobs"].shape == data["blobs"].shape

        assert out["accept"].shape == ()
        assert out["steps"].shape == ()
        assert out["calls"].shape == ()
        assert out["proposal_scale"].shape == ()

        assert bool(jnp.all(jnp.isfinite(out["u"])))
        assert bool(jnp.all(jnp.isfinite(out["x"])))
        assert bool(jnp.all(jnp.isfinite(out["logdetj"])))
        assert bool(jnp.all(jnp.isfinite(out["logdetj_flow"])))
        assert bool(jnp.all(jnp.isfinite(out["logl"])))
        assert bool(jnp.all(jnp.isfinite(out["logp"])))
        assert bool(jnp.all(jnp.isfinite(out["blobs"])))
        assert bool(jnp.isfinite(out["accept"]))
        assert bool(jnp.isfinite(out["proposal_scale"]))

        assert 0 <= int(out["steps"]) <= 3
        assert 0 <= int(out["calls"]) <= 3 * data["u"].shape[0]
        assert 0.0 <= float(out["accept"]) <= 1.0

    @chex.all_variants(with_pmap=False)
    def test_repeat(self):
        data = self.data()

        def f(key):
            return self._run(key, data=data, n_max=3)

        out1 = self.variant(f)(self.key)
        out2 = self.variant(f)(self.key)

        np.testing.assert_array_equal(
            jax.random.key_data(out1["key"]), jax.random.key_data(out2["key"])
        )
        np.testing.assert_allclose(out1["u"], out2["u"])
        np.testing.assert_allclose(out1["x"], out2["x"])
        np.testing.assert_allclose(out1["logdetj"], out2["logdetj"])
        np.testing.assert_allclose(out1["logdetj_flow"], out2["logdetj_flow"])
        np.testing.assert_allclose(out1["logl"], out2["logl"])
        np.testing.assert_allclose(out1["logp"], out2["logp"])
        np.testing.assert_allclose(out1["blobs"], out2["blobs"])
        np.testing.assert_allclose(out1["accept"], out2["accept"])
        np.testing.assert_array_equal(out1["steps"], out2["steps"])
        np.testing.assert_array_equal(out1["calls"], out2["calls"])
        np.testing.assert_allclose(out1["proposal_scale"], out2["proposal_scale"])

    @chex.all_variants(with_pmap=False)
    def test_key(self):
        data = self.data()

        out = self.variant(lambda key: self._run(key, data=data, n_max=2))(self.key)

        assert not np.array_equal(
            np.asarray(jax.random.key_data(out["key"])),
            np.asarray(jax.random.key_data(self.key)),
        )

    @chex.all_variants(with_pmap=False)
    def test_blob0(self):
        data = self.data(N=5, D=3, B=0)

        out = self.variant(
            lambda key: self._run(
                key,
                data=data,
                n_max=2,
                loglike_fn=self.like0,
            )
        )(self.key)

        self.check_kernel(out)
        assert out["blobs"].shape == (5, 0)
        np.testing.assert_allclose(
            out["blobs"], jnp.zeros((5, 0), dtype=data["u"].dtype)
        )

    @chex.all_variants(with_pmap=False)
    def test_da(self):
        data = self.data()

        out = self.variant(
            lambda key: self._run(
                key,
                data=data,
                n_max=3,
                use_delayed_acceptance=jnp.asarray(True),
            )
        )(self.key)

        self.check_kernel(out)
        assert out["u"].shape == data["u"].shape
        assert out["x"].shape == data["x"].shape
        assert out["blobs"].shape == data["blobs"].shape
        assert 0 <= int(out["steps"]) <= 3
        assert 0 <= int(out["calls"]) <= 3 * data["u"].shape[0]
        assert 0.0 <= float(out["accept"]) <= 1.0
        assert bool(jnp.isfinite(out["proposal_scale"]))

    @chex.all_variants(with_pmap=False)
    def test_flowdet(self):
        data = self.data(D=3)
        flow = Flow(AffBij(scale=2.0, shift=1.0))
        expected = -3.0 * jnp.log(jnp.asarray(2.0, dtype=data["u"].dtype))

        out = self.variant(
            lambda key: self._run(
                key,
                data=data,
                flow=flow,
                n_max=0,
            )
        )(self.key)

        np.testing.assert_allclose(
            out["logdetj_flow"],
            jnp.full((data["u"].shape[0],), expected, dtype=data["u"].dtype),
            rtol=1e-6,
            atol=1e-6,
        )

    @chex.all_variants(with_pmap=False)
    def test_dtype(self):
        data = self.data(dtype=jnp.float64)

        out = self.variant(lambda key: self._run(key, data=data, n_max=2))(self.key)

        assert out["u"].dtype == jnp.float64
        assert out["x"].dtype == jnp.float64
        assert out["logdetj"].dtype == jnp.float64
        assert out["logdetj_flow"].dtype == jnp.float64
        assert out["logl"].dtype == jnp.float64
        assert out["logp"].dtype == jnp.float64
        assert out["blobs"].dtype == jnp.float64
        assert out["accept"].dtype == jnp.float64
        assert out["proposal_scale"].dtype == jnp.float64

    @chex.all_variants(with_pmap=False)
    def test_mutate_noop(self):
        cur = self.cur()
        cfg, msk = self.cfg(cur["u"].shape[1])
        center, basis, post_var, cov_ref = self.geom(
            cur["u"].shape[1],
            rank=2,
            dtype=cur["u"].dtype,
        )

        def f(k, c):
            return mutate(
                k,
                c,
                use_preconditioned_pcn=jnp.asarray(False),
                loglike_single_fn=self.like,
                loglike_approx_single_fn=self.approx,
                logprior_fn=self.prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((3,), dtype=c["u"].dtype),
                geom_cov=jnp.eye(3, dtype=c["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=c["u"].dtype),
                dili_center=center,
                dili_basis=basis,
                dili_post_var=post_var,
                dili_cov_ref=cov_ref,
                kernel="dili_pcn",
                n_max=0,
                n_steps=1,
            )

        key_out, out, info = self.variant(f)(self.key, cur)

        self.check_mutate(out)
        np.testing.assert_array_equal(
            jax.random.key_data(key_out), jax.random.key_data(self.key)
        )
        np.testing.assert_allclose(out["u"], cur["u"])
        np.testing.assert_allclose(out["x"], cur["x"])
        np.testing.assert_allclose(out["logdetj"], cur["logdetj"])
        np.testing.assert_allclose(out["logdetj_flow"], cur["logdetj_flow"])
        np.testing.assert_allclose(out["logl"], cur["logl"])
        np.testing.assert_allclose(out["logp"], cur["logp"])
        np.testing.assert_allclose(out["blobs"], cur["blobs"])
        np.testing.assert_array_equal(out["calls"], cur["calls"])
        np.testing.assert_array_equal(out["steps"], jnp.asarray(0, dtype=jnp.int32))
        np.testing.assert_allclose(out["accept"], 0.0)
        np.testing.assert_array_equal(
            info["calls_increment"], jnp.asarray(0, dtype=jnp.int32)
        )
        np.testing.assert_allclose(out["proposal_scale"], cur["proposal_scale"])

    @chex.all_variants(with_pmap=False)
    def test_mutate(self):
        cur = self.cur()
        cfg, msk = self.cfg(cur["u"].shape[1])
        center, basis, post_var, cov_ref = self.geom(
            cur["u"].shape[1],
            rank=2,
            dtype=cur["u"].dtype,
        )

        def f(k, c):
            return mutate(
                k,
                c,
                use_preconditioned_pcn=jnp.asarray(True),
                loglike_single_fn=self.like,
                loglike_approx_single_fn=self.approx,
                logprior_fn=self.prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((3,), dtype=c["u"].dtype),
                geom_cov=jnp.eye(3, dtype=c["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=c["u"].dtype),
                dili_center=center,
                dili_basis=basis,
                dili_post_var=post_var,
                dili_cov_ref=cov_ref,
                kernel="dili_pcn",
                dili_lis_scale=1.0,
                dili_cs_scale=0.5,
                n_max=2,
                n_steps=1,
            )

        key_out, out, info = self.variant(f)(self.key, cur)

        self.check_mutate(out)
        assert not np.array_equal(
            jax.random.key_data(key_out), jax.random.key_data(self.key)
        )
        assert out["u"].shape == cur["u"].shape
        assert out["x"].shape == cur["x"].shape
        assert out["blobs"].shape == cur["blobs"].shape
        assert bool(jnp.all(jnp.isfinite(out["u"])))
        assert bool(jnp.all(jnp.isfinite(out["x"])))
        assert bool(jnp.isfinite(out["proposal_scale"]))
        assert 0 <= int(out["steps"]) <= 2
        assert int(out["calls"]) >= int(cur["calls"])
        np.testing.assert_array_equal(
            out["calls"], cur["calls"] + info["calls_increment"]
        )
        np.testing.assert_allclose(
            out["proposal_scale"], info["proposal_scale"], rtol=1e-6, atol=1e-6
        )
        assert 0.0 <= float(out["accept"]) <= 1.0

    def test_missing_geom(self):
        cur = self.cur()
        cfg, msk = self.cfg(cur["u"].shape[1])

        with self.assertRaisesRegex(ValueError, "kernel='dili_pcn' requires"):
            mutate(
                self.key,
                cur,
                use_preconditioned_pcn=jnp.asarray(True),
                loglike_single_fn=self.like,
                loglike_approx_single_fn=self.approx,
                logprior_fn=self.prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((3,), dtype=cur["u"].dtype),
                geom_cov=jnp.eye(3, dtype=cur["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=cur["u"].dtype),
                kernel="dili_pcn",
                n_max=1,
                n_steps=1,
            )


if __name__ == "__main__":
    absltest.main()
