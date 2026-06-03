import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.mcmc import preconditioned_pcn_jax
from jaxpsmc.mcmc.flow_jax import (
    _flow_u_to_theta_jax,
    _flow_theta_to_u_jax,
)
from jaxpsmc.scaler_jax import init_bounds_config_jax, masks_jax


class IdentityBijection:
    def transform_and_log_det(self, u, condition=None):
        return u, jnp.zeros((), dtype=u.dtype)

    def inverse_and_log_det(self, theta, condition=None):
        return theta, jnp.zeros((), dtype=theta.dtype)


class AffineBijection:
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
    def __init__(self, bijection):
        self.bijection = bijection


class PcnTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(1)
        self.flow = Flow(IdentityBijection())

    def _cfg(self, dim):
        cfg = init_bounds_config_jax(dim, scale=False)
        msk = masks_jax(cfg["low"], cfg["high"])
        return cfg, msk

    def _data(self, N=4, D=2, B=1, dtype=jnp.float64):
        u = jnp.asarray(
            [[0.0, 0.0], [0.5, -0.5], [1.0, 1.0], [-0.5, 0.2]],
            dtype=dtype,
        )
        u = u[:N, :D]
        x = u
        logdetj = jnp.zeros((N,), dtype=dtype)
        logdetj_flow = jnp.zeros((N,), dtype=dtype)
        logp = -0.5 * jnp.sum(x * x, axis=1)
        logl = -0.5 * jnp.sum((x - 0.5) ** 2, axis=1)
        blobs = jnp.zeros((N, B), dtype=dtype)
        beta = jnp.asarray(0.5, dtype=dtype)

        return {
            "u": u,
            "x": x,
            "logdetj": logdetj,
            "logl": logl,
            "logp": logp,
            "logdetj_flow": logdetj_flow,
            "blobs": blobs,
            "beta": beta,
        }

    def _loglike(self, x):
        ll = -0.5 * jnp.sum((x - 0.5) ** 2)
        blob = jnp.array([jnp.sum(x)], dtype=x.dtype)
        return ll, blob

    def _loglike0(self, x):
        ll = -0.5 * jnp.sum((x - 0.5) ** 2)
        blob = jnp.zeros((0,), dtype=x.dtype)
        return ll, blob

    def _approx(self, x):
        return -0.25 * jnp.sum((x - 0.5) ** 2)

    def _prior(self, x):
        return -0.5 * jnp.sum(x * x)

    def _run(self, key, data=None, **kwargs):
        if data is None:
            data = self._data()

        dim = data["u"].shape[1]
        cfg, msk = self._cfg(dim)

        return preconditioned_pcn_jax(
            key,
            u=data["u"],
            x=data["x"],
            logdetj=data["logdetj"],
            logl=data["logl"],
            logp=data["logp"],
            logdetj_flow=data["logdetj_flow"],
            blobs=data["blobs"],
            beta=data["beta"],
            loglike_fn=kwargs.pop("loglike_fn", self._loglike),
            loglike_approx_fn=kwargs.pop("loglike_approx_fn", self._approx),
            logprior_fn=kwargs.pop("logprior_fn", self._prior),
            flow=kwargs.pop("flow", self.flow),
            scaler_cfg=cfg,
            scaler_masks=msk,
            geom_mu=kwargs.pop("geom_mu", jnp.zeros((dim,), dtype=data["u"].dtype)),
            geom_cov=kwargs.pop("geom_cov", jnp.eye(dim, dtype=data["u"].dtype)),
            geom_nu=kwargs.pop("geom_nu", jnp.asarray(10.0, dtype=data["u"].dtype)),
            n_max=kwargs.pop("n_max", 4),
            n_steps=kwargs.pop("n_steps", 2),
            proposal_scale=kwargs.pop(
                "proposal_scale", jnp.asarray(0.2, dtype=data["u"].dtype)
            ),
            use_delayed_acceptance=kwargs.pop(
                "use_delayed_acceptance", jnp.asarray(False)
            ),
            da_c_const=kwargs.pop("da_c_const", jnp.asarray(0.01)),
            da_d_const=kwargs.pop("da_d_const", jnp.asarray(2.0)),
            condition=kwargs.pop("condition", None),
        )

    def _assert_keys(self, out):
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

    def test_flow(self):
        flow = Flow(AffineBijection(scale=2.0, shift=1.0))
        u = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64)

        theta, logdet = _flow_u_to_theta_jax(flow, u)
        u_back, inv_logdet = _flow_theta_to_u_jax(flow, theta)

        np.testing.assert_allclose(theta, 2.0 * u + 1.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(u_back, u, rtol=1e-6, atol=1e-6)

        expected = 3.0 * np.log(2.0)
        np.testing.assert_allclose(logdet, -expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(inv_logdet, -expected, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_noop(self):
        data = self._data()

        out = self.variant(lambda key: self._run(key, data=data, n_max=0))(self.key)

        self._assert_keys(out)
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
        np.testing.assert_array_equal(out["steps"], jnp.asarray(0, dtype=jnp.int64))
        np.testing.assert_array_equal(out["calls"], jnp.asarray(0, dtype=jnp.int64))
        np.testing.assert_allclose(out["proposal_scale"], 0.2)

    @chex.all_variants(with_pmap=False)
    def test_cap(self):
        data = self._data()

        out = self.variant(
            lambda key: self._run(
                key,
                data=data,
                n_max=0,
                proposal_scale=jnp.asarray(2.0, dtype=data["u"].dtype),
            )
        )(self.key)

        np.testing.assert_allclose(out["proposal_scale"], 0.99, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out["efficiency"], out["proposal_scale"])

    @chex.all_variants(with_pmap=False)
    def test_shapes(self):
        data = self._data(N=4, D=2, B=1)

        out = self.variant(lambda key: self._run(key, data=data, n_max=3))(self.key)

        self._assert_keys(out)
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

    @chex.all_variants(with_pmap=False)
    def test_bounds(self):
        data = self._data()

        out = self.variant(lambda key: self._run(key, data=data, n_max=4))(self.key)

        assert bool(jnp.all(jnp.isfinite(out["u"])))
        assert bool(jnp.all(jnp.isfinite(out["x"])))
        assert bool(jnp.all(jnp.isfinite(out["logdetj"])))
        assert bool(jnp.all(jnp.isfinite(out["logdetj_flow"])))
        assert bool(jnp.all(jnp.isfinite(out["logl"])))
        assert bool(jnp.all(jnp.isfinite(out["logp"])))
        assert bool(jnp.all(jnp.isfinite(out["blobs"])))
        assert bool(jnp.isfinite(out["proposal_scale"]))

        assert 0 <= int(out["steps"]) <= 4
        assert 0.0 <= float(out["accept"]) <= 1.0
        assert 0.0 <= float(out["proposal_scale"]) <= 0.99

    @chex.all_variants(with_pmap=False)
    def test_repro(self):
        data = self._data()

        def run(key):
            return self._run(key, data=data, n_max=3)

        out1 = self.variant(run)(self.key)
        out2 = self.variant(run)(self.key)

        np.testing.assert_array_equal(
            jax.random.key_data(out1["key"]),
            jax.random.key_data(out2["key"]),
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
        np.testing.assert_allclose(out1["proposal_scale"], out2["proposal_scale"])

    @chex.all_variants(with_pmap=False)
    def test_key(self):
        data = self._data()

        out = self.variant(lambda key: self._run(key, data=data, n_max=2))(self.key)

        assert not np.array_equal(
            np.asarray(jax.random.key_data(out["key"])),
            np.asarray(jax.random.key_data(self.key)),
        )

    @chex.all_variants(with_pmap=False)
    def test_blob0(self):
        data = self._data(N=4, D=2, B=0)

        out = self.variant(
            lambda key: self._run(
                key,
                data=data,
                n_max=2,
                loglike_fn=self._loglike0,
            )
        )(self.key)

        assert out["blobs"].shape == (4, 0)
        np.testing.assert_allclose(out["blobs"], jnp.zeros((4, 0), dtype=data["u"].dtype))

    @chex.all_variants(with_pmap=False)
    def test_da(self):
        data = self._data()

        out = self.variant(
            lambda key: self._run(
                key,
                data=data,
                n_max=3,
                use_delayed_acceptance=jnp.asarray(True),
            )
        )(self.key)

        self._assert_keys(out)
        assert out["u"].shape == data["u"].shape
        assert out["x"].shape == data["x"].shape
        assert out["blobs"].shape == data["blobs"].shape
        assert 0 <= int(out["steps"]) <= 3
        assert 0.0 <= float(out["accept"]) <= 1.0
        assert bool(jnp.isfinite(out["proposal_scale"]))

    @chex.all_variants(with_pmap=False)
    def test_da_noop(self):
        data = self._data()

        out = self.variant(
            lambda key: self._run(
                key,
                data=data,
                n_max=0,
                use_delayed_acceptance=jnp.asarray(True),
            )
        )(self.key)

        np.testing.assert_allclose(out["u"], data["u"])
        np.testing.assert_allclose(out["x"], data["x"])
        np.testing.assert_allclose(out["logl"], data["logl"])
        np.testing.assert_allclose(out["logp"], data["logp"])
        np.testing.assert_array_equal(out["steps"], jnp.asarray(0, dtype=jnp.int64))
        np.testing.assert_allclose(out["accept"], 0.0)

    @chex.all_variants(with_pmap=False)
    def test_geom(self):
        data = self._data()
        geom_mu = jnp.array([0.1, -0.1], dtype=data["u"].dtype)
        geom_cov = jnp.array([[1.0, 0.2], [0.2, 1.5]], dtype=data["u"].dtype)

        out = self.variant(
            lambda key: self._run(
                key,
                data=data,
                n_max=3,
                geom_mu=geom_mu,
                geom_cov=geom_cov,
                geom_nu=jnp.asarray(5.0, dtype=data["u"].dtype),
            )
        )(self.key)

        assert out["u"].shape == data["u"].shape
        assert out["x"].shape == data["x"].shape
        assert bool(jnp.all(jnp.isfinite(out["u"])))
        assert bool(jnp.all(jnp.isfinite(out["x"])))
        assert 0.0 <= float(out["accept"]) <= 1.0

    @chex.all_variants(with_pmap=False)
    def test_dtype(self):
        data = self._data(dtype=jnp.float64)

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


if __name__ == "__main__":
    absltest.main()