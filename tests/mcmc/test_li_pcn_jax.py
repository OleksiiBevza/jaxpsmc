import chex
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.mcmc.li_pcn_jax import (
    _empirical_li_geometry_from_cov,
    _li_log_reference,
    _li_pcn_proposal,
    likelihood_informed_pcn_jax,
)
from jaxpsmc.mcmc.flow_jax import (
    _flow_u_to_theta_jax,
    _flow_theta_to_u_jax,
)
from jaxpsmc.sampler.sampler_helper_jax import mutate
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
    def __init__(self, bijection=None):
        self.bijection = IdentityBijection() if bijection is None else bijection


class LikelihoodInformedPcnTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(7)
        self.flow = Flow()

    def _cfg(self, dim):
        cfg = init_bounds_config_jax(dim, scale=False)
        msk = masks_jax(cfg["low"], cfg["high"])
        return cfg, msk

    def _data(self, N=5, D=3, B=1, dtype=jnp.float64):
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

    def _current(self, N=5, D=3, B=1, dtype=jnp.float64):
        data = self._data(N=N, D=D, B=B, dtype=dtype)
        data["calls"] = jnp.asarray(5, dtype=jnp.int32)
        data["proposal_scale"] = jnp.asarray(0.2, dtype=dtype)
        return data

    def _loglike(self, x):
        ll = -0.5 * jnp.sum((x - 0.25) ** 2)
        blob = jnp.asarray([jnp.sum(x)], dtype=x.dtype)
        return ll, blob

    def _loglike0(self, x):
        ll = -0.5 * jnp.sum((x - 0.25) ** 2)
        blob = jnp.zeros((0,), dtype=x.dtype)
        return ll, blob

    def _approx(self, x):
        return -0.25 * jnp.sum((x - 0.25) ** 2)

    def _prior(self, x):
        return -0.5 * jnp.sum(x * x)

    def _run(self, key, data=None, **kwargs):
        if data is None:
            data = self._data()

        dim = data["u"].shape[1]
        cfg, msk = self._cfg(dim)

        return likelihood_informed_pcn_jax(
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
            geom_cov=kwargs.pop(
                "geom_cov",
                jnp.asarray(
                    [[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.75]],
                    dtype=data["u"].dtype,
                )[:dim, :dim],
            ),
            n_max=kwargs.pop("n_max", 4),
            n_steps=kwargs.pop("n_steps", 2),
            proposal_scale=kwargs.pop(
                "proposal_scale", jnp.asarray(0.2, dtype=data["u"].dtype)
            ),
            li_rank=kwargs.pop("li_rank", min(1, dim)),
            li_lis_scale=kwargs.pop("li_lis_scale", 1.0),
            li_cs_scale=kwargs.pop("li_cs_scale", 0.5),
            li_var_floor=kwargs.pop("li_var_floor", 1e-8),
            li_complement_var=kwargs.pop("li_complement_var", 1.0),
            use_delayed_acceptance=kwargs.pop(
                "use_delayed_acceptance", jnp.asarray(False)
            ),
            da_c_const=kwargs.pop("da_c_const", jnp.asarray(0.01)),
            da_d_const=kwargs.pop("da_d_const", jnp.asarray(2.0)),
            condition=kwargs.pop("condition", None),
        )

    def _assert_kernel_keys(self, out):
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

    def _assert_mutate_keys(self, out):
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

    def test_empirical_geometry_diagonal_covariance_uses_rank_and_complement(self):
        cov = jnp.diag(jnp.asarray([1.0, 3.0, 2.0], dtype=jnp.float64))

        eigvecs, var_dir, active, cov_ref = _empirical_li_geometry_from_cov(
            cov,
            li_rank=2,
            li_var_floor=0.1,
            li_complement_var=0.5,
        )

        np.testing.assert_array_equal(active, jnp.asarray([True, True, False]))
        np.testing.assert_allclose(var_dir, jnp.asarray([3.1, 2.1, 0.5], dtype=jnp.float64))
        np.testing.assert_allclose(eigvecs.T @ eigvecs, jnp.eye(3, dtype=jnp.float64), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cov_ref, jnp.diag(jnp.asarray([0.6, 3.2, 2.2], dtype=jnp.float64)), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cov_ref, cov_ref.T, rtol=1e-6, atol=1e-6)

    def test_empirical_geometry_clips_rank_to_dimension(self):
        cov = jnp.diag(jnp.asarray([1.0, 4.0], dtype=jnp.float64))

        _eigvecs, var_dir, active, cov_ref = _empirical_li_geometry_from_cov(
            cov,
            li_rank=10,
            li_var_floor=1e-6,
            li_complement_var=0.25,
        )

        np.testing.assert_array_equal(active, jnp.asarray([True, True]))
        assert bool(jnp.all(var_dir > 0.0))
        assert cov_ref.shape == (2, 2)
        assert bool(jnp.all(jnp.linalg.eigvalsh(cov_ref) > 0.0))

    def test_log_reference_matches_manual_quadratic_form(self):
        theta = jnp.asarray([[1.0, -1.0], [3.0, 1.0]], dtype=jnp.float64)
        mu = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
        eigvecs = jnp.eye(2, dtype=jnp.float64)
        var_dir = jnp.asarray([1.0, 4.0], dtype=jnp.float64)

        out = _li_log_reference(theta, mu, eigvecs, var_dir)

        expected = jnp.asarray([0.0, -2.5], dtype=jnp.float64)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_li_proposal_zero_sigma_is_effectively_identity(self):
        theta = jnp.asarray([[0.0, 1.0], [2.0, -1.0]], dtype=jnp.float64)
        mu = jnp.asarray([0.5, -0.5], dtype=jnp.float64)
        eigvecs = jnp.eye(2, dtype=jnp.float64)
        var_dir = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
        active = jnp.asarray([True, False])

        out = _li_pcn_proposal(
            self.key,
            theta,
            mu,
            eigvecs,
            var_dir,
            active,
            jnp.asarray(0.0, dtype=jnp.float64),
            li_lis_scale=1.0,
            li_cs_scale=1.0,
        )

        np.testing.assert_allclose(out, theta, rtol=1e-9, atol=1e-9)

    @chex.all_variants(with_pmap=False)
    def test_noop(self):
        data = self._data()

        out = self.variant(lambda key: self._run(key, data=data, n_max=0))(self.key)

        self._assert_kernel_keys(out)
        np.testing.assert_allclose(out["u"], data["u"])
        np.testing.assert_allclose(out["x"], data["x"])
        np.testing.assert_allclose(out["logdetj"], data["logdetj"])
        np.testing.assert_allclose(out["logdetj_flow"], data["logdetj_flow"])
        np.testing.assert_allclose(out["logl"], data["logl"])
        np.testing.assert_allclose(out["logp"], data["logp"])
        np.testing.assert_allclose(out["blobs"], data["blobs"])
        np.testing.assert_array_equal(jax.random.key_data(out["key"]), jax.random.key_data(self.key))
        np.testing.assert_allclose(out["accept"], 0.0)
        np.testing.assert_array_equal(out["steps"], jnp.asarray(0, dtype=jnp.int32))
        np.testing.assert_array_equal(out["calls"], jnp.asarray(0, dtype=jnp.int32))
        np.testing.assert_allclose(out["proposal_scale"], 0.2)

    @chex.all_variants(with_pmap=False)
    def test_proposal_scale_is_capped_even_when_no_mutation_steps_run(self):
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
        np.testing.assert_allclose(out["efficiency"], out["proposal_scale"], rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_shapes_and_finite_outputs(self):
        data = self._data(N=5, D=3, B=1)

        out = self.variant(lambda key: self._run(key, data=data, n_max=3))(self.key)

        self._assert_kernel_keys(out)
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
        assert bool(jnp.isfinite(out["proposal_scale"]))

        assert 0 <= int(out["steps"]) <= 3
        assert 0 <= int(out["calls"]) <= 3 * data["u"].shape[0]
        assert 0.0 <= float(out["accept"]) <= 1.0
        assert 0.0 <= float(out["proposal_scale"]) <= 0.99

    @chex.all_variants(with_pmap=False)
    def test_reproducible_for_same_key(self):
        data = self._data()

        def run(key):
            return self._run(key, data=data, n_max=3)

        out1 = self.variant(run)(self.key)
        out2 = self.variant(run)(self.key)

        np.testing.assert_array_equal(jax.random.key_data(out1["key"]), jax.random.key_data(out2["key"]))
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
    def test_key_advances_when_mutation_loop_runs(self):
        data = self._data()

        out = self.variant(lambda key: self._run(key, data=data, n_max=2))(self.key)

        assert not np.array_equal(
            np.asarray(jax.random.key_data(out["key"])),
            np.asarray(jax.random.key_data(self.key)),
        )

    @chex.all_variants(with_pmap=False)
    def test_blob0(self):
        data = self._data(N=5, D=3, B=0)

        out = self.variant(
            lambda key: self._run(
                key,
                data=data,
                n_max=2,
                loglike_fn=self._loglike0,
            )
        )(self.key)

        assert out["blobs"].shape == (5, 0)
        np.testing.assert_allclose(out["blobs"], jnp.zeros((5, 0), dtype=data["u"].dtype))

    @chex.all_variants(with_pmap=False)
    def test_delayed_acceptance_path(self):
        data = self._data()

        out = self.variant(
            lambda key: self._run(
                key,
                data=data,
                n_max=3,
                use_delayed_acceptance=jnp.asarray(True),
            )
        )(self.key)

        self._assert_kernel_keys(out)
        assert out["u"].shape == data["u"].shape
        assert out["x"].shape == data["x"].shape
        assert out["blobs"].shape == data["blobs"].shape
        assert 0 <= int(out["steps"]) <= 3
        assert 0 <= int(out["calls"]) <= 3 * data["u"].shape[0]
        assert 0.0 <= float(out["accept"]) <= 1.0
        assert bool(jnp.isfinite(out["proposal_scale"]))

    @chex.all_variants(with_pmap=False)
    def test_affine_flow_refreshes_flow_logdet_before_noop_exit(self):
        data = self._data(D=3)
        flow = Flow(AffineBijection(scale=2.0, shift=1.0))
        expected_logdet = -3.0 * jnp.log(jnp.asarray(2.0, dtype=data["u"].dtype))

        out = self.variant(
            lambda key: self._run(key, data=data, flow=flow, n_max=0)
        )(self.key)

        np.testing.assert_allclose(
            out["logdetj_flow"],
            expected_logdet * jnp.ones((data["u"].shape[0],), dtype=data["u"].dtype),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(out["u"], data["u"])
        np.testing.assert_allclose(out["x"], data["x"])

    @chex.all_variants(with_pmap=False)
    def test_geometry_options_rank_zero_and_full_rank_are_valid(self):
        data = self._data()
        geom_mu = jnp.asarray([0.1, -0.1, 0.2], dtype=data["u"].dtype)
        geom_cov = jnp.asarray(
            [[1.0, 0.3, 0.1], [0.3, 2.0, 0.0], [0.1, 0.0, 0.5]],
            dtype=data["u"].dtype,
        )

        out_rank0 = self.variant(
            lambda key: self._run(
                key,
                data=data,
                n_max=2,
                geom_mu=geom_mu,
                geom_cov=geom_cov,
                li_rank=0,
            )
        )(self.key)

        out_full = self.variant(
            lambda key: self._run(
                key,
                data=data,
                n_max=2,
                geom_mu=geom_mu,
                geom_cov=geom_cov,
                li_rank=3,
            )
        )(self.key)

        assert out_rank0["u"].shape == data["u"].shape
        assert out_full["u"].shape == data["u"].shape
        assert bool(jnp.all(jnp.isfinite(out_rank0["u"])))
        assert bool(jnp.all(jnp.isfinite(out_full["u"])))
        assert 0.0 <= float(out_rank0["accept"]) <= 1.0
        assert 0.0 <= float(out_full["accept"]) <= 1.0

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

    @chex.all_variants(with_pmap=False)
    def test_mutate_dispatch_li_pcn_noop_when_preconditioned_false(self):
        cur = self._current()
        cfg, msk = self._cfg(cur["u"].shape[1])

        def run(k, c):
            return mutate(
                k,
                c,
                use_preconditioned_pcn=jnp.asarray(False),
                loglike_single_fn=self._loglike,
                loglike_approx_single_fn=self._approx,
                logprior_fn=self._prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((3,), dtype=c["u"].dtype),
                geom_cov=jnp.eye(3, dtype=c["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=c["u"].dtype),
                li_geom_mu=jnp.asarray([1.0, 2.0, 3.0], dtype=c["u"].dtype),
                li_geom_cov=2.0 * jnp.eye(3, dtype=c["u"].dtype),
                kernel="li_pcn",
                n_max=0,
                n_steps=1,
            )

        key_out, out, info = self.variant(run)(self.key, cur)

        self._assert_mutate_keys(out)
        np.testing.assert_array_equal(jax.random.key_data(key_out), jax.random.key_data(self.key))
        np.testing.assert_allclose(out["u"], cur["u"])
        np.testing.assert_allclose(out["x"], cur["x"])
        np.testing.assert_allclose(out["logl"], cur["logl"])
        np.testing.assert_allclose(out["logp"], cur["logp"])
        np.testing.assert_array_equal(out["calls"], cur["calls"])
        np.testing.assert_array_equal(out["steps"], jnp.asarray(0, dtype=jnp.int32))
        np.testing.assert_allclose(out["accept"], 0.0)
        np.testing.assert_array_equal(info["calls_increment"], jnp.asarray(0, dtype=jnp.int32))

    @chex.all_variants(with_pmap=False)
    def test_mutate_dispatch_li_pcn_active(self):
        cur = self._current()
        cfg, msk = self._cfg(cur["u"].shape[1])

        def run(k, c):
            return mutate(
                k,
                c,
                use_preconditioned_pcn=jnp.asarray(True),
                loglike_single_fn=self._loglike,
                loglike_approx_single_fn=self._approx,
                logprior_fn=self._prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((3,), dtype=c["u"].dtype),
                geom_cov=jnp.eye(3, dtype=c["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=c["u"].dtype),
                li_geom_mu=jnp.asarray([0.1, -0.1, 0.2], dtype=c["u"].dtype),
                li_geom_cov=jnp.asarray(
                    [[1.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 0.75]],
                    dtype=c["u"].dtype,
                ),
                kernel="li_pcn",
                li_rank=1,
                li_lis_scale=1.0,
                li_cs_scale=0.5,
                li_var_floor=1e-8,
                li_complement_var=1.0,
                n_max=2,
                n_steps=1,
            )

        key_out, out, info = self.variant(run)(self.key, cur)

        self._assert_mutate_keys(out)
        assert not np.array_equal(jax.random.key_data(key_out), jax.random.key_data(self.key))
        assert out["u"].shape == cur["u"].shape
        assert out["x"].shape == cur["x"].shape
        assert out["blobs"].shape == cur["blobs"].shape
        assert bool(jnp.all(jnp.isfinite(out["u"])))
        assert bool(jnp.all(jnp.isfinite(out["x"])))
        assert bool(jnp.isfinite(out["proposal_scale"]))
        assert 0 <= int(out["steps"]) <= 2
        assert int(out["calls"]) >= int(cur["calls"])
        np.testing.assert_array_equal(out["calls"], cur["calls"] + info["calls_increment"])
        np.testing.assert_allclose(out["proposal_scale"], info["proposal_scale"], rtol=1e-6, atol=1e-6)
        assert 0.0 <= float(out["accept"]) <= 1.0

    def test_mutate_rejects_unknown_kernel(self):
        cur = self._current()
        cfg, msk = self._cfg(cur["u"].shape[1])

        with self.assertRaisesRegex(ValueError, "kernel must be one of"):
            mutate(
                self.key,
                cur,
                use_preconditioned_pcn=jnp.asarray(True),
                loglike_single_fn=self._loglike,
                loglike_approx_single_fn=self._approx,
                logprior_fn=self._prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((3,), dtype=cur["u"].dtype),
                geom_cov=jnp.eye(3, dtype=cur["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=cur["u"].dtype),
                kernel="bad",
                n_max=1,
                n_steps=1,
            )


if __name__ == "__main__":
    absltest.main()
