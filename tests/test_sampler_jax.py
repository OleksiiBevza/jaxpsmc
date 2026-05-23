import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.particles_jax import ParticlesState, ParticlesStep, init_particles_state_jax
from jaxpsmc.prior_jax import NORMAL, Prior
from jaxpsmc.sampler_jax import (
    IdentityBijectionJAX,
    IdentityFlowJAX,
    RunOutputJAX,
    SamplerConfigJAX,
    SamplerJAX,
    _build_step_from_particles,
    _metric_code,
    _replace_inf_rows,
    _resample_code,
    make_run_fn,
)


class SamplerTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(0)

    def _prior(self, dtype=jnp.float64):
        kinds = jnp.array([NORMAL, NORMAL], dtype=jnp.int32)
        params = jnp.array([[0.0, 1.0], [0.5, 1.5]], dtype=dtype)
        return Prior.create(kinds, params)
   
    def _cfg(self, **kwargs):
        base = dict(
            n_dim=2,
            n_effective=2,
            n_active=4,
            n_prior=4,
            n_total=2,
            n_steps=0,
            n_max_steps=0,
            proposal_scale=0.0,
            kernel="pcn",
            keep_max=4,
            trim_ess=0.99,
            bins=16,
            bisect_steps=8,
            sampling_mode="truncated_persistent",
            preconditioned=False,
            dynamic=False,
            metric="ess",
            resample="mult",
            transform="probit",
            blob_dim=0,
        )
        base.update(kwargs)
        return SamplerConfigJAX(**base)

    def _like_scalar(self, x):
        return -0.5 * jnp.sum(x * x)

    def _like_blob(self, x):
        ll = -0.5 * jnp.sum(x * x)
        blob = jnp.array([jnp.sum(x)], dtype=x.dtype)
        return ll, blob

    def _approx(self, x):
        return -0.25 * jnp.sum(x * x)

    def _arrays(self, dtype=jnp.float64):
        x = jnp.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, -1.0], [-1.0, 0.5]],
            dtype=dtype,
        )
        u = x + jnp.asarray(0.1, dtype=dtype)
        logdetj = jnp.arange(4, dtype=dtype)
        logp = -0.5 * jnp.sum(x * x, axis=1)
        logl = jnp.array([0.0, jnp.inf, -jnp.inf, -1.0], dtype=dtype)
        blobs = jnp.arange(4, dtype=dtype).reshape(4, 1)
        return x, u, logdetj, logp, logl, blobs

    def _assert_valid_run_output(self, out, *, max_steps, n_active, n_dim, blob_dim):
        assert isinstance(out, RunOutputJAX)
        assert isinstance(out.state, ParticlesState)

        t = int(out.state.t)
        assert 1 <= t <= max_steps

        assert out.state.u.shape == (max_steps, n_active, n_dim)
        assert out.state.x.shape == (max_steps, n_active, n_dim)
        assert out.state.logdetj.shape == (max_steps, n_active)
        assert out.state.logl.shape == (max_steps, n_active)
        assert out.state.logp.shape == (max_steps, n_active)
        assert out.state.blobs.shape == (max_steps, n_active, blob_dim)

        assert bool(jnp.all(jnp.isfinite(out.state.u[:t])))
        assert bool(jnp.all(jnp.isfinite(out.state.x[:t])))
        assert bool(jnp.all(jnp.isfinite(out.state.logl[:t])))
        assert bool(jnp.all(jnp.isfinite(out.state.logp[:t])))
        assert bool(jnp.all(jnp.isfinite(out.logz)))
        assert bool(jnp.isnan(out.logz_err))

    def _one_outer_step_cfg(self, *, kernel, preconditioned, **kwargs):
        base = dict(
            kernel=kernel,
            preconditioned=preconditioned,
            blob_dim=1,
            n_prior=4,
            n_active=4,
            n_effective=2,
            n_total=8,
            n_max_steps=1,
            n_steps=1,
            proposal_scale=0.05,
            keep_max=4,
            dynamic=False,
        )
        base.update(kwargs)
        return self._cfg(**base)

    def _assert_one_outer_step_run(self, out, *, blob_dim=1):
        self._assert_valid_run_output(
            out,
            max_steps=2,
            n_active=4,
            n_dim=2,
            blob_dim=blob_dim,
        )
        assert int(out.state.t) == 2

    def _assert_noop_mutation_step(self, out):
        self._assert_one_outer_step_run(out, blob_dim=1)
        np.testing.assert_allclose(out.state.steps[1], 0.0)
        np.testing.assert_allclose(out.state.accept[1], 0.0)
        np.testing.assert_allclose(out.state.calls[1], out.state.calls[0])

    def _assert_active_mutation_step(self, out):
        self._assert_one_outer_step_run(out, blob_dim=1)
        assert bool(out.state.steps[1] > 0)
        assert bool(out.state.calls[1] > 0)
        assert bool(jnp.isfinite(out.state.accept[1]))
        assert bool(jnp.isfinite(out.state.efficiency[1]))

    def test_codes(self):
        assert int(_metric_code("ess")) == 0
        assert int(_metric_code("ESS")) == 0
        assert int(_metric_code("uss")) == 1
        assert int(_metric_code("USS")) == 1

        assert int(_resample_code("mult")) == 0
        assert int(_resample_code("MULT")) == 0
        assert int(_resample_code("syst")) == 1
        assert int(_resample_code("SYST")) == 1

        with self.assertRaises(ValueError):
            _metric_code("bad")
        with self.assertRaises(ValueError):
            _resample_code("bad")

    def test_config(self):
        cfg = self._cfg()

        assert cfg.n_dim == 2
        assert cfg.n_effective == 2
        assert cfg.n_active == 4
        assert cfg.n_prior == 4
        assert cfg.n_total == 2
        assert cfg.kernel == "pcn"
        assert cfg.preconditioned is False
        assert cfg.dynamic is False
        assert cfg.metric == "ess"
        assert cfg.resample == "mult"
        assert cfg.sampling_mode == "truncated_persistent" 

    def test_config_kernel_api(self):
        cfg_pcn = self._cfg(kernel="pcn")
        cfg_li_pcn = self._cfg(kernel="li_pcn")

        assert cfg_pcn.kernel == "pcn"
        assert cfg_li_pcn.kernel == "li_pcn"

        with self.assertRaisesRegex(ValueError, "kernel must be one of"):
            self._cfg(kernel="none")
        with self.assertRaisesRegex(ValueError, "kernel must be one of"):
            self._cfg(kernel="bad")

    def test_config_kernel_is_independent_from_preconditioned_flag(self):
        cfg_pcn_noop = self._cfg(kernel="pcn", preconditioned=False)
        cfg_li_noop = self._cfg(kernel="li_pcn", preconditioned=False)
        cfg_pcn_active = self._cfg(kernel="pcn", preconditioned=True)
        cfg_li_active = self._cfg(kernel="li_pcn", preconditioned=True)

        assert cfg_pcn_noop.kernel == "pcn"
        assert cfg_li_noop.kernel == "li_pcn"
        assert cfg_pcn_noop.preconditioned is False
        assert cfg_li_noop.preconditioned is False

        assert cfg_pcn_active.kernel == "pcn"
        assert cfg_li_active.kernel == "li_pcn"
        assert cfg_pcn_active.preconditioned is True
        assert cfg_li_active.preconditioned is True

    def test_config_sampling_mode_persistent(self):
        cfg = self._cfg(sampling_mode="persistent")

        assert cfg.sampling_mode == "persistent"

    def test_config_sampling_mode_truncated_persistent(self):
        cfg = self._cfg(sampling_mode="truncated_persistent")

        assert cfg.sampling_mode == "truncated_persistent"      

    def test_config_bad(self):
        with self.assertRaises(ValueError):
            self._cfg(n_dim=0)
        with self.assertRaises(ValueError):
            self._cfg(n_active=0)
        with self.assertRaises(ValueError):
            self._cfg(n_effective=0)
        with self.assertRaises(ValueError):
            self._cfg(n_prior=5)
        with self.assertRaises(ValueError):
            self._cfg(keep_max=0)
        with self.assertRaises(ValueError):
            self._cfg(sampling_mode="bad")        

    def test_bijection(self):
        bij = IdentityBijectionJAX()
        u = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)

        theta, logdet = bij.transform_and_log_det(u)
        back, inv_logdet = bij.inverse_and_log_det(theta)

        np.testing.assert_allclose(theta, u)
        np.testing.assert_allclose(back, u)
        np.testing.assert_allclose(logdet, 0.0)
        np.testing.assert_allclose(inv_logdet, 0.0)
        assert theta.dtype == u.dtype
        assert logdet.shape == ()

    def test_bijection_batch(self):
        bij = IdentityBijectionJAX()
        u = jnp.array([[1.0, -2.0], [0.5, 0.25]], dtype=jnp.float32)

        theta, logdet = bij.transform_and_log_det(u)
        back, inv_logdet = bij.inverse_and_log_det(theta)

        np.testing.assert_allclose(theta, u)
        np.testing.assert_allclose(back, u)
        np.testing.assert_allclose(logdet, jnp.zeros((2,), dtype=u.dtype))
        np.testing.assert_allclose(inv_logdet, jnp.zeros((2,), dtype=u.dtype))

    def test_bijection_tree(self):
        bij = IdentityBijectionJAX()
        leaves, treedef = jax.tree_util.tree_flatten(bij)
        out = jax.tree_util.tree_unflatten(treedef, leaves)

        assert isinstance(out, IdentityBijectionJAX)
        assert len(leaves) == 0

    def test_flow(self):
        flow = IdentityFlowJAX(dim=3)
        key = jax.random.key(11)

        samples = flow.sample(key, 5)
        samples2 = flow.sample(key, 5)

        assert samples.shape == (5, 3)
        np.testing.assert_allclose(samples, samples2)
        assert isinstance(flow.bijection, IdentityBijectionJAX)
        assert flow.fit(jnp.ones((2, 3))) is flow

    def test_flow_tree(self):
        flow = IdentityFlowJAX(dim=7)
        leaves, treedef = jax.tree_util.tree_flatten(flow)
        out = jax.tree_util.tree_unflatten(treedef, leaves)

        assert isinstance(out, IdentityFlowJAX)
        assert out.dim == 7
        assert len(leaves) == 0

    def test_output_tree(self):
        state = init_particles_state_jax(
            max_steps=2,
            n_particles=3,
            n_dim=2,
            blob_dim=1,
            dtype=jnp.float32,
        )
        out = RunOutputJAX(
            state=state,
            logz=jnp.asarray(1.25, dtype=jnp.float32),
            logz_err=jnp.asarray(jnp.nan, dtype=jnp.float32),
        )

        leaves, treedef = jax.tree_util.tree_flatten(out)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

        assert isinstance(rebuilt, RunOutputJAX)
        assert isinstance(rebuilt.state, ParticlesState)
        np.testing.assert_allclose(rebuilt.state.u, state.u)
        np.testing.assert_allclose(rebuilt.logz, out.logz)
        assert bool(jnp.isnan(rebuilt.logz_err))

    @chex.all_variants(with_pmap=False)
    def test_replace(self):
        x, u, logdetj, logp, logl, blobs = self._arrays(dtype=jnp.float32)

        key2, x2, u2, logdetj2, logp2, logl2, blobs2 = self.variant(
            lambda key, x, u, logdetj, logp, logl, blobs: _replace_inf_rows(
                key, x, u, logdetj, logp, logl, blobs
            )
        )(self.key, x, u, logdetj, logp, logl, blobs)

        assert x2.shape == x.shape
        assert u2.shape == u.shape
        assert logdetj2.shape == logdetj.shape
        assert logp2.shape == logp.shape
        assert logl2.shape == logl.shape
        assert blobs2.shape == blobs.shape
        assert bool(jnp.all(jnp.isfinite(logl2)))

        np.testing.assert_allclose(x2[0], x[0])
        np.testing.assert_allclose(x2[3], x[3])
        np.testing.assert_allclose(logl2[0], logl[0])
        np.testing.assert_allclose(logl2[3], logl[3])
        assert not np.array_equal(
            np.asarray(jax.random.key_data(key2)),
            np.asarray(jax.random.key_data(self.key)),
        )

        finite_x = np.asarray(
            jnp.take(x, jnp.asarray([0, 3], dtype=jnp.int32), axis=0)
        )
        for row in np.asarray(x2):
            assert np.any(np.all(np.isclose(row[None, :], finite_x), axis=1))

    @chex.all_variants(with_pmap=False)
    def test_replace_none(self):
        x, u, logdetj, logp, _logl, blobs = self._arrays(dtype=jnp.float32)
        logl = jnp.array([0.0, -0.5, -1.0, -1.5], dtype=jnp.float32)

        key2, x2, u2, logdetj2, logp2, logl2, blobs2 = self.variant(
            lambda key, x, u, logdetj, logp, logl, blobs: _replace_inf_rows(
                key, x, u, logdetj, logp, logl, blobs
            )
        )(self.key, x, u, logdetj, logp, logl, blobs)

        np.testing.assert_allclose(x2, x)
        np.testing.assert_allclose(u2, u)
        np.testing.assert_allclose(logdetj2, logdetj)
        np.testing.assert_allclose(logp2, logp)
        np.testing.assert_allclose(logl2, logl)
        np.testing.assert_allclose(blobs2, blobs)
        assert not np.array_equal(
            np.asarray(jax.random.key_data(key2)),
            np.asarray(jax.random.key_data(self.key)),
        )

    def test_step(self):
        x, u, logdetj, logp, _logl, blobs = self._arrays(dtype=jnp.float32)
        logl = jnp.array([0.0, -0.5, -1.0, -1.5], dtype=jnp.float32)

        step = _build_step_from_particles(
            u=u,
            x=x,
            logdetj=logdetj,
            logl=logl,
            logp=logp,
            blobs=blobs,
            iter_idx=jnp.asarray(3, dtype=jnp.int64),
            beta=jnp.asarray(0.75, dtype=jnp.float32),
            logz=jnp.asarray(-1.25, dtype=jnp.float32),
            calls=jnp.asarray(9.0, dtype=jnp.float32),
            steps=jnp.asarray(2.0, dtype=jnp.float32),
            efficiency=jnp.asarray(0.4, dtype=jnp.float32),
            ess=jnp.asarray(3.5, dtype=jnp.float32),
            accept=jnp.asarray(0.25, dtype=jnp.float32),
        )

        assert isinstance(step, ParticlesStep)
        np.testing.assert_allclose(step.u, u)
        np.testing.assert_allclose(step.x, x)
        np.testing.assert_allclose(step.logdetj, logdetj)
        np.testing.assert_allclose(step.logl, logl)
        np.testing.assert_allclose(step.logp, logp)
        np.testing.assert_allclose(step.logw, jnp.zeros_like(logl))
        np.testing.assert_allclose(step.blobs, blobs)
        np.testing.assert_array_equal(step.iter, jnp.asarray(3, dtype=jnp.int32))
        np.testing.assert_allclose(step.beta, 0.75)
        np.testing.assert_allclose(step.logz, -1.25)
        np.testing.assert_allclose(step.calls, 9.0)
        np.testing.assert_allclose(step.steps, 2.0)
        np.testing.assert_allclose(step.efficiency, 0.4)
        np.testing.assert_allclose(step.ess, 3.5)
        np.testing.assert_allclose(step.accept, 0.25)

    @chex.all_variants(with_pmap=False)
    def test_run_scalar(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(blob_dim=0, n_max_steps=0)
        run = make_run_fn(
            prior=prior,
            loglike_single_fn=self._like_scalar,
            loglike_approx_single_fn=None,
            cfg=cfg,
        )

        out = self.variant(lambda key: run(key))(self.key)

        assert isinstance(out, RunOutputJAX)
        assert isinstance(out.state, ParticlesState)
        assert int(out.state.t) == 1
        assert out.state.u.shape == (1, 4, 2)
        assert out.state.x.shape == (1, 4, 2)
        assert out.state.blobs.shape == (1, 4, 0)
        assert bool(jnp.all(jnp.isfinite(out.state.logl[0])))
        assert bool(jnp.all(jnp.isfinite(out.state.logp[0])))
        assert bool(jnp.isfinite(out.logz))
        assert bool(jnp.isnan(out.logz_err))

    @chex.all_variants(with_pmap=False)
    def test_run_blob(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(blob_dim=1, n_max_steps=0)
        run = make_run_fn(
            prior=prior,
            loglike_single_fn=self._like_blob,
            loglike_approx_single_fn=None,
            cfg=cfg,
        )

        out = self.variant(lambda key: run(key))(self.key)

        assert int(out.state.t) == 1
        assert out.state.blobs.shape == (1, 4, 1)
        np.testing.assert_allclose(out.state.blobs[0, :, 0], jnp.sum(out.state.x[0], axis=1))
        np.testing.assert_allclose(out.state.calls[0], 4.0)

    @chex.all_variants(with_pmap=False)
    def test_run_outer(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(blob_dim=1, n_max_steps=1, n_steps=0, preconditioned=False)
        run = make_run_fn(
            prior=prior,
            loglike_single_fn=self._like_blob,
            loglike_approx_single_fn=self._approx,
            cfg=cfg,
        )

        out = self.variant(lambda key: run(key))(self.key)

        assert isinstance(out, RunOutputJAX)
        assert 1 <= int(out.state.t) <= 2
        assert out.state.u.shape == (2, 4, 2)
        assert out.state.blobs.shape == (2, 4, 1)
        assert bool(jnp.all(jnp.isfinite(out.state.u[: out.state.t])))
        assert bool(jnp.all(jnp.isfinite(out.state.x[: out.state.t])))
        assert bool(jnp.isfinite(out.logz))

    @chex.all_variants(with_pmap=False)
    def test_run_outer_truncated_persistent_mode(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(
            sampling_mode="truncated_persistent",
            blob_dim=1,
            n_prior=8,
            n_active=4,
            n_effective=2,
            n_total=2,
            n_max_steps=1,
            n_steps=0,
            keep_max=4,
            preconditioned=False,
            dynamic=False,
        )
        run = make_run_fn(
            prior=prior,
            loglike_single_fn=self._like_blob,
            loglike_approx_single_fn=self._approx,
            cfg=cfg,
        )

        out = self.variant(lambda key: run(key))(self.key)

        # max_steps_total = n_prior // n_active + n_max_steps = 2 + 1
        self._assert_valid_run_output(
            out,
            max_steps=3,
            n_active=4,
            n_dim=2,
            blob_dim=1,
        )
        assert cfg.sampling_mode == "truncated_persistent"

    @chex.all_variants(with_pmap=False)
    def test_run_outer_persistent_mode(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(
            sampling_mode="persistent",
            blob_dim=1,
            n_prior=8,
            n_active=4,
            n_effective=2,
            n_total=2,
            n_max_steps=1,
            n_steps=0,
            keep_max=4,
            preconditioned=False,
            dynamic=False,
        )
        run = make_run_fn(
            prior=prior,
            loglike_single_fn=self._like_blob,
            loglike_approx_single_fn=self._approx,
            cfg=cfg,
        )

        out = self.variant(lambda key: run(key))(self.key)

        # max_steps_total = n_prior // n_active + n_max_steps = 2 + 1
        self._assert_valid_run_output(
            out,
            max_steps=3,
            n_active=4,
            n_dim=2,
            blob_dim=1,
        )
        assert cfg.sampling_mode == "persistent"

    @chex.all_variants(with_pmap=False)
    def test_run_outer_persistent_ignores_small_keep_max(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(
            sampling_mode="persistent",
            blob_dim=1,
            n_prior=8,
            n_active=4,
            n_effective=2,
            n_total=2,
            n_max_steps=1,
            n_steps=0,
            keep_max=1,
            trim_ess=0.10,
            preconditioned=False,
            dynamic=False,
        )
        run = make_run_fn(
            prior=prior,
            loglike_single_fn=self._like_blob,
            loglike_approx_single_fn=self._approx,
            cfg=cfg,
        )

        out = self.variant(lambda key: run(key))(self.key)

        self._assert_valid_run_output(
            out,
            max_steps=3,
            n_active=4,
            n_dim=2,
            blob_dim=1,
        )
        assert int(out.state.t) >= 2
        assert bool(jnp.isfinite(out.logz))





    @chex.all_variants(with_pmap=False)
    def test_run_outer_noop_pcn_kernel(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._one_outer_step_cfg(kernel="pcn", preconditioned=False)
        run = make_run_fn(
            prior=prior,
            loglike_single_fn=self._like_blob,
            loglike_approx_single_fn=self._approx,
            cfg=cfg,
        )

        out = self.variant(lambda key: run(key))(self.key)

        self._assert_noop_mutation_step(out)

    @chex.all_variants(with_pmap=False)
    def test_run_outer_noop_li_pcn_kernel(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._one_outer_step_cfg(kernel="li_pcn", preconditioned=False)
        run = make_run_fn(
            prior=prior,
            loglike_single_fn=self._like_blob,
            loglike_approx_single_fn=self._approx,
            cfg=cfg,
        )

        out = self.variant(lambda key: run(key))(self.key)

        self._assert_noop_mutation_step(out)

    @chex.all_variants(with_pmap=False)
    def test_run_outer_preconditioned_pcn_kernel(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._one_outer_step_cfg(kernel="pcn", preconditioned=True)
        run = make_run_fn(
            prior=prior,
            loglike_single_fn=self._like_blob,
            loglike_approx_single_fn=self._approx,
            cfg=cfg,
        )

        out = self.variant(lambda key: run(key))(self.key)

        self._assert_active_mutation_step(out)

    @chex.all_variants(with_pmap=False)
    def test_run_outer_preconditioned_li_pcn_kernel(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._one_outer_step_cfg(kernel="li_pcn", preconditioned=True)
        run = make_run_fn(
            prior=prior,
            loglike_single_fn=self._like_blob,
            loglike_approx_single_fn=self._approx,
            cfg=cfg,
        )

        out = self.variant(lambda key: run(key))(self.key)

        self._assert_active_mutation_step(out)


    def test_sampler_init(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(blob_dim=0, n_max_steps=0)

        sampler = SamplerJAX(prior, self._like_scalar, cfg)

        assert sampler.prior is prior
        assert sampler.cfg is cfg
        assert isinstance(sampler.flow, IdentityFlowJAX)

    def test_sampler_da_requires_approx(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(delayed_acceptance=True, blob_dim=0, n_max_steps=0)

        with self.assertRaises(ValueError):
            SamplerJAX(prior, self._like_scalar, cfg)

    def test_sampler_da_init(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(delayed_acceptance=True, blob_dim=0, n_max_steps=0)

        sampler = SamplerJAX(
            prior,
            self._like_scalar,
            cfg,
            loglike_approx_single_fn=self._approx,
        )

        assert isinstance(sampler.flow, IdentityFlowJAX)

    def test_sampler_da_init_persistent_mode(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(
            sampling_mode="persistent",
            delayed_acceptance=True,
            blob_dim=0,
            n_max_steps=0,
        )

        sampler = SamplerJAX(
            prior,
            self._like_scalar,
            cfg,
            loglike_approx_single_fn=self._approx,
        )

        assert isinstance(sampler.flow, IdentityFlowJAX)
        assert sampler.cfg.sampling_mode == "persistent"
        assert sampler.cfg.delayed_acceptance is True

    @chex.all_variants(with_pmap=False)
    def test_sampler_run(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(blob_dim=0, n_max_steps=0)
        sampler = SamplerJAX(prior, self._like_scalar, cfg)

        out = self.variant(lambda key: sampler.run(key, n_total=3))(self.key)

        assert isinstance(out, RunOutputJAX)
        assert int(out.state.t) == 1
        assert out.state.u.shape == (1, 4, 2)
        assert out.state.blobs.shape == (1, 4, 0)
        assert bool(jnp.isfinite(out.logz))

    @chex.all_variants(with_pmap=False)
    def test_sampler_run_persistent_mode(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(
            sampling_mode="persistent",
            blob_dim=0,
            n_prior=8,
            n_active=4,
            n_effective=2,
            n_total=2,
            n_max_steps=1,
            n_steps=0,
            keep_max=1,
            preconditioned=False,
            dynamic=False,
        )
        sampler = SamplerJAX(
            prior,
            self._like_scalar,
            cfg,
            loglike_approx_single_fn=self._approx,
        )

        out = self.variant(lambda key: sampler.run(key, n_total=2))(self.key)

        self._assert_valid_run_output(
            out,
            max_steps=3,
            n_active=4,
            n_dim=2,
            blob_dim=0,
        )
        assert sampler.cfg.sampling_mode == "persistent"

    def test_repro(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(blob_dim=1, n_max_steps=0)
        sampler = SamplerJAX(prior, self._like_blob, cfg)

        out1 = sampler.run(self.key)
        out2 = sampler.run(self.key)

        np.testing.assert_allclose(out1.state.u, out2.state.u)
        np.testing.assert_allclose(out1.state.x, out2.state.x)
        np.testing.assert_allclose(out1.state.logl, out2.state.logl)
        np.testing.assert_allclose(out1.state.logp, out2.state.logp)
        np.testing.assert_allclose(out1.state.blobs, out2.state.blobs)
        np.testing.assert_allclose(out1.logz, out2.logz)

    def test_repro_persistent_mode(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(
            sampling_mode="persistent",
            blob_dim=1,
            n_prior=8,
            n_active=4,
            n_effective=2,
            n_total=2,
            n_max_steps=1,
            n_steps=0,
            keep_max=1,
            preconditioned=False,
            dynamic=False,
        )
        sampler = SamplerJAX(
            prior,
            self._like_blob,
            cfg,
            loglike_approx_single_fn=self._approx,
        )

        out1 = sampler.run(self.key)
        out2 = sampler.run(self.key)

        np.testing.assert_allclose(out1.state.u, out2.state.u)
        np.testing.assert_allclose(out1.state.x, out2.state.x)
        np.testing.assert_allclose(out1.state.logl, out2.state.logl)
        np.testing.assert_allclose(out1.state.logp, out2.state.logp)
        np.testing.assert_allclose(out1.state.blobs, out2.state.blobs)
        np.testing.assert_allclose(out1.logz, out2.logz)


    @chex.all_variants(with_pmap=False)
    def test_dtype(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(blob_dim=1, n_max_steps=0)
        run = make_run_fn(
            prior=prior,
            loglike_single_fn=self._like_blob,
            loglike_approx_single_fn=None,
            cfg=cfg,
        )

        out = self.variant(lambda key: run(key))(self.key)
        expected_dtype = jnp.result_type(prior.params, jnp.float64)

        assert out.state.u.dtype == expected_dtype
        assert out.state.x.dtype == expected_dtype
        assert out.state.logl.dtype == expected_dtype
        assert out.state.logp.dtype == expected_dtype
        assert out.state.blobs.dtype == expected_dtype
        assert out.logz.dtype == expected_dtype
        assert out.logz_err.dtype == expected_dtype

    @chex.all_variants(with_pmap=False)
    def test_dtype_persistent_mode(self):
        prior = self._prior(dtype=jnp.float64)
        cfg = self._cfg(
            sampling_mode="persistent",
            blob_dim=1,
            n_prior=8,
            n_active=4,
            n_effective=2,
            n_total=2,
            n_max_steps=1,
            n_steps=0,
            keep_max=1,
            preconditioned=False,
            dynamic=False,
        )
        run = make_run_fn(
            prior=prior,
            loglike_single_fn=self._like_blob,
            loglike_approx_single_fn=self._approx,
            cfg=cfg,
        )

        out = self.variant(lambda key: run(key))(self.key)
        expected_dtype = jnp.result_type(prior.params, jnp.float64)

        assert out.state.u.dtype == expected_dtype
        assert out.state.x.dtype == expected_dtype
        assert out.state.logl.dtype == expected_dtype
        assert out.state.logp.dtype == expected_dtype
        assert out.state.blobs.dtype == expected_dtype
        assert out.logz.dtype == expected_dtype
        assert out.logz_err.dtype == expected_dtype


if __name__ == "__main__":
    absltest.main()
