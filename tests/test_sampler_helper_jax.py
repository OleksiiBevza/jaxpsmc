import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.particles_jax import ParticlesStep, init_particles_state_jax, record_step_jax
from jaxpsmc.sampler_helper_jax import (
    _metric_value,
    mutate,
    not_termination_jax,
    posterior_jax,
    reweight_step_jax,
    resample_particles_jax,
)
from jaxpsmc.sampler_jax import IdentityFlowJAX
from jaxpsmc.scaler_jax import init_bounds_config_jax, masks_jax


class HelperTest(chex.TestCase):
    def _step(self, logl, beta, logz):
        u = jnp.array([[0.0], [1.0]])
        x = u
        z = jnp.zeros((2,))
        return ParticlesStep(
            u=u,
            x=x,
            logdetj=z,
            logl=jnp.asarray(logl),
            logp=jnp.array([0.0, 0.0]),
            logw=z,
            blobs=jnp.zeros((2, 0)),
            iter=jnp.array(0, dtype=jnp.int32),
            logz=jnp.asarray(logz),
            calls=jnp.array(2.0),
            steps=jnp.array(1.0),
            efficiency=jnp.array(1.0),
            ess=jnp.array(2.0),
            accept=jnp.array(1.0),
            beta=jnp.asarray(beta),
        )

    def _state(self):
        state = init_particles_state_jax(3, 2, 1)
        state = record_step_jax(state, self._step([0.0, 0.0], 0.0, 0.0))
        state = record_step_jax(state, self._step([1.0, 2.0], 0.5, 0.1))
        return state

    @chex.all_variants(with_pmap=False)
    def test_metric_value(self):
        w = jnp.array([0.5, 0.5])
        ess = self.variant(lambda z: _metric_value(z, jnp.int32(0), 2))(w)
        uss = self.variant(lambda z: _metric_value(z, jnp.int32(1), 2))(w)
        np.testing.assert_allclose(ess, 2.0)
        np.testing.assert_allclose(uss, 1.5)

    @chex.all_variants(with_pmap=False)
    def test_reweight(self):
        run = lambda state: reweight_step_jax(
            state,
            n_effective=jnp.int32(2),
            metric_id=jnp.int32(0),
            dynamic=jnp.asarray(False),
            n_active=jnp.int32(2),
            dynamic_ratio=jnp.array(1.0),
            bins=32,
            bisect_steps=16,
            keep_max=3,
            trim_ess=0.95,
        )

        cur, n_eff, stats = self.variant(run)(self._state())

        assert cur["x"].shape[0] == 3
        np.testing.assert_allclose(jnp.sum(cur["weights"]), 1.0)
        assert 0.5 <= float(cur["beta"]) <= 1.0
        assert int(n_eff) == 2
        assert jnp.isfinite(stats["logz"])

    @chex.all_variants(with_pmap=False)
    def test_resample(self):
        cur = {
            "u": jnp.arange(4.0)[:, None],
            "x": jnp.arange(4.0)[:, None],
            "logdetj": jnp.zeros((4,)),
            "logl": jnp.zeros((4,)),
            "logp": jnp.zeros((4,)),
            "weights": jnp.array([0.7, 0.2, 0.1, 0.0]),
            "blobs": jnp.zeros((4, 0)),
        }
        out, status, _ = self.variant(
            lambda d: resample_particles_jax(
                d,
                key=jax.random.key(3),
                n_active=3,
                method_code=jnp.int32(1),
            )
        )(cur)

        assert out["x"].shape == (3, 1)
        np.testing.assert_allclose(out["weights"], jnp.full((3,), 1.0 / 3.0))
        assert status == 0

    def test_mutate_noop(self):
        cfg = init_bounds_config_jax(1, scale=False)
        msk = masks_jax(cfg["low"], cfg["high"])
        cur = {
            "u": jnp.array([[0.0], [1.0]]),
            "x": jnp.array([[0.0], [1.0]]),
            "logdetj": jnp.zeros((2,)),
            "logl": jnp.zeros((2,)),
            "logp": jnp.zeros((2,)),
            "logdetj_flow": jnp.zeros((2,)),
            "blobs": jnp.zeros((2, 0)),
            "beta": jnp.array(0.5),
            "calls": jnp.array(3, dtype=jnp.int32),
            "proposal_scale": jnp.array(0.2),
        }

        _, out, info = mutate(
            jax.random.key(0),
            cur,
            use_preconditioned_pcn=jnp.asarray(False),
            loglike_single_fn=lambda x: (jnp.array(0.0), jnp.zeros((0,))),
            loglike_approx_single_fn=lambda x: jnp.array(0.0),
            logprior_fn=lambda x: jnp.array(0.0),
            flow=IdentityFlowJAX(1),
            scaler_cfg=cfg,
            scaler_masks=msk,
            geom_mu=jnp.zeros((1,)),
            geom_cov=jnp.eye(1),
            geom_nu=jnp.array(10.0),
            n_max=4,
            n_steps=2,
        )

        np.testing.assert_allclose(out["x"], cur["x"])
        assert info["calls_increment"] == 0
        assert out["steps"] == 0

    @chex.all_variants(with_pmap=False)
    def test_not_termination(self):
        state = self._state()
        keep_going = self.variant(not_termination_jax)(
            state,
            jnp.array(0.5),
            jnp.array(10.0),
            jnp.int32(0),
            jnp.int32(2),
        )
        assert bool(keep_going)

    @chex.all_variants(with_pmap=False)
    def test_posterior(self):
        out = self.variant(
            lambda s: posterior_jax(s, jax.random.key(4), do_resample=True)
        )(self._state())

        assert out.samples.shape[0] == 6
        assert out.weights.shape == (6,)
        assert out.samples_resampled.shape == out.samples.shape
        assert jnp.isfinite(out.logz_new)
        np.testing.assert_allclose(jnp.sum(out.weights), 1.0, atol=1e-6)


if __name__ == "__main__":
    absltest.main()
