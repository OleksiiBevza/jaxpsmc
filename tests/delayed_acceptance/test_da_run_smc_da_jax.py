import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.delayed_acceptance.da_run_smc_da_jax import (
    SMCDACarry,
    _step_mutated_particles_jax,
    run_smc_da_scan_jax,
    smc_da_step_jax,
)
from jaxpsmc.geometry.geometry_jax import Geometry
from jaxpsmc.particles_jax import (
    ParticlesStep,
    init_particles_state_jax,
    record_step_jax,
)
from jaxpsmc.sampler.sampler_jax import IdentityFlowJAX
from jaxpsmc.scaler_jax import init_bounds_config_jax, masks_jax


def _fake_mutate(
    key,
    cur,
    *,
    use_preconditioned_pcn,
    loglike_single_fn,
    logprior_fn,
    flow,
    scaler_cfg,
    scaler_masks,
    geom_mu,
    geom_cov,
    geom_nu,
    n_max,
    n_steps,
    condition=None,
):
    del (
        use_preconditioned_pcn,
        loglike_single_fn,
        logprior_fn,
        flow,
        scaler_cfg,
        scaler_masks,
        geom_mu,
        geom_cov,
        geom_nu,
        n_max,
        condition,
    )

    dtype = cur["logl"].dtype

    mutated = {
        "u": cur["u"],
        "x": cur["x"],
        "logdetj": cur["logdetj"],
        "logl": cur["logl"] + jnp.asarray(0.1, dtype=dtype),
        "logp": cur["logp"],
        "logdetj_flow": cur["logdetj_flow"],
        "blobs": cur["blobs"],
        "calls": cur["calls"] + jnp.asarray(5.0, dtype=cur["calls"].dtype),
        "steps": jnp.asarray(n_steps, dtype=jnp.int32),
        "efficiency": jnp.asarray(0.25, dtype=dtype),
        "accept": jnp.asarray(0.75, dtype=dtype),
        "proposal_scale": cur["proposal_scale"] + jnp.asarray(0.01, dtype=dtype),
    }
    info = {"calls_increment": jnp.asarray(5, dtype=jnp.int32)}
    return key, mutated, info


def _loglike(x):
    return -0.5 * jnp.sum((x - 0.2) ** 2), jnp.zeros((0,))


def _logprior(x):
    return -0.5 * jnp.sum(x * x)


class RunSMCDATest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.n_active = 2
        self.n_dim = 1
        self.keep_max = 4
        self.dtype = jnp.float32

        self.flow = IdentityFlowJAX(self.n_dim)
        self.cfg = init_bounds_config_jax(self.n_dim, scale=False)
        self.masks = masks_jax(self.cfg["low"], self.cfg["high"])

    def _step(self, logl, beta, logz):
        u = jnp.array([[0.0], [1.0]], dtype=self.dtype)
        z = jnp.zeros((2,), dtype=self.dtype)

        return ParticlesStep(
            u=u,
            x=u,
            logdetj=z,
            logl=jnp.asarray(logl, dtype=self.dtype),
            logp=z,
            logw=z,
            blobs=jnp.zeros((2, 0), dtype=self.dtype),
            iter=jnp.array(0, dtype=jnp.int32),
            logz=jnp.asarray(logz, dtype=self.dtype),
            calls=jnp.array(2.0, dtype=self.dtype),
            steps=jnp.array(1.0, dtype=self.dtype),
            efficiency=jnp.array(1.0, dtype=self.dtype),
            ess=jnp.array(2.0, dtype=self.dtype),
            accept=jnp.array(1.0, dtype=self.dtype),
            beta=jnp.asarray(beta, dtype=self.dtype),
        )

    def _state(self):
        state = init_particles_state_jax(self.keep_max, self.n_active, self.n_dim)
        state = record_step_jax(state, self._step([0.0, 0.0], 0.0, 0.0))
        state = record_step_jax(state, self._step([1.0, 2.0], 0.5, 0.1))
        return state

    def _geom(self):
        return Geometry(
            normal_mean=jnp.zeros((1,), dtype=self.dtype),
            normal_cov=jnp.eye(1, dtype=self.dtype),
            t_mean=jnp.zeros((1,), dtype=self.dtype),
            t_cov=jnp.eye(1, dtype=self.dtype),
            t_nu=jnp.array(10.0, dtype=self.dtype),
        )

    def _cur(self, beta=0.5):
        u = jnp.array([[0.0], [1.0]], dtype=self.dtype)

        return {
            "u": u,
            "x": u,
            "logdetj": jnp.zeros((2,), dtype=self.dtype),
            "logl": jnp.array([1.0, 2.0], dtype=self.dtype),
            "logp": jnp.zeros((2,), dtype=self.dtype),
            "blobs": jnp.zeros((2, 0), dtype=self.dtype),
            "beta": jnp.asarray(beta, dtype=self.dtype),
            "calls": jnp.array(3.0, dtype=self.dtype),
            "proposal_scale": jnp.array(0.2, dtype=self.dtype),
        }

    def _carry(self, beta=0.5):
        return SMCDACarry(
            key=jax.random.key(0),
            state=self._state(),
            current_particles=self._cur(beta=beta),
            geom=self._geom(),
            n_effective=jnp.asarray(2, dtype=jnp.int32),
            iteration=jnp.asarray(0, dtype=jnp.int32),
        )

    def _kwargs(
        self,
        n_outer_max_steps=3,
        n_mutation_steps=2,
        *,
        sampling_mode="truncated_persistent",
        keep_max=None,
        trim_ess=0.95,
        dynamic=False,
    ):
        return dict(
            n_total=jnp.asarray(20.0, dtype=self.dtype),
            metric_id=jnp.asarray(0, dtype=jnp.int32),
            dynamic=jnp.asarray(dynamic),
            n_active=self.n_active,
            n_outer_max_steps=n_outer_max_steps,
            n_mutation_max_steps=4,
            n_mutation_steps=n_mutation_steps,
            n_active_i32=jnp.asarray(self.n_active, dtype=jnp.int32),
            dynamic_ratio=jnp.asarray(1.0, dtype=self.dtype),
            resample_code=jnp.asarray(1, dtype=jnp.int32),
            use_preconditioned_pcn=jnp.asarray(False),
            keep_max=self.keep_max if keep_max is None else int(keep_max),
            bins=32,
            bisect_steps=16,
            trim_ess=float(trim_ess),
            sampling_mode=str(sampling_mode),
            flow=self.flow,
            scaler_cfg=self.cfg,
            scaler_masks=self.masks,
            mutation_fn=_fake_mutate,
            loglike_single_fn=_loglike,
            logprior_fn=_logprior,
        )

    def _assert_active_step(
        self,
        *,
        carry0,
        carry1,
        stats,
        expected_steps,
    ):
        assert bool(stats.active)
        assert int(carry1.iteration) == int(carry0.iteration) + 1
        assert int(carry1.state.t) == int(carry0.state.t) + 1

        assert carry1.current_particles["u"].shape == (self.n_active, self.n_dim)
        assert carry1.current_particles["x"].shape == (self.n_active, self.n_dim)
        assert carry1.current_particles["logdetj"].shape == (self.n_active,)
        assert carry1.current_particles["logl"].shape == (self.n_active,)
        assert carry1.current_particles["logp"].shape == (self.n_active,)
        assert carry1.current_particles["blobs"].shape == (self.n_active, 0)

        assert bool(jnp.all(jnp.isfinite(carry1.current_particles["u"])))
        assert bool(jnp.all(jnp.isfinite(carry1.current_particles["x"])))
        assert bool(jnp.all(jnp.isfinite(carry1.current_particles["logl"])))
        assert bool(jnp.all(jnp.isfinite(carry1.current_particles["logp"])))

        np.testing.assert_allclose(stats.accept, 0.75)
        assert int(stats.steps) == int(expected_steps)
        np.testing.assert_allclose(
            carry1.current_particles["proposal_scale"],
            0.21,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(stats.calls, carry1.current_particles["calls"])

        assert bool(jnp.isfinite(stats.beta))
        assert bool(jnp.isfinite(stats.logz))
        assert bool(jnp.isfinite(stats.ess))
        assert bool(jnp.isfinite(carry1.current_particles["beta"]))
        assert bool(jnp.isfinite(carry1.current_particles["calls"]))

    def test_record(self):
        mutated = {
            "u": jnp.array([[0.0], [1.0]]),
            "x": jnp.array([[2.0], [3.0]]),
            "logdetj": jnp.array([0.1, 0.2]),
            "logl": jnp.array([1.0, 2.0]),
            "logp": jnp.array([0.5, 0.6]),
            "blobs": jnp.zeros((2, 0)),
            "calls": jnp.array(7.0),
            "steps": jnp.array(3.0),
            "efficiency": jnp.array(0.4),
            "accept": jnp.array(0.8),
        }

        step = _step_mutated_particles_jax(
            mutated=mutated,
            iter_idx=jnp.asarray(5),
            beta=jnp.asarray(0.9),
            logz=jnp.asarray(1.5),
            ess=jnp.asarray(1.7),
        )

        np.testing.assert_allclose(step.u, mutated["u"])
        np.testing.assert_allclose(step.x, mutated["x"])
        np.testing.assert_allclose(step.logdetj, mutated["logdetj"])
        np.testing.assert_allclose(step.logl, mutated["logl"])
        np.testing.assert_allclose(step.logp, mutated["logp"])
        np.testing.assert_allclose(step.logw, jnp.zeros((2,)))
        assert int(step.iter) == 5
        np.testing.assert_allclose(step.beta, 0.9)
        np.testing.assert_allclose(step.logz, 1.5)
        np.testing.assert_allclose(step.ess, 1.7)
        np.testing.assert_allclose(step.accept, 0.8)

    @chex.all_variants(with_pmap=False)
    def test_skip(self):
        carry0 = self._carry(beta=0.5)

        carry1, stats = self.variant(
            lambda c: smc_da_step_jax(
                c,
                **self._kwargs(n_outer_max_steps=0),
            )
        )(carry0)

        assert not bool(stats.active)
        assert int(carry1.iteration) == int(carry0.iteration)
        assert int(carry1.state.t) == int(carry0.state.t)

        np.testing.assert_allclose(stats.beta, carry0.current_particles["beta"])
        np.testing.assert_allclose(stats.logz, 0.0)
        np.testing.assert_allclose(stats.ess, 0.0)
        np.testing.assert_allclose(stats.accept, 0.0)
        assert int(stats.steps) == 0
        np.testing.assert_allclose(stats.calls, carry0.current_particles["calls"])

    @chex.all_variants(with_pmap=False)
    def test_truncated_persistent(self):
        carry0 = self._carry(beta=0.5)

        carry1, stats = self.variant(
            lambda c: smc_da_step_jax(
                c,
                **self._kwargs(
                    n_outer_max_steps=3,
                    n_mutation_steps=2,
                    sampling_mode="truncated_persistent",
                ),
            )
        )(carry0)

        self._assert_active_step(
            carry0=carry0,
            carry1=carry1,
            stats=stats,
            expected_steps=2,
        )

    @chex.all_variants(with_pmap=False)
    def test_step_persistent(self):
        carry0 = self._carry(beta=0.5)

        carry1, stats = self.variant(
            lambda c: smc_da_step_jax(
                c,
                **self._kwargs(
                    n_outer_max_steps=3,
                    n_mutation_steps=2,
                    sampling_mode="persistent",
                ),
            )
        )(carry0)

        self._assert_active_step(
            carry0=carry0,
            carry1=carry1,
            stats=stats,
            expected_steps=2,
        )

    @chex.all_variants(with_pmap=False)
    def test_persistent_small_max(self):
        carry0 = self._carry(beta=0.5)

        carry1, stats = self.variant(
            lambda c: smc_da_step_jax(
                c,
                **self._kwargs(
                    n_outer_max_steps=3,
                    n_mutation_steps=2,
                    sampling_mode="persistent",
                    keep_max=1,
                    trim_ess=0.10,
                ),
            )
        )(carry0)

        self._assert_active_step(
            carry0=carry0,
            carry1=carry1,
            stats=stats,
            expected_steps=2,
        )

    def test_reject_invalid_mode(self):
        carry0 = self._carry(beta=0.5)

        with self.assertRaises(ValueError):
            smc_da_step_jax(
                carry0,
                **self._kwargs(
                    n_outer_max_steps=3,
                    n_mutation_steps=2,
                    sampling_mode="bad",
                ),
            )

    @chex.all_variants(with_pmap=False)
    def test_scan(self):
        carry0 = self._carry(beta=0.5)

        carry1, stats = self.variant(
            lambda c: run_smc_da_scan_jax(
                c,
                n_scan_steps=3,
                **self._kwargs(n_outer_max_steps=1, n_mutation_steps=1),
            )
        )(carry0)

        assert int(carry1.iteration) == 1
        assert int(carry1.state.t) == int(carry0.state.t) + 1

        np.testing.assert_array_equal(
            stats.active,
            jnp.array([True, False, False]),
        )
        assert stats.beta.shape == (3,)
        assert stats.logz.shape == (3,)
        assert stats.ess.shape == (3,)
        assert stats.accept.shape == (3,)
        assert stats.steps.shape == (3,)
        assert stats.calls.shape == (3,)

        assert int(stats.steps[0]) == 1
        np.testing.assert_allclose(stats.accept[0], 0.75)
        assert int(stats.steps[1]) == 0
        assert int(stats.steps[2]) == 0

    @chex.all_variants(with_pmap=False)
    def test_scan_truncated_persistent(self):
        carry0 = self._carry(beta=0.5)

        carry1, stats = self.variant(
            lambda c: run_smc_da_scan_jax(
                c,
                n_scan_steps=3,
                **self._kwargs(
                    n_outer_max_steps=1,
                    n_mutation_steps=1,
                    sampling_mode="truncated_persistent",
                ),
            )
        )(carry0)

        assert int(carry1.iteration) == 1
        assert int(carry1.state.t) == int(carry0.state.t) + 1

        np.testing.assert_array_equal(
            stats.active,
            jnp.array([True, False, False]),
        )
        assert stats.beta.shape == (3,)
        assert stats.logz.shape == (3,)
        assert stats.ess.shape == (3,)
        assert stats.accept.shape == (3,)
        assert stats.steps.shape == (3,)
        assert stats.calls.shape == (3,)

        assert int(stats.steps[0]) == 1
        np.testing.assert_allclose(stats.accept[0], 0.75)
        assert int(stats.steps[1]) == 0
        assert int(stats.steps[2]) == 0

    @chex.all_variants(with_pmap=False)
    def test_scan_persistent(self):
        carry0 = self._carry(beta=0.5)

        carry1, stats = self.variant(
            lambda c: run_smc_da_scan_jax(
                c,
                n_scan_steps=3,
                **self._kwargs(
                    n_outer_max_steps=1,
                    n_mutation_steps=1,
                    sampling_mode="persistent",
                ),
            )
        )(carry0)

        assert int(carry1.iteration) == 1
        assert int(carry1.state.t) == int(carry0.state.t) + 1

        np.testing.assert_array_equal(
            stats.active,
            jnp.array([True, False, False]),
        )
        assert stats.beta.shape == (3,)
        assert stats.logz.shape == (3,)
        assert stats.ess.shape == (3,)
        assert stats.accept.shape == (3,)
        assert stats.steps.shape == (3,)
        assert stats.calls.shape == (3,)

        assert int(stats.steps[0]) == 1
        np.testing.assert_allclose(stats.accept[0], 0.75)
        assert int(stats.steps[1]) == 0
        assert int(stats.steps[2]) == 0
        assert bool(jnp.isfinite(stats.beta[0]))
        assert bool(jnp.isfinite(stats.logz[0]))
        assert bool(jnp.isfinite(stats.ess[0]))

    @chex.all_variants(with_pmap=False)
    def test_scan_persistent(self):
        carry0 = self._carry(beta=0.5)

        carry1, stats = self.variant(
            lambda c: run_smc_da_scan_jax(
                c,
                n_scan_steps=3,
                **self._kwargs(
                    n_outer_max_steps=1,
                    n_mutation_steps=1,
                    sampling_mode="persistent",
                    keep_max=1,
                    trim_ess=0.10,
                ),
            )
        )(carry0)

        assert int(carry1.iteration) == 1
        assert int(carry1.state.t) == int(carry0.state.t) + 1
        np.testing.assert_array_equal(
            stats.active,
            jnp.array([True, False, False]),
        )
        assert int(stats.steps[0]) == 1
        assert bool(jnp.isfinite(stats.beta[0]))
        assert bool(jnp.isfinite(stats.logz[0]))
        assert bool(jnp.isfinite(stats.ess[0]))

    @chex.all_variants(with_pmap=False)
    def test_pers_trunc_valid_diagnostics(self):
        carry0 = self._carry(beta=0.5)

        carry_trunc, stats_trunc = self.variant(
            lambda c: smc_da_step_jax(
                c,
                **self._kwargs(
                    n_outer_max_steps=3,
                    n_mutation_steps=1,
                    sampling_mode="truncated_persistent",
                ),
            )
        )(carry0)

        carry_persist, stats_persist = self.variant(
            lambda c: smc_da_step_jax(
                c,
                **self._kwargs(
                    n_outer_max_steps=3,
                    n_mutation_steps=1,
                    sampling_mode="persistent",
                    keep_max=1,
                    trim_ess=0.10,
                ),
            )
        )(carry0)

        self._assert_active_step(
            carry0=carry0,
            carry1=carry_trunc,
            stats=stats_trunc,
            expected_steps=1,
        )
        self._assert_active_step(
            carry0=carry0,
            carry1=carry_persist,
            stats=stats_persist,
            expected_steps=1,
        )

        assert (
            carry_trunc.current_particles["u"].shape
            == carry_persist.current_particles["u"].shape
        )
        assert stats_trunc.beta.shape == stats_persist.beta.shape
        assert stats_trunc.logz.shape == stats_persist.logz.shape
        assert stats_trunc.ess.shape == stats_persist.ess.shape


if __name__ == "__main__":
    absltest.main()
