import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.delayed_acceptance.da_conservative_damh_jax import (
    _clean_log_ratio_jax,
    _log_accept_prob_jax,
    conservative_damh_step_parts_jax,
    conservative_damh_step_jax,
    mahalanobis_distance_jax,
)


class ConservativeDAMHTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(7)
        self.old = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [-1.0, 2.0],
            ]
        )
        self.new = jnp.array(
            [
                [1.0, 0.0],
                [1.0, 3.0],
                [1.0, 2.0],
            ]
        )
        self.cov = jnp.array(
            [
                [4.0, 0.0],
                [0.0, 1.0],
            ]
        )

    @chex.all_variants(with_pmap=False)
    def test_clean(self):
        log_ratio = jnp.array([jnp.nan, -jnp.inf, jnp.inf, -2.0, 3.0])

        out = self.variant(_clean_log_ratio_jax)(log_ratio)

        assert jnp.isneginf(out[0])
        assert jnp.isneginf(out[1])
        assert jnp.isposinf(out[2])
        np.testing.assert_allclose(out[3:], jnp.array([-2.0, 3.0]))

    @chex.all_variants(with_pmap=False)
    def test_logprob(self):
        log_ratio = jnp.array([jnp.nan, -2.0, 0.0, 3.0])

        out = self.variant(_log_accept_prob_jax)(log_ratio)

        assert jnp.isneginf(out[0])
        np.testing.assert_allclose(
            out[1:],
            jnp.array([-2.0, 0.0, 0.0]),
        )
        assert bool(jnp.all(out <= 0.0))

    @chex.all_variants(with_pmap=False)
    def test_maha(self):
        out = self.variant(
            lambda new, old: mahalanobis_distance_jax(
                new_particles=new,
                old_particles=old,
                cov=self.cov,
            )
        )(self.new, self.old)

        expected = jnp.array([0.5, 2.0, 1.0])
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_step(self):
        log_ratio_surrogate = jnp.array([-10.0, -1.0, 10.0])
        log_ratio_full = jnp.array([-3.0, 0.0, 2.0])

        out = self.variant(
            lambda key: conservative_damh_step_jax(
                key=key,
                new_particles=self.new,
                old_particles=self.old,
                cov=self.cov,
                log_ratio_surrogate=log_ratio_surrogate,
                log_ratio_full=log_ratio_full,
                c_const=jnp.array(0.01),
                d_const=jnp.array(2.0),
            )
        )(self.key)

        log_b = jnp.log(0.01)
        expected_stage1 = jnp.clip(log_ratio_surrogate, log_b, -log_b)
        expected_stage2 = log_ratio_full - expected_stage1
        expected_pre_prob = jnp.exp(jnp.minimum(expected_stage1, 0.0))
        expected_prob = expected_pre_prob * jnp.exp(jnp.minimum(expected_stage2, 0.0))

        np.testing.assert_allclose(out.log_ratio_stage1, expected_stage1)
        np.testing.assert_allclose(out.log_ratio_stage2, expected_stage2)
        np.testing.assert_allclose(out.expected_pre_accept, expected_pre_prob)
        np.testing.assert_allclose(out.prob_accept, expected_prob)

        assert out.pre_accept.shape == (3,)
        assert out.stage2_accept.shape == (3,)
        assert out.accept.shape == (3,)
        assert out.full_eval_mask.shape == (3,)
        assert out.proposal_dist.shape == (3,)

        np.testing.assert_array_equal(out.accept, out.pre_accept & out.stage2_accept)
        np.testing.assert_array_equal(out.full_eval_mask, out.pre_accept)
        assert int(out.full_calls) == int(jnp.sum(out.pre_accept))

        np.testing.assert_allclose(
            out.actual_dist,
            out.proposal_dist * out.accept.astype(out.proposal_dist.dtype),
        )
        np.testing.assert_allclose(
            out.expected_dist,
            out.proposal_dist * out.prob_accept,
        )

    @chex.all_variants(with_pmap=False)
    def test_reject(self):
        log_ratio_surrogate = jnp.array([jnp.nan, jnp.nan, jnp.nan])
        log_ratio_full = jnp.array([0.0, 1.0, -1.0])

        out = self.variant(
            lambda key: conservative_damh_step_jax(
                key=key,
                new_particles=self.new,
                old_particles=self.old,
                cov=self.cov,
                log_ratio_surrogate=log_ratio_surrogate,
                log_ratio_full=log_ratio_full,
            )
        )(self.key)

        np.testing.assert_array_equal(
            out.pre_accept,
            jnp.array([False, False, False]),
        )
        np.testing.assert_array_equal(
            out.accept,
            jnp.array([False, False, False]),
        )
        np.testing.assert_allclose(out.expected_pre_accept, jnp.zeros((3,)))
        np.testing.assert_allclose(out.prob_accept, jnp.zeros((3,)))
        np.testing.assert_allclose(out.actual_dist, jnp.zeros((3,)))
        assert int(out.full_calls) == 0

    @chex.all_variants(with_pmap=False)
    def test_parts(self):
        approx_posterior_old = jnp.array([0.0, 1.0, -1.0])
        approx_posterior_new = jnp.array([1.0, 0.5, 2.0])

        full_likelihood_old = jnp.array([0.0, -1.0, 1.0])
        full_likelihood_new = jnp.array([2.0, -0.5, 0.5])

        approx_likelihood_old = jnp.array([0.5, -0.5, 0.0])
        approx_likelihood_new = jnp.array([1.5, 0.0, -1.0])

        out = self.variant(
            lambda key: conservative_damh_step_parts_jax(
                key=key,
                new_particles=self.new,
                old_particles=self.old,
                cov=self.cov,
                approx_posterior_new=approx_posterior_new,
                approx_posterior_old=approx_posterior_old,
                full_likelihood_new=full_likelihood_new,
                full_likelihood_old=full_likelihood_old,
                approx_likelihood_new=approx_likelihood_new,
                approx_likelihood_old=approx_likelihood_old,
                c_const=jnp.array(0.01),
                d_const=jnp.array(2.0),
            )
        )(self.key)

        expected_surrogate = approx_posterior_new - approx_posterior_old
        expected_full = expected_surrogate + (
            full_likelihood_new
            - full_likelihood_old
            - approx_likelihood_new
            + approx_likelihood_old
        )

        np.testing.assert_allclose(out.log_ratio_surrogate_raw, expected_surrogate)
        np.testing.assert_allclose(out.log_ratio_full, expected_full)

    @chex.all_variants(with_pmap=False)
    def test_deterministic(self):
        log_ratio_surrogate = jnp.array([-0.5, 0.0, 0.5])
        log_ratio_full = jnp.array([-0.25, 0.25, 1.0])

        def run(key):
            return conservative_damh_step_jax(
                key=key,
                new_particles=self.new,
                old_particles=self.old,
                cov=self.cov,
                log_ratio_surrogate=log_ratio_surrogate,
                log_ratio_full=log_ratio_full,
            )

        out1 = self.variant(run)(self.key)
        out2 = self.variant(run)(self.key)

        np.testing.assert_array_equal(out1.pre_accept, out2.pre_accept)
        np.testing.assert_array_equal(out1.stage2_accept, out2.stage2_accept)
        np.testing.assert_array_equal(out1.accept, out2.accept)
        np.testing.assert_allclose(out1.prob_accept, out2.prob_accept)
        np.testing.assert_allclose(out1.proposal_dist, out2.proposal_dist)


if __name__ == "__main__":
    absltest.main()
