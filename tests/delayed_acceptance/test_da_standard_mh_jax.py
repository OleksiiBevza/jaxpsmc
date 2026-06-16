from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.delayed_acceptance.da_standard_mh_jax import (
    _proposal_distance_jax,
    standard_mh_step_logtargets_jax,
    standard_mh_step_jax,
)


class _TargetEval(NamedTuple):
    value: jax.Array
    full_calls: jax.Array
    approx_calls: jax.Array
    prior_calls: jax.Array


def _log_target_fn(particles, beta, type_code):
    n = jnp.asarray(particles.shape[0], dtype=jnp.int32)
    dtype = jnp.result_type(particles, beta, jnp.asarray(1.0))

    value = jnp.asarray(beta, dtype=dtype) * jnp.sum(particles, axis=1) + jnp.asarray(
        type_code, dtype=dtype
    ) * jnp.asarray(0.01, dtype=dtype)

    return _TargetEval(
        value=value,
        full_calls=n,
        approx_calls=n + jnp.asarray(1, dtype=jnp.int32),
        prior_calls=2 * n,
    )


class StandardMHTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(11)
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
    def test_dist(self):
        out = self.variant(
            lambda new, old: _proposal_distance_jax(
                new_particles=new,
                old_particles=old,
                cov=self.cov,
            )
        )(self.new, self.old)

        expected = jnp.array([0.5, 2.0, 1.0])
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_logtargets(self):
        old_particles = jnp.zeros((6, 2))
        new_particles = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        )
        cov = jnp.eye(2)

        new_logtarget = jnp.array([1.0, -2.0, 0.0, jnp.nan, jnp.inf, -jnp.inf])
        old_logtarget = jnp.zeros((6,))

        out = self.variant(
            lambda key: standard_mh_step_logtargets_jax(
                key=key,
                new_particles=new_particles,
                old_particles=old_particles,
                cov=cov,
                new_logtarget=new_logtarget,
                old_logtarget=old_logtarget,
            )
        )(self.key)

        expected_ratio = new_logtarget - old_logtarget
        expected_log_prob = jnp.minimum(expected_ratio, 0.0)
        expected_prob = jnp.exp(expected_log_prob)
        expected_prob = jnp.where(jnp.isfinite(expected_prob), expected_prob, 0.0)

        _, subkey = jax.random.split(self.key)
        log_u = jnp.log(
            jax.random.uniform(
                subkey,
                shape=expected_prob.shape,
                dtype=expected_prob.dtype,
            )
        )
        expected_accept = log_u < expected_log_prob

        expected_dist = jnp.array([1.0, 2.0, 2.0, 1.0, jnp.sqrt(2.0), 0.0])

        np.testing.assert_allclose(out.log_accept_ratio, expected_ratio, equal_nan=True)
        np.testing.assert_allclose(out.prob_accept, expected_prob)
        np.testing.assert_array_equal(out.accept, expected_accept)
        np.testing.assert_allclose(out.proposal_dist, expected_dist, rtol=1e-6)
        np.testing.assert_allclose(
            out.actual_dist,
            out.proposal_dist * out.accept.astype(out.proposal_dist.dtype),
        )
        np.testing.assert_allclose(
            out.expected_dist,
            out.proposal_dist * out.prob_accept,
        )

        assert int(out.full_calls) == 0
        assert int(out.approx_calls) == 0
        assert int(out.prior_calls) == 0

    @chex.all_variants(with_pmap=False)
    def test_wrapper(self):
        old_particles = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        new_particles = jnp.array(
            [
                [1.0, 1.0],
                [2.0, 0.0],
                [0.0, 0.0],
            ]
        )
        cov = jnp.eye(2)
        beta = jnp.array(0.5)
        type_code = jnp.array(3, dtype=jnp.int32)

        out = self.variant(
            lambda key: standard_mh_step_jax(
                key=key,
                new_particles=new_particles,
                old_particles=old_particles,
                cov=cov,
                beta=beta,
                log_target_fn=_log_target_fn,
                type_code=type_code,
            )
        )(self.key)

        expected_new = beta * jnp.sum(new_particles, axis=1) + 0.03
        expected_old = beta * jnp.sum(old_particles, axis=1) + 0.03
        expected_ratio = expected_new - expected_old
        expected_prob = jnp.exp(jnp.minimum(expected_ratio, 0.0))

        np.testing.assert_allclose(out.new_logtarget, expected_new)
        np.testing.assert_allclose(out.old_logtarget, expected_old)
        np.testing.assert_allclose(out.log_accept_ratio, expected_ratio)
        np.testing.assert_allclose(out.prob_accept, expected_prob)

        assert int(out.full_calls) == 6
        assert int(out.approx_calls) == 8
        assert int(out.prior_calls) == 12

    @chex.all_variants(with_pmap=False)
    def test_deterministic(self):
        old_particles = jnp.zeros((3, 2))
        new_particles = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cov = jnp.eye(2)
        new_logtarget = jnp.array([0.5, -0.5, 1.0])
        old_logtarget = jnp.zeros((3,))

        def run(key):
            return standard_mh_step_logtargets_jax(
                key=key,
                new_particles=new_particles,
                old_particles=old_particles,
                cov=cov,
                new_logtarget=new_logtarget,
                old_logtarget=old_logtarget,
            )

        out1 = self.variant(run)(self.key)
        out2 = self.variant(run)(self.key)

        np.testing.assert_array_equal(out1.accept, out2.accept)
        np.testing.assert_allclose(out1.prob_accept, out2.prob_accept)
        np.testing.assert_allclose(out1.proposal_dist, out2.proposal_dist)
        np.testing.assert_allclose(out1.actual_dist, out2.actual_dist)
        np.testing.assert_allclose(out1.expected_dist, out2.expected_dist)
        np.testing.assert_allclose(out1.log_accept_ratio, out2.log_accept_ratio)

    @chex.all_variants(with_pmap=False)
    def test_shapes(self):
        old_particles = jnp.zeros((4, 2))
        new_particles = jnp.ones((4, 2))
        cov = jnp.eye(2)
        new_logtarget = jnp.array([1.0, -1.0, 0.0, 2.0])
        old_logtarget = jnp.zeros((4,))

        out = self.variant(
            lambda key: standard_mh_step_logtargets_jax(
                key=key,
                new_particles=new_particles,
                old_particles=old_particles,
                cov=cov,
                new_logtarget=new_logtarget,
                old_logtarget=old_logtarget,
            )
        )(self.key)

        assert out.accept.shape == (4,)
        assert out.prob_accept.shape == (4,)
        assert out.proposal_dist.shape == (4,)
        assert out.actual_dist.shape == (4,)
        assert out.expected_dist.shape == (4,)
        assert out.log_accept_ratio.shape == (4,)
        assert out.new_logtarget.shape == (4,)
        assert out.old_logtarget.shape == (4,)

        assert out.accept.dtype == jnp.bool_
        assert jnp.isfinite(out.prob_accept).all()
        assert jnp.isfinite(out.proposal_dist).all()


if __name__ == "__main__":
    absltest.main()
