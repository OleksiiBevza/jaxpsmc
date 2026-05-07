import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.delayed_acceptance.da_required_mh_steps_jax import (
    expected_squared_jump_jax,
    time_steps_jax,
    time_steps_to_min_quantile_dist_median_batched_jax,
)


class RequiredMHTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    def test_jump(self):
        proposal_dist = jnp.array([2.0, 3.0, 4.0])
        prob_accept = jnp.array([0.5, 0.0, 1.0])

        out = self.variant(expected_squared_jump_jax)(
            proposal_dist,
            prob_accept,
        )

        expected = jnp.array([2.0, 0.0, 16.0])
        np.testing.assert_allclose(out, expected)

    @chex.all_variants(with_pmap=False)
    def test_steps(self):
        proposal_dist = jnp.array([2.0, 4.0, 6.0])
        prob_accept = jnp.array([0.5, 0.25, 1.0])

        out = self.variant(
            lambda d, p: time_steps_jax(
                proposal_dist=d,
                prob_accept=p,
                threshold=jnp.array(10.0),
                rho=jnp.array(0.7),
                max_t=jnp.array(10, dtype=jnp.int32),
            )
        )(proposal_dist, prob_accept)

        assert int(out.iter) == 3
        assert bool(out.sufficient_iter)
        np.testing.assert_allclose(out.prob, 0.7)
        np.testing.assert_allclose(out.median_expected_dist, 4.0)
        np.testing.assert_allclose(out.steps_float, 2.5)

    @chex.all_variants(with_pmap=False)
    def test_min(self):
        proposal_dist = jnp.array([1.0, 2.0, 3.0])
        prob_accept = jnp.array([1.0, 1.0, 1.0])

        out = self.variant(
            lambda d, p: time_steps_jax(
                proposal_dist=d,
                prob_accept=p,
                threshold=jnp.array(0.0),
                rho=jnp.array(0.5),
                max_t=jnp.array(10, dtype=jnp.int32),
            )
        )(proposal_dist, prob_accept)

        assert int(out.iter) == 1
        assert bool(out.sufficient_iter)
        np.testing.assert_allclose(out.prob, 0.5)
        np.testing.assert_allclose(out.steps_float, 0.0)

    @chex.all_variants(with_pmap=False)
    def test_cap(self):
        proposal_dist = jnp.array([2.0, 4.0, 6.0])
        prob_accept = jnp.array([0.5, 0.25, 1.0])

        out = self.variant(
            lambda d, p: time_steps_jax(
                proposal_dist=d,
                prob_accept=p,
                threshold=jnp.array(100.0),
                rho=jnp.array(0.7),
                max_t=jnp.array(5, dtype=jnp.int32),
            )
        )(proposal_dist, prob_accept)

        assert int(out.iter) == 5
        assert not bool(out.sufficient_iter)
        assert bool(jnp.isnan(out.prob))
        np.testing.assert_allclose(out.median_expected_dist, 4.0)
        np.testing.assert_allclose(out.steps_float, 25.0)

    @chex.all_variants(with_pmap=False)
    def test_bad(self):
        proposal_dist = jnp.array([1.0, 2.0, 3.0])
        prob_accept = jnp.array([0.0, 0.0, 0.0])

        out = self.variant(
            lambda d, p: time_steps_jax(
                proposal_dist=d,
                prob_accept=p,
                threshold=jnp.array(1.0),
                rho=jnp.array(0.5),
                max_t=jnp.array(8, dtype=jnp.int32),
            )
        )(proposal_dist, prob_accept)

        assert int(out.iter) == 8
        assert not bool(out.sufficient_iter)
        assert bool(jnp.isnan(out.prob))
        np.testing.assert_allclose(out.median_expected_dist, 0.0)
        assert bool(jnp.isinf(out.steps_float))

    @chex.all_variants(with_pmap=False)
    def test_negative(self):
        proposal_dist = jnp.array([1.0, 2.0, 3.0])
        prob_accept = jnp.array([1.0, 1.0, 1.0])

        out = self.variant(
            lambda d, p: time_steps_jax(
                proposal_dist=d,
                prob_accept=p,
                threshold=jnp.array(-1.0),
                rho=jnp.array(0.5),
                max_t=jnp.array(8, dtype=jnp.int32),
            )
        )(proposal_dist, prob_accept)

        assert int(out.iter) == 8
        assert not bool(out.sufficient_iter)
        assert bool(jnp.isnan(out.prob))
        assert bool(jnp.isinf(out.steps_float))

    @chex.all_variants(with_pmap=False)
    def test_batch(self):
        proposal_dist = jnp.array(
            [
                [2.0, 4.0, 6.0],
                [1.0, 2.0, 3.0],
            ]
        )
        prob_accept = jnp.array(
            [
                [0.5, 0.25, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        out = self.variant(
            lambda d, p: time_steps_to_min_quantile_dist_median_batched_jax(
                d,
                p,
                jnp.array(10.0),
                jnp.array(0.6),
                jnp.array(10, dtype=jnp.int32),
            )
        )(proposal_dist, prob_accept)

        np.testing.assert_array_equal(out.iter, jnp.array([3, 3], dtype=jnp.int32))
        np.testing.assert_array_equal(
            out.sufficient_iter,
            jnp.array([True, True]),
        )
        np.testing.assert_allclose(out.prob, jnp.array([0.6, 0.6]))
        np.testing.assert_allclose(out.median_expected_dist, jnp.array([4.0, 4.0]))
        np.testing.assert_allclose(out.steps_float, jnp.array([2.5, 2.5]))


if __name__ == "__main__":
    absltest.main()