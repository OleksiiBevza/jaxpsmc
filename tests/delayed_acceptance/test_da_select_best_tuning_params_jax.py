import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.delayed_acceptance.da_select_best_tuning_params_jax import (
    da_mh_costs_jax,
    mh_costs_jax,
    min_da_mh_cost_jax,
    min_mh_cost_jax,
)


class TuningParamsTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    def test_da_costs(self):
        min_steps = jnp.array([1.0, 2.0, 4.0])
        surrogate_acceptance = jnp.array([0.5, 0.25, 1.0])

        out = self.variant(
            lambda s, a: da_mh_costs_jax(
                min_steps=s,
                surrogate_acceptance=a,
                surrogate_cost=jnp.array(2.0),
                full_cost=jnp.array(10.0),
            )
        )(min_steps, surrogate_acceptance)

        expected = jnp.array(
            [
                1.0 * (2.0 + 0.5 * 10.0),
                2.0 * (2.0 + 0.25 * 10.0),
                4.0 * (2.0 + 1.0 * 10.0),
            ]
        )
        np.testing.assert_allclose(out, expected)

    @chex.all_variants(with_pmap=False)
    def test_da_mask(self):
        min_steps = jnp.array([1.0, 2.0, 4.0])
        surrogate_acceptance = jnp.array([0.5, 0.25, 1.0])
        valid_mask = jnp.array([True, False, True])

        out = self.variant(
            lambda s, a, m: da_mh_costs_jax(
                min_steps=s,
                surrogate_acceptance=a,
                surrogate_cost=jnp.array(2.0),
                full_cost=jnp.array(10.0),
                valid_mask=m,
            )
        )(min_steps, surrogate_acceptance, valid_mask)

        assert jnp.isfinite(out[0])
        assert jnp.isinf(out[1])
        assert jnp.isfinite(out[2])

    @chex.all_variants(with_pmap=False)
    def test_da_invalid(self):
        min_steps = jnp.array([1.0, 0.0, -1.0, jnp.inf, 2.0, 3.0])
        surrogate_acceptance = jnp.array([0.5, 0.5, 0.5, 0.5, -0.1, 1.1])

        out = self.variant(
            lambda s, a: da_mh_costs_jax(
                min_steps=s,
                surrogate_acceptance=a,
                surrogate_cost=jnp.array(2.0),
                full_cost=jnp.array(10.0),
            )
        )(min_steps, surrogate_acceptance)

        assert jnp.isfinite(out[0])
        assert bool(jnp.isinf(out[1:]).all())

    @chex.all_variants(with_pmap=False)
    def test_da_bad_costs(self):
        min_steps = jnp.array([1.0, 2.0])
        surrogate_acceptance = jnp.array([0.5, 0.25])

        out_bad_surrogate = self.variant(
            lambda s, a: da_mh_costs_jax(
                min_steps=s,
                surrogate_acceptance=a,
                surrogate_cost=jnp.array(-1.0),
                full_cost=jnp.array(10.0),
            )
        )(min_steps, surrogate_acceptance)

        out_bad_full = self.variant(
            lambda s, a: da_mh_costs_jax(
                min_steps=s,
                surrogate_acceptance=a,
                surrogate_cost=jnp.array(2.0),
                full_cost=jnp.array(-10.0),
            )
        )(min_steps, surrogate_acceptance)

        assert bool(jnp.isinf(out_bad_surrogate).all())
        assert bool(jnp.isinf(out_bad_full).all())

    @chex.all_variants(with_pmap=False)
    def test_min_da(self):
        min_steps = jnp.array([10.0, 2.0, 4.0])
        surrogate_acceptance = jnp.array([0.1, 0.25, 1.0])

        out = self.variant(
            lambda s, a: min_da_mh_cost_jax(
                min_steps=s,
                surrogate_acceptance=a,
                surrogate_cost=jnp.array(2.0),
                full_cost=jnp.array(10.0),
            )
        )(min_steps, surrogate_acceptance)

        expected_costs = jnp.array(
            [
                10.0 * (2.0 + 0.1 * 10.0),
                2.0 * (2.0 + 0.25 * 10.0),
                4.0 * (2.0 + 1.0 * 10.0),
            ]
        )

        assert int(out.index) == 1
        assert bool(out.valid)
        np.testing.assert_allclose(out.costs, expected_costs)
        np.testing.assert_allclose(out.min_cost, expected_costs[1])

    @chex.all_variants(with_pmap=False)
    def test_min_da_none(self):
        min_steps = jnp.array([0.0, -1.0, jnp.inf])
        surrogate_acceptance = jnp.array([0.5, 0.5, 0.5])

        out = self.variant(
            lambda s, a: min_da_mh_cost_jax(
                min_steps=s,
                surrogate_acceptance=a,
                surrogate_cost=jnp.array(2.0),
                full_cost=jnp.array(10.0),
            )
        )(min_steps, surrogate_acceptance)

        assert int(out.index) == 0
        assert not bool(out.valid)
        assert bool(jnp.isinf(out.costs).all())
        assert bool(jnp.isinf(out.min_cost))

    @chex.all_variants(with_pmap=False)
    def test_mh_costs(self):
        min_steps = jnp.array([3.0, 1.0, 2.0])

        out = self.variant(mh_costs_jax)(min_steps)

        np.testing.assert_allclose(out, min_steps)

    @chex.all_variants(with_pmap=False)
    def test_mh_invalid(self):
        min_steps = jnp.array([3.0, 0.0, -1.0, jnp.inf])
        valid_mask = jnp.array([True, True, True, True])

        out = self.variant(
            lambda s, m: mh_costs_jax(
                min_steps=s,
                valid_mask=m,
            )
        )(min_steps, valid_mask)

        assert jnp.isfinite(out[0])
        assert bool(jnp.isinf(out[1:]).all())

    @chex.all_variants(with_pmap=False)
    def test_mh_mask(self):
        min_steps = jnp.array([3.0, 1.0, 2.0])
        valid_mask = jnp.array([True, False, True])

        out = self.variant(
            lambda s, m: mh_costs_jax(
                min_steps=s,
                valid_mask=m,
            )
        )(min_steps, valid_mask)

        np.testing.assert_allclose(out[0], 3.0)
        assert bool(jnp.isinf(out[1]))
        np.testing.assert_allclose(out[2], 2.0)

    @chex.all_variants(with_pmap=False)
    def test_min_mh(self):
        min_steps = jnp.array([3.0, 1.0, 2.0])
        valid_mask = jnp.array([True, False, True])

        out = self.variant(
            lambda s, m: min_mh_cost_jax(
                min_steps=s,
                valid_mask=m,
            )
        )(min_steps, valid_mask)

        assert int(out.index) == 2
        assert bool(out.valid)
        np.testing.assert_allclose(out.costs, jnp.array([3.0, jnp.inf, 2.0]))
        np.testing.assert_allclose(out.min_cost, 2.0)

    @chex.all_variants(with_pmap=False)
    def test_min_mh_none(self):
        min_steps = jnp.array([0.0, -1.0, jnp.inf])

        out = self.variant(min_mh_cost_jax)(min_steps)

        assert int(out.index) == 0
        assert not bool(out.valid)
        assert bool(jnp.isinf(out.costs).all())
        assert bool(jnp.isinf(out.min_cost))


if __name__ == "__main__":
    absltest.main()
