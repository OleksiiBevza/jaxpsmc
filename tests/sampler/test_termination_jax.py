import chex
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _sampler_test_utils import SamplerHelperBase
from absl.testing import absltest

from jaxpsmc.sampler.constants_jax import METRIC_ESS, METRIC_USS
from jaxpsmc.sampler.termination_jax import not_termination_jax


class TerminationTest(SamplerHelperBase):
    @chex.all_variants(with_pmap=False)
    def test_term_beta(self):
        state = self._state()

        out = self.variant(
            lambda s: not_termination_jax(
                s,
                beta_current=jnp.asarray(0.5, dtype=s.logl.dtype),
                n_total=jnp.asarray(1, dtype=jnp.int32),
                metric_code=METRIC_ESS,
                n_active=jnp.asarray(2, dtype=jnp.int32),
            )
        )(state)

        assert bool(out)

    @chex.all_variants(with_pmap=False)
    def test_term_done(self):
        state = self._state()

        out = self.variant(
            lambda s: not_termination_jax(
                s,
                beta_current=jnp.asarray(1.0, dtype=s.logl.dtype),
                n_total=jnp.asarray(1, dtype=jnp.int32),
                metric_code=METRIC_ESS,
                n_active=jnp.asarray(2, dtype=jnp.int32),
            )
        )(state)

        assert not bool(out)

    @chex.all_variants(with_pmap=False)
    def test_term_metric(self):
        state = self._state()

        out = self.variant(
            lambda s: not_termination_jax(
                s,
                beta_current=jnp.asarray(1.0, dtype=s.logl.dtype),
                n_total=jnp.asarray(100, dtype=jnp.int32),
                metric_code=METRIC_USS,
                n_active=jnp.asarray(2, dtype=jnp.int32),
            )
        )(state)

        assert bool(out)


if __name__ == "__main__":
    absltest.main()
