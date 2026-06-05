import chex
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.sampler.constants_jax import _ECONVERGED, _EVALUEERR
from jaxpsmc.sampler.resample_jax import resample_particles_jax

from _sampler_test_utils import SamplerHelperBase


class ResampleTest(SamplerHelperBase):
    @chex.all_variants(with_pmap=False)
    def test_resample_mult(self):
        cur = self._resample_input()

        out, status, key_out = self.variant(
            lambda c, k: resample_particles_jax(
                c,
                key=k,
                n_active=4,
                method_code=jnp.asarray(0, dtype=jnp.int32),
                reset_weights=True,
            )
        )(cur, self.key)

        assert int(status) == int(_ECONVERGED)
        np.testing.assert_allclose(out["u"], jnp.ones((4, 1), dtype=cur["u"].dtype))
        np.testing.assert_allclose(out["x"], 11.0 * jnp.ones((4, 1), dtype=cur["x"].dtype))
        np.testing.assert_allclose(out["weights"], 0.25 * jnp.ones((4,), dtype=cur["u"].dtype))
        assert not np.array_equal(jax.random.key_data(key_out), jax.random.key_data(self.key))

    @chex.all_variants(with_pmap=False)
    def test_resample_syst(self):
        cur = self._resample_input()

        out, status, _key_out = self.variant(
            lambda c, k: resample_particles_jax(
                c,
                key=k,
                n_active=4,
                method_code=jnp.asarray(1, dtype=jnp.int32),
                reset_weights=True,
            )
        )(cur, self.key)

        assert int(status) == int(_ECONVERGED)
        np.testing.assert_allclose(out["u"], jnp.ones((4, 1), dtype=cur["u"].dtype))
        np.testing.assert_allclose(out["blobs"], jnp.ones((4, 1), dtype=cur["u"].dtype))
        np.testing.assert_allclose(out["weights"], 0.25 * jnp.ones((4,), dtype=cur["u"].dtype))

    @chex.all_variants(with_pmap=False)
    def test_resample_keep(self):
        cur = self._resample_input()

        out, status, _key_out = self.variant(
            lambda c, k: resample_particles_jax(
                c,
                key=k,
                n_active=3,
                method_code=jnp.asarray(0, dtype=jnp.int32),
                reset_weights=False,
            )
        )(cur, self.key)

        assert int(status) == int(_ECONVERGED)
        np.testing.assert_allclose(out["weights"], jnp.ones((3,), dtype=cur["u"].dtype))

    @chex.all_variants(with_pmap=False)
    def test_resample_bad(self):
        cur = self._resample_input()
        cur = dict(cur)
        cur["weights"] = jnp.zeros_like(cur["weights"])

        out, status, _key_out = self.variant(
            lambda c, k: resample_particles_jax(
                c,
                key=k,
                n_active=5,
                method_code=jnp.asarray(0, dtype=jnp.int32),
                reset_weights=True,
            )
        )(cur, self.key)

        assert int(status) == int(_EVALUEERR)
        expected = jnp.asarray([[0.0], [1.0], [2.0], [0.0], [1.0]], dtype=cur["u"].dtype)
        np.testing.assert_allclose(out["u"], expected)
        np.testing.assert_allclose(out["weights"], 0.2 * jnp.ones((5,), dtype=cur["u"].dtype))


if __name__ == "__main__":
    absltest.main()
