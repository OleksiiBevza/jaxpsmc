# ruff: noqa: E402
import chex
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.sampler.constants_jax import _ECONVERGED, _EVALUEERR
from jaxpsmc.sampler.posterior_jax import (
    _systematic_resample_impl,
    posterior_jax,
    trim_weights_scan_jax,
)

from _sampler_test_utils import SamplerHelperBase


class PosteriorTest(SamplerHelperBase):
    @chex.all_variants(with_pmap=False)
    def test_systematic(self):
        weights = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32)

        idx, status, key_out = self.variant(
            lambda k, w: _systematic_resample_impl(k, w, size=5)
        )(self.key, weights)

        assert int(status) == int(_ECONVERGED)
        np.testing.assert_array_equal(idx, jnp.ones((5,), dtype=jnp.int32))
        assert not np.array_equal(
            jax.random.key_data(key_out), jax.random.key_data(self.key)
        )

    @chex.all_variants(with_pmap=False)
    def test_systematic_bad(self):
        weights = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32)

        idx, status, _key_out = self.variant(
            lambda k, w: _systematic_resample_impl(k, w, size=4)
        )(self.key, weights)

        assert int(status) == int(_EVALUEERR)
        np.testing.assert_array_equal(idx, -jnp.ones((4,), dtype=jnp.int64))

    @chex.all_variants(with_pmap=False)
    def test_trim(self):
        weights = jnp.asarray([0.7, 0.2, 0.1, 0.0], dtype=jnp.float32)

        mask, w_trim, threshold, ratio, i_final = self.variant(
            lambda w: trim_weights_scan_jax(
                w, ess=jnp.asarray(0.5, dtype=w.dtype), bins=16
            )
        )(weights)

        assert mask.shape == weights.shape
        assert w_trim.shape == weights.shape
        assert bool(jnp.isfinite(threshold))
        assert bool(jnp.isfinite(ratio))
        assert 0 <= int(i_final) < 16
        assert float(ratio) >= 0.5
        np.testing.assert_allclose(jnp.sum(w_trim), 1.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(w_trim[~mask], 0.0, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_trim_bad(self):
        weights = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32)

        mask, w_trim, threshold, ratio, _i_final = self.variant(
            lambda w: trim_weights_scan_jax(
                w, ess=jnp.asarray(0.5, dtype=w.dtype), bins=8
            )
        )(weights)

        np.testing.assert_array_equal(mask, jnp.zeros((3,), dtype=bool))
        assert bool(jnp.all(jnp.isnan(w_trim)))
        assert bool(jnp.isnan(threshold))
        assert bool(jnp.isnan(ratio))

    @chex.all_variants(with_pmap=False)
    def test_posterior(self):
        state = self._state(T=3, N=2, D=2, B=1)

        out = self.variant(
            lambda s, k: posterior_jax(
                s,
                k,
                do_resample=jnp.asarray(False),
                trim_importance_weights=jnp.asarray(False),
                bins_trim=16,
                beta_final=jnp.asarray(1.0, dtype=s.logl.dtype),
            )
        )(state, self.key)

        assert out.samples.shape == (6, 2)
        assert out.logl.shape == (6,)
        assert out.logp.shape == (6,)
        assert out.blobs.shape == (6, 1)
        assert out.weights.shape == (6,)
        assert out.logw.shape == (6,)
        assert out.idx_resampled.shape == (6,)
        np.testing.assert_array_equal(
            out.mask_valid, jnp.array([True, True, True, True, False, False])
        )
        np.testing.assert_array_equal(out.mask_trim, out.mask_valid)
        np.testing.assert_array_equal(out.idx_resampled, jnp.arange(6, dtype=jnp.int32))
        np.testing.assert_allclose(jnp.sum(out.weights), 1.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out.samples[-2:], 0.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out.logl[-2:], 0.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(
            jax.random.key_data(out.key_out), jax.random.key_data(self.key)
        )
        assert int(out.resample_status) == 0
        assert bool(jnp.isfinite(out.logz_new))

    @chex.all_variants(with_pmap=False)
    def test_posterior_trim(self):
        state = self._state(T=3, N=2, D=2, B=1)

        out = self.variant(
            lambda s, k: posterior_jax(
                s,
                k,
                do_resample=jnp.asarray(False),
                trim_importance_weights=jnp.asarray(True),
                ess_trim=jnp.asarray(0.5, dtype=s.logl.dtype),
                bins_trim=16,
                beta_final=jnp.asarray(1.0, dtype=s.logl.dtype),
            )
        )(state, self.key)

        assert out.mask_trim.shape == (6,)
        assert bool(jnp.all(out.mask_trim <= out.mask_valid))
        assert bool(jnp.isfinite(out.threshold))
        assert bool(jnp.isfinite(out.ess_ratio))
        np.testing.assert_allclose(jnp.sum(out.weights), 1.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            out.weights[~out.mask_trim], 0.0, rtol=1e-6, atol=1e-6
        )

    @chex.all_variants(with_pmap=False)
    def test_posterior_resample(self):
        state = self._state(T=3, N=2, D=2, B=1)

        out = self.variant(
            lambda s, k: posterior_jax(
                s,
                k,
                do_resample=jnp.asarray(True),
                resample_method=jnp.asarray(1, dtype=jnp.int32),
                trim_importance_weights=jnp.asarray(False),
                bins_trim=16,
                beta_final=jnp.asarray(1.0, dtype=s.logl.dtype),
            )
        )(state, self.key)

        assert int(out.resample_status) == 0
        assert out.samples_resampled.shape == out.samples.shape
        assert out.logl_resampled.shape == out.logl.shape
        assert out.logp_resampled.shape == out.logp.shape
        assert out.blobs_resampled.shape == out.blobs.shape
        assert bool(jnp.all(out.idx_resampled >= 0))
        assert bool(jnp.all(out.idx_resampled < 6))
        assert not np.array_equal(
            jax.random.key_data(out.key_out), jax.random.key_data(self.key)
        )

    @chex.all_variants(with_pmap=False)
    def test_posterior_blob0(self):
        state = self._state(T=3, N=2, D=2, B=0)

        out = self.variant(
            lambda s, k: posterior_jax(
                s,
                k,
                do_resample=jnp.asarray(False),
                trim_importance_weights=jnp.asarray(False),
                bins_trim=16,
            )
        )(state, self.key)

        assert out.blobs.shape == (6, 0)
        assert out.blobs_resampled.shape == (6, 0)
        np.testing.assert_allclose(out.blobs, jnp.zeros((6, 0), dtype=state.logl.dtype))


if __name__ == "__main__":
    absltest.main()
