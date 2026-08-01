import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.tools_jax import (
    _ECONVERGED,
    _EVALUEERR,
    _systematic_resample_impl,
    compute_ess_jax,
    effective_sample_size_jax,
    increment_logz_jax,
    systematic_resample_jax,
    systematic_resample_jax_size,
    trim_weights_jax,
    unique_sample_size_jax,
)


class ToolsTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(17)

    def _manual_ess(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / np.sum(weights)
        return 1.0 / np.sum(weights * weights)

    def _manual_uss(self, weights, k):
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        return np.sum(1.0 - np.power(1.0 - weights, k), axis=-1)

    @chex.all_variants(with_pmap=False)
    def test_ess(self):
        weights = jnp.asarray([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

        ess = self.variant(lambda w: effective_sample_size_jax(w))(weights)

        np.testing.assert_allclose(ess, 4.0, rtol=1e-6, atol=1e-6)
        assert ess.shape == ()

    @chex.all_variants(with_pmap=False)
    def test_ess_scale(self):
        weights = jnp.asarray([2.0, 4.0, 6.0, 8.0], dtype=jnp.float32)
        weights_scaled = 10.0 * weights

        ess = self.variant(lambda w: effective_sample_size_jax(w))(weights)
        ess_scaled = self.variant(lambda w: effective_sample_size_jax(w))(
            weights_scaled
        )

        expected = self._manual_ess(np.asarray(weights))
        np.testing.assert_allclose(ess, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ess_scaled, expected, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_ess_degenerate(self):
        weights = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

        ess = self.variant(lambda w: effective_sample_size_jax(w))(weights)

        np.testing.assert_allclose(ess, 1.0, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_ess_bad(self):
        zeros = jnp.zeros((4,), dtype=jnp.float32)
        infs = jnp.asarray([1.0, jnp.inf, 2.0, 3.0], dtype=jnp.float32)

        ess_zero = self.variant(lambda w: effective_sample_size_jax(w))(zeros)
        ess_inf = self.variant(lambda w: effective_sample_size_jax(w))(infs)

        assert bool(jnp.isnan(ess_zero))
        assert bool(jnp.isnan(ess_inf))

    @chex.all_variants(with_pmap=False)
    def test_uss(self):
        weights = jnp.asarray([0.25, 0.25, 0.25, 0.25], dtype=jnp.float32)

        uss = self.variant(lambda w: unique_sample_size_jax(w, k=4))(weights)

        expected = self._manual_uss(np.asarray(weights), k=4)
        np.testing.assert_allclose(uss, expected, rtol=1e-6, atol=1e-6)
        assert uss.shape == ()

    @chex.all_variants(with_pmap=False)
    def test_uss_default(self):
        weights = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)

        uss_default = self.variant(lambda w: unique_sample_size_jax(w))(weights)
        uss_explicit = self.variant(lambda w: unique_sample_size_jax(w, k=4))(weights)

        np.testing.assert_allclose(uss_default, uss_explicit, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_uss_batch(self):
        weights = jnp.asarray(
            [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.2, 0.3, 0.5]],
            dtype=jnp.float32,
        )

        uss = self.variant(lambda w: unique_sample_size_jax(w, k=2))(weights)

        expected = self._manual_uss(np.asarray(weights), k=2)
        assert uss.shape == (3,)
        np.testing.assert_allclose(uss, expected, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_uss_k0(self):
        weights = jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float32)

        uss = self.variant(lambda w: unique_sample_size_jax(w, k=0))(weights)

        np.testing.assert_allclose(uss, 0.0, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_uss_bad(self):
        weights = jnp.zeros((2, 3), dtype=jnp.float32)

        uss = self.variant(lambda w: unique_sample_size_jax(w, k=3))(weights)

        assert uss.shape == (2,)
        assert bool(jnp.all(jnp.isnan(uss)))

    @chex.all_variants(with_pmap=False)
    def test_logess(self):
        logw = jnp.asarray([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

        ess_frac = self.variant(lambda lw: compute_ess_jax(lw))(logw)

        np.testing.assert_allclose(ess_frac, 1.0, rtol=1e-6, atol=1e-6)
        assert ess_frac.shape == ()

    @chex.all_variants(with_pmap=False)
    def test_logess_weighted(self):
        weights = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
        logw = jnp.log(weights)

        ess_frac = self.variant(lambda lw: compute_ess_jax(lw))(logw)

        expected = self._manual_ess(np.asarray(weights)) / weights.shape[0]
        np.testing.assert_allclose(ess_frac, expected, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_logess_batch(self):
        logw = jnp.asarray(
            [[0.0, 0.0, 0.0], [0.0, -jnp.inf, -jnp.inf]],
            dtype=jnp.float32,
        )

        ess_frac = self.variant(lambda lw: compute_ess_jax(lw))(logw)

        expected = jnp.asarray([1.0, 1.0 / 3.0], dtype=jnp.float32)
        assert ess_frac.shape == (2,)
        np.testing.assert_allclose(ess_frac, expected, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_logess_stable(self):
        logw = jnp.asarray([1000.0, 1000.0], dtype=jnp.float32)
        logw_one = jnp.asarray([1000.0, -1000.0], dtype=jnp.float32)

        ess_equal = self.variant(lambda lw: compute_ess_jax(lw))(logw)
        ess_one = self.variant(lambda lw: compute_ess_jax(lw))(logw_one)

        np.testing.assert_allclose(ess_equal, 1.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ess_one, 0.5, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_logess_bad(self):
        logw = jnp.asarray([-jnp.inf, -jnp.inf], dtype=jnp.float32)

        ess_frac = self.variant(lambda lw: compute_ess_jax(lw))(logw)

        assert bool(jnp.isnan(ess_frac))

    @chex.all_variants(with_pmap=False)
    def test_logz(self):
        logw = jnp.asarray([0.0, jnp.log(2.0), jnp.log(3.0)], dtype=jnp.float32)

        out = self.variant(lambda lw: increment_logz_jax(lw))(logw)

        expected = jax.nn.logsumexp(logw)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
        assert out.shape == ()

    @chex.all_variants(with_pmap=False)
    def test_logz_batch(self):
        logw = jnp.asarray(
            [[0.0, 0.0], [jnp.log(2.0), jnp.log(3.0)]],
            dtype=jnp.float32,
        )

        out = self.variant(lambda lw: increment_logz_jax(lw))(logw)

        expected = jax.nn.logsumexp(logw, axis=-1)
        assert out.shape == (2,)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_logz_stable(self):
        logw = jnp.asarray([1000.0, 1000.0], dtype=jnp.float32)

        out = self.variant(lambda lw: increment_logz_jax(lw))(logw)

        np.testing.assert_allclose(out, 1000.0 + np.log(2.0), rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_logz_bad(self):
        logw = jnp.asarray([-jnp.inf, -jnp.inf], dtype=jnp.float32)

        out = self.variant(lambda lw: increment_logz_jax(lw))(logw)

        assert bool(jnp.isnan(out))

    @chex.all_variants(with_pmap=False)
    def test_trim_uniform(self):
        samples = jnp.arange(4, dtype=jnp.int32)
        weights = jnp.ones((4,), dtype=jnp.float32)

        mask, w_trim, threshold, ratio, i_final = self.variant(
            lambda s, w: trim_weights_jax(s, w, ess=0.99, bins=10)
        )(samples, weights)

        np.testing.assert_array_equal(mask, jnp.ones((4,), dtype=bool))
        np.testing.assert_allclose(w_trim, 0.25, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(threshold, 0.25, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ratio, 1.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(i_final, jnp.asarray(9, dtype=jnp.int32))

    @chex.all_variants(with_pmap=False)
    def test_trim(self):
        samples = jnp.arange(5, dtype=jnp.int32)
        weights = jnp.asarray([0.001, 0.009, 0.09, 0.2, 0.7], dtype=jnp.float32)

        mask, w_trim, threshold, ratio, i_final = self.variant(
            lambda s, w: trim_weights_jax(s, w, ess=0.5, bins=20)
        )(samples, weights)

        assert mask.shape == weights.shape
        assert w_trim.shape == weights.shape
        assert threshold.shape == ()
        assert ratio.shape == ()
        assert i_final.shape == ()
        assert bool(jnp.all(w_trim >= 0.0))
        np.testing.assert_allclose(jnp.sum(w_trim), 1.0, rtol=1e-6, atol=1e-6)
        assert bool(ratio >= 0.5)
        np.testing.assert_array_equal(mask, w_trim > 0.0)

    @chex.all_variants(with_pmap=False)
    def test_trim_scale(self):
        samples = jnp.arange(5, dtype=jnp.int32)
        weights = jnp.asarray([0.001, 0.009, 0.09, 0.2, 0.7], dtype=jnp.float32)

        out1 = self.variant(lambda s, w: trim_weights_jax(s, w, ess=0.75, bins=20))(
            samples, weights
        )
        out2 = self.variant(lambda s, w: trim_weights_jax(s, w, ess=0.75, bins=20))(
            samples + 10, 10.0 * weights
        )

        np.testing.assert_array_equal(out1[0], out2[0])
        np.testing.assert_allclose(out1[1], out2[1], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out1[2], out2[2], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out1[3], out2[3], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(out1[4], out2[4])

    @chex.all_variants(with_pmap=False)
    def test_trim_samples(self):
        samples1 = jnp.arange(4, dtype=jnp.int32)
        samples2 = jnp.arange(4, dtype=jnp.int32) + 100
        weights = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)

        out1 = self.variant(lambda s, w: trim_weights_jax(s, w, ess=0.9, bins=10))(
            samples1, weights
        )
        out2 = self.variant(lambda s, w: trim_weights_jax(s, w, ess=0.9, bins=10))(
            samples2, weights
        )

        for a, b in zip(out1, out2, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_trim_bad(self):
        samples = jnp.arange(4, dtype=jnp.int32)
        weights = jnp.zeros((4,), dtype=jnp.float32)

        mask, w_trim, threshold, ratio, _ = self.variant(
            lambda s, w: trim_weights_jax(s, w, ess=0.99, bins=10)
        )(samples, weights)

        np.testing.assert_array_equal(mask, jnp.zeros((4,), dtype=bool))
        assert bool(jnp.all(jnp.isnan(w_trim)))
        assert bool(jnp.isnan(threshold))
        assert bool(jnp.isnan(ratio))

    @chex.all_variants(with_pmap=False)
    def test_resample_impl(self):
        weights = jnp.asarray([0.1, 0.2, 0.7], dtype=jnp.float32)

        idx, status, key_out = self.variant(
            lambda key, w: _systematic_resample_impl(key, w, size=5)
        )(self.key, weights)

        assert idx.shape == (5,)
        np.testing.assert_array_equal(status, _ECONVERGED)
        assert bool(jnp.all(idx >= 0))
        assert bool(jnp.all(idx < weights.shape[0]))
        assert not np.array_equal(
            np.asarray(jax.random.key_data(key_out)),
            np.asarray(jax.random.key_data(self.key)),
        )

    @chex.all_variants(with_pmap=False)
    def test_resample_onehot(self):
        weights = jnp.asarray([0.0, 0.0, 1.0, 0.0], dtype=jnp.float32)

        idx, status, _ = self.variant(
            lambda key, w: _systematic_resample_impl(key, w, size=6)
        )(self.key, weights)

        np.testing.assert_array_equal(status, _ECONVERGED)
        np.testing.assert_array_equal(idx, jnp.full((6,), 2, dtype=jnp.int32))

    @chex.all_variants(with_pmap=False)
    def test_resample_bad(self):
        weights = jnp.zeros((4,), dtype=jnp.float32)

        idx, status, key_out = self.variant(
            lambda key, w: _systematic_resample_impl(key, w, size=5)
        )(self.key, weights)

        np.testing.assert_array_equal(idx, jnp.full((5,), -1, dtype=jnp.int32))
        np.testing.assert_array_equal(status, _EVALUEERR)
        assert not np.array_equal(
            np.asarray(jax.random.key_data(key_out)),
            np.asarray(jax.random.key_data(self.key)),
        )

    @chex.all_variants(with_pmap=False)
    def test_resample_negative(self):
        weights = jnp.asarray([0.5, -0.1, 0.6], dtype=jnp.float32)

        idx, status, _ = self.variant(
            lambda key, w: _systematic_resample_impl(key, w, size=3)
        )(self.key, weights)

        np.testing.assert_array_equal(idx, jnp.full((3,), -1, dtype=jnp.int32))
        np.testing.assert_array_equal(status, _EVALUEERR)

    def test_resample_wrap(self):
        weights = jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float32)

        idx, status, key_out = systematic_resample_jax(weights, key=self.key)

        assert idx.shape == weights.shape
        np.testing.assert_array_equal(status, _ECONVERGED)
        assert bool(jnp.all(idx >= 0))
        assert bool(jnp.all(idx < weights.shape[0]))
        assert not np.array_equal(
            np.asarray(jax.random.key_data(key_out)),
            np.asarray(jax.random.key_data(self.key)),
        )

    def test_resample_size(self):
        weights = jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float32)

        idx, status, _ = systematic_resample_jax_size(weights, key=self.key, size=8)

        assert idx.shape == (8,)
        np.testing.assert_array_equal(status, _ECONVERGED)
        assert bool(jnp.all(idx >= 0))
        assert bool(jnp.all(idx < weights.shape[0]))

    def test_resample_repro(self):
        weights = jnp.asarray([0.1, 0.2, 0.7], dtype=jnp.float32)

        out1 = systematic_resample_jax_size(weights, key=self.key, size=7)
        out2 = systematic_resample_jax_size(weights, key=self.key, size=7)

        np.testing.assert_array_equal(out1[0], out2[0])
        np.testing.assert_array_equal(out1[1], out2[1])
        np.testing.assert_array_equal(
            jax.random.key_data(out1[2]),
            jax.random.key_data(out2[2]),
        )

    @chex.all_variants(with_pmap=False)
    def test_dtype(self):
        weights = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
        samples = jnp.arange(3, dtype=jnp.int32)

        ess = self.variant(lambda w: effective_sample_size_jax(w))(weights)
        uss = self.variant(lambda w: unique_sample_size_jax(w, k=3))(weights)
        logess = self.variant(lambda w: compute_ess_jax(jnp.log(w)))(weights)
        logz = self.variant(lambda w: increment_logz_jax(jnp.log(w)))(weights)
        _, w_trim, threshold, ratio, _ = self.variant(
            lambda s, w: trim_weights_jax(s, w, ess=0.99, bins=10)
        )(samples, weights)
        idx, status, _ = self.variant(
            lambda key, w: _systematic_resample_impl(key, w, size=3)
        )(self.key, weights)

        assert ess.dtype == jnp.float64
        assert uss.dtype == jnp.float64
        assert logess.dtype == jnp.float64
        assert logz.dtype == jnp.float64
        assert w_trim.dtype == jnp.float64
        assert threshold.dtype == jnp.float64
        assert ratio.dtype == jnp.float64
        assert idx.dtype == jnp.int32
        assert status.dtype == jnp.int64


if __name__ == "__main__":
    absltest.main()
