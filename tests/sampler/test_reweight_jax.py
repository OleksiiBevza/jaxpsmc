# ruff: noqa: E402
import chex
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.sampler.constants_jax import METRIC_ESS, METRIC_USS
from jaxpsmc.sampler.reweight_jax import (
    _bisect_beta_scan,
    _dynamic_neff,
    _metric_value,
    _weights_metric_logz,
    reweight_step_jax,
)
from jaxpsmc.tools_jax import effective_sample_size_jax, unique_sample_size_jax

from _sampler_test_utils import SamplerHelperBase


class ReweightTest(SamplerHelperBase):
    def test_metric(self):
        weights = jnp.asarray([0.5, 0.5], dtype=jnp.float32)

        ess = _metric_value(weights, METRIC_ESS, jnp.asarray(2, dtype=jnp.int32))
        uss = _metric_value(weights, METRIC_USS, jnp.asarray(2, dtype=jnp.int32))

        np.testing.assert_allclose(ess, 2.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(uss, 1.5, rtol=1e-6, atol=1e-6)

    def test_dynamic(self):
        weights = jnp.asarray([0.7, 0.2, 0.1, 0.0], dtype=jnp.float32)
        n_eff = jnp.asarray(3, dtype=jnp.int32)
        n_active = jnp.asarray(4, dtype=jnp.int32)
        ratio = jnp.asarray(0.5, dtype=jnp.float32)

        out = _dynamic_neff(n_eff, weights, n_active, ratio)

        nuniq = unique_sample_size_jax(weights, k=n_active)
        n_eff_f = n_eff.astype(weights.dtype)
        n_act_f = n_active.astype(weights.dtype)
        low = n_act_f * (0.95 * ratio)
        high = n_act_f * jnp.minimum(1.05 * ratio, 1.0)
        down = (n_act_f / (nuniq + 1e-12)) * n_eff_f
        up = ((nuniq + 1e-12) / n_act_f) * n_eff_f
        expected = jnp.where(nuniq < low, down, n_eff_f)
        expected = jnp.where(nuniq > high, up, expected)
        expected = jnp.floor(expected).astype(jnp.int32)

        np.testing.assert_array_equal(out, expected)

    @chex.all_variants(with_pmap=False)
    def test_weights(self):
        state = self._state()

        def run(s):
            return _weights_metric_logz(
                s,
                jnp.asarray(1.0, dtype=s.logl.dtype),
                METRIC_ESS,
                jnp.asarray(2, dtype=jnp.int32),
            )

        weights, metric, logz, logw = self.variant(run)(state)

        assert weights.shape == (6,)
        assert logw.shape == (6,)
        np.testing.assert_allclose(jnp.sum(weights), 1.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            metric, effective_sample_size_jax(weights), rtol=1e-6, atol=1e-6
        )
        assert bool(jnp.isfinite(logz))
        np.testing.assert_array_equal(jnp.isneginf(logw[-2:]), jnp.array([True, True]))

    @chex.all_variants(with_pmap=False)
    def test_bisect(self):
        state = self._state()

        def run(s):
            return _bisect_beta_scan(
                s,
                jnp.asarray(0.0, dtype=s.logl.dtype),
                jnp.asarray(1.0, dtype=s.logl.dtype),
                jnp.asarray(2.0, dtype=s.logl.dtype),
                METRIC_ESS,
                jnp.asarray(2, dtype=jnp.int32),
                steps=16,
                tol=jnp.asarray(1e-2, dtype=s.logl.dtype),
            )

        beta = self.variant(run)(state)

        assert 0.0 <= float(beta) <= 1.0
        assert bool(jnp.isfinite(beta))

    @chex.all_variants(with_pmap=False)
    def test_reweight_truncated_persistent(self):
        state = self._state()

        def run(s):
            return reweight_step_jax(
                s,
                jnp.asarray(2, dtype=jnp.int32),
                METRIC_ESS,
                jnp.asarray(False),
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=s.logl.dtype),
                bins=16,
                bisect_steps=8,
                keep_max=4,
                trim_ess=0.95,
            )

        cur, n_eff, stats = self.variant(run)(state)

        self._assert_current_keys(cur)
        assert cur["u"].shape == (4, 2)
        assert cur["x"].shape == (4, 2)
        assert cur["blobs"].shape == (4, 1)
        assert cur["weights"].shape == (4,)
        assert cur["idx"].shape == (4,)
        assert cur["keep_mask"].shape == (4,)
        assert cur["trim_mask_full"].shape == (6,)
        assert int(n_eff) == 2
        assert set(stats.keys()) == {"beta", "logz", "ess", "n_effective"}
        assert 0.0 <= float(cur["beta"]) <= 1.0
        assert bool(jnp.isfinite(cur["logz"]))
        np.testing.assert_allclose(jnp.sum(cur["weights"]), 1.0, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_reweight_truncated_persistent_dynamic(self):
        state = self._state()

        def run(s):
            return reweight_step_jax(
                s,
                jnp.asarray(2, dtype=jnp.int32),
                METRIC_USS,
                jnp.asarray(True),
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=s.logl.dtype),
                bins=16,
                bisect_steps=8,
                keep_max=4,
                trim_ess=0.95,
            )

        cur, n_eff, stats = self.variant(run)(state)

        assert n_eff.dtype == jnp.int32
        np.testing.assert_array_equal(stats["n_effective"], n_eff)
        assert cur["u"].shape == (4, 2)
        assert bool(jnp.all(jnp.isfinite(cur["weights"])))

    @chex.all_variants(with_pmap=False)
    def test_dtype(self):
        state = self._state(dtype=jnp.float32)

        cur, _n_eff, _stats = self.variant(
            lambda s: reweight_step_jax(
                s,
                jnp.asarray(2, dtype=jnp.int32),
                METRIC_ESS,
                jnp.asarray(False),
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=s.logl.dtype),
                bins=16,
                bisect_steps=8,
                keep_max=4,
                trim_ess=0.95,
            )
        )(state)

        assert cur["u"].dtype == jnp.float32
        assert cur["x"].dtype == jnp.float32
        assert cur["logl"].dtype == jnp.float32
        assert cur["logp"].dtype == jnp.float32
        assert cur["weights"].dtype == jnp.float32
        assert cur["beta"].dtype == jnp.float32


if __name__ == "__main__":
    absltest.main()
