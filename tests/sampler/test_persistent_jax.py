# ruff: noqa: E402
import chex
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.sampler.constants_jax import METRIC_ESS, METRIC_USS, _ECONVERGED
from jaxpsmc.sampler.persistent_jax import reweight_step_persistent_jax
from jaxpsmc.sampler.resample_jax import resample_particles_jax
from jaxpsmc.sampler.reweight_jax import _weights_metric_logz, reweight_step_jax

from _sampler_test_utils import SamplerHelperBase


class PersistentTest(SamplerHelperBase):
    @chex.all_variants(with_pmap=False)
    def test_reweight_persistent_exact_full_history(self):
        state = self._state(T=3, N=2, D=2, B=1)

        def run(s):
            return reweight_step_persistent_jax(
                s,
                jnp.asarray(2, dtype=jnp.int32),
                METRIC_ESS,
                jnp.asarray(False),
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=s.logl.dtype),
                bins=16,
                bisect_steps=8,
                keep_max=1,  # must be ignored by exact persistent mode
                trim_ess=0.25,  # must be ignored by exact persistent mode
            )

        cur, n_eff, stats = self.variant(run)(state)

        self._assert_current_keys(cur)

        T, N, D = state.u.shape
        B = state.blobs.shape[-1]
        K = T * N

        mask_t = jnp.arange(T, dtype=state.t.dtype) < state.t
        mask_flat = jnp.repeat(mask_t, N)

        u_flat = state.u.reshape((K, D))
        x_flat = state.x.reshape((K, D))
        logdetj_flat = state.logdetj.reshape((K,))
        logl_flat = state.logl.reshape((K,))
        logp_flat = state.logp.reshape((K,))
        blobs_flat = state.blobs.reshape((K, B))

        assert cur["u"].shape == (K, D)
        assert cur["x"].shape == (K, D)
        assert cur["logdetj"].shape == (K,)
        assert cur["logl"].shape == (K,)
        assert cur["logp"].shape == (K,)
        assert cur["blobs"].shape == (K, B)
        assert cur["weights"].shape == (K,)
        assert cur["idx"].shape == (K,)
        assert cur["keep_mask"].shape == (K,)
        assert cur["trim_mask_full"].shape == (K,)

        np.testing.assert_array_equal(cur["idx"], jnp.arange(K, dtype=jnp.int32))
        np.testing.assert_array_equal(cur["keep_mask"], mask_flat)
        np.testing.assert_array_equal(cur["trim_mask_full"], mask_flat)

        np.testing.assert_allclose(
            cur["u"],
            jnp.where(mask_flat[:, None], u_flat, jnp.asarray(0.0, u_flat.dtype)),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            cur["x"],
            jnp.where(mask_flat[:, None], x_flat, jnp.asarray(0.0, x_flat.dtype)),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            cur["logdetj"],
            jnp.where(mask_flat, logdetj_flat, jnp.asarray(0.0, logdetj_flat.dtype)),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            cur["logl"],
            jnp.where(mask_flat, logl_flat, jnp.asarray(0.0, logl_flat.dtype)),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            cur["logp"],
            jnp.where(mask_flat, logp_flat, jnp.asarray(0.0, logp_flat.dtype)),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            cur["blobs"],
            jnp.where(
                mask_flat[:, None], blobs_flat, jnp.asarray(0.0, blobs_flat.dtype)
            ),
            rtol=1e-6,
            atol=1e-6,
        )

        expected_weights, expected_metric, expected_logz, _ = _weights_metric_logz(
            state,
            cur["beta"],
            METRIC_ESS,
            jnp.asarray(2, dtype=jnp.int32),
        )
        expected_weights = jnp.where(mask_flat, expected_weights, 0.0)
        expected_weights = expected_weights / jnp.sum(expected_weights)

        np.testing.assert_allclose(
            cur["weights"], expected_weights, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(jnp.sum(cur["weights"]), 1.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            cur["weights"],
            jnp.where(mask_flat, cur["weights"], 0.0),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(cur["ess"], expected_metric, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cur["logz"], expected_logz, rtol=1e-6, atol=1e-6)

        assert int(n_eff) == 2
        assert set(stats.keys()) == {"beta", "logz", "ess", "n_effective"}
        np.testing.assert_allclose(stats["beta"], cur["beta"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(stats["logz"], cur["logz"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(stats["ess"], cur["ess"], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(stats["n_effective"], n_eff)

        np.testing.assert_allclose(cur["trim_threshold"], 0.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cur["trim_ratio"], 1.0, rtol=1e-6, atol=1e-6)
        assert 0.0 <= float(cur["beta"]) <= 1.0
        assert bool(jnp.isfinite(cur["logz"]))

    @chex.all_variants(with_pmap=False)
    def test_reweight_persistent_ignores_keep_max_and_trim_ess(self):
        state = self._state(T=3, N=2, D=2, B=1)

        def run_small_keep(s):
            return reweight_step_persistent_jax(
                s,
                jnp.asarray(2, dtype=jnp.int32),
                METRIC_ESS,
                jnp.asarray(False),
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=s.logl.dtype),
                bins=16,
                bisect_steps=8,
                keep_max=1,
                trim_ess=0.25,
            )

        def run_large_keep(s):
            return reweight_step_persistent_jax(
                s,
                jnp.asarray(2, dtype=jnp.int32),
                METRIC_ESS,
                jnp.asarray(False),
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=s.logl.dtype),
                bins=16,
                bisect_steps=8,
                keep_max=6,
                trim_ess=0.99,
            )

        cur_small, n_eff_small, stats_small = self.variant(run_small_keep)(state)
        cur_large, n_eff_large, stats_large = self.variant(run_large_keep)(state)

        assert cur_small["u"].shape == cur_large["u"].shape == (6, 2)
        assert cur_small["weights"].shape == cur_large["weights"].shape == (6,)

        for key in (
            "u",
            "x",
            "logdetj",
            "logl",
            "logp",
            "blobs",
            "weights",
            "idx",
            "keep_mask",
            "trim_mask_full",
        ):
            np.testing.assert_allclose(
                cur_small[key], cur_large[key], rtol=1e-6, atol=1e-6
            )

        np.testing.assert_allclose(
            cur_small["beta"], cur_large["beta"], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            cur_small["logz"], cur_large["logz"], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            cur_small["ess"], cur_large["ess"], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(n_eff_small, n_eff_large)

        for key in ("beta", "logz", "ess", "n_effective"):
            np.testing.assert_allclose(
                stats_small[key], stats_large[key], rtol=1e-6, atol=1e-6
            )

    @chex.all_variants(with_pmap=False)
    def test_reweight_persistent_and_truncated_share_beta_logz_and_metric(self):
        state = self._state(T=3, N=2, D=2, B=1)

        def run_persistent(s):
            return reweight_step_persistent_jax(
                s,
                jnp.asarray(2, dtype=jnp.int32),
                METRIC_ESS,
                jnp.asarray(False),
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=s.logl.dtype),
                bins=16,
                bisect_steps=8,
                keep_max=1,
                trim_ess=0.25,
            )

        def run_truncated(s):
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

        cur_p, n_eff_p, stats_p = self.variant(run_persistent)(state)
        cur_t, n_eff_t, stats_t = self.variant(run_truncated)(state)

        # Candidate-pool shapes must differ: exact persistent keeps T*N,
        # truncated persistent keeps keep_max.
        assert cur_p["u"].shape == (6, 2)
        assert cur_t["u"].shape == (4, 2)

        # These quantities are computed before truncation, so they should match.
        np.testing.assert_allclose(cur_p["beta"], cur_t["beta"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cur_p["logz"], cur_t["logz"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cur_p["ess"], cur_t["ess"], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(n_eff_p, n_eff_t)

        for key in ("beta", "logz", "ess", "n_effective"):
            np.testing.assert_allclose(stats_p[key], stats_t[key], rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_reweight_persistent_dynamic(self):
        state = self._state(T=3, N=2, D=2, B=1)

        def run(s):
            return reweight_step_persistent_jax(
                s,
                jnp.asarray(2, dtype=jnp.int32),
                METRIC_USS,
                jnp.asarray(True),
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=s.logl.dtype),
                bins=16,
                bisect_steps=8,
                keep_max=1,
                trim_ess=0.25,
            )

        cur, n_eff, stats = self.variant(run)(state)

        assert n_eff.dtype == jnp.int32
        np.testing.assert_array_equal(stats["n_effective"], n_eff)
        assert cur["u"].shape == (6, 2)
        assert cur["weights"].shape == (6,)
        np.testing.assert_allclose(jnp.sum(cur["weights"]), 1.0, rtol=1e-6, atol=1e-6)
        assert bool(jnp.all(jnp.isfinite(cur["weights"])))
        assert bool(jnp.isfinite(cur["logz"]))

    @chex.all_variants(with_pmap=False)
    def test_reweight_persistent_resample_one_hot_weight(self):
        state = self._state(T=3, N=2, D=2, B=1)

        def build_cur(s):
            cur, _, _ = reweight_step_persistent_jax(
                s,
                jnp.asarray(2, dtype=jnp.int32),
                METRIC_ESS,
                jnp.asarray(False),
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=s.logl.dtype),
                bins=16,
                bisect_steps=8,
                keep_max=1,
                trim_ess=0.25,
            )
            one_hot = jnp.zeros_like(cur["weights"]).at[1].set(1.0)
            cur = dict(cur)
            cur["weights"] = one_hot
            return cur

        cur = self.variant(build_cur)(state)

        out, status, _key_out = self.variant(
            lambda c, k: resample_particles_jax(
                c,
                key=k,
                n_active=4,
                method_code=jnp.asarray(0, dtype=jnp.int32),
                reset_weights=True,
            )
        )(cur, self.key)

        assert int(status) == int(_ECONVERGED)
        np.testing.assert_allclose(
            out["u"],
            jnp.repeat(cur["u"][1:2], repeats=4, axis=0),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            out["x"],
            jnp.repeat(cur["x"][1:2], repeats=4, axis=0),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            out["logl"],
            jnp.repeat(cur["logl"][1:2], repeats=4, axis=0),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            out["weights"], 0.25 * jnp.ones((4,), dtype=cur["u"].dtype)
        )

    @chex.all_variants(with_pmap=False)
    def test_reweight_persistent_blob0(self):
        state = self._state(T=3, N=2, D=2, B=0)

        def run(s):
            return reweight_step_persistent_jax(
                s,
                jnp.asarray(2, dtype=jnp.int32),
                METRIC_ESS,
                jnp.asarray(False),
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=s.logl.dtype),
                bins=16,
                bisect_steps=8,
                keep_max=1,
                trim_ess=0.25,
            )

        cur, _n_eff, _stats = self.variant(run)(state)

        assert cur["u"].shape == (6, 2)
        assert cur["blobs"].shape == (6, 0)
        np.testing.assert_allclose(
            cur["blobs"], jnp.zeros((6, 0), dtype=state.logl.dtype)
        )
        np.testing.assert_allclose(jnp.sum(cur["weights"]), 1.0, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_reweight_persistent_dtype(self):
        state = self._state(dtype=jnp.float32)

        def run(s):
            return reweight_step_persistent_jax(
                s,
                jnp.asarray(2, dtype=jnp.int32),
                METRIC_ESS,
                jnp.asarray(False),
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=s.logl.dtype),
                bins=16,
                bisect_steps=8,
                keep_max=1,
                trim_ess=0.25,
            )

        cur, _n_eff, _stats = self.variant(run)(state)

        assert cur["u"].dtype == jnp.float32
        assert cur["x"].dtype == jnp.float32
        assert cur["logdetj"].dtype == jnp.float32
        assert cur["logl"].dtype == jnp.float32
        assert cur["logp"].dtype == jnp.float32
        assert cur["weights"].dtype == jnp.float32
        assert cur["beta"].dtype == jnp.float32
        assert cur["logz"].dtype == jnp.float32


if __name__ == "__main__":
    absltest.main()
