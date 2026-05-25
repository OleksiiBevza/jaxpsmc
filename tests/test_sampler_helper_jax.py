import chex
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.particles_jax import (
    ParticlesStep,
    init_particles_state_jax,
    record_step_jax,
)
from jaxpsmc.scaler_jax import init_bounds_config_jax, masks_jax
from jaxpsmc.sampler_helper_jax import (
    METRIC_ESS,
    METRIC_USS,
    _ECONVERGED,
    _EVALUEERR,
    _bisect_beta_scan,
    _dynamic_neff,
    _log_like,
    _log_like_batched,
    _metric_value,
    _systematic_resample_impl,
    _weights_metric_logz,
    mutate,
    not_termination_jax,
    posterior_jax,
    resample_particles_jax,
    reweight_step_jax,
    reweight_step_persistent_jax,
    trim_weights_scan_jax,
)
from jaxpsmc.tools_jax import effective_sample_size_jax, unique_sample_size_jax


class IdentityBijection:
    def transform_and_log_det(self, u, condition=None):
        return u, jnp.zeros((), dtype=u.dtype)

    def inverse_and_log_det(self, theta, condition=None):
        return theta, jnp.zeros((), dtype=theta.dtype)


class Flow:
    def __init__(self):
        self.bijection = IdentityBijection()


class SamplerHelperTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(0)

    def _cfg(self, dim):
        cfg = init_bounds_config_jax(dim, scale=False)
        msk = masks_jax(cfg["low"], cfg["high"])
        return cfg, msk

    def _step(self, u, logl, *, beta, logz, value=0.0, B=1, dtype=jnp.float32):
        u = jnp.asarray(u, dtype=dtype)
        logl = jnp.asarray(logl, dtype=dtype)
        N, D = u.shape
        x = u
        logp = -0.5 * jnp.sum(x * x, axis=1)
        blobs = jnp.full((N, B), jnp.asarray(value, dtype=dtype), dtype=dtype)

        return ParticlesStep(
            u=u,
            x=x,
            logdetj=jnp.zeros((N,), dtype=dtype),
            logl=logl,
            logp=logp,
            logw=jnp.zeros((N,), dtype=dtype),
            blobs=blobs,
            iter=jnp.asarray(value, dtype=jnp.int32),
            logz=jnp.asarray(logz, dtype=dtype),
            calls=jnp.asarray(value + 1.0, dtype=dtype),
            steps=jnp.asarray(value + 2.0, dtype=dtype),
            efficiency=jnp.asarray(value + 3.0, dtype=dtype),
            ess=jnp.asarray(value + 4.0, dtype=dtype),
            accept=jnp.asarray(value + 5.0, dtype=dtype),
            beta=jnp.asarray(beta, dtype=dtype),
        )

    def _state(self, T=3, N=2, D=2, B=1, dtype=jnp.float32):
        state = init_particles_state_jax(
            max_steps=T,
            n_particles=N,
            n_dim=D,
            blob_dim=B,
            dtype=dtype,
        )
        s1 = self._step(
            [[0.0, 0.0], [1.0, -1.0]],
            [0.0, -1.0],
            beta=0.0,
            logz=0.0,
            value=0.0,
            B=B,
            dtype=dtype,
        )
        s2 = self._step(
            [[0.5, 0.2], [-0.5, 1.0]],
            [-2.0, -3.0],
            beta=0.5,
            logz=-0.2,
            value=1.0,
            B=B,
            dtype=dtype,
        )

        state = record_step_jax(state, s1)
        state = record_step_jax(state, s2)
        return state

    def _current(self, N=4, D=2, B=1, dtype=jnp.float64):
        u = jnp.asarray(
            [[0.0, 0.0], [1.0, -1.0], [0.5, 0.25], [-0.5, 0.5]],
            dtype=dtype,
        )[:N, :D]
        x = u
        logl = -0.5 * jnp.sum((x - 0.25) ** 2, axis=1)
        logp = -0.5 * jnp.sum(x * x, axis=1)
        return {
            "u": u,
            "x": x,
            "logdetj": jnp.zeros((N,), dtype=dtype),
            "logl": logl,
            "logp": logp,
            "logdetj_flow": jnp.zeros((N,), dtype=dtype),
            "blobs": jnp.zeros((N, B), dtype=dtype),
            "beta": jnp.asarray(0.5, dtype=dtype),
            "calls": jnp.asarray(5, dtype=jnp.int32),
            "proposal_scale": jnp.asarray(0.2, dtype=dtype),
        }

    def _resample_input(self, dtype=jnp.float32):
        u = jnp.asarray([[0.0], [1.0], [2.0]], dtype=dtype)
        x = 10.0 + u
        return {
            "u": u,
            "x": x,
            "logdetj": jnp.asarray([0.0, 0.1, 0.2], dtype=dtype),
            "logl": jnp.asarray([-1.0, -2.0, -3.0], dtype=dtype),
            "logp": jnp.asarray([-0.1, -0.2, -0.3], dtype=dtype),
            "weights": jnp.asarray([0.0, 1.0, 0.0], dtype=dtype),
            "blobs": jnp.asarray([[0.0], [1.0], [2.0]], dtype=dtype),
        }

    def _loglike(self, x):
        ll = -0.5 * jnp.sum((x - 0.25) ** 2)
        blob = jnp.array([jnp.sum(x)], dtype=x.dtype)
        return ll, blob

    def _approx(self, x):
        return -0.25 * jnp.sum((x - 0.25) ** 2)

    def _prior(self, x):
        return -0.5 * jnp.sum(x * x)

    def _dili_geom(self, dtype=jnp.float64):
        center = jnp.asarray([0.0, 0.0], dtype=dtype)
        basis = jnp.asarray([[1.0], [0.0]], dtype=dtype)
        post_var = jnp.asarray([0.75], dtype=dtype)
        cov_ref = jnp.eye(2, dtype=dtype)
        return center, basis, post_var, cov_ref

    def _assert_current_keys(self, out):
        expected = {
            "u",
            "x",
            "logdetj",
            "logl",
            "logp",
            "blobs",
            "logz",
            "beta",
            "weights",
            "ess",
            "idx",
            "keep_mask",
            "trim_threshold",
            "trim_ratio",
            "trim_mask_full",
        }
        assert set(out.keys()) == expected

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
        np.testing.assert_allclose(metric, effective_sample_size_jax(weights), rtol=1e-6, atol=1e-6)
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
                keep_max=1,      # must be ignored by exact persistent mode
                trim_ess=0.25,   # must be ignored by exact persistent mode
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
            jnp.where(mask_flat[:, None], blobs_flat, jnp.asarray(0.0, blobs_flat.dtype)),
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

        np.testing.assert_allclose(cur["weights"], expected_weights, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(jnp.sum(cur["weights"]), 1.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cur["weights"], jnp.where(mask_flat, cur["weights"], 0.0), rtol=1e-6, atol=1e-6)
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
            np.testing.assert_allclose(cur_small[key], cur_large[key], rtol=1e-6, atol=1e-6)

        np.testing.assert_allclose(cur_small["beta"], cur_large["beta"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cur_small["logz"], cur_large["logz"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cur_small["ess"], cur_large["ess"], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(n_eff_small, n_eff_large)

        for key in ("beta", "logz", "ess", "n_effective"):
            np.testing.assert_allclose(stats_small[key], stats_large[key], rtol=1e-6, atol=1e-6)

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
        np.testing.assert_allclose(out["weights"], 0.25 * jnp.ones((4,), dtype=cur["u"].dtype))

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
        np.testing.assert_allclose(cur["blobs"], jnp.zeros((6, 0), dtype=state.logl.dtype))
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

    def test_loglike(self):
        x = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

        ll, blob = _log_like(x, self._loglike)

        np.testing.assert_allclose(ll, -0.5 * np.sum((np.asarray(x) - 0.25) ** 2), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(blob, jnp.asarray([3.0], dtype=jnp.float32))

    def test_loglike_batch(self):
        x = jnp.asarray([[1.0, 2.0], [0.0, 0.5]], dtype=jnp.float32)

        ll, blob = _log_like_batched(x, self._loglike)

        expected_ll = -0.5 * jnp.sum((x - 0.25) ** 2, axis=1)
        expected_blob = jnp.sum(x, axis=1, keepdims=True)
        np.testing.assert_allclose(ll, expected_ll, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(blob, expected_blob, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_mutate(self):
        cur = self._current(dtype=jnp.float64)
        cfg, msk = self._cfg(cur["u"].shape[1])

        def run(k, c):
            return mutate(
                k,
                c,
                use_preconditioned_pcn=jnp.asarray(False),
                loglike_single_fn=self._loglike,
                loglike_approx_single_fn=self._approx,
                logprior_fn=self._prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((2,), dtype=c["u"].dtype),
                geom_cov=jnp.eye(2, dtype=c["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=c["u"].dtype),
                n_max=0,
                n_steps=1,
            )

        key_out, out, info = self.variant(run)(self.key, cur)

        expected = {
            "u",
            "x",
            "logdetj",
            "logl",
            "logp",
            "logdetj_flow",
            "blobs",
            "beta",
            "calls",
            "proposal_scale",
            "efficiency",
            "steps",
            "accept",
        }
        assert set(out.keys()) == expected
        np.testing.assert_array_equal(jax.random.key_data(key_out), jax.random.key_data(self.key))
        np.testing.assert_allclose(out["u"], cur["u"])
        np.testing.assert_allclose(out["x"], cur["x"])
        np.testing.assert_allclose(out["logl"], cur["logl"])
        np.testing.assert_allclose(out["logp"], cur["logp"])
        np.testing.assert_array_equal(out["calls"], cur["calls"])
        np.testing.assert_array_equal(out["steps"], jnp.asarray(0, dtype=jnp.int32))
        np.testing.assert_allclose(out["accept"], 0.0)
        np.testing.assert_array_equal(info["calls_increment"], jnp.asarray(0, dtype=jnp.int32))


    @chex.all_variants(with_pmap=False)
    def test_mutate_pcn_kernel_active(self):
        cur = self._current(dtype=jnp.float64)
        cfg, msk = self._cfg(cur["u"].shape[1])

        def run(k, c):
            return mutate(
                k,
                c,
                use_preconditioned_pcn=jnp.asarray(True),
                loglike_single_fn=self._loglike,
                loglike_approx_single_fn=self._approx,
                logprior_fn=self._prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((2,), dtype=c["u"].dtype),
                geom_cov=jnp.eye(2, dtype=c["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=c["u"].dtype),
                kernel="pcn",
                n_max=2,
                n_steps=1,
            )

        key_out, out, info = self.variant(run)(self.key, cur)

        expected = {
            "u",
            "x",
            "logdetj",
            "logl",
            "logp",
            "logdetj_flow",
            "blobs",
            "beta",
            "calls",
            "proposal_scale",
            "efficiency",
            "steps",
            "accept",
        }
        assert set(out.keys()) == expected
        assert set(info.keys()) == {
            "efficiency_raw",
            "proposal_scale",
            "accept",
            "steps",
            "calls_increment",
        }

        assert out["u"].shape == cur["u"].shape
        assert out["x"].shape == cur["x"].shape
        assert out["logl"].shape == cur["logl"].shape
        assert out["logp"].shape == cur["logp"].shape
        assert out["blobs"].shape == cur["blobs"].shape
        assert out["logdetj_flow"].shape == cur["logdetj_flow"].shape

        assert not np.array_equal(
            np.asarray(jax.random.key_data(key_out)),
            np.asarray(jax.random.key_data(self.key)),
        )
        assert 1 <= int(out["steps"]) <= 2
        assert int(info["calls_increment"]) >= 0
        np.testing.assert_array_equal(out["calls"], cur["calls"] + info["calls_increment"])
        np.testing.assert_array_equal(out["steps"], info["steps"])
        np.testing.assert_allclose(out["accept"], info["accept"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out["proposal_scale"], info["proposal_scale"], rtol=1e-6, atol=1e-6)
        assert 0.0 <= float(out["accept"]) <= 1.0
        assert 0.0 <= float(out["proposal_scale"]) <= 0.99
        assert bool(jnp.all(jnp.isfinite(out["u"])))
        assert bool(jnp.all(jnp.isfinite(out["x"])))
        assert bool(jnp.all(jnp.isfinite(out["logl"])))
        assert bool(jnp.all(jnp.isfinite(out["logp"])))
        assert bool(jnp.isfinite(out["efficiency"]))

    @chex.all_variants(with_pmap=False)
    def test_mutate_li_pcn_kernel_noop(self):
        cur = self._current(dtype=jnp.float64)
        cfg, msk = self._cfg(cur["u"].shape[1])

        def run(k, c):
            return mutate(
                k,
                c,
                use_preconditioned_pcn=jnp.asarray(False),
                loglike_single_fn=self._loglike,
                loglike_approx_single_fn=self._approx,
                logprior_fn=self._prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((2,), dtype=c["u"].dtype),
                geom_cov=jnp.eye(2, dtype=c["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=c["u"].dtype),
                li_geom_mu=jnp.array([0.1, -0.1], dtype=c["u"].dtype),
                li_geom_cov=jnp.array(
                    [[1.0, 0.2], [0.2, 0.5]],
                    dtype=c["u"].dtype,
                ),
                kernel="li_pcn",
                li_rank=1,
                li_lis_scale=1.0,
                li_cs_scale=1.0,
                li_var_floor=1e-8,
                li_complement_var=1.0,
                n_max=2,
                n_steps=1,
            )

        key_out, out, info = self.variant(run)(self.key, cur)

        np.testing.assert_array_equal(jax.random.key_data(key_out), jax.random.key_data(self.key))
        np.testing.assert_allclose(out["u"], cur["u"])
        np.testing.assert_allclose(out["x"], cur["x"])
        np.testing.assert_allclose(out["logdetj"], cur["logdetj"])
        np.testing.assert_allclose(out["logdetj_flow"], cur["logdetj_flow"])
        np.testing.assert_allclose(out["logl"], cur["logl"])
        np.testing.assert_allclose(out["logp"], cur["logp"])
        np.testing.assert_allclose(out["blobs"], cur["blobs"])
        np.testing.assert_array_equal(out["calls"], cur["calls"])
        np.testing.assert_array_equal(out["steps"], jnp.asarray(0, dtype=jnp.int32))
        np.testing.assert_allclose(out["accept"], 0.0)
        np.testing.assert_array_equal(info["calls_increment"], jnp.asarray(0, dtype=jnp.int32))
        np.testing.assert_allclose(out["proposal_scale"], cur["proposal_scale"])

    @chex.all_variants(with_pmap=False)
    def test_mutate_li_pcn_kernel_active(self):
        cur = self._current(dtype=jnp.float64)
        cfg, msk = self._cfg(cur["u"].shape[1])

        def run(k, c):
            return mutate(
                k,
                c,
                use_preconditioned_pcn=jnp.asarray(True),
                loglike_single_fn=self._loglike,
                loglike_approx_single_fn=self._approx,
                logprior_fn=self._prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((2,), dtype=c["u"].dtype),
                geom_cov=jnp.eye(2, dtype=c["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=c["u"].dtype),
                li_geom_mu=jnp.array([0.1, -0.1], dtype=c["u"].dtype),
                li_geom_cov=jnp.array(
                    [[1.0, 0.2], [0.2, 0.5]],
                    dtype=c["u"].dtype,
                ),
                kernel="li_pcn",
                li_rank=1,
                li_lis_scale=1.0,
                li_cs_scale=1.0,
                li_var_floor=1e-8,
                li_complement_var=1.0,
                n_max=2,
                n_steps=1,
            )

        key_out, out, info = self.variant(run)(self.key, cur)

        expected = {
            "u",
            "x",
            "logdetj",
            "logl",
            "logp",
            "logdetj_flow",
            "blobs",
            "beta",
            "calls",
            "proposal_scale",
            "efficiency",
            "steps",
            "accept",
        }
        assert set(out.keys()) == expected
        assert set(info.keys()) == {
            "efficiency_raw",
            "proposal_scale",
            "accept",
            "steps",
            "calls_increment",
        }

        assert out["u"].shape == cur["u"].shape
        assert out["x"].shape == cur["x"].shape
        assert out["logl"].shape == cur["logl"].shape
        assert out["logp"].shape == cur["logp"].shape
        assert out["blobs"].shape == cur["blobs"].shape
        assert out["logdetj_flow"].shape == cur["logdetj_flow"].shape

        assert not np.array_equal(
            np.asarray(jax.random.key_data(key_out)),
            np.asarray(jax.random.key_data(self.key)),
        )
        assert 1 <= int(out["steps"]) <= 2
        assert int(info["calls_increment"]) >= 0
        np.testing.assert_array_equal(out["calls"], cur["calls"] + info["calls_increment"])
        np.testing.assert_array_equal(out["steps"], info["steps"])
        np.testing.assert_allclose(out["accept"], info["accept"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out["proposal_scale"], info["proposal_scale"], rtol=1e-6, atol=1e-6)
        assert 0.0 <= float(out["accept"]) <= 1.0
        assert 0.0 <= float(out["proposal_scale"]) <= 0.99
        assert bool(jnp.all(jnp.isfinite(out["u"])))
        assert bool(jnp.all(jnp.isfinite(out["x"])))
        assert bool(jnp.all(jnp.isfinite(out["logl"])))
        assert bool(jnp.all(jnp.isfinite(out["logp"])))
        assert bool(jnp.isfinite(out["efficiency"]))

    @chex.all_variants(with_pmap=False)
    def test_mutate_dili_pcn_kernel_noop(self):
        cur = self._current(dtype=jnp.float64)
        cfg, msk = self._cfg(cur["u"].shape[1])
        center, basis, post_var, cov_ref = self._dili_geom(dtype=cur["u"].dtype)

        def run(k, c):
            return mutate(
                k,
                c,
                use_preconditioned_pcn=jnp.asarray(False),
                loglike_single_fn=self._loglike,
                loglike_approx_single_fn=self._approx,
                logprior_fn=self._prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((2,), dtype=c["u"].dtype),
                geom_cov=jnp.eye(2, dtype=c["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=c["u"].dtype),
                dili_center=center,
                dili_basis=basis,
                dili_post_var=post_var,
                dili_cov_ref=cov_ref,
                kernel="dili_pcn",
                dili_lis_scale=1.0,
                dili_cs_scale=1.0,
                n_max=2,
                n_steps=1,
            )

        key_out, out, info = self.variant(run)(self.key, cur)

        np.testing.assert_array_equal(jax.random.key_data(key_out), jax.random.key_data(self.key))
        np.testing.assert_allclose(out["u"], cur["u"])
        np.testing.assert_allclose(out["x"], cur["x"])
        np.testing.assert_allclose(out["logdetj"], cur["logdetj"])
        np.testing.assert_allclose(out["logdetj_flow"], cur["logdetj_flow"])
        np.testing.assert_allclose(out["logl"], cur["logl"])
        np.testing.assert_allclose(out["logp"], cur["logp"])
        np.testing.assert_allclose(out["blobs"], cur["blobs"])
        np.testing.assert_array_equal(out["calls"], cur["calls"])
        np.testing.assert_array_equal(out["steps"], jnp.asarray(0, dtype=jnp.int32))
        np.testing.assert_allclose(out["accept"], 0.0)
        np.testing.assert_array_equal(info["calls_increment"], jnp.asarray(0, dtype=jnp.int32))
        np.testing.assert_allclose(out["proposal_scale"], cur["proposal_scale"])

    @chex.all_variants(with_pmap=False)
    def test_mutate_dili_pcn_kernel_active(self):
        cur = self._current(dtype=jnp.float64)
        cfg, msk = self._cfg(cur["u"].shape[1])
        center, basis, post_var, cov_ref = self._dili_geom(dtype=cur["u"].dtype)

        def run(k, c):
            return mutate(
                k,
                c,
                use_preconditioned_pcn=jnp.asarray(True),
                loglike_single_fn=self._loglike,
                loglike_approx_single_fn=self._approx,
                logprior_fn=self._prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((2,), dtype=c["u"].dtype),
                geom_cov=jnp.eye(2, dtype=c["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=c["u"].dtype),
                dili_center=center,
                dili_basis=basis,
                dili_post_var=post_var,
                dili_cov_ref=cov_ref,
                kernel="dili_pcn",
                dili_lis_scale=1.0,
                dili_cs_scale=1.0,
                n_max=2,
                n_steps=1,
            )

        key_out, out, info = self.variant(run)(self.key, cur)

        expected = {
            "u",
            "x",
            "logdetj",
            "logl",
            "logp",
            "logdetj_flow",
            "blobs",
            "beta",
            "calls",
            "proposal_scale",
            "efficiency",
            "steps",
            "accept",
        }
        assert set(out.keys()) == expected
        assert set(info.keys()) == {
            "efficiency_raw",
            "proposal_scale",
            "accept",
            "steps",
            "calls_increment",
        }

        assert out["u"].shape == cur["u"].shape
        assert out["x"].shape == cur["x"].shape
        assert out["logl"].shape == cur["logl"].shape
        assert out["logp"].shape == cur["logp"].shape
        assert out["blobs"].shape == cur["blobs"].shape
        assert out["logdetj_flow"].shape == cur["logdetj_flow"].shape

        assert not np.array_equal(
            np.asarray(jax.random.key_data(key_out)),
            np.asarray(jax.random.key_data(self.key)),
        )
        assert 0 <= int(out["steps"]) <= 2
        assert int(info["calls_increment"]) >= 0
        np.testing.assert_array_equal(out["calls"], cur["calls"] + info["calls_increment"])
        np.testing.assert_array_equal(out["steps"], info["steps"])
        np.testing.assert_allclose(out["accept"], info["accept"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out["proposal_scale"], info["proposal_scale"], rtol=1e-6, atol=1e-6)
        assert 0.0 <= float(out["accept"]) <= 1.0
        assert 0.0 <= float(out["proposal_scale"]) <= 0.99
        assert bool(jnp.all(jnp.isfinite(out["u"])))
        assert bool(jnp.all(jnp.isfinite(out["x"])))
        assert bool(jnp.all(jnp.isfinite(out["logl"])))
        assert bool(jnp.all(jnp.isfinite(out["logp"])))
        assert bool(jnp.isfinite(out["efficiency"]))

    def test_mutate_dili_pcn_requires_geometry(self):
        cur = self._current(dtype=jnp.float64)
        cfg, msk = self._cfg(cur["u"].shape[1])

        with self.assertRaisesRegex(ValueError, "kernel='dili_pcn' requires"):
            mutate(
                self.key,
                cur,
                use_preconditioned_pcn=jnp.asarray(True),
                loglike_single_fn=self._loglike,
                loglike_approx_single_fn=self._approx,
                logprior_fn=self._prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((2,), dtype=cur["u"].dtype),
                geom_cov=jnp.eye(2, dtype=cur["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=cur["u"].dtype),
                kernel="dili_pcn",
                n_max=1,
                n_steps=1,
            )

    def test_mutate_rejects_bad_kernel(self):
        cur = self._current(dtype=jnp.float64)
        cfg, msk = self._cfg(cur["u"].shape[1])

        with self.assertRaisesRegex(ValueError, "kernel must be one of"):
            mutate(
                self.key,
                cur,
                use_preconditioned_pcn=jnp.asarray(False),
                loglike_single_fn=self._loglike,
                loglike_approx_single_fn=self._approx,
                logprior_fn=self._prior,
                flow=Flow(),
                scaler_cfg=cfg,
                scaler_masks=msk,
                geom_mu=jnp.zeros((2,), dtype=cur["u"].dtype),
                geom_cov=jnp.eye(2, dtype=cur["u"].dtype),
                geom_nu=jnp.asarray(10.0, dtype=cur["u"].dtype),
                kernel="bad",
                n_max=0,
                n_steps=1,
            )

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

    @chex.all_variants(with_pmap=False)
    def test_systematic(self):
        weights = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32)

        idx, status, key_out = self.variant(
            lambda k, w: _systematic_resample_impl(k, w, size=5)
        )(self.key, weights)

        assert int(status) == int(_ECONVERGED)
        np.testing.assert_array_equal(idx, jnp.ones((5,), dtype=jnp.int32))
        assert not np.array_equal(jax.random.key_data(key_out), jax.random.key_data(self.key))

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
            lambda w: trim_weights_scan_jax(w, ess=jnp.asarray(0.5, dtype=w.dtype), bins=16)
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
            lambda w: trim_weights_scan_jax(w, ess=jnp.asarray(0.5, dtype=w.dtype), bins=8)
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
        np.testing.assert_array_equal(out.mask_valid, jnp.array([True, True, True, True, False, False]))
        np.testing.assert_array_equal(out.mask_trim, out.mask_valid)
        np.testing.assert_array_equal(out.idx_resampled, jnp.arange(6, dtype=jnp.int32))
        np.testing.assert_allclose(jnp.sum(out.weights), 1.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out.samples[-2:], 0.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out.logl[-2:], 0.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(jax.random.key_data(out.key_out), jax.random.key_data(self.key))
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
        np.testing.assert_allclose(out.weights[~out.mask_trim], 0.0, rtol=1e-6, atol=1e-6)

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
        assert not np.array_equal(jax.random.key_data(out.key_out), jax.random.key_data(self.key))

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
