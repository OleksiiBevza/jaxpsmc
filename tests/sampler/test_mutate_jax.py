import chex
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from _sampler_test_utils import Flow, SamplerHelperBase
from absl.testing import absltest

from jaxpsmc.sampler.mutate_jax import _log_like, _log_like_batched, mutate


class MutateTest(SamplerHelperBase):
    def test_loglike(self):
        x = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

        ll, blob = _log_like(x, self._loglike)

        np.testing.assert_allclose(
            ll, -0.5 * np.sum((np.asarray(x) - 0.25) ** 2), rtol=1e-6, atol=1e-6
        )
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
        np.testing.assert_array_equal(
            jax.random.key_data(key_out), jax.random.key_data(self.key)
        )
        np.testing.assert_allclose(out["u"], cur["u"])
        np.testing.assert_allclose(out["x"], cur["x"])
        np.testing.assert_allclose(out["logl"], cur["logl"])
        np.testing.assert_allclose(out["logp"], cur["logp"])
        np.testing.assert_array_equal(out["calls"], cur["calls"])
        np.testing.assert_array_equal(out["steps"], jnp.asarray(0, dtype=jnp.int32))
        np.testing.assert_allclose(out["accept"], 0.0)
        np.testing.assert_array_equal(
            info["calls_increment"], jnp.asarray(0, dtype=jnp.int32)
        )

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
        np.testing.assert_array_equal(
            out["calls"], cur["calls"] + info["calls_increment"]
        )
        np.testing.assert_array_equal(out["steps"], info["steps"])
        np.testing.assert_allclose(out["accept"], info["accept"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            out["proposal_scale"], info["proposal_scale"], rtol=1e-6, atol=1e-6
        )
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

        np.testing.assert_array_equal(
            jax.random.key_data(key_out), jax.random.key_data(self.key)
        )
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
        np.testing.assert_array_equal(
            info["calls_increment"], jnp.asarray(0, dtype=jnp.int32)
        )
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
        np.testing.assert_array_equal(
            out["calls"], cur["calls"] + info["calls_increment"]
        )
        np.testing.assert_array_equal(out["steps"], info["steps"])
        np.testing.assert_allclose(out["accept"], info["accept"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            out["proposal_scale"], info["proposal_scale"], rtol=1e-6, atol=1e-6
        )
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

        np.testing.assert_array_equal(
            jax.random.key_data(key_out), jax.random.key_data(self.key)
        )
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
        np.testing.assert_array_equal(
            info["calls_increment"], jnp.asarray(0, dtype=jnp.int32)
        )
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
        np.testing.assert_array_equal(
            out["calls"], cur["calls"] + info["calls_increment"]
        )
        np.testing.assert_array_equal(out["steps"], info["steps"])
        np.testing.assert_allclose(out["accept"], info["accept"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            out["proposal_scale"], info["proposal_scale"], rtol=1e-6, atol=1e-6
        )
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


if __name__ == "__main__":
    absltest.main()
