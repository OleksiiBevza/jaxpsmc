import chex
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxpsmc.particles_jax import (
    ParticlesStep,
    init_particles_state_jax,
    record_step_jax,
)
from jaxpsmc.scaler_jax import init_bounds_config_jax, masks_jax


class IdentityBijection:
    def transform_and_log_det(self, u, condition=None):
        return u, jnp.zeros((), dtype=u.dtype)

    def inverse_and_log_det(self, theta, condition=None):
        return theta, jnp.zeros((), dtype=theta.dtype)


class Flow:
    def __init__(self):
        self.bijection = IdentityBijection()


class SamplerHelperBase(chex.TestCase):
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
        N, _D = u.shape
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
