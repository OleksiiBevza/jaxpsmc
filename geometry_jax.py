from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax

from tools_jax import *
from student_jax import *






@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Geometry:
    normal_mean: jax.Array  # (D,)
    normal_cov:  jax.Array  # (D,D)
    t_mean:      jax.Array  # (D,)
    t_cov:       jax.Array  # (D,D)
    t_nu:        jax.Array  # ()

    def tree_flatten(self):
        return (self.normal_mean, self.normal_cov, self.t_mean, self.t_cov, self.t_nu), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        nm, nc, tm, tc, tnu = children
        return cls(nm, nc, tm, tc, tnu)

    @classmethod
    def init(cls, dim: int, *, dtype=jnp.float64):
        z1 = jnp.zeros((dim,), dtype=dtype)
        z2 = jnp.zeros((dim, dim), dtype=dtype)
        nu = jnp.asarray(1e6, dtype=dtype)
        return cls(z1, z2, z1, z2, nu)


@jax.jit
def _cov_unweighted(theta: jax.Array, *, jitter: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Unweighted mean + sample covariance
    """
    theta = jnp.asarray(theta)
    n, d = theta.shape
    mu = jnp.mean(theta, axis=0)
    xc = theta - mu[None, :]
    denom = jnp.asarray(n - 1, theta.dtype)
    cov = (xc.T @ xc) / jnp.where(denom > 0, denom, jnp.asarray(1.0, theta.dtype))
    cov = 0.5 * (cov + cov.T)
    cov = cov + jitter * jnp.eye(d, dtype=theta.dtype)
    return mu, cov


@jax.jit
def _cov_weighted_aweights(theta: jax.Array, weights: jax.Array, *, jitter: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Weighted mean + covariance
    theta:   (N,D)
    weights: (N,)
    """
    theta = jnp.asarray(theta)
    w = jnp.asarray(weights)

    n, d = theta.shape
    dtype = theta.dtype

    # validate weights 
    wsum = jnp.sum(w)
    bad = (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(w)) | jnp.any(w < 0)

    # normalize weights
    w = w / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    mu = jnp.sum(theta * w[:, None], axis=0)
    xc = theta - mu[None, :]

    # aweights: normalization factor:
    # fact = 1 / (1 - sum(w^2))   because w is normalized to sum=1
    w2sum = jnp.sum(w * w)
    denom = (jnp.asarray(1.0, dtype) - w2sum)
    fact = jnp.where(denom > 0, jnp.asarray(1.0, dtype) / denom, jnp.asarray(0.0, dtype))

    cov = (xc * w[:, None]).T @ xc
    cov = cov * fact
    cov = 0.5 * (cov + cov.T)
    cov = cov + jitter * jnp.eye(d, dtype=dtype)

    # if bad, fall back to unweighted stats
    mu_u, cov_u = _cov_unweighted(theta, jitter=jitter)
    mu = jnp.where(bad, mu_u, mu)
    cov = jnp.where(bad, cov_u, cov)
    return mu, cov


@partial(jax.jit, static_argnames=("nu_cap",))
def _sanitize_nu(nu: jax.Array, nu_cap: float) -> jax.Array:
    cap = jnp.asarray(nu_cap, dtype=nu.dtype)
    return jnp.where(jnp.isfinite(nu), nu, cap)


@partial(jax.jit, static_argnames=("nu_cap",))
def geometry_fit_jax(
    geom: Geometry,
    theta: jax.Array,          # (N,D)
    weights: jax.Array,        # (N,)
    use_weights: jax.Array,    # bool scalar: True => use weights logic
    key: jax.Array,            # PRNGKey
    *,
    nu_cap: float = 1e6,
    jitter: float = 1e-9,
):
    """
    Pure JAX Geometry.fit

    Returns:
      geom_new, key_out, resample_status
    """
    theta = jnp.asarray(theta)
    weights = jnp.asarray(weights)
    use_weights = jnp.asarray(use_weights, dtype=bool)

    jitter = jnp.asarray(jitter, dtype=theta.dtype)

    # Normal fit: choose weighted vs unweighted via lax.cond (instead of python if)
    def _do_weighted(_):
        return _cov_weighted_aweights(theta, weights, jitter=jitter)

    def _do_unweighted(_):
        return _cov_unweighted(theta, jitter=jitter)

    normal_mean, normal_cov = lax.cond(use_weights, _do_weighted, _do_unweighted, operand=None)

    # T fit: if use_weights -> resample then fit; else fit directly
    n = theta.shape[0]

    def _t_fit_resampled(_):
        # normalize weights for resampling
        wsum = jnp.sum(weights)
        bad = (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0)
        w_norm = weights / jnp.where(bad, jnp.asarray(1.0, theta.dtype), wsum)

        idx, status, key_out = systematic_resample_jax(w_norm, key=key)
        idx_safe = jnp.clip(idx, 0, n - 1)
        theta_rs = theta[idx_safe]

        t_mean, t_cov, t_nu, _info = fit_mvstud_jax(theta_rs)  
        return t_mean, t_cov, _sanitize_nu(t_nu, nu_cap), key_out, status

    def _t_fit_direct(_):
        t_mean, t_cov, t_nu, _info = fit_mvstud_jax(theta)
        status = jnp.int64(0)
        return t_mean, t_cov, _sanitize_nu(t_nu, nu_cap), key, status

    t_mean, t_cov, t_nu, key_out, resample_status = lax.cond(use_weights, _t_fit_resampled, _t_fit_direct, operand=None)

    geom_new = Geometry(
        normal_mean=normal_mean,
        normal_cov=normal_cov,
        t_mean=t_mean,
        t_cov=t_cov,
        t_nu=t_nu,
    )
    return geom_new, key_out, resample_status