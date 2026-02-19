from __future__ import annotations

from typing import Callable, Mapping, Tuple, Any, Optional, Dict

import jax
import jax.numpy as jnp

from .scaler_jax import *


Array = jax.Array

# remember that these guys is for identity flow only 
def _flow_u_to_theta_jax(flow, u: Array, condition: Optional[Array] = None) -> Tuple[Array, Array]:
    """
        * mapping u into theta using FlowJAX bijection
        * return log|det du/dtheta|.

    FlowJAX:
        * transform_and_log_det(u) returns (theta, log|det dtheta/du|)
        * get log|det du/dtheta|.
    """
    theta, fwd_logdet = flow.bijection.transform_and_log_det(u, condition)
    return theta, -fwd_logdet


def _flow_theta_to_u_jax(flow, theta: Array, condition: Optional[Array] = None) -> Tuple[Array, Array]:
    """
        * mapping theta into u using FlowJAX bijection
        * return log|det du/dtheta| directly.
    """
    u, inv_logdet = flow.bijection.inverse_and_log_det(theta, condition)
    return u, inv_logdet


def preconditioned_pcn_jax(
    key: Array,
    *,
    # current state (all arrays; no None)
    u: Array,                 # (N, D)
    x: Array,                 # (N, D)
    logdetj: Array,           # (N,)
    logl: Array,              # (N,)
    logp: Array,              # (N,)
    logdetj_flow: Array,      # (N,)
    blobs: Array,             # (N, B...) ; use shape (N, 0) if no blobs

    beta: Array,              # scalar

    # functions
    loglike_fn: Callable[[Array], Tuple[Array, Array]],
    logprior_fn: Callable[[Array], Array],
    flow: Any,                # FlowJAX Transformed-like object with .bijection
    scaler_cfg: Mapping[str, Array],
    scaler_masks: Mapping[str, Array],

    # geometry (Student-t)
    geom_mu: Array,           # (D,)
    geom_cov: Array,          # (D, D)
    geom_nu: Array,           # scalar

    # options
    n_max: int,
    n_steps: int,
    proposal_scale: Array,    # scalar
    condition: Optional[Array] = None,
) -> Dict[str, Array]:
    """
    Doubly Preconditioned Crank–Nicolson (PCN), JAX version.

    Requirements:
      - logprior_fn(x_i): x_i has shape (D,), returns scalar
      - loglike_fn(x_i): x_i has shape (D,), returns (scalar_loglike, blob_i)
        where blob_i has fixed shape (B...) matching blobs[0].

    Info:
      - All randomness uses `key` and returns updated `key` (make sure it is pure and check randomness).
      - FlowJAX bijections are vmapped across walkers.
    """
    u = jnp.asarray(u)
    x = jnp.asarray(x)
    logdetj = jnp.asarray(logdetj)
    logl = jnp.asarray(logl)
    logp = jnp.asarray(logp)
    logdetj_flow = jnp.asarray(logdetj_flow)
    blobs = jnp.asarray(blobs)
    beta = jnp.asarray(beta)
    proposal_scale = jnp.asarray(proposal_scale)

    geom_mu = jnp.asarray(geom_mu)
    geom_cov = jnp.asarray(geom_cov)
    geom_nu = jnp.asarray(geom_nu)

    n_walkers, n_dim = u.shape

    inv_cov = jnp.linalg.inv(geom_cov)
    chol_cov = jnp.linalg.cholesky(geom_cov)

    # Flow: u -> theta (batched via vmap)
    def _u2t_single(ui: Array) -> Tuple[Array, Array]:
        return _flow_u_to_theta_jax(flow, ui, condition)

    theta, logdetj_flow0 = jax.vmap(_u2t_single, in_axes=0, out_axes=(0, 0))(u)

    # initial mean and counter and objective
    mu = geom_mu
    sigma0 = jnp.minimum(proposal_scale, jnp.asarray(0.99, dtype=u.dtype))
    logp2_best = jnp.mean(logl + logp)
    cnt0 = jnp.asarray(0, dtype=jnp.int32)
    i0 = jnp.asarray(0, dtype=jnp.int32)
    calls0 = jnp.asarray(0, dtype=jnp.int32)
    accept0 = jnp.asarray(0.0, dtype=u.dtype)
    done0 = jnp.asarray(False)

    # update initial flow logdet with computed one 
    logdetj_flow = logdetj_flow0

    blob_template = jnp.zeros_like(blobs[0])

    # helpers: Student-t form
    def _quad(diff_: Array) -> Array:
        tmp = diff_ @ inv_cov
        return jnp.sum(tmp * diff_, axis=1)

    # skip invalid walkers
    def _prior_or_neginf(xi: Array, ok: Array) -> Array:
        return jax.lax.cond(
            ok,
            lambda z: logprior_fn(z),
            lambda z: jnp.asarray(-jnp.inf, dtype=xi.dtype),
            xi,
        )

    def _like_or_neginf(xi: Array, ok: Array) -> Tuple[Array, Array]:
        def _do(z: Array) -> Tuple[Array, Array]:
            ll, bb = loglike_fn(z)
            return ll, bb

        def _skip(z: Array) -> Tuple[Array, Array]:
            return jnp.asarray(-jnp.inf, dtype=xi.dtype), blob_template

        return jax.lax.cond(ok, _do, _skip, xi)

    
    # (key, u, x, theta, logdetj, logdetj_flow, logl, logp, blobs, mu, sigma, logp2_best, cnt, i, calls, accept, done)
    carry0 = (
        key, u, x, theta, logdetj, logdetj_flow, logl, logp, blobs,
        mu, sigma0, logp2_best, cnt0, i0, calls0, accept0, done0
    )

    max_sigma_cap = jnp.minimum(jnp.asarray(2.38, dtype=u.dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=u.dtype)),
                                jnp.asarray(0.99, dtype=u.dtype))

    def cond_fn(carry):
        (_, _, _, _, _, _, _, _, _, _, _, _, _, i, _, _, done) = carry
        return (i < jnp.asarray(n_max, dtype=i.dtype)) & (~done)

    def body_fn(carry):
        (key, u, x, theta, logdetj, logdetj_flow, logl, logp, blobs,
         mu, sigma, logp2_best, cnt, i, calls, accept, done) = carry

        i1 = i + jnp.asarray(1, dtype=i.dtype)

        key, k_gamma, k_norm, k_unif = jax.random.split(key, 4)

        diff = theta - mu
        quad = _quad(diff)

        a = (jnp.asarray(n_dim, dtype=u.dtype) + geom_nu) / jnp.asarray(2.0, dtype=u.dtype)
        z = jax.random.gamma(k_gamma, a, shape=(n_walkers,))  # unit scale
        s = (geom_nu + quad) / (jnp.asarray(2.0, dtype=u.dtype) * z)

        eps = jax.random.normal(k_norm, shape=(n_walkers, n_dim), dtype=u.dtype)
        noise = eps @ chol_cov.T

        theta_prime = (
            mu
            + jnp.sqrt(jnp.asarray(1.0, dtype=u.dtype) - sigma * sigma) * diff
            + sigma * jnp.sqrt(s)[:, None] * noise
        )

        # --- Flow: theta into u (batched via vmap) ---
        def _t2u_single(ti: Array) -> Tuple[Array, Array]:
            return _flow_theta_to_u_jax(flow, ti, condition)

        u_prime, logdetj_flow_prime = jax.vmap(_t2u_single, in_axes=0, out_axes=(0, 0))(theta_prime)

        # --- Scaler inverse: u into x, ---
        #TODO check boundary handling here 
        x_prime, logdetj_prime = inverse_jax(u_prime, scaler_cfg, scaler_masks)

        x_prime_bc = apply_boundary_conditions_x_jax(x_prime, dict(scaler_cfg))
        u_prime_bc = forward_jax(x_prime_bc, scaler_cfg, scaler_masks)
        x_prime, logdetj_prime = inverse_jax(u_prime_bc, scaler_cfg, scaler_masks)

        u_prime = u_prime_bc

        finite0 = jnp.isfinite(logdetj_prime) & jnp.all(jnp.isfinite(x_prime), axis=1)

        logp_prime = jax.vmap(_prior_or_neginf, in_axes=(0, 0), out_axes=0)(x_prime, finite0)
        finite1 = finite0 & jnp.isfinite(logp_prime)

        logl_prime, blobs_prime = jax.vmap(_like_or_neginf, in_axes=(0, 0), out_axes=(0, 0))(x_prime, finite1)

        # calls = calls + jnp.sum(finite1.astype(jnp.int32))
        calls_inc = jnp.sum(finite1.astype(calls.dtype), dtype=calls.dtype)
        calls = calls + calls_inc

        diff_prime = theta_prime - mu
        quad_prime = _quad(diff_prime)

        coef = -(jnp.asarray(n_dim, dtype=u.dtype) + geom_nu) / jnp.asarray(2.0, dtype=u.dtype)
        A = coef * jnp.log1p(quad_prime / geom_nu)
        B = coef * jnp.log1p(quad / geom_nu)

        log_alpha = (
            (logl_prime - logl) * beta
            + (logp_prime - logp)
            + (logdetj_prime - logdetj)
            + (logdetj_flow_prime - logdetj_flow)
            - A + B
        )

        alpha = jnp.exp(jnp.minimum(jnp.asarray(0.0, dtype=u.dtype), log_alpha))
        alpha = jnp.where(jnp.isnan(alpha), jnp.asarray(0.0, dtype=u.dtype), alpha)

        u_rand = jax.random.uniform(k_unif, shape=(n_walkers,), dtype=u.dtype)
        accept_mask = u_rand < alpha

        # accept / reject
        # TODO check this
        theta = jnp.where(accept_mask[:, None], theta_prime, theta)
        u = jnp.where(accept_mask[:, None], u_prime, u)
        x = jnp.where(accept_mask[:, None], x_prime, x)

        logdetj = jnp.where(accept_mask, logdetj_prime, logdetj)
        logdetj_flow = jnp.where(accept_mask, logdetj_flow_prime, logdetj_flow)
        logl = jnp.where(accept_mask, logl_prime, logl)
        logp = jnp.where(accept_mask, logp_prime, logp)
        blobs = jnp.where(accept_mask.reshape((n_walkers,) + (1,) * (blobs.ndim - 1)), blobs_prime, blobs)

        accept = jnp.mean(alpha)

        # TODO check
        step = jnp.asarray(1.0, dtype=u.dtype) / jnp.power(jnp.asarray(i1 + 1, dtype=u.dtype), jnp.asarray(0.75, dtype=u.dtype))
        sigma = sigma + step * (accept - jnp.asarray(0.234, dtype=u.dtype))
        sigma = jnp.abs(jnp.minimum(sigma, max_sigma_cap))

        mu_step = jnp.asarray(1.0, dtype=u.dtype) / jnp.asarray(i1 + 1, dtype=u.dtype)
        mu = mu + mu_step * (jnp.mean(theta, axis=0) - mu)

        logp2_new = jnp.mean(logl + logp)
        improved = logp2_new > logp2_best
        cnt = jnp.where(improved, jnp.asarray(0, dtype=cnt.dtype), cnt + jnp.asarray(1, dtype=cnt.dtype))
        logp2_best = jnp.where(improved, logp2_new, logp2_best)

        thresh = jnp.asarray(n_steps, dtype=u.dtype) * jnp.power(
            (jnp.asarray(2.38, dtype=u.dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=u.dtype))) / sigma,
            jnp.asarray(2.0, dtype=u.dtype),
        )
        done = cnt.astype(u.dtype) >= thresh

        return (
            key, u, x, theta, logdetj, logdetj_flow, logl, logp, blobs,
            mu, sigma, logp2_best, cnt, i1, calls, accept, done
        )

    carry_f = jax.lax.while_loop(cond_fn, body_fn, carry0)

    (key, u, x, theta, logdetj, logdetj_flow, logl, logp, blobs,
     mu, sigma, logp2_best, cnt, i, calls, accept, done) = carry_f

    return {
        "key": key,
        "u": u,
        "x": x,
        "logdetj": logdetj,
        "logdetj_flow": logdetj_flow,
        "logl": logl,
        "logp": logp,
        "blobs": blobs,
        "efficiency": sigma,
        "accept": accept,
        "steps": i,
        "calls": calls,
        "proposal_scale": sigma,
    }