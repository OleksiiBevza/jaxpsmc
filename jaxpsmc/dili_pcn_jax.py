from __future__ import annotations

from typing import Callable, Mapping, Tuple, Any, Optional, Dict

import jax
import jax.numpy as jnp

from .dili_geometry_jax import DILIPCNGeometry
from .scaler_jax import *
from .pcn_jax import _flow_u_to_theta_jax, _flow_theta_to_u_jax
from .delayed_acceptance.da_conservative_damh_jax import (
    conservative_damh_step_jax,
)

Array = jax.Array


def _standard_normal_log_reference(theta: Array, center: Array) -> Array:
    """
    Log reference density up to an additive constant.

    DILI LI-Prior is reversible with respect to a Gaussian reference in
    whitened/theta coordinates. We include the reference-density correction
    explicitly so this remains valid in the sampler's transformed coordinates.
    """
    diff = theta - center[None, :]
    return -0.5 * jnp.sum(diff * diff, axis=1)


def _dili_li_prior_proposal(
    key: Array,
    theta: Array,
    center: Array,
    basis: Array,
    post_var: Array,
    sigma: Array,
    *,
    dili_lis_scale: float,
    dili_cs_scale: float,
) -> Array:
    """
    Hessian/GNH-based LI-Prior proposal.

    This implements the DILI idea:
      - use the GNH-based LIS basis in `basis`;
      - use the projected posterior variances `post_var` inside the LIS;
      - use pCN/CN movement in the complement.

    The proposal itself is gradient-free. Derivatives only enter through
    the construction of `basis` and `post_var`.
    """
    theta = jnp.asarray(theta)
    center = jnp.asarray(center)
    basis = jnp.asarray(basis)
    post_var = jnp.asarray(post_var)

    dtype = theta.dtype

    # Convert adaptive sigma into positive CN time steps.
    # sigma is kept bounded by the mutation loop; squaring keeps dt positive.
    sigma_lis = jnp.clip(jnp.abs(sigma * jnp.asarray(dili_lis_scale, dtype=dtype)), 1e-12, 0.99)
    sigma_cs = jnp.clip(jnp.abs(sigma * jnp.asarray(dili_cs_scale, dtype=dtype)), 1e-12, 0.99)

    tau_lis = sigma_lis * sigma_lis
    tau_cs = sigma_cs * sigma_cs

    # LI-Prior / CN coefficients in the LIS.
    # a_i = (2 - tau * D_i) / (2 + tau * D_i)
    # b_i = sqrt(1 - a_i^2)
    post_var = jnp.maximum(post_var, jnp.asarray(1e-12, dtype=dtype))
    a_lis = (2.0 - tau_lis * post_var) / (2.0 + tau_lis * post_var)
    a_lis = jnp.clip(a_lis, -0.999999, 0.999999)
    b_lis = jnp.sqrt(jnp.maximum(1.0 - a_lis * a_lis, 0.0))

    # Complement pCN coefficient.
    a_cs = (2.0 - tau_cs) / (2.0 + tau_cs)
    a_cs = jnp.clip(a_cs, -0.999999, 0.999999)
    b_cs = jnp.sqrt(jnp.maximum(1.0 - a_cs * a_cs, 0.0))

    key_lis, key_full = jax.random.split(key)

    diff = theta - center[None, :]

    # LIS coordinates.
    z_lis = diff @ basis                       # (N, r)
    eps_lis = jax.random.normal(key_lis, shape=z_lis.shape, dtype=dtype)
    z_lis_prime = a_lis[None, :] * z_lis + b_lis[None, :] * eps_lis
    diff_lis_prime = z_lis_prime @ basis.T

    # Complement coordinates.
    diff_lis = z_lis @ basis.T
    diff_cs = diff - diff_lis

    eps_full = jax.random.normal(key_full, shape=theta.shape, dtype=dtype)
    eps_lis_part = (eps_full @ basis) @ basis.T
    eps_cs = eps_full - eps_lis_part

    diff_cs_prime = a_cs * diff_cs + b_cs * eps_cs

    return center[None, :] + diff_lis_prime + diff_cs_prime


def _li_log_reference(theta: Array, mu: Array, eigvecs: Array, var_dir: Array) -> Array:
    """
    Log reference density up to an additive constant.

    The proposal is reversible with respect to this Gaussian reference.
    The missing log-determinant constant cancels in MH ratios because the
    geometry is fixed during one mutation call.
    """
    diff = theta - mu[None, :]
    z = diff @ eigvecs
    q = jnp.sum((z * z) / var_dir[None, :], axis=1)
    return -0.5 * q


def _li_pcn_proposal(
    key: Array,
    theta: Array,
    mu: Array,
    eigvecs: Array,
    var_dir: Array,
    active: Array,
    sigma: Array,
    *,
    li_lis_scale: float,
    li_cs_scale: float,
) -> Array:
    """
    Direction-wise pCN proposal in empirical LIS coordinates.

    In active LIS directions, the proposal uses the empirical covariance
    eigenvalues. In the complement, it uses the scalar complement variance.
    """
    dtype = theta.dtype

    lis_scale = jnp.asarray(li_lis_scale, dtype=dtype)
    cs_scale = jnp.asarray(li_cs_scale, dtype=dtype)

    sigma_lis = jnp.clip(jnp.abs(sigma * lis_scale), 1e-12, 0.99)
    sigma_cs = jnp.clip(jnp.abs(sigma * cs_scale), 1e-12, 0.99)

    sigma_dir = jnp.where(active, sigma_lis, sigma_cs)
    a_dir = jnp.sqrt(jnp.maximum(1.0 - sigma_dir * sigma_dir, 0.0))

    diff = theta - mu[None, :]
    z = diff @ eigvecs

    eps = jax.random.normal(key, shape=z.shape, dtype=dtype)
    z_prime = (
        a_dir[None, :] * z
        + sigma_dir[None, :] * jnp.sqrt(var_dir)[None, :] * eps
    )

    theta_prime = mu[None, :] + z_prime @ eigvecs.T
    return theta_prime


def dili_pcn_jax(
    key: Array,
    *,
    # current state
    u: Array,
    x: Array,
    logdetj: Array,
    logl: Array,
    logp: Array,
    logdetj_flow: Array,
    blobs: Array,
    beta: Array,

    # functions
    loglike_fn: Callable[[Array], Tuple[Array, Array]],
    loglike_approx_fn: Callable[[Array], Array],
    logprior_fn: Callable[[Array], Array],
    flow: Any,
    scaler_cfg: Mapping[str, Array],
    scaler_masks: Mapping[str, Array],

    # Hessian/GNH-based DILI geometry
    dili_center: Array,
    dili_basis: Array,
    dili_post_var: Array,
    dili_cov_ref: Array,

    # options
    n_max: int,
    n_steps: int,
    proposal_scale: Array,
    dili_lis_scale: float = 1.0,
    dili_cs_scale: float = 1.0,


    use_delayed_acceptance: Array = jnp.asarray(False),
    da_c_const: Array = jnp.asarray(0.01),
    da_d_const: Array = jnp.asarray(2.0),
    condition: Optional[Array] = None,
) -> Dict[str, Array]:
    """
    Empirical likelihood-informed pCN mutation kernel.

    This is not full Hessian-based DILI. It is a DILI-inspired
    empirical LI-pCN kernel:
      - empirical covariance eigenvectors define an LIS proxy;
      - pCN is applied in the eigenbasis;
      - complement directions use scalar reference variance;
      - MH correction includes the Gaussian reference-density ratio.
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

    dili_center = jnp.asarray(dili_center)
    dili_basis = jnp.asarray(dili_basis)
    dili_post_var = jnp.asarray(dili_post_var)
    dili_cov_ref = jnp.asarray(dili_cov_ref)

    n_walkers, n_dim = u.shape
    dtype = u.dtype

    def _u2t_single(ui: Array) -> Tuple[Array, Array]:
        return _flow_u_to_theta_jax(flow, ui, condition)

    theta, logdetj_flow0 = jax.vmap(
        _u2t_single,
        in_axes=0,
        out_axes=(0, 0),
    )(u)

    logdetj_flow = logdetj_flow0

    mu0 = dili_center  
    sigma0 = jnp.minimum(jnp.abs(proposal_scale), jnp.asarray(0.99, dtype=dtype))
    logp2_best0 = jnp.mean(logl + logp)
    cnt0 = jnp.asarray(0, dtype=jnp.int32)
    i0 = jnp.asarray(0, dtype=jnp.int32)
    calls0 = jnp.asarray(0, dtype=jnp.int32)
    accept0 = jnp.asarray(0.0, dtype=dtype)
    done0 = jnp.asarray(False)

    blob_template = jnp.zeros_like(blobs[0])

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

    def _approx_or_neginf(xi: Array, ok: Array) -> Array:
        def _do(z: Array) -> Array:
            return jnp.asarray(loglike_approx_fn(z), dtype=xi.dtype)

        def _skip(z: Array) -> Array:
            return jnp.asarray(-jnp.inf, dtype=xi.dtype)

        return jax.lax.cond(ok, _do, _skip, xi)

    finite_current = jnp.isfinite(logp) & jnp.isfinite(logl)

    def _init_approx(_):
        return jax.vmap(_approx_or_neginf, in_axes=(0, 0), out_axes=0)(
            x,
            finite_current,
        )

    def _zero_approx(_):
        return jnp.zeros_like(logl)

    logl_approx0 = jax.lax.cond(
        jnp.asarray(use_delayed_acceptance),
        _init_approx,
        _zero_approx,
        operand=None,
    )

    carry0 = (
        key, u, x, theta, logdetj, logdetj_flow, logl, logl_approx0, logp, blobs,
        mu0, sigma0, logp2_best0, cnt0, i0, calls0, accept0, done0,
    )

    max_sigma_cap = jnp.minimum(
        jnp.asarray(2.38, dtype=dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=dtype)),
        jnp.asarray(0.99, dtype=dtype),
    )

    def cond_fn(carry):
        (_, _, _, _, _, _, _, _, _, _, _, _, _, _, i, _, _, done) = carry
        return (i < jnp.asarray(n_max, dtype=i.dtype)) & (~done)

    def body_fn(carry):
        (
            key, u, x, theta, logdetj, logdetj_flow, logl, logl_approx, logp, blobs,
            mu, sigma, logp2_best, cnt, i, calls, accept, done,
        ) = carry

        i1 = i + jnp.asarray(1, dtype=i.dtype)
        key, k_prop, k_unif = jax.random.split(key, 3)

        theta_prime = _dili_li_prior_proposal(
            k_prop,
            theta,
            mu,
            dili_basis,
            dili_post_var,
            sigma,
            dili_lis_scale=dili_lis_scale,
            dili_cs_scale=dili_cs_scale,
        )

        def _t2u_single(ti: Array) -> Tuple[Array, Array]:
            return _flow_theta_to_u_jax(flow, ti, condition)

        u_prime, logdetj_flow_prime = jax.vmap(
            _t2u_single,
            in_axes=0,
            out_axes=(0, 0),
        )(theta_prime)

        x_prime, logdetj_prime = inverse_jax(u_prime, scaler_cfg, scaler_masks)

        x_prime_bc = apply_boundary_conditions_x_jax(x_prime, dict(scaler_cfg))
        u_prime_bc = forward_jax(x_prime_bc, scaler_cfg, scaler_masks)
        x_prime, logdetj_prime = inverse_jax(u_prime_bc, scaler_cfg, scaler_masks)
        u_prime = u_prime_bc

        finite0 = jnp.isfinite(logdetj_prime) & jnp.all(jnp.isfinite(x_prime), axis=1)

        logp_prime = jax.vmap(_prior_or_neginf, in_axes=(0, 0), out_axes=0)(
            x_prime,
            finite0,
        )
        finite1 = finite0 & jnp.isfinite(logp_prime)

        logl_prime, blobs_prime = jax.vmap(
            _like_or_neginf,
            in_axes=(0, 0),
            out_axes=(0, 0),
        )(x_prime, finite1)

        calls = calls + jnp.sum(finite1.astype(jnp.int32), dtype=jnp.int32)

        def _eval_approx_prime(_):
            return jax.vmap(_approx_or_neginf, in_axes=(0, 0), out_axes=0)(
                x_prime,
                finite1,
            )

        def _zero_approx_prime(_):
            return jnp.zeros_like(logl_prime)

        logl_approx_prime = jax.lax.cond(
            jnp.asarray(use_delayed_acceptance),
            _eval_approx_prime,
            _zero_approx_prime,
            operand=None,
        )

        log_ref = _standard_normal_log_reference(theta, mu)
        log_ref_prime = _standard_normal_log_reference(theta_prime, mu)


        shared_terms = (
            (logp_prime - logp)
            + (logdetj_prime - logdetj)
            + (logdetj_flow_prime - logdetj_flow)
            + (log_ref - log_ref_prime)
        )

        log_ratio_full = beta * (logl_prime - logl) + shared_terms
        log_ratio_surrogate = beta * (logl_approx_prime - logl_approx) + shared_terms

        def _mh_accept(_):
            log_alpha = log_ratio_full
            alpha = jnp.exp(jnp.minimum(jnp.asarray(0.0, dtype=dtype), log_alpha))
            alpha = jnp.where(jnp.isnan(alpha), jnp.asarray(0.0, dtype=dtype), alpha)

            u_rand = jax.random.uniform(k_unif, shape=(n_walkers,), dtype=dtype)
            accept_mask = u_rand < alpha
            accept_value = jnp.mean(alpha)
            return accept_mask, accept_value

        def _da_accept(_):
            da = conservative_damh_step_jax(
                key=k_unif,
                new_particles=theta_prime,
                old_particles=theta,
                cov=dili_cov_ref,
                log_ratio_surrogate=log_ratio_surrogate,
                log_ratio_full=log_ratio_full,
                c_const=da_c_const,
                d_const=da_d_const,
            )
            return da.accept, jnp.mean(da.prob_accept)

        accept_mask, accept_value = jax.lax.cond(
            jnp.asarray(use_delayed_acceptance),
            _da_accept,
            _mh_accept,
            operand=None,
        )

        theta = jnp.where(accept_mask[:, None], theta_prime, theta)
        u = jnp.where(accept_mask[:, None], u_prime, u)
        x = jnp.where(accept_mask[:, None], x_prime, x)

        logdetj = jnp.where(accept_mask, logdetj_prime, logdetj)
        logdetj_flow = jnp.where(accept_mask, logdetj_flow_prime, logdetj_flow)
        logl = jnp.where(accept_mask, logl_prime, logl)
        logl_approx = jnp.where(accept_mask, logl_approx_prime, logl_approx)
        logp = jnp.where(accept_mask, logp_prime, logp)

        blobs = jnp.where(
            accept_mask.reshape((n_walkers,) + (1,) * (blobs.ndim - 1)),
            blobs_prime,
            blobs,
        )

        accept = accept_value

        step = jnp.asarray(1.0, dtype=dtype) / jnp.power(
            jnp.asarray(i1 + 1, dtype=dtype),
            jnp.asarray(0.75, dtype=dtype),
        )
        sigma = sigma + step * (accept - jnp.asarray(0.234, dtype=dtype))
        sigma = jnp.abs(jnp.minimum(sigma, max_sigma_cap))

        mu_step = jnp.asarray(1.0, dtype=dtype) / jnp.asarray(i1 + 1, dtype=dtype)
        mu = mu + mu_step * (jnp.mean(theta, axis=0) - mu)

        logp2_new = jnp.mean(logl + logp)
        improved = logp2_new > logp2_best
        cnt = jnp.where(improved, jnp.asarray(0, dtype=cnt.dtype), cnt + 1)
        logp2_best = jnp.where(improved, logp2_new, logp2_best)

        thresh = jnp.asarray(n_steps, dtype=dtype) * jnp.power(
            (jnp.asarray(2.38, dtype=dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=dtype))) / sigma,
            jnp.asarray(2.0, dtype=dtype),
        )
        done = cnt.astype(dtype) >= thresh

        return (
            key, u, x, theta, logdetj, logdetj_flow, logl, logl_approx, logp, blobs,
            mu, sigma, logp2_best, cnt, i1, calls, accept, done,
        )

    carry_f = jax.lax.while_loop(cond_fn, body_fn, carry0)

    (
        key, u, x, theta, logdetj, logdetj_flow, logl, logl_approx, logp, blobs,
        mu, sigma, logp2_best, cnt, i, calls, accept, done,
    ) = carry_f

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