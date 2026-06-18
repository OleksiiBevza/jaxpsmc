from __future__ import annotations

from typing import Callable, Mapping, Tuple, Any, Optional, Dict

import jax
import jax.numpy as jnp

from ..scaler_jax import (
    apply_boundary_conditions_x_jax,
    forward_jax,
    inverse_jax,
)
from ..delayed_acceptance.da_conservative_damh_jax import (
    conservative_damh_step_jax,
)
from .flow_jax import _flow_u_to_theta_jax, _flow_theta_to_u_jax

Array = jax.Array


def _standard_normal_log_reference(theta: Array, center: Array) -> Array:
    """
    Computes standard Gaussian reference log density.
    The constant normalization term is not included.

    Parameters
    ----------
    theta:
        Particle positions in theta-space, shape (N, D).
    center:
        Reference center, shape (D,).

    Returns
    -------
    Array:
        Log reference density for each particle, shape (N,).
    """
    # shift particles so that Gaussian reference is centered at `center`
    diff = theta - center[None, :]
    # return -0.5 * ||theta - center||^2 for each particle
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
    Builds one DILI LI-prior proposal in theta-space.

    The proposal splits current particles into two parts:
        * one part inside likelihood-informed subspace (LIS),
        * one part in the complement space.

    Inside the LIS, each direction uses its estimated posterior variance.
    Outside the LIS, the proposal uses a scalar pCN-style move.

    Parameters
    ----------
    key:
        JAX random key.
    theta:
        Current particles in theta-space, shape (N, D).
    center:
        Center of the proposal/reference distribution, shape (D,).
    basis:
        Orthonormal LIS basis, shape (D, r).
    post_var:
        Posterior variance estimates inside the LIS, shape (r,).
    sigma:
        Current adaptive proposal scale, scalar.
    dili_lis_scale:
        Extra multiplier for the LIS proposal scale.
    dili_cs_scale:
        Extra multiplier for the complement-space proposal scale.

    Returns
    -------
    Array:
        Proposed particles in theta-space, shape (N, D).
    """
    theta = jnp.asarray(theta)
    center = jnp.asarray(center)
    basis = jnp.asarray(basis)
    post_var = jnp.asarray(post_var)

    dtype = theta.dtype

    # build separate proposal scales for LIS and its complement
    # scale is clipped to keep the CN coefficients numerically stable
    sigma_lis = jnp.clip(
        jnp.abs(sigma * jnp.asarray(dili_lis_scale, dtype=dtype)), 1e-12, 0.99
    )
    sigma_cs = jnp.clip(
        jnp.abs(sigma * jnp.asarray(dili_cs_scale, dtype=dtype)), 1e-12, 0.99
    )

    # convert proposal scales to CN time-step parameters
    tau_lis = sigma_lis * sigma_lis
    tau_cs = sigma_cs * sigma_cs

    # CN coefficient inside each LIS direction
    # large posterior variance produces a different step size per direction
    # LI-Prior / CN coefficients in the LIS
    # a_i = (2 - tau * D_i) / (2 + tau * D_i)
    # b_i = sqrt(1 - a_i^2)
    post_var = jnp.maximum(post_var, jnp.asarray(1e-12, dtype=dtype))
    a_lis = (2.0 - tau_lis * post_var) / (2.0 + tau_lis * post_var)
    a_lis = jnp.clip(a_lis, -0.999999, 0.999999)
    # noise coefficient inside LIS
    b_lis = jnp.sqrt(jnp.maximum(1.0 - a_lis * a_lis, 0.0))

    # CN coefficient in complement space
    a_cs = (2.0 - tau_cs) / (2.0 + tau_cs)
    a_cs = jnp.clip(a_cs, -0.999999, 0.999999)
    # noise coefficient in complement space
    b_cs = jnp.sqrt(jnp.maximum(1.0 - a_cs * a_cs, 0.0))
    # use separate random keys for LIS noise and full-space noise
    key_lis, key_full = jax.random.split(key)
    # work with centered particles
    diff = theta - center[None, :]

    # project particles into the LIS
    z_lis = diff @ basis  # (N, r)
    # draw independent Gaussian noise in the LIS
    eps_lis = jax.random.normal(key_lis, shape=z_lis.shape, dtype=dtype)
    # apply CN update in LIS coordinates
    z_lis_prime = a_lis[None, :] * z_lis + b_lis[None, :] * eps_lis
    # map proposed LIS coordinates back to theta-space
    diff_lis_prime = z_lis_prime @ basis.T

    # compute current LIS component in theta-space
    diff_lis = z_lis @ basis.T
    # complement is after subtracting LIS component
    diff_cs = diff - diff_lis

    # draw full-space noise, then remove its LIS projection
    # so noise left only in the complement space
    eps_full = jax.random.normal(key_full, shape=theta.shape, dtype=dtype)
    eps_lis_part = (eps_full @ basis) @ basis.T
    eps_cs = eps_full - eps_lis_part

    # apply CN update in complement space
    diff_cs_prime = a_cs * diff_cs + b_cs * eps_cs

    return center[None, :] + diff_lis_prime + diff_cs_prime


def _li_log_reference(theta: Array, mu: Array, eigvecs: Array, var_dir: Array) -> Array:
    """
    Computes a Gaussian reference log density in an eigenbasis

    This helper is for an empirical LI-pCN reference density
    The normalization constant is omitted because it cancels in MH ratios

    Parameters
    ----------
    theta:
        Particle positions, shape (N, D).
    mu:
        Reference mean, shape (D,).
    eigvecs:
        Eigenvector matrix, shape (D, D).
    var_dir:
        Variance for each eigen-direction, shape (D,).

    Returns
    -------
    Array:
        Log reference density for each particle, shape (N,).
    """
    diff = theta - mu[None, :]
    # express the shift in the eigenbasis
    z = diff @ eigvecs
    # compute diagonal Gaussian quadratic form
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
    Builds a direction-wise empirical LI-pCN proposal.

    Active directions use the LIS proposal scale.
    Inactive directions use the complement-space proposal scale.

    Parameters
    ----------
    key:
        JAX random key.
    theta:
        Current particles, shape (N, D).
    mu:
        Reference mean, shape (D,).
    eigvecs:
        Eigenvectors defining the proposal coordinates, shape (D, D).
    var_dir:
        Variance for each proposal direction, shape (D,).
    active:
        Boolean mask selecting active LIS directions, shape (D,).
    sigma:
        Current adaptive proposal scale, scalar.
    li_lis_scale:
        Extra multiplier for active directions.
    li_cs_scale:
        Extra multiplier for inactive directions.

    Returns
    -------
    Array:
        Proposed particles, shape (N, D).
    """
    dtype = theta.dtype

    lis_scale = jnp.asarray(li_lis_scale, dtype=dtype)
    cs_scale = jnp.asarray(li_cs_scale, dtype=dtype)
    # use separate scales for active and inactive directions
    sigma_lis = jnp.clip(jnp.abs(sigma * lis_scale), 1e-12, 0.99)
    sigma_cs = jnp.clip(jnp.abs(sigma * cs_scale), 1e-12, 0.99)
    # pick scale per eigen direction
    sigma_dir = jnp.where(active, sigma_lis, sigma_cs)
    # standard pCN autoregressive coefficient
    a_dir = jnp.sqrt(jnp.maximum(1.0 - sigma_dir * sigma_dir, 0.0))
    # move particles to centered eigen coordinates
    diff = theta - mu[None, :]
    z = diff @ eigvecs
    # draw independent Gaussian noise in eigen coordinates
    eps = jax.random.normal(key, shape=z.shape, dtype=dtype)
    # apply directionwise pCN update
    z_prime = a_dir[None, :] * z + sigma_dir[None, :] * jnp.sqrt(var_dir)[None, :] * eps
    # map back to theta space
    theta_prime = mu[None, :] + z_prime @ eigvecs.T
    return theta_prime


def dili_pcn_jax(
    key: Array,
    *,
    # current particle state
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
    # mutation loop options
    n_max: int,
    n_steps: int,
    proposal_scale: Array,
    dili_lis_scale: float = 1.0,
    dili_cs_scale: float = 1.0,
    # delayed-acceptance controls
    use_delayed_acceptance: Array = jnp.asarray(False),
    da_c_const: Array = jnp.asarray(0.01),
    da_d_const: Array = jnp.asarray(2.0),
    condition: Optional[Array] = None,
) -> Dict[str, Array]:
    """
    Runs a Hessian/GNH-based DILI-pCN mutation kernel.

    The kernel proposes particles in theta-space using a DILI-style split:
    the likelihood-informed subspace uses the fitted DILI basis and
    posterior variances, while the complement uses a pCN-style move.

    The proposal is then mapped back through the flow and scaler.
    A Metropolis-Hastings correction accepts or rejects each proposed particle.

    If delayed acceptance is enabled, the kernel first uses a surrogate
    likelihood ratio and then applies the conservative DA-MH correction.

    Parameters
    ----------
    key:
        JAX random key.
    u:
        Current particles in unconstrained sampler space, shape (N, D).
    x:
        Current particles in physical/constrained space, shape (N, D).
    logdetj:
        Current scaler log-Jacobian values, shape (N,).
    logl:
        Current exact log-likelihood values, shape (N,).
    logp:
        Current log-prior values, shape (N,).
    logdetj_flow:
        Current flow log-Jacobian values, shape (N,).
    blobs:
        Extra likelihood outputs stored per particle.
    beta:
        Tempering parameter.
    loglike_fn:
        Exact log-likelihood function. It returns ``(loglike, blob)``.
    loglike_approx_fn:
        Approximate log-likelihood used only when delayed acceptance is enabled.
    logprior_fn:
        Log-prior function evaluated in x-space.
    flow:
        Normalizing flow object or flow parameters used by flow helpers.
    scaler_cfg:
        Scaler configuration dictionary.
    scaler_masks:
        Scaler mask dictionary.
    dili_center:
        Center of the DILI geometry in theta-space, shape (D,).
    dili_basis:
        Orthonormal DILI basis, shape (D, r).
    dili_post_var:
        Posterior variance estimates in the DILI basis, shape (r,).
    dili_cov_ref:
        Reference covariance used by conservative delayed acceptance, shape (D, D).
    n_max:
        Maximum number of mutation-loop iterations.
    n_steps:
        Baseline number of mutation steps used in the stopping rule.
    proposal_scale:
        Initial adaptive proposal scale.
    dili_lis_scale:
        Extra scale multiplier for the LIS move.
    dili_cs_scale:
        Extra scale multiplier for the complement-space move.
    use_delayed_acceptance:
        If True, use conservative delayed acceptance.
    da_c_const:
        Conservative DA constant c.
    da_d_const:
        Conservative DA constant d.
    condition:
        Optional conditioning input passed to flow helpers.

    Returns
    -------
    Dict[str, Array]:
        Updated state dictionary containing particles, log densities,
        blobs, proposal scale, acceptance estimate, step count, and call count.
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

    # read particle count and dimension
    n_walkers, n_dim = u.shape
    dtype = u.dtype

    def _u2t_single(ui: Array) -> Tuple[Array, Array]:
        """
        Maps one particle from u-space to theta-space.

        Parameters
        ----------
        ui:
            One particle in unconstrained sampler space, shape ``(D,)``.

        Returns
        -------
        Tuple[Array, Array]:
            Theta-space particle and flow log-Jacobian value.
        """
        # map one particle from u space to theta space
        return _flow_u_to_theta_jax(flow, ui, condition)

    # convert all current particles from u space to theta space
    theta, logdetj_flow0 = jax.vmap(
        _u2t_single,
        in_axes=0,
        out_axes=(0, 0),
    )(u)

    logdetj_flow = logdetj_flow0

    # initialize adaptive proposal state
    mu0 = dili_center
    sigma0 = jnp.minimum(jnp.abs(proposal_scale), jnp.asarray(0.99, dtype=dtype))
    # track best average untempered target value for stopping
    logp2_best0 = jnp.mean(logl + logp)
    # initialize loop counters and diagnostics
    cnt0 = jnp.asarray(0, dtype=jnp.int32)
    i0 = jnp.asarray(0, dtype=jnp.int32)
    calls0 = jnp.asarray(0, dtype=jnp.int32)
    accept0 = jnp.asarray(0.0, dtype=dtype)
    done0 = jnp.asarray(False)
    # template blob used when a likelihood evaluation is skipped
    blob_template = jnp.zeros_like(blobs[0])

    def _prior_or_neginf(xi: Array, ok: Array) -> Array:
        """
        Evaluates prior for one particle, or returns ``-inf``.

        Prior is evaluated only when ``ok`` is true. If ``ok`` is false,
        the function returns ``-inf`` so proposed particle cannot be accepted.

        Parameters
        ----------
        xi:
            One proposed particle in x-space, shape ``(D,)``.
        ok:
            Boolean scalar showing whether the particle is valid.

        Returns
        -------
        Array:
            Log-prior value, or ``-inf``.
        """
        # evaluate prior only when proposed point is valid
        # Otherwise return -inf so particle cannot be accepted
        return jax.lax.cond(
            ok,
            lambda z: logprior_fn(z),
            lambda z: jnp.asarray(-jnp.inf, dtype=xi.dtype),
            xi,
        )

    def _like_or_neginf(xi: Array, ok: Array) -> Tuple[Array, Array]:
        """
        Evaluates exact likelihood for one particle, or returns ``-inf``.

        Exact likelihood is evaluated only for valid proposed particles.
        Invalid particles receive ``-inf`` likelihood and a zero-like blob.

        Parameters
        ----------
        xi:
            One proposed particle in x-space, shape ``(D,)``.
        ok:
            Boolean scalar showing whether the likelihood should be evaluated.

        Returns
        -------
        Tuple[Array, Array]:
            Exact log-likelihood value and blob output.
        """

        # evaluate exact likelihood only when proposed point is valid
        def _do(z: Array) -> Tuple[Array, Array]:
            """
            Runs exact likelihood function for one valid particle.

            Parameters
            ----------
            z:
                One valid particle in x-space.

            Returns
            -------
            Tuple[Array, Array]:
                Exact log-likelihood value and blob output.
            """
            ll, bb = loglike_fn(z)
            return ll, bb

        # if invalid, return -inf likelihood and an empty blob
        def _skip(z: Array) -> Tuple[Array, Array]:
            """
            Returns fallback likelihood output for an invalid particle.

            Parameters
            ----------
            z:
                Unused invalid particle.

            Returns
            -------
            Tuple[Array, Array]:
                ``-inf`` likelihood and a zero-like blob.
            """
            return jnp.asarray(-jnp.inf, dtype=xi.dtype), blob_template

        return jax.lax.cond(ok, _do, _skip, xi)

    def _approx_or_neginf(xi: Array, ok: Array) -> Array:
        """
        Evaluates approximate likelihood for one particle, or returns ``-inf``.

        This is used for delayed acceptance. It keeps invalid particles out
        of the surrogate likelihood ratio.

        Parameters
        ----------
        xi:
            One proposed particle in x-space, shape ``(D,)``.
        ok:
            Boolean scalar showing whether approximate likelihood should be
            evaluated.

        Returns
        -------
        Array:
            Approximate log-likelihood value, or ``-inf``.
        """

        # evaluate approximate likelihood only when needed and valid
        def _do(z: Array) -> Array:
            """
            Runs approximate likelihood function for one valid particle.

            Parameters
            ----------
            z:
                One valid particle in x-space.

            Returns
            -------
            Array:
                Approximate log-likelihood value.
            """
            return jnp.asarray(loglike_approx_fn(z), dtype=xi.dtype)

        def _skip(z: Array) -> Array:
            """
            Returns fallback approximate likelihood for an invalid particle.

            Parameters
            ----------
            z:
                Unused invalid particle.

            Returns
            -------
            Array:
                ``-inf`` approximate likelihood.
            """
            return jnp.asarray(-jnp.inf, dtype=xi.dtype)

        return jax.lax.cond(ok, _do, _skip, xi)

    # current particles are valid only if both prior and likelihood are finite
    finite_current = jnp.isfinite(logp) & jnp.isfinite(logl)

    def _init_approx(_):
        """
        Initializes surrogate likelihood values for current particles.
        Branch is used when delayed acceptance is enabled.

        Parameters
        ----------
        _:
            Unused operand required by ``jax.lax.cond``.

        Returns
        -------
        Array:
            Approximate log-likelihood values for current particles, shape ``(N,)``.
        """
        # initialize surrogate likelihood values for current particles
        return jax.vmap(_approx_or_neginf, in_axes=(0, 0), out_axes=0)(
            x,
            finite_current,
        )

    def _zero_approx(_):
        """
        Returns dummy surrogate likelihood values.
        Branch is used when delayed acceptance is disabled.

        Parameters
        ----------
        _:
            Unused operand required by ``jax.lax.cond``.

        Returns
        -------
        Array:
            Zero array with the same shape as ``logl``.
        """
        # if delayed acceptance is disabled, these values are unused
        return jnp.zeros_like(logl)

    # store current approximate likelihood values for DA
    logl_approx0 = jax.lax.cond(
        jnp.asarray(use_delayed_acceptance),
        _init_approx,
        _zero_approx,
        operand=None,
    )

    # carry object for lax.while_loop
    carry0 = (
        key,
        u,
        x,
        theta,
        logdetj,
        logdetj_flow,
        logl,
        logl_approx0,
        logp,
        blobs,
        mu0,
        sigma0,
        logp2_best0,
        cnt0,
        i0,
        calls0,
        accept0,
        done0,
    )

    # adaptive proposal scale: 2.38 / sqrt(D)
    max_sigma_cap = jnp.minimum(
        jnp.asarray(2.38, dtype=dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=dtype)),
        jnp.asarray(0.99, dtype=dtype),
    )

    def cond_fn(carry):
        """
        Checks whether the mutation loop should continue.

        The loop continues until either ``n_max`` is reached or the adaptive
        stopping rule marks the loop as done.

        Parameters
        ----------
        carry:
            Current ``jax.lax.while_loop`` state.

        Returns
        -------
        Array:
            Boolean scalar. True means another mutation step should run.
        """
        # continue while iterating
        (_, _, _, _, _, _, _, _, _, _, _, _, _, _, i, _, _, done) = carry
        return (i < jnp.asarray(n_max, dtype=i.dtype)) & (~done)

    def body_fn(carry):
        """
        Performs one DILI-pCN mutation-loop iteration.
        Function proposes new theta-space particles, maps them back to x-space,
        evaluates prior and likelihood values, performs MH or delayed-acceptance
        correction, and updates adaptation diagnostics.

        Parameters
        ----------
        carry:
            Current ``jax.lax.while_loop`` state.

        Returns
        -------
        tuple:
            Updated loop state.
        """
        # unpack current mutation loop state
        (
            key,
            u,
            x,
            theta,
            logdetj,
            logdetj_flow,
            logl,
            logl_approx,
            logp,
            blobs,
            mu,
            sigma,
            logp2_best,
            cnt,
            i,
            calls,
            accept,
            done,
        ) = carry

        # iteration counter
        i1 = i + jnp.asarray(1, dtype=i.dtype)
        key, k_prop, k_unif = jax.random.split(key, 3)

        # propose new particles in theta space using DILI LI-prior move
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
            """
            Maps one particle from theta-space back to u-space.

            Parameters
            ----------
            ti:
                One proposed particle in theta-space, shape ``(D,)``.

            Returns
            -------
            Tuple[Array, Array]:
                U-space particle and flow log-Jacobian value.
            """
            # map one particle from theta space back to u space
            return _flow_theta_to_u_jax(flow, ti, condition)

        # convert proposed theta particles back to u space
        u_prime, logdetj_flow_prime = jax.vmap(
            _t2u_single,
            in_axes=0,
            out_axes=(0, 0),
        )(theta_prime)

        # convert proposed u particles to physical x space
        x_prime, logdetj_prime = inverse_jax(u_prime, scaler_cfg, scaler_masks)

        # apply periodic/reflective boundary conditions in x space
        x_prime_bc = apply_boundary_conditions_x_jax(x_prime, dict(scaler_cfg))
        # recompute consistent u and x values after boundary correction
        u_prime_bc = forward_jax(x_prime_bc, scaler_cfg, scaler_masks)
        x_prime, logdetj_prime = inverse_jax(u_prime_bc, scaler_cfg, scaler_masks)
        u_prime = u_prime_bc

        # check if proposed x values and scaler log-Jacobians are finite
        finite0 = jnp.isfinite(logdetj_prime) & jnp.all(jnp.isfinite(x_prime), axis=1)

        # evaluate prior only for finite proposed points
        logp_prime = jax.vmap(_prior_or_neginf, in_axes=(0, 0), out_axes=0)(
            x_prime,
            finite0,
        )

        # proposal is valid for likelihood evaluation only if its prior is finite
        finite1 = finite0 & jnp.isfinite(logp_prime)

        # evaluate exact likelihood for valid proposed points
        logl_prime, blobs_prime = jax.vmap(
            _like_or_neginf,
            in_axes=(0, 0),
            out_axes=(0, 0),
        )(x_prime, finite1)

        # count exact likelihood evaluations
        calls = calls + jnp.sum(finite1.astype(jnp.int32), dtype=jnp.int32)

        def _eval_approx_prime(_):
            """
            Evaluates approximate likelihoods for proposed particles.

            This branch is used when delayed acceptance is enabled.

            Parameters
            ----------
            _:
                Unused operand required by ``jax.lax.cond``.

            Returns
            -------
            Array:
                Approximate log-likelihood values for proposed particles, shape ``(N,)``.
            """
            # evaluate approximate likelihood for proposed particles
            return jax.vmap(_approx_or_neginf, in_axes=(0, 0), out_axes=0)(
                x_prime,
                finite1,
            )

        def _zero_approx_prime(_):
            """
            Returns dummy approximate likelihoods for proposed particles.
            Branch is used when delayed acceptance is disabled.

            Parameters
            ----------
            _:
                Unused operand required by ``jax.lax.cond``.

            Returns
            -------
            Array:
                Zero array with the same shape as ``logl_prime``.
            """
            # if DA is disabled, approximate likelihood values are unused
            return jnp.zeros_like(logl_prime)

        # proposed approximate likelihood values for DA
        logl_approx_prime = jax.lax.cond(
            jnp.asarray(use_delayed_acceptance),
            _eval_approx_prime,
            _zero_approx_prime,
            operand=None,
        )

        # compute Gaussian reference density correction for asymmetry
        log_ref = _standard_normal_log_reference(theta, mu)
        log_ref_prime = _standard_normal_log_reference(theta_prime, mu)

        # terms shared by full and surrogate MH ratios
        shared_terms = (
            (logp_prime - logp)
            + (logdetj_prime - logdetj)
            + (logdetj_flow_prime - logdetj_flow)
            + (log_ref - log_ref_prime)
        )

        # full MH ratio uses exact likelihood
        log_ratio_full = beta * (logl_prime - logl) + shared_terms
        # surrogate ratio uses approximate likelihood
        log_ratio_surrogate = beta * (logl_approx_prime - logl_approx) + shared_terms

        def _mh_accept(_):
            """
            Performs standard Metropolis-Hastings acceptance step.

            Parameters
            ----------
            _:
                Unused operand required by ``jax.lax.cond``.

            Returns
            -------
            Tuple[Array, Array]:
                Boolean acceptance mask, shape ``(N,)``, and mean acceptance probability.
            """
            # standard Metropolis-Hastings acceptance step
            log_alpha = log_ratio_full
            alpha = jnp.exp(jnp.minimum(jnp.asarray(0.0, dtype=dtype), log_alpha))
            # treat NaN acceptance probabilities as rejection
            alpha = jnp.where(jnp.isnan(alpha), jnp.asarray(0.0, dtype=dtype), alpha)
            # draw one uniform random number per walker
            u_rand = jax.random.uniform(k_unif, shape=(n_walkers,), dtype=dtype)
            accept_mask = u_rand < alpha
            # store mean acceptance probability as the adaptation signal
            accept_value = jnp.mean(alpha)
            return accept_mask, accept_value

        def _da_accept(_):
            """
            Performs conservative delayed-acceptance MH step.

            This branch uses surrogate ratio, the full ratio, and the
            covariance-based proposal-distance correction.

            Parameters
            ----------
            _:
                Unused operand required by ``jax.lax.cond``.

            Returns
            -------
            Tuple[Array, Array]:
                Boolean acceptance mask, shape ``(N,)``, and mean delayed-acceptance
                probability.
            """
            # conservative delayed-acceptance MH step
            # uses both surrogate and full ratios plus proposal-distance control
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

        # choose standard MH or delayed acceptance
        accept_mask, accept_value = jax.lax.cond(
            jnp.asarray(use_delayed_acceptance),
            _da_accept,
            _mh_accept,
            operand=None,
        )

        # keep proposed theta/u/x only for accepted particles
        theta = jnp.where(accept_mask[:, None], theta_prime, theta)
        u = jnp.where(accept_mask[:, None], u_prime, u)
        x = jnp.where(accept_mask[:, None], x_prime, x)

        # update scalar particle quantities only where accepted
        logdetj = jnp.where(accept_mask, logdetj_prime, logdetj)
        logdetj_flow = jnp.where(accept_mask, logdetj_flow_prime, logdetj_flow)
        logl = jnp.where(accept_mask, logl_prime, logl)
        logl_approx = jnp.where(accept_mask, logl_approx_prime, logl_approx)
        logp = jnp.where(accept_mask, logp_prime, logp)

        # update blobs only where accepted
        blobs = jnp.where(
            accept_mask.reshape((n_walkers,) + (1,) * (blobs.ndim - 1)),
            blobs_prime,
            blobs,
        )

        # save current mean acceptance probability
        accept = accept_value

        # Robbins-Monro style adaptation toward target acceptance 0.234
        step = jnp.asarray(1.0, dtype=dtype) / jnp.power(
            jnp.asarray(i1 + 1, dtype=dtype),
            jnp.asarray(0.75, dtype=dtype),
        )
        sigma = sigma + step * (accept - jnp.asarray(0.234, dtype=dtype))
        sigma = jnp.abs(jnp.minimum(sigma, max_sigma_cap))
        # adapt proposal center toward current particle mean
        mu_step = jnp.asarray(1.0, dtype=dtype) / jnp.asarray(i1 + 1, dtype=dtype)
        mu = mu + mu_step * (jnp.mean(theta, axis=0) - mu)

        # update simple improvement based stopping diagnostic
        logp2_new = jnp.mean(logl + logp)
        improved = logp2_new > logp2_best
        cnt = jnp.where(improved, jnp.asarray(0, dtype=cnt.dtype), cnt + 1)
        logp2_best = jnp.where(improved, logp2_new, logp2_best)

        # stop earlier if chain has not improved
        thresh = jnp.asarray(n_steps, dtype=dtype) * jnp.power(
            (jnp.asarray(2.38, dtype=dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=dtype)))
            / sigma,
            jnp.asarray(2.0, dtype=dtype),
        )
        done = cnt.astype(dtype) >= thresh

        # return updated loop state
        return (
            key,
            u,
            x,
            theta,
            logdetj,
            logdetj_flow,
            logl,
            logl_approx,
            logp,
            blobs,
            mu,
            sigma,
            logp2_best,
            cnt,
            i1,
            calls,
            accept,
            done,
        )

    # run mutation loop using JAX control flow
    carry_f = jax.lax.while_loop(cond_fn, body_fn, carry0)

    # unpack final loop state
    (
        key,
        u,
        x,
        theta,
        logdetj,
        logdetj_flow,
        logl,
        logl_approx,
        logp,
        blobs,
        mu,
        sigma,
        logp2_best,
        cnt,
        i,
        calls,
        accept,
        done,
    ) = carry_f

    # return updated sampler state and diagnostics
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
