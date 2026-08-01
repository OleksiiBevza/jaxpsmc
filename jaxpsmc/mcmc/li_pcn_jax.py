from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from ..delayed_acceptance.da_conservative_damh_jax import (
    conservative_damh_step_jax,
)
from ..scaler_jax import (
    apply_boundary_conditions_x_jax,
    forward_jax,
    inverse_jax,
)
from .flow_jax import _flow_theta_to_u_jax, _flow_u_to_theta_jax

Array = jax.Array


def _empirical_li_geometry_from_cov(
    cov: Array,
    *,
    li_rank: int,
    li_var_floor: float,
    li_complement_var: float,
) -> tuple[Array, Array, Array, Array]:
    """
    Builds empirical likelihood-informed geometry from a covariance matrix.

    The function uses the eigenvectors of the covariance matrix to define
    approximate likelihood-informed directions.

    The first li_rank directions are treated as likelihood-informed
    directions. The remaining directions are treated as complement-space
    directions and receive the same reference variance.

    A small variance floor is added for numerical stability.

    Parameters:
    -----------
    cov:
        empirical covariance matrix, shape (D, D).
    li_rank:
        number of likelihood-informed directions to use.
    li_var_floor:
        smallest allowed variance value.
        This prevents zero or negative variances.
    li_complement_var:
        reference variance used for directions outside the
        likelihood-informed subspace.

    Returns:
    --------
    Tuple[Array, Array, Array, Array]:
        eigvecs:
            eigenvectors stored as columns, shape (D, D).
            They are sorted from largest to smallest eigenvalue.
        var_dir:
            reference variance for each eigendirection, shape (D,).
        active:
            Boolean mask showing which directions are likelihood-informed,
            shape (D,).
        cov_ref:
            full reference covariance matrix, shape (D, D).
            It is used for diagnostics and delayed-acceptance distance
            calculations.
    """
    cov = jnp.asarray(cov)
    d = cov.shape[0]
    dtype = cov.dtype

    floor = jnp.asarray(li_var_floor, dtype=dtype)
    comp_var = jnp.maximum(jnp.asarray(li_complement_var, dtype=dtype), floor)

    cov = 0.5 * (cov + cov.T)
    cov = cov + floor * jnp.eye(d, dtype=dtype)

    eigvals, eigvecs = jnp.linalg.eigh(cov)

    # Descending order: largest empirical covariance directions first.
    order = jnp.argsort(eigvals)[::-1]
    eigvals = jnp.take(eigvals, order, axis=0)
    eigvecs = jnp.take(eigvecs, order, axis=1)

    eigvals = jnp.maximum(eigvals, floor)

    rank = jnp.clip(jnp.asarray(li_rank, dtype=jnp.int32), 0, d)
    active = jnp.arange(d, dtype=jnp.int32) < rank

    var_dir = jnp.where(active, eigvals, comp_var)
    var_dir = jnp.maximum(var_dir, floor)

    cov_ref = (eigvecs * var_dir[None, :]) @ eigvecs.T
    cov_ref = 0.5 * (cov_ref + cov_ref.T) + floor * jnp.eye(d, dtype=dtype)

    return eigvecs, var_dir, active, cov_ref


def _li_log_reference(theta: Array, mu: Array, eigvecs: Array, var_dir: Array) -> Array:
    """
    Computes the Gaussian reference log-density up to a constant.

    This function evaluates how far each particle is from the reference
    mean in the likelihood-informed eigenbasis.

    The missing normalizing constant is not included.
    The same fixed geometry is used during one
    mutation call, so the constant cancels in the Metropolis-Hastings ratio.

    Parameters:
    -----------
    theta:
        particles in theta-space, shape (N, D).
    mu:
        reference mean, shape (D,).
    eigvecs:
        eigenvectors of the reference covariance, shape (D, D).
        The eigenvectors are stored as columns.
    var_dir:
        reference variance for each eigendirection, shape (D,).

    Returns:
    --------
    Array:
        reference log-density values up to a constant, shape (N,).
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
    Proposes new particles with the likelihood-informed pCN rule.

    The function works in the eigenvector basis of the empirical
    covariance matrix. In this basis, each direction can use its own
    proposal scale and variance.

    Directions marked as active are treated as likelihood-informed
    directions. The remaining directions are treated as complement-space
    directions.

    Parameters:
    -----------
    key:
        JAX random key used to draw Gaussian noise.
    theta:
        current particles in theta-space, shape (N, D).
    mu:
        reference mean, shape (D,).
    eigvecs:
        eigenvectors of the reference covariance, shape (D, D).
        The eigenvectors are stored as columns.
    var_dir:
        reference variance for each eigendirection, shape (D,).
    active:
        Boolean mask showing which directions are likelihood-informed,
        shape (D,).
    sigma:
        base proposal scale.
    li_lis_scale:
        scale multiplier for likelihood-informed directions.
    li_cs_scale:
        scale multiplier for complement-space directions.

    Returns:
    --------
    Array:
        proposed particles in theta-space, shape (N, D).
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
    z_prime = a_dir[None, :] * z + sigma_dir[None, :] * jnp.sqrt(var_dir)[None, :] * eps

    theta_prime = mu[None, :] + z_prime @ eigvecs.T
    return theta_prime


def likelihood_informed_pcn_jax(
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
    loglike_fn: Callable[[Array], tuple[Array, Array]],
    loglike_approx_fn: Callable[[Array], Array],
    logprior_fn: Callable[[Array], Array],
    flow: Any,
    scaler_cfg: Mapping[str, Array],
    scaler_masks: Mapping[str, Array],
    # empirical LI geometry
    geom_mu: Array,
    geom_cov: Array,
    # options
    n_max: int,
    n_steps: int,
    proposal_scale: Array,
    li_rank: int = 16,
    li_lis_scale: float = 1.0,
    li_cs_scale: float = 1.0,
    li_var_floor: float = 1e-8,
    li_complement_var: float = 1.0,
    use_delayed_acceptance: Array = jnp.asarray(False),
    da_c_const: Array = jnp.asarray(0.01),
    da_d_const: Array = jnp.asarray(2.0),
    condition: Array | None = None,
) -> dict[str, Array]:
    """
    Runs one likelihood-informed pCN mutation step.

    This function moves the current particles with an empirical
    likelihood-informed pCN kernel. The likelihood-informed directions
    are estimated from the empirical covariance matrix.

    This is not full Hessian-based DILI. It is a simpler DILI-inspired
    kernel. It uses covariance eigenvectors as an approximate
    likelihood-informed subspace.

    The function works in theta-space. It proposes new theta values,
    maps them back to u-space and x-space, evaluates the prior and
    likelihood, and then accepts or rejects the proposal.

    If delayed acceptance is enabled, the function uses the approximate
    likelihood before applying the full likelihood correction. If delayed
    acceptance is disabled, it uses the normal Metropolis-Hastings rule.

    Parameters:
    -----------
    key:
        JAX random key used by the mutation kernel.
    u:
        current particles in u-space, shape (N, D).
    x:
        current particles in x-space, shape (N, D).
    logdetj:
        current scaler log-determinant values, shape (N,).
    logl:
        current full log-likelihood values, shape (N,).
    logp:
        current log-prior values, shape (N,).
    logdetj_flow:
        current flow log-determinant values, shape (N,).
    blobs:
        extra likelihood outputs for the current particles.
    beta:
        current SMC annealing value.
    loglike_fn:
        function that evaluates the full log-likelihood for one particle.
        It must return a log-likelihood value and a blob output.
    loglike_approx_fn:
        function that evaluates the approximate log-likelihood
        for one particle. It is used by delayed acceptance.
    logprior_fn:
        function that evaluates the log-prior for one particle.
    flow:
        flow object used to map between u-space and theta-space.
    scaler_cfg:
        scaler configuration used to map between u-space and x-space.
    scaler_masks:
        masks used by the scaler.
    geom_mu:
        empirical mean used as the center of the LI-pCN proposal.
    geom_cov:
        empirical covariance used to build the likelihood-informed basis.
    n_max:
        maximum number of mutation iterations.
    n_steps:
        stopping-rule value used by the mutation loop.
    proposal_scale:
        current proposal scale.
    li_rank:
        number of likelihood-informed directions to use.
    li_lis_scale:
        scale multiplier for likelihood-informed directions.
    li_cs_scale:
        scale multiplier for complement-space directions.
    li_var_floor:
        smallest allowed variance value.
        This prevents numerical problems.
    li_complement_var:
        reference variance used outside the likelihood-informed subspace.
    use_delayed_acceptance:
        Boolean flag saying whether delayed acceptance is used.
    da_c_const:
        clipping constant used by conservative delayed acceptance.
    da_d_const:
        exponent constant used by conservative delayed acceptance.
    condition:
        optional conditioning value passed to the flow.

    Returns:
    --------
    Dict[str, Array]:
        dictionary with the updated mutation state.
        It contains the updated random key, particles, log values,
        blobs, proposal scale, acceptance value, number of steps,
        number of likelihood calls, and efficiency value.
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

    n_walkers, n_dim = u.shape
    dtype = u.dtype

    eigvecs, var_dir, active, cov_ref = _empirical_li_geometry_from_cov(
        geom_cov,
        li_rank=li_rank,
        li_var_floor=li_var_floor,
        li_complement_var=li_complement_var,
    )

    def _u2t_single(ui: Array) -> tuple[Array, Array]:
        """
        Maps one particle from u-space to theta-space.

        Parameters:
        -----------
        ui:
            one particle in u-space, shape (D,).

        Returns:
        --------
        Tuple[Array, Array]:
            particle in theta-space and flow log-determinant.
        """
        return _flow_u_to_theta_jax(flow, ui, condition)

    theta, logdetj_flow0 = jax.vmap(
        _u2t_single,
        in_axes=0,
        out_axes=(0, 0),
    )(u)

    logdetj_flow = logdetj_flow0

    mu0 = geom_mu
    sigma0 = jnp.minimum(jnp.abs(proposal_scale), jnp.asarray(0.99, dtype=dtype))
    logp2_best0 = jnp.mean(logl + logp)
    cnt0 = jnp.asarray(0, dtype=jnp.int32)
    i0 = jnp.asarray(0, dtype=jnp.int32)
    calls0 = jnp.asarray(0, dtype=jnp.int32)
    accept0 = jnp.asarray(0.0, dtype=dtype)
    done0 = jnp.asarray(False)

    blob_template = jnp.zeros_like(blobs[0])

    def _prior_or_neginf(xi: Array, ok: Array) -> Array:
        """
        Evaluates the prior only when the particle is valid.

        If the particle is not valid, the function returns -inf.
        This prevents invalid proposals from being accepted.

        Parameters:
        -----------
        xi:
            one particle in x-space, shape (D,).
        ok:
            Boolean flag saying whether the particle is valid.

        Returns:
        --------
        Array:
            log-prior value, or -inf for an invalid particle.
        """
        return jax.lax.cond(
            ok,
            lambda z: logprior_fn(z),
            lambda z: jnp.asarray(-jnp.inf, dtype=xi.dtype),
            xi,
        )

    def _like_or_neginf(xi: Array, ok: Array) -> tuple[Array, Array]:
        """
        Evaluates the full likelihood only when the particle is valid.

        If the particle is not valid, the function returns -inf and
        an empty blob with the correct shape.

        Parameters:
        -----------
        xi:
            one particle in x-space, shape (D,).
        ok:
            Boolean flag saying whether the particle is valid.

        Returns:
        --------
        Tuple[Array, Array]:
            log-likelihood value and blob output.
        """

        def _do(z: Array) -> tuple[Array, Array]:
            ll, bb = loglike_fn(z)
            return ll, bb

        def _skip(z: Array) -> tuple[Array, Array]:
            return jnp.asarray(-jnp.inf, dtype=xi.dtype), blob_template

        return jax.lax.cond(ok, _do, _skip, xi)

    def _approx_or_neginf(xi: Array, ok: Array) -> Array:
        """
        Evaluates the approximate likelihood only when the particle is valid.

        If the particle is not valid, the function returns -inf.
        This helper is used by delayed acceptance.

        Parameters:
        -----------
        xi:
            one particle in x-space, shape (D,).
        ok:
            Boolean flag saying whether the particle is valid.

        Returns:
        --------
        Array:
            approximate log-likelihood value, or -inf.
        """

        def _do(z: Array) -> Array:
            return jnp.asarray(loglike_approx_fn(z), dtype=xi.dtype)

        def _skip(z: Array) -> Array:
            return jnp.asarray(-jnp.inf, dtype=xi.dtype)

        return jax.lax.cond(ok, _do, _skip, xi)

    finite_current = jnp.isfinite(logp) & jnp.isfinite(logl)

    def _init_approx(_):
        """
        Initializes approximate likelihood values for current particles.

        This is used only when delayed acceptance is enabled.

        Parameters:
        -----------
        _:
            unused input required by lax.cond.

        Returns:
        --------
        Array:
            approximate log-likelihood values, shape (N,).
        """
        return jax.vmap(_approx_or_neginf, in_axes=(0, 0), out_axes=0)(
            x,
            finite_current,
        )

    def _zero_approx(_):
        """
        Builds zero approximate likelihood values.

        This is used when delayed acceptance is disabled.

        Parameters:
        -----------
        _:
            unused input required by lax.cond.

        Returns:
        --------
        Array:
            zero array with the same shape as logl.
        """
        return jnp.zeros_like(logl)

    logl_approx0 = jax.lax.cond(
        jnp.asarray(use_delayed_acceptance),
        _init_approx,
        _zero_approx,
        operand=None,
    )

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

    max_sigma_cap = jnp.minimum(
        jnp.asarray(2.38, dtype=dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=dtype)),
        jnp.asarray(0.99, dtype=dtype),
    )

    def cond_fn(carry):
        """
        Checks whether the mutation loop should continue.

        The loop continues while the maximum number of iterations
        has not been reached and the stopping rule has not triggered.

        Parameters:
        -----------
        carry:
            current mutation-loop state.

        Returns:
        --------
        Array:
            Boolean scalar.
            True means continue the loop.
            False means stop the loop.
        """
        (_, _, _, _, _, _, _, _, _, _, _, _, _, _, i, _, _, done) = carry
        return (i < jnp.asarray(n_max, dtype=i.dtype)) & (~done)

    def body_fn(carry):
        """
        Performs one LI-pCN mutation iteration.

        The function proposes new particles in theta-space, maps them
        back to u-space and x-space, evaluates the prior and likelihood,
        accepts or rejects the proposal, and updates the proposal scale.

        It also updates the empirical proposal center and checks whether
        the mutation loop should stop.

        Parameters:
        -----------
        carry:
            current mutation-loop state.
            It contains the random key, particles, log values, blobs,
            proposal center, proposal scale, counters, acceptance value,
            and stopping flag.

        Returns:
        --------
        tuple:
            updated mutation-loop state.
        """
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

        i1 = i + jnp.asarray(1, dtype=i.dtype)
        key, k_prop, k_unif = jax.random.split(key, 3)

        theta_prime = _li_pcn_proposal(
            k_prop,
            theta,
            mu,
            eigvecs,
            var_dir,
            active,
            sigma,
            li_lis_scale=li_lis_scale,
            li_cs_scale=li_cs_scale,
        )

        def _t2u_single(ti: Array) -> tuple[Array, Array]:
            """
            Maps one particle from theta-space back to u-space.

            Parameters:
            -----------
            ti:
                one particle in theta-space, shape (D,).

            Returns:
            --------
            Tuple[Array, Array]:
                particle in u-space and flow log-determinant.
            """
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
            """
            Evaluates approximate likelihood values for proposed particles.

            This is used only when delayed acceptance is enabled.

            Parameters:
            -----------
            _:
                unused input required by lax.cond.

            Returns:
            --------
            Array:
                approximate log-likelihood values for proposed particles.
            """
            return jax.vmap(_approx_or_neginf, in_axes=(0, 0), out_axes=0)(
                x_prime,
                finite1,
            )

        def _zero_approx_prime(_):
            """
            Builds zero approximate likelihood values for proposed particles.

            This is used when delayed acceptance is disabled.

            Parameters:
            -----------
            _:
                unused input required by lax.cond.

            Returns:
            --------
            Array:
                zero array with the same shape as logl_prime.
            """
            return jnp.zeros_like(logl_prime)

        logl_approx_prime = jax.lax.cond(
            jnp.asarray(use_delayed_acceptance),
            _eval_approx_prime,
            _zero_approx_prime,
            operand=None,
        )

        log_ref = _li_log_reference(theta, mu, eigvecs, var_dir)
        log_ref_prime = _li_log_reference(theta_prime, mu, eigvecs, var_dir)

        shared_terms = (
            (logp_prime - logp)
            + (logdetj_prime - logdetj)
            + (logdetj_flow_prime - logdetj_flow)
            + (log_ref - log_ref_prime)
        )

        log_ratio_full = beta * (logl_prime - logl) + shared_terms
        log_ratio_surrogate = beta * (logl_approx_prime - logl_approx) + shared_terms

        def _mh_accept(_):
            """
            Applies the standard Metropolis-Hastings acceptance rule.

            This branch is used when delayed acceptance is disabled.

            Parameters:
            -----------
            _:
                unused input required by lax.cond.

            Returns:
            --------
            tuple:
                accept_mask:
                    Boolean mask showing which particles are accepted.
                accept_value:
                    mean acceptance probability.
            """
            log_alpha = log_ratio_full
            alpha = jnp.exp(jnp.minimum(jnp.asarray(0.0, dtype=dtype), log_alpha))
            alpha = jnp.where(jnp.isnan(alpha), jnp.asarray(0.0, dtype=dtype), alpha)

            u_rand = jax.random.uniform(k_unif, shape=(n_walkers,), dtype=dtype)
            accept_mask = u_rand < alpha
            accept_value = jnp.mean(alpha)
            return accept_mask, accept_value

        def _da_accept(_):
            """
            Applies the conservative delayed-acceptance rule.

            This branch is used when delayed acceptance is enabled.
            It combines the surrogate likelihood ratio and the full
            likelihood ratio.

            Parameters:
            -----------
            _:
                unused input required by lax.cond.

            Returns:
            --------
            tuple:
                accept_mask:
                    Boolean mask showing which particles are accepted.
                accept_value:
                    mean delayed-acceptance probability.
            """
            da = conservative_damh_step_jax(
                key=k_unif,
                new_particles=theta_prime,
                old_particles=theta,
                cov=cov_ref,
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
            (jnp.asarray(2.38, dtype=dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=dtype)))
            / sigma,
            jnp.asarray(2.0, dtype=dtype),
        )
        done = cnt.astype(dtype) >= thresh

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

    carry_f = jax.lax.while_loop(cond_fn, body_fn, carry0)

    (
        key,
        u,
        x,
        theta,
        logdetj,
        logdetj_flow,
        logl,
        _logl_approx,
        logp,
        blobs,
        _mu,
        sigma,
        _logp2_best,
        _cnt,
        i,
        calls,
        accept,
        _done,
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
