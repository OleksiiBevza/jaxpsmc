from __future__ import annotations

from typing import Callable, Mapping, Tuple, Any, Optional, Dict

import jax
import jax.numpy as jnp

from ..scaler_jax import *
from ..delayed_acceptance.da_conservative_damh_jax import (
    conservative_damh_step_jax,)
from .flow_jax import _flow_u_to_theta_jax, _flow_theta_to_u_jax


Array = jax.Array

 

def preconditioned_pcn_jax(
    key: Array,
    *,

    # current state (all arrays; no None)
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

    # geometry (Student-t)
    geom_mu: Array,
    geom_cov: Array,
    geom_nu: Array,

    # options
    n_max: int,
    n_steps: int,
    proposal_scale: Array,
    use_delayed_acceptance: Array = jnp.asarray(False),
    da_c_const: Array = jnp.asarray(0.01),
    da_d_const: Array = jnp.asarray(2.0),
    condition: Optional[Array] = None,
) -> Dict[str, Array]:
    """
    Runs a preconditioned pCN mutation kernel for a batch of particles.

    The function updates particles using a proposal in theta-space.
    The proposal is preconditioned by a fitted Student-t geometry.
    This means the proposal uses an estimated mean, covariance,
    and degrees of freedom from previous particles.

    The current particles are first mapped from u-space to theta-space.
    A new theta proposal is generated with a pCN-style update.
    The proposed theta values are then mapped back to u-space.
    Finally, they are converted to x-space and tested with an MH rule.

    The function can also use delayed acceptance.
    In that case, it builds both a surrogate acceptance ratio
    and a full acceptance ratio.
    The final accept/reject decision is then made by
    conservative_damh_step_jax.

    Parameters:
    -----------
    key:
        JAX random key used for all random draws in the mutation loop.
    u:
        current particles in u-space, shape (N, D).
        This is the scaled latent representation.
    x:
        current particles in x-space, shape (N, D).
        This is the physical or model input representation.
    logdetj:
        current scaler log determinant values, shape (N,).
        These correct the density after scaling transformations.
    logl:
        current full log-likelihood values, shape (N,).
    logp:
        current log-prior values, shape (N,).
    logdetj_flow:
        current flow log determinant values, shape (N,).
        These are recomputed at the start of the function.
    blobs:
        extra outputs stored from the likelihood, shape (N, B...)
        or shape (N, 0) when no extra values are used.
    beta:
        tempering value used in the likelihood part of the MH ratio.
    loglike_fn:
        full likelihood function for one x-space particle.
        It must return a pair: log-likelihood and blob output.
    loglike_approx_fn:
        approximate likelihood function for one x-space particle.
        It is used only when delayed acceptance is enabled.
    logprior_fn:
        prior function for one x-space particle.
        It must return one scalar log-prior value.
    flow:
        flow object used to move between u-space and theta-space.
        It must provide a bijection with forward and inverse methods.
    scaler_cfg:
        configuration for the scaler transformation.
    scaler_masks:
        masks used by the scaler transformation.
    geom_mu:
        Student-t geometry mean, shape (D,).
        This is the center of the proposal.
    geom_cov:
        Student-t geometry covariance matrix, shape (D, D).
        This must be positive definite for Cholesky factorization.
    geom_nu:
        Student-t degrees of freedom.
        Smaller values give heavier-tailed proposal scaling.
    n_max:
        maximum number of inner pCN iterations.
    n_steps:
        value used in the stopping rule.
        Larger values allow more failed-improvement iterations.
    proposal_scale:
        initial pCN proposal scale.
        It is adapted during the mutation loop.
    use_delayed_acceptance:
        if True, use delayed-acceptance logic.
        If False, use a standard MH accept/reject rule.
    da_c_const:
        conservative delayed-acceptance clipping constant.
        It must be positive.
    da_d_const:
        conservative delayed-acceptance exponent constant.
        It must be greater than 1.
    condition:
        optional conditioning value passed to the flow.
        Use None when the flow is unconditional.

    Returns:
    --------
    Dict[str, Array]:
        dictionary with updated particles and diagnostics.
        It contains updated u, x, log determinants, log-likelihoods,
        log-priors, blobs, acceptance value, number of steps,
        call count, and final proposal scale.
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
    # define nrof walkers and parameter dimension
    n_walkers, n_dim = u.shape
    # precompute inverse covariance and Cholesky factor
    inv_cov = jnp.linalg.inv(geom_cov)
    chol_cov = jnp.linalg.cholesky(geom_cov)

    # Flow: u -> theta (batched via vmap)
    def _u2t_single(ui: Array) -> Tuple[Array, Array]:
        """
        Maps one particle from u-space to theta-space.

        This helper is used with jax.vmap.
        It applies the flow transformation to one particle at a time.

        Parameters:
        -----------
        ui:
            one particle in u-space, shape (D,).

        Returns:
        --------
        Tuple[Array, Array]:
            transformed particle in theta-space,
            and its flow log determinant correction.
        """
        # apply forward flow to one walker
        return _flow_u_to_theta_jax(flow, ui, condition)

    # map all walkers from u-space to theta-space
    theta, logdetj_flow0 = jax.vmap(_u2t_single, in_axes=0, out_axes=(0, 0))(u)

    # initialize running mean and adaptation values
    mu = geom_mu
    sigma0 = jnp.minimum(proposal_scale, jnp.asarray(0.99, dtype=u.dtype))
    logp2_best = jnp.mean(logl + logp)
    cnt0 = jnp.asarray(0, dtype=jnp.int32)
    i0 = jnp.asarray(0, dtype=jnp.int32)
    calls0 = jnp.asarray(0, dtype=jnp.int32)
    accept0 = jnp.asarray(0.0, dtype=u.dtype)
    done0 = jnp.asarray(False)

    # replace old flow log det with recomputed one
    logdetj_flow = logdetj_flow0
    # Use first blob shape as template for skipped likelihood calls
    blob_template = jnp.zeros_like(blobs[0])

    # helpers: Student-t form
    def _quad(diff_: Array) -> Array:
        """
        Computes the geometry-scaled squared distance for each particle.

        The input is already centered around the proposal mean.
        The covariance inverse defines the scale of the distance.
        This is the quadratic form diff.T @ inv_cov @ diff.

        Parameters:
        -----------
        diff_:
            centered particle values, shape (N, D).

        Returns:
        --------
        Array:
            one quadratic-form value per particle, shape (N,).
        """
        # compute diff^T inv_cov diff row by row
        tmp = diff_ @ inv_cov
        return jnp.sum(tmp * diff_, axis=1)

    # skip invalid walkers
    def _prior_or_neginf(xi: Array, ok: Array) -> Array:
        """
        Evaluates the prior only when the proposal is valid.

        Invalid proposals should not be passed to the prior.
        For those proposals, the function returns -inf.
        This makes the MH probability equal to zero.

        Parameters:
        -----------
        xi:
            one proposed particle in x-space, shape (D,).
        ok:
            Boolean flag showing whether xi is finite and usable.

        Returns:
        --------
        Array:
            log-prior value if ok is True.
            Otherwise, -inf.
        """
        # skip the prior if the input already failed earlier checks.
        return jax.lax.cond(
            ok,
            lambda z: logprior_fn(z),
            lambda z: jnp.asarray(-jnp.inf, dtype=xi.dtype),
            xi,
        )

    def _like_or_neginf(xi: Array, ok: Array) -> Tuple[Array, Array]:
        """
        Evaluates the full likelihood only when the proposal is valid.

        Invalid proposals receive -inf likelihood.
        They also receive a zero blob with the correct shape.
        This keeps all JAX branches shape-compatible.

        Parameters:
        -----------
        xi:
            one proposed particle in x-space, shape (D,).
        ok:
            Boolean flag showing whether xi passed previous checks.

        Returns:
        --------
        Tuple[Array, Array]:
            log-likelihood value and blob output.
            Invalid proposals receive -inf and a zero blob.
        """
        def _do(z: Array) -> Tuple[Array, Array]:
            """
            Evaluates the full likelihood for one valid particle.

            Parameters:
            -----------
            z:
                one valid particle in x-space, shape (D,).

            Returns:
            --------
            Tuple[Array, Array]:
                full log-likelihood and blob output.
            """
            # evaluate user-provided likelihood.
            ll, bb = loglike_fn(z)
            return ll, bb

        def _skip(z: Array) -> Tuple[Array, Array]:
            """
            Returns fallback values for one invalid particle.

            The likelihood is skipped.
            The returned values keep the same shapes as the valid branch.

            Parameters:
            -----------
            z:
                one invalid particle in x-space.
                It is unused except for dtype compatibility.

            Returns:
            --------
            Tuple[Array, Array]:
                -inf likelihood and zero blob.
            """
            # keep shapes consistent when likelihood is skipped
            return jnp.asarray(-jnp.inf, dtype=xi.dtype), blob_template
        # choose between real likelihood and fallback branch.
        return jax.lax.cond(ok, _do, _skip, xi)


    def _approx_or_neginf(xi: Array, ok: Array) -> Array:
        """
        Evaluates the approximate likelihood only when the proposal is valid.

        This helper is used for delayed acceptance.
        Invalid proposals receive -inf.
        That makes them automatically rejected by the surrogate stage.

        Parameters:
        -----------
        xi:
            one proposed particle in x-space, shape (D,).
        ok:
            Boolean flag showing whether xi passed previous checks.

        Returns:
        --------
        Array:
            approximate log-likelihood if ok is True.
            Otherwise, -inf.
        """
        def _do(z: Array) -> Array:
            """
            Evaluates the approximate likelihood for one valid particle.

            Parameters:
            -----------
            z:
                one valid particle in x-space, shape (D,).

            Returns:
            --------
            Array:
                approximate log-likelihood value.
            """
            return jnp.asarray(loglike_approx_fn(z), dtype=xi.dtype)

        def _skip(z: Array) -> Array:
            """
            Returns a fallback approximate likelihood for invalid input.

            Parameters:
            -----------
            z:
                one invalid particle in x-space.
                It is unused except for dtype compatibility.

            Returns:
            --------
            Array:
                -inf value.
            """
            return jnp.asarray(-jnp.inf, dtype=xi.dtype)

        return jax.lax.cond(ok, _do, _skip, xi)
    
    finite_current = jnp.isfinite(logp) & jnp.isfinite(logl)

    def _init_approx(_):
        """
        Computes approximate likelihoods for the current particles.

        This branch is used only when delayed acceptance is enabled.
        It initializes the surrogate likelihood values needed
        for the surrogate acceptance ratio.

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
            x, finite_current
        )

    def _zero_approx(_):
        """
        Creates zero approximate likelihood values.

        This branch is used when delayed acceptance is disabled.
        The values are placeholders and are not used by standard MH.

        Parameters:
        -----------
        _:
            unused input required by lax.cond.

        Returns:
        --------
        Array:
            zero vector with the same shape as logl.
        """
        return jnp.zeros_like(logl)

    logl_approx0 = jax.lax.cond(
        jnp.asarray(use_delayed_acceptance),
        _init_approx,
        _zero_approx,
        operand=None,
    )    

    # pack loop state into one tuple for lax.while_loop.
    carry0 = (
        key, u, x, theta, logdetj, logdetj_flow, logl, logl_approx0, logp, blobs,
        mu, sigma0, logp2_best, cnt0, i0, calls0, accept0, done0
    )

    # cap adaptive proposal scale
    max_sigma_cap = jnp.minimum(jnp.asarray(2.38, dtype=u.dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=u.dtype)),
                                jnp.asarray(0.99, dtype=u.dtype))

    def cond_fn(carry):
        """
        Checks whether the inner pCN loop should continue.

        The loop stops when the maximum iteration count is reached.
        It also stops when the adaptive no-improvement rule marks
        the mutation as done.

        Parameters:
        -----------
        carry:
            current loop state.

        Returns:
        --------
        Array:
            Boolean scalar.
            True means another pCN update should run.
        """
        # stop when iteration limit is reached or done becomes True
        (_, _, _, _, _, _, _, _, _, _, _, _, _, _, i, _, _, done) = carry
        return (i < jnp.asarray(n_max, dtype=i.dtype)) & (~done)

    def body_fn(carry):
        """
        Performs one inner pCN proposal and accept/reject update.

        The step draws a Student-t-scaled pCN proposal in theta-space.
        It maps the proposal back to x-space.
        It evaluates prior and likelihood values.
        It then accepts or rejects each walker.

        The proposal scale is adapted using the average acceptance value.
        The running proposal mean is also updated.
        The loop can stop when the objective no longer improves.

        Parameters:
        -----------
        carry:
            current loop state.

        Returns:
        --------
        tuple:
            updated loop state after one pCN iteration.
        """
        # unpack current loop state
        (key, u, x, theta, logdetj, logdetj_flow, logl, logl_approx, logp, blobs,
         mu, sigma, logp2_best, cnt, i, calls, accept, done) = carry        

        # move to the next inner iteration
        i1 = i + jnp.asarray(1, dtype=i.dtype)
        # split random key for all random draws in this step
        key, k_gamma, k_norm, k_unif = jax.random.split(key, 4)
        # compute current centered values and quadratic term
        diff = theta - mu
        quad = _quad(diff)
        # draw Student-t scaling factors
        a = (jnp.asarray(n_dim, dtype=u.dtype) + geom_nu) / jnp.asarray(2.0, dtype=u.dtype)
        z = jax.random.gamma(k_gamma, a, shape=(n_walkers,))  # unit scale
        s = (geom_nu + quad) / (jnp.asarray(2.0, dtype=u.dtype) * z)
        # draw Gaussian noise in geometry covariance
        eps = jax.random.normal(k_norm, shape=(n_walkers, n_dim), dtype=u.dtype)
        noise = eps @ chol_cov.T
        # build proposal in theta-space
        theta_prime = (
            mu
            + jnp.sqrt(jnp.asarray(1.0, dtype=u.dtype) - sigma * sigma) * diff
            + sigma * jnp.sqrt(s)[:, None] * noise
        )


        def _t2u_single(ti: Array) -> Tuple[Array, Array]:
            """
            Maps one proposed particle from theta-space to u-space.

            This helper is used with jax.vmap.
            It applies the inverse flow transformation to one particle.

            Parameters:
            -----------
            ti:
                one proposed particle in theta-space, shape (D,).

            Returns:
            --------
            Tuple[Array, Array]:
                transformed particle in u-space,
                and its inverse-flow log determinant.
            """
            # apply inverse flow to one walker
            return _flow_theta_to_u_jax(flow, ti, condition)
        # map all walkers from theta-space back to u-space
        u_prime, logdetj_flow_prime = jax.vmap(_t2u_single, in_axes=0, out_axes=(0, 0))(theta_prime)

        # apply scaler inverse to move from u-space to x-space.
        x_prime, logdetj_prime = inverse_jax(u_prime, scaler_cfg, scaler_masks)

        # apply boundary handling and recompute the consistent transformed values
        x_prime_bc = apply_boundary_conditions_x_jax(x_prime, dict(scaler_cfg))
        u_prime_bc = forward_jax(x_prime_bc, scaler_cfg, scaler_masks)
        x_prime, logdetj_prime = inverse_jax(u_prime_bc, scaler_cfg, scaler_masks)

        # keep boundary-corrected u values
        u_prime = u_prime_bc

        # mark proposals that are still finite after the scaler step
        finite0 = jnp.isfinite(logdetj_prime) & jnp.all(jnp.isfinite(x_prime), axis=1)

        # evaluate prior only for finite proposals
        logp_prime = jax.vmap(_prior_or_neginf, in_axes=(0, 0), out_axes=0)(x_prime, finite0)
        finite1 = finite0 & jnp.isfinite(logp_prime)

        # evaluate full likelihood only for proposals that passed the prior step
        logl_prime, blobs_prime = jax.vmap(
            _like_or_neginf,
            in_axes=(0, 0),
            out_axes=(0, 0),
        )(x_prime, finite1)

        # evaluate surrogate likelihood only when delayed acceptance is enabled
        def _eval_approx_prime(_):
            """
            Computes approximate likelihoods for proposed particles.

            This branch is used only when delayed acceptance is enabled.
            It evaluates the surrogate likelihood after prior validation.

            Parameters:
            -----------
            _:
                unused input required by lax.cond.

            Returns:
            --------
            Array:
                approximate log-likelihood values for proposals, shape (N,).
            """
            return jax.vmap(
                _approx_or_neginf,
                in_axes=(0, 0),
                out_axes=0,
            )(x_prime, finite1)

        def _zero_approx_prime(_):
            """
            Creates zero approximate likelihoods for proposed particles.

            This branch is used when delayed acceptance is disabled.
            The values are placeholders for shape consistency.

            Parameters:
            -----------
            _:
                unused input required by lax.cond.

            Returns:
            --------
            Array:
                zero vector with the same shape as logl_prime.
            """
            return jnp.zeros_like(logl_prime)   


        logl_approx_prime = jax.lax.cond(
            jnp.asarray(use_delayed_acceptance),
            _eval_approx_prime,
            _zero_approx_prime,
            operand=None,
        )  
   

        # compute quadratic term for the proposal
        diff_prime = theta_prime - mu
        quad_prime = _quad(diff_prime)

        # build Student-t correction terms
        coef = -(jnp.asarray(n_dim, dtype=u.dtype) + geom_nu) / jnp.asarray(
            2.0, dtype=u.dtype
        )
        A = coef * jnp.log1p(quad_prime / geom_nu)
        B = coef * jnp.log1p(quad / geom_nu)

        # shared terms used by both the full and surrogate acceptance ratios
        shared_terms = (
            (logp_prime - logp)
            + (logdetj_prime - logdetj)
            + (logdetj_flow_prime - logdetj_flow)
            - A
            + B
        )

        # full MH ratio
        log_ratio_full = beta * (logl_prime - logl) + shared_terms

        # surrogate-stage ratio
        log_ratio_surrogate = beta * (logl_approx_prime - logl_approx) + shared_terms

        def _mh_accept(_):
            """
            Applies the standard MH accept/reject rule.

            This branch uses the full log acceptance ratio directly.
            Each walker receives its own uniform random draw.

            Parameters:
            -----------
            _:
                unused input required by lax.cond.

            Returns:
            --------
            Tuple[Array, Array]:
                Boolean accept mask for all walkers,
                and mean acceptance probability.
            """
            log_alpha = log_ratio_full
            alpha = jnp.exp(jnp.minimum(jnp.asarray(0.0, dtype=u.dtype), log_alpha))
            alpha = jnp.where(jnp.isnan(alpha), jnp.asarray(0.0, dtype=u.dtype), alpha)

            u_rand = jax.random.uniform(k_unif, shape=(n_walkers,), dtype=u.dtype)
            accept_mask_mh = u_rand < alpha
            accept_value_mh = jnp.mean(alpha)

            return accept_mask_mh, accept_value_mh

        def _da_accept(_):
            """
            Applies the conservative delayed-acceptance rule.

            This branch uses a surrogate ratio for the first stage.
            It then uses the full ratio for the second-stage correction.
            A proposal is accepted only if it passes both stages.

            Parameters:
            -----------
            _:
                unused input required by lax.cond.

            Returns:
            --------
            Tuple[Array, Array]:
                Boolean accept mask for all walkers,
                and mean total acceptance probability.
            """
            da = conservative_damh_step_jax(
                key=k_unif,
                new_particles=theta_prime,
                old_particles=theta,
                cov=geom_cov,
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


        # apply accept/reject updates to all particle values
        theta = jnp.where(accept_mask[:, None], theta_prime, theta)
        u = jnp.where(accept_mask[:, None], u_prime, u)
        x = jnp.where(accept_mask[:, None], x_prime, x)

        logdetj = jnp.where(accept_mask, logdetj_prime, logdetj)
        logdetj_flow = jnp.where(accept_mask, logdetj_flow_prime, logdetj_flow)
        logl = jnp.where(accept_mask, logl_prime, logl)
        logl_approx = jnp.where(accept_mask, logl_approx_prime, logl_approx)
        logp = jnp.where(accept_mask, logp_prime, logp)
        blobs = jnp.where(accept_mask.reshape((n_walkers,) + (1,) * (blobs.ndim - 1)), blobs_prime, blobs)

        accept = accept_value

        # adapt proposal scale toward the target acceptance rate
        step = jnp.asarray(1.0, dtype=u.dtype) / jnp.power(jnp.asarray(i1 + 1, dtype=u.dtype), jnp.asarray(0.75, dtype=u.dtype))
        sigma = sigma + step * (accept - jnp.asarray(0.234, dtype=u.dtype))
        sigma = jnp.abs(jnp.minimum(sigma, max_sigma_cap))
        
        # update the running mean in theta-space.
        mu_step = jnp.asarray(1.0, dtype=u.dtype) / jnp.asarray(i1 + 1, dtype=u.dtype)
        mu = mu + mu_step * (jnp.mean(theta, axis=0) - mu)

        # track whether average objective improved
        logp2_new = jnp.mean(logl + logp)
        improved = logp2_new > logp2_best
        cnt = jnp.where(improved, jnp.asarray(0, dtype=cnt.dtype), cnt + jnp.asarray(1, dtype=cnt.dtype))
        logp2_best = jnp.where(improved, logp2_new, logp2_best)

        # stop when no-improvement count reaches the threshold
        thresh = jnp.asarray(n_steps, dtype=u.dtype) * jnp.power(
            (jnp.asarray(2.38, dtype=u.dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=u.dtype))) / sigma,
            jnp.asarray(2.0, dtype=u.dtype),
        )
        done = cnt.astype(u.dtype) >= thresh
   
        return (
            key, u, x, theta, logdetj, logdetj_flow, logl, logl_approx, logp, blobs,
            mu, sigma, logp2_best, cnt, i1, calls, accept, done
        )    

    # run iterative PCN update loop
    carry_f = jax.lax.while_loop(cond_fn, body_fn, carry0)

    # unpack the final state
    (key, u, x, theta, logdetj, logdetj_flow, logl, logl_approx, logp, blobs,
     mu, sigma, logp2_best, cnt, i, calls, accept, done) = carry_f

    # return updated state and summary values
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


