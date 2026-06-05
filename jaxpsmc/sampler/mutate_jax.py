from __future__ import annotations
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp

from ..mcmc.dili_pcn_jax import dili_pcn_jax
from ..mcmc.li_pcn_jax import likelihood_informed_pcn_jax
from ..mcmc.pcn_jax import preconditioned_pcn_jax




#################################################################
# 3. MUTATE
#################################################################

Array = jax.Array


def _log_like(
    x_i: Array,
    loglike_single_fn: Callable[[Array], Tuple[Array, Array]],
) -> Tuple[Array, Array]:
    """
    Calls the single-particle log-likelihood function.

    This wrapper gives a consistent interface for batched likelihood
    evaluation with vmap. It does not change the likelihood value.

    Parameters:
    -----------
    x_i:
        one particle in x-space, shape (D,).
    loglike_single_fn:
        function that evaluates one particle.
        It must return a log-likelihood value and blob output.

    Returns:
    --------
    Tuple[Array, Array]:
        log-likelihood value and blob output for one particle.
    """
    
    return loglike_single_fn(x_i)

# map single-particle likelihood wrapper over a batch of particles
_log_like_batched = jax.vmap(_log_like, in_axes=(0, None), out_axes=(0, 0))


  


def mutate(
    key: Array,
    current_particles: Dict[str, Array],
    *,
    use_preconditioned_pcn: Array,

    # functions required by mutation kernels
    loglike_single_fn: Callable[[Array], Tuple[Array, Array]],
    loglike_approx_single_fn: Optional[Callable[[Array], Array]] = None,
    logprior_fn: Callable[[Array], Array],
    flow: Any,
    scaler_cfg: Mapping[str, Array],
    scaler_masks: Mapping[str, Array],

    # default pCN geometry
    geom_mu: Array,
    geom_cov: Array,
    geom_nu: Array,

    # empirical LI-pCN geometry; if omitted, geom_mu/geom_cov are reused
    li_geom_mu: Optional[Array] = None,
    li_geom_cov: Optional[Array] = None,

    # kernel choice
    kernel: str = "pcn",

    # empirical LI-pCN options
    li_rank: int = 16,
    li_lis_scale: float = 1.0,
    li_cs_scale: float = 1.0,
    li_var_floor: float = 1e-8,
    li_complement_var: float = 1.0,

    # Hessian/GNH-based DILI-pCN geometry
    dili_center: Optional[Array] = None,
    dili_basis: Optional[Array] = None,
    dili_post_var: Optional[Array] = None,
    dili_cov_ref: Optional[Array] = None,

    # DILI-pCN options
    dili_lis_scale: float = 1.0,
    dili_cs_scale: float = 1.0,    

    # choice form
    n_max: int,
    n_steps: int,
    use_delayed_acceptance: Array = jnp.asarray(False),
    da_c_const: Array = jnp.asarray(0.01),
    da_d_const: Array = jnp.asarray(2.0),
    condition: Optional[Array] = None,
) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
    """
    Runs the mutation step to the selected pCN-type kernel.

    This function chooses between three mutation kernels:
        * ``kernel="pcn"`` runs the standard preconditioned pCN kernel.
        * ``kernel="li_pcn"`` runs the empirical likelihood-informed pCN kernel.
        * ``kernel="dili_pcn"`` runs the Hessian/GNH-based DILI-pCN kernel.

    The ``kernel`` argument is static Python configuration. It is
    a Python string and should not be passed as a traced JAX value.

    Parameters
    ----------
    key:
        JAX random key used by the selected mutation kernel.
    current_particles:
        Dictionary containing the current particle state. It must contain
        ``u``, ``x``, ``logdetj``, ``logl``, ``logp``, ``logdetj_flow``,
        ``blobs``, ``beta``, ``calls``, and ``proposal_scale``.
    use_preconditioned_pcn:
        Boolean scalar controlling whether mutation is active.
    loglike_single_fn:
        Exact single-particle log-likelihood function.
    loglike_approx_single_fn:
        Optional approximate single-particle log-likelihood function used
        by delayed acceptance.
    logprior_fn:
        Single-particle log-prior function.
    flow:
        Flow object or flow parameters used to move between ``u`` and
        ``theta`` spaces.
    scaler_cfg:
        Scaler configuration dictionary.
    scaler_masks:
        Scaler mask dictionary.
    geom_mu:
        Mean vector used by the standard preconditioned pCN geometry.
    geom_cov:
        Covariance matrix used by the standard preconditioned pCN geometry.
    geom_nu:
        Degrees of freedom used by the standard pCN/Student-t geometry.
    li_geom_mu:
        Optional mean vector for the empirical LI-pCN kernel. If omitted,
        ``geom_mu`` is reused.
    li_geom_cov:
        Optional covariance matrix for the empirical LI-pCN kernel. If
        omitted, ``geom_cov`` is reused.
    kernel:
        Mutation kernel name. Must be ``"pcn"``, ``"li_pcn"``, or
        ``"dili_pcn"``.
    li_rank:
        Number of empirical likelihood-informed directions used by LI-pCN.
    li_lis_scale:
        Proposal-scale multiplier for LI-pCN active directions.
    li_cs_scale:
        Proposal-scale multiplier for LI-pCN complement directions.
    li_var_floor:
        Minimum variance used in the empirical LI-pCN geometry.
    li_complement_var:
        Reference variance used outside the empirical LIS.
    dili_center:
        Center of the DILI-pCN geometry in theta-space.
    dili_basis:
        Orthonormal DILI basis, shape ``(D, r)``.
    dili_post_var:
        Posterior variance estimates in the DILI basis, shape ``(r,)``.
    dili_cov_ref:
        Reference covariance used by conservative delayed acceptance.
    dili_lis_scale:
        Proposal-scale multiplier for the DILI LIS move.
    dili_cs_scale:
        Proposal-scale multiplier for the DILI complement-space move.
    n_max:
        Maximum number of mutation iterations.
    n_steps:
        Baseline number of mutation steps used by the stopping rule.
    use_delayed_acceptance:
        If true, the selected kernel uses delayed acceptance.
    da_c_const:
        Conservative delayed-acceptance constant ``c``.
    da_d_const:
        Conservative delayed-acceptance constant ``d``.
    condition:
        Optional condition passed to flow helper functions.

    Returns
    -------
    Tuple[Array, Dict[str, Array], Dict[str, Array]]:
        Updated random key, updated particle dictionary, and diagnostics.
    """
    u = current_particles["u"]
    n_dim = u.shape[1]
    norm_ref = jnp.asarray(2.38, dtype=u.dtype) / jnp.sqrt(
        jnp.asarray(n_dim, dtype=u.dtype)
    )

    payload = (
        key,
        current_particles["u"],
        current_particles["x"],
        current_particles["logdetj"],
        current_particles["logl"],
        current_particles["logp"],
        current_particles["logdetj_flow"],
        current_particles["blobs"],
        current_particles["beta"],
        current_particles["proposal_scale"],
    )

    def loglike_fn_single(x_i: Array) -> Tuple[Array, Array]:
        """
        Evaluates the exact likelihood wrapper for one particle.

        Parameters
        ----------
        x_i:
            One particle in physical space, shape ``(D,)``.

        Returns
        -------
        Tuple[Array, Array]:
            Exact log-likelihood value and blob output for this particle.
        """
        return _log_like(x_i, loglike_single_fn)

    def loglike_approx_fn_single(x_i: Array) -> Array:
        """
        Evaluates approximate likelihood for one particle.

        If no approximate likelihood function is supplied, this returns zero.
        That keeps the delayed-acceptance kernel interface valid even when
        delayed acceptance is disabled.

        Parameters
        ----------
        x_i:
            One particle in physical space, shape ``(D,)``.

        Returns
        -------
        Array:
            Approximate log-likelihood value for this particle.
        """
        if loglike_approx_single_fn is None:
            return jnp.asarray(0.0, dtype=x_i.dtype)
        return jnp.asarray(loglike_approx_single_fn(x_i), dtype=x_i.dtype)

    def _do_pcn(op):
        """
        Runs the standard preconditioned pCN mutation branch.

        Parameters
        ----------
        op:
            Packed mutation state containing the random key, particles,
            log-density values, blobs, temperature, and proposal scale.

        Returns
        -------
        Dict[str, Array]:
            Result dictionary returned by ``preconditioned_pcn_jax``.
        """
        (
            key0, u0, x0, logdetj0, logl0, logp0,
            logdetj_flow0, blobs0, beta0, proposal_scale0,
        ) = op

        return preconditioned_pcn_jax(
            key0,
            u=u0,
            x=x0,
            logdetj=logdetj0,
            logp=logp0,
            logl=logl0,
            logdetj_flow=logdetj_flow0,
            blobs=blobs0,
            beta=beta0,
            loglike_fn=loglike_fn_single,
            loglike_approx_fn=loglike_approx_fn_single,
            logprior_fn=logprior_fn,
            flow=flow,
            scaler_cfg=scaler_cfg,
            scaler_masks=scaler_masks,
            geom_mu=geom_mu,
            geom_cov=geom_cov,
            geom_nu=geom_nu,
            n_max=n_max,
            n_steps=n_steps,
            proposal_scale=proposal_scale0,
            use_delayed_acceptance=use_delayed_acceptance,
            da_c_const=da_c_const,
            da_d_const=da_d_const,
            condition=condition,
        )

    def _do_li_pcn(op):
        """
        Runs empirical likelihood-informed pCN mutation branch.

        If LI-specific geometry is not supplied, this branch reuses the
        default geometry from ``geom_mu`` and ``geom_cov``.

        Parameters
        ----------
        op:
            Packed mutation state containing the random key, particles,
            log-density values, blobs, temperature, and proposal scale.

        Returns
        -------
        Dict[str, Array]:
            Result dictionary returned by ``likelihood_informed_pcn_jax``.
        """
        (
            key0, u0, x0, logdetj0, logl0, logp0,
            logdetj_flow0, blobs0, beta0, proposal_scale0,
        ) = op

        li_mu = geom_mu if li_geom_mu is None else li_geom_mu
        li_cov = geom_cov if li_geom_cov is None else li_geom_cov

        return likelihood_informed_pcn_jax(
            key0,
            u=u0,
            x=x0,
            logdetj=logdetj0,
            logp=logp0,
            logl=logl0,
            logdetj_flow=logdetj_flow0,
            blobs=blobs0,
            beta=beta0,
            loglike_fn=loglike_fn_single,
            loglike_approx_fn=loglike_approx_fn_single,
            logprior_fn=logprior_fn,
            flow=flow,
            scaler_cfg=scaler_cfg,
            scaler_masks=scaler_masks,
            geom_mu=li_mu,
            geom_cov=li_cov,
            n_max=n_max,
            n_steps=n_steps,
            proposal_scale=proposal_scale0,
            li_rank=li_rank,
            li_lis_scale=li_lis_scale,
            li_cs_scale=li_cs_scale,
            li_var_floor=li_var_floor,
            li_complement_var=li_complement_var,
            use_delayed_acceptance=use_delayed_acceptance,
            da_c_const=da_c_const,
            da_d_const=da_d_const,
            condition=condition,
        )
    
    def _do_dili_pcn(op):
        """
        Runs the Hessian/GNH-based DILI-pCN mutation branch.

        This branch requires precomputed DILI geometry. The required objects
        are ``dili_center``, ``dili_basis``, ``dili_post_var``, and
        ``dili_cov_ref``.

        Parameters
        ----------
        op:
            Packed mutation state containing the random key, particles,
            log-density values, blobs, temperature, and proposal scale.

        Returns
        -------
        Dict[str, Array]:
            Result dictionary returned by ``dili_pcn_jax``.

        Raises
        ------
        ValueError:
            If any required DILI geometry object is missing.
        """
        (
            key0, u0, x0, logdetj0, logl0, logp0,
            logdetj_flow0, blobs0, beta0, proposal_scale0,
        ) = op

        if (
            dili_center is None
            or dili_basis is None
            or dili_post_var is None
            or dili_cov_ref is None
        ):
            raise ValueError(
                "kernel='dili_pcn' requires dili_center, dili_basis, "
                "dili_post_var, and dili_cov_ref."
            )

        return dili_pcn_jax(
            key0,
            u=u0,
            x=x0,
            logdetj=logdetj0,
            logp=logp0,
            logl=logl0,
            logdetj_flow=logdetj_flow0,
            blobs=blobs0,
            beta=beta0,
            loglike_fn=loglike_fn_single,
            loglike_approx_fn=loglike_approx_fn_single,
            logprior_fn=logprior_fn,
            flow=flow,
            scaler_cfg=scaler_cfg,
            scaler_masks=scaler_masks,
            dili_center=dili_center,
            dili_basis=dili_basis,
            dili_post_var=dili_post_var,
            dili_cov_ref=dili_cov_ref,
            n_max=n_max,
            n_steps=n_steps,
            proposal_scale=proposal_scale0,
            dili_lis_scale=dili_lis_scale,
            dili_cs_scale=dili_cs_scale,
            use_delayed_acceptance=use_delayed_acceptance,
            da_c_const=da_c_const,
            da_d_const=da_d_const,
            condition=condition,
        )

    def _do_noop(op):
        """
        Returns the current particle state without applying mutation.

        Branch is used when ``use_preconditioned_pcn`` is false.
        It preserves all particles and log-density values and returns zero
        mutation diagnostics.

        Parameters
        ----------
        op:
            Packed mutation state containing the random key, particles,
            log-density values, blobs, temperature, and proposal scale.

        Returns
        -------
        Dict[str, Array]:
            Result dictionary with unchanged particles and zero diagnostics.
        """
        (
            key0, u0, x0, logdetj0, logl0, logp0,
            logdetj_flow0, blobs0, _beta0, proposal_scale0,
        ) = op

        z0f = jnp.asarray(0.0, dtype=u0.dtype)
        z0i = jnp.asarray(0, dtype=jnp.int32)

        return {
            "key": key0,
            "u": u0,
            "x": x0,
            "logdetj": logdetj0,
            "logdetj_flow": logdetj_flow0,
            "logl": logl0,
            "logp": logp0,
            "blobs": blobs0,
            "efficiency": proposal_scale0,
            "accept": z0f,
            "steps": z0i,
            "calls": z0i,
            "proposal_scale": proposal_scale0,
        }

    kernel_l = str(kernel).lower()

    if kernel_l == "pcn":
        results = jax.lax.cond(
            jnp.asarray(use_preconditioned_pcn),
            _do_pcn,
            _do_noop,
            payload,
        )
    elif kernel_l == "li_pcn":
        results = jax.lax.cond(
            jnp.asarray(use_preconditioned_pcn),
            _do_li_pcn,
            _do_noop,
            payload,
        )
    elif kernel_l == "dili_pcn":
        results = jax.lax.cond(
            jnp.asarray(use_preconditioned_pcn),
            _do_dili_pcn,
            _do_noop,
            payload,
        )
    else:
        raise ValueError("kernel must be one of: 'pcn', 'li_pcn', 'dili_pcn'.")

    new_calls = current_particles["calls"] + results["calls"]
    new_proposal_scale = results["proposal_scale"]

    new_particles = {
        "u": results["u"],
        "x": results["x"],
        "logdetj": results["logdetj"],
        "logl": results["logl"],
        "logp": results["logp"],
        "logdetj_flow": results["logdetj_flow"],
        "blobs": results["blobs"],
        "beta": current_particles["beta"],
        "calls": new_calls,
        "proposal_scale": new_proposal_scale,
        "efficiency": results["efficiency"] / norm_ref,
        "steps": results["steps"],
        "accept": results["accept"],
    }

    info = {
        "efficiency_raw": results["efficiency"],
        "proposal_scale": results["proposal_scale"],
        "accept": results["accept"],
        "steps": results["steps"],
        "calls_increment": results["calls"],
    }

    return results["key"], new_particles, info