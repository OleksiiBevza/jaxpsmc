from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ..particles_jax import ParticlesState, compute_logw_and_logz_jax
from ..tools_jax import effective_sample_size_jax, unique_sample_size_jax

#################################################################
# 4. _not_termination part
#################################################################
Array = jax.Array


@jax.jit
def not_termination_jax(
    state: ParticlesState,
    beta_current: Array,
    n_total: Array,
    metric_code: Array,
    n_active: Array,
    beta_tol: Array = jnp.asarray(1e-4),
) -> Array:
    """
    Checks whether the SMC sampler should continue.

    The sampler continues if beta has not reached one.
    It also continues if the final-weight metric is still too small.
    The metric can be ESS or USS, depending on metric_code.

    Parameters:
    -----------
    state:
        particle history state.
    beta_current:
        current annealing value.
    n_total:
        required ESS or USS threshold.
    metric_code:
        integer code selecting the metric.
        0 means ESS.
        1 means USS.
    n_active:
        number of active particles used by USS.
    beta_tol:
        tolerance used to decide whether beta is close enough to one.

    Returns:
    --------
    Array:
        Boolean scalar.
        True means the sampler should continue.
        False means the termination condition has been reached.
    """
    # do final-step log-weights at beta = 1
    logw_flat, _, mask_flat = compute_logw_and_logz_jax(
        state, beta_final=jnp.asarray(1.0, dtype=state.logl.dtype), normalize=False
    )

    # keep only valid entries, invalid entries make -inf
    logw_valid = jnp.where(mask_flat, logw_flat, -jnp.inf)

    # safe maximum before exponentiating
    m = jnp.max(logw_valid)
    m_safe = jnp.where(jnp.isfinite(m), m, jnp.asarray(0.0, dtype=logw_flat.dtype))

    # transform valid log-weights into positive weights
    weights = jnp.where(
        mask_flat,
        jnp.exp(logw_valid - m_safe),
        jnp.asarray(0.0, dtype=logw_flat.dtype),
    )

    # convert active particle count to int32 for USS
    n_active_i32 = jnp.asarray(n_active, dtype=jnp.int32)

    # select ESS or USS based on metric code
    ess_or_uss = lax.cond(
        jnp.asarray(metric_code, dtype=jnp.int32) == jnp.int32(0),
        lambda w: effective_sample_size_jax(w),
        lambda w: unique_sample_size_jax(w, k=n_active_i32),
        weights,
    )

    # continue when beta is not close to 1 or metric is small
    beta_not_close = (
        jnp.asarray(1.0, dtype=beta_current.dtype) - beta_current
    ) >= jnp.asarray(beta_tol, dtype=beta_current.dtype)
    ess_too_small = ess_or_uss < jnp.asarray(n_total, dtype=ess_or_uss.dtype)

    return jnp.logical_or(beta_not_close, ess_too_small)
