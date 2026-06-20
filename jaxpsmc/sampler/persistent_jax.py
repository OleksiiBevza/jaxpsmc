from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .reweight_jax import (
    _bisect_beta_scan,
    _dynamic_neff,
    _weights_metric_logz,
)


###################################################################
# PERSISTENT SAMPLING
###################################################################
@partial(
    jax.jit,
    static_argnames=("bins", "bisect_steps", "keep_max", "trim_ess"),
)
def reweight_step_persistent_jax(
    state,
    n_effective,
    metric_id,
    dynamic,
    n_active,
    dynamic_ratio,
    bins=1000,
    bisect_steps=32,
    keep_max=4096,
    trim_ess=0.99,
):
    """
    Performs one exact persistent-sampling reweighting step.

    The function chooses the next beta value.
    It then computes importance weights for all active particles stored
    in the particle history. Unlike the truncated reweighting step, this
    version does not trim the weighted particle set before resampling.

    The beta choice has three cases.
    If the previous beta already gives a small metric, beta is unchanged.
    If beta equal to one is still safe, beta jumps to one.
    Otherwise, a bisection scan finds an intermediate beta.

    This is the exact persistent sampling variant.
    The candidate pool is the full active particle history.
    Inactive entries are kept in the fixed-size arrays but are masked out
    and assigned zero weight.

    Parameters:
    -----------
    state:
        particle history state.
    n_effective:
        target ESS or USS value.
    metric_id:
        integer code selecting the metric.
        METRIC_ESS selects effective sample size.
        METRIC_USS selects unique sample size.
    dynamic:
        Boolean flag.
        If True, update n_effective using the dynamic rule.
    n_active:
        number of active particles used by later stages.
        This is also used by the USS metric.
    dynamic_ratio:
        ratio used by the dynamic target update.
    bins:
        unused trimming argument kept for API compatibility.
        This is static for JAX compilation.
    bisect_steps:
        number of fixed bisection steps used to search beta.
        This is static for JAX compilation.
    keep_max:
        unused truncation argument kept for API compatibility.
        Exact persistent sampling keeps all active historical particles.
    trim_ess:
        unused trimming argument kept for API compatibility.
        Exact persistent sampling does not trim the candidate pool.

    Returns:
    --------
    tuple:
        current_particles:
            dictionary with active historical particles, persistent weights,
            beta, logz, and compatibility diagnostics.
        n_eff_new:
            updated target effective sample size.
        stats:
            dictionary with beta, logz, metric value, and n_effective.
    """
    # keep the same signature as reweight_step_jax for mode switching
    _ = (bins, keep_max, trim_ess)

    # define the most recent beta and logz from the state
    t_idx = jnp.maximum(state.t - jnp.int32(1), jnp.int32(0))
    beta_prev = lax.dynamic_index_in_dim(
        state.beta,
        t_idx,
        axis=0,
        keepdims=False,
    )
    logz_prev = lax.dynamic_index_in_dim(
        state.logz,
        t_idx,
        axis=0,
        keepdims=False,
    )

    # evaluate chosen metric at previous beta and at beta = 1
    beta_one = jnp.asarray(1.0, dtype=beta_prev.dtype)

    _, m_prev, _, _ = _weights_metric_logz(
        state,
        beta_prev,
        metric_id,
        n_active,
    )
    _, m_one, _, _ = _weights_metric_logz(
        state,
        beta_one,
        metric_id,
        n_active,
    )

    # build target value and tolerance for the metric
    target = jnp.asarray(n_effective, dtype=m_prev.dtype)
    tol = jnp.asarray(0.01, dtype=m_prev.dtype) * target

    # decide whether to keep beta_prev, jump to 1, or bisect
    c0 = m_prev <= target
    c1 = (~c0) & (m_one >= target)
    cid = jnp.where(
        c0,
        jnp.int32(0),
        jnp.where(c1, jnp.int32(1), jnp.int32(2)),
    )

    # run bisection scan for the middle case
    beta_bis = _bisect_beta_scan(
        state=state,
        lo=beta_prev,
        hi=beta_one,
        target=target,
        metric_id=metric_id,
        n_active=n_active,
        steps=bisect_steps,
        tol=tol,
    )

    # select beta for this reweight step
    beta = lax.switch(
        cid,
        (
            lambda _: beta_prev,
            lambda _: beta_one,
            lambda _: beta_bis,
        ),
        operand=None,
    )

    # compute persistent weights and logz for chosen beta
    w_full, ess_est, logz_new, _ = _weights_metric_logz(
        state,
        beta,
        metric_id,
        n_active,
    )
    logz = jnp.where(cid == jnp.int32(0), logz_prev, logz_new)

    # update target effective size
    n_eff_new = lax.cond(
        dynamic,
        lambda ne: _dynamic_neff(
            ne,
            w_full,
            n_active,
            jnp.asarray(dynamic_ratio, w_full.dtype),
        ),
        lambda ne: jnp.asarray(ne, dtype=jnp.int32),
        n_effective,
    )

    # read fixed history sizes
    T, N = state.logl.shape
    D = state.u.shape[-1]
    B = state.blobs.shape[-1]

    # build fixed-size active-history mask for JAX indexing
    mask_t = jnp.arange(T, dtype=state.t.dtype) < state.t
    mask_flat = jnp.repeat(mask_t, N)

    # flatten all particle history arrays
    u_flat = state.u.reshape((T * N, D))
    x_flat = state.x.reshape((T * N, D))
    logdetj_flat = state.logdetj.reshape((T * N,))
    logl_flat = state.logl.reshape((T * N,))
    logp_flat = state.logp.reshape((T * N,))
    blobs_flat = state.blobs.reshape((T * N, B))

    # keep all active historical particles as candidates
    weights = jnp.where(
        mask_flat,
        w_full,
        jnp.asarray(0.0, dtype=w_full.dtype),
    )
    # renormalize weights over active historical particles only
    wsum = jnp.sum(weights)
    weights = weights / jnp.where(
        wsum > jnp.asarray(0.0, dtype=w_full.dtype),
        wsum,
        jnp.asarray(1.0, dtype=w_full.dtype),
    )
    weights = jnp.where(
        mask_flat,
        weights,
        jnp.asarray(0.0, dtype=weights.dtype),
    )
    # build current particle dictionary used by the next stages
    current_particles = {
        "u": jnp.where(mask_flat[:, None], u_flat, jnp.asarray(0.0, u_flat.dtype)),
        "x": jnp.where(mask_flat[:, None], x_flat, jnp.asarray(0.0, x_flat.dtype)),
        "logdetj": jnp.where(
            mask_flat, logdetj_flat, jnp.asarray(0.0, logdetj_flat.dtype)
        ),
        "logl": jnp.where(mask_flat, logl_flat, jnp.asarray(0.0, logl_flat.dtype)),
        "logp": jnp.where(mask_flat, logp_flat, jnp.asarray(0.0, logp_flat.dtype)),
        "blobs": jnp.where(
            mask_flat[:, None], blobs_flat, jnp.asarray(0.0, blobs_flat.dtype)
        ),
        "logz": logz,
        "beta": beta,
        "weights": weights,
        "ess": ess_est,
        "idx": jnp.arange(T * N, dtype=jnp.int32),
        "keep_mask": mask_flat,
        "trim_threshold": jnp.asarray(0.0, dtype=w_full.dtype),
        "trim_ratio": jnp.asarray(1.0, dtype=w_full.dtype),
        "trim_mask_full": mask_flat,
    }
    # return summary dictionary with the particles
    stats = {
        "beta": beta,
        "logz": logz,
        "ess": ess_est,
        "n_effective": n_eff_new,
    }

    return current_particles, n_eff_new, stats
