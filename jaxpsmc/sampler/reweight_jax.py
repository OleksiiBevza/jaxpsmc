from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from ..particles_jax import compute_logw_and_logz_jax
from ..tools_jax import (
    effective_sample_size_jax,
    trim_weights_jax,
    unique_sample_size_jax,
)
from .constants_jax import METRIC_ESS, METRIC_USS



#################################################################
# 1. REWEIGHT
#################################################################

def _metric_value(weights, metric_id, n_active):
    """
    Computes the selected particle-quality metric.

    The function chooses between two metrics.
    ESS measures how evenly the particle weights are spread.
    USS measures how many unique particles are expected after resampling.

    Parameters:
    -----------
    weights:
        normalized particle weights, shape (K,).
    metric_id:
        integer code selecting the metric.
        METRIC_ESS selects effective sample size.
        METRIC_USS selects unique sample size.
    n_active:
        number of active particles.
        This is used by the USS calculation.

    Returns:
    --------
    Array:
        selected metric value.
    """
    return lax.cond(
        metric_id == METRIC_ESS,
        lambda w: effective_sample_size_jax(w),
        lambda w: unique_sample_size_jax(w, k=n_active),
        weights,
    )


def _weights_metric_logz(state, beta, metric_id, n_active):
    """
    Computes weights, metric value, and log evidence for one beta.

    The function builds final weights from the stored particle history.
    It then normalizes those weights and evaluates the selected metric.
    It also returns the log normalizing constant estimate for this beta.

    Parameters:
    -----------
    state:
        particle history state.
    beta:
        annealing value used to build the weights.
    metric_id:
        integer code selecting ESS or USS.
    n_active:
        number of active particles.
        This is used by the USS metric.

    Returns:
    --------
    tuple:
        w_full:
            normalized flattened particle weights, shape (K,).
        m_val:
            selected metric value.
        logz_new:
            log normalizing constant estimate at this beta.
        logw_flat:
            unnormalized flattened log weights, shape (K,).
    """
    # build flattened log-weights for the chosen beta
    logw_flat, logz_new, mask_flat = compute_logw_and_logz_jax(
        state, beta_final=beta, normalize=False
    )
    # force invalid entries to -inf before turning log-weights into weights
    logw_flat = jnp.where(mask_flat, logw_flat, -jnp.inf)
    # convert log-weights into normalized positive weights
    w_full = jax.nn.softmax(logw_flat)
    # compute selected metric from the full weights
    m_val = _metric_value(w_full, metric_id, n_active)
    return w_full, m_val, logz_new, logw_flat


def _bisect_beta_scan(state, lo, hi, target, metric_id, n_active, steps, tol):
    """
    Finds a beta value using a fixed-length bisection scan.

    The goal is to find a beta where the selected metric is close
    to the requested target. The function works inside JAX because
    it uses a fixed number of scan steps instead of a Python loop.

    Parameters:
    -----------
    state:
        particle history state.
    lo:
        lower beta bound.
    hi:
        upper beta bound.
    target:
        target value for the selected metric.
    metric_id:
        integer code selecting ESS or USS.
    n_active:
        number of active particles.
        This is used by the USS metric.
    steps:
        number of bisection scan steps.
    tol:
        tolerance for accepting the target match.

    Returns:
    --------
    Array:
        beta value selected by the bisection scan.
    """
    # keep all temporary values in the same dtype as the bounds
    dtype = jnp.asarray(lo).dtype

    def scan_step(carry, _):
        """
        Performs one bisection update for beta.

        The midpoint beta is tested.
        If the metric is close enough to the target, the scan marks
        the search as done. Otherwise, the beta interval is narrowed.

        Parameters:
        -----------
        carry:
            tuple containing lower bound, upper bound, done flag,
            and current beta estimate.
        _:
            unused scan input.

        Returns:
        --------
        tuple:
            updated carry and no scan output.
        """
        # unpack current bounds and status
        lo_c, hi_c, done_c, beta_c = carry
        # test midpoint of current interval
        mid = (lo_c + hi_c) * jnp.asarray(0.5, dtype)
        # evaluate selected metric at midpoint
        _, m_mid, _, _ = _weights_metric_logz(state, mid, metric_id, n_active)
        # check if midpoint is close enough to the target
        close = jnp.abs(m_mid - target) <= tol
        done2 = done_c | close

        # update interval based on the midpoint metric value
        hi2 = jnp.where((~done2) & (m_mid < target), mid, hi_c)
        lo2 = jnp.where((~done2) & (m_mid >= target), mid, lo_c)
        # save midpoint when target has been reached
        beta2 = jnp.where((~done_c) & close, mid, beta_c)

        
        return (lo2, hi2, done2, beta2), None

    # start from midpoint of the initial interval
    beta0 = (lo + hi) * jnp.asarray(0.5, dtype)
    carry0 = (lo, hi, jnp.asarray(False), beta0)
   
    # run fixed number of scan steps
    (lo_f, hi_f, done_f, beta_f), _ = lax.scan(
        scan_step,
        carry0,
        xs=jnp.arange(steps, dtype=jnp.int32),
    )

    # if scan never hit tolerance, return final midpoint
    mid_f = (lo_f + hi_f) * jnp.asarray(0.5, dtype)
    return jnp.where(done_f, beta_f, mid_f)


def _dynamic_neff(n_eff, weights_full, n_active, ratio):
    """
    Updates the target effective sample size dynamically.

    The update uses unique sample size as feedback.
    If the unique sample size is too small, the target is reduced.
    If the unique sample size is too large, the target is increased.
    This helps control how aggressive the next reweighting step is.

    Parameters:
    -----------
    n_eff:
        current target effective sample size.
    weights_full:
        normalized particle weights, shape (K,).
    n_active:
        number of active particles.
    ratio:
        target ratio for the unique sample size.

    Returns:
    --------
    Array:
        updated target effective sample size as int32.
    """
    # convert scalar inputs to weight dtype for stable arithmetic   
    n_eff_f = jnp.asarray(n_eff, dtype=weights_full.dtype)
    n_act_f = jnp.asarray(n_active, dtype=weights_full.dtype)
    # compute unique sample size from the weights
    nuniq = unique_sample_size_jax(weights_full, k=n_active)

    # build acceptance band around requested ratio
    low = n_act_f * (jnp.asarray(0.95, n_eff_f.dtype) * ratio)
    high = n_act_f * jnp.minimum(
        jnp.asarray(1.05, n_eff_f.dtype) * ratio,
        jnp.asarray(1.0, n_eff_f.dtype),
    )

    # use a tiny constant to avoid division by zero
    eps = jnp.asarray(1e-12, n_eff_f.dtype)

    # move target down or up depending on unique sample size
    down = (n_act_f / (nuniq + eps)) * n_eff_f
    up = ((nuniq + eps) / n_act_f) * n_eff_f
    # apply update only when unique sample size is outside band
    n2 = jnp.where(nuniq < low, down, n_eff_f)
    n3 = jnp.where(nuniq > high, up, n2)
    return jnp.floor(n3).astype(jnp.int32)


@partial(
    jax.jit,
    static_argnames=("bins", "bisect_steps", "keep_max", "trim_ess"),
)
def reweight_step_jax(
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
    Performs one SMC reweighting step.

    The function chooses the next beta value.
    It then computes importance weights for all stored particles.
    After that, it trims very small weights and keeps the strongest
    particles for the next resampling and mutation steps.

    The beta choice has three cases.
    If the previous beta already gives a small metric, beta is unchanged.
    If beta equal to one is still safe, beta jumps to one.
    Otherwise, a bisection scan finds an intermediate beta.

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
    dynamic_ratio:
        ratio used by the dynamic target update.
    bins:
        number of bins used by weight trimming.
        This is static for JAX compilation.
    bisect_steps:
        number of fixed bisection steps used to search beta.
        This is static for JAX compilation.
    keep_max:
        maximum number of particles kept after trimming.
        This is static for JAX compilation.
    trim_ess:
        ESS ratio target used by weight trimming.
        This is static for JAX compilation.

    Returns:
    --------
    tuple:
        current_particles:
            dictionary with kept particles, weights, beta, logz,
            and trimming diagnostics.
        n_eff_new:
            updated target effective sample size.
        stats:
            dictionary with beta, logz, metric value, and n_effective.
    """
    # define the most recent beta and logz from the state
    t_idx = jnp.maximum(state.t - jnp.int32(1), jnp.int32(0))
    beta_prev = lax.dynamic_index_in_dim(state.beta, t_idx, axis=0, keepdims=False)
    logz_prev = lax.dynamic_index_in_dim(state.logz, t_idx, axis=0, keepdims=False)

    # the largest beta in this step is 1
    beta_one = jnp.asarray(1.0, dtype=beta_prev.dtype)

    # evaluate chosen metric at previous beta and at beta = 1
    _, m_prev, _, _ = _weights_metric_logz(state, beta_prev, metric_id, n_active)
    _, m_one, _, _ = _weights_metric_logz(state, beta_one, metric_id, n_active)
    
    # build target value and tolerance for the metric
    target = jnp.asarray(n_effective, dtype=m_prev.dtype)
    tol = jnp.asarray(0.01, dtype=m_prev.dtype) * target

    # decide whether to keep beta_prev, jump to 1, or bisect
    c0 = m_prev <= target
    c1 = (~c0) & (m_one >= target)
    cid = jnp.where(c0, jnp.int32(0), jnp.where(c1, jnp.int32(1), jnp.int32(2)))

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

    # compute weights and logz for chosen beta
    w_full, ess_est, logz_new, _ = _weights_metric_logz(state, beta, metric_id, n_active)
    logz = jnp.where(cid == jnp.int32(0), logz_prev, logz_new)

    # optionally update target effective size
    n_eff_new = lax.cond(
        dynamic,
        lambda ne: _dynamic_neff(ne, w_full, n_active, jnp.asarray(dynamic_ratio, w_full.dtype)),
        lambda ne: jnp.asarray(ne, dtype=jnp.int32),
        n_effective,
    )

    # build sample indices for the full flattened particle set
    n_tot = w_full.shape[0]
    samples = jnp.arange(n_tot, dtype=jnp.int32)

    # trim weights before selecting top particles
    mask_trim, w_trim, thr, ratio, _ = trim_weights_jax(
        samples=samples,
        weights=w_full,
        ess=jnp.asarray(trim_ess, dtype=w_full.dtype),
        bins=bins,
    )

    # read flattened history sizes
    T, N = state.logl.shape
    D = state.u.shape[-1]
    B = state.blobs.shape[-1]

    # flatten all particle history arrays
    u_flat = state.u.reshape((T * N, D))
    x_flat = state.x.reshape((T * N, D))
    logdetj_flat = state.logdetj.reshape((T * N,))
    logl_flat = state.logl.reshape((T * N,))
    logp_flat = state.logp.reshape((T * N,))
    blobs_flat = state.blobs.reshape((T * N, B))

    # keep highest trimmed weights
    order = jnp.argsort(w_trim)
    start = jnp.int32(n_tot - keep_max)
    idx = lax.dynamic_slice_in_dim(order, start_index=start, slice_size=keep_max, axis=0)[::-1]
    
    # extract kept weights and mask out zeros
    w_keep = w_trim[idx]
    keep_mask = w_keep > jnp.asarray(0.0, w_keep.dtype)

    # renormalize kept weights
    wsum = jnp.sum(w_keep)
    wnorm = w_keep / jnp.where(wsum > 0, wsum, jnp.asarray(1.0, w_keep.dtype))
    wnorm = jnp.where(keep_mask, wnorm, jnp.asarray(0.0, wnorm.dtype))

    # gather kept particles and zero out dropped entries
    u_keep = jnp.where(keep_mask[:, None], u_flat[idx], jnp.asarray(0.0, u_flat.dtype))
    x_keep = jnp.where(keep_mask[:, None], x_flat[idx], jnp.asarray(0.0, x_flat.dtype))
    logdetj_keep = jnp.where(keep_mask, logdetj_flat[idx], jnp.asarray(0.0, logdetj_flat.dtype))
    logl_keep = jnp.where(keep_mask, logl_flat[idx], jnp.asarray(0.0, logl_flat.dtype))
    logp_keep = jnp.where(keep_mask, logp_flat[idx], jnp.asarray(0.0, logp_flat.dtype))
    blobs_keep = jnp.where(keep_mask[:, None], blobs_flat[idx], jnp.asarray(0.0, blobs_flat.dtype))

    # build current particle dictionary used by the next stages
    current_particles = {
        "u": u_keep,
        "x": x_keep,
        "logdetj": logdetj_keep,
        "logl": logl_keep,
        "logp": logp_keep,
        "blobs": blobs_keep,
        "logz": logz,
        "beta": beta,
        "weights": wnorm,
        "ess": ess_est,
        "idx": idx,
        "keep_mask": keep_mask,
        "trim_threshold": thr,
        "trim_ratio": ratio,
        "trim_mask_full": mask_trim,
    }

    # return summary dictionary with the particles
    stats = {"beta": beta, "logz": logz, "ess": ess_est, "n_effective": n_eff_new}

    return current_particles, n_eff_new, stats