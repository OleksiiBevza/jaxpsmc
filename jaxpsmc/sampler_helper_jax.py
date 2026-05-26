from __future__ import annotations

from .tools_jax import *
from .particles_jax import *

from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

from typing import Callable, Mapping, Tuple, Any, Optional, Dict, NamedTuple
from .scaler_jax import *
from .pcn_jax import *
from .dili_pcn_jax import dili_pcn_jax
from .li_pcn_jax import likelihood_informed_pcn_jax









#################################################################
# 1. REWEIGHT
#################################################################

METRIC_ESS = jnp.int32(0)
METRIC_USS = jnp.int32(1)

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
        "logdetj": jnp.where(mask_flat, logdetj_flat, jnp.asarray(0.0, logdetj_flat.dtype)),
        "logl": jnp.where(mask_flat, logl_flat, jnp.asarray(0.0, logl_flat.dtype)),
        "logp": jnp.where(mask_flat, logp_flat, jnp.asarray(0.0, logp_flat.dtype)),
        "blobs": jnp.where(mask_flat[:, None], blobs_flat, jnp.asarray(0.0, blobs_flat.dtype)),
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







#################################################################
# 2. RESAMPLE
#################################################################

_ECONVERGED = jnp.int32(0)
_EVALUEERR  = jnp.int32(-3)

@partial(jax.jit, static_argnames=("n_active", "reset_weights"))
def resample_particles_jax(current_particles, *, key, n_active: int, method_code: jnp.int32, reset_weights: bool = True):
    """
    Resamples particles from the current weighted particle set.

    The function draws n_active particles according to their weights.
    It supports multinomial resampling and systematic resampling.
    After resampling, weights can be reset to a uniform distribution.

    Parameters:
    -----------
    current_particles:
        dictionary with particle arrays and weights.
        Expected keys are "u", "x", "logdetj", "logl",
        "logp", "blobs", and "weights".
    key:
        JAX random key used for resampling.
    n_active:
        number of particles to draw.
        This is static for JAX compilation.
    method_code:
        integer code selecting the resampling method.
        0 means multinomial resampling.
        1 means systematic resampling.
    reset_weights:
        if True, reset output weights to uniform values.
        If False, keep the selected input weights.

    Returns:
    --------
    tuple:
        new_particles:
            dictionary with resampled particles and output weights.
        status:
            resampling status code.
            Zero means success.
            A negative value means invalid weights.
        key_out:
            updated JAX random key.
    """
    # read current weights and total number of stored particles
    w = jnp.asarray(current_particles["weights"])
    n_total = w.shape[0]

    def _multinomial(args):
        """
        Performs multinomial resampling.

        Each output index is drawn independently from the categorical
        distribution defined by the particle weights.

        Parameters:
        -----------
        args:
            tuple containing input key and weights.

        Returns:
        --------
        tuple:
            sampled indices, status code, and output key.
        """
        # unpack inputs and split the random key
        key_in, weights = args
        key_out, subkey = jax.random.split(key_in)

        # validate weights before sampling
        wsum = jnp.sum(weights)
        bad = (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0)

        # draw indices from categorical distribution when weights are valid
        logits = jnp.where(weights > 0, jnp.log(weights), -jnp.inf)
        idx_samp = jax.random.categorical(subkey, logits, shape=(n_active,), axis=0).astype(jnp.int32)

        # fall back to a simple repeating pattern when weights are invalid
        idx_fallback = (jnp.arange(n_active, dtype=jnp.int32) % jnp.int32(n_total)).astype(jnp.int32)

        # select real indices or the fallback indices
        idx = jnp.where(bad, idx_fallback, idx_samp)
        status = jnp.where(bad, _EVALUEERR, _ECONVERGED)
        return idx, status, key_out

    def _systematic(args):
        """
        Performs systematic resampling.

        Systematic resampling uses one random offset and an evenly spaced
        grid over the cumulative weight distribution. It usually has lower
        variance than multinomial resampling.

        Parameters:
        -----------
        args:
            tuple containing input key and weights.

        Returns:
        --------
        tuple:
            sampled indices, status code, and output key.
        """
        key_in, weights = args
        #idx, status, key_out = systematic_resample_jax(weights, key=key_in, size=n_active)
        idx, status, key_out = systematic_resample_jax_size(weights, key=key_in, size=n_active)
        return idx.astype(jnp.int32), status.astype(jnp.int32), key_out

    # select resampling method from integer code
    idx, status, key_out = lax.switch(
        method_code.astype(jnp.int32),
        (_multinomial, _systematic),
        (key, w),
    )
    
    # gather resampled particle arrays
    u_out       = jnp.take(current_particles["u"],       idx, axis=0)
    x_out       = jnp.take(current_particles["x"],       idx, axis=0)
    logdetj_out = jnp.take(current_particles["logdetj"], idx, axis=0)
    logl_out    = jnp.take(current_particles["logl"],    idx, axis=0)
    logp_out    = jnp.take(current_particles["logp"],    idx, axis=0)
    blobs_out   = jnp.take(current_particles["blobs"],   idx, axis=0)

    # reset weights to uniform or keep resampled weights
    w_res = jnp.take(w, idx, axis=0)
    w_uni = jnp.full((n_active,), jnp.asarray(1.0, w.dtype) / jnp.asarray(n_active, w.dtype), dtype=w.dtype)
    w_out = lax.cond(jnp.asarray(reset_weights), lambda _: w_uni, lambda _: w_res, operand=None)

    # build new particle dictionary
    new_particles = {
        "u": u_out,
        "x": x_out,
        "logdetj": logdetj_out,
        "logl": logl_out,
        "logp": logp_out,
        "weights": w_out,
        "blobs": blobs_out,
    }
    return new_particles, status, key_out









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
    beta_not_close = (jnp.asarray(1.0, dtype=beta_current.dtype) - beta_current) >= jnp.asarray(
        beta_tol, dtype=beta_current.dtype
    )
    ess_too_small = ess_or_uss < jnp.asarray(n_total, dtype=ess_or_uss.dtype)

    return jnp.logical_or(beta_not_close, ess_too_small)








#################################################################
# 5. POSTERIOR
#################################################################
_ECONVERGED = jnp.int32(0)
_EVALUEERR  = jnp.int32(-3)

@partial(jax.jit, static_argnames=("size",))
def _systematic_resample_impl(key, weights, size: int):
    """
    Runs the JIT-compiled core of systematic resampling.

    The function validates and normalizes the weights.
    It then builds a cumulative distribution and samples indices
    using one random offset and an evenly spaced grid.

    Parameters:
    -----------
    key:
        JAX random key used for the offset draw.
    weights:
        input weights, shape (K,).
        They should be finite, non-negative, and have positive sum.
    size:
        number of resampled indices to return.
        This is static for JAX compilation.

    Returns:
    --------
    tuple:
        idx:
            resampled indices, shape (size,).
            Invalid weights produce dummy index -1.
        status:
            status code.
            Zero means success.
            A negative value means invalid weights.
        key_out:
            updated JAX random key.
    """
    # convert weights to an array and dtype
    w = jnp.asarray(weights)
    dtype = jnp.result_type(w, jnp.float64)

    # validate and normalize weights 
    wsum = jnp.sum(w)
    bad = (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(w)) | jnp.any(w < 0)

    w_norm = w / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    # build CDF and force last value to 1 when weights are valid
    cdf = jnp.cumsum(w_norm)
    cdf = cdf / jnp.where(bad, jnp.asarray(1.0, dtype), cdf[-1])

    # draw one uniform offset for the whole systematic grid
    key_out, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(), dtype=dtype)  # scalar
    positions = (u + jnp.arange(size, dtype=dtype)) / jnp.asarray(size, dtype=dtype)

    # convert positions into CDF indices
    idx = jnp.searchsorted(cdf, positions, side="left")  # :contentReference[oaicite:2]{index=2}
    idx = jnp.clip(idx, 0, w.shape[0] - 1).astype(jnp.int32)

    # mark invalid weights with a failure code and dummy indices
    idx = jnp.where(bad, jnp.full((size,), jnp.int64(-1)), idx)
    status = jnp.where(bad, _EVALUEERR, _ECONVERGED)

    return idx, status, key_out




class PosteriorOut(NamedTuple):
    """
    Stores fixed-shape posterior outputs.

    This object contains the flattened posterior sample arrays.
    It also contains importance weights, trimming diagnostics,
    optional resampled outputs, and the final evidence estimate.

    The arrays keep fixed shapes because this is useful for JAX.
    Masks show which entries are valid and which entries were kept
    after trimming.

    Parameters:
    -----------
    samples:
        flattened posterior samples, shape (K, D).
    logl:
        flattened log-likelihood values, shape (K,).
    logp:
        flattened log-prior values, shape (K,).
    blobs:
        flattened extra likelihood outputs, shape (K, B).
    mask_valid:
        Boolean mask for filled history entries, shape (K,).
    weights:
        normalized importance weights over kept entries, shape (K,).
        Dropped or invalid entries have weight zero.
    logw:
        log of the kept weights, shape (K,).
        Entries with zero weight have value -inf.
    mask_trim:
        Boolean mask for entries kept after trimming, shape (K,).
    threshold:
        trimming threshold used for the weights.
    ess_ratio:
        ESS ratio achieved by the trimming step.
    i_final:
        final scan index used by the trimming routine.
    idx_resampled:
        posterior resampling indices, shape (K,).
    resample_status:
        status code from posterior resampling.
    samples_resampled:
        resampled posterior samples, shape (K, D).
    logl_resampled:
        resampled log-likelihood values, shape (K,).
    logp_resampled:
        resampled log-prior values, shape (K,).
    blobs_resampled:
        resampled blob values, shape (K, B).
    logz_new:
        final log evidence estimate.
    key_out:
        updated JAX random key after optional resampling.

    Returns:
    --------
    PosteriorOut:
        posterior samples, weights, diagnostics, and optional resamples.
    """
    # flattened, fixed-size (T_max * N) arrays
    samples: jax.Array              # (K, D)
    logl: jax.Array                 # (K,)
    logp: jax.Array                 # (K,)
    blobs: jax.Array                # (K, B)
    mask_valid: jax.Array           # (K,) bool (True for filled steps)

    # importance weights 
    weights: jax.Array              # (K,) normalized over kept entries and zeros where dropped/invalid
    logw: jax.Array                 # (K,) log(weights); -inf where weights==0
    mask_trim: jax.Array            # (K,) bool
    threshold: jax.Array            # scalar
    ess_ratio: jax.Array            # scalar
    i_final: jax.Array              # scalar int32

    # optional resampling
    idx_resampled: jax.Array        # (K,) int32
    resample_status: jax.Array      # scalar int64 (0 ok; nonzero indicates invalid weights in systematic)
    samples_resampled: jax.Array    # (K, D)
    logl_resampled: jax.Array       # (K,)
    logp_resampled: jax.Array       # (K,)
    blobs_resampled: jax.Array      # (K, B)

    # evidence (from compute_logw_and_logz_jax) 
    logz_new: jax.Array             # scalar
    key_out: jax.Array              # PRNGKey


@partial(jax.jit, static_argnames=("bins",))
def trim_weights_scan_jax(
    weights: jax.Array,
    ess: float | jax.Array = 0.99,
    bins: int = 1000,
):
    """
    Trims importance weights by scanning percentile thresholds.

    The function searches for a weight threshold.
    Weights below the threshold are dropped.
    The threshold is chosen so the retained weights preserve
    at least the requested ESS ratio.

    This is useful when many tiny weights add cost but little value.
    The output weights are renormalized over the kept entries.

    Parameters:
    -----------
    weights:
        input importance weights, shape (K,).
        They should be finite, non-negative, and have positive sum.
    ess:
        target ratio between trimmed ESS and full ESS.
        Values close to one keep more particles.
    bins:
        number of percentile grid points to scan.
        This is static for JAX compilation.

    Returns:
    --------
    tuple:
        mask:
            Boolean mask showing which weights are kept.
        w_trim:
            trimmed and normalized weights.
        threshold:
            selected weight threshold.
        ratio:
            achieved ESS ratio after trimming.
        i_final:
            selected percentile-grid index.
    """
    # convert inputs to arrays with a common dtype
    w = jnp.asarray(weights)
    dtype = w.dtype
    ess = jnp.asarray(ess, dtype=dtype)

    # validate and normalize weights
    wsum = jnp.sum(w)
    bad = (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(w)) | jnp.any(w < 0)

    # normalize 
    w = w / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    # compute ESS of the full normalized weights
    ess_total = 1.0 / jnp.sum(w * w)

    # build percentile grid and sorted weights
    percentiles = jnp.linspace(jnp.asarray(0.0, dtype), jnp.asarray(99.0, dtype), bins)
    sorted_w = jnp.sort(w)

    # precompute constants used by percentile interpolation
    n = w.shape[0]
    n_minus_1 = jnp.asarray(n - 1, dtype)

    def ratio_for_i(i: jax.Array):
        """
        Computes the trimming threshold and ESS ratio for one grid index.

        The index selects a percentile.
        The percentile gives a threshold.
        The function then computes the ESS of the weights above that threshold.

        Parameters:
        -----------
        i:
            percentile grid index.

        Returns:
        --------
        tuple:
            threshold and achieved ESS ratio.
        """
        # define percentile and convert it to a fraction
        p = lax.dynamic_index_in_dim(percentiles, i, axis=0, keepdims=False)
        frac = p / jnp.asarray(100.0, dtype)

        # interpolate percentile value from sorted weights
        pos = frac * n_minus_1
        lo = jnp.floor(pos).astype(jnp.int32)
        hi = jnp.minimum(lo + 1, jnp.int32(n - 1))
        alpha = pos - lo.astype(dtype)

        w_lo = sorted_w[lo]
        w_hi = sorted_w[hi]
        threshold = (1.0 - alpha) * w_lo + alpha * w_hi

        # keep only weights above the threshold
        mask = w >= threshold
        w_kept = jnp.where(mask, w, 0.0)

        # compute ESS of the kept and renormalized weights
        kept_sum = jnp.sum(w_kept)
        kept_sumsq = jnp.sum(w_kept * w_kept)

        # ESS of normalized  weights that are kept:
        # w_trim = w_kept / kept_sum is sum(w_trim^2) = kept_sumsq / kept_sum^2
        kept_sum_safe = jnp.where(kept_sum > 0, kept_sum, jnp.asarray(1.0, dtype))
        ess_trim = (kept_sum_safe * kept_sum_safe) / jnp.where(kept_sumsq > 0, kept_sumsq, jnp.asarray(jnp.inf, dtype))

        # return threshold and ESS ratio
        ratio = ess_trim / ess_total
        return threshold, ratio

    # scan percentile grid from high to low: 
    # i from bins-1 down to 0 and pick first i with ratio >= ess
    idxs = jnp.arange(bins - 1, -1, -1, dtype=jnp.int32)

    def scan_step(carry, i):
        """
        Updates the selected trimming index during the scan.

        The scan moves from high percentiles to low percentiles.
        It stores the first index that achieves the requested ESS ratio.

        Parameters:
        -----------
        carry:
            tuple with found flag and current best index.
        i:
            current percentile grid index.

        Returns:
        --------
        tuple:
            updated carry and ESS ratio at this index.
        """
        # unpack scan state and evaluate current index
        found, i_best = carry
        _, r = ratio_for_i(i)
        # save first index whose ratio reaches target
        update = (~found) & (r >= ess)
        found2 = found | update
        i_best2 = jnp.where(update, i, i_best)
        return (found2, i_best2), r

    # run scan over all percentile indices
    (found_final, i_final), _ = lax.scan(scan_step, (jnp.asarray(False), jnp.asarray(0, jnp.int32)), idxs)

    # rebuild trimming result from chosen index
    # if not found, i_final = 0
    threshold, ratio = ratio_for_i(i_final)
    mask = w >= threshold

    # build trimmed and renormalized weights
    w_kept = jnp.where(mask, w, 0.0)
    kept_sum = jnp.sum(w_kept)
    kept_sum_safe = jnp.where(kept_sum > 0, kept_sum, jnp.asarray(1.0, dtype))
    w_trim = jnp.where(mask, w_kept / kept_sum_safe, 0.0)

    # works with invalid-input behavior 
    mask = jnp.where(bad, jnp.zeros_like(mask), mask)
    w_trim = jnp.where(bad, jnp.full_like(w_trim, jnp.nan), w_trim)
    threshold = jnp.where(bad, jnp.asarray(jnp.nan, dtype), threshold)
    ratio = jnp.where(bad, jnp.asarray(jnp.nan, dtype), ratio)

    return mask, w_trim, threshold, ratio, i_final


@partial(jax.jit, static_argnames=("bins_trim",))
def posterior_jax(
    state: ParticlesState,
    key: jax.Array,
    *,
    do_resample: bool | jax.Array = False,
    resample_method: int | jax.Array = 1,  # 1=syst, 0=mult
    trim_importance_weights: bool | jax.Array = True,
    ess_trim: float | jax.Array = 0.99,
    bins_trim: int = 1000,
    beta_final: float | jax.Array = 1.0,
) -> PosteriorOut:
    """
    Builds posterior arrays from the stored particle history.

    The function flattens all recorded samples into one posterior array.
    It computes final importance weights at beta_final.
    It can trim small weights to reduce the effective output set.
    It can also produce a resampled posterior sample.

    The returned object keeps fixed shapes.
    Invalid or inactive entries are marked by masks and zeroed arrays.

    Parameters:
    -----------
    state:
        particle history state.
    key:
        JAX random key used if posterior resampling is enabled.
    do_resample:
        Boolean flag.
        If True, draw resampled posterior particles.
        If False, keep identity indices.
    resample_method:
        integer code for posterior resampling.
        1 means systematic resampling.
        0 means multinomial resampling.
    trim_importance_weights:
        Boolean flag.
        If True, trim low-weight entries before posterior output.
        If False, keep all valid entries.
    ess_trim:
        ESS ratio target used by trimming.
        Values close to one keep more weighted particles.
    bins_trim:
        number of percentile bins used by trimming.
        This is static for JAX compilation.
    beta_final:
        final beta value used to compute posterior weights.
        Usually this is 1.0.

    Returns:
    --------
    PosteriorOut:
        fixed-shape posterior samples, weights, masks,
        optional resampled arrays, evidence estimate, and output key.
    """
    # flatten history arrays into one sample axis
    T, N, D = state.x.shape
    K = T * N

    samples = state.x.reshape((K, D))
    logl = state.logl.reshape((K,))
    logp = state.logp.reshape((K,))
    blobs = state.blobs.reshape((K, state.blobs.shape[-1]))

    # compute normalized log-weights (normalized), evidence, and valid-entry mask
    logw0, logz_new, mask_valid = compute_logw_and_logz_jax(state, beta_final=beta_final, normalize=True)
    w0 = jnp.exp(logw0)

    # zero out entries that are not part of filled history
    samples = jnp.where(mask_valid[:, None], samples, jnp.zeros_like(samples))
    logl = jnp.where(mask_valid, logl, jnp.zeros_like(logl))
    logp = jnp.where(mask_valid, logp, jnp.zeros_like(logp))
    blobs = jnp.where(mask_valid[:, None], blobs, jnp.zeros_like(blobs))

    # convert trim flag to a JAX boolean
    trim_flag = jnp.asarray(trim_importance_weights, dtype=bool)

    def _do_trim(_):
        """
        Applies trimming to the posterior importance weights.

        The trimming routine selects a threshold.
        Invalid history entries are removed from the trim mask.
        The remaining weights are renormalized.

        Parameters:
        -----------
        _:
            unused operand required by lax.cond.

        Returns:
        --------
        tuple:
            trim mask, normalized trimmed weights,
            threshold, ESS ratio, and trim index.
        """
        # run trimming scan on the full weight vector      
        mask_trim, w_trim, thr, ratio, i_final = trim_weights_scan_jax(
            w0, ess=ess_trim, bins=bins_trim
        )
        # do not keep invalid history entries
        mask_trim = mask_trim & mask_valid
        # make sure trimmed weights are normalized over and keep valid entries only
        w_trim = jnp.where(mask_trim, w_trim, 0.0)
        # safe renormalization (if nothing is kept, then all zeros)
        s = jnp.sum(w_trim)
        s_safe = jnp.where(s > 0, s, jnp.asarray(1.0, w_trim.dtype))
        w_trim = jnp.where(mask_trim, w_trim / s_safe, 0.0)
        return mask_trim, w_trim, thr, ratio, i_final

    def _no_trim(_):
        """
        Keeps all valid posterior importance weights.

        This branch is used when trimming is disabled.
        It keeps the normalized weights from compute_logw_and_logz_jax.

        Parameters:
        -----------
        _:
            unused operand required by lax.cond.

        Returns:
        --------
        tuple:
            valid mask, untrimmed weights, default threshold,
            default ESS ratio, and default index.
        """
        # keep all valid entries and leave normalized weights unchanged
        mask_trim = mask_valid
        w_trim = jnp.where(mask_trim, w0, 0.0)
        # w0 already sums to 1 over valid entries (invalid are 0)
        thr = jnp.asarray(-jnp.inf, w0.dtype)
        ratio = jnp.asarray(1.0, w0.dtype)
        i_final = jnp.asarray(-1, jnp.int32)
        return mask_trim, w_trim, thr, ratio, i_final

    # choose trimmed or untrimmed weights
    mask_trim, weights, threshold, ess_ratio, i_final = lax.cond(trim_flag, _do_trim, _no_trim, operand=None)
    logw = jnp.log(weights)  # -inf where weights==0

    # convert resampling options to JAX arrays
    do_resample_arr = jnp.asarray(do_resample, dtype=bool)
    resample_method = jnp.asarray(resample_method)

    def _resample(key_in):
        """
        Resamples posterior particles from the final weights.

        The branch chooses systematic or multinomial resampling.
        It returns a full set of K resampled indices.

        Parameters:
        -----------
        key_in:
            input JAX random key.

        Returns:
        --------
        tuple:
            resampled indices, status code, and output key.
        """
        
        def _syst(k):
            """
            Applies systematic posterior resampling.

            Parameters:
            -----------
            k:
                input JAX random key.

            Returns:
            --------
            tuple:
                resampled indices, status code, and output key.
            """
            # run systematic resampler on posterior weights
            idx, status, k_out = _systematic_resample_impl(k, weights, size=K)
            return idx.astype(jnp.int32), status.astype(jnp.int64), k_out
       
        def _mult(k):
            """
            Applies multinomial posterior resampling.

            Parameters:
            -----------
            k:
                input JAX random key.

            Returns:
            --------
            tuple:
                resampled indices, zero status code, and output key.
            """
            # draw indices directly from posterior weight vector
            k_out, sub = jax.random.split(k)
            idx = jax.random.choice(sub, a=K, shape=(K,), replace=True, p=weights)
            status = jnp.asarray(0, jnp.int64)
            return idx.astype(jnp.int32), status, k_out

        # choose systematic or multinomial resampling
        use_syst = resample_method == jnp.asarray(1, resample_method.dtype)
        return lax.cond(use_syst, _syst, _mult, key_in)

    def _no_resample(key_in):
        """
        Skips posterior resampling.

        The output indices are the identity order.
        The random key is returned unchanged.

        Parameters:
        -----------
        key_in:
            input JAX random key.

        Returns:
        --------
        tuple:
            identity indices, zero status code, and unchanged key.
        """
        # keep original order when resampling is disabled
        idx = jnp.arange(K, dtype=jnp.int32)
        status = jnp.asarray(0, jnp.int64)
        return idx, status, key_in

    # either resample posterior or keep it unchanged
    idx_resampled, resample_status, key_out = lax.cond(do_resample_arr, _resample, _no_resample, key)

    # clip indices (needed for array gathering)
    idx_safe = jnp.clip(idx_resampled, 0, K - 1)

    # build optional resampled outputs
    samples_res = samples[idx_safe]
    logl_res = logl[idx_safe]
    logp_res = logp[idx_safe]
    blobs_res = blobs[idx_safe]

    return PosteriorOut(
        samples=samples,
        logl=logl,
        logp=logp,
        blobs=blobs,
        mask_valid=mask_valid,
        weights=weights,
        logw=logw,
        mask_trim=mask_trim,
        threshold=threshold,
        ess_ratio=ess_ratio,
        i_final=i_final,
        idx_resampled=idx_resampled,
        resample_status=resample_status,
        samples_resampled=samples_res,
        logl_resampled=logl_res,
        logp_resampled=logp_res,
        blobs_resampled=blobs_res,
        logz_new=logz_new,
        key_out=key_out,
    )



















