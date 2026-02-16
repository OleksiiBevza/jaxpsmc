from __future__ import annotations

from tools_jax import *
from particles_jax import *

from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

from typing import Callable, Mapping, Tuple, Any, Optional, Dict, NamedTuple
from scaler_jax import inverse_jax, forward_jax, apply_boundary_conditions_x_jax
from pcn_jax import preconditioned_pcn_jax











#################################################################
# 1. reweight part
#################################################################

METRIC_ESS = jnp.int32(0)
METRIC_USS = jnp.int32(1)

def _metric_value(weights, metric_id, n_active):
    return lax.cond(
        metric_id == METRIC_ESS,
        lambda w: effective_sample_size_jax(w),
        lambda w: unique_sample_size_jax(w, k=n_active),
        weights,
    )


def _weights_metric_logz(state, beta, metric_id, n_active):
    logw_flat, logz_new, mask_flat = compute_logw_and_logz_jax(
        state, beta_final=beta, normalize=False
    )
    logw_flat = jnp.where(mask_flat, logw_flat, -jnp.inf)
    w_full = jax.nn.softmax(logw_flat)
    m_val = _metric_value(w_full, metric_id, n_active)
    return w_full, m_val, logz_new, logw_flat


def _bisect_beta_scan(state, lo, hi, target, metric_id, n_active, steps, tol):
    dtype = jnp.asarray(lo).dtype

    def scan_step(carry, _):
        lo_c, hi_c, done_c, beta_c = carry
        mid = (lo_c + hi_c) * jnp.asarray(0.5, dtype)

        _, m_mid, _, _ = _weights_metric_logz(state, mid, metric_id, n_active)

        close = jnp.abs(m_mid - target) <= tol
        done2 = done_c | close

        hi2 = jnp.where((~done2) & (m_mid < target), mid, hi_c)
        lo2 = jnp.where((~done2) & (m_mid >= target), mid, lo_c)
        beta2 = jnp.where((~done_c) & close, mid, beta_c)

        return (lo2, hi2, done2, beta2), None

    beta0 = (lo + hi) * jnp.asarray(0.5, dtype)
    carry0 = (lo, hi, jnp.asarray(False), beta0)

    (lo_f, hi_f, done_f, beta_f), _ = lax.scan(
        scan_step,
        carry0,
        xs=jnp.arange(steps, dtype=jnp.int32),
    )

    mid_f = (lo_f + hi_f) * jnp.asarray(0.5, dtype)
    return jnp.where(done_f, beta_f, mid_f)


def _dynamic_neff(n_eff, weights_full, n_active, ratio):
    n_eff_f = jnp.asarray(n_eff, dtype=weights_full.dtype)
    n_act_f = jnp.asarray(n_active, dtype=weights_full.dtype)
    nuniq = unique_sample_size_jax(weights_full, k=n_active)

    low = n_act_f * (jnp.asarray(0.95, n_eff_f.dtype) * ratio)
    high = n_act_f * jnp.minimum(
        jnp.asarray(1.05, n_eff_f.dtype) * ratio,
        jnp.asarray(1.0, n_eff_f.dtype),
    )

    eps = jnp.asarray(1e-12, n_eff_f.dtype)
    down = (n_act_f / (nuniq + eps)) * n_eff_f
    up = ((nuniq + eps) / n_act_f) * n_eff_f

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
    t_idx = jnp.maximum(state.t - jnp.int32(1), jnp.int32(0))
    beta_prev = lax.dynamic_index_in_dim(state.beta, t_idx, axis=0, keepdims=False)
    logz_prev = lax.dynamic_index_in_dim(state.logz, t_idx, axis=0, keepdims=False)

    beta_one = jnp.asarray(1.0, dtype=beta_prev.dtype)

    _, m_prev, _, _ = _weights_metric_logz(state, beta_prev, metric_id, n_active)
    _, m_one, _, _ = _weights_metric_logz(state, beta_one, metric_id, n_active)

    target = jnp.asarray(n_effective, dtype=m_prev.dtype)
    tol = jnp.asarray(0.01, dtype=m_prev.dtype) * target

    c0 = m_prev <= target
    c1 = (~c0) & (m_one >= target)
    cid = jnp.where(c0, jnp.int32(0), jnp.where(c1, jnp.int32(1), jnp.int32(2)))

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

    beta = lax.switch(
        cid,
        (
            lambda _: beta_prev,
            lambda _: beta_one,
            lambda _: beta_bis,
        ),
        operand=None,
    )

    w_full, ess_est, logz_new, _ = _weights_metric_logz(state, beta, metric_id, n_active)
    logz = jnp.where(cid == jnp.int32(0), logz_prev, logz_new)

    n_eff_new = lax.cond(
        dynamic,
        lambda ne: _dynamic_neff(ne, w_full, n_active, jnp.asarray(dynamic_ratio, w_full.dtype)),
        lambda ne: jnp.asarray(ne, dtype=jnp.int32),
        n_effective,
    )

    n_tot = w_full.shape[0]
    samples = jnp.arange(n_tot, dtype=jnp.int32)

    mask_trim, w_trim, thr, ratio, _ = trim_weights_jax(
        samples=samples,
        weights=w_full,
        ess=jnp.asarray(trim_ess, dtype=w_full.dtype),
        bins=bins,
    )

    T, N = state.logl.shape
    D = state.u.shape[-1]
    B = state.blobs.shape[-1]

    u_flat = state.u.reshape((T * N, D))
    x_flat = state.x.reshape((T * N, D))
    logdetj_flat = state.logdetj.reshape((T * N,))
    logl_flat = state.logl.reshape((T * N,))
    logp_flat = state.logp.reshape((T * N,))
    blobs_flat = state.blobs.reshape((T * N, B))

    order = jnp.argsort(w_trim)
    start = jnp.int32(n_tot - keep_max)
    idx = lax.dynamic_slice_in_dim(order, start_index=start, slice_size=keep_max, axis=0)[::-1]

    w_keep = w_trim[idx]
    keep_mask = w_keep > jnp.asarray(0.0, w_keep.dtype)

    wsum = jnp.sum(w_keep)
    wnorm = w_keep / jnp.where(wsum > 0, wsum, jnp.asarray(1.0, w_keep.dtype))
    wnorm = jnp.where(keep_mask, wnorm, jnp.asarray(0.0, wnorm.dtype))

    u_keep = jnp.where(keep_mask[:, None], u_flat[idx], jnp.asarray(0.0, u_flat.dtype))
    x_keep = jnp.where(keep_mask[:, None], x_flat[idx], jnp.asarray(0.0, x_flat.dtype))
    logdetj_keep = jnp.where(keep_mask, logdetj_flat[idx], jnp.asarray(0.0, logdetj_flat.dtype))
    logl_keep = jnp.where(keep_mask, logl_flat[idx], jnp.asarray(0.0, logl_flat.dtype))
    logp_keep = jnp.where(keep_mask, logp_flat[idx], jnp.asarray(0.0, logp_flat.dtype))
    blobs_keep = jnp.where(keep_mask[:, None], blobs_flat[idx], jnp.asarray(0.0, blobs_flat.dtype))

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

    stats = {"beta": beta, "logz": logz, "ess": ess_est, "n_effective": n_eff_new}

    return current_particles, n_eff_new, stats






#################################################################
# 2. resample part
#################################################################

_ECONVERGED = jnp.int32(0)
_EVALUEERR  = jnp.int32(-3)

@partial(jax.jit, static_argnames=("n_active", "reset_weights"))
def resample_particles_jax(current_particles, *, key, n_active: int, method_code: jnp.int32, reset_weights: bool = True):
    """
    Pure JAX resampling.
    Inputs:
      current_particles: dict with keys
        "u","x","logdetj","logl","logp","weights","blobs"
        (If no blobs, set them to a dummy array with shape (N, 0).)
      key: jax.random.PRNGKey
      n_active: int (static)
      method_code: 0 -> multinomial, 1 -> systematic
      reset_weights: if True, set weights to uniform after resampling

    Returns:
      new_particles (dict), status (int32), key_out
    """
    w = jnp.asarray(current_particles["weights"])
    n_total = w.shape[0]

    def _multinomial(args):
        key_in, weights = args
        key_out, subkey = jax.random.split(key_in)

        wsum = jnp.sum(weights)
        bad = (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0)

        logits = jnp.where(weights > 0, jnp.log(weights), -jnp.inf)
        idx_samp = jax.random.categorical(subkey, logits, shape=(n_active,), axis=0).astype(jnp.int32)
        idx_fallback = (jnp.arange(n_active, dtype=jnp.int32) % jnp.int32(n_total)).astype(jnp.int32)

        idx = jnp.where(bad, idx_fallback, idx_samp)
        status = jnp.where(bad, _EVALUEERR, _ECONVERGED)
        return idx, status, key_out

    def _systematic(args):
        key_in, weights = args
        #idx, status, key_out = systematic_resample_jax(weights, key=key_in, size=n_active)
        idx, status, key_out = systematic_resample_jax_size(weights, key=key_in, size=n_active)
        return idx.astype(jnp.int32), status.astype(jnp.int32), key_out

    idx, status, key_out = lax.switch(
        method_code.astype(jnp.int32),
        (_multinomial, _systematic),
        (key, w),
    )

    u_out       = jnp.take(current_particles["u"],       idx, axis=0)
    x_out       = jnp.take(current_particles["x"],       idx, axis=0)
    logdetj_out = jnp.take(current_particles["logdetj"], idx, axis=0)
    logl_out    = jnp.take(current_particles["logl"],    idx, axis=0)
    logp_out    = jnp.take(current_particles["logp"],    idx, axis=0)
    blobs_out   = jnp.take(current_particles["blobs"],   idx, axis=0)

    w_res = jnp.take(w, idx, axis=0)
    w_uni = jnp.full((n_active,), jnp.asarray(1.0, w.dtype) / jnp.asarray(n_active, w.dtype), dtype=w.dtype)
    w_out = lax.cond(jnp.asarray(reset_weights), lambda _: w_uni, lambda _: w_res, operand=None)

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
# 3. mutate part
#################################################################

Array = jax.Array


# ---------------------------------------------------------------------
# 1) JAX-native _log_like (SINGLE walker)
#    CHECK IF matches preconditioned_pcn_jax expectation:
#      loglike_fn(x_i: (D,)) -> (scalar_logl, blob_i: (B...))
# ---------------------------------------------------------------------
def _log_like(
    x_i: Array,
    loglike_single_fn: Callable[[Array], Tuple[Array, Array]],
) -> Tuple[Array, Array]:
    """
    Pure JAX log-likelihood wrapper for a SINGLE particle.

    Requirements:
      - x_i has shape (D,)
      - loglike_single_fn(x_i) returns (logl_i, blob_i) with fixed blob shape (B...)
        (If no blobs, use blob_i with shape (0,) and keep blobs as (N, 0))
    """
    
    return loglike_single_fn(x_i)



_log_like_batched = jax.vmap(_log_like, in_axes=(0, None), out_axes=(0, 0))


# ---------------------------------------------------------------------
# 2) JAX-native _mutate 
#    CHECK IF works nicely with preconditioned_pcn_jax 
# ---------------------------------------------------------------------
def mutate(
    key: Array,
    current_particles: Dict[str, Array],
    *,
    # as a BOOLEAN 
    use_preconditioned_pcn: Array,  # scalar bool (jnp.bool_ or python bool ok)

    # functions required by preconditioned_pcn_jax
    loglike_single_fn: Callable[[Array], Tuple[Array, Array]],
    logprior_fn: Callable[[Array], Array],
    flow: Any,
    scaler_cfg: Mapping[str, Array],
    scaler_masks: Mapping[str, Array],

    # geometry (Student-t)
    geom_mu: Array,    # (D,)
    geom_cov: Array,   # (D, D)
    geom_nu: Array,    # scalar

    # choose from
    n_max: int,
    n_steps: int,
    condition: Optional[Array] = None,
) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
    """
    Pure JAX mutate step.

    Expected current_particles keys (ALL should be as input and also fixed PyTree structure):
      - "u":            (N, D)
      - "x":            (N, D)
      - "logdetj":      (N,)
      - "logl":         (N,)
      - "logp":         (N,)
      - "logdetj_flow": (N,)      (can be zeros initially cause pcn_jax overwrites it)
      - "blobs":        (N, B...) (use shape (N, 0) if no blobs)
      - "beta":         () scalar
      - "proposal_scale": () scalar
      - "calls":        () int scalar

    Returns:
      new_key, new_particles, info_dict
    """
    u = current_particles["u"]
    n_dim = u.shape[1]
    norm_ref = jnp.asarray(2.38, dtype=u.dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=u.dtype))

    # args for cond without Python branching
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

    def _do_pcn(op):
        (
            key0, u0, x0, logdetj0, logl0, logp0, logdetj_flow0, blobs0, beta0, proposal_scale0
        ) = op

        # preconditioned_pcn_jax expects a SINGLE-walker loglike_fn;
        # it vmaps internally so  pass _log_like partially will be used... LALALA CHECK IF SO 
        def loglike_fn_single(x_i: Array) -> Tuple[Array, Array]:
            return _log_like(x_i, loglike_single_fn)

        out = preconditioned_pcn_jax(
            key0,
            u=u0,
            x=x0,
            logdetj=logdetj0,
            logl=logl0,
            logp=logp0,
            logdetj_flow=logdetj_flow0,
            blobs=blobs0,
            beta=beta0,
            loglike_fn=loglike_fn_single,
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
            condition=condition,
        )
        return out

    def _do_noop(op):
        # CHECK IF SAME output structure as preconditioned_pcn_jax
        (
            key0, u0, x0, logdetj0, logl0, logp0, logdetj_flow0, blobs0, _beta0, proposal_scale0
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

    results = jax.lax.cond(
        jnp.asarray(use_preconditioned_pcn),
        _do_pcn,
        _do_noop,
        payload,
    )

    new_calls = current_particles["calls"] + results["calls"]
    new_proposal_scale = results["proposal_scale"]

    # new particles dict CHECK IF KEYS ARE FIXED
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
        "efficiency": results["efficiency"] / norm_ref,    # CHECK HERE ALSO
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
    beta_current: Array,         # scalar, e.g. current_particles["beta"]
    n_total: Array,              # scalar threshold (ESS/USS target)
    metric_code: Array,          # int scalar: 0 -> ESS, 1 -> USS
    n_active: Array,             # int scalar (used as k for USS)
    beta_tol: Array = jnp.asarray(1e-4),
) -> Array:
    """
    Returns True if we walk (i.e., "not terminated"),
    
        (1 - beta) >= 1e-4  OR  ess < n_total

    Notes:
      - Uses compute_logw_and_logz_jax(state, beta_final=1.0) to get log-weights
      - Builds positive weights from log-weights 
      - Chooses ESS vs USS via lax.cond (instead of if/elif)
    """
    # logw_flat: (T*N,), mask_flat: (T*N,)
    logw_flat, _, mask_flat = compute_logw_and_logz_jax(
        state, beta_final=jnp.asarray(1.0, dtype=state.logl.dtype), normalize=False
    )

    # keep only valid entries; invalid -> -inf CHECK IF IT IS DOING SO
    logw_valid = jnp.where(mask_flat, logw_flat, -jnp.inf)

    # stable exponentiation CHECK IF IT avoids (-inf) - (-inf) => NaN by using m_safe
    m = jnp.max(logw_valid)
    m_safe = jnp.where(jnp.isfinite(m), m, jnp.asarray(0.0, dtype=logw_flat.dtype))

    weights = jnp.where(
        mask_flat,
        jnp.exp(logw_valid - m_safe),
        jnp.asarray(0.0, dtype=logw_flat.dtype),
    )

    # n_active is a JAX int scalar for unique_sample_size_jax(k=...)
    n_active_i32 = jnp.asarray(n_active, dtype=jnp.int32)

    # metric_code: 0 => ESS, 1 => USS
    ess_or_uss = lax.cond(
        jnp.asarray(metric_code, dtype=jnp.int32) == jnp.int32(0),
        lambda w: effective_sample_size_jax(w),
        lambda w: unique_sample_size_jax(w, k=n_active_i32),
        weights,
    )

    beta_not_close = (jnp.asarray(1.0, dtype=beta_current.dtype) - beta_current) >= jnp.asarray(
        beta_tol, dtype=beta_current.dtype
    )
    ess_too_small = ess_or_uss < jnp.asarray(n_total, dtype=ess_or_uss.dtype)

    return jnp.logical_or(beta_not_close, ess_too_small)








#################################################################
# 5. posterior part 
#################################################################
_ECONVERGED = jnp.int32(0)
_EVALUEERR  = jnp.int32(-3)

@partial(jax.jit, static_argnames=("size",))
def _systematic_resample_impl(key, weights, size: int):
    """
    JIT-compiled systematic resampling core.
    Returns: indices (size,), status (scalar int32), key_out
    """
    w = jnp.asarray(weights)
    dtype = jnp.result_type(w, jnp.float64)

    # validate and normalize weights 
    wsum = jnp.sum(w)
    bad = (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(w)) | jnp.any(w < 0)

    w_norm = w / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    # CDF (ensure last element is 1.0)
    cdf = jnp.cumsum(w_norm)
    cdf = cdf / jnp.where(bad, jnp.asarray(1.0, dtype), cdf[-1])

    # a single uniform u in [0, 1)
    key_out, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(), dtype=dtype)  # scalar
    positions = (u + jnp.arange(size, dtype=dtype)) / jnp.asarray(size, dtype=dtype)

    # find indices via jnp.searchsorted 
    idx = jnp.searchsorted(cdf, positions, side="left")  # :contentReference[oaicite:2]{index=2}
    idx = jnp.clip(idx, 0, w.shape[0] - 1).astype(jnp.int32)

    # on invalid weights, return indices and status
    idx = jnp.where(bad, jnp.full((size,), jnp.int64(-1)), idx)
    status = jnp.where(bad, _EVALUEERR, _ECONVERGED)

    return idx, status, key_out






class PosteriorOut(NamedTuple):
    # flattened, fixed-size (T_max * N) arrays
    samples: jax.Array        # (K, D)
    logl: jax.Array           # (K,)
    logp: jax.Array           # (K,)
    blobs: jax.Array          # (K, B)
    mask_valid: jax.Array     # (K,) bool (True for filled steps)

    # mmportance weights 
    weights: jax.Array        # (K,) normalized over kept entries and zeros where dropped/invalid
    logw: jax.Array           # (K,) log(weights); -inf where weights==0
    mask_trim: jax.Array      # (K,) bool
    threshold: jax.Array      # scalar
    ess_ratio: jax.Array      # scalar
    i_final: jax.Array        # scalar int32

    # optional resampling
    idx_resampled: jax.Array  # (K,) int32
    resample_status: jax.Array# scalar int64 (0 ok; nonzero indicates invalid weights in systematic)
    samples_resampled: jax.Array  # (K, D)
    logl_resampled: jax.Array     # (K,)
    logp_resampled: jax.Array     # (K,)
    blobs_resampled: jax.Array    # (K, B)

    # evidence (from compute_logw_and_logz_jax) 
    logz_new: jax.Array       # scalar
    key_out: jax.Array        # PRNGKey


@partial(jax.jit, static_argnames=("bins",))
def trim_weights_scan_jax(
    weights: jax.Array,
    ess: float | jax.Array = 0.99,
    bins: int = 1000,
):
    """
    Pure JAX trimming:
    scans percentile thresholds from high->low and picks the first satisfying ESS ratio >= ess.

    Returns:
      mask_trim: (K,) bool
      w_trim:    (K,) normalized; zeros are dropped
      threshold: scalar
      ratio:     scalar ESS_trim/ESS_total
      i_final:   scalar int32 (index into percentile grid)
    """
    w = jnp.asarray(weights)
    dtype = w.dtype
    ess = jnp.asarray(ess, dtype=dtype)

    wsum = jnp.sum(w)
    bad = (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(w)) | jnp.any(w < 0)

    # normalize 
    w = w / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    ess_total = 1.0 / jnp.sum(w * w)

    percentiles = jnp.linspace(jnp.asarray(0.0, dtype), jnp.asarray(99.0, dtype), bins)
    sorted_w = jnp.sort(w)

    n = w.shape[0]
    n_minus_1 = jnp.asarray(n - 1, dtype)

    def ratio_for_i(i: jax.Array):
        # i in [0, bins-1]
        p = lax.dynamic_index_in_dim(percentiles, i, axis=0, keepdims=False)
        frac = p / jnp.asarray(100.0, dtype)

        # linear interpolation percentile
        pos = frac * n_minus_1
        lo = jnp.floor(pos).astype(jnp.int32)
        hi = jnp.minimum(lo + 1, jnp.int32(n - 1))
        alpha = pos - lo.astype(dtype)

        w_lo = sorted_w[lo]
        w_hi = sorted_w[hi]
        threshold = (1.0 - alpha) * w_lo + alpha * w_hi

        mask = w >= threshold
        w_kept = jnp.where(mask, w, 0.0)

        kept_sum = jnp.sum(w_kept)
        kept_sumsq = jnp.sum(w_kept * w_kept)

        # ESS of normalized kept weights:
        # w_trim = w_kept / kept_sum  => sum(w_trim^2) = kept_sumsq / kept_sum^2
        kept_sum_safe = jnp.where(kept_sum > 0, kept_sum, jnp.asarray(1.0, dtype))
        ess_trim = (kept_sum_safe * kept_sum_safe) / jnp.where(kept_sumsq > 0, kept_sumsq, jnp.asarray(jnp.inf, dtype))

        ratio = ess_trim / ess_total
        return threshold, ratio

    # Scan i from bins-1 down to 0; pick first i with ratio >= ess
    idxs = jnp.arange(bins - 1, -1, -1, dtype=jnp.int32)

    def scan_step(carry, i):
        found, i_best = carry
        _, r = ratio_for_i(i)
        update = (~found) & (r >= ess)
        found2 = found | update
        i_best2 = jnp.where(update, i, i_best)
        return (found2, i_best2), r

    (found_final, i_final), _ = lax.scan(scan_step, (jnp.asarray(False), jnp.asarray(0, jnp.int32)), idxs)

    # if no found, i_final =0
    threshold, ratio = ratio_for_i(i_final)
    mask = w >= threshold

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
    JAX posterior() equivalent with fixed-shape outputs.

    - returns flattened arrays of size K = T_max * N.
    - mask_valid marks which entries correspond to filled steps (t prefix).
    - If trimming = true, mask_trim/weights/logw reflect trimming. otherwise mask_trim==mask_valid.
    - If do_resample=true, returns resampled arrays + indices + status. Otherwise idx_resampled=arange(K).

    NOTE: resampling is discrete (non-differentiable); weights/logw computation is differentiable.
    """
    # flatten history 
    T, N, D = state.x.shape
    K = T * N

    samples = state.x.reshape((K, D))
    logl = state.logl.reshape((K,))
    logp = state.logp.reshape((K,))
    blobs = state.blobs.reshape((K, state.blobs.shape[-1]))

    # Compute logw (normalized) + logz_new + valid mask
    logw0, logz_new, mask_valid = compute_logw_and_logz_jax(state, beta_final=beta_final, normalize=True)
    w0 = jnp.exp(logw0)

    # Mask-out invalid entries in returned arrays
    samples = jnp.where(mask_valid[:, None], samples, jnp.zeros_like(samples))
    logl = jnp.where(mask_valid, logl, jnp.zeros_like(logl))
    logp = jnp.where(mask_valid, logp, jnp.zeros_like(logp))
    blobs = jnp.where(mask_valid[:, None], blobs, jnp.zeros_like(blobs))

    trim_flag = jnp.asarray(trim_importance_weights, dtype=bool)

    def _do_trim(_):
        mask_trim, w_trim, thr, ratio, i_final = trim_weights_scan_jax(
            w0, ess=ess_trim, bins=bins_trim
        )
        # dont keep invalid entries
        mask_trim = mask_trim & mask_valid
        # make sure trimmed weights are normalized over and keep entries only
        w_trim = jnp.where(mask_trim, w_trim, 0.0)
        # safe renormalization (if nothing iskept, then all zeros)
        s = jnp.sum(w_trim)
        s_safe = jnp.where(s > 0, s, jnp.asarray(1.0, w_trim.dtype))
        w_trim = jnp.where(mask_trim, w_trim / s_safe, 0.0)
        return mask_trim, w_trim, thr, ratio, i_final

    def _no_trim(_):
        # keep everything valid
        mask_trim = mask_valid
        w_trim = jnp.where(mask_trim, w0, 0.0)
        # w0 already sums to 1 over valid (invalid are 0). no renormalization is needed
        thr = jnp.asarray(-jnp.inf, w0.dtype)
        ratio = jnp.asarray(1.0, w0.dtype)
        i_final = jnp.asarray(-1, jnp.int32)
        return mask_trim, w_trim, thr, ratio, i_final

    mask_trim, weights, threshold, ess_ratio, i_final = lax.cond(trim_flag, _do_trim, _no_trim, operand=None)
    logw = jnp.log(weights)  # -inf where weights==0

    do_resample_arr = jnp.asarray(do_resample, dtype=bool)
    resample_method = jnp.asarray(resample_method)

    def _resample(key_in):
        # systematic
        def _syst(k):
            idx, status, k_out = _systematic_resample_impl(k, weights, size=K)
            return idx.astype(jnp.int32), status.astype(jnp.int64), k_out

        # multinomial
        def _mult(k):
            k_out, sub = jax.random.split(k)
            idx = jax.random.choice(sub, a=K, shape=(K,), replace=True, p=weights)
            status = jnp.asarray(0, jnp.int64)
            return idx.astype(jnp.int32), status, k_out

        use_syst = resample_method == jnp.asarray(1, resample_method.dtype)
        return lax.cond(use_syst, _syst, _mult, key_in)

    def _no_resample(key_in):
        idx = jnp.arange(K, dtype=jnp.int32)
        status = jnp.asarray(0, jnp.int64)
        return idx, status, key_in

    idx_resampled, resample_status, key_out = lax.cond(do_resample_arr, _resample, _no_resample, key)

    # safe indexing even if systematic returns -1 on invalid weights
    idx_safe = jnp.clip(idx_resampled, 0, K - 1)

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



















