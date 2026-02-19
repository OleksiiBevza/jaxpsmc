from functools import partial
import jax.numpy as jnp
from jax import lax
import jax
jax.config.update("jax_enable_x64", True)
print("x64 enabled?", jax.config.jax_enable_x64)



@partial(jax.jit, static_argnames=("bins",))
def trim_weights_jax(samples, weights, ess=0.99, bins=1000):
    """
    trim_weights.

    Returns:
      mask:            (N,) bool
      weights_trimmed: (N,) weights after trimming, renormalized (zeros for dropped)
      threshold:       scalar weight threshold is used here
      ess_ratio:       ess_trimmed / ess_total
      i_final:         index in percentile grid that was selected

    Notes:
      - No Python control flow, no mutation, no NumPy.
      - JIT/vmap/cond  = yes
      - Does not return variable-length arrays; uses `mask` for selection
    """
    samples = jnp.asarray(samples)
    weights = jnp.asarray(weights)

    dtype = jnp.result_type(weights, jnp.asarray(ess))
    weights = weights.astype(dtype)
    ess = jnp.asarray(ess, dtype=dtype)

    # Normalize weights
    wsum = jnp.sum(weights)
    bad = (wsum <= 0) | jnp.isnan(wsum)

    w = weights / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    ess_total = 1.0 / jnp.sum(w * w)

    percentiles = jnp.linspace(jnp.asarray(0.0, dtype), jnp.asarray(99.0, dtype), bins)
    sorted_w = jnp.sort(w)  # 

    n = w.shape[0]
    n_minus_1 = jnp.asarray(n - 1, dtype)

    def stats_for_i(i):
        # p in [0, 99]
        p = lax.dynamic_index_in_dim(percentiles, i, axis=0, keepdims=False)
        frac = p / jnp.asarray(100.0, dtype)

        # linear interpolation percentile like in  NumPy's default ("linear")
        pos = frac * n_minus_1                    # in [0, n-1]
        lo = jnp.floor(pos).astype(jnp.int32)
        hi = jnp.minimum(lo + 1, jnp.int32(n - 1))
        alpha = pos - lo.astype(dtype)

        w_lo = sorted_w[lo]
        w_hi = sorted_w[hi]
        threshold = (1.0 - alpha) * w_lo + alpha * w_hi

        mask = w >= threshold
        w_kept = jnp.where(mask, w, 0.0)
        kept_sum = jnp.sum(w_kept)

        kept_sum_safe = jnp.where(kept_sum > 0, kept_sum, jnp.asarray(1.0, dtype))
        w_trim = jnp.where(mask, w_kept / kept_sum_safe, 0.0)

        ess_trim = 1.0 / jnp.sum(w_trim * w_trim)
        ratio = ess_trim / ess_total
        return threshold, mask, w_trim, ratio

    # search from high percentile downward until ratio >= ess (or is zero)
    def cond(state):
        i, done = state
        return ~done

    def body(state):
        i, done = state
        _, _, _, ratio = stats_for_i(i)
        satisfied = ratio >= ess
        done2 = done | satisfied | (i == 0)
        i2 = jnp.where(done2, i, i - 1)
        return i2, done2

    i0 = jnp.int32(bins - 1)
    i_final, _ = lax.while_loop(cond, body, (i0, False))

    threshold, mask, w_trim, ratio = stats_for_i(i_final)

    # If weights were invalid, return "all dropped" + NaNs for weights_trimmed
    mask = jnp.where(bad, jnp.zeros_like(mask), mask)
    w_trim = jnp.where(bad, jnp.full_like(w_trim, jnp.nan), w_trim)
    threshold = jnp.where(bad, jnp.asarray(jnp.nan, dtype), threshold)
    ratio = jnp.where(bad, jnp.asarray(jnp.nan, dtype), ratio)

    return mask, w_trim, threshold, ratio, i_final


@jax.jit
def effective_sample_size_jax(weights):
    """
    ESS

    Returns a scalar ESS (float dtype), or NaN if inputs are invalid.
    """
    w = jnp.asarray(weights)
    wsum = jnp.sum(w)

    # invalid if sum<=0 or non-finite
    bad = (wsum <= 0) | jnp.isnan(wsum) | jnp.isinf(wsum)

    w_norm = w / jnp.where(bad, jnp.asarray(1.0, w.dtype), wsum)
    ess = 1.0 / jnp.sum(w_norm * w_norm)

    return jnp.where(bad, jnp.asarray(jnp.nan, ess.dtype), ess)


@jax.jit
def unique_sample_size_jax(weights, k=-1):
    """
    unique sample size.

    Args:
      weights: array (..., N) or (N,)
      k: int. If k < 0, uses k = N (last dimension).

    Returns:
      uss: scalar if weights is (N,), or array (...) if weights is (..., N).
           Returns NaN where weights are invalid (sum<=0 or non-finite).
    """
    w = jnp.asarray(weights)
    wsum = jnp.sum(w, axis=-1, keepdims=True)

    bad = (wsum <= 0) | jnp.isnan(wsum) | jnp.isinf(wsum)

    # normalize without mutation
    w_norm = w / jnp.where(bad, jnp.asarray(1.0, w.dtype), wsum)

    # resolve k purely in JAX
    N = w.shape[-1]
    k_eff = lax.cond(k < 0, lambda _: jnp.int32(N), lambda _: jnp.int32(k), operand=None)

    # compute: sum_i [ 1 - (1 - w_i)^k ]
    # works for k=0 too -> term becomes 0
    term = 1.0 - jnp.power(1.0 - w_norm, k_eff)
    uss = jnp.sum(term, axis=-1)

    # bad -> NaN 
    uss = jnp.where(jnp.squeeze(bad, axis=-1), jnp.asarray(jnp.nan, uss.dtype), uss)
    return uss


@jax.jit
def compute_ess_jax(logw):
    """
    Compute ESS fraction = (1 / sum(w^2)) / N, with w = softmax(logw).

    Args:
      logw: array (N,) or (..., N) of log-weights.

    Returns:
      ess_frac: scalar if logw is (N,), or array (...) if logw is (..., N).
                Returns NaN if inputs are non-finite
    """
    lw = jnp.asarray(logw)

    # stabilize: subtract max along last axis
    lw_max = jnp.max(lw, axis=-1, keepdims=True)
    lw0 = lw - lw_max

    # exponentiate + normalize using softmax weights
    w_unnorm = jnp.exp(lw0)
    wsum = jnp.sum(w_unnorm, axis=-1, keepdims=True)

    bad = (wsum <= 0) | jnp.isnan(wsum) | jnp.isinf(wsum)

    w = w_unnorm / jnp.where(bad, jnp.asarray(1.0, w_unnorm.dtype), wsum)

    ess = 1.0 / jnp.sum(w * w, axis=-1)          # ESS
    N = lw.shape[-1]
    ess_frac = ess / jnp.asarray(N, ess.dtype)   # ESS / N

    # propagate invalid -> NaN    BUT IS IT INSIDE JAX
    ess_frac = jnp.where(jnp.squeeze(bad, axis=-1), jnp.asarray(jnp.nan, ess_frac.dtype), ess_frac)
    return ess_frac


@jax.jit
def increment_logz_jax(logw):
    """
    Compute logZ increment: logsumexp(logw).

    Args:
      logw: array (N,) or (..., N) of log-weights.

    Returns:
      logz_inc: scalar if logw is (N,), or array (...) if logw is (..., N).
                Returns NaN if inputs are all -inf
    """
    lw = jnp.asarray(logw)

    lw_max = jnp.max(lw, axis=-1, keepdims=True)
    lw0 = lw - lw_max

    # logsumexp = max + log(sum(exp(lw - max)))
    lse = lw_max + jnp.log(jnp.sum(jnp.exp(lw0), axis=-1, keepdims=True))

    # reduce axis
    lse = jnp.squeeze(lse, axis=-1)

    lse = jnp.where(jnp.isfinite(lse), lse, jnp.nan)

    return lse


_ECONVERGED = jnp.int64(0)
_EVALUEERR  = jnp.int64(-3)

@partial(jax.jit, static_argnames=("size",))
def _systematic_resample_impl(key, weights, size: int):
    w = jnp.asarray(weights)
    dtype = jnp.result_type(w, jnp.float64)

    wsum = jnp.sum(w)
    bad = (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(w)) | jnp.any(w < 0)

    w_norm = w / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    cdf = jnp.cumsum(w_norm)
    cdf = cdf / jnp.where(bad, jnp.asarray(1.0, dtype), cdf[-1])

    key_out, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(), dtype=dtype)
    positions = (u + jnp.arange(size, dtype=dtype)) / jnp.asarray(size, dtype=dtype)

    idx = jnp.searchsorted(cdf, positions, side="left")
    idx = jnp.clip(idx, 0, w.shape[0] - 1).astype(jnp.int32)

    idx = jnp.where(bad, jnp.full((size,), jnp.int32(-1)), idx)
    status = jnp.where(bad, _EVALUEERR, _ECONVERGED)
    return idx, status, key_out


def systematic_resample_jax(weights, *, key):
    """Resample exactly len(weights)"""
    w = jnp.asarray(weights)
    return _systematic_resample_impl(key, w, w.shape[0])


def systematic_resample_jax_size(weights, *, key, size: int):
    """Resample explicit static size"""
    w = jnp.asarray(weights)
    return _systematic_resample_impl(key, w, size)