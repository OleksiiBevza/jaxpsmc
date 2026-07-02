from functools import partial
import jax.numpy as jnp
from jax import lax
import jax

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("bins",))
def trim_weights_jax(samples, weights, ess=0.99, bins=1000):
    """
    Trims importance weights by scanning percentile thresholds.

    The function searches for a weight threshold.
    Weights below the threshold are dropped.
    The remaining weights are renormalized to sum to one.

    The chosen threshold should keep enough effective sample size.
    More precisely, the function searches for a threshold where the ESS
    after trimming is at least the requested fraction of the original ESS.

    The samples argument is not used in the computation.
    It is kept only to match the expected interface elsewhere in the code.

    Parameters:
    -----------
    samples:
        sample index array.
        This input is currently unused.
    weights:
        input importance weights, shape (N,).
        They are normalized inside the function.
    ess:
        target ESS ratio after trimming.
        For example, 0.99 means the trimmed weights should preserve
        at least 99 percent of the original ESS.
    bins:
        number of percentile grid points used in the threshold search.
        This is static for JAX compilation.

    Returns:
    --------
    tuple:
        mask:
            Boolean mask, shape (N,).
            True means the weight was kept.
        weights_trimmed:
            trimmed and renormalized weights, shape (N,).
            Dropped entries are zero.
            Invalid input returns NaN weights.
        threshold:
            selected weight threshold.
        ess_ratio:
            ratio between trimmed ESS and original ESS.
        i_final:
            selected percentile-grid index.
    """
    samples = jnp.asarray(samples)
    weights = jnp.asarray(weights)

    dtype = jnp.result_type(weights, jnp.asarray(ess))
    weights = weights.astype(dtype)
    ess = jnp.asarray(ess, dtype=dtype)

    # normalize weights
    wsum = jnp.sum(weights)
    bad = (wsum <= 0) | jnp.isnan(wsum)

    w = weights / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    # compute ESS before trimming
    ess_total = 1.0 / jnp.sum(w * w)

    # define percentile grid and sorted weights
    percentiles = jnp.linspace(jnp.asarray(0.0, dtype), jnp.asarray(99.0, dtype), bins)
    sorted_w = jnp.sort(w)  #

    # nr of weights
    n = w.shape[0]
    n_minus_1 = jnp.asarray(n - 1, dtype)

    def stats_for_i(i):
        """
        Computes trimming statistics for one percentile-grid index.

        The index selects one percentile from the grid.
        That percentile defines a candidate threshold.
        The function then keeps weights above the threshold,
        renormalizes them, and computes the ESS ratio.

        Parameters:
        -----------
        i:
            percentile-grid index.

        Returns:
        --------
        tuple:
            threshold:
                candidate weight threshold.
            mask:
                Boolean mask showing which weights are kept.
            w_trim:
                trimmed and renormalized weights.
            ratio:
                ESS ratio after trimming.
        """
        # current percentile: p in [0, 99]
        p = lax.dynamic_index_in_dim(percentiles, i, axis=0, keepdims=False)
        frac = p / jnp.asarray(100.0, dtype)

        # linear interpolation percentile threshold from sorted weights
        pos = frac * n_minus_1  # in [0, n-1]
        lo = jnp.floor(pos).astype(jnp.int32)
        hi = jnp.minimum(lo + 1, jnp.int32(n - 1))
        alpha = pos - lo.astype(dtype)

        w_lo = sorted_w[lo]
        w_hi = sorted_w[hi]
        threshold = (1.0 - alpha) * w_lo + alpha * w_hi

        # keep only weights above threshold
        mask = w >= threshold
        w_kept = jnp.where(mask, w, 0.0)
        kept_sum = jnp.sum(w_kept)

        # renormalize kept weights
        kept_sum_safe = jnp.where(kept_sum > 0, kept_sum, jnp.asarray(1.0, dtype))
        w_trim = jnp.where(mask, w_kept / kept_sum_safe, 0.0)

        # compute ESS ratio after trimming
        ess_trim = 1.0 / jnp.sum(w_trim * w_trim)
        ratio = ess_trim / ess_total
        return threshold, mask, w_trim, ratio

    # search from high percentile downward until ratio >= ess (or is zero)
    def cond(state):
        """
        Checks whether the threshold search should continue.

        The loop continues until a threshold is accepted.
        It also stops when the search reaches the lowest grid index.

        Parameters:
        -----------
        state:
            tuple containing the current index and a done flag.

        Returns:
        --------
        jax.Array:
            Boolean scalar.
            True means another search step should run.
        """
        # continue until valid threshold has been found
        i, done = state
        return ~done

    def body(state):
        """
        Performs one threshold-search step.

        The current threshold is tested.
        If it preserves enough ESS, the search stops.
        Otherwise, the search moves to the next lower percentile.

        Parameters:
        -----------
        state:
            tuple containing the current index and a done flag.

        Returns:
        --------
        tuple:
            updated index and updated done flag.
        """
        # current search state
        i, done = state

        # # check if current percentile reaches target ESS ratio
        _, _, _, ratio = stats_for_i(i)
        satisfied = ratio >= ess

        # stop when target is reached or grid start is reached
        done2 = done | satisfied | (i == 0)
        i2 = jnp.where(done2, i, i - 1)
        return i2, done2

    # start from largest percentile and move downward
    i0 = jnp.int32(bins - 1)
    i_final, _ = lax.while_loop(cond, body, (i0, False))

    # rebuild trimming result for  selected index
    threshold, mask, w_trim, ratio = stats_for_i(i_final)

    # if weights were invalid, return "empty outputs" + NaNs for weights_trimmed
    mask = jnp.where(bad, jnp.zeros_like(mask), mask)
    w_trim = jnp.where(bad, jnp.full_like(w_trim, jnp.nan), w_trim)
    threshold = jnp.where(bad, jnp.asarray(jnp.nan, dtype), threshold)
    ratio = jnp.where(bad, jnp.asarray(jnp.nan, dtype), ratio)

    return mask, w_trim, threshold, ratio, i_final


@jax.jit
def effective_sample_size_jax(weights):
    """
    Computes effective sample size from importance weights.

    Effective sample size measures weight concentration.
    If all weights are equal, the ESS is close to the number of weights.
    If one weight dominates, the ESS is close to one.

    The function normalizes the weights before computing ESS.
    Invalid total weight returns NaN.

    Parameters:
    -----------
    weights:
        input weight array, shape (N,).

    Returns:
    --------
    jax.Array:
        effective sample size.
        Returns NaN if the total weight is invalid.
    """
    w = jnp.asarray(weights)
    wsum = jnp.sum(w)

    # invalid inputs if sum<=0 or non-finite
    bad = (wsum <= 0) | jnp.isnan(wsum) | jnp.isinf(wsum)

    # normalize weights
    w_norm = w / jnp.where(bad, jnp.asarray(1.0, w.dtype), wsum)
    ess = 1.0 / jnp.sum(w_norm * w_norm)

    # return NaN for invalid inputs
    return jnp.where(bad, jnp.asarray(jnp.nan, ess.dtype), ess)


@jax.jit
def unique_sample_size_jax(weights, k=-1):
    """
    Computes the expected number of unique samples after resampling.

    The formula estimates how many distinct particles would appear
    after drawing k times with replacement from the weight distribution.

    If k is negative, the function uses the number of weights N.
    The function supports a single weight vector or batched weight vectors.

    Parameters:
    -----------
    weights:
        input weights, shape (N,) or (..., N).
    k:
        number of resampling draws.
        If k < 0, use N, where N is the last dimension of weights.

    Returns:
    --------
    jax.Array:
        expected unique sample size.
        Shape is scalar for input shape (N,),
        or shape (...) for input shape (..., N).
        Invalid total weights return NaN.
    """
    w = jnp.asarray(weights)
    wsum = jnp.sum(w, axis=-1, keepdims=True)

    bad = (wsum <= 0) | jnp.isnan(wsum) | jnp.isinf(wsum)

    # normalize weights without mutation
    w_norm = w / jnp.where(bad, jnp.asarray(1.0, w.dtype), wsum)

    # choose k from input or from last axis length
    N = w.shape[-1]
    k_eff = lax.cond(
        k < 0, lambda _: jnp.int32(N), lambda _: jnp.int32(k), operand=None
    )

    # compute unique sample size formula: sum_i [ 1 - (1 - w_i)^k ]
    # works for k=0 too then term becomes 0
    term = 1.0 - jnp.power(1.0 - w_norm, k_eff)
    uss = jnp.sum(term, axis=-1)

    # return NaN for invalid inputs
    uss = jnp.where(jnp.squeeze(bad, axis=-1), jnp.asarray(jnp.nan, uss.dtype), uss)
    return uss


@jax.jit
def compute_ess_jax(logw):
    """
    Computes the ESS fraction from log-weights.

    The function first converts log-weights into normalized weights.
    It then computes ESS and divides by the number of weights.
    The result is therefore an ESS fraction between zero and one
    when the input is valid.

    This is useful when weights are stored in log space for stability.

    Parameters:
    -----------
    logw:
        input log-weights, shape (N,) or (..., N).

    Returns:
    --------
    jax.Array:
        ESS fraction.
        Shape is scalar for input shape (N,),
        or shape (...) for input shape (..., N).
        Invalid inputs return NaN.
    """
    lw = jnp.asarray(logw)

    # stabilize exponentials by subtracting maximum
    lw_max = jnp.max(lw, axis=-1, keepdims=True)
    lw0 = lw - lw_max

    # convert log-weights into normalized weights:
    # exponentiate and normalize using softmax weights
    w_unnorm = jnp.exp(lw0)
    wsum = jnp.sum(w_unnorm, axis=-1, keepdims=True)

    # invalid inputs
    bad = (wsum <= 0) | jnp.isnan(wsum) | jnp.isinf(wsum)

    # normalize weights and compute ESS
    w = w_unnorm / jnp.where(bad, jnp.asarray(1.0, w_unnorm.dtype), wsum)
    ess = 1.0 / jnp.sum(w * w, axis=-1)  # ESS

    # divide ESS by nr of weights
    N = lw.shape[-1]
    ess_frac = ess / jnp.asarray(N, ess.dtype)  # ESS / N

    # NaN for invalid inputs
    ess_frac = jnp.where(
        jnp.squeeze(bad, axis=-1), jnp.asarray(jnp.nan, ess_frac.dtype), ess_frac
    )
    return ess_frac


@jax.jit
def increment_logz_jax(logw):
    """
    Computes a log normalizing-constant increment from log-weights.

    The function computes logsumexp(logw).
    This is the log of the sum of exponentiated log-weights.
    The computation subtracts the maximum log-weight first for stability.

    Parameters:
    -----------
    logw:
        input log-weights, shape (N,) or (..., N).

    Returns:
    --------
    jax.Array:
        log-sum-exp value.
        Shape is scalar for input shape (N,),
        or shape (...) for input shape (..., N).
        Non-finite results are returned as NaN.
    """
    lw = jnp.asarray(logw)

    lw_max = jnp.max(lw, axis=-1, keepdims=True)
    lw0 = lw - lw_max

    # compute logsumexp as max plus log of the summed exponentials:
    # logsumexp = max + log(sum(exp(lw - max)))
    lse = lw_max + jnp.log(jnp.sum(jnp.exp(lw0), axis=-1, keepdims=True))

    # remove last singleton axis
    lse = jnp.squeeze(lse, axis=-1)
    # NaN when the result is not finite
    lse = jnp.where(jnp.isfinite(lse), lse, jnp.nan)

    return lse


_ECONVERGED = jnp.int64(0)
_EVALUEERR = jnp.int64(-3)


@partial(jax.jit, static_argnames=("size",))
def _systematic_resample_impl(key, weights, size: int):
    """
    Runs the core systematic resampling algorithm.

    Systematic resampling draws many indices from a weighted particle set.
    It uses one random offset and an evenly spaced grid over the cumulative
    weight distribution.

    This usually has lower variance than multinomial resampling.
    If the weights are invalid, the function returns dummy indices
    and an error status.

    Parameters:
    -----------
    key:
        JAX random key used to draw the random offset.
    weights:
        input particle weights, shape (N,).
        They should be finite, non-negative, and have positive sum.
    size:
        number of resampled indices to draw.
        This is static for JAX compilation.

    Returns:
    --------
    tuple:
        idx:
            resampled indices, shape (size,).
            Invalid weights produce index -1.
        status:
            status code.
            0 means success.
            -3 means invalid weights.
        key_out:
            updated JAX random key.
    """
    w = jnp.asarray(weights)
    dtype = jnp.result_type(w, jnp.float64)

    # weights validation
    wsum = jnp.sum(w)
    bad = (
        (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(w)) | jnp.any(w < 0)
    )
    # weights normalization
    w_norm = w / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    # cumulative distribution function
    cdf = jnp.cumsum(w_norm)
    cdf = cdf / jnp.where(bad, jnp.asarray(1.0, dtype), cdf[-1])

    # draw one random offset and define systematic positions
    key_out, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(), dtype=dtype)
    positions = (u + jnp.arange(size, dtype=dtype)) / jnp.asarray(size, dtype=dtype)

    # dummy indices when you have invalid weights
    idx = jnp.searchsorted(cdf, positions, side="left")
    idx = jnp.clip(idx, 0, w.shape[0] - 1).astype(jnp.int32)

    idx = jnp.where(bad, jnp.full((size,), jnp.int32(-1)), idx)
    status = jnp.where(bad, _EVALUEERR, _ECONVERGED)
    return idx, status, key_out


def systematic_resample_jax(weights, *, key):
    """
    Resamples len(weights) indices using systematic resampling.

    This is a convenience wrapper around _systematic_resample_impl.
    The output size is set to the number of input weights.

    Parameters:
    -----------
    weights:
        input particle weights, shape (N,).
    key:
        JAX random key.

    Returns:
    --------
    tuple:
        resampled indices, status code, and updated random key.
    """
    w = jnp.asarray(weights)
    return _systematic_resample_impl(key, w, w.shape[0])


def systematic_resample_jax_size(weights, *, key, size: int):
    """
    Resamples a fixed number of indices using systematic resampling.

    This wrapper is used when the requested output size is not necessarily
    equal to the number of input weights.

    Parameters:
    -----------
    weights:
        input particle weights, shape (N,).
    key:
        JAX random key.
    size:
        number of indices to draw.

    Returns:
    --------
    tuple:
        resampled indices, status code, and updated random key.
    """
    w = jnp.asarray(weights)
    return _systematic_resample_impl(key, w, size)
