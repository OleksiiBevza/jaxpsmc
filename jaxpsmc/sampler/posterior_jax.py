from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from ..particles_jax import ParticlesState, compute_logw_and_logz_jax
from .constants_jax import _ECONVERGED, _EVALUEERR


#################################################################
# 5. POSTERIOR
#################################################################
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
    bad = (
        (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(w)) | jnp.any(w < 0)
    )

    w_norm = w / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    # build CDF and force last value to 1 when weights are valid
    cdf = jnp.cumsum(w_norm)
    cdf = cdf / jnp.where(bad, jnp.asarray(1.0, dtype), cdf[-1])

    # draw one uniform offset for the whole systematic grid
    key_out, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(), dtype=dtype)  # scalar
    positions = (u + jnp.arange(size, dtype=dtype)) / jnp.asarray(size, dtype=dtype)

    # convert positions into CDF indices
    idx = jnp.searchsorted(
        cdf, positions, side="left"
    )  # :contentReference[oaicite:2]{index=2}
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
    samples: jax.Array  # (K, D)
    logl: jax.Array  # (K,)
    logp: jax.Array  # (K,)
    blobs: jax.Array  # (K, B)
    mask_valid: jax.Array  # (K,) bool (True for filled steps)

    # importance weights
    weights: (
        jax.Array
    )  # (K,) normalized over kept entries and zeros where dropped/invalid
    logw: jax.Array  # (K,) log(weights); -inf where weights==0
    mask_trim: jax.Array  # (K,) bool
    threshold: jax.Array  # scalar
    ess_ratio: jax.Array  # scalar
    i_final: jax.Array  # scalar int32

    # optional resampling
    idx_resampled: jax.Array  # (K,) int32
    resample_status: (
        jax.Array
    )  # scalar int64 (0 ok; nonzero indicates invalid weights in systematic)
    samples_resampled: jax.Array  # (K, D)
    logl_resampled: jax.Array  # (K,)
    logp_resampled: jax.Array  # (K,)
    blobs_resampled: jax.Array  # (K, B)

    # evidence (from compute_logw_and_logz_jax)
    logz_new: jax.Array  # scalar
    key_out: jax.Array  # PRNGKey


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
    bad = (
        (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(w)) | jnp.any(w < 0)
    )

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
        ess_trim = (kept_sum_safe * kept_sum_safe) / jnp.where(
            kept_sumsq > 0, kept_sumsq, jnp.asarray(jnp.inf, dtype)
        )

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
    (found_final, i_final), _ = lax.scan(
        scan_step, (jnp.asarray(False), jnp.asarray(0, jnp.int32)), idxs
    )

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
    logw0, logz_new, mask_valid = compute_logw_and_logz_jax(
        state, beta_final=beta_final, normalize=True
    )
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
    mask_trim, weights, threshold, ess_ratio, i_final = lax.cond(
        trim_flag, _do_trim, _no_trim, operand=None
    )
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
    idx_resampled, resample_status, key_out = lax.cond(
        do_resample_arr, _resample, _no_resample, key
    )

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
