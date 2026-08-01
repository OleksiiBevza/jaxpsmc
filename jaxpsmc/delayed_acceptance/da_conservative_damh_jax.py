from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class ConservativeDAMHStep(NamedTuple):
    """
    Stores the result of one conservative delayed-acceptance MH step.

    The step has two acceptance stages.
    The first stage uses the cheap surrogate ratio.
    The second stage corrects this decision using the full ratio.

    Parameters:
    -----------
    key:
        updated JAX random key after splitting.
    pre_accept:
        Boolean array showing which proposals passed stage 1.
    stage2_accept:
        Boolean array showing which proposals passed stage 2.
    accept:
        Boolean array showing which proposals passed both stages.

    expected_pre_accept:
        stage-1 acceptance probability.
    prob_accept:
        total acceptance probability from both stages.

    log_ratio_surrogate_raw:
        original surrogate log-ratio before clipping.
    log_ratio_stage1:
        clipped surrogate log-ratio used in stage 1.
    log_ratio_stage2:
        correction log-ratio used in stage 2.
    log_ratio_full:
        full log-ratio for the exact target.

    proposal_dist:
        Mahalanobis distance between old and proposed particles.
    actual_dist:
        proposal distance counted only for accepted proposals.
    expected_dist:
        proposal distance weighted by acceptance probability.

    full_eval_mask:
        Boolean mask showing which proposals need a full evaluation.
        This is the same as pre_accept.
    full_calls:
        number of proposals that passed stage 1.

    Returns:
    --------
    ConservativeDAMHStep:
        stores all acceptance decisions, probabilities, log-ratios,
        distances, and full-evaluation counts for one DAMH step.
    """

    key: Array
    pre_accept: Array
    stage2_accept: Array
    accept: Array

    expected_pre_accept: Array
    prob_accept: Array

    log_ratio_surrogate_raw: Array
    log_ratio_stage1: Array
    log_ratio_stage2: Array
    log_ratio_full: Array

    proposal_dist: Array
    actual_dist: Array
    expected_dist: Array

    full_eval_mask: Array
    full_calls: Array


@jax.jit
def _clean_log_ratio_jax(log_ratio: Array) -> Array:
    """
    Replaces invalid log-ratios with automatic rejection values.

    A NaN log-ratio is unsafe.
    This function changes NaN values to negative infinity.
    Negative infinity gives acceptance probability zero.

    Parameters:
    -----------
    log_ratio:
        log acceptance ratio.

    Returns:
    --------
    Array:
        log-ratio with NaN values replaced by -inf.
    """
    log_ratio = jnp.asarray(log_ratio)
    return jnp.where(jnp.isnan(log_ratio), -jnp.inf, log_ratio)


@jax.jit
def _log_accept_prob_jax(log_ratio: Array) -> Array:
    """
    Converts a log-ratio into a log acceptance probability.

    Metropolis-Hastings accepts with probability min(1, exp(log_ratio)).
    In log space, this is min(0, log_ratio).

    Parameters:
    -----------
    log_ratio:
        log acceptance ratio.

    Returns:
    --------
    Array:
        log acceptance probability.
        Values are always less than or equal to zero.
    """
    log_ratio = _clean_log_ratio_jax(log_ratio)
    return jnp.minimum(log_ratio, jnp.asarray(0.0, dtype=log_ratio.dtype))


@jax.jit
def mahalanobis_distance_jax(
    new_particles: Array,
    old_particles: Array,
    cov: Array,
) -> Array:
    """
    Computes Mahalanobis distances between proposed and old particles.

    The distance uses the covariance matrix as a scale.
    This makes movement in high-variance directions count less.
    It makes movement in low-variance directions count more.

    Parameters:
    -----------
    new_particles:
        proposed particles, shape (N, D).
    old_particles:
        current particles, shape (N, D).
    cov:
        covariance matrix, shape (D, D).
        It must be invertible.
        In practice, it should usually be positive definite.

    Returns:
    --------
    Array:
        Mahalanobis distance for each particle, shape (N,).
    """
    new_particles = jnp.asarray(new_particles)
    old_particles = jnp.asarray(old_particles)
    cov = jnp.asarray(cov)

    diff = new_particles - old_particles
    solved = jnp.linalg.solve(cov, diff.T).T
    maha = jnp.sum(diff * solved, axis=1)

    return jnp.sqrt(jnp.maximum(maha, jnp.asarray(0.0, dtype=maha.dtype)))


@jax.jit
def conservative_damh_step_jax(
    key: Array,
    new_particles: Array,
    old_particles: Array,
    cov: Array,
    log_ratio_surrogate: Array,
    log_ratio_full: Array,
    c_const: Array = jnp.asarray(0.01),
    d_const: Array = jnp.asarray(2.0),
) -> ConservativeDAMHStep:
    """
    Runs one conservative delayed-acceptance MH step.

    The method tests each proposal in two stages.
    Stage 1 uses a cheap surrogate log-ratio.
    The surrogate ratio is clipped to avoid extreme decisions.
    Stage 2 corrects the decision using the full log-ratio.

    A proposal is accepted only if it passes both stages.
    The function also reports distances and full-model call counts.

    Parameters:
    -----------
    key:
        JAX random key used for stage-1 and stage-2 decisions.
    new_particles:
        proposed particles, shape (N, D).
    old_particles:
        current particles, shape (N, D).
    cov:
        covariance matrix used for Mahalanobis distance, shape (D, D).
        It must be invertible.
    log_ratio_surrogate:
        surrogate log acceptance ratio, shape (N,).
        This is the cheap first-stage ratio.
    log_ratio_full:
        full log acceptance ratio, shape (N,).
        This is the exact or expensive ratio.
    c_const:
        lower control constant for conservative clipping.
        It must be positive.
    d_const:
        exponent-like control constant for conservative clipping.
        It must be greater than 1.

    Returns:
    --------
    ConservativeDAMHStep:
        stores the updated key, acceptance decisions, probabilities,
        log-ratios, proposal distances, and number of full evaluations.
    """
    new_particles = jnp.asarray(new_particles)
    old_particles = jnp.asarray(old_particles)
    log_ratio_surrogate = jnp.asarray(log_ratio_surrogate)
    log_ratio_full = jnp.asarray(log_ratio_full)

    dtype = jnp.result_type(
        new_particles,
        old_particles,
        cov,
        log_ratio_surrogate,
        log_ratio_full,
        c_const,
        d_const,
        jnp.asarray(1.0),
    )

    log_ratio_surrogate = log_ratio_surrogate.astype(dtype)
    log_ratio_full = log_ratio_full.astype(dtype)
    c_const = jnp.asarray(c_const, dtype=dtype)
    d_const = jnp.asarray(d_const, dtype=dtype)

    key, key_stage1, key_stage2 = jax.random.split(key, 3)

    log_b_const = jnp.log(c_const) / (d_const - jnp.asarray(1.0, dtype=dtype))

    log_ratio_stage1 = jnp.clip(
        log_ratio_surrogate,
        log_b_const,
        -log_b_const,
    )

    log_prob_stage1 = _log_accept_prob_jax(log_ratio_stage1)

    u1 = jax.random.uniform(
        key_stage1,
        shape=log_prob_stage1.shape,
        dtype=dtype,
    )
    pre_accept = jnp.log(u1) < log_prob_stage1

    log_ratio_stage2 = _clean_log_ratio_jax(log_ratio_full - log_ratio_stage1)
    log_prob_stage2 = _log_accept_prob_jax(log_ratio_stage2)

    u2 = jax.random.uniform(
        key_stage2,
        shape=log_prob_stage2.shape,
        dtype=dtype,
    )
    stage2_accept = jnp.log(u2) < log_prob_stage2

    accept = pre_accept & stage2_accept

    expected_pre_accept = jnp.exp(log_prob_stage1)
    prob_accept = expected_pre_accept * jnp.exp(log_prob_stage2)

    proposal_dist = mahalanobis_distance_jax(
        new_particles=new_particles,
        old_particles=old_particles,
        cov=cov,
    ).astype(dtype)

    actual_dist = proposal_dist * accept.astype(dtype)
    expected_dist = proposal_dist * prob_accept

    full_calls = jnp.sum(pre_accept.astype(jnp.int32), dtype=jnp.int32)

    return ConservativeDAMHStep(
        key=key,
        pre_accept=pre_accept,
        stage2_accept=stage2_accept,
        accept=accept,
        expected_pre_accept=expected_pre_accept,
        prob_accept=prob_accept,
        log_ratio_surrogate_raw=log_ratio_surrogate,
        log_ratio_stage1=log_ratio_stage1,
        log_ratio_stage2=log_ratio_stage2,
        log_ratio_full=log_ratio_full,
        proposal_dist=proposal_dist,
        actual_dist=actual_dist,
        expected_dist=expected_dist,
        full_eval_mask=pre_accept,
        full_calls=full_calls,
    )


@jax.jit
def conservative_damh_step_parts_jax(
    key: Array,
    new_particles: Array,
    old_particles: Array,
    cov: Array,
    approx_posterior_new: Array,
    approx_posterior_old: Array,
    full_likelihood_new: Array,
    full_likelihood_old: Array,
    approx_likelihood_new: Array,
    approx_likelihood_old: Array,
    c_const: Array = jnp.asarray(0.01),
    d_const: Array = jnp.asarray(2.0),
) -> ConservativeDAMHStep:
    """
    Builds the DAMH log-ratios from model parts and runs one step.

    The surrogate ratio is made from approximate posterior values.
    The full ratio adds a correction term.
    The correction compares the full likelihood with the approximate likelihood.

    This wrapper is useful when the caller has separate log-density pieces.
    It avoids requiring the caller to build the final ratios manually.

    Parameters:
    -----------
    key:
        JAX random key used for stage-1 and stage-2 decisions.
    new_particles:
        proposed particles, shape (N, D).
    old_particles:
        current particles, shape (N, D).
    cov:
        covariance matrix used for Mahalanobis distance, shape (D, D).
        It must be invertible.
    approx_posterior_new:
        approximate posterior value at proposed particles, shape (N,).
    approx_posterior_old:
        approximate posterior value at current particles, shape (N,).
    full_likelihood_new:
        full likelihood value at proposed particles, shape (N,).
    full_likelihood_old:
        full likelihood value at current particles, shape (N,).
    approx_likelihood_new:
        approximate likelihood value at proposed particles, shape (N,).
    approx_likelihood_old:
        approximate likelihood value at current particles, shape (N,).
    c_const:
        lower control constant for conservative clipping.
        It must be positive.
    d_const:
        exponent-like control constant for conservative clipping.
        It must be greater than 1.

    Returns:
    --------
    ConservativeDAMHStep:
        result of one conservative delayed-acceptance MH step.
    """
    log_ratio_surrogate = approx_posterior_new - approx_posterior_old

    log_ratio_full = log_ratio_surrogate + (
        full_likelihood_new
        - full_likelihood_old
        - approx_likelihood_new
        + approx_likelihood_old
    )

    return conservative_damh_step_jax(
        key=key,
        new_particles=new_particles,
        old_particles=old_particles,
        cov=cov,
        log_ratio_surrogate=log_ratio_surrogate,
        log_ratio_full=log_ratio_full,
        c_const=c_const,
        d_const=d_const,
    )
