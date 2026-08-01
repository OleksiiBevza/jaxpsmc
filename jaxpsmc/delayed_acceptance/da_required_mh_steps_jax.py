from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class RequiredMHSteps(NamedTuple):
    """
    Stores the estimated number of MH steps needed.

    The estimate is based on expected squared jump distance.
    If the required number of steps is too large, the result is capped.

    Parameters:
    -----------
    prob:
        probability value returned for the MH move.
        It is rho when the required number of steps is acceptable.
        It is NaN when the required number of steps is too large.
    iter:
        number of MH steps to use.
        This is capped at max_t.
    sufficient_iter:
        Boolean value showing whether the required number of steps
        was less than or equal to max_t.
    median_expected_dist:
        median expected squared jump distance.
    steps_float:
        raw estimated number of steps before rounding.

    Returns:
    --------
    RequiredMHSteps:
        stores the MH step estimate, the capped iteration count,
        and diagnostic values used to make the decision.
    """

    prob: Array
    iter: Array
    sufficient_iter: Array
    median_expected_dist: Array
    steps_float: Array


@jax.jit
def expected_squared_jump_jax(
    proposal_dist: Array,
    prob_accept: Array,
) -> Array:
    """
    Computes the expected squared jump distance.

    The proposal distance is squared.
    The result is then weighted by the acceptance probability.
    Large proposed moves matter less if they are rarely accepted.

    Parameters:
    -----------
    proposal_dist:
        proposal distances, shape (...).
    prob_accept:
        acceptance probabilities, shape (...).
        Must be broadcast-compatible with proposal_dist.

    Returns:
    --------
    Array:
        expected squared jump distance, shape (...).
    """
    proposal_dist = jnp.asarray(proposal_dist)
    prob_accept = jnp.asarray(prob_accept)

    dtype = jnp.result_type(proposal_dist, prob_accept, jnp.asarray(1.0))
    proposal_dist = proposal_dist.astype(dtype)
    prob_accept = prob_accept.astype(dtype)

    return prob_accept * jnp.square(proposal_dist)


@jax.jit
def time_steps_jax(
    proposal_dist: Array,
    prob_accept: Array,
    threshold: Array,
    rho: Array = jnp.asarray(0.5),
    max_t: Array = jnp.asarray(10, dtype=jnp.int32),
) -> RequiredMHSteps:
    """
    Estimates how many MH steps are needed to reach a distance threshold.

    The function first computes expected squared jump distances.
    It then takes their median as a typical expected movement size.
    The threshold is divided by this median movement size.
    This gives an estimated number of MH steps.

    If the estimate is valid and no larger than max_t,
    the function returns that estimate after rounding up.
    Otherwise, it returns max_t and marks the estimate as insufficient.

    Parameters:
    -----------
    proposal_dist:
        proposal distances for the particles, shape (N,).
    prob_accept:
        acceptance probabilities for the particles, shape (N,).
    threshold:
        target distance threshold.
        Must be finite and non-negative to give a valid estimate.
    rho:
        probability value returned when the step estimate is sufficient.
    max_t:
        maximum allowed number of MH steps.

    Returns:
    --------
    RequiredMHSteps:
        stores the returned probability, selected number of steps,
        sufficiency flag, median expected distance, and raw step estimate.
    """
    proposal_dist = jnp.asarray(proposal_dist)
    prob_accept = jnp.asarray(prob_accept)

    dtype = jnp.result_type(
        proposal_dist,
        prob_accept,
        threshold,
        rho,
        jnp.asarray(1.0),
    )

    threshold = jnp.asarray(threshold, dtype=dtype)
    rho = jnp.asarray(rho, dtype=dtype)
    max_t = jnp.asarray(max_t, dtype=jnp.int32)

    expected_dist = expected_squared_jump_jax(
        proposal_dist=proposal_dist,
        prob_accept=prob_accept,
    ).astype(dtype)

    median_expected_dist = jnp.median(expected_dist)

    valid = (
        jnp.isfinite(median_expected_dist)
        & (median_expected_dist > jnp.asarray(0.0, dtype=dtype))
        & jnp.isfinite(threshold)
        & (threshold >= jnp.asarray(0.0, dtype=dtype))
    )

    steps_float_raw = threshold / jnp.where(
        valid,
        median_expected_dist,
        jnp.asarray(1.0, dtype=dtype),
    )

    steps_float = jnp.where(
        valid,
        steps_float_raw,
        jnp.asarray(jnp.inf, dtype=dtype),
    )

    iter_raw = jnp.ceil(steps_float).astype(jnp.int32)
    iter_raw = jnp.maximum(iter_raw, jnp.asarray(1, dtype=jnp.int32))

    sufficient_iter = valid & (iter_raw <= max_t)

    iter_out = jnp.where(
        sufficient_iter,
        iter_raw,
        max_t,
    )

    prob_out = jnp.where(
        sufficient_iter,
        rho,
        jnp.asarray(jnp.nan, dtype=dtype),
    )

    return RequiredMHSteps(
        prob=prob_out,
        iter=iter_out,
        sufficient_iter=sufficient_iter,
        median_expected_dist=median_expected_dist,
        steps_float=steps_float,
    )


time_steps_to_min_quantile_dist_median_batched_jax = jax.jit(
    jax.vmap(
        time_steps_jax,
        in_axes=(0, 0, None, None, None),
        out_axes=0,
    )
)
