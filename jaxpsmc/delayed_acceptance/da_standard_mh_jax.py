from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

try:
    from .da_likelihood_interface_jax import TYPE_FULL_POSTERIOR
except Exception:  
    TYPE_FULL_POSTERIOR = jnp.int32(3)


Array = jax.Array


class StandardMHStep(NamedTuple):
    """
    Stores the result of one standard Metropolis-Hastings step.

    The object contains the accept/reject decision.
    It also stores the acceptance probability, proposal distance,
    log-target values, and model-call counts.

    Parameters:
    -----------
    key:
        updated JAX random key after the accept/reject draw.
    accept:
        Boolean array showing which proposals were accepted.
    prob_accept:
        MH acceptance probability for each proposal.
    proposal_dist:
        Mahalanobis distance between proposed and old particles.
    actual_dist:
        proposal distance counted only for accepted proposals.
    expected_dist:
        proposal distance weighted by the acceptance probability.
    log_accept_ratio:
        log acceptance ratio for each proposal.
    new_logtarget:
        log-target value at proposed particles.
    old_logtarget:
        log-target value at old particles.
    full_calls:
        number of full likelihood evaluations used by this step.
    approx_calls:
        number of approximate likelihood evaluations used by this step.
    prior_calls:
        number of prior evaluations used by this step.

    Returns:
    --------
    StandardMHStep:
        stores MH decisions, probabilities, distances, log-targets,
        and evaluation counts for one batch of proposals.
    """
    key: Array
    accept: Array
    prob_accept: Array
    proposal_dist: Array
    actual_dist: Array
    expected_dist: Array
    log_accept_ratio: Array
    new_logtarget: Array
    old_logtarget: Array
    full_calls: Array
    approx_calls: Array
    prior_calls: Array


@jax.jit
def _proposal_distance_jax(
    new_particles: Array,
    old_particles: Array,
    cov: Array,
) -> Array:
    """
    Computes Mahalanobis proposal distances.

    The function measures how far each proposed particle moved.
    The covariance matrix defines the scale of the distance.
    Movement in high-variance directions counts less.
    Movement in low-variance directions counts more.

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
        Mahalanobis distance for each proposal, shape (N,).
    """
    new_particles = jnp.asarray(new_particles)
    old_particles = jnp.asarray(old_particles)
    cov = jnp.asarray(cov)

    diff = new_particles - old_particles
    solved = jnp.linalg.solve(cov, diff.T).T
    maha = jnp.sum(diff * solved, axis=1)

    return jnp.sqrt(jnp.maximum(maha, jnp.asarray(0.0, dtype=maha.dtype)))


@jax.jit
def standard_mh_step_from_logtargets_jax(
    key: Array,
    new_particles: Array,
    old_particles: Array,
    cov: Array,
    new_logtarget: Array,
    old_logtarget: Array,
) -> StandardMHStep:
    """
    Runs one MH accept/reject step from precomputed log-target values.

    The function compares the proposed and old log-target values.
    It accepts each proposal with probability min(1, exp(log ratio)).
    It also computes proposal distances and expected movement.

    This function does not evaluate the target itself.
    Therefore, the returned call counts are all zero.

    Parameters:
    -----------
    key:
        JAX random key used for the accept/reject draw.
    new_particles:
        proposed particles, shape (N, D).
    old_particles:
        current particles, shape (N, D).
    cov:
        covariance matrix used for Mahalanobis distance, shape (D, D).
        It must be invertible.
    new_logtarget:
        log-target values at proposed particles, shape (N,).
    old_logtarget:
        log-target values at current particles, shape (N,).

    Returns:
    --------
    StandardMHStep:
        stores the updated key, acceptance decisions, probabilities,
        distances, log-target values, and zero evaluation counts.
    """
    new_logtarget = jnp.asarray(new_logtarget)
    old_logtarget = jnp.asarray(old_logtarget)

    dtype = jnp.result_type(
        new_particles,
        old_particles,
        cov,
        new_logtarget,
        old_logtarget,
        jnp.asarray(1.0),
    )

    key, subkey = jax.random.split(key)

    log_accept_ratio = (new_logtarget - old_logtarget).astype(dtype)
    log_prob_accept = jnp.minimum(log_accept_ratio, jnp.asarray(0.0, dtype=dtype))

    prob_accept = jnp.exp(log_prob_accept)
    prob_accept = jnp.where(
        jnp.isfinite(prob_accept),
        prob_accept,
        jnp.asarray(0.0, dtype=dtype),
    )

    log_u = jnp.log(
        jax.random.uniform(subkey, shape=prob_accept.shape, dtype=dtype)
    )
    accept = log_u < log_prob_accept

    proposal_dist = _proposal_distance_jax(
        new_particles,
        old_particles,
        cov,
    ).astype(dtype)

    actual_dist = proposal_dist * accept.astype(dtype)
    expected_dist = proposal_dist * prob_accept

    zero_calls = jnp.asarray(0, dtype=jnp.int32)

    return StandardMHStep(
        key=key,
        accept=accept,
        prob_accept=prob_accept,
        proposal_dist=proposal_dist,
        actual_dist=actual_dist,
        expected_dist=expected_dist,
        log_accept_ratio=log_accept_ratio,
        new_logtarget=new_logtarget,
        old_logtarget=old_logtarget,
        full_calls=zero_calls,
        approx_calls=zero_calls,
        prior_calls=zero_calls,
    )


@partial(jax.jit, static_argnames=("log_target_fn",))
def standard_mh_step_jax(
    key: Array,
    new_particles: Array,
    old_particles: Array,
    cov: Array,
    beta: Array,
    log_target_fn: Callable[..., object],
    type_code: Array = TYPE_FULL_POSTERIOR,
) -> StandardMHStep:
    """
    Runs one standard MH step by evaluating the log target.

    The function evaluates the selected log target at proposed particles.
    It also evaluates the same log target at old particles.
    It then calls standard_mh_step_from_logtargets_jax
    to perform the accept/reject decision.

    The returned call counts are the sum of the calls used for
    the proposed particles and the old particles.

    Parameters:
    -----------
    key:
        JAX random key used for the accept/reject draw.
    new_particles:
        proposed particles, shape (N, D).
    old_particles:
        current particles, shape (N, D).
    cov:
        covariance matrix used for Mahalanobis distance, shape (D, D).
        It must be invertible.
    beta:
        annealing value passed to the log-target evaluator.
    log_target_fn:
        function that evaluates the selected log target.
        It must return an object with value, full_calls,
        approx_calls, and prior_calls fields.
    type_code:
        integer code selecting which target to evaluate.
        The default is the full posterior.

    Returns:
    --------
    StandardMHStep:
        stores the updated key, acceptance decisions, probabilities,
        distances, log-target values, and model-call counts.
    """
    new_eval = log_target_fn(
        new_particles,
        beta=beta,
        type_code=type_code,
    )
    old_eval = log_target_fn(
        old_particles,
        beta=beta,
        type_code=type_code,
    )

    core = standard_mh_step_from_logtargets_jax(
        key=key,
        new_particles=new_particles,
        old_particles=old_particles,
        cov=cov,
        new_logtarget=new_eval.value,
        old_logtarget=old_eval.value,
    )

    return StandardMHStep(
        key=core.key,
        accept=core.accept,
        prob_accept=core.prob_accept,
        proposal_dist=core.proposal_dist,
        actual_dist=core.actual_dist,
        expected_dist=core.expected_dist,
        log_accept_ratio=core.log_accept_ratio,
        new_logtarget=new_eval.value,
        old_logtarget=old_eval.value,
        full_calls=new_eval.full_calls + old_eval.full_calls,
        approx_calls=new_eval.approx_calls + old_eval.approx_calls,
        prior_calls=new_eval.prior_calls + old_eval.prior_calls,
    )