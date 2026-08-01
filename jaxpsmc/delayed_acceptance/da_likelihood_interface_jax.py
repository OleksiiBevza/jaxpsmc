from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array

TYPE_APPROX_POSTERIOR = jnp.int32(0)
TYPE_APPROX_LIKELIHOOD = jnp.int32(1)
TYPE_FULL_LIKELIHOOD = jnp.int32(2)
TYPE_FULL_POSTERIOR = jnp.int32(3)
TYPE_PRIOR = jnp.int32(4)


class DALogTargetEval(NamedTuple):
    """
    Stores the result of one delayed-acceptance target evaluation.

    The object contains the selected log-target value.
    It also stores the log-likelihood, approximate likelihood,
    prior values, and the number of model calls used.

    Parameters:
    -----------
    value:
        evaluated log-target value, shape (N,).
    logl_full:
        full log-likelihood values, shape (N,).
        Entries are NaN when the full likelihood was not evaluated.
    logl_approx:
        approximate log-likelihood values, shape (N,).
        Entries are NaN when the approximate likelihood was not evaluated.
    logl_approx_base:
        base approximate log-likelihood values, shape (N,).
        Used when annealing starts from the approximate model.
    logp:
        log-prior values, shape (N,).
        Entries are NaN when the prior was not evaluated.
    full_calls:
        number of full likelihood evaluations.
    approx_calls:
        number of approximate likelihood evaluations.
    prior_calls:
        number of prior evaluations.

    Returns:
    --------
    DALogTargetEval:
        stores log-target values and bookkeeping information
        for one batch of particles.
    """

    value: Array
    logl_full: Array
    logl_approx: Array
    logl_approx_base: Array
    logp: Array
    full_calls: Array
    approx_calls: Array
    prior_calls: Array


def da_target_type(name: str) -> jnp.int32:
    """
    Converts a target type name into an integer code.

    The code is used by JAX control flow.
    This avoids branching on Python strings inside jitted code.

    Parameters:
    -----------
    name:
        name of the target type.
        Allowed values are:
        "approx_posterior",
        "approx_likelihood",
        "full_likelihood",
        "full_posterior",
        and "prior".

    Returns:
    --------
    jnp.int32:
        integer code for the requested target type.

    Raises:
    -------
    ValueError:
        raised when the target type name is unknown.
    """
    name = str(name).lower()
    table = {
        "approx_posterior": TYPE_APPROX_POSTERIOR,
        "approx_likelihood": TYPE_APPROX_LIKELIHOOD,
        "full_likelihood": TYPE_FULL_LIKELIHOOD,
        "full_posterior": TYPE_FULL_POSTERIOR,
        "prior": TYPE_PRIOR,
    }
    if name not in table:
        raise ValueError(
            "type must be one of: approx_posterior, approx_likelihood, "
            "full_likelihood, full_posterior, prior."
        )
    return table[name]


@jax.jit
def annealed_log_target_jax(
    *,
    logl_full: Array,
    logl_approx: Array,
    logl_approx_base: Array,
    logp: Array,
    beta: Array,
    type_code: Array,
    start_from_approx: Array = jnp.asarray(False),
    max_approx_anneal: Array = jnp.asarray(1.0),
) -> Array:
    """
    Builds an annealed log-target from already computed log-density parts.

    The function selects which target to use through type_code.
    It can return an approximate posterior, approximate likelihood,
    full likelihood, full posterior, or prior.

    The annealing value beta controls how strongly the likelihood is used.
    When beta is zero, the likelihood contribution is removed.
    When beta is one, the selected likelihood is fully used.

    If start_from_approx is True, the target also includes a base
    approximate likelihood term during annealing.
    This can make the path start closer to the approximate model.

    Parameters:
    -----------
    logl_full:
        full log-likelihood values, shape (N,).
    logl_approx:
        approximate log-likelihood values, shape (N,).
    logl_approx_base:
        base approximate log-likelihood values, shape (N,).
        Used only when start_from_approx is True.
    logp:
        log-prior values, shape (N,).
    beta:
        annealing value.
        Usually between 0 and 1.
    type_code:
        integer code selecting the target type.
        Use da_target_type to create this safely.
    start_from_approx:
        whether to include the base approximate likelihood
        in the annealing path.
    max_approx_anneal:
        scale applied to logl_approx_base when start_from_approx is True.

    Returns:
    --------
    Array:
        selected annealed log-target value, shape (N,).
    """
    beta = jnp.asarray(beta)
    type_code = jnp.asarray(type_code, dtype=jnp.int32)
    start_from_approx = jnp.asarray(start_from_approx, dtype=bool)
    max_approx_anneal = jnp.asarray(max_approx_anneal, dtype=beta.dtype)

    logl_full = jnp.asarray(logl_full, dtype=beta.dtype)
    logl_approx = jnp.asarray(logl_approx, dtype=beta.dtype)
    logl_approx_base = jnp.asarray(logl_approx_base, dtype=beta.dtype)
    logp = jnp.asarray(logp, dtype=beta.dtype)

    operand = (
        logl_full,
        logl_approx,
        logl_approx_base,
        logp,
        beta,
        start_from_approx,
        max_approx_anneal,
    )

    def approx_posterior(op):
        _, la, la0, lp, b, sfa, max_a = op

        def sfa_branch(_):
            return b * la + (1.0 - b) * max_a * la0 + lp

        def normal_branch(_):
            return b * la + lp

        return lax.cond(sfa, sfa_branch, normal_branch, operand=None)

    def approx_likelihood(op):
        _, la, _, _, b, _, _ = op
        return b * la

    def full_likelihood(op):
        lf, _, _, _, b, _, _ = op
        return b * lf

    def full_posterior(op):
        lf, _, la0, lp, b, sfa, max_a = op

        def sfa_branch(_):
            return b * lf + (1.0 - b) * max_a * la0 + lp

        def normal_branch(_):
            return b * lf + lp

        return lax.cond(sfa, sfa_branch, normal_branch, operand=None)

    def prior(op):
        _, _, _, lp, _, _, _ = op
        return lp

    return lax.switch(
        type_code,
        (
            approx_posterior,
            approx_likelihood,
            full_likelihood,
            full_posterior,
            prior,
        ),
        operand,
    )


def make_evaluator_jax(
    *,
    log_likelihood_single: Callable[[Array], Array],
    log_like_approx_single: Callable[[Array], Array],
    log_prior_single: Callable[[Array], Array],
    transform_single: Callable[[Array], Array] | None = None,
) -> Callable[..., DALogTargetEval]:
    """
    Creates a batched JAX evaluator for delayed-acceptance targets.

    The input functions evaluate one particle at a time.
    This function vectorizes them over a batch of particles.
    It then returns a jitted evaluate function.

    The returned evaluator can compute different target types.
    It also counts how many full, approximate, and prior calls were used.

    Parameters:
    -----------
    log_likelihood_single:
        function that computes the full log-likelihood for one particle.
    log_like_approx_single:
        function that computes the approximate log-likelihood for one particle.
    log_prior_single:
        function that computes the log-prior for one particle.
    transform_single:
        optional function applied before the approximate likelihood.
        If None, particles are used without transformation.

    Returns:
    --------
    Callable[..., DALogTargetEval]:
        jitted batched evaluator.
        It maps particles to selected log-target values
        and model-call counts.
    """
    if transform_single is None:

        def transform_single(x: Array) -> Array:
            return x

    log_likelihood_batch = jax.vmap(log_likelihood_single, in_axes=0, out_axes=0)
    log_like_approx_batch = jax.vmap(log_like_approx_single, in_axes=0, out_axes=0)
    log_prior_batch = jax.vmap(log_prior_single, in_axes=0, out_axes=0)
    transform_batch = jax.vmap(transform_single, in_axes=0, out_axes=0)

    @jax.jit
    def evaluate(
        particles: Array,
        beta: Array = jnp.asarray(1.0),
        type_code: Array = TYPE_FULL_POSTERIOR,
        start_from_approx: Array = jnp.asarray(False),
        max_approx_anneal: Array = jnp.asarray(1.0),
    ) -> DALogTargetEval:
        """
        Evaluates one selected delayed-acceptance target for many particles.

        The function chooses the target using type_code.
        It evaluates only the log-density parts needed for that target.
        Unused values are filled with NaN.
        Call counters record which model parts were evaluated.

        Parameters:
        -----------
        particles:
            particle batch, shape (N, D).
        beta:
            annealing value.
            Usually between 0 and 1.
        type_code:
            integer code selecting the target type.
            Use da_target_type to create this safely.
        start_from_approx:
            whether to include the base approximate likelihood
            in the annealing path.
        max_approx_anneal:
            scale applied to the base approximate likelihood
            when start_from_approx is True.

        Returns:
        --------
        DALogTargetEval:
            selected log-target values, component values,
            and model-call counts.
        """
        particles = jnp.asarray(particles)
        beta = jnp.asarray(beta)
        dtype = jnp.result_type(particles, beta, jnp.asarray(1.0))
        beta = beta.astype(dtype)

        type_code_i = jnp.asarray(type_code, dtype=jnp.int32)
        start_from_approx_b = jnp.asarray(start_from_approx, dtype=bool)
        max_approx_anneal_f = jnp.asarray(max_approx_anneal, dtype=dtype)

        n = jnp.asarray(particles.shape[0], dtype=jnp.int32)
        zero_i = jnp.asarray(0, dtype=jnp.int32)
        nan_vec = jnp.full((particles.shape[0],), jnp.nan, dtype=dtype)

        operand = (
            particles,
            beta,
            start_from_approx_b,
            max_approx_anneal_f,
        )

        def eval_approx_posterior(op):
            x, b, sfa, max_a = op
            x_t = transform_batch(x)
            la = jnp.asarray(log_like_approx_batch(x_t), dtype=dtype)
            lp = jnp.asarray(log_prior_batch(x), dtype=dtype)

            def sfa_branch(_):
                la0 = jnp.asarray(log_like_approx_batch(x), dtype=dtype)
                val = b * la + (1.0 - b) * max_a * la0 + lp
                return DALogTargetEval(val, nan_vec, la, la0, lp, zero_i, n + n, n)

            def normal_branch(_):
                val = b * la + lp
                return DALogTargetEval(val, nan_vec, la, nan_vec, lp, zero_i, n, n)

            return lax.cond(sfa, sfa_branch, normal_branch, operand=None)

        def eval_approx_likelihood(op):
            x, b, _, _ = op
            x_t = transform_batch(x)
            la = jnp.asarray(log_like_approx_batch(x_t), dtype=dtype)
            val = b * la
            return DALogTargetEval(
                val, nan_vec, la, nan_vec, nan_vec, zero_i, n, zero_i
            )

        def eval_full_likelihood(op):
            x, b, _, _ = op
            lf = jnp.asarray(log_likelihood_batch(x), dtype=dtype)
            val = b * lf
            return DALogTargetEval(
                val, lf, nan_vec, nan_vec, nan_vec, n, zero_i, zero_i
            )

        def eval_full_posterior(op):
            x, b, sfa, max_a = op
            lf = jnp.asarray(log_likelihood_batch(x), dtype=dtype)
            lp = jnp.asarray(log_prior_batch(x), dtype=dtype)

            def sfa_branch(_):
                la0 = jnp.asarray(log_like_approx_batch(x), dtype=dtype)
                val = b * lf + (1.0 - b) * max_a * la0 + lp
                return DALogTargetEval(val, lf, nan_vec, la0, lp, n, n, n)

            def normal_branch(_):
                val = b * lf + lp
                return DALogTargetEval(val, lf, nan_vec, nan_vec, lp, n, zero_i, n)

            return lax.cond(sfa, sfa_branch, normal_branch, operand=None)

        def eval_prior(op):
            x, _, _, _ = op
            lp = jnp.asarray(log_prior_batch(x), dtype=dtype)
            return DALogTargetEval(lp, nan_vec, nan_vec, nan_vec, lp, zero_i, zero_i, n)

        return lax.switch(
            type_code_i,
            (
                eval_approx_posterior,
                eval_approx_likelihood,
                eval_full_likelihood,
                eval_full_posterior,
                eval_prior,
            ),
            operand,
        )

    return evaluate
