from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class TuningSelection(NamedTuple):
    """
    Stores the result of selecting the best tuning option.

    The best option is the one with the smallest valid cost.
    Invalid options are assigned infinite cost before selection.

    Parameters:
    -----------
    index:
        index of the option with the smallest cost.
    costs:
        cost value for each tuning option.
        Invalid options have cost infinity.
    min_cost:
        smallest cost among all options.
    valid:
        Boolean value showing whether at least one option was valid.

    Returns:
    --------
    TuningSelection:
        stores the selected index, all costs, the minimum cost,
        and whether the selection is usable.
    """

    index: Array
    costs: Array
    min_cost: Array
    valid: Array


@jax.jit
def da_mh_costs_jax(
    min_steps: Array,
    surrogate_acceptance: Array,
    surrogate_cost: Array,
    full_cost: Array,
    valid_mask: Array = jnp.asarray(True),
) -> Array:
    """
    Computes delayed-acceptance MH costs for tuning options.

    Each option has a number of required MH steps.
    Each step has a surrogate-model cost.
    A full-model cost is paid only with the surrogate acceptance probability.

    Invalid options are assigned infinite cost.
    This makes them impossible to select as the best option.

    Parameters:
    -----------
    min_steps:
        required number of MH steps for each option, shape (...).
        Values must be positive and finite.
    surrogate_acceptance:
        probability of passing the surrogate stage, shape (...).
        Values must be between 0 and 1.
    surrogate_cost:
        cost of one surrogate evaluation.
        Must be finite and non-negative.
    full_cost:
        cost of one full evaluation.
        Must be finite and non-negative.
    valid_mask:
        Boolean mask showing which options are allowed.
        Must be broadcast-compatible with the computed costs.

    Returns:
    --------
    Array:
        delayed-acceptance MH cost for each option, shape (...).
        Invalid options are returned as infinity.
    """
    min_steps = jnp.asarray(min_steps)
    surrogate_acceptance = jnp.asarray(surrogate_acceptance)

    dtype = jnp.result_type(
        min_steps,
        surrogate_acceptance,
        surrogate_cost,
        full_cost,
        jnp.asarray(1.0),
    )

    min_steps = min_steps.astype(dtype)
    surrogate_acceptance = surrogate_acceptance.astype(dtype)
    surrogate_cost = jnp.asarray(surrogate_cost, dtype=dtype)
    full_cost = jnp.asarray(full_cost, dtype=dtype)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)

    costs = min_steps * (surrogate_cost + surrogate_acceptance * full_cost)

    finite = (
        jnp.isfinite(costs)
        & jnp.isfinite(min_steps)
        & jnp.isfinite(surrogate_acceptance)
        & jnp.isfinite(surrogate_cost)
        & jnp.isfinite(full_cost)
        & (min_steps > jnp.asarray(0.0, dtype=dtype))
        & (surrogate_acceptance >= jnp.asarray(0.0, dtype=dtype))
        & (surrogate_acceptance <= jnp.asarray(1.0, dtype=dtype))
        & (surrogate_cost >= jnp.asarray(0.0, dtype=dtype))
        & (full_cost >= jnp.asarray(0.0, dtype=dtype))
    )

    valid = valid_mask & finite

    return jnp.where(
        valid,
        costs,
        jnp.asarray(jnp.inf, dtype=dtype),
    )


@jax.jit
def min_da_mh_cost_jax(
    min_steps: Array,
    surrogate_acceptance: Array,
    surrogate_cost: Array,
    full_cost: Array,
    valid_mask: Array = jnp.asarray(True),
) -> TuningSelection:
    """
    Selects the cheapest delayed-acceptance MH tuning option.

    The function first computes costs for all options.
    It then selects the option with the smallest valid cost.
    If no option is valid, the minimum cost is infinity.

    Parameters:
    -----------
    min_steps:
        required number of MH steps for each option, shape (...).
        Values must be positive and finite.
    surrogate_acceptance:
        probability of passing the surrogate stage, shape (...).
        Values must be between 0 and 1.
    surrogate_cost:
        cost of one surrogate evaluation.
        Must be finite and non-negative.
    full_cost:
        cost of one full evaluation.
        Must be finite and non-negative.
    valid_mask:
        Boolean mask showing which options are allowed.
        Must be broadcast-compatible with the computed costs.

    Returns:
    --------
    TuningSelection:
        selected option index, all option costs, minimum cost,
        and validity flag.
    """
    costs = da_mh_costs_jax(
        min_steps=min_steps,
        surrogate_acceptance=surrogate_acceptance,
        surrogate_cost=surrogate_cost,
        full_cost=full_cost,
        valid_mask=valid_mask,
    )

    index = jnp.argmin(costs).astype(jnp.int32)
    min_cost = jnp.min(costs)

    valid = jnp.isfinite(min_cost)

    return TuningSelection(
        index=index,
        costs=costs,
        min_cost=min_cost,
        valid=valid,
    )


@jax.jit
def mh_costs_jax(
    min_steps: Array,
    valid_mask: Array = jnp.asarray(True),
) -> Array:
    """
    Computes standard MH costs for tuning options.

    For standard MH, the cost is represented by the required number
    of MH steps. No surrogate/full cost split is used here.

    Invalid options are assigned infinite cost.
    This makes them impossible to select as the best option.

    Parameters:
    -----------
    min_steps:
        required number of MH steps for each option, shape (...).
        Values must be positive and finite.
    valid_mask:
        Boolean mask showing which options are allowed.
        Must be broadcast-compatible with min_steps.

    Returns:
    --------
    Array:
        MH cost for each option, shape (...).
        Invalid options are returned as infinity.
    """
    min_steps = jnp.asarray(min_steps)
    dtype = jnp.result_type(min_steps, jnp.asarray(1.0))

    min_steps = min_steps.astype(dtype)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)

    valid = (
        valid_mask
        & jnp.isfinite(min_steps)
        & (min_steps > jnp.asarray(0.0, dtype=dtype))
    )

    return jnp.where(
        valid,
        min_steps,
        jnp.asarray(jnp.inf, dtype=dtype),
    )


@jax.jit
def min_mh_cost_jax(
    min_steps: Array,
    valid_mask: Array = jnp.asarray(True),
) -> TuningSelection:
    """
    Selects the cheapest standard MH tuning option.

    The function first computes standard MH costs.
    It then selects the option with the smallest valid cost.
    If no option is valid, the minimum cost is infinity.

    Parameters:
    -----------
    min_steps:
        required number of MH steps for each option, shape (...).
        Values must be positive and finite.
    valid_mask:
        Boolean mask showing which options are allowed.
        Must be broadcast-compatible with min_steps.

    Returns:
    --------
    TuningSelection:
        selected option index, all option costs, minimum cost,
        and validity flag.
    """
    costs = mh_costs_jax(
        min_steps=min_steps,
        valid_mask=valid_mask,
    )

    index = jnp.argmin(costs).astype(jnp.int32)
    min_cost = jnp.min(costs)

    valid = jnp.isfinite(min_cost)

    return TuningSelection(
        index=index,
        costs=costs,
        min_cost=min_cost,
        valid=valid,
    )
