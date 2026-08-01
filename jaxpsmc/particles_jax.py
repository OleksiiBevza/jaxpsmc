from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class ParticlesState(NamedTuple):
    """
    Stores the full particle history across recorded SMC steps.

    This object contains fixed-size buffers.
    Each new particle step is written into the next free position.
    The field t stores how many steps have been recorded.

    Parameters:
    -----------
    t:
        number of recorded steps.
    u:
        stored latent particles, shape (T, N, D).
    x:
        stored transformed particles, shape (T, N, D).
    logdetj:
        stored log absolute Jacobian terms, shape (T, N).
    logl:
        stored log-likelihood values, shape (T, N).
    logp:
        stored log-prior values, shape (T, N).
    logw:
        stored log-weight values, shape (T, N).
    blobs:
        stored extra per-particle values, shape (T, N, B).
        B can be zero.
    iter:
        stored iteration index for each step, shape (T,).
    logz:
        stored log normalizing constant values, shape (T,).
    calls:
        stored function-call counts, shape (T,).
    steps:
        stored inner-step counts, shape (T,).
    efficiency:
        stored efficiency values, shape (T,).
    ess:
        stored effective sample size values, shape (T,).
    accept:
        stored acceptance values, shape (T,).
    beta:
        stored annealing values, shape (T,).

    Returns:
    --------
    ParticlesState:
        fixed-size particle history buffers and the current write index.
    """

    t: jax.Array  # nr of filled steps: [0 to max_steps)
    u: jax.Array  # (T, N, D)   latent values for each step
    x: jax.Array  # (T, N, D)   transformed values for each step
    logdetj: jax.Array  # (T, N)      log absolute Jacobian terms
    logl: jax.Array  # (T, N)      log-likelihood values
    logp: jax.Array  # (T, N)      log-prior values
    logw: jax.Array  # (T, N)      log-weight values

    blobs: jax.Array  # (T, N, B)   where B may be 0. Extra per-particle values
    iter: jax.Array  # (T,)        iteration count per step
    logz: jax.Array  # (T,)        log normalizing constant per step
    calls: jax.Array  # (T,)        function call count per step
    steps: jax.Array  # (T,)        inner step count per step
    efficiency: jax.Array  # (T,)        efficiency value per step
    ess: jax.Array  # (T,)        effective sample size per step
    accept: jax.Array  # (T,)        acceptance value per step
    beta: jax.Array  # (T,)        beta value per step


class ParticlesStep(NamedTuple):
    """
    Stores particle values for one recorded step.

    This object contains one batch of particles.
    It is written into a ParticlesState by record_step_jax.

    Parameters:
    -----------
    u:
        latent particles, shape (N, D).
    x:
        transformed particles, shape (N, D).
    logdetj:
        log absolute Jacobian terms, shape (N,).
    logl:
        log-likelihood values, shape (N,).
    logp:
        log-prior values, shape (N,).
    logw:
        log-weight values, shape (N,).
    blobs:
        extra per-particle values, shape (N, B).
    iter:
        iteration index for this step.
    logz:
        log normalizing constant for this step.
    calls:
        function-call count for this step.
    steps:
        number of inner mutation steps for this step.
    efficiency:
        efficiency value for this step.
    ess:
        effective sample size for this step.
    accept:
        acceptance value for this step.
    beta:
        annealing value for this step.

    Returns:
    --------
    ParticlesStep:
        particle values and diagnostics for one step.
    """

    # single-step values (no Python dicts)
    u: jax.Array  # (N, D)
    x: jax.Array  # (N, D)
    logdetj: jax.Array  # (N,)
    logl: jax.Array  # (N,)
    logp: jax.Array  # (N,)
    logw: jax.Array  # (N,)

    blobs: jax.Array  # (N, B)
    iter: jax.Array  # () or (1,) int
    logz: jax.Array  # () float
    calls: jax.Array  # () float/int
    steps: jax.Array  # () float/int
    efficiency: jax.Array  # () float
    ess: jax.Array  # () float
    accept: jax.Array  # () float
    beta: jax.Array  # () float


def init_particles_state_jax(
    max_steps: int,
    n_particles: int,
    n_dim: int,
    blob_dim: int = 0,
    dtype=jnp.float32,
) -> ParticlesState:
    """
    Creates an empty particle history state.

    The function allocates fixed-size arrays.
    These arrays store particle values and diagnostics for later steps.
    The initial number of recorded steps is zero.

    Parameters:
    -----------
    max_steps:
        maximum number of steps to store.
    n_particles:
        number of particles stored at each step.
    n_dim:
        dimension of each particle.
    blob_dim:
        dimension of the extra blob vector.
        Use zero if no extra values are stored.
    dtype:
        numeric dtype used for floating-point buffers.

    Returns:
    --------
    ParticlesState:
        zero-filled particle history buffers.
        The log-weight buffer is initialized to -inf.
    """
    # names for buffer sizes.
    T, N, D, B = max_steps, n_particles, n_dim, blob_dim
    # buffers for arrays with shape (T, N, D)
    zeros_TND = jnp.zeros((T, N, D), dtype=dtype)
    # buffers for arrays with shape (T, N)
    zeros_TN = jnp.zeros((T, N), dtype=dtype)
    # buffers for arrays with shape (T, N, B)
    zeros_TNB = jnp.zeros((T, N, B), dtype=dtype)
    # buffers for arrays with shape (T,)
    zeros_T = jnp.zeros((T,), dtype=dtype)

    # build the initial state
    return ParticlesState(
        t=jnp.array(0, dtype=jnp.int32),
        u=zeros_TND,
        x=zeros_TND,
        logdetj=zeros_TN,
        logl=zeros_TN,
        logp=zeros_TN,
        logw=jnp.full((T, N), -jnp.inf, dtype=dtype),
        blobs=zeros_TNB,
        iter=jnp.zeros((T,), dtype=jnp.int32),
        logz=zeros_T,
        calls=zeros_T,
        steps=zeros_T,
        efficiency=zeros_T,
        ess=zeros_T,
        accept=zeros_T,
        beta=zeros_T,
    )


@jax.jit
def record_step_jax(state: ParticlesState, step: ParticlesStep) -> ParticlesState:
    """
    Writes one particle step into the history state.

    The step is written at the current index state.t.
    The counter t is then increased by one.
    If the buffer is already full, the last slot is overwritten.

    Parameters:
    -----------
    state:
        current particle history state.
    step:
        particle values and diagnostics to record.

    Returns:
    --------
    ParticlesState:
        updated particle history state.
    """
    # define total buffer length
    T = state.logl.shape[0]
    # index stays inside the allocated buffers
    idx = jnp.minimum(state.t, jnp.array(T - 1, dtype=state.t.dtype))
    # write new step into every buffer and advance t
    state2 = ParticlesState(
        t=jnp.minimum(state.t + 1, jnp.array(T, dtype=state.t.dtype)),
        u=state.u.at[idx].set(step.u),
        x=state.x.at[idx].set(step.x),
        logdetj=state.logdetj.at[idx].set(step.logdetj),
        logl=state.logl.at[idx].set(step.logl),
        logp=state.logp.at[idx].set(step.logp),
        logw=state.logw.at[idx].set(step.logw),
        blobs=state.blobs.at[idx].set(step.blobs),
        iter=state.iter.at[idx].set(step.iter.astype(state.iter.dtype)),
        logz=state.logz.at[idx].set(step.logz),
        calls=state.calls.at[idx].set(step.calls),
        steps=state.steps.at[idx].set(step.steps),
        efficiency=state.efficiency.at[idx].set(step.efficiency),
        ess=state.ess.at[idx].set(step.ess),
        accept=state.accept.at[idx].set(step.accept),
        beta=state.beta.at[idx].set(step.beta),
    )
    return state2


@jax.jit
def pop_step_jax(state: ParticlesState) -> ParticlesState:
    """
    Removes the most recent recorded step from the active history.

    The stored arrays are not cleared.
    Only the counter t is reduced by one.
    Values after the new t are treated as inactive.

    Parameters:
    -----------
    state:
        current particle history state.

    Returns:
    --------
    ParticlesState:
        same buffers with t reduced by one.
    """
    # Lower t but not below zero
    t_new = jnp.maximum(state.t - 1, jnp.array(0, dtype=state.t.dtype))
    # keep buffers unchanged and only update t
    return state._replace(t=t_new)


@jax.jit
def step_mask_jax(state: ParticlesState) -> jax.Array:
    """
    Builds a mask for recorded steps.

    Entries before state.t are active.
    Entries at or after state.t are inactive.

    Parameters:
    -----------
    state:
        current particle history state.

    Returns:
    --------
    jax.Array:
        Boolean mask of shape (T,).
        True means the step has been recorded.
    """
    T = state.logl.shape[0]
    return jnp.arange(T) < state.t  # (T,) bool


@jax.jit
def compute_logw_and_logz_jax(
    state: ParticlesState,
    beta_final: float | jax.Array = 1.0,
    normalize: bool | jax.Array = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Computes final flattened log weights and log normalizing constant.

    The function uses the recorded particle history.
    It builds weights for all active stored particles.
    Inactive steps are assigned log weight -inf.

    If normalize is True, the returned log weights are normalized.
    Then their exponentiated values sum to one.

    Parameters:
    -----------
    state:
        current particle history state.
    beta_final:
        final annealing value used in the weight formula.
    normalize:
        if True, normalize the flattened log weights.

    Returns:
    --------
    tuple[jax.Array, jax.Array, jax.Array]:
        logw_flat:
            flattened final log weights, shape (T * N,).
        logz_new:
            final log normalizing constant estimate.
        mask_flat:
            Boolean mask for active flattened entries, shape (T * N,).
    """
    # read stored arrays used in the calculation.
    logl = state.logl  # (T, N)
    beta = state.beta  # (T,)
    logz = state.logz  # (T,)
    # read history sizes
    T, N = logl.shape
    n_steps = state.t  # CHECK cause dynamic int32 scalar

    # build masks for valid steps and valid flattened entries
    mask_t = jnp.arange(T) < n_steps  # (T,)
    mask_i = mask_t  # (T,)  sum over i uses active prefix
    mask_flat = jnp.repeat(mask_t, N)  # (T*N,)

    # Build tensor used for log-mean-exp term
    # b[i, j, n] = logl[j, n] * beta[i] - logz[i]
    b = logl[None, :, :] * beta[:, None, None] - logz[:, None, None]  # (T, T, N)

    # ignore inactive i steps from the log-mean-exp (in the average)
    b = jnp.where(mask_i[:, None, None], b, -jnp.inf)

    # use at least one step in denominator to avoid log(0)
    denom_steps = jnp.maximum(n_steps, jnp.array(1, dtype=n_steps.dtype)).astype(
        logl.dtype
    )
    # compute the log-mean-exp term.
    B = jax.nn.logsumexp(b, axis=0) - jnp.log(denom_steps)  # (T, N)
    # compute unnormalized log weights
    A = logl * jnp.asarray(beta_final, dtype=logl.dtype)  # (T, N)
    logw = A - B  # (T, N)

    # ignore inactive steps j in final weights
    logw = jnp.where(mask_t[:, None], logw, -jnp.inf)  # (T, N)
    logw_flat = logw.reshape(-1)  # (T*N,)
    # compute final log normalizing constant
    denom_particles = denom_steps.astype(logl.dtype) * jnp.asarray(N, dtype=logl.dtype)
    logz_new = jax.nn.logsumexp(logw_flat) - jnp.log(denom_particles)
    # convert normalize to a JAX value for lax.cond
    normalize_arr = jnp.asarray(normalize)

    def _norm(lw):
        """
        Normalizes flattened log weights.

        Parameters:
        -----------
        lw:
            unnormalized flattened log weights.

        Returns:
        --------
        jax.Array:
            normalized flattened log weights.
        """
        # subtract logsumexp so the weights sum to one
        return lw - jax.nn.logsumexp(lw)

    # normalize
    logw_flat = lax.cond(normalize_arr, _norm, lambda lw: lw, logw_flat)

    return logw_flat, logz_new, mask_flat


@jax.jit
def compute_results_jax(
    state: ParticlesState,
    beta_final: float | jax.Array = 1.0,
    normalize: bool | jax.Array = True,
) -> dict[str, jax.Array]:
    """
    Builds a result dictionary from the particle history state.

    The dictionary contains masks, final weights, final logz,
    and all stored particle history arrays.
    This is the main output format used after sampling.

    Parameters:
    -----------
    state:
        current particle history state.
    beta_final:
        final annealing value used in the weight formula.
    normalize:
        if True, normalize the flattened log weights.

    Returns:
    --------
    Dict[str, jax.Array]:
        result dictionary with particle history, final weights,
        final logz, and masks for active entries.
    """
    # compute final flattened weights and masks
    logw_flat, logz_new, mask_flat = compute_logw_and_logz_jax(
        state, beta_final, normalize
    )
    mask_t = step_mask_jax(state)

    # Return all values as a JAX pytree dictionary
    return {
        "t": state.t,
        "mask_t": mask_t,
        "mask_flat": mask_flat,
        "logz_new": logz_new,
        "logw_flat": logw_flat,
        "u": state.u,
        "x": state.x,
        "logdetj": state.logdetj,
        "logl": state.logl,
        "logp": state.logp,
        "logw_hist": state.logw,
        "blobs": state.blobs,
        "iter": state.iter,
        "logz": state.logz,
        "calls": state.calls,
        "steps": state.steps,
        "efficiency": state.efficiency,
        "ess": state.ess,
        "accept": state.accept,
        "beta": state.beta,
    }
