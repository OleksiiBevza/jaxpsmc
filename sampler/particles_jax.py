from __future__ import annotations

from typing import NamedTuple, Dict
import jax
import jax.numpy as jnp
from jax import lax


class ParticlesState(NamedTuple):
    # how many steps filled (0....max_steps)
    t: jax.Array  # int32 scalar

    # history buffers (preallocated)
    u: jax.Array          # (T, N, D)
    x: jax.Array          # (T, N, D)
    logdetj: jax.Array    # (T, N)
    logl: jax.Array       # (T, N)
    logp: jax.Array       # (T, N)
    logw: jax.Array       # (T, N)  

    blobs: jax.Array      # (T, N, B)  (B may be 0)
    iter: jax.Array       # (T,)
    logz: jax.Array       # (T,)
    calls: jax.Array      # (T,)
    steps: jax.Array      # (T,)
    efficiency: jax.Array # (T,)
    ess: jax.Array        # (T,)
    accept: jax.Array     # (T,)
    beta: jax.Array       # (T,)


class ParticlesStep(NamedTuple):
    # single-step values (no Python dicts)
    u: jax.Array          # (N, D)
    x: jax.Array          # (N, D)
    logdetj: jax.Array    # (N,)
    logl: jax.Array       # (N,)
    logp: jax.Array       # (N,)
    logw: jax.Array       # (N,)

    blobs: jax.Array      # (N, B)
    iter: jax.Array       # () or (1,) int
    logz: jax.Array       # () float
    calls: jax.Array      # () float/int
    steps: jax.Array      # () float/int
    efficiency: jax.Array # () float
    ess: jax.Array        # () float
    accept: jax.Array     # () float
    beta: jax.Array       # () float


def init_particles_state_jax(
    max_steps: int,
    n_particles: int,
    n_dim: int,
    blob_dim: int = 0,
    dtype=jnp.float32,
) -> ParticlesState:
    T, N, D, B = max_steps, n_particles, n_dim, blob_dim

    zeros_TND = jnp.zeros((T, N, D), dtype=dtype)
    zeros_TN  = jnp.zeros((T, N), dtype=dtype)
    zeros_TNB = jnp.zeros((T, N, B), dtype=dtype)
    zeros_T   = jnp.zeros((T,), dtype=dtype)

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
    # index to avoid OOB without Python branching
    T = state.logl.shape[0]
    idx = jnp.minimum(state.t, jnp.array(T - 1, dtype=state.t.dtype))

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
    # pure pop step: decrement t (buffers are not changed)
    t_new = jnp.maximum(state.t - 1, jnp.array(0, dtype=state.t.dtype))
    return state._replace(t=t_new)


@jax.jit
def step_mask_jax(state: ParticlesState) -> jax.Array:
    T = state.logl.shape[0]
    return jnp.arange(T) < state.t  # (T,) bool


@jax.jit
def compute_logw_and_logz_jax(
    state: ParticlesState,
    beta_final: float | jax.Array = 1.0,
    normalize: bool | jax.Array = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    JAX version of compute_logw_and_logz.

    Returns:
      logw_flat: (T*N,) with -inf for unused steps
      logz_new: scalar
      mask_flat: (T*N,) True if logw_flat is valid
    """
    logl = state.logl  # (T, N)
    beta = state.beta  # (T,)
    logz = state.logz  # (T,)

    T, N = logl.shape
    n_steps = state.t  # CHECK cause dynamic int32 scalar

    mask_t = jnp.arange(T) < n_steps              # (T,)
    mask_i = mask_t                                # (T,)  sum over i uses active prefix
    mask_flat = jnp.repeat(mask_t, N)              # (T*N,)

    # b[i, j, n] = logl[j, n] * beta[i] - logz[i]
    b = logl[None, :, :] * beta[:, None, None] - logz[:, None, None]  # (T, T, N)

    # exclude inactive i from the log-mean-exp 
    b = jnp.where(mask_i[:, None, None], b, -jnp.inf)

    # log-mean-exp over i (axis=0). Use max(1, n_steps) to avoid log(0) in degenerate case.
    denom_steps = jnp.maximum(n_steps, jnp.array(1, dtype=n_steps.dtype)).astype(logl.dtype)
    B = jax.nn.logsumexp(b, axis=0) - jnp.log(denom_steps)            # (T, N)

    A = logl * jnp.asarray(beta_final, dtype=logl.dtype)              # (T, N)
    logw = A - B                                                      # (T, N)

    # exclude inactive j from final weights
    logw = jnp.where(mask_t[:, None], logw, -jnp.inf)                 # (T, N)
    logw_flat = logw.reshape(-1)                                      # (T*N,)

    denom_particles = denom_steps.astype(logl.dtype) * jnp.asarray(N, dtype=logl.dtype)
    logz_new = jax.nn.logsumexp(logw_flat) - jnp.log(denom_particles)

    normalize_arr = jnp.asarray(normalize)

    def _norm(lw):
        return lw - jax.nn.logsumexp(lw)

    logw_flat = lax.cond(normalize_arr, _norm, lambda lw: lw, logw_flat)

    return logw_flat, logz_new, mask_flat


@jax.jit
def compute_results_jax(
    state: ParticlesState,
    beta_final: float | jax.Array = 1.0,
    normalize: bool | jax.Array = True,
) -> Dict[str, jax.Array]:
    logw_flat, logz_new, mask_flat = compute_logw_and_logz_jax(state, beta_final, normalize)
    mask_t = step_mask_jax(state)

    # Pure result dictionary (a pytree)
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