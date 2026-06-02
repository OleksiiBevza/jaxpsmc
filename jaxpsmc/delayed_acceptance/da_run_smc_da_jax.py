from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Mapping, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from ..geometry.geometry_jax import Geometry, geometry_fit_jax
from ..particles_jax import ParticlesState, ParticlesStep, record_step_jax
from ..sampler_helper_jax import (
    mutate,
    not_termination_jax,
    resample_particles_jax,
    reweight_step_jax,
    reweight_step_persistent_jax,
)


Array = jax.Array


class SMCDACarry(NamedTuple):
    """
    Stores the state carried through the SMC-DA scan.

    This object is passed from one SMC iteration to the next.
    It contains the random key, particle history, current particles,
    fitted geometry, effective sample size information, and iteration count.

    Parameters:
    -----------
    key:
        JAX random key used by the next SMC step.
    state:
        full particle state and recorded particle history.
    current_particles:
        dictionary with the current active particles and their values.
        Expected keys include "u", "x", "logdetj", "logl", "logp",
        "blobs", "beta", "calls", and "proposal_scale".
    geom:
        fitted geometry used by the mutation kernel.
    n_effective:
        current effective sample size information used by reweighting.
    iteration:
        current outer SMC iteration counter.

    Returns:
    --------
    SMCDACarry:
        stores all values that must be carried between SMC-DA iterations.
    """
    key: Array
    state: ParticlesState
    current_particles: Dict[str, Array]
    geom: Geometry
    n_effective: Array
    iteration: Array


class SMCDAStepStats(NamedTuple):
    """
    Stores summary statistics from one SMC-DA step.

    These values are useful for diagnostics.
    They show whether the step was active, what beta was reached,
    and how expensive the mutation was.

    Parameters:
    -----------
    active:
        Boolean value showing whether an SMC step was actually run.
    beta:
        current annealing value after reweighting.
    logz:
        log normalizing constant estimate from the step.
    ess:
        effective sample size after reweighting.
    accept:
        mutation acceptance information.
    steps:
        number of mutation steps used.
    calls:
        number of likelihood calls used so far.

    Returns:
    --------
    SMCDAStepStats:
        stores compact diagnostics for one SMC-DA step.
    """
    active: Array
    beta: Array
    logz: Array
    ess: Array
    accept: Array
    steps: Array
    calls: Array


def _step_mutated_particles_jax(
    *,
    mutated: Dict[str, Array],
    iter_idx: Array,
    beta: Array,
    logz: Array,
    ess: Array,
) -> ParticlesStep:
    """
    Converts mutated particles into a recorded particle step.

    The mutation output is stored in the ParticlesStep format.
    The log weights are reset to zero because this function records
    the particles after resampling and mutation.

    Parameters:
    -----------
    mutated:
        dictionary containing mutated particles and diagnostics.
        Expected keys include "u", "x", "logdetj", "logl", "logp",
        "blobs", "calls", "steps", "efficiency", and "accept".
    iter_idx:
        iteration index to store in the particle record.
    beta:
        annealing value for this particle record.
    logz:
        log normalizing constant estimate for this step.
    ess:
        effective sample size for this step.

    Returns:
    --------
    ParticlesStep:
        particle record that can be appended to the SMC particle history.
    """
    logw = jnp.zeros_like(mutated["logl"])

    return ParticlesStep(
        u=mutated["u"],
        x=mutated["x"],
        logdetj=mutated["logdetj"],
        logl=mutated["logl"],
        logp=mutated["logp"],
        logw=logw,
        blobs=mutated["blobs"],
        iter=iter_idx.astype(jnp.int32),
        logz=logz,
        calls=mutated["calls"],
        steps=mutated["steps"],
        efficiency=mutated["efficiency"],
        ess=ess,
        accept=mutated["accept"],
        beta=beta,
    )


@partial(
    jax.jit,
    static_argnames=(
        "n_active",
        "n_outer_max_steps",
        "n_mutation_max_steps",
        "n_mutation_steps",
        "keep_max",
        "bins",
        "bisect_steps",
        "trim_ess",
        "sampling_mode",
        "mutation_fn",
        "loglike_single_fn",
        "logprior_fn",
    ),
)
def smc_da_step_jax(
    carry: SMCDACarry,
    *,
    n_total: Array,
    metric_id: Array,
    dynamic: Array,
    n_active: int,
    n_outer_max_steps: int,
    n_mutation_max_steps: int,
    n_mutation_steps: int,
    n_active_i32: Array,
    dynamic_ratio: Array,
    resample_code: Array,
    use_preconditioned_pcn: Array,
    keep_max: int,
    bins: int,
    bisect_steps: int,
    trim_ess: float,
    sampling_mode: str = "truncated_persistent",
    flow: Any,
    scaler_cfg: Mapping[str, Array],
    scaler_masks: Mapping[str, Array],
    mutation_fn: Callable[..., Tuple[Array, Dict[str, Array], Dict[str, Array]]] = mutate,
    loglike_single_fn: Callable[[Array], Tuple[Array, Array]],
    logprior_fn: Callable[[Array], Array],
) -> Tuple[SMCDACarry, SMCDAStepStats]:
    """
    Runs one outer SMC delayed-acceptance step.

    The function first checks whether the sampler should continue.
    If the sampler is still active, it performs one SMC update.

    The active update has four main parts.
    First, particles are reweighted and beta is updated.
    Second, geometry is fitted from the weighted particles.
    Third, particles are resampled.
    Fourth, particles are mutated with the chosen mutation kernel.

    If the sampler is no longer active, the carry is returned unchanged.
    In that case, the statistics are marked as inactive.

    Parameters:
    -----------
    carry:
        current SMC-DA carry.
        It contains the random key, particles, geometry, ESS state,
        and iteration counter.
    n_total:
        total number of particles or target total count used
        by the termination rule.
    metric_id:
        integer code selecting the termination or progress metric.
    dynamic:
        flag controlling whether the beta update is dynamic.
    n_active:
        number of active particles as a Python integer.
        This is static for JAX compilation.
    n_outer_max_steps:
        maximum number of outer SMC iterations.
    n_mutation_max_steps:
        maximum number of mutation steps allowed inside mutation_fn.
    n_mutation_steps:
        requested number of mutation steps inside mutation_fn.
    n_active_i32:
        number of active particles as a JAX int32 scalar.
    dynamic_ratio:
        ratio used by the dynamic reweighting rule.
    resample_code:
        integer code selecting the resampling method.
    use_preconditioned_pcn:
        flag selecting whether to use the preconditioned pCN kernel.
    keep_max:
        maximum number of particles or bins kept by reweighting.
        This is static for JAX compilation.
    bins:
        number of bins used by the beta-search or reweighting routine.
        This is static for JAX compilation.
    bisect_steps:
        number of bisection steps used by the beta-search routine.
        This is static for JAX compilation.
    trim_ess:
        ESS trimming value used inside reweighting.
        This is static for JAX compilation.
    flow:
        flow object used to transform particles from u-space to x-space.
        It must provide flow.bijection.transform_and_log_det.
    scaler_cfg:
        scaler configuration passed to the mutation function.
    scaler_masks:
        scaler masks passed to the mutation function.
    mutation_fn:
        mutation function used after resampling.
        It must return an updated key, mutated particles, and info.
    loglike_single_fn:
        function that evaluates the likelihood for one particle.
    logprior_fn:
        function that evaluates the prior for one particle.

    Returns:
    --------
    Tuple[SMCDACarry, SMCDAStepStats]:
        updated carry and diagnostic statistics for this SMC-DA step.
    """
    key, state, cur, geom, n_eff, it = carry
    dtype = state.logl.dtype

    active = (
        not_termination_jax(
            state,
            beta_current=cur["beta"],
            n_total=n_total,
            metric_code=metric_id,
            n_active=n_active_i32,
        )
        & (it < jnp.asarray(n_outer_max_steps, dtype=it.dtype))
    )

    def do_step(op: SMCDACarry) -> Tuple[SMCDACarry, SMCDAStepStats]:
        """
        Performs one active SMC update.

        The update reweights particles, fits geometry, resamples particles,
        mutates particles, records the result, and returns new diagnostics.

        Parameters:
        -----------
        op:
            current SMC-DA carry.

        Returns:
        --------
        Tuple[SMCDACarry, SMCDAStepStats]:
            updated carry and active step statistics.
        """
        key_c, state_c, cur_c, geom_c, n_eff_c, it_c = op

        sampling_mode_l = str(sampling_mode).lower()
        if sampling_mode_l == "persistent":
            reweight_fn = reweight_step_persistent_jax
        elif sampling_mode_l == "truncated_persistent":
            reweight_fn = reweight_step_jax
        else:
            raise ValueError(
                "sampling_mode must be 'persistent' or 'truncated_persistent'."
            )

        cur_rw, n_eff_new, rw_stats = reweight_fn(
            state_c,
            n_eff_c,
            metric_id,
            dynamic,
            n_active_i32,
            dynamic_ratio,
            bins=bins,
            bisect_steps=bisect_steps,
            keep_max=keep_max,
            trim_ess=trim_ess,
        )

        def u_to_theta(ui: Array) -> Tuple[Array, Array]:
            """
            Transforms one particle from u-space to theta-space.

            Parameters:
            -----------
            ui:
                one particle in u-space, shape (D,).

            Returns:
            --------
            Tuple[Array, Array]:
                transformed particle and log determinant value.
            """
            theta_i, logdet_i = flow.bijection.transform_and_log_det(ui, None)
            return theta_i, logdet_i

        theta_keep, _ = jax.vmap(u_to_theta, in_axes=0, out_axes=(0, 0))(
            cur_rw["u"]
        )

        geom_new, key_c, _ = geometry_fit_jax(
            geom_c,
            theta_keep,
            cur_rw["weights"],
            use_weights=jnp.asarray(True),
            key=key_c,
        )

        rs_out, _rs_status, key_c = resample_particles_jax(
            cur_rw,
            key=key_c,
            n_active=n_active,
            method_code=resample_code,
            reset_weights=True,
        )

        cur_for_mut = {
            "u": rs_out["u"],
            "x": rs_out["x"],
            "logdetj": rs_out["logdetj"],
            "logl": rs_out["logl"],
            "logp": rs_out["logp"],
            "logdetj_flow": jnp.zeros((n_active,), dtype=dtype),
            "blobs": rs_out["blobs"],
            "beta": cur_rw["beta"],
            "calls": cur_c["calls"],
            "proposal_scale": cur_c["proposal_scale"],
        }

        key_c, mutated, _info = mutation_fn(
            key_c,
            cur_for_mut,
            use_preconditioned_pcn=use_preconditioned_pcn,
            loglike_single_fn=loglike_single_fn,
            logprior_fn=logprior_fn,
            flow=flow,
            scaler_cfg=scaler_cfg,
            scaler_masks=scaler_masks,
            geom_mu=geom_new.t_mean,
            geom_cov=geom_new.t_cov,
            geom_nu=geom_new.t_nu,
            n_max=n_mutation_max_steps,
            n_steps=n_mutation_steps,
            condition=None,
        )

        step = _step_mutated_particles_jax(
            mutated=mutated,
            iter_idx=state_c.t,
            beta=cur_rw["beta"],
            logz=cur_rw["logz"],
            ess=rw_stats["ess"],
        )

        state_new = record_step_jax(state_c, step)

        #cur_next = {
        #    **mutated,
        #    "beta": cur_rw["beta"],
        #    "calls": mutated["calls"],
        #    "proposal_scale": mutated["proposal_scale"],
        #}

        cur_next = {
            "u": mutated["u"],
            "x": mutated["x"],
            "logdetj": mutated["logdetj"],
            "logl": mutated["logl"],
            "logp": mutated["logp"],
            "blobs": mutated["blobs"],
            "beta": cur_rw["beta"],
            "calls": mutated["calls"],
            "proposal_scale": mutated["proposal_scale"],
        }

        carry_next = SMCDACarry(
            key=key_c,
            state=state_new,
            current_particles=cur_next,
            geom=geom_new,
            n_effective=n_eff_new,
            iteration=it_c + jnp.asarray(1, dtype=it_c.dtype),
        )

        stats = SMCDAStepStats(
            active=jnp.asarray(True),
            beta=cur_rw["beta"],
            logz=cur_rw["logz"],
            ess=rw_stats["ess"],
            accept=mutated["accept"],
            steps=mutated["steps"],
            calls=mutated["calls"],
        )

        return carry_next, stats

    def skip_step(op: SMCDACarry) -> Tuple[SMCDACarry, SMCDAStepStats]:
        """
        Returns the carry unchanged when the sampler is inactive.

        This branch is used after termination.
        It keeps scan shapes stable while avoiding extra SMC work.

        Parameters:
        -----------
        op:
            current SMC-DA carry.

        Returns:
        --------
        Tuple[SMCDACarry, SMCDAStepStats]:
            unchanged carry and inactive step statistics.
        """
        _key, _state, cur_c, _geom, _n_eff, _it = op
        zero_f = jnp.asarray(0.0, dtype=dtype)
        zero_i = jnp.asarray(0, dtype=jnp.int32)

        stats = SMCDAStepStats(
            active=jnp.asarray(False),
            beta=cur_c["beta"],
            logz=zero_f,
            ess=zero_f,
            accept=zero_f,
            steps=zero_i,
            calls=cur_c["calls"],
        )

        return op, stats

    return lax.cond(active, do_step, skip_step, carry)


@partial(
    jax.jit,
    static_argnames=(
        "n_scan_steps",
        "n_active",
        "n_outer_max_steps",
        "n_mutation_max_steps",
        "n_mutation_steps",
        "keep_max",
        "bins",
        "bisect_steps",
        "sampling_mode",
        "trim_ess",
        "mutation_fn",
        "loglike_single_fn",
        "logprior_fn",
    ),
)
def run_smc_da_scan_jax(
    carry0: SMCDACarry,
    *,
    n_scan_steps: int,
    n_total: Array,
    metric_id: Array,
    dynamic: Array,
    n_active: int,
    n_outer_max_steps: int,
    n_mutation_max_steps: int,
    n_mutation_steps: int,
    n_active_i32: Array,
    dynamic_ratio: Array,
    resample_code: Array,
    use_preconditioned_pcn: Array,
    keep_max: int,
    bins: int,
    bisect_steps: int,
    trim_ess: float,
    sampling_mode: str = "truncated_persistent",    
    flow: Any,
    scaler_cfg: Mapping[str, Array],
    scaler_masks: Mapping[str, Array],
    mutation_fn: Callable[..., Tuple[Array, Dict[str, Array], Dict[str, Array]]] = mutate,
    loglike_single_fn: Callable[[Array], Tuple[Array, Array]],
    logprior_fn: Callable[[Array], Array],
) -> Tuple[SMCDACarry, SMCDAStepStats]:
    """
    Runs several SMC-DA steps with JAX scan.

    The scan repeatedly calls smc_da_step_jax.
    Each call updates the carry and returns step statistics.
    Once the termination rule is reached, later scan steps are skipped.

    Parameters:
    -----------
    carry0:
        initial SMC-DA carry.
    n_scan_steps:
        number of scan iterations to run.
        This is static for JAX compilation.
    n_total:
        total number of particles or target total count used
        by the termination rule.
    metric_id:
        integer code selecting the termination or progress metric.
    dynamic:
        flag controlling whether the beta update is dynamic.
    n_active:
        number of active particles as a Python integer.
        This is static for JAX compilation.
    n_outer_max_steps:
        maximum number of outer SMC iterations.
    n_mutation_max_steps:
        maximum number of mutation steps allowed inside mutation_fn.
    n_mutation_steps:
        requested number of mutation steps inside mutation_fn.
    n_active_i32:
        number of active particles as a JAX int32 scalar.
    dynamic_ratio:
        ratio used by the dynamic reweighting rule.
    resample_code:
        integer code selecting the resampling method.
    use_preconditioned_pcn:
        flag selecting whether to use the preconditioned pCN kernel.
    keep_max:
        maximum number of particles or bins kept by reweighting.
        This is static for JAX compilation.
    bins:
        number of bins used by the beta-search or reweighting routine.
        This is static for JAX compilation.
    bisect_steps:
        number of bisection steps used by the beta-search routine.
        This is static for JAX compilation.
    trim_ess:
        ESS trimming value used inside reweighting.
        This is static for JAX compilation.
    flow:
        flow object used to transform particles from u-space to x-space.
        It must provide flow.bijection.transform_and_log_det.
    scaler_cfg:
        scaler configuration passed to the mutation function.
    scaler_masks:
        scaler masks passed to the mutation function.
    mutation_fn:
        mutation function used after resampling.
        It must return an updated key, mutated particles, and info.
    loglike_single_fn:
        function that evaluates the likelihood for one particle.
    logprior_fn:
        function that evaluates the prior for one particle.

    Returns:
    --------
    Tuple[SMCDACarry, SMCDAStepStats]:
        final carry and statistics for all scan steps.
    """
    def scan_body(
        carry: SMCDACarry,
        _: Array,
    ) -> Tuple[SMCDACarry, SMCDAStepStats]:
        """
        Runs one scan iteration.

        The second input is unused.
        It only exists because lax.scan expects a sequence input.

        Parameters:
        -----------
        carry:
            current SMC-DA carry.
        _:
            unused scan index.

        Returns:
        --------
        Tuple[SMCDACarry, SMCDAStepStats]:
            updated carry and one-step statistics.
        """
        return smc_da_step_jax(
            carry,
            n_total=n_total,
            metric_id=metric_id,
            dynamic=dynamic,
            n_active=n_active,
            n_outer_max_steps=n_outer_max_steps,
            n_mutation_max_steps=n_mutation_max_steps,
            n_mutation_steps=n_mutation_steps,
            n_active_i32=n_active_i32,
            dynamic_ratio=dynamic_ratio,
            resample_code=resample_code,
            use_preconditioned_pcn=use_preconditioned_pcn,
            keep_max=keep_max,
            bins=bins,
            bisect_steps=bisect_steps,
            trim_ess=trim_ess,
            sampling_mode=sampling_mode,
            flow=flow,
            scaler_cfg=scaler_cfg,
            scaler_masks=scaler_masks,
            mutation_fn=mutation_fn,
            loglike_single_fn=loglike_single_fn,
            logprior_fn=logprior_fn,
        )

    return lax.scan(
        scan_body,
        carry0,
        xs=jnp.arange(n_scan_steps, dtype=jnp.int32),
    )