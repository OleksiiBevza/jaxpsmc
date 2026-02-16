from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax



# my things 
from bisect_jax import *
from geometry_jax import *
from input_validation_jax import *
from particles_jax import *
from pcn_jax import *
from prior_jax import *
from sampler_helper_jax import *
from scaler_jax import *
from student_jax import *
from tools_jax import *













Array = jax.Array


# ---------------------------------------------------------------------------
# JUST EMPTY NF (all data go through it without training)
# will be substituted with real guy
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IdentityBijectionJAX:
    """bijection with zero log-det, matching FlowJAX interface"""

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()

    def transform_and_log_det(self, u: Array, condition: Optional[Array] = None) -> Tuple[Array, Array]:
        u = jnp.asarray(u)
        return u, jnp.zeros(u.shape[:-1], dtype=u.dtype)

    def inverse_and_log_det(self, theta: Array, condition: Optional[Array] = None) -> Tuple[Array, Array]:
        theta = jnp.asarray(theta)
        return theta, jnp.zeros(theta.shape[:-1], dtype=theta.dtype)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IdentityFlowJAX:
    """sth like flow object providing bijection used by preconditioned_pcn_jax."""

    dim: int

    def tree_flatten(self):
        return (), (self.dim,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (dim,) = aux
        return cls(dim=dim)

    @property
    def bijection(self) -> IdentityBijectionJAX:
        return IdentityBijectionJAX()

    # 
    def fit(self, *args, **kwargs):
        return self

    def sample(self, key: Array, n: int, condition: Optional[Array] = None) -> Array:
        return jax.random.normal(key, (n, self.dim))


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def _metric_code(metric: str) -> jnp.int32:
    # host mapping. it is not in jit REMEMBR IT 
    metric_l = str(metric).lower()
    if metric_l == "ess":
        return METRIC_ESS
    if metric_l == "uss":
        return METRIC_USS
    raise ValueError("metric must be 'ess' or 'uss'.")


def _resample_code(resample: str) -> jnp.int32:
    res_l = str(resample).lower()
    if res_l == "mult":
        return jnp.int32(0)
    if res_l == "syst":
        return jnp.int32(1)
    raise ValueError("resample must be 'mult' or 'syst'.")


@dataclass(frozen=True)
class SamplerConfigJAX:
    # dimensions
    n_dim: int
    n_effective: int = 512
    n_active: int = 256
    n_prior: int = 512

    # SMC termination
    n_total: int = 4096

    # MCMC kernel
    n_steps: int = 8
    n_max_steps: int = 80
    proposal_scale: float = 0.0  # if 0 -> set to 2.38/sqrt(D)

    # reweight / trim
    keep_max: int = 4096
    trim_ess: float = 0.99
    bins: int = 1000
    bisect_steps: int = 32

    # ess or  uss and syst or mult
    preconditioned: bool = True
    dynamic: bool = True
    metric: str = "ess"         # "ess" or "uss" 
    resample: str = "mult"      # "mult" or "syst" 

    # scaler
    transform: str = "probit"   # "probit" or "logit" (validate inside init_bounds_config_jax)
    periodic: Optional[jnp.ndarray] = None
    reflective: Optional[jnp.ndarray] = None

    # initiate blobs
    blob_dim: int = 0

    # evidence
    enable_flow_evidence: bool = False

    def __post_init__(self):
        if self.n_active <= 0 or self.n_effective <= 0 or self.n_dim <= 0:
            raise ValueError("n_dim, n_active, n_effective must be positive.")
        if self.n_prior % self.n_active != 0:
            raise ValueError("n_prior must be a multiple of n_active for warmup batching.")
        if self.keep_max <= 0:
            raise ValueError("keep_max must be positive.")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RunOutputJAX:
    """PyTree for results.

    JAX-jitted functions must return JAX PyTrees. this
    dataclass makes it a valid return type from `jax.jit`.
    NEED TO CHECK IT HERE SOMEHOW
    """

    state: ParticlesState
    logz: Array
    logz_err: Array

    def tree_flatten(self):
        # `ParticlesState` is a NamedTuple. NamedTuple is already a PyTree.
        return (self.state, self.logz, self.logz_err), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        state, logz, logz_err = children
        return cls(state=state, logz=logz, logz_err=logz_err)


class SamplerJAX:
    """convenience wrapper.

    Usage:
        sampler = SamplerJAX(prior, loglike_fn, cfg, flow=IdentityFlowJAX(cfg.n_dim))
        out = sampler.run(jax.random.PRNGKey(0))

    Notes:
      - `run` is the jitted pure function created by `make_run_fn`.
      - wrapper is not mutating internal state. it only keeps static objects.
    """

    def __init__(
        self,
        prior: Prior,
        loglike_single_fn: Callable[[Array], Any],
        cfg: SamplerConfigJAX,
        *,
        flow: Optional[Any] = None,
    ):
        self.prior = prior
        self.cfg = cfg
        self.flow = IdentityFlowJAX(cfg.n_dim) if flow is None else flow
        self._run_fn = make_run_fn(prior=prior, loglike_single_fn=loglike_single_fn, cfg=cfg, flow=self.flow)

    def run(self, key: Array, n_total: Optional[int] = None) -> RunOutputJAX:
        return self._run_fn(key, n_total=n_total)


# ---------------------------------------------------------------------------
# core JAX run loop
# ---------------------------------------------------------------------------


def _replace_inf_rows(
    key: Array,
    x: Array,
    u: Array,
    logdetj: Array,
    logp: Array,
    logl: Array,
    blobs: Array,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Replace rows with +/-inf likelihood by resampling finite rows. shapes are fixed"""
    n = x.shape[0]
    inf_mask = jnp.isinf(logl)
    finite_mask = ~inf_mask

    # if everything is inf, just keep it this way (no replacement needed). probabilities become uniform
    probs = finite_mask.astype(x.dtype)
    psum = jnp.sum(probs)
    probs = probs / jnp.where(psum > 0, psum, jnp.asarray(1.0, x.dtype))
    logits = jnp.where(probs > 0, jnp.log(probs), -jnp.inf)

    key, sub = jax.random.split(key)
    idx_rep = jax.random.categorical(sub, logits, shape=(n,), axis=0).astype(jnp.int32)
    idx_self = jnp.arange(n, dtype=jnp.int32)
    idx = jnp.where(inf_mask, idx_rep, idx_self)

    x2 = jnp.take(x, idx, axis=0)
    u2 = jnp.take(u, idx, axis=0)
    logdetj2 = jnp.take(logdetj, idx, axis=0)
    logp2 = jnp.take(logp, idx, axis=0)
    logl2 = jnp.take(logl, idx, axis=0)
    blobs2 = jnp.take(blobs, idx, axis=0)
    return key, x2, u2, logdetj2, logp2, logl2, blobs2


def _build_step_from_particles(
    *,
    u: Array,
    x: Array,
    logdetj: Array,
    logl: Array,
    logp: Array,
    blobs: Array,
    iter_idx: Array,
    beta: Array,
    logz: Array,
    calls: Array,
    steps: Array,
    efficiency: Array,
    ess: Array,
    accept: Array,
) -> ParticlesStep:
    # logw is not used by compute_logw_and_logz_jax; keep placeholder.
    logw = jnp.zeros_like(logl)
    return ParticlesStep(
        u=u,
        x=x,
        logdetj=logdetj,
        logl=logl,
        logp=logp,
        logw=logw,
        blobs=blobs,
        iter=iter_idx.astype(jnp.int32),
        logz=logz,
        calls=calls,
        steps=steps,
        efficiency=efficiency,
        ess=ess,
        accept=accept,
        beta=beta,
    )


def make_run_fn(
    *,
    prior: Prior,
    loglike_single_fn: Callable[[Array], Tuple[Array, Array]],
    cfg: SamplerConfigJAX,
    flow: Optional[Any] = None,
) -> Callable[[Array], RunOutputJAX]:
    """returns a single-argument, jitted run(key) function.

    Captures `prior`, `loglike_single_fn`, `cfg`, and `flow` in a closure so they are static for jit.
    """

    flow_obj = IdentityFlowJAX(cfg.n_dim) if flow is None else flow

    # wrap likelihood into a fixed-signature JAX function:
    #   loglike_wrapped(x: (D,)) -> (scalar, (B,))
    # supports two versions:
    #   - loglike_single_fn(x) -> scalar
    #   - loglike_single_fn(x) -> (scalar, blob)
    blob_dim = int(cfg.blob_dim)

    def loglike_wrapped(x: Array) -> Tuple[Array, Array]:
        out = loglike_single_fn(x)

        # CHECK IT HERE
        if isinstance(out, tuple) and len(out) == 2:
            ll, blob = out
        else:
            ll, blob = out, jnp.zeros((blob_dim,), dtype=jnp.result_type(out, jnp.float64))

        if blob_dim == 0:
            blob_vec = jnp.zeros((0,), dtype=jnp.result_type(ll, jnp.float64))
        else:
            blob_vec = jnp.asarray(blob).reshape((blob_dim,))
        return jnp.asarray(ll), blob_vec
    metric_code = _metric_code(cfg.metric)
    res_code = _resample_code(cfg.resample)

    periodic = jnp.asarray(cfg.periodic if cfg.periodic is not None else jnp.zeros((0,), dtype=jnp.int64))
    reflective = jnp.asarray(cfg.reflective if cfg.reflective is not None else jnp.zeros((0,), dtype=jnp.int64))

    bounds = prior.bounds()
    scaler_cfg0 = init_bounds_config_jax(
        cfg.n_dim,
        bounds=bounds,
        periodic=periodic,
        reflective=reflective,
        transform=cfg.transform,
        scale=True,
        diagonal=True,
    )
    scaler_masks = masks_jax(scaler_cfg0["low"], scaler_cfg0["high"])

    # compute dynamic ratio (host-side, but pure). 
    # WELL IT USES FLOAT64 AND EVERYWHERE IS FLOAT 32 ????
    w_ones = jnp.ones((cfg.n_effective,), dtype=jnp.float64)
    dyn_ratio = (unique_sample_size_jax(w_ones, k=cfg.n_active) / jnp.asarray(cfg.n_active, jnp.float64)).astype(jnp.float64)

    # proposal scale by default
    prop_scale = (
        (2.38 / (cfg.n_dim ** 0.5))
        if (cfg.proposal_scale is None or cfg.proposal_scale == 0.0)
        else float(cfg.proposal_scale)
    )

    max_steps_total = int((cfg.n_prior // cfg.n_active) + cfg.n_max_steps)

    @partial(
        jax.jit,
        static_argnames=(
            "n_active",
            "n_prior",
            "n_steps",
            "n_max_steps",
            "keep_max",
            "bins",
            "bisect_steps",
            "trim_ess",
            "blob_dim",
        ),
    )
    def _run(
        key: Array,
        n_total_dyn: Array,
        *,
        n_active: int,
        n_prior: int,
        n_steps: int,
        n_max_steps: int,
        keep_max: int,
        bins: int,
        bisect_steps: int,
        trim_ess: float,
        blob_dim: int,
    ) -> RunOutputJAX:
        key = jnp.asarray(key)
        dtype = jnp.result_type(prior.params, jnp.float64)

        # (i) sample prior for scaler fit 
        key, k_prior = jax.random.split(key)
        prior_samples = prior.sample(k_prior, n_prior).astype(dtype)  # (n_prior, D)

        # (ii) fit scaler (returns new cfg dict)
        scaler_cfg = fit_jax(prior_samples, scaler_cfg0, scaler_masks)

        # (iii) init particle history buffers
        state = init_particles_state_jax(
            max_steps=max_steps_total,
            n_particles=n_active,
            n_dim=cfg.n_dim,
            blob_dim=blob_dim,
            dtype=dtype,
        )

        #  scan over prior batches
        n_warm = n_prior // n_active
        geom0 = Geometry.init(cfg.n_dim, dtype=dtype)

        calls0 = jnp.asarray(0, dtype=jnp.int32)
        beta0 = jnp.asarray(0.0, dtype=dtype)
        logz0 = jnp.asarray(0.0, dtype=dtype)
        ess0 = jnp.asarray(cfg.n_effective, dtype=dtype)
        accept0 = jnp.asarray(1.0, dtype=dtype)
        steps0 = jnp.asarray(1, dtype=jnp.int32)
        eff0 = jnp.asarray(1.0, dtype=dtype)

        def warm_body(carry, i):
            key_c, state_c, calls_c = carry

            start = (i * n_active)
            x = lax.dynamic_slice_in_dim(prior_samples, start_index=start, slice_size=n_active, axis=0)

            # u = forward(x); logdetj from inverse(u)
            u = forward_jax(x, scaler_cfg, scaler_masks)
            _x_back, logdetj = inverse_jax(u, scaler_cfg, scaler_masks)

            logp = prior.logpdf(x)
            logl, blobs = jax.vmap(loglike_wrapped, in_axes=0, out_axes=(0, 0))(x)
            blobs = blobs.astype(dtype)

            calls_c = calls_c + jnp.asarray(n_active, dtype=calls_c.dtype)

            key_c, x, u, logdetj, logp, logl, blobs = _replace_inf_rows(
                key_c, x, u, logdetj, logp, logl, blobs
            )

            step = _build_step_from_particles(
                u=u,
                x=x,
                logdetj=logdetj,
                logl=logl,
                logp=logp,
                blobs=blobs,
                iter_idx=state_c.t,
                beta=beta0,
                logz=logz0,
                calls=calls_c.astype(dtype),
                steps=steps0.astype(dtype),
                efficiency=eff0,
                ess=ess0,
                accept=accept0,
            )
            state_c = record_step_jax(state_c, step)
            return (key_c, state_c, calls_c), None

        (key, state, calls_w), _ = lax.scan(
            warm_body,
            (key, state, calls0),
            xs=jnp.arange(n_warm, dtype=jnp.int32),
        )

        # variables for SMC loop
        n_eff_c = jnp.asarray(cfg.n_effective, dtype=jnp.int32)
        iter0 = jnp.asarray(0, dtype=jnp.int32)

        # <current particles> is only used for beta/calls/proposal_scale  in function  _mutate
        # actual particles for reweight/resample come from `state`.
        last_u = lax.dynamic_index_in_dim(state.u, state.t - 1, axis=0, keepdims=False)
        last_x = lax.dynamic_index_in_dim(state.x, state.t - 1, axis=0, keepdims=False)
        last_logdetj = lax.dynamic_index_in_dim(state.logdetj, state.t - 1, axis=0, keepdims=False)
        last_logl = lax.dynamic_index_in_dim(state.logl, state.t - 1, axis=0, keepdims=False)
        last_logp = lax.dynamic_index_in_dim(state.logp, state.t - 1, axis=0, keepdims=False)
        last_blobs = lax.dynamic_index_in_dim(state.blobs, state.t - 1, axis=0, keepdims=False)

        current_particles0: Dict[str, Array] = {
            "u": last_u,
            "x": last_x,
            "logdetj": last_logdetj,
            "logl": last_logl,
            "logp": last_logp,
            "logdetj_flow": jnp.zeros((n_active,), dtype=dtype),
            "blobs": last_blobs,
            "beta": beta0,
            "calls": calls_w,
            "proposal_scale": jnp.asarray(prop_scale, dtype=dtype),
            # IMPORTANT: 
            # PyTree structureis fixed across SMC while_loop.
            # _mutate() always returns scalar diagnostics, so 
            # include them in the initial carry as well.
            "efficiency": jnp.asarray(1.0, dtype=dtype),
            "accept": jnp.asarray(1.0, dtype=dtype),
            "steps": jnp.asarray(0, dtype=jnp.int32),
        }

        # Initialise geometry from warmup particles which are unweighted
        def _u2t_single(ui: Array) -> Tuple[Array, Array]:
            theta, logdet = flow_obj.bijection.transform_and_log_det(ui, None)
            return theta, logdet

        theta0, _ = jax.vmap(_u2t_single, in_axes=0, out_axes=(0, 0))(current_particles0["u"])
        w0 = jnp.full((n_active,), jnp.asarray(1.0, dtype) / jnp.asarray(n_active, dtype), dtype=dtype)
        geom, key, _ = geometry_fit_jax(geom0, theta0, w0, use_weights=jnp.asarray(False), key=key)

        # SMC while loop (JAAX version)
        # CHECK IT HERE VERY CAREFULLY 
        # OTHERWISE STH BAD MIGH THAPPEN 
        n_total = jnp.asarray(n_total_dyn, dtype=dtype)
        metric_id = jnp.asarray(metric_code, dtype=jnp.int32)
        n_active_i32 = jnp.asarray(n_active, dtype=jnp.int32)
        res_code_i32 = jnp.asarray(res_code, dtype=jnp.int32)
        dyn_ratio_arr = jnp.asarray(dyn_ratio, dtype=dtype)
        use_pcn = jnp.asarray(cfg.preconditioned)
        dynamic = jnp.asarray(cfg.dynamic)

        def cond_fn(carry):
            key_c, state_c, cur_c, geom_c, n_eff_c2, it = carry
            not_done = not_termination_jax(
                state_c,
                beta_current=cur_c["beta"],
                n_total=n_total,
                metric_code=metric_id,
                n_active=n_active_i32,
            )
            within_cap = it < jnp.asarray(n_max_steps, dtype=it.dtype)
            return not_done & within_cap

        def body_fn(carry):
            key_c, state_c, cur_c, geom_c, n_eff_c2, it = carry

            # Reweight (it is deterministic)
            cur_rw, n_eff_new, stats = reweight_step_jax(
                state_c,
                n_eff_c2,
                metric_id,
                dynamic,
                n_active_i32,
                dyn_ratio_arr,
                bins=bins,
                bisect_steps=bisect_steps,
                keep_max=keep_max,
                trim_ess=trim_ess,
            )

            # Geometry update on theta-space, weighted
            def _u2t_keep(ui: Array) -> Tuple[Array, Array]:
                th, ld = flow_obj.bijection.transform_and_log_det(ui, None)
                return th, ld

            theta_keep, _ = jax.vmap(_u2t_keep, in_axes=0, out_axes=(0, 0))(cur_rw["u"])
            geom_new, key_c, _ = geometry_fit_jax(
                geom_c,
                theta_keep,
                cur_rw["weights"],
                use_weights=jnp.asarray(True),
                key=key_c,
            )

            # resample to active set
            rs_out, _status, key_c = resample_particles_jax(
                cur_rw,
                key=key_c,
                n_active=n_active,
                method_code=res_code_i32,
                reset_weights=True,
            )

            # prepare dict expected by function  _mutate
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

            # mutate (PCN or noop via lax.cond inside _mutate)
            key_c, mutated, info = mutate(
                key_c,
                cur_for_mut,
                use_preconditioned_pcn=use_pcn,
                loglike_single_fn=loglike_wrapped,
                logprior_fn=prior.logpdf1,
                flow=flow_obj,
                scaler_cfg=scaler_cfg,
                scaler_masks=scaler_masks,
                geom_mu=geom_new.t_mean,
                geom_cov=geom_new.t_cov,
                geom_nu=geom_new.t_nu,
                n_max=n_max_steps,
                n_steps=n_steps,
                condition=None,
            )

            # record step
            step = _build_step_from_particles(
                u=mutated["u"],
                x=mutated["x"],
                logdetj=mutated["logdetj"],
                logl=mutated["logl"],
                logp=mutated["logp"],
                blobs=mutated["blobs"],
                iter_idx=state_c.t,
                beta=cur_rw["beta"],
                logz=cur_rw["logz"],
                calls=mutated["calls"].astype(dtype),
                steps=mutated["steps"].astype(dtype),
                efficiency=mutated["efficiency"],
                ess=stats["ess"],
                accept=mutated["accept"],
            )
            state_c = record_step_jax(state_c, step)

            # update carry
            cur_next = {
                **mutated,
                "beta": cur_rw["beta"],
                "calls": mutated["calls"],
                "proposal_scale": mutated["proposal_scale"],
            }

            return (key_c, state_c, cur_next, geom_new, n_eff_new, it + jnp.int32(1))

        key, state, cur, geom, n_eff_c, itf = lax.while_loop(
            cond_fn,
            body_fn,
            (key, state, current_particles0, geom, n_eff_c, iter0),
        )

        # evidence via flow is requested but not working. Use standard logZ from history instead!!!!!!!!
        _logw_flat, logz_final, _mask = compute_logw_and_logz_jax(
            state,
            beta_final=jnp.asarray(1.0, dtype=dtype),
            normalize=False,
        )
        logz_err = jnp.asarray(jnp.nan, dtype=dtype)

        return RunOutputJAX(state=state, logz=logz_final, logz_err=logz_err)

    def run(key: Array, n_total: Optional[int] = None) -> RunOutputJAX:
        n_total_use = cfg.n_total if n_total is None else int(n_total)
        return _run(
            key,
            n_total_dyn=jnp.asarray(n_total_use, dtype=jnp.float64),
            n_active=cfg.n_active,
            n_prior=cfg.n_prior,
            n_steps=cfg.n_steps,
            n_max_steps=cfg.n_max_steps,
            keep_max=cfg.keep_max,
            bins=cfg.bins,
            bisect_steps=cfg.bisect_steps,
            trim_ess=cfg.trim_ess,
            blob_dim=cfg.blob_dim,
        )

    return run