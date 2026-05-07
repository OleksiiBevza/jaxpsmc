from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

# helper modules used by the sampler
from .bisect_jax import *
from .geometry_jax import *
from .input_validation_jax import *
from .particles_jax import *
from .pcn_jax import *
from .prior_jax import *
from .sampler_helper_jax import *
from .scaler_jax import *
from .student_jax import *
from .tools_jax import *





Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IdentityBijectionJAX:
    """
    Stores an identity bijection for the sampler.

    A bijection is a reversible transformation.
    This class represents the simplest possible bijection:
    it returns the input unchanged.

    The log determinant is always zero.
    This is correct because the identity transformation does not stretch
    or shrink volume.

    This class is registered as a JAX pytree.
    It has no trainable parameters and no array fields.

    Parameters:
    -----------
    None:
        this class has no stored fields.

    Returns:
    --------
    IdentityBijectionJAX:
        identity bijection object compatible with the sampler flow interface.
    """

    def tree_flatten(self):
        """
        Converts the identity bijection into JAX pytree parts.

        Since this object stores no arrays, the children tuple is empty.
        The auxiliary data is also None.

        Parameters:
        -----------
        None:
            this method uses the current object.

        Returns:
        --------
        tuple:
            empty children tuple and no auxiliary data.
        """
        # This object has no array fields.
        return (), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """
        Rebuilds the identity bijection from pytree parts.

        JAX calls this when reconstructing the object after transformations.
        The inputs are unused because the identity bijection has no state.

        Parameters:
        -----------
        aux:
            auxiliary pytree data.
            It is unused here.
        children:
            pytree children.
            It is unused here.

        Returns:
        --------
        IdentityBijectionJAX:
            rebuilt identity bijection object.
        """
        return cls()

    def transform_and_log_det(self, u: Array, condition: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Applies the forward identity transformation.

        The input is returned unchanged.
        The log determinant is zero for every input row.
        The condition argument is accepted only to match the flow interface.

        Parameters:
        -----------
        u:
            input array in latent space, shape (..., D).
        condition:
            optional conditioning input.
            It is unused by the identity transformation.

        Returns:
        --------
        Tuple[Array, Array]:
            unchanged input array and zero log determinant, shape (...,).
        """
        # convert input to a JAX array
        u = jnp.asarray(u)
        return u, jnp.zeros(u.shape[:-1], dtype=u.dtype)

    def inverse_and_log_det(self, theta: Array, condition: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Applies the inverse identity transformation.

        The inverse of the identity map is also the identity map.
        The input is returned unchanged.
        The log determinant is zero for every input row.

        Parameters:
        -----------
        theta:
            input array in transformed space, shape (..., D).
        condition:
            optional conditioning input.
            It is unused by the identity transformation.

        Returns:
        --------
        Tuple[Array, Array]:
            unchanged input array and zero log determinant, shape (...,).
        """
        # convert input to a JAX array
        theta = jnp.asarray(theta)
        return theta, jnp.zeros(theta.shape[:-1], dtype=theta.dtype)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IdentityFlowJAX:
    """
    Stores a flow object that uses the identity bijection.

    A flow normally maps between a simple latent space and a transformed space.
    This identity flow performs no transformation.
    It is useful as a safe default when no learned flow is supplied.

    The object stores only the dimension of the space.
    It has no trainable parameters.

    Parameters:
    -----------
    dim:
        dimension of the latent space.

    Returns:
    --------
    IdentityFlowJAX:
        flow object with an identity bijection.
    """

    dim: int

    def tree_flatten(self):
        """
        Converts the identity flow into JAX pytree parts.

        The flow has no array children.
        The dimension is stored as auxiliary data because it is static.

        Parameters:
        -----------
        None:
            this method uses the current object.

        Returns:
        --------
        tuple:
            empty children tuple and auxiliary data containing dim.
        """
        # store dimension as auxiliary data
        return (), (self.dim,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """
        Rebuilds the identity flow from pytree parts.

        JAX uses this method after pytree transformations.
        The dimension is recovered from auxiliary data.

        Parameters:
        -----------
        aux:
            auxiliary data containing the dimension.
        children:
            pytree children.
            It is unused because this flow has no array fields.

        Returns:
        --------
        IdentityFlowJAX:
            rebuilt identity flow object.
        """
        # define stored dimension and rebuild object
        (dim,) = aux
        return cls(dim=dim)

    @property
    def bijection(self) -> IdentityBijectionJAX:
        """
        Returns the identity bijection used by this flow.

        Parameters:
        -----------
        None:
            this property uses the current flow object.

        Returns:
        --------
        IdentityBijectionJAX:
            identity bijection object.
        """
        # return a fresh identity bijection
        return IdentityBijectionJAX()

    def fit(self, *args, **kwargs):
        """
        Returns the same identity flow without fitting anything.

        A learned flow would update its parameters here.
        This identity flow has no parameters, so calling fit changes nothing.

        Parameters:
        -----------
        *args:
            unused positional arguments.
        **kwargs:
            unused keyword arguments.

        Returns:
        --------
        IdentityFlowJAX:
            unchanged identity flow object.
        """
        # flow has no trainable state, so return itself
        return self

    def sample(self, key: Array, n: int, condition: Optional[Array] = None) -> Array:
        """
        Draws standard normal samples in the latent space.

        Since the flow is identity, samples are drawn directly
        from a standard normal distribution with dimension dim.

        Parameters:
        -----------
        key:
            JAX random key.
        n:
            number of samples to draw.
        condition:
            optional conditioning input.
            It is unused by the identity flow.

        Returns:
        --------
        Array:
            standard normal samples, shape (n, dim).
        """
        # draw standard normal samples in the latent space
        return jax.random.normal(key, (n, self.dim))






##############################################################
# 1. CONFIGURATION HELPERS
##############################################################


def _metric_code(metric: str) -> jnp.int32:
    """
    Converts a metric name into an integer code.

    The sampler uses integer codes inside JAX control flow.
    This helper keeps string handling outside the jitted code.

    Parameters:
    -----------
    metric:
        metric name.
        Use "ess" for effective sample size.
        Use "uss" for unique sample size.

    Returns:
    --------
    jnp.int32:
        integer code for the selected metric.

    Raises:
    -------
    ValueError:
        raised when the metric name is not "ess" or "uss".
    """
    # normalize input string before checking it
    metric_l = str(metric).lower()

    # map metric name to internal code
    if metric_l == "ess":
        return METRIC_ESS
    if metric_l == "uss":
        return METRIC_USS
    raise ValueError("metric must be 'ess' or 'uss'.")


def _resample_code(resample: str) -> jnp.int32:
    """
    Converts a resampling method name into an integer code.

    The sampler uses integer codes inside JAX control flow.
    This helper keeps string handling outside the jitted code.

    Parameters:
    -----------
    resample:
        resampling method name.
        Use "mult" for multinomial resampling.
        Use "syst" for systematic resampling.

    Returns:
    --------
    jnp.int32:
        integer code for the selected resampling method.

    Raises:
    -------
    ValueError:
        raised when the resampling name is not "mult" or "syst".
    """
    # normalize input string before checking it
    res_l = str(resample).lower()

    # map method name to internal code
    if res_l == "mult":
        return jnp.int32(0)
    if res_l == "syst":
        return jnp.int32(1)
    raise ValueError("resample must be 'mult' or 'syst'.")


@dataclass(frozen=True)
class SamplerConfigJAX:
    """
    Stores all user-facing settings for the JAX sampler.

    This class controls the sampler size, stopping rules,
    mutation behavior, reweighting behavior, resampling method,
    scaler options, and delayed-acceptance options.

    The values are mostly static configuration values.
    Many of them are used as static arguments to jitted functions.
    Changing these values usually causes JAX to compile a new function.

    Parameters:
    -----------
    n_dim:
        dimension of the parameter space.
    n_effective:
        target effective sample size or unique sample size.
        The meaning depends on metric.
    n_active:
        number of active particles used in each live SMC batch.
    n_prior:
        number of prior samples used during warmup.
        It must be a multiple of n_active.
    n_total:
        stopping target used by the outer sampler.
    n_steps:
        stopping-rule value used inside the mutation kernel.
    n_max_steps:
        maximum number of outer SMC iterations.
        This value is also passed as the maximum number of pCN iterations
        in the current implementation.
    proposal_scale:
        initial proposal scale.
        If this is 0.0 or None, the sampler uses 2.38 / sqrt(D).
    delayed_acceptance:
        if True, use delayed-acceptance logic in the mutation kernel.
    da_c_const:
        conservative delayed-acceptance clipping constant.
        It must be positive.
    da_d_const:
        conservative delayed-acceptance exponent constant.
        It must be greater than 1.
    keep_max:
        maximum number of particles kept after trimming.
    trim_ess:
        ESS ratio used when trimming importance weights.
    bins:
        number of bins used by trimming.
    bisect_steps:
        number of bisection steps used when choosing beta.
    preconditioned:
        if True, use the preconditioned pCN mutation kernel.
        If False, mutation becomes a no-op.
    dynamic:
        if True, update the effective-size target dynamically.
    metric:
        metric used for beta selection and stopping.
        Must be "ess" or "uss".
    resample:
        resampling method.
        Must be "mult" or "syst".
    transform:
        scaler transform name.
        Expected values depend on scaler_jax.
    periodic:
        optional indices for periodic coordinates.
    reflective:
        optional indices for reflective coordinates.
    blob_dim:
        size of the extra blob output returned by the likelihood.
        Use 0 when the likelihood returns no extra values.
    enable_flow_evidence:
        flag reserved for flow-based evidence logic.
        It is stored here but not used in this file.

    Returns:
    --------
    SamplerConfigJAX:
        validated sampler configuration object.
    """
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
    proposal_scale: float = 0.0     # if 0 then set to 2.38/sqrt(D)

    # delayed acceptance
    delayed_acceptance: bool = False
    da_c_const: float = 0.01
    da_d_const: float = 2.0

    # reweight and trim
    keep_max: int = 4096
    trim_ess: float = 0.99
    bins: int = 1000
    bisect_steps: int = 32

    # resampling options: ess or uss and syst or mult
    preconditioned: bool = True
    dynamic: bool = True
    metric: str = "ess"             # "ess" or "uss" 
    resample: str = "mult"          # "mult" or "syst" 

    # scaler options
    transform: str = "probit"       # "probit" or "logit"
    periodic: Optional[jnp.ndarray] = None
    reflective: Optional[jnp.ndarray] = None

    # initiate blobs
    blob_dim: int = 0

    # evidence option
    enable_flow_evidence: bool = False


    def __post_init__(self):
        """
        Validates the most important configuration values.

        This method catches basic invalid settings before JAX compilation.
        It checks positive dimensions and particle counts.
        It also checks that warmup samples split evenly into batches.

        Parameters:
        -----------
        None:
            this method uses the current configuration object.

        Returns:
        --------
        None:
            the method returns nothing if validation passes.

        Raises:
        -------
        ValueError:
            raised when dimensions or particle counts are invalid.
        """
        # require positive dimensions and particle counts
        if self.n_active <= 0 or self.n_effective <= 0 or self.n_dim <= 0:
            raise ValueError("n_dim, n_active, n_effective must be positive.")
        # warmup samples must split evenly into active-particle batches
        if self.n_prior % self.n_active != 0:
            raise ValueError("n_prior must be a multiple of n_active for warmup batching.")
        # keep limit must also be positive.
        if self.keep_max <= 0:
            raise ValueError("keep_max must be positive.")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RunOutputJAX:
    """
    Stores the final output of one sampler run.

    The object contains the full particle history.
    It also stores the final log evidence estimate.
    The log evidence error is currently a placeholder value.

    This class is registered as a JAX pytree.
    That allows it to be returned from jitted code.

    Parameters:
    -----------
    state:
        full particle history recorded during warmup and SMC.
    logz:
        final log evidence estimate.
    logz_err:
        error estimate for logz.
        In this implementation it is set to NaN.

    Returns:
    --------
    RunOutputJAX:
        final sampler state and evidence values.
    """
    state: ParticlesState
    logz: Array
    logz_err: Array

    def tree_flatten(self):
        """
        Converts the output object into JAX pytree children.

        The particle state, log evidence, and error value are array children.
        No auxiliary data is needed.

        Parameters:
        -----------
        None:
            this method uses the current output object.

        Returns:
        --------
        tuple:
            children tuple and no auxiliary data.
        """
        # ParticlesState is already a pytree, so return it directly
        return (self.state, self.logz, self.logz_err), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Rebuilds the output object from JAX pytree children.

        JAX calls this after transformations involving the output object.

        Parameters:
        -----------
        aux_data:
            auxiliary pytree data.
            It is unused here.
        children:
            tuple containing state, logz, and logz_err.

        Returns:
        --------
        RunOutputJAX:
            rebuilt sampler output object.
        """
        # read stored fields and rebuild object
        state, logz, logz_err = children
        return cls(state=state, logz=logz, logz_err=logz_err)


class SamplerJAX:
    """
    Stores the objects needed to run the JAX SMC sampler.

    This is the user-facing sampler wrapper.
    It stores the prior, likelihood, configuration, and flow.
    It also builds the jitted run function during initialization.

    The actual sampling logic is built by make_run_fn.
    Calling run forwards to that compiled run function.

    Parameters:
    -----------
    prior:
        prior distribution object.
    loglike_single_fn:
        likelihood function for one point.
        It can return either a scalar log-likelihood or
        a pair of log-likelihood and blob output.
    cfg:
        sampler configuration.
    flow:
        optional flow object.
        If None, an identity flow is used.
    loglike_approx_single_fn:
        optional approximate likelihood function.
        Required when cfg.delayed_acceptance is True.

    Returns:
    --------
    SamplerJAX:
        sampler wrapper with a compiled run function.
    """

    def __init__(
        self,
        prior: Prior,
        loglike_single_fn: Callable[[Array], Any],
        cfg: SamplerConfigJAX,
        *,
        flow: Optional[Any] = None,
        loglike_approx_single_fn: Optional[Callable[[Array], Any]] = None,
    ):
        """
        Initializes the sampler wrapper.

        The constructor stores the prior, configuration, and flow.
        It also checks that an approximate likelihood is provided
        when delayed acceptance is requested.

        Finally, it creates the jitted run function.
        This means later calls to run can reuse the same compiled logic.

        Parameters:
        -----------
        prior:
            prior distribution object.
        loglike_single_fn:
            likelihood function for one point.
        cfg:
            sampler configuration.
        flow:
            optional flow object.
            If None, IdentityFlowJAX is used.
        loglike_approx_single_fn:
            optional approximate likelihood function.
            Required when cfg.delayed_acceptance is True.

        Returns:
        --------
        None:
            the initialized object stores the run function internally.
        """
        # store main objects used by sampler
        self.prior = prior
        self.cfg = cfg
        self.flow = IdentityFlowJAX(cfg.n_dim) if flow is None else flow

        if cfg.delayed_acceptance and loglike_approx_single_fn is None:
            raise ValueError(
                "cfg.delayed_acceptance=True requires loglike_approx_single_fn."
            )
        
        # build jitted run function once during initialization
        self._run_fn = make_run_fn(
            prior=prior, 
            loglike_single_fn=loglike_single_fn, 
            loglike_approx_single_fn=loglike_approx_single_fn,
            cfg=cfg, 
            flow=self.flow)

    def run(self, key: Array, n_total: Optional[int] = None) -> RunOutputJAX:
        """
        Runs the sampler from a random key.

        This method calls the compiled run function created in __init__.
        The optional n_total argument can override the stopping target
        stored in the configuration.

        Parameters:
        -----------
        key:
            JAX random key.
        n_total:
            optional stopping target.
            If None, cfg.n_total is used.

        Returns:
        --------
        RunOutputJAX:
            final particle history and evidence estimate.
        """ 
        # forward call to stored jitted run function
        return self._run_fn(key, n_total=n_total)







##############################################################
# 2. CORE JAX RUN LOOP 
##############################################################

def _replace_inf_rows(
    key: Array,
    x: Array,
    u: Array,
    logdetj: Array,
    logp: Array,
    logl: Array,
    blobs: Array,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:
    """
    Replaces rows with infinite likelihood values.

    The function identifies particles whose likelihood is infinite.
    These particles are replaced by copies of finite-likelihood particles.
    This prevents invalid rows from entering the particle history.

    The replacement is random and uses only rows with finite likelihood.
    If no finite rows exist, the sampling probabilities become invalid.

    Parameters:
    -----------
    key:
        JAX random key used to sample replacement rows.
    x:
        particle values in x-space, shape (N, D).
    u:
        particle values in u-space, shape (N, D).
    logdetj:
        scaler log-determinant values, shape (N,).
    logp:
        log-prior values, shape (N,).
    logl:
        log-likelihood values, shape (N,).
    blobs:
        extra likelihood outputs, shape (N, B).

    Returns:
    --------
    Tuple[Array, Array, Array, Array, Array, Array, Array]:
        updated key and particle arrays after invalid rows are replaced.
    """
    # define number of rows
    n = x.shape[0]
    # mark rows with infinite likelihood values
    inf_mask = jnp.isinf(logl)
    finite_mask = ~inf_mask

    # build sampling probabilities from finite rows only
    probs = finite_mask.astype(x.dtype)
    psum = jnp.sum(probs)
    probs = probs / jnp.where(psum > 0, psum, jnp.asarray(1.0, x.dtype))
    logits = jnp.where(probs > 0, jnp.log(probs), -jnp.inf)

    # draw replacement rows for invalid entries
    key, sub = jax.random.split(key)
    idx_rep = jax.random.categorical(sub, logits, shape=(n,), axis=0).astype(jnp.int32)
    idx_self = jnp.arange(n, dtype=jnp.int32)
    idx = jnp.where(inf_mask, idx_rep, idx_self)

    # gather updated arrays
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
    """
    Builds one particle-history record.

    The sampler stores particles as ParticlesStep objects.
    This helper takes raw arrays and scalar diagnostics,
    then packages them into the expected structure.

    The log-weight vector is set to zero here.
    Final importance weights are computed later from the full history.

    Parameters:
    -----------
    u:
        particle values in u-space, shape (N, D).
    x:
        particle values in x-space, shape (N, D).
    logdetj:
        scaler log-determinant values, shape (N,).
    logl:
        log-likelihood values, shape (N,).
    logp:
        log-prior values, shape (N,).
    blobs:
        extra likelihood outputs, shape (N, B).
    iter_idx:
        iteration index stored for this step.
    beta:
        annealing value for this step.
    logz:
        log evidence estimate stored for this step.
    calls:
        likelihood call count stored for this step.
    steps:
        number of mutation steps stored for this step.
    efficiency:
        mutation efficiency diagnostic.
    ess:
        effective sample size diagnostic.
    accept:
        acceptance diagnostic.

    Returns:
    --------
    ParticlesStep:
        one particle-history record.
    """
    # store a placeholder log-weight vector for the history record
    logw = jnp.zeros_like(logl)
    # buils ParticlesStep object
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
    loglike_approx_single_fn: Optional[Callable[[Array], Any]],
    cfg: SamplerConfigJAX,
    flow: Optional[Any] = None,
) -> Callable[[Array], RunOutputJAX]:
    """
    Builds the sampler run function.

    This function closes over the prior, likelihood, configuration,
    scaler setup, and flow object.
    It then returns a callable run function.

    The returned function samples prior warmup particles,
    fits the scaler, records warmup history, runs the outer SMC loop,
    and finally computes the log evidence estimate.

    Parameters:
    -----------
    prior:
        prior distribution object.
    loglike_single_fn:
        likelihood function for one x-space point.
        It can return either a scalar or a pair of scalar and blob output.
    loglike_approx_single_fn:
        optional approximate likelihood function.
        Used by delayed acceptance.
        If None, the full likelihood value is used as a fallback wrapper.
    cfg:
        sampler configuration.
    flow:
        optional flow object.
        If None, IdentityFlowJAX is used.

    Returns:
    --------
    Callable[[Array], RunOutputJAX]:
        run function that takes a random key and returns sampler output.
    """
    # useidentity flow when no flow object is given

    flow_obj = IdentityFlowJAX(cfg.n_dim) if flow is None else flow

    # read fixed blob size from config
    blob_dim = int(cfg.blob_dim)

    def loglike_wrapped(x: Array) -> Tuple[Array, Array]:
        """
        Converts the user likelihood output into a fixed format.

        The sampler expects every likelihood call to return two values:
        a scalar log-likelihood and a blob vector.
        This wrapper accepts either that pair or only a scalar likelihood.

        Parameters:
        -----------
        x:
            one input point in x-space, shape (D,).

        Returns:
        --------
        Tuple[Array, Array]:
            scalar log-likelihood and blob vector, shape (blob_dim,).
        """
        # call user-provided likelihood function
        out = loglike_single_fn(x)

        # accept either a scalar return or a pair with a blob
        if isinstance(out, tuple) and len(out) == 2:
            ll, blob = out
        else:
            ll, blob = out, jnp.zeros((blob_dim,), dtype=jnp.result_type(out, jnp.float64))

        # build a blob vector with configured size
        if blob_dim == 0:
            blob_vec = jnp.zeros((0,), dtype=jnp.result_type(ll, jnp.float64))
        else:
            blob_vec = jnp.asarray(blob).reshape((blob_dim,))
        return jnp.asarray(ll), blob_vec
    
    def loglike_approx_wrapped(x: Array) -> Array:
        """
        Converts the approximate likelihood output into a scalar array.

        If no approximate likelihood is provided, the wrapper uses
        the full likelihood value. This fallback is convenient,
        but it removes the computational benefit of delayed acceptance.

        Parameters:
        -----------
        x:
            one input point in x-space, shape (D,).

        Returns:
        --------
        Array:
            scalar approximate log-likelihood.
        """
        if loglike_approx_single_fn is None:
            return loglike_wrapped(x)[0]

        out = loglike_approx_single_fn(x)
        if isinstance(out, tuple) and len(out) == 2:
            ll, _ = out
        else:
            ll = out

        return jnp.asarray(ll)


    # convert string options into integer codes
    metric_code = _metric_code(cfg.metric)
    res_code = _resample_code(cfg.resample)

    # build periodic and reflective index arrays
    periodic = jnp.asarray(cfg.periodic if cfg.periodic is not None else jnp.zeros((0,), dtype=jnp.int64))
    reflective = jnp.asarray(cfg.reflective if cfg.reflective is not None else jnp.zeros((0,), dtype=jnp.int64))

    # build scaler configuration from prior bounds
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

    # precompute dynamic ratio used by reweight step
    w_ones = jnp.ones((cfg.n_effective,), dtype=jnp.float64)
    dyn_ratio = (unique_sample_size_jax(w_ones, k=cfg.n_active) / jnp.asarray(cfg.n_active, jnp.float64)).astype(jnp.float64)

    # choose initial proposal scale
    prop_scale = (
        (2.38 / (cfg.n_dim ** 0.5))
        if (cfg.proposal_scale is None or cfg.proposal_scale == 0.0)
        else float(cfg.proposal_scale)
    )

    # preallocate particle history slots for warmup and SMC
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
        """
        Runs the full jitted sampler.

        The run has four main phases.
        First, it samples warmup points from the prior.
        Second, it fits the scaler and records warmup batches.
        Third, it runs the outer SMC loop with reweighting,
        geometry fitting, resampling, and mutation.
        Fourth, it computes the final log evidence estimate.

        Parameters:
        -----------
        key:
            JAX random key.
        n_total_dyn:
            stopping target as a JAX scalar.
        n_active:
            number of active particles per batch.
            This is static for JAX compilation.
        n_prior:
            number of prior samples used in warmup.
            This is static for JAX compilation.
        n_steps:
            stopping-rule value passed to the mutation kernel.
            This is static for JAX compilation.
        n_max_steps:
            maximum number of outer SMC iterations.
            This is static for JAX compilation.
        keep_max:
            maximum number of kept particles after trimming.
            This is static for JAX compilation.
        bins:
            number of trimming bins.
            This is static for JAX compilation.
        bisect_steps:
            number of bisection steps for beta selection.
            This is static for JAX compilation.
        trim_ess:
            ESS ratio used by trimming.
            This is static for JAX compilation.
        blob_dim:
            size of likelihood blob output.
            This is static for JAX compilation.

        Returns:
        --------
        RunOutputJAX:
            final particle history and evidence estimate.
        """
        # convert key and choose dtype
        key = jnp.asarray(key)
        dtype = jnp.result_type(prior.params, jnp.float64)

        # (i) sample prior points used to fit the scaler
        key, k_prior = jax.random.split(key)
        prior_samples = prior.sample(k_prior, n_prior).astype(dtype)  # (n_prior, D)

        # (ii) fit scaler on the prior samples
        scaler_cfg = fit_jax(prior_samples, scaler_cfg0, scaler_masks)

        # (iii) create particle-history buffers
        state = init_particles_state_jax(
            max_steps=max_steps_total,
            n_particles=n_active,
            n_dim=cfg.n_dim,
            blob_dim=blob_dim,
            dtype=dtype,
        )

        #  compute number of warmup batches
        n_warm = n_prior // n_active
        geom0 = Geometry.init(cfg.n_dim, dtype=dtype)

        # build initial scalar values used during warmup
        calls0 = jnp.asarray(0, dtype=jnp.int32)
        beta0 = jnp.asarray(0.0, dtype=dtype)
        logz0 = jnp.asarray(0.0, dtype=dtype)
        ess0 = jnp.asarray(cfg.n_effective, dtype=dtype)
        accept0 = jnp.asarray(1.0, dtype=dtype)
        steps0 = jnp.asarray(1, dtype=jnp.int32)
        eff0 = jnp.asarray(1.0, dtype=dtype)

        def warm_body(carry, i):
            """
            Processes one warmup batch of prior samples.

            The batch is transformed into u-space.
            The prior and likelihood are evaluated.
            Infinite-likelihood rows are replaced.
            The resulting batch is recorded in the particle history.

            Parameters:
            -----------
            carry:
                tuple containing current key, particle state, and call count.
            i:
                warmup batch index.

            Returns:
            --------
            tuple:
                updated carry and no scan output.
            """
            # unpack current warmup state
            key_c, state_c, calls_c = carry

            # slice out one batch of prior samples
            start = (i * n_active)
            x = lax.dynamic_slice_in_dim(prior_samples, start_index=start, slice_size=n_active, axis=0)

            # map x into u-space and recompute log-determinant
            u = forward_jax(x, scaler_cfg, scaler_masks)
            _x_back, logdetj = inverse_jax(u, scaler_cfg, scaler_masks)

            # evaluate prior and likelihood values for batch
            logp = prior.logpdf(x)
            logl, blobs = jax.vmap(loglike_wrapped, in_axes=0, out_axes=(0, 0))(x)
            blobs = blobs.astype(dtype)

            # count likelihood calls from this batch
            calls_c = calls_c + jnp.asarray(n_active, dtype=calls_c.dtype)

            # replace rows with infinite likelihood values
            key_c, x, u, logdetj, logp, logl, blobs = _replace_inf_rows(
                key_c, x, u, logdetj, logp, logl, blobs
            )

            # build and record one history step
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

        # run warmup over all prior batches
        (key, state, calls_w), _ = lax.scan(
            warm_body,
            (key, state, calls0),
            xs=jnp.arange(n_warm, dtype=jnp.int32),
        )

        # initialize values used by outer SMC loop
        n_eff_c = jnp.asarray(cfg.n_effective, dtype=jnp.int32)
        iter0 = jnp.asarray(0, dtype=jnp.int32)

        # initiate the most recent warmup particles
        last_u = lax.dynamic_index_in_dim(state.u, state.t - 1, axis=0, keepdims=False)
        last_x = lax.dynamic_index_in_dim(state.x, state.t - 1, axis=0, keepdims=False)
        last_logdetj = lax.dynamic_index_in_dim(state.logdetj, state.t - 1, axis=0, keepdims=False)
        last_logl = lax.dynamic_index_in_dim(state.logl, state.t - 1, axis=0, keepdims=False)
        last_logp = lax.dynamic_index_in_dim(state.logp, state.t - 1, axis=0, keepdims=False)
        last_blobs = lax.dynamic_index_in_dim(state.blobs, state.t - 1, axis=0, keepdims=False)

        # build initial current-particle dictionary
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
            # PyTree structure is fixed across SMC while_loop
            # _mutate() always returns scalar diagnostics, so 
            # include them in the initial carry as well
            "efficiency": jnp.asarray(1.0, dtype=dtype),
            "accept": jnp.asarray(1.0, dtype=dtype),
            "steps": jnp.asarray(0, dtype=jnp.int32),
        }

        def _u2t_single(ui: Array) -> Tuple[Array, Array]:
            """
            Maps one current particle from u-space to theta-space.

            This helper is used to build the initial fitted geometry.
            It applies the flow bijection to one particle.

            Parameters:
            -----------
            ui:
                one particle in u-space, shape (D,).

            Returns:
            --------
            Tuple[Array, Array]:
                transformed particle in theta-space
                and flow log determinant.
            """
            # use flow bijection to one latent vector
            theta, logdet = flow_obj.bijection.transform_and_log_det(ui, None)
            return theta, logdet

        # build an initial geometry object from warmup particles
        theta0, _ = jax.vmap(_u2t_single, in_axes=0, out_axes=(0, 0))(current_particles0["u"])
        w0 = jnp.full((n_active,), jnp.asarray(1.0, dtype) / jnp.asarray(n_active, dtype), dtype=dtype)
        geom, key, _ = geometry_fit_jax(geom0, theta0, w0, use_weights=jnp.asarray(False), key=key)

        # convert loop settings to JAX arrays
        n_total = jnp.asarray(n_total_dyn, dtype=dtype)
        metric_id = jnp.asarray(metric_code, dtype=jnp.int32)
        n_active_i32 = jnp.asarray(n_active, dtype=jnp.int32)
        res_code_i32 = jnp.asarray(res_code, dtype=jnp.int32)
        dyn_ratio_arr = jnp.asarray(dyn_ratio, dtype=dtype)
        use_pcn = jnp.asarray(cfg.preconditioned)
        dynamic = jnp.asarray(cfg.dynamic)

        def cond_fn(carry):
            """
            Checks whether the outer SMC loop should continue.

            The loop continues while the sampler has not met the stopping rule
            and the maximum number of outer iterations has not been reached.

            Parameters:
            -----------
            carry:
                tuple containing key, state, current particles,
                geometry, effective-size target, and iteration count.

            Returns:
            --------
            Array:
                Boolean scalar.
                True means another outer SMC step should run.
            """
            # unpack values needed by the stop rule
            key_c, state_c, cur_c, geom_c, n_eff_c2, it = carry
            
            # continue while sampler has not met stop rule
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
            """
            Runs one outer SMC step.

            One outer step performs reweighting, geometry fitting,
            resampling, mutation, and history recording.
            The resulting particles become the current particles
            for the next outer iteration.

            Parameters:
            -----------
            carry:
                tuple containing key, state, current particles,
                geometry, effective-size target, and iteration count.

            Returns:
            --------
            tuple:
                updated loop carry for the next SMC iteration.
            """
            # unpack current loop state
            key_c, state_c, cur_c, geom_c, n_eff_c2, it = carry

            # reweight history and keep the highest weight particles
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

            def _u2t_keep(ui: Array) -> Tuple[Array, Array]:
                """
                Maps one kept particle from u-space to theta-space.

                The transformed particles are used to update
                the fitted geometry before mutation.

                Parameters:
                -----------
                ui:
                    one kept particle in u-space, shape (D,).

                Returns:
                --------
                Tuple[Array, Array]:
                    transformed particle and flow log determinant.
                """
                # apply flow bijection to one kept particle
                th, ld = flow_obj.bijection.transform_and_log_det(ui, None)
                return th, ld

            # update geometry using kept weighted particles            
            theta_keep, _ = jax.vmap(_u2t_keep, in_axes=0, out_axes=(0, 0))(cur_rw["u"])
            geom_new, key_c, _ = geometry_fit_jax(
                geom_c,
                theta_keep,
                cur_rw["weights"],
                use_weights=jnp.asarray(True),
                key=key_c,
            )

            # resample kept particles down to active set
            rs_out, _status, key_c = resample_particles_jax(
                cur_rw,
                key=key_c,
                n_active=n_active,
                method_code=res_code_i32,
                reset_weights=True,
            )

            # build input dictionary expected by mutation step
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

            # run mutation step
            key_c, mutated, info = mutate(
                key_c,
                cur_for_mut,
                use_preconditioned_pcn=use_pcn,
                loglike_single_fn=loglike_wrapped,
                loglike_approx_single_fn=loglike_approx_wrapped,
                logprior_fn=prior.logpdf1,
                flow=flow_obj,
                scaler_cfg=scaler_cfg,
                scaler_masks=scaler_masks,
                geom_mu=geom_new.t_mean,
                geom_cov=geom_new.t_cov,
                geom_nu=geom_new.t_nu,
                n_max=n_max_steps,
                n_steps=n_steps,
                use_delayed_acceptance=jnp.asarray(cfg.delayed_acceptance),
                da_c_const=jnp.asarray(cfg.da_c_const, dtype=dtype),
                da_d_const=jnp.asarray(cfg.da_d_const, dtype=dtype),
                condition=None,
            )

            # record new mutated particles into history
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

            # define next current particle dictionary
            cur_next = {
                **mutated,
                "beta": cur_rw["beta"],
                "calls": mutated["calls"],
                "proposal_scale": mutated["proposal_scale"],
            }

            return (key_c, state_c, cur_next, geom_new, n_eff_new, it + jnp.int32(1))

        # run outer SMC loop
        key, state, cur, geom, n_eff_c, itf = lax.while_loop(
            cond_fn,
            body_fn,
            (key, state, current_particles0, geom, n_eff_c, iter0),
        )

        # compute final log-evidence from stored history
        _logw_flat, logz_final, _mask = compute_logw_and_logz_jax(
            state,
            beta_final=jnp.asarray(1.0, dtype=dtype),
            normalize=False,
        )
        logz_err = jnp.asarray(jnp.nan, dtype=dtype)

        return RunOutputJAX(state=state, logz=logz_final, logz_err=logz_err)

    def run(key: Array, n_total: Optional[int] = None) -> RunOutputJAX:
        """
        Runs the compiled sampler with captured static settings.

        This wrapper chooses the stopping target.
        If n_total is not provided, it uses cfg.n_total.
        It then calls the jitted _run function.

        Parameters:
        -----------
        key:
            JAX random key.
        n_total:
            optional stopping target.
            If None, cfg.n_total is used.

        Returns:
        --------
        RunOutputJAX:
            final particle history and evidence estimate.
        """
        # use stopping or fall back to config value
        n_total_use = cfg.n_total if n_total is None else int(n_total)
        
        # run jitted core run function
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




