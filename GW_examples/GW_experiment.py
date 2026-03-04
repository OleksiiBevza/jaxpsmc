
##################################################################################
# 1. PACKAGES
##################################################################################

# ! pip install -e "C:\Users\oleks\jim"
import jimgw
print(f" Package location: {jimgw.__file__}")

from GW150914_IMRPhenomPV2 import *

# my sampler here
from sampler.sampler_jax import *
from sampler.sampler_helper_jax import *
from sampler.sampler_jax import *
from sampler.sampler_helper_jax import *
from sampler.prior_jax import *
jax.config.update("jax_enable_x64", True)


# diagnostics
import os
import json
import re
import sys
import argparse


import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

from jax.tree_util import tree_map
import time
import numpy as np
import matplotlib.pyplot as plt
import corner
import matplotlib as mpl
mpl.rcParams["axes.grid"] = False

import h5py











##################################################################################
# 2. ARGUMENTS
##################################################################################
### The argparse is used to store and process any user input we want to pass on
parser = argparse.ArgumentParser(description="Run experiment with specified parameters.")


parser.add_argument(
    "--outdir",
    type=str,
    required=True,
    help="The output directory, where things will be stored"
)


parser.add_argument(
    "--nr-of-samples",
    type=int,
    default=10000,
    help="Number of samples to be geerated"
)



# Everything below here are hyperparameters for sampler
parser.add_argument(
    "--prior-low",
    type=float,
    default=-20.0,
    help="Prior lower bound."
)
parser.add_argument(
    "--prior-high",
    type=float,
    default=20.0,
    help="Prior upper bound."
)

parser.add_argument("--n-total", type=int, default=4096)
parser.add_argument("--pc-n-steps", type=int, default=8)
parser.add_argument("--pc-n-max-steps", type=int, default=80)
parser.add_argument("--keep-max", type=int, default=4096)
parser.add_argument("--random-state", type=int, default=0)

parser.add_argument("--precondition", action="store_true", default=True)  # True by default
parser.add_argument("--no-precondition", action="store_false", dest="precondition")

parser.add_argument("--dynamic", action="store_true", default=True)
parser.add_argument("--no-dynamic", action="store_false", dest="dynamic")

parser.add_argument("--metric", type=str, default="ess", choices=["ess", "uss"])
parser.add_argument("--resample", type=str, default="mult", choices=["mult", "syst"])
parser.add_argument("--transform", type=str, default="probit", choices=["probit", "logit"])


# parser.add_argument("--use-identity-flow", action="store_true", default=True)

parser.add_argument("--n-effective", type=int, required=True)
parser.add_argument("--n-active", type=int, required=True)
parser.add_argument("--n-prior", type=int, required=True)


parser.add_argument("--proposal-scale", type=float, default=0.0)
parser.add_argument("--trim-ess", type=float, default=0.99)
parser.add_argument("--bins", type=int, default=1000)
parser.add_argument("--bisect-steps", type=int, default=1000)

























import time
import jax
import jax.numpy as jnp
from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
)
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import BaseTransientLikelihoodFD
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.transforms import BoundToUnbound
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914
# for the analysis
gps = 1126259462.4
start = gps - 2
end = gps + 2

# fetch 4096s of data to estimate the PSD (to be
# careful we should avoid the on-source segment,
# but we don't do this in this example)
psd_start = gps - 2048
psd_end = gps + 2048

# define frequency integration bounds for the likelihood
# we set fmax to 87.5% of the Nyquist frequency to avoid
# data corrupted by the GWOSC antialiasing filter
# (Note that Data.from_gwosc will pull data sampled at
# 4096 Hz by default)
fmin = 20.0
fmax = 1024

# initialize detectors
ifos = [get_H1(), get_L1()]

for ifo in ifos:
    # set analysis data
    data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(data)

    # set PSD (Welch estimate)
    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    # set an NFFT corresponding to the analysis segment duration
    psd_fftlength = data.duration * data.sampling_frequency
    ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))

###########################################
########## Set up waveform ################
###########################################

# initialize waveform
waveform = RippleIMRPhenomPv2(f_ref=20)

###########################################
########## Set up priors ##################
###########################################

prior = []

# Mass prior
M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

prior = prior + [Mc_prior, q_prior]

# Spin prior
s1_prior = UniformSpherePrior(parameter_names=["s1"])
s2_prior = UniformSpherePrior(parameter_names=["s2"])
iota_prior = SinePrior(parameter_names=["iota"])

prior = prior + [
    s1_prior,
    s2_prior,
    iota_prior,
]

# Extrinsic prior
dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = prior + [
    dL_prior,
    t_c_prior,
    phase_c_prior,
    psi_prior,
    ra_prior,
    dec_prior,
]

prior = CombinePrior(prior)

# Defining Transforms

sample_transforms = [
    DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(gps_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
    BoundToUnbound(
        name_mapping=(["M_c"], ["M_c_unbounded"]),
        original_lower_bound=M_c_min,
        original_upper_bound=M_c_max,
    ),
    BoundToUnbound(
        name_mapping=(["q"], ["q_unbounded"]),
        original_lower_bound=q_min,
        original_upper_bound=q_max,
    ),
    BoundToUnbound(
        name_mapping=(["s1_phi"], ["s1_phi_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s2_phi"], ["s2_phi_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["iota"], ["iota_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s1_theta"], ["s1_theta_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s2_theta"], ["s2_theta_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s1_mag"], ["s1_mag_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=1.0,
    ),
    BoundToUnbound(
        name_mapping=(["s2_mag"], ["s2_mag_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=1.0,
    ),
    BoundToUnbound(
        name_mapping=(["phase_det"], ["phase_det_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["psi"], ["psi_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["zenith"], ["zenith_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["azimuth"], ["azimuth_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]


likelihood = BaseTransientLikelihoodFD(
    ifos,
    waveform=waveform,
    trigger_time=gps,
    f_min=fmin,
    f_max=fmax,
)







# Import the necessary transform classes for type checking
from GW_prior import (
    ScaleTransform,
    OffsetTransform,
    CosineTransform,
    PowerLawTransform,
    RayleighTransform,
    CompositePrior,
    UniformDistribution,
    StandardNormalDistribution,
    LogisticDistribution,
    PowerLawPrior,
    GaussianPrior,
)







import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Iterable, Tuple
import numpy as np



class JimPriorAdapter(eqx.Module):
    """
    Thin adapter to use a Jim prior with the SMC sampler.

    - base_prior: raw Jim prior (CombinePrior / CompositePrior).
    - sample(key, n) -> (n, D) array.
    - logpdf(x) uses base_prior.log_prob on a dict.
    """

    base_prior: 'JimPriorBase'
    parameter_names: tuple[str, ...]
    params: jax.Array  # only used for dtype/shape in SMC, not for computation

    def __init__(self, base_prior: 'JimPriorBase'):
        # base_prior must be the ORIGINAL Jim prior (e.g. CombinePrior)
        self.base_prior = base_prior
        self.parameter_names = tuple(base_prior.parameter_names)
        dim = len(self.parameter_names)
        # dummy params just to satisfy sampler_jax expectations
        self.params = jnp.zeros((dim, 2), dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return len(self.parameter_names)

    # ------------------------------------------------------------------
    # Bounds: (D, 2) array, consistent with Jim prior support
    # ------------------------------------------------------------------
    def bounds(self) -> jax.Array:
        """Return per-parameter bounds in the SMC ordering."""
        bound_dict = self._get_bounds(self.base_prior)
        # Ensure order matches self.parameter_names
        out = []
        for name in self.parameter_names:
            low, high = bound_dict.get(name, (-jnp.inf, jnp.inf))
            out.append(jnp.array([low, high], dtype=jnp.float32))
        return jnp.stack(out)  # (D, 2)

    def _get_bounds(self, prior):
        """Recursively compute bounds for a prior, returning dict {name: (low, high)}."""
        # CompositePrior includes CombinePrior and SequentialTransformPrior
        if isinstance(prior, CompositePrior):
            # CombinePrior: multiple independent priors
            if hasattr(prior, 'base_prior') and isinstance(prior.base_prior, (list, tuple)):
                bound_dict = {}
                for child in prior.base_prior:
                    child_bounds = self._get_bounds(child)
                    bound_dict.update(child_bounds)
                return bound_dict
            # SequentialTransformPrior: base prior + transforms
            elif hasattr(prior, 'transforms') and hasattr(prior, 'base_prior'):
                # base_prior is a list of one element
                base_bounds = self._get_bounds(prior.base_prior[0])
                # Apply transforms in forward order
                for transform in prior.transforms:
                    base_bounds = self._apply_transform_to_bounds(transform, base_bounds)
                return base_bounds
            else:
                # Fallback (should not happen): trace leaf priors
                leaf_priors = prior.trace_prior_parent([]) if hasattr(prior, 'trace_prior_parent') else [prior]
                bound_dict = {}
                for p in leaf_priors:
                    low, high = self._bounds_for_leaf(p)
                    for name in p.parameter_names:
                        bound_dict[name] = (low, high)
                return bound_dict
        else:
            # Single leaf prior (e.g., UniformDistribution, StandardNormalDistribution)
            low, high = self._bounds_for_leaf(prior)
            return {name: (low, high) for name in prior.parameter_names}

    def _apply_transform_to_bounds(self, transform, bounds_dict):
        """Apply a single transform to an interval dictionary."""
        input_names, output_names = transform.name_mapping
        new_bounds = {}
        for in_name, out_name in zip(input_names, output_names):
            if in_name in bounds_dict:
                low_in, high_in = bounds_dict[in_name]

                # Handle known transform types
                if isinstance(transform, ScaleTransform):
                    scale = transform.scale
                    low_out = low_in * scale
                    high_out = high_in * scale
                elif isinstance(transform, OffsetTransform):
                    offset = transform.offset
                    low_out = low_in + offset
                    high_out = high_in + offset
                elif isinstance(transform, CosineTransform):
                    # input: cos(theta) in [-1, 1]; output: theta in [0, pi]
                    # The base prior should have given [-1,1].
                    # arccos is decreasing, so we swap after mapping.
                    low_out = jnp.arccos(high_in)
                    high_out = jnp.arccos(low_in)
                    # Ensure low_out <= high_out (swap if necessary)
                    low_out, high_out = jnp.minimum(low_out, high_out), jnp.maximum(low_out, high_out)
                elif isinstance(transform, PowerLawTransform):
                    # input: u in [0,1]; output: x in [xmin, xmax]
                    low_out = transform.xmin
                    high_out = transform.xmax
                elif isinstance(transform, RayleighTransform):
                    # input: u in [0,1]; output: x in [0, inf)
                    low_out = 0.0
                    high_out = jnp.inf
                else:
                    # Unsupported transform → assume unbounded
                    low_out, high_out = -jnp.inf, jnp.inf

                new_bounds[out_name] = (low_out, high_out)
            else:
                # Input name not found → output is unbounded
                new_bounds[out_name] = (-jnp.inf, jnp.inf)
        return new_bounds

    @staticmethod
    def _bounds_for_leaf(p):
        """Return (low, high) for a leaf distribution."""
        # Prefer explicit xmin/xmax if present (BoundedMixin, UniformPrior, PowerLawPrior, etc.)
        if hasattr(p, "xmin") and hasattr(p, "xmax"):
            return float(p.xmin), float(p.xmax)

        # Distribution-specific defaults
        if isinstance(p, UniformDistribution):
            return 0.0, 1.0
        if isinstance(p, (StandardNormalDistribution, GaussianPrior, LogisticDistribution)):
            return -jnp.inf, jnp.inf
        if isinstance(p, PowerLawPrior):
            return float(p.xmin), float(p.xmax)

        # Fallback: unbounded
        return -jnp.inf, jnp.inf

    # ------------------------------------------------------------------
    # Sampling: (n, D) interface on top of Jim dict samples
    # ------------------------------------------------------------------
    def sample(self, key: jax.Array, n: int) -> jax.Array:
        """
        Draw n samples from the Jim prior and return as (n, D) array
        in the order of `self.parameter_names`.
        """
        samples_dict = self.base_prior.sample(key, n)  # dict[name] -> (n,)
        cols = [samples_dict[name] for name in self.parameter_names]
        return jnp.stack(cols, axis=1)  # (n, D)

    def sample1(self, key: jax.Array) -> jax.Array:
        """Single sample: (D,)"""
        return self.sample(key, 1)[0]

    # ------------------------------------------------------------------
    # Log density: use Jim's log_prob, per-sample, then vmap
    # ------------------------------------------------------------------
    def logpdf(self, x: jax.Array) -> jax.Array:
        """
        x: shape (n, D) or (D,)
        returns: shape (n,)
        """
        x = jnp.atleast_2d(x)

        def _logprob_one(xi: jax.Array) -> jax.Array:
            # xi: shape (D,)
            z = {name: xi[i] for i, name in enumerate(self.parameter_names)}
            return self.base_prior.log_prob(z)

        return jax.vmap(_logprob_one)(x)

    def logpdf1(self, x: jax.Array) -> jax.Array:
        """
        Single-point log-density: x shape (D,) -> scalar.
        """
        x = jnp.asarray(x)
        assert x.ndim == 1 and x.shape[0] == self.dim
        return self.logpdf(x)[0]


















###################################################
# OUTDIR
###################################################
def next_run_dir(root: str, prefix: str = "run") -> str:
    os.makedirs(root, exist_ok=True)
    k = 0
    while True:
        outdir = os.path.join(root, f"{prefix}_{k:03d}")
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=False)
            return outdir
        k += 1


def _block_tree(x):
    def _b(a):
        return a.block_until_ready() if hasattr(a, "block_until_ready") else a
    return tree_map(_b, x)


###################################################
# RUNNER
###################################################
def run_event_and_save_posteriors(
    *,
    event_name: str,
    prior_u,
    loglike_x,
    D: int,
    names: list[str],
    ranges,
    periodic_idx=None,
    args,
):
    
    
    # read sampler params 
    n_effective = int(args.n_effective)
    n_active    = int(args.n_active)
    n_prior_in  = int(args.n_prior)

    # n_prior = multiple of n_active for this sampler
    n_prior = int(np.ceil(n_prior_in / n_active) * n_active)

    n_total     = int(args.n_total)
    n_steps     = int(args.pc_n_steps)
    n_max_steps = int(args.pc_n_max_steps)
    keep_max    = int(args.keep_max)

    precond   = bool(args.precondition)
    dynamic   = bool(args.dynamic)
    metric    = str(args.metric)
    resample  = str(args.resample)
    transform = str(args.transform)

    proposal_scale = float(args.proposal_scale)
    trim_ess       = float(args.trim_ess)
    bins           = int(args.bins)
    bisect_steps   = int(args.bisect_steps)

    seed     = int(args.random_state)
    n_keep   = int(args.nr_of_samples)   
    out_root = str(args.outdir)          

    # define sampler
    cfg = SamplerConfigJAX(
        n_dim=D,
        n_active=n_active,
        n_effective=n_effective,
        n_prior=n_prior,
        n_total=n_total,
        n_steps=n_steps,
        n_max_steps=n_max_steps,
        proposal_scale=proposal_scale,
        keep_max=keep_max,
        trim_ess=trim_ess,
        bins=bins,
        bisect_steps=bisect_steps,
        preconditioned=precond,
        dynamic=dynamic,
        metric=metric,
        resample=resample,
        transform=transform,
        periodic=periodic_idx,
        reflective=None,
        blob_dim=0,
    )



    # run sampler
    t0 = time.time()


    # ----------------------------------------------------------
    # Build SMC prior from the ORIGINAL Jim prior (prior_u)
    # ----------------------------------------------------------
    prior_smc = JimPriorAdapter(prior_u)   # adapter around raw CombinePrior

    def gw_loglike_single(x: jax.Array) -> jax.Array:
        # x: shape (D,)
        x = jnp.atleast_1d(x)
        assert x.ndim == 1 and x.shape[0] == D

        # 1) vector -> dict in PRIOR parameter space (M_c, q, s1_mag, ...)
        params = {name: x[i] for i, name in enumerate(names)}

        # 2) apply the *original* likelihood transforms to get eta, s1_x, ...
        for t in likelihood_transforms:
            params = t.forward(params)

        # 3) call the unchanged GW likelihood
        return loglike_x.evaluate(params, {})

    likelihood = gw_loglike_single

    sampler = SamplerJAX(
        prior_smc,
        likelihood,
        cfg,
        flow=IdentityFlowJAX(D),
    )

    out = sampler.run(jax.random.PRNGKey(seed))
    out = _block_tree(out)
    print(f"[{event_name}] sampler.run: {(time.time()-t0)/60:.2f} min")

    # draw posterior samples
    key_post = jax.random.PRNGKey(seed + 1)
    resample_method = jnp.int32(0 if resample == "mult" else 1)

    t1 = time.time()
    post = posterior_jax(
        out.state,
        key=key_post,
        do_resample=True,
        resample_method=resample_method,
        trim_importance_weights=True,
        ess_trim=jnp.asarray(cfg.trim_ess, dtype=jnp.float64),
        bins_trim=int(cfg.bins),
        beta_final=jnp.asarray(1.0, dtype=jnp.float64),
    )
    post = _block_tree(post)
    print(f"[{event_name}] posterior_jax: {(time.time()-t1)/60:.2f} min")



    # 
    theta = np.asarray(post.samples_resampled[:n_keep])  

    logZ = float(np.asarray(out.logz))
    logZerr = float(np.asarray(out.logz_err))

    print("Sampling complete!")
    print("n_prior (adjusted) =", n_prior, "(input was", n_prior_in, ")")
    print("samples.shape =", theta.shape)
    print("logZ =", logZ, "logZerr =", logZerr)

    # save
    outdir = next_run_dir(os.path.join(out_root, event_name))

    # ------------------------------------------------------------------
    # SAVE *YOUR* POSTERIOR IN ONE HDF5 FILE (like GW_15params.py)
    # ------------------------------------------------------------------
    h5_path = os.path.join(outdir, "posterior.hdf5")

    meta = {
        "event": event_name,
        "parameter_names": names,
        "n_samples": int(theta.shape[0]),
        "logz": float(logZ),
        "logz_err": float(logZerr),
        "seed": int(seed),
        "note": "Posterior draws from sampler",
    }

    with h5py.File(h5_path, "w") as f_h5:
        # samples: shape (n_samples, D)
        f_h5.create_dataset("samples", data=theta)

        # parameter names as fixed-length ASCII strings
        f_h5.create_dataset("names", data=np.asarray(names, dtype="S"))

        # metadata as HDF5 attributes
        for k, v in meta.items():
            # HDF5 attrs can't store Python lists nicely -> store names list as a single string
            if k == "parameter_names":
                f_h5.attrs[k] = ",".join(v)
            else:
                f_h5.attrs[k] = v

    print(f"[{event_name}] wrote posterior HDF5: {h5_path}")








    # --- 1) TRUE posterior HDF5 path (you said it's the same) ---
    TRUE_FILE = "/home/obevza/jaxpsmc/GW_examples/GW150914_095045_data0_1126259462-391_analysis_H1L1_result.hdf5"

    # --- 2) mapping from YOUR parameter names -> TRUE posterior dataset names ---
    name_map = {
        "M_c":      "chirp_mass",
        "q":        "mass_ratio",
        "s1_mag":   "a_1",
        "s1_theta": "tilt_1",
        "s1_phi":   "phi_1",
        "s2_mag":   "a_2",
        "s2_theta": "tilt_2",
        "s2_phi":   "phi_2",
        "iota":     "iota",
        "d_L":      "luminosity_distance",
        "t_c":      "geocent_time",
        "phase_c":  "phase",
        "psi":      "psi",
        "ra":       "ra",
        "dec":      "dec",
    }

    # IMPORTANT: in the true file, geocent_time is absolute GPS.
    # In your sampler, t_c is usually stored as an offset around gps_ref.
    gps_ref = 1126259462.4  # GW150914 trigger time used in your scripts

    def load_true_samples(true_file: str, names: list[str]) -> np.ndarray:
        with h5py.File(true_file, "r") as f_true:
            post_true = f_true["posterior"]
            cols = []
            for nm in names:
                true_nm = name_map[nm]  # will KeyError if nm not in map (good: forces correctness)
                arr = post_true[true_nm][:]

                # Convert absolute geocent_time -> offset t_c (seconds)
                if nm == "t_c":
                    arr = arr - gps_ref

                cols.append(arr)

        return np.column_stack(cols)

    samples_true = load_true_samples(TRUE_FILE, names)
    samples_ours = np.asarray(theta)

    # Optional: nicer LaTeX labels (must match your `names` order!)
    labels_latex = [
        r"$\mathcal{M}_c\ [M_\odot]$",
        r"$q$",
        r"$s_{1,\mathrm{mag}}$",
        r"$\theta_1$",
        r"$\phi_1$",
        r"$s_{2,\mathrm{mag}}$",
        r"$\theta_2$",
        r"$\phi_2$",
        r"$\iota$",
        r"$d_L\ \mathrm{[Mpc]}$",
        r"$t_c$",
        r"$\phi_c$",
        r"$\psi$",
        r"$\alpha$",
        r"$\delta$",
    ]

    # --- 3) Build overlay: TRUE first (sets axis limits), then OURS on same fig ---
    fig = plt.figure(figsize=(16, 16))

    fig = corner.corner(
        samples_true,
        fig=fig,
        labels=labels_latex if len(labels_latex) == len(names) else names,
        show_titles=True,
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
        bins=30,
        color="red",
        hist_kwargs={"density": True},
    )

    corner.corner(
        samples_ours,
        fig=fig,
        plot_datapoints=False,
        plot_density=True,
        fill_contours=False,
        bins=30,
        color="blue",
        hist_kwargs={"density": True},
    )

    for ax in fig.get_axes():
        ax.grid(False)

    save_path = os.path.join(outdir, "corner_true_vs_mine.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved overlay corner:", save_path)




    print(f"[{event_name}] saved {theta.shape[0]} samples to: {outdir}")
    return outdir, theta




















import sys
import argparse


# prior is your original CombinePrior (from prior_jim.py)
prior_u = prior                      # raw GW prior, unchanged
prior_smc = JimPriorAdapter(prior_u) # thin wrapper for SMC

bounds = np.asarray(prior_smc.bounds())       # (D, 2)
ranges = [tuple(map(float, b)) for b in bounds]



sys.argv = [
    "notebook",
    "--outdir", "/home/obevza/jaxpsmc/GW_examples",   
    "--nr-of-samples", "10000",        

    "--n-effective", "6500",
    "--n-active", "6500",
    "--n-prior", "149500",

    "--n-total", "11000",
    "--pc-n-steps", "450",
    "--pc-n-max-steps", "500",
    "--keep-max", "20000",
    "--random-state", "0",

    "--metric", "ess",
    "--resample", "mult",
    "--transform", "probit",

    "--proposal-scale", "0.0",
    "--trim-ess", "0.99",
    "--bins", "1000",
    "--bisect-steps", "1000",
]

args = parser.parse_args()









outdir, theta = run_event_and_save_posteriors(
    event_name="GW150914",
    prior_u=prior_u,                 # raw CombinePrior
    loglike_x=likelihood,            # original GW likelihood
    D=len(prior_u.parameter_names),
    names=list(prior_u.parameter_names),
    ranges=ranges,
    periodic_idx=None,
    args=args,
)



