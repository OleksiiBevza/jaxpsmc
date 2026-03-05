##################################################################################
# 1. PACKAGES
##################################################################################

# my sampler here
from sampler.sampler_jax import *
from sampler.sampler_helper_jax import *
from sampler.sampler_jax import *
from sampler.sampler_helper_jax import *
from sampler.prior_jax import *
jax.config.update("jax_enable_x64", True)

# configurations
import jimgw
print(f" Package location: {jimgw.__file__}")
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

# diagnostics
import os
import json
import re
import sys
import argparse
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from jax.tree_util import tree_map
import numpy as np
import matplotlib.pyplot as plt
import corner
import matplotlib as mpl
mpl.rcParams["axes.grid"] = False
import h5py







##############################################################################################
# 2. ARGUMENTS
##############################################################################################
### The argparse is used to store and process any user input we want to pass on
parser = argparse.ArgumentParser(description="Run experiment with specified parameters.")
parser.add_argument("--outdir", type=str, required=True, help="The output directory")
parser.add_argument("--nr-of-samples", type=int, default=10000, help="Number of samples to generate")

# Everything below here are hyperparameters for sampler
parser.add_argument("--prior-low", type=float, default=-20.0, help="Prior lower bound.")
parser.add_argument("--prior-high",  type=float, default=20.0, help="Prior upper bound.")
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












##############################################################################################
# 3. CONFIGURATIONS OF THE EXPERIMENT
##############################################################################################
# grab data
total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914 for the analysis
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









##############################################################################################
# 4. TRANSFORM PRIOR FOR SMS IN JIM's UNCONSTARINT SPACE
##############################################################################################
# compute final parameter names after all sample_transforms
current_names = list(prior.parameter_names)
for t in sample_transforms:
    current_names = t.propagate_name(current_names)
final_names = current_names
D_transformed = len(final_names)
print(f"Physical parameters ({len(prior.parameter_names)}): {prior.parameter_names}")
print(f"Transformed parameters ({D_transformed}): {final_names}")


# helper functions to convert between dict and array
def dict_to_array(d, names):
    return jnp.array([d[name] for name in names])

def array_to_dict(x, names):
    return {name: x[i] for i, name in enumerate(names)}


# sample from the transformed prior (physical -> forward transforms)
def sample_transformed_prior(key, n):
    phys_dict = prior.sample(key, n)                     # dict of physical samples, each value shape (n,)
    # apply forward transforms sequentially
    current_dict = phys_dict
    for t in sample_transforms:
        current_dict = jax.vmap(t.forward)(current_dict) # vmap over batch
    # convert to array in final_names order
    return jnp.stack([current_dict[name] for name in final_names], axis=1)


# log‑probability in transformed space (physical log‑prob + log|J|)
def logpdf_transformed(x):
    """
    x : array (D_transformed,) or (n, D_transformed)
    returns log probability of x in the transformed space.
    """
    def _logpdf_one(xi):
        d = array_to_dict(xi, final_names)
        # inverse transforms (reverse order) accumulate log Jacobian
        logjac = 0.0
        for t in reversed(sample_transforms):
            d, ld = t.inverse(d)
            logjac += ld
        phys_logprob = prior.log_prob(d)
        return phys_logprob + logjac

    if x.ndim == 1:
        return _logpdf_one(x)
    else:
        return jax.vmap(_logpdf_one)(x)


# likelihood for unconstrained samples
def gw_loglike_unconstrained(x):
    """
    x : array (D_transformed,) or (n, D_transformed)
    returns log likelihood.
    """
    def _like_one(xi):
        d = array_to_dict(xi, final_names)
        # inverse transforms to get physical parameters
        for t in reversed(sample_transforms):
            d, _ = t.inverse(d)          # ignore logjac
        # apply likelihood transforms (original list, already instances)
        for t in likelihood_transforms:
            d = t.forward(d)
        return likelihood.evaluate(d, {})

    if x.ndim == 1:
        return _like_one(x)
    else:
        return jax.vmap(_like_one)(x)


# convert unconstrained samples back to physical space (for posterior comparison)
def unconstrained_to_physical(x):
    """
    x : array (n, D_transformed) or (D_transformed,)
    returns physical samples in same order as prior.parameter_names.
    """
    def _to_phys_one(xi):
        d = array_to_dict(xi, final_names)
        for t in reversed(sample_transforms):
            d, _ = t.inverse(d)
        # return array in original physical order
        return jnp.array([d[name] for name in prior.parameter_names])

    if x.ndim == 1:
        return _to_phys_one(x)
    else:
        return jax.vmap(_to_phys_one)(x)


# create a prior object that matches the SMC sampler expected input
class TransformedPrior:
    def __init__(self, sample_fn, logpdf_fn, dim):
        self.sample = sample_fn
        self.logpdf = logpdf_fn
        self.dim = dim
        self.params = jnp.zeros((dim, 2))   # dummy, not used

    def bounds(self):
        # all parameters are unbounded after the transforms
        return jnp.array([[-jnp.inf, jnp.inf]] * self.dim)
    
    def logpdf1(self, x):
        """
        Log probability for a single point (D,). Returns a scalar.
        """
        return self.logpdf(x)
    
prior_smc = TransformedPrior(
    sample_fn=lambda key, n: sample_transformed_prior(key, n),
    logpdf_fn=logpdf_transformed,
    dim=D_transformed
)






##############################################################################################
# 5. OUTDIR AND PLOTS
##############################################################################################
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



def plot_diagnostics(out, n_active, n_dims, outdir,
                                  filename="diagnostics_core_long.pdf"):
    """
    Two-page PDF with one plot per row (full-width subplots):

    Page 1:
        1. beta(t)
        2. ESS(t)
        3. logZ(t)

    Page 2:
        4. acceptance(t)
        5. sigma(t)  (reconstructed from efficiency, beta>0 only)
    """
    import os
    from matplotlib.backends.backend_pdf import PdfPages

    T = int(np.asarray(out.state.t))
    if T < 2:
        raise ValueError(f"Not enough iterations recorded (t={T}).")

    it     = np.arange(T)
    beta   = np.asarray(out.state.beta[:T]).reshape(-1)
    ess    = np.asarray(out.state.ess[:T]).reshape(-1)
    accept = np.asarray(out.state.accept[:T]).reshape(-1)
    logz   = np.asarray(out.state.logz[:T]).reshape(-1)
    eff    = np.asarray(out.state.efficiency[:T]).reshape(-1)

    # proposal scale normalisation (same as in mutate())
    norm_ref = 2.38 / np.sqrt(n_dims)

    # sigma only meaningful once beta > 0
    mask_sigma = beta > 0.0
    it_sigma   = it[mask_sigma]
    sigma      = eff[mask_sigma] * norm_ref

    # useful ratios
    ess_ratio = ess / max(1, n_active)

    # a bit taller than A4
    figsize = (8.27, 13.0)  # width, height in inches

    save_path = os.path.join(outdir, filename)

    with PdfPages(save_path) as pdf:
        # PAGE 1: beta, ESS, logZ
        # ------------------------------
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=figsize,
            sharex=False,
            constrained_layout=True,
        )
        fig.suptitle("SMC core diagnostics – page 1/2", fontsize=14)

        # 1. beta(t)
        ax = axes[0]
        ax.plot(it, beta, marker="o", linewidth=1)
        ax.set_title("beta(t)")
        ax.set_xlabel("SMC iteration")
        ax.set_ylabel("beta")
        ax.set_ylim(min(-0.02, beta.min()), max(1.02, beta.max()))
        ax.grid(True, alpha=0.3)

        # 2. ESS(t) and ESS/N_active
        ax = axes[1]
        ax.plot(it, ess, marker="o", linewidth=1, label="ESS")
        ax.plot(it, ess_ratio * n_active,
                linestyle="--", linewidth=1,
                label="ESS/N_active × N_active")
        ax.axhline(n_active, linestyle=":", linewidth=1, label="N_active")
        ax.set_title("ESS(t)")
        ax.set_xlabel("SMC iteration")
        ax.set_ylabel("ESS")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # 3. logZ(t)
        ax = axes[2]
        ax.plot(it, logz, marker="o", linewidth=1)
        ax.set_title("logZ(t)")
        ax.set_xlabel("SMC iteration")
        ax.set_ylabel("logZ")
        ax.grid(True, alpha=0.3)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 2: accept, sigma
        # ------------------------------
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=figsize,
            sharex=False,
            constrained_layout=True,
        )
        fig.suptitle("SMC core diagnostics – page 2/2", fontsize=14)

        # 4) acceptance(t)
        ax = axes[0]
        ax.plot(it, accept, marker="o", linewidth=1)
        ax.set_title("acceptance rate")
        ax.set_xlabel("SMC iteration")
        ax.set_ylabel("accept")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

        # 5) sigma(t) (beta > 0)
        ax = axes[1]
        ax.plot(it_sigma, sigma, marker="o", linewidth=1)
        ax.set_title("proposal scale sigma(t)  (beta > 0)")
        ax.set_xlabel("SMC iteration")
        ax.set_ylabel("sigma")
        ax.grid(True, alpha=0.3)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved core diagnostics PDF to {save_path}")








##############################################################################################
# 6. EXPERIMENT RUNNER
##############################################################################################
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

    t0 = time.time()


    sampler = SamplerJAX(
        prior_u,          # <-- use the transformed prior
        loglike_x,
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

    # samples are in the transformed (unconstrained) space
    theta_transformed = np.asarray(post.samples_resampled[:n_keep])

    # convert to physical space for saving and comparison
    theta_physical = np.asarray(unconstrained_to_physical(theta_transformed))  

    logZ = float(np.asarray(out.logz))
    logZerr = float(np.asarray(out.logz_err))

    print("Sampling complete!")
    print("n_prior (adjusted) =", n_prior, "(input was", n_prior_in, ")")
    print("samples.shape =", theta_physical.shape)
    print("logZ =", logZ, "logZerr =", logZerr)

    # save
    outdir = next_run_dir(os.path.join(out_root, event_name))

    # create diagnostics PDF
    plot_diagnostics(out, n_active=n_active, n_dims=D, outdir=outdir)

    # ------------------------------------------------------------------
    # SAVE POSTERIOR IN HDF5 FILE (physical parameters)
    # ------------------------------------------------------------------
    h5_path = os.path.join(outdir, "posterior.hdf5")

    meta = {
        "event": event_name,
        "parameter_names": list(prior.parameter_names),   # physical names
        "n_samples": int(theta_physical.shape[0]),
        "logz": float(logZ),
        "logz_err": float(logZerr),
        "seed": int(seed),
        "note": "Posterior draws from sampler (physical space)",
    }

    with h5py.File(h5_path, "w") as f_h5:
        f_h5.create_dataset("samples", data=theta_physical)
        f_h5.create_dataset("names", data=np.asarray(list(prior.parameter_names), dtype="S"))
        for k, v in meta.items():
            if k == "parameter_names":
                f_h5.attrs[k] = ",".join(v)
            else:
                f_h5.attrs[k] = v

    print(f"[{event_name}] wrote posterior HDF5: {h5_path}")

    #1. TRUE posterior HDF5 path

    TRUE_FILE = "/home/obevza/jaxpsmc/GW_examples/GW150914_095045_data0_1126259462-391_analysis_H1L1_result.hdf5"

    # match my parameters with true posteriors
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
    gps_ref = 1126259462.4

    def load_true_samples(true_file: str, names: list[str]) -> np.ndarray:
        with h5py.File(true_file, "r") as f_true:
            post_true = f_true["posterior"]
            cols = []
            for nm in names:
                true_nm = name_map[nm]
                arr = post_true[true_nm][:]
                if nm == "t_c":
                    arr = arr - gps_ref
                cols.append(arr)
        return np.column_stack(cols)

    # load true samples and convert our samples
    samples_true = load_true_samples(TRUE_FILE, list(prior.parameter_names))
    samples_ours = theta_physical

    # print parameter ranges
    for i, name in enumerate(prior.parameter_names):
        col_true = samples_true[:, i]
        col_ours = samples_ours[:, i]
        print(f"True {name}: min={col_true.min():.4f}, max={col_true.max():.4f}, unique={np.unique(col_true).size}")
        print(f"Ours {name}: min={col_ours.min():.4f}, max={col_ours.max():.4f}, unique={np.unique(col_ours).size}")


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

    fig = plt.figure(figsize=(16, 16))

    fig = corner.corner(
        samples_true,
        fig=fig,
        labels=labels_latex if len(labels_latex) == len(prior.parameter_names) else list(prior.parameter_names),
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

    handles = [
        plt.Line2D([], [], color="blue", label="sampler"),
        plt.Line2D([], [], color="red", label="True Normal"),
    ]
    fig.legend(handles=handles, loc="upper right")

    for ax in fig.get_axes():
        ax.grid(False)

    save_path = os.path.join(outdir, "corner_true_vs_mine.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved overlay corner:", save_path)

    print(f"[{event_name}] saved {theta_physical.shape[0]} samples to: {outdir}")
    return outdir, theta_physical








##############################################################################################
# 7. RUN
##############################################################################################

import sys
import argparse

prior_u = prior                      

bounds = np.asarray(prior_smc.bounds())       # (D, 2)
ranges = [tuple(map(float, b)) for b in bounds]
prior_smc = TransformedPrior(
    sample_fn=lambda key, n: sample_transformed_prior(key, n),
    logpdf_fn=logpdf_transformed,
    dim=D_transformed
)



sys.argv = [
    "notebook",
    "--outdir", "/home/obevza/jaxpsmc/GW_examples",       
    "--nr-of-samples", "10000",        

    "--n-effective", "7000",
    "--n-active", "7000",
    "--n-prior", "175000",

    "--n-total", "10000",
    "--pc-n-steps", "450",
    "--pc-n-max-steps", "850",
    "--keep-max", "30000",
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
    prior_u=prior_smc,                     # use transformed prior
    loglike_x=gw_loglike_unconstrained,    # use likelihood
    D=D_transformed,                       # dimension in transformed space
    names=final_names,                     # transformed parameter names
    ranges=None,                           # will be computed inside or set to None
    periodic_idx=None,
    args=args,
)












