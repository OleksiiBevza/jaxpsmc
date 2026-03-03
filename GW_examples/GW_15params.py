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






##################################################################################
# 3. PRIOR 
##################################################################################

names = [
    "M_c",
    "q",
    "s1_mag",
    "s1_theta",
    "s1_phi",
    "s2_mag",
    "s2_theta",
    "s2_phi",
    "iota",
    "d_L",
    "t_c",
    "phase_c",
    "psi",
    "ra",
    "dec",
]
D = len(names)

# (matches GW150914_IMRPhenomPV2 ranges)
bounds = np.array([
    [10.0, 80.0],            # M_c
    [0.125, 1.0],            # q
    [0.0, 1.0],              # s1_mag
    [0.0, np.pi],            # s1_theta
    [0.0, 2*np.pi],          # s1_phi
    [0.0, 1.0],              # s2_mag
    [0.0, np.pi],            # s2_theta
    [0.0, 2*np.pi],          # s2_phi
    [0.0, np.pi],            # iota
    [1.0, 2000.0],           # d_L
    [-0.05, 0.05],           # t_c
    [0.0, 2*np.pi],          # phase_c
    [0.0, np.pi],            # psi
    [0.0, 2*np.pi],          # ra
    [-np.pi/2, np.pi/2],     # dec
], dtype=np.float64)

# box prior used by the SMC sampler (uniform in the above bounds)
kinds = jnp.full((D,), UNIFORM, dtype=jnp.int32)
prior_u = Prior.create(kinds, jnp.asarray(bounds, dtype=jnp.float64))

# periodic angles (for sampler)
periodic_idx = jnp.array(
    [names.index(k) for k in ["s1_phi", "s2_phi", "phase_c", "ra"]],
    dtype=jnp.int64
)




def logprior_phys(x: jax.Array) -> jax.Array:
    """
    Prior that directly calls the CombinePrior 'prior' from GW150914_IMRPhenomPV2.py.
    x is the 15D parameter vector in the order used by your sampler.
    """
    (
        M_c, q,
        s1_mag, s1_th, s1_ph,
        s2_mag, s2_th, s2_ph,
        iota,
        d_L, t_c,
        phase_c, psi,
        ra, dec,
    ) = x

    # Build the named dict in the SAME parameterization that GW150914_IMRPhenomPV2 uses.
    params = {
        "M_c": M_c,
        "q": q,
        # UniformSpherePrior("s1") / "s2" expand to these components
        "s1_mag": s1_mag,
        "s1_theta": s1_th,
        "s1_phi": s1_ph,
        "s2_mag": s2_mag,
        "s2_theta": s2_th,
        "s2_phi": s2_ph,
        "iota": iota,
        "d_L": d_L,
        "t_c": t_c,
        "phase_c": phase_c,
        "psi": psi,
        "ra": ra,
        "dec": dec,
    }

    # This is EXACTLY what Jim does before adding Jacobians from sample_transforms:
    # prior = self.prior.log_prob(named_params)
    # (we are not using sample_transforms, so no Jacobian term here)
    return prior.log_prob(params)








# three detector-frame transforms were defined earlier in your file as
#   frame_transforms = sample_transforms[1:4]
# we keep those as-is for the distance / tc / sky stuff.

def loglike_x(x: jax.Array) -> jax.Array:
    (
        M_c, q,
        s1_mag, s1_th, s1_ph,
        s2_mag, s2_th, s2_ph,
        iota,
        d_L, t_c,
        phase_c, psi,
        ra, dec,
    ) = x

    # 1. Start in the PRIOR coordinate system (same as in GW150914_IMRPhenomPV2)
    params = {
        "M_c": M_c,
        "q": q,
        "s1_mag": s1_mag,
        "s1_theta": s1_th,
        "s1_phi": s1_ph,
        "s2_mag": s2_mag,
        "s2_theta": s2_th,
        "s2_phi": s2_ph,
        "iota": iota,
        "d_L": d_L,
        "t_c": t_c,
        "phase_c": phase_c,
        "psi": psi,
        "ra": ra,
        "dec": dec,
    }

    # 2. Apply the SAME likelihood_transforms as Jim does
    #    (MassRatioToSymmetricMassRatioTransform + two SphereSpinToCartesianSpinTransform)
    named_for_like = params
    for tr in LIKELIHOOD_TRANSFORMS_FOR_SAMPLER:
        named_for_like = tr.forward(named_for_like)

    # 3. Your frame_transforms (distance to d_hat, tc/phase/sky transforms)
    #    – this is what Jim does in the waveform/likelihood, you already have it set up.
    #    If you previously applied frame_transforms *inside* loglike_x, keep that logic here,
    #    but now operate on 'named_for_like' instead of the old manual dict.
    for tr in frame_transforms:
        named_for_like, _ = tr.inverse(named_for_like)  # or .forward/.backward depending on how you used them before

    # 4. Evaluate the Jim likelihood on the fully transformed dict
    return likelihood.evaluate(named_for_like, {})








def logtarget_x(x: jax.Array) -> jax.Array:
    return loglike_x(x) + logprior_phys(x)










##################################################################################
# 5. RUNNER
##################################################################################



# -------------------------------------------------
# OUTDIR
#--------------------------------------------------
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






def plot_core_diagnostics_longpdf(out, n_active, n_dims, outdir,
                                  filename="diagnostics_core_long.pdf"):
    """
    Two-page PDF with one plot per row (full-width subplots):

    Page 1:
        1) beta(t)
        2) ESS(t)
        3) logZ(t)

    Page 2:
        4) acceptance(t)
        5) sigma(t)  (reconstructed from efficiency, beta>0 only)
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
        # ==========================
        # PAGE 1: beta, ESS, logZ
        # ==========================
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=figsize,
            sharex=False,
            constrained_layout=True,
        )
        fig.suptitle("SMC core diagnostics – page 1/2", fontsize=14)

        # 1) beta(t)
        ax = axes[0]
        ax.plot(it, beta, marker="o", linewidth=1)
        ax.set_title("beta(t)")
        ax.set_xlabel("SMC iteration")
        ax.set_ylabel("beta")
        ax.set_ylim(min(-0.02, beta.min()), max(1.02, beta.max()))
        ax.grid(True, alpha=0.3)

        # 2) ESS(t) and ESS/N_active
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

        # 3) logZ(t)
        ax = axes[2]
        ax.plot(it, logz, marker="o", linewidth=1)
        ax.set_title("logZ(t)")
        ax.set_xlabel("SMC iteration")
        ax.set_ylabel("logZ")
        ax.grid(True, alpha=0.3)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ==========================
        # PAGE 2: accept, sigma
        # ==========================
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



















# -------------------------------------------------
# RUNNER
# -------------------------------------------------

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
    # sampler = SamplerJAX(prior_u, loglike_x, cfg, flow=IdentityFlowJAX(D))
    sampler = SamplerJAX(prior_u, logtarget_x, cfg, flow=IdentityFlowJAX(D))
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

    # create diagnostics PDF (beta, ESS, logZ, acceptance, sigma)
    plot_core_diagnostics_longpdf(out, n_active=n_active, n_dims=D, outdir=outdir)

    # ------------------------------------------------------------------
    # 1) SAVE *YOUR* POSTERIOR IN ONE HDF5 FILE
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
        # parameter names as fixed-length strings
        f_h5.create_dataset("names", data=np.asarray(names, dtype="S"))
        # simple attributes
        for k, v in meta.items():
            f_h5.attrs[k] = v

    # ------------------------------------------------------------------
    # 2) LOAD TRUE POSTERIOR FROM BILBY/JIMGW HDF5
    # ------------------------------------------------------------------
    true_file = "/home/obevza/jaxpsmc/GW_examples/GW150914_095045_data0_1126259462-391_analysis_H1L1_result.hdf5"

    # mapping from *your* parameter names -> names in true posterior
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



    gps_ref = 1126259462.4  # same GPS time as in GW150914_IMRPhenomPV2.py

    with h5py.File(true_file, "r") as f_true:
        post_true = f_true["posterior"]
        true_cols = []
        for nm in names:
            true_nm = name_map[nm]
            arr = post_true[true_nm][:]

            # convert absolute geocent_time -> offset t_c (seconds)
            if nm == "t_c":
                arr = arr - gps_ref

            true_cols.append(arr)

        samples_true = np.column_stack(true_cols)



    # ------------------------------------------------------------------
    # 3) BUILD OVERLAY: TRUE (BLACK) VS YOUR SAMPLER (ORANGE)
    # ------------------------------------------------------------------
    samples_ours = theta  # same order as `names`

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

    # First: corner plot of the TRUE posterior ONLY.
    # This sets the axis limits from the true samples
    # in the same way as in the original "true post" plot.
    fig = plt.figure(figsize=(16, 16))
    fig = corner.corner(
        samples_true,
        fig=fig,
        labels=labels_latex,
        show_titles=True,
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
        bins=30,
        color="black",
        hist_kwargs={"density": True},
    )

    # Second: overlay YOUR posterior on the SAME figure,
    # without specifying `range`, so the limits stay identical.
    corner.corner(
        samples_ours,
        fig=fig,
        plot_datapoints=False,
        plot_density=True,
        fill_contours=False,
        bins=30,
        color="tab:orange",
        hist_kwargs={"density": True},
    )

    for ax in fig.get_axes():
        ax.grid(False)

    fig.savefig(
        os.path.join(outdir, "corner_true_vs_mine.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[{event_name}] saved {theta.shape[0]} samples to: {outdir}")
    print(f"  HDF5 posterior: {h5_path}")
    print("  Corner overlay: corner_true_vs_mine.png")

    return outdir, theta














##################################################################################
# 6. RUN EXPERIMENT
##################################################################################


sys.argv = [
    "notebook",
    "--outdir", "/home/obevza/jaxpsmc/GW_examples",   
    "--nr-of-samples", "10000",        

    "--n-effective", "1000",
    "--n-active", "1000",
    "--n-prior", "5000",

    "--n-total", "1000",
    "--pc-n-steps", "50",
    "--pc-n-max-steps", "50",
    "--keep-max", "2000",
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
ranges = [tuple(map(float, b)) for b in bounds]


outdir, theta = run_event_and_save_posteriors(
    event_name="GW150914",
    prior_u=prior_u,
    loglike_x=logtarget_x,  # <-- use logtarget_x here
    D=D,
    names=names,
    ranges=ranges,
    periodic_idx=periodic_idx,
    args=args,
)


print("Saved to:", outdir)
print("theta.shape =", theta.shape)











