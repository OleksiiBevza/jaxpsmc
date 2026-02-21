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
import argparse

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

from jax.tree_util import tree_map
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import corner
import matplotlib as mpl
mpl.rcParams["axes.grid"] = False









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

kinds = jnp.full((D,), UNIFORM, dtype=jnp.int32)
prior_u = Prior.create(kinds, jnp.asarray(bounds, dtype=jnp.float64))

periodic_idx = jnp.array(
    [names.index(k) for k in ["s1_phi", "s2_phi", "phase_c", "ra"]],
    dtype=jnp.int64
)












##################################################################################
# 4. LIKELIHOOD
##################################################################################

def loglike_x(x: jax.Array) -> jax.Array:
    M_c = x[0]
    q = x[1]

    s1_mag = x[2]
    s1_th = x[3]
    s1_ph = x[4]

    s2_mag = x[5]
    s2_th = x[6]
    s2_ph = x[7]

    iota = x[8]

    d_L = x[9]
    t_c = x[10]

    phase_c = x[11]
    psi = x[12]

    ra = x[13]
    dec = x[14]

    eta = q / (1.0 + q) ** 2

    s1x = s1_mag * jnp.sin(s1_th) * jnp.cos(s1_ph)
    s1y = s1_mag * jnp.sin(s1_th) * jnp.sin(s1_ph)
    s1z = s1_mag * jnp.cos(s1_th)

    s2x = s2_mag * jnp.sin(s2_th) * jnp.cos(s2_ph)
    s2y = s2_mag * jnp.sin(s2_th) * jnp.sin(s2_ph)
    s2z = s2_mag * jnp.cos(s2_th)

    params = {
        "M_c": M_c,
        "q": q,
        "eta": eta,

        "iota": iota,
        "d_L": d_L,
        "t_c": t_c,
        "phase_c": phase_c,
        "psi": psi,
        "ra": ra,
        "dec": dec,

        "s1_mag": s1_mag,
        "s1_theta": s1_th,
        "s1_phi": s1_ph,

        "s2_mag": s2_mag,
        "s2_theta": s2_th,
        "s2_phi": s2_ph,

        "s1_x": s1x,
        "s1_y": s1y,
        "s1_z": s1z,

        "s2_x": s2x,
        "s2_y": s2y,
        "s2_z": s2z,
    }

    return likelihood.evaluate(params, {})












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
    sampler = SamplerJAX(prior_u, loglike_x, cfg, flow=IdentityFlowJAX(D))
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

    meta = {
        "event": event_name,
        "parameter_names": names,
        "n_samples": int(theta.shape[0]),
        "logz": logZ,
        "logz_err": logZerr,
        "seed": int(seed),
        "note": "Posterior draws from sampler",
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for j, nm in enumerate(names):
        with open(os.path.join(outdir, f"{nm}_posterior.json"), "w") as f:
            json.dump(theta[:, j].astype(float).tolist(), f)

    # corner plot 
    fig = corner.corner(
        theta,
        labels=names,
        range=ranges,
        show_titles=True,
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
        bins=30,
    )
    for ax in fig.get_axes():
        ax.grid(False)

    fig.savefig(os.path.join(outdir, "corner.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[{event_name}] saved {theta.shape[0]} samples to: {outdir}")
    return outdir, theta










##################################################################################
# 6. RUN EXPERIMENT
##################################################################################


import sys
import argparse

sys.argv = [
    "notebook",
    "--outdir", "/home/obevza/jaxpsmc/GW_examples",   
    "--nr-of-samples", "10000",        

    "--n-effective", "100",
    "--n-active", "100",
    "--n-prior", "15",

    "--n-total", "100",
    "--pc-n-steps", "5",
    "--pc-n-max-steps", "5",
    "--keep-max", "80",
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
    loglike_x=loglike_x,
    D=D,
    names=names,
    ranges=ranges,
    periodic_idx=periodic_idx,
    args=args,
)

print("Saved to:", outdir)
print("theta.shape =", theta.shape)













