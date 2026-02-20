##################################################################################
# 1. PACKAGES
##################################################################################

# diagnostics
import os
import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import corner
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Preconditioned Monte Carlo
import numpy as np

from scipy.stats import uniform
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import norm




import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
from jax.scipy.linalg import solve_triangular


# my classes
from NUMlikelihood import *
from NUMgaussian_mixture import *
from sampler.sampler_jax import *

import numpy as np
import jax
import jax.numpy as jnp

from sampler.prior_jax import *
#from sampler_jax import SamplerConfigJAX, SamplerJAX, IdentityFlowJAX
from sampler.sampler_helper_jax import *














"""
Code for generating and running toy problems with pocomc sampler
"""




SUPPORTED_EXPERIMENTS = ["gaussian"]
##################################################################################
# 2. ARGUMENTS
##################################################################################
### The argparse is used to store and process any user input we want to pass on
parser = argparse.ArgumentParser(description="Run experiment with specified parameters.")
parser.add_argument(
    "--experiment-type",
    choices=["gaussian", "dualmoon", "rosenbrock"],
    required=True,
    help="Which experiment to run."
)
parser.add_argument(
    "--n-dims",
    type=int,
    required=True,
    help="Number of dimensions."
)
parser.add_argument(
    "--outdir",
    type=str,
    required=True,
    help="The output directory, where things will be stored"
)

# Everything below here are hyperparameters for the Gaussian experiment. 
parser.add_argument(
    "--nr-of-samples",
    type=int,
    default=10000,
    help="Number of samples to be geerated"
)
parser.add_argument(
    "--nr-of-components",
    type=int,
    default=2,
    help="Number of components to be geerated"
)
parser.add_argument(
    "--width-mean",
    type=float,
    default=10.0,
    help="The width of mean"
)
parser.add_argument(
    "--width-cov",
    type=float,
    default=3.0,
    help="The width of cov"
)
parser.add_argument(
    "--weights-of-components",
    nargs="+",          
    type=float,
    default=None,
    help="Mixture weights (--weights-of-components 0.3 0.7). If omitted, uses equal weights."
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
# JAX sampler controls
parser.add_argument("--n-total", type=int, default=4096)
parser.add_argument("--pc-n-steps", type=int, default=8)
parser.add_argument("--pc-n-max-steps", type=int, default=80)
parser.add_argument("--keep-max", type=int, default=4096)
parser.add_argument("--random-state", type=int, default=0)

parser.add_argument("--precondition", action="store_true", default=True)  # keep True by default
parser.add_argument("--no-precondition", action="store_false", dest="precondition")

parser.add_argument("--dynamic", action="store_true", default=True)
parser.add_argument("--no-dynamic", action="store_false", dest="dynamic")

parser.add_argument("--metric", type=str, default="ess", choices=["ess", "uss"])
parser.add_argument("--resample", type=str, default="mult", choices=["mult", "syst"])
parser.add_argument("--transform", type=str, default="probit", choices=["probit", "logit"])

# optional: for now, you’re using identity flow for “untrained but flow-shaped”
parser.add_argument("--use-identity-flow", action="store_true", default=True)




parser.add_argument("--n-effective", type=int, required=True)
parser.add_argument("--n-active", type=int, required=True)
parser.add_argument("--n-prior", type=int, required=True)








##################################################################################
# 3. EXPERIMENT RUNNER
##################################################################################
class SequentialMCExperimentRunner:
    """
    Base class storing everything shared between different run experiments
    """
    def __init__(self, args):
        # Process the argparse args into params:
        self.params = vars(args)

        # Automatically create a unique output directory: results_1, results_2, ...
        base_results_dir = self.params["outdir"]
        unique_outdir = self.get_next_available_outdir(base_results_dir)
        print(f"Using output directory: {unique_outdir}")
        os.makedirs(unique_outdir, exist_ok=False)
        self.params["outdir"] = unique_outdir

        # Check if experiment type is allowed/supported:
        if self.params["experiment_type"] not in SUPPORTED_EXPERIMENTS:
            raise ValueError(
                f"Experiment type {self.params['experiment_type']} is not supported. "
                f"Supported types are: {SUPPORTED_EXPERIMENTS}"
            )

        # Show the parameters to the screen/log file
        print("Passed parameters:")
        for key, value in self.params.items():
            print(f"{key}: {value}")



        # Specify the desired target function based on the experiment type
        # ... your outdir logic etc ...
        if self.params["experiment_type"] == "gaussian":
            print("Setting the target function to a standard Gaussian distribution.")

            # defining parameters for mcmc sampler 
            np.random.seed(501)

            D = self.params["n_dims"]
            
            true_samples, means, covariances, weights = GaussianMixtureGenerator.generate_gaussian_mixture(
                n_dim=D,
                n_gaussians=args.nr_of_components,
                n_samples = args.nr_of_samples,
                width_mean = args.width_mean,
                width_cov = args.width_cov,
                weights= args.weights_of_components,
            ) 

            # Store true samples for diagnostics later on
            self.true_samples = true_samples

            # Convert to JAX arrays 
            self.mcmc_means   = jnp.stack(means, axis=0)         # (K, D)
            self.mcmc_covs    = jnp.stack(covariances, axis=0)   # (K, D, D)
            self.mcmc_weights = jnp.asarray(weights)             # (K,)

            # Define Likelihood 
            self.likelihood = GaussianMixtureLikelihood(
                means=self.mcmc_means,
                covs=self.mcmc_covs,
                weights=self.mcmc_weights,
            )

            self.target_fn = self.target_normal

        

    def get_next_available_outdir(self, base_dir: str, prefix: str = "results") -> str:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        matches = [re.match(rf"{prefix}_(\d+)", name) for name in existing]
        numbers = [int(m.group(1)) for m in matches if m]
        next_number = max(numbers, default=0) + 1
        return os.path.join(base_dir, f"{prefix}_{next_number}")



    def target_normal(self, x):
        # x can be shape (D,) or (..., D); GaussianMixtureLikelihood.log_prob supports both
        return self.likelihood.log_prob(x)
    

    @staticmethod
    def make_auto_bounds_inflated(means, covs, inflate=9.0, nsig=12.0, pad=1e-6,
                                    prior_low=None, prior_high=None):
        """
        Per-dimension uniform bounds [low[d], high[d]] that 
        contain all mass of a Gaussian mixture:
            - Bounds per component are wide enough to cover every single component;
            - Created by means and cov  
        achieved by:
            - find smallest and largest component means in dim k:
            - for each component k, take the marginal sd in dim k 
            - set bounds
        """
        means = np.asarray(means, dtype=float)           # (K, D)
        covs  = np.asarray(covs,  dtype=float) * float(inflate)   # make variance bigger

        # find smallest and largest component means in dim k
        mu_min = means.min(axis=0)                       # (D,)
        mu_max = means.max(axis=0)                       # (D,)

        # for each component k, take the marginal sd in dim k 
        std_max = np.sqrt(np.stack([np.diag(C) for C in covs], axis=0)).max(axis=0)  # (D,)

        # set bounds
        low  = mu_min - nsig * std_max - pad
        high = mu_max + nsig * std_max + pad

        # if bounds are provided, make bigger bounds, if necessary
        if prior_low  is not None: low  = np.minimum(low,  float(prior_low))
        if prior_high is not None: high = np.maximum(high, float(prior_high))
        return low, high
    




    #-----------------------------------------------------------------------------
    # 3.1. RUN EXPERIMENT
    #-----------------------------------------------------------------------------

    def run_experiment(self):
        """
        Run the JAX SMC sampler.
        Stores results on self and returns a dict (same structure as before).
        """


        dim = int(self.params["n_dims"])

        # -------------------------
        # Prior bounds 
        # -------------------------
        means_np = np.asarray(self.mcmc_means)
        covs_np  = np.asarray(self.mcmc_covs)

        low_np, high_np = self.make_auto_bounds_inflated(
            means_np, covs_np,
            inflate=9.0,
            nsig=12.0,
            prior_low=float(self.params["prior_low"]),
            prior_high=float(self.params["prior_high"]),
        )

        low  = jnp.asarray(low_np, dtype=jnp.float64)
        high = jnp.asarray(high_np, dtype=jnp.float64)

        # JAX Uniform Prior over [low[d], high[d]]
        kinds  = jnp.full((dim,), UNIFORM, dtype=jnp.int32)        # UNIFORM constant from prior_jax
        params = jnp.stack([low, high], axis=1)                    # (D,2): [low, high]
        prior  = Prior.create(kinds, params)
        self.prior = prior

        # -------------------------
        # Likelihood: pure JAX (D,) -> (scalar, blobs)
        # -------------------------
        if hasattr(self, "likelihood") and self.likelihood is not None:
            loglike_single = self.likelihood.loglike_single
        else:
            # likelihood goes here
            def loglike_single(x_1d):
                ll = self.target_normal(x_1d)  # 
                return ll, jnp.zeros((0,), dtype=ll.dtype)

        # -------------------------
        # Read sampler params
        # -------------------------
        n_effective = int(self.params.get("n_effective", 512))
        n_active    = int(self.params.get("n_active", 256))
        n_prior_in  = int(self.params.get("n_prior", 512))

        # IMPORTANT: n_prior must be multiple of n_active for this sampler
        n_prior = int(np.ceil(n_prior_in / n_active) * n_active)

        n_total     = int(self.params.get("n_total", 4096))
        n_steps     = int(self.params.get("pc_n_steps", 8))
        n_max_steps = int(self.params.get("pc_n_max_steps", 80))
        keep_max    = int(self.params.get("keep_max", 4096))

        precond  = bool(self.params.get("precondition", True))
        dynamic  = bool(self.params.get("dynamic", True))
        metric   = str(self.params.get("metric", "ess"))
        resample = str(self.params.get("resample", "mult"))
        transform = str(self.params.get("transform", "probit"))

        # -------------------------
        # build sampler
        # -------------------------
        cfg = SamplerConfigJAX(
            n_dim=dim,
            n_effective=n_effective,
            n_active=n_active,
            n_prior=n_prior,
            n_total=n_total,
            n_steps=n_steps,
            n_max_steps=n_max_steps,
            keep_max=keep_max,
            blob_dim=0,
            preconditioned=precond,
            dynamic=dynamic,
            metric=metric,
            resample=resample,
            transform=transform,
            enable_flow_evidence=False,   # as for now not used 
        )

        # use dummy flow
        flow_obj = IdentityFlowJAX(cfg.n_dim) if bool(self.params.get("use_identity_flow", True)) else self.flow

        sampler = SamplerJAX(prior, loglike_single, cfg, flow=flow_obj)

        # -------------------------
        # Run
        # -------------------------
        random_state = int(self.params.get("random_state", 0))
        key = jax.random.PRNGKey(random_state)
        out = sampler.run(key, n_total)   

        # -------------------------
        # posterior draws for diagnostics
        # -------------------------
        key_post = jax.random.fold_in(key, 1)
        resample_method = jnp.int32(0 if resample == "mult" else 1)

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

        # choose how many samples you want to keep
        n_keep = int(self.params.get("nr_of_samples", min(cfg.keep_max, int(post.samples_resampled.shape[0]))))

        samples = np.asarray(post.samples_resampled[:n_keep])
        logl    = np.asarray(post.logl_resampled[:n_keep])
        logp    = np.asarray(post.logp_resampled[:n_keep])

        logZ    = float(np.asarray(out.logz))
        logZerr = float(np.asarray(out.logz_err))  # will be nan cause flow is not trained

        # save results
        self.samples = samples
        self.logl = logl
        self.logp = logp
        self.logZ = logZ
        self.logZerr = logZerr
        self.out = out
        self.posterior = post

        self.results = {
            "samples": samples,
            "logl": logl,
            "logp": logp,
            "logZ": logZ,
            "logZerr": logZerr,
            "out": out,
            "posterior": post,
            "params": self.params,
        }

        print("Sampling complete!")
        print("n_prior (adjusted) =", n_prior, "(input was", n_prior_in, ")")
        print("samples.shape =", samples.shape)
        print("logZ =", logZ, "logZerr =", logZerr)

        return self.results




    




    #-----------------------------------------------------------------------------
    # 3.2. PLOT DIAGNOSTICS 
    #-----------------------------------------------------------------------------
    def get_true_and_mcmc_samples(self, discard=0, thin=1):
        dim = int(self.params["n_dims"])

        # True samples
        if not hasattr(self, "true_samples") or self.true_samples is None:
            raise ValueError("No true samples found. Ensure self.true_samples is set (gaussian experiment).")

        true_np = np.asarray(self.true_samples).reshape(-1, dim)

        # Sampler samples
        if hasattr(self, "samples") and self.samples is not None:
            # pocoMC: self.samples is already (N, dim)
            samp = np.asarray(self.samples).reshape(-1, dim)
            samp = samp[int(discard)::int(thin), :]
            mcmc_np = samp
        else:
            raise ValueError(
                "No sampler samples found. Run run_experiment() first. "
                "Expected either self.samples (pocoMC) or self.chains (BlackJAX)."
            )

        return true_np, mcmc_np
    


    def plot_true_vs_mcmc_corner(self, seed=2046):
        """
        Overlay corner plot:
        - MCMC production samples (black)
        - true samples (red)
        Saves: true_vs_mcmc_corner_plot.pdf
        """
        # Get samples 
        true_np, mcmc_np = self.get_true_and_mcmc_samples()

        dim = int(self.params["n_dims"])
        labels = [f"x{i}" for i in range(dim)]

        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        # Plot MCMC first 
        fig = corner.corner(
            mcmc_np,
            color="blue",
            hist_kwargs={"color": "blue", "density": True},
            show_titles=True,
            labels=labels,
        )

        # Overlay true samples 
        corner.corner(
            true_np,
            fig=fig,
            color="red",
            hist_kwargs={"color": "red", "density": True},
            show_titles=True,
            labels=labels,
        )

        # Legend
        handles = [
            plt.Line2D([], [], color="blue", label="sampler"),
            plt.Line2D([], [], color="red", label="True Normal"),
        ]
        fig.legend(handles=handles, loc="upper right")

        save_name = os.path.join(outdir, "true_vs_mcmc_corner_plot.pdf")
        fig.savefig(save_name, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved overlay corner plot to {save_name}")



    def plot_acceptance_rate(self):
        outdir = self.params["outdir"]

        # --- JAX sampler path ---
        if hasattr(self, "out") and self.out is not None:
            T = int(np.asarray(self.out.state.t))
            accept = np.asarray(self.out.state.accept[:T]).reshape(-1)

            plt.figure(figsize=(6, 4))
            plt.plot(np.arange(T), accept)
            plt.xlabel("SMC iteration")
            plt.ylabel("Acceptance rate")
            plt.title("Acceptance rate over SMC iterations (JAX)")
            plt.ylim(0.0, 1.0)
            path = os.path.join(outdir, "acceptance_rate.pdf")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            print(f"Saved acceptance-rate plot to {path}")
            return
        
    
    def plot_sigma(self):
        if not hasattr(self, "out") or self.out is None:
            raise ValueError("No JAX sampler output found. Run run_experiment() first.")

        dim = int(self.params["n_dims"])
        norm_ref = 2.38 / np.sqrt(dim)  # same normalization used in mutate()

        T = int(np.asarray(self.out.state.t))

        # recorded efficiency is sigma / norm_ref in this implementation
        eff = np.asarray(self.out.state.efficiency[:T]).reshape(-1)
        beta = np.asarray(self.out.state.beta[:T]).reshape(-1)

        # drop warmup (beta == 0) so sigma is meaningful
        mask = beta > 0.0
        it = np.arange(T)[mask]
        sigma = (eff[mask] * norm_ref)

        plt.figure(figsize=(6, 4))
        plt.plot(it, sigma)
        plt.xlabel("SMC iteration")
        plt.ylabel("proposal_scale (sigma)")
        plt.title("Preconditioned pCN sigma adaptation (JAX)")
        save_name = os.path.join(self.params["outdir"], "sigma_curve.pdf")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        print(f"Saved to {save_name}")




    def plot_top6_diagnostics_a4_2pages(self, filename="diagnostics_top6_2pages.pdf"):
        """
        Save the 'top 6' sampler diagnostics as a TWO-PAGE A4 PDF (no inset plots).

        Page 1:
        (1) beta(t)       | Δ beta(t)
        (2) ESS(t)        | ESS/N_active
        (5) logZ(t)       | Δ logZ(t)

        Page 2:
        (3) accept(t)     | steps(t)
        (4) sigma(t)      | accept vs sigma scatter
        (6) logl quantiles| final logl histogram

        Requires: self.out (JAX sampler output) from run_experiment().
        """
        import os
        from matplotlib.backends.backend_pdf import PdfPages

        if not hasattr(self, "out") or self.out is None:
            raise ValueError("No JAX sampler output found. Run run_experiment() first.")

        outdir = self.params["outdir"]
        save_path = os.path.join(outdir, filename)

        # ---- pull history ----
        T = int(np.asarray(self.out.state.t))
        if T < 2:
            raise ValueError(f"Not enough iterations recorded (t={T}).")

        it = np.arange(T)

        beta   = np.asarray(self.out.state.beta[:T]).reshape(-1)
        ess    = np.asarray(self.out.state.ess[:T]).reshape(-1)
        accept = np.asarray(self.out.state.accept[:T]).reshape(-1)
        steps  = np.asarray(self.out.state.steps[:T]).reshape(-1)
        logz   = np.asarray(self.out.state.logz[:T]).reshape(-1)
        logl_hist = np.asarray(self.out.state.logl[:T])  # (T, N)

        dbeta = np.diff(beta, prepend=beta[0])
        dlogz = np.diff(logz, prepend=logz[0])

        n_active = int(self.params["n_active"])
        ess_ratio = ess / max(1, n_active)

        # sigma reconstruction from efficiency (exclude warmup beta==0)
        dim = int(self.params["n_dims"])
        norm_ref = 2.38 / np.sqrt(dim)
        eff = np.asarray(self.out.state.efficiency[:T]).reshape(-1)

        mask = beta > 0.0
        it_m = it[mask]
        sigma = eff[mask] * norm_ref
        accept_m = accept[mask]

        # logl quantiles
        q_lo, q_hi = 0.10, 0.90
        q50 = np.quantile(logl_hist, 0.50, axis=1)
        qL  = np.quantile(logl_hist, q_lo, axis=1)
        qH  = np.quantile(logl_hist, q_hi, axis=1)

        # ---- figure helpers ----
        def _page():
            # A4 portrait in inches
            fig = plt.figure(figsize=(8.27, 11.69))
            gs = fig.add_gridspec(
                3, 2,
                left=0.09, right=0.97, top=0.92, bottom=0.06,
                hspace=0.35, wspace=0.28
            )
            return fig, gs

        with PdfPages(save_path) as pdf:
            # ==========================
            # PAGE 1: beta, ESS, logZ
            # ==========================
            fig, gs = _page()
            fig.suptitle("SMC Diagnostics: Page 1/2", fontsize=14)

            # (1) beta(t)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(it, beta, marker="o", linewidth=1)
            ax.set_title("1) beta(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("beta")
            ax.set_ylim(min(-0.02, beta.min()), max(1.02, beta.max()))

            # (1b) Δbeta(t)
            ax = fig.add_subplot(gs[0, 1])
            ax.plot(it, dbeta, marker="o", linewidth=1)
            ax.set_title("Δ beta(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("Δ beta")

            # (2) ESS(t)
            ax = fig.add_subplot(gs[1, 0])
            ax.plot(it, ess, marker="o", linewidth=1)
            ax.axhline(n_active, linestyle="--", linewidth=1)
            ax.set_title("2) ESS(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("ESS")

            # (2b) ESS/N_active
            ax = fig.add_subplot(gs[1, 1])
            ax.plot(it, ess_ratio, marker="o", linewidth=1)
            ax.axhline(1.0, linestyle="--", linewidth=1)
            ax.set_title("ESS / N_active")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("ESS/N")

            # (5) logZ(t)
            ax = fig.add_subplot(gs[2, 0])
            ax.plot(it, logz, marker="o", linewidth=1)
            ax.set_title("3) logZ(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("logZ")

            # (5b) ΔlogZ(t)
            ax = fig.add_subplot(gs[2, 1])
            ax.plot(it, dlogz, marker="o", linewidth=1)
            ax.set_title("Δ logZ(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("Δ logZ")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ==========================
            # PAGE 2: accept/steps, sigma, logl
            # ==========================
            fig, gs = _page()
            fig.suptitle("SMC Diagnostics: Page 2/2", fontsize=14)

            # (3) accept(t)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(it, accept, marker="o", linewidth=1)
            ax.set_title("4) accept(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("accept")
            ax.set_ylim(0.0, 1.0)

            # (3b) steps(t)
            ax = fig.add_subplot(gs[0, 1])
            ax.plot(it, steps, marker="o", linewidth=1)
            ax.set_title("steps(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("steps used")

            # (4) sigma(t)
            ax = fig.add_subplot(gs[1, 0])
            ax.plot(it_m, sigma, marker="o", linewidth=1)
            ax.set_title("5) sigma(t) (reconstructed, beta>0)")
            ax.set_xlabel("SMC iteration (beta>0)")
            ax.set_ylabel("sigma")

            # (4b) accept vs sigma scatter
            ax = fig.add_subplot(gs[1, 1])
            ax.scatter(sigma, accept_m, s=16, alpha=0.7)
            ax.set_title("accept vs sigma")
            ax.set_xlabel("sigma")
            ax.set_ylabel("accept")
            ax.set_ylim(0.0, 1.0)

            # (6) logl quantiles(t)
            ax = fig.add_subplot(gs[2, 0])
            ax.plot(it, q50, linewidth=1.5, label="median")
            ax.fill_between(it, qL, qH, alpha=0.3, label="10–90%")
            ax.set_title("6) logl quantiles(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("log-likelihood")
            ax.legend(fontsize=9)

            # (6b) final logl histogram
            ax = fig.add_subplot(gs[2, 1])
            ax.hist(logl_hist[-1], bins=25)
            ax.set_title("final logl histogram")
            ax.set_xlabel("log-likelihood")
            ax.set_ylabel("count")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved two-page diagnostics to {save_path}")








    #-----------------------------------------------------------------------------
    # 3.3. SAMPLE STATISTICS
    #-----------------------------------------------------------------------------
    def save_samples_json(self):
        # output directory 
        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        # get samples once
        true_np, mcmc_np = self.get_true_and_mcmc_samples()

        # save generated samples
        mcmc_path = os.path.join(outdir, "mcmc_samples.json")
        with open(mcmc_path, "w", encoding="utf-8") as f:
            json.dump(mcmc_np.tolist(), f)
        print(f"MCMC samples saved to {mcmc_path}")

        # save true samples
        true_path = os.path.join(outdir, "true_samples.json")
        with open(true_path, "w", encoding="utf-8") as f:
            json.dump(true_np.tolist(), f)
        print(f"True samples saved to {true_path}")




    def compute_and_save_sample_statistics(self):
        """
        Computes and saves per-dimension statistics for:
        - MCMC production samples
        - true samples
        Saves: sample_statistics.txt in self.params["outdir"]
        """

        # get samples 
        true_samples, mcmc_samples = self.get_true_and_mcmc_samples()

        # MCMC stats
        self.pm = mcmc_samples.mean(axis=0)
        self.pv = mcmc_samples.var(axis=0)
        self.ps = mcmc_samples.std(axis=0)

        # True stats
        self.qm = true_samples.mean(axis=0)
        self.qv = true_samples.var(axis=0)
        self.qs = true_samples.std(axis=0)

        # store arrays 
        self.mcmc_samples = mcmc_samples
        self.true_samples_np = true_samples

        np.set_printoptions(precision=4, suppress=True)

        stats_str = (
            "pm (mean of MCMC samples):\n" + str(self.pm) +
            "\n\npv (variance of MCMC samples):\n" + str(self.pv) +
            "\n\nps (std dev of MCMC samples):\n" + str(self.ps) +
            "\n\nqm (mean of true samples):\n" + str(self.qm) +
            "\n\nqv (variance of true samples):\n" + str(self.qv) +
            "\n\nqs (std dev of true samples):\n" + str(self.qs) + "\n"
        )

        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        stats_path = os.path.join(outdir, "sample_statistics.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(stats_str)

        print(f"Sample statistics saved to {stats_path}")



    #-----------------------------------------------------------------------------
    # 3.4. KL DIVERGENCE
    #-----------------------------------------------------------------------------
    import numpy as np, warnings, os
    from typing import Tuple


    @staticmethod
    def gau_kl(pm: np.ndarray, pv: np.ndarray,
               qm: np.ndarray, qv: np.ndarray) -> float:
        """
        Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Also computes KL divergence from a single Gaussian pm,pv to a set
         of Gaussians qm,qv.
        Diagonal covariances are assumed.  Divergence is expressed in nats.
        """
        if (len(qm.shape) == 2):
            axis = 1
        else:
            axis = 0
        # Determinants of diagonal covariances pv, qv
        dpv = pv.prod()
        dqv = qv.prod(axis)
        # Inverse of diagonal covariance qv
        iqv = 1. / qv
        # Difference between means pm, qm
        diff = qm - pm
        return (0.5 * (
            np.log(dqv / dpv)                 # log |\Sigma_q| / |\Sigma_p|
            + (iqv * pv).sum(axis)            # + tr(\Sigma_q^{-1} * \Sigma_p)
            + (diff * iqv * diff).sum(axis)   # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
            - len(pm)                         # - N
        ))
    

    def kl_metrics(
        self,
        outdir: str | None = None,
        filename: str = "kl_metrics.txt",
    ) -> None:
        import os
        import numpy as np

        # define outdir
        outdir = (
            outdir
            or (getattr(self, "params", {}) or {}).get("outdir", None)
            or getattr(self, "outdir", None)
        )
        if outdir is None:
            raise ValueError("No output directory specified (pass outdir=... or set params['outdir']).")
        os.makedirs(outdir, exist_ok=True)

        
        true_np, mcmc_np = self.get_true_and_mcmc_samples() 

        # Parametric Gaussian stats (diagonal covariance assumed)
        pm = mcmc_np.mean(axis=0)
        pv = mcmc_np.var(axis=0)
        qm = true_np.mean(axis=0)
        qv = true_np.var(axis=0)

        kl_val = self.gau_kl(pm, pv, qm, qv)  # scalar for 1D qm/qv

        out_path = os.path.join(outdir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            if np.isscalar(kl_val):
                f.write(f"Parametric KL (Gaussian): {float(kl_val):.8f}\n")
            else:
                kl_arr = np.asarray(kl_val).ravel()
                f.write("Parametric KL (Gaussian):\n")
                for i, v in enumerate(kl_arr):
                    f.write(f"  [{i}] {float(v):.8f}\n")

        print(f"KL metrics saved to {out_path}")





import sys
sys.argv = [

    # where to save
    "notebook",
    "--experiment-type", "gaussian",
    "--outdir", "/home/obevza/jaxpsmc/numerical_experiments/gaussian_10",

    # parameters of the experiments
    "--n-dims", "14",
    "--nr-of-samples", "10000",
    "--nr-of-components", "4",
    "--width-mean", "10.0",
    "--width-cov", "1.3",
    "--weights-of-components", "0.25", "0.25", "0.25", "0.25", 

    # define bounds
    "--prior-low", "-30.0",
    "--prior-high", "30.0",

    # define number of particles
    "--n-effective", "10000",
    "--n-active", "10000",
    "--n-prior", "280000",

    # define steps
    "--n-total", "10000",
    "--pc-n-steps", "550",
    "--pc-n-max-steps", "850",
    "--keep-max", "30000",
    "--random-state", "0",

    # define metrics
    "--metric", "ess",
    "--resample", "mult",
    "--transform", "probit",
    "--use-identity-flow",
]






def main():
    args = parser.parse_args()
    runner = SequentialMCExperimentRunner(args)
    runner.run_experiment()
    runner.plot_true_vs_mcmc_corner()
    runner.plot_acceptance_rate()
    runner.save_samples_json()
    runner.compute_and_save_sample_statistics()
    runner.kl_metrics()
    runner.plot_sigma()
    runner.plot_top6_diagnostics_a4_2pages()


main()
