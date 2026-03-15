from typing import Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp

Array = jax.Array


class GaussianMixtureParams(eqx.Module):
    means: Array      # (K, D)
    chols: Array      # (K, D, D)
    log_w: Array      # (K,)
    log_norms: Array  # (K,)


def gmm_init_params(
    means: Array,      # (K, D)
    covs: Array,       # (K, D, D)
    weights: Array,    # (K,)
    *,
    logits: bool = False,
    eps: float = 1e-30,
) -> GaussianMixtureParams:
    means = jnp.asarray(means)
    covs = jnp.asarray(covs)
    weights = jnp.asarray(weights)

    K, D = means.shape

    chols = jax.vmap(jnp.linalg.cholesky)(covs)  # (K, D, D)

    log_dets = 2.0 * jnp.sum(
        jnp.log(jnp.diagonal(chols, axis1=-2, axis2=-1)),
        axis=-1,
    )  # (K,)

    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=means.dtype)
    log_norms = -0.5 * (jnp.asarray(D, dtype=means.dtype) * jnp.log(two_pi) + log_dets)  # (K,)

    log_w = jax.lax.cond(
        jnp.asarray(logits),
        lambda w: jax.nn.log_softmax(w),
        lambda w: jnp.log((w / jnp.sum(w)) + jnp.asarray(eps, dtype=w.dtype)),
        weights,
    )

    return GaussianMixtureParams(means=means, chols=chols, log_w=log_w, log_norms=log_norms)


def gmm_log_prob_single(params: GaussianMixtureParams, x: Array) -> Array:
    # x: (D,)
    diffs = x - params.means  # (K, D)

    def quad_form(diff: Array, L: Array) -> Array:
        y = solve_triangular(L, diff, lower=True)
        return jnp.sum(y * y)

    quad = jax.vmap(quad_form)(diffs, params.chols)     # (K,)
    log_comp = params.log_norms - 0.5 * quad            # (K,)
    return logsumexp(params.log_w + log_comp)           # scalar


def gmm_log_prob(params: GaussianMixtureParams, xs: Array) -> Array:
    # xs: (..., D) -> (...,)
    xs = jnp.asarray(xs)
    D = params.means.shape[1]
    flat = xs.reshape((-1, D))
    flat_lp = jax.vmap(lambda z: gmm_log_prob_single(params, z))(flat)
    return flat_lp.reshape(xs.shape[:-1])




class GaussianMixtureLikelihood(eqx.Module):
    params: GaussianMixtureParams  # from your code

    def __init__(self, means, covs, weights, *, logits: bool = False):
        self.params = gmm_init_params(means=means, covs=covs, weights=weights, logits=logits)

    def loglike_single(self, x_1d: Array) -> Tuple[Array, Array]:
        ll = gmm_log_prob_single(self.params, x_1d)
        return ll, jnp.zeros((0,), dtype=ll.dtype)  # blobs_dim=0