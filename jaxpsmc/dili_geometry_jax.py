from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class DILIPCNGeometry(NamedTuple):
    """
    Geometry for Hessian/GNH-based DILI-pCN.

    center:
        Proposal/reference center in theta-space, shape (D,).

    basis:
        DILI basis Psi_r, shape (D, r), orthonormal columns.

    post_var:
        Empirical posterior covariance eigenvalues inside the LIS,
        shape (r,). These enter the LI-Prior CN time discretization.

    gnh_eigvals:
        Leading eigenvalues of the expected local GNH, shape (r,).

    cov_ref:
        Positive definite reference covariance for diagnostics / DA
        proposal-distance computations, shape (D, D).
    """
    center: Array
    basis: Array
    post_var: Array
    gnh_eigvals: Array
    cov_ref: Array


def _normalize_weights_jax(weights: Array) -> Array:
    weights = jnp.asarray(weights)
    dtype = weights.dtype

    wsum = jnp.sum(weights)
    bad = (
        (wsum <= jnp.asarray(0.0, dtype=dtype))
        | (~jnp.isfinite(wsum))
        | jnp.any(~jnp.isfinite(weights))
        | jnp.any(weights < jnp.asarray(0.0, dtype=dtype))
    )

    n = weights.shape[0]
    w_uniform = jnp.full((n,), jnp.asarray(1.0, dtype=dtype) / jnp.asarray(n, dtype=dtype))
    w_norm = weights / jnp.where(bad, jnp.asarray(1.0, dtype=dtype), wsum)
    return jnp.where(bad, w_uniform, w_norm)


def _symmetrize_jax(mat: Array) -> Array:
    mat = jnp.asarray(mat)
    return jnp.asarray(0.5, dtype=mat.dtype) * (mat + mat.T)


def _project_psd_jax(mat: Array, floor: Array) -> Array:
    """
    Symmetrize and project a matrix to PSD by clipping eigenvalues.

    This is useful because an autodiff Hessian of negative log-likelihood
    can be indefinite away from a local Gaussian/Gauss-Newton regime.
    """
    mat = _symmetrize_jax(mat)
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    eigvals = jnp.maximum(eigvals, floor)
    out = (eigvecs * eigvals[None, :]) @ eigvecs.T
    return _symmetrize_jax(out)


@partial(jax.jit, static_argnames=("local_gnh_fn", "rank"))
def build_dili_pcn_geometry_jax(
    theta: Array,
    weights: Array,
    *,
    local_gnh_fn: Callable[[Array], Array],
    rank: int,
    gnh_floor: float = 1e-10,
    cov_floor: float = 1e-8,
    complement_var: float = 1.0,
) -> DILIPCNGeometry:
    """
    Builds Hessian/GNH-based DILI-pCN geometry from weighted particles.

    Parameters
    ----------
    theta:
        Selected particles in theta-space, shape (K, D).

    weights:
        Non-negative weights for selected particles, shape (K,).

    local_gnh_fn:
        Pure JAX function mapping one theta particle to a local
        GNH / PSD Hessian approximation, shape (D, D).

    rank:
        Static DILI rank r. Must satisfy 1 <= rank <= D.

    Returns
    -------
    DILIPCNGeometry:
        DILI center, basis, LIS posterior variances, GNH eigenvalues,
        and a reference covariance.
    """
    theta = jnp.asarray(theta)
    weights = _normalize_weights_jax(weights)

    dtype = theta.dtype
    k, d = theta.shape

    gnh_floor_arr = jnp.asarray(gnh_floor, dtype=dtype)
    cov_floor_arr = jnp.asarray(cov_floor, dtype=dtype)
    complement_var_arr = jnp.maximum(jnp.asarray(complement_var, dtype=dtype), cov_floor_arr)

    # Weighted center in theta-space.
    center = jnp.sum(theta * weights[:, None], axis=0)

    # Expected local GNH: S = E[H(theta)].
    Hs = jax.vmap(local_gnh_fn, in_axes=0, out_axes=0)(theta)
    Hs = jax.vmap(lambda h: _project_psd_jax(h, gnh_floor_arr), in_axes=0, out_axes=0)(Hs)
    S = jnp.sum(Hs * weights[:, None, None], axis=0)
    S = _project_psd_jax(S, gnh_floor_arr)

    # Leading global LIS basis from expected GNH.
    evals, evecs = jnp.linalg.eigh(S)
    order = jnp.argsort(evals)[::-1]
    evals = jnp.take(evals, order, axis=0)
    evecs = jnp.take(evecs, order, axis=1)

    theta_basis = evecs[:, :rank]          # (D, r)
    gnh_eigvals = evals[:rank]             # (r,)

    # Project particles into the LIS and estimate posterior covariance there.
    z = (theta - center[None, :]) @ theta_basis     # (K, r)
    z_mean = jnp.sum(z * weights[:, None], axis=0)
    zc = z - z_mean[None, :]

    w2 = jnp.sum(weights * weights)
    denom = jnp.maximum(jnp.asarray(1.0, dtype=dtype) - w2, jnp.asarray(1e-12, dtype=dtype))
    Sigma_r = (zc * weights[:, None]).T @ zc / denom
    Sigma_r = _symmetrize_jax(Sigma_r) + cov_floor_arr * jnp.eye(rank, dtype=dtype)

    # Diagonalize projected posterior covariance as in the DILI paper:
    # Psi_r = Theta_r W_r, D_r = eig(Sigma_r).
    cov_evals, W = jnp.linalg.eigh(Sigma_r)
    cov_evals = jnp.maximum(cov_evals, cov_floor_arr)

    # Sort projected covariance directions descending for stable diagnostics.
    cov_order = jnp.argsort(cov_evals)[::-1]
    cov_evals = jnp.take(cov_evals, cov_order, axis=0)
    W = jnp.take(W, cov_order, axis=1)

    basis = theta_basis @ W
    # Re-orthonormalize against numerical drift.
    basis, _ = jnp.linalg.qr(basis, mode="reduced")

    # Reference covariance for diagnostics / proposal distance.
    P = basis @ basis.T
    cov_ref = (
        basis @ (jnp.diag(cov_evals) @ basis.T)
        + complement_var_arr * (jnp.eye(d, dtype=dtype) - P)
        + cov_floor_arr * jnp.eye(d, dtype=dtype)
    )
    cov_ref = _symmetrize_jax(cov_ref)

    return DILIPCNGeometry(
        center=center,
        basis=basis,
        post_var=cov_evals,
        gnh_eigvals=gnh_eigvals,
        cov_ref=cov_ref,
    )