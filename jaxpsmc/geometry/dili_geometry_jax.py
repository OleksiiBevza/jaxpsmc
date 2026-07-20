from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class DILIPCNGeometry(NamedTuple):
    """
    Stores all geometry objects needed by the Hessian/GNH-based DILI-pCN kernel.

    DILI-pCN uses a low-dimensional likelihood-informed subspace (LIS).
    Inside this subspace, the proposal uses adapted posterior variances.
    Outside this subspace, the proposal uses a simpler reference variance.

    Fields:
    -------
    center:
        Weighted particle mean in theta-space.
        Shape is (D,).

    basis:
        Orthonormal DILI/LIS basis.
        Shape is (D, r), where r is the selected DILI rank.

    post_var:
        Estimated posterior variances inside the LIS.
        Shape is (r,).

    gnh_eigvals:
        Leading eigenvalues of the expected local GNH matrix.
        Shape is (r,).

    cov_ref:
        Reference covariance matrix used for diagnostics or proposal-distance
        calculations, for example in delayed acceptance.
        Shape is (D, D).
    """

    center: Array
    basis: Array
    post_var: Array
    gnh_eigvals: Array
    cov_ref: Array


def _normalize_weights_jax(weights: Array) -> Array:
    """
    Normalizes particle weights safely.

    If the weights are valid, they are divided by their sum.
    If the weights are invalid, the function returns uniform weights.

    Invalid if:
    * the sum is zero or negative,
    * the sum is not finite,
    * at least one weight is not finite,
    * at least one weight is negative.

    Parameters:
    -----------
    weights:
        Particle weights, shape (K,).

    Returns:
    --------
    Array:
        Normalized weights, shape (K,).
    """
    weights = jnp.asarray(weights)
    dtype = weights.dtype

    # compute total weight
    wsum = jnp.sum(weights)
    # check if weights can be used safely
    bad = (
        (wsum <= jnp.asarray(0.0, dtype=dtype))
        | (~jnp.isfinite(wsum))
        | jnp.any(~jnp.isfinite(weights))
        | jnp.any(weights < jnp.asarray(0.0, dtype=dtype))
    )

    # number of particles
    n = weights.shape[0]
    # fallback weights: every particle gets the same weight
    w_uniform = jnp.full(
        (n,), jnp.asarray(1.0, dtype=dtype) / jnp.asarray(n, dtype=dtype)
    )
    # normalize valid weights. If weights bad, divide by 1 to avoid NaNs
    w_norm = weights / jnp.where(bad, jnp.asarray(1.0, dtype=dtype), wsum)
    # return normalized weights if valid, if not, use uniform weights
    return jnp.where(bad, w_uniform, w_norm)


def _symmetrize_jax(mat: Array) -> Array:
    """
    Makes a square matrix symmetric.
    Replaces the matrix by 0.5 * (mat + mat.T).

    Parameters:
    -----------
    mat:
        Square matrix, shape (D, D).

    Returns:
    --------
    Array:
        Symmetric matrix, shape (D, D).
    """
    mat = jnp.asarray(mat)
    # average matrix with its transpose
    return jnp.asarray(0.5, dtype=mat.dtype) * (mat + mat.T)


def _project_psd_jax(mat: Array, floor: Array) -> Array:
    """
    Projects a matrix to be positive semi-definite.
    The matrix is first symmetrized.
    Then its eigenvalues are clipped from below by floor.

    Parameters:
    -----------
    mat:
        Input square matrix, shape (D, D).

    floor:
        Minimum allowed eigenvalue.

    Returns:
    --------
    Array:
        Symmetric positive semi-definite matrix, shape (D, D).
    """
    # remove numerical asymmetry before eigendecomposition
    mat = _symmetrize_jax(mat)
    # compute eigenvalues and eigenvectors of the symmetric matrix
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    # clip eigenvalues so that the matrix is PSD
    eigvals = jnp.maximum(eigvals, floor)
    # rebuild matrix
    out = (eigvecs * eigvals[None, :]) @ eigvecs.T
    # symmetrize again to remove small errors
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
    Builds the geometry used by the Hessian/GNH-based DILI-pCN proposal.

    The function uses weighted particles to estimate:
    * a center in theta-space,
    * an expected local GNH matrix,
    * a low-dimensional likelihood-informed subspace,
    * posterior variances inside that subspace,
    * a full reference covariance for diagnostics or delayed acceptance.

    Parameters:
    -----------
    theta:
        Selected particles in theta-space.
        Shape is (K, D), where K is the number of particles
        and D is the parameter dimension.

    weights:
        Particle weights, shape (K,).
        The weights are normalized internally.
        If they are invalid, uniform weights are used.

    local_gnh_fn:
        Pure JAX function that maps one particle to a local GNH matrix.
        Input shape is (D,).
        Output shape must be (D, D).

    rank:
        DILI rank r.
        This is static for JIT compilation.
        It should satisfy 1 <= rank <= D.

    gnh_floor:
        Minimum eigenvalue used when projecting GNH matrices to PSD.

    cov_floor:
        Minimum variance/eigenvalue used for covariance stabilization.

    complement_var:
        Reference variance used outside the DILI/LIS subspace.

    Returns:
    --------
    DILIPCNGeometry:
        Geometry object containing the center, basis, posterior variances,
        GNH eigenvalues, and reference covariance.
    """
    theta = jnp.asarray(theta)
    # normalize weights. If invalid, return uniform weights
    weights = _normalize_weights_jax(weights)

    dtype = theta.dtype
    k, d = theta.shape

    gnh_floor_arr = jnp.asarray(gnh_floor, dtype=dtype)
    cov_floor_arr = jnp.asarray(cov_floor, dtype=dtype)
    # check complement variance is not smaller than covariance floor
    complement_var_arr = jnp.maximum(
        jnp.asarray(complement_var, dtype=dtype), cov_floor_arr
    )

    # weighted center in theta-space: point around which the LIS coordinates are centered
    center = jnp.sum(theta * weights[:, None], axis=0)

    # evaluate local GNH matrix at every selected particle
    Hs = jax.vmap(local_gnh_fn, in_axes=0, out_axes=0)(theta)
    # make every local GNH matrix symmetric PSD
    Hs = jax.vmap(lambda h: _project_psd_jax(h, gnh_floor_arr), in_axes=0, out_axes=0)(
        Hs
    )
    # compute weighted expected local GNH matrix, it defines first approximation of the LIS
    S = jnp.sum(Hs * weights[:, None, None], axis=0)
    # stabilize expected GNH matrix
    S = _project_psd_jax(S, gnh_floor_arr)

    # diagonalize the expected GNH matrix
    evals, evecs = jnp.linalg.eigh(S)

    # sort eigenvalues from largest to smallest
    # the largest eigenvalues define the most likelihood-informed directions
    order = jnp.argsort(evals)[::-1]
    evals = jnp.take(evals, order, axis=0)
    evecs = jnp.take(evecs, order, axis=1)

    # take first r eigenvectors as preliminary LIS basis
    theta_basis = evecs[:, :rank]  # (D, r)
    # store corresponding GNH eigenvalues
    gnh_eigvals = evals[:rank]  # (r,)

    # project centered particles into preliminary LIS
    z = (theta - center[None, :]) @ theta_basis  # (K, r)
    # compute weighted mean in LIS coordinates
    z_mean = jnp.sum(z * weights[:, None], axis=0)
    # center projected particles
    zc = z - z_mean[None, :]

    # weighted covariance correction 1 / (1 - sum(w^2))
    w2 = jnp.sum(weights * weights)
    denom = jnp.maximum(
        jnp.asarray(1.0, dtype=dtype) - w2, jnp.asarray(1e-12, dtype=dtype)
    )

    # estimate posterior covariance inside preliminary LIS
    Sigma_r = (zc * weights[:, None]).T @ zc / denom
    # symmetrize and add small diagonal floor for stability
    Sigma_r = _symmetrize_jax(Sigma_r) + cov_floor_arr * jnp.eye(rank, dtype=dtype)

    # diagonalize projected posterior covariance
    # to get final rotated LIS basis and variances
    cov_evals, W = jnp.linalg.eigh(Sigma_r)
    cov_evals = jnp.maximum(cov_evals, cov_floor_arr)

    # sort projected covariance directions from largest variance to smallest
    cov_order = jnp.argsort(cov_evals)[::-1]
    cov_evals = jnp.take(cov_evals, cov_order, axis=0)
    W = jnp.take(W, cov_order, axis=1)

    # rotate preliminary LIS basis by the covariance eigenvectors
    basis = theta_basis @ W
    # re-orthonormalize basis to remove small numerical errors
    basis, _ = jnp.linalg.qr(basis, mode="reduced")

    # build projector onto final LIS
    P = basis @ basis.T

    # build full reference covariance:
    # inside LIS, use estimated posterior variances
    # outside LIS, use complement_var
    cov_ref = (
        basis @ (jnp.diag(cov_evals) @ basis.T)
        + complement_var_arr * (jnp.eye(d, dtype=dtype) - P)
        + cov_floor_arr * jnp.eye(d, dtype=dtype)
    )
    # make final covariance exactly symmetric up to numerical precision
    cov_ref = _symmetrize_jax(cov_ref)

    return DILIPCNGeometry(
        center=center,
        basis=basis,
        post_var=cov_evals,
        gnh_eigvals=gnh_eigvals,
        cov_ref=cov_ref,
    )
