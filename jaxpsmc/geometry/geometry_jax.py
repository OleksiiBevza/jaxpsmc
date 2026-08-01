from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from ..student_jax import fit_mvstud_jax
from ..tools_jax import systematic_resample_jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Geometry:
    """
    Stores fitted geometry parameters for proposal adaptation.

    The class stores two fitted shapes.
    One shape is a normal approximation.
    The other shape is a Student-t approximation.

    These values are used later by mutation kernels.
    The class is also registered as a JAX pytree.
    This lets JAX pass it through jit, scan.

    Parameters:
    -----------
    normal_mean:
        mean vector for the normal fit, shape (D,).
    normal_cov:
        covariance matrix for the normal fit, shape (D, D).
    t_mean:
        mean vector for the Student-t fit, shape (D,).
    t_cov:
        covariance matrix for the Student-t fit, shape (D, D).
    t_nu:
        degrees of freedom for the Student-t fit.

    Returns:
    --------
    Geometry:
        stores normal and Student-t geometry parameters
        in one JAX-compatible object.
    """

    normal_mean: jax.Array  # (D,)
    normal_cov: jax.Array  # (D,D)
    t_mean: jax.Array  # (D,)
    t_cov: jax.Array  # (D,D)
    t_nu: jax.Array  # ()

    def tree_flatten(self):
        """
        Converts the Geometry object into JAX pytree children.

        JAX needs this method to know which fields are arrays.
        The returned children are the values that JAX can transform.

        Parameters:
        -----------
        None:
            this method uses the current Geometry object.

        Returns:
        --------
        tuple:
            tuple containing the array fields and auxiliary data.
            Auxiliary data is None here.
        """
        return (
            self.normal_mean,
            self.normal_cov,
            self.t_mean,
            self.t_cov,
            self.t_nu,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """
        Rebuilds a Geometry object from pytree children.

        JAX uses this after transforming or moving the object.
        The auxiliary value is unused because all fields are arrays.

        Parameters:
        -----------
        aux:
            auxiliary pytree data.
            It is unused here.
        children:
            tuple containing the stored Geometry fields.

        Returns:
        --------
        Geometry:
            rebuilt Geometry object.
        """
        # unpack saved fields
        nm, nc, tm, tc, tnu = children
        # rebuild dataclass
        return cls(nm, nc, tm, tc, tnu)

    @classmethod
    def init(cls, dim: int, *, dtype=jnp.float64):
        """
        Creates an initial Geometry object for a given dimension.

        The means and covariances are initialized to zero.
        The Student-t degrees of freedom are set very large.
        A large degrees of freedom value makes the Student-t close to normal.

        Parameters:
        -----------
        dim:
            dimension of the parameter space.
        dtype:
            numeric dtype used for the stored arrays.

        Returns:
        --------
        Geometry:
            initial Geometry object with zero means,
            zero covariance matrices, and large Student-t degrees of freedom.
        """
        # create zero mean vectors
        z1 = jnp.zeros((dim,), dtype=dtype)
        # create zero covariance matrice
        z2 = jnp.zeros((dim, dim), dtype=dtype)
        # use large value so the initial t distribution is close to normal
        nu = jnp.asarray(1e6, dtype=dtype)
        # return initialized object
        return cls(z1, z2, z1, z2, nu)


@jax.jit
def _cov_unweighted(
    theta: jax.Array, *, jitter: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the unweighted sample mean and covariance.

    Each sample has the same importance.
    The covariance uses the sample covariance denominator N - 1.
    A small diagonal jitter is added for numerical stability.

    Parameters:
    -----------
    theta:
        sample matrix, shape (N, D).
    jitter:
        small value added to the covariance diagonal.

    Returns:
    --------
    Tuple[jax.Array, jax.Array]:
        sample mean, shape (D,),
        and sample covariance, shape (D, D).
    """
    # convert input to JAX array
    theta = jnp.asarray(theta)
    # read sample count and dimension
    n, d = theta.shape
    # compute sample mean
    mu = jnp.mean(theta, axis=0)
    # center samples around mean
    xc = theta - mu[None, :]
    # use n - 1 for sample covariance denominator
    denom = jnp.asarray(n - 1, theta.dtype)
    # avoid division by zero when n <= 1
    cov = (xc.T @ xc) / jnp.where(denom > 0, denom, jnp.asarray(1.0, theta.dtype))
    # force matrix to be symmetric
    cov = 0.5 * (cov + cov.T)
    # add diagonal jitter for stability
    cov = cov + jitter * jnp.eye(d, dtype=theta.dtype)
    return mu, cov


@jax.jit
def _cov_weighted_aweights(
    theta: jax.Array, weights: jax.Array, *, jitter: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the weighted sample mean and covariance.

    Samples with larger weights influence the fit more.
    The weights are normalized before use.
    If the weights are invalid, the function falls back
    to the unweighted mean and covariance.

    Parameters:
    -----------
    theta:
        sample matrix, shape (N, D).
    weights:
        non-negative sample weights, shape (N,).
    jitter:
        small value added to the covariance diagonal.

    Returns:
    --------
    Tuple[jax.Array, jax.Array]:
        weighted mean, shape (D,),
        and weighted covariance, shape (D, D).
    """
    # convert inputs to JAX arrays
    theta = jnp.asarray(theta)
    w = jnp.asarray(weights)
    # read the sample count and dimension
    _n, d = theta.shape
    dtype = theta.dtype

    # validate weights
    wsum = jnp.sum(w)
    bad = (
        (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(w)) | jnp.any(w < 0)
    )

    # normalize weights if they valid
    w = w / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    # calculate weighted mean
    mu = jnp.sum(theta * w[:, None], axis=0)
    # center samples around weighted mean.
    xc = theta - mu[None, :]

    # correction factor for normalized analytical weights.
    # normalization: fact = 1 / (1 - sum(w^2))   because w is normalized to sum=1
    w2sum = jnp.sum(w * w)
    denom = jnp.asarray(1.0, dtype) - w2sum
    fact = jnp.where(
        denom > 0, jnp.asarray(1.0, dtype) / denom, jnp.asarray(0.0, dtype)
    )

    # compute weighted covariance
    cov = (xc * w[:, None]).T @ xc
    cov = cov * fact
    # force matrix to be symmetric
    cov = 0.5 * (cov + cov.T)
    # diagonal jitter gives stability here
    cov = cov + jitter * jnp.eye(d, dtype=dtype)

    # go back to unweighted result if the weights are bad
    mu_u, cov_u = _cov_unweighted(theta, jitter=jitter)
    mu = jnp.where(bad, mu_u, mu)
    cov = jnp.where(bad, cov_u, cov)
    return mu, cov


@partial(jax.jit, static_argnames=("nu_cap",))
def _sanitize_nu(nu: jax.Array, nu_cap: float) -> jax.Array:
    """
    Replaces invalid Student-t degrees of freedom values.

    The Student-t fit may return a non-finite value.
    This function replaces that value with a fixed cap.
    Finite values are kept unchanged.

    Parameters:
    -----------
    nu:
        fitted degrees of freedom value.
    nu_cap:
        fallback value used when nu is not finite.

    Returns:
    --------
    jax.Array:
        finite degrees of freedom value.
    """
    # convert cap to same dtype as nu
    cap = jnp.asarray(nu_cap, dtype=nu.dtype)
    # keep nu if finite, otherwise use cap
    return jnp.where(jnp.isfinite(nu), nu, cap)


@partial(jax.jit, static_argnames=("nu_cap",))
def geometry_fit_jax(
    geom: Geometry,
    theta: jax.Array,  # (N,D)
    weights: jax.Array,  # (N,)
    use_weights: jax.Array,  # bool scalar: if True, use weights logic
    key: jax.Array,  # PRNGKey
    *,
    nu_cap: float = 1e6,
    jitter: float = 1e-9,
):
    """
    Fits normal and Student-t geometry from sample points.

    The normal geometry is fitted from the sample mean and covariance.
    It can use either weighted or unweighted covariance.

    The Student-t geometry is fitted directly when weights are not used.
    When weights are used, the samples are first resampled according
    to the weights. The Student-t fit is then computed on resampled data.

    The input geom is not used in the calculation.
    It is kept in the function signature for interface consistency.

    Parameters:
    -----------
    geom:
        current Geometry object.
        It is not used directly in this fit.
    theta:
        sample matrix, shape (N, D).
    weights:
        sample weights, shape (N,).
        Used only when use_weights is True.
    use_weights:
        Boolean flag.
        If True, weighted normal statistics and resampled Student-t fit are used.
        If False, unweighted fitting is used.
    key:
        JAX random key used for weighted resampling.
    nu_cap:
        fallback value for non-finite Student-t degrees of freedom.
    jitter:
        small value added to covariance diagonals.

    Returns:
    --------
    tuple:
        geom_new:
            fitted Geometry object.
        key_out:
            updated JAX random key.
        resample_status:
            status code from the resampling step.
            If no resampling is used, this is zero.
    """
    # convert to jax
    theta = jnp.asarray(theta)
    weights = jnp.asarray(weights)
    use_weights = jnp.asarray(use_weights, dtype=bool)
    # match jitter dtype to sample array
    jitter_array = jnp.asarray(jitter, dtype=theta.dtype)

    def _do_weighted(_):
        """
        Computes weighted normal mean and covariance.

        Parameters:
        -----------
        _:
            unused input required by lax.cond.

        Returns:
        --------
        Tuple[jax.Array, jax.Array]:
            weighted mean and weighted covariance.
        """
        # use weighted covariance
        return _cov_weighted_aweights(theta, weights, jitter=jitter_array)

    def _do_unweighted(_):
        """
        Computes unweighted normal mean and covariance.

        Parameters:
        -----------
        _:
            unused input required by lax.cond.

        Returns:
        --------
        Tuple[jax.Array, jax.Array]:
            unweighted mean and unweighted covariance.
        """
        # use unweighted covariance
        return _cov_unweighted(theta, jitter=jitter_array)

    # choose weighted or unweighted normal fit
    normal_mean, normal_cov = lax.cond(
        use_weights, _do_weighted, _do_unweighted, operand=None
    )

    # read nr of samples
    n = theta.shape[0]

    def _t_fit_resampled(_):
        """
        Fits Student-t geometry after weighted resampling.

        The weights are normalized.
        Systematic resampling is then used to create an unweighted sample.
        The Student-t distribution is fitted to that resampled data.

        Parameters:
        -----------
        _:
            unused input required by lax.cond.

        Returns:
        --------
        tuple:
            fitted Student-t mean,
            fitted Student-t covariance,
            cleaned degrees of freedom,
            updated random key,
            and resampling status code.
        """
        # normalize weights for resampling
        wsum = jnp.sum(weights)
        bad = (
            (wsum <= 0)
            | (~jnp.isfinite(wsum))
            | jnp.any(~jnp.isfinite(weights))
            | jnp.any(weights < 0)
        )
        w_norm = weights / jnp.where(bad, jnp.asarray(1.0, theta.dtype), wsum)

        # resample indices using systematic resampling
        idx, status, key_out = systematic_resample_jax(w_norm, key=key)
        # keep indices inside valid bounds
        idx_safe = jnp.clip(idx, 0, n - 1)
        # build resampled sample matrix
        theta_rs = theta[idx_safe]
        # fit multivariate Student-t model on resampled data
        t_mean, t_cov, t_nu, _info = fit_mvstud_jax(theta_rs)
        return t_mean, t_cov, _sanitize_nu(t_nu, nu_cap), key_out, status

    def _t_fit_direct(_):
        """
        Fits Student-t geometry without weighted resampling.

        The Student-t distribution is fitted directly to theta.
        The random key is returned unchanged.

        Parameters:
        -----------
        _:
            unused input required by lax.cond.

        Returns:
        --------
        tuple:
            fitted Student-t mean,
            fitted Student-t covariance,
            cleaned degrees of freedom,
            unchanged random key,
            and zero status code.
        """
        # fit multivariate Student-t model directly
        t_mean, t_cov, t_nu, _info = fit_mvstud_jax(theta)
        # use zero to mark that no resampling error occurred
        status = jnp.int64(0)
        return t_mean, t_cov, _sanitize_nu(t_nu, nu_cap), key, status

    # choose direct fit or resampled fit for the Student-t model
    t_mean, t_cov, t_nu, key_out, resample_status = lax.cond(
        use_weights, _t_fit_resampled, _t_fit_direct, operand=None
    )
    # build new Geometry object from the fitted values
    geom_new = Geometry(
        normal_mean=normal_mean,
        normal_cov=normal_cov,
        t_mean=t_mean,
        t_cov=t_cov,
        t_nu=t_nu,
    )
    return geom_new, key_out, resample_status
