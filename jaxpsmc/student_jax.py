# ruff: noqa: E402
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import digamma

jax.config.update("jax_enable_x64", True)
from .bisect_jax import bisect_jax


#####################################################################
# Nu UPDATE-HELPER
#####################################################################


def _nu_fixed_point_objective(
    nu: jnp.ndarray, delta: jnp.ndarray, dim: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the fixed-point objective used to update Student-t nu.

    The parameter nu is the degrees of freedom of the Student-t model.
    Smaller nu gives heavier tails.
    Larger nu makes the Student-t closer to a normal distribution.

    This function returns the scalar objective whose root defines
    the next nu value in the EM update.

    Parameters:
    -----------
    nu:
        current degrees-of-freedom value.
    delta:
        Mahalanobis squared distances for all data points, shape (n,).
        These distances measure how far each point is from the current mean.
    dim:
        data dimension as a scalar.

    Returns:
    --------
    jnp.ndarray:
        scalar objective value.
        A root of this objective is used as the updated nu.
    """
    # build Student-t weights for current nu
    w = (nu + dim) / (nu + delta)  # shape (n,)

    # return fixed point objective value
    return (
        -digamma(nu / 2)
        + jnp.log(nu / 2)
        + jnp.mean(jnp.log(w))
        - jnp.mean(w)
        + 1.0
        + digamma((nu + dim) / 2)
        - jnp.log((nu + dim) / 2)
    )


def _opt_nu_bisect(
    delta: jnp.ndarray,
    dim: int,
    nu_old: jnp.ndarray,
    *,
    xtol: jnp.ndarray,
    bisect_maxiter: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.int64, jnp.bool_]:
    """
    Updates Student-t nu by solving a scalar equation with bisection.

    The update searches for a root of the fixed-point objective.
    If the objective suggests an effectively infinite nu,
    the function returns a very large value and marks nu as infinite.

    If bisection fails, the previous nu value is kept.
    The status code reports whether the update succeeded.

    Parameters:
    -----------
    delta:
        Mahalanobis squared distances for all data points, shape (n,).
    dim:
        data dimension.
    nu_old:
        previous degrees-of-freedom value.
        Used as a fallback if bisection fails.
    xtol:
        absolute stopping tolerance for the bisection solver.
    bisect_maxiter:
        maximum number of bisection iterations.

    Returns:
    --------
    Tuple[jnp.ndarray, jnp.int64, jnp.bool_]:
        nu_new:
            updated degrees-of-freedom value.
        status:
            bisection status code.
            Zero means success.
            Negative values come from the bisection solver.
        nu_is_inf:
            Boolean flag.
            True means nu was treated as effectively infinite.
    """
    dtype = delta.dtype
    dim_f = jnp.asarray(dim, dtype=dtype)

    # use very small and very large positive bracket for nu
    a = jnp.asarray(1e-300, dtype=dtype)
    b = jnp.asarray(1e300, dtype=dtype)

    # check large-nu side first
    f_large = _nu_fixed_point_objective(b, delta, dim_f)

    def _set_inf(_: Any):
        """
        Returns the large-nu shortcut result.

        This branch is used when the upper bracket already indicates
        that nu should be treated as effectively infinite.

        Parameters:
        -----------
        _:
            unused operand required by lax.cond.

        Returns:
        --------
        Tuple[jnp.ndarray, jnp.int64, jnp.bool_]:
            large nu value, success status, and True infinite-nu flag.
        """
        # treat very large upper bound as infinite nu case
        # return (b, jnp.int64(0), jnp.bool_(True))
        return (
            b.astype(dtype),
            jnp.asarray(0, dtype=jnp.int64),
            jnp.bool_(True),
        )

    def _do_bisect(_: Any):
        """
        Runs bisection to solve the nu fixed-point equation.

        If the solve succeeds, the root becomes the new nu.
        If the solve fails, the old nu is kept.

        Parameters:
        -----------
        _:
            unused operand required by lax.cond.

        Returns:
        --------
        Tuple[jnp.ndarray, jnp.int64, jnp.bool_]:
            updated nu, bisection status code, and False infinite-nu flag.
        """
        # solve fixed point equation with bisection
        # root, status, _, _ = bisect_jax(
        #    _nu_fixed_point_objective,
        #    a,
        #    b,
        #    xtol=xtol,
        #    maxiter=bisect_maxiter,
        #    args=(delta, dim_f),
        # )
        # keep the previous nu when bisection fails
        # nu_new = jnp.where(status == 0, root, nu_old)
        # return (nu_new, status, jnp.bool_(False))
        # return (nu_new, status.astype(jnp.int64), jnp.bool_(False))
        root, status, _, _ = bisect_jax(
            _nu_fixed_point_objective,
            a,
            b,
            xtol=xtol,
            maxiter=bisect_maxiter,
            args=(delta, dim_f),
        )

        root = root.astype(dtype)
        nu_new = jnp.where(status == 0, root, nu_old).astype(dtype)

        return (
            nu_new,
            status.astype(jnp.int64),
            jnp.bool_(False),
        )

    # large-nu shortcut when objective is already nonnegative there
    return lax.cond(f_large >= 0, _set_inf, _do_bisect, operand=None)


#####################################################################
# INITIALIZATION HELPER
#####################################################################
def _init_mu_sigma(data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Builds initial mean and covariance estimates for the Student-t fit.

    The initial mean is the coordinate-wise median.
    This is more robust to outliers than the ordinary mean.

    The initial covariance is built from the centered data.
    A small diagonal variance contribution is added through the formula used here.
    This gives a more stable starting matrix for the EM iterations.

    Parameters:
    -----------
    data:
        input data matrix, shape (n, dim).
        n is the number of samples.
        dim is the number of dimensions.

    Returns:
    --------
    Tuple[jnp.ndarray, jnp.ndarray]:
        mu:
            initial location vector, shape (dim,).
        Sigma:
            initial covariance matrix, shape (dim, dim).
    """
    # sample count and dimension
    n, dim = data.shape
    # coordinate wise median as initial location
    mu = jnp.median(data, axis=0)

    # stable covariance like starting matrix
    centered = data - jnp.mean(data, axis=0, keepdims=True)
    n_f = jnp.asarray(n, dtype=data.dtype)
    # equivalent to: cov*(n-1)/n + (1/n)*diag(var)
    cov_mle = (centered.T @ centered) / n_f
    var = jnp.var(data, axis=0)  # ddof=0
    Sigma = cov_mle + jnp.diag(var) / n_f

    # Sigma = 0.5 * (Sigma + Sigma.T)
    return mu, Sigma


#####################################################################
# EM (EXPECTATION MAXIMIZATION CORE)
#####################################################################


@jax.jit
def _fit_mvstud_core(
    data: jnp.ndarray,  # (n, dim)
    tol: jnp.ndarray,  # scalar float
    max_iter: jnp.ndarray,  # scalar int32
    nu_init: jnp.ndarray,  # scalar float
    xtol: jnp.ndarray,  # scalar float
    bisect_maxiter: jnp.ndarray,  # scalar int32
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.int64, jnp.int64]:
    """
    Fits a multivariate Student-t distribution using EM updates.

    The fitted distribution has three main parameters.
    The mean controls the center.
    The covariance controls the spread and dependence between dimensions.
    The degrees of freedom nu controls tail heaviness.

    The algorithm alternates between two ideas.
    First, it computes weights from the current Student-t model.
    Points far from the center get smaller weights.
    Second, it updates the mean, covariance, and nu using those weights.

    The loop stops when nu changes by at most tol,
    when max_iter is reached, or when the nu update fails.

    Status codes:
    -------------
    0:
        converged.
    1:
        maximum number of EM iterations was reached.
    2:
        nu was treated as effectively infinite.
    negative value:
        error code propagated from the bisection solver.

    Parameters:
    -----------
    data:
        input data matrix, shape (n, dim).
    tol:
        stopping tolerance for the change in nu.
    max_iter:
        maximum number of EM iterations.
    nu_init:
        initial degrees-of-freedom value.
    xtol:
        absolute stopping tolerance for the bisection step used to update nu.
    bisect_maxiter:
        maximum number of bisection iterations used to update nu.

    Returns:
    --------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.int64, jnp.int64]:
        mu:
            fitted mean vector, shape (dim,).
        Sigma:
            fitted covariance matrix, shape (dim, dim).
        nu:
            fitted degrees-of-freedom value.
        iters:
            number of EM iterations used.
        status:
            status code describing convergence or failure.
    """
    n, dim = data.shape
    dtype = data.dtype

    # initial parameters
    mu0, Sigma0 = _init_mu_sigma(data)
    nu0 = nu_init.astype(dtype)
    last_nu0 = jnp.asarray(0.0, dtype=dtype)

    # initial loop state
    i0 = jnp.int64(0)
    stop0 = jnp.bool_(False)
    status0 = jnp.int64(0)

    def cond_fun(state):
        """
        Checks whether the EM loop should continue.

        The loop continues only when the maximum iteration count
        has not been reached, no stop flag is set, and nu has not converged.

        Parameters:
        -----------
        state:
            tuple containing current parameters, counters, and status flags.

        Returns:
        --------
        jnp.ndarray:
            Boolean scalar.
            True means another EM iteration should run.
        """
        # current loop state
        mu, Sigma, nu, last_nu, i, stop, status = state
        not_done = jnp.logical_and(i < max_iter, jnp.logical_not(stop))
        not_converged = jnp.abs(nu - last_nu) > tol
        return jnp.logical_and(not_done, not_converged)

    def body_fun(state):
        """
        Performs one EM update for the Student-t parameters.

        The update first computes Mahalanobis distances.
        These distances are used to update nu.
        Then Student-t weights are computed.
        Those weights are used to update the mean and covariance.

        Parameters:
        -----------
        state:
            tuple containing current mean, covariance, nu,
            previous nu, iteration count, stop flag, and status.

        Returns:
        --------
        tuple:
            updated EM loop state.
        """
        # current loop state
        mu, Sigma, nu, last_nu, i, stop, status = state

        # compute Mahalanobis distances under current parameters
        diffs = data - mu[None, :]  # (n, dim)
        sol = jnp.linalg.solve(Sigma, diffs.T)  # (dim, n)
        delta = jnp.sum(diffs.T * sol, axis=0)  # (n,)

        # update nu with bisection helper
        nu_old = nu
        nu_new, nu_bisect_status, nu_is_inf = _opt_nu_bisect(
            delta, dim, nu_old, xtol=xtol, bisect_maxiter=bisect_maxiter
        )

        # failures in nu update (if exists)
        bisect_error = nu_bisect_status != 0

        # compute Student-t weights for the updated nu
        dim_f = jnp.asarray(dim, dtype=dtype)
        w = (nu_new + dim_f) / (nu_new + delta)  # (n,)

        def _keep_params(_: Any):
            """
            Keeps the current mean and covariance unchanged.

            This branch is used when nu is treated as effectively infinite.
            In that case, the function stops without applying the weighted
            mean and covariance update.

            Parameters:
            -----------
            _:
                unused operand required by lax.cond.

            Returns:
            --------
            Tuple[jnp.ndarray, jnp.ndarray]:
                current mean and covariance.
            """
            return (mu, Sigma)

        def _update_params(_: Any):
            """
            Updates the mean and covariance using Student-t weights.

            Larger weights give a data point more influence.
            Smaller weights reduce the influence of points that look far
            from the current center.

            Parameters:
            -----------
            _:
                unused operand required by lax.cond.

            Returns:
            --------
            Tuple[jnp.ndarray, jnp.ndarray]:
                updated mean and covariance.
            """
            # weighted mean
            w_sum = jnp.sum(w)
            mu_upd = jnp.sum(w[:, None] * data, axis=0) / w_sum

            # weighted covariance
            diffs2 = data - mu_upd[None, :]
            Sigma_upd = (diffs2.T * w[None, :]) @ diffs2 / jnp.asarray(n, dtype=dtype)
            Sigma_upd = 0.5 * (Sigma_upd + Sigma_upd.T)
            return (mu_upd, Sigma_upd)

        # match original behavior: if nu becomes inf, return *current* mu/Sigma (don’t update them)
        mu_new2, Sigma_new2 = lax.cond(
            nu_is_inf, _keep_params, _update_params, operand=None
        )

        # update stop flag and status code
        stop2 = jnp.logical_or(stop, jnp.logical_or(nu_is_inf, bisect_error))
        status2 = lax.cond(
            status != 0,
            lambda _: status,  # already has an error code
            lambda _: lax.cond(
                bisect_error,
                lambda __: nu_bisect_status,  # negative error code
                lambda __: lax.cond(
                    nu_is_inf, lambda ___: jnp.int64(2), lambda ___: jnp.int64(0), None
                ),
                None,
            ),
            operand=None,
        )

        return (mu_new2, Sigma_new2, nu_new, nu_old, i + jnp.int64(1), stop2, status2)

    # run EM loop
    mu, Sigma, nu, last_nu, iters, stop, status = lax.while_loop(
        cond_fun, body_fun, (mu0, Sigma0, nu0, last_nu0, i0, stop0, status0)
    )

    # decide if convergence is in max_iter
    converged = jnp.abs(nu - last_nu) <= tol
    status = lax.cond(
        status != 0,
        lambda _: status,
        lambda _: lax.cond(
            converged, lambda __: jnp.int64(0), lambda __: jnp.int64(1), None
        ),
        operand=None,
    )

    return mu, Sigma, nu, iters, status


#####################################################################
# WRAPPER
#####################################################################
def fit_mvstud_jax(
    data,
    tolerance: float = 1e-6,
    max_iter: int = 100,
    nu_init: float = 20.0,
    xtol: float = 2e-12,
    bisect_maxiter: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """
    Fits a multivariate Student-t distribution to data.

    This is the public wrapper around the jitted EM core.
    It converts Python inputs into JAX arrays and returns
    the fitted Student-t parameters with a small info dictionary.

    The fitted Student-t model is useful when the data may have outliers
    or heavier tails than a normal distribution.

    Parameters:
    -----------
    data:
        input data matrix, shape (n, dim).
        Each row is one sample.
    tolerance:
        stopping tolerance for the change in nu.
    max_iter:
        maximum number of EM iterations.
    nu_init:
        initial degrees-of-freedom value.
    xtol:
        absolute stopping tolerance for the bisection step used to update nu.
    bisect_maxiter:
        maximum number of bisection iterations used to update nu.

    Returns:
    --------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        mu:
            fitted mean vector, shape (dim,).
        Sigma:
            fitted covariance matrix, shape (dim, dim).
        nu:
            fitted degrees-of-freedom value.
        info:
            dictionary with diagnostic information.
            It contains "iters" and "status".
    """
    data = jnp.asarray(data)
    tol = jnp.asarray(tolerance, dtype=data.dtype)
    max_iter_j = jnp.asarray(max_iter, dtype=jnp.int64)
    nu_init_j = jnp.asarray(nu_init, dtype=data.dtype)
    xtol_j = jnp.asarray(xtol, dtype=data.dtype)
    bisect_maxiter_j = jnp.asarray(bisect_maxiter, dtype=jnp.int64)

    # run EM core
    mu, Sigma, nu, iters, status = _fit_mvstud_core(
        data, tol, max_iter_j, nu_init_j, xtol_j, bisect_maxiter_j
    )

    # info dictionary returned to caller
    info = {"iters": iters, "status": status}
    return mu, Sigma, nu, info
