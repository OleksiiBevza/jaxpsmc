from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import digamma
jax.config.update("jax_enable_x64", True)




_ECONVERGED  = jnp.int64(0)
_ESIGNERR    = jnp.int64(-1)
_ECONVERR    = jnp.int64(-2)
_EVALUEERR   = jnp.int64(-3)

def _bisect_impl(f, a, b, *, xtol, rtol, maxiter, args):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    dtype = jnp.result_type(a, b, jnp.asarray(xtol), jnp.asarray(rtol))
    a = a.astype(dtype)
    b = b.astype(dtype)
    xtol = jnp.asarray(xtol, dtype=dtype)
    rtol = jnp.asarray(rtol, dtype=dtype)
    maxiter = jnp.asarray(maxiter, dtype=jnp.int64)

    # value error condition 
    bad_maxiter = maxiter < 0

    fa = f(a, *args)
    fb = f(b, *args)
    funcalls0 = jnp.int64(2)
    it0 = jnp.int64(0)

    a_is_root = (fa == jnp.asarray(0, dtype=dtype))
    b_is_root = (fb == jnp.asarray(0, dtype=dtype))
    converged0 = a_is_root | b_is_root

    any_nan0 = jnp.isnan(fa) | jnp.isnan(fb)
    bracketed0 = (jnp.sign(fa) != jnp.sign(fb))

    x0 = jnp.asarray(0.5, dtype=dtype) * (a + b)
    x0 = jnp.where(a_is_root, a, jnp.where(b_is_root, b, x0))

    # only iterate if numbers are OK for bisection
    need_loop0 = (~bad_maxiter) & (~any_nan0) & (~converged0) & bracketed0 & (maxiter > 0)
    nan_seen0 = any_nan0

    left, right = a, b
    fleft, fright = fa, fb


    def cond(state):
        left, right, fleft, fright, x, it, funcalls, converged, nan_seen, need_loop = state
        return need_loop & (~converged) & (~nan_seen) & (it < maxiter)


    def body(state):
        left, right, fleft, fright, x, it, funcalls, converged, nan_seen, need_loop = state

        it = it + jnp.int64(1)
        x = jnp.asarray(0.5, dtype=dtype) * (left + right)
        fx = f(x, *args)
        funcalls = funcalls + jnp.int64(1)

        fx_nan = jnp.isnan(fx)
        nan_seen = nan_seen | fx_nan
        fx_is_root = (fx == jnp.asarray(0, dtype=dtype))

        same_sign = (jnp.sign(fleft) == jnp.sign(fx))
        go_left = same_sign & (~fx_is_root) & (~fx_nan)

        left2  = jnp.where(go_left, x, left)
        fleft2 = jnp.where(go_left, fx, fleft)
        right2  = jnp.where(go_left, right, x)
        fright2 = jnp.where(go_left, fright, fx)

        width = jnp.abs(right2 - left2)
        tol = xtol + rtol * jnp.abs(x)
        converged = fx_is_root | (width <= tol)

        return (left2, right2, fleft2, fright2, x, it, funcalls, converged, nan_seen, need_loop)

    left, right, fleft, fright, x, it, funcalls, converged, nan_seen, need_loop = lax.while_loop(
        cond,
        body,
        (left, right, fleft, fright, x0, it0, funcalls0, converged0, nan_seen0, need_loop0),
    )

    # status
    status = lax.select(
        bad_maxiter,
        _EVALUEERR,
        lax.select(
            nan_seen,
            _EVALUEERR,
            lax.select(
                converged0 | (need_loop0 & converged),
                _ECONVERGED,
                lax.select((~any_nan0) & (~converged0) & (~bracketed0), _ESIGNERR, _ECONVERR),
            ),
        ),
    )

    # root will be NaN on any failure status
    x = jnp.where(status == _ECONVERGED, x, jnp.nan)
    return x, status, it, funcalls


_bisect_jit = jax.jit(_bisect_impl, static_argnames=("f",))




def bisect_jax(
    f, a, b, *,
    xtol=2e-12,
    rtol=4 * jnp.finfo(jnp.float64).eps,
    maxiter=100,
    args=(),
):
    return _bisect_jit(f, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter, args=args)




def bisect_jax_batch(f, a, b, *, args=(), **kwargs):
    a = jnp.asarray(a)
    b = jnp.asarray(b)

    # should i  per-arg whether to map over axis 0 (arrays) or keep constant (scalars)
    def _axis_for(arg):
        arg = jnp.asarray(arg)
        return 0 if arg.ndim > 0 else None

    args_axes = tuple(_axis_for(arg) for arg in args)

    def solve_one(ai, bi, *args_i):
        return bisect_jax(f, ai, bi, args=args_i, **kwargs)

    in_axes = (0, 0) + args_axes
    return jax.vmap(solve_one, in_axes=in_axes)(a, b, *args)


















# bisect_jax(f, a, b, *, xtol=..., rtol=..., maxiter=..., args=()) -> (root, status, iters, funcalls)

# ----------------------------
# ν update: root-finding target
# ----------------------------
def _nu_fixed_point_objective(nu: jnp.ndarray, delta: jnp.ndarray, dim: jnp.ndarray) -> jnp.ndarray:
    """
    Scalar objective in nu whose root is solved by bisection.

    Args:
      nu: scalar > 0
      delta: shape (n,)
      dim: scalar (same dtype as nu)
    """
    w = (nu + dim) / (nu + delta)  # shape (n,)
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
    Returns (nu_new, status, nu_is_inf).

    status:
      0  -> ok (bisection converged OR set nu=inf via the "large nu" test)
      <0 -> bisect_jax error codes (e.g. sign error, convergence error, etc.)
    """
    dtype = delta.dtype
    dim_f = jnp.asarray(dim, dtype=dtype)

    a = jnp.asarray(1e-300, dtype=dtype)
    b = jnp.asarray(1e300, dtype=dtype)

    f_large = _nu_fixed_point_objective(b, delta, dim_f)

    def _set_inf(_: Any):
        return (b, jnp.int64(0), jnp.bool_(True))
        #return (jnp.asarray(jnp.inf, dtype=dtype), jnp.int32(0), jnp.bool_(True))

    # jnp.inf, dtype=dtype
    def _do_bisect(_: Any):
        root, status, _, _ = bisect_jax(
            _nu_fixed_point_objective,
            a,
            b,
            xtol=xtol,
            maxiter=bisect_maxiter,
            args=(delta, dim_f),
        )
        # if bisect fails, keep the previous nu (pure + stable) and propagate status
        nu_new = jnp.where(status == 0, root, nu_old)
        return (nu_new, status, jnp.bool_(False))

    # match if f(1e300) >= 0 => nu = inf else bisect.
    return lax.cond(f_large >= 0, _set_inf, _do_bisect, operand=None)


# -----------------------------------------------------------------------
# Initialization helpers
# -----------------------------------------------------------------------
def _init_mu_sigma(data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    data: (n, dim)
    Returns:
      mu:    (dim,)
      Sigma: (dim, dim)
    """
    n, dim = data.shape
    mu = jnp.median(data, axis=0)  

    centered = data - jnp.mean(data, axis=0, keepdims=True)
    n_f = jnp.asarray(n, dtype=data.dtype)

    # equivalent to: cov*(n-1)/n + (1/n)*diag(var)
    # dso not rely on jnp.cov behavior
    cov_mle = (centered.T @ centered) / n_f
    var = jnp.var(data, axis=0)  # ddof=0
    Sigma = cov_mle + jnp.diag(var) / n_f

    # 
    #Sigma = 0.5 * (Sigma + Sigma.T)
    return mu, Sigma


# ---------------------------------------------------
# EM (Expectation-Maximization) core
# ---------------------------------------------------
@jax.jit
def _fit_mvstud_core(
    data: jnp.ndarray,              # (n, dim)
    tol: jnp.ndarray,               # scalar float
    max_iter: jnp.ndarray,          # scalar int32
    nu_init: jnp.ndarray,           # scalar float
    xtol: jnp.ndarray,              # scalar float
    bisect_maxiter: jnp.ndarray,    # scalar int32
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.int64, jnp.int64]:
    """
    Returns:
      mu (dim,), Sigma (dim,dim), nu (scalar), iters (int32), status (int32)

    status:
      0  -> converged (|nu - last_nu| <= tol)
      1  -> max_iter reached (no convergence)
      2  -> nu set to +inf (early-stop condition which is to be specified)
      -k -> propagated bisect_jax error codes 
    """
    n, dim = data.shape
    dtype = data.dtype

    mu0, Sigma0 = _init_mu_sigma(data)
    nu0 = nu_init.astype(dtype)
    last_nu0 = jnp.asarray(0.0, dtype=dtype)

    i0 = jnp.int64(0)
    stop0 = jnp.bool_(False)
    status0 = jnp.int64(0)

    def cond_fun(state):
        mu, Sigma, nu, last_nu, i, stop, status = state
        not_done = jnp.logical_and(i < max_iter, jnp.logical_not(stop))
        not_converged = jnp.abs(nu - last_nu) > tol
        return jnp.logical_and(not_done, not_converged)

    def body_fun(state):
        mu, Sigma, nu, last_nu, i, stop, status = state

        diffs = data - mu[None, :]                    # (n, dim)
        sol = jnp.linalg.solve(Sigma, diffs.T)        # (dim, n)
        delta = jnp.sum(diffs.T * sol, axis=0)        # (n,)

        nu_old = nu
        nu_new, nu_bisect_status, nu_is_inf = _opt_nu_bisect(
            delta, dim, nu_old, xtol=xtol, bisect_maxiter=bisect_maxiter
        )

        # if bisection errored, stop and propagate error code.
        bisect_error = (nu_bisect_status != 0)

        # compute w using nu_new
        dim_f = jnp.asarray(dim, dtype=dtype)
        w = (nu_new + dim_f) / (nu_new + delta)       # (n,)

        def _keep_params(_: Any):
            return (mu, Sigma)

        def _update_params(_: Any):
            w_sum = jnp.sum(w)
            mu_upd = jnp.sum(w[:, None] * data, axis=0) / w_sum
            diffs2 = data - mu_upd[None, :]
            Sigma_upd = (diffs2.T * w[None, :]) @ diffs2 / jnp.asarray(n, dtype=dtype)
            Sigma_upd = 0.5 * (Sigma_upd + Sigma_upd.T)
            return (mu_upd, Sigma_upd)

        # match original behavior: if nu becomes inf, return *current* mu/Sigma (don’t update them).
        mu_new2, Sigma_new2 = lax.cond(nu_is_inf, _keep_params, _update_params, operand=None)

        # update stop + status
        stop2 = jnp.logical_or(stop, jnp.logical_or(nu_is_inf, bisect_error))
        status2 = lax.cond(
            status != 0,
            lambda _: status,                                   # already has an error code
            lambda _: lax.cond(
                bisect_error,
                lambda __: nu_bisect_status,                    # negative error code
                lambda __: lax.cond(nu_is_inf, lambda ___: jnp.int64(2), lambda ___: jnp.int64(0), None),
                None,
            ),
            operand=None,
        )

        return (mu_new2, Sigma_new2, nu_new, nu_old, i + jnp.int64(1), stop2, status2)

    mu, Sigma, nu, last_nu, iters, stop, status = lax.while_loop(
        cond_fun, body_fun, (mu0, Sigma0, nu0, last_nu0, i0, stop0, status0)
    )

    # decide if convergence in max_iter.
    converged = jnp.abs(nu - last_nu) <= tol
    status = lax.cond(
        status != 0,
        lambda _: status,
        lambda _: lax.cond(converged, lambda __: jnp.int64(0), lambda __: jnp.int64(1), None),
        operand=None,
    )

    return mu, Sigma, nu, iters, status


# ----------------------------
# wrapper
# ----------------------------
def fit_mvstud_jax(
    data,
    tolerance: float = 1e-6,
    max_iter: int = 100,
    nu_init: float = 20.0,
    xtol: float = 2e-12,
    bisect_maxiter: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """
    EM fit

    Args:
      data: array-like, shape (n, dim)
    Returns:
      mu: (dim,)
      Sigma: (dim, dim)
      nu: scalar
      info: dict with iters and status code
    """
    data = jnp.asarray(data)
    tol = jnp.asarray(tolerance, dtype=data.dtype)
    max_iter_j = jnp.asarray(max_iter, dtype=jnp.int64)
    nu_init_j = jnp.asarray(nu_init, dtype=data.dtype)
    xtol_j = jnp.asarray(xtol, dtype=data.dtype)
    bisect_maxiter_j = jnp.asarray(bisect_maxiter, dtype=jnp.int64)

    mu, Sigma, nu, iters, status = _fit_mvstud_core(
        data, tol, max_iter_j, nu_init_j, xtol_j, bisect_maxiter_j
    )

    info = {"iters": iters, "status": status}
    return mu, Sigma, nu, info