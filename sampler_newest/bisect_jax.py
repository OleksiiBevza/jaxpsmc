from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
jax.config.update("jax_enable_x64", True)

_ECONVERGED  = jnp.int32(0)
_ESIGNERR    = jnp.int32(-1)
_ECONVERR    = jnp.int32(-2)
_EVALUEERR   = jnp.int32(-3)

def _bisect_impl(f, a, b, *, xtol, rtol, maxiter, args):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    dtype = jnp.result_type(a, b, jnp.asarray(xtol), jnp.asarray(rtol))
    a = a.astype(dtype)
    b = b.astype(dtype)
    xtol = jnp.asarray(xtol, dtype=dtype)
    rtol = jnp.asarray(rtol, dtype=dtype)
    maxiter = jnp.asarray(maxiter, dtype=jnp.int32)

    # Early value error condition 
    bad_maxiter = maxiter < 0

    fa = f(a, *args)
    fb = f(b, *args)
    funcalls0 = jnp.int32(2)
    it0 = jnp.int32(0)

    a_is_root = (fa == jnp.asarray(0, dtype=dtype))
    b_is_root = (fb == jnp.asarray(0, dtype=dtype))
    converged0 = a_is_root | b_is_root

    any_nan0 = jnp.isnan(fa) | jnp.isnan(fb)
    bracketed0 = (jnp.sign(fa) != jnp.sign(fb))

    x0 = jnp.asarray(0.5, dtype=dtype) * (a + b)
    x0 = jnp.where(a_is_root, a, jnp.where(b_is_root, b, x0))

    # only iterate if everything is okay for bisection
    need_loop0 = (~bad_maxiter) & (~any_nan0) & (~converged0) & bracketed0 & (maxiter > 0)
    nan_seen0 = any_nan0

    left, right = a, b
    fleft, fright = fa, fb


    def cond(state):
        left, right, fleft, fright, x, it, funcalls, converged, nan_seen, need_loop = state
        return need_loop & (~converged) & (~nan_seen) & (it < maxiter)


    def body(state):
        left, right, fleft, fright, x, it, funcalls, converged, nan_seen, need_loop = state

        it = it + jnp.int32(1)
        x = jnp.asarray(0.5, dtype=dtype) * (left + right)
        fx = f(x, *args)
        funcalls = funcalls + jnp.int32(1)

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

    # Status
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

    # Root becomes NaN on any failure status
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

    # check:  per-arg whether to map over axis 0 (arrays) or keep constant (scalars)
    def _axis_for(arg):
        arg = jnp.asarray(arg)
        return 0 if arg.ndim > 0 else None

    args_axes = tuple(_axis_for(arg) for arg in args)

    def solve_one(ai, bi, *args_i):
        return bisect_jax(f, ai, bi, args=args_i, **kwargs)

    in_axes = (0, 0) + args_axes
    return jax.vmap(solve_one, in_axes=in_axes)(a, b, *args)