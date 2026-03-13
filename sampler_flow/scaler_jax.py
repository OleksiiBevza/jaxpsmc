from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import checkify


Array = jax.Array


# ----------------------------------------------------------------
# Static checks
# ----------------------------------------------------------------
# depend only on shape/dtype data which are static under jit. so safe
# and cheap. They raise Python errors immediately when the function is traced.

def assert_array_ndim(x: Array, ndim: int, *, name: str = "x") -> Array:
    # make ndim usable even if it is JAX scalar (still static conceptually)
    ndim_arr = jnp.asarray(ndim, dtype=jnp.int64)
    x_ndim_arr = jnp.asarray(x.ndim, dtype=jnp.int64)  # x.ndim is static -> becomes a constant scalar array

    ok = (x_ndim_arr == ndim_arr)
    checkify.check(ok, f"{name} should have {{}} dimensions, but got {x.ndim}", ndim_arr)
    return x


def assert_array_2d(x: Array, *, name: str = "x") -> Array:
    return assert_array_ndim(x, 2, name=name)


def assert_array_1d(x: Array, *, name: str = "x") -> Array:
    return assert_array_ndim(x, 1, name=name)


def assert_arrays_equal_shape(
    x: Array, y: Array, *, x_name: str = "x", y_name: str = "y"
) -> Tuple[Array, Array]:
    # shape comparison is static -> boolean constant. IS IT OK WHEN TRACING?
    ok = jnp.asarray(x.shape == y.shape)
    checkify.check(ok, f"{x_name} and {y_name} should have equal shape, but got {x.shape} and {y.shape}")
    return x, y


def assert_equal_type(
    x: Array, y: Array, *, x_name: str = "x", y_name: str = "y"
) -> Tuple[Array, Array]:
    ok = jnp.asarray(x.dtype == y.dtype)
    checkify.check(ok, f"{x_name} and {y_name} should have equal dtype, but got {x.dtype} and {y.dtype}")
    return x, y


def assert_array_float(x: Array, *, name: str = "x") -> Array:
    ok = jnp.asarray(jnp.issubdtype(x.dtype, jnp.floating))
    checkify.check(ok, f"{name} should have a floating dtype, but got {x.dtype}")
    return x


# -----------------------------------------------------------------
#  runtime value predicates
# -----------------------------------------------------------------
# return booleans / masks which are fully jittable and  vmappable.

def within_interval_mask(
    x: Array,
    left: Array,
    right: Array,
    *,
    left_open: bool = False,
    right_open: bool = False,
) -> Array:
    left_ = jnp.where(jnp.isnan(left), -jnp.inf, left)
    right_ = jnp.where(jnp.isnan(right),  jnp.inf, right)

    closed = (left_ <= x) & (x <= right_)      # [ , ]
    lo_only = (left_ <  x) & (x <= right_)     # ( , ]
    ro_only = (left_ <= x) & (x <  right_)     # [ , )
    open_  = (left_ <  x) & (x <  right_)      # ( , )

    lo = jnp.asarray(left_open)   # becomes a JAX bool[] tracer under tracing 
    ro = jnp.asarray(right_open)

    return jnp.where(
        lo,
        jnp.where(ro, open_, lo_only),
        jnp.where(ro, ro_only, closed),
    )


# ---------------------------------------------------
# Runtime assertions that work under jit via checkify
# ---------------------------------------------------
# NOTE: checkify.check is working if checkify.checkify(...) used.
# everything is pure :contentReference[oaicite:1]{index=1}

def assert_array_within_interval(
    x: Array,
    left: Array,
    right: Array,
    *,
    left_open: bool = False,
    right_open: bool = False,
    name: str = "x",
) -> Array:
    mask = within_interval_mask(x, left, right, left_open=left_open, right_open=right_open)
    ok = jnp.all(mask)

    # SHOULD I KEEP THIS
    xmin = jnp.min(x)
    xmax = jnp.max(x)

    # 
    checkify.check(ok, f"{name} has values outside the required interval. min={{}} max={{}}", xmin, xmax)
    return x


# -------------------------
# SHOULD I KEEP CHECKS???????????????????????????
# -------------------------
def jit_with_checks(
    fn,
    *,
    errors: Any = (checkify.user_checks),
    static_argnames: Tuple[str, ...] = (),
):
    """
    Returns a jitted version of `fn` that supports checkify BUT IF IT IS NEEDED AT ALL 

    Usage:
        checked = jit_with_checks(my_fn)
        out = checked(...)
    """
    checked_fn = checkify.checkify(fn, errors=errors)
    jitted = jax.jit(checked_fn, static_argnames=static_argnames)

    def wrapped(*args, **kwargs):
        err, out = jitted(*args, **kwargs)
        err.throw()   # raises ValueError if any check failed
        return out

    return wrapped





import jax
import jax.numpy as jnp
from jax.experimental import checkify

Array = jax.Array


_EMPTY_I32 = jnp.zeros((0,), dtype=jnp.int64)
_DEFAULT_BOUNDS = jnp.array([jnp.inf, jnp.inf], dtype=jnp.float64)  # matches bounds


def init_bounds_config_jax(
    n_dim: int,
    bounds: Array = _DEFAULT_BOUNDS,          # allowed shapes: (2,) or (n_dim, 2)
    periodic: Array = _EMPTY_I32,             # indices, shape (k,)
    reflective: Array = _EMPTY_I32,           # indices, shape (m,)
    *,
    transform: str = "probit",                # "logit" or "probit" 
    scale: bool = True,
    diagonal: bool = True,
) -> dict[str, Array]:
    # n_dim must be static under jit. it sets array shapes.
    checkify.check(jnp.asarray(n_dim > 0), "n_dim must be a positive integer.")
    n_dim32 = jnp.asarray(n_dim, dtype=jnp.int64)

    # bounds
    bounds = jnp.asarray(bounds, dtype=jnp.float64)
    bounds = assert_array_float(bounds, name="bounds")

    ok_bounds_shape = jnp.asarray(
        ((bounds.ndim == 1) & (bounds.shape == (2,))) |
        ((bounds.ndim == 2) & (bounds.shape == (n_dim, 2)))
    )
    checkify.check(ok_bounds_shape, "bounds must have shape (2,) or (n_dim, 2).")

    # (2,) -> (n_dim, 2); (n_dim, 2) stays (n_dim, 2)
    bounds = jnp.broadcast_to(bounds, (n_dim, 2))
    low = bounds[:, 0]
    high = bounds[:, 1]
    checkify.check(jnp.all(low <= high), "bounds[:,0] must be <= bounds[:,1] elementwise.")

    # periodic or reflective indices turns into masks 
    periodic = jnp.asarray(periodic, dtype=jnp.int64).reshape((-1,))
    reflective = jnp.asarray(reflective, dtype=jnp.int64).reshape((-1,))

    checkify.check(jnp.all((periodic >= 0) & (periodic < n_dim)), "periodic indices must be in [0, n_dim).")
    checkify.check(jnp.all((reflective >= 0) & (reflective < n_dim)), "reflective indices must be in [0, n_dim).")

    dims = jnp.arange(n_dim, dtype=jnp.int64)
    periodic_mask = jnp.any(dims[:, None] == periodic[None, :], axis=1)
    reflective_mask = jnp.any(dims[:, None] == reflective[None, :], axis=1)

    checkify.check(jnp.all(~(periodic_mask & reflective_mask)),
                   "A dimension cannot be both periodic and reflective.")

    # transform: make it as binary logit=0, probit=1
    is_logit = transform == "logit"     # static
    is_probit = transform == "probit"
    checkify.check(jnp.asarray(is_logit | is_probit), "transform must be 'logit' or 'probit'.")
    transform_id = jnp.asarray(is_probit, dtype=jnp.int64)  # 0=logit, 1=probit

    # keep uninitialized parameters as NaNs
    dtype = bounds.dtype
    nan_vec = jnp.full((n_dim,), jnp.nan, dtype=dtype)
    nan_mat = jnp.full((n_dim, n_dim), jnp.nan, dtype=dtype)
    nan_scalar = jnp.asarray(jnp.nan, dtype=dtype)

    return {
        "ndim": n_dim32,
        "low": low,
        "high": high,
        "periodic_mask": periodic_mask,
        "reflective_mask": reflective_mask,
        "transform_id": transform_id,
        "scale": jnp.asarray(scale),
        "diagonal": jnp.asarray(diagonal),
        "mu": nan_vec,
        "sigma": nan_vec,
        "cov": nan_mat,
        "L": nan_mat,
        "L_inv": nan_mat,
        "log_det_L": nan_scalar,
    }



def masks_jax(low: Array, high: Array) -> dict[str, Array]:
    """  

    low, high: shape (ndim,)
    returns boolean masks, each shape (ndim,)
    """
    low = jnp.asarray(low)
    high = jnp.asarray(high)

    fin_low = jnp.isfinite(low)
    fin_high = jnp.isfinite(high)

    mask_none = (~fin_low) & (~fin_high)
    mask_right = (~fin_low) & (fin_high)
    mask_left = (fin_low) & (~fin_high)
    mask_both = (fin_low) & (fin_high)

    return {
        "mask_left": mask_left,
        "mask_right": mask_right,
        "mask_both": mask_both,
        "mask_none": mask_none,
    }


def _create_masks_jax(n_dim: int, bounds: Array) -> dict[str, Array]:
    # Use init_bounds_config_jax
    cfg = init_bounds_config_jax(n_dim, bounds)
    return masks_jax(cfg["low"], cfg["high"])


def _inverse_none_jax(u: Array, mask_none: Array) -> tuple[Array, Array]:
    """
    
      return u[:, mask_none], np.zeros(u.shape)[:, mask_none]
    """
    u = jnp.asarray(u)
    mask_none = jnp.asarray(mask_none, dtype=bool)

    x = u[:, mask_none]
    log_det_J = jnp.zeros_like(u)[:, mask_none]
    return x, log_det_J



def _forward_none_jax(x: Array, mask_none: Array) -> Array:
    """
    return x[:, mask_none]

    Parameters
    ----------
    x : Array, shape (N, D)
    mask_none : Array, shape (D,), bool

    Returns
    -------
    u : Array, shape (N, K) where K = sum(mask_none)
    """
    x = jnp.asarray(x)
    mask_none = jnp.asarray(mask_none, dtype=bool)

    return x[:, mask_none]



import jax.scipy as jsp

def _inverse_both_jax(
    u: Array,
    low: Array,
    high: Array,
    mask_both: Array,
    transform_id: Array,   # 0 = logit, 1 = probit
) -> tuple[Array, Array]:
    """
    both low and high finite

    Returns:
      x : (N, K) transformed values for bounded dims
      J : (N, K) log-diagonal Jacobian terms
    """
    u = jnp.asarray(u)
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    mask_both = jnp.asarray(mask_both, dtype=bool)
    transform_id = jnp.asarray(transform_id, dtype=jnp.int64)

    # only bounded dims
    u_sel = u[:, mask_both]        # (N, K)
    low_sel = low[mask_both]       # (K,)
    high_sel = high[mask_both]     # (K,)
    span = high_sel - low_sel      # (K,)
    log_span = jnp.log(span)       # (K,)  (invalid bounds will yield nan/-inf)

    def _logit_branch(op):
        u_s, low_s, log_span_s, span_s = op
        p = jax.nn.sigmoid(u_s)
        x = p * span_s + low_s
        J = log_span_s + jnp.log(p) + jnp.log1p(-p)
        return x, J

    def _probit_branch(op):
        u_s, low_s, log_span_s, span_s = op
        p = jsp.special.ndtr(u_s)  # Phi(u)
        x = p * span_s + low_s
        # log phi(u) = -0.5 u^2 - log(sqrt(2*pi))
        J = log_span_s + (-0.5 * u_s**2) - jnp.log(jnp.sqrt(2.0 * jnp.pi))
        return x, J

    x, J = jax.lax.switch(
        transform_id,
        (_logit_branch, _probit_branch),
        (u_sel, low_sel, log_span, span),
    )
    return x, J



def _forward_both_jax(
    x: Array,
    low: Array,
    high: Array,
    mask_both: Array,
    transform_id: Array,   # 0 = logit, 1 = probit
    *,
    eps: float = 1e-13,
) -> Array:
    """
    Parameters
    ----------
    x : Array, shape (N, D)
    low, high : Array, shape (D,)
    mask_both : Array, shape (D,), bool
    transform_id : scalar int array, 0=logit, 1=probit
    eps : float
        assign probabilities to [eps, 1-eps] for numerical stability.

    Returns
    -------
    u : Array, shape (N, K) where K = sum(mask_both)
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    mask_both = jnp.asarray(mask_both, dtype=bool)
    transform_id = jnp.asarray(transform_id, dtype=jnp.int64)

    x_sel = x[:, mask_both]          # (N, K)
    low_sel = low[mask_both]         # (K,)
    high_sel = high[mask_both]       # (K,)
    span = high_sel - low_sel        # (K,)

    p = (x_sel - low_sel) / span
    eps_t = jnp.asarray(eps, dtype=x_sel.dtype)
    p = jnp.clip(p, eps_t, 1.0 - eps_t)

    def _logit_branch(p_in: Array) -> Array:
        # logit(p) = log(p) - log(1-p) 
        return jnp.log(p_in) - jnp.log1p(-p_in)

    def _probit_branch(p_in: Array) -> Array:
        # probit(p) = sqrt(2) * erfinv(2p - 1)
        return jnp.sqrt(jnp.asarray(2.0, dtype=p_in.dtype)) * jsp.special.erfinv(
            2.0 * p_in - 1.0
        )

    u = jax.lax.switch(transform_id, (_logit_branch, _probit_branch), p)
    return u



def _inverse_right_jax(u: Array, high: Array, mask_right: Array) -> tuple[Array, Array]:
    """
    
      return high[mask_right] - exp(u[:, mask_right]), u[:, mask_right]

    Parameters
    ----------
    u : Array, shape (N, D)
    high : Array, shape (D,)
    mask_right : Array, shape (D,), bool
        True if only the upper bound is finite.

    Returns
    -------
    x : Array, shape (N, K)
    J : Array, shape (N, K)   (diagonal log-Jacobian terms, CHECK IF matches original: u[:, mask_right])
    """
    u = jnp.asarray(u)
    high = jnp.asarray(high)
    mask_right = jnp.asarray(mask_right, dtype=bool)

    u_sel = u[:, mask_right]        # (N, K)
    high_sel = high[mask_right]     # (K,)

    x = high_sel - jnp.exp(u_sel)   # (N, K) through  broadcasting ???? 
    J = u_sel                       # 
    return x, J



def _forward_right_jax(x: Array, high: Array, mask_right: Array) -> Array:
    """
    return log(high[mask_right] - x[:, mask_right])

    Parameters
    ----------
    x : Array, shape (N, D)
    high : Array, shape (D,)
    mask_right : Array, shape (D,), bool

    Returns
    -------
    u : Array, shape (N, K) where K = sum(mask_right)
    """
    x = jnp.asarray(x)
    high = jnp.asarray(high)
    mask_right = jnp.asarray(mask_right, dtype=bool)

    x_sel = x[:, mask_right]        # (N, K)
    high_sel = high[mask_right]     # (K,)

    return jnp.log(high_sel - x_sel)



def _inverse_left_jax(u: Array, low: Array, mask_left: Array) -> tuple[Array, Array]:
    """
    p = exp(u[:, mask_left])
    return exp(u[:, mask_left]) + low[mask_left], u[:, mask_left]

    Parameters
    ----------
    u : Array, shape (N, D)
    low : Array, shape (D,)
    mask_left : Array, shape (D,), bool
        True if only the lower bound is finite.

    Returns
    -------
    x : Array, shape (N, K)
    J : Array, shape (N, K)   (matches original: u[:, mask_left])
    """
    u = jnp.asarray(u)
    low = jnp.asarray(low)
    mask_left = jnp.asarray(mask_left, dtype=bool)

    u_sel = u[:, mask_left]      # (N, K)
    low_sel = low[mask_left]     # (K,)

    x = jnp.exp(u_sel) + low_sel
    J = u_sel                    
    return x, J



def _forward_left_jax(x: Array, low: Array, mask_left: Array) -> Array:
    """

    return log(x[:, mask_left] - low[mask_left])

    Parameters
    ----------
    x : Array, shape (N, D)
    low : Array, shape (D,)
    mask_left : Array, shape (D,), bool

    Returns
    -------
    u : Array, shape (N, K) where K = sum(mask_left)
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    mask_left = jnp.asarray(mask_left, dtype=bool)

    x_sel = x[:, mask_left]     # (N, K)
    low_sel = low[mask_left]    # (K,)

    return jnp.log(x_sel - low_sel)




def _inverse_affine_jax(
    u: Array,                 # (N, D)
    mu: Array,                # (D,)
    sigma: Array,             # (D,)  (use it if diagonal=True)
    L: Array,                 # (D, D) (use it if diagonal=False)
    log_det_L: Array,         # scalar (use it if diagonal=False)
    diagonal: Array | bool,   # scalar bool
) -> tuple[Array, Array]:
    """
    inverse affine transform

    math behind:
      - diagonal: x = mu + sigma * u, log_det = sum(log(sigma)) repeated N times
      - full:     x = mu + (L @ u_i) per row i (vectorized), log_det = log_det_L repeated N times

    Returns
    -------
    x : Array, shape (N, D)
    log_det : Array, shape (N,)
    """
    u = jnp.asarray(u)
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)
    L = jnp.asarray(L)
    log_det_L = jnp.asarray(log_det_L)
    diagonal = jnp.asarray(diagonal, dtype=bool)

    n = u.shape[0]
    ones_n = jnp.ones((n,), dtype=jnp.result_type(u, mu, sigma, L, log_det_L))

    def _diag_branch(_):
        x = mu + sigma * u
        log_det = jnp.sum(jnp.log(sigma)) * ones_n
        return x, log_det

    def _full_branch(_):
        # vectorized version of: mu + np.array([L @ ui for ui in u])
        x = mu + (u @ L.T)
        log_det = log_det_L * ones_n
        return x, log_det

    x, log_det = jax.lax.cond(diagonal, _diag_branch, _full_branch, operand=None)
    return x, log_det



def _forward_affine_jax(
    x: Array,          # (N, D)
    mu: Array,         # (D,)
    sigma: Array,      # (D,)    used if diagonal=True
    L_inv: Array,      # (D, D)  used if diagonal=False
    diagonal: Array,   # scalar bool
) -> Array:
    """
    affine transform.

    diagonal=True:
        u = (x - mu) / sigma
    diagonal=False:
        u_i = L_inv @ (x_i - mu)   for each row i, vectorized as (x-mu) @ L_inv.T
    """
    x = jnp.asarray(x)
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)
    L_inv = jnp.asarray(L_inv)
    diagonal = jnp.asarray(diagonal, dtype=bool)

    def _diag_branch(_):
        return (x - mu) / sigma

    def _full_branch(_):
        # vectorized version of: np.array([L_inv @ (xi - mu) for xi in x])
        return (x - mu) @ L_inv.T

    return jax.lax.cond(diagonal, _diag_branch, _full_branch, operand=None)



import jax
import jax.numpy as jnp
import jax.scipy as jsp

_LOG_SQRT_2PI = jnp.log(jnp.sqrt(2.0 * jnp.pi))

def _inverse_jax(
    u: jax.Array,             # (N, D)
    low: jax.Array,           # (D,)
    high: jax.Array,          # (D,)
    mask_none: jax.Array,     # (D,) bool
    mask_left: jax.Array,     # (D,) bool
    mask_right: jax.Array,    # (D,) bool
    mask_both: jax.Array,     # (D,) bool
    transform_id: jax.Array,  # scalar int: 0=logit, 1=probit
) -> tuple[jax.Array, jax.Array]:
    u = jnp.asarray(u)
    low = jnp.asarray(low)
    high = jnp.asarray(high)

    mask_none = jnp.asarray(mask_none, dtype=bool)[None, :]   # (1, D)
    mask_left = jnp.asarray(mask_left, dtype=bool)[None, :]
    mask_right = jnp.asarray(mask_right, dtype=bool)[None, :]
    mask_both = jnp.asarray(mask_both, dtype=bool)[None, :]

    transform_id = jnp.asarray(transform_id, dtype=jnp.int64)
    is_probit = (transform_id == 1)  # scalar bool 

    # bounded branch from 2 sides (computed for all dims, only used where mask_both=True)
    span = jnp.where(mask_both[0], high - low, 1.0)           # (D,) 
    log_span = jnp.log(span)                                  # (D,)

    # logit inverse
    p_sig = jax.nn.sigmoid(u)                                 # (N, D)
    x_logit = low + p_sig * span                              # (N, D)
    J_logit = log_span + jnp.log(p_sig) + jnp.log1p(-p_sig)    # (N, D)

    # probit inverse
    p_phi = jsp.special.ndtr(u)                               # (N, D)
    x_probit = low + p_phi * span                             # (N, D)
    J_probit = log_span + (-0.5 * u * u) - _LOG_SQRT_2PI       # (N, D)

    x_both = jnp.where(is_probit, x_probit, x_logit)           # (N, D)
    J_both = jnp.where(is_probit, J_probit, J_logit)           # (N, D)

    # one-sided branches (computed for all dims, only used where their mask=True)
    exp_u = jnp.exp(u)
    x_left = exp_u + low
    J_left = u

    x_right = high - exp_u
    J_right = u

    # assemble full (N, D) output with static shapes
    x = jnp.zeros_like(u)
    J = jnp.zeros_like(u)

    x = jnp.where(mask_none, u, x)
    x = jnp.where(mask_left, x_left, x)
    x = jnp.where(mask_right, x_right, x)
    x = jnp.where(mask_both, x_both, x)

    # mask_none contributes 0 to J, so we only set left/right/both
    J = jnp.where(mask_left, J_left, J)
    J = jnp.where(mask_right, J_right, J)
    J = jnp.where(mask_both, J_both, J)

    log_det_J = jnp.sum(J, axis=1)  # (N,)
    return x, log_det_J



def _forward_jax(
    x: Array,                  # (N, D)
    low: Array,                # (D,)
    high: Array,               # (D,)
    mask_none: Array,          # (D,) bool
    mask_left: Array,          # (D,) bool
    mask_right: Array,         # (D,) bool
    mask_both: Array,          # (D,) bool
    transform_id: Array,       # scalar int: 0=logit, 1=probit
    *,
    eps: float = 1e-13,
) -> Array:
    """Shape forward bounds transform: returns u with shape (N, D)."""
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    high = jnp.asarray(high)

    mask_none  = jnp.asarray(mask_none,  dtype=bool)[None, :]
    mask_left  = jnp.asarray(mask_left,  dtype=bool)[None, :]
    mask_right = jnp.asarray(mask_right, dtype=bool)[None, :]
    mask_both  = jnp.asarray(mask_both,  dtype=bool)[None, :]

    transform_id = jnp.asarray(transform_id, dtype=jnp.int64)
    is_probit = (transform_id == 1)

    # none: identity
    u_none = x

    # left: log(x - low)
    u_left = jnp.log(x - low)

    # right: log(high - x)
    u_right = jnp.log(high - x)

    # both: p=(x-low)/(high-low) then logit/probit
    span = jnp.where(mask_both[0], high - low, 1.0)       # (D,)
    low_safe = jnp.where(mask_both[0], low, 0.0)          # (D,)
    p = (x - low_safe) / span                              # (N, D)

    eps_t = jnp.asarray(eps, dtype=x.dtype)
    p = jnp.clip(p, eps_t, 1.0 - eps_t)

    u_logit = jnp.log(p) - jnp.log1p(-p)
    u_probit = jnp.sqrt(jnp.asarray(2.0, dtype=x.dtype)) * jsp.special.erfinv(2.0 * p - 1.0)
    u_both = jnp.where(is_probit, u_probit, u_logit)

    # assemble full (N, D)
    u = jnp.zeros_like(x)
    u = jnp.where(mask_none,  u_none,  u)
    u = jnp.where(mask_left,  u_left,  u)
    u = jnp.where(mask_right, u_right, u)
    u = jnp.where(mask_both,  u_both,  u)
    return u


from typing import Mapping

def inverse_jax(u: Array, cfg: Mapping[str, Array], masks: Mapping[str, Array]) -> tuple[Array, Array]:
    """
    inverse transformation.

    Parameters
    ----------
    u : Array
        Shape (N, D) unconstrained input.
    cfg : dictionary pytree of JAX arrays
        Must contain: low, high, transform_id, scale, diagonal, mu, sigma, L, log_det_L
    masks : dict-like pytree of JAX bool arrays
        Must contain: mask_none, mask_left, mask_right, mask_both

    Returns
    -------
    x : Array, shape (N, D)
    log_det_J : Array, shape (N,)
    """
    u = jnp.asarray(u)

    low = cfg["low"]
    high = cfg["high"]
    transform_id = cfg["transform_id"]

    scale = jnp.asarray(cfg["scale"], dtype=bool)
    diagonal = jnp.asarray(cfg["diagonal"], dtype=bool)

    mu = cfg["mu"]
    sigma = cfg["sigma"]
    L = cfg["L"]
    log_det_L = cfg["log_det_L"]

    mask_none = masks["mask_none"]
    mask_left = masks["mask_left"]
    mask_right = masks["mask_right"]
    mask_both = masks["mask_both"]

    def _scaled(u_in: Array) -> tuple[Array, Array]:
        x1, ld1 = _inverse_affine_jax(u_in, mu, sigma, L, log_det_L, diagonal)
        x2, ld2 = _inverse_jax(
            x1, low, high,
            mask_none, mask_left, mask_right, mask_both,
            transform_id,
        )
        return x2, ld1 + ld2

    def _unscaled(u_in: Array) -> tuple[Array, Array]:
        return _inverse_jax(
            u_in, low, high,
            mask_none, mask_left, mask_right, mask_both,
            transform_id,
        )

    x, log_det_J = jax.lax.cond(scale, _scaled, _unscaled, u)
    return x, log_det_J



def forward_jax(
    x: Array,
    cfg: Mapping[str, Array],
    masks: Mapping[str, Array],
    *,
    eps: float = 1e-13,
) -> Array:
    """
    forward transformation:
      u = bounds_forward(x)
      if scale: u = affine_forward(u)
    """
    x = jnp.asarray(x)

    u0 = _forward_jax(
        x,
        cfg["low"], cfg["high"],
        masks["mask_none"], masks["mask_left"], masks["mask_right"], masks["mask_both"],
        cfg["transform_id"],
        eps=eps,
    )

    scale = jnp.asarray(cfg["scale"], dtype=bool)
    diagonal = jnp.asarray(cfg["diagonal"], dtype=bool)

    def _scaled(u_in: Array) -> Array:
        return _forward_affine_jax(u_in, cfg["mu"], cfg["sigma"], cfg["L_inv"], diagonal)

    return jax.lax.cond(scale, _scaled, lambda z: z, u0)



def forward_jax_checked(
    x: Array,
    cfg: Mapping[str, Array],
    masks: Mapping[str, Array],
    *,
    eps: float = 1e-13,
) -> Array:
    """
    with checkify-based bounds check.   SHOULD I KEEP IT?????????? 
    IMPORTANT: call this via jit_with_checks(...) HMMM?? 
    """
    x = assert_array_within_interval(x, cfg["low"], cfg["high"], name="x")
    return forward_jax(x, cfg, masks, eps=eps)




def fit_jax(
    x: Array,
    cfg: Mapping[str, Array],
    masks: Mapping[str, Array],
    *,
    eps: float = 1e-13,
    jitter: float = 0.0,   # set e.g. 1e-6 if you ever get Cholesky issues
) -> dict[str, Array]:
    """
    fit: learns mu and (sigma if diagonal) or (cov/L/L_inv/log_det_L if full).
    Returns new cfg dict (whichis not mutated anywheer)
    """
    x = jnp.asarray(x)

    # (i) forward bounds only (matches original: u = self._forward(x))
    u = _forward_jax(
        x,
        cfg["low"], cfg["high"],
        masks["mask_none"], masks["mask_left"], masks["mask_right"], masks["mask_both"],
        cfg["transform_id"],
        eps=eps,
    )

    mu = jnp.mean(u, axis=0)
    diagonal = jnp.asarray(cfg["diagonal"], dtype=bool)

    D = u.shape[1]
    dtype = u.dtype
    I = jnp.eye(D, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)

    def _diag_branch(_):
        sigma = jnp.std(u, axis=0)   # ddof=0 (matches np.std default)
        cov = I
        L = I
        L_inv = I
        log_det_L = zero
        return sigma, cov, L, L_inv, log_det_L

    def _full_branch(_):
        # np.cov(u.T) equivalent: centered.T @ centered / (N-1)
        n = u.shape[0]
        denom = jnp.asarray(jnp.maximum(n - 1, 1), dtype=dtype)  # avoid divide-by-zero if n==1
        centered = u - mu
        cov = (centered.T @ centered) / denom

        # numerical stabilization (keep jitter=0.0 for exactness)
        cov = cov + jnp.asarray(jitter, dtype=dtype) * I

        L = jnp.linalg.cholesky(cov)

        # L_inv = inv(L)
        L_inv = jsp.linalg.solve_triangular(L, I, lower=True)

        # log(det(L)) for triangular L
        log_det_L = jnp.sum(jnp.log(jnp.diag(L)))

        # original numpy code doesn't define sigma here
        sigma = cfg.get("sigma", jnp.ones((D,), dtype=dtype))
        return sigma, cov, L, L_inv, log_det_L

    sigma, cov, L, L_inv, log_det_L = jax.lax.cond(
        diagonal, _diag_branch, _full_branch, operand=None
    )

    # IMPORTANT: return a dict (don't do dict(cfg).update(...)!)
    cfg_out = dict(cfg)
    cfg_out.update(
        mu=mu,
        sigma=sigma,
        cov=cov,
        L=L,
        L_inv=L_inv,
        log_det_L=log_det_L,
    )
    return cfg_out



def apply_reflective_boundary_conditions_x_jax(
    x: Array,
    low: Array,
    high: Array,
    reflective_mask: Array,
) -> Array:
    """
    boundary conditions for selected dimensions

    For each reflective dimension i with finite bounds [low[i], high[i]],
    values are reflected back into the interval, equivalent to repeatedly applying:
      while x > high: x = 2*high - x
      while x < low:  x = 2*low  - x

    Parameters
    ----------
    x : Array, shape (N, D)
    low, high : Array, shape (D,)
    reflective_mask : Array, shape (D,), bool
        True if dimensions should use reflective boundaries.

    Returns
    -------
    x_ref : Array, shape (N, D)
        Reflected x (unchanged on non-reflective dims).
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    reflective_mask = jnp.asarray(reflective_mask, dtype=bool)

    # quick exit if no reflective dimensions 
    has_reflect = jnp.any(reflective_mask)

    def _do_reflect(x_in: Array) -> Array:
        m = reflective_mask[None, :]  # (1, D)

        # dont touch non-reflective dims while computing (prevents inf/nan propagation)
        x_safe = jnp.where(m, x_in, 0.0)

        low_safe = jnp.where(reflective_mask, low, 0.0)     # (D,)
        span = jnp.where(reflective_mask, high - low, 1.0)  # (D,)

        # keep bounds numerically safe
        tiny = jnp.asarray(jnp.finfo(x_in.dtype).tiny, dtype=x_in.dtype)
        span = jnp.where(span > tiny, span, 1.0)

        period = 2.0 * span  # (D,)

        # fold into [0, 2*span)
        y = jnp.mod(x_safe - low_safe, period)  # (N, D), in [0, period)

        # reflect second half back: [span, 2*span) -> [span, 0]
        y = jnp.where(y > span, period - y, y)

        x_ref = low_safe + y  # (N, D)
        return jnp.where(m, x_ref, x_in)

    return jax.lax.cond(has_reflect, _do_reflect, lambda z: z, x)



def apply_periodic_boundary_conditions_x_jax(
    x: Array,
    low: Array,
    high: Array,
    periodic_mask: Array,
) -> Array:
    """
    periodic boundary conditions for selected dimensions.

    Matches the original Python while-loop:
      while x > high: x = low + x - high   (subtract period)
      while x < low:  x = high + x - low   (add period)

    TO CHECK? 
      - x == high stays high
      - x == low stays low
      - x == low + k*(high-low), k=1,2,... maps to high (like the while-loop code)
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    periodic_mask = jnp.asarray(periodic_mask, dtype=bool)

    has_periodic = jnp.any(periodic_mask)

    def _wrap(x_in: Array) -> Array:
        m = periodic_mask[None, :]  # (1, D)

        # periodic for finite bounds only
        fin = jnp.isfinite(low) & jnp.isfinite(high)
        m = m & fin[None, :]

        # placeholders to avoid inf/nan propagation in non-periodic dims
        x_safe = jnp.where(m, x_in, 0.0)

        low_safe = jnp.where(periodic_mask, low, 0.0)     # (D,)
        high_safe = jnp.where(periodic_mask, high, 1.0)   # (D,)
        span = jnp.where(periodic_mask, high - low, 1.0)  # (D,)

        # protect against degenerate span
        tiny = jnp.asarray(jnp.finfo(x_in.dtype).tiny, dtype=x_in.dtype)
        span = jnp.where(span > tiny, span, 1.0)

        # wrap to [low, high)
        y = jnp.mod(x_safe - low_safe, span)              # in [0, span)
        x_wrap = low_safe + y                             # in [low, high)

        # map positive multiples to 'high' instead of 'low'
        pos = (x_safe - low_safe) > 0                     # excludes x==low, includes x==high and above
        x_wrap = jnp.where((y == 0.0) & pos, high_safe, x_wrap)

        return jnp.where(m, x_wrap, x_in)

    return jax.lax.cond(has_periodic, _wrap, lambda z: z, x)



def apply_boundary_conditions_x_jax(
    x: Array,
    cfg: dict[str, Array],
) -> Array:
    """
    Apply boundary conditions (periodic and/or reflective) to x
    if both: reflective(periodic(x))
    """
    x = jnp.asarray(x)

    low = cfg["low"]
    high = cfg["high"]
    periodic_mask = jnp.asarray(cfg["periodic_mask"], dtype=bool)
    reflective_mask = jnp.asarray(cfg["reflective_mask"], dtype=bool)

    has_periodic = jnp.any(periodic_mask)
    has_reflective = jnp.any(reflective_mask)

    def _apply_periodic(x_in: Array) -> Array:
        return apply_periodic_boundary_conditions_x_jax(x_in, low, high, periodic_mask)

    def _apply_reflective(x_in: Array) -> Array:
        return apply_reflective_boundary_conditions_x_jax(x_in, low, high, reflective_mask)

    # periodic first
    x1 = jax.lax.cond(has_periodic, _apply_periodic, lambda z: z, x)
    # then reflective
    x2 = jax.lax.cond(has_reflective, _apply_reflective, lambda z: z, x1)

    return x2


