from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import checkify


Array = jax.Array




def assert_array_ndim(x: Array, ndim: int, *, name: str = "x") -> Array:
    # Make ndim usable even if JAX scalar is being passed.
    ndim_arr = jnp.asarray(ndim, dtype=jnp.int32)
    x_ndim_arr = jnp.asarray(x.ndim, dtype=jnp.int32)  # x.ndim is static -> constant scalar array

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
    # shape comparison is static -> boolean constant -> check if ok under tracing
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



# These return booleans / masks and fully: jittable and vmappable.

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

    lo = jnp.asarray(left_open)   # becomes a JAX bool tracer  CHECK IT 
    ro = jnp.asarray(right_open)

    return jnp.where(
        lo,
        jnp.where(ro, open_, lo_only),
        jnp.where(ro, ro_only, closed),
    )


# ---------------------------------------------------
# Runtime assertions that work under jit via checkify
# ---------------------------------------------------
# NOTE: checkify.check is working when you use checkify.checkify(...)
# the whole code works with jit/vmap/grad

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

    # put runtime values in the error message (arrays/scalars)
    xmin = jnp.min(x)
    xmax = jnp.max(x)

    
    checkify.check(ok, f"{name} has values outside the required interval. min={{}} max={{}}", xmin, xmax)
    return x


# -------------------------
#  jit + checks
# -------------------------
def jit_with_checks(
    fn,
    *,
    errors: Any = (checkify.user_checks),
    static_argnames: Tuple[str, ...] = (),
):
    """
    Returns a jitted version of `fn` that supports checkify.

    Use:
        checked = jit_with_checks(my_fn)
        out = checked(...)
    """
    checked_fn = checkify.checkify(fn, errors=errors)
    jitted = jax.jit(checked_fn, static_argnames=static_argnames)

    def wrapped(*args, **kwargs):
        err, out = jitted(*args, **kwargs)
        err.throw()   # raises ValueError if any check fails
        return out

    return wrapped