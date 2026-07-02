from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import checkify


Array = jax.Array


def assert_array_ndim(x: Array, ndim: int, *, name: str = "x") -> Array:
    """
    Checks whether an array has the expected number of dimensions.

    This function is used for input validation.
    It raises a checkify error if the array dimension count is wrong.
    If the check passes, the original array is returned unchanged.

    Parameters:
    -----------
    x:
        array to check.
    ndim:
        expected number of dimensions.
    name:
        variable name used in the error message.

    Returns:
    --------
    Array:
        same input array if the check passes.
    """
    # convert ndim to JAX scalar so it can be used inside JAX
    ndim_arr = jnp.asarray(ndim, dtype=jnp.int32)
    # x.ndim is static, so convert it to a scalar array for check
    x_ndim_arr = jnp.asarray(x.ndim, dtype=jnp.int32)
    # check whether dimension count matches
    ok = x_ndim_arr == ndim_arr
    checkify.check(
        ok, f"{name} should have {{}} dimensions, but got {x.ndim}", ndim_arr
    )
    return x


def assert_array_2d(x: Array, *, name: str = "x") -> Array:
    """
    Checks whether an array is two-dimensional.

    This is a small wrapper around assert_array_ndim.
    It requires the input to have exactly two dimensions.

    Parameters:
    -----------
    x:
        array to check.
    name:
        variable name used in the error message.

    Returns:
    --------
    Array:
        same input array if the check passes.
    """
    return assert_array_ndim(x, 2, name=name)


def assert_array_1d(x: Array, *, name: str = "x") -> Array:
    """
    Checks whether an array is one-dimensional.

    This is a small wrapper around assert_array_ndim.
    It requires the input to have exactly one dimension.

    Parameters:
    -----------
    x:
        array to check.
    name:
        variable name used in the error message.

    Returns:
    --------
    Array:
        same input array if the check passes.
    """
    return assert_array_ndim(x, 1, name=name)


def assert_arrays_equal_shape(
    x: Array, y: Array, *, x_name: str = "x", y_name: str = "y"
) -> Tuple[Array, Array]:
    """
    Checks whether two arrays have the same shape.

    The function compares the full shape of both arrays.
    If the shapes are different, it raises a checkify error.
    If the check passes, both arrays are returned unchanged.

    Parameters:
    -----------
    x:
        first array to check.
    y:
        second array to check.
    x_name:
        name of the first array used in the error message.
    y_name:
        name of the second array used in the error message.

    Returns:
    --------
    Tuple[Array, Array]:
        same two input arrays if the check passes.
    """
    # shape comparison is static, so convert result to JAX scalar
    ok = jnp.asarray(x.shape == y.shape)
    checkify.check(
        ok,
        f"{x_name} and {y_name} should have equal shape, but got {x.shape} and {y.shape}",
    )
    return x, y


def assert_equal_type(
    x: Array, y: Array, *, x_name: str = "x", y_name: str = "y"
) -> Tuple[Array, Array]:
    """
    Checks whether two arrays have the same dtype.

    The function compares the dtype of both arrays.
    If the dtypes are different, it raises a checkify error.
    If the check passes, both arrays are returned unchanged.

    Parameters:
    -----------
    x:
        first array to check.
    y:
        second array to check.
    x_name:
        name of the first array used in the error message.
    y_name:
        name of the second array used in the error message.

    Returns:
    --------
    Tuple[Array, Array]:
        same two input arrays if the check passes.
    """
    # compare dtypes of both arrays
    ok = jnp.asarray(x.dtype == y.dtype)
    checkify.check(
        ok,
        f"{x_name} and {y_name} should have equal dtype, but got {x.dtype} and {y.dtype}",
    )
    return x, y


def assert_array_float(x: Array, *, name: str = "x") -> Array:
    """
    Checks whether an array has a floating-point dtype.

    This is useful before operations that require real-valued arrays.
    Integer and Boolean arrays fail this check.

    Parameters:
    -----------
    x:
        array to check.
    name:
        variable name used in the error message.

    Returns:
    --------
    Array:
        same input array if the check passes.
    """
    # check if dtype is floating-point type
    ok = jnp.asarray(jnp.issubdtype(x.dtype, jnp.floating))
    checkify.check(ok, f"{name} should have a floating dtype, but got {x.dtype}")
    return x


def within_interval_mask(
    x: Array,
    left: Array,
    right: Array,
    *,
    left_open: bool = False,
    right_open: bool = False,
) -> Array:
    """
    Builds a Boolean mask for values inside an interval.

    The interval can be closed or open on either side.
    A NaN left bound is treated as negative infinity.
    A NaN right bound is treated as positive infinity.

    Parameters:
    -----------
    x:
        values to test.
    left:
        left endpoint of the interval.
    right:
        right endpoint of the interval.
    left_open:
        if True, the left endpoint is excluded.
        If False, the left endpoint is included.
    right_open:
        if True, the right endpoint is excluded.
        If False, the right endpoint is included.

    Returns:
    --------
    Array:
        Boolean mask showing which values are inside the interval.
    """
    # left NaN bound as negative infinity
    left_ = jnp.where(jnp.isnan(left), -jnp.inf, left)
    # right NaN bound as positive infinity
    right_ = jnp.where(jnp.isnan(right), jnp.inf, right)
    # build masks for the four interval types.
    closed = (left_ <= x) & (x <= right_)  # [left, right]
    lo_only = (left_ < x) & (x <= right_)  # (left, right]
    ro_only = (left_ <= x) & (x < right_)  # [left, right)
    open_ = (left_ < x) & (x < right_)  # (left, right)

    # python booleans to JAX booleans
    lo = jnp.asarray(left_open)
    ro = jnp.asarray(right_open)

    # choose correct mask based on open or closed endpoints
    return jnp.where(
        lo,
        jnp.where(ro, open_, lo_only),
        jnp.where(ro, ro_only, closed),
    )


def assert_array_within_interval(
    x: Array,
    left: Array,
    right: Array,
    *,
    left_open: bool = False,
    right_open: bool = False,
    name: str = "x",
) -> Array:
    """
    Checks whether all array values are inside an interval.

    The function builds an interval mask.
    The check passes only if every value is inside the interval.
    If at least one value is outside, it raises a checkify error.

    Parameters:
    -----------
    x:
        array to check.
    left:
        left endpoint of the interval.
    right:
        right endpoint of the interval.
    left_open:
        if True, the left endpoint is excluded.
        If False, the left endpoint is included.
    right_open:
        if True, the right endpoint is excluded.
        If False, the right endpoint is included.
    name:
        variable name used in the error message.

    Returns:
    --------
    Array:
        same input array if the check passes.
    """
    # get boolean mask for all elements
    mask = within_interval_mask(
        x, left, right, left_open=left_open, right_open=right_open
    )
    # check passes only if all elements are inside the interval
    ok = jnp.all(mask)

    # compute min and max for error message
    xmin = jnp.min(x)
    xmax = jnp.max(x)

    # runtime error if at least one value is outside the interval
    checkify.check(
        ok,
        f"{name} has values outside the required interval. min={{}} max={{}}",
        xmin,
        xmax,
    )
    return x


def jit_with_checks(
    fn,
    *,
    errors: Any = (checkify.user_checks),
    static_argnames: Tuple[str, ...] = (),
):
    """
    Creates a jitted function that keeps runtime checks.

    Normal JAX jit can make validation errors harder to handle.
    This wrapper uses checkify to track checks inside jitted code.
    When a check fails, the wrapped function raises the error.

    Parameters:
    -----------
    fn:
        function to wrap.
    errors:
        checkify error categories to enable.
    static_argnames:
        names of arguments that should be treated as static by jit.

    Returns:
    --------
    Callable:
        wrapped function.
        It runs with jit and raises an error if a check fails.
    """

    checked_fn = checkify.checkify(fn, errors=errors)
    jitted = jax.jit(checked_fn, static_argnames=static_argnames)

    def wrapped(*args, **kwargs):
        """
        Runs the checked and jitted function.

        The function first receives checkify output.
        It then throws any recorded error.
        If no error exists, it returns the original function output.

        Parameters:
        -----------
        *args:
            positional arguments passed to the wrapped function.
        **kwargs:
            keyword arguments passed to the wrapped function.

        Returns:
        --------
        Any:
            output of the original function.
        """
        err, out = jitted(*args, **kwargs)
        err.throw()
        return out

    return wrapped
