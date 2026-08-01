from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

from .input_validation_jax import assert_array_within_interval

_EMPTY_I32 = jnp.zeros((0,), dtype=jnp.int64)
_DEFAULT_BOUNDS = jnp.array([jnp.inf, jnp.inf], dtype=jnp.float64)  # matches bounds


def init_bounds_config_jax(
    n_dim: int,
    bounds: Array = _DEFAULT_BOUNDS,
    periodic: Array = _EMPTY_I32,
    reflective: Array = _EMPTY_I32,
    *,
    transform: str = "probit",
    scale: bool = True,
    diagonal: bool = True,
) -> dict[str, Array]:
    """
    Builds the initial configuration for bounded-to-unbounded scaling.

    The sampler works in an unconstrained latent space.
    This function stores the information needed to map between
    bounded physical coordinates and unconstrained coordinates.

    The configuration contains lower and upper bounds, masks for
    periodic and reflective boundary conditions, the selected bounded
    transform, scaling flags, and placeholder affine-scaling parameters.

    This function performs static Python validation.

    Parameters:
    -----------
    n_dim:
        number of dimensions.
        Must be a positive Python integer.

    bounds:
        lower and upper bounds.
        Allowed shapes are (2,) for shared bounds
        or (n_dim, 2) for dimension-specific bounds.
        Column 0 contains lower bounds.
        Column 1 contains upper bounds.

    periodic:
        indices of dimensions with periodic boundary conditions.
        These dimensions wrap around their interval.

    reflective:
        indices of dimensions with reflective boundary conditions.
        These dimensions are reflected back into their interval.

    transform:
        transform used for dimensions with two finite bounds.
        Must be "logit" or "probit".

    scale:
        if True, affine scaling is enabled after the bounds transform.

    diagonal:
        if True, affine scaling uses per-dimension standard deviations.
        If False, affine scaling uses a full covariance matrix.

    Returns:
    --------
    dict[str, Array]:
        configuration dictionary containing bounds, boundary masks,
        transform code, scaling flags, and affine-scaling placeholders.
    """
    # static validation
    if not isinstance(n_dim, int):
        raise TypeError("n_dim must be a Python int.")

    if n_dim <= 0:
        raise ValueError("n_dim must be a positive integer.")

    if transform not in ("logit", "probit"):
        raise ValueError("transform must be 'logit' or 'probit'.")

    bounds = jnp.asarray(bounds, dtype=jnp.float64)

    # shape validation is static and therefore safe under jit
    if bounds.ndim == 1:
        if bounds.shape != (2,):
            raise ValueError("bounds must have shape (2,) or (n_dim, 2).")
    elif bounds.ndim == 2:
        if bounds.shape != (n_dim, 2):
            raise ValueError("bounds must have shape (2,) or (n_dim, 2).")
    else:
        raise ValueError("bounds must have shape (2,) or (n_dim, 2).")

    bounds = jnp.broadcast_to(bounds, (n_dim, 2))
    low = bounds[:, 0]
    high = bounds[:, 1]

    periodic = jnp.asarray(periodic, dtype=jnp.int64).reshape((-1,))
    reflective = jnp.asarray(reflective, dtype=jnp.int64).reshape((-1,))

    dims = jnp.arange(n_dim, dtype=jnp.int64)
    periodic_mask = jnp.any(dims[:, None] == periodic[None, :], axis=1)
    reflective_mask = jnp.any(dims[:, None] == reflective[None, :], axis=1)

    transform_id = jnp.asarray(transform == "probit", dtype=jnp.int64)

    dtype = bounds.dtype
    nan_vec = jnp.full((n_dim,), jnp.nan, dtype=dtype)
    nan_mat = jnp.full((n_dim, n_dim), jnp.nan, dtype=dtype)
    nan_scalar = jnp.asarray(jnp.nan, dtype=dtype)

    return {
        "ndim": jnp.asarray(n_dim, dtype=jnp.int64),
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
    Builds masks that describe which kind of bounds each dimension has.

    A dimension can be unbounded, left-bounded, right-bounded,
    or bounded on both sides. These masks are used to choose
    the correct transformation for each coordinate.

    Parameters:
    -----------
    low:
        lower bounds, shape (D,).
    high:
        upper bounds, shape (D,).

    Returns:
    --------
    dict[str, Array]:
        dictionary with four Boolean masks:
        mask_left, mask_right, mask_both, and mask_none.
    """
    low = jnp.asarray(low)
    high = jnp.asarray(high)

    # mark finite lower and upper bounds
    fin_low = jnp.isfinite(low)
    fin_high = jnp.isfinite(high)

    # define four bound masks
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
    """
    Builds bound masks directly from a dimension count and bounds.

    This is a convenience helper.
    It first creates a bounds configuration.
    It then extracts the four bound masks from that configuration.

    Parameters:
    -----------
    n_dim:
        number of dimensions.
    bounds:
        lower and upper bounds.
        Allowed shapes are (2,) or (n_dim, 2).

    Returns:
    --------
    dict[str, Array]:
        dictionary with masks for each bound type.
    """
    cfg = init_bounds_config_jax(n_dim, bounds)
    return masks_jax(cfg["low"], cfg["high"])


def _inverse_none_jax(u: Array, mask_none: Array) -> tuple[Array, Array]:
    """
    Applies the inverse transform for unbounded dimensions.

    Unbounded dimensions do not need a bounds transform.
    They pass from u-space to x-space unchanged.
    Their log-Jacobian contribution is zero.

    Parameters:
    -----------
    u:
        unconstrained values, shape (N, D).
    mask_none:
        Boolean mask for unbounded dimensions, shape (D,).

    Returns:
    --------
    tuple[Array, Array]:
        selected x values and zero log-Jacobian terms.
    """
    u = jnp.asarray(u)
    mask_none = jnp.asarray(mask_none, dtype=bool)

    # unbounded dimensions pass through unchanged
    x = u[:, mask_none]
    log_det_J = jnp.zeros_like(u)[:, mask_none]
    return x, log_det_J


def _forward_none_jax(x: Array, mask_none: Array) -> Array:
    """
    Applies the forward transform for unbounded dimensions.

    Unbounded dimensions do not need any transformation.
    They pass from x-space to u-space unchanged.

    Parameters:
    -----------
    x:
        constrained-space values, shape (N, D).
    mask_none:
        Boolean mask for unbounded dimensions, shape (D,).

    Returns:
    --------
    Array:
        selected u values for unbounded dimensions.
    """
    x = jnp.asarray(x)
    mask_none = jnp.asarray(mask_none, dtype=bool)

    return x[:, mask_none]


def _inverse_both_jax(
    u: Array,
    low: Array,
    high: Array,
    mask_both: Array,
    transform_id: Array,  # 0 = logit, 1 = probit
) -> tuple[Array, Array]:
    """
    Applies the inverse transform for dimensions with two finite bounds.

    These dimensions have support [low, high].
    The inverse transform maps unconstrained u values into that interval.
    It can use either a logit-based transform or a probit-based transform.

    Parameters:
    -----------
    u:
        unconstrained values, shape (N, D).
    low:
        lower bounds, shape (D,).
    high:
        upper bounds, shape (D,).
    mask_both:
        Boolean mask for two-sided bounded dimensions, shape (D,).
    transform_id:
        integer transform code.
        0 means logit.
        1 means probit.

    Returns:
    --------
    tuple[Array, Array]:
        transformed x values and diagonal log-Jacobian terms
        for the selected bounded dimensions.
    """
    # inputs to arrays
    u = jnp.asarray(u)
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    mask_both = jnp.asarray(mask_both, dtype=bool)
    transform_id = jnp.asarray(transform_id, dtype=jnp.int64)

    # choose only bounded dimensions
    u_sel = u[:, mask_both]  # (N, K)
    low_sel = low[mask_both]  # (K,)
    high_sel = high[mask_both]  # (K,)
    span = high_sel - low_sel  # (K,)
    log_span = jnp.log(span)  # (K,)  (invalid bounds will yield nan/-inf)

    def _logit_branch(op):
        """
        Applies the inverse logit bounded transform.

        The sigmoid maps u into (0, 1).
        The result is then stretched to [low, high].

        Parameters:
        -----------
        op:
            tuple with selected u values, lower bounds,
            log spans, and spans.

        Returns:
        --------
        tuple[Array, Array]:
            transformed values and log-Jacobian terms.
        """
        # define selected arrays
        u_s, low_s, log_span_s, span_s = op

        # map probabilities from sigmoid values into bounded interval
        p = jax.nn.sigmoid(u_s)
        x = p * span_s + low_s
        J = log_span_s + jnp.log(p) + jnp.log1p(-p)
        return x, J

    def _probit_branch(op):
        """
        Applies the inverse probit bounded transform.

        The Gaussian CDF maps u into (0, 1).
        The result is then stretched to [low, high].

        Parameters:
        -----------
        op:
            tuple with selected u values, lower bounds,
            log spans, and spans.

        Returns:
        --------
        tuple[Array, Array]:
            transformed values and log-Jacobian terms.
        """
        # define selected arrays
        u_s, low_s, log_span_s, span_s = op

        # map Gaussian CDF values into bounded interval
        p = jsp.special.ndtr(u_s)  # Phi(u)
        x = p * span_s + low_s
        # log phi(u) = -0.5 u^2 - log(sqrt(2*pi))
        J = log_span_s + (-0.5 * u_s**2) - jnp.log(jnp.sqrt(2.0 * jnp.pi))
        return x, J

    # choose requested two-sided transform
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
    transform_id: Array,  # 0 = logit, 1 = probit
    *,
    eps: float = 1e-13,
) -> Array:
    """
    Applies the forward transform for dimensions with two finite bounds.

    The forward transform maps values from [low, high] to the real line.
    First, x is converted to a probability in (0, 1).
    Then either logit or probit maps that probability to u-space.

    Parameters:
    -----------
    x:
        constrained values, shape (N, D).
    low:
        lower bounds, shape (D,).
    high:
        upper bounds, shape (D,).
    mask_both:
        Boolean mask for two-sided bounded dimensions, shape (D,).
    transform_id:
        integer transform code.
        0 means logit.
        1 means probit.
    eps:
        clipping value used to keep probabilities away from 0 and 1.

    Returns:
    --------
    Array:
        transformed u values for two-sided bounded dimensions.
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    mask_both = jnp.asarray(mask_both, dtype=bool)
    transform_id = jnp.asarray(transform_id, dtype=jnp.int64)

    # select only bounded dimensions
    x_sel = x[:, mask_both]  # (N, K)
    low_sel = low[mask_both]  # (K,)
    high_sel = high[mask_both]  # (K,)
    span = high_sel - low_sel  # (K,)

    # convert bounded values to probabilities and clip them away from 0 and 1
    p = (x_sel - low_sel) / span
    eps_t = jnp.asarray(eps, dtype=x_sel.dtype)
    p = jnp.clip(p, eps_t, 1.0 - eps_t)

    def _logit_branch(p_in: Array) -> Array:
        """
        Applies the logit transform.

        Parameters:
        -----------
        p_in:
            probability values in (0, 1).

        Returns:
        --------
        Array:
            logit-transformed values.
        """
        # apply logit transform: logit(p) = log(p) - log(1-p)
        return jnp.log(p_in) - jnp.log1p(-p_in)

    def _probit_branch(p_in: Array) -> Array:
        """
        Applies the probit transform.

        Parameters:
        -----------
        p_in:
            probability values in (0, 1).

        Returns:
        --------
        Array:
            probit-transformed values.
        """
        # apply inverse Gaussian CDF: probit(p) = sqrt(2) * erfinv(2p - 1)
        return jnp.sqrt(jnp.asarray(2.0, dtype=p_in.dtype)) * jsp.special.erfinv(
            2.0 * p_in - 1.0
        )

    # choose requested two-sided transform
    u = jax.lax.switch(transform_id, (_logit_branch, _probit_branch), p)
    return u


def _inverse_right_jax(u: Array, high: Array, mask_right: Array) -> tuple[Array, Array]:
    """
    Applies the inverse transform for dimensions with only an upper bound.

    A right-bounded dimension has support (-inf, high].
    The inverse transform maps unconstrained u values into this support
    using x = high - exp(u).

    Parameters:
    -----------
    u:
        unconstrained values, shape (N, D).
    high:
        upper bounds, shape (D,).
    mask_right:
        Boolean mask for right-bounded dimensions, shape (D,).

    Returns:
    --------
    tuple[Array, Array]:
        transformed x values and log-Jacobian terms.
    """
    u = jnp.asarray(u)
    high = jnp.asarray(high)
    mask_right = jnp.asarray(mask_right, dtype=bool)

    # choose only right-bounded dimensions
    u_sel = u[:, mask_right]  # (N, K)
    high_sel = high[mask_right]  # (K,)

    # apply x = high - exp(u) and keep u as log-Jacobian term
    x = high_sel - jnp.exp(u_sel)  # (N, K) through  broadcasting
    J = u_sel
    return x, J


def _forward_right_jax(x: Array, high: Array, mask_right: Array) -> Array:
    """
    Applies the forward transform for dimensions with only an upper bound.

    A right-bounded dimension has support (-inf, high].
    The forward transform maps it to the real line using u = log(high - x).

    Parameters:
    -----------
    x:
        constrained values, shape (N, D).
    high:
        upper bounds, shape (D,).
    mask_right:
        Boolean mask for right-bounded dimensions, shape (D,).

    Returns:
    --------
    Array:
        transformed u values for right-bounded dimensions.
    """
    x = jnp.asarray(x)
    high = jnp.asarray(high)
    mask_right = jnp.asarray(mask_right, dtype=bool)

    # select only right-bounded dimensions
    x_sel = x[:, mask_right]  # (N, K)
    high_sel = high[mask_right]  # (K,)

    # apply u = log(high - x)
    return jnp.log(high_sel - x_sel)


def _inverse_left_jax(u: Array, low: Array, mask_left: Array) -> tuple[Array, Array]:
    """
    Applies the inverse transform for dimensions with only a lower bound.

    A left-bounded dimension has support [low, inf).
    The inverse transform maps unconstrained u values into this support
    using x = exp(u) + low.

    Parameters:
    -----------
    u:
        unconstrained values, shape (N, D).
    low:
        lower bounds, shape (D,).
    mask_left:
        Boolean mask for left-bounded dimensions, shape (D,).

    Returns:
    --------
    tuple[Array, Array]:
        transformed x values and log-Jacobian terms.
    """
    u = jnp.asarray(u)
    low = jnp.asarray(low)
    mask_left = jnp.asarray(mask_left, dtype=bool)

    # select only left-bounded dimensions
    u_sel = u[:, mask_left]  # (N, K)
    low_sel = low[mask_left]  # (K,)

    # apply x = exp(u) + low and keep u as the log-Jacobian term
    x = jnp.exp(u_sel) + low_sel
    J = u_sel
    return x, J


def _forward_left_jax(x: Array, low: Array, mask_left: Array) -> Array:
    """
    Applies the forward transform for dimensions with only a lower bound.

    A left-bounded dimension has support [low, inf).
    The forward transform maps it to the real line using u = log(x - low).

    Parameters:
    -----------
    x:
        constrained values, shape (N, D).
    low:
        lower bounds, shape (D,).
    mask_left:
        Boolean mask for left-bounded dimensions, shape (D,).

    Returns:
    --------
    Array:
        transformed u values for left-bounded dimensions.
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    mask_left = jnp.asarray(mask_left, dtype=bool)

    # select only left-bounded dimensions
    x_sel = x[:, mask_left]  # (N, K)
    low_sel = low[mask_left]  # (K,)

    # apply u = log(x - low)
    return jnp.log(x_sel - low_sel)


def _inverse_affine_jax(
    u: Array,  # (N, D)
    mu: Array,  # (D,)
    sigma: Array,  # (D,)  (use it if diagonal=True)
    L: Array,  # (D, D) (use it if diagonal=False)
    log_det_L: Array,  # scalar (use it if diagonal=False)
    diagonal: Array | bool,  # scalar bool
) -> tuple[Array, Array]:
    """
    Applies the inverse affine scaling transform.

    This transform undoes the fitted centering and scaling.
    If diagonal is True, each dimension is scaled independently.
    If diagonal is False, a full Cholesky factor is used.

    Parameters:
    -----------
    u:
        scaled unconstrained values, shape (N, D).
    mu:
        fitted mean vector, shape (D,).
    sigma:
        fitted diagonal scale vector, shape (D,).
        Used only when diagonal is True.
    L:
        fitted Cholesky factor, shape (D, D).
        Used only when diagonal is False.
    log_det_L:
        log determinant of L.
        Used only when diagonal is False.
    diagonal:
        Boolean flag selecting diagonal or full affine scaling.

    Returns:
    --------
    tuple[Array, Array]:
        unscaled values and affine log-determinant vector.
    """
    u = jnp.asarray(u)
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)
    L = jnp.asarray(L)
    log_det_L = jnp.asarray(log_det_L)
    diagonal = jnp.asarray(diagonal, dtype=bool)

    # build a length-N vector used to broadcast log-determinant
    n = u.shape[0]
    ones_n = jnp.ones((n,), dtype=jnp.result_type(u, mu, sigma, L, log_det_L))

    def _diag_branch(_):
        """
        Applies diagonal inverse affine scaling.

        Parameters:
        -----------
        _:
            unused operand required by lax.cond.

        Returns:
        --------
        tuple[Array, Array]:
            unscaled values and log determinant.
        """
        # apply x = mu + sigma * u
        x = mu + sigma * u
        log_det = jnp.sum(jnp.log(sigma)) * ones_n
        return x, log_det

    def _full_branch(_):
        """
        Applies full inverse affine scaling.

        Parameters:
        -----------
        _:
            unused operand required by lax.cond.

        Returns:
        --------
        tuple[Array, Array]:
            unscaled values and log determinant.
        """
        # vectorized version of: mu + np.array([L @ ui for ui in u])
        # apply x = mu + u @ L.T.
        x = mu + (u @ L.T)
        log_det = log_det_L * ones_n
        return x, log_det

    # choose diagonal or full affine scaling
    x, log_det = jax.lax.cond(diagonal, _diag_branch, _full_branch, operand=None)
    return x, log_det


def _forward_affine_jax(
    x: Array,  # (N, D)
    mu: Array,  # (D,)
    sigma: Array,  # (D,)    used if diagonal=True
    L_inv: Array,  # (D, D)  used if diagonal=False
    diagonal: Array,  # scalar bool
) -> Array:
    """
    Applies the forward affine scaling transform.

    This transform centers and scales values after the bounds transform.
    If diagonal is True, scaling is done dimension by dimension.
    If diagonal is False, the inverse Cholesky factor is used.

    Parameters:
    -----------
    x:
        unscaled values, shape (N, D).
    mu:
        fitted mean vector, shape (D,).
    sigma:
        fitted diagonal scale vector, shape (D,).
        Used only when diagonal is True.
    L_inv:
        inverse Cholesky factor, shape (D, D).
        Used only when diagonal is False.
    diagonal:
        Boolean flag selecting diagonal or full affine scaling.

    Returns:
    --------
    Array:
        scaled values, shape (N, D).
    """
    x = jnp.asarray(x)
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)
    L_inv = jnp.asarray(L_inv)
    diagonal = jnp.asarray(diagonal, dtype=bool)

    def _diag_branch(_):
        """
        Applies diagonal forward affine scaling.

        Parameters:
        -----------
        _:
            unused operand required by lax.cond.

        Returns:
        --------
        Array:
            centered and diagonally scaled values.
        """
        # apply u = (x - mu) / sigma
        return (x - mu) / sigma

    def _full_branch(_):
        """
        Applies full forward affine scaling.

        Parameters:
        -----------
        _:
            unused operand required by lax.cond.

        Returns:
        --------
        Array:
            centered and fully scaled values.
        """
        # vectorized version of: np.array([L_inv @ (xi - mu) for xi in x])
        # apply u = (x - mu) @ L_inv.T.
        return (x - mu) @ L_inv.T

    # choose diagonal or full affine scaling
    return jax.lax.cond(diagonal, _diag_branch, _full_branch, operand=None)


_LOG_SQRT_2PI = jnp.log(jnp.sqrt(2.0 * jnp.pi))


def _inverse_jax(
    u: jax.Array,  # (N, D)
    low: jax.Array,  # (D,)
    high: jax.Array,  # (D,)
    mask_none: jax.Array,  # (D,) bool
    mask_left: jax.Array,  # (D,) bool
    mask_right: jax.Array,  # (D,) bool
    mask_both: jax.Array,  # (D,) bool
    transform_id: jax.Array,  # scalar int: 0=logit, 1=probit
) -> tuple[jax.Array, jax.Array]:
    """
    Applies the inverse bounds transform with fixed output shape.

    This function maps unconstrained u-space values into constrained x-space.
    It handles all bound types in one static-shape calculation.
    This is useful for JAX because the output shape does not depend
    on how many dimensions have each bound type.

    Parameters:
    -----------
    u:
        unconstrained values, shape (N, D).
    low:
        lower bounds, shape (D,).
    high:
        upper bounds, shape (D,).
    mask_none:
        Boolean mask for unbounded dimensions, shape (D,).
    mask_left:
        Boolean mask for left-bounded dimensions, shape (D,).
    mask_right:
        Boolean mask for right-bounded dimensions, shape (D,).
    mask_both:
        Boolean mask for two-sided bounded dimensions, shape (D,).
    transform_id:
        integer transform code.
        0 means logit.
        1 means probit.

    Returns:
    --------
    tuple[jax.Array, jax.Array]:
        constrained x values, shape (N, D),
        and summed log-Jacobian values, shape (N,).
    """
    u = jnp.asarray(u)
    low = jnp.asarray(low)
    high = jnp.asarray(high)

    # expand masks so they broadcast over rows
    mask_none = jnp.asarray(mask_none, dtype=bool)[None, :]  # (1, D)
    mask_left = jnp.asarray(mask_left, dtype=bool)[None, :]
    mask_right = jnp.asarray(mask_right, dtype=bool)[None, :]
    mask_both = jnp.asarray(mask_both, dtype=bool)[None, :]

    # initiate transform choice
    transform_id = jnp.asarray(transform_id, dtype=jnp.int64)
    is_probit = transform_id == 1  # scalar bool

    # build span and log-span only where both bounds are finite
    span = jnp.where(mask_both[0], high - low, 1.0)  # (D,)
    log_span = jnp.log(span)  # (D,)

    # logit inverse branch for all dimensions
    p_sig = jax.nn.sigmoid(u)  # (N, D)
    x_logit = low + p_sig * span  # (N, D)
    J_logit = log_span + jnp.log(p_sig) + jnp.log1p(-p_sig)  # (N, D)

    # probit inverse branch for all dimensions
    p_phi = jsp.special.ndtr(u)  # (N, D)
    x_probit = low + p_phi * span  # (N, D)
    J_probit = log_span + (-0.5 * u * u) - _LOG_SQRT_2PI  # (N, D)

    # select requested two-sided branch
    x_both = jnp.where(is_probit, x_probit, x_logit)  # (N, D)
    J_both = jnp.where(is_probit, J_probit, J_logit)  # (N, D)

    # one-sided branches (computed for all dims, only used where their mask=True)
    exp_u = jnp.exp(u)
    x_left = exp_u + low
    J_left = u

    x_right = high - exp_u
    J_right = u

    # create full (N, D) output arrays with static shapes
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

    # sum diagonal log-Jacobian terms over dimensions
    log_det_J = jnp.sum(J, axis=1)  # (N,)
    return x, log_det_J


def _forward_jax(
    x: Array,  # (N, D)
    low: Array,  # (D,)
    high: Array,  # (D,)
    mask_none: Array,  # (D,) bool
    mask_left: Array,  # (D,) bool
    mask_right: Array,  # (D,) bool
    mask_both: Array,  # (D,) bool
    transform_id: Array,  # scalar int: 0=logit, 1=probit
    *,
    eps: float = 1e-13,
) -> Array:
    """
    Applies the forward bounds transform with fixed output shape.

    This function maps constrained x-space values into unconstrained u-space.
    It handles unbounded, one-sided bounded, and two-sided bounded
    dimensions in one static-shape calculation.

    Parameters:
    -----------
    x:
        constrained values, shape (N, D).
    low:
        lower bounds, shape (D,).
    high:
        upper bounds, shape (D,).
    mask_none:
        Boolean mask for unbounded dimensions, shape (D,).
    mask_left:
        Boolean mask for left-bounded dimensions, shape (D,).
    mask_right:
        Boolean mask for right-bounded dimensions, shape (D,).
    mask_both:
        Boolean mask for two-sided bounded dimensions, shape (D,).
    transform_id:
        integer transform code.
        0 means logit.
        1 means probit.
    eps:
        clipping value used for two-sided bounded dimensions.
        It prevents logit or probit from receiving exactly 0 or 1.

    Returns:
    --------
    Array:
        unconstrained u values, shape (N, D).
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    high = jnp.asarray(high)

    # expand masks so they broadcast over rows
    mask_none = jnp.asarray(mask_none, dtype=bool)[None, :]
    mask_left = jnp.asarray(mask_left, dtype=bool)[None, :]
    mask_right = jnp.asarray(mask_right, dtype=bool)[None, :]
    mask_both = jnp.asarray(mask_both, dtype=bool)[None, :]

    # read transform choice
    transform_id = jnp.asarray(transform_id, dtype=jnp.int64)
    is_probit = transform_id == 1

    # unbounded dimensions pass through unchanged
    u_none = x

    # left: log(x - low)
    u_left = jnp.log(x - low)

    # one-sided forward transforms: right: log(high - x)
    u_right = jnp.log(high - x)

    # two-sided probabilities: p=(x-low)/(high-low) then logit/probit
    span = jnp.where(mask_both[0], high - low, 1.0)  # (D,)
    low_safe = jnp.where(mask_both[0], low, 0.0)  # (D,)
    p = (x - low_safe) / span  # (N, D)

    # clip probabilities away from 0 and 1
    eps_t = jnp.asarray(eps, dtype=x.dtype)
    p = jnp.clip(p, eps_t, 1.0 - eps_t)

    # initialize two-sided logit and probit transforms
    u_logit = jnp.log(p) - jnp.log1p(-p)
    u_probit = jnp.sqrt(jnp.asarray(2.0, dtype=x.dtype)) * jsp.special.erfinv(
        2.0 * p - 1.0
    )
    u_both = jnp.where(is_probit, u_probit, u_logit)

    # assemble full output array (N, D)
    u = jnp.zeros_like(x)
    u = jnp.where(mask_none, u_none, u)
    u = jnp.where(mask_left, u_left, u)
    u = jnp.where(mask_right, u_right, u)
    u = jnp.where(mask_both, u_both, u)
    return u


def inverse_jax(
    u: Array, cfg: Mapping[str, Array], masks: Mapping[str, Array]
) -> tuple[Array, Array]:
    """
    Applies the full inverse scaler transformation.

    This maps u-space values back to x-space.
    If affine scaling is enabled, the affine transform is undone first.
    Then the bounds transform is undone.
    The function also returns the total log-Jacobian correction.

    Parameters:
    -----------
    u:
        unconstrained input values, shape (N, D).
    cfg:
        scaler configuration dictionary.
        It must contain bounds, transform_id, scale flag,
        diagonal flag, and fitted affine parameters.
    masks:
        dictionary with bound masks.
        It must contain mask_none, mask_left, mask_right, and mask_both.

    Returns:
    --------
    tuple[Array, Array]:
        x:
            transformed x-space values, shape (N, D).
        log_det_J:
            total log-Jacobian correction, shape (N,).
    """
    u = jnp.asarray(u)

    # values from the configuration and mask dictionaries
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
        """
        Applies inverse affine scaling and inverse bounds transformation.

        Parameters:
        -----------
        u_in:
            unconstrained input values, shape (N, D).

        Returns:
        --------
        tuple[Array, Array]:
            x-space values and total log-Jacobian correction.
        """
        # undo affine scaling, then undo bounds transform
        x1, ld1 = _inverse_affine_jax(u_in, mu, sigma, L, log_det_L, diagonal)
        x2, ld2 = _inverse_jax(
            x1,
            low,
            high,
            mask_none,
            mask_left,
            mask_right,
            mask_both,
            transform_id,
        )
        return x2, ld1 + ld2

    def _unscaled(u_in: Array) -> tuple[Array, Array]:
        """
        Applies only the inverse bounds transformation.

        Parameters:
        -----------
        u_in:
            unconstrained input values, shape (N, D).

        Returns:
        --------
        tuple[Array, Array]:
            x-space values and bounds-transform log-Jacobian correction.
        """
        # skip affine scaling when it is disabled
        return _inverse_jax(
            u_in,
            low,
            high,
            mask_none,
            mask_left,
            mask_right,
            mask_both,
            transform_id,
        )

    # choose scaled or unscaled inverse transformation
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
    Applies the full forward scaler transformation.

    This maps x-space values into u-space.
    First, the bounds transform maps constrained coordinates
    into an unconstrained representation.
    If affine scaling is enabled, fitted centering and scaling
    are then applied.

    Parameters:
    -----------
    x:
        constrained input values, shape (N, D).
    cfg:
        scaler configuration dictionary.
        It must contain bounds, transform_id, scale flag,
        diagonal flag, and fitted affine parameters.
    masks:
        dictionary with bound masks.
        It must contain mask_none, mask_left, mask_right, and mask_both.
    eps:
        clipping value for two-sided bounded dimensions.

    Returns:
    --------
    Array:
        transformed u-space values, shape (N, D).
    """
    x = jnp.asarray(x)

    # apply the bounds transform first
    u0 = _forward_jax(
        x,
        cfg["low"],
        cfg["high"],
        masks["mask_none"],
        masks["mask_left"],
        masks["mask_right"],
        masks["mask_both"],
        cfg["transform_id"],
        eps=eps,
    )

    # affine-scaling options
    scale = jnp.asarray(cfg["scale"], dtype=bool)
    diagonal = jnp.asarray(cfg["diagonal"], dtype=bool)

    def _scaled(u_in: Array) -> Array:
        """
        Applies forward affine scaling.

        Parameters:
        -----------
        u_in:
            bounds-transformed values, shape (N, D).

        Returns:
        --------
        Array:
            fully transformed u-space values.
        """
        # apply affine scaling when it is enabled
        return _forward_affine_jax(
            u_in, cfg["mu"], cfg["sigma"], cfg["L_inv"], diagonal
        )

    # choose scaled or unscaled forward transformation
    return jax.lax.cond(scale, _scaled, lambda z: z, u0)


def forward_jax_checked(
    x: Array,
    cfg: Mapping[str, Array],
    masks: Mapping[str, Array],
    *,
    eps: float = 1e-13,
) -> Array:
    """
    Checks bounds and then applies the full forward scaler transform.

    This function is the safer version of forward_jax.
    It first checks that all x values lie inside the configured interval.
    If the check passes, it maps x-space values into u-space.

    Parameters:
    -----------
    x:
        constrained input values, shape (N, D).
    cfg:
        scaler configuration dictionary.
    masks:
        dictionary with bound masks.
    eps:
        clipping value for two-sided bounded dimensions.

    Returns:
    --------
    Array:
        transformed u-space values, shape (N, D).
    """
    # check that x stays inside the configured bounds
    x = assert_array_within_interval(x, cfg["low"], cfg["high"], name="x")
    return forward_jax(x, cfg, masks, eps=eps)


def fit_jax(
    x: Array,
    cfg: Mapping[str, Array],
    masks: Mapping[str, Array],
    *,
    eps: float = 1e-13,
    jitter: float = 0.0,  # set 1e-6 if problems with Cholevsky
) -> dict[str, Array]:
    """
    Fits affine scaling parameters after the bounds transform.

    The function first maps x-space values through the bounds transform.
    It then estimates affine scaling parameters in that transformed space.

    If diagonal scaling is enabled, it stores a mean and per-dimension
    standard deviation.
    If full scaling is enabled, it stores a mean, covariance matrix,
    Cholesky factor, inverse Cholesky factor, and log determinant.

    Parameters:
    -----------
    x:
        constrained input values, shape (N, D).
    cfg:
        scaler configuration dictionary with bounds and placeholders.
    masks:
        dictionary with bound masks.
    eps:
        clipping value used during the bounds transform.
    jitter:
        diagonal jitter added before Cholesky factorization
        in the full-covariance case.

    Returns:
    --------
    dict[str, Array]:
        updated scaler configuration with fitted affine parameters.
    """
    x = jnp.asarray(x)

    # (i) forward bounds transform before fitting affine scaling
    u = _forward_jax(
        x,
        cfg["low"],
        cfg["high"],
        masks["mask_none"],
        masks["mask_left"],
        masks["mask_right"],
        masks["mask_both"],
        cfg["transform_id"],
        eps=eps,
    )

    # compute mean of the transformed data
    mu = jnp.mean(u, axis=0)
    diagonal = jnp.asarray(cfg["diagonal"], dtype=bool)

    # build common constants for diagonal and full branches
    D = u.shape[1]
    dtype = u.dtype
    eye = jnp.eye(D, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)

    def _diag_branch(_):
        """
        Fits diagonal affine scaling.

        The scale is the standard deviation of each transformed dimension.
        The covariance-related fields are set to identity placeholders.

        Parameters:
        -----------
        _:
            unused operand required by lax.cond.

        Returns:
        --------
        tuple:
            sigma, covariance, Cholesky factor,
            inverse Cholesky factor, and log determinant.
        """
        # use per-dimension standard deviation
        sigma = jnp.std(u, axis=0)  # ddof=0 (matches np.std default)
        cov = eye
        L = eye
        L_inv = eye
        log_det_L = zero
        return sigma, cov, L, L_inv, log_det_L

    def _full_branch(_):
        """
        Fits full affine scaling from the covariance matrix.

        The covariance is estimated from the transformed samples.
        A Cholesky factor is then computed.
        The inverse Cholesky factor is used by the forward transform.

        Parameters:
        -----------
        _:
            unused operand required by lax.cond.

        Returns:
        --------
        tuple:
            sigma placeholder, covariance, Cholesky factor,
            inverse Cholesky factor, and log determinant.
        """
        # compute sample covariance of transformed data:
        # np.cov(u.T) equivalent: centered.T @ centered / (N-1)
        n = u.shape[0]
        denom = jnp.asarray(
            jnp.maximum(n - 1, 1), dtype=dtype
        )  # avoid divide-by-zero if n==1
        centered = u - mu
        cov = (centered.T @ centered) / denom

        # numerical stabilization: optional jitter before Cholesky
        cov = cov + jnp.asarray(jitter, dtype=dtype) * eye

        # factorize covariance and build its inverse factor
        L = jnp.linalg.cholesky(cov)

        # L_inv = inv(L)
        L_inv = jsp.linalg.solve_triangular(L, eye, lower=True)

        # log(det(L)) for triangular L
        log_det_L = jnp.sum(jnp.log(jnp.diag(L)))

        # keep sigma for consistency with config structure
        sigma = cfg.get("sigma", jnp.ones((D,), dtype=dtype))
        return sigma, cov, L, L_inv, log_det_L

    # choose diagonal or full affine fitting
    sigma, cov, L, L_inv, log_det_L = jax.lax.cond(
        diagonal, _diag_branch, _full_branch, operand=None
    )

    # return new config dictionary with updated fitted values
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
    Applies reflective boundary conditions to selected dimensions.

    Reflective dimensions are folded back into their finite interval.
    Values that go above the upper bound are reflected downward.
    Values that go below the lower bound are reflected upward.

    This is useful for parameters where crossing a boundary should
    behave like bouncing off a wall.

    Parameters:
    -----------
    x:
        input values, shape (N, D).
    low:
        lower bounds, shape (D,).
    high:
        upper bounds, shape (D,).
    reflective_mask:
        Boolean mask for reflective dimensions, shape (D,).

    Returns:
    --------
    Array:
        values after reflective boundary handling, shape (N, D).
        Non-reflective dimensions are unchanged.
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    reflective_mask = jnp.asarray(reflective_mask, dtype=bool)

    # skip when no reflective dimensions exist
    has_reflect = jnp.any(reflective_mask)

    def _do_reflect(x_in: Array) -> Array:
        """
        Reflects values back into their intervals.

        Parameters:
        -----------
        x_in:
            input values, shape (N, D).

        Returns:
        --------
        Array:
            reflected values, shape (N, D).
        """
        # expand mask across rows
        m = reflective_mask[None, :]  # (1, D)

        # dont touch non-reflective dims while computing (prevents inf/nan propagation)
        x_safe = jnp.where(m, x_in, 0.0)

        # build safe lower bounds and interval widths
        low_safe = jnp.where(reflective_mask, low, 0.0)  # (D,)
        span = jnp.where(reflective_mask, high - low, 1.0)  # (D,)

        # keep bound numerically stable: guard against zero-width intervals
        tiny = jnp.asarray(jnp.finfo(x_in.dtype).tiny, dtype=x_in.dtype)
        span = jnp.where(span > tiny, span, 1.0)

        # define period
        period = 2.0 * span  # (D,)

        # fold into a period: [0, 2*span)
        y = jnp.mod(x_safe - low_safe, period)  # (N, D), in [0, period)

        # reflect second half back into interval: [span, 2*span) into [span, 0]
        y = jnp.where(y > span, period - y, y)

        # restore non-reflective dimensions unchanged
        x_ref = low_safe + y  # (N, D)
        return jnp.where(m, x_ref, x_in)

    # apply reflection only when needed
    return jax.lax.cond(has_reflect, _do_reflect, lambda z: z, x)


def apply_periodic_boundary_conditions_x_jax(
    x: Array,
    low: Array,
    high: Array,
    periodic_mask: Array,
) -> Array:
    """
    Applies periodic boundary conditions to selected dimensions.

    Periodic dimensions wrap around their finite interval.
    For example, a value slightly above the upper bound is moved
    back near the lower bound.
    This is useful for circular variables such as angles.

    Parameters:
    -----------
    x:
        input values, shape (N, D).
    low:
        lower bounds, shape (D,).
    high:
        upper bounds, shape (D,).
    periodic_mask:
        Boolean mask for periodic dimensions, shape (D,).

    Returns:
    --------
    Array:
        values after periodic wrapping, shape (N, D).
        Non-periodic dimensions are unchanged.
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    periodic_mask = jnp.asarray(periodic_mask, dtype=bool)

    # skip when no periodic dimensions exist
    has_periodic = jnp.any(periodic_mask)

    def _wrap(x_in: Array) -> Array:
        """
        Wraps periodic values back into their intervals.

        Parameters:
        -----------
        x_in:
            input values, shape (N, D).

        Returns:
        --------
        Array:
            wrapped values, shape (N, D).
        """
        # expand mask across rows and keep only finite intervals
        m = periodic_mask[None, :]  # (1, D)

        # periodic for finite bounds only
        fin = jnp.isfinite(low) & jnp.isfinite(high)
        m = m & fin[None, :]

        # placeholders to avoid inf/nan propagation in non-periodic dims
        x_safe = jnp.where(m, x_in, 0.0)

        # safe lower bounds, upper bounds, and interval widths
        low_safe = jnp.where(periodic_mask, low, 0.0)  # (D,)
        high_safe = jnp.where(periodic_mask, high, 1.0)  # (D,)
        span = jnp.where(periodic_mask, high - low, 1.0)  # (D,)

        # protect against zero-width intervals
        tiny = jnp.asarray(jnp.finfo(x_in.dtype).tiny, dtype=x_in.dtype)
        span = jnp.where(span > tiny, span, 1.0)

        # wrap to [low, high)
        y = jnp.mod(x_safe - low_safe, span)  # in [0, span)
        x_wrap = low_safe + y  # in [low, high)

        # map positive multiples to 'high' instead of 'low'
        pos = (x_safe - low_safe) > 0  # excludes x==low, includes x==high and above
        x_wrap = jnp.where((y == 0.0) & pos, high_safe, x_wrap)

        # restore non-periodic dimensions unchanged
        return jnp.where(m, x_wrap, x_in)

    # apply wrapping only when needed
    return jax.lax.cond(has_periodic, _wrap, lambda z: z, x)


def apply_boundary_conditions_x_jax(
    x: Array,
    cfg: dict[str, Array],
) -> Array:
    """
    Applies configured boundary conditions to x-space values.

    Periodic boundary handling is applied first.
    Reflective boundary handling is applied second.
    Dimensions without either rule are left unchanged.

    Parameters:
    -----------
    x:
        input values, shape (N, D).
    cfg:
        configuration dictionary containing low, high,
        periodic_mask, and reflective_mask.

    Returns:
    --------
    Array:
        values after boundary handling, shape (N, D).
    """
    x = jnp.asarray(x)

    # bounds and boundary masks from config
    low = cfg["low"]
    high = cfg["high"]
    periodic_mask = jnp.asarray(cfg["periodic_mask"], dtype=bool)
    reflective_mask = jnp.asarray(cfg["reflective_mask"], dtype=bool)

    # check whether each type of boundary rule is present
    has_periodic = jnp.any(periodic_mask)
    has_reflective = jnp.any(reflective_mask)

    def _apply_periodic(x_in: Array) -> Array:
        """
        Applies periodic boundary handling.

        Parameters:
        -----------
        x_in:
            input values, shape (N, D).

        Returns:
        --------
        Array:
            periodically wrapped values.
        """
        # apply periodic boundary rule
        return apply_periodic_boundary_conditions_x_jax(x_in, low, high, periodic_mask)

    def _apply_reflective(x_in: Array) -> Array:
        """
        Applies reflective boundary handling.

        Parameters:
        -----------
        x_in:
            input values, shape (N, D).

        Returns:
        --------
        Array:
            reflectively corrected values.
        """
        # apply reflective boundary rule
        return apply_reflective_boundary_conditions_x_jax(
            x_in, low, high, reflective_mask
        )

    # apply periodic first
    x1 = jax.lax.cond(has_periodic, _apply_periodic, lambda z: z, x)
    # then apply reflective second
    x2 = jax.lax.cond(has_reflective, _apply_reflective, lambda z: z, x1)

    return x2
