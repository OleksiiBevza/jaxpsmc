from __future__ import annotations
from typing import Optional, Tuple
from jax import Array


# helpers assume the flow is used to move between u-space and theta-space
def _flow_u_to_theta_jax(
    flow, u: Array, condition: Optional[Array] = None
) -> Tuple[Array, Array]:
    """
    Maps values from u-space to theta-space using the flow.

    The sampler works with several coordinate systems.
    The variable u is the scaled latent representation.
    The variable theta is the geometry space used by the pCN proposal.

    This function applies the forward flow transformation.
    It also returns the log determinant needed to correct densities
    after changing coordinates.

    Parameters:
    -----------
    flow:
        flow object used for the coordinate transformation.
        It must have flow.bijection.transform_and_log_det.
    u:
        input values in u-space, shape (D,) for one particle
        or shape (N, D) for many particles.
    condition:
        optional conditioning value passed to the flow.
        Use None when the flow is unconditional.

    Returns:
    --------
    theta:
        transformed values in theta-space.
    logdet:
        log absolute determinant for the change of variables.
        The sign is changed so it represents the needed correction
        in the u-to-theta direction used by this sampler.
    """
    theta, fwd_logdet = flow.bijection.transform_and_log_det(u, condition)
    return theta, -fwd_logdet


def _flow_theta_to_u_jax(
    flow, theta: Array, condition: Optional[Array] = None
) -> Tuple[Array, Array]:
    """
    Maps values from theta-space back to u-space using the flow.

    The pCN proposal is built in theta-space.
    After proposing a new theta value, the sampler must convert it
    back to u-space before applying the scaler and likelihood.

    This function applies the inverse flow transformation.
    It also returns the log determinant needed for the MH ratio.

    Parameters:
    -----------
    flow:
        flow object used for the coordinate transformation.
        It must have flow.bijection.inverse_and_log_det.
    theta:
        input values in theta-space, shape (D,) for one particle
        or shape (N, D) for many particles.
    condition:
        optional conditioning value passed to the flow.
        Use None when the flow is unconditional.

    Returns:
    --------
    u:
        transformed values in u-space.
    logdet:
        log absolute determinant for the inverse flow transformation.
    """
    u, inv_logdet = flow.bijection.inverse_and_log_det(theta, condition)
    return u, inv_logdet
