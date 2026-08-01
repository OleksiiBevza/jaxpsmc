from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from ..tools_jax import systematic_resample_jax_size
from .constants_jax import _ECONVERGED, _EVALUEERR


#################################################################
# 2. RESAMPLE
#################################################################
@partial(jax.jit, static_argnames=("n_active", "reset_weights"))
def resample_particles_jax(
    current_particles,
    *,
    key,
    n_active: int,
    method_code: jnp.int32,
    reset_weights: bool = True,
):
    """
    Resamples particles from the current weighted particle set.

    The function draws n_active particles according to their weights.
    It supports multinomial resampling and systematic resampling.
    After resampling, weights can be reset to a uniform distribution.

    Parameters:
    -----------
    current_particles:
        dictionary with particle arrays and weights.
        Expected keys are "u", "x", "logdetj", "logl",
        "logp", "blobs", and "weights".
    key:
        JAX random key used for resampling.
    n_active:
        number of particles to draw.
        This is static for JAX compilation.
    method_code:
        integer code selecting the resampling method.
        0 means multinomial resampling.
        1 means systematic resampling.
    reset_weights:
        if True, reset output weights to uniform values.
        If False, keep the selected input weights.

    Returns:
    --------
    tuple:
        new_particles:
            dictionary with resampled particles and output weights.
        status:
            resampling status code.
            Zero means success.
            A negative value means invalid weights.
        key_out:
            updated JAX random key.
    """
    # read current weights and total number of stored particles
    w = jnp.asarray(current_particles["weights"])
    n_total = w.shape[0]

    def _multinomial(args):
        """
        Performs multinomial resampling.

        Each output index is drawn independently from the categorical
        distribution defined by the particle weights.

        Parameters:
        -----------
        args:
            tuple containing input key and weights.

        Returns:
        --------
        tuple:
            sampled indices, status code, and output key.
        """
        # unpack inputs and split the random key
        key_in, weights = args
        key_out, subkey = jax.random.split(key_in)

        # validate weights before sampling
        wsum = jnp.sum(weights)
        bad = (
            (wsum <= 0)
            | (~jnp.isfinite(wsum))
            | jnp.any(~jnp.isfinite(weights))
            | jnp.any(weights < 0)
        )

        # draw indices from categorical distribution when weights are valid
        logits = jnp.where(weights > 0, jnp.log(weights), -jnp.inf)
        idx_samp = jax.random.categorical(
            subkey, logits, shape=(n_active,), axis=0
        ).astype(jnp.int32)

        # fall back to a simple repeating pattern when weights are invalid
        idx_fallback = (
            jnp.arange(n_active, dtype=jnp.int32) % jnp.int32(n_total)
        ).astype(jnp.int32)

        # select real indices or the fallback indices
        idx = jnp.where(bad, idx_fallback, idx_samp)
        status = jnp.where(bad, _EVALUEERR, _ECONVERGED)
        return idx, status, key_out

    def _systematic(args):
        """
        Performs systematic resampling.

        Systematic resampling uses one random offset and an evenly spaced
        grid over the cumulative weight distribution. It usually has lower
        variance than multinomial resampling.

        Parameters:
        -----------
        args:
            tuple containing input key and weights.

        Returns:
        --------
        tuple:
            sampled indices, status code, and output key.
        """
        key_in, weights = args
        # idx, status, key_out = systematic_resample_jax(weights, key=key_in, size=n_active)
        idx, status, key_out = systematic_resample_jax_size(
            weights, key=key_in, size=n_active
        )
        return idx.astype(jnp.int32), status.astype(jnp.int32), key_out

    # select resampling method from integer code
    idx, status, key_out = lax.switch(
        method_code.astype(jnp.int32),
        (_multinomial, _systematic),
        (key, w),
    )

    # gather resampled particle arrays
    u_out = jnp.take(current_particles["u"], idx, axis=0)
    x_out = jnp.take(current_particles["x"], idx, axis=0)
    logdetj_out = jnp.take(current_particles["logdetj"], idx, axis=0)
    logl_out = jnp.take(current_particles["logl"], idx, axis=0)
    logp_out = jnp.take(current_particles["logp"], idx, axis=0)
    blobs_out = jnp.take(current_particles["blobs"], idx, axis=0)

    # reset weights to uniform or keep resampled weights
    w_res = jnp.take(w, idx, axis=0)
    w_uni = jnp.full(
        (n_active,),
        jnp.asarray(1.0, w.dtype) / jnp.asarray(n_active, w.dtype),
        dtype=w.dtype,
    )
    w_out = lax.cond(
        jnp.asarray(reset_weights), lambda _: w_uni, lambda _: w_res, operand=None
    )

    # build new particle dictionary
    new_particles = {
        "u": u_out,
        "x": x_out,
        "logdetj": logdetj_out,
        "logl": logl_out,
        "logp": logp_out,
        "weights": w_out,
        "blobs": blobs_out,
    }
    return new_particles, status, key_out
