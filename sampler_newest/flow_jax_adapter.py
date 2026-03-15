from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import inspect

import jax
import jax.numpy as jnp
from jax import lax

import equinox as eqx
import optax
from paramax import unwrap

from flowjax.flows import masked_autoregressive_flow
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal

import numpy as np
from tqdm import tqdm





Array = jax.Array


def create_flow(
    key: Array,
    n_dim: int,
    transforms: int = 6,
    knots: int = 8,
    interval: float = 4.0,
    nn_width: int = 128,
    nn_depth: int = 3,
):
    """
    FlowJAX flow constructor.
    """
    sig = inspect.signature(masked_autoregressive_flow)
    params = sig.parameters

    kwargs = dict(
        key=key,
        base_dist=Normal(jnp.zeros(n_dim)),
        transformer=RationalQuadraticSpline(knots=knots, interval=interval),
        nn_width=nn_width,
        nn_depth=nn_depth,
    )

    if "transforms" in params:
        kwargs["transforms"] = transforms
    elif "flow_layers" in params:
        kwargs["flow_layers"] = transforms
    else:
        raise TypeError(
            "Unsupported FlowJAX signature for masked_autoregressive_flow. "
            f"Detected signature: {sig}"
        )

    return masked_autoregressive_flow(**kwargs)


def _sanitize_weights(weights: Array, min_weight: float) -> Array:
    """
    Keep shape fixed.
    Invalid / tiny weights are zeroed rather than filtered out.
    """
    w = jnp.asarray(weights)
    thr = jnp.asarray(min_weight, dtype=w.dtype)
    good = jnp.isfinite(w) & (w > thr)
    return jnp.where(good, w, jnp.asarray(0.0, dtype=w.dtype))


def weighted_maximum_likelihood(
    params,
    static,
    x: Array,
    weights: Array,
    *,
    min_weight: float = 1e-12,
) -> Array:
    """
    Weighted negative log-likelihood with fixed-shape masking.
    """
    dist = unwrap(eqx.combine(params, static))
    logp = dist.log_prob(x)  # shape (N,)

    w = _sanitize_weights(weights, min_weight=min_weight).astype(logp.dtype)
    wsum = jnp.sum(w)

    # If a batch has no valid mass, return zero loss contribution.
    def _good_branch(op):
        logp_, w_ = op
        w_ = w_ / wsum
        return -(w_ * logp_).sum()

    def _bad_branch(op):
        logp_, _w = op
        return jnp.asarray(0.0, dtype=logp_.dtype)

    return lax.cond(wsum > 0, _good_branch, _bad_branch, (logp, w))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FlowJAXAdapter:
    """
    JAX-friendly flow state.

    params:   differentiable/trainable leaves
    static:   non-trainable part of the flow architecture
    opt_state: optimizer state carried through SMC
    dim:      latent dimension
    """

    params: Any
    static: Any
    opt_state: Any
    dim: int


    def tree_flatten(self):        
        return (self.params, self.opt_state), (self.static, self.dim)

    
    @classmethod
    def tree_unflatten(cls, aux, children):
        static, dim = aux
        params, opt_state = children
        return cls(
            params=params,
            static=static,
            opt_state=opt_state,
            dim=dim,            
        )

    @property
    def dist(self):
        return unwrap(eqx.combine(self.params, self.static))

    @property
    def bijection(self):
        return self.dist.bijection

    def sample(self, key: Array, n: int, condition: Optional[Array] = None) -> Array:
        del condition
        return self.dist.sample(key, (n,))
    

    def transform_batch(
        self,
        x: Array,
        condition: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        x = jnp.asarray(x)

        def _single(xi: Array) -> Tuple[Array, Array]:
            return self.bijection.transform_and_log_det(xi, condition)

        return jax.vmap(_single, in_axes=0, out_axes=(0, 0))(x)






    def fit(
        self,
        key: Array,
        data: Array,
        weights: Optional[Array] = None,
        *,
        learning_rate: float = 5e-3,
        max_epochs: int = 200,
        batch_size: int = 256,
        min_weight: float = 1e-12        
    ) -> Tuple["FlowJAXAdapter", Array]:
        """
        Pure JAX training: no NumPy, no device_get, no host-side fit_to_data.

        Returns
        -------
        new_flow : FlowJAXAdapter
        losses   : jax.Array, shape (max_epochs,)
        """
        x = jnp.asarray(data)

        if weights is None:
            n = x.shape[0]
            weights = jnp.full(
                (n,),
                jnp.asarray(1.0, dtype=x.dtype) / jnp.asarray(n, dtype=x.dtype),
                dtype=x.dtype,
            )
        else:
            weights = jnp.asarray(weights, dtype=x.dtype)

        optimizer = optax.adam(learning_rate)
        batch_size = int(min(batch_size, x.shape[0]))
        max_epochs = int(max_epochs)


        
        w_all = _sanitize_weights(weights, min_weight=min_weight)
        total_mass = jnp.sum(w_all)

        def _train_batch(
            flow_state: "FlowJAXAdapter",
            x_batch: Array,
            w_batch: Array,
        ) -> Tuple["FlowJAXAdapter", Array]:
            loss_fn = lambda params: weighted_maximum_likelihood(
                params,
                flow_state.static,
                x_batch,
                w_batch,
                min_weight=min_weight,
            )

            loss, grads = eqx.filter_value_and_grad(loss_fn)(flow_state.params)
            updates, new_opt_state = optimizer.update(
                grads,
                flow_state.opt_state,
                flow_state.params,
            )
            new_params = eqx.apply_updates(flow_state.params, updates)
     

            return (
                FlowJAXAdapter(
                    params=new_params,
                    static=flow_state.static,
                    opt_state=new_opt_state,
                    dim=flow_state.dim,                    
                ),
                    loss,
            )
        



        def _train_one_epoch(
            flow_state: "FlowJAXAdapter",
            key_epoch: Array,
        ) -> Tuple["FlowJAXAdapter", Array]:
            n_items = x.shape[0]
            n_batches = (n_items + batch_size - 1) // batch_size

            perm = jax.random.permutation(key_epoch, n_items)
            total = n_batches * batch_size
            base = jnp.arange(total, dtype=jnp.int32)
            batch_idx = perm[base % n_items].reshape((n_batches, batch_size))

            x_batches = x[batch_idx]
            w_batches = w_all[batch_idx]

            def _scan_body(flow_state_inner, batch):
                xb, wb = batch
                wsum = jnp.sum(wb)

                def _do_train(op):
                    fs, xb2, wb2 = op
                    return _train_batch(fs, xb2, wb2)

                def _skip_train(op):
                    fs, xb2, wb2 = op
                    del xb2, wb2
                    return fs, jnp.asarray(0.0, dtype=x.dtype)

                return lax.cond(
                    wsum > 0,
                    _do_train,
                    _skip_train,
                    (flow_state_inner, xb, wb),
                )

            flow_state, batch_losses = lax.scan(
                _scan_body,
                flow_state,
                (x_batches, w_batches),
            )

            return flow_state, jnp.mean(batch_losses)      


        def _do_fit(op):
            flow_state0, key0 = op
            epoch_keys = jax.random.split(key0, max_epochs)

            def _epoch_body(flow_state_inner, key_epoch):
                flow_state_inner, epoch_loss = _train_one_epoch(flow_state_inner, key_epoch)
                return flow_state_inner, epoch_loss

            flow_state_f, losses = lax.scan(
            _epoch_body,
            flow_state0,
            epoch_keys,
            )
            return flow_state_f, losses



        def _skip_fit(op):
            flow_state0, _key0 = op
            losses = jnp.full((max_epochs,), jnp.nan, dtype=x.dtype)
            return flow_state0, losses

        return lax.cond(
            total_mass > 0,
            _do_fit,
            _skip_fit,
            (self, key),
        )


def init_flow_state(
    key: Array,
    n_dim: int,
    *,
    transforms: int = 6,
    knots: int = 8,
    interval: float = 4.0,
    nn_width: int = 128,
    nn_depth: int = 3,
    learning_rate: float = 5e-3    
) -> FlowJAXAdapter:
    """
    Create the flow and initialize optimizer state.
    """
    dist = create_flow(
        key=key,
        n_dim=n_dim,
        transforms=transforms,
        knots=knots,
        interval=interval,
        nn_width=nn_width,
        nn_depth=nn_depth,
    )

    params, static = eqx.partition(dist, eqx.is_inexact_array)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    return FlowJAXAdapter(
        params=params,
        static=static,
        opt_state=opt_state,
        dim=n_dim        
    )






