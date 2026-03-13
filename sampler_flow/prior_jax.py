from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import lax, random

# Distribution kinds
NORMAL = jnp.int32(0)
UNIFORM = jnp.int32(1)


# distribution primitives 
def _normal_logpdf(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    loc, scale = params[0], params[1]
    z = (x - loc) / scale
    return -0.5 * (jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(scale) + z * z)


def _uniform_logpdf(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    low, high = params[0], params[1]
    logZ = jnp.log(high - low)
    in_support = (x >= low) & (x <= high)
    return jnp.where(in_support, -logZ, -jnp.inf)


def _normal_sample(key: jax.Array, params: jnp.ndarray, n: int) -> jnp.ndarray:
    loc, scale = params[0], params[1]
    return loc + scale * random.normal(key, shape=(n,))


def _uniform_sample(key: jax.Array, params: jnp.ndarray, n: int) -> jnp.ndarray:
    low, high = params[0], params[1]
    return random.uniform(key, shape=(n,), minval=low, maxval=high)


def _support_bounds(kind: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    def normal_bounds(_p):
        return jnp.array([-jnp.inf, jnp.inf])

    def uniform_bounds(p):
        return jnp.array([p[0], p[1]])

    return lax.switch(kind, [normal_bounds, uniform_bounds], params)


def _logpdf_one_dim(kind: jnp.ndarray, params: jnp.ndarray, x_col: jnp.ndarray) -> jnp.ndarray:
    return lax.switch(kind, [_normal_logpdf, _uniform_logpdf], params, x_col)


def _sample_one_dim(key: jax.Array, kind: jnp.ndarray, params: jnp.ndarray, n: int) -> jnp.ndarray:
    return lax.switch(
        kind,
        [
            lambda p: _normal_sample(key, p, n),
            lambda p: _uniform_sample(key, p, n),
        ],
        params,
    )


# Prior pytree
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Prior:
    kinds: jnp.ndarray   # (D,) int32
    params: jnp.ndarray  # (D, 2)

    def tree_flatten(self):
        return (self.kinds, self.params), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kinds, params = children
        return cls(kinds=kinds, params=params)

    @property
    def dim(self) -> int:
        return int(self.kinds.shape[0])

    @staticmethod
    def create(kinds, params) -> "Prior":
        # No branching; just translate into JAX arrays
        return Prior(
            kinds=jnp.asarray(kinds, dtype=jnp.int32),
            params=jnp.asarray(params),
        )

    def bounds(self) -> jnp.ndarray:
        return jax.vmap(_support_bounds, in_axes=(0, 0))(self.kinds, self.params)

    # Batch logpdf: x is (N, D) -> (N,)
    def logpdf(self, x: jnp.ndarray) -> jnp.ndarray:
        per_dim = jax.vmap(
            _logpdf_one_dim,
            in_axes=(0, 0, 1),   # kinds (D,), params (D,2), x (N,D) => column per dim
            out_axes=1,          # (N, D)
        )(self.kinds, self.params, x)
        return jnp.sum(per_dim, axis=1)

    # Single-point logpdf: x is (D,) -> scalar
    def logpdf1(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.logpdf(x[jnp.newaxis, :])[0]

    # Batch sampling: returns (n, D)
    def sample(self, key: jax.Array, n: int) -> jnp.ndarray:
        keys = random.split(key, self.kinds.shape[0])
        return jax.vmap(
            lambda k, kind, p: _sample_one_dim(k, kind, p, n),
            in_axes=(0, 0, 0),
            out_axes=1,
        )(keys, self.kinds, self.params)

    # Single sample: returns (D,)
    def sample1(self, key: jax.Array) -> jnp.ndarray:
        return self.sample(key, n=1)[0]