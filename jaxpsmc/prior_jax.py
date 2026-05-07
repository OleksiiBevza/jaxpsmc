from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import lax, random

# kinds of distribution
NORMAL = jnp.int32(0)
UNIFORM = jnp.int32(1)



def _normal_logpdf(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the log-density of a normal distribution.

    The distribution is defined by a mean and a scale.
    The scale is the standard deviation.
    This function can evaluate one value or many values.

    Parameters:
    -----------
    params:
        array with two values: [loc, scale].
        loc is the mean.
        scale is the standard deviation.
    x:
        value or array of values where the log-density is evaluated.

    Returns:
    --------
    jnp.ndarray:
        normal log-density evaluated at x.
    """
    # define mean and SD
    loc, scale = params[0], params[1]
    # standardize x
    z = (x - loc) / scale
    # return normal log-density
    return -0.5 * (jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(scale) + z * z)


def _uniform_logpdf(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the log-density of a uniform distribution.

    The distribution is constant inside the interval [low, high].
    Values outside this interval get log-density -inf.
    This makes them impossible under the prior.

    Parameters:
    -----------
    params:
        array with two values: [low, high].
        These define the support interval.
    x:
        value or array of values where the log-density is evaluated.

    Returns:
    --------
    jnp.ndarray:
        uniform log-density evaluated at x.
        Values outside the interval return -inf.
    """
    # define the interval endpoints
    low, high = params[0], params[1]
    # compute log of the interval width
    logZ = jnp.log(high - low)
    # check whether x is inside the interval
    in_support = (x >= low) & (x <= high)
    # return log-density inside interval and -inf outside it
    return jnp.where(in_support, -logZ, -jnp.inf)


def _normal_sample(key: jax.Array, params: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Draws samples from a normal distribution.

    The function first draws standard normal samples.
    It then rescales them using the requested mean and scale.

    Parameters:
    -----------
    key:
        JAX random key used for sampling.
    params:
        array with two values: [loc, scale].
        loc is the mean.
        scale is the standard deviation.
    n:
        number of samples to draw.

    Returns:
    --------
    jnp.ndarray:
        array of samples from the normal distribution, shape (n,).
    """
    # define mean and sd
    loc, scale = params[0], params[1]
    # draw standard normal samples and rescale them
    return loc + scale * random.normal(key, shape=(n,))


def _uniform_sample(key: jax.Array, params: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Draws samples from a uniform distribution.

    The samples are drawn from the interval [low, high].
    Each value inside the interval is equally likely.

    Parameters:
    -----------
    key:
        JAX random key used for sampling.
    params:
        array with two values: [low, high].
        These define the sampling interval.
    n:
        number of samples to draw.

    Returns:
    --------
    jnp.ndarray:
        array of samples from the uniform distribution, shape (n,).
    """
    low, high = params[0], params[1]
    return random.uniform(key, shape=(n,), minval=low, maxval=high)


def _support_bounds(kind: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the support bounds for one distribution.

    The support is the set of values allowed by the distribution.
    A normal distribution allows all real values.
    A uniform distribution allows only values inside its interval.

    Parameters:
    -----------
    kind:
        integer code for the distribution type.
        NORMAL means normal distribution.
        UNIFORM means uniform distribution.
    params:
        parameter array for the distribution.
        For NORMAL, this is [loc, scale].
        For UNIFORM, this is [low, high].

    Returns:
    --------
    jnp.ndarray:
        array with two values: [lower_bound, upper_bound].
    """
    def normal_bounds(_p):
        """
        Returns the support of a normal distribution.

        A normal distribution has support on the whole real line.
        The parameter array is not needed for the bounds.

        Parameters:
        -----------
        _p:
            unused parameter array.

        Returns:
        --------
        jnp.ndarray:
            array [-inf, inf].
        """
        # normal distribution is supported on the whole real line
        return jnp.array([-jnp.inf, jnp.inf])

    def uniform_bounds(p):
        """
        Returns the support of a uniform distribution.

        The support is exactly the interval [low, high].

        Parameters:
        -----------
        p:
            parameter array with [low, high].

        Returns:
        --------
        jnp.ndarray:
            array [low, high].
        """
        # uniform distribution is supported only on [low, high]
        return jnp.array([p[0], p[1]])

    return lax.switch(kind, [normal_bounds, uniform_bounds], params)


def _logpdf_one_dim(kind: jnp.ndarray, params: jnp.ndarray, x_col: jnp.ndarray) -> jnp.ndarray:
    """
    Computes prior log-density values for one dimension.

    Each dimension can use a different distribution.
    This function selects the correct log-density formula
    using the distribution code.

    Parameters:
    -----------
    kind:
        integer code for the distribution type.
        NORMAL means normal distribution.
        UNIFORM means uniform distribution.
    params:
        parameter array for this dimension.
        For NORMAL, this is [loc, scale].
        For UNIFORM, this is [low, high].
    x_col:
        values from one dimension, shape (N,).

    Returns:
    --------
    jnp.ndarray:
        log-density values for this dimension, shape (N,).
    """
    # choose correct log-density function from the distribution code
    return lax.switch(kind, [_normal_logpdf, _uniform_logpdf], params, x_col)


def _sample_one_dim(key: jax.Array, kind: jnp.ndarray, params: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Draws samples for one prior dimension.

    The function chooses the correct sampling rule
    from the distribution code.

    Parameters:
    -----------
    key:
        JAX random key used for this dimension.
    kind:
        integer code for the distribution type.
        NORMAL means normal distribution.
        UNIFORM means uniform distribution.
    params:
        parameter array for this dimension.
        For NORMAL, this is [loc, scale].
        For UNIFORM, this is [low, high].
    n:
        number of samples to draw.

    Returns:
    --------
    jnp.ndarray:
        samples for one dimension, shape (n,).
    """
    # choose correct sampling function from distribution
    return lax.switch(
        kind,
        [
            lambda p: _normal_sample(key, p, n),
            lambda p: _uniform_sample(key, p, n),
        ],
        params,
    )



@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Prior:
    """
    Stores a factorized prior distribution across dimensions.

    A factorized prior means each dimension has its own distribution.
    The full prior density is the product of these one-dimensional densities.
    In log space, this becomes a sum over dimensions.

    The class supports two distribution types:
    normal and uniform.
    Each dimension stores one integer code telling which distribution is used.
    It also stores two parameters for that distribution.

    The class is registered as a JAX pytree.
    This allows JAX to pass it through transformations such as jit and vmap.

    Parameters:
    -----------
    kinds:
        integer distribution codes, shape (D,).
        Each entry selects the distribution for one dimension.
        NORMAL means normal distribution.
        UNIFORM means uniform distribution.
    params:
        distribution parameters, shape (D, 2).
        For a normal dimension, the row is [loc, scale].
        For a uniform dimension, the row is [low, high].

    Returns:
    --------
    Prior:
        factorized prior object with one distribution per dimension.
    """
    kinds: jnp.ndarray   # (D,) int32
    params: jnp.ndarray  # (D, 2)

    def tree_flatten(self):
        """
        Converts the Prior object into JAX pytree children.

        JAX needs this method to know which fields are arrays.
        The distribution codes and parameters are the array children.
        No auxiliary data is needed.

        Parameters:
        -----------
        None:
            this method uses the current Prior object.

        Returns:
        --------
        tuple:
            tuple containing the array children and auxiliary data.
            Auxiliary data is None here.
        """
        return (self.kinds, self.params), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Rebuilds a Prior object from JAX pytree children.

        JAX uses this after transforming or moving the object.
        The auxiliary data is unused because all information is stored
        in the array children.

        Parameters:
        -----------
        aux_data:
            auxiliary pytree data.
            It is unused here.
        children:
            tuple containing kinds and params.

        Returns:
        --------
        Prior:
            rebuilt Prior object.
        """
        kinds, params = children
        return cls(kinds=kinds, params=params)

    @property
    def dim(self) -> int:
        """
        Returns the number of prior dimensions.

        This is the number of one-dimensional distributions
        stored in the prior.

        Parameters:
        -----------
        None:
            this property uses the current Prior object.

        Returns:
        --------
        int:
            number of dimensions.
        """
        return int(self.kinds.shape[0])

    @staticmethod
    def create(kinds, params) -> "Prior":
        """
        Creates a Prior object from distribution codes and parameters.

        The inputs are converted to JAX arrays.
        The distribution codes are stored as int32 values.
        The parameter values are stored as a JAX array.

        Parameters:
        -----------
        kinds:
            distribution codes for each dimension, shape (D,).
            Use NORMAL for normal dimensions.
            Use UNIFORM for uniform dimensions.
        params:
            distribution parameters, shape (D, 2).
            For NORMAL, each row should be [loc, scale].
            For UNIFORM, each row should be [low, high].

        Returns:
        --------
        Prior:
            prior object with one distribution per dimension.
        """
        # no branching; just translate into JAX arrays
        return Prior(
            kinds=jnp.asarray(kinds, dtype=jnp.int32),
            params=jnp.asarray(params),
        )

    def bounds(self) -> jnp.ndarray:
        """
        Returns support bounds for all prior dimensions.

        For normal dimensions, the bounds are [-inf, inf].
        For uniform dimensions, the bounds are [low, high].

        Parameters:
        -----------
        None:
            this method uses the current Prior object.

        Returns:
        --------
        jnp.ndarray:
            support bounds for all dimensions, shape (D, 2).
            Column 0 contains lower bounds.
            Column 1 contains upper bounds.
        """
        return jax.vmap(_support_bounds, in_axes=(0, 0))(self.kinds, self.params)

    # Batch logpdf: x is (N, D) -> (N,)
    def logpdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the prior log-density for a batch of points.

        The input x contains many points.
        Each row is one point in D dimensions.
        The function evaluates the one-dimensional prior in each column.
        It then sums the log-density values across dimensions.

        Parameters:
        -----------
        x:
            input points, shape (N, D).
            N is the number of points.
            D must match the number of prior dimensions.

        Returns:
        --------
        jnp.ndarray:
            prior log-density for each row, shape (N,).
        """
        per_dim = jax.vmap(
            _logpdf_one_dim,
            in_axes=(0, 0, 1),   # kinds (D,), params (D,2), x (N,D) into column per dim
            out_axes=1,          # (N, D)
        )(self.kinds, self.params, x)
        # sum the per-dimension terms across columns
        return jnp.sum(per_dim, axis=1)

    # Single-point logpdf: x is (D,) -> scalar
    def logpdf1(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the prior log-density for one point.

        This is a convenience wrapper around logpdf.
        It adds a batch dimension, evaluates the batch log-density,
        and returns the first scalar result.

        Parameters:
        -----------
        x:
            one input point, shape (D,).

        Returns:
        --------
        jnp.ndarray:
            scalar prior log-density for the point.
        """
        return self.logpdf(x[jnp.newaxis, :])[0]

    # Batch sampling: returns (n, D)
    def sample(self, key: jax.Array, n: int) -> jnp.ndarray:
        """
        Draws a batch of samples from the prior.

        Each dimension is sampled independently.
        A separate random key is used for each dimension.
        The sampled columns are stacked into an array of shape (n, D).

        Parameters:
        -----------
        key:
            JAX random key used for sampling.
        n:
            number of samples to draw.

        Returns:
        --------
        jnp.ndarray:
            sampled points from the prior, shape (n, D).
        """
        keys = random.split(key, self.kinds.shape[0])
        # sample each dimension separately and stack the results by column
        return jax.vmap(
            lambda k, kind, p: _sample_one_dim(k, kind, p, n),
            in_axes=(0, 0, 0),
            out_axes=1,
        )(keys, self.kinds, self.params)

    # Single sample: returns (D,)
    def sample1(self, key: jax.Array) -> jnp.ndarray:
        """
        Draws one sample from the prior.

        This is a convenience wrapper around sample.
        It draws a batch of size one and returns the first row.

        Parameters:
        -----------
        key:
            JAX random key used for sampling.

        Returns:
        --------
        jnp.ndarray:
            one sampled point from the prior, shape (D,).
        """
        return self.sample(key, n=1)[0]
    




    