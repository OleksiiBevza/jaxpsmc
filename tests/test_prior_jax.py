import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.prior_jax import (
    NORMAL,
    UNIFORM,
    Prior,
    _logpdf_one_dim,
    _normal_logpdf,
    _normal_sample,
    _sample_one_dim,
    _support_bounds,
    _uniform_logpdf,
    _uniform_sample,
)


class PriorTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(0)

    def _prior(self):
        kinds = jnp.array([0, 1, 0], dtype=jnp.int32)
        params = jnp.array(
            [
                [0.0, 1.0],    # normal(loc=0, scale=1)
                [-2.0, 2.0],   # uniform(low=-2, high=2)
                [1.0, 0.5],    # normal(loc=1, scale=0.5)
            ],
            dtype=jnp.float32,
        )
        return Prior.create(kinds, params)

    def _manual_normal(self, x, loc, scale):
        x = np.asarray(x, dtype=np.float64)
        z = (x - loc) / scale
        return -0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(scale) + z * z)

    def _manual_logpdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        out = (
            self._manual_normal(x[:, 0], 0.0, 1.0)
            + self._manual_normal(x[:, 2], 1.0, 0.5)
            - np.log(4.0)
        )
        out = np.where((x[:, 1] >= -2.0) & (x[:, 1] <= 2.0), out, -np.inf)
        return out

    def test_constants(self):
        assert int(NORMAL) == 0
        assert int(UNIFORM) == 1
        assert NORMAL.dtype == jnp.int32
        assert UNIFORM.dtype == jnp.int32

    def test_create(self):
        prior = self._prior()

        assert isinstance(prior, Prior)
        assert prior.dim == 3
        assert prior.kinds.shape == (3,)
        assert prior.params.shape == (3, 2)
        assert prior.kinds.dtype == jnp.int32
        assert prior.params.dtype == jnp.float32

        np.testing.assert_array_equal(prior.kinds, jnp.array([0, 1, 0], dtype=jnp.int32))
        np.testing.assert_allclose(
            prior.params,
            jnp.array([[0.0, 1.0], [-2.0, 2.0], [1.0, 0.5]], dtype=jnp.float32),
        )

    def test_tree(self):
        prior = self._prior()

        leaves, treedef = jax.tree_util.tree_flatten(prior)
        out = jax.tree_util.tree_unflatten(treedef, leaves)

        assert isinstance(out, Prior)
        assert len(leaves) == 2
        np.testing.assert_array_equal(out.kinds, prior.kinds)
        np.testing.assert_allclose(out.params, prior.params)
        assert out.dim == prior.dim

    @chex.all_variants(with_pmap=False)
    def test_bounds(self):
        prior = self._prior()

        bounds = self.variant(lambda p: p.bounds())(prior)

        expected = jnp.array(
            [
                [-jnp.inf, jnp.inf],
                [-2.0, 2.0],
                [-jnp.inf, jnp.inf],
            ],
            dtype=bounds.dtype,
        )
        np.testing.assert_allclose(bounds, expected)

    def test_helpers(self):
        normal_params = jnp.array([1.0, 2.0], dtype=jnp.float32)
        x = jnp.array([-1.0, 1.0, 3.0], dtype=jnp.float32)

        out_normal = _normal_logpdf(normal_params, x)
        expected_normal = self._manual_normal(np.asarray(x), loc=1.0, scale=2.0)
        np.testing.assert_allclose(out_normal, expected_normal, rtol=1e-6, atol=1e-6)

        uniform_params = jnp.array([-2.0, 2.0], dtype=jnp.float32)
        y = jnp.array([-3.0, -2.0, 0.0, 2.0, 3.0], dtype=jnp.float32)

        out_uniform = _uniform_logpdf(uniform_params, y)
        expected_uniform = jnp.array(
            [-jnp.inf, -jnp.log(4.0), -jnp.log(4.0), -jnp.log(4.0), -jnp.inf],
            dtype=jnp.float32,
        )
        np.testing.assert_allclose(out_uniform, expected_uniform, rtol=1e-6, atol=1e-6)

    def test_support(self):
        normal = _support_bounds(NORMAL, jnp.array([0.0, 1.0], dtype=jnp.float32))
        uniform = _support_bounds(UNIFORM, jnp.array([-1.5, 2.5], dtype=jnp.float32))

        np.testing.assert_allclose(normal, jnp.array([-jnp.inf, jnp.inf]))
        np.testing.assert_allclose(uniform, jnp.array([-1.5, 2.5]))

    def test_onedim(self):
        x = jnp.array([-2.0, 0.0, 2.0, 4.0], dtype=jnp.float32)
        params = jnp.array([-2.0, 2.0], dtype=jnp.float32)

        out = _logpdf_one_dim(UNIFORM, params, x)
        expected = jnp.array(
            [-jnp.log(4.0), -jnp.log(4.0), -jnp.log(4.0), -jnp.inf],
            dtype=jnp.float32,
        )
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_logpdf(self):
        prior = self._prior()
        x = jnp.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, -2.0, 1.5],
                [0.0, 2.0, 0.5],
                [0.0, 3.0, 1.0],
            ],
            dtype=jnp.float32,
        )

        out = self.variant(lambda p, z: p.logpdf(z))(prior, x)
        expected = self._manual_logpdf(x)

        assert out.shape == (4,)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
        assert bool(jnp.isneginf(out[-1]))

    @chex.all_variants(with_pmap=False)
    def test_logpdf1(self):
        prior = self._prior()
        x = jnp.array([1.0, -2.0, 1.5], dtype=jnp.float32)

        one = self.variant(lambda p, z: p.logpdf1(z))(prior, x)
        batch = prior.logpdf(x[jnp.newaxis, :])[0]

        np.testing.assert_allclose(one, batch, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_sample(self):
        prior = self._prior()

        samples = self.variant(lambda key: prior.sample(key, n=32))(self.key)

        assert samples.shape == (32, 3)
        assert jnp.issubdtype(samples.dtype, jnp.floating)
        assert bool(jnp.all(jnp.isfinite(samples[:, 0])))
        assert bool(jnp.all(jnp.isfinite(samples[:, 2])))
        assert bool(jnp.all(samples[:, 1] >= -2.0))
        assert bool(jnp.all(samples[:, 1] <= 2.0))

    @chex.all_variants(with_pmap=False)
    def test_sample1(self):
        prior = self._prior()

        one = self.variant(lambda key: prior.sample1(key))(self.key)
        batch_one = prior.sample(self.key, n=1)[0]

        assert one.shape == (3,)
        np.testing.assert_allclose(one, batch_one, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_repro(self):
        prior = self._prior()

        def run(key):
            return prior.sample(key, n=16)

        out1 = self.variant(run)(self.key)
        out2 = self.variant(run)(self.key)

        np.testing.assert_allclose(out1, out2, rtol=0.0, atol=0.0)

    def test_sample_helpers(self):
        normal_params = jnp.array([2.0, 0.25], dtype=jnp.float32)
        uniform_params = jnp.array([-1.0, 1.0], dtype=jnp.float32)

        normal = _normal_sample(self.key, normal_params, n=5)
        uniform = _uniform_sample(self.key, uniform_params, n=5)
        via_switch = _sample_one_dim(self.key, UNIFORM, uniform_params, n=5)

        assert normal.shape == (5,)
        assert uniform.shape == (5,)
        assert via_switch.shape == (5,)
        assert bool(jnp.all(jnp.isfinite(normal)))
        assert bool(jnp.all(uniform >= -1.0))
        assert bool(jnp.all(uniform <= 1.0))
        np.testing.assert_allclose(uniform, via_switch, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    absltest.main()
