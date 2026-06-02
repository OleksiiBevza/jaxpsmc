import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.geometry.geometry_jax import (
    Geometry,
    _cov_unweighted,
    _cov_weighted_aweights,
    _sanitize_nu,
    geometry_fit_jax,
)


class GeometryTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(7)
        self.theta = jnp.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [2.0, 2.0],
                [1.0, 3.0],
                [3.0, 1.0],
            ],
            dtype=jnp.float64,
        )
        self.weights = jnp.array(
            [0.10, 0.15, 0.20, 0.25, 0.20, 0.10],
            dtype=jnp.float64,
        )
        self.geom0 = Geometry.init(2, dtype=jnp.float64)

    def test_init(self):
        geom = Geometry.init(3, dtype=jnp.float32)

        assert geom.normal_mean.shape == (3,)
        assert geom.normal_cov.shape == (3, 3)
        assert geom.t_mean.shape == (3,)
        assert geom.t_cov.shape == (3, 3)
        assert geom.t_nu.shape == ()

        assert geom.normal_mean.dtype == jnp.float32
        assert geom.normal_cov.dtype == jnp.float32
        assert geom.t_mean.dtype == jnp.float32
        assert geom.t_cov.dtype == jnp.float32
        assert geom.t_nu.dtype == jnp.float32

        np.testing.assert_allclose(geom.normal_mean, jnp.zeros((3,)))
        np.testing.assert_allclose(geom.normal_cov, jnp.zeros((3, 3)))
        np.testing.assert_allclose(geom.t_mean, jnp.zeros((3,)))
        np.testing.assert_allclose(geom.t_cov, jnp.zeros((3, 3)))
        np.testing.assert_allclose(geom.t_nu, 1e6)

    def test_tree(self):
        geom = Geometry(
            normal_mean=jnp.array([1.0, 2.0]),
            normal_cov=jnp.eye(2),
            t_mean=jnp.array([3.0, 4.0]),
            t_cov=2.0 * jnp.eye(2),
            t_nu=jnp.array(5.0),
        )

        leaves, treedef = jax.tree_util.tree_flatten(geom)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)

        assert isinstance(restored, Geometry)
        np.testing.assert_allclose(restored.normal_mean, geom.normal_mean)
        np.testing.assert_allclose(restored.normal_cov, geom.normal_cov)
        np.testing.assert_allclose(restored.t_mean, geom.t_mean)
        np.testing.assert_allclose(restored.t_cov, geom.t_cov)
        np.testing.assert_allclose(restored.t_nu, geom.t_nu)

    @chex.all_variants(with_pmap=False)
    def test_unweighted(self):
        theta = jnp.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=jnp.float64,
        )

        mu, cov = self.variant(
            lambda x: _cov_unweighted(x, jitter=jnp.array(0.0, dtype=x.dtype))
        )(theta)

        expected_mu = jnp.array([2.0 / 3.0, 2.0 / 3.0], dtype=jnp.float64)
        expected_cov = jnp.array(
            [
                [4.0 / 3.0, -2.0 / 3.0],
                [-2.0 / 3.0, 4.0 / 3.0],
            ],
            dtype=jnp.float64,
        )

        np.testing.assert_allclose(mu, expected_mu, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cov, expected_cov, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cov, cov.T, rtol=1e-12, atol=1e-12)

    @chex.all_variants(with_pmap=False)
    def test_jitter(self):
        theta = jnp.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=jnp.float64,
        )
        jitter = jnp.array(1e-3, dtype=jnp.float64)

        mu, cov = self.variant(
            lambda x: _cov_unweighted(x, jitter=jitter)
        )(theta)

        np.testing.assert_allclose(mu, jnp.array([1.0, 1.0]))
        np.testing.assert_allclose(cov, jitter * jnp.eye(2), rtol=1e-12)

    @chex.all_variants(with_pmap=False)
    def test_weighted(self):
        theta = jnp.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=jnp.float64,
        )
        weights = jnp.array([0.5, 0.25, 0.25], dtype=jnp.float64)

        mu, cov = self.variant(
            lambda x, w: _cov_weighted_aweights(
                x,
                w,
                jitter=jnp.array(0.0, dtype=x.dtype),
            )
        )(theta, weights)

        expected_mu = jnp.array([0.5, 0.5], dtype=jnp.float64)
        expected_cov = jnp.array(
            [
                [1.2, -0.4],
                [-0.4, 1.2],
            ],
            dtype=jnp.float64,
        )

        np.testing.assert_allclose(mu, expected_mu, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cov, expected_cov, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cov, cov.T, rtol=1e-12, atol=1e-12)

    @chex.all_variants(with_pmap=False)
    def test_badweights(self):
        theta = jnp.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=jnp.float64,
        )
        bad = jnp.array([1.0, -1.0, 1.0], dtype=jnp.float64)

        mu_bad, cov_bad = self.variant(
            lambda x, w: _cov_weighted_aweights(
                x,
                w,
                jitter=jnp.array(0.0, dtype=x.dtype),
            )
        )(theta, bad)

        mu_ref, cov_ref = _cov_unweighted(
            theta,
            jitter=jnp.array(0.0, dtype=theta.dtype),
        )

        np.testing.assert_allclose(mu_bad, mu_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cov_bad, cov_ref, rtol=1e-12, atol=1e-12)

    @chex.all_variants(with_pmap=False)
    def test_zero(self):
        theta = jnp.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=jnp.float64,
        )
        bad = jnp.zeros((3,), dtype=jnp.float64)

        mu_bad, cov_bad = self.variant(
            lambda x, w: _cov_weighted_aweights(
                x,
                w,
                jitter=jnp.array(0.0, dtype=x.dtype),
            )
        )(theta, bad)

        mu_ref, cov_ref = _cov_unweighted(
            theta,
            jitter=jnp.array(0.0, dtype=theta.dtype),
        )

        np.testing.assert_allclose(mu_bad, mu_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cov_bad, cov_ref, rtol=1e-12, atol=1e-12)

    @chex.all_variants(with_pmap=False)
    def test_nu(self):
        nu = jnp.array([2.5, jnp.inf, -jnp.inf, jnp.nan], dtype=jnp.float64)

        out = self.variant(
            lambda x: _sanitize_nu(x, nu_cap=100.0)
        )(nu)

        expected = jnp.array([2.5, 100.0, 100.0, 100.0], dtype=jnp.float64)
        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)

    @chex.all_variants(with_pmap=False)
    def test_direct(self):
        geom, key_out, status = self.variant(
            lambda key: geometry_fit_jax(
                self.geom0,
                self.theta,
                self.weights,
                use_weights=jnp.asarray(False),
                key=key,
                jitter=0.0,
            )
        )(self.key)

        mu_ref, cov_ref = _cov_unweighted(
            self.theta,
            jitter=jnp.array(0.0, dtype=self.theta.dtype),
        )

        np.testing.assert_allclose(geom.normal_mean, mu_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(geom.normal_cov, cov_ref, rtol=1e-10, atol=1e-10)
        #np.testing.assert_allclose(key_out, self.key)
        np.testing.assert_array_equal(
            jax.random.key_data(key_out),
            jax.random.key_data(self.key),
        )
        assert int(status) == 0

        assert geom.t_mean.shape == (2,)
        assert geom.t_cov.shape == (2, 2)
        assert geom.t_nu.shape == ()
        assert bool(jnp.all(jnp.isfinite(geom.t_mean)))
        assert bool(jnp.all(jnp.isfinite(geom.t_cov)))
        assert bool(jnp.isfinite(geom.t_nu))

    @chex.all_variants(with_pmap=False)
    def test_fitw(self):
        geom, key_out, status = self.variant(
            lambda key: geometry_fit_jax(
                self.geom0,
                self.theta,
                self.weights,
                use_weights=jnp.asarray(True),
                key=key,
                jitter=0.0,
            )
        )(self.key)

        mu_ref, cov_ref = _cov_weighted_aweights(
            self.theta,
            self.weights,
            jitter=jnp.array(0.0, dtype=self.theta.dtype),
        )

        np.testing.assert_allclose(geom.normal_mean, mu_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(geom.normal_cov, cov_ref, rtol=1e-10, atol=1e-10)
        assert int(status) == 0

        assert key_out.shape == self.key.shape
        assert geom.t_mean.shape == (2,)
        assert geom.t_cov.shape == (2, 2)
        assert geom.t_nu.shape == ()
        assert bool(jnp.all(jnp.isfinite(geom.t_mean)))
        assert bool(jnp.all(jnp.isfinite(geom.t_cov)))
        assert bool(jnp.isfinite(geom.t_nu))

    @chex.all_variants(with_pmap=False)
    def test_repeat(self):
        run = self.variant(
            lambda key: geometry_fit_jax(
                self.geom0,
                self.theta,
                self.weights,
                use_weights=jnp.asarray(True),
                key=key,
                jitter=1e-9,
            )
        )

        geom1, key1, status1 = run(self.key)
        geom2, key2, status2 = run(self.key)

        np.testing.assert_allclose(geom1.normal_mean, geom2.normal_mean)
        np.testing.assert_allclose(geom1.normal_cov, geom2.normal_cov)
        np.testing.assert_allclose(geom1.t_mean, geom2.t_mean)
        np.testing.assert_allclose(geom1.t_cov, geom2.t_cov)
        np.testing.assert_allclose(geom1.t_nu, geom2.t_nu)
        #np.testing.assert_allclose(key1, key2)
        np.testing.assert_array_equal(
            jax.random.key_data(key1),
            jax.random.key_data(key2),
        )
        assert int(status1) == int(status2)

    @chex.all_variants(with_pmap=False)
    def test_dtype(self):
        theta = self.theta.astype(jnp.float32)
        weights = self.weights.astype(jnp.float32)
        geom0 = Geometry.init(2, dtype=jnp.float32)

        geom, _, status = self.variant(
            lambda key: geometry_fit_jax(
                geom0,
                theta,
                weights,
                use_weights=jnp.asarray(False),
                key=key,
                jitter=0.0,
            )
        )(self.key)

        assert geom.normal_mean.dtype == jnp.float32
        assert geom.normal_cov.dtype == jnp.float32
        assert geom.t_mean.dtype == jnp.float32
        assert geom.t_cov.dtype == jnp.float32
        assert geom.t_nu.dtype == jnp.float32
        assert int(status) == 0

    @chex.all_variants(with_pmap=False)
    def test_one(self):
        theta = jnp.array([[2.0, -1.0]], dtype=jnp.float64)
        weights = jnp.array([1.0], dtype=jnp.float64)

        mu, cov = self.variant(
            lambda x: _cov_unweighted(
                x,
                jitter=jnp.array(1e-4, dtype=x.dtype),
            )
        )(theta)

        np.testing.assert_allclose(mu, theta[0])
        np.testing.assert_allclose(cov, 1e-4 * jnp.eye(2), rtol=1e-12)


if __name__ == "__main__":
    absltest.main()
