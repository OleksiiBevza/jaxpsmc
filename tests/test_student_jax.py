import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.student_jax import (
    _fit_mvstud_core,
    _init_mu_sigma,
    _nu_fixed_point_objective,
    _opt_nu_bisect,
    fit_mvstud_jax,
)


class StudentTest(chex.TestCase):
    def _data(self, dtype=jnp.float64):
        return jnp.asarray(
            [
                [-1.0, -0.5],
                [-0.5, 0.0],
                [0.0, 0.2],
                [0.5, 0.7],
                [1.0, 1.2],
                [4.0, -3.0],
            ],
            dtype=dtype,
        )

    def _clean_data(self, dtype=jnp.float64):
        return jnp.asarray(
            [
                [-1.0, -1.0],
                [-0.5, -0.4],
                [0.0, 0.1],
                [0.4, 0.5],
                [0.9, 1.0],
                [1.2, 1.3],
            ],
            dtype=dtype,
        )

    def _manual_init(self, data):
        data_np = np.asarray(data)
        mu = np.median(data_np, axis=0)
        centered = data_np - np.mean(data_np, axis=0, keepdims=True)
        n = data_np.shape[0]
        cov_mle = centered.T @ centered / n
        var = np.var(data_np, axis=0)
        sigma = cov_mle + np.diag(var) / n
        return mu, sigma

    def test_objective(self):
        delta = jnp.asarray([0.2, 1.0, 2.5, 5.0], dtype=jnp.float64)
        dim = jnp.asarray(2.0, dtype=jnp.float64)
        nu = jnp.asarray(7.0, dtype=jnp.float64)

        out = _nu_fixed_point_objective(nu, delta, dim)

        w = (nu + dim) / (nu + delta)
        expected = (
            -jax.scipy.special.digamma(nu / 2.0)
            + jnp.log(nu / 2.0)
            + jnp.mean(jnp.log(w))
            - jnp.mean(w)
            + 1.0
            + jax.scipy.special.digamma((nu + dim) / 2.0)
            - jnp.log((nu + dim) / 2.0)
        )

        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
        assert out.shape == ()
        assert out.dtype == jnp.float64

    @chex.all_variants(with_pmap=False)
    def test_objective_jit(self):
        delta = jnp.asarray([0.1, 0.3, 1.1, 2.7], dtype=jnp.float64)
        dim = jnp.asarray(3.0, dtype=jnp.float64)
        nu = jnp.asarray(5.0, dtype=jnp.float64)

        out = self.variant(lambda n, d, p: _nu_fixed_point_objective(n, d, p))(
            nu, delta, dim
        )

        assert out.shape == ()
        assert bool(jnp.isfinite(out))

    def test_init(self):
        data = self._data()
        mu, sigma = _init_mu_sigma(data)
        expected_mu, expected_sigma = self._manual_init(data)

        np.testing.assert_allclose(mu, expected_mu, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(sigma, expected_sigma, rtol=1e-12, atol=1e-12)
        assert mu.shape == (2,)
        assert sigma.shape == (2, 2)
        assert mu.dtype == data.dtype
        assert sigma.dtype == data.dtype
        np.testing.assert_allclose(sigma, sigma.T, rtol=1e-12, atol=1e-12)

    def test_init_one(self):
        data = jnp.asarray([[2.0, -1.0]], dtype=jnp.float64)
        mu, sigma = _init_mu_sigma(data)

        np.testing.assert_allclose(mu, jnp.asarray([2.0, -1.0], dtype=jnp.float64))
        np.testing.assert_allclose(sigma, jnp.zeros((2, 2), dtype=jnp.float64))

    @chex.all_variants(with_pmap=False)
    def test_nu_bisect(self):
        delta = jnp.asarray([0.1, 0.5, 1.0, 3.0, 8.0], dtype=jnp.float64)
        nu_old = jnp.asarray(20.0, dtype=jnp.float64)

        nu, status, nu_is_inf = self.variant(
            lambda d, n: _opt_nu_bisect(
                d,
                2,
                n,
                xtol=jnp.asarray(2e-12, dtype=d.dtype),
                bisect_maxiter=jnp.asarray(100, dtype=jnp.int64),
            )
        )(delta, nu_old)

        assert nu.shape == ()
        assert status.shape == ()
        assert nu_is_inf.shape == ()
        assert status.dtype == jnp.int64
        assert nu_is_inf.dtype == jnp.bool_
        assert bool(jnp.isfinite(nu)) or bool(nu_is_inf)
        assert float(nu) > 0.0

        if int(status) == 0 and not bool(nu_is_inf):
            obj = _nu_fixed_point_objective(nu, delta, jnp.asarray(2.0, dtype=delta.dtype))
            np.testing.assert_allclose(obj, 0.0, rtol=1e-6, atol=1e-6)

    def test_nu_fallback(self):
        delta = jnp.asarray([0.1, 0.5, 1.0, 3.0, 8.0], dtype=jnp.float64)
        nu_old = jnp.asarray(13.0, dtype=jnp.float64)

        nu, status, nu_is_inf = _opt_nu_bisect(
            delta,
            2,
            nu_old,
            xtol=jnp.asarray(2e-12, dtype=jnp.float64),
            bisect_maxiter=jnp.asarray(0, dtype=jnp.int64),
        )

        assert status.shape == ()
        assert nu.shape == ()
        assert nu_is_inf.shape == ()
        if int(status) != 0:
            np.testing.assert_allclose(nu, nu_old, rtol=1e-12, atol=1e-12)

    @chex.all_variants(with_pmap=False)
    def test_core_shapes(self):
        data = self._data()

        mu, sigma, nu, iters, status = self.variant(
            lambda x: _fit_mvstud_core(
                x,
                jnp.asarray(1e-6, dtype=x.dtype),
                jnp.asarray(20, dtype=jnp.int64),
                jnp.asarray(10.0, dtype=x.dtype),
                jnp.asarray(2e-12, dtype=x.dtype),
                jnp.asarray(100, dtype=jnp.int64),
            )
        )(data)

        assert mu.shape == (2,)
        assert sigma.shape == (2, 2)
        assert nu.shape == ()
        assert iters.shape == ()
        assert status.shape == ()
        assert mu.dtype == data.dtype
        assert sigma.dtype == data.dtype
        assert nu.dtype == data.dtype
        assert iters.dtype == jnp.int64
        assert status.dtype == jnp.int64
        assert bool(jnp.all(jnp.isfinite(mu)))
        assert bool(jnp.all(jnp.isfinite(sigma)))
        assert bool(jnp.isfinite(nu))
        assert float(nu) > 0.0
        assert 0 <= int(iters) <= 20
        assert int(status) in {0, 1, 2, -1, -2, -3}
        np.testing.assert_allclose(sigma, sigma.T, rtol=1e-10, atol=1e-10)

    def test_core_zero_iter(self):
        data = self._data()
        expected_mu, expected_sigma = _init_mu_sigma(data)

        mu, sigma, nu, iters, status = _fit_mvstud_core(
            data,
            jnp.asarray(1e-6, dtype=data.dtype),
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(20.0, dtype=data.dtype),
            jnp.asarray(2e-12, dtype=data.dtype),
            jnp.asarray(100, dtype=jnp.int64),
        )

        np.testing.assert_allclose(mu, expected_mu, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(sigma, expected_sigma, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(nu, 20.0, rtol=1e-12, atol=1e-12)
        assert int(iters) == 0
        assert int(status) == 1

    def test_core_tolerance(self):
        data = self._clean_data()

        mu1, sigma1, nu1, iters1, status1 = _fit_mvstud_core(
            data,
            jnp.asarray(1e3, dtype=data.dtype),
            jnp.asarray(100, dtype=jnp.int64),
            jnp.asarray(20.0, dtype=data.dtype),
            jnp.asarray(2e-12, dtype=data.dtype),
            jnp.asarray(100, dtype=jnp.int64),
        )

        expected_mu, expected_sigma = _init_mu_sigma(data)
        np.testing.assert_allclose(mu1, expected_mu, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(sigma1, expected_sigma, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(nu1, 20.0, rtol=1e-12, atol=1e-12)
        assert int(iters1) == 0
        assert int(status1) == 0

    def test_wrapper(self):
        data = self._data()
        mu, sigma, nu, info = fit_mvstud_jax(
            data,
            tolerance=1e-6,
            max_iter=30,
            nu_init=10.0,
            xtol=2e-12,
            bisect_maxiter=100,
        )

        assert set(info.keys()) == {"iters", "status"}
        assert mu.shape == (2,)
        assert sigma.shape == (2, 2)
        assert nu.shape == ()
        assert info["iters"].shape == ()
        assert info["status"].shape == ()
        assert bool(jnp.all(jnp.isfinite(mu)))
        assert bool(jnp.all(jnp.isfinite(sigma)))
        assert bool(jnp.isfinite(nu))
        assert float(nu) > 0.0
        assert 0 <= int(info["iters"]) <= 30
        assert int(info["status"]) in {0, 1, 2, -1, -2, -3}

    def test_wrapper_numpy(self):
        data = np.asarray(
            [[-1.0, 0.0], [-0.5, 0.3], [0.2, 0.5], [0.9, 1.0]],
            dtype=np.float64,
        )

        mu, sigma, nu, info = fit_mvstud_jax(data, max_iter=5)

        assert mu.shape == (2,)
        assert sigma.shape == (2, 2)
        assert nu.shape == ()
        assert set(info.keys()) == {"iters", "status"}
        assert bool(jnp.all(jnp.isfinite(mu)))
        assert bool(jnp.all(jnp.isfinite(sigma)))
        assert bool(jnp.isfinite(nu))

    def test_outlier(self):
        base = jnp.asarray(
            [
                [-1.0, -1.0],
                [-0.8, -0.9],
                [-0.4, -0.3],
                [0.0, 0.1],
                [0.5, 0.4],
                [0.8, 0.9],
                [1.0, 1.1],
            ],
            dtype=jnp.float64,
        )
        outlier = jnp.asarray([[20.0, -20.0]], dtype=jnp.float64)
        data = jnp.concatenate([base, outlier], axis=0)

        init_mu, _ = _init_mu_sigma(data)
        mean = jnp.mean(data, axis=0)

        assert float(jnp.linalg.norm(init_mu)) < float(jnp.linalg.norm(mean))

        mu, sigma, nu, info = fit_mvstud_jax(data, max_iter=20, nu_init=10.0)

        assert mu.shape == (2,)
        assert sigma.shape == (2, 2)
        assert bool(jnp.all(jnp.isfinite(mu)))
        assert bool(jnp.all(jnp.isfinite(sigma)))
        assert bool(jnp.isfinite(nu))
        assert float(nu) > 0.0
        assert int(info["status"]) in {0, 1, 2, -1, -2, -3}

    def test_dtype64(self):
        data = self._data(dtype=jnp.float64)
        mu, sigma, nu, info = fit_mvstud_jax(data, max_iter=5)

        assert mu.dtype == jnp.float64
        assert sigma.dtype == jnp.float64
        assert nu.dtype == jnp.float64
        assert info["iters"].dtype == jnp.int64
        assert info["status"].dtype == jnp.int64

    def test_dtype64(self):
        data = self._data(dtype=jnp.float64)
        mu, sigma, nu, info = fit_mvstud_jax(
            data,
            max_iter=5,
            xtol=1e-6,
            bisect_maxiter=50,
        )

        assert mu.dtype == jnp.float64
        assert sigma.dtype == jnp.float64
        assert nu.dtype == jnp.float64
        assert info["iters"].dtype == jnp.int64
        assert info["status"].dtype == jnp.int64
        assert bool(jnp.all(jnp.isfinite(mu)))
        assert bool(jnp.all(jnp.isfinite(sigma)))
        assert bool(jnp.isfinite(nu))

    def test_repro(self):
        data = self._data()

        out1 = fit_mvstud_jax(data, max_iter=10, nu_init=12.0)
        out2 = fit_mvstud_jax(data, max_iter=10, nu_init=12.0)

        np.testing.assert_allclose(out1[0], out2[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out1[1], out2[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out1[2], out2[2], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(out1[3]["iters"], out2[3]["iters"])
        np.testing.assert_array_equal(out1[3]["status"], out2[3]["status"])


if __name__ == "__main__":
    absltest.main()
