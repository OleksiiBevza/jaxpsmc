import chex
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from absl.testing import absltest

from jaxpsmc.scaler_jax import (
    _create_masks_jax,
    _forward_affine_jax,
    _forward_both_jax,
    _forward_jax,
    _forward_left_jax,
    _forward_none_jax,
    _forward_right_jax,
    _inverse_affine_jax,
    _inverse_both_jax,
    _inverse_jax,
    _inverse_left_jax,
    _inverse_none_jax,
    _inverse_right_jax,
    apply_boundary_conditions_x_jax,
    apply_periodic_boundary_conditions_x_jax,
    apply_reflective_boundary_conditions_x_jax,
    fit_jax,
    forward_jax,
    init_bounds_config_jax,
    inverse_jax,
    masks_jax,
)


class ScalerTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(0)

    def _bounds(self, dtype=jnp.float64):
        return jnp.asarray(
            [
                [-jnp.inf, jnp.inf],
                [0.0, jnp.inf],
                [-jnp.inf, 5.0],
                [-2.0, 2.0],
            ],
            dtype=dtype,
        )

    def _cfg(self, *, transform="logit", scale=False, diagonal=True, dtype=jnp.float64):
        cfg = init_bounds_config_jax(
            4,
            bounds=self._bounds(dtype),
            transform=transform,
            scale=scale,
            diagonal=diagonal,
        )
        msk = masks_jax(cfg["low"], cfg["high"])
        return cfg, msk

    def _x(self, dtype=jnp.float64):
        return jnp.asarray(
            [
                [-1.0, 1.0, 4.0, -1.0],
                [0.5, 2.0, 3.0, 0.0],
                [2.0, 0.25, 4.5, 1.0],
            ],
            dtype=dtype,
        )

    def _u(self, dtype=jnp.float64):
        return jnp.asarray(
            [
                [-1.0, 0.0, 0.0, -1.0],
                [0.5, 0.7, -0.2, 0.0],
                [2.0, -0.5, 0.4, 1.0],
            ],
            dtype=dtype,
        )

    def test_config(self):
        bounds = self._bounds()
        cfg = init_bounds_config_jax(
            4,
            bounds=bounds,
            periodic=jnp.asarray([3]),
            reflective=jnp.asarray([1]),
            transform="probit",
            scale=True,
            diagonal=False,
        )

        expected = {
            "ndim",
            "low",
            "high",
            "periodic_mask",
            "reflective_mask",
            "transform_id",
            "scale",
            "diagonal",
            "mu",
            "sigma",
            "cov",
            "L",
            "L_inv",
            "log_det_L",
        }
        assert set(cfg.keys()) == expected
        assert int(cfg["ndim"]) == 4
        assert cfg["low"].shape == (4,)
        assert cfg["high"].shape == (4,)
        assert cfg["mu"].shape == (4,)
        assert cfg["cov"].shape == (4, 4)
        assert bool(cfg["scale"])
        assert not bool(cfg["diagonal"])
        assert int(cfg["transform_id"]) == 1
        np.testing.assert_array_equal(
            cfg["periodic_mask"], jnp.array([False, False, False, True])
        )
        np.testing.assert_array_equal(
            cfg["reflective_mask"], jnp.array([False, True, False, False])
        )
        assert bool(jnp.all(jnp.isnan(cfg["mu"])))
        assert bool(jnp.all(jnp.isnan(cfg["cov"])))

    def test_config_shared(self):
        cfg = init_bounds_config_jax(
            3,
            bounds=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
            transform="logit",
        )

        np.testing.assert_allclose(cfg["low"], jnp.zeros((3,)))
        np.testing.assert_allclose(cfg["high"], jnp.ones((3,)))
        assert int(cfg["transform_id"]) == 0

    def test_config_errors(self):
        with self.assertRaises(TypeError):
            init_bounds_config_jax(jnp.asarray(2))
        with self.assertRaises(ValueError):
            init_bounds_config_jax(0)
        with self.assertRaises(ValueError):
            init_bounds_config_jax(2, bounds=jnp.ones((3, 2)))
        with self.assertRaises(ValueError):
            init_bounds_config_jax(2, bounds=jnp.ones((3,)))
        with self.assertRaises(ValueError):
            init_bounds_config_jax(2, transform="bad")

    def test_masks(self):
        low = jnp.asarray([-jnp.inf, 0.0, -jnp.inf, -2.0])
        high = jnp.asarray([jnp.inf, jnp.inf, 5.0, 2.0])

        msk = masks_jax(low, high)

        np.testing.assert_array_equal(msk["mask_none"], [True, False, False, False])
        np.testing.assert_array_equal(msk["mask_left"], [False, True, False, False])
        np.testing.assert_array_equal(msk["mask_right"], [False, False, True, False])
        np.testing.assert_array_equal(msk["mask_both"], [False, False, False, True])

    def test_create_masks(self):
        msk = _create_masks_jax(4, self._bounds())

        np.testing.assert_array_equal(msk["mask_none"], [True, False, False, False])
        np.testing.assert_array_equal(msk["mask_left"], [False, True, False, False])
        np.testing.assert_array_equal(msk["mask_right"], [False, False, True, False])
        np.testing.assert_array_equal(msk["mask_both"], [False, False, False, True])

    def test_none(self):
        x = self._x()
        u = self._u()
        mask = jnp.asarray([True, False, False, False])

        x_sel, logdet = _inverse_none_jax(u, mask)
        u_sel = _forward_none_jax(x, mask)

        np.testing.assert_allclose(x_sel, u[:, :1])
        np.testing.assert_allclose(logdet, jnp.zeros((3, 1), dtype=u.dtype))
        np.testing.assert_allclose(u_sel, x[:, :1])

    def test_left(self):
        x = self._x()
        u = self._u()
        low = self._bounds()[:, 0]
        mask = jnp.asarray([False, True, False, False])

        x_sel, logdet = _inverse_left_jax(u, low, mask)
        u_sel = _forward_left_jax(x, low, mask)

        np.testing.assert_allclose(x_sel, jnp.exp(u[:, 1:2]), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(logdet, u[:, 1:2])
        np.testing.assert_allclose(u_sel, jnp.log(x[:, 1:2]), rtol=1e-6, atol=1e-6)

    def test_right(self):
        x = self._x()
        u = self._u()
        high = self._bounds()[:, 1]
        mask = jnp.asarray([False, False, True, False])

        x_sel, logdet = _inverse_right_jax(u, high, mask)
        u_sel = _forward_right_jax(x, high, mask)

        np.testing.assert_allclose(
            x_sel,
            5.0 - jnp.exp(u[:, 2:3]),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(logdet, u[:, 2:3])
        np.testing.assert_allclose(
            u_sel,
            jnp.log(5.0 - x[:, 2:3]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_both_logit(self):
        x = self._x()
        u = self._u()
        bounds = self._bounds()
        low, high = bounds[:, 0], bounds[:, 1]
        mask = jnp.asarray([False, False, False, True])
        transform_id = jnp.asarray(0, dtype=jnp.int64)

        x_sel, logdet = _inverse_both_jax(u, low, high, mask, transform_id)
        u_sel = _forward_both_jax(x, low, high, mask, transform_id)

        p = jax.nn.sigmoid(u[:, 3:4])
        expected_x = -2.0 + 4.0 * p
        expected_logdet = jnp.log(jnp.asarray(4.0, dtype=u.dtype)) + jnp.log(p) + jnp.log1p(-p)
        expected_p = (x[:, 3:4] + 2.0) / 4.0
        expected_u = jnp.log(expected_p) - jnp.log1p(-expected_p)

        np.testing.assert_allclose(x_sel, expected_x, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(logdet, expected_logdet, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(u_sel, expected_u, rtol=1e-6, atol=1e-6)

    def test_both_probit(self):
        x = jnp.asarray([[0.0], [0.5], [-0.5]], dtype=jnp.float64)
        u = jnp.asarray([[0.0], [1.0], [-1.0]], dtype=jnp.float64)
        low = jnp.asarray([-2.0], dtype=jnp.float64)
        high = jnp.asarray([2.0], dtype=jnp.float64)
        mask = jnp.asarray([True])
        transform_id = jnp.asarray(1, dtype=jnp.int64)

        x_sel, logdet = _inverse_both_jax(u, low, high, mask, transform_id)
        u_sel = _forward_both_jax(x, low, high, mask, transform_id)

        expected_x = -2.0 + 4.0 * jsp.special.ndtr(u)
        expected_logdet = jnp.log(jnp.asarray(4.0, dtype=u.dtype)) - 0.5 * u * u - jnp.log(jnp.sqrt(jnp.asarray(2.0 * np.pi, dtype=u.dtype)))
        p = (x + 2.0) / 4.0
        expected_u = jnp.sqrt(jnp.asarray(2.0, dtype=x.dtype)) * jsp.special.erfinv(2.0 * p - 1.0)

        np.testing.assert_allclose(x_sel, expected_x, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(logdet, expected_logdet, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(u_sel, expected_u, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_core(self):
        cfg, msk = self._cfg(transform="logit", scale=False)
        u = self._u()

        def run(u_in):
            x, logdet = _inverse_jax(
                u_in,
                cfg["low"],
                cfg["high"],
                msk["mask_none"],
                msk["mask_left"],
                msk["mask_right"],
                msk["mask_both"],
                cfg["transform_id"],
            )
            u_back = _forward_jax(
                x,
                cfg["low"],
                cfg["high"],
                msk["mask_none"],
                msk["mask_left"],
                msk["mask_right"],
                msk["mask_both"],
                cfg["transform_id"],
            )
            return x, logdet, u_back

        x, logdet, u_back = self.variant(run)(u)

        assert x.shape == u.shape
        assert logdet.shape == (u.shape[0],)
        np.testing.assert_allclose(u_back, u, rtol=1e-5, atol=1e-5)
        assert bool(jnp.all(jnp.isfinite(x)))
        assert bool(jnp.all(jnp.isfinite(logdet)))

    @chex.all_variants(with_pmap=False)
    def test_public(self):
        cfg, msk = self._cfg(transform="probit", scale=False)
        x = self._x()

        def run(x_in):
            u = forward_jax(x_in, cfg, msk)
            x_back, logdet = inverse_jax(u, cfg, msk)
            return u, x_back, logdet

        u, x_back, logdet = self.variant(run)(x)

        np.testing.assert_allclose(x_back, x, rtol=1e-5, atol=1e-5)
        assert u.shape == x.shape
        assert logdet.shape == (x.shape[0],)
        assert bool(jnp.all(jnp.isfinite(u)))
        assert bool(jnp.all(jnp.isfinite(logdet)))

    def test_affine_diag(self):
        u = jnp.asarray([[0.0, 1.0], [2.0, -1.0]], dtype=jnp.float64)
        mu = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        sigma = jnp.asarray([2.0, 0.5], dtype=jnp.float64)
        L = jnp.eye(2, dtype=jnp.float64)
        log_det_L = jnp.asarray(0.0, dtype=jnp.float64)

        x, logdet = _inverse_affine_jax(
            u, mu, sigma, L, log_det_L, jnp.asarray(True)
        )
        u_back = _forward_affine_jax(
            x, mu, sigma, L, jnp.asarray(True)
        )

        np.testing.assert_allclose(x, mu + sigma * u, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(u_back, u, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            logdet,
            jnp.full((2,), jnp.sum(jnp.log(sigma)), dtype=x.dtype),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_affine_full(self):
        u = jnp.asarray([[0.0, 1.0], [2.0, -1.0]], dtype=jnp.float64)
        mu = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        L = jnp.asarray([[2.0, 0.0], [0.5, 1.5]], dtype=jnp.float64)
        L_inv = jsp.linalg.solve_triangular(L, jnp.eye(2, dtype=jnp.float64), lower=True)
        sigma = jnp.ones((2,), dtype=jnp.float64)
        log_det_L = jnp.sum(jnp.log(jnp.diag(L)))

        x, logdet = _inverse_affine_jax(
            u, mu, sigma, L, log_det_L, jnp.asarray(False)
        )
        u_back = _forward_affine_jax(
            x, mu, sigma, L_inv, jnp.asarray(False)
        )

        np.testing.assert_allclose(x, mu + u @ L.T, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(u_back, u, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            logdet,
            jnp.full((2,), log_det_L, dtype=x.dtype),
            rtol=1e-6,
            atol=1e-6,
        )

    @chex.all_variants(with_pmap=False)
    def test_fit_diag(self):
        bounds = jnp.asarray([[-jnp.inf, jnp.inf], [-jnp.inf, jnp.inf]], dtype=jnp.float64)
        cfg = init_bounds_config_jax(2, bounds=bounds, scale=True, diagonal=True)
        msk = masks_jax(cfg["low"], cfg["high"])
        x = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]], dtype=jnp.float64)

        def run(x_in):
            cfg_fit = fit_jax(x_in, cfg, msk)
            u = forward_jax(x_in, cfg_fit, msk)
            x_back, logdet = inverse_jax(u, cfg_fit, msk)
            return cfg_fit["mu"], cfg_fit["sigma"], cfg_fit["cov"], u, x_back, logdet

        mu, sigma, cov, u, x_back, logdet = self.variant(run)(x)

        np.testing.assert_allclose(mu, jnp.mean(x, axis=0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(sigma, jnp.std(x, axis=0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cov, jnp.eye(2, dtype=cov.dtype), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(u, (x - mu) / sigma, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(x_back, x, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            logdet,
            jnp.full((x.shape[0],), jnp.sum(jnp.log(sigma)), dtype=logdet.dtype),
            rtol=1e-6,
            atol=1e-6,
        )

    @chex.all_variants(with_pmap=False)
    def test_fit_full(self):
        bounds = jnp.asarray([[-jnp.inf, jnp.inf], [-jnp.inf, jnp.inf]], dtype=jnp.float64)
        cfg = init_bounds_config_jax(2, bounds=bounds, scale=True, diagonal=False)
        msk = masks_jax(cfg["low"], cfg["high"])
        x = jnp.asarray(
            [[1.0, 2.0], [3.0, 5.0], [5.0, 7.0], [8.0, 11.0]],
            dtype=jnp.float64,
        )

        def run(x_in):
            cfg_fit = fit_jax(x_in, cfg, msk, jitter=1e-5)
            u = forward_jax(x_in, cfg_fit, msk)
            x_back, logdet = inverse_jax(u, cfg_fit, msk)
            return cfg_fit, u, x_back, logdet

        cfg_fit, u, x_back, logdet = self.variant(run)(x)

        assert cfg_fit["cov"].shape == (2, 2)
        assert cfg_fit["L"].shape == (2, 2)
        assert cfg_fit["L_inv"].shape == (2, 2)
        assert bool(jnp.all(jnp.isfinite(cfg_fit["cov"])))
        assert bool(jnp.all(jnp.isfinite(cfg_fit["L"])))
        assert bool(jnp.all(jnp.isfinite(u)))
        assert bool(jnp.all(jnp.isfinite(logdet)))
        np.testing.assert_allclose(x_back, x, rtol=1e-5, atol=1e-5)

    @chex.all_variants(with_pmap=False)
    def test_periodic(self):
        x = jnp.asarray(
            [[12.0, 0.5], [-2.0, 0.25], [10.0, 0.0], [0.0, 0.75]],
            dtype=jnp.float64,
        )
        low = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
        high = jnp.asarray([10.0, 1.0], dtype=jnp.float64)
        mask = jnp.asarray([True, False])

        out = self.variant(
            lambda z: apply_periodic_boundary_conditions_x_jax(z, low, high, mask)
        )(x)

        expected = jnp.asarray(
            [[2.0, 0.5], [8.0, 0.25], [10.0, 0.0], [0.0, 0.75]],
            dtype=x.dtype,
        )
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_reflective(self):
        x = jnp.asarray(
            [[12.0, 0.5], [-2.0, 0.25], [15.0, 0.0], [0.0, 0.75]],
            dtype=jnp.float64,
        )
        low = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
        high = jnp.asarray([10.0, 1.0], dtype=jnp.float64)
        mask = jnp.asarray([True, False])

        out = self.variant(
            lambda z: apply_reflective_boundary_conditions_x_jax(z, low, high, mask)
        )(x)

        expected = jnp.asarray(
            [[8.0, 0.5], [2.0, 0.25], [5.0, 0.0], [0.0, 0.75]],
            dtype=x.dtype,
        )
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_boundary(self):
        cfg = init_bounds_config_jax(
            3,
            bounds=jnp.asarray([[0.0, 10.0], [0.0, 5.0], [-jnp.inf, jnp.inf]], dtype=jnp.float64),
            periodic=jnp.asarray([0]),
            reflective=jnp.asarray([1]),
            scale=False,
        )
        x = jnp.asarray(
            [[12.0, 7.0, -3.0], [-2.0, -1.0, 4.0], [10.0, 2.5, 0.0]],
            dtype=jnp.float64,
        )

        out = self.variant(lambda z: apply_boundary_conditions_x_jax(z, cfg))(x)

        expected = jnp.asarray(
            [[2.0, 3.0, -3.0], [8.0, 1.0, 4.0], [10.0, 2.5, 0.0]],
            dtype=x.dtype,
        )
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_no_boundary(self):
        cfg = init_bounds_config_jax(
            2,
            bounds=jnp.asarray([[-jnp.inf, jnp.inf], [-jnp.inf, jnp.inf]], dtype=jnp.float64),
            scale=False,
        )
        x = jnp.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=jnp.float64)

        out = apply_boundary_conditions_x_jax(x, cfg)

        np.testing.assert_allclose(out, x)

    @chex.all_variants(with_pmap=False)
    def test_dtype(self):
        cfg, msk = self._cfg(transform="logit", scale=False, dtype=jnp.float64)
        x = self._x(dtype=jnp.float64)

        def run(x_in):
            u = forward_jax(x_in, cfg, msk)
            x_back, logdet = inverse_jax(u, cfg, msk)
            return u, x_back, logdet

        u, x_back, logdet = self.variant(run)(x)

        assert u.dtype == cfg["low"].dtype
        assert x_back.dtype == cfg["low"].dtype
        assert logdet.dtype == cfg["low"].dtype


if __name__ == "__main__":
    absltest.main()
