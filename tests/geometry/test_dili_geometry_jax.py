import chex
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.geometry.dili_geometry_jax import (
    DILIPCNGeometry,
    _normalize_weights_jax,
    _project_psd_jax,
    _symmetrize_jax,
    build_dili_pcn_geometry_jax,
)


def _diag_gnh(theta):
    del theta
    return jnp.diag(jnp.asarray([5.0, 2.0, 1.0], dtype=jnp.float64))


def _bad_gnh(theta):
    del theta
    return jnp.asarray(
        [
            [2.0, 0.3],
            [0.3, -1.0],
        ],
        dtype=jnp.float64,
    )


class DiliGeometryTest(chex.TestCase):
    def test_normalize_weights(self):
        weights = jnp.asarray([1.0, 2.0, 1.0], dtype=jnp.float64)

        out = _normalize_weights_jax(weights)

        np.testing.assert_allclose(out, jnp.asarray([0.25, 0.5, 0.25]))
        np.testing.assert_allclose(jnp.sum(out), 1.0)
        assert out.dtype == weights.dtype

    def test_bad_weights(self):
        cases = [
            jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64),
            jnp.asarray([1.0, -1.0, 2.0], dtype=jnp.float64),
            jnp.asarray([1.0, jnp.nan, 2.0], dtype=jnp.float64),
            jnp.asarray([1.0, jnp.inf, 2.0], dtype=jnp.float64),
        ]

        expected = jnp.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=jnp.float64)

        for weights in cases:
            out = _normalize_weights_jax(weights)
            np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_symmetrize(self):
        mat = jnp.asarray(
            [
                [1.0, 4.0],
                [2.0, 3.0],
            ],
            dtype=jnp.float64,
        )

        out = _symmetrize_jax(mat)

        expected = jnp.asarray(
            [
                [1.0, 3.0],
                [3.0, 3.0],
            ],
            dtype=jnp.float64,
        )
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out, out.T, rtol=1e-6, atol=1e-6)

    def test_project_psd(self):
        mat = jnp.asarray(
            [
                [1.0, 2.0],
                [2.0, -3.0],
            ],
            dtype=jnp.float64,
        )
        floor = jnp.asarray(0.1, dtype=jnp.float64)

        out = _project_psd_jax(mat, floor)

        np.testing.assert_allclose(out, out.T, rtol=1e-6, atol=1e-6)
        eigvals = jnp.linalg.eigvalsh(out)
        assert bool(jnp.all(eigvals >= floor - 1e-8))
        assert bool(jnp.all(jnp.isfinite(out)))

    @chex.all_variants(with_pmap=False)
    def test_build_shapes(self):
        theta = jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.5],
                [0.0, 2.0, -0.5],
                [1.0, 1.0, 1.0],
            ],
            dtype=jnp.float64,
        )
        weights = jnp.asarray([1.0, 2.0, 1.0, 0.0], dtype=jnp.float64)

        out = self.variant(
            lambda th, w: build_dili_pcn_geometry_jax(
                th,
                w,
                local_gnh_fn=_diag_gnh,
                rank=2,
                gnh_floor=1e-8,
                cov_floor=1e-6,
                complement_var=0.5,
            )
        )(theta, weights)

        assert isinstance(out, DILIPCNGeometry)
        assert out.center.shape == (3,)
        assert out.basis.shape == (3, 2)
        assert out.post_var.shape == (2,)
        assert out.gnh_eigvals.shape == (2,)
        assert out.cov_ref.shape == (3, 3)

        assert out.center.dtype == theta.dtype
        assert out.basis.dtype == theta.dtype
        assert out.post_var.dtype == theta.dtype
        assert out.gnh_eigvals.dtype == theta.dtype
        assert out.cov_ref.dtype == theta.dtype

    @chex.all_variants(with_pmap=False)
    def test_build_center(self):
        theta = jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.5],
                [0.0, 2.0, -0.5],
                [1.0, 1.0, 1.0],
            ],
            dtype=jnp.float64,
        )
        weights = jnp.asarray([1.0, 2.0, 1.0, 0.0], dtype=jnp.float64)

        out = self.variant(
            lambda th, w: build_dili_pcn_geometry_jax(
                th,
                w,
                local_gnh_fn=_diag_gnh,
                rank=2,
                gnh_floor=1e-8,
                cov_floor=1e-6,
                complement_var=0.5,
            )
        )(theta, weights)

        w = weights / jnp.sum(weights)
        expected_center = jnp.sum(theta * w[:, None], axis=0)

        np.testing.assert_allclose(out.center, expected_center, rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_build_basis(self):
        theta = jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.5],
                [0.0, 2.0, -0.5],
                [1.0, 1.0, 1.0],
            ],
            dtype=jnp.float64,
        )
        weights = jnp.asarray([0.25, 0.25, 0.25, 0.25], dtype=jnp.float64)

        out = self.variant(
            lambda th, w: build_dili_pcn_geometry_jax(
                th,
                w,
                local_gnh_fn=_diag_gnh,
                rank=2,
                gnh_floor=1e-8,
                cov_floor=1e-6,
                complement_var=0.5,
            )
        )(theta, weights)

        gram = out.basis.T @ out.basis
        np.testing.assert_allclose(gram, jnp.eye(2, dtype=jnp.float64), rtol=1e-6, atol=1e-6)

        assert bool(jnp.all(out.post_var > 0.0))
        assert bool(jnp.all(out.gnh_eigvals > 0.0))
        np.testing.assert_allclose(out.gnh_eigvals, jnp.asarray([5.0, 2.0]), rtol=1e-6, atol=1e-6)

    @chex.all_variants(with_pmap=False)
    def test_cov_ref(self):
        theta = jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.5],
                [0.0, 2.0, -0.5],
                [1.0, 1.0, 1.0],
            ],
            dtype=jnp.float64,
        )
        weights = jnp.asarray([0.25, 0.25, 0.25, 0.25], dtype=jnp.float64)

        out = self.variant(
            lambda th, w: build_dili_pcn_geometry_jax(
                th,
                w,
                local_gnh_fn=_diag_gnh,
                rank=2,
                gnh_floor=1e-8,
                cov_floor=1e-6,
                complement_var=0.5,
            )
        )(theta, weights)

        np.testing.assert_allclose(out.cov_ref, out.cov_ref.T, rtol=1e-6, atol=1e-6)
        eigvals = jnp.linalg.eigvalsh(out.cov_ref)
        assert bool(jnp.all(eigvals > 0.0))
        assert bool(jnp.all(jnp.isfinite(out.cov_ref)))

    @chex.all_variants(with_pmap=False)
    def test_bad_gnh_projected(self):
        theta = jnp.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=jnp.float64,
        )
        weights = jnp.asarray([1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)

        out = self.variant(
            lambda th, w: build_dili_pcn_geometry_jax(
                th,
                w,
                local_gnh_fn=_bad_gnh,
                rank=1,
                gnh_floor=1e-4,
                cov_floor=1e-6,
                complement_var=1.0,
            )
        )(theta, weights)

        assert out.basis.shape == (2, 1)
        assert out.post_var.shape == (1,)
        assert out.gnh_eigvals.shape == (1,)
        assert bool(jnp.all(out.gnh_eigvals >= 1e-4 - 1e-8))
        assert bool(jnp.all(out.post_var > 0.0))
        assert bool(jnp.all(jnp.linalg.eigvalsh(out.cov_ref) > 0.0))

    @chex.all_variants(with_pmap=False)
    def test_bad_weights_center(self):
        theta = jnp.asarray(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 4.0],
            ],
            dtype=jnp.float64,
        )
        weights = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)

        out = self.variant(
            lambda th, w: build_dili_pcn_geometry_jax(
                th,
                w,
                local_gnh_fn=_bad_gnh,
                rank=1,
                gnh_floor=1e-4,
                cov_floor=1e-6,
                complement_var=1.0,
            )
        )(theta, weights)

        expected_center = jnp.mean(theta, axis=0)
        np.testing.assert_allclose(out.center, expected_center, rtol=1e-6, atol=1e-6)

    def test_tree(self):
        geom = DILIPCNGeometry(
            center=jnp.zeros((2,), dtype=jnp.float64),
            basis=jnp.eye(2, dtype=jnp.float64),
            post_var=jnp.ones((2,), dtype=jnp.float64),
            gnh_eigvals=jnp.asarray([2.0, 1.0], dtype=jnp.float64),
            cov_ref=jnp.eye(2, dtype=jnp.float64),
        )

        leaves, treedef = jax.tree_util.tree_flatten(geom)
        out = jax.tree_util.tree_unflatten(treedef, leaves)

        assert isinstance(out, DILIPCNGeometry)
        np.testing.assert_allclose(out.center, geom.center)
        np.testing.assert_allclose(out.basis, geom.basis)
        np.testing.assert_allclose(out.post_var, geom.post_var)
        np.testing.assert_allclose(out.gnh_eigvals, geom.gnh_eigvals)
        np.testing.assert_allclose(out.cov_ref, geom.cov_ref)


if __name__ == "__main__":
    absltest.main()