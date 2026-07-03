import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax.experimental import checkify

from jaxpsmc.input_validation_jax import (
    assert_array_1d,
    assert_array_2d,
    assert_array_float,
    assert_array_ndim,
    assert_array_within_interval,
    assert_arrays_equal_shape,
    assert_equal_type,
    jit_with_checks,
    within_interval_mask,
)


def _checked(fn):
    return checkify.checkify(fn, errors=checkify.user_checks)


class InputValidationTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    def test_ndim(self):
        x = jnp.ones((2, 3))

        err, out = self.variant(
            lambda z: _checked(lambda y: assert_array_ndim(y, 2, name="x"))(z)
        )(x)

        err.throw()
        np.testing.assert_array_equal(out, x)

    @chex.all_variants(with_pmap=False)
    def test_ndim_bad(self):
        x = jnp.ones((2, 3))

        err, _ = self.variant(
            lambda z: _checked(lambda y: assert_array_ndim(y, 1, name="x"))(z)
        )(x)

        with self.assertRaisesRegex(Exception, "x should have"):
            err.throw()

    @chex.all_variants(with_pmap=False)
    def test_1d(self):
        x = jnp.array([1.0, 2.0, 3.0])

        err, out = self.variant(
            lambda z: _checked(lambda y: assert_array_1d(y, name="vec"))(z)
        )(x)

        err.throw()
        np.testing.assert_array_equal(out, x)

    @chex.all_variants(with_pmap=False)
    def test_1d_bad(self):
        x = jnp.ones((2, 2))

        err, _ = self.variant(
            lambda z: _checked(lambda y: assert_array_1d(y, name="vec"))(z)
        )(x)

        with self.assertRaisesRegex(Exception, "vec should have"):
            err.throw()

    @chex.all_variants(with_pmap=False)
    def test_2d(self):
        x = jnp.ones((2, 3))

        err, out = self.variant(
            lambda z: _checked(lambda y: assert_array_2d(y, name="mat"))(z)
        )(x)

        err.throw()
        np.testing.assert_array_equal(out, x)

    @chex.all_variants(with_pmap=False)
    def test_2d_bad(self):
        x = jnp.ones((3,))

        err, _ = self.variant(
            lambda z: _checked(lambda y: assert_array_2d(y, name="mat"))(z)
        )(x)

        with self.assertRaisesRegex(Exception, "mat should have"):
            err.throw()

    @chex.all_variants(with_pmap=False)
    def test_shape(self):
        x = jnp.ones((2, 3))
        y = jnp.zeros((2, 3))

        err, out = self.variant(
            lambda a, b: _checked(
                lambda u, v: assert_arrays_equal_shape(
                    u,
                    v,
                    x_name="a",
                    y_name="b",
                )
            )(a, b)
        )(x, y)

        err.throw()
        out_x, out_y = out
        np.testing.assert_array_equal(out_x, x)
        np.testing.assert_array_equal(out_y, y)

    @chex.all_variants(with_pmap=False)
    def test_shape_bad(self):
        x = jnp.ones((2, 3))
        y = jnp.zeros((3, 2))

        err, _ = self.variant(
            lambda a, b: _checked(
                lambda u, v: assert_arrays_equal_shape(
                    u,
                    v,
                    x_name="a",
                    y_name="b",
                )
            )(a, b)
        )(x, y)

        with self.assertRaisesRegex(Exception, "a and b should have equal shape"):
            err.throw()

    @chex.all_variants(with_pmap=False)
    def test_dtype(self):
        x = jnp.ones((2,), dtype=jnp.float32)
        y = jnp.zeros((2,), dtype=jnp.float32)

        err, out = self.variant(
            lambda a, b: _checked(
                lambda u, v: assert_equal_type(
                    u,
                    v,
                    x_name="a",
                    y_name="b",
                )
            )(a, b)
        )(x, y)

        err.throw()
        out_x, out_y = out
        assert out_x.dtype == jnp.float32
        assert out_y.dtype == jnp.float32

    @chex.all_variants(with_pmap=False)
    def test_dtype_bad(self):
        x = jnp.ones((2,), dtype=jnp.float32)
        y = jnp.zeros((2,), dtype=jnp.int32)

        err, _ = self.variant(
            lambda a, b: _checked(
                lambda u, v: assert_equal_type(
                    u,
                    v,
                    x_name="a",
                    y_name="b",
                )
            )(a, b)
        )(x, y)

        with self.assertRaisesRegex(Exception, "a and b should have equal dtype"):
            err.throw()

    @chex.all_variants(with_pmap=False)
    def test_float(self):
        x = jnp.ones((3,), dtype=jnp.float32)

        err, out = self.variant(
            lambda z: _checked(lambda y: assert_array_float(y, name="x"))(z)
        )(x)

        err.throw()
        assert out.dtype == jnp.float32
        np.testing.assert_array_equal(out, x)

    @chex.all_variants(with_pmap=False)
    def test_float_bad(self):
        x = jnp.ones((3,), dtype=jnp.int32)

        err, _ = self.variant(
            lambda z: _checked(lambda y: assert_array_float(y, name="x"))(z)
        )(x)

        with self.assertRaisesRegex(Exception, "x should have a floating dtype"):
            err.throw()

    @chex.all_variants(with_pmap=False)
    def test_mask_closed(self):
        x = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])

        out = self.variant(
            lambda z: within_interval_mask(
                z,
                jnp.array(0.0),
                jnp.array(1.0),
            )
        )(x)

        np.testing.assert_array_equal(
            out,
            jnp.array([False, True, True, True, False]),
        )

    @chex.all_variants(with_pmap=False)
    def test_mask_open(self):
        x = jnp.array([0.0, 0.5, 1.0])

        out = self.variant(
            lambda z: within_interval_mask(
                z,
                jnp.array(0.0),
                jnp.array(1.0),
                left_open=True,
                right_open=True,
            )
        )(x)

        np.testing.assert_array_equal(
            out,
            jnp.array([False, True, False]),
        )

    @chex.all_variants(with_pmap=False)
    def test_mask_left(self):
        x = jnp.array([0.0, 0.5, 1.0])

        out = self.variant(
            lambda z: within_interval_mask(
                z,
                jnp.array(0.0),
                jnp.array(1.0),
                left_open=True,
                right_open=False,
            )
        )(x)

        np.testing.assert_array_equal(
            out,
            jnp.array([False, True, True]),
        )

    @chex.all_variants(with_pmap=False)
    def test_mask_right(self):
        x = jnp.array([0.0, 0.5, 1.0])

        out = self.variant(
            lambda z: within_interval_mask(
                z,
                jnp.array(0.0),
                jnp.array(1.0),
                left_open=False,
                right_open=True,
            )
        )(x)

        np.testing.assert_array_equal(
            out,
            jnp.array([True, True, False]),
        )

    @chex.all_variants(with_pmap=False)
    def test_mask_nan(self):
        x = jnp.array([-10.0, 0.0, 10.0])

        out_left = self.variant(
            lambda z: within_interval_mask(
                z,
                jnp.array(jnp.nan),
                jnp.array(0.0),
            )
        )(x)

        out_right = self.variant(
            lambda z: within_interval_mask(
                z,
                jnp.array(0.0),
                jnp.array(jnp.nan),
            )
        )(x)

        np.testing.assert_array_equal(
            out_left,
            jnp.array([True, True, False]),
        )
        np.testing.assert_array_equal(
            out_right,
            jnp.array([False, True, True]),
        )

    @chex.all_variants(with_pmap=False)
    def test_interval(self):
        x = jnp.array([0.0, 0.25, 1.0])

        err, out = self.variant(
            lambda z: _checked(
                lambda y: assert_array_within_interval(
                    y,
                    jnp.array(0.0),
                    jnp.array(1.0),
                    name="prob",
                )
            )(z)
        )(x)

        err.throw()
        np.testing.assert_array_equal(out, x)

    @chex.all_variants(with_pmap=False)
    def test_interval_bad(self):
        x = jnp.array([-0.1, 0.5, 1.1])

        err, _ = self.variant(
            lambda z: _checked(
                lambda y: assert_array_within_interval(
                    y,
                    jnp.array(0.0),
                    jnp.array(1.0),
                    name="prob",
                )
            )(z)
        )(x)

        with self.assertRaisesRegex(Exception, "prob has values outside"):
            err.throw()

    @chex.all_variants(with_pmap=False)
    def test_interval_open(self):
        x = jnp.array([0.25, 0.5, 0.75])

        err, out = self.variant(
            lambda z: _checked(
                lambda y: assert_array_within_interval(
                    y,
                    jnp.array(0.0),
                    jnp.array(1.0),
                    left_open=True,
                    right_open=True,
                    name="prob",
                )
            )(z)
        )(x)

        err.throw()
        np.testing.assert_array_equal(out, x)

    @chex.all_variants(with_pmap=False)
    def test_interval_open_bad(self):
        x = jnp.array([0.0, 0.5, 1.0])

        err, _ = self.variant(
            lambda z: _checked(
                lambda y: assert_array_within_interval(
                    y,
                    jnp.array(0.0),
                    jnp.array(1.0),
                    left_open=True,
                    right_open=True,
                    name="prob",
                )
            )(z)
        )(x)

        with self.assertRaisesRegex(Exception, "prob has values outside"):
            err.throw()

    def test_jit(self):
        def fn(x):
            x = assert_array_1d(x, name="x")
            x = assert_array_float(x, name="x")
            x = assert_array_within_interval(
                x,
                jnp.array(0.0),
                jnp.array(1.0),
                name="x",
            )
            return jnp.sum(x)

        wrapped = jit_with_checks(fn)

        out = wrapped(jnp.array([0.25, 0.75], dtype=jnp.float32))
        np.testing.assert_allclose(out, 1.0)

    def test_jit_bad(self):
        def fn(x):
            x = assert_array_1d(x, name="x")
            x = assert_array_within_interval(
                x,
                jnp.array(0.0),
                jnp.array(1.0),
                name="x",
            )
            return jnp.sum(x)

        wrapped = jit_with_checks(fn)

        with self.assertRaisesRegex(Exception, "x has values outside"):
            wrapped(jnp.array([-1.0, 0.5], dtype=jnp.float32))

    def test_static(self):
        def fn(x, *, ndim):
            x = assert_array_ndim(x, ndim, name="x")
            return x + 1.0

        wrapped = jit_with_checks(fn, static_argnames=("ndim",))

        out = wrapped(jnp.ones((2, 2)), ndim=2)
        np.testing.assert_allclose(out, jnp.full((2, 2), 2.0))

        with self.assertRaisesRegex(Exception, "x should have"):
            wrapped(jnp.ones((2, 2)), ndim=1)


if __name__ == "__main__":
    absltest.main()
