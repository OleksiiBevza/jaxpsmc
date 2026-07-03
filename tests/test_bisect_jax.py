import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.bisect_jax import bisect_jax, bisect_jax_batch


class BisectTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    def test_root(self):
        def f(x):
            return x * x - 2.0

        root, status, it, calls = self.variant(
            lambda a, b: bisect_jax(
                f,
                a,
                b,
                xtol=jnp.array(1e-10),
                maxiter=jnp.array(100, dtype=jnp.int32),
            )
        )(jnp.array(0.0), jnp.array(2.0))

        np.testing.assert_allclose(root, jnp.sqrt(2.0), atol=1e-8)
        assert int(status) == 0
        assert int(it) > 0
        assert int(calls) == int(it) + 2

    @chex.all_variants(with_pmap=False)
    def test_left(self):
        def f(x):
            return x - 1.0

        root, status, it, calls = self.variant(lambda a, b: bisect_jax(f, a, b))(
            jnp.array(1.0), jnp.array(4.0)
        )

        np.testing.assert_allclose(root, 1.0)
        assert int(status) == 0
        assert int(it) == 0
        assert int(calls) == 2

    @chex.all_variants(with_pmap=False)
    def test_right(self):
        def f(x):
            return x - 4.0

        root, status, it, calls = self.variant(lambda a, b: bisect_jax(f, a, b))(
            jnp.array(1.0), jnp.array(4.0)
        )

        np.testing.assert_allclose(root, 4.0)
        assert int(status) == 0
        assert int(it) == 0
        assert int(calls) == 2

    @chex.all_variants(with_pmap=False)
    def test_sign(self):
        def f(x):
            return x * x + 1.0

        root, status, it, calls = self.variant(lambda a, b: bisect_jax(f, a, b))(
            jnp.array(-1.0), jnp.array(1.0)
        )

        assert bool(jnp.isnan(root))
        assert int(status) == -1
        assert int(it) == 0
        assert int(calls) == 2

    @chex.all_variants(with_pmap=False)
    def test_nan(self):
        def f(x):
            return jnp.where(x < 0.0, jnp.nan, x - 1.0)

        root, status, it, calls = self.variant(lambda a, b: bisect_jax(f, a, b))(
            jnp.array(-1.0), jnp.array(2.0)
        )

        assert bool(jnp.isnan(root))
        assert int(status) == -3
        assert int(it) == 0
        assert int(calls) == 2

    @chex.all_variants(with_pmap=False)
    def test_mid(self):
        def f(x):
            return x

        root, status, it, calls = self.variant(
            lambda a, b: bisect_jax(
                f,
                a,
                b,
                xtol=jnp.array(1e-12),
                maxiter=jnp.array(20, dtype=jnp.int32),
            )
        )(jnp.array(-1.0), jnp.array(1.0))

        np.testing.assert_allclose(root, 0.0)
        assert int(status) == 0
        assert int(it) == 1
        assert int(calls) == 3

    @chex.all_variants(with_pmap=False)
    def test_reverse(self):
        def f(x):
            return 2.0 - x

        root, status, it, calls = self.variant(
            lambda a, b: bisect_jax(
                f,
                a,
                b,
                xtol=jnp.array(1e-10),
                maxiter=jnp.array(100, dtype=jnp.int32),
            )
        )(jnp.array(0.0), jnp.array(4.0))

        np.testing.assert_allclose(root, 2.0, atol=1e-8)
        assert int(status) == 0
        assert int(it) > 0
        assert int(calls) == int(it) + 2

    @chex.all_variants(with_pmap=False)
    def test_args(self):
        def f(x, scale, shift):
            return scale * (x - shift)

        root, status, it, calls = self.variant(
            lambda a, b, scale, shift: bisect_jax(
                f,
                a,
                b,
                args=(scale, shift),
                xtol=jnp.array(1e-10),
                maxiter=jnp.array(100, dtype=jnp.int32),
            )
        )(
            jnp.array(0.0),
            jnp.array(3.0),
            jnp.array(2.5),
            jnp.array(1.25),
        )

        np.testing.assert_allclose(root, 1.25, atol=1e-8)
        assert int(status) == 0
        assert int(it) > 0
        assert int(calls) == int(it) + 2

    @chex.all_variants(with_pmap=False)
    def test_tol(self):
        def f(x):
            return x * x - 2.0

        loose = self.variant(
            lambda a, b: bisect_jax(
                f,
                a,
                b,
                xtol=jnp.array(1e-2),
                maxiter=jnp.array(100, dtype=jnp.int32),
            )
        )(jnp.array(0.0), jnp.array(2.0))

        tight = self.variant(
            lambda a, b: bisect_jax(
                f,
                a,
                b,
                xtol=jnp.array(1e-10),
                maxiter=jnp.array(100, dtype=jnp.int32),
            )
        )(jnp.array(0.0), jnp.array(2.0))

        root_loose, status_loose, it_loose, calls_loose = loose
        root_tight, status_tight, it_tight, calls_tight = tight

        assert int(status_loose) == 0
        assert int(status_tight) == 0
        assert int(it_tight) >= int(it_loose)
        assert int(calls_tight) >= int(calls_loose)

        err_loose = jnp.abs(root_loose - jnp.sqrt(2.0))
        err_tight = jnp.abs(root_tight - jnp.sqrt(2.0))
        assert bool(err_tight <= err_loose + 1e-12)

    @chex.all_variants(with_pmap=False)
    def test_cap(self):
        def f(x):
            return x - 0.3

        root, status, it, calls = self.variant(
            lambda a, b: bisect_jax(
                f,
                a,
                b,
                xtol=jnp.array(1e-12),
                maxiter=jnp.array(1, dtype=jnp.int32),
            )
        )(jnp.array(0.0), jnp.array(1.0))

        assert bool(jnp.isnan(root))
        assert int(status) == -2
        assert int(it) == 1
        assert int(calls) == 3

    @chex.all_variants(with_pmap=False)
    def test_badmax(self):
        def f(x):
            return x - 0.5

        root, status, it, calls = self.variant(
            lambda a, b: bisect_jax(
                f,
                a,
                b,
                maxiter=jnp.array(-1, dtype=jnp.int32),
            )
        )(jnp.array(0.0), jnp.array(1.0))

        assert bool(jnp.isnan(root))
        assert int(status) == -3
        assert int(it) == 0
        assert int(calls) == 2

    @chex.all_variants(with_pmap=False)
    def test_zeroiter(self):
        def f(x):
            return x - 0.25

        root, status, it, calls = self.variant(
            lambda a, b: bisect_jax(
                f,
                a,
                b,
                maxiter=jnp.array(0, dtype=jnp.int32),
            )
        )(jnp.array(0.0), jnp.array(1.0))

        assert bool(jnp.isnan(root))
        assert int(status) == -2
        assert int(it) == 0
        assert int(calls) == 2

    @chex.all_variants(with_pmap=False)
    def test_dtype(self):
        def f(x):
            return x - jnp.asarray(0.25, dtype=x.dtype)

        root, status, _, _ = self.variant(
            lambda a, b: bisect_jax(
                f,
                a,
                b,
                xtol=jnp.array(1e-6, dtype=jnp.float32),
                rtol=jnp.array(0.0, dtype=jnp.float32),
                maxiter=jnp.array(80, dtype=jnp.int32),
            )
        )(
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
        )

        assert root.dtype == jnp.float32
        np.testing.assert_allclose(root, 0.25, atol=1e-5)
        assert int(status) == 0

    @chex.all_variants(with_pmap=False)
    def test_batch(self):
        def f(x, shift):
            return x - shift

        a = jnp.zeros((3,))
        b = jnp.full((3,), 3.0)
        shift = jnp.array([0.25, 1.0, 2.5])

        roots, status, it, calls = self.variant(
            lambda aa, bb, ss: bisect_jax_batch(
                f,
                aa,
                bb,
                args=(ss,),
                xtol=jnp.array(1e-10),
                maxiter=jnp.array(100, dtype=jnp.int32),
            )
        )(a, b, shift)

        np.testing.assert_allclose(roots, shift, atol=1e-8)
        np.testing.assert_array_equal(status, jnp.zeros((3,), dtype=jnp.int32))
        assert bool(jnp.all(it > 0))
        np.testing.assert_array_equal(calls, it + 2)

    @chex.all_variants(with_pmap=False)
    def test_mixed(self):
        def f(x, shift):
            return x - shift

        a = jnp.zeros((4,))
        b = jnp.full((4,), 3.0)
        shift = jnp.array([0.25, 4.0, 2.5, -1.0])

        roots, status, it, calls = self.variant(
            lambda aa, bb, ss: bisect_jax_batch(
                f,
                aa,
                bb,
                args=(ss,),
                xtol=jnp.array(1e-10),
                maxiter=jnp.array(100, dtype=jnp.int32),
            )
        )(a, b, shift)

        np.testing.assert_allclose(roots[0], 0.25, atol=1e-8)
        np.testing.assert_allclose(roots[2], 2.5, atol=1e-8)
        assert bool(jnp.isnan(roots[1]))
        assert bool(jnp.isnan(roots[3]))

        np.testing.assert_array_equal(
            status,
            jnp.array([0, -1, 0, -1], dtype=jnp.int32),
        )
        assert int(it[1]) == 0
        assert int(it[3]) == 0
        assert int(calls[1]) == 2
        assert int(calls[3]) == 2

    @chex.all_variants(with_pmap=False)
    def test_scalar(self):
        def f(x, scale, shift):
            return scale * (x - shift)

        a = jnp.zeros((3,))
        b = jnp.full((3,), 3.0)
        scale = jnp.array(2.0)
        shift = jnp.array([0.25, 1.0, 2.5])

        roots, status, it, calls = self.variant(
            lambda aa, bb, sc, sh: bisect_jax_batch(
                f,
                aa,
                bb,
                args=(sc, sh),
                xtol=jnp.array(1e-10),
                maxiter=jnp.array(100, dtype=jnp.int32),
            )
        )(a, b, scale, shift)

        np.testing.assert_allclose(roots, shift, atol=1e-8)
        np.testing.assert_array_equal(status, jnp.zeros((3,), dtype=jnp.int32))
        assert bool(jnp.all(it > 0))
        np.testing.assert_array_equal(calls, it + 2)

    @chex.all_variants(with_pmap=False)
    def test_endbatch(self):
        def f(x, shift):
            return x - shift

        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([3.0, 2.0, 4.0])
        shift = jnp.array([1.0, 2.0, 3.0])

        roots, status, it, calls = self.variant(
            lambda aa, bb, ss: bisect_jax_batch(
                f,
                aa,
                bb,
                args=(ss,),
            )
        )(a, b, shift)

        np.testing.assert_allclose(roots, shift, atol=1e-8)
        np.testing.assert_array_equal(status, jnp.zeros((3,), dtype=jnp.int32))

        assert int(it[0]) == 0
        assert int(it[1]) == 0
        assert int(calls[0]) == 2
        assert int(calls[1]) == 2

        assert int(it[2]) > 0
        assert int(calls[2]) == int(it[2]) + 2


if __name__ == "__main__":
    absltest.main()
