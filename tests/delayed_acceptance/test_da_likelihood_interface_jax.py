import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.delayed_acceptance.da_likelihood_interface_jax import (
    TYPE_APPROX_LIKELIHOOD,
    TYPE_APPROX_POSTERIOR,
    TYPE_FULL_LIKELIHOOD,
    TYPE_FULL_POSTERIOR,
    TYPE_PRIOR,
    annealed_log_target_jax,
    da_target_type,
    make_evaluator_jax,
)


class DALikelihoodTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.x = jnp.array(
            [
                [1.0, 2.0],
                [-1.0, 0.5],
                [0.25, -0.75],
            ],
            dtype=jnp.float32,
        )
        self.beta = jnp.array(0.3, dtype=jnp.float32)
        self.max_a = jnp.array(0.7, dtype=jnp.float32)

        self.logl_full = jnp.array([3.0, -1.0, 0.5], dtype=jnp.float32)
        self.logl_approx = jnp.array([1.0, 2.0, -2.0], dtype=jnp.float32)
        self.logl_base = jnp.array([0.5, -0.5, 1.5], dtype=jnp.float32)
        self.logp = jnp.array([-0.2, -0.4, -0.6], dtype=jnp.float32)

    def _full(self, x):
        return 5.0 - jnp.sum((x - 1.0) ** 2)

    def _approx(self, x):
        return 2.0 + 0.5 * jnp.sum(x)

    def _prior(self, x):
        return -0.1 * jnp.sum(x * x)

    def _transform(self, x):
        return 2.0 * x - 1.0

    def _eval(self, transform=True):
        transform_single = self._transform if transform else None
        return make_evaluator_jax(
            log_likelihood_single=self._full,
            log_like_approx_single=self._approx,
            log_prior_single=self._prior,
            transform_single=transform_single,
        )

    def _full_batch(self, x):
        return jax.vmap(self._full)(x)

    def _approx_batch(self, x):
        return jax.vmap(self._approx)(x)

    def _prior_batch(self, x):
        return jax.vmap(self._prior)(x)

    def _transform_batch(self, x):
        return jax.vmap(self._transform)(x)

    def test_code(self):
        assert int(da_target_type("approx_posterior")) == int(
            TYPE_APPROX_POSTERIOR
        )
        assert int(da_target_type("approx_likelihood")) == int(
            TYPE_APPROX_LIKELIHOOD
        )
        assert int(da_target_type("full_likelihood")) == int(TYPE_FULL_LIKELIHOOD)
        assert int(da_target_type("full_posterior")) == int(TYPE_FULL_POSTERIOR)
        assert int(da_target_type("prior")) == int(TYPE_PRIOR)

        with self.assertRaises(ValueError):
            da_target_type("bad_type")

    @chex.all_variants(with_pmap=False)
    def test_anneal(self):
        def run(code):
            return annealed_log_target_jax(
                logl_full=self.logl_full,
                logl_approx=self.logl_approx,
                logl_approx_base=self.logl_base,
                logp=self.logp,
                beta=self.beta,
                type_code=code,
                start_from_approx=False,
                max_approx_anneal=self.max_a,
            )

        approx_post = self.variant(run)(TYPE_APPROX_POSTERIOR)
        approx_like = self.variant(run)(TYPE_APPROX_LIKELIHOOD)
        full_like = self.variant(run)(TYPE_FULL_LIKELIHOOD)
        full_post = self.variant(run)(TYPE_FULL_POSTERIOR)
        prior = self.variant(run)(TYPE_PRIOR)

        np.testing.assert_allclose(
            approx_post,
            self.beta * self.logl_approx + self.logp,
        )
        np.testing.assert_allclose(
            approx_like,
            self.beta * self.logl_approx,
        )
        np.testing.assert_allclose(
            full_like,
            self.beta * self.logl_full,
        )
        np.testing.assert_allclose(
            full_post,
            self.beta * self.logl_full + self.logp,
        )
        np.testing.assert_allclose(prior, self.logp)

    @chex.all_variants(with_pmap=False)
    def test_sfa(self):

        def run(code):
            return annealed_log_target_jax(
                logl_full=self.logl_full,
                logl_approx=self.logl_approx,
                logl_approx_base=self.logl_base,
                logp=self.logp,
                beta=self.beta,
                type_code=code,
                start_from_approx=True,
                max_approx_anneal=self.max_a,
            )

        approx_post = self.variant(run)(TYPE_APPROX_POSTERIOR)
        full_post = self.variant(run)(TYPE_FULL_POSTERIOR)

        np.testing.assert_allclose(
            approx_post,
            self.beta * self.logl_approx
            + (1.0 - self.beta) * self.max_a * self.logl_base
            + self.logp,
        )
        np.testing.assert_allclose(
            full_post,
            self.beta * self.logl_full
            + (1.0 - self.beta) * self.max_a * self.logl_base
            + self.logp,
        )

    @chex.all_variants(with_pmap=False)
    def test_eval_approx(self):
        evaluator = self._eval(transform=True)

        out = self.variant(
            lambda x: evaluator(
                x,
                beta=self.beta,
                type_code=TYPE_APPROX_POSTERIOR,
                start_from_approx=False,
                max_approx_anneal=self.max_a,
            )
        )(self.x)

        x_t = self._transform_batch(self.x)
        expected_approx = self._approx_batch(x_t)
        expected_prior = self._prior_batch(self.x)
        expected_value = self.beta * expected_approx + expected_prior

        np.testing.assert_allclose(out.value, expected_value)
        np.testing.assert_allclose(out.logl_approx, expected_approx)
        np.testing.assert_allclose(out.logp, expected_prior)

        assert bool(jnp.isnan(out.logl_full).all())
        assert bool(jnp.isnan(out.logl_approx_base).all())
        assert int(out.full_calls) == 0
        assert int(out.approx_calls) == self.x.shape[0]
        assert int(out.prior_calls) == self.x.shape[0]

    @chex.all_variants(with_pmap=False)
    def test_eval_like(self):
        evaluator = self._eval(transform=True)

        out = self.variant(
            lambda x: evaluator(
                x,
                beta=self.beta,
                type_code=TYPE_APPROX_LIKELIHOOD,
            )
        )(self.x)

        expected_approx = self._approx_batch(self._transform_batch(self.x))

        np.testing.assert_allclose(out.value, self.beta * expected_approx)
        np.testing.assert_allclose(out.logl_approx, expected_approx)

        assert bool(jnp.isnan(out.logl_full).all())
        assert bool(jnp.isnan(out.logl_approx_base).all())
        assert bool(jnp.isnan(out.logp).all())
        assert int(out.full_calls) == 0
        assert int(out.approx_calls) == self.x.shape[0]
        assert int(out.prior_calls) == 0

    @chex.all_variants(with_pmap=False)
    def test_eval_full(self):
        evaluator = self._eval(transform=True)

        out_like = self.variant(
            lambda x: evaluator(
                x,
                beta=self.beta,
                type_code=TYPE_FULL_LIKELIHOOD,
            )
        )(self.x)

        out_post = self.variant(
            lambda x: evaluator(
                x,
                beta=self.beta,
                type_code=TYPE_FULL_POSTERIOR,
                start_from_approx=False,
                max_approx_anneal=self.max_a,
            )
        )(self.x)

        expected_full = self._full_batch(self.x)
        expected_prior = self._prior_batch(self.x)

        np.testing.assert_allclose(out_like.value, self.beta * expected_full)
        np.testing.assert_allclose(out_like.logl_full, expected_full)

        assert bool(jnp.isnan(out_like.logl_approx).all())
        assert bool(jnp.isnan(out_like.logl_approx_base).all())
        assert bool(jnp.isnan(out_like.logp).all())
        assert int(out_like.full_calls) == self.x.shape[0]
        assert int(out_like.approx_calls) == 0
        assert int(out_like.prior_calls) == 0

        np.testing.assert_allclose(
            out_post.value,
            self.beta * expected_full + expected_prior,
        )
        np.testing.assert_allclose(out_post.logl_full, expected_full)
        np.testing.assert_allclose(out_post.logp, expected_prior)

        assert bool(jnp.isnan(out_post.logl_approx).all())
        assert bool(jnp.isnan(out_post.logl_approx_base).all())
        assert int(out_post.full_calls) == self.x.shape[0]
        assert int(out_post.approx_calls) == 0
        assert int(out_post.prior_calls) == self.x.shape[0]

    @chex.all_variants(with_pmap=False)
    def test_eval_sfa(self):
        evaluator = self._eval(transform=True)

        out = self.variant(
            lambda x: evaluator(
                x,
                beta=self.beta,
                type_code=TYPE_FULL_POSTERIOR,
                start_from_approx=True,
                max_approx_anneal=self.max_a,
            )
        )(self.x)

        expected_full = self._full_batch(self.x)
        expected_base = self._approx_batch(self.x)
        expected_prior = self._prior_batch(self.x)
        expected_value = (
            self.beta * expected_full
            + (1.0 - self.beta) * self.max_a * expected_base
            + expected_prior
        )

        np.testing.assert_allclose(out.value, expected_value)
        np.testing.assert_allclose(out.logl_full, expected_full)
        np.testing.assert_allclose(out.logl_approx_base, expected_base)
        np.testing.assert_allclose(out.logp, expected_prior)

        assert bool(jnp.isnan(out.logl_approx).all())
        assert int(out.full_calls) == self.x.shape[0]
        assert int(out.approx_calls) == self.x.shape[0]
        assert int(out.prior_calls) == self.x.shape[0]

    @chex.all_variants(with_pmap=False)
    def test_eval_prior(self):
        evaluator = self._eval(transform=True)

        out = self.variant(
            lambda x: evaluator(
                x,
                beta=self.beta,
                type_code=TYPE_PRIOR,
            )
        )(self.x)

        expected_prior = self._prior_batch(self.x)

        np.testing.assert_allclose(out.value, expected_prior)
        np.testing.assert_allclose(out.logp, expected_prior)

        assert bool(jnp.isnan(out.logl_full).all())
        assert bool(jnp.isnan(out.logl_approx).all())
        assert bool(jnp.isnan(out.logl_approx_base).all())
        assert int(out.full_calls) == 0
        assert int(out.approx_calls) == 0
        assert int(out.prior_calls) == self.x.shape[0]

    @chex.all_variants(with_pmap=False)
    def test_no_transform(self):
        evaluator = self._eval(transform=False)

        out = self.variant(
            lambda x: evaluator(
                x,
                beta=self.beta,
                type_code=TYPE_APPROX_LIKELIHOOD,
            )
        )(self.x)

        expected_approx = self._approx_batch(self.x)

        np.testing.assert_allclose(out.value, self.beta * expected_approx)
        np.testing.assert_allclose(out.logl_approx, expected_approx)
        assert int(out.approx_calls) == self.x.shape[0]


if __name__ == "__main__":
    absltest.main()
