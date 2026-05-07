import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest

from jaxpsmc.pcn_jax import (
    _flow_theta_to_u_jax,
    _flow_u_to_theta_jax,
    preconditioned_pcn_jax,
)
from jaxpsmc.sampler_jax import IdentityFlowJAX
from jaxpsmc.scaler_jax import init_bounds_config_jax, masks_jax


class PcnTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(1)
        self.flow = IdentityFlowJAX(2)
        self.cfg = init_bounds_config_jax(2, scale=False)
        self.masks = masks_jax(self.cfg["low"], self.cfg["high"])

    @chex.all_variants(with_pmap=False)
    def test_flow_helpers(self):
        u = jnp.array([[0.5, -1.0], [1.0, 2.0]])

        theta, ld1 = self.variant(
            lambda z: _flow_u_to_theta_jax(self.flow, z)
        )(u[0])
        u2, ld2 = self.variant(
            lambda z: _flow_theta_to_u_jax(self.flow, z)
        )(theta)

        chex.assert_trees_all_close(u[0], theta)
        chex.assert_trees_all_close(u[0], u2)
        assert ld1 == 0.0
        assert ld2 == 0.0

    @chex.all_variants(with_pmap=False)
    def test_step(self):
        u = jnp.array([[0.0, 0.0], [0.5, -0.5], [1.0, 1.0], [-0.5, 0.2]])
        x = u
        logdetj = jnp.zeros((4,))
        logdetj_flow = jnp.zeros((4,))
        logp = -0.5 * jnp.sum(x * x, axis=1)
        logl = -0.5 * jnp.sum((x - 0.5) ** 2, axis=1)
        blobs = jnp.zeros((4, 1))

        def loglike_fn(xi):
            return -0.5 * jnp.sum((xi - 0.5) ** 2), jnp.array([jnp.sum(xi)])

        def logprior_fn(xi):
            return -0.5 * jnp.sum(xi * xi)

        run = lambda key: preconditioned_pcn_jax(
            key,
            u=u,
            x=x,
            logdetj=logdetj,
            logl=logl,
            logp=logp,
            logdetj_flow=logdetj_flow,
            blobs=blobs,
            beta=jnp.array(0.5),
            loglike_fn=loglike_fn,
            loglike_approx_fn=lambda xi: -0.5 * jnp.sum((xi - 0.5) ** 2),
            logprior_fn=logprior_fn,
            flow=self.flow,
            scaler_cfg=self.cfg,
            scaler_masks=self.masks,
            geom_mu=jnp.zeros((2,)),
            geom_cov=jnp.eye(2),
            geom_nu=jnp.array(10.0),
            n_max=4,
            n_steps=2,
            proposal_scale=jnp.array(0.2),
        )

        out = self.variant(run)(self.key)

        assert out["u"].shape == u.shape
        assert out["x"].shape == x.shape
        assert out["logl"].shape == (4,)
        assert out["blobs"].shape == (4, 1)
        assert jnp.isfinite(out["accept"])
        assert jnp.isfinite(out["proposal_scale"])
        assert out["steps"] <= 4
        assert out["calls"] >= 0


if __name__ == "__main__":
    absltest.main()
