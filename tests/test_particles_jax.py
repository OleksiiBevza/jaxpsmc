import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.particles_jax import (
    ParticlesState,
    ParticlesStep,
    compute_logw_and_logz_jax,
    compute_results_jax,
    init_particles_state_jax,
    pop_step_jax,
    record_step_jax,
    step_mask_jax,
)


class ParticlesTest(chex.TestCase):
    def _state(self, T=3, N=2, D=2, B=1, dtype=jnp.float32):
        return init_particles_state_jax(
            max_steps=T,
            n_particles=N,
            n_dim=D,
            blob_dim=B,
            dtype=dtype,
        )

    def _step(self, value=1.0, N=2, D=2, B=1, dtype=jnp.float32):
        value = jnp.asarray(value, dtype=dtype)
        u = value + jnp.arange(N * D, dtype=dtype).reshape(N, D)
        x = 2.0 * u

        return ParticlesStep(
            u=u,
            x=x,
            logdetj=jnp.full((N,), value + 0.1, dtype=dtype),
            logl=jnp.full((N,), value + 0.2, dtype=dtype),
            logp=jnp.full((N,), value + 0.3, dtype=dtype),
            logw=jnp.full((N,), value + 0.4, dtype=dtype),
            blobs=jnp.full((N, B), value + 0.5, dtype=dtype),
            iter=jnp.asarray(value, dtype=jnp.int32),
            logz=jnp.asarray(value + 0.6, dtype=dtype),
            calls=jnp.asarray(value + 0.7, dtype=dtype),
            steps=jnp.asarray(value + 0.8, dtype=dtype),
            efficiency=jnp.asarray(value + 0.9, dtype=dtype),
            ess=jnp.asarray(value + 1.0, dtype=dtype),
            accept=jnp.asarray(value + 1.1, dtype=dtype),
            beta=jnp.asarray(value + 1.2, dtype=dtype),
        )

    def _manual(self, logl, beta, logz, t, beta_final, normalize):
        logl = np.asarray(logl, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
        logz = np.asarray(logz, dtype=np.float64)

        T, N = logl.shape
        mask_t = np.arange(T) < int(t)
        mask_flat = np.repeat(mask_t, N)

        b = logl[None, :, :] * beta[:, None, None] - logz[:, None, None]
        b = np.where(mask_t[:, None, None], b, -np.inf)

        denom_steps = max(int(t), 1)

        B = np.log(np.sum(np.exp(b), axis=0)) - np.log(denom_steps)
        A = logl * float(beta_final)
        logw = A - B
        logw = np.where(mask_t[:, None], logw, -np.inf)
        logw_flat = logw.reshape(-1)

        logz_new = np.log(np.sum(np.exp(logw_flat))) - np.log(denom_steps * N)

        if normalize:
            logw_flat = logw_flat - np.log(np.sum(np.exp(logw_flat)))

        return logw_flat, logz_new, mask_flat

    def test_init(self):
        state = self._state(T=4, N=3, D=2, B=5, dtype=jnp.float32)

        assert isinstance(state, ParticlesState)
        assert state.t.shape == ()
        assert state.t.dtype == jnp.int32
        assert int(state.t) == 0

        assert state.u.shape == (4, 3, 2)
        assert state.x.shape == (4, 3, 2)
        assert state.logdetj.shape == (4, 3)
        assert state.logl.shape == (4, 3)
        assert state.logp.shape == (4, 3)
        assert state.logw.shape == (4, 3)
        assert state.blobs.shape == (4, 3, 5)

        assert state.u.dtype == jnp.float32
        assert state.logw.dtype == jnp.float32
        assert state.iter.dtype == jnp.int32

        np.testing.assert_allclose(state.u, 0.0)
        np.testing.assert_allclose(state.x, 0.0)
        np.testing.assert_allclose(state.logdetj, 0.0)
        np.testing.assert_allclose(state.logl, 0.0)
        np.testing.assert_allclose(state.logp, 0.0)
        np.testing.assert_allclose(state.logw, -jnp.inf)
        np.testing.assert_allclose(state.blobs, 0.0)

    def test_blob0(self):
        state = self._state(T=2, N=3, D=4, B=0, dtype=jnp.float32)

        assert state.blobs.shape == (2, 3, 0)
        assert state.blobs.dtype == jnp.float32
        np.testing.assert_allclose(state.blobs, jnp.zeros((2, 3, 0)))

    def test_tree(self):
        state = self._state()
        leaves, treedef = jax.tree_util.tree_flatten(state)
        out = jax.tree_util.tree_unflatten(treedef, leaves)

        assert isinstance(out, ParticlesState)
        assert len(leaves) == len(state)
        np.testing.assert_array_equal(out.u, state.u)
        np.testing.assert_array_equal(out.logw, state.logw)
        np.testing.assert_array_equal(out.beta, state.beta)

    @chex.all_variants(with_pmap=False)
    def test_record(self):
        state0 = self._state()
        step = self._step(value=3.0)

        state1 = self.variant(lambda s, st: record_step_jax(s, st))(state0, step)

        assert int(state1.t) == 1
        np.testing.assert_allclose(state1.u[0], step.u)
        np.testing.assert_allclose(state1.x[0], step.x)
        np.testing.assert_allclose(state1.logdetj[0], step.logdetj)
        np.testing.assert_allclose(state1.logl[0], step.logl)
        np.testing.assert_allclose(state1.logp[0], step.logp)
        np.testing.assert_allclose(state1.logw[0], step.logw)
        np.testing.assert_allclose(state1.blobs[0], step.blobs)
        np.testing.assert_array_equal(state1.iter[0], step.iter)
        np.testing.assert_allclose(state1.logz[0], step.logz)
        np.testing.assert_allclose(state1.calls[0], step.calls)
        np.testing.assert_allclose(state1.steps[0], step.steps)
        np.testing.assert_allclose(state1.efficiency[0], step.efficiency)
        np.testing.assert_allclose(state1.ess[0], step.ess)
        np.testing.assert_allclose(state1.accept[0], step.accept)
        np.testing.assert_allclose(state1.beta[0], step.beta)

        np.testing.assert_allclose(state1.u[1:], 0.0)
        np.testing.assert_allclose(state1.logw[1:], -jnp.inf)

    @chex.all_variants(with_pmap=False)
    def test_many(self):
        state0 = self._state(T=3)
        s1 = self._step(value=1.0)
        s2 = self._step(value=2.0)

        def run(state):
            state = record_step_jax(state, s1)
            state = record_step_jax(state, s2)
            return state

        state2 = self.variant(run)(state0)

        assert int(state2.t) == 2
        np.testing.assert_allclose(state2.u[0], s1.u)
        np.testing.assert_allclose(state2.u[1], s2.u)
        np.testing.assert_allclose(state2.logl[0], s1.logl)
        np.testing.assert_allclose(state2.logl[1], s2.logl)
        np.testing.assert_allclose(state2.logw[2], -jnp.inf)

    @chex.all_variants(with_pmap=False)
    def test_full(self):
        state0 = self._state(T=2)
        s1 = self._step(value=1.0)
        s2 = self._step(value=2.0)
        s3 = self._step(value=3.0)

        def run(state):
            state = record_step_jax(state, s1)
            state = record_step_jax(state, s2)
            state = record_step_jax(state, s3)
            return state

        state3 = self.variant(run)(state0)

        assert int(state3.t) == 2
        np.testing.assert_allclose(state3.u[0], s1.u)
        np.testing.assert_allclose(state3.u[1], s3.u)
        np.testing.assert_allclose(state3.logl[0], s1.logl)
        np.testing.assert_allclose(state3.logl[1], s3.logl)

    @chex.all_variants(with_pmap=False)
    def test_pop(self):
        state0 = self._state(T=3)
        s1 = self._step(value=1.0)
        s2 = self._step(value=2.0)

        def run(state):
            state = record_step_jax(state, s1)
            state = record_step_jax(state, s2)
            state = pop_step_jax(state)
            return state

        state1 = self.variant(run)(state0)

        assert int(state1.t) == 1
        np.testing.assert_allclose(state1.u[0], s1.u)
        np.testing.assert_allclose(state1.u[1], s2.u)

    @chex.all_variants(with_pmap=False)
    def test_pop0(self):
        state0 = self._state(T=3)

        state1 = self.variant(lambda s: pop_step_jax(s))(state0)

        assert int(state1.t) == 0
        np.testing.assert_allclose(state1.u, state0.u)
        np.testing.assert_allclose(state1.logw, state0.logw)

    @chex.all_variants(with_pmap=False)
    def test_mask(self):
        state0 = self._state(T=4)
        s1 = self._step(value=1.0)
        s2 = self._step(value=2.0)

        def run(state):
            state = record_step_jax(state, s1)
            state = record_step_jax(state, s2)
            return step_mask_jax(state)

        mask = self.variant(run)(state0)

        np.testing.assert_array_equal(
            mask,
            jnp.array([True, True, False, False]),
        )

    @chex.all_variants(with_pmap=False)
    def test_logw_one(self):
        state0 = self._state(T=3, N=2, D=1, B=0)
        step = ParticlesStep(
            u=jnp.array([[0.0], [1.0]], dtype=jnp.float32),
            x=jnp.array([[0.0], [1.0]], dtype=jnp.float32),
            logdetj=jnp.zeros((2,), dtype=jnp.float32),
            logl=jnp.array([-1.0, -2.0], dtype=jnp.float32),
            logp=jnp.zeros((2,), dtype=jnp.float32),
            logw=jnp.zeros((2,), dtype=jnp.float32),
            blobs=jnp.zeros((2, 0), dtype=jnp.float32),
            iter=jnp.asarray(0, dtype=jnp.int32),
            logz=jnp.asarray(0.1, dtype=jnp.float32),
            calls=jnp.asarray(2.0, dtype=jnp.float32),
            steps=jnp.asarray(1.0, dtype=jnp.float32),
            efficiency=jnp.asarray(0.5, dtype=jnp.float32),
            ess=jnp.asarray(1.5, dtype=jnp.float32),
            accept=jnp.asarray(0.25, dtype=jnp.float32),
            beta=jnp.asarray(0.5, dtype=jnp.float32),
        )

        def run(state):
            state = record_step_jax(state, step)
            return compute_logw_and_logz_jax(
                state,
                beta_final=jnp.asarray(1.0, dtype=jnp.float32),
                normalize=False,
            )

        logw, logz_new, mask = self.variant(run)(state0)

        expected_logw = jnp.array([-0.4, -0.9, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf])
        expected_logz = jax.nn.logsumexp(expected_logw) - jnp.log(2.0)

        np.testing.assert_allclose(logw, expected_logw, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(logz_new, expected_logz, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(
            mask,
            jnp.array([True, True, False, False, False, False]),
        )

    @chex.all_variants(with_pmap=False)
    def test_logw_two(self):
        state0 = self._state(T=3, N=2, D=1, B=0)

        s1 = ParticlesStep(
            u=jnp.zeros((2, 1), dtype=jnp.float32),
            x=jnp.zeros((2, 1), dtype=jnp.float32),
            logdetj=jnp.zeros((2,), dtype=jnp.float32),
            logl=jnp.array([0.0, -1.0], dtype=jnp.float32),
            logp=jnp.zeros((2,), dtype=jnp.float32),
            logw=jnp.zeros((2,), dtype=jnp.float32),
            blobs=jnp.zeros((2, 0), dtype=jnp.float32),
            iter=jnp.asarray(0, dtype=jnp.int32),
            logz=jnp.asarray(0.0, dtype=jnp.float32),
            calls=jnp.asarray(1.0, dtype=jnp.float32),
            steps=jnp.asarray(1.0, dtype=jnp.float32),
            efficiency=jnp.asarray(0.0, dtype=jnp.float32),
            ess=jnp.asarray(2.0, dtype=jnp.float32),
            accept=jnp.asarray(0.0, dtype=jnp.float32),
            beta=jnp.asarray(0.0, dtype=jnp.float32),
        )

        s2 = s1._replace(
            logl=jnp.array([-2.0, -3.0], dtype=jnp.float32),
            logz=jnp.asarray(-0.2, dtype=jnp.float32),
            beta=jnp.asarray(0.5, dtype=jnp.float32),
            iter=jnp.asarray(1, dtype=jnp.int32),
        )

        def run(state):
            state = record_step_jax(state, s1)
            state = record_step_jax(state, s2)
            return state, compute_logw_and_logz_jax(
                state,
                beta_final=jnp.asarray(1.0, dtype=jnp.float32),
                normalize=False,
            )

        state2, out = self.variant(run)(state0)
        logw, logz_new, mask = out

        expected_logw, expected_logz, expected_mask = self._manual(
            state2.logl,
            state2.beta,
            state2.logz,
            t=2,
            beta_final=1.0,
            normalize=False,
        )

        np.testing.assert_allclose(logw, expected_logw, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(logz_new, expected_logz, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(mask, expected_mask)

    @chex.all_variants(with_pmap=False)
    def test_norm(self):
        state0 = self._state(T=3, N=2, D=1, B=0)
        s1 = self._step(value=1.0, N=2, D=1, B=0)
        s2 = self._step(value=2.0, N=2, D=1, B=0)

        def run(state):
            state = record_step_jax(state, s1)
            state = record_step_jax(state, s2)
            return compute_logw_and_logz_jax(
                state,
                beta_final=jnp.asarray(1.0, dtype=jnp.float32),
                normalize=True,
            )

        logw, _logz_new, mask = self.variant(run)(state0)

        active = np.asarray(mask)
        np.testing.assert_allclose(
            np.sum(np.exp(np.asarray(logw)[active])),
            1.0,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(np.asarray(logw)[~active], -np.inf)

    @chex.all_variants(with_pmap=False)
    def test_nonorm(self):
        state0 = self._state(T=3, N=2, D=1, B=0)
        s1 = self._step(value=1.0, N=2, D=1, B=0)

        def run(state):
            state = record_step_jax(state, s1)
            return compute_logw_and_logz_jax(
                state,
                beta_final=jnp.asarray(1.0, dtype=jnp.float32),
                normalize=False,
            )

        logw, _logz_new, mask = self.variant(run)(state0)

        active = np.asarray(mask)
        assert not np.isclose(np.sum(np.exp(np.asarray(logw)[active])), 1.0)

    @chex.all_variants(with_pmap=False)
    def test_empty(self):
        state = self._state(T=2, N=3, D=1, B=0)

        logw, logz_new, mask = self.variant(
            lambda s: compute_logw_and_logz_jax(
                s,
                beta_final=jnp.asarray(1.0, dtype=jnp.float32),
                normalize=False,
            )
        )(state)

        np.testing.assert_array_equal(mask, jnp.zeros((6,), dtype=bool))
        np.testing.assert_allclose(logw, -jnp.inf)
        assert bool(jnp.isneginf(logz_new))

    @chex.all_variants(with_pmap=False)
    def test_results(self):
        state0 = self._state(T=3, N=2, D=1, B=0)
        s1 = self._step(value=1.0, N=2, D=1, B=0)
        s2 = self._step(value=2.0, N=2, D=1, B=0)

        def run(state):
            state = record_step_jax(state, s1)
            state = record_step_jax(state, s2)
            return state, compute_results_jax(
                state,
                beta_final=jnp.asarray(1.0, dtype=jnp.float32),
                normalize=True,
            )

        state2, out = self.variant(run)(state0)

        expected = {
            "t",
            "mask_t",
            "mask_flat",
            "logz_new",
            "logw_flat",
            "u",
            "x",
            "logdetj",
            "logl",
            "logp",
            "logw_hist",
            "blobs",
            "iter",
            "logz",
            "calls",
            "steps",
            "efficiency",
            "ess",
            "accept",
            "beta",
        }

        assert set(out.keys()) == expected
        assert int(out["t"]) == 2

        np.testing.assert_array_equal(out["mask_t"], jnp.array([True, True, False]))
        np.testing.assert_array_equal(
            out["mask_flat"],
            jnp.array([True, True, True, True, False, False]),
        )

        np.testing.assert_allclose(out["u"], state2.u)
        np.testing.assert_allclose(out["x"], state2.x)
        np.testing.assert_allclose(out["logdetj"], state2.logdetj)
        np.testing.assert_allclose(out["logl"], state2.logl)
        np.testing.assert_allclose(out["logp"], state2.logp)
        np.testing.assert_allclose(out["logw_hist"], state2.logw)
        np.testing.assert_allclose(out["blobs"], state2.blobs)
        np.testing.assert_array_equal(out["iter"], state2.iter)
        np.testing.assert_allclose(out["logz"], state2.logz)
        np.testing.assert_allclose(out["calls"], state2.calls)
        np.testing.assert_allclose(out["steps"], state2.steps)
        np.testing.assert_allclose(out["efficiency"], state2.efficiency)
        np.testing.assert_allclose(out["ess"], state2.ess)
        np.testing.assert_allclose(out["accept"], state2.accept)
        np.testing.assert_allclose(out["beta"], state2.beta)

    @chex.all_variants(with_pmap=False)
    def test_shapes(self):
        state0 = self._state(T=5, N=4, D=3, B=2)
        step = self._step(value=1.0, N=4, D=3, B=2)

        def run(state):
            state = record_step_jax(state, step)
            return compute_results_jax(state, normalize=True)

        out = self.variant(run)(state0)

        assert out["logw_flat"].shape == (20,)
        assert out["mask_flat"].shape == (20,)
        assert out["mask_t"].shape == (5,)
        assert out["u"].shape == (5, 4, 3)
        assert out["x"].shape == (5, 4, 3)
        assert out["blobs"].shape == (5, 4, 2)


if __name__ == "__main__":
    absltest.main()
