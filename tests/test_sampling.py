"""Tests for :mod:`jax_bm.sampling`.

``sample_chain`` always returns a tuple whose **first element is the final
state of the Markov chain**, with subsequent elements depending on the
``corr`` / ``avg`` flags:

==============  ==============  =====================================================
``corr``        ``avg``         Return value
==============  ==============  =====================================================
``False``       ``False``       ``(final_x, samples)``
                                with ``samples.shape == (n_samples, n)``
``True``        ``False``       ``(final_x, (outers, states))``
                                with ``outers.shape == (n_samples, n, n)`` and
                                ``states.shape == (n_samples, n)``
``False``       ``True``        ``(final_x, x_mean)`` with ``x_mean.shape == (n,)``
``True``        ``True``        ``(final_x, outer_mean, x_mean)`` (a 3-tuple)
==============  ==============  =====================================================

The shared ``key`` fixture is provided by ``tests/conftest.py``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jax_bm.bm import BoltzmannMachine
from jax_bm.sampling import sample_chain


# =========================================================================== #
# Default mode (corr=False, avg=False): emit a stack of states                #
# =========================================================================== #


class TestSampleChainShapes:
    def test_basic_shape(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        final_x, samples = sample_chain(
            bm, key, jnp.ones(5), jnp.arange(5), burn_in_steps=10, n_samples=20
        )
        assert final_x.shape == (5,)
        assert samples.shape == (20, 5)

    def test_n_samples_one(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        final_x, samples = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), burn_in_steps=0, n_samples=1
        )
        assert final_x.shape == (4,)
        assert samples.shape == (1, 4)

    def test_zero_burn_in(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        final_x, samples = sample_chain(bm, key, jnp.ones(4), jnp.arange(4), 0, 5)
        assert final_x.shape == (4,)
        assert samples.shape == (5, 4)

    def test_x0_size_propagates(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=10)
        final_x, samples = sample_chain(bm, key, jnp.ones(10), jnp.arange(10), 3, 4)
        assert final_x.shape == (10,)
        assert samples.shape == (4, 10)


class TestSampleChainFinalState:
    """The first element of the return is the chain's terminal state. It must
    agree with the last emitted sample (so callers can resume the chain)."""

    def test_final_state_matches_last_sample(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        final_x, samples = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 0, 6, 1,
        )
        np.testing.assert_array_equal(np.asarray(final_x), np.asarray(samples[-1]))

    def test_final_state_matches_last_state_under_corr(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        final_x, (_, states) = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 0, 6, 1, corr=True,
        )
        np.testing.assert_array_equal(np.asarray(final_x), np.asarray(states[-1]))


class TestSampleChainValueSet:
    def test_spin_outputs_in_pm_one(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=6, spin_style=True)
        _, samples = sample_chain(bm, key, jnp.ones(6), jnp.arange(6), 5, 30)
        vals = set(np.unique(np.asarray(samples)).tolist())
        assert vals.issubset({-1.0, 1.0})

    def test_binary_outputs_in_zero_one(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=6, spin_style=False)
        _, samples = sample_chain(bm, key, jnp.zeros(6), jnp.arange(6), 5, 30)
        vals = set(np.unique(np.asarray(samples)).tolist())
        assert vals.issubset({0.0, 1.0})


class TestSampleChainTrajectory:
    """Pin down the exact correspondence between the sampler and a
    hand-rolled sequence of ``machine.update_state`` calls."""

    def test_steps_per_sample_matches_manual(self, key: jax.Array) -> None:
        # With burn_in=0, n_samples=1, steps_per_sample=k, the single sample
        # must equal the state after k manual ``update_state`` calls.
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        k_steps = 5
        _, samples = sample_chain(
            bm, key, x0, free, burn_in_steps=0, n_samples=1,
            steps_per_sample=k_steps,
        )
        kk, xx = key, x0
        for _ in range(k_steps):
            kk, xx = bm.update_state(kk, xx, free)
        np.testing.assert_array_equal(np.asarray(samples[0]), np.asarray(xx))

    def test_burn_in_then_one_sample(self, key: jax.Array) -> None:
        # burn_in=B, n_samples=1, steps_per_sample=1 should equal state after
        # B+1 manual ``update_state`` calls.
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        B = 7
        _, samples = sample_chain(bm, key, x0, free, B, 1, 1)
        kk, xx = key, x0
        for _ in range(B + 1):
            kk, xx = bm.update_state(kk, xx, free)
        np.testing.assert_array_equal(np.asarray(samples[0]), np.asarray(xx))

    def test_full_trajectory_matches_manual(self, key: jax.Array) -> None:
        # burn=0, n_samples=N, steps_per_sample=1 must equal the chain's
        # consecutive states after step 1, 2, ..., N.
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        N = 6
        _, samples = sample_chain(bm, key, x0, free, 0, N, 1)
        kk, xx = key, x0
        expected = []
        for _ in range(N):
            kk, xx = bm.update_state(kk, xx, free)
            expected.append(np.asarray(xx))
        np.testing.assert_array_equal(np.asarray(samples), np.stack(expected))


class TestSampleChainDeterminism:
    def test_same_key_gives_same_samples(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        x0 = jnp.ones(5)
        free = jnp.arange(5)
        _, s1 = sample_chain(bm, key, x0, free, 5, 10)
        _, s2 = sample_chain(bm, key, x0, free, 5, 10)
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))

    def test_different_keys_differ(self) -> None:
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)
        bm = BoltzmannMachine.init_random(key1, n=5)
        x0 = jnp.ones(5)
        free = jnp.arange(5)
        _, s1 = sample_chain(bm, key1, x0, free, 5, 20)
        _, s2 = sample_chain(bm, key2, x0, free, 5, 20)
        assert not np.array_equal(np.asarray(s1), np.asarray(s2))


class TestSampleChainFreeUnits:
    def test_unfree_units_never_change(self, key: jax.Array) -> None:
        # With free_units = {0, 1, 2}, positions 3..7 must equal x0 in every sample.
        bm = BoltzmannMachine.init_random(key, n=8)
        x0 = -jnp.ones(8)
        free = jnp.array([0, 1, 2])
        _, samples = sample_chain(bm, key, x0, free, 20, 30)
        protected = np.asarray(samples[:, 3:])
        expected = np.broadcast_to(np.asarray(x0[3:]), protected.shape)
        np.testing.assert_array_equal(protected, expected)


class TestSampleChainCompilation:
    def test_jit(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        f = jax.jit(
            partial(
                sample_chain,
                burn_in_steps=3, n_samples=5, steps_per_sample=1,
            )
        )
        final_x, samples = f(bm, key, x0, free)
        assert final_x.shape == (4,)
        assert samples.shape == (5, 4)

    def test_jit_matches_eager(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        eager_x, eager = sample_chain(bm, key, x0, free, 3, 5, 1)
        jit_x, jitted = jax.jit(
            partial(
                sample_chain,
                burn_in_steps=3, n_samples=5, steps_per_sample=1,
            )
        )(bm, key, x0, free)
        np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))
        np.testing.assert_array_equal(np.asarray(eager_x), np.asarray(jit_x))


class TestSampleChainNoBias:
    def test_no_bias_model_runs(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, bias=False)
        final_x, samples = sample_chain(bm, key, jnp.zeros(4), jnp.arange(4), 5, 10)
        assert final_x.shape == (4,)
        assert samples.shape == (10, 4)


# =========================================================================== #
# corr=True (avg=False): emit (outer-product, state) per sample               #
# =========================================================================== #


class TestSampleChainCorr:
    """``corr=True`` (with ``avg=False``) emits two stacked arrays per call:
    the outer products ``x_t x_t^T`` and the underlying states ``x_t``."""

    def test_output_is_pair_of_outers_and_states(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        final_x, (outers, states) = sample_chain(
            bm, key, jnp.ones(5), jnp.arange(5),
            burn_in_steps=2, n_samples=7, steps_per_sample=1, corr=True,
        )
        assert final_x.shape == (5,)
        assert outers.shape == (7, 5, 5)
        assert states.shape == (7, 5)

    def test_states_match_default_trajectory(self, key: jax.Array) -> None:
        # The state trajectory emitted under ``corr=True`` must be identical
        # to the trajectory under ``corr=False`` for the same key/init/params.
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        _, default_states = sample_chain(bm, key, x0, free, 3, 6, 1, corr=False)
        _, (_, corr_states) = sample_chain(bm, key, x0, free, 3, 6, 1, corr=True)
        np.testing.assert_array_equal(
            np.asarray(corr_states), np.asarray(default_states),
        )

    def test_outers_are_outer_products_of_states(self, key: jax.Array) -> None:
        # outers[t] must equal outer(states[t], states[t]) entry-by-entry.
        bm = BoltzmannMachine.init_random(key, n=4)
        _, (outers, states) = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 3, 6, 1, corr=True,
        )
        expected = np.einsum("ti,tj->tij", np.asarray(states), np.asarray(states))
        np.testing.assert_array_equal(np.asarray(outers), expected)

    def test_outers_are_symmetric(self, key: jax.Array) -> None:
        # x x^T is by definition symmetric in its trailing two axes.
        bm = BoltzmannMachine.init_random(key, n=4)
        _, (outers, _) = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 0, 5, 1, corr=True,
        )
        outers_np = np.asarray(outers)
        np.testing.assert_array_equal(outers_np, np.swapaxes(outers_np, -1, -2))

    def test_outers_diagonal_in_spin_mode_is_one(self, key: jax.Array) -> None:
        # In spin mode every entry is +/-1, so x_i^2 == 1 for all i.
        bm = BoltzmannMachine.init_random(key, n=4, spin_style=True)
        _, (outers, _) = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 0, 8, 1, corr=True,
        )
        diag = np.diagonal(np.asarray(outers), axis1=1, axis2=2)
        np.testing.assert_array_equal(diag, np.ones_like(diag))

    def test_jit(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        f = jax.jit(
            partial(
                sample_chain,
                burn_in_steps=2, n_samples=3, steps_per_sample=1, corr=True,
            )
        )
        final_x, (outers, states) = f(bm, key, jnp.ones(4), jnp.arange(4))
        assert final_x.shape == (4,)
        assert outers.shape == (3, 4, 4)
        assert states.shape == (3, 4)


# =========================================================================== #
# avg=True (corr=False): return only the running mean of the chain states     #
# =========================================================================== #


class TestSampleChainAvg:
    def test_output_is_pair_of_final_state_and_mean(self, key: jax.Array) -> None:
        # avg=True does not stack samples; it returns the per-coordinate mean.
        bm = BoltzmannMachine.init_random(key, n=5)
        final_x, x_mean = sample_chain(
            bm, key, jnp.ones(5), jnp.arange(5),
            burn_in_steps=2, n_samples=10, steps_per_sample=1, avg=True,
        )
        assert final_x.shape == (5,)
        assert x_mean.shape == (5,)

    def test_mean_matches_mean_of_default_run(self, key: jax.Array) -> None:
        # Same key/init/params: avg=True must equal jnp.mean of avg=False
        # samples (both paths advance the chain through the same fori_loops).
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        _, states = sample_chain(bm, key, x0, free, 3, 8, 1, avg=False)
        _, x_mean = sample_chain(bm, key, x0, free, 3, 8, 1, avg=True)
        np.testing.assert_allclose(
            np.asarray(x_mean), np.asarray(jnp.mean(states, axis=0)),
            rtol=1e-6, atol=1e-6,
        )

    def test_final_state_matches_default_run(self, key: jax.Array) -> None:
        # The final_x returned by the avg path must equal the avg=False
        # path's final_x (same chain advance, same key).
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        avg_final, _ = sample_chain(bm, key, x0, free, 3, 8, 1, avg=True)
        default_final, _ = sample_chain(bm, key, x0, free, 3, 8, 1, avg=False)
        np.testing.assert_array_equal(
            np.asarray(avg_final), np.asarray(default_final),
        )

    def test_mean_within_value_range(self, key: jax.Array) -> None:
        # The empirical mean of {-1,+1}-valued samples is in [-1, 1].
        bm = BoltzmannMachine.init_random(key, n=4, spin_style=True)
        _, x_mean = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 5, 50, 1, avg=True,
        )
        x_mean_np = np.asarray(x_mean)
        assert np.all(x_mean_np >= -1.0 - 1e-6)
        assert np.all(x_mean_np <= 1.0 + 1e-6)

    def test_strong_positive_bias(self) -> None:
        # Strong positive bias should drive the empirical mean toward +1
        # in every coordinate.
        n = 4
        bm = BoltzmannMachine.init_from_matrix(jnp.zeros((n, n)), 5.0 * jnp.ones(n))
        _, x_mean = sample_chain(
            bm, jax.random.PRNGKey(0), -jnp.ones(n), jnp.arange(n),
            burn_in_steps=50, n_samples=200, steps_per_sample=1, avg=True,
        )
        assert np.all(np.asarray(x_mean) > 0.9)

    def test_jit(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        f = jax.jit(
            partial(
                sample_chain,
                burn_in_steps=2, n_samples=10, steps_per_sample=1, avg=True,
            )
        )
        final_x, x_mean = f(bm, key, jnp.ones(4), jnp.arange(4))
        assert final_x.shape == (4,)
        assert x_mean.shape == (4,)

    def test_no_bias_model_runs(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, bias=False)
        final_x, x_mean = sample_chain(
            bm, key, jnp.zeros(4), jnp.arange(4), 5, 10, 1, avg=True,
        )
        assert final_x.shape == (4,)
        assert x_mean.shape == (4,)


# =========================================================================== #
# corr=True, avg=True: return (final_x, mean of outer products, mean state)   #
# =========================================================================== #


class TestSampleChainCorrAvg:
    """The combined ``corr=True, avg=True`` mode does not stack samples;
    it returns the running means of both the outer-product and the state
    sequences as a 3-tuple ``(final_x, mean_outer, mean_state)``."""

    def test_output_is_triple(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        out = sample_chain(
            bm, key, jnp.ones(5), jnp.arange(5),
            burn_in_steps=2, n_samples=8, steps_per_sample=1,
            corr=True, avg=True,
        )
        assert isinstance(out, tuple)
        assert len(out) == 3
        final_x, outer_mean, x_mean = out
        assert final_x.shape == (5,)
        assert outer_mean.shape == (5, 5)
        assert x_mean.shape == (5,)

    def test_means_match_means_of_corr_only_run(self, key: jax.Array) -> None:
        # Same key/init/params: corr=True+avg=True must equal the pointwise
        # means of the corr=True+avg=False stacks.
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        _, (outers, states) = sample_chain(
            bm, key, x0, free, 3, 8, 1, corr=True, avg=False,
        )
        _, outer_mean, x_mean = sample_chain(
            bm, key, x0, free, 3, 8, 1, corr=True, avg=True,
        )
        np.testing.assert_allclose(
            np.asarray(outer_mean), np.asarray(jnp.mean(outers, axis=0)),
            rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(x_mean), np.asarray(jnp.mean(states, axis=0)),
            rtol=1e-6, atol=1e-6,
        )

    def test_x_mean_matches_avg_only_run(self, key: jax.Array) -> None:
        # The state mean returned in corr+avg mode must agree with the
        # avg-only mode (corr=False, avg=True): both advance the chain
        # through the same fori_loops with the same key.
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        _, avg_only_mean = sample_chain(bm, key, x0, free, 3, 8, 1, corr=False, avg=True)
        _, _, x_mean = sample_chain(bm, key, x0, free, 3, 8, 1, corr=True, avg=True)
        np.testing.assert_array_equal(
            np.asarray(x_mean), np.asarray(avg_only_mean),
        )

    def test_final_state_matches_avg_only_run(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        avg_only_final, _ = sample_chain(bm, key, x0, free, 3, 8, 1, corr=False, avg=True)
        final_x, _, _ = sample_chain(bm, key, x0, free, 3, 8, 1, corr=True, avg=True)
        np.testing.assert_array_equal(
            np.asarray(final_x), np.asarray(avg_only_final),
        )

    def test_outer_mean_is_symmetric(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        _, outer_mean, _ = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 0, 5, 1, corr=True, avg=True,
        )
        outer_mean_np = np.asarray(outer_mean)
        np.testing.assert_allclose(outer_mean_np, outer_mean_np.T, rtol=1e-6, atol=1e-6)

    def test_outer_mean_diagonal_in_spin_mode_is_one(self, key: jax.Array) -> None:
        # In spin mode every emitted state has x_i^2 == 1, so the diagonal
        # of the running mean of outer products must be exactly 1.
        bm = BoltzmannMachine.init_random(key, n=4, spin_style=True)
        _, outer_mean, _ = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 0, 12, 1, corr=True, avg=True,
        )
        diag = np.diagonal(np.asarray(outer_mean))
        np.testing.assert_allclose(diag, np.ones_like(diag), rtol=1e-6, atol=1e-6)

    def test_jit(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        f = jax.jit(
            partial(
                sample_chain,
                burn_in_steps=2, n_samples=10, steps_per_sample=1,
                corr=True, avg=True,
            )
        )
        final_x, outer_mean, x_mean = f(bm, key, jnp.ones(4), jnp.arange(4))
        assert final_x.shape == (4,)
        assert outer_mean.shape == (4, 4)
        assert x_mean.shape == (4,)


# =========================================================================== #
# Statistical sanity                                                          #
# =========================================================================== #


class TestSamplingStatistics:
    """Mild statistical checks. Tolerances are loose to keep tests stable;
    they would fail dramatically if the sampler were obviously broken
    (e.g., if it ignored the bias or never updated the state)."""

    def test_zero_model_uniform_marginals(self) -> None:
        # W=0 and no bias: each unit's conditional is Bernoulli(0.5), so
        # the empirical mean over many samples should be ~0 in spin mode.
        bm = BoltzmannMachine.init_from_matrix(jnp.zeros((5, 5)))
        _, samples = sample_chain(
            bm, jax.random.PRNGKey(0), jnp.ones(5), jnp.arange(5),
            burn_in_steps=10, n_samples=4000,
        )
        empirical_mean = np.asarray(jnp.mean(samples, axis=0))
        np.testing.assert_allclose(empirical_mean, np.zeros(5), atol=0.06)

    def test_strong_positive_bias_aligns_state(self) -> None:
        # A strong positive bias should drive every coordinate toward +1.
        n = 4
        bm = BoltzmannMachine.init_from_matrix(jnp.zeros((n, n)), 5.0 * jnp.ones(n))
        _, samples = sample_chain(
            bm, jax.random.PRNGKey(0), -jnp.ones(n), jnp.arange(n),
            burn_in_steps=50, n_samples=200,
        )
        empirical_mean = np.asarray(jnp.mean(samples, axis=0))
        assert np.all(empirical_mean > 0.9)

    def test_strong_negative_bias_aligns_state(self) -> None:
        n = 4
        bm = BoltzmannMachine.init_from_matrix(jnp.zeros((n, n)), -5.0 * jnp.ones(n))
        _, samples = sample_chain(
            bm, jax.random.PRNGKey(0), jnp.ones(n), jnp.arange(n),
            burn_in_steps=50, n_samples=200,
        )
        empirical_mean = np.asarray(jnp.mean(samples, axis=0))
        assert np.all(empirical_mean < -0.9)
