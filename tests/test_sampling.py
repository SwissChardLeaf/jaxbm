"""Tests for :mod:`jax_bm.sampling`.

Covers ``sample_chain``: output shapes under all combinations of the
``corr`` / ``avg`` / ``steps_per_sample`` / ``burn_in_steps`` knobs, the
value-set invariants implied by ``spin_style``, the ``free_units``
invariant, equivalence of the chain trajectory with hand-rolled
``update_state`` calls, ``jit``-compatibility, and a couple of mild
statistical sanity checks.

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
        out = sample_chain(
            bm, key, jnp.ones(5), jnp.arange(5), burn_in_steps=10, n_samples=20
        )
        assert out.shape == (20, 5)

    def test_n_samples_one(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        out = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), burn_in_steps=0, n_samples=1
        )
        assert out.shape == (1, 4)

    def test_zero_burn_in(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        out = sample_chain(bm, key, jnp.ones(4), jnp.arange(4), 0, 5)
        assert out.shape == (5, 4)

    def test_x0_size_propagates(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=10)
        out = sample_chain(bm, key, jnp.ones(10), jnp.arange(10), 3, 4)
        assert out.shape == (4, 10)


class TestSampleChainValueSet:
    def test_spin_outputs_in_pm_one(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=6, spin_style=True)
        out = sample_chain(bm, key, jnp.ones(6), jnp.arange(6), 5, 30)
        vals = set(np.unique(np.asarray(out)).tolist())
        assert vals.issubset({-1.0, 1.0})

    def test_binary_outputs_in_zero_one(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=6, spin_style=False)
        out = sample_chain(bm, key, jnp.zeros(6), jnp.arange(6), 5, 30)
        vals = set(np.unique(np.asarray(out)).tolist())
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
        sample = sample_chain(
            bm, key, x0, free, burn_in_steps=0, n_samples=1,
            steps_per_sample=k_steps,
        )
        kk, xx = key, x0
        for _ in range(k_steps):
            kk, xx = bm.update_state(kk, xx, free)
        np.testing.assert_array_equal(np.asarray(sample[0]), np.asarray(xx))

    def test_burn_in_then_one_sample(self, key: jax.Array) -> None:
        # burn_in=B, n_samples=1, steps_per_sample=1 should equal state after
        # B+1 manual ``update_state`` calls.
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        B = 7
        sample = sample_chain(bm, key, x0, free, B, 1, 1)
        kk, xx = key, x0
        for _ in range(B + 1):
            kk, xx = bm.update_state(kk, xx, free)
        np.testing.assert_array_equal(np.asarray(sample[0]), np.asarray(xx))

    def test_full_trajectory_matches_manual(self, key: jax.Array) -> None:
        # burn=0, n_samples=N, steps_per_sample=1 must equal the chain's
        # consecutive states after step 1, 2, ..., N.
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        N = 6
        out = sample_chain(bm, key, x0, free, 0, N, 1)
        kk, xx = key, x0
        expected = []
        for _ in range(N):
            kk, xx = bm.update_state(kk, xx, free)
            expected.append(np.asarray(xx))
        np.testing.assert_array_equal(np.asarray(out), np.stack(expected))


class TestSampleChainDeterminism:
    def test_same_key_gives_same_samples(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        x0 = jnp.ones(5)
        free = jnp.arange(5)
        s1 = sample_chain(bm, key, x0, free, 5, 10)
        s2 = sample_chain(bm, key, x0, free, 5, 10)
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))

    def test_different_keys_differ(self) -> None:
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)
        bm = BoltzmannMachine.init_random(key1, n=5)
        x0 = jnp.ones(5)
        free = jnp.arange(5)
        s1 = sample_chain(bm, key1, x0, free, 5, 20)
        s2 = sample_chain(bm, key2, x0, free, 5, 20)
        assert not np.array_equal(np.asarray(s1), np.asarray(s2))


class TestSampleChainFreeUnits:
    def test_unfree_units_never_change(self, key: jax.Array) -> None:
        # With free_units = {0, 1, 2}, positions 3..7 must equal x0 in every sample.
        bm = BoltzmannMachine.init_random(key, n=8)
        x0 = -jnp.ones(8)
        free = jnp.array([0, 1, 2])
        out = sample_chain(bm, key, x0, free, 20, 30)
        protected = np.asarray(out[:, 3:])
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
        out = f(bm, key, x0, free)
        assert out.shape == (5, 4)

    def test_jit_matches_eager(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        eager = sample_chain(bm, key, x0, free, 3, 5, 1)
        jitted = jax.jit(
            partial(
                sample_chain,
                burn_in_steps=3, n_samples=5, steps_per_sample=1,
            )
        )(bm, key, x0, free)
        np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))


class TestSampleChainNoBias:
    def test_no_bias_model_runs(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, bias=False)
        out = sample_chain(bm, key, jnp.zeros(4), jnp.arange(4), 5, 10)
        assert out.shape == (10, 4)


# =========================================================================== #
# corr=True (avg=False): emit (outer-product, state) per sample               #
# =========================================================================== #


class TestSampleChainCorr:
    """``corr=True`` (with ``avg=False``) emits two stacked arrays per call:
    the outer products ``x_t x_t^T`` and the underlying states ``x_t``."""

    def test_output_is_tuple_of_outers_and_states(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        out = sample_chain(
            bm, key, jnp.ones(5), jnp.arange(5),
            burn_in_steps=2, n_samples=7, steps_per_sample=1, corr=True,
        )
        assert isinstance(out, tuple)
        assert len(out) == 2
        outers, states = out
        assert outers.shape == (7, 5, 5)
        assert states.shape == (7, 5)

    def test_states_match_default_trajectory(self, key: jax.Array) -> None:
        # The state trajectory emitted under ``corr=True`` must be identical
        # to the trajectory under ``corr=False`` for the same key/init/params.
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        default_states = sample_chain(bm, key, x0, free, 3, 6, 1, corr=False)
        _, corr_states = sample_chain(bm, key, x0, free, 3, 6, 1, corr=True)
        np.testing.assert_array_equal(
            np.asarray(corr_states), np.asarray(default_states),
        )

    def test_outers_are_outer_products_of_states(self, key: jax.Array) -> None:
        # outers[t] must equal outer(states[t], states[t]) entry-by-entry.
        bm = BoltzmannMachine.init_random(key, n=4)
        outers, states = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 3, 6, 1, corr=True,
        )
        expected = np.einsum("ti,tj->tij", np.asarray(states), np.asarray(states))
        np.testing.assert_array_equal(np.asarray(outers), expected)

    def test_outers_are_symmetric(self, key: jax.Array) -> None:
        # x x^T is by definition symmetric in its trailing two axes.
        bm = BoltzmannMachine.init_random(key, n=4)
        outers, _ = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 0, 5, 1, corr=True,
        )
        outers_np = np.asarray(outers)
        np.testing.assert_array_equal(outers_np, np.swapaxes(outers_np, -1, -2))

    def test_outers_diagonal_in_spin_mode_is_one(self, key: jax.Array) -> None:
        # In spin mode every entry is +/-1, so x_i^2 == 1 for all i.
        bm = BoltzmannMachine.init_random(key, n=4, spin_style=True)
        outers, _ = sample_chain(
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
        outers, states = f(bm, key, jnp.ones(4), jnp.arange(4))
        assert outers.shape == (3, 4, 4)
        assert states.shape == (3, 4)


# =========================================================================== #
# avg=True: return only the running mean of the chain states                  #
# =========================================================================== #


class TestSampleChainAvg:
    def test_output_shape_is_a_single_vector(self, key: jax.Array) -> None:
        # avg=True does not stack samples; it returns the per-coordinate mean.
        bm = BoltzmannMachine.init_random(key, n=5)
        out = sample_chain(
            bm, key, jnp.ones(5), jnp.arange(5),
            burn_in_steps=2, n_samples=10, steps_per_sample=1, avg=True,
        )
        assert out.shape == (5,)

    def test_matches_mean_of_default_run(self, key: jax.Array) -> None:
        # Same key/init/params: avg=True must equal jnp.mean of avg=False
        # samples (both paths advance the chain through the same fori_loops).
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        states = sample_chain(bm, key, x0, free, 3, 8, 1, avg=False)
        avg = sample_chain(bm, key, x0, free, 3, 8, 1, avg=True)
        np.testing.assert_allclose(
            np.asarray(avg), np.asarray(jnp.mean(states, axis=0)), rtol=1e-6, atol=1e-6,
        )

    def test_mean_within_value_range(self, key: jax.Array) -> None:
        # The empirical mean of {-1,+1}-valued samples is in [-1, 1].
        bm = BoltzmannMachine.init_random(key, n=4, spin_style=True)
        out = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 5, 50, 1, avg=True,
        )
        out_np = np.asarray(out)
        assert np.all(out_np >= -1.0 - 1e-6)
        assert np.all(out_np <= 1.0 + 1e-6)

    def test_strong_positive_bias(self) -> None:
        # Strong positive bias should drive the empirical mean toward +1
        # in every coordinate. End-to-end check that avg=True actually
        # accumulates samples (not, e.g., x0).
        n = 4
        bm = BoltzmannMachine.init_from_matrix(jnp.zeros((n, n)), 5.0 * jnp.ones(n))
        out = sample_chain(
            bm, jax.random.PRNGKey(0), -jnp.ones(n), jnp.arange(n),
            burn_in_steps=50, n_samples=200, steps_per_sample=1, avg=True,
        )
        assert np.all(np.asarray(out) > 0.9)

    def test_jit(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        f = jax.jit(
            partial(
                sample_chain,
                burn_in_steps=2, n_samples=10, steps_per_sample=1, avg=True,
            )
        )
        out = f(bm, key, jnp.ones(4), jnp.arange(4))
        assert out.shape == (4,)

    def test_no_bias_model_runs(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, bias=False)
        out = sample_chain(
            bm, key, jnp.zeros(4), jnp.arange(4), 5, 10, 1, avg=True,
        )
        assert out.shape == (4,)


# =========================================================================== #
# corr=True, avg=True: return (mean of outer products, mean of states)        #
# =========================================================================== #


class TestSampleChainCorrAvg:
    """The combined ``corr=True, avg=True`` mode does not stack samples;
    it returns the running means of both the outer-product and the state
    sequences as a tuple ``(mean_outer, mean_state)``."""

    def test_output_is_tuple_of_means(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        out = sample_chain(
            bm, key, jnp.ones(5), jnp.arange(5),
            burn_in_steps=2, n_samples=8, steps_per_sample=1,
            corr=True, avg=True,
        )
        assert isinstance(out, tuple)
        assert len(out) == 2
        outer_mean, x_mean = out
        assert outer_mean.shape == (5, 5)
        assert x_mean.shape == (5,)

    def test_means_match_means_of_corr_only_run(self, key: jax.Array) -> None:
        # Same key/init/params: corr=True+avg=True must equal the pointwise
        # means of the corr=True+avg=False stacks.
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        outers, states = sample_chain(
            bm, key, x0, free, 3, 8, 1, corr=True, avg=False,
        )
        outer_mean, x_mean = sample_chain(
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
        avg_only = sample_chain(bm, key, x0, free, 3, 8, 1, corr=False, avg=True)
        _, x_mean = sample_chain(bm, key, x0, free, 3, 8, 1, corr=True, avg=True)
        np.testing.assert_array_equal(np.asarray(x_mean), np.asarray(avg_only))

    def test_outer_mean_is_symmetric(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        outer_mean, _ = sample_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 0, 5, 1, corr=True, avg=True,
        )
        outer_mean_np = np.asarray(outer_mean)
        np.testing.assert_allclose(outer_mean_np, outer_mean_np.T, rtol=1e-6, atol=1e-6)

    def test_outer_mean_diagonal_in_spin_mode_is_one(self, key: jax.Array) -> None:
        # In spin mode every emitted state has x_i^2 == 1, so the diagonal
        # of the running mean of outer products must be exactly 1.
        bm = BoltzmannMachine.init_random(key, n=4, spin_style=True)
        outer_mean, _ = sample_chain(
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
        outer_mean, x_mean = f(bm, key, jnp.ones(4), jnp.arange(4))
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
        out = sample_chain(
            bm, jax.random.PRNGKey(0), jnp.ones(5), jnp.arange(5),
            burn_in_steps=10, n_samples=4000,
        )
        empirical_mean = np.asarray(jnp.mean(out, axis=0))
        np.testing.assert_allclose(empirical_mean, np.zeros(5), atol=0.06)

    def test_strong_positive_bias_aligns_state(self) -> None:
        # A strong positive bias should drive every coordinate toward +1.
        n = 4
        bm = BoltzmannMachine.init_from_matrix(jnp.zeros((n, n)), 5.0 * jnp.ones(n))
        out = sample_chain(
            bm, jax.random.PRNGKey(0), -jnp.ones(n), jnp.arange(n),
            burn_in_steps=50, n_samples=200,
        )
        empirical_mean = np.asarray(jnp.mean(out, axis=0))
        assert np.all(empirical_mean > 0.9)

    def test_strong_negative_bias_aligns_state(self) -> None:
        n = 4
        bm = BoltzmannMachine.init_from_matrix(jnp.zeros((n, n)), -5.0 * jnp.ones(n))
        out = sample_chain(
            bm, jax.random.PRNGKey(0), jnp.ones(n), jnp.arange(n),
            burn_in_steps=50, n_samples=200,
        )
        empirical_mean = np.asarray(jnp.mean(out, axis=0))
        assert np.all(empirical_mean < -0.9)
