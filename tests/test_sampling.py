"""Tests for :mod:`jax_bm.sampling`.

Covers ``sample_single_chain`` and ``sample_multiple_chains``: output shapes
under all combinations of ``carry_fn`` / ``steps_per_sample`` / ``burn_in_steps``,
the value-set invariants implied by ``spin_style``, the ``free_units``
invariant, equivalence of the chain trajectory with hand-rolled
``update_state`` calls, ``vmap``-vs-loop consistency between the two
samplers, ``jit``-compatibility, and a couple of mild statistical sanity
checks (uniform marginals on the trivial model, alignment under a strong
bias).

The shared ``key`` fixture is provided by ``tests/conftest.py``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_bm.bm import BoltzmannMachine
from jax_bm.sampling import sample_multiple_chains, sample_single_chain


# =========================================================================== #
# sample_single_chain                                                         #
# =========================================================================== #


class TestSampleSingleChainShapes:
    def test_basic_shape(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        out = sample_single_chain(
            bm, key, jnp.ones(5), jnp.arange(5), burn_in_steps=10, n_samples=20
        )
        assert out.shape == (20, 5)

    def test_n_samples_one(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        out = sample_single_chain(
            bm, key, jnp.ones(4), jnp.arange(4), burn_in_steps=0, n_samples=1
        )
        assert out.shape == (1, 4)

    def test_zero_burn_in(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        out = sample_single_chain(bm, key, jnp.ones(4), jnp.arange(4), 0, 5)
        assert out.shape == (5, 4)

    def test_x0_size_propagates(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=10)
        out = sample_single_chain(bm, key, jnp.ones(10), jnp.arange(10), 3, 4)
        assert out.shape == (4, 10)

    def test_carry_fn_scalar(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        out = sample_single_chain(
            bm, key, jnp.ones(5), jnp.arange(5), 0, 8, 1, carry_fn=bm.energy
        )
        assert out.shape == (8,)

    def test_carry_fn_custom_scalar(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        out = sample_single_chain(
            bm, key, jnp.ones(4), jnp.arange(4), 0, 5, 1,
            carry_fn=lambda x: jnp.sum(x),
        )
        assert out.shape == (5,)

    def test_carry_fn_vector(self, key: jax.Array) -> None:
        # carry_fn returning a vector (e.g. the first two coordinates).
        bm = BoltzmannMachine.init_random(key, n=6)
        out = sample_single_chain(
            bm, key, jnp.ones(6), jnp.arange(6), 0, 4, 1,
            carry_fn=lambda x: x[:2],
        )
        assert out.shape == (4, 2)


class TestSampleSingleChainValueSet:
    def test_spin_outputs_in_pm_one(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=6, spin_style=True)
        out = sample_single_chain(bm, key, jnp.ones(6), jnp.arange(6), 5, 30)
        vals = set(np.unique(np.asarray(out)).tolist())
        assert vals.issubset({-1.0, 1.0})

    def test_binary_outputs_in_zero_one(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=6, spin_style=False)
        out = sample_single_chain(bm, key, jnp.zeros(6), jnp.arange(6), 5, 30)
        vals = set(np.unique(np.asarray(out)).tolist())
        assert vals.issubset({0.0, 1.0})


class TestSampleSingleChainTrajectory:
    """Tests that pin down the exact correspondence between the sampler and
    a hand-rolled sequence of ``machine.update_state`` calls."""

    def test_steps_per_sample_matches_manual(self, key: jax.Array) -> None:
        # With burn_in=0, n_samples=1, steps_per_sample=k, the single sample
        # must equal the state after k manual ``update_state`` calls.
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        k_steps = 5
        sample = sample_single_chain(
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
        sample = sample_single_chain(bm, key, x0, free, B, 1, 1)
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
        out = sample_single_chain(bm, key, x0, free, 0, N, 1)
        kk, xx = key, x0
        expected = []
        for _ in range(N):
            kk, xx = bm.update_state(kk, xx, free)
            expected.append(np.asarray(xx))
        np.testing.assert_array_equal(np.asarray(out), np.stack(expected))


class TestSampleSingleChainDeterminism:
    def test_same_key_gives_same_samples(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        x0 = jnp.ones(5)
        free = jnp.arange(5)
        s1 = sample_single_chain(bm, key, x0, free, 5, 10)
        s2 = sample_single_chain(bm, key, x0, free, 5, 10)
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))

    def test_different_keys_differ(self) -> None:
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)
        bm = BoltzmannMachine.init_random(key1, n=5)
        x0 = jnp.ones(5)
        free = jnp.arange(5)
        s1 = sample_single_chain(bm, key1, x0, free, 5, 20)
        s2 = sample_single_chain(bm, key2, x0, free, 5, 20)
        assert not np.array_equal(np.asarray(s1), np.asarray(s2))


class TestSampleSingleChainFreeUnits:
    def test_unfree_units_never_change(self, key: jax.Array) -> None:
        # With free_units = {0, 1, 2}, positions 3..7 must equal x0 in every sample.
        bm = BoltzmannMachine.init_random(key, n=8)
        x0 = -jnp.ones(8)
        free = jnp.array([0, 1, 2])
        out = sample_single_chain(bm, key, x0, free, 20, 30)
        protected = np.asarray(out[:, 3:])
        expected = np.broadcast_to(np.asarray(x0[3:]), protected.shape)
        np.testing.assert_array_equal(protected, expected)


class TestSampleSingleChainCompilation:
    def test_jit(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        f = jax.jit(
            partial(
                sample_single_chain,
                burn_in_steps=3, n_samples=5, steps_per_sample=1,
            )
        )
        out = f(bm, key, x0, free)
        assert out.shape == (5, 4)

    def test_jit_matches_eager(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones(4)
        free = jnp.arange(4)
        eager = sample_single_chain(bm, key, x0, free, 3, 5, 1)
        jitted = jax.jit(
            partial(
                sample_single_chain,
                burn_in_steps=3, n_samples=5, steps_per_sample=1,
            )
        )(bm, key, x0, free)
        np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))


class TestSampleSingleChainNoBias:
    def test_no_bias_model_runs(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, bias=False)
        out = sample_single_chain(bm, key, jnp.zeros(4), jnp.arange(4), 5, 10)
        assert out.shape == (10, 4)


# =========================================================================== #
# sample_multiple_chains                                                      #
# =========================================================================== #


class TestSampleMultipleChainsShapes:
    def test_basic_shape(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        x0 = jnp.ones((3, 5))
        out = sample_multiple_chains(bm, key, x0, jnp.arange(5), 2, 7, 1)
        assert out.shape == (3, 7, 5)

    def test_n_chains_one(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones((1, 4))
        out = sample_multiple_chains(bm, key, x0, jnp.arange(4), 0, 5, 1)
        assert out.shape == (1, 5, 4)

    def test_carry_fn_scalar(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones((3, 4))
        out = sample_multiple_chains(
            bm, key, x0, jnp.arange(4), 0, 6, 1, carry_fn=bm.energy,
        )
        assert out.shape == (3, 6)

    def test_carry_fn_vector(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        x0 = jnp.ones((2, 4))
        out = sample_multiple_chains(
            bm, key, x0, jnp.arange(4), 0, 5, 1,
            carry_fn=lambda x: x[:2],
        )
        assert out.shape == (2, 5, 2)


class TestSampleMultipleChainsValueSet:
    def test_spin_outputs(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, spin_style=True)
        x0 = jnp.ones((2, 4))
        out = sample_multiple_chains(bm, key, x0, jnp.arange(4), 5, 20, 1)
        vals = set(np.unique(np.asarray(out)).tolist())
        assert vals.issubset({-1.0, 1.0})

    def test_binary_outputs(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, spin_style=False)
        x0 = jnp.zeros((2, 4))
        out = sample_multiple_chains(bm, key, x0, jnp.arange(4), 5, 20, 1)
        vals = set(np.unique(np.asarray(out)).tolist())
        assert vals.issubset({0.0, 1.0})


class TestSampleMultipleChainsConsistency:
    """Cross-check that ``sample_multiple_chains`` is just a vmap'd
    ``sample_single_chain`` over split sub-keys."""

    def test_each_chain_matches_single_with_subkey(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        n_chains = 3
        x0 = jnp.tile(jnp.ones(4), (n_chains, 1))
        free = jnp.arange(4)
        burn, n, steps = 3, 5, 1

        multi = sample_multiple_chains(bm, key, x0, free, burn, n, steps)
        sub_keys = jax.random.split(key, n_chains)
        for i in range(n_chains):
            single = sample_single_chain(bm, sub_keys[i], x0[i], free, burn, n, steps)
            np.testing.assert_array_equal(
                np.asarray(multi[i]), np.asarray(single),
            )

    def test_chains_with_same_x0_diverge_via_subkeys(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        x0 = jnp.tile(jnp.ones(5), (2, 1))
        out = sample_multiple_chains(bm, key, x0, jnp.arange(5), 5, 30, 1)
        assert not np.array_equal(np.asarray(out[0]), np.asarray(out[1]))

    def test_per_chain_x0_respected(self, key: jax.Array) -> None:
        # Use ``free_units = [0]`` so positions 1..4 stay at their initial values.
        # Then out[c, :, 1:] must equal x0[c, 1:] for every chain c.
        bm = BoltzmannMachine.init_random(key, n=5)
        x0 = jnp.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0],
             [-1.0, -1.0, -1.0, -1.0, -1.0],
             [0.0, 1.0, 0.0, 1.0, 0.0]]
        )
        free = jnp.array([0])
        out = sample_multiple_chains(bm, key, x0, free, 2, 4, 1)
        assert out.shape == (3, 4, 5)
        for c in range(3):
            expected = np.broadcast_to(np.asarray(x0[c, 1:]), (4, 4))
            np.testing.assert_array_equal(
                np.asarray(out[c, :, 1:]), expected,
            )


class TestSampleMultipleChainsNoBias:
    def test_no_bias_model_runs(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, bias=False)
        x0 = jnp.zeros((2, 4))
        out = sample_multiple_chains(bm, key, x0, jnp.arange(4), 3, 5, 1)
        assert out.shape == (2, 5, 4)


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
        out = sample_single_chain(
            bm, jax.random.PRNGKey(0), jnp.ones(5), jnp.arange(5),
            burn_in_steps=10, n_samples=4000,
        )
        empirical_mean = np.asarray(jnp.mean(out, axis=0))
        np.testing.assert_allclose(empirical_mean, np.zeros(5), atol=0.06)

    def test_strong_positive_bias_aligns_state(self) -> None:
        # A strong positive bias should drive every coordinate toward +1.
        n = 4
        bm = BoltzmannMachine.init_from_matrix(jnp.zeros((n, n)), 5.0 * jnp.ones(n))
        out = sample_single_chain(
            bm, jax.random.PRNGKey(0), -jnp.ones(n), jnp.arange(n),
            burn_in_steps=50, n_samples=200,
        )
        empirical_mean = np.asarray(jnp.mean(out, axis=0))
        assert np.all(empirical_mean > 0.9)

    def test_strong_negative_bias_aligns_state(self) -> None:
        n = 4
        bm = BoltzmannMachine.init_from_matrix(jnp.zeros((n, n)), -5.0 * jnp.ones(n))
        out = sample_single_chain(
            bm, jax.random.PRNGKey(0), jnp.ones(n), jnp.arange(n),
            burn_in_steps=50, n_samples=200,
        )
        empirical_mean = np.asarray(jnp.mean(out, axis=0))
        assert np.all(empirical_mean < -0.9)

    def test_multi_chain_pooled_marginals(self) -> None:
        # Same uniform check, but pooling samples across multiple chains
        # exercises sample_multiple_chains. Tolerance is loose because the
        # single-site Gibbs chain only updates ~1 of 4 sites per step, so
        # the effective sample size is ~n/4.
        bm = BoltzmannMachine.init_from_matrix(jnp.zeros((4, 4)))
        x0 = jnp.ones((4, 4))  # 4 chains, 4 dims
        out = sample_multiple_chains(
            bm, jax.random.PRNGKey(0), x0, jnp.arange(4),
            burn_in_steps=50, n_samples=2000, steps_per_sample=1,
        )
        pooled = np.asarray(out).reshape(-1, 4)
        empirical_mean = pooled.mean(axis=0)
        np.testing.assert_allclose(empirical_mean, np.zeros(4), atol=0.1)
