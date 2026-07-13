"""Tests for :mod:`jax_bm.sample` (the ``sample_chain`` array-in entry point).

``sample_chain`` always returns the final state ``x`` first; what else comes
back depends on ``n_samples`` / ``avg`` / ``corr`` -- see its docstring for
the full table. The shared ``key`` fixture is provided by ``tests/conftest.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_bm.sample import (
    _restricted_n_visible,
    _validate_state,
    _validate_weights,
    sample_chain,
)


def _random_symmetric_weights(key, n, minval=-1.0, maxval=1.0):
    W = jax.random.uniform(key, (n, n), minval=0.5 * minval, maxval=0.5 * maxval)
    W = W + W.T
    W = W - jnp.diag(jnp.diagonal(W))
    return W


def _bipartite_weights(key, n_visible, n_hidden, minval=-1.0, maxval=1.0):
    n = n_visible + n_hidden
    block = jax.random.uniform(key, (n_visible, n_hidden), minval=minval, maxval=maxval)
    W = jnp.zeros((n, n))
    W = W.at[:n_visible, n_visible:].set(block)
    W = W.at[n_visible:, :n_visible].set(block.T)
    return W


# =========================================================================== #
# Validation                                                                  #
# =========================================================================== #


class TestValidateWeights:
    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            _validate_weights(jnp.ones((2, 3)), jnp.zeros(2))

    def test_asymmetric_raises(self) -> None:
        W = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="symmetric"):
            _validate_weights(W, jnp.zeros(2))

    def test_bias_size_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same number of rows"):
            _validate_weights(jnp.zeros((3, 3)), jnp.zeros(4))

    def test_none_bias_is_allowed(self) -> None:
        W = jnp.zeros((3, 3))
        assert _validate_weights(W, None) == (W, None)

    def test_valid_weights_pass_through(self) -> None:
        W = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        b = jnp.array([0.1, 0.2])
        assert _validate_weights(W, b) == (W, b)

    def test_sample_chain_propagates_validation_error(self, key: jax.Array) -> None:
        with pytest.raises(ValueError):
            sample_chain(key, jnp.ones(2), jnp.ones((2, 3)), jnp.zeros(2), steps=1)


class TestValidateState:
    def test_wrong_length_raises(self) -> None:
        W = jnp.zeros((3, 3))
        with pytest.raises(ValueError, match="size"):
            _validate_state(jnp.ones(2), W, n_visible=None, spin=True)

    def test_non_vector_raises(self) -> None:
        W = jnp.zeros((2, 2))
        with pytest.raises(ValueError, match="size"):
            _validate_state(jnp.ones((2, 2)), W, n_visible=None, spin=True)

    def test_spin_true_rejects_binary_state(self) -> None:
        W = jnp.zeros((3, 3))
        with pytest.raises(ValueError, match=r"\{-1, \+1\}|-1.0, 1.0"):
            _validate_state(jnp.zeros(3), W, n_visible=None, spin=True)

    def test_spin_true_accepts_pm_one_state(self) -> None:
        W = jnp.zeros((3, 3))
        x = jnp.array([-1.0, 1.0, -1.0])
        assert _validate_state(x, W, n_visible=None, spin=True) is x

    def test_spin_false_rejects_pm_one_state(self) -> None:
        W = jnp.zeros((3, 3))
        with pytest.raises(ValueError):
            _validate_state(jnp.array([-1.0, 1.0, -1.0]), W, n_visible=None, spin=False)

    def test_spin_false_accepts_binary_state(self) -> None:
        W = jnp.zeros((3, 3))
        x = jnp.array([0.0, 1.0, 0.0])
        assert _validate_state(x, W, n_visible=None, spin=False) is x

    def test_restricted_requires_binary_regardless_of_spin(self) -> None:
        # n_visible is not None => restricted => always {0, 1}, even though
        # spin=True (the default) would otherwise mean {-1, +1}.
        W = jnp.zeros((5, 5))
        with pytest.raises(ValueError):
            _validate_state(jnp.array([-1.0, 1.0, -1.0, 1.0, 1.0]), W, n_visible=3, spin=True)
        x = jnp.array([0.0, 1.0, 0.0, 1.0, 1.0])
        assert _validate_state(x, W, n_visible=3, spin=True) is x

    def test_sample_chain_propagates_state_validation_error(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        with pytest.raises(ValueError):
            sample_chain(key, jnp.zeros(n), W, jnp.zeros(n), steps=1)


# =========================================================================== #
# Restricted (bipartite) detection                                           #
# =========================================================================== #


class TestRestrictedDetection:
    def test_fully_connected_is_not_restricted(self, key: jax.Array) -> None:
        W = _random_symmetric_weights(key, 5)
        assert _restricted_n_visible(W) is None

    def test_zero_matrix_is_not_restricted(self) -> None:
        # An all-zero matrix trivially has no within-block couplings for any
        # split, but it also has no *cross*-block coupling -- without that
        # extra requirement it would be ambiguous with a fully-connected BM
        # that happens to have zero weights, which matters because the two
        # samplers use different unit conventions ({-1,+1} vs. {0,1}).
        W = jnp.zeros((4, 4))
        assert _restricted_n_visible(W) is None

    def test_bipartite_is_restricted(self, key: jax.Array) -> None:
        W = _bipartite_weights(key, n_visible=3, n_hidden=2)
        assert _restricted_n_visible(W) == 3

    def test_bipartite_with_different_split(self, key: jax.Array) -> None:
        W = _bipartite_weights(key, n_visible=2, n_hidden=4)
        assert _restricted_n_visible(W) == 2


# =========================================================================== #
# Shapes: n_samples is None (burn-in only)                                    #
# =========================================================================== #


class TestSampleChainBurnIn:
    def test_returns_bare_state(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x = sample_chain(key, jnp.ones(n), W, b, steps=10)
        assert x.shape == (n,)

    def test_zero_steps_is_identity(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        x = sample_chain(key, x0, W, b, steps=0)
        np.testing.assert_array_equal(np.asarray(x), np.asarray(x0))

    def test_no_bias_runs(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        x = sample_chain(key, jnp.ones(n), W, None, steps=5)
        assert x.shape == (n,)


# =========================================================================== #
# Shapes: n_samples given, avg=False, corr=False                              #
# =========================================================================== #


class TestSampleChainSamples:
    def test_basic_shape(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x, samples = sample_chain(key, jnp.ones(n), W, b, steps=1, n_samples=20)
        assert x.shape == (n,)
        assert samples.shape == (20, n)

    def test_final_state_matches_last_sample(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x, samples = sample_chain(key, jnp.ones(n), W, b, steps=1, n_samples=6)
        np.testing.assert_array_equal(np.asarray(x), np.asarray(samples[-1]))

    def test_spin_outputs_in_pm_one(self, key: jax.Array) -> None:
        n = 6
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        _, samples = sample_chain(key, jnp.ones(n), W, b, steps=1, n_samples=30)
        vals = set(np.unique(np.asarray(samples)).tolist())
        assert vals.issubset({-1.0, 1.0})

    def test_same_key_gives_same_samples(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        _, s1 = sample_chain(key, x0, W, b, steps=1, n_samples=10)
        _, s2 = sample_chain(key, x0, W, b, steps=1, n_samples=10)
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))

    def test_different_keys_differ(self) -> None:
        key1, key2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
        n = 5
        W = _random_symmetric_weights(key1, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        _, s1 = sample_chain(key1, x0, W, b, steps=1, n_samples=20)
        _, s2 = sample_chain(key2, x0, W, b, steps=1, n_samples=20)
        assert not np.array_equal(np.asarray(s1), np.asarray(s2))


# =========================================================================== #
# spin=False: fully-connected machine sampled in {0, 1} instead of {-1, +1}   #
# =========================================================================== #


class TestSampleChainSpinFalse:
    def test_binary_initial_state_required(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError):
            sample_chain(key, -jnp.ones(n), W, b, steps=1, spin=False)

    def test_outputs_are_binary(self, key: jax.Array) -> None:
        n = 6
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.zeros(n)
        _, samples = sample_chain(key, x0, W, b, steps=1, n_samples=30, spin=False)
        vals = set(np.unique(np.asarray(samples)).tolist())
        assert vals.issubset({0.0, 1.0})

    def test_strong_positive_bias_aligns_state(self) -> None:
        n = 4
        W = jnp.zeros((n, n))
        b = 5.0 * jnp.ones(n)
        key = jax.random.PRNGKey(0)
        x0 = sample_chain(key, jnp.zeros(n), W, b, steps=50, spin=False)
        _, samples = sample_chain(key, x0, W, b, steps=1, n_samples=200, spin=False)
        empirical_mean = np.asarray(jnp.mean(samples, axis=0))
        assert np.all(empirical_mean > 0.9)


# =========================================================================== #
# corr=True (avg=False): emit (outers, states)                                #
# =========================================================================== #


class TestSampleChainCorr:
    def test_output_shapes(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x, (outers, states) = sample_chain(
            key, jnp.ones(n), W, b, steps=1, n_samples=7, corr=True,
        )
        assert x.shape == (n,)
        assert outers.shape == (7, n, n)
        assert states.shape == (7, n)

    def test_outers_are_outer_products_of_states(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        _, (outers, states) = sample_chain(
            key, jnp.ones(n), W, b, steps=1, n_samples=6, corr=True,
        )
        expected = np.einsum("ti,tj->tij", np.asarray(states), np.asarray(states))
        np.testing.assert_array_equal(np.asarray(outers), expected)

    def test_states_match_default_trajectory(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        _, default_states = sample_chain(key, x0, W, b, steps=1, n_samples=6, corr=False)
        _, (_, corr_states) = sample_chain(key, x0, W, b, steps=1, n_samples=6, corr=True)
        np.testing.assert_array_equal(np.asarray(corr_states), np.asarray(default_states))


# =========================================================================== #
# avg=True (corr=False): return only the running mean of the chain states     #
# =========================================================================== #


class TestSampleChainAvg:
    def test_output_shapes(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x, x_mean = sample_chain(
            key, jnp.ones(n), W, b, steps=1, n_samples=15, avg=True,
        )
        assert x.shape == (n,)
        assert x_mean.shape == (n,)

    def test_mean_matches_mean_of_default_run(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        _, states = sample_chain(key, x0, W, b, steps=1, n_samples=8, avg=False)
        _, x_mean = sample_chain(key, x0, W, b, steps=1, n_samples=8, avg=True)
        np.testing.assert_allclose(
            np.asarray(x_mean), np.asarray(jnp.mean(states, axis=0)),
            rtol=1e-6, atol=1e-6,
        )

    def test_final_state_matches_default_run(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        avg_final, _ = sample_chain(key, x0, W, b, steps=1, n_samples=8, avg=True)
        default_final, _ = sample_chain(key, x0, W, b, steps=1, n_samples=8, avg=False)
        np.testing.assert_array_equal(np.asarray(avg_final), np.asarray(default_final))

    def test_mean_within_value_range(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        _, x_mean = sample_chain(key, jnp.ones(n), W, b, steps=1, n_samples=50, avg=True)
        x_mean_np = np.asarray(x_mean)
        assert np.all(x_mean_np >= -1.0 - 1e-6)
        assert np.all(x_mean_np <= 1.0 + 1e-6)


# =========================================================================== #
# corr=True, avg=True: return (x, outer_mean, x_mean)                         #
# =========================================================================== #


class TestSampleChainCorrAvg:
    def test_output_is_triple(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        out = sample_chain(
            key, jnp.ones(n), W, b, steps=1, n_samples=8, avg=True, corr=True,
        )
        assert isinstance(out, tuple)
        assert len(out) == 3
        x, outer_mean, x_mean = out
        assert x.shape == (n,)
        assert outer_mean.shape == (n, n)
        assert x_mean.shape == (n,)

    def test_means_match_means_of_corr_only_run(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        _, (outers, states) = sample_chain(
            key, x0, W, b, steps=1, n_samples=8, corr=True, avg=False,
        )
        _, outer_mean, x_mean = sample_chain(
            key, x0, W, b, steps=1, n_samples=8, corr=True, avg=True,
        )
        np.testing.assert_allclose(
            np.asarray(outer_mean), np.asarray(jnp.mean(outers, axis=0)),
            rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(x_mean), np.asarray(jnp.mean(states, axis=0)),
            rtol=1e-6, atol=1e-6,
        )

    def test_outer_mean_is_symmetric(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        _, outer_mean, _ = sample_chain(
            key, jnp.ones(n), W, b, steps=1, n_samples=6, avg=True, corr=True,
        )
        outer_mean_np = np.asarray(outer_mean)
        np.testing.assert_allclose(outer_mean_np, outer_mean_np.T, rtol=1e-6, atol=1e-6)


# =========================================================================== #
# Restricted machines actually get sampled via the block sampler              #
# =========================================================================== #


class TestSampleChainRestricted:
    def test_basic_shape(self, key: jax.Array) -> None:
        W = _bipartite_weights(key, n_visible=3, n_hidden=2)
        b = jnp.zeros(5)
        x, samples = sample_chain(key, jnp.zeros(5), W, b, steps=1, n_samples=10)
        assert x.shape == (5,)
        assert samples.shape == (10, 5)

    def test_outputs_are_binary(self, key: jax.Array) -> None:
        # The block-conditional RBM sampler operates in {0, 1}.
        W = _bipartite_weights(key, n_visible=3, n_hidden=2)
        b = jnp.zeros(5)
        _, samples = sample_chain(key, jnp.zeros(5), W, b, steps=1, n_samples=20)
        vals = set(np.unique(np.asarray(samples)).tolist())
        assert vals.issubset({0.0, 1.0})

    def test_strong_positive_bias_aligns_state(self) -> None:
        # Tiny (but nonzero) cross-block coupling keeps this genuinely
        # detected as restricted, while the strong bias dominates the
        # dynamics and drives every unit toward 1 (binary {0, 1} units).
        n_visible, n_hidden = 3, 2
        n = n_visible + n_hidden
        W = _bipartite_weights(
            jax.random.PRNGKey(1), n_visible, n_hidden, minval=-0.01, maxval=0.01,
        )
        b = 5.0 * jnp.ones(n)
        x, x_mean = sample_chain(
            jax.random.PRNGKey(0), jnp.zeros(n), W, b,
            steps=1, n_samples=200, avg=True,
        )
        assert np.all(np.asarray(x_mean) > 0.9)


# =========================================================================== #
# Statistical sanity                                                          #
# =========================================================================== #


class TestSampleChainStatistics:
    def test_zero_model_uniform_marginals(self) -> None:
        n = 5
        W = jnp.zeros((n, n))
        _, samples = sample_chain(
            jax.random.PRNGKey(0), jnp.ones(n), W, jnp.zeros(n),
            steps=1, n_samples=4000,
        )
        empirical_mean = np.asarray(jnp.mean(samples, axis=0))
        np.testing.assert_allclose(empirical_mean, np.zeros(n), atol=0.06)

    def test_strong_positive_bias_aligns_state(self) -> None:
        # Unlike the old ``sample_chain``, this API has no separate
        # burn-in phase built into the sampling call -- ``steps`` is purely
        # "steps between samples". So burn in explicitly first (composing
        # two calls, as the docstring's "advance the chain between calls"
        # note describes) before collecting samples.
        n = 4
        W = jnp.zeros((n, n))
        b = 5.0 * jnp.ones(n)
        key = jax.random.PRNGKey(0)
        x0 = sample_chain(key, -jnp.ones(n), W, b, steps=50)
        _, samples = sample_chain(key, x0, W, b, steps=1, n_samples=200)
        empirical_mean = np.asarray(jnp.mean(samples, axis=0))
        assert np.all(empirical_mean > 0.9)

    def test_strong_negative_bias_aligns_state(self) -> None:
        n = 4
        W = jnp.zeros((n, n))
        b = -5.0 * jnp.ones(n)
        key = jax.random.PRNGKey(0)
        x0 = sample_chain(key, jnp.ones(n), W, b, steps=50)
        _, samples = sample_chain(key, x0, W, b, steps=1, n_samples=200)
        empirical_mean = np.asarray(jnp.mean(samples, axis=0))
        assert np.all(empirical_mean < -0.9)
