"""Tests for :mod:`jaxbm.sample` (the ``BM_chain`` / ``RBM_chain`` array-in
entry points, and their ``_bm_validate`` / ``_rbm_validate`` helpers).

``BM_chain`` / ``RBM_chain`` always return the final state first; what else
comes back depends on ``mode`` (and the ``n_samples`` it requires) -- see
their docstrings for the full table. The shared ``key`` fixture is provided
by ``tests/conftest.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from jaxbm.sample import (
    BM_chain,
    RBM_chain,
    _bm_validate,
    _rbm_validate,
)


def _random_symmetric_weights(key, n, minval=-1.0, maxval=1.0):
    W = jax.random.uniform(key, (n, n), minval=0.5 * minval, maxval=0.5 * maxval)
    W = W + W.T
    W = W - jnp.diag(jnp.diagonal(W))
    return W


# =========================================================================== #
# Validation                                                                  #
# =========================================================================== #


class TestBMValidate:
    def test_non_square_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            _bm_validate(jnp.ones(2), jnp.ones((2, 3)), jnp.zeros(2), spin=True, clamp=None)

    def test_asymmetric_weights_raises(self) -> None:
        W = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="symmetric"):
            _bm_validate(jnp.ones(2), W, jnp.zeros(2), spin=True, clamp=None)

    def test_bias_size_mismatch_raises(self) -> None:
        W = jnp.zeros((3, 3))
        with pytest.raises(ValueError, match="bias"):
            _bm_validate(jnp.ones(3), W, jnp.zeros(4), spin=True, clamp=None)

    def test_none_bias_is_allowed(self) -> None:
        W = jnp.zeros((3, 3))
        _bm_validate(jnp.ones(3), W, None, spin=False, clamp=None)

    def test_wrong_length_state_raises(self) -> None:
        W = jnp.zeros((3, 3))
        with pytest.raises(ValueError, match="size"):
            _bm_validate(jnp.ones(2), W, None, spin=True, clamp=None)

    def test_non_vector_state_raises(self) -> None:
        W = jnp.zeros((2, 2))
        with pytest.raises(ValueError, match="size"):
            _bm_validate(jnp.ones((2, 2)), W, None, spin=True, clamp=None)

    def test_spin_true_rejects_binary_state(self) -> None:
        W = jnp.zeros((3, 3))
        with pytest.raises(ValueError, match=r"\{-1, \+1\}|-1.0, 1.0"):
            _bm_validate(jnp.zeros(3), W, None, spin=True, clamp=None)

    def test_spin_true_accepts_pm_one_state(self) -> None:
        W = jnp.zeros((3, 3))
        x = jnp.array([-1.0, 1.0, -1.0])
        _bm_validate(x, W, None, spin=True, clamp=None)

    def test_spin_false_rejects_pm_one_state(self) -> None:
        W = jnp.zeros((3, 3))
        with pytest.raises(ValueError):
            _bm_validate(jnp.array([-1.0, 1.0, -1.0]), W, None, spin=False, clamp=None)

    def test_spin_false_accepts_binary_state(self) -> None:
        W = jnp.zeros((3, 3))
        x = jnp.array([0.0, 1.0, 0.0])
        _bm_validate(x, W, None, spin=False, clamp=None)

    def test_clamp_none_is_allowed(self) -> None:
        W = jnp.zeros((3, 3))
        _bm_validate(jnp.ones(3), W, None, spin=True, clamp=None)

    def test_clamp_rejects_non_integer_array(self) -> None:
        W = jnp.zeros((4, 4))
        with pytest.raises(ValueError, match="integer"):
            _bm_validate(jnp.ones(4), W, None, spin=True, clamp=jnp.array([0.0]))

    def test_clamp_rejects_out_of_range_index(self) -> None:
        W = jnp.zeros((4, 4))
        with pytest.raises(ValueError, match=r"\[0, 4\)"):
            _bm_validate(jnp.ones(4), W, None, spin=True, clamp=jnp.array([4], dtype=jnp.int32))

    def test_clamp_accepts_valid_indices(self) -> None:
        W = jnp.zeros((4, 4))
        _bm_validate(jnp.ones(4), W, None, spin=True, clamp=jnp.array([0, 2], dtype=jnp.int32))

    def test_clamp_rejects_duplicate_indices(self) -> None:
        W = jnp.zeros((4, 4))
        with pytest.raises(ValueError, match="duplicate"):
            _bm_validate(
                jnp.ones(4), W, None, spin=True, clamp=jnp.array([0, 2, 0], dtype=jnp.int32)
            )

    def test_composes_with_checkify_under_jit(self) -> None:
        # The value-dependent checks (symmetry, unit convention, clamp bounds)
        # go through ``checkify.check`` instead of a bare ``if``/``raise``, so
        # ``_bm_validate`` itself must stay traceable: wrapping it in
        # ``checkify.checkify`` and calling that under ``jit`` must not raise
        # at trace time, and the resulting ``Error`` must carry the failure
        # (only surfaced once ``.throw()`` is called, outside the trace).
        W = jnp.array([[0.0, 1.0], [0.0, 0.0]])  # asymmetric

        @jax.jit
        def checked(x, W):
            err, _ = checkify.checkify(_bm_validate)(x, W, None, True, None)
            return err

        err = checked(jnp.ones(2), W)
        assert err.get() is not None
        with pytest.raises(ValueError, match="symmetric"):
            err.throw()

    def test_bm_chain_propagates_validation_error(self, key: jax.Array) -> None:
        with pytest.raises(ValueError):
            BM_chain(key, jnp.ones(2), jnp.ones((2, 3)), jnp.zeros(2), n_samples=None, steps=1)

    def test_bm_chain_propagates_state_validation_error(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        with pytest.raises(ValueError):
            BM_chain(key, jnp.zeros(n), W, jnp.zeros(n), n_samples=None, steps=1)


class TestRBMValidate:
    def test_non_2d_W_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            _rbm_validate(jnp.ones(3), jnp.ones(2), jnp.ones(3), None, None, spin=True, clamp=None)

    def test_b_v_size_mismatch_raises(self) -> None:
        W = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match="bias_v"):
            _rbm_validate(jnp.ones(3), jnp.ones(2), W, jnp.zeros(4), None, spin=True, clamp=None)

    def test_b_h_size_mismatch_raises(self) -> None:
        W = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match="bias_h"):
            _rbm_validate(jnp.ones(3), jnp.ones(2), W, None, jnp.zeros(5), spin=True, clamp=None)

    def test_none_biases_are_allowed(self) -> None:
        W = jnp.zeros((3, 2))
        _rbm_validate(jnp.ones(3), jnp.ones(2), W, None, None, spin=True, clamp=None)

    def test_wrong_length_x_v_raises(self) -> None:
        W = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match="x_v"):
            _rbm_validate(jnp.ones(2), jnp.ones(2), W, None, None, spin=True, clamp=None)

    def test_wrong_length_x_h_raises(self) -> None:
        W = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match="x_h"):
            _rbm_validate(jnp.ones(3), jnp.ones(3), W, None, None, spin=True, clamp=None)

    def test_spin_true_rejects_binary_state(self) -> None:
        W = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match=r"\{-1, \+1\}|-1.0, 1.0"):
            _rbm_validate(jnp.zeros(3), jnp.ones(2), W, None, None, spin=True, clamp=None)

    def test_spin_true_accepts_pm_one_state(self) -> None:
        W = jnp.zeros((3, 2))
        x_v = jnp.array([-1.0, 1.0, -1.0])
        x_h = jnp.array([1.0, -1.0])
        _rbm_validate(x_v, x_h, W, None, None, spin=True, clamp=None)

    def test_spin_false_rejects_pm_one_state(self) -> None:
        W = jnp.zeros((3, 2))
        with pytest.raises(ValueError):
            _rbm_validate(
                jnp.array([-1.0, 1.0, -1.0]), jnp.ones(2), W, None, None, spin=False, clamp=None,
            )

    def test_spin_false_accepts_binary_state(self) -> None:
        W = jnp.zeros((3, 2))
        x_v = jnp.array([0.0, 1.0, 0.0])
        x_h = jnp.array([1.0, 0.0])
        _rbm_validate(x_v, x_h, W, None, None, spin=False, clamp=None)

    def test_clamp_none_is_allowed(self) -> None:
        W = jnp.zeros((3, 2))
        _rbm_validate(jnp.ones(3), jnp.ones(2), W, None, None, spin=True, clamp=None)

    def test_clamp_rejects_non_integer_array(self) -> None:
        W = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match="integer"):
            _rbm_validate(
                jnp.ones(3), jnp.ones(2), W, None, None, spin=True, clamp=jnp.array([0.0]),
            )

    def test_clamp_rejects_out_of_range_index(self) -> None:
        W = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match=r"\[0, 3\)"):
            _rbm_validate(
                jnp.ones(3), jnp.ones(2), W, None, None,
                spin=True, clamp=jnp.array([3], dtype=jnp.int32),
            )

    def test_clamp_accepts_valid_indices(self) -> None:
        W = jnp.zeros((3, 2))
        _rbm_validate(
            jnp.ones(3), jnp.ones(2), W, None, None,
            spin=True, clamp=jnp.array([0, 2], dtype=jnp.int32),
        )

    def test_clamp_rejects_duplicate_indices(self) -> None:
        W = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match="duplicate"):
            _rbm_validate(
                jnp.ones(3), jnp.ones(2), W, None, None,
                spin=True, clamp=jnp.array([0, 0], dtype=jnp.int32),
            )

    def test_composes_with_checkify_under_jit(self) -> None:
        W = jnp.zeros((3, 2))

        @jax.jit
        def checked(x_v, x_h, W):
            err, _ = checkify.checkify(_rbm_validate)(x_v, x_h, W, None, None, True, None)
            return err

        err = checked(jnp.zeros(3), jnp.ones(2), W)  # spin=True rejects {0,1}
        assert err.get() is not None
        with pytest.raises(ValueError):
            err.throw()


# =========================================================================== #
# mode / n_samples agreement, clamp                                          #
# =========================================================================== #


class TestModeValidation:
    def test_invalid_mode_raises(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError, match="mode"):
            BM_chain(key, jnp.ones(n), W, b, steps=1, n_samples=5, mode="BOGUS")

    def test_given_n_samples_without_mode_defaults_to_hist(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        default_x, default_xs = BM_chain(key, jnp.ones(n), W, b, steps=1, n_samples=5)
        hist_x, hist_xs = BM_chain(key, jnp.ones(n), W, b, steps=1, n_samples=5, mode="HIST")
        np.testing.assert_array_equal(np.asarray(default_x), np.asarray(hist_x))
        np.testing.assert_array_equal(np.asarray(default_xs), np.asarray(hist_xs))

    def test_no_n_samples_forbids_mode(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError, match="mode"):
            BM_chain(key, jnp.ones(n), W, b, n_samples=None, steps=1, mode="HIST")

    def test_rejects_zero_n_samples(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError, match="positive"):
            BM_chain(key, jnp.ones(n), W, b, steps=1, n_samples=0, mode="HIST")

    def test_rejects_negative_n_samples(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError, match="positive"):
            BM_chain(key, jnp.ones(n), W, b, steps=1, n_samples=-3, mode="HIST")

    def test_rejects_non_int_n_samples(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError, match="int"):
            BM_chain(key, jnp.ones(n), W, b, steps=1, n_samples=2.5, mode="HIST")

    def test_clamp_rejects_non_integer_array(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError, match="integer"):
            BM_chain(key, jnp.ones(n), W, b, n_samples=None, steps=1, clamp=jnp.array([0.0]))

    def test_clamp_rejects_out_of_range_index(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError, match=r"\[0, 4\)"):
            BM_chain(
                key, jnp.ones(n), W, b, n_samples=None, steps=1,
                clamp=jnp.array([4], dtype=jnp.int32),
            )

    def test_clamp_rejects_duplicate_indices(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError, match="duplicate"):
            BM_chain(
                key, jnp.ones(n), W, b, n_samples=None, steps=1,
                clamp=jnp.array([1, 1], dtype=jnp.int32),
            )


# =========================================================================== #
# n_samples=None (default): burn-in / advance-and-discard                    #
# =========================================================================== #


class TestBMChainLast:
    def test_returns_bare_state(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x = BM_chain(key, jnp.ones(n), W, b, n_samples=None, steps=10)
        assert x.shape == (n,)

    def test_zero_steps_is_identity(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        x = BM_chain(key, x0, W, b, n_samples=None, steps=0)
        np.testing.assert_array_equal(np.asarray(x), np.asarray(x0))

    def test_no_bias_runs(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        x = BM_chain(key, jnp.ones(n), W, None, n_samples=None, steps=5)
        assert x.shape == (n,)


# =========================================================================== #
# clamp: hold a subset of units fixed throughout sampling                    #
# =========================================================================== #


class TestBMChainClamp:
    def test_clamped_units_never_change(self, key: jax.Array) -> None:
        n = 6
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
        clamp = jnp.array([0, 2, 4], dtype=jnp.int32)
        _, samples = BM_chain(
            key, x0, W, b, steps=1, n_samples=50, mode="HIST", clamp=clamp
        )
        np.testing.assert_array_equal(
            np.asarray(samples[:, clamp]),
            np.broadcast_to(np.asarray(x0[clamp]), (50, clamp.shape[0])),
        )

    def test_non_clamped_units_still_vary(self, key: jax.Array) -> None:
        n = 6
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        clamp = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)  # all but unit 5
        _, samples = BM_chain(
            key, x0, W, b, steps=1, n_samples=50, mode="HIST", clamp=clamp
        )
        assert np.unique(np.asarray(samples[:, 5])).size > 1

    def test_clamping_all_units_is_identity(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        clamp = jnp.arange(n, dtype=jnp.int32)
        x = BM_chain(key, x0, W, b, n_samples=None, steps=10, clamp=clamp)
        np.testing.assert_array_equal(np.asarray(x), np.asarray(x0))


class TestRBMChainClamp:
    def test_clamped_visible_units_never_change(self, key: jax.Array) -> None:
        n_v, n_h = 5, 3
        W = jax.random.normal(key, (n_v, n_h))
        b_v = jnp.zeros(n_v)
        b_h = jnp.zeros(n_h)
        x_v0 = jnp.where(jnp.arange(n_v) % 2 == 0, 1.0, -1.0)
        x_h0 = jnp.ones(n_h)
        clamp = jnp.array([0, 2, 4], dtype=jnp.int32)
        (_, _), (vs, _hs) = RBM_chain(
            key, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=50, mode="HIST", clamp=clamp
        )
        np.testing.assert_array_equal(
            np.asarray(vs[:, clamp]),
            np.broadcast_to(np.asarray(x_v0[clamp]), (50, clamp.shape[0])),
        )

    def test_non_clamped_visible_units_still_vary(self, key: jax.Array) -> None:
        n_v, n_h = 5, 3
        W = jax.random.normal(key, (n_v, n_h))
        b_v = jnp.zeros(n_v)
        b_h = jnp.zeros(n_h)
        x_v0 = jnp.ones(n_v)
        x_h0 = jnp.ones(n_h)
        clamp = jnp.array([0, 1, 2, 3], dtype=jnp.int32)  # all but unit 4
        (_, _), (vs, _hs) = RBM_chain(
            key, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=50, mode="HIST", clamp=clamp
        )
        assert np.unique(np.asarray(vs[:, 4])).size > 1

    def test_hidden_update_unaffected_by_clamp(self, key: jax.Array) -> None:
        n_v, n_h = 5, 3
        W = jax.random.normal(key, (n_v, n_h))
        b_v = jnp.zeros(n_v)
        b_h = jnp.zeros(n_h)
        x_v0 = jnp.ones(n_v)
        x_h0 = jnp.ones(n_h)
        clamp = jnp.arange(n_v, dtype=jnp.int32)  # clamp every visible unit
        (v, h), (_vs, hs) = RBM_chain(
            key, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=50, mode="HIST", clamp=clamp
        )
        np.testing.assert_array_equal(np.asarray(v), np.asarray(x_v0))
        assert np.unique(np.asarray(hs)).size > 1


# =========================================================================== #
# mode='HIST': stacked trajectory                                             #
# =========================================================================== #


class TestBMChainHist:
    def test_basic_shape(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x, samples = BM_chain(key, jnp.ones(n), W, b, steps=1, n_samples=20, mode="HIST")
        assert x.shape == (n,)
        assert samples.shape == (20, n)

    def test_final_state_matches_last_sample(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x, samples = BM_chain(key, jnp.ones(n), W, b, steps=1, n_samples=6, mode="HIST")
        np.testing.assert_array_equal(np.asarray(x), np.asarray(samples[-1]))

    def test_spin_outputs_in_pm_one(self, key: jax.Array) -> None:
        n = 6
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        _, samples = BM_chain(key, jnp.ones(n), W, b, steps=1, n_samples=30, mode="HIST")
        vals = set(np.unique(np.asarray(samples)).tolist())
        assert vals.issubset({-1.0, 1.0})

    def test_same_key_gives_same_samples(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        _, s1 = BM_chain(key, x0, W, b, steps=1, n_samples=10, mode="HIST")
        _, s2 = BM_chain(key, x0, W, b, steps=1, n_samples=10, mode="HIST")
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))

    def test_different_keys_differ(self) -> None:
        key1, key2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
        n = 5
        W = _random_symmetric_weights(key1, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        _, s1 = BM_chain(key1, x0, W, b, steps=1, n_samples=20, mode="HIST")
        _, s2 = BM_chain(key2, x0, W, b, steps=1, n_samples=20, mode="HIST")
        assert not np.array_equal(np.asarray(s1), np.asarray(s2))


# =========================================================================== #
# spin=False: fully-connected machine sampled in {0, 1} instead of {-1, +1}   #
# =========================================================================== #


class TestBMChainSpinFalse:
    def test_binary_initial_state_required(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError):
            BM_chain(key, -jnp.ones(n), W, b, n_samples=None, steps=1, spin=False)

    def test_outputs_are_binary(self, key: jax.Array) -> None:
        n = 6
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.zeros(n)
        _, samples = BM_chain(
            key, x0, W, b, steps=1, n_samples=30, mode="HIST", spin=False,
        )
        vals = set(np.unique(np.asarray(samples)).tolist())
        assert vals.issubset({0.0, 1.0})

    def test_strong_positive_bias_aligns_state(self) -> None:
        n = 4
        W = jnp.zeros((n, n))
        b = 5.0 * jnp.ones(n)
        key = jax.random.PRNGKey(0)
        x0 = BM_chain(key, jnp.zeros(n), W, b, n_samples=None, steps=50, spin=False)
        _, x_mean = BM_chain(
            key, x0, W, b, steps=1, n_samples=200, mode="MEAN", spin=False,
        )
        assert np.all(np.asarray(x_mean) > 0.9)


# =========================================================================== #
# mode='MEAN': running mean of the chain states, no stacked trajectory       #
# =========================================================================== #


class TestBMChainMean:
    def test_output_shapes(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x, x_mean = BM_chain(
            key, jnp.ones(n), W, b, steps=1, n_samples=15, mode="MEAN",
        )
        assert x.shape == (n,)
        assert x_mean.shape == (n,)

    def test_mean_matches_mean_of_hist_run(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        _, states = BM_chain(key, x0, W, b, steps=1, n_samples=8, mode="HIST")
        _, x_mean = BM_chain(key, x0, W, b, steps=1, n_samples=8, mode="MEAN")
        np.testing.assert_allclose(
            np.asarray(x_mean), np.asarray(jnp.mean(states, axis=0)),
            rtol=1e-6, atol=1e-6,
        )

    def test_final_state_matches_hist_run(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        mean_final, _ = BM_chain(key, x0, W, b, steps=1, n_samples=8, mode="MEAN")
        hist_final, _ = BM_chain(key, x0, W, b, steps=1, n_samples=8, mode="HIST")
        np.testing.assert_array_equal(np.asarray(mean_final), np.asarray(hist_final))

    def test_mean_within_value_range(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        _, x_mean = BM_chain(
            key, jnp.ones(n), W, b, steps=1, n_samples=50, mode="MEAN",
        )
        x_mean_np = np.asarray(x_mean)
        assert np.all(x_mean_np >= -1.0 - 1e-6)
        assert np.all(x_mean_np <= 1.0 + 1e-6)


# =========================================================================== #
# mode='CORR': running mean of outer(x, x)                                    #
# =========================================================================== #


class TestBMChainCorr:
    def test_output_shapes(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x, outer_mean = BM_chain(
            key, jnp.ones(n), W, b, steps=1, n_samples=8, mode="CORR",
        )
        assert x.shape == (n,)
        assert outer_mean.shape == (n, n)

    def test_matches_mean_of_outer_products_of_hist_run(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        _, states = BM_chain(key, x0, W, b, steps=1, n_samples=8, mode="HIST")
        _, outer_mean = BM_chain(key, x0, W, b, steps=1, n_samples=8, mode="CORR")
        states_np = np.asarray(states)
        expected = np.einsum("ti,tj->ij", states_np, states_np) / states_np.shape[0]
        np.testing.assert_allclose(np.asarray(outer_mean), expected, rtol=1e-6, atol=1e-6)

    def test_final_state_matches_hist_run(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        corr_final, _ = BM_chain(key, x0, W, b, steps=1, n_samples=8, mode="CORR")
        hist_final, _ = BM_chain(key, x0, W, b, steps=1, n_samples=8, mode="HIST")
        np.testing.assert_array_equal(np.asarray(corr_final), np.asarray(hist_final))

    def test_outer_mean_is_symmetric(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        _, outer_mean = BM_chain(
            key, jnp.ones(n), W, b, steps=1, n_samples=6, mode="CORR",
        )
        outer_mean_np = np.asarray(outer_mean)
        np.testing.assert_allclose(outer_mean_np, outer_mean_np.T, rtol=1e-6, atol=1e-6)


# =========================================================================== #
# Statistical sanity                                                          #
# =========================================================================== #


class TestBMChainStatistics:
    def test_zero_model_uniform_marginals(self) -> None:
        n = 5
        W = jnp.zeros((n, n))
        _, samples = BM_chain(
            jax.random.PRNGKey(0), jnp.ones(n), W, jnp.zeros(n),
            steps=1, n_samples=4000, mode="HIST",
        )
        empirical_mean = np.asarray(jnp.mean(samples, axis=0))
        np.testing.assert_allclose(empirical_mean, np.zeros(n), atol=0.06)

    def test_strong_positive_bias_aligns_state(self) -> None:
        # ``BM_chain`` has no separate burn-in phase built into the sampling
        # call -- ``steps`` is purely "steps between samples". So burn in
        # explicitly first (composing two calls, n_samples=None then
        # mode='HIST') before collecting samples.
        n = 4
        W = jnp.zeros((n, n))
        b = 5.0 * jnp.ones(n)
        key = jax.random.PRNGKey(0)
        x0 = BM_chain(key, -jnp.ones(n), W, b, n_samples=None, steps=50)
        _, samples = BM_chain(key, x0, W, b, steps=1, n_samples=200, mode="HIST")
        empirical_mean = np.asarray(jnp.mean(samples, axis=0))
        assert np.all(empirical_mean > 0.9)

    def test_strong_negative_bias_aligns_state(self) -> None:
        n = 4
        W = jnp.zeros((n, n))
        b = -5.0 * jnp.ones(n)
        key = jax.random.PRNGKey(0)
        x0 = BM_chain(key, jnp.ones(n), W, b, n_samples=None, steps=50)
        _, samples = BM_chain(key, x0, W, b, steps=1, n_samples=200, mode="HIST")
        empirical_mean = np.asarray(jnp.mean(samples, axis=0))
        assert np.all(empirical_mean < -0.9)


# =========================================================================== #
# jit / vmap compatibility (in_jit=True)                                     #
# =========================================================================== #

# ``steps``, ``n_samples``, ``mode``, and ``spin`` all control Python-level
# branching inside ``BM_chain`` / ``RBM_chain`` (or downstream in
# ``_sampler.py`` / ``_loop.py``), so they must be static under ``jit``.
# ``in_jit`` is a plain Python bool used the same way, so it must be static
# too. ``weights``/``bias``/``clamp``/``x`` stay dynamic -- ``clamp``'s
# "is it None" / "how many indices" branching only ever depends on its
# *shape*, which is static even for a traced array, so it needs no special
# treatment.
_STATIC_ARGNAMES = ("steps", "n_samples", "mode", "spin", "in_jit")


class TestBMChainJit:
    def test_plain_jit_without_in_jit_raises(self, key: jax.Array) -> None:
        # ``BM_chain``'s default validation calls ``checkify.checkify(...)``
        # then eagerly ``.throw()``s -- that needs a concrete answer to "did
        # a check fail?", which isn't available while still being traced by
        # an *outer* ``jit``. This is exactly why ``in_jit=True`` exists;
        # see the tests below.
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        jitted = jax.jit(BM_chain, static_argnames=("steps", "n_samples", "mode", "spin"))
        with pytest.raises(ValueError, match="checkify"):
            jitted(key, jnp.ones(n), W, b, steps=1, n_samples=5, mode="HIST")

    def test_jit_matches_eager_last(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        jitted = jax.jit(BM_chain, static_argnames=_STATIC_ARGNAMES)
        x_eager = BM_chain(key, x0, W, b, n_samples=None, steps=5, in_jit=False)
        x_jit = jitted(key, x0, W, b, n_samples=None, steps=5, in_jit=True)
        np.testing.assert_array_equal(np.asarray(x_eager), np.asarray(x_jit))

    def test_jit_matches_eager_hist(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        jitted = jax.jit(BM_chain, static_argnames=_STATIC_ARGNAMES)
        x_e, s_e = BM_chain(key, x0, W, b, steps=1, n_samples=10, mode="HIST", in_jit=False)
        x_j, s_j = jitted(key, x0, W, b, steps=1, n_samples=10, mode="HIST", in_jit=True)
        np.testing.assert_array_equal(np.asarray(x_e), np.asarray(x_j))
        np.testing.assert_array_equal(np.asarray(s_e), np.asarray(s_j))

    def test_jit_matches_eager_mean(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        jitted = jax.jit(BM_chain, static_argnames=_STATIC_ARGNAMES)
        x_e, m_e = BM_chain(key, x0, W, b, steps=1, n_samples=10, mode="MEAN", in_jit=False)
        x_j, m_j = jitted(key, x0, W, b, steps=1, n_samples=10, mode="MEAN", in_jit=True)
        np.testing.assert_array_equal(np.asarray(x_e), np.asarray(x_j))
        np.testing.assert_array_equal(np.asarray(m_e), np.asarray(m_j))

    def test_jit_matches_eager_corr(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        jitted = jax.jit(BM_chain, static_argnames=_STATIC_ARGNAMES)
        x_e, c_e = BM_chain(key, x0, W, b, steps=1, n_samples=10, mode="CORR", in_jit=False)
        x_j, c_j = jitted(key, x0, W, b, steps=1, n_samples=10, mode="CORR", in_jit=True)
        np.testing.assert_array_equal(np.asarray(x_e), np.asarray(x_j))
        np.testing.assert_array_equal(np.asarray(c_e), np.asarray(c_j))

    def test_jit_respects_clamp(self, key: jax.Array) -> None:
        # ``clamp`` is passed as a normal (dynamic) traced argument here --
        # no need to mark it static.
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
        clamp = jnp.array([0, 2, 4], dtype=jnp.int32)
        jitted = jax.jit(BM_chain, static_argnames=_STATIC_ARGNAMES)
        _, samples = jitted(
            key, x0, W, b, steps=1, n_samples=20, mode="HIST", clamp=clamp, in_jit=True
        )
        np.testing.assert_array_equal(
            np.asarray(samples[:, clamp]),
            np.broadcast_to(np.asarray(x0[clamp]), (20, clamp.shape[0])),
        )


class TestBMChainVmap:
    def test_vmap_over_key_matches_sequential_calls(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        keys = jax.random.split(key, 6)

        def one(k):
            return BM_chain(k, x0, W, b, steps=1, n_samples=8, mode="HIST", in_jit=True)

        batched_x, batched_samples = jax.vmap(one)(keys)
        assert batched_x.shape == (6, n)
        assert batched_samples.shape == (6, 8, n)
        for i in range(6):
            x_i, s_i = BM_chain(keys[i], x0, W, b, steps=1, n_samples=8, mode="HIST")
            np.testing.assert_array_equal(np.asarray(batched_x[i]), np.asarray(x_i))
            np.testing.assert_array_equal(np.asarray(batched_samples[i]), np.asarray(s_i))

    def test_vmap_over_key_and_x0(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        keys = jax.random.split(key, 5)
        x0s = jnp.where(jax.random.bernoulli(key, shape=(5, n)), 1.0, -1.0)

        def one(k, x0):
            return BM_chain(k, x0, W, b, steps=1, n_samples=6, mode="MEAN", in_jit=True)

        batched_x, batched_mean = jax.vmap(one)(keys, x0s)
        assert batched_x.shape == (5, n)
        assert batched_mean.shape == (5, n)
        for i in range(5):
            x_i, m_i = BM_chain(keys[i], x0s[i], W, b, steps=1, n_samples=6, mode="MEAN")
            np.testing.assert_array_equal(np.asarray(batched_x[i]), np.asarray(x_i))
            np.testing.assert_array_equal(np.asarray(batched_mean[i]), np.asarray(m_i))

    def test_vmap_over_key_x0_weights_bias(self, key: jax.Array) -> None:
        n = 4
        n_models = 3
        keys = jax.random.split(key, n_models)
        Ws = jnp.stack([_random_symmetric_weights(k, n) for k in keys])
        bs = jnp.zeros((n_models, n))
        x0s = jnp.ones((n_models, n))

        def one(k, x0, W, b):
            return BM_chain(k, x0, W, b, steps=1, n_samples=5, mode="CORR", in_jit=True)

        batched_x, batched_corr = jax.vmap(one)(keys, x0s, Ws, bs)
        assert batched_x.shape == (n_models, n)
        assert batched_corr.shape == (n_models, n, n)
        for i in range(n_models):
            x_i, c_i = BM_chain(keys[i], x0s[i], Ws[i], bs[i], steps=1, n_samples=5, mode="CORR")
            np.testing.assert_array_equal(np.asarray(batched_x[i]), np.asarray(x_i))
            np.testing.assert_array_equal(np.asarray(batched_corr[i]), np.asarray(c_i))

    def test_jit_of_vmap(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        keys = jax.random.split(key, 4)

        def one(k):
            return BM_chain(k, x0, W, b, steps=1, n_samples=5, mode="HIST", in_jit=True)

        jitted_vmapped = jax.jit(jax.vmap(one))
        batched_x, batched_samples = jitted_vmapped(keys)
        assert batched_x.shape == (4, n)
        assert batched_samples.shape == (4, 5, n)

    def test_bare_vmap_does_not_need_in_jit(self, key: jax.Array) -> None:
        """Unlike ``jax.jit``, a bare ``jax.vmap`` (no enclosing ``jit``)
        executes eagerly on concrete, batched values, so the default
        ``in_jit=False`` validation works -- and still catches bad input.
        """
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        keys = jax.random.split(key, 4)

        def one(k):
            return BM_chain(k, x0, W, b, steps=1, n_samples=5, mode="HIST")

        batched_x, batched_samples = jax.vmap(one)(keys)
        assert batched_x.shape == (4, n)
        assert batched_samples.shape == (4, 5, n)

        bad_x0 = jnp.full((n,), 5.0)

        def bad(k):
            return BM_chain(k, bad_x0, W, b, steps=1, n_samples=5, mode="HIST")

        with pytest.raises(Exception, match="only contain values"):
            jax.vmap(bad)(keys)


class TestRBMChainJit:
    def test_plain_jit_without_in_jit_raises(self, key: jax.Array) -> None:
        n_v, n_h = 3, 2
        W = jax.random.normal(key, (n_v, n_h))
        b_v, b_h = jnp.zeros(n_v), jnp.zeros(n_h)
        jitted = jax.jit(RBM_chain, static_argnames=("steps", "n_samples", "mode", "spin"))
        with pytest.raises(ValueError, match="checkify"):
            jitted(
                key, jnp.ones(n_v), jnp.ones(n_h), W, b_v, b_h,
                steps=1, n_samples=5, mode="HIST",
            )

    def test_jit_matches_eager_last(self, key: jax.Array) -> None:
        n_v, n_h = 3, 2
        W = jax.random.normal(key, (n_v, n_h))
        b_v, b_h = jnp.zeros(n_v), jnp.zeros(n_h)
        x_v0, x_h0 = jnp.ones(n_v), jnp.ones(n_h)
        jitted = jax.jit(RBM_chain, static_argnames=_STATIC_ARGNAMES)
        v_e, h_e = RBM_chain(key, x_v0, x_h0, W, b_v, b_h, n_samples=None, steps=5, in_jit=False)
        v_j, h_j = jitted(key, x_v0, x_h0, W, b_v, b_h, n_samples=None, steps=5, in_jit=True)
        np.testing.assert_array_equal(np.asarray(v_e), np.asarray(v_j))
        np.testing.assert_array_equal(np.asarray(h_e), np.asarray(h_j))

    def test_jit_matches_eager_hist(self, key: jax.Array) -> None:
        n_v, n_h = 3, 2
        W = jax.random.normal(key, (n_v, n_h))
        b_v, b_h = jnp.zeros(n_v), jnp.zeros(n_h)
        x_v0, x_h0 = jnp.ones(n_v), jnp.ones(n_h)
        jitted = jax.jit(RBM_chain, static_argnames=_STATIC_ARGNAMES)
        (v_e, h_e), (vs_e, hs_e) = RBM_chain(
            key, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=10, mode="HIST", in_jit=False
        )
        (v_j, h_j), (vs_j, hs_j) = jitted(
            key, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=10, mode="HIST", in_jit=True
        )
        np.testing.assert_array_equal(np.asarray(v_e), np.asarray(v_j))
        np.testing.assert_array_equal(np.asarray(h_e), np.asarray(h_j))
        np.testing.assert_array_equal(np.asarray(vs_e), np.asarray(vs_j))
        np.testing.assert_array_equal(np.asarray(hs_e), np.asarray(hs_j))

    def test_jit_matches_eager_mean(self, key: jax.Array) -> None:
        n_v, n_h = 3, 2
        W = jax.random.normal(key, (n_v, n_h))
        b_v, b_h = jnp.zeros(n_v), jnp.zeros(n_h)
        x_v0, x_h0 = jnp.ones(n_v), jnp.ones(n_h)
        jitted = jax.jit(RBM_chain, static_argnames=_STATIC_ARGNAMES)
        (v_e, h_e), (vm_e, hm_e) = RBM_chain(
            key, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=10, mode="MEAN", in_jit=False
        )
        (v_j, h_j), (vm_j, hm_j) = jitted(
            key, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=10, mode="MEAN", in_jit=True
        )
        np.testing.assert_array_equal(np.asarray(v_e), np.asarray(v_j))
        np.testing.assert_array_equal(np.asarray(h_e), np.asarray(h_j))
        np.testing.assert_array_equal(np.asarray(vm_e), np.asarray(vm_j))
        np.testing.assert_array_equal(np.asarray(hm_e), np.asarray(hm_j))

    def test_jit_matches_eager_corr(self, key: jax.Array) -> None:
        n_v, n_h = 3, 2
        W = jax.random.normal(key, (n_v, n_h))
        b_v, b_h = jnp.zeros(n_v), jnp.zeros(n_h)
        x_v0, x_h0 = jnp.ones(n_v), jnp.ones(n_h)
        jitted = jax.jit(RBM_chain, static_argnames=_STATIC_ARGNAMES)
        (v_e, h_e), c_e = RBM_chain(
            key, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=10, mode="CORR", in_jit=False
        )
        (v_j, h_j), c_j = jitted(
            key, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=10, mode="CORR", in_jit=True
        )
        np.testing.assert_array_equal(np.asarray(v_e), np.asarray(v_j))
        np.testing.assert_array_equal(np.asarray(h_e), np.asarray(h_j))
        np.testing.assert_array_equal(np.asarray(c_e), np.asarray(c_j))

    def test_jit_respects_clamp(self, key: jax.Array) -> None:
        n_v, n_h = 5, 3
        W = jax.random.normal(key, (n_v, n_h))
        b_v, b_h = jnp.zeros(n_v), jnp.zeros(n_h)
        x_v0 = jnp.where(jnp.arange(n_v) % 2 == 0, 1.0, -1.0)
        x_h0 = jnp.ones(n_h)
        clamp = jnp.array([0, 2, 4], dtype=jnp.int32)
        jitted = jax.jit(RBM_chain, static_argnames=_STATIC_ARGNAMES)
        (_, _), (vs, _hs) = jitted(
            key, x_v0, x_h0, W, b_v, b_h,
            steps=1, n_samples=20, mode="HIST", clamp=clamp, in_jit=True,
        )
        np.testing.assert_array_equal(
            np.asarray(vs[:, clamp]),
            np.broadcast_to(np.asarray(x_v0[clamp]), (20, clamp.shape[0])),
        )


class TestRBMChainVmap:
    def test_vmap_over_key_matches_sequential_calls(self, key: jax.Array) -> None:
        n_v, n_h = 3, 2
        W = jax.random.normal(key, (n_v, n_h))
        b_v, b_h = jnp.zeros(n_v), jnp.zeros(n_h)
        x_v0, x_h0 = jnp.ones(n_v), jnp.ones(n_h)
        keys = jax.random.split(key, 6)

        def one(k):
            return RBM_chain(
                k, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=8, mode="HIST", in_jit=True
            )

        (batched_v, batched_h), (batched_vs, batched_hs) = jax.vmap(one)(keys)
        assert batched_v.shape == (6, n_v)
        assert batched_h.shape == (6, n_h)
        assert batched_vs.shape == (6, 8, n_v)
        assert batched_hs.shape == (6, 8, n_h)
        for i in range(6):
            (v_i, h_i), (vs_i, hs_i) = RBM_chain(
                keys[i], x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=8, mode="HIST"
            )
            np.testing.assert_array_equal(np.asarray(batched_v[i]), np.asarray(v_i))
            np.testing.assert_array_equal(np.asarray(batched_h[i]), np.asarray(h_i))
            np.testing.assert_array_equal(np.asarray(batched_vs[i]), np.asarray(vs_i))
            np.testing.assert_array_equal(np.asarray(batched_hs[i]), np.asarray(hs_i))

    def test_vmap_over_key_and_x0(self, key: jax.Array) -> None:
        n_v, n_h = 3, 2
        W = jax.random.normal(key, (n_v, n_h))
        b_v, b_h = jnp.zeros(n_v), jnp.zeros(n_h)
        keys = jax.random.split(key, 5)
        x_v0s = jnp.where(jax.random.bernoulli(key, shape=(5, n_v)), 1.0, -1.0)
        x_h0s = jnp.where(jax.random.bernoulli(key, shape=(5, n_h)), 1.0, -1.0)

        def one(k, x_v0, x_h0):
            return RBM_chain(
                k, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=6, mode="MEAN", in_jit=True
            )

        (batched_v, batched_h), (batched_vm, batched_hm) = jax.vmap(one)(keys, x_v0s, x_h0s)
        assert batched_v.shape == (5, n_v)
        assert batched_vm.shape == (5, n_v)
        assert batched_hm.shape == (5, n_h)
        for i in range(5):
            (v_i, h_i), (vm_i, hm_i) = RBM_chain(
                keys[i], x_v0s[i], x_h0s[i], W, b_v, b_h, steps=1, n_samples=6, mode="MEAN"
            )
            np.testing.assert_array_equal(np.asarray(batched_v[i]), np.asarray(v_i))
            np.testing.assert_array_equal(np.asarray(batched_h[i]), np.asarray(h_i))
            np.testing.assert_array_equal(np.asarray(batched_vm[i]), np.asarray(vm_i))
            np.testing.assert_array_equal(np.asarray(batched_hm[i]), np.asarray(hm_i))

    def test_vmap_over_key_x0_weights_bias(self, key: jax.Array) -> None:
        n_v, n_h = 3, 2
        n_models = 3
        keys = jax.random.split(key, n_models)
        Ws = jnp.stack([jax.random.normal(k, (n_v, n_h)) for k in keys])
        bvs = jnp.zeros((n_models, n_v))
        bhs = jnp.zeros((n_models, n_h))
        x_v0s = jnp.ones((n_models, n_v))
        x_h0s = jnp.ones((n_models, n_h))

        def one(k, x_v0, x_h0, W, b_v, b_h):
            return RBM_chain(
                k, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=5, mode="CORR", in_jit=True
            )

        (batched_v, batched_h), batched_corr = jax.vmap(one)(
            keys, x_v0s, x_h0s, Ws, bvs, bhs
        )
        assert batched_v.shape == (n_models, n_v)
        assert batched_corr.shape == (n_models, n_v, n_h)
        for i in range(n_models):
            (v_i, h_i), c_i = RBM_chain(
                keys[i], x_v0s[i], x_h0s[i], Ws[i], bvs[i], bhs[i],
                steps=1, n_samples=5, mode="CORR",
            )
            np.testing.assert_array_equal(np.asarray(batched_v[i]), np.asarray(v_i))
            np.testing.assert_array_equal(np.asarray(batched_h[i]), np.asarray(h_i))
            np.testing.assert_array_equal(np.asarray(batched_corr[i]), np.asarray(c_i))

    def test_jit_of_vmap(self, key: jax.Array) -> None:
        n_v, n_h = 3, 2
        W = jax.random.normal(key, (n_v, n_h))
        b_v, b_h = jnp.zeros(n_v), jnp.zeros(n_h)
        x_v0, x_h0 = jnp.ones(n_v), jnp.ones(n_h)
        keys = jax.random.split(key, 4)

        def one(k):
            return RBM_chain(
                k, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=5, mode="HIST", in_jit=True
            )

        jitted_vmapped = jax.jit(jax.vmap(one))
        (batched_v, batched_h), (batched_vs, batched_hs) = jitted_vmapped(keys)
        assert batched_v.shape == (4, n_v)
        assert batched_h.shape == (4, n_h)
        assert batched_vs.shape == (4, 5, n_v)
        assert batched_hs.shape == (4, 5, n_h)

    def test_bare_vmap_does_not_need_in_jit(self, key: jax.Array) -> None:
        """Unlike ``jax.jit``, a bare ``jax.vmap`` (no enclosing ``jit``)
        executes eagerly on concrete, batched values, so the default
        ``in_jit=False`` validation works -- and still catches bad input.
        """
        n_v, n_h = 3, 2
        W = jax.random.normal(key, (n_v, n_h))
        b_v, b_h = jnp.zeros(n_v), jnp.zeros(n_h)
        x_v0, x_h0 = jnp.ones(n_v), jnp.ones(n_h)
        keys = jax.random.split(key, 4)

        def one(k):
            return RBM_chain(k, x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=5, mode="HIST")

        (batched_v, batched_h), (batched_vs, batched_hs) = jax.vmap(one)(keys)
        assert batched_v.shape == (4, n_v)
        assert batched_vs.shape == (4, 5, n_v)

        bad_x_v0 = jnp.full((n_v,), 5.0)

        def bad(k):
            return RBM_chain(k, bad_x_v0, x_h0, W, b_v, b_h, steps=1, n_samples=5, mode="HIST")

        with pytest.raises(Exception, match="only contain values"):
            jax.vmap(bad)(keys)
