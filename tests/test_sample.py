"""Tests for :mod:`jax_bm.sample` (the ``BM_chain`` array-in entry point).

``BM_chain`` always returns the final state ``x`` first; what else comes
back depends on ``mode`` (and the ``n_samples`` it requires) -- see its
docstring for the full table. The shared ``key`` fixture is provided by
``tests/conftest.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.experimental import checkify

from jax_bm.sample import (
    _bm_validate,
    _rbm_validate,
    BM_chain,
    RBM_chain,
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
            BM_chain(key, jnp.ones(2), jnp.ones((2, 3)), jnp.zeros(2), steps=1)

    def test_bm_chain_propagates_state_validation_error(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        with pytest.raises(ValueError):
            BM_chain(key, jnp.zeros(n), W, jnp.zeros(n), steps=1)


class TestRBMValidate:
    def test_non_2d_W_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            _rbm_validate(jnp.ones(3), jnp.ones(2), jnp.ones(3), None, None, spin=True, clamp=None)

    def test_b_v_size_mismatch_raises(self) -> None:
        W = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match="b_v"):
            _rbm_validate(jnp.ones(3), jnp.ones(2), W, jnp.zeros(4), None, spin=True, clamp=None)

    def test_b_h_size_mismatch_raises(self) -> None:
        W = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match="b_h"):
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

    def test_given_n_samples_requires_mode(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError, match="mode"):
            BM_chain(key, jnp.ones(n), W, b, steps=1, n_samples=5)

    def test_no_n_samples_forbids_mode(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError, match="mode"):
            BM_chain(key, jnp.ones(n), W, b, steps=1, mode="HIST")

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
            BM_chain(key, jnp.ones(n), W, b, steps=1, clamp=jnp.array([0.0]))

    def test_clamp_rejects_out_of_range_index(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError, match=r"\[0, 4\)"):
            BM_chain(key, jnp.ones(n), W, b, steps=1, clamp=jnp.array([4], dtype=jnp.int32))

    def test_clamp_rejects_duplicate_indices(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        with pytest.raises(ValueError, match="duplicate"):
            BM_chain(key, jnp.ones(n), W, b, steps=1, clamp=jnp.array([1, 1], dtype=jnp.int32))


# =========================================================================== #
# n_samples=None (default): burn-in / advance-and-discard                    #
# =========================================================================== #


class TestBMChainLast:
    def test_returns_bare_state(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x = BM_chain(key, jnp.ones(n), W, b, steps=10)
        assert x.shape == (n,)

    def test_zero_steps_is_identity(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        x0 = jnp.ones(n)
        x = BM_chain(key, x0, W, b, steps=0)
        np.testing.assert_array_equal(np.asarray(x), np.asarray(x0))

    def test_no_bias_runs(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        x = BM_chain(key, jnp.ones(n), W, None, steps=5)
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
        x = BM_chain(key, x0, W, b, steps=10, clamp=clamp)
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
            BM_chain(key, -jnp.ones(n), W, b, steps=1, spin=False)

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
        x0 = BM_chain(key, jnp.zeros(n), W, b, steps=50, spin=False)
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
        x0 = BM_chain(key, -jnp.ones(n), W, b, steps=50)
        _, samples = BM_chain(key, x0, W, b, steps=1, n_samples=200, mode="HIST")
        empirical_mean = np.asarray(jnp.mean(samples, axis=0))
        assert np.all(empirical_mean > 0.9)

    def test_strong_negative_bias_aligns_state(self) -> None:
        n = 4
        W = jnp.zeros((n, n))
        b = -5.0 * jnp.ones(n)
        key = jax.random.PRNGKey(0)
        x0 = BM_chain(key, jnp.ones(n), W, b, steps=50)
        _, samples = BM_chain(key, x0, W, b, steps=1, n_samples=200, mode="HIST")
        empirical_mean = np.asarray(jnp.mean(samples, axis=0))
        assert np.all(empirical_mean < -0.9)
