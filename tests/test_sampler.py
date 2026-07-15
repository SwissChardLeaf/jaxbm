"""Tests for `jaxbm._sampler` (the low-level conditional sampler builders).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxbm._sampler import (
    _BM_sampler,
    _bm_update,
    _RBM_sampler,
    _rbm_update_hidden,
    _rbm_update_visible,
)


def _random_symmetric_weights(key, n, minval=-1.0, maxval=1.0):
    W = jax.random.uniform(key, (n, n), minval=0.5 * minval, maxval=0.5 * maxval)
    W = W + W.T
    W = W - jnp.diag(jnp.diagonal(W))
    return W


def _vmap_trials(fn, key, n_trials):
    """Call ``fn(subkey)`` for ``n_trials`` independent subkeys of ``key``.

    Vectorized via ``vmap`` -- a single dispatch for all trials -- instead of
    a slow, un-jitted Python loop calling ``fn`` one key at a time.
    """
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(n_trials))
    return jax.vmap(fn)(keys)


# =========================================================================== #
# _bm_update: one single-unit Gibbs update                                   #
# =========================================================================== #


class TestBmUpdate:
    def test_only_the_chosen_unit_can_change(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        unit_p = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])  # always "choose" unit 2
        x0 = jnp.ones(n)
        _, x1 = _bm_update(W, None, key, x0, unit_p, spin=True)
        changed = np.flatnonzero(np.asarray(x0) != np.asarray(x1))
        assert set(changed.tolist()).issubset({2})

    def test_strong_positive_field_sets_plus_one(self, key: jax.Array) -> None:
        n = 3
        W = jnp.zeros((n, n))
        bias = jnp.array([0.0, 50.0, 0.0])
        unit_p = jnp.array([0.0, 1.0, 0.0])
        x0 = -jnp.ones(n)
        _, x1 = _bm_update(W, bias, key, x0, unit_p, spin=True)
        assert x1[1] == 1.0

    def test_strong_negative_field_sets_minus_one(self, key: jax.Array) -> None:
        n = 3
        W = jnp.zeros((n, n))
        bias = jnp.array([0.0, -50.0, 0.0])
        unit_p = jnp.array([0.0, 1.0, 0.0])
        x0 = jnp.ones(n)
        _, x1 = _bm_update(W, bias, key, x0, unit_p, spin=True)
        assert x1[1] == -1.0

    def test_field_comes_from_neighbors_not_own_value(self, key: jax.Array) -> None:
        # The unit's own (about-to-be-overwritten) value must not leak into its
        # own field -- `_bm_update` zeroes it out (`x.at[unit].set(0)`) before
        # taking the dot product with `weights[unit]`.
        W = jnp.array([[0.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 0.0]])
        unit_p = jnp.array([0.0, 1.0, 0.0])
        x0 = jnp.array([0.0, 1.0, 0.0])
        # If x[1]'s own value (1.0) leaked into its field via W[1, 1] = 100, the
        # update would be forced to +1 regardless of key; since it's zeroed out
        # first, the field is actually 0 and the outcome is a fair coin flip.
        outcomes = {
            float(_bm_update(W, None, jax.random.fold_in(key, i), x0, unit_p, True)[1][1])
            for i in range(20)
        }
        assert outcomes == {-1.0, 1.0}

    def test_spin_false_writes_binary(self, key: jax.Array) -> None:
        n = 3
        W = jnp.zeros((n, n))
        bias = jnp.array([0.0, 50.0, 0.0])
        unit_p = jnp.array([0.0, 1.0, 0.0])
        x0 = jnp.zeros(n)
        _, x1 = _bm_update(W, bias, key, x0, unit_p, spin=False)
        assert x1[1] == 1.0
        assert set(np.unique(np.asarray(x1)).tolist()).issubset({0.0, 1.0})

    def test_none_bias_is_allowed(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        unit_p = jnp.full((n,), 1.0 / n)
        _, x1 = _bm_update(W, None, key, jnp.ones(n), unit_p, spin=True)
        assert x1.shape == (n,)

    def test_advances_the_key(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        unit_p = jnp.full((n,), 1.0 / n)
        new_key, _ = _bm_update(W, None, key, jnp.ones(n), unit_p, spin=True)
        assert not np.array_equal(np.asarray(new_key), np.asarray(key))

    def test_same_key_gives_same_result(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        unit_p = jnp.full((n,), 1.0 / n)
        x0 = jnp.ones(n)
        _, x1 = _bm_update(W, None, key, x0, unit_p, spin=True)
        _, x2 = _bm_update(W, None, key, x0, unit_p, spin=True)
        np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))


# =========================================================================== #
# _BM_sampler: single-unit Gibbs sampler builder                             #
# =========================================================================== #


class TestBMSampler:
    def test_shape_and_dtype_preserved(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        sampler = _BM_sampler(W, b, sampler_steps=3, spin=True, clamp=None)
        x0 = jnp.ones(n)
        _, x1 = sampler(key, x0)
        assert x1.shape == (n,)
        assert x1.dtype == x0.dtype

    def test_zero_steps_is_identity(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        sampler = _BM_sampler(W, None, sampler_steps=0, spin=True, clamp=None)
        x0 = jnp.ones(n)
        _, x1 = sampler(key, x0)
        np.testing.assert_array_equal(np.asarray(x0), np.asarray(x1))

    def test_same_key_gives_same_result(self, key: jax.Array) -> None:
        n = 5
        W = _random_symmetric_weights(key, n)
        sampler = _BM_sampler(W, None, sampler_steps=5, spin=True, clamp=None)
        _, x1 = sampler(key, jnp.ones(n))
        _, x2 = sampler(key, jnp.ones(n))
        np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))

    def test_different_keys_can_differ(self) -> None:
        key1, key2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
        n = 6
        W = _random_symmetric_weights(key1, n)
        sampler = _BM_sampler(W, None, sampler_steps=10, spin=True, clamp=None)
        _, x1 = sampler(key1, jnp.ones(n))
        _, x2 = sampler(key2, jnp.ones(n))
        assert not np.array_equal(np.asarray(x1), np.asarray(x2))

    def test_spin_true_outputs_in_pm_one(self, key: jax.Array) -> None:
        n = 6
        W = _random_symmetric_weights(key, n)
        sampler = _BM_sampler(W, None, sampler_steps=30, spin=True, clamp=None)
        _, x1 = sampler(key, jnp.ones(n))
        assert set(np.unique(np.asarray(x1)).tolist()).issubset({-1.0, 1.0})

    def test_spin_false_outputs_binary(self, key: jax.Array) -> None:
        n = 6
        W = _random_symmetric_weights(key, n)
        sampler = _BM_sampler(W, None, sampler_steps=30, spin=False, clamp=None)
        _, x1 = sampler(key, jnp.zeros(n))
        assert set(np.unique(np.asarray(x1)).tolist()).issubset({0.0, 1.0})

    def test_clamp_none_can_change_every_position(self, key: jax.Array) -> None:
        n = 5
        W = jnp.zeros((n, n))  # zero field everywhere -> fair coin flips
        sampler = _BM_sampler(W, None, sampler_steps=1, spin=True, clamp=None)
        x0 = jnp.ones(n)
        _, xs = _vmap_trials(lambda k: sampler(k, x0), key, 300)
        touched = np.any(np.asarray(xs) != np.asarray(x0), axis=0)
        assert touched.all()

    def test_clamped_units_never_change(self, key: jax.Array) -> None:
        n = 6
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        clamp = jnp.array([1, 3, 5], dtype=jnp.int32)
        sampler = _BM_sampler(W, b, sampler_steps=300, spin=True, clamp=clamp)
        x0 = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
        _, x1 = sampler(key, x0)
        np.testing.assert_array_equal(np.asarray(x1[clamp]), np.asarray(x0[clamp]))

    def test_non_clamped_units_can_still_change(self, key: jax.Array) -> None:
        n = 5
        W = jnp.zeros((n, n))  # zero field -> fair coin flips
        clamp = jnp.array([0, 1, 2, 3], dtype=jnp.int32)  # all but unit 4
        sampler = _BM_sampler(W, None, sampler_steps=1, spin=True, clamp=clamp)
        x0 = jnp.ones(n)
        _, xs = _vmap_trials(lambda k: sampler(k, x0), key, 50)
        assert set(np.unique(np.asarray(xs[:, 4])).tolist()) == {-1.0, 1.0}

    def test_clamping_every_unit_is_identity(self, key: jax.Array) -> None:
        n = 4
        W = _random_symmetric_weights(key, n)
        b = jnp.zeros(n)
        clamp = jnp.arange(n, dtype=jnp.int32)
        sampler = _BM_sampler(W, b, sampler_steps=20, spin=True, clamp=clamp)
        x0 = jnp.ones(n)
        _, x1 = sampler(key, x0)
        np.testing.assert_array_equal(np.asarray(x0), np.asarray(x1))


# =========================================================================== #
# _rbm_update_visible: block-resample the visible layer                      #
# =========================================================================== #


class TestRbmUpdateVisible:
    def test_strong_positive_field_sets_plus_one(self, key: jax.Array) -> None:
        n_v, n_h = 3, 2
        W = jnp.zeros((n_v, n_h))
        b_v = jnp.array([0.0, 50.0, 0.0])
        x_v0 = -jnp.ones(n_v)
        x_h = jnp.ones(n_h)
        new_v = _rbm_update_visible(W, b_v, key, x_v0, x_h, spin=True, free_mask=None)
        assert new_v[1] == 1.0

    def test_strong_negative_field_sets_minus_one(self, key: jax.Array) -> None:
        n_v, n_h = 3, 2
        W = jnp.zeros((n_v, n_h))
        b_v = jnp.array([0.0, -50.0, 0.0])
        x_v0 = jnp.ones(n_v)
        x_h = jnp.ones(n_h)
        new_v = _rbm_update_visible(W, b_v, key, x_v0, x_h, spin=True, free_mask=None)
        assert new_v[1] == -1.0

    def test_field_comes_from_hidden_via_W(self, key: jax.Array) -> None:
        n_v = 2
        W = jnp.array([[0.0, 0.0], [50.0, 0.0]])  # v1 driven by h0 only
        x_v0 = -jnp.ones(n_v)
        new_v = _rbm_update_visible(W, None, key, x_v0, jnp.array([1.0, -1.0]), True, None)
        assert new_v[1] == 1.0
        new_v = _rbm_update_visible(W, None, key, x_v0, jnp.array([-1.0, 1.0]), True, None)
        assert new_v[1] == -1.0

    def test_spin_false_writes_binary(self, key: jax.Array) -> None:
        n_v, n_h = 3, 2
        W = jnp.zeros((n_v, n_h))
        b_v = jnp.array([0.0, 50.0, 0.0])
        new_v = _rbm_update_visible(W, b_v, key, jnp.zeros(n_v), jnp.ones(n_h), False, None)
        assert new_v[1] == 1.0
        assert set(np.unique(np.asarray(new_v)).tolist()).issubset({0.0, 1.0})

    def test_none_bias_is_allowed(self, key: jax.Array) -> None:
        n_v, n_h = 4, 3
        W = jax.random.normal(key, (n_v, n_h))
        new_v = _rbm_update_visible(W, None, key, jnp.ones(n_v), jnp.ones(n_h), True, None)
        assert new_v.shape == (n_v,)

    def test_free_mask_none_can_change_every_unit(self, key: jax.Array) -> None:
        n_v, n_h = 4, 3
        W = jnp.zeros((n_v, n_h))  # zero field -> fair coin flips
        x_v0 = jnp.ones(n_v)
        x_h = jnp.ones(n_h)
        touched = set()
        for i in range(100):
            subkey = jax.random.fold_in(key, i)
            new_v = _rbm_update_visible(W, None, subkey, x_v0, x_h, True, None)
            touched.update(np.flatnonzero(np.asarray(x_v0) != np.asarray(new_v)).tolist())
        assert touched == set(range(n_v))

    def test_free_mask_keeps_clamped_positions_at_old_value(self, key: jax.Array) -> None:
        n_v, n_h = 5, 3
        W = jax.random.normal(key, (n_v, n_h))
        x_v0 = jnp.where(jnp.arange(n_v) % 2 == 0, 1.0, -1.0)
        x_h = jnp.ones(n_h)
        free_mask = jnp.array([True, False, True, False, True])
        new_v = _rbm_update_visible(W, None, key, x_v0, x_h, True, free_mask)
        clamped = np.flatnonzero(~np.asarray(free_mask))
        np.testing.assert_array_equal(np.asarray(new_v)[clamped], np.asarray(x_v0)[clamped])

    def test_free_mask_all_free_matches_no_mask(self, key: jax.Array) -> None:
        n_v, n_h = 4, 3
        W = jax.random.normal(key, (n_v, n_h))
        x_v0 = jnp.ones(n_v)
        x_h = jnp.ones(n_h)
        new_v_no_mask = _rbm_update_visible(W, None, key, x_v0, x_h, True, None)
        new_v_all_free = _rbm_update_visible(
            W, None, key, x_v0, x_h, True, jnp.ones(n_v, dtype=bool)
        )
        np.testing.assert_array_equal(np.asarray(new_v_no_mask), np.asarray(new_v_all_free))


# =========================================================================== #
# _rbm_update_hidden: block-resample the hidden layer                        #
# =========================================================================== #


class TestRbmUpdateHidden:
    def test_strong_positive_field_sets_plus_one(self, key: jax.Array) -> None:
        n_v, n_h = 2, 3
        W = jnp.zeros((n_v, n_h))
        b_h = jnp.array([0.0, 50.0, 0.0])
        new_h = _rbm_update_hidden(W, b_h, key, -jnp.ones(n_v), spin=True)
        assert new_h[1] == 1.0

    def test_strong_negative_field_sets_minus_one(self, key: jax.Array) -> None:
        n_v, n_h = 2, 3
        W = jnp.zeros((n_v, n_h))
        b_h = jnp.array([0.0, -50.0, 0.0])
        new_h = _rbm_update_hidden(W, b_h, key, jnp.ones(n_v), spin=True)
        assert new_h[1] == -1.0

    def test_field_comes_from_visible_via_W(self, key: jax.Array) -> None:
        W = jnp.array([[0.0, 50.0], [0.0, 0.0]])  # h1 driven by v0 only
        new_h = _rbm_update_hidden(W, None, key, jnp.array([1.0, -1.0]), spin=True)
        assert new_h[1] == 1.0
        new_h = _rbm_update_hidden(W, None, key, jnp.array([-1.0, 1.0]), spin=True)
        assert new_h[1] == -1.0

    def test_spin_false_writes_binary(self, key: jax.Array) -> None:
        n_v, n_h = 2, 3
        W = jnp.zeros((n_v, n_h))
        b_h = jnp.array([0.0, 50.0, 0.0])
        new_h = _rbm_update_hidden(W, b_h, key, jnp.zeros(n_v), spin=False)
        assert new_h[1] == 1.0
        assert set(np.unique(np.asarray(new_h)).tolist()).issubset({0.0, 1.0})

    def test_none_bias_is_allowed(self, key: jax.Array) -> None:
        n_v, n_h = 4, 3
        W = jax.random.normal(key, (n_v, n_h))
        new_h = _rbm_update_hidden(W, None, key, jnp.ones(n_v), spin=True)
        assert new_h.shape == (n_h,)

    def test_can_change_every_unit(self, key: jax.Array) -> None:
        n_v, n_h = 3, 4
        W = jnp.zeros((n_v, n_h))  # zero field -> fair coin flips
        x_v = jnp.ones(n_v)
        touched = set()
        for i in range(100):
            subkey = jax.random.fold_in(key, i)
            new_h = _rbm_update_hidden(W, None, subkey, x_v, True)
            touched.update(np.flatnonzero(np.ones(n_h) != np.asarray(new_h)).tolist())
        assert touched == set(range(n_h))


# =========================================================================== #
# _RBM_sampler: block-conditional Gibbs sampler builder                      #
# =========================================================================== #


class TestRBMSampler:
    def test_shapes_and_dtypes_preserved(self, key: jax.Array) -> None:
        n_v, n_h = 5, 3
        W = jax.random.normal(key, (n_v, n_h))
        b_v, b_h = jnp.zeros(n_v), jnp.zeros(n_h)
        sampler = _RBM_sampler(W, b_v, b_h, sampler_steps=3, spin=True)
        x_v0, x_h0 = jnp.ones(n_v), jnp.ones(n_h)
        _, (x_v1, x_h1) = sampler(key, (x_v0, x_h0))
        assert x_v1.shape == (n_v,)
        assert x_h1.shape == (n_h,)
        assert x_v1.dtype == x_v0.dtype
        assert x_h1.dtype == x_h0.dtype

    def test_zero_steps_is_identity(self, key: jax.Array) -> None:
        n_v, n_h = 4, 2
        W = jax.random.normal(key, (n_v, n_h))
        sampler = _RBM_sampler(W, None, None, sampler_steps=0, spin=True)
        x_v0, x_h0 = jnp.ones(n_v), jnp.ones(n_h)
        _, (x_v1, x_h1) = sampler(key, (x_v0, x_h0))
        np.testing.assert_array_equal(np.asarray(x_v0), np.asarray(x_v1))
        np.testing.assert_array_equal(np.asarray(x_h0), np.asarray(x_h1))

    def test_same_key_gives_same_result(self, key: jax.Array) -> None:
        n_v, n_h = 5, 3
        W = jax.random.normal(key, (n_v, n_h))
        sampler = _RBM_sampler(W, None, None, sampler_steps=5, spin=True)
        x0 = (jnp.ones(n_v), jnp.ones(n_h))
        _, (v1, h1) = sampler(key, x0)
        _, (v2, h2) = sampler(key, x0)
        np.testing.assert_array_equal(np.asarray(v1), np.asarray(v2))
        np.testing.assert_array_equal(np.asarray(h1), np.asarray(h2))

    def test_different_keys_can_differ(self) -> None:
        key1, key2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
        n_v, n_h = 5, 3
        W = jax.random.normal(key1, (n_v, n_h))
        sampler = _RBM_sampler(W, None, None, sampler_steps=5, spin=True)
        x0 = (jnp.ones(n_v), jnp.ones(n_h))
        _, (v1, h1) = sampler(key1, x0)
        _, (v2, h2) = sampler(key2, x0)
        assert not (np.array_equal(np.asarray(v1), np.asarray(v2))
                    and np.array_equal(np.asarray(h1), np.asarray(h2)))

    def test_spin_true_outputs_in_pm_one(self, key: jax.Array) -> None:
        n_v, n_h = 5, 3
        W = jax.random.normal(key, (n_v, n_h))
        sampler = _RBM_sampler(W, None, None, sampler_steps=30, spin=True)
        _, (v1, h1) = sampler(key, (jnp.ones(n_v), jnp.ones(n_h)))
        assert set(np.unique(np.asarray(v1)).tolist()).issubset({-1.0, 1.0})
        assert set(np.unique(np.asarray(h1)).tolist()).issubset({-1.0, 1.0})

    def test_spin_false_outputs_binary(self, key: jax.Array) -> None:
        n_v, n_h = 5, 3
        W = jax.random.normal(key, (n_v, n_h))
        sampler = _RBM_sampler(W, None, None, sampler_steps=30, spin=False)
        _, (v1, h1) = sampler(key, (jnp.zeros(n_v), jnp.zeros(n_h)))
        assert set(np.unique(np.asarray(v1)).tolist()).issubset({0.0, 1.0})
        assert set(np.unique(np.asarray(h1)).tolist()).issubset({0.0, 1.0})

    def test_clamp_none_visible_can_change_every_unit(self, key: jax.Array) -> None:
        n_v, n_h = 4, 3
        W = jnp.zeros((n_v, n_h))  # zero field -> fair coin flips
        sampler = _RBM_sampler(W, None, None, sampler_steps=1, spin=True)
        x0 = (jnp.ones(n_v), jnp.ones(n_h))
        _, (vs, _hs) = _vmap_trials(lambda k: sampler(k, x0), key, 200)
        touched = np.any(np.asarray(vs) != np.ones(n_v), axis=0)
        assert touched.all()

    def test_clamped_visible_units_never_change(self, key: jax.Array) -> None:
        n_v, n_h = 5, 3
        W = jax.random.normal(key, (n_v, n_h))
        clamp = jnp.array([0, 2, 4], dtype=jnp.int32)
        sampler = _RBM_sampler(W, None, None, sampler_steps=100, spin=True, clamp=clamp)
        x_v0 = jnp.where(jnp.arange(n_v) % 2 == 0, 1.0, -1.0)
        x_h0 = jnp.ones(n_h)
        _, (v1, _h1) = sampler(key, (x_v0, x_h0))
        np.testing.assert_array_equal(np.asarray(v1[clamp]), np.asarray(x_v0[clamp]))

    def test_non_clamped_visible_units_can_still_change(self, key: jax.Array) -> None:
        n_v, n_h = 5, 3
        W = jnp.zeros((n_v, n_h))  # zero field -> fair coin flips
        clamp = jnp.array([0, 1, 2, 3], dtype=jnp.int32)  # all but unit 4
        sampler = _RBM_sampler(W, None, None, sampler_steps=1, spin=True, clamp=clamp)
        x0 = (jnp.ones(n_v), jnp.ones(n_h))
        _, (vs, _hs) = _vmap_trials(lambda k: sampler(k, x0), key, 50)
        assert set(np.unique(np.asarray(vs[:, 4])).tolist()) == {-1.0, 1.0}

    def test_hidden_update_unaffected_by_visible_clamp(self, key: jax.Array) -> None:
        n_v, n_h = 5, 3
        W = jax.random.normal(key, (n_v, n_h))
        clamp = jnp.arange(n_v, dtype=jnp.int32)  # clamp every visible unit
        sampler = _RBM_sampler(W, None, None, sampler_steps=1, spin=True, clamp=clamp)
        x_v0 = jnp.ones(n_v)
        x_h0 = jnp.ones(n_h)
        _, (v1, h1) = sampler(key, (x_v0, x_h0))
        np.testing.assert_array_equal(np.asarray(v1), np.asarray(x_v0))
        # the hidden update should still see the same effective field it would
        # without clamping (since v0 is unchanged either way), and still varies
        # across draws -- it isn't just frozen along with v.
        _, (_vs, hs) = _vmap_trials(lambda k: sampler(k, (x_v0, x_h0)), key, 20)
        assert len({tuple(row) for row in np.asarray(hs).tolist()}) > 1

    def test_clamping_every_visible_unit_freezes_only_visible(self, key: jax.Array) -> None:
        n_v, n_h = 4, 2
        W = jax.random.normal(key, (n_v, n_h))
        clamp = jnp.arange(n_v, dtype=jnp.int32)
        sampler = _RBM_sampler(W, None, None, sampler_steps=20, spin=True, clamp=clamp)
        x_v0, x_h0 = jnp.ones(n_v), jnp.ones(n_h)
        _, (v1, h1) = sampler(key, (x_v0, x_h0))
        np.testing.assert_array_equal(np.asarray(v1), np.asarray(x_v0))
        assert h1.shape == (n_h,)
