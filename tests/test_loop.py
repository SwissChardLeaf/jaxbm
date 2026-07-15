"""Tests for `jaxbm._loop` (the generic JAX loops shared by
`BM_chain` / `RBM_chain`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxbm._loop import _bm_outer_self, _for_loop, _for_loop_stat, _rbm_outer_self, _scan


def _increment_sampler(key, x):
    """Deterministic sampler: leaves ``key`` untouched, increments ``x`` by 1."""
    return key, x + 1


def _increment_pytree_sampler(key, x):
    """Same idea as ``_increment_sampler``, but for a ``(v, h)`` pytree state,
    with a different (still deterministic) increment per leaf."""
    v, h = x
    return key, (v + 1, h + 2)


def _random_sign_sampler(key, x):
    """Sampler that redraws every unit of ``x`` to a fresh, independent
    ``{-1, +1}`` sign each call, ignoring its previous value. Advances
    ``key`` via ``jax.random.split``, so each call is independent -- useful
    for closed-form statistical checks (mean 0, ``outer(x, x)`` diagonal 1).
    """
    key, subkey = jax.random.split(key)
    signs = jnp.where(jax.random.bernoulli(subkey, shape=x.shape), 1.0, -1.0)
    return key, signs


# =========================================================================== #
# _for_loop: advance-and-discard (no accumulation, no stacking)              #
# =========================================================================== #


class TestForLoop:
    def test_zero_calls_returns_initial_state(self, key: jax.Array) -> None:
        x0 = jnp.zeros(3)
        result = _for_loop(_increment_sampler, key, x0, 0)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(x0))

    def test_calls_sampler_num_calls_times(self, key: jax.Array) -> None:
        x0 = jnp.zeros(4)
        result = _for_loop(_increment_sampler, key, x0, 5)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(x0) + 5)

    def test_shape_and_dtype_preserved(self, key: jax.Array) -> None:
        x0 = jnp.ones(5)
        result = _for_loop(_increment_sampler, key, x0, 3)
        assert result.shape == x0.shape
        assert result.dtype == x0.dtype

    def test_matches_manual_unrolled_calls(self, key: jax.Array) -> None:
        n = 4
        x0 = jnp.ones(n)
        num_calls = 7
        x, k = x0, key
        for _ in range(num_calls):
            k, x = _random_sign_sampler(k, x)
        result = _for_loop(_random_sign_sampler, key, x0, num_calls)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(x))

    def test_works_with_pytree_state(self, key: jax.Array) -> None:
        v0, h0 = jnp.zeros(3), jnp.zeros(2)
        v1, h1 = _for_loop(_increment_pytree_sampler, key, (v0, h0), 4)
        np.testing.assert_array_equal(np.asarray(v1), np.asarray(v0) + 4)
        np.testing.assert_array_equal(np.asarray(h1), np.asarray(h0) + 8)


# =========================================================================== #
# _bm_outer_self / _rbm_outer_self: stat_fn for mode='CORR'                   #
# =========================================================================== #


class TestOuterSelfHelpers:
    def test_bm_outer_self_matches_outer_product(self) -> None:
        x = jnp.array([1.0, -1.0, 2.0])
        result = _bm_outer_self(x)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(jnp.outer(x, x)))

    def test_bm_outer_self_shape_is_n_by_n(self) -> None:
        x = jnp.ones(5)
        assert _bm_outer_self(x).shape == (5, 5)

    def test_rbm_outer_self_matches_outer_of_v_and_h(self) -> None:
        v = jnp.array([1.0, -1.0])
        h = jnp.array([2.0, 3.0, -1.0])
        result = _rbm_outer_self((v, h))
        np.testing.assert_array_equal(np.asarray(result), np.asarray(jnp.outer(v, h)))

    def test_rbm_outer_self_shape_is_n_v_by_n_h(self) -> None:
        v, h = jnp.ones(4), jnp.ones(3)
        assert _rbm_outer_self((v, h)).shape == (4, 3)


# =========================================================================== #
# _for_loop_stat: running mean of stat_fn(x), used by mode='MEAN' / 'CORR'    #
# =========================================================================== #


class TestForLoopStat:
    def test_mean_matches_manual_average_with_stat_fn_none(self, key: jax.Array) -> None:
        x0 = jnp.zeros(3)
        num_samples = 5
        x_final, x_mean = _for_loop_stat(_increment_sampler, key, x0, num_samples)
        # visited states are x0 + 1, ..., x0 + num_samples
        expected_mean = x0 + (num_samples + 1) / 2
        np.testing.assert_array_equal(np.asarray(x_final), np.asarray(x0) + num_samples)
        np.testing.assert_allclose(np.asarray(x_mean), np.asarray(expected_mean))

    def test_stat_fn_none_is_equivalent_to_explicit_identity(self, key: jax.Array) -> None:
        def identity(x):
            return x

        x0 = jnp.ones(4)
        num_samples = 6
        x1, mean1 = _for_loop_stat(_random_sign_sampler, key, x0, num_samples)
        x2, mean2 = _for_loop_stat(_random_sign_sampler, key, x0, num_samples, stat_fn=identity)
        np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))
        np.testing.assert_array_equal(np.asarray(mean1), np.asarray(mean2))

    def test_custom_stat_fn_is_applied(self, key: jax.Array) -> None:
        x0 = jnp.zeros(3)

        def square(x):
            return x**2

        _, stat_mean = _for_loop_stat(_increment_sampler, key, x0, 4, stat_fn=square)
        # visited states are 1, 2, 3, 4 -> squares are 1, 4, 9, 16 -> mean 7.5
        np.testing.assert_allclose(np.asarray(stat_mean), np.full(3, 7.5))

    def test_initial_state_excluded_from_mean(self, key: jax.Array) -> None:
        x0 = jnp.array([100.0])
        _, x_mean = _for_loop_stat(_increment_sampler, key, x0, 1)
        # the only visited state is x0 + 1 = 101 -- if x0 leaked in, the mean
        # would be (100 + 101) / 2 = 100.5 instead
        np.testing.assert_array_equal(np.asarray(x_mean), np.asarray([101.0]))

    def test_pytree_state_accumulates_each_leaf_independently(self, key: jax.Array) -> None:
        v0, h0 = jnp.zeros(2), jnp.zeros(3)
        (v_final, h_final), (v_mean, h_mean) = _for_loop_stat(
            _increment_pytree_sampler, key, (v0, h0), 4
        )
        np.testing.assert_array_equal(np.asarray(v_final), np.full(2, 4.0))
        np.testing.assert_array_equal(np.asarray(h_final), np.full(3, 8.0))
        np.testing.assert_allclose(np.asarray(v_mean), np.full(2, 2.5))  # mean(1, 2, 3, 4)
        np.testing.assert_allclose(np.asarray(h_mean), np.full(3, 5.0))  # mean(2, 4, 6, 8)

    def test_same_key_gives_same_result(self, key: jax.Array) -> None:
        x0 = jnp.ones(4)
        x1, mean1 = _for_loop_stat(_random_sign_sampler, key, x0, 10)
        x2, mean2 = _for_loop_stat(_random_sign_sampler, key, x0, 10)
        np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))
        np.testing.assert_array_equal(np.asarray(mean1), np.asarray(mean2))

    def test_outer_self_stat_fn_recovers_expected_second_moment(self, key: jax.Array) -> None:
        # a fresh, independent +-1 sign each draw: outer(x, x)'s diagonal is
        # exactly 1 (since sign**2 == 1 always), and its off-diagonal entries
        # have mean 0 (independent units) -- check both, with enough samples
        # for the off-diagonal check to be reliable.
        n = 4
        x0 = jnp.ones(n)
        _, outer_mean = _for_loop_stat(_random_sign_sampler, key, x0, 4000, stat_fn=_bm_outer_self)
        outer_mean = np.asarray(outer_mean)
        np.testing.assert_array_equal(np.diagonal(outer_mean), np.ones(n))
        off_diag = outer_mean[~np.eye(n, dtype=bool)]
        assert np.all(np.abs(off_diag) < 0.1)


# =========================================================================== #
# _scan: stack every visited state into a trajectory                         #
# =========================================================================== #


class TestScan:
    def test_stacks_every_intermediate_state(self, key: jax.Array) -> None:
        x0 = jnp.zeros(3)
        num_samples = 5
        (_, x_final), xs = _scan(_increment_sampler, key, x0, num_samples)
        expected = jnp.stack([x0 + i for i in range(1, num_samples + 1)])
        np.testing.assert_array_equal(np.asarray(xs), np.asarray(expected))
        np.testing.assert_array_equal(np.asarray(x_final), np.asarray(x0) + num_samples)

    def test_final_carry_matches_last_row_of_trajectory(self, key: jax.Array) -> None:
        x0 = jnp.ones(4)
        (_, x_final), xs = _scan(_increment_sampler, key, x0, 6)
        np.testing.assert_array_equal(np.asarray(x_final), np.asarray(xs[-1]))

    def test_matches_manual_unrolled_calls(self, key: jax.Array) -> None:
        n = 3
        x0 = jnp.ones(n)
        num_samples = 5
        x, k = x0, key
        expected_states = []
        for _ in range(num_samples):
            k, x = _random_sign_sampler(k, x)
            expected_states.append(x)
        (k_final, x_final), xs = _scan(_random_sign_sampler, key, x0, num_samples)
        np.testing.assert_array_equal(np.asarray(xs), np.asarray(jnp.stack(expected_states)))
        np.testing.assert_array_equal(np.asarray(x_final), np.asarray(x))
        np.testing.assert_array_equal(np.asarray(k_final), np.asarray(k))

    def test_input_state_is_not_included_in_trajectory(self, key: jax.Array) -> None:
        x0 = jnp.array([100.0])
        (_, _), xs = _scan(_increment_sampler, key, x0, 1)
        # the only sample is x0 + 1 = 101 -- the trajectory must not contain
        # the input state (100) anywhere
        np.testing.assert_array_equal(np.asarray(xs), np.asarray([[101.0]]))

    def test_pytree_state_stacks_each_leaf(self, key: jax.Array) -> None:
        v0, h0 = jnp.zeros(2), jnp.zeros(3)
        (_, (v_final, h_final)), (vs, hs) = _scan(_increment_pytree_sampler, key, (v0, h0), 4)
        assert vs.shape == (4, 2)
        assert hs.shape == (4, 3)
        np.testing.assert_array_equal(np.asarray(vs[-1]), np.asarray(v_final))
        np.testing.assert_array_equal(np.asarray(hs[-1]), np.asarray(h_final))
        np.testing.assert_array_equal(
            np.asarray(vs), np.stack([np.full(2, float(i)) for i in range(1, 5)])
        )
        np.testing.assert_array_equal(
            np.asarray(hs), np.stack([np.full(3, float(2 * i)) for i in range(1, 5)])
        )

    def test_zero_samples_returns_empty_trajectory_and_unchanged_carry(
        self, key: jax.Array
    ) -> None:
        x0 = jnp.ones(3)
        (k_final, x_final), xs = _scan(_increment_sampler, key, x0, 0)
        assert xs.shape == (0, 3)
        np.testing.assert_array_equal(np.asarray(x_final), np.asarray(x0))
        np.testing.assert_array_equal(np.asarray(k_final), np.asarray(key))
