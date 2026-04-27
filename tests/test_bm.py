"""Tests for :mod:`jax_bm.bm`.

Covers the abstract base class, both concrete model classes, all
classmethod constructors, the pytree registration, and every public
method (``energy``, ``update_state``, ``update_visible``, ``update_hidden``,
``update_params``).

Tests are organized by class and by method/concern. The shared ``key``
fixture is defined in ``tests/conftest.py``.
"""

from __future__ import annotations

import dataclasses
from abc import ABC
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_bm.bm import (
    AbstractBoltzmannMachine,
    BoltzmannMachine,
    RestrictedBoltzmannMachine,
)


# =========================================================================== #
# AbstractBoltzmannMachine                                                    #
# =========================================================================== #


class TestAbstractBoltzmannMachine:
    def test_is_actual_class(self) -> None:
        # Regression: previously written as ``def AbstractBoltzmannMachine(ABC):``
        # which would silently be a no-op function.
        assert isinstance(AbstractBoltzmannMachine, type)

    def test_inherits_from_abc(self) -> None:
        assert issubclass(AbstractBoltzmannMachine, ABC)

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            AbstractBoltzmannMachine()  # type: ignore[abstract]

    def test_required_abstract_methods(self) -> None:
        expected = {"energy", "update_state", "update_params"}
        assert expected.issubset(AbstractBoltzmannMachine.__abstractmethods__)

    def test_boltzmann_machine_is_subclass(self) -> None:
        assert issubclass(BoltzmannMachine, AbstractBoltzmannMachine)

    def test_rbm_is_subclass(self) -> None:
        assert issubclass(RestrictedBoltzmannMachine, AbstractBoltzmannMachine)


# =========================================================================== #
# BoltzmannMachine: construction                                              #
# =========================================================================== #


class TestBoltzmannMachineInit:
    def test_init_random_shapes(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=8)
        assert bm.W.shape == (8, 8)
        assert bm.b is not None
        assert bm.b.shape == (8,)

    def test_W_is_symmetric(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=10)
        np.testing.assert_allclose(np.asarray(bm.W), np.asarray(bm.W.T), atol=1e-7)

    def test_W_has_zero_diagonal(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=12)
        np.testing.assert_allclose(np.diag(np.asarray(bm.W)), np.zeros(12), atol=1e-7)

    def test_init_random_default_flags(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        assert bm.bias is True
        assert bm.spin_style is True

    def test_init_random_no_bias(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5, bias=False)
        assert bm.b is None
        assert bm.bias is False

    def test_init_random_spin_style_false(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, spin_style=False)
        assert bm.spin_style is False

    def test_init_random_absnorm_with_bias(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=20, absnorm=True)
        total = float(jnp.sum(jnp.abs(bm.W)) + jnp.sum(jnp.abs(bm.b)))
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_init_random_absnorm_no_bias(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=20, absnorm=True, bias=False)
        assert float(jnp.sum(jnp.abs(bm.W))) == pytest.approx(1.0, abs=1e-5)

    def test_init_random_minmax_bounds(self, key: jax.Array) -> None:
        # After symmetrization (A + A.T) with A from U[0.5*min, 0.5*max], the
        # off-diagonal entries are in [min, max]; biases are in [min, max].
        minval, maxval = -2.0, 2.0
        bm = BoltzmannMachine.init_random(key, n=20, minval=minval, maxval=maxval)
        W = np.asarray(bm.W)
        offdiag = W[~np.eye(20, dtype=bool)]
        assert offdiag.min() >= minval - 1e-5
        assert offdiag.max() <= maxval + 1e-5
        b = np.asarray(bm.b)
        assert b.min() >= minval - 1e-5
        assert b.max() <= maxval + 1e-5

    def test_init_random_deterministic_in_key(self, key: jax.Array) -> None:
        bm1 = BoltzmannMachine.init_random(key, n=6)
        bm2 = BoltzmannMachine.init_random(key, n=6)
        np.testing.assert_array_equal(np.asarray(bm1.W), np.asarray(bm2.W))
        np.testing.assert_array_equal(np.asarray(bm1.b), np.asarray(bm2.b))

    def test_init_random_different_keys_differ(self) -> None:
        bm1 = BoltzmannMachine.init_random(jax.random.PRNGKey(0), n=6)
        bm2 = BoltzmannMachine.init_random(jax.random.PRNGKey(1), n=6)
        assert not np.allclose(np.asarray(bm1.W), np.asarray(bm2.W))

    def test_init_random_W_and_b_independent_of_each_other(self, key: jax.Array) -> None:
        # Sanity: W and b should be drawn from independent random streams,
        # so a column of W and the bias vector should not be linearly related.
        bm = BoltzmannMachine.init_random(key, n=200)
        col = np.asarray(bm.W[:, 0])
        b = np.asarray(bm.b)
        corr = np.corrcoef(col, b)[0, 1]
        assert abs(corr) < 0.3  # huge slack; a key-reuse bug would give |corr|≈1.

    def test_init_from_matrix_with_bias(self) -> None:
        W = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        b = jnp.array([0.1, 0.2, 0.3])
        bm = BoltzmannMachine.init_from_matrix(W, b)
        np.testing.assert_array_equal(np.asarray(bm.W), np.asarray(W))
        np.testing.assert_array_equal(np.asarray(bm.b), np.asarray(b))
        assert bm.bias is True

    def test_init_from_matrix_no_bias(self) -> None:
        W = jnp.zeros((3, 3))
        bm = BoltzmannMachine.init_from_matrix(W)
        assert bm.b is None
        assert bm.bias is False

    def test_dataclass_is_frozen(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        with pytest.raises(dataclasses.FrozenInstanceError):
            bm.W = jnp.zeros((4, 4))  # type: ignore[misc]


# =========================================================================== #
# BoltzmannMachine: pytree registration                                       #
# =========================================================================== #


class TestBoltzmannMachinePytree:
    def test_leaves_with_bias(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        leaves = jax.tree_util.tree_leaves(bm)
        assert len(leaves) == 2  # W and b
        shapes = sorted(leaf.shape for leaf in leaves)
        assert shapes == [(4,), (4, 4)]

    def test_leaves_no_bias(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, bias=False)
        leaves = jax.tree_util.tree_leaves(bm)
        # b is None and contributes no leaves.
        assert len(leaves) == 1
        assert leaves[0].shape == (4, 4)

    def test_tree_map_only_touches_data_fields(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, spin_style=False)
        bm2 = jax.tree_util.tree_map(lambda x: x * 2, bm)
        np.testing.assert_allclose(np.asarray(bm2.W), 2 * np.asarray(bm.W))
        np.testing.assert_allclose(np.asarray(bm2.b), 2 * np.asarray(bm.b))
        # Meta fields preserved unchanged.
        assert bm2.spin_style is False
        assert bm2.bias is True

    def test_jit_passthrough(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        x = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0])
        eager = float(bm.energy(x))
        jitted = float(jax.jit(lambda m, s: m.energy(s))(bm, x))
        assert jitted == pytest.approx(eager, rel=1e-5)

    def test_vmap_over_models(self, key: jax.Array) -> None:
        keys = jax.random.split(key, 4)
        bms = jax.vmap(lambda k: BoltzmannMachine.init_random(k, n=4))(keys)
        assert bms.W.shape == (4, 4, 4)
        assert bms.b.shape == (4, 4)
        # Meta fields are static and shared across the batch.
        assert bms.bias is True
        assert bms.spin_style is True

    def test_round_trip_flatten_unflatten(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5, spin_style=False)
        leaves, treedef = jax.tree_util.tree_flatten(bm)
        bm2 = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_array_equal(np.asarray(bm2.W), np.asarray(bm.W))
        np.testing.assert_array_equal(np.asarray(bm2.b), np.asarray(bm.b))
        assert bm2.spin_style is False
        assert bm2.bias is True

    def test_different_bias_flags_have_different_treedefs(self, key: jax.Array) -> None:
        bm_with = BoltzmannMachine.init_random(key, n=4, bias=True)
        bm_without = BoltzmannMachine.init_random(key, n=4, bias=False)
        td_with = jax.tree_util.tree_structure(bm_with)
        td_without = jax.tree_util.tree_structure(bm_without)
        assert td_with != td_without


# =========================================================================== #
# BoltzmannMachine: energy                                                    #
# =========================================================================== #


class TestBoltzmannMachineEnergy:
    def test_manual_with_bias(self) -> None:
        W = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        b = jnp.array([0.5, -0.5, 1.0])
        bm = BoltzmannMachine.init_from_matrix(W, b)
        x = jnp.array([1.0, -1.0, 1.0])
        expected = -0.5 * float(x @ W @ x) - float(b @ x)
        assert float(bm.energy(x)) == pytest.approx(expected, abs=1e-5)

    def test_manual_no_bias(self) -> None:
        W = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        bm = BoltzmannMachine.init_from_matrix(W)
        # x = (1, 1): x^T W x = 2 -> E = -1
        assert float(bm.energy(jnp.array([1.0, 1.0]))) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_state_binary(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5, spin_style=False)
        # E(0) = -0.5 * 0 - b·0 = 0
        assert float(bm.energy(jnp.zeros(5))) == 0.0

    def test_random_state_matches_quadratic_form(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=8)
        x = jax.random.normal(jax.random.PRNGKey(7), (8,))
        expected = -0.5 * float(x @ bm.W @ x) - float(bm.b @ x)
        assert float(bm.energy(x)) == pytest.approx(expected, abs=1e-5)

    def test_returns_scalar(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        assert bm.energy(jnp.zeros(4)).shape == ()

    def test_energy_jit(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        x = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0])
        np.testing.assert_allclose(
            float(jax.jit(bm.energy)(x)), float(bm.energy(x)), rtol=1e-5
        )

    def test_no_bias_energy_jit(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, bias=False)
        x = jnp.array([1.0, -1.0, 1.0, -1.0])
        np.testing.assert_allclose(
            float(jax.jit(bm.energy)(x)), float(bm.energy(x)), rtol=1e-5
        )


# =========================================================================== #
# BoltzmannMachine: update_state                                              #
# =========================================================================== #


class TestBoltzmannMachineUpdateState:
    def test_returns_key_and_state(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        x = jnp.ones(4)
        free = jnp.arange(4)
        new_key, new_x = bm.update_state(key, x, free)
        assert new_key.shape == key.shape
        assert new_x.shape == x.shape

    def test_at_most_one_unit_changes_per_step(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        x0 = -jnp.ones(5)
        free = jnp.arange(5)
        for k in jax.random.split(key, 30):
            _, x1 = bm.update_state(k, x0, free)
            diff = int(np.sum(np.asarray(x1) != np.asarray(x0)))
            assert diff <= 1

    def test_spin_outputs_in_pm_one(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=6, spin_style=True)
        x = jnp.ones(6)
        free = jnp.arange(6)
        for k in jax.random.split(key, 40):
            _, x = bm.update_state(k, x, free)
        vals = set(np.unique(np.asarray(x)).tolist())
        assert vals.issubset({-1.0, 1.0})

    def test_binary_outputs_in_zero_one(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=6, spin_style=False)
        x = jnp.zeros(6)
        free = jnp.arange(6)
        for k in jax.random.split(key, 40):
            _, x = bm.update_state(k, x, free)
        vals = set(np.unique(np.asarray(x)).tolist())
        assert vals.issubset({0.0, 1.0})

    def test_free_units_subset_protects_others(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=8)
        x0 = -jnp.ones(8)
        free = jnp.array([0, 1, 2])
        x = x0
        for k in jax.random.split(key, 50):
            _, x = bm.update_state(k, x, free)
        # Positions 3..7 must remain at their initial values.
        np.testing.assert_array_equal(np.asarray(x[3:]), np.asarray(x0[3:]))

    def test_jit_update_state(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        x = jnp.ones(4)
        free = jnp.arange(4)
        new_key, new_x = jax.jit(bm.update_state)(key, x, free)
        assert new_key.shape == key.shape
        assert new_x.shape == x.shape

    def test_deterministic_with_same_key(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=5)
        x = jnp.ones(5)
        free = jnp.arange(5)
        same_key = jax.random.PRNGKey(42)
        _, x1 = bm.update_state(same_key, x, free)
        _, x2 = bm.update_state(same_key, x, free)
        np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))

    def test_rng_advances(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        new_key, _ = bm.update_state(key, jnp.ones(4), jnp.arange(4))
        assert not np.array_equal(np.asarray(new_key), np.asarray(key))

    def test_update_state_no_bias(self, key: jax.Array) -> None:
        # No-bias path must not error on a None ``b``.
        bm = BoltzmannMachine.init_random(key, n=5, bias=False)
        new_key, new_x = bm.update_state(key, jnp.ones(5), jnp.arange(5))
        assert new_x.shape == (5,)


# =========================================================================== #
# BoltzmannMachine: update_params                                             #
# =========================================================================== #


class TestBoltzmannMachineUpdateParams:
    def test_returns_new_instance(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        new_W = jnp.zeros((4, 4))
        new_b = jnp.ones(4)
        bm2 = bm.update_params(new_W, new_b)
        assert isinstance(bm2, BoltzmannMachine)
        np.testing.assert_array_equal(np.asarray(bm2.W), np.asarray(new_W))
        np.testing.assert_array_equal(np.asarray(bm2.b), np.asarray(new_b))

    def test_original_unchanged(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4)
        W_before = np.asarray(bm.W).copy()
        b_before = np.asarray(bm.b).copy()
        _ = bm.update_params(jnp.zeros((4, 4)), jnp.ones(4))
        np.testing.assert_array_equal(np.asarray(bm.W), W_before)
        np.testing.assert_array_equal(np.asarray(bm.b), b_before)

    def test_meta_preserved(self, key: jax.Array) -> None:
        bm = BoltzmannMachine.init_random(key, n=4, spin_style=False)
        bm2 = bm.update_params(jnp.zeros((4, 4)), jnp.ones(4))
        assert bm2.spin_style is False
        assert bm2.bias is True


# =========================================================================== #
# RestrictedBoltzmannMachine: construction                                    #
# =========================================================================== #


class TestRBMInit:
    def test_init_random_shapes(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=5, n_hidden=3)
        assert rbm.W.shape == (5, 3)
        assert rbm.b_v.shape == (5,)
        assert rbm.b_h.shape == (3,)
        assert rbm.n_visible == 5

    def test_init_random_default_flags(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=3, n_hidden=2)
        assert rbm.bias is True
        assert rbm.spin_style is True

    def test_init_random_no_bias(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(
            key, n_visible=4, n_hidden=2, bias=False
        )
        assert rbm.b_v is None
        assert rbm.b_h is None
        assert rbm.bias is False

    def test_init_random_absnorm_with_bias(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(
            key, n_visible=4, n_hidden=3, absnorm=True
        )
        total = float(
            jnp.sum(jnp.abs(rbm.W))
            + jnp.sum(jnp.abs(rbm.b_v))
            + jnp.sum(jnp.abs(rbm.b_h))
        )
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_init_random_absnorm_no_bias(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(
            key, n_visible=4, n_hidden=3, absnorm=True, bias=False
        )
        assert float(jnp.sum(jnp.abs(rbm.W))) == pytest.approx(1.0, abs=1e-5)

    def test_init_random_minmax_bounds(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(
            key, n_visible=5, n_hidden=4, minval=-3.0, maxval=3.0
        )
        W = np.asarray(rbm.W)
        assert W.min() >= -3.0 - 1e-5
        assert W.max() <= 3.0 + 1e-5

    def test_init_random_deterministic_in_key(self, key: jax.Array) -> None:
        rbm1 = RestrictedBoltzmannMachine.init_random(key, n_visible=4, n_hidden=2)
        rbm2 = RestrictedBoltzmannMachine.init_random(key, n_visible=4, n_hidden=2)
        np.testing.assert_array_equal(np.asarray(rbm1.W), np.asarray(rbm2.W))
        np.testing.assert_array_equal(np.asarray(rbm1.b_v), np.asarray(rbm2.b_v))
        np.testing.assert_array_equal(np.asarray(rbm1.b_h), np.asarray(rbm2.b_h))

    def test_init_random_W_bv_bh_independent(self, key: jax.Array) -> None:
        # If the same key were reused for W, b_v, b_h (a known footgun),
        # these would be perfectly correlated. Use a large size for power.
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=400, n_hidden=400)
        col = np.asarray(rbm.W[:, 0])
        bv = np.asarray(rbm.b_v)
        assert abs(np.corrcoef(col, bv)[0, 1]) < 0.2

    def test_init_from_matrix_with_biases(self) -> None:
        W = jnp.ones((3, 2))
        v = jnp.array([0.1, 0.2, 0.3])
        h = jnp.array([1.0, 2.0])
        rbm = RestrictedBoltzmannMachine.init_from_matrix(W, v, h)
        np.testing.assert_array_equal(np.asarray(rbm.W), np.asarray(W))
        np.testing.assert_array_equal(np.asarray(rbm.b_v), np.asarray(v))
        np.testing.assert_array_equal(np.asarray(rbm.b_h), np.asarray(h))
        assert rbm.n_visible == 3
        assert rbm.bias is True

    def test_init_from_matrix_no_biases(self) -> None:
        W = jnp.ones((3, 2))
        rbm = RestrictedBoltzmannMachine.init_from_matrix(W)
        assert rbm.b_v is None
        assert rbm.b_h is None
        assert rbm.bias is False
        assert rbm.n_visible == 3

    def test_dataclass_is_frozen(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=3, n_hidden=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            rbm.W = jnp.zeros((3, 2))  # type: ignore[misc]


# =========================================================================== #
# RestrictedBoltzmannMachine: pytree                                          #
# =========================================================================== #


class TestRBMPytree:
    def test_leaves_with_bias(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=4, n_hidden=3)
        leaves = jax.tree_util.tree_leaves(rbm)
        assert len(leaves) == 3  # W, b_v, b_h

    def test_leaves_no_bias(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(
            key, n_visible=4, n_hidden=3, bias=False
        )
        leaves = jax.tree_util.tree_leaves(rbm)
        assert len(leaves) == 1
        assert leaves[0].shape == (4, 3)

    def test_tree_map_only_touches_data(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(
            key, n_visible=3, n_hidden=2, spin_style=False
        )
        rbm2 = jax.tree_util.tree_map(lambda x: x * 2, rbm)
        np.testing.assert_allclose(np.asarray(rbm2.W), 2 * np.asarray(rbm.W))
        np.testing.assert_allclose(np.asarray(rbm2.b_v), 2 * np.asarray(rbm.b_v))
        np.testing.assert_allclose(np.asarray(rbm2.b_h), 2 * np.asarray(rbm.b_h))
        assert rbm2.n_visible == 3
        assert rbm2.bias is True
        assert rbm2.spin_style is False

    def test_jit_passthrough(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=3, n_hidden=2)
        x = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])
        eager = float(rbm.energy(x))
        jitted = float(jax.jit(lambda m, s: m.energy(s))(rbm, x))
        assert jitted == pytest.approx(eager, rel=1e-5)

    def test_vmap_over_models(self, key: jax.Array) -> None:
        keys = jax.random.split(key, 3)
        rbms = jax.vmap(
            lambda k: RestrictedBoltzmannMachine.init_random(k, n_visible=4, n_hidden=2)
        )(keys)
        assert rbms.W.shape == (3, 4, 2)
        assert rbms.b_v.shape == (3, 4)
        assert rbms.b_h.shape == (3, 2)
        # Static metadata is shared across the vmap batch.
        assert rbms.n_visible == 4

    def test_round_trip_flatten_unflatten(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(
            key, n_visible=3, n_hidden=2, spin_style=False
        )
        leaves, treedef = jax.tree_util.tree_flatten(rbm)
        rbm2 = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_array_equal(np.asarray(rbm2.W), np.asarray(rbm.W))
        np.testing.assert_array_equal(np.asarray(rbm2.b_v), np.asarray(rbm.b_v))
        np.testing.assert_array_equal(np.asarray(rbm2.b_h), np.asarray(rbm.b_h))
        assert rbm2.n_visible == 3
        assert rbm2.bias is True
        assert rbm2.spin_style is False


# =========================================================================== #
# RestrictedBoltzmannMachine: energy                                          #
# =========================================================================== #


class TestRBMEnergy:
    def test_manual_with_bias(self) -> None:
        W = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b_v = jnp.array([0.5, 1.0, -0.5])
        b_h = jnp.array([0.1, -0.1])
        rbm = RestrictedBoltzmannMachine.init_from_matrix(W, b_v, b_h)
        v = jnp.array([1.0, 0.0, 1.0])
        h = jnp.array([1.0, 1.0])
        x = jnp.concatenate([v, h])
        expected = -float(v @ W @ h) - float(b_v @ v) - float(b_h @ h)
        assert float(rbm.energy(x)) == pytest.approx(expected, abs=1e-5)

    def test_manual_no_bias(self) -> None:
        W = jnp.ones((2, 2))
        rbm = RestrictedBoltzmannMachine.init_from_matrix(W)
        x = jnp.array([1.0, 1.0, 1.0, 1.0])
        # E = -v^T W h = -(1+1+1+1) = -4
        assert float(rbm.energy(x)) == pytest.approx(-4.0, abs=1e-5)

    def test_zero_state_zero_energy(self) -> None:
        W = jnp.zeros((2, 3))
        b_v = jnp.zeros(2)
        b_h = jnp.zeros(3)
        rbm = RestrictedBoltzmannMachine.init_from_matrix(W, b_v, b_h)
        assert float(rbm.energy(jnp.zeros(5))) == 0.0

    def test_returns_scalar(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=3, n_hidden=2)
        assert rbm.energy(jnp.zeros(5)).shape == ()

    def test_energy_jit(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=3, n_hidden=2)
        x = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(
            float(jax.jit(rbm.energy)(x)), float(rbm.energy(x)), rtol=1e-5
        )

    def test_no_bias_energy_jit(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(
            key, n_visible=3, n_hidden=2, bias=False
        )
        x = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(
            float(jax.jit(rbm.energy)(x)), float(rbm.energy(x)), rtol=1e-5
        )


# =========================================================================== #
# RestrictedBoltzmannMachine: update_visible / update_hidden / update_state   #
# =========================================================================== #


class TestRBMUpdates:
    def test_update_visible_preserves_hidden(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=4, n_hidden=3)
        x = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        new_x = rbm.update_visible(key, x)
        assert new_x.shape == x.shape
        np.testing.assert_array_equal(np.asarray(new_x[4:]), np.asarray(x[4:]))

    def test_update_hidden_preserves_visible(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=4, n_hidden=3)
        x = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        new_x = rbm.update_hidden(key, x)
        assert new_x.shape == x.shape
        np.testing.assert_array_equal(np.asarray(new_x[:4]), np.asarray(x[:4]))

    def test_update_visible_outputs_binary(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=5, n_hidden=3)
        for k in jax.random.split(key, 20):
            new_x = rbm.update_visible(k, jnp.zeros(8))
            v_part = np.asarray(new_x[:5])
            assert set(np.unique(v_part).tolist()).issubset({0.0, 1.0})

    def test_update_hidden_outputs_binary(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=5, n_hidden=3)
        for k in jax.random.split(key, 20):
            new_x = rbm.update_hidden(k, jnp.zeros(8))
            h_part = np.asarray(new_x[5:])
            assert set(np.unique(h_part).tolist()).issubset({0.0, 1.0})

    def test_update_state_default_returns_pair(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=4, n_hidden=3)
        new_key, new_x = rbm.update_state(key, jnp.zeros(7))
        assert new_key.shape == key.shape
        assert new_x.shape == (7,)

    def test_update_state_wake_only_changes_hidden(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=4, n_hidden=3)
        x = jnp.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        for k in jax.random.split(key, 20):
            _, new_x = rbm.update_state(k, x, wake=True)
            np.testing.assert_array_equal(np.asarray(new_x[:4]), np.asarray(x[:4]))

    def test_update_visible_jit(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=4, n_hidden=3)
        x = jnp.zeros(7)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(rbm.update_visible)(key, x)),
            np.asarray(rbm.update_visible(key, x)),
        )

    def test_update_hidden_jit(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=4, n_hidden=3)
        x = jnp.ones(7)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(rbm.update_hidden)(key, x)),
            np.asarray(rbm.update_hidden(key, x)),
        )

    def test_update_state_jit_wake_true(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=3, n_hidden=2)
        x = jnp.zeros(5)
        f = jax.jit(partial(rbm.update_state, wake=True))
        new_key, new_x = f(key, x)
        # Visible part is preserved under wake.
        np.testing.assert_array_equal(np.asarray(new_x[:3]), np.asarray(x[:3]))

    def test_update_state_jit_wake_false(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=3, n_hidden=2)
        x = jnp.zeros(5)
        new_key, new_x = jax.jit(rbm.update_state)(key, x)
        assert new_x.shape == (5,)

    def test_update_state_no_bias(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(
            key, n_visible=4, n_hidden=3, bias=False
        )
        new_key, new_x = rbm.update_state(key, jnp.zeros(7))
        assert new_x.shape == (7,)

    def test_update_visible_field_formula(self) -> None:
        # Pin down the conditional formula: P(v_i = 1 | h) = sigmoid(W_i · h + b_v[i]).
        # Construct an RBM with extreme couplings so the bernoulli is basically
        # deterministic, then verify the output matches the sign-of-field rule.
        W = jnp.array([[10.0, 10.0], [-10.0, -10.0]])
        b_v = jnp.array([0.0, 0.0])
        b_h = jnp.zeros(2)
        rbm = RestrictedBoltzmannMachine.init_from_matrix(W, b_v, b_h)
        x = jnp.array([0.0, 0.0, 1.0, 1.0])
        # field for v: W @ h = [20, -20] -> sigmoid -> ~[1, 0]
        new_x = rbm.update_visible(jax.random.PRNGKey(0), x)
        v_new = np.asarray(new_x[:2])
        # Should almost certainly be [1, 0] regardless of key noise.
        np.testing.assert_array_equal(v_new, np.array([1.0, 0.0]))

    def test_update_hidden_field_formula(self) -> None:
        # Same idea, but for the hidden conditional: P(h_j=1|v) = sigmoid(v^T W_:j + b_h[j]).
        W = jnp.array([[10.0, -10.0], [10.0, -10.0]])
        b_v = jnp.zeros(2)
        b_h = jnp.array([0.0, 0.0])
        rbm = RestrictedBoltzmannMachine.init_from_matrix(W, b_v, b_h)
        x = jnp.array([1.0, 1.0, 0.0, 0.0])
        new_x = rbm.update_hidden(jax.random.PRNGKey(1), x)
        h_new = np.asarray(new_x[2:])
        # field for h: v @ W = [20, -20] -> sigmoid -> ~[1, 0]
        np.testing.assert_array_equal(h_new, np.array([1.0, 0.0]))

    def test_deterministic_with_same_key(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=3, n_hidden=2)
        x = jnp.zeros(5)
        same_key = jax.random.PRNGKey(123)
        _, x1 = rbm.update_state(same_key, x)
        _, x2 = rbm.update_state(same_key, x)
        np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))

    def test_rng_advances(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=3, n_hidden=2)
        new_key, _ = rbm.update_state(key, jnp.zeros(5))
        assert not np.array_equal(np.asarray(new_key), np.asarray(key))


# =========================================================================== #
# RestrictedBoltzmannMachine: update_params                                   #
# =========================================================================== #


class TestRBMUpdateParams:
    def test_returns_new_instance(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=3, n_hidden=2)
        new_W = jnp.ones((3, 2))
        new_b = jnp.arange(5).astype(jnp.float32)  # length n_visible + n_hidden
        rbm2 = rbm.update_params(new_W, new_b)
        assert isinstance(rbm2, RestrictedBoltzmannMachine)
        np.testing.assert_array_equal(np.asarray(rbm2.W), np.asarray(new_W))
        np.testing.assert_array_equal(np.asarray(rbm2.b_v), np.asarray(new_b[:3]))
        np.testing.assert_array_equal(np.asarray(rbm2.b_h), np.asarray(new_b[3:]))

    def test_meta_preserved(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(
            key, n_visible=3, n_hidden=2, spin_style=False
        )
        rbm2 = rbm.update_params(jnp.ones((3, 2)), jnp.zeros(5))
        assert rbm2.spin_style is False
        assert rbm2.n_visible == 3
        assert rbm2.bias is True

    def test_original_unchanged(self, key: jax.Array) -> None:
        rbm = RestrictedBoltzmannMachine.init_random(key, n_visible=3, n_hidden=2)
        W_before = np.asarray(rbm.W).copy()
        bv_before = np.asarray(rbm.b_v).copy()
        _ = rbm.update_params(jnp.zeros((3, 2)), jnp.zeros(5))
        np.testing.assert_array_equal(np.asarray(rbm.W), W_before)
        np.testing.assert_array_equal(np.asarray(rbm.b_v), bv_before)
