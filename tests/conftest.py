"""Shared pytest fixtures for the jax_bm test suite."""

from __future__ import annotations

import jax
import pytest


@pytest.fixture
def key() -> jax.Array:
    """A deterministic PRNG key for tests."""
    return jax.random.PRNGKey(0)
