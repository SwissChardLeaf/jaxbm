"""Tests for the top-level :mod:`jax_bm` package surface.

These exercise everything that a user gets from ``import jax_bm``:

- the bare import succeeds and exposes ``__version__`` / ``__all__``,
- every name in ``__all__`` is reachable as an attribute of the package,
- the re-exported symbol is the *same* object as its source-of-truth
  definition in ``jax_bm.sample``, and
- the re-exported public API actually works through the top-level namespace
  (smoke test).

As of this writing the only top-level export is ``BM_chain`` (see
``jax_bm/sample.py``).
"""

from __future__ import annotations

import importlib
import re

import jax
import jax.numpy as jnp


# =========================================================================== #
# Package metadata                                                            #
# =========================================================================== #


class TestPackageMetadata:
    def test_import_succeeds(self) -> None:
        # Re-import to make sure the package can be loaded from a clean state
        # without raising.
        mod = importlib.import_module("jax_bm")
        assert mod.__name__ == "jax_bm"

    def test_has_version(self) -> None:
        import jax_bm

        assert isinstance(jax_bm.__version__, str)
        # Looks like a PEP 440-ish version (digits separated by dots, plus an
        # optional pre/post/dev suffix). We don't try to be strict here.
        assert re.match(r"^\d+\.\d+\.\d+", jax_bm.__version__)

    def test_has_all(self) -> None:
        import jax_bm

        assert hasattr(jax_bm, "__all__")
        assert isinstance(jax_bm.__all__, list)
        assert all(isinstance(name, str) for name in jax_bm.__all__)

    def test_all_is_a_superset_of_expected_api(self) -> None:
        # The expected public surface as of today. The test fails loudly if a
        # name is removed; new names are fine.
        import jax_bm

        expected = {"BM_chain"}
        assert expected.issubset(set(jax_bm.__all__))


# =========================================================================== #
# Re-exports                                                                  #
# =========================================================================== #


class TestPackageReExports:
    def test_all_symbols_are_attributes(self) -> None:
        import jax_bm

        missing = [name for name in jax_bm.__all__ if not hasattr(jax_bm, name)]
        assert missing == [], f"names listed in __all__ but missing from module: {missing}"

    def test_bm_chain_is_same_object_as_in_sample_module(self) -> None:
        import jax_bm
        import jax_bm.sample as sample_module

        assert jax_bm.BM_chain is sample_module.BM_chain

    def test_from_import_works(self) -> None:
        # ``from jax_bm import ...`` must work for every name in __all__.
        from jax_bm import BM_chain  # noqa: F401


# =========================================================================== #
# Submodules                                                                  #
# =========================================================================== #


class TestSubmodules:
    """Each submodule should be importable on its own (so users aren't forced
    to go through the top-level package), and importing a submodule should not
    have surprising side effects."""

    def test_sample_submodule_importable(self) -> None:
        mod = importlib.import_module("jax_bm.sample")
        assert hasattr(mod, "BM_chain")

    def test_sampler_submodule_importable(self) -> None:
        mod = importlib.import_module("jax_bm._sampler")
        assert hasattr(mod, "_BM_sampler")
        assert hasattr(mod, "_RBM_sampler")

    def test_loop_submodule_importable(self) -> None:
        mod = importlib.import_module("jax_bm._loop")
        assert hasattr(mod, "_scan")
        assert hasattr(mod, "_for_loop")


# =========================================================================== #
# Smoke test of the public API                                                #
# =========================================================================== #


class TestPublicApiSmoke:
    """End-to-end exercise of the symbol re-exported at the top level. If
    this breaks, ``import jax_bm; jax_bm.BM_chain(...)`` is also broken."""

    def test_bm_chain_through_top_level(self) -> None:
        import jax_bm

        key = jax.random.PRNGKey(0)
        n = 4
        W = jnp.zeros((n, n))
        b = jnp.zeros(n)
        final_x, samples = jax_bm.BM_chain(
            key, jnp.ones(n), W, b, steps=2, n_samples=3, mode="HIST",
        )
        assert final_x.shape == (n,)
        assert samples.shape == (3, n)
