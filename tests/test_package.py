"""Tests for the top-level :mod:`jaxbm` package surface.

These exercise everything that a user gets from ``import jaxbm``:

- the bare import succeeds and exposes ``__version__`` / ``__all__``,
- every name in ``__all__`` is reachable as an attribute of the package,
- the re-exported symbol is the *same* object as its source-of-truth
  definition in ``jaxbm.sample``, and
- the re-exported public API actually works through the top-level namespace
  (smoke test).

As of this writing the top-level exports are ``BM_chain`` and ``RBM_chain``
(see ``jaxbm/sample.py``).
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
        mod = importlib.import_module("jaxbm")
        assert mod.__name__ == "jaxbm"

    def test_has_version(self) -> None:
        import jaxbm

        assert isinstance(jaxbm.__version__, str)
        # Looks like a PEP 440-ish version (digits separated by dots, plus an
        # optional pre/post/dev suffix). We don't try to be strict here.
        assert re.match(r"^\d+\.\d+\.\d+", jaxbm.__version__)

    def test_has_all(self) -> None:
        import jaxbm

        assert hasattr(jaxbm, "__all__")
        assert isinstance(jaxbm.__all__, list)
        assert all(isinstance(name, str) for name in jaxbm.__all__)

    def test_all_is_a_superset_of_expected_api(self) -> None:
        # The expected public surface as of today. The test fails loudly if a
        # name is removed; new names are fine.
        import jaxbm

        expected = {"BM_chain", "RBM_chain"}
        assert expected.issubset(set(jaxbm.__all__))


# =========================================================================== #
# Re-exports                                                                  #
# =========================================================================== #


class TestPackageReExports:
    def test_all_symbols_are_attributes(self) -> None:
        import jaxbm

        missing = [name for name in jaxbm.__all__ if not hasattr(jaxbm, name)]
        assert missing == [], f"names listed in __all__ but missing from module: {missing}"

    def test_bm_chain_is_same_object_as_in_sample_module(self) -> None:
        import jaxbm
        import jaxbm.sample as sample_module

        assert jaxbm.BM_chain is sample_module.BM_chain

    def test_from_import_works(self) -> None:
        # ``from jaxbm import ...`` must work for every name in __all__.
        from jaxbm import BM_chain  # noqa: F401


# =========================================================================== #
# Submodules                                                                  #
# =========================================================================== #


class TestSubmodules:
    """Each submodule should be importable on its own (so users aren't forced
    to go through the top-level package), and importing a submodule should not
    have surprising side effects."""

    def test_sample_submodule_importable(self) -> None:
        mod = importlib.import_module("jaxbm.sample")
        assert hasattr(mod, "BM_chain")

    def test_sampler_submodule_importable(self) -> None:
        mod = importlib.import_module("jaxbm._sampler")
        assert hasattr(mod, "_BM_sampler")
        assert hasattr(mod, "_RBM_sampler")

    def test_loop_submodule_importable(self) -> None:
        mod = importlib.import_module("jaxbm._loop")
        assert hasattr(mod, "_scan")
        assert hasattr(mod, "_for_loop")


# =========================================================================== #
# Smoke test of the public API                                                #
# =========================================================================== #


class TestPublicApiSmoke:
    """End-to-end exercise of the symbol re-exported at the top level. If
    this breaks, ``import jaxbm; jaxbm.BM_chain(...)`` is also broken."""

    def test_bm_chain_through_top_level(self) -> None:
        import jaxbm

        key = jax.random.PRNGKey(0)
        n = 4
        W = jnp.zeros((n, n))
        b = jnp.zeros(n)
        final_x, samples = jaxbm.BM_chain(
            key, jnp.ones(n), W, b, steps=2, n_samples=3, mode="HIST",
        )
        assert final_x.shape == (n,)
        assert samples.shape == (3, n)
