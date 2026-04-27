"""Tests for the top-level :mod:`jax_bm` package surface.

These exercise everything that a user gets from ``import jax_bm``:

- the bare import succeeds and exposes ``__version__`` / ``__all__``,
- every name in ``__all__`` is reachable as an attribute of the package,
- the re-exported symbols are the *same* objects as their source-of-truth
  definitions in the submodules,
- the submodules themselves are importable, and
- the re-exported public API actually works through the top-level
  namespace (smoke test).
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

        expected = {
            "BoltzmannMachine",
            "RestrictedBoltzmannMachine",
            "sample_single_chain",
            "sample_multiple_chains",
        }
        assert expected.issubset(set(jax_bm.__all__))


# =========================================================================== #
# Re-exports                                                                  #
# =========================================================================== #


class TestPackageReExports:
    def test_all_symbols_are_attributes(self) -> None:
        import jax_bm

        missing = [name for name in jax_bm.__all__ if not hasattr(jax_bm, name)]
        assert missing == [], f"names listed in __all__ but missing from module: {missing}"

    def test_classes_are_same_object_as_in_bm_module(self) -> None:
        import jax_bm
        import jax_bm.bm as bm

        assert jax_bm.BoltzmannMachine is bm.BoltzmannMachine
        assert jax_bm.RestrictedBoltzmannMachine is bm.RestrictedBoltzmannMachine

    def test_samplers_are_same_object_as_in_sampling_module(self) -> None:
        import jax_bm
        import jax_bm.sampling as sampling

        assert jax_bm.sample_single_chain is sampling.sample_single_chain
        assert jax_bm.sample_multiple_chains is sampling.sample_multiple_chains

    def test_from_import_works(self) -> None:
        # ``from jax_bm import ...`` must work for every name in __all__.
        from jax_bm import (  # noqa: F401
            BoltzmannMachine,
            RestrictedBoltzmannMachine,
            sample_multiple_chains,
            sample_single_chain,
        )


# =========================================================================== #
# Submodules                                                                  #
# =========================================================================== #


class TestSubmodules:
    """Each submodule should be importable on its own (so users aren't forced
    to go through the top-level package), and importing a submodule should not
    have surprising side effects."""

    def test_bm_submodule_importable(self) -> None:
        mod = importlib.import_module("jax_bm.bm")
        assert hasattr(mod, "BoltzmannMachine")
        assert hasattr(mod, "RestrictedBoltzmannMachine")
        assert hasattr(mod, "AbstractBoltzmannMachine")

    def test_sampling_submodule_importable(self) -> None:
        mod = importlib.import_module("jax_bm.sampling")
        assert hasattr(mod, "sample_single_chain")
        assert hasattr(mod, "sample_multiple_chains")

    def test_statistics_submodule_importable(self) -> None:
        mod = importlib.import_module("jax_bm.statistics")
        assert hasattr(mod, "compute_rhat")
        assert hasattr(mod, "compute_ess")

    def test_utils_submodule_importable(self) -> None:
        mod = importlib.import_module("jax_bm.utils")
        # ``utils`` should expose at least the basic helpers.
        assert hasattr(mod, "sigmoid")
        assert hasattr(mod, "logit")


# =========================================================================== #
# Smoke test of the public API                                                #
# =========================================================================== #


class TestPublicApiSmoke:
    """End-to-end exercise of the symbols re-exported at the top level. If
    any of these break, ``import jax_bm; jax_bm.xxx(...)`` is also broken."""

    def test_construct_boltzmann_machine_from_top_level(self) -> None:
        import jax_bm

        bm = jax_bm.BoltzmannMachine.init_random(jax.random.PRNGKey(0), n=4)
        assert isinstance(bm, jax_bm.BoltzmannMachine)
        assert bm.W.shape == (4, 4)

    def test_construct_rbm_from_top_level(self) -> None:
        import jax_bm

        rbm = jax_bm.RestrictedBoltzmannMachine.init_random(
            jax.random.PRNGKey(0), n_visible=3, n_hidden=2,
        )
        assert isinstance(rbm, jax_bm.RestrictedBoltzmannMachine)
        assert rbm.W.shape == (3, 2)

    def test_sample_single_chain_through_top_level(self) -> None:
        import jax_bm

        key = jax.random.PRNGKey(0)
        bm = jax_bm.BoltzmannMachine.init_random(key, n=4)
        out = jax_bm.sample_single_chain(
            bm, key, jnp.ones(4), jnp.arange(4),
            burn_in_steps=2, n_samples=3,
        )
        assert out.shape == (3, 4)

    def test_sample_multiple_chains_through_top_level(self) -> None:
        import jax_bm

        key = jax.random.PRNGKey(0)
        bm = jax_bm.BoltzmannMachine.init_random(key, n=4)
        out = jax_bm.sample_multiple_chains(
            bm, key, jnp.ones((2, 4)), jnp.arange(4),
            burn_in_steps=1, n_samples=3, steps_per_sample=1,
        )
        assert out.shape == (2, 3, 4)
