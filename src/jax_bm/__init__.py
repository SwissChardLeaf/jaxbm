"""jax_bm: JAX-based simulations of Boltzmann machines."""

from jax_bm.bm import BoltzmannMachine, RestrictedBoltzmannMachine
from jax_bm.sampling import sample_single_chain, sample_multiple_chains

__all__ = ["BoltzmannMachine", "RestrictedBoltzmannMachine", "sample_single_chain", "sample_multiple_chains"]

__version__ = "0.0.1"
