"""jax_bm: JAX-based simulations of Boltzmann machines."""

from jax_bm.bm import BoltzmannMachine, RestrictedBoltzmannMachine
from jax_bm.sampling import sample_chain

__all__ = ["BoltzmannMachine", "RestrictedBoltzmannMachine", "sample_chain"]

__version__ = "0.0.1"
