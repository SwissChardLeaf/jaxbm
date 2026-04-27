"""Sampling primitives for fully-connected Boltzmann machines.

A fully-connected Boltzmann machine becomes a Restricted Boltzmann Machine
when its coupling matrix `W` has bipartite block structure, i.e. the units
split into a "visible" set `V` and a "hidden" set `H` with `W[V, V] = 0`
and `W[H, H] = 0`. In that case all units in `V` are conditionally
independent given `H` (and vice versa), which lets us update them as a
single block in one step instead of one site at a time.

This module exposes both:

- `gibbs_step` / `gibbs_chain`: generic single-site Gibbs over all `n` units.
- `block_gibbs_step` / `block_gibbs_chain`: bipartite block Gibbs that
  alternately resamples a "visible" and "hidden" partition. This is the
  sampler one uses to train / sample from RBMs.
"""

from __future__ import annotations

import jax

from .bm import BoltzmannMachine


def sample_single_chain(
    machine: BoltzmannMachine,
    key,
    x0: jax.Array,
    free_units: jax.Array,
    burn_in_steps: int,
    n_samples: int,
    steps_per_sample: int = 1,
    carry_fn = None
):

    key, x = jax.lax.fori_loop(
        0, burn_in_steps, lambda i, val: machine.update_state(val[0], val[1], free_units), (key, x0)
    )

    def scan_helper(val, _):
        key, x = val
        key, x = jax.lax.fori_loop(
            0,
            steps_per_sample,
            lambda i, val: machine.update_state(val[0], val[1], free_units),
            (key, x),
        )
        if carry_fn:
            return (key, x), carry_fn(x)
        else:
            return (key, x), x

    (key, x), samples = jax.lax.scan(scan_helper, (key, x), length=n_samples)

    return samples

def sample_multiple_chains(machine, key, x0, free_units, burn_in_steps, n_samples, steps_per_sample, carry_fn=None):
    n_chains = x0.shape[0]
    keys = jax.random.split(key, n_chains)
    in_axes = (None, 0, 0, None, None, None, None, None)
    return jax.vmap(sample_single_chain, in_axes=in_axes)(machine, keys, x0, free_units, burn_in_steps, n_samples, steps_per_sample, carry_fn)