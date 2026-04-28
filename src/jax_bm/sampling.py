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
import jax.numpy as jnp

from .bm import BoltzmannMachine

def sample_chain(
    machine: BoltzmannMachine,
    key,
    x0: jax.Array,
    free_units: jax.Array,
    burn_in_steps: int,
    n_samples: int,
    steps_per_sample: int = 1,
    corr = False,
    avg = False
):

    key, x = jax.lax.fori_loop(
        0, burn_in_steps, lambda i, val: machine.update_state(val[0], val[1], free_units), (key, x0)
    )

    def helper(val, _):
        key, x = val
        key, x = jax.lax.fori_loop(
            0,
            steps_per_sample,
            lambda i, val: machine.update_state(val[0], val[1], free_units),
            (key, x),
        )
        if corr:
            return (key, x), (jnp.outer(x, x), x)
        else:
            return (key, x), x

    def avg_helper(_, val):
        key, x, running_sum = val
        key, x = jax.lax.fori_loop(
            0,
            steps_per_sample,
            lambda i, val: machine.update_state(val[0], val[1], free_units),
            (key, x),
        )

        if corr:
            return (key, x, (running_sum[0] + jnp.outer(x, x), running_sum[1] + x))
        else:
            return (key, x, running_sum + x)

    if avg:
        init_running_sum = (jnp.zeros_like(jnp.outer(x, x)), jnp.zeros_like(x)) if corr else jnp.zeros_like(x)
        (key, x, running_sum) = jax.lax.fori_loop(0, n_samples, avg_helper, (key, x, init_running_sum))
        return jax.tree.map(lambda v: v / n_samples, running_sum)
    else:
        (key, x), samples = jax.lax.scan(helper, (key, x), length=n_samples)
        return samples