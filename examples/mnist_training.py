"""Example: wake-sleep training of a Boltzmann machine on MNIST.

This is legacy prototype code, kept here as a worked example rather than as
part of the installable ``jax_bm`` package. It predates the new array-in
``sample()`` entry point (see ``tests/sample.py``) and will need to be
updated to call that (or ``jax_bm.sampling.sample_chain``) once it lands.
"""

from __future__ import annotations

import jax.numpy as jnp

from mnist_config import ALL_UNITS, HIDDEN_UNITS


def sample_correlations(samples):
    return jnp.einsum('BNi,BNj->BNij', samples, samples)

def mean_sample_correlations(samples):
    return jnp.mean(sample_correlations(samples), axis=(0, 1))

def mnist_wake_sleep_update(
    key,
    dataset,
    bm,
    learning_rate,
    wake_n_samples,
    wake_burn_in_steps,
    wake_steps_per_sample,
    sleep_n_samples,
    sleep_burn_in_steps,
    sleep_steps_per_sample,
):
    clamp_state = dataset.get_clamp()
    wake_samples = sample(key, bm, clamp_state, jnp.arange(HIDDEN_UNITS), wake_burn_in_steps, wake_steps_per_sample)
    wake_correlations = mean_sample_correlations(wake_samples)

    sleep_samples = sample(key, bm, wake_samples, jnp.arange(ALL_UNITS), sleep_burn_in_steps, sleep_steps_per_sample)
    sleep_correlations = mean_sample_correlations(sleep_samples)

    wake_correlations_delta = wake_correlations - sleep_correlations
    new_W = bm.W + learning_rate * wake_correlations_delta

    if bm.bias:
        ...

    return init_machine.replace(W=new_W)
