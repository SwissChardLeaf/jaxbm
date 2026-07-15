"""Drivers for ``sample_chain()``: advance/accumulate/stack a sampler's chain.

Each driver takes a ``sampler(key, x) -> (key, x)`` callable (see
``_sampler.py``) and calls it repeatedly, differing only in what they do with
the intermediate states:

- ``_for_loop``: discard them, return only the final ``(key, x)``.
- ``_for_loop_mean``: accumulate a running mean of ``x``.
- ``_for_loop_corr``: accumulate a running mean of ``stat_fn(x)`` (defaults
  to ``outer(x, x)``).
- ``_scan``: stack every ``x`` into a trajectory.

``x`` need not be a single array -- it can be any pytree (e.g. ``RBM_chain``
drives a ``(v, h)`` tuple through the same four functions unchanged).

``sample_chain()`` (see ``sample.py``) picks between these based on ``mode``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _for_loop(sampler, key, x, num_calls):
    """Call ``sampler`` ``num_calls`` times, returning only the final ``(key, x)``.

    Used for plain burn-in / "advance and discard" (``mode='LAST'``).
    """
    def body(i, carry):
        key, x = carry
        return sampler(key, x)

    _, result = jax.lax.fori_loop(0, num_calls, body, (key, x))
    return result

def _for_loop_mean(sampler, key, x, num_samples):
    """Call ``sampler`` ``num_samples`` times, accumulating a running mean of ``x``.

    ``x`` may be any pytree (e.g. a ``(v, h)`` tuple); the mean is taken
    leaf-wise via ``jax.tree_util.tree_map``.

    Returns ``(key, x, x_mean)`` where ``x`` is the final state and ``x_mean``
    is the elementwise average of the ``num_samples`` visited states.
    """
    def body(i, carry):
        key, x, sum_x = carry
        key, x = sampler(key, x)
        sum_x = jax.tree_util.tree_map(jnp.add, sum_x, x)
        return key, x, sum_x

    init = (key, x, jax.tree_util.tree_map(jnp.zeros_like, x))
    _, x, sum_x = jax.lax.fori_loop(0, num_samples, body, init)
    x_mean = jax.tree_util.tree_map(lambda s: s / num_samples, sum_x)
    return x, x_mean


def _outer_self(x):
    """Default ``stat_fn`` for ``_for_loop_corr``: ``outer(x, x)``."""
    return jnp.outer(x, x)


def _for_loop_corr(sampler, key, x, num_samples, stat_fn=_outer_self):
    """Call ``sampler`` ``num_samples`` times, accumulating a running mean of
    ``stat_fn(x)``.

    ``stat_fn`` defaults to ``outer(x, x)`` (a BM's pairwise sufficient
    statistic); ``RBM_chain`` passes ``stat_fn=lambda xh: outer(xh[0], xh[1])``
    instead, since its sufficient statistic is ``outer(v, h)``.

    Returns ``(key, x, stat_mean)`` where ``x`` is the final state and
    ``stat_mean`` is the average of ``stat_fn(x)`` over the ``num_samples``
    visited states.
    """
    def body(i, carry):
        key, x, sum_stat = carry
        key, x = sampler(key, x)
        return key, x, sum_stat + stat_fn(x)

    init = (key, x, jnp.zeros_like(stat_fn(x)))
    _, x, sum_stat = jax.lax.fori_loop(0, num_samples, body, init)
    return x, sum_stat / num_samples


def _scan(sampler, key, x, num_samples):
    """Call ``sampler`` ``num_samples`` times via ``jax.lax.scan``, stacking every state.

    Returns ``(key, x), xs`` where ``xs`` has shape ``(num_samples, *x.shape)``.
    """
    def body(carry, _):
        key, x = carry
        key, x = sampler(key, x)
        return (key, x), x

    return jax.lax.scan(body, (key, x), xs=None, length=num_samples)
