"""JAX loops for ``BM_chain`` / ``RBM_chain``: advance/accumulate/stack a
sampler's chain. Each loop takes a ``sampler(key, x) -> (key, x)`` callable
(see ``_sampler.py``) and calls it repeatedly, differing only in what they do
with the intermediate states:

- ``_for_loop``: discard them, return only the final ``(key, x)``.
- ``_for_loop_stat``: accumulate a running mean of ``stat_fn(x)`` -- or of
  ``x`` itself, if ``stat_fn`` is ``None`` -- used for both ``mode='MEAN'``
  (``stat_fn=None``) and ``mode='CORR'`` (``stat_fn=_bm_outer_self`` /
  ``_rbm_outer_self``).
- ``_scan``: stack every ``x`` into a trajectory.

``x`` need not be a single array -- it can be any pytree (e.g. ``RBM_chain``
drives a ``(v, h)`` tuple through the same four functions unchanged).

``BM_chain`` / ``RBM_chain`` (see ``sample.py``) pick between these based on
``n_samples`` and ``mode``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _for_loop(sampler, key, x, num_calls):
    """Call ``sampler`` ``num_calls`` times, returning only the final ``x``
    (the key is advanced internally but discarded too, along with every
    intermediate state).

    Used for plain burn-in / "advance and discard" (``n_samples=None``).
    """
    def body(i, carry):
        key, x = carry
        return sampler(key, x)

    _, result = jax.lax.fori_loop(0, num_calls, body, (key, x))
    return result

def _bm_outer_self(x):
    """``stat_fn`` for a BM's ``mode='CORR'``: ``outer(x, x)``."""
    return jnp.outer(x, x)

def _rbm_outer_self(x):
    """``stat_fn`` for an RBM's ``mode='CORR'``, with ``x = (x_v, x_h)``:
    ``outer(x_v, x_h)``."""
    return jnp.outer(x[0], x[1])

def _for_loop_stat(sampler, key, x, num_samples, stat_fn=None):
    """Call ``sampler`` ``num_samples`` times, accumulating a running mean of
    ``stat_fn(x)`` -- or of ``x`` itself if ``stat_fn`` is ``None``.

    ``stat_fn=None`` (the default) is what ``mode='MEAN'`` uses: it's
    equivalent to passing the identity function, so the accumulated
    statistic is just ``x``. ``mode='CORR'`` instead passes a real
    ``stat_fn`` -- ``_bm_outer_self`` / ``_rbm_outer_self`` above, or
    equivalent -- to accumulate a pairwise statistic instead.

    ``x`` (and whatever ``stat_fn`` returns) may be any pytree, e.g. a
    ``(v, h)`` tuple for an RBM; accumulation is done leaf-wise via
    ``jax.tree_util.tree_map``, so both cases share the same code path here.

    Returns ``(x, stat_mean)`` where ``x`` is the final state and
    ``stat_mean`` is the average of ``stat_fn(x)`` (or of ``x``) over the
    ``num_samples`` visited states.
    """
    if stat_fn is None:
        def stat_fn(x):
            return x

    def body(i, carry):
        key, x, sum_stat = carry
        key, x = sampler(key, x)
        sum_stat = jax.tree_util.tree_map(jnp.add, sum_stat, stat_fn(x))
        return key, x, sum_stat

    init = (key, x, jax.tree_util.tree_map(jnp.zeros_like, stat_fn(x)))
    _, x, sum_stat = jax.lax.fori_loop(0, num_samples, body, init)
    stat_mean = jax.tree_util.tree_map(lambda s: s / num_samples, sum_stat)
    return x, stat_mean


def _scan(sampler, key, x, num_samples):
    """Call ``sampler`` ``num_samples`` times via ``jax.lax.scan``, stacking every state.

    Returns ``(key, x), xs``, where ``x`` is the final state and ``xs`` is
    the stacked trajectory: if ``x`` is a single array, ``xs.shape ==
    (num_samples, *x.shape)``; if ``x`` is a pytree (e.g. a ``(v, h)``
    tuple), ``xs`` is a pytree of the same structure with a leading
    ``num_samples`` axis added to each leaf (e.g. ``(vs, hs)``).
    """
    def body(carry, _):
        key, x = carry
        key, x = sampler(key, x)
        return (key, x), x

    return jax.lax.scan(body, (key, x), xs=None, length=num_samples)
