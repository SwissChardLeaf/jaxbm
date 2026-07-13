"""Drivers for ``sample_chain()``: ``_for_loop`` (no-stack) and ``_scan`` (stacking).

Both drivers dispatch on ``sampler.corr`` -- the ``corr`` flag baked onto the
sampler closure by ``_sampler.py`` -- rather than inspecting the runtime
shape of ``sampler``'s return value on every call.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _for_loop(sampler, key, x, num_samples, avg=False):
    """Call ``sampler`` ``num_samples`` times without stacking a trajectory.

    With ``avg=False`` (default) this just returns the final ``(key, x)`` --
    used for plain burn-in / "advance and discard" (``n_samples is None``).
    With ``avg=True`` it instead accumulates a running sum of the visited
    states (and, if ``sampler.corr``, of ``outer(x, x)``) across ``num_samples``
    calls and divides by ``num_samples`` at the end, returning
    ``(key, x, x_mean)`` or ``(key, x, x_mean, outer_mean)``.
    """
    if not avg:
        def body(i, carry):
            key, x = carry
            result = sampler(key, x)
            return result[0], result[1]

        return jax.lax.fori_loop(0, num_samples, body, (key, x))

    if sampler.corr:
        def body(i, carry):
            key, x, sum_x, sum_outer = carry
            key, x, outer = sampler(key, x)
            return key, x, sum_x + x, sum_outer + outer

        n = x.shape[0]
        init = (key, x, jnp.zeros_like(x), jnp.zeros((n, n), dtype=x.dtype))
        key, x, sum_x, sum_outer = jax.lax.fori_loop(0, num_samples, body, init)
        return key, x, sum_x / num_samples, sum_outer / num_samples

    def body(i, carry):
        key, x, sum_x = carry
        key, x = sampler(key, x)
        return key, x, sum_x + x

    init = (key, x, jnp.zeros_like(x))
    key, x, sum_x = jax.lax.fori_loop(0, num_samples, body, init)
    return key, x, sum_x / num_samples


def _scan(sampler, key, x, num_samples):
    """Call ``sampler`` ``num_samples`` times via ``jax.lax.scan``, stacking outputs.

    Returns ``(key, x), stacked`` where ``stacked`` has a leading axis of
    length ``num_samples`` (one entry per call to ``sampler``). If
    ``sampler.corr`` is set (see ``_sampler.py``), ``stacked`` is the pair
    ``(outers, states)``; otherwise it is just the stacked states.
    """
    if sampler.corr:
        def body(carry, _):
            key, x = carry
            key, x, outer = sampler(key, x)
            return (key, x), (outer, x)
    else:
        def body(carry, _):
            key, x = carry
            key, x = sampler(key, x)
            return (key, x), x

    return jax.lax.scan(body, (key, x), xs=None, length=num_samples)
