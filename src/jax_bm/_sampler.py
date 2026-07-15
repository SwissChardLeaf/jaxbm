"""Sampler builders for ``sample_chain()`` (see ``sample.py``).

Each builder closes over ``(weights, bias)`` and returns a ``sampler(key, x)``
callable that advances a chain by ``sampler_steps`` update(s) and returns the
resulting ``(new_key, new_x)``. ``sample_chain()`` picks between the two
builders below based on whether ``weights`` has (contiguous) bipartite
structure -- see ``sample_chain()``'s docstring for the detection rule.
Any accumulation across calls (history, mean, correlation) is the
responsibility of the drivers in ``_loop.py``, not of the sampler itself.

Units are spin-valued (``{-1, +1}``) by default, matching the ``E(x) = -1/2
x^T W x - b^T x`` energy convention; pass ``spin=False`` to work in binary
(``{0, 1}``) units instead. Both ``_BM_sampler`` and ``_RBM_sampler`` accept
``spin`` and use it the same way. Both also accept ``clamp``, an optional
1-D integer array of unit indices to hold fixed at their current value
throughout sampling -- for ``_BM_sampler``, indices into the full state;
for ``_RBM_sampler``, indices into the visible state only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _bm_update(weights, bias, key, x, unit_p, spin):
    """One single-site Gibbs update of a unit chosen according to ``unit_p``.

    ``unit_p`` is a probability vector over ``[0, n)``; giving clamped units
    zero probability (see ``_BM_sampler``) means they can never be the one
    picked, so they keep whatever value they started with.

    The new value is drawn from its conditional distribution given the rest
    of the configuration: ``p(x_i = +1 | x_{-i}) = sigmoid(2 * field_i)``.
    It is then written back as ``{-1, +1}`` if ``spin`` else ``{0, 1}``.
    """
    unit_key, bern_key, new_key = jax.random.split(key, 3)
    unit = jax.random.choice(unit_key, unit_p.shape[0], shape=(), p=unit_p)

    x_zeroed = x.at[unit].set(0)
    field = jnp.dot(weights[unit], x_zeroed)
    if bias is not None:
        field = field + bias[unit]

    p = 1.0 / (1.0 + jnp.exp(-2.0 * field))
    sampled = jax.random.bernoulli(bern_key, p=p)
    new_val = sampled.astype(x.dtype) * 2 - 1 if spin else sampled.astype(x.dtype)
    return new_key, x.at[unit].set(new_val)


def _BM_sampler(weights, bias, sampler_steps, spin, clamp):
    """Build a single-site Gibbs sampler for a fully-connected BM.

    ``clamp`` is ``None`` or a 1-D integer array of unit indices to hold
    fixed: those units are simply never picked to be resampled, so they
    keep whatever value they had in the state passed to ``sampler(key, x)``.

    Returned ``sampler(key, x)`` resamples one (non-clamped) unit at a time,
    ``sampler_steps`` times, from its conditional distribution given the
    rest of the state, writing units back as ``{-1, +1}`` if ``spin`` else
    ``{0, 1}``.
    """
    n = weights.shape[0]
    if clamp is None:
        unit_p = jnp.full((n,), 1.0 / n)
    else:
        free_mask = jnp.ones((n,), dtype=jnp.float32).at[clamp].set(0.0)
        unit_p = free_mask / free_mask.sum()

    def sampler(key, x):
        return jax.lax.fori_loop(
            0,
            sampler_steps,
            lambda i, val: _bm_update(weights, bias, val[0], val[1], unit_p, spin),
            (key, x),
        )

    return sampler


def _rbm_update_visible(W, b_v, key, x_v, x_h, spin, free_mask):
    """Block-resample the (non-clamped) visible units given the hidden configuration.

    ``p(v_i = +1 | h) = sigmoid(2 * field_i)``; written back as ``{-1, +1}``
    if ``spin`` else ``{0, 1}`` -- same convention as ``_bm_update``. The
    whole visible layer's new values are still computed together as one
    block; ``free_mask`` (``None``, or boolean, shape ``(n_v,)``, ``True``
    at free positions) then selects, per unit, between that fresh value and
    the previous ``x_v`` -- so clamped units keep their old value.
    """
    field = W @ x_h
    if b_v is not None:
        field = field + b_v
    p = 1.0 / (1.0 + jnp.exp(-2.0 * field))
    sampled = jax.random.bernoulli(key, p=p)
    new_v = sampled.astype(x_h.dtype) * 2 - 1 if spin else sampled.astype(x_h.dtype)
    if free_mask is None:
        return new_v
    return jnp.where(free_mask, new_v, x_v)


def _rbm_update_hidden(W, b_h, key, x_v, spin):
    """Block-resample the hidden units given the visible configuration.

    ``p(h_j = +1 | v) = sigmoid(2 * field_j)``; written back as ``{-1, +1}``
    if ``spin`` else ``{0, 1}`` -- same convention as ``_bm_update``.
    """
    field = x_v @ W
    if b_h is not None:
        field = field + b_h
    p = 1.0 / (1.0 + jnp.exp(-2.0 * field))
    sampled = jax.random.bernoulli(key, p=p)
    return sampled.astype(x_v.dtype) * 2 - 1 if spin else sampled.astype(x_v.dtype)


def _RBM_sampler(W, b_v, b_h, sampler_steps, spin, clamp=None):
    """Build a block-conditional Gibbs sampler for a restricted BM.

    ``W`` is the ``(n_v, n_h)`` visible-hidden coupling matrix; ``b_v`` /
    ``b_h`` are the visible / hidden bias vectors, or ``None``. ``clamp`` is
    ``None`` or a 1-D integer array of *visible* unit indices to hold fixed:
    the non-clamped visible units are still block-resampled together (see
    ``_rbm_update_visible``), but the clamped ones keep their previous
    value. The hidden update is unaffected by ``clamp``.

    Returned ``sampler(key, x)``, with ``x = (x_v, x_h)``, alternately
    block-resamples the (non-clamped) visible units given the hidden units
    and then the hidden units given the visible units, ``sampler_steps``
    times, writing units back as ``{-1, +1}`` if ``spin`` else ``{0, 1}``.
    """
    free_mask = None
    if clamp is not None:
        free_mask = jnp.ones((W.shape[0],), dtype=bool).at[clamp].set(False)

    def block_update(key, x):
        x_v, x_h = x
        v_key, h_key, new_key = jax.random.split(key, 3)
        x_v = _rbm_update_visible(W, b_v, v_key, x_v, x_h, spin, free_mask)
        x_h = _rbm_update_hidden(W, b_h, h_key, x_v, spin)
        return new_key, (x_v, x_h)

    def sampler(key, x):
        return jax.lax.fori_loop(
            0,
            sampler_steps,
            lambda i, val: block_update(val[0], val[1]),
            (key, x),
        )

    return sampler
