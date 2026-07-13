"""Sampler builders for ``sample_chain()`` (see ``sample.py``).

Each builder closes over ``(weights, bias)`` and returns a ``sampler(key, x)``
callable that advances a chain by ``sampler_steps`` update(s) and returns the
resulting ``(new_key, new_x)`` (or ``(new_key, new_x, outer(new_x, new_x))``
when ``corr=True``). ``sample_chain()`` picks between the two builders below
based on whether ``weights`` has (contiguous) bipartite structure -- see
``sample_chain()``'s docstring for the detection rule.

Units are spin-valued (``{-1, +1}``) by default, matching the ``E(x) = -1/2
x^T W x - b^T x`` energy convention; pass ``spin=False`` to ``_BM_sampler`` to
instead work in binary (``{0, 1}``) units. Restricted machines (below) always
use binary units, regardless of ``spin`` -- see ``_RBM_sampler``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _bm_update(weights, bias, key, x, free_units, spin):
    """One single-site Gibbs update of a randomly chosen unit in ``free_units``.

    The new value is drawn from its conditional distribution given the rest
    of the configuration: ``p(x_i = +1 | x_{-i}) = sigmoid(2 * field_i)``.
    It is then written back as ``{-1, +1}`` if ``spin`` else ``{0, 1}``.
    """
    unit_key, bern_key, new_key = jax.random.split(key, 3)
    unit = jax.random.choice(unit_key, free_units, shape=())

    x_zeroed = x.at[unit].set(0)
    field = jnp.dot(weights[unit], x_zeroed)
    if bias is not None:
        field = field + bias[unit]

    p = 1.0 / (1.0 + jnp.exp(-2.0 * field))
    sampled = jax.random.bernoulli(bern_key, p=p)
    new_val = sampled.astype(x.dtype) * 2 - 1 if spin else sampled.astype(x.dtype)
    return new_key, x.at[unit].set(new_val)


def _BM_sampler(weights, bias, sampler_steps, corr, spin):
    """Build a single-site Gibbs sampler for a fully-connected BM.

    Returned ``sampler(key, x)`` resamples one unit at a time, ``sampler_steps``
    times, from its conditional distribution given the rest of the state,
    writing units back as ``{-1, +1}`` if ``spin`` else ``{0, 1}``.
    """
    free_units = jnp.arange(weights.shape[0])

    def sampler(key, x):
        key, x = jax.lax.fori_loop(
            0,
            sampler_steps,
            lambda i, val: _bm_update(weights, bias, val[0], val[1], free_units, spin),
            (key, x),
        )
        if corr:
            return key, x, jnp.outer(x, x)
        return key, x

    return sampler


def _rbm_update_visible(W_vh, b_v, key, x, n_visible):
    """Block-resample the visible units given the hidden configuration."""
    x_h = x[n_visible:]
    field = W_vh @ x_h
    if b_v is not None:
        field = field + b_v
    p = 1.0 / (1.0 + jnp.exp(-field))
    v_new = jax.random.bernoulli(key, p=p).astype(x.dtype)
    return jnp.concatenate([v_new, x_h])


def _rbm_update_hidden(W_vh, b_h, key, x, n_visible):
    """Block-resample the hidden units given the visible configuration."""
    x_v = x[:n_visible]
    field = x_v @ W_vh
    if b_h is not None:
        field = field + b_h
    p = 1.0 / (1.0 + jnp.exp(-field))
    h_new = jax.random.bernoulli(key, p=p).astype(x.dtype)
    return jnp.concatenate([x_v, h_new])


def _RBM_sampler(weights, bias, n_visible, sampler_steps, corr, spin):
    """Build a block-conditional sampler for a restricted BM.

    ``weights`` is the *full* ``(n, n)`` coupling matrix (bipartite, with
    zero within-block couplings); ``n_visible`` is the size of the leading
    visible block, so a state ``x`` is laid out as ``concat(v, h)``.

    Returned ``sampler(key, x)`` alternately block-resamples the visible
    units given the hidden units and vice versa, ``sampler_steps`` times.
    """
    W_vh = weights[:n_visible, n_visible:]
    b_v = bias[:n_visible] if bias is not None else None
    b_h = bias[n_visible:] if bias is not None else None

    def block_update(key, x):
        v_key, h_key, new_key = jax.random.split(key, 3)
        x = _rbm_update_visible(W_vh, b_v, v_key, x, n_visible)
        x = _rbm_update_hidden(W_vh, b_h, h_key, x, n_visible)
        return new_key, x

    def sampler(key, x):
        key, x = jax.lax.fori_loop(
            0,
            sampler_steps,
            lambda i, val: block_update(val[0], val[1]),
            (key, x),
        )
        if corr:
            return key, x, jnp.outer(x, x)
        return key, x

    sampler.corr = corr
    # Restricted machines always sample in binary {0, 1} units -- there is no
    # spin/binary choice here, unlike ``_BM_sampler``.
    sampler.spin = False
    return sampler
