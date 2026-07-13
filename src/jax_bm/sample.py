"""Array-in / array-out sampling entry point for Boltzmann machines.

This is the sole public surface of :mod:`jax_bm` right now: given a raw
``(weights, bias)`` pair, :func:`sample_chain` validates them, detects
whether they describe a restricted (bipartite) machine, and drives the
right conditional sampler (see ``_sampler.py``) with the right driver
(``_loop.py``'s ``_scan`` for a stacked trajectory, ``_for_loop`` otherwise).
"""

from __future__ import annotations

import jax.numpy as jnp

from ._loop import _for_loop, _scan
from ._sampler import _BM_sampler, _RBM_sampler


def _validate_weights(weights, bias):
    """Check that ``weights`` is square and symmetric, and agrees with ``bias``.

    ``bias`` may be ``None`` (no bias term), in which case only ``weights``
    itself is checked. Raises ``ValueError`` on any violation.
    """
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"weights must be square (n, n); got shape {weights.shape}")
    if not jnp.allclose(weights, weights.T):
        raise ValueError("weights must be symmetric")
    if bias is not None and bias.shape[0] != weights.shape[0]:
        raise ValueError(
            "weights and bias must have the same number of rows; got "
            f"weights.shape[0]={weights.shape[0]} and bias.shape[0]={bias.shape[0]}"
        )
    return weights, bias


def _restricted_n_visible(weights, tol=1e-9):
    """Return the visible-block size if ``weights`` is contiguously bipartite.

    Looks for a split point ``0 < k < n`` such that ``weights[:k, :k] == 0``
    and ``weights[k:, k:] == 0`` (no within-block couplings) *and* there is
    at least one actual coupling crossing the split (``weights[:k, k:]`` is
    not all-zero) -- otherwise an all-zero ``weights`` would be ambiguous
    between "fully-connected with no couplings" and "restricted with no
    couplings", which matters here because the two samplers use different
    unit conventions (spin ``{-1, +1}`` vs. binary ``{0, 1}``). Returns
    ``None`` if no such split exists, i.e. the machine is not restricted.
    """
    n = weights.shape[0]
    nonzero = jnp.abs(weights) > tol
    for k in range(1, n):
        within_zero = not bool(jnp.any(nonzero[:k, :k])) and not bool(jnp.any(nonzero[k:, k:]))
        has_cross_coupling = bool(jnp.any(nonzero[:k, k:]))
        if within_zero and has_cross_coupling:
            return k
    return None


def _validate_state(x, weights, n_visible, spin):
    """Check that the initial state ``x`` agrees with ``weights`` in size and
    only takes values in whichever convention the sampler will actually use.

    Restricted machines (``n_visible is not None``) always sample in binary
    ``{0, 1}`` units (see ``_RBM_sampler``), regardless of ``spin``. A
    fully-connected machine samples in ``{-1, +1}`` if ``spin`` else
    ``{0, 1}`` (see ``_BM_sampler``). Raises ``ValueError`` on any violation.
    """
    if x.ndim != 1 or x.shape[0] != weights.shape[0]:
        raise ValueError(
            "x must be a vector matching weights in size; got x.shape="
            f"{x.shape} and weights.shape[0]={weights.shape[0]}"
        )
    use_spin = spin and n_visible is None
    allowed = (-1.0, 1.0) if use_spin else (0.0, 1.0)
    allowed_arr = jnp.asarray(allowed, dtype=x.dtype)
    if not bool(jnp.all(jnp.isin(x, allowed_arr))):
        raise ValueError(f"x must only contain values in {allowed}; got {x}")
    return x


def _build_sampler(weights, bias, n_visible, sampler_steps=1, corr=False, spin=True):
    """Build the appropriate conditional sampler for ``(weights, bias)``.

    ``n_visible`` is the (already-computed) result of ``_restricted_n_visible``.
    """
    if n_visible is not None:
        return _RBM_sampler(weights, bias, n_visible, sampler_steps=sampler_steps, corr=corr)
    return _BM_sampler(weights, bias, sampler_steps=sampler_steps, corr=corr, spin=spin)


def sample_chain(key, x, weights, bias, steps, n_samples=None, avg=False, corr=False, spin=True):
    """Sample from a Boltzmann machine defined by raw ``(weights, bias)``.

    This is the top-level, array-in/array-out entry point for sampling: it
    does not require the caller to build a model object first. Everything
    needed is inferred from ``weights`` and ``bias``, and the right
    conditional sampler and driver are chosen automatically.

    Behavior
    --------
    1. **Validation.** ``weights`` must be square (``(n, n)``) and symmetric,
       and ``bias`` must agree with it in size (``bias.shape == (n,)``), or be
       ``None``. Bad shapes raise ``ValueError`` before any sampling happens
       (see ``_validate_weights``). The initial state ``x`` is checked
       similarly: it must be a vector of length ``n``, and its entries must
       lie in whichever convention the sampler will actually use -- ``{-1,
       +1}`` for a fully-connected machine with ``spin=True``, or ``{0, 1}``
       otherwise (restricted machines are always binary; see
       ``_validate_state``).
    2. **Restricted detection.** Whether the machine is *restricted* is read
       directly off the structure of ``weights`` -- there is no separate flag
       for this. If there is a split point ``k`` such that ``weights[:k, :k]``
       and ``weights[k:, k:]`` are both all-zero (i.e. the ``n`` units split
       into a "visible" prefix of length ``k`` and a "hidden" suffix with no
       within-block couplings), the machine is treated as restricted, with
       ``x`` laid out as ``concat(v, h)``. See ``_restricted_n_visible``.
    3. **Sampler selection**, via ``_sampler.py``:
       - Restricted: use the block-conditional sampler (``_RBM_sampler``),
         which alternately resamples the whole visible block given the hidden
         block and vice versa.
       - Not restricted: use the single-site Gibbs sampler (``_BM_sampler``),
         which resamples one unit at a time from its conditional
         distribution.
       Either sampler advances the chain by ``steps`` update(s) per call, so
       ``steps`` doubles as "burn-in length" when ``n_samples is None`` and as
       "steps between consecutive samples" when ``n_samples`` is given.
    4. **Driver selection**, via ``_loop.py``:
       - ``n_samples`` is an int and ``avg=False``: drive the sampler with
         ``_scan`` (``jax.lax.scan``), calling it ``n_samples`` times and
         stacking every resulting state (shape ``(n_samples, n)``).
       - ``n_samples is None``: drive the sampler with ``_for_loop`` for a
         single call, returning just the final state -- e.g. for burn-in, or
         to advance the chain between calls.
       - ``avg=True`` (requires ``n_samples``): drive the sampler with
         ``_for_loop`` in accumulating mode, which does not stack a
         trajectory but instead sums the ``n_samples`` draws and divides by
         ``n_samples`` at the end.
    5. **``avg``.** When set, samples are not stacked; instead the function
       returns the elementwise average of the sampled states (and, if
       ``corr=True``, the average outer product) over the ``n_samples``
       draws.
    6. **``corr``.** When set, each sampled state ``x`` additionally reports
       its pairwise correlation matrix ``jnp.outer(x, x)`` alongside the
       state itself (or, combined with ``avg=True``, the average of those
       outer products rather than one outer product per sample).

    Parameters
    ----------
    key:
        JAX PRNG key.
    x:
        Initial state, shape ``(n,)`` (for a restricted machine, laid out as
        ``concat(v, h)``).
    weights:
        Square coupling matrix, shape ``(n, n)``.
    bias:
        Bias vector, shape ``(n,)``, or ``None`` for no bias.
    steps:
        Number of chain-update steps to take: either the total burn-in
        length (when ``n_samples is None``) or the number of steps between
        consecutive samples (when ``n_samples`` is given).
    n_samples:
        Number of samples to draw. ``None`` (default) means "just advance
        the chain and return the final state" -- no trajectory is collected.
    avg:
        If ``True``, return the average of the sampled states (and outer
        products, if ``corr=True``) instead of the full stacked trajectory.
        Requires ``n_samples`` to be given.
    corr:
        If ``True``, also compute/report ``outer(x, x)`` for the sampled
        state(s).
    spin:
        Only affects fully-connected (non-restricted) machines. If ``True``
        (default), units are ``{-1, +1}``-valued; if ``False``, ``{0, 1}``.
        Restricted machines are always ``{0, 1}``-valued regardless of
        ``spin``. ``x`` is validated against whichever convention actually
        applies (see ``_validate_state``).

    Returns
    -------
    The final state ``x`` (shape ``(n,)``) is always returned; what else is
    returned alongside it depends on ``n_samples`` / ``avg`` / ``corr``:

    ==============  ==============  ==============  =================================
    ``n_samples``   ``avg``         ``corr``        Return value
    ==============  ==============  ==============  =================================
    ``None``        --              --              ``x``
    given           ``False``       ``False``       ``(x, samples)``
    given           ``False``       ``True``        ``(x, (outers, states))``
    given           ``True``        ``False``       ``(x, x_mean)``
    given           ``True``        ``True``        ``(x, outer_mean, x_mean)``
    ==============  ==============  ==============  =================================

    where ``samples.shape == (n_samples, n)``, ``outers.shape ==
    (n_samples, n, n)``, and ``x_mean`` / ``outer_mean`` are the elementwise
    averages of ``states`` / ``outers`` over the ``n_samples`` draws.
    """
    _validate_weights(weights, bias)
    n_visible = _restricted_n_visible(weights)
    _validate_state(x, weights, n_visible, spin)
    sampler = _build_sampler(weights, bias, n_visible, sampler_steps=steps, corr=corr, spin=spin)

    if n_samples is None:
        _, x = _for_loop(sampler, key, x, 1)
        return x

    if avg:
        if corr:
            _, x, x_mean, outer_mean = _for_loop(sampler, key, x, n_samples, avg=True)
            return x, outer_mean, x_mean
        _, x, x_mean = _for_loop(sampler, key, x, n_samples, avg=True)
        return x, x_mean

    (_, x), stacked = _scan(sampler, key, x, n_samples)
    return x, stacked
