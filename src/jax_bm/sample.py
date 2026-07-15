"""Array-in / array-out sampling entry points for Boltzmann machines.

:func:`BM_chain` (fully-connected) and, eventually, ``RBM_chain``
(restricted) are the public surface of :mod:`jax_bm`: given raw
``(weights, bias)`` arrays, they validate them and drive the right
conditional sampler (see ``_sampler.py``) with the right driver (``_loop.py``'s
``_scan`` for a stacked trajectory, ``_for_loop`` variants otherwise).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.experimental import checkify

from ._loop import _for_loop, _for_loop_corr, _for_loop_mean, _scan
from ._sampler import _BM_sampler, _RBM_sampler

_MODES = ("HIST", "MEAN", "CORR")


def _bm_validate(x, weights, bias, spin, clamp):
    """Validate all of ``BM_chain``'s array inputs together.

    - ``weights`` must be square, shape ``(n, n)``, and symmetric.
    - ``bias`` must be ``None`` or a vector of shape ``(n,)``.
    - ``x`` must be a vector of shape ``(n,)`` with entries in ``{-1, +1}``
      if ``spin`` else ``{0, 1}`` -- the convention ``_BM_sampler`` actually
      samples in.
    - ``clamp`` must be ``None`` or a 1-D integer array of *distinct* indices
      into ``[0, n)`` (a *subset* of unit indices, arbitrary length, not
      necessarily ``n`` -- i.e. something you could hand to ``x[clamp]``).

    Shape/dtype checks (static under ``jit``) raise ``ValueError`` directly.
    Checks that depend on array *values* (symmetry, unit convention, clamp
    bounds) go through ``checkify.check`` instead of a plain ``if``/``raise``,
    so this function itself can be traced -- e.g. ``checkify.checkify(
    _bm_validate)(...)`` works under ``jit``/``scan``, unlike a bare
    ``if not jnp.allclose(...): raise ...``, which would try to convert a
    traced array to a Python ``bool`` and fail. ``BM_chain`` calls this via
    ``checkify.checkify`` and immediately ``.throw()``s, so from the outside
    it still just raises ``ValueError`` like before.
    """
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"weights must be square (n, n); got shape {weights.shape}")
    n = weights.shape[0]
    checkify.check(jnp.allclose(weights, weights.T), "weights must be symmetric")

    if bias is not None and bias.shape != (n,):
        raise ValueError(
            f"bias must have shape ({n},) matching weights; got bias.shape={bias.shape}"
        )

    if x.ndim != 1 or x.shape[0] != n:
        raise ValueError(
            f"x must be a vector matching weights in size; got x.shape={x.shape} and n={n}"
        )
    allowed = (-1.0, 1.0) if spin else (0.0, 1.0)
    allowed_arr = jnp.asarray(allowed, dtype=x.dtype)
    checkify.check(
        jnp.all(jnp.isin(x, allowed_arr)),
        f"x must only contain values in {allowed}; got x={{x}}", x=x,
    )

    if clamp is not None:
        if not jnp.issubdtype(clamp.dtype, jnp.integer):
            raise ValueError(f"clamp must be an integer array; got dtype {clamp.dtype}")
        if clamp.ndim != 1:
            raise ValueError(f"clamp must be 1-D; got shape {clamp.shape}")
        checkify.check(
            jnp.all((clamp >= 0) & (clamp < n)),
            f"clamp indices must be in [0, {n}); got clamp={{clamp}}", clamp=clamp,
        )
        sorted_clamp = jnp.sort(clamp)
        checkify.check(
            jnp.all(sorted_clamp[1:] != sorted_clamp[:-1]),
            "clamp must not contain duplicate indices; got clamp={clamp}", clamp=clamp,
        )


def _check_sampling_mode(mode, n_samples):
    """Check that ``mode`` and ``n_samples`` agree with each other.

    - ``n_samples=None`` (default) means "no accumulation, just advance the
      chain and return the final state" -- ``mode`` must then also be
      ``None`` (there is nothing to summarize).
    - Otherwise, ``n_samples`` must be a positive ``int``, and ``mode`` must
      be one of ``_MODES``, selecting how to summarize the ``n_samples``
      visited states.

    Raises ``ValueError`` on any violation.
    """
    if n_samples is None:
        if mode is not None:
            raise ValueError(f"mode must be None when n_samples is None; got mode={mode!r}")
        return

    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int; got n_samples={n_samples!r}")
    if mode not in _MODES:
        raise ValueError(
            f"mode must be one of {_MODES} when n_samples is given; got {mode!r}"
        )


def BM_chain(key, x, weights, bias, steps, n_samples=None, mode=None, spin=True, clamp=None):
    """Sample from a fully-connected Boltzmann machine (single-site Gibbs).

    Behavior
    --------
    1. **Validation**, via ``_bm_validate``: ``weights`` must be square
       (``(n, n)``) and symmetric, and ``bias`` must agree with it in size
       (``bias.shape == (n,)``), or be ``None``. The initial state ``x``
       must be a vector of length ``n`` with entries in ``{-1, +1}`` if
       ``spin`` else ``{0, 1}``. ``clamp`` must be ``None`` or a 1-D integer
       array of indices into ``[0, n)``. ``mode`` and ``n_samples`` must
       also agree (see below). Bad inputs raise ``ValueError`` before any
       sampling happens.
    2. **Sampling**, via ``_sampler.py``'s ``_BM_sampler``: resamples one
       (non-clamped) unit at a time from its conditional distribution given
       the rest of the state, ``steps`` times per call -- either once
       (``n_samples is None``) or ``n_samples`` times, so ``steps`` is the
       number of update steps between the input state and the state(s)
       returned. If ``clamp`` is given, those units are never chosen, so
       they stay fixed at their value in ``x`` for the whole call.
    3. **``mode`` / driver selection**, via ``_loop.py``:
       - ``n_samples is None`` (default; ``mode`` must also be ``None``): a
         single call to ``_for_loop``, returning just the final state --
         e.g. for burn-in, or to advance the chain between calls.
       - ``mode='HIST'`` (requires ``n_samples``): ``_scan`` (``jax.lax.scan``),
         stacking every resulting state (shape ``(n_samples, n)``).
       - ``mode='MEAN'`` (requires ``n_samples``): ``_for_loop_mean``, which
         accumulates the elementwise mean of the ``n_samples`` visited
         states without stacking a trajectory.
       - ``mode='CORR'`` (requires ``n_samples``): ``_for_loop_corr``, which
         accumulates the mean of ``outer(x, x)`` over the ``n_samples``
         visited states (the sufficient statistic for the pairwise term of
         a BM's log-likelihood gradient).

    Parameters
    ----------
    key:
        JAX PRNG key.
    x:
        Initial state, shape ``(n,)``.
    weights:
        Symmetric coupling matrix, shape ``(n, n)``, zero diagonal.
    bias:
        Bias vector, shape ``(n,)``, or ``None`` for no bias.
    steps:
        Number of chain-update steps to take between the input state and
        each returned state.
    n_samples:
        Number of samples to draw. ``None`` (default) means "just advance
        the chain and return the final state" -- no trajectory is collected,
        and ``mode`` must be ``None``. Otherwise a positive ``int``, and
        ``mode`` must be given (see above).
    mode:
        ``None`` (default, requires ``n_samples is None``), or one of
        ``'HIST'``, ``'MEAN'``, ``'CORR'`` (each requires ``n_samples``) --
        see above.
    spin:
        If ``True`` (default), units are ``{-1, +1}``-valued; if ``False``,
        ``{0, 1}``. ``x`` is validated against this convention.
    clamp:
        ``None``, or an optional integer array of indices into ``[0, n)``
        naming units to hold fixed at their value in ``x``: those units are
        simply never chosen to be resampled, for the whole call.

    Returns
    -------
    The final state ``x`` (shape ``(n,)``) is always returned; what else is
    returned alongside it depends on ``mode``:

    ==============  ==============  =================================
    ``n_samples``   ``mode``        Return value
    ==============  ==============  =================================
    ``None``        ``None``        ``x``
    given           ``'HIST'``      ``(x, xs)``
    given           ``'MEAN'``      ``(x, x_mean)``
    given           ``'CORR'``      ``(x, outer_mean)``
    ==============  ==============  =================================

    where ``xs.shape == (n_samples, n)`` is the full stacked trajectory, and
    ``x_mean`` / ``outer_mean`` are the elementwise averages of ``x`` /
    ``outer(x, x)`` over the ``n_samples`` draws.
    """
    _check_sampling_mode(mode, n_samples)
    err, _ = checkify.checkify(_bm_validate)(x, weights, bias, spin, clamp)
    err.throw()

    sampler = _BM_sampler(weights, bias, sampler_steps=steps, spin=spin, clamp=clamp)

    if n_samples is None:
        return _for_loop(sampler, key, x, 1)
    if mode == "HIST":
        (_, x), xs = _scan(sampler, key, x, n_samples)
        return x, xs
    if mode == "MEAN":
        return _for_loop_mean(sampler, key, x, n_samples)
    return _for_loop_corr(sampler, key, x, n_samples)


def _rbm_validate(x_v, x_h, W, b_v, b_h, spin, clamp):
    """Validate all of ``RBM_chain``'s array inputs together.

    - ``W`` must be 2-D, shape ``(n_v, n_h)``.
    - ``b_v`` must be ``None`` or a vector of shape ``(n_v,)``; likewise
      ``b_h`` against ``(n_h,)``.
    - ``x_v`` must be a vector of shape ``(n_v,)``, and ``x_h`` a vector of
      shape ``(n_h,)``, both with entries in ``{-1, +1}`` if ``spin`` else
      ``{0, 1}`` -- same convention as ``_bm_validate``.
    - ``clamp`` must be ``None`` or a 1-D integer array of *distinct*
      indices into ``[0, n_v)`` -- clamping only ever applies to the visible
      units.

    Same ``checkify``/``ValueError`` split as ``_bm_validate``: shape/dtype
    checks (static under ``jit``) raise directly, and value-dependent checks
    go through ``checkify.check`` so this function can be traced.

    Raises ``ValueError`` on any violation.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D (n_v, n_h); got shape {W.shape}")
    n_v, n_h = W.shape

    if b_v is not None and b_v.shape != (n_v,):
        raise ValueError(
            f"b_v must have shape ({n_v},) matching W; got b_v.shape={b_v.shape}"
        )
    if b_h is not None and b_h.shape != (n_h,):
        raise ValueError(
            f"b_h must have shape ({n_h},) matching W; got b_h.shape={b_h.shape}"
        )

    if x_v.ndim != 1 or x_v.shape[0] != n_v:
        raise ValueError(
            f"x_v must be a vector matching W in size; got x_v.shape={x_v.shape} and n_v={n_v}"
        )
    if x_h.ndim != 1 or x_h.shape[0] != n_h:
        raise ValueError(
            f"x_h must be a vector matching W in size; got x_h.shape={x_h.shape} and n_h={n_h}"
        )
    allowed = (-1.0, 1.0) if spin else (0.0, 1.0)
    v_allowed = jnp.asarray(allowed, dtype=x_v.dtype)
    checkify.check(
        jnp.all(jnp.isin(x_v, v_allowed)),
        f"x_v must only contain values in {allowed}; got x_v={{x_v}}", x_v=x_v,
    )
    h_allowed = jnp.asarray(allowed, dtype=x_h.dtype)
    checkify.check(
        jnp.all(jnp.isin(x_h, h_allowed)),
        f"x_h must only contain values in {allowed}; got x_h={{x_h}}", x_h=x_h,
    )

    if clamp is not None:
        if not jnp.issubdtype(clamp.dtype, jnp.integer):
            raise ValueError(f"clamp must be an integer array; got dtype {clamp.dtype}")
        if clamp.ndim != 1:
            raise ValueError(f"clamp must be 1-D; got shape {clamp.shape}")
        checkify.check(
            jnp.all((clamp >= 0) & (clamp < n_v)),
            f"clamp indices must be in [0, {n_v}); got clamp={{clamp}}", clamp=clamp,
        )
        sorted_clamp = jnp.sort(clamp)
        checkify.check(
            jnp.all(sorted_clamp[1:] != sorted_clamp[:-1]),
            "clamp must not contain duplicate indices; got clamp={clamp}", clamp=clamp,
        )


def RBM_chain(
    key, x_v, x_h, W, b_v, b_h, steps, n_samples=None, mode=None, spin=True, clamp=None
):
    """Sample from a restricted Boltzmann machine (block-conditional Gibbs).

    Unlike ``BM_chain``, the visible/hidden split is explicit (separate ``v``
    and ``h`` arguments) rather than auto-detected from a full ``(n, n)``
    matrix -- there is no ``concat(v, h)`` layout and no restricted-detection
    step, since a caller of ``RBM_chain`` has already committed to the
    bipartite structure by construction.

    Behavior
    --------
    1. **Validation**, via ``_rbm_validate``. ``W`` must be 2-D, shape
       ``(n_v, n_h)``. ``b_v`` must agree with it in size (``b_v.shape ==
       (n_v,)``), or be ``None``; likewise ``b_h`` against ``(n_h,)``. ``v``
       and ``h`` must be vectors of length ``n_v`` / ``n_h`` respectively,
       with entries in ``{-1, +1}`` if ``spin`` else ``{0, 1}`` -- same
       convention as ``BM_chain``. ``mode`` and ``n_samples`` must agree
       (see below). Bad inputs raise ``ValueError`` before any sampling
       happens.
    2. **Sampling**, via ``_sampler.py``'s ``_RBM_sampler``: alternately
       block-resamples the non-clamped visible units given the hidden block
       and then the whole hidden block given the visible block, ``steps``
       times per call -- either once (``n_samples is None``) or
       ``n_samples`` times, so ``steps`` is the number of update steps
       between the input state and the state(s) returned. If ``clamp`` is
       given, those visible units stay fixed at their value in ``v`` for
       the whole call; the hidden update is unaffected.
    3. **``mode`` / driver selection**, via ``_loop.py`` -- same modes as
       ``BM_chain``, but driving the ``(v, h)`` pair instead of a single
       state, and with ``'CORR'`` computing ``outer(v, h)`` (the actual
       sufficient statistic for RBM contrastive-divergence training) instead
       of the full ``outer(x, x)``:
       - ``n_samples is None`` (default; ``mode`` must also be ``None``): a
         single call to ``_for_loop``, returning just the final ``(v, h)``
         -- e.g. for burn-in, or to advance the chain between calls.
       - ``mode='HIST'`` (requires ``n_samples``): ``_scan`` (``jax.lax.scan``),
         stacking every resulting ``(v, h)`` (shapes ``(n_samples, n_v)`` /
         ``(n_samples, n_h)``).
       - ``mode='MEAN'`` (requires ``n_samples``): ``_for_loop_mean``, which
         accumulates the elementwise mean of the ``n_samples`` visited
         ``v`` / ``h`` without stacking a trajectory.
       - ``mode='CORR'`` (requires ``n_samples``): ``_for_loop_corr``, which
         accumulates the mean of ``outer(v, h)`` over the ``n_samples``
         visited states.

    Parameters
    ----------
    key:
        JAX PRNG key.
    v:
        Initial visible state, shape ``(n_v,)``.
    h:
        Initial hidden state, shape ``(n_h,)``.
    W:
        Visible-hidden coupling matrix, shape ``(n_v, n_h)``.
    b_v:
        Visible bias vector, shape ``(n_v,)``, or ``None`` for no bias.
    b_h:
        Hidden bias vector, shape ``(n_h,)``, or ``None`` for no bias.
    steps:
        Number of block-update steps to take between the input state and
        each returned state.
    n_samples:
        Number of samples to draw. ``None`` (default) means "just advance
        the chain and return the final state" -- no trajectory is collected,
        and ``mode`` must be ``None``. Otherwise a positive ``int``, and
        ``mode`` must be given (see above).
    mode:
        ``None`` (default, requires ``n_samples is None``), or one of
        ``'HIST'``, ``'MEAN'``, ``'CORR'`` (each requires ``n_samples``) --
        see above.
    spin:
        If ``True`` (default), units are ``{-1, +1}``-valued; if ``False``,
        ``{0, 1}``. ``v`` and ``h`` are validated against this convention.
    clamp:
        ``None``, or an optional integer array of indices into ``v``, i.e.
        ``[0, n_v)``, naming visible units to hold fixed at their value;
        ``h`` is always resampled normally.

    Returns
    -------
    The final ``(v, h)`` pair is always returned first; what else is
    returned alongside it depends on ``mode``:

    ==============  ==============  =================================
    ``n_samples``   ``mode``        Return value
    ==============  ==============  =================================
    ``None``        ``None``        ``(v, h)``
    given           ``'HIST'``      ``((v, h), (vs, hs))``
    given           ``'MEAN'``      ``((v, h), (v_mean, h_mean))``
    given           ``'CORR'``      ``((v, h), vh_mean)``
    ==============  ==============  =================================

    where ``vs.shape == (n_samples, n_v)`` / ``hs.shape == (n_samples, n_h)``
    is the full stacked trajectory, ``v_mean`` / ``h_mean`` are the
    elementwise averages of ``v`` / ``h`` over the ``n_samples`` draws, and
    ``vh_mean`` is the average of ``outer(v, h)``, shape ``(n_v, n_h)``.
    """
    _check_sampling_mode(mode, n_samples)
    err, _ = checkify.checkify(_rbm_validate)(x_v, x_h, W, b_v, b_h, spin, clamp)
    err.throw()

    sampler = _RBM_sampler(W, b_v, b_h, sampler_steps=steps, spin=spin, clamp=clamp)
    x = (x_v, x_h)

    if n_samples is None:
        return _for_loop(sampler, key, x, 1)
    if mode == "HIST":
        (_, x), xs = _scan(sampler, key, x, n_samples)
        return x, xs
    if mode == "MEAN":
        return _for_loop_mean(sampler, key, x, n_samples)
    return _for_loop_corr(sampler, key, x, n_samples, stat_fn=lambda xh: jnp.outer(xh[0], xh[1]))