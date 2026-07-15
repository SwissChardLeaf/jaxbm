"""Array-in / array-out for Boltzmann machines and restricted Boltzmann machines.

Both ``BM_chain`` (fully-connected) and ``RBM_chain`` (restricted) are
the public surface of ``jaxbm``: they validate inputs using runtime JAX
checks, build the right sampler (see ``_sampler.py``) and select the right
JAX loop for optimal memory usage. Both allow for clamping and
configuration of spin.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.experimental import checkify

from ._loop import _bm_outer_self, _for_loop, _for_loop_stat, _rbm_outer_self, _scan
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
    it still just raises ``ValueError``.
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
    """Check that ``mode`` and ``n_samples`` agree with each other, and
    resolve ``mode``'s default.

    - ``n_samples=None`` means "no accumulation, just advance the chain and
      return the final state" -- ``mode`` must then also be ``None`` (there
      is nothing to summarize).
    - Otherwise, ``n_samples`` must be a positive ``int``, and ``mode``
      selects how to summarize the ``n_samples`` visited states: one of
      ``_MODES``, or ``None`` (default), which resolves to ``'HIST'``.

    Returns the resolved ``mode``. Raises ``ValueError`` on any violation.
    """
    if n_samples is None:
        if mode is not None:
            raise ValueError(f"mode must be None when n_samples is None; got mode={mode!r}")
        return None

    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int; got n_samples={n_samples!r}")
    if mode is None:
        mode = "HIST"
    if mode not in _MODES:
        raise ValueError(
            f"mode must be one of {_MODES} when n_samples is given; got {mode!r}"
        )
    return mode


def BM_chain(
    key, x, weights, bias, n_samples, steps=1, mode=None, spin=True, clamp=None, in_jit=False
):
    """Sample from a fully-connected Boltzmann machine (single-unit Gibbs).

    Behavior
    --------
    1. **Validation**, via ``_bm_validate``: ``weights`` must be square
       (``(n, n)``) and symmetric, and ``bias`` must agree with it in size
       (``bias.shape == (n,)``), or be ``None``. The initial state ``x``
       must be a vector of length ``n`` with entries in ``{-1, +1}`` if
       ``spin`` else ``{0, 1}``. ``clamp`` must be ``None`` or a 1-D integer
       array of *distinct* indices into ``[0, n)``. ``mode`` and
       ``n_samples`` must also agree (see below). Bad inputs raise
       ``ValueError`` before any sampling happens. Value-based checks require
       ``checkify.check`` so this function itself can be traced -- but the
       eager ``.throw()`` used to surface them still isn't safe under an
       *outer* ``jit``/``vmap`` (it needs a concrete answer to "did a check
       fail?", and there isn't one while still being traced). Passing
       ``in_jit=True`` skips validation entirely instead, which is what
       makes this function traceable; see ``in_jit`` below.
    2. **Sampling**, via ``_sampler.py``'s ``_BM_sampler``: resamples one
       (non-clamped) unit at a time from its conditional distribution given
       the rest of the state, ``steps`` times per call -- either once
       (``n_samples is None``) or ``n_samples`` times, so ``steps`` is the
       number of update steps between the input state and the state(s)
       returned. If ``clamp`` is given, those units are never chosen, so
       they stay fixed at their value in ``x`` for the whole call.
    3. **``mode`` / JAX loop selection**, via ``_loop.py``:
       - ``n_samples=None`` (``mode`` must also be ``None``): a single call
         to ``_for_loop``, returning just the final state -- e.g. for
         burn-in, or to advance the chain between calls.
       - ``mode='HIST'`` (requires ``n_samples``; also the default ``mode``
         whenever ``n_samples`` is given): ``_scan``, stacking every
         resulting state (shape ``(n_samples, n)``).
       - ``mode='MEAN'`` (requires ``n_samples``): ``_for_loop_stat`` with
         ``stat_fn=None``, which accumulates the elementwise mean of the
         ``n_samples`` visited states without stacking a trajectory.
       - ``mode='CORR'`` (requires ``n_samples``): ``_for_loop_stat`` with
         ``stat_fn=_bm_outer_self``, which accumulates the mean of
         ``outer(x, x)`` over the ``n_samples`` visited states (the
         sufficient statistic for the pairwise term of a BM's
         log-likelihood gradient).

       In all three cases, the ``n_samples`` visited states are the ones
       produced *by* the ``n_samples`` sampler calls -- the input ``x`` you
       passed in is never itself one of them (it's only ever used as the
       starting point for the first call).

    Static arguments under ``jax.jit``
    -----------------------------------
    ``n_samples``, ``steps``, ``mode``, ``spin``, and ``in_jit`` all control
    Python-level branching in this function (or downstream in
    ``_sampler.py`` / ``_loop.py``), so they must be passed as static
    arguments, e.g. ``jax.jit(BM_chain, static_argnames=("n_samples",
    "steps", "mode", "spin", "in_jit"))``. ``key``, ``x``, ``weights``,
    ``bias``, and ``clamp`` stay dynamic (traced) -- ``clamp``'s branching
    only ever depends on its *shape* (``None`` or not, how many indices),
    which is static even for a traced array, so it needs no special
    treatment. Also requires ``in_jit=True``, since the default eager
    validation isn't traceable (see ``in_jit`` below).

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
    n_samples:
        Number of samples to draw -- required, no default, so every call
        site makes an explicit choice. ``None`` means "just advance the
        chain and return the final state" -- no trajectory is collected,
        and ``mode`` must be ``None``. Otherwise a positive ``int``; ``mode``
        then defaults to ``'HIST'`` if not given (see above). Static under
        ``jax.jit``.
    steps:
        Number of chain-update steps to take between each sampled state.
        Defaults to ``1``. Static under ``jax.jit``.
    mode:
        ``None`` (default) -- requires ``n_samples is None`` (in which case
        there's nothing to summarize), *unless* ``n_samples`` is given, in
        which case ``None`` resolves to ``'HIST'``. Otherwise one of
        ``'HIST'``, ``'MEAN'``, ``'CORR'`` (each requires ``n_samples``) --
        see above. Static under ``jax.jit``.
    spin:
        If ``True`` (default), units are ``{-1, +1}``-valued; if ``False``,
        ``{0, 1}``. ``x`` is validated against this convention. Static
        under ``jax.jit``.
    clamp:
        ``None``, or an optional integer array of *distinct* indices into
        ``[0, n)`` naming units to hold fixed at their value in ``x``: those
        units are never chosen to be resampled.
    in_jit:
        If ``False`` (default), ``x``/``weights``/``bias``/``clamp`` are
        validated eagerly as described above. If ``True``, that validation
        is skipped entirely -- pass ``in_jit=True`` whenever ``jit`` is
        anywhere in the call stack (directly, e.g. ``jax.jit(BM_chain)``, or
        wrapping a ``vmap``, e.g. ``jax.jit(jax.vmap(BM_chain))``), and take
        responsibility for the inputs being valid yourself. A bare
        ``jax.vmap(BM_chain)`` (no enclosing ``jit``) does *not* need
        ``in_jit=True`` -- it still validates eagerly and raises correctly,
        since unlike ``jit``, plain ``vmap`` executes each op immediately on
        concrete (batched) values rather than staging the whole function
        into an abstract program first. Static under ``jax.jit``.

    Returns
    -------
    The final state ``x`` (shape ``(n,)``) is always returned; what else is
    returned alongside it depends on ``mode``. Note that the input ``x`` is
    never itself included in ``xs`` / ``x_mean`` / ``outer_mean`` below --
    those only ever summarize the ``n_samples`` states *produced by*
    sampling, not the state you started from:

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
    mode = _check_sampling_mode(mode, n_samples)
    if not in_jit:
        err, _ = checkify.checkify(_bm_validate)(x, weights, bias, spin, clamp)
        err.throw()

    sampler = _BM_sampler(weights, bias, sampler_steps=steps, spin=spin, clamp=clamp)

    if n_samples is None:
        return _for_loop(sampler, key, x, 1)
    if mode == "HIST":
        (_, x), xs = _scan(sampler, key, x, n_samples)
        return x, xs
    if mode == "MEAN":
        return _for_loop_stat(sampler, key, x, n_samples)
    return _for_loop_stat(sampler, key, x, n_samples, stat_fn=_bm_outer_self)


def _rbm_validate(x_v, x_h, weights, bias_v, bias_h, spin, clamp):
    """Validate all of ``RBM_chain``'s array inputs together.

    - ``weights`` must be 2-D, shape ``(n_v, n_h)``.
    - ``bias_v`` must be ``None`` or a vector of shape ``(n_v,)``; likewise
      ``bias_h`` against ``(n_h,)``.
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
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2-D (n_v, n_h); got shape {weights.shape}")
    n_v, n_h = weights.shape

    if bias_v is not None and bias_v.shape != (n_v,):
        raise ValueError(
            f"bias_v must have shape ({n_v},) matching weights; got bias_v.shape={bias_v.shape}"
        )
    if bias_h is not None and bias_h.shape != (n_h,):
        raise ValueError(
            f"bias_h must have shape ({n_h},) matching weights; got bias_h.shape={bias_h.shape}"
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
    key, x_v, x_h, weights, bias_v, bias_h, n_samples,
    steps=1, mode=None, spin=True, clamp=None, in_jit=False,
):
    """Sample from a restricted Boltzmann machine (block-conditional Gibbs).

    Unlike ``BM_chain``, the visible and hidden units are always kept as two
    separate arrays (``x_v``, ``x_h``) rather than one combined state.

    Behavior
    --------
    1. **Validation**, via ``_rbm_validate``. ``weights`` must be 2-D, shape
       ``(n_v, n_h)``. ``bias_v`` must agree with it in size
       (``bias_v.shape == (n_v,)``), or be ``None``; likewise ``bias_h``
       against ``(n_h,)``. ``x_v`` and ``x_h`` must be vectors of length
       ``n_v`` / ``n_h`` respectively, with entries in ``{-1, +1}`` if
       ``spin`` else ``{0, 1}``. ``clamp`` must be ``None`` or a 1-D integer
       array of *distinct* indices into ``[0, n_v)``. ``mode`` and
       ``n_samples`` must agree (see below). Bad inputs raise ``ValueError``
       before any sampling happens -- unless ``in_jit=True``, which skips
       validation entirely so this function can be traced (see ``in_jit``
       below; same caveat as ``BM_chain``'s).
    2. **Sampling**, via ``_sampler.py``'s ``_RBM_sampler``: alternately
       block-resamples the non-clamped visible units given the hidden block
       and then the whole hidden block given the visible block, ``steps``
       times per call -- either once (``n_samples is None``) or
       ``n_samples`` times, so ``steps`` is the number of update steps
       between the input state and the state(s) returned. If ``clamp`` is
       given, those visible units stay fixed at their value in ``x_v`` for
       the whole call.
    3. **``mode`` / JAX loop selection**, via ``_loop.py``: operating on the
       ``(x_v, x_h)`` pair, with ``'CORR'`` computing ``outer(x_v, x_h)``:
       - ``n_samples=None`` (``mode`` must also be ``None``): a single call
         to ``_for_loop``, returning just the final ``(x_v, x_h)`` -- e.g.
         for burn-in, or to advance the chain between calls.
       - ``mode='HIST'`` (requires ``n_samples``; also the default ``mode``
         whenever ``n_samples`` is given): ``_scan``, stacking every
         resulting ``(x_v, x_h)`` (shapes ``(n_samples, n_v)`` /
         ``(n_samples, n_h)``).
       - ``mode='MEAN'`` (requires ``n_samples``): ``_for_loop_stat`` with
         ``stat_fn=None``, which accumulates the elementwise mean of the
         ``n_samples`` visited ``x_v`` / ``x_h`` without stacking a
         trajectory.
       - ``mode='CORR'`` (requires ``n_samples``): ``_for_loop_stat`` with
         ``stat_fn=_rbm_outer_self``, which accumulates the mean of
         ``outer(x_v, x_h)`` over the ``n_samples`` visited states.

       In all three cases, the ``n_samples`` visited states are the ones
       produced *by* the ``n_samples`` sampler calls -- the input
       ``(x_v, x_h)`` you passed in is never itself one of them (it's only
       ever used as the starting point for the first call).

    Static arguments under ``jax.jit``
    -----------------------------------
    ``n_samples``, ``steps``, ``mode``, ``spin``, and ``in_jit`` all control
    Python-level branching in this function (or downstream in
    ``_sampler.py`` / ``_loop.py``), so they must be passed as static
    arguments, e.g. ``jax.jit(RBM_chain, static_argnames=("n_samples",
    "steps", "mode", "spin", "in_jit"))``. ``key``, ``x_v``, ``x_h``,
    ``weights``, ``bias_v``, ``bias_h``, and ``clamp`` stay dynamic (traced)
    -- ``clamp``'s branching only ever depends on its *shape* (``None`` or
    not, how many indices), which is static even for a traced array, so it
    needs no special treatment. Also requires ``in_jit=True``, since the
    default eager validation isn't traceable (see ``in_jit`` below).

    Parameters
    ----------
    key:
        JAX PRNG key.
    x_v:
        Initial visible state, shape ``(n_v,)``.
    x_h:
        Initial hidden state, shape ``(n_h,)``.
    weights:
        Visible-hidden coupling matrix, shape ``(n_v, n_h)``.
    bias_v:
        Visible bias vector, shape ``(n_v,)``, or ``None`` for no bias.
    bias_h:
        Hidden bias vector, shape ``(n_h,)``, or ``None`` for no bias.
    n_samples:
        Number of samples to draw -- required, no default, so every call
        site makes an explicit choice. ``None`` means "just advance the
        chain and return the final state" -- no trajectory is collected,
        and ``mode`` must be ``None``. Otherwise a positive ``int``; ``mode``
        then defaults to ``'HIST'`` if not given (see above). Static under
        ``jax.jit``.
    steps:
        Number of block-update steps to take between the input state and
        each returned state. Defaults to ``1``. Static under ``jax.jit``.
    mode:
        ``None`` (default) -- requires ``n_samples is None`` (in which case
        there's nothing to summarize), *unless* ``n_samples`` is given, in
        which case ``None`` resolves to ``'HIST'``. Otherwise one of
        ``'HIST'``, ``'MEAN'``, ``'CORR'`` (each requires ``n_samples``) --
        see above. Static under ``jax.jit``.
    spin:
        If ``True`` (default), units are ``{-1, +1}``-valued; if ``False``,
        ``{0, 1}``. ``x_v`` and ``x_h`` are validated against this convention.
        Static under ``jax.jit``.
    clamp:
        ``None``, or an optional integer array of *distinct* indices into
        ``[0, n_v)``, naming visible units to hold fixed at their value in ``x_v`` for
        the whole call.
    in_jit:
        If ``False`` (default), inputs are validated eagerly as described
        above. If ``True``, that validation is skipped entirely -- pass
        ``in_jit=True`` whenever ``jit`` is anywhere in the call stack
        (directly, e.g. ``jax.jit(RBM_chain)``, or wrapping a ``vmap``, e.g.
        ``jax.jit(jax.vmap(RBM_chain))``), and take responsibility for the
        inputs being valid yourself. A bare ``jax.vmap(RBM_chain)`` (no
        enclosing ``jit``) does *not* need ``in_jit=True`` -- it still
        validates eagerly and raises correctly, since unlike ``jit``, plain
        ``vmap`` executes each op immediately on concrete (batched) values
        rather than staging the whole function into an abstract program
        first. Static under ``jax.jit``.

    Returns
    -------
    The final ``(x_v, x_h)`` pair is always returned first; what else is
    returned alongside it depends on ``mode``. Note that the input
    ``(x_v, x_h)`` is never itself included in ``(xs_v, xs_h)`` /
    ``(x_v_mean, x_h_mean)`` / ``outer_mean`` below -- those only ever
    summarize the ``n_samples`` states *produced by* sampling, not the state
    you started from:

    ==============  ==============  =================================
    ``n_samples``   ``mode``        Return value
    ==============  ==============  =================================
    ``None``        ``None``        ``(x_v, x_h)``
    given           ``'HIST'``      ``((x_v, x_h), (xs_v, xs_h))``
    given           ``'MEAN'``      ``((x_v, x_h), (x_v_mean, x_h_mean))``
    given           ``'CORR'``      ``((x_v, x_h), outer_mean)``
    ==============  ==============  =================================

    where ``xs_v.shape == (n_samples, n_v)`` and ``xs_h.shape == (n_samples, n_h)``
    is the full stacked trajectory, ``x_v_mean`` / ``x_h_mean`` are the
    elementwise averages of ``x_v`` / ``x_h`` over the ``n_samples`` draws, and
    ``outer_mean`` is the average of ``outer(x_v, x_h)``, shape ``(n_v, n_h)``.
    """
    mode = _check_sampling_mode(mode, n_samples)
    if not in_jit:
        err, _ = checkify.checkify(_rbm_validate)(x_v, x_h, weights, bias_v, bias_h, spin, clamp)
        err.throw()

    sampler = _RBM_sampler(weights, bias_v, bias_h, sampler_steps=steps, spin=spin, clamp=clamp)
    x = (x_v, x_h)

    if n_samples is None:
        return _for_loop(sampler, key, x, 1)
    if mode == "HIST":
        (_, x), xs = _scan(sampler, key, x, n_samples)
        return x, xs
    if mode == "MEAN":
        return _for_loop_stat(sampler, key, x, n_samples)
    return _for_loop_stat(sampler, key, x, n_samples, stat_fn=_rbm_outer_self)