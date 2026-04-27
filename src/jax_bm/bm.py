"""Boltzmann machine models: fully-connected and restricted."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


class AbstractBoltzmannMachine(ABC):
    """Common interface for Boltzmann-machine models in :mod:`jax_bm`.

    Concrete subclasses must implement an ``energy`` function, an MCMC
    ``update_state`` step, and an ``update_params`` helper that returns a
    new instance with replaced parameters.
    """

    @abstractmethod
    def energy(self, x: jax.Array) -> jax.Array:
        """Return the (scalar) energy of a configuration ``x``."""

    @abstractmethod
    def update_state(self, key: jax.Array, x: jax.Array, *args, **kwargs):
        """Run one MCMC update step and return ``(new_key, new_x)``."""

    @abstractmethod
    def update_params(self, W: jax.Array, b: jax.Array) -> "AbstractBoltzmannMachine":
        """Return a new instance with replaced parameters."""


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("W", "b"),
    meta_fields=("bias", "spin_style"),
)
@dataclass(frozen=True)
class BoltzmannMachine(AbstractBoltzmannMachine):
    """Fully-connected Boltzmann machine over ``n`` binary units.

    Energy of a configuration ``s`` is

        E(s) = -1/2 * s^T W s - b^T s,

    where ``W`` is symmetric with zero diagonal and ``b`` is a bias
    vector. Units take values in ``{-1, +1}`` when ``spin_style=True``
    (default) and in ``{0, 1}`` otherwise.

    Parameters
    ----------
    W:
        Symmetric coupling matrix of shape ``(n, n)`` with zero diagonal.
    b:
        Bias vector of shape ``(n,)``, or ``None`` when ``bias=False``.
    bias:
        Static flag indicating whether ``b`` is used. Stored separately
        from ``b`` so that ``jit``-compiled methods can dispatch on it
        without inspecting a possibly-``None`` array.
    spin_style:
        If ``True``, units are in ``{-1, +1}``; otherwise in ``{0, 1}``.
    """

    W: jax.Array
    b: jax.Array | None
    bias: bool = True
    spin_style: bool = True

    @classmethod
    def init_random(
        cls,
        key: jax.Array,
        n: int,
        absnorm: bool = False,
        bias: bool = True,
        minval: float = -1.0,
        maxval: float = 1.0,
        spin_style: bool = True,
    ) -> "BoltzmannMachine":
        """Initialize a fully-connected BM with uniformly-random symmetric couplings."""
        kW, kb = jax.random.split(key)
        W = jax.random.uniform(kW, (n, n), minval=0.5 * minval, maxval=0.5 * maxval)
        W = W + W.T
        W = W - jnp.diag(jnp.diagonal(W))

        b: jax.Array | None = None
        if bias:
            b = jax.random.uniform(kb, (n,), minval=minval, maxval=maxval)

        if absnorm:
            abssum = jnp.sum(jnp.abs(W))
            if bias:
                abssum = abssum + jnp.sum(jnp.abs(b))
            W = W / abssum
            if bias:
                b = b / abssum

        return cls(W=W, b=b, bias=bias, spin_style=spin_style)

    @classmethod
    def init_from_matrix(
        cls,
        W: jax.Array,
        b: jax.Array | None = None,
        spin_style: bool = True,
    ) -> "BoltzmannMachine":
        """Build a BM from an explicit coupling matrix and (optional) bias vector."""
        return cls(W=W, b=b, bias=b is not None, spin_style=spin_style)

    def energy(self, x: jax.Array) -> jax.Array:
        """Energy of a configuration ``x`` of shape ``(n,)``."""
        e = -0.5 * x @ self.W @ x
        if self.bias:
            e = e - self.b @ x
        return e

    def update_state(
        self,
        key: jax.Array,
        x: jax.Array,
        free_units: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """One Gibbs update of a randomly chosen unit drawn from ``free_units``.

        Returns ``(new_key, new_x)``. The unit's new value is sampled from
        its conditional distribution given the rest of the configuration.
        """
        unit_key, bern_key, new_key = jax.random.split(key, 3)
        unit = jax.random.choice(unit_key, free_units, shape=())

        x_zeroed = x.at[unit].set(0)
        field = jnp.dot(self.W[unit], x_zeroed)
        if self.bias:
            field = field + self.b[unit]

        p = 1.0 / (1.0 + jnp.exp(-2.0 * field))
        sampled = jax.random.bernoulli(bern_key, p=p)
        if self.spin_style:
            new_val = sampled.astype(x.dtype) * 2 - 1
        else:
            new_val = sampled.astype(x.dtype)
        return new_key, x.at[unit].set(new_val)

    def update_params(self, W: jax.Array, b: jax.Array) -> "BoltzmannMachine":
        """Return a new BM with replaced ``W`` and ``b`` (other fields preserved)."""
        return dataclasses.replace(self, W=W, b=b)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("W", "b_v", "b_h"),
    meta_fields=("n_visible", "bias", "spin_style"),
)
@dataclass(frozen=True)
class RestrictedBoltzmannMachine(AbstractBoltzmannMachine):
    """Bipartite (restricted) Boltzmann machine over visible and hidden units.

    Energy of a configuration ``(v, h)`` is

        E(v, h) = -v^T W h - b_v^T v - b_h^T h,

    where ``W`` couples only visible to hidden units (no intra-layer
    couplings). The state vector passed to ``energy`` and ``update_*`` is
    laid out as ``concat(v, h)`` with ``v`` of length ``n_visible``.

    Parameters
    ----------
    W:
        Coupling matrix of shape ``(n_visible, n_hidden)``.
    b_v:
        Visible bias of shape ``(n_visible,)``, or ``None``.
    b_h:
        Hidden bias of shape ``(n_hidden,)``, or ``None``.
    n_visible:
        Number of visible units. Stored as static metadata so it is
        available inside ``jit``-compiled code.
    bias:
        Static flag indicating whether biases are used.
    spin_style:
        Currently only affects the energy convention; the conditional
        update samplers operate in ``{0, 1}`` regardless.
    """

    W: jax.Array
    b_v: jax.Array | None
    b_h: jax.Array | None
    n_visible: int
    bias: bool = True
    spin_style: bool = True

    @classmethod
    def init_random(
        cls,
        key: jax.Array,
        n_visible: int,
        n_hidden: int,
        absnorm: bool = False,
        bias: bool = True,
        minval: float = -1.0,
        maxval: float = 1.0,
        spin_style: bool = True,
    ) -> "RestrictedBoltzmannMachine":
        """Initialize an RBM with uniformly-random couplings (and optional biases)."""
        kW, kbv, kbh = jax.random.split(key, 3)
        W = jax.random.uniform(kW, (n_visible, n_hidden), minval=minval, maxval=maxval)

        b_v: jax.Array | None = None
        b_h: jax.Array | None = None
        if bias:
            b_v = jax.random.uniform(kbv, (n_visible,), minval=minval, maxval=maxval)
            b_h = jax.random.uniform(kbh, (n_hidden,), minval=minval, maxval=maxval)

        if absnorm:
            abssum = jnp.sum(jnp.abs(W))
            if bias:
                abssum = abssum + jnp.sum(jnp.abs(b_v)) + jnp.sum(jnp.abs(b_h))
            W = W / abssum
            if bias:
                b_v = b_v / abssum
                b_h = b_h / abssum

        return cls(
            W=W,
            b_v=b_v,
            b_h=b_h,
            n_visible=n_visible,
            bias=bias,
            spin_style=spin_style,
        )

    @classmethod
    def init_from_matrix(
        cls,
        W: jax.Array,
        vbias: jax.Array | None = None,
        hbias: jax.Array | None = None,
        spin_style: bool = True,
    ) -> "RestrictedBoltzmannMachine":
        """Build an RBM from an explicit coupling matrix and (optional) biases."""
        bias_flag = (vbias is not None) and (hbias is not None)
        return cls(
            W=W,
            b_v=vbias,
            b_h=hbias,
            n_visible=W.shape[0],
            bias=bias_flag,
            spin_style=spin_style,
        )

    def energy(self, x: jax.Array) -> jax.Array:
        """Joint energy of a state ``x = concat(v, h)`` of length ``n_visible + n_hidden``."""
        x_v = x[: self.n_visible]
        x_h = x[self.n_visible :]
        e = -x_v @ self.W @ x_h
        if self.bias:
            e = e - x_v @ self.b_v - x_h @ self.b_h
        return e

    def update_visible(self, key: jax.Array, x: jax.Array) -> jax.Array:
        """Block-resample the visible units given the hidden configuration."""
        x_h = x[self.n_visible :]
        field = self.W @ x_h
        if self.bias:
            field = field + self.b_v
        p = 1.0 / (1.0 + jnp.exp(-field))
        x_v_new = jax.random.bernoulli(key, p=p).astype(x.dtype)
        return jnp.concatenate([x_v_new, x_h])

    def update_hidden(self, key: jax.Array, x: jax.Array) -> jax.Array:
        """Block-resample the hidden units given the visible configuration."""
        x_v = x[: self.n_visible]
        field = x_v @ self.W
        if self.bias:
            field = field + self.b_h
        p = 1.0 / (1.0 + jnp.exp(-field))
        x_h_new = jax.random.bernoulli(key, p=p).astype(x.dtype)
        return jnp.concatenate([x_v, x_h_new])

    def update_state(
        self,
        key: jax.Array,
        x: jax.Array,
        wake: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Block-Gibbs update.

        With ``wake=False`` (default) resamples ``v`` then ``h``. With
        ``wake=True`` resamples only ``h`` (a "wake-phase" half-step,
        useful in CD-style training).
        """
        if wake:
            hidden_key, new_key = jax.random.split(key, 2)
            return new_key, self.update_hidden(hidden_key, x)
        visible_key, hidden_key, new_key = jax.random.split(key, 3)
        x = self.update_visible(visible_key, x)
        x = self.update_hidden(hidden_key, x)
        return new_key, x

    def update_params(self, W: jax.Array, b: jax.Array) -> "RestrictedBoltzmannMachine":
        """Return a new RBM with replaced ``W`` and stacked bias ``b = concat(b_v, b_h)``."""
        b_v = b[: self.n_visible]
        b_h = b[self.n_visible :]
        return dataclasses.replace(self, W=W, b_v=b_v, b_h=b_h)
