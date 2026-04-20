"""Boltzmann machine models: fully-connected and restricted."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

def AbstractBoltzmannMachine(ABC):
    @abstractmethod
    def energy(self, x: jax.Array) -> jax.Array:
        pass

    @abstractmethod
    def update_state(self, key: jax.Array, x: jax.Array, free_units: jax.Array):
        pass

    @abstractmethod
    def update_params(self, W: jax.Array, b: jax.Array):
        pass

@partial(
    jax.tree_util.register_dataclass,
    data_fields=("W", "b"),
    meta_fields=("spin_style",),
)
@dataclass(frozen=True)
class BoltzmannMachine:
    """Fully-connected Boltzmann machine over `n` binary units in `{0, 1}`.

    Energy of a configuration `s` is

        E(s) = -1/2 * s^T W s - b^T s,

    where `W` is symmetric with zero diagonal and `b` is a bias vector.

    Parameters
    ----------
    W:
        Symmetric coupling matrix of shape `(n, n)` with zero diagonal.
    b:
        Bias vector of shape `(n,)`.
    """

    W: jax.Array
    b: jax.Array
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
        """Initialize a fully-connected BM with small symmetric Gaussian couplings."""
        W = jax.random.uniform(key, (n, n), minval=0.5 * minval, maxval=0.5 * maxval)

        W = W + W.T
        W -= jnp.diag(jnp.diagonal(W))

        b = None
        if bias:
            b = jax.random.uniform(key, (n,), minval=minval, maxval=maxval)

        if absnorm:
            abssum = jnp.sum(jnp.abs(W))

            if b:
                abssum += jnp.sum(jnp.abs(b))

            W = W / abssum

            if b:
                b = b / abssum

        return cls(W=W, b=b, spin_style=spin_style)

    @classmethod
    def init_from_matrix(
        cls, W: jax.Array, b: jax.Array, spin_style: bool = True
    ) -> "BoltzmannMachine":
        return cls(W=W, b=b, spin_style=spin_style)

    def energy(self, x: jax.Array) -> jax.Array:
        """Energy of configurations `s` of shape `(..., n)`."""
        if self.b:
            return -0.5 * x @ self.W @ x - self.b @ x
        else:
            return -0.5 * x @ self.W @ x

    def update_state(self, key: jax.Array, x: jax.Array, free_units: jax.Array):
        unit_key, bern_key, new_key = jax.random.split(key, 3)
        unit = jax.random.choice(unit_key, free_units, shape=())

        x = x.at[unit].set(0)
        delta_E = jnp.sum(jnp.dot(self.W.at[unit].get(), x))
        if self.b is not None:
            delta_E += self.b.at[unit].get()

        p = 1 / (1 + jnp.exp(-2 * delta_E))
        if self.spin_style:
            return new_key, x.at[unit].set(jax.random.bernoulli(bern_key, p=p) * 2 - 1)
        else:
            return new_key, x.at[unit].set(jax.random.bernoulli(bern_key, p=p))

    def update_params(self, W: jax.Array, b: jax.Array):
        return self.replace(W=W, b=b)

@partial(
    jax.tree_util.register_dataclass,
    data_fields=("W", "b_v", "b_h"),
    meta_fields=("spin_style",),
)
@dataclass(frozen=True)
class RestrictedBoltzmannMachine:
    """Bipartite (restricted) Boltzmann machine over visible and hidden binary units.

    Energy of a configuration `(v, h)` is

        E(v, h) = -v^T W h - vbias^T v - hbias^T h,

    where `W` couples only visible to hidden units (no intra-layer couplings).

    Parameters
    ----------
    W:
        Coupling matrix of shape `(n_visible, n_hidden)`.
    vbias:
        Visible bias of shape `(n_visible,)`. May be `None` for no bias.
    hbias:
        Hidden bias of shape `(n_hidden,)`. May be `None` for no bias.
    spin_style:
        If `True`, units take values in `{-1, +1}`; otherwise in `{0, 1}`.
    """

    W: jax.Array  # v x h matrix
    bias: bool = True
    n_visible: int
    b_v: jax.Array
    b_h: jax.Array
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
        W = jax.random.uniform(key, (n_visible, n_hidden), minval=minval, maxval=maxval)

        b_v = None
        b_h = None

        if bias:
            b_v = jax.random.uniform(key, (n_visible,), minval=minval, maxval=maxval)
            b_h = jax.random.uniform(key, (n_hidden,), minval=minval, maxval=maxval)

        if absnorm:
            abssum = jnp.sum(jnp.abs(W))

            if bias:
                abssum += jnp.sum(jnp.abs(b_v))
                abssum += jnp.sum(jnp.abs(b_h))

            W = W / abssum

            if bias:
                b_v = b_v / abssum
                b_h = b_h / abssum

        return cls(W=W, n_visible=n_visible, b_v=b_v, b_h=b_h, spin_style=spin_style)

    @classmethod
    def init_from_matrix(
        cls,
        W: jax.Array,
        vbias: jax.Array | None = None,
        hbias: jax.Array | None = None,
        spin_style: bool = True,
    ) -> "RestrictedBoltzmannMachine":
        """Build an RBM from an explicit coupling matrix and (optional) biases."""
        return cls(W=W, n_visible=W.shape[0], b_v=vbias, b_h=hbias, spin_style=spin_style)

    def energy(self, x) -> jax.Array:
        """Joint energy of visible/hidden configurations."""
        x_v = x[:self.n_visible]
        x_h = x[self.n_visible:]

        sum = - x_v @ self.W @ x_h

        if self.b_v:
            sum -= x_v @ self.b_v
        if self.b_h:
            sum -= x_h @ self.b_h

        return sum

    def update_visible(self, key: jax.Array, x: jax.Array):
        x_v = x[:self.n_visible]
        x_h = x[self.n_visible:]

        p = 1 / (1 + jnp.exp(- (self.b_v + self.W @ x_h)))
        x_v = jax.random.bernoulli(key, p=p)

        return jnp.concatenate([x_v, x_h])

    def update_hidden(self, key: jax.Array, x: jax.Array):
        x_v = x[:self.n_visible]
        x_h = x[self.n_visible:]

        p = 1 / (1 + jnp.exp(- (self.b_h + x_v @ self.W)))
        x_h = jax.random.bernoulli(key, p=p)

        return jnp.concatenate([x_v, x_h])

    def update_state(self, key: jax.Array, x, wake = False):
        if wake:
            hidden_key, new_key = jax.random.split(key, 2)
            return new_key, self.update_hidden(hidden_key, x)
        else:
            visible_key, hidden_key, new_key = jax.random.split(key, 3)
            x = self.update_visible(visible_key, x)
            x = self.update_hidden(hidden_key, x)
            return new_key, x

    def update_params(self, W, b):
        b_v = b[:self.n_visible]
        b_h = b[self.n_visible:]
        return self.replace(W=W, b_v=b_v, b_h=b_h)
