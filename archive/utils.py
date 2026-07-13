"""Miscellaneous utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sigmoid(x: jax.Array) -> jax.Array:
    """Numerically stable sigmoid."""
    return jax.nn.sigmoid(x)


def logit(p: jax.Array, eps: float = 1e-7) -> jax.Array:
    """Inverse of `sigmoid`, with a small clamp for numerical safety."""
    p = jnp.clip(p, eps, 1.0 - eps)
    return jnp.log(p) - jnp.log1p(-p)
