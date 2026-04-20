"""MCMC convergence diagnostics for Boltzmann-machine samplers.

Currently exposes:

- ``compute_rhat``: rank-normalized split-R-hat (Vehtari et al., 2021),
  which is the modern Stan/ArviZ default. Works with one or more chains.
- ``compute_ess``: split effective sample size with bulk / tail / mean
  variants, using Geyer's initial-positive-sequence estimator on the
  multi-chain autocorrelation function.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm


def _split_chains(samples: jax.Array) -> jax.Array:
    """Split each chain in half: ``(M, N, ...) -> (2M, N // 2, ...)``.

    The trailing odd sample (when ``N`` is odd) is dropped.
    """
    n_samples = samples.shape[1]
    half = n_samples // 2
    first = samples[:, :half]
    second = samples[:, half : 2 * half]
    return jnp.concatenate([first, second], axis=0)


def _ordinal_ranks(x: jax.Array) -> jax.Array:
    """1-based ordinal ranks along the leading axis.

    Ties are broken by index order (the same convention as
    ``scipy.stats.rankdata(method='ordinal')``). For continuous samples
    with no ties this matches average ranks exactly. If you feed in heavily
    tied data (e.g. raw binary spins) you should pre-aggregate to a
    real-valued observable first; otherwise the diagnostic loses power.
    """
    sort_idx = jnp.argsort(x, axis=0)
    inv = jnp.argsort(sort_idx, axis=0)
    return inv.astype(jnp.float32) + 1.0


def _rank_normalize(samples: jax.Array) -> jax.Array:
    """Rank-normalize jointly over chains and draws, per parameter.

    Implements the Blom transformation: ranks are mapped through
    ``Φ⁻¹((r − 3/8) / (n − 1/4))``, which yields approximately N(0, 1)
    samples under the null of identically-distributed draws.
    """
    n_chains, n_samples = samples.shape[:2]
    n = n_chains * n_samples
    flat = samples.reshape((n,) + samples.shape[2:])
    ranks = _ordinal_ranks(flat)
    z = jnorm.ppf((ranks - 3.0 / 8.0) / (n - 1.0 / 4.0))
    return z.reshape(samples.shape)


def _rhat_plain(samples: jax.Array) -> jax.Array:
    """Classical Gelman–Rubin R-hat on shape ``(n_chains, n_samples, ...)``.

    Returns an array of shape ``samples.shape[2:]``.
    """
    n_samples = samples.shape[1]
    chain_mean = samples.mean(axis=1)
    chain_var = samples.var(axis=1, ddof=1)
    within = chain_var.mean(axis=0)
    between = n_samples * chain_mean.var(axis=0, ddof=1)
    var_hat = (n_samples - 1) / n_samples * within + between / n_samples
    return jnp.sqrt(var_hat / within)


def compute_rhat(samples: jax.Array) -> jax.Array:
    """Rank-normalized split-R-hat (Vehtari et al., 2021).

    Splits each chain in half, then returns the maximum of:

    - **bulk-R-hat**: R-hat on rank-normalized samples (sensitive to drift
      in the body of the distribution).
    - **tail-R-hat**: R-hat on rank-normalized samples folded around the
      median (sensitive to drift in the scale / tails).

    Parameters
    ----------
    samples
        Array of shape ``(n_chains, n_samples, *param_shape)``. Each
        chain should be at length-2 or longer; chains will be split into
        halves before the diagnostic is applied.

    Returns
    -------
    jax.Array
        Array of shape ``param_shape`` with one R-hat per scalar parameter.
        Values close to 1.0 (e.g. < 1.01) indicate the chains have mixed;
        values noticeably above 1 indicate non-convergence.

    Notes
    -----
    With ``n_chains == 1`` this still returns a meaningful split-R-hat, but
    it can only diagnose **within-chain** non-stationarity. It cannot
    detect mode-trapping: if the true distribution is multimodal and the
    one chain is stuck in a single mode, this R-hat will read ≈ 1 and
    silently give a false sense of convergence. For Boltzmann machines —
    which are routinely multimodal — running multiple chains from
    overdispersed initial states (cheap with ``jax.vmap``) is strongly
    recommended.

    The implementation follows the rank-normalized split-R-hat described in

        Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021),
        "Rank-Normalization, Folding, and Localization: An Improved R-hat
        for Assessing Convergence of MCMC", *Bayesian Analysis*.
    """
    if samples.ndim < 2:
        raise ValueError(
            "compute_rhat expects samples of shape (n_chains, n_samples, ...); "
            f"got shape {samples.shape}."
        )

    split = _split_chains(samples)

    z_bulk = _rank_normalize(split)
    rhat_bulk = _rhat_plain(z_bulk)

    median = jnp.median(split.reshape((-1,) + split.shape[2:]), axis=0)
    folded = jnp.abs(split - median)
    z_tail = _rank_normalize(folded)
    rhat_tail = _rhat_plain(z_tail)

    return jnp.maximum(rhat_bulk, rhat_tail)


def _autocov_fft(x: jax.Array, axis: int = 0) -> jax.Array:
    """Biased sample autocovariance along ``axis`` via FFT.

    Returns an array of the same shape as ``x``; the value at index ``t``
    along ``axis`` is the lag-``t`` autocovariance estimated as
    ``(1/n) * sum_{i=0}^{n-t-1} (x_i - x_bar)(x_{i+t} - x_bar)``.
    """
    n = x.shape[axis]
    centered = x - x.mean(axis=axis, keepdims=True)
    n_pad = 1 << ((2 * n - 1).bit_length())  # next power of two >= 2n
    pad_widths = [(0, 0)] * x.ndim
    pad_widths[axis] = (0, n_pad - n)
    padded = jnp.pad(centered, pad_widths)
    f = jnp.fft.fft(padded, axis=axis)
    acov_full = jnp.fft.ifft(f * jnp.conj(f), axis=axis).real
    acov = jnp.take(acov_full, jnp.arange(n), axis=axis)
    return acov / n


def _ess_ips(samples: jax.Array) -> jax.Array:
    """Geyer's initial-positive-sequence ESS for shape ``(M, N, ...)``.

    Uses the multi-chain autocorrelation estimate from Vehtari et al. (2021):

        rho_hat_t = 1 - (W - mean_m s_{m,t}) / var_plus,

    where ``s_{m,t}`` is the lag-``t`` autocovariance of chain ``m``,
    ``W`` is the mean within-chain variance, and ``var_plus`` is the
    R-hat-style overestimate ``(N-1)/N * W + B/N``. Lags are paired
    via ``Gamma_k = rho_hat_{2k} + rho_hat_{2k+1}`` and summed up to
    the first non-positive ``Gamma_k`` (initial-positive-sequence).
    """
    n_chains, n_samples = samples.shape[:2]

    acov = _autocov_fft(samples, axis=1)  # (M, N, ...)
    mean_acov = acov.mean(axis=0)  # (N, ...)

    chain_var = samples.var(axis=1, ddof=1)  # (M, ...)
    within = chain_var.mean(axis=0)  # (...)
    if n_chains > 1:
        chain_mean = samples.mean(axis=1)
        between = n_samples * chain_mean.var(axis=0, ddof=1)
    else:
        between = jnp.zeros_like(within)
    var_plus = (n_samples - 1) / n_samples * within + between / n_samples

    rho_hat = 1.0 - (within - mean_acov) / var_plus  # (N, ...)
    rho_hat = rho_hat.at[0].set(1.0)

    n_pairs = n_samples // 2
    even = rho_hat[0 : 2 * n_pairs : 2]
    odd = rho_hat[1 : 2 * n_pairs : 2]
    gamma = even + odd  # (n_pairs, ...)

    positive_run = jnp.cumprod((gamma > 0).astype(gamma.dtype), axis=0)
    tau = -1.0 + 2.0 * jnp.sum(gamma * positive_run, axis=0)
    tau = jnp.maximum(tau, 1.0)  # guarantees ESS <= total samples

    return n_chains * n_samples / tau


def _bulk_ess(samples: jax.Array) -> jax.Array:
    """Bulk-ESS: ESS on rank-normalized samples."""
    return _ess_ips(_rank_normalize(samples))


def _tail_ess(samples: jax.Array) -> jax.Array:
    """Tail-ESS: min of ESS at the empirical 5% and 95% quantile indicators."""
    flat = samples.reshape((-1,) + samples.shape[2:])
    q05 = jnp.quantile(flat, 0.05, axis=0)
    q95 = jnp.quantile(flat, 0.95, axis=0)
    lower = (samples <= q05).astype(jnp.float32)
    upper = (samples >= q95).astype(jnp.float32)
    return jnp.minimum(_ess_ips(lower), _ess_ips(upper))


def compute_ess(samples: jax.Array, kind: str = "bulk") -> jax.Array:
    """Split effective sample size (Vehtari et al., 2021).

    Each chain is split in half before the diagnostic is applied, matching
    :func:`compute_rhat`. The autocorrelation function is estimated by
    pooling autocovariances across chains (Vehtari et al. 2021 Eq. 10),
    and the variance sum is truncated using Geyer's initial-positive-
    sequence rule.

    Parameters
    ----------
    samples
        Array of shape ``(n_chains, n_samples, *param_shape)``.
    kind
        Which ESS variant to compute:

        - ``"bulk"`` *(default)*: ESS on rank-normalized samples, which
          measures how well the body of the distribution is sampled and
          is the right number to compare against the usual "ESS > 400"
          rule of thumb.
        - ``"tail"``: minimum of the ESSs of the indicators
          ``I[x <= q_05]`` and ``I[x >= q_95]``, measuring how well
          the tails are sampled.
        - ``"mean"``: ESS computed directly on the raw samples (no rank
          transform). Appropriate when you specifically want the
          variance of the sample mean.

    Returns
    -------
    jax.Array
        Array of shape ``param_shape`` with one ESS per scalar parameter.

    Notes
    -----
    Single-chain ESS is well-defined and meaningful (unlike single-chain
    R-hat), but it shares R-hat's blind spot for **mode-trapping**: a
    chain stuck in one basin of a multimodal distribution will report
    high ESS regardless of how badly it is missing the rest of the
    target. For Boltzmann machines, run multiple ``vmap``-parallel
    chains from overdispersed initial states.

    References
    ----------
    Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021),
    "Rank-Normalization, Folding, and Localization: An Improved R-hat
    for Assessing Convergence of MCMC", *Bayesian Analysis*.

    Geyer (1992), "Practical Markov Chain Monte Carlo",
    *Statistical Science* 7(4): 473-483.
    """
    if samples.ndim < 2:
        raise ValueError(
            "compute_ess expects samples of shape (n_chains, n_samples, ...); "
            f"got shape {samples.shape}."
        )

    split = _split_chains(samples)

    if kind == "bulk":
        return _bulk_ess(split)
    if kind == "tail":
        return _tail_ess(split)
    if kind == "mean":
        return _ess_ips(split)
    raise ValueError(f"Unknown ESS kind {kind!r}; expected 'bulk', 'tail', or 'mean'.")
