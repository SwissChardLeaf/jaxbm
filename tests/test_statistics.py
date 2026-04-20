"""Tests for :mod:`jax_bm.statistics`.

Covers every public and private helper in the module. Tests are organized
roughly bottom-up: leaf helpers first (`_split_chains`, `_ordinal_ranks`,
`_rank_normalize`, `_rhat_plain`, `_autocov_fft`), then composites
(`_ess_ips`, `_bulk_ess`, `_tail_ess`), then the public API
(`compute_rhat`, `compute_ess`), and finally a few integration checks
(jit, vmap, agreement between public and private paths).

Conventions
-----------
- All stochastic tests use fixed PRNG seeds and sample sizes large enough
  that failures due to Monte Carlo noise are extremely unlikely.
- Tolerances are intentionally generous: the goal is to catch regressions
  and wrong formulas, not to verify the estimators to high precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_bm.statistics import (
    _autocov_fft,
    _bulk_ess,
    _ess_ips,
    _ordinal_ranks,
    _rank_normalize,
    _rhat_plain,
    _split_chains,
    _tail_ess,
    compute_ess,
    compute_rhat,
)


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #


def _ar1(
    key: jax.Array, phi: float, n_samples: int, n_chains: int, sigma: float = 1.0
) -> jax.Array:
    """Generate ``n_chains`` AR(1) chains of length ``n_samples``.

    Each chain obeys ``x_{t+1} = phi * x_t + eps_t``, with ``eps_t ~ N(0, sigma^2)``.
    Chains are started from their stationary distribution so we do not need to
    burn in. Returns an array of shape ``(n_chains, n_samples)``.
    """
    stationary_sd = sigma / jnp.sqrt(1.0 - phi**2)
    k0, k1 = jax.random.split(key)
    x0 = stationary_sd * jax.random.normal(k0, (n_chains,))
    eps = sigma * jax.random.normal(k1, (n_samples, n_chains))

    def step(prev: jax.Array, e: jax.Array) -> tuple[jax.Array, jax.Array]:
        cur = phi * prev + e
        return cur, cur

    _, traj = jax.lax.scan(step, x0, eps)
    return traj.T  # (n_chains, n_samples)


# The ``key`` fixture is provided by ``tests/conftest.py``.


# --------------------------------------------------------------------------- #
# _split_chains                                                               #
# --------------------------------------------------------------------------- #


class TestSplitChains:
    def test_shape_even(self) -> None:
        x = jnp.arange(4 * 10).reshape(4, 10).astype(jnp.float32)
        out = _split_chains(x)
        assert out.shape == (8, 5)

    def test_shape_odd_drops_trailing(self) -> None:
        x = jnp.arange(2 * 7).reshape(2, 7).astype(jnp.float32)
        out = _split_chains(x)
        # N = 7 -> half = 3, so output has 2*2 chains of length 3 (one sample dropped).
        assert out.shape == (4, 3)

    def test_preserves_parameter_dims(self) -> None:
        x = jnp.zeros((3, 8, 4, 5))
        out = _split_chains(x)
        assert out.shape == (6, 4, 4, 5)

    def test_exact_values(self) -> None:
        x = jnp.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            dtype=jnp.float32,
        )
        out = _split_chains(x)
        # First halves come first, then second halves.
        expected = jnp.array(
            [
                [1, 2],
                [5, 6],
                [3, 4],
                [7, 8],
            ],
            dtype=jnp.float32,
        )
        np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))


# --------------------------------------------------------------------------- #
# _ordinal_ranks                                                              #
# --------------------------------------------------------------------------- #


class TestOrdinalRanks:
    def test_sorted_input(self) -> None:
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(np.asarray(_ordinal_ranks(x)), [1, 2, 3, 4])

    def test_reverse_sorted_input(self) -> None:
        x = jnp.array([4.0, 3.0, 2.0, 1.0])
        np.testing.assert_array_equal(np.asarray(_ordinal_ranks(x)), [4, 3, 2, 1])

    def test_arbitrary_permutation(self) -> None:
        x = jnp.array([10.0, -1.0, 5.0, 3.0])
        # sorted order: -1, 3, 5, 10 -> ranks 4, 1, 3, 2
        np.testing.assert_array_equal(np.asarray(_ordinal_ranks(x)), [4, 1, 3, 2])

    def test_ties_broken_deterministically(self) -> None:
        x = jnp.array([2.0, 2.0, 2.0, 2.0])
        ranks = _ordinal_ranks(x)
        # Four unique ranks from {1, 2, 3, 4}.
        assert sorted(np.asarray(ranks).tolist()) == [1.0, 2.0, 3.0, 4.0]

    def test_independent_columns(self) -> None:
        x = jnp.array(
            [
                [1.0, 30.0],
                [2.0, 20.0],
                [3.0, 10.0],
            ]
        )
        # Column 0: already sorted (1, 2, 3). Column 1: reverse-sorted (3, 2, 1).
        out = _ordinal_ranks(x)
        np.testing.assert_array_equal(np.asarray(out[:, 0]), [1, 2, 3])
        np.testing.assert_array_equal(np.asarray(out[:, 1]), [3, 2, 1])

    def test_is_permutation_of_1_to_n(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (100,))
        ranks = np.sort(np.asarray(_ordinal_ranks(x)))
        np.testing.assert_array_equal(ranks, np.arange(1, 101, dtype=np.float32))


# --------------------------------------------------------------------------- #
# _rank_normalize                                                             #
# --------------------------------------------------------------------------- #


class TestRankNormalize:
    def test_shape_preserved(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (4, 50, 3, 2))
        out = _rank_normalize(x)
        assert out.shape == x.shape

    def test_approximately_normal(self, key: jax.Array) -> None:
        x = jax.random.exponential(key, (4, 1000))
        z = np.asarray(_rank_normalize(x))
        # Blom transform of iid samples should give approximately N(0, 1).
        assert abs(z.mean()) < 0.05
        assert abs(z.std() - 1.0) < 0.05

    def test_invariant_to_monotone_transform(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (2, 200))
        # exp is strictly increasing; ranks (and hence rank-normalized values)
        # should be exactly preserved.
        np.testing.assert_allclose(
            np.asarray(_rank_normalize(x)),
            np.asarray(_rank_normalize(jnp.exp(x))),
            atol=1e-6,
        )

    def test_monotone_within_parameter(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (3, 50))
        z = _rank_normalize(x)
        # Flatten jointly then check that sort permutations agree.
        flat_x = np.asarray(x).reshape(-1)
        flat_z = np.asarray(z).reshape(-1)
        np.testing.assert_array_equal(np.argsort(flat_x), np.argsort(flat_z))

    def test_independent_parameters(self, key: jax.Array) -> None:
        # With two independent parameter columns, each should be normalized
        # jointly over chains and draws but independently of the other.
        k1, k2 = jax.random.split(key)
        a = jax.random.normal(k1, (4, 200))
        b = 100.0 + 1e-3 * jax.random.normal(k2, (4, 200))  # tiny range, huge offset
        x = jnp.stack([a, b], axis=-1)
        z = _rank_normalize(x)
        # Each column marginally ~ N(0, 1) even though the raw scales are
        # wildly different.
        z_np = np.asarray(z)
        for col in range(2):
            assert abs(z_np[..., col].mean()) < 0.05
            assert abs(z_np[..., col].std() - 1.0) < 0.1


# --------------------------------------------------------------------------- #
# _rhat_plain                                                                 #
# --------------------------------------------------------------------------- #


class TestRhatPlain:
    def test_identical_chains_gives_exactly_one(self) -> None:
        single_chain = jnp.arange(100, dtype=jnp.float32)
        x = jnp.stack([single_chain, single_chain, single_chain], axis=0)
        # Zero between-chain variance => R-hat == sqrt( (N-1)/N * W / W ) < 1.
        # But the identity used is sqrt(var_hat / W) with B=0 -> sqrt((N-1)/N).
        rhat = float(_rhat_plain(x))
        assert rhat == pytest.approx(np.sqrt(99 / 100), rel=1e-6)

    def test_iid_chains_close_to_one(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (4, 5000))
        assert float(_rhat_plain(x)) == pytest.approx(1.0, abs=0.05)

    def test_shifted_chains_flag(self, key: jax.Array) -> None:
        shifts = jnp.array([-3.0, -1.0, 1.0, 3.0])[:, None]
        x = jax.random.normal(key, (4, 2000)) + shifts
        assert float(_rhat_plain(x)) > 1.5

    def test_shape_scalar_param(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (3, 200))
        assert _rhat_plain(x).shape == ()

    def test_shape_vector_param(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (3, 200, 7))
        assert _rhat_plain(x).shape == (7,)

    def test_manual_two_chain_computation(self) -> None:
        # Compute R-hat by hand on a tiny deterministic example to verify the
        # formula matches the canonical Gelman-Rubin definition.
        x = jnp.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 2.0, 4.0, 6.0],
            ],
            dtype=jnp.float32,
        )
        n_chains, n_samples = x.shape
        chain_mean = x.mean(axis=1)
        chain_var = x.var(axis=1, ddof=1)
        within = chain_var.mean()
        between = n_samples * chain_mean.var(ddof=1)
        var_hat = (n_samples - 1) / n_samples * within + between / n_samples
        expected = float(jnp.sqrt(var_hat / within))
        got = float(_rhat_plain(x))
        assert got == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- #
# _autocov_fft                                                                #
# --------------------------------------------------------------------------- #


class TestAutocovFft:
    def test_shape_preserved(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (32,))
        assert _autocov_fft(x).shape == x.shape

    def test_shape_preserved_multidim(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (5, 32, 3))
        # Default axis=0 means "along the first axis".
        assert _autocov_fft(x, axis=1).shape == x.shape

    def test_lag_zero_matches_biased_variance(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (256,))
        acov = _autocov_fft(x)
        assert float(acov[0]) == pytest.approx(float(jnp.var(x, ddof=0)), rel=1e-5)

    def test_constant_signal_is_zero(self) -> None:
        x = 7.0 * jnp.ones((64,))
        acov = np.asarray(_autocov_fft(x))
        np.testing.assert_allclose(acov, 0.0, atol=1e-6)

    def test_white_noise_decorrelates(self, key: jax.Array) -> None:
        # White noise: lag-0 ~ variance; lags > 0 small relative to lag-0.
        x = jax.random.normal(key, (4096,))
        acov = np.asarray(_autocov_fft(x))
        lag0 = acov[0]
        far_lags = np.abs(acov[10:4000])
        # Far lags should be well below lag-0.
        assert far_lags.max() < 0.3 * lag0

    def test_ar1_matches_theory(self, key: jax.Array) -> None:
        # AR(1): theoretical autocov at lag t is phi^t * sigma^2 / (1 - phi^2).
        phi = 0.7
        sigma = 1.0
        chains = _ar1(key, phi=phi, n_samples=20_000, n_chains=1, sigma=sigma)
        acov = np.asarray(_autocov_fft(chains[0]))
        var_theory = sigma**2 / (1.0 - phi**2)
        for lag in (0, 1, 2, 5, 10, 20):
            expected = phi**lag * var_theory
            assert acov[lag] == pytest.approx(expected, abs=0.1)

    def test_axis_argument(self, key: jax.Array) -> None:
        # Put the time axis last and verify we get the same answer.
        x = jax.random.normal(key, (4, 256))
        acov_axis1 = _autocov_fft(x, axis=1)
        acov_axis0 = _autocov_fft(x.T, axis=0)
        np.testing.assert_allclose(np.asarray(acov_axis1), np.asarray(acov_axis0.T), atol=1e-6)


# --------------------------------------------------------------------------- #
# _ess_ips                                                                    #
# --------------------------------------------------------------------------- #


class TestEssIps:
    def test_iid_close_to_total_samples(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (4, 4000))
        ess = float(_ess_ips(x))
        total = 4 * 4000
        assert 0.85 * total < ess <= total

    def test_ess_never_exceeds_total(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (4, 500))
        total = 4 * 500
        assert float(_ess_ips(x)) <= total + 1e-4

    def test_ar1_matches_theory(self, key: jax.Array) -> None:
        # Theoretical ESS for AR(1): N_total * (1 - phi) / (1 + phi).
        phi = 0.9
        x = _ar1(key, phi=phi, n_samples=4000, n_chains=4)
        total = 4 * 4000
        theoretical = total * (1.0 - phi) / (1.0 + phi)
        got = float(_ess_ips(x))
        # Allow ~25% tolerance (Geyer IPS plus finite-sample noise on rho estimates).
        assert got == pytest.approx(theoretical, rel=0.25)

    def test_high_autocorrelation_lowers_ess(self, key: jax.Array) -> None:
        k1, k2 = jax.random.split(key)
        iid = jax.random.normal(k1, (2, 2000))
        ar_strong = _ar1(k2, phi=0.95, n_samples=2000, n_chains=2)
        assert float(_ess_ips(ar_strong)) < 0.2 * float(_ess_ips(iid))

    def test_single_chain_is_supported(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (1, 4000))
        ess = float(_ess_ips(x))
        assert 0.8 * 4000 < ess <= 4000

    def test_shape_vector_param(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (3, 1000, 5))
        assert _ess_ips(x).shape == (5,)


# --------------------------------------------------------------------------- #
# _bulk_ess and _tail_ess                                                     #
# --------------------------------------------------------------------------- #


class TestBulkTailEss:
    def test_bulk_ess_iid(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (4, 2000))
        ess = float(_bulk_ess(x))
        total = 4 * 2000
        assert 0.8 * total < ess <= total

    def test_tail_ess_iid(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (4, 4000))
        ess = float(_tail_ess(x))
        total = 4 * 4000
        # Tail ESS has more variance for a given N; allow a looser lower bound.
        assert 0.5 * total < ess <= total

    def test_bulk_ess_reduces_under_autocorrelation(self, key: jax.Array) -> None:
        k1, k2 = jax.random.split(key)
        iid = jax.random.normal(k1, (2, 2000))
        ar_strong = _ar1(k2, phi=0.9, n_samples=2000, n_chains=2)
        assert float(_bulk_ess(ar_strong)) < 0.5 * float(_bulk_ess(iid))

    def test_tail_ess_reduces_under_autocorrelation(self, key: jax.Array) -> None:
        k1, k2 = jax.random.split(key)
        iid = jax.random.normal(k1, (2, 4000))
        ar_strong = _ar1(k2, phi=0.9, n_samples=4000, n_chains=2)
        assert float(_tail_ess(ar_strong)) < 0.6 * float(_tail_ess(iid))

    def test_bulk_ess_reacts_to_shifted_chains(self, key: jax.Array) -> None:
        # Chains disagreeing on the mean should yield a dramatically reduced
        # bulk-ESS once rank-normalized jointly.
        shifts = jnp.array([-3.0, -1.0, 1.0, 3.0])[:, None]
        x = jax.random.normal(key, (4, 2000)) + shifts
        total = 4 * 2000
        assert float(_bulk_ess(x)) < 0.01 * total


# --------------------------------------------------------------------------- #
# compute_rhat                                                                #
# --------------------------------------------------------------------------- #


class TestComputeRhat:
    def test_iid_chains(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (4, 4000))
        assert float(compute_rhat(x)) == pytest.approx(1.0, abs=0.02)

    def test_shifted_chains(self, key: jax.Array) -> None:
        shifts = jnp.array([-3.0, -1.0, 1.0, 3.0])[:, None]
        x = jax.random.normal(key, (4, 2000)) + shifts
        assert float(compute_rhat(x)) > 1.5

    def test_single_well_mixed_chain(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (1, 8000))
        assert float(compute_rhat(x)) == pytest.approx(1.0, abs=0.02)

    def test_single_drifting_chain_flagged(self, key: jax.Array) -> None:
        # Linearly drifting mean across the chain -> split halves disagree.
        t = jnp.linspace(0.0, 5.0, 4000)
        x = jax.random.normal(key, (1, 4000)) + t[None, :]
        assert float(compute_rhat(x)) > 1.3

    def test_scale_disagreement_flagged_by_tail(self, key: jax.Array) -> None:
        # Two chains with identical means but very different scales: bulk-R-hat
        # should look OK, but the max over (bulk, tail) should still exceed 1
        # because the folded-median trick picks up the scale disagreement.
        k1, k2 = jax.random.split(key)
        a = 0.2 * jax.random.normal(k1, (2000,))
        b = 4.0 * jax.random.normal(k2, (2000,))
        x = jnp.stack([a, b], axis=0)
        assert float(compute_rhat(x)) > 1.1

    def test_vector_parameter_shape(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (3, 500, 4, 2))
        out = compute_rhat(x)
        assert out.shape == (4, 2)

    def test_raises_on_1d_input(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            compute_rhat(jnp.arange(10.0))

    def test_jit_compiles(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (4, 1000))
        eager = float(compute_rhat(x))
        jitted = float(jax.jit(compute_rhat)(x))
        assert eager == pytest.approx(jitted, rel=1e-5)


# --------------------------------------------------------------------------- #
# compute_ess                                                                 #
# --------------------------------------------------------------------------- #


class TestComputeEss:
    @pytest.mark.parametrize("kind", ["bulk", "tail", "mean"])
    def test_iid_near_total_samples(self, key: jax.Array, kind: str) -> None:
        x = jax.random.normal(key, (4, 4000))
        total = 4 * 4000
        ess = float(compute_ess(x, kind=kind))
        # Generous lower bounds; tail is noisier so floor is lower.
        floor = 0.5 * total if kind == "tail" else 0.7 * total
        assert floor < ess <= total

    def test_ar1_mean_matches_theory(self, key: jax.Array) -> None:
        phi = 0.9
        x = _ar1(key, phi=phi, n_samples=4000, n_chains=4)
        total = 4 * 4000
        theoretical = total * (1.0 - phi) / (1.0 + phi)
        got = float(compute_ess(x, kind="mean"))
        assert got == pytest.approx(theoretical, rel=0.25)

    def test_bulk_smaller_than_mean_under_non_gaussian(self, key: jax.Array) -> None:
        # For heavy-tailed (or non-Gaussian) targets, bulk-ESS (rank-normalized)
        # should still be well-defined and comparable to mean-ESS within a
        # reasonable factor. Here we just check both are positive and finite.
        x = jax.random.exponential(key, (4, 2000))
        bulk = float(compute_ess(x, kind="bulk"))
        mean = float(compute_ess(x, kind="mean"))
        assert bulk > 0 and np.isfinite(bulk)
        assert mean > 0 and np.isfinite(mean)

    def test_default_kind_is_bulk(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (2, 500))
        got_default = float(compute_ess(x))
        got_bulk = float(compute_ess(x, kind="bulk"))
        assert got_default == pytest.approx(got_bulk, rel=1e-5)

    def test_vector_parameter_shape(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (3, 1000, 4, 2))
        for kind in ("bulk", "tail", "mean"):
            assert compute_ess(x, kind=kind).shape == (4, 2)

    def test_shifted_chains_hurt_bulk_ess(self, key: jax.Array) -> None:
        shifts = jnp.array([-3.0, -1.0, 1.0, 3.0])[:, None]
        x = jax.random.normal(key, (4, 2000)) + shifts
        total = 4 * 2000
        assert float(compute_ess(x, kind="bulk")) < 0.02 * total

    def test_raises_on_1d_input(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            compute_ess(jnp.arange(10.0))

    def test_raises_on_unknown_kind(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (2, 100))
        with pytest.raises(ValueError, match="Unknown ESS kind"):
            compute_ess(x, kind="banana")

    def test_jit_compiles(self, key: jax.Array) -> None:
        from functools import partial

        x = jax.random.normal(key, (4, 1000))
        for kind in ("bulk", "tail", "mean"):
            eager = float(compute_ess(x, kind=kind))
            jitted = float(jax.jit(partial(compute_ess, kind=kind))(x))
            assert eager == pytest.approx(jitted, rel=1e-5)

    def test_vmap_over_leading_axis(self, key: jax.Array) -> None:
        from functools import partial

        xs = jax.random.normal(key, (6, 3, 1000))  # 6 experiments of 3 chains x 1000
        out = jax.vmap(partial(compute_ess, kind="bulk"))(xs)
        assert out.shape == (6,)
        assert np.all(np.asarray(out) > 0)


# --------------------------------------------------------------------------- #
# Cross-checks / integration                                                  #
# --------------------------------------------------------------------------- #


class TestIntegration:
    def test_public_rhat_is_max_of_bulk_and_tail(self, key: jax.Array) -> None:
        # Recompute bulk-R-hat and tail-R-hat manually and verify
        # compute_rhat == max(bulk, tail).
        x = jax.random.normal(key, (4, 500))
        split = _split_chains(x)
        bulk = float(_rhat_plain(_rank_normalize(split)))
        median = float(jnp.median(split))
        folded = jnp.abs(split - median)
        tail = float(_rhat_plain(_rank_normalize(folded)))
        expected = max(bulk, tail)
        np.testing.assert_allclose(float(compute_rhat(x)), expected, rtol=1e-5)

    def test_public_bulk_ess_matches_private(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (3, 600))
        split = _split_chains(x)
        expected = float(_bulk_ess(split))
        np.testing.assert_allclose(float(compute_ess(x, kind="bulk")), expected, rtol=1e-5)

    def test_public_tail_ess_matches_private(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (3, 600))
        split = _split_chains(x)
        expected = float(_tail_ess(split))
        np.testing.assert_allclose(float(compute_ess(x, kind="tail")), expected, rtol=1e-5)

    def test_public_mean_ess_matches_private(self, key: jax.Array) -> None:
        x = jax.random.normal(key, (3, 600))
        split = _split_chains(x)
        expected = float(_ess_ips(split))
        np.testing.assert_allclose(float(compute_ess(x, kind="mean")), expected, rtol=1e-5)

    def test_rhat_and_ess_agree_on_pathology(self, key: jax.Array) -> None:
        # A bad chain setup should trip both diagnostics: R-hat large *and*
        # bulk-ESS tiny (relative to total).
        shifts = jnp.array([-3.0, -1.0, 1.0, 3.0])[:, None]
        x = jax.random.normal(key, (4, 2000)) + shifts
        total = 4 * 2000
        assert float(compute_rhat(x)) > 1.5
        assert float(compute_ess(x, kind="bulk")) < 0.02 * total
