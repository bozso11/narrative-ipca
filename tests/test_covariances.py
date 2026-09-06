"""Tests for narrative_ipca.covariances: Eq. 6 recursion vs an O(days) brute-force reference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from narrative_ipca.config import CovarianceConfig
from narrative_ipca.covariances import (
    brute_force_covariance,
    build_covariance_panel,
    kernel_weights,
    window_bounds,
)
from narrative_ipca.data import period_end_index
from narrative_ipca.types import ShockPanel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_data(seed: int = 0, n_days: int = 300, N: int = 5, L: int = 3, missing: float = 0.1):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2021-01-04", periods=n_days)
    R = rng.standard_normal((n_days, N))
    R[rng.random((n_days, N)) < missing] = np.nan
    R[:40, 0] = np.nan  # asset 0 enters late
    R[-30:, 1] = np.nan  # asset 1 exits early
    Z = rng.standard_normal((n_days, L)) + 0.3  # non-zero mean so the mean correction matters
    Z[:5] = np.nan  # shock warm-up rows
    if n_days > 123:
        Z[123, 1] = np.nan  # an isolated missing shock
    returns = pd.DataFrame(R, index=cal, columns=[f"a{i}" for i in range(N)])
    z = pd.DataFrame(Z, index=cal, columns=[f"z{l}" for l in range(L)])
    return returns, ShockPanel(z=z, window=5)


def reference_panel(shocks: ShockPanel, returns: pd.DataFrame, cfg: CovarianceConfig, period: str):
    """O(T N days L) brute force: explicit kernel per (t, i) with the window-end cutoff."""
    cal = returns.index
    z = shocks.z.reindex(cal)
    pid, ends = period_end_index(cal, period)
    start, stop, cut = window_bounds(pid, cfg.skip_days)
    T, N, L = len(ends), returns.shape[1], z.shape[1]
    ref = np.full((T, N, L), np.nan)
    n_days = np.zeros((T, N), dtype=int)
    zok = np.isfinite(z.to_numpy()).all(axis=1)
    for t in range(T):
        w = kernel_weights(pid, t, cfg.xi, cfg.lookback_periods)
        w[cut[t] :] = 0.0  # exclude the tail of period t (and everything after)
        for i in range(N):
            r = returns.iloc[:, i]
            obs = np.isfinite(r.to_numpy()) & zok & (w > 0)
            n_days[t, i] = obs.sum()
            if obs.sum() >= cfg.min_days:
                ref[t, i] = brute_force_covariance(r, z, w)
    return ref, n_days, ends


# ---------------------------------------------------------------------------
# kernel_weights / window_bounds
# ---------------------------------------------------------------------------
def test_kernel_weights_values():
    pid = np.array([0, 0, 1, 1, 2])
    np.testing.assert_allclose(kernel_weights(pid, 2, 0.5, None), [0.25, 0.25, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(kernel_weights(pid, 2, 0.5, 2), [0.0, 0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(kernel_weights(pid, 2, 0.5, 1), [0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(kernel_weights(pid, 1, 0.5, None), [0.5, 0.5, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(kernel_weights(pid, 1, 1.0, None), [1.0, 1.0, 1.0, 1.0, 0.0])


def test_kernel_weights_rejects_bad_args():
    with pytest.raises(ValueError):
        kernel_weights(np.array([0, 1]), 1, 0.0, None)
    with pytest.raises(ValueError):
        kernel_weights(np.array([0, 1]), 1, 0.9, 0)


def test_window_bounds_skip_days():
    pid = np.array([0, 0, 0, 1, 1, 2])
    start, stop, cut = window_bounds(pid, 1)
    np.testing.assert_array_equal(start, [0, 3, 5])
    np.testing.assert_array_equal(stop, [3, 5, 6])
    np.testing.assert_array_equal(cut, [2, 4, 5])
    _, _, cut3 = window_bounds(pid, 3)
    np.testing.assert_array_equal(cut3, [0, 3, 5])  # never before the period start


# ---------------------------------------------------------------------------
# brute_force_covariance
# ---------------------------------------------------------------------------
def test_brute_force_matches_weighted_covariance_formula():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.standard_normal(50))
    z = pd.DataFrame(rng.standard_normal((50, 2)))
    w = rng.random(50)
    out = brute_force_covariance(r, z, w)
    wn = w / w.sum()
    expected = np.array([np.sum(wn * (r - wn @ r) * (z[c] - wn @ z[c])) for c in z.columns])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_brute_force_drops_missing_and_zero_weight_days():
    r = pd.Series([1.0, np.nan, 3.0, 4.0])
    z = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
    w = np.array([1.0, 1.0, 1.0, 0.0])
    out = brute_force_covariance(r, z, w)
    np.testing.assert_allclose(out, [0.0], atol=1e-15)  # only day 0 survives: r z - (r)(z) = 0
    assert np.isnan(brute_force_covariance(r, z, np.zeros(4))).all()  # no day survives
    r2 = pd.Series([1.0, 2.0, np.nan, 4.0])
    out2 = brute_force_covariance(r2, z, np.array([1.0, 1.0, 1.0, 1.0]))
    expected = np.cov(np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.0, 4.0]), ddof=0)[0, 1]
    np.testing.assert_allclose(out2, [expected], atol=1e-12)


# ---------------------------------------------------------------------------
# build_covariance_panel vs brute force
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("period", ["M", "W"])
@pytest.mark.parametrize("lookback", [None, 1, 3, 7])
@pytest.mark.parametrize("skip_days", [0, 1, 3])
def test_panel_equals_brute_force(period, lookback, skip_days):
    returns, shocks = make_data(seed=3)
    # a weekly window truncated to one period holds at most 5 - skip_days days
    min_days = 2 if (period == "W" and lookback == 1) else 4
    cfg = CovarianceConfig(xi=0.9, skip_days=skip_days, lookback_periods=lookback, min_days=min_days)
    got = build_covariance_panel(shocks, returns, cfg, period)
    ref, n_days, ends = reference_panel(shocks, returns, cfg, period)
    assert got.shape == ref.shape
    assert got.periods.equals(ends)
    np.testing.assert_array_equal(got.n_days, n_days)
    np.testing.assert_allclose(got.values, ref, atol=1e-10, rtol=1e-10, equal_nan=True)
    assert np.isfinite(got.values).any(), "no valid asset-period in the test data"
    assert np.isnan(got.values).any(), "test data should also exercise the min_days branch"


def test_panel_bks_defaults_vs_brute_force():
    returns, shocks = make_data(seed=11, n_days=520, N=4, L=2)
    cfg = CovarianceConfig()  # xi=0.99, skip_days=1, no lookback, min_days=60
    got = build_covariance_panel(shocks, returns, cfg, "M")
    ref, n_days, _ = reference_panel(shocks, returns, cfg, "M")
    np.testing.assert_allclose(got.values, ref, atol=1e-10, rtol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(got.n_days, n_days)


def test_xi_one_full_window_is_plain_available_case_covariance():
    returns, shocks = make_data(seed=5, missing=0.0)
    cfg = CovarianceConfig(xi=1.0, skip_days=0, lookback_periods=None, min_days=2)
    got = build_covariance_panel(shocks, returns, cfg, "M")
    cal = returns.index
    pid, ends = period_end_index(cal, "M")
    z = shocks.z.to_numpy()
    for t in range(len(ends)):
        rows = np.flatnonzero(pid <= t)
        for i in range(returns.shape[1]):
            r = returns.iloc[rows, i].to_numpy()
            ok = np.isfinite(r) & np.isfinite(z[rows]).all(axis=1)
            if ok.sum() < 2:
                assert np.isnan(got.values[t, i]).all()
                continue
            for l in range(z.shape[1]):
                c = np.cov(r[ok], z[rows][ok, l], ddof=0)[0, 1]
                assert got.values[t, i, l] == pytest.approx(c, abs=1e-10)


def test_scale_equivariance_per_asset():
    returns, shocks = make_data(seed=7)
    cfg = CovarianceConfig(xi=0.95, min_days=4)
    base = build_covariance_panel(shocks, returns, cfg, "M")
    scaled = returns.copy()
    scaled.iloc[:, 2] *= 3.5
    got = build_covariance_panel(shocks, scaled, cfg, "M")
    np.testing.assert_allclose(got.values[:, 2], 3.5 * base.values[:, 2], rtol=1e-12, equal_nan=True)
    others = [0, 1, 3, 4]
    # other assets untouched (up to BLAS rounding, which may differ by code path)
    np.testing.assert_allclose(got.values[:, others], base.values[:, others], rtol=1e-12, equal_nan=True)


def test_shift_invariance_in_r_and_z():
    returns, shocks = make_data(seed=8)
    cfg = CovarianceConfig(xi=0.95, min_days=4)
    base = build_covariance_panel(shocks, returns, cfg, "M")
    shifted = build_covariance_panel(ShockPanel(z=shocks.z + 7.0, window=5), returns + 2.5, cfg, "M")
    np.testing.assert_allclose(shifted.values, base.values, atol=1e-10, equal_nan=True)


def test_window_end_dates():
    returns, shocks = make_data(seed=2)
    pid, ends = period_end_index(returns.index, "M")
    got1 = build_covariance_panel(shocks, returns, CovarianceConfig(skip_days=1, min_days=2), "M")
    got0 = build_covariance_panel(shocks, returns, CovarianceConfig(skip_days=0, min_days=2), "M")
    assert got0.window_end.equals(ends)
    for t in range(len(ends)):
        days_t = returns.index[pid == t]
        assert got1.window_end[t] == days_t[-2]
        assert got1.window_end[t] < got1.periods[t]


def test_window_end_falls_back_to_previous_period_when_period_too_short():
    returns, shocks = make_data(seed=2, n_days=60)
    got = build_covariance_panel(shocks, returns, CovarianceConfig(skip_days=10, min_days=2), "W")
    pid, ends = period_end_index(returns.index, "W")
    assert pd.isna(got.window_end[0])
    assert got.window_end[1] == ends[0]


def test_min_days_and_n_days():
    returns, shocks = make_data(seed=4)
    cfg = CovarianceConfig(xi=0.9, skip_days=1, lookback_periods=2, min_days=25)
    got = build_covariance_panel(shocks, returns, cfg, "M")
    nan_rows = np.isnan(got.values).all(axis=2)
    np.testing.assert_array_equal(nan_rows, got.n_days < cfg.min_days)
    assert nan_rows.any() and (~nan_rows).any()


def test_missing_shock_days_are_excluded_for_every_asset():
    returns, shocks = make_data(seed=9, missing=0.0)
    cfg = CovarianceConfig(xi=0.9, min_days=2)
    got = build_covariance_panel(shocks, returns, cfg, "M")
    # rebuild with the missing-shock days removed from the returns entirely: same result
    ok = np.isfinite(shocks.z.to_numpy()).all(axis=1)
    r2 = returns.copy()
    r2.loc[~ok] = np.nan
    got2 = build_covariance_panel(shocks, r2, cfg, "M")
    np.testing.assert_allclose(got.values, got2.values, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(got.n_days, got2.n_days)


def test_storage_dtype_and_labels():
    returns, shocks = make_data(seed=6)
    got = build_covariance_panel(shocks, returns, CovarianceConfig(min_days=4), "M", dtype="float32")
    assert got.values.dtype == np.float32
    assert list(got.assets) == list(returns.columns)
    assert list(got.topics) == list(shocks.z.columns)
    assert got.xi == pytest.approx(0.99)
    frame = got.frame(5)
    assert frame.shape == (returns.shape[1], shocks.z.shape[1])
    assert list(frame.index) == list(returns.columns)


def test_shocks_on_a_different_calendar_are_reindexed():
    returns, shocks = make_data(seed=10)
    cfg = CovarianceConfig(xi=0.9, min_days=4)
    base = build_covariance_panel(shocks, returns, cfg, "M")
    # add weekend rows to z (must be ignored) and drop one trading day (becomes a missing shock)
    extra = pd.DataFrame(
        np.ones((2, shocks.z.shape[1])), index=pd.DatetimeIndex(["2021-01-09", "2021-01-10"]), columns=shocks.z.columns
    )
    z2 = pd.concat([shocks.z, extra]).sort_index()
    got = build_covariance_panel(ShockPanel(z=z2, window=5), returns, cfg, "M")
    np.testing.assert_allclose(got.values, base.values, atol=1e-12, equal_nan=True)


def test_rejects_unsorted_returns():
    returns, shocks = make_data(seed=1, n_days=50)
    with pytest.raises(ValueError):
        build_covariance_panel(shocks, returns.iloc[::-1], CovarianceConfig(min_days=2), "M")


@pytest.mark.parametrize("lookback", [None, 12, 7])
def test_long_horizon_recursion_has_no_drift(lookback):
    """20 years of monthly periods: the closed-form recursion (with periodic rebuilds when truncated)
    must still agree with the brute force at 1e-10 on the last periods."""
    rng = np.random.default_rng(42)
    cal = pd.bdate_range("2000-01-03", periods=252 * 20)
    R = rng.standard_normal((len(cal), 3))
    R[rng.random(R.shape) < 0.1] = np.nan
    Z = rng.standard_normal((len(cal), 2)) + 0.5
    Z[:5] = np.nan
    returns = pd.DataFrame(R, index=cal, columns=list("abc"))
    shocks = ShockPanel(z=pd.DataFrame(Z, index=cal, columns=["u", "v"]), window=5)
    cfg = CovarianceConfig(xi=0.99, skip_days=1, lookback_periods=lookback, min_days=10)
    got = build_covariance_panel(shocks, returns, cfg, "M")
    ref, n_days, _ = reference_panel(shocks, returns, cfg, "M")
    assert got.shape[0] > 200
    np.testing.assert_allclose(got.values, ref, atol=1e-10, rtol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(got.n_days, n_days)


def test_window_matches_app_b1_definition_without_module_helpers():
    """BKS App. B.1: kappa(tau; t) = xi^(t - t_tau) for tau < tau_t (tau_t the last day of month t), 0 otherwise.

    Independent reference: explicit day loop with its own period ids; no kernel_weights / window_bounds.
    Guards the window boundary (the second-to-last trading day of period t is the last day inside the
    window with skip_days=1) and the per-period decay, which the other tests share with the code under test.
    """
    rng = np.random.default_rng(21)
    cal = pd.bdate_range("2021-01-04", periods=260)
    N, L, xi = 3, 2, 0.9
    R = rng.standard_normal((len(cal), N))
    R[rng.random(R.shape) < 0.1] = np.nan
    Z = rng.standard_normal((len(cal), L)) + 0.4
    Z[:5] = np.nan
    returns = pd.DataFrame(R, index=cal, columns=list("abc"))
    shocks = ShockPanel(z=pd.DataFrame(Z, index=cal, columns=["u", "v"]), window=5)
    got = build_covariance_panel(shocks, returns, CovarianceConfig(xi=xi, skip_days=1, min_days=3), "M")

    months = sorted({(d.year, d.month) for d in cal})
    pid = np.array([months.index((d.year, d.month)) for d in cal])
    for t in range(len(months)):
        last_day_of_t = np.flatnonzero(pid == t)[-1]
        for i in range(N):
            num_rz, num_r, num_z, wsum, n = np.zeros(L), 0.0, np.zeros(L), 0.0, 0
            for tau in range(len(cal)):
                if pid[tau] > t or tau >= last_day_of_t:  # tau < tau_t: strictly before the last day of month t
                    continue
                if not (np.isfinite(R[tau, i]) and np.isfinite(Z[tau]).all()):
                    continue
                k = xi ** (t - pid[tau])
                num_rz += k * R[tau, i] * Z[tau]
                num_r += k * R[tau, i]
                num_z += k * Z[tau]
                wsum += k
                n += 1
            assert got.n_days[t, i] == n
            if n < 3:
                assert np.isnan(got.values[t, i]).all()
            else:
                ref = num_rz / wsum - (num_r / wsum) * (num_z / wsum)
                np.testing.assert_allclose(got.values[t, i], ref, atol=1e-12)
        assert got.window_end[t] == cal[last_day_of_t - 1]
