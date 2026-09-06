"""Tests for narrative_ipca.panel: Eq. 7 lag alignment, burn-in, min-assets filter, moments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from narrative_ipca.config import CovarianceConfig, DataConfig, ShockConfig
from narrative_ipca.covariances import build_covariance_panel
from narrative_ipca.data import period_end_index, period_returns
from narrative_ipca.panel import build_panel, panel_summary, split_periods
from narrative_ipca.shocks import attention_shocks
from narrative_ipca.types import CovariancePanel, compute_sigma_c


# ---------------------------------------------------------------------------
# a hand-built covariance panel with distinguishable entries
# ---------------------------------------------------------------------------
def _tiny(seed: int = 0):
    """Three months (Jan-Mar 2020), three assets, two topics; cov[t, i, l] = 100 t + 10 i + l."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2020-01-01", "2020-03-31")
    N, L = 3, 2
    returns = pd.DataFrame(rng.standard_normal((len(cal), N)) * 0.01, index=cal, columns=["A", "B", "C"])
    pid, ends = period_end_index(cal, "M")
    T = len(ends)
    values = np.empty((T, N, L))
    for t in range(T):
        for i in range(N):
            for l in range(L):
                values[t, i, l] = 100 * t + 10 * i + l
    cov = CovariancePanel(
        values=values,
        periods=ends,
        window_end=ends,
        assets=np.array(["A", "B", "C"], dtype=object),
        topics=np.array(["z0", "z1"], dtype=object),
        n_days=np.full((T, N), 20),
        xi=0.99,
    )
    return cov, returns, cal, ends


def test_lag_alignment_instrument_t_minus_1_with_return_t():
    cov, returns, cal, ends = _tiny()
    panel = build_panel(cov, returns, DataConfig(min_assets_per_period=1), CovarianceConfig(burn_in_periods=0))
    pr = period_returns(returns, "M", "sum")
    # return periods Feb and Mar (instrument periods Jan and Feb); Jan has no lagged instrument
    assert list(panel.periods) == [ends[1], ends[2]]
    assert panel.instrument_names == ["const", "z0", "z1"]
    assert panel.n_obs == 6 and panel.p == 3 and panel.L == 2 and panel.T == 2 and panel.N == 3
    np.testing.assert_array_equal(panel.t_idx, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(panel.asset_idx, [0, 1, 2, 0, 1, 2])
    assert (panel.X[:, 0] == 1.0).all()
    np.testing.assert_allclose(panel.X[:3, 1:], cov.values[0])  # Jan instruments -> Feb returns
    np.testing.assert_allclose(panel.X[3:, 1:], cov.values[1])  # Feb instruments -> Mar returns
    np.testing.assert_allclose(panel.y[:3], pr.loc[ends[1]].to_numpy())
    np.testing.assert_allclose(panel.y[3:], pr.loc[ends[2]].to_numpy())
    np.testing.assert_allclose(panel.sigma_c, compute_sigma_c(panel.X))
    assert panel.sigma_c[0] == 1.0 and (panel.sigma_c[1:] > 0).all()


def test_burn_in_drops_first_instrument_periods():
    cov, returns, cal, ends = _tiny()
    panel = build_panel(cov, returns, DataConfig(min_assets_per_period=1), CovarianceConfig(burn_in_periods=1))
    assert list(panel.periods) == [ends[2]]
    np.testing.assert_allclose(panel.X[:, 1:], cov.values[1])
    with pytest.raises(ValueError):
        build_panel(cov, returns, DataConfig(min_assets_per_period=1), CovarianceConfig(burn_in_periods=2))


def test_min_assets_filter_and_nonfinite_rows():
    cov, returns, cal, ends = _tiny()
    cov.values[1, 0, :] = np.nan  # asset A has no Feb instrument
    returns.loc[returns.index.month == 2, "B"] = np.nan  # asset B has no Feb return
    pr = period_returns(returns, "M", "sum")
    p2 = build_panel(cov, returns, DataConfig(min_assets_per_period=2), CovarianceConfig(burn_in_periods=0))
    # Feb return period: A, C (B has no return); Mar return period: B, C (A has no instrument)
    assert list(p2.periods) == [ends[1], ends[2]]
    np.testing.assert_array_equal(p2.asset_idx, [0, 2, 1, 2])
    np.testing.assert_array_equal(p2.t_idx, [0, 0, 1, 1])
    np.testing.assert_allclose(p2.X[:2, 1:], cov.values[0][[0, 2]])
    np.testing.assert_allclose(p2.X[2:, 1:], cov.values[1][[1, 2]])
    np.testing.assert_allclose(p2.y, [pr.loc[ends[1], "A"], pr.loc[ends[1], "C"], pr.loc[ends[2], "B"], pr.loc[ends[2], "C"]])
    # with min_assets=3 every period has only two rows -> no period survives
    with pytest.raises(ValueError):
        build_panel(cov, returns, DataConfig(min_assets_per_period=3), CovarianceConfig(burn_in_periods=0))


def test_compound_aggregation():
    cov, returns, cal, ends = _tiny()
    panel = build_panel(
        cov, returns, DataConfig(min_assets_per_period=1, return_aggregation="compound"), CovarianceConfig(burn_in_periods=0)
    )
    feb = returns[returns.index.month == 2]
    np.testing.assert_allclose(panel.y[:3], (np.prod(1 + feb, axis=0) - 1).to_numpy())


def test_moments_equal_brute_force():
    cov, returns, cal, ends = _tiny(seed=3)
    panel = build_panel(cov, returns, DataConfig(min_assets_per_period=1), CovarianceConfig(burn_in_periods=0))
    m = panel.moments()
    assert m.T == panel.T and m.p == panel.p
    for t in range(panel.T):
        rows = panel.t_idx == t
        Xt, yt = panel.X[rows], panel.y[rows]
        np.testing.assert_allclose(m.S[t], Xt.T @ Xt, rtol=1e-13)
        np.testing.assert_allclose(m.V[t], Xt.T @ yt, rtol=1e-13)
        assert m.yy[t] == pytest.approx(yt @ yt)
        assert m.n[t] == rows.sum()


def test_split_periods_and_summary():
    cov, returns, cal, ends = _tiny()
    panel = build_panel(cov, returns, DataConfig(min_assets_per_period=1), CovarianceConfig(burn_in_periods=0))
    train, test = split_periods(panel, pd.Timestamp("2020-03-01"))
    np.testing.assert_array_equal(train, [True, False])
    np.testing.assert_array_equal(test, [False, True])
    train2, _ = split_periods(panel, ends[1])  # boundary is inclusive on the test side
    np.testing.assert_array_equal(train2, [False, False])
    s = panel_summary(panel)
    assert list(s.index) == list(panel.periods)
    assert (s["n_assets"] == 3).all()
    assert s["mean_y"].iloc[0] == pytest.approx(panel.y[:3].mean())
    assert s["std_y"].iloc[1] == pytest.approx(panel.y[3:].std(ddof=1))


def test_rejects_returns_from_another_calendar():
    cov, returns, cal, ends = _tiny()
    shifted = returns.copy()
    shifted.index = shifted.index + pd.DateOffset(years=1)
    with pytest.raises(ValueError):
        build_panel(cov, shifted, DataConfig(min_assets_per_period=1), CovarianceConfig(burn_in_periods=0))


# ---------------------------------------------------------------------------
# end to end with the covariance stage
# ---------------------------------------------------------------------------
def test_end_to_end_rows_match_covariance_frames():
    rng = np.random.default_rng(1)
    cal = pd.bdate_range("2019-01-01", periods=380)
    att = pd.DataFrame(rng.random((len(cal), 3)), index=cal, columns=["n1", "n2", "n3"])
    ret = pd.DataFrame(rng.standard_normal((len(cal), 6)) * 0.01, index=cal, columns=[f"a{i}" for i in range(6)])
    ret[rng.random(ret.shape) < 0.05] = np.nan
    ret.iloc[:150, 5] = np.nan  # late entrant
    shocks = attention_shocks(att, ShockConfig())
    cov_cfg = CovarianceConfig(burn_in_periods=3, min_days=40)
    data_cfg = DataConfig(min_assets_per_period=4)
    cov = build_covariance_panel(shocks, ret, cov_cfg, "M")
    panel = build_panel(cov, ret, data_cfg, cov_cfg)
    pr = period_returns(ret, "M", "sum")
    cov_pos = {p: j for j, p in enumerate(cov.periods)}
    assert panel.instrument_names == ["const", "n1", "n2", "n3"]
    for t, sl in panel.period_slices():
        ret_period = panel.periods[t]
        j = cov_pos[cov.periods[cov.periods < ret_period][-1]]  # the instrument period right before
        assert j >= cov_cfg.burn_in_periods
        frame = cov.frame(j)
        y_all = pr.loc[ret_period]
        expected_assets = [a for a in cov.assets if np.isfinite(frame.loc[a]).all() and np.isfinite(y_all[a])]
        assert len(expected_assets) >= data_cfg.min_assets_per_period
        got_assets = list(panel.assets[panel.asset_idx[sl]])
        assert got_assets == expected_assets
        np.testing.assert_allclose(panel.X[sl, 1:], frame.loc[expected_assets].to_numpy())
        np.testing.assert_allclose(panel.y[sl], y_all[expected_assets].to_numpy())
    # no row uses an instrument period at or after its return period (no look-ahead)
    for t in range(panel.T):
        assert cov.window_end[cov_pos[cov.periods[cov.periods < panel.periods[t]][-1]]] < panel.periods[t]
