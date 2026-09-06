"""Tests of the simulation DGP (DESIGN.md Part D) against its own mathematics.

The checks are population identities of the data-generating process evaluated
on long samples (20 years by default, sampling error a few per cent), the
exact bookkeeping identities (period sums, masks, labels) and the ground-truth
formulas of Part D step 7.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from narrative_ipca import simulation as sim
from narrative_ipca.config import AssetClassSpec, SimulationConfig
from narrative_ipca.simulation import scenario_config, simulate
from narrative_ipca.types import SimulatedData, annualized_sharpe

FAST = SimulationConfig(n_assets=60, n_topics=20, n_relevant=5, n_placebo=5, n_years=3)


@pytest.fixture(scope="module")
def baseline() -> SimulatedData:
    return simulate(scenario_config("baseline"))


@pytest.fixture(scope="module")
def balanced() -> SimulatedData:
    """Balanced panel with constant loadings, so population identities are exact."""
    return simulate(replace(scenario_config("balanced"), beta_innov_sd=0.0))


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ---------------------------------------------------------------------------
# Reproducibility and shapes
# ---------------------------------------------------------------------------
def test_reproducible_by_seed_and_independent_of_global_state():
    a = simulate(FAST)
    np.random.seed(12345)  # the simulator must not touch the legacy global stream
    b = simulate(FAST)
    pd.testing.assert_frame_equal(a.attention.levels, b.attention.levels)
    pd.testing.assert_frame_equal(a.returns.returns, b.returns.returns)
    pd.testing.assert_frame_equal(a.truth.z_daily, b.truth.z_daily)
    np.testing.assert_array_equal(a.truth.A, b.truth.A)
    np.testing.assert_array_equal(a.truth.beta, b.truth.beta)
    pd.testing.assert_frame_equal(a.returns.asset_meta, b.returns.asset_meta)
    c = simulate(replace(FAST, seed=1))
    assert not np.allclose(a.returns.returns.fillna(0.0).values, c.returns.returns.fillna(0.0).values)
    assert not np.array_equal(a.truth.relevant, c.truth.relevant) or not np.allclose(a.truth.A, c.truth.A)


def test_shapes_labels_and_masks(baseline: SimulatedData):
    cfg, tr = baseline.config, baseline.truth
    n_days = cfg.n_years * cfg.days_per_year
    L, N, K = cfg.n_topics, cfg.n_assets, cfg.K
    T = len(tr.periods)
    assert baseline.attention.levels.shape == (n_days, L)
    assert baseline.returns.returns.shape == (n_days, N)
    assert list(baseline.attention.levels.columns) == [f"topic_{l + 1:03d}" for l in range(L)]
    assert list(baseline.returns.returns.columns) == [f"A{i + 1:04d}" for i in range(N)]
    assert baseline.attention.levels.index.equals(baseline.returns.returns.index)
    assert isinstance(baseline.attention.levels.index, pd.DatetimeIndex)
    assert baseline.attention.levels.index[0] == pd.Timestamp("2005-01-03")
    assert tr.A.shape == (L, K)
    assert tr.relevant.shape == (L,) and tr.placebo.shape == (L,)
    assert tr.relevant.sum() == cfg.n_relevant and tr.placebo.sum() == cfg.n_placebo
    assert not np.any(tr.relevant & tr.placebo), "placebo must be a subset of ~relevant"
    assert tr.f_daily.shape == (n_days, K) and tr.x_daily.shape == (n_days, K)
    assert tr.z_daily.shape == (n_days, L)
    assert tr.f_period.shape == (T, K)
    assert tr.beta.shape == (T, N, K)
    assert tr.mu_f_period.shape == (K,) and tr.Sigma_ff_period.shape == (K, K) and tr.Sigma_ff_daily.shape == (K, K)
    assert tr.Gamma_tilde_true.shape == (L, K) and tr.impact_z_to_mve_true.shape == (L,)
    assert tr.asset_class.shape == (N,)
    assert list(tr.z_daily.columns) == list(baseline.attention.levels.columns)
    assert tr.f_period.index.equals(tr.periods)
    meta = baseline.returns.asset_meta
    assert list(meta.columns) == ["asset_class", "entry", "exit"]
    assert list(meta.index) == list(baseline.returns.returns.columns)
    assert set(baseline.attention.topic_labels) == set(baseline.attention.levels.columns)
    assert tr.meta["scenario"] == "baseline"


def test_periods_are_calendar_months_stamped_at_last_trading_day(baseline: SimulatedData):
    cal = baseline.attention.levels.index
    tr = baseline.truth
    expected = cal.to_series().groupby(cal.to_period("M")).max()
    assert tr.periods.equals(pd.DatetimeIndex(expected.values))
    assert len(tr.periods) == cal.to_period("M").nunique()
    assert tr.periods.is_monotonic_increasing and tr.periods.is_unique


# ---------------------------------------------------------------------------
# Step 1: factors
# ---------------------------------------------------------------------------
def test_daily_factor_sums_equal_period_factors(baseline: SimulatedData):
    tr = baseline.truth
    cal = tr.f_daily.index
    sums = tr.f_daily.groupby(cal.to_period("M")).sum()
    np.testing.assert_allclose(sums.values, tr.f_period.values, rtol=0, atol=1e-14)


def test_population_factor_moments_match_config(baseline: SimulatedData):
    cfg, tr = baseline.config, baseline.truth
    vols = np.asarray(cfg.factor_vol_annual[: cfg.K])
    np.testing.assert_allclose(tr.Sigma_ff_period, np.diag(vols**2 / 12.0), atol=1e-16)
    # daily moments use the realised mean number of trading days per period of the business-day
    # calendar (about 21.7), not the nominal 252 / 12 = 21
    n_days, T = tr.f_daily.shape[0], len(tr.periods)
    d_bar = n_days / T
    assert 21.5 < d_bar < 22.0 and d_bar != 21.0
    np.testing.assert_allclose(tr.Sigma_ff_daily, tr.Sigma_ff_period / d_bar, atol=1e-18)
    # sqrt(12 mu' Sigma^-1 mu) = mve_sharpe_annual, and mu along Sigma^{1/2} 1
    assert annualized_sharpe(tr.mu_f_period, tr.Sigma_ff_period, 12.0) == pytest.approx(cfg.mve_sharpe_annual, abs=1e-12)
    assert tr.sharpe_mve_true == pytest.approx(cfg.mve_sharpe_annual, abs=1e-12)
    direction = tr.mu_f_period / np.sqrt(np.diag(tr.Sigma_ff_period))
    np.testing.assert_allclose(direction, direction[0], rtol=1e-12)
    assert tr.meta["d_bar"] == pytest.approx(d_bar) == pytest.approx(tr.meta["mean_days_per_period"])
    assert tr.meta["periods_per_year"] == 12.0


def test_realised_daily_factor_moments(baseline: SimulatedData):
    tr = baseline.truth
    f = tr.f_daily.values
    n = f.shape[0]
    S = np.cov(f.T, ddof=1)
    # variances: relative sampling std sqrt(2/n) ~ 2 %
    np.testing.assert_allclose(np.diag(S), np.diag(tr.Sigma_ff_daily), rtol=0.10)
    corr = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    assert np.all(np.abs(corr - np.eye(3)) < 4.0 / np.sqrt(n))
    # means: sampling std of the mean is sd/sqrt(n); allow 4 sigma
    mu_d = tr.mu_f_period / tr.meta["d_bar"]
    assert np.all(np.abs(f.mean(axis=0) - mu_d) < 4.0 * np.sqrt(np.diag(tr.Sigma_ff_daily) / n))


def test_factor_ar1_keeps_unconditional_moments():
    cfg = replace(scenario_config("balanced"), n_assets=10, n_topics=5, n_relevant=3, n_placebo=0, factor_ar1=0.5)
    tr = simulate(cfg).truth
    f = tr.f_daily.values
    np.testing.assert_allclose(f.var(axis=0, ddof=1), np.diag(tr.Sigma_ff_daily), rtol=0.15)
    dev = f - f.mean(axis=0)
    ac1 = (dev[1:] * dev[:-1]).sum(axis=0) / (dev**2).sum(axis=0)
    np.testing.assert_allclose(ac1, 0.5, atol=0.06)


def test_states_are_factors_plus_independent_noise(baseline: SimulatedData):
    cfg, tr = baseline.config, baseline.truth
    nu = tr.x_daily.values - tr.f_daily.values
    np.testing.assert_allclose(nu.var(axis=0, ddof=1), cfg.nontradable_share * np.diag(tr.Sigma_ff_daily), rtol=0.10)
    n = nu.shape[0]
    for k in range(cfg.K):
        assert abs(np.corrcoef(nu[:, k], tr.f_daily.values[:, k])[0, 1]) < 4.0 / np.sqrt(n)


# ---------------------------------------------------------------------------
# Step 3: topics, BKS Eq. 1 in the DGP
# ---------------------------------------------------------------------------
def test_eq1_shocks_load_on_states(balanced: SimulatedData):
    """cov(z, x) = A Sigma_x (BKS Eq. 1 with eta independent of x)."""
    cfg, tr = balanced.config, balanced.truth
    z, x = tr.z_daily.values, tr.x_daily.values
    L = z.shape[1]
    C = np.cov(z.T, x.T, ddof=1)[:L, L:]
    Sigma_x = (1.0 + cfg.nontradable_share) * tr.Sigma_ff_daily
    target = tr.A @ Sigma_x
    rel = tr.relevant
    assert _rel_err(C[rel], target[rel]) < 0.12
    # irrelevant and placebo rows: zero loading, so their covariance is pure sampling noise
    scale = np.linalg.norm(target[rel]) / np.sqrt(rel.sum())
    assert np.abs(C[~rel]).max() < 0.5 * scale
    assert np.all(tr.A[~rel] == 0.0)
    assert np.all(np.linalg.norm(tr.A[rel], axis=1) > 0.0)
    # Cov with the tradable part only uses Sigma_ff_daily (nu is orthogonal to f)
    Cf = np.cov(z.T, tr.f_daily.values.T, ddof=1)[:L, L:]
    assert _rel_err(Cf[rel], (tr.A @ tr.Sigma_ff_daily)[rel]) < 0.15


def test_signal_to_noise_of_relevant_topics_is_order_one(balanced: SimulatedData):
    """At signal_strength = 1 the expected signal share of a relevant topic's variance is
    1 / (1 + (topic_noise_vol s_l)^2) with s_l ~ U(0.5, 1.5): between 0.31 and 0.8."""
    cfg, tr = balanced.config, balanced.truth
    z, x = tr.z_daily.values, tr.x_daily.values
    Sigma_x = (1.0 + cfg.nontradable_share) * tr.Sigma_ff_daily
    rel = np.flatnonzero(tr.relevant)
    signal_var = np.einsum("lk,kj,lj->l", tr.A[rel], Sigma_x, tr.A[rel])
    total_var = z[:, rel].var(axis=0, ddof=1)
    share = signal_var / total_var
    # individual rows vary with the chi-squared-like a' Sigma_x a; the statement is about the typical row
    assert share.min() > 0.01 and share.max() < 0.97
    assert 0.2 < np.median(share) < 0.85


def test_placebo_topics_are_noise_with_matched_variance(balanced: SimulatedData):
    tr = balanced.truth
    scale = np.asarray(tr.meta["shock_scale"])
    zeta = tr.z_daily.values / scale[None, :]  # relative units
    v = zeta.var(axis=0, ddof=1)
    lo, hi = v[tr.relevant].min(), v[tr.relevant].max()
    assert np.all(v[tr.placebo] > 0.8 * lo) and np.all(v[tr.placebo] < 1.2 * hi)
    x = tr.x_daily.values
    n = x.shape[0]
    for l in np.flatnonzero(tr.placebo):
        for k in range(x.shape[1]):
            assert abs(np.corrcoef(zeta[:, l], x[:, k])[0, 1]) < 4.0 / np.sqrt(n)
    # placebo shocks are serially uncorrelated
    d = zeta[:, tr.placebo]
    ac1 = (d[1:] * d[:-1]).mean(axis=0) / d.var(axis=0)
    assert np.all(np.abs(ac1) < 4.0 / np.sqrt(n))


def test_placebos_without_relevant_topics_match_a_real_topic():
    """n_relevant = 0 is legal under the null; placebos then match a random non-placebo topic
    (BKS App. C.2: 'a randomly chosen real narrative'), and the random stream of every
    configuration with relevant topics is unchanged by the fallback."""
    cfg = replace(FAST, n_relevant=0, n_placebo=5, signal_strength=0.0)
    d = simulate(cfg)  # used to raise: rng.choice on an empty pool
    tr = d.truth
    assert tr.relevant.sum() == 0 and tr.placebo.sum() == 5 and np.all(tr.A == 0.0)
    scale = np.asarray(tr.meta["shock_scale"])
    zeta = tr.z_daily.values / scale[None, :]
    v = zeta.var(axis=0, ddof=1)
    real = ~tr.placebo
    assert np.all(v[tr.placebo] > 0.8 * v[real].min()) and np.all(v[tr.placebo] < 1.2 * v[real].max())
    # all placebos: nothing to match, the noise draw is kept
    d2 = simulate(replace(FAST, n_topics=4, n_relevant=0, n_placebo=4, signal_strength=0.0))
    assert np.all(d2.truth.placebo) and np.all(np.isfinite(d2.truth.z_daily.values))
    # unchanged draws when relevant topics exist
    pd.testing.assert_frame_equal(simulate(FAST).truth.z_daily, simulate(FAST).truth.z_daily)


def test_topic_noise_ar1_applies_to_non_placebo_topics_only():
    cfg = replace(FAST, n_years=8, topic_noise_ar1=0.5, unbalanced_fraction=0.0, missing_day_fraction=0.0)
    tr = simulate(cfg).truth
    z = tr.z_daily.values - tr.z_daily.values.mean(axis=0)
    ac1 = (z[1:] * z[:-1]).sum(axis=0) / (z**2).sum(axis=0)
    noise = ~tr.relevant & ~tr.placebo
    np.testing.assert_allclose(ac1[noise], 0.5, atol=0.08)
    assert np.all(np.abs(ac1[tr.placebo]) < 0.1)


def test_topic_null_scenario_has_no_signal_but_keeps_the_factor_structure():
    d = simulate(scenario_config("topic_null"))
    tr = d.truth
    assert np.all(tr.A == 0.0)
    assert np.all(tr.Gamma_tilde_true == 0.0)
    assert np.all(tr.impact_z_to_mve_true == 0.0)
    assert tr.meta["n_relevant_effective"] == 0
    assert tr.meta["scenario"] == "topic_null"
    assert tr.relevant.sum() == d.config.n_relevant  # nominal mask kept for the harness's chance level
    assert np.all(tr.z_daily.values.std(axis=0) > 0.0)
    # returns keep their priced factor structure: loadings and the population systematic R2 of the baseline
    assert tr.meta["factor_structure"] is True
    assert np.nanstd(tr.beta) > 0.0 and tr.systematic_r2 > 0.1
    assert tr.sharpe_mve_true == pytest.approx(d.config.mve_sharpe_annual)
    # no topic covaries with the states beyond sampling noise
    z, x = tr.z_daily.values, tr.x_daily.values
    n = z.shape[0]
    zs = (z - z.mean(0)) / z.std(0)
    xs = (x - x.mean(0)) / x.std(0)
    corr = zs.T @ xs / n
    assert np.abs(corr).max() < 5.0 / np.sqrt(n)


def test_null_is_an_alias_of_topic_null():
    assert scenario_config("null") == scenario_config("topic_null")
    assert scenario_config("NULL-fast") == scenario_config("topic_null", fast=True)
    assert scenario_config("null", base=FAST) == replace(FAST, signal_strength=0.0)
    assert sim.SCENARIO_ALIASES == {"null": "topic_null"} and "null" not in sim.SCENARIOS
    # the recorded name is the inferred canonical one unless given explicitly
    assert simulate(scenario_config("null", base=FAST)).truth.meta["scenario"] == "topic_null"
    assert simulate(scenario_config("null", base=FAST), scenario="null").truth.meta["scenario"] == "null"


def test_no_factor_scenario_has_no_signal_and_no_factor_structure():
    cfg = scenario_config("no_factor", base=FAST)
    assert cfg.signal_strength == 0.0 and cfg.beta_innov_sd == 0.0
    for spec, base_spec in zip(cfg.asset_classes, FAST.asset_classes):
        assert spec.name == base_spec.name and spec.share == base_spec.share
        assert spec.idio_vol_annual == base_spec.idio_vol_annual
        assert all(b == 0.0 for b in spec.beta_mean) and spec.beta_sd == 0.0
    d = simulate(cfg)
    tr = d.truth
    assert tr.meta["scenario"] == "no_factor" and tr.meta["factor_structure"] is False
    assert np.all(tr.A == 0.0) and np.all(tr.Gamma_tilde_true == 0.0)
    # beta is identically zero inside the universe and NaN outside it
    finite = np.isfinite(tr.beta)
    assert np.all(tr.beta[finite] == 0.0) and finite.any() and not finite.all()
    period_id = pd.factorize(tr.f_daily.index.to_period("M"), sort=True)[0]
    obs = d.returns.returns.notna().values
    in_universe = np.zeros((len(tr.periods), cfg.n_assets), dtype=bool)
    np.logical_or.at(in_universe, period_id, obs)
    assert np.array_equal(finite.all(axis=2), in_universe)
    assert tr.systematic_r2 == 0.0
    # returns are pure idiosyncratic noise with the class volatilities: no correlation with the factors,
    # daily std idio_vol / sqrt(12 d_bar)
    r = d.returns.returns
    vol = pd.Series(tr.asset_class).map({c.name: c.idio_vol_annual for c in cfg.asset_classes}).values
    np.testing.assert_allclose(r.std(ddof=1).values, vol / np.sqrt(12.0 * tr.meta["d_bar"]), rtol=0.15)
    corr = np.array([[r[a].corr(tr.f_daily[k]) for k in tr.f_daily.columns] for a in r.columns])  # pairwise, NaN-aware
    n_obs = r.notna().sum().to_numpy()  # unbalanced assets have short lives: bound per asset with its own count
    assert np.all(np.isfinite(corr)) and np.all(np.abs(corr) < 5.0 / np.sqrt(n_obs)[:, None])
    # same random stream as topic_null: identical attention levels and calendar, only the returns differ
    tn = simulate(scenario_config("topic_null", base=FAST))
    pd.testing.assert_frame_equal(d.attention.levels, tn.attention.levels)
    pd.testing.assert_frame_equal(tr.f_daily, tn.truth.f_daily)
    assert not np.allclose(r.fillna(0.0).values, tn.returns.returns.fillna(0.0).values)
    assert r.notna().equals(tn.returns.returns.notna())


# ---------------------------------------------------------------------------
# Step 4: attention levels
# ---------------------------------------------------------------------------
def test_additive_levels_positive_and_clipping_below_threshold(caplog):
    with caplog.at_level(logging.WARNING, logger="narrative_ipca.simulation"):
        d = simulate(SimulationConfig(seed=3))
    theta = d.attention.levels.values
    assert np.all(theta >= sim.CLIP_FLOOR)
    assert d.truth.meta["clipped_fraction"] < sim.CLIP_WARN_FRACTION
    assert not any("clipped" in r.getMessage() for r in caplog.records)
    # mean level per topic is m_l (sum_l m_l = attention_level_mean)
    m = np.asarray(d.truth.meta["attention_mean_level"])
    assert m.sum() == pytest.approx(d.config.attention_level_mean)
    np.testing.assert_allclose(theta.mean(axis=0), m, rtol=0.2)


def test_clipping_warning_fires_when_levels_go_negative(caplog):
    cfg = replace(FAST, attention_slow_vol=0.5)
    with caplog.at_level(logging.WARNING, logger="narrative_ipca.simulation"):
        d = simulate(cfg)
    assert d.truth.meta["clipped_fraction"] > sim.CLIP_WARN_FRACTION
    assert any("clipped" in r.getMessage() for r in caplog.records)
    assert np.all(d.attention.levels.values >= sim.CLIP_FLOOR)


def test_softmax_rows_sum_to_one():
    d = simulate(scenario_config("softmax"))
    theta = d.attention.levels.values
    np.testing.assert_allclose(theta.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(theta > 0.0)
    assert d.truth.meta["scenario"] == "softmax"
    assert d.truth.meta["clipped_fraction"] == 0.0


@pytest.mark.parametrize("model", ["additive", "softmax"])
def test_truth_is_in_attention_units(model):
    """cov(theta, x) = A Sigma_x on relevant rows: the truth A maps states to the
    shock in the *observed* attention levels (exactly for additive, first order for softmax)."""
    cfg = replace(scenario_config("balanced"), attention_model=model, beta_innov_sd=0.0)
    d = simulate(cfg)
    tr = d.truth
    theta, x = d.attention.levels.values, tr.x_daily.values
    L = theta.shape[1]
    C = np.cov(theta.T, x.T, ddof=1)[:L, L:]
    Sigma_x = (1.0 + cfg.nontradable_share) * tr.Sigma_ff_daily
    target = tr.A @ Sigma_x
    rel = tr.relevant
    assert _rel_err(C[rel], target[rel]) < (0.15 if model == "additive" else 0.25)
    # z_daily is exactly what enters theta additively: theta - z has no covariance with x
    resid = theta - tr.z_daily.values
    Cr = np.cov(resid.T, x.T, ddof=1)[:L, L:]
    tol = 0.15 if model == "additive" else 0.35
    assert np.linalg.norm(Cr[rel]) < tol * np.linalg.norm(target[rel])


# ---------------------------------------------------------------------------
# Step 5: assets, BKS Eq. 4-5 in the DGP
# ---------------------------------------------------------------------------
def test_eq5_covariance_instruments_equal_beta_sigma_a(balanced: SimulatedData):
    """Cov(r_i, z) = beta_i Sigma_ff_daily A' (BKS Eq. 5) for assets of every class."""
    tr = balanced.truth
    r = balanced.returns.returns.values
    z = tr.z_daily.values
    beta_bar = np.nanmean(tr.beta, axis=0)  # constant over time here (beta_innov_sd = 0)
    np.testing.assert_allclose(np.nanstd(tr.beta, axis=0), 0.0, atol=1e-14)
    rel = tr.relevant
    n = r.shape[0]
    var_z = z.var(axis=0, ddof=1)
    classes = pd.Series(tr.asset_class)
    errors = []
    for name in classes.unique():
        for i in classes.index[classes == name][:2]:
            emp = np.cov(r[:, i], z.T, ddof=1)[0, 1:]
            theo = beta_bar[i] @ tr.Sigma_ff_daily @ tr.A.T
            err = _rel_err(emp[rel], theo[rel])
            # CLT size of the sampling error of the covariance vector, relative to the target;
            # assets with a small loading draw have a small target and a larger relative error
            predicted = np.sqrt((r[:, i].var(ddof=1) * var_z[rel] / n).sum()) / np.linalg.norm(theo[rel])
            assert err < max(0.15, 3.0 * predicted), (name, i, err, predicted)
            errors.append(err)
            # Gamma_tilde_true inverts Eq. 5: cov Gamma_tilde = beta
            np.testing.assert_allclose(theo @ tr.Gamma_tilde_true, beta_bar[i], atol=1e-12)
            assert _rel_err(emp @ tr.Gamma_tilde_true, beta_bar[i]) < max(0.35, 5.0 * predicted)
    assert np.mean(errors) < 0.15


def test_returns_follow_factor_model(balanced: SimulatedData):
    cfg, tr = balanced.config, balanced.truth
    r = balanced.returns.returns.values
    f = tr.f_daily.values
    cal = tr.f_daily.index
    period_id = pd.factorize(cal.to_period("M"), sort=True)[0]
    beta_daily = tr.beta[period_id]  # (n_days, N, K)
    resid = r - np.einsum("tik,tk->ti", beta_daily, f)
    vol = pd.Series(tr.asset_class).map({c.name: c.idio_vol_annual for c in cfg.asset_classes}).values
    d_bar = tr.meta["d_bar"]
    sd_daily = vol / np.sqrt(12.0 * d_bar)  # so that the period idiosyncratic variance is exactly vol^2 / 12
    np.testing.assert_allclose(resid.std(axis=0, ddof=1), sd_daily, rtol=0.06)
    assert np.abs(resid.mean(axis=0)).max() < 4.0 * sd_daily.max() / np.sqrt(r.shape[0])
    # the summed period residual has variance vol^2 / 12 (relative sampling std sqrt(2 / T) ~ 9 %)
    period_resid = pd.DataFrame(resid, index=cal).groupby(cal.to_period("M")).sum().to_numpy()
    np.testing.assert_allclose(period_resid.var(axis=0, ddof=1).mean() / np.mean(vol**2 / 12.0), 1.0, rtol=0.1)
    # residuals are orthogonal to the factors
    corr = np.corrcoef(resid.T, f.T)[: r.shape[1], r.shape[1] :]
    assert np.abs(corr).max() < 5.0 / np.sqrt(r.shape[0])


def test_long_run_loadings_follow_class_means(balanced: SimulatedData):
    cfg, tr = balanced.config, balanced.truth
    beta_bar = np.nanmean(tr.beta, axis=0)
    for spec in cfg.asset_classes:
        rows = beta_bar[tr.asset_class == spec.name]
        mean = np.asarray(spec.beta_mean, dtype=float)[: cfg.K]
        assert np.all(np.abs(rows.mean(axis=0) - mean) < 4.0 * spec.beta_sd / np.sqrt(len(rows)))
        np.testing.assert_allclose(rows.std(axis=0, ddof=1), spec.beta_sd, rtol=0.3)


def test_period_loadings_are_ar1_around_long_run(baseline: SimulatedData):
    cfg, tr = baseline.config, baseline.truth
    full = tr.beta[:, ~np.isnan(tr.beta).any(axis=(0, 2)), :]  # always-in-universe assets
    dev = full - full.mean(axis=0, keepdims=True)
    stationary_sd = cfg.beta_innov_sd / np.sqrt(1.0 - cfg.beta_ar1**2)
    assert dev.std(ddof=1) == pytest.approx(stationary_sd, rel=0.15)
    ac1 = (dev[1:] * dev[:-1]).sum() / (dev**2).sum()
    assert ac1 == pytest.approx(cfg.beta_ar1, abs=0.05)


def test_fat_tails_keep_the_idiosyncratic_std():
    cfg = replace(scenario_config("balanced"), n_assets=40, n_topics=10, n_relevant=3, n_placebo=2, fat_tails_df=4.0, beta_innov_sd=0.0)
    d = simulate(cfg)
    tr = d.truth
    r = d.returns.returns.values
    period_id = pd.factorize(tr.f_daily.index.to_period("M"), sort=True)[0]
    resid = r - np.einsum("tik,tk->ti", tr.beta[period_id], tr.f_daily.values)
    vol = pd.Series(tr.asset_class).map({c.name: c.idio_vol_annual for c in cfg.asset_classes}).values
    np.testing.assert_allclose(resid.std(axis=0, ddof=1), vol / np.sqrt(12.0 * tr.meta["d_bar"]), rtol=0.15)
    s = resid / resid.std(axis=0)
    assert (s**4).mean() > 4.0  # excess kurtosis (normal = 3)
    with pytest.raises(ValueError):
        simulate(replace(FAST, fat_tails_df=2.0))


def test_asset_class_counts_match_shares(baseline: SimulatedData):
    cfg = baseline.config
    counts = baseline.returns.asset_meta["asset_class"].value_counts()
    total = sum(c.share for c in cfg.asset_classes)
    for spec in cfg.asset_classes:
        assert abs(counts[spec.name] - spec.share / total * cfg.n_assets) <= 1
    assert counts.sum() == cfg.n_assets
    np.testing.assert_array_equal(baseline.truth.asset_class, baseline.returns.asset_meta["asset_class"].values)
    # unnormalised shares and rounding remainder
    classes = (AssetClassSpec("a", 2.0, 0.2, (1.0,)), AssetClassSpec("b", 1.0, 0.2, (1.0,)), AssetClassSpec("c", 1.0, 0.2, (1.0,)))
    d = simulate(replace(FAST, n_assets=7, asset_classes=classes))
    assert d.truth.meta["class_counts"] == {"a": 3, "b": 2, "c": 2}


# ---------------------------------------------------------------------------
# Step 6: unbalanced panel
# ---------------------------------------------------------------------------
def test_unbalanced_panel_entry_exit_and_missing_days(baseline: SimulatedData):
    cfg, tr = baseline.config, baseline.truth
    r = baseline.returns.returns
    meta = baseline.returns.asset_meta
    obs = r.notna().values
    cal = r.index
    first = cal[np.argmax(obs, axis=0)]
    last = cal[len(cal) - 1 - np.argmax(obs[::-1], axis=0)]
    assert (meta["entry"].values == first.values).all()
    assert (meta["exit"].values == last.values).all()
    # inside the life window, the missing-day rate matches the config
    day = np.arange(len(cal))[:, None]
    alive = (day >= np.argmax(obs, axis=0)[None, :]) & (day <= (len(cal) - 1 - np.argmax(obs[::-1], axis=0))[None, :])
    missing_rate = 1.0 - obs[alive].mean()
    assert missing_rate == pytest.approx(cfg.missing_day_fraction, rel=0.1)
    # beta is NaN exactly in the periods where the asset has no observed day
    period_id = pd.factorize(cal.to_period("M"), sort=True)[0]
    in_universe = np.zeros((len(tr.periods), cfg.n_assets), dtype=bool)
    np.logical_or.at(in_universe, period_id, obs)
    assert np.array_equal(~np.isnan(tr.beta).any(axis=2), in_universe)
    assert 0.5 < tr.meta["observed_fraction"] < 1.0


def test_half_of_unbalanced_assets_enter_late_and_half_exit_early():
    cfg = replace(FAST, n_assets=101, unbalanced_fraction=0.3, missing_day_fraction=0.0)
    d = simulate(cfg)
    meta, cal = d.returns.asset_meta, d.returns.returns.index
    n_unb = round(cfg.unbalanced_fraction * cfg.n_assets)  # 30
    late = (meta["entry"] > cal[0]).sum()
    early = (meta["exit"] < cal[-1]).sum()
    assert late == n_unb // 2 and early == n_unb - n_unb // 2
    assert ((meta["entry"] == cal[0]) & (meta["exit"] == cal[-1])).sum() == cfg.n_assets - n_unb
    obs = d.returns.returns.notna().values
    # membership is one contiguous block per asset (no gaps without missing days)
    changes = np.abs(np.diff(obs.astype(int), axis=0)).sum(axis=0)
    assert np.all(changes <= 1)


def test_balanced_scenario_has_no_missing_values():
    d = simulate(scenario_config("balanced", base=FAST))
    assert not d.returns.returns.isna().any().any()
    assert not np.isnan(d.truth.beta).any()
    assert (d.returns.asset_meta["entry"] == d.returns.returns.index[0]).all()
    assert (d.returns.asset_meta["exit"] == d.returns.returns.index[-1]).all()
    assert d.truth.meta["scenario"] == "balanced"


# ---------------------------------------------------------------------------
# Step 7: truth objects
# ---------------------------------------------------------------------------
def test_truth_identities(baseline: SimulatedData):
    cfg, tr = baseline.config, baseline.truth
    A = tr.A
    AtA_inv = np.linalg.inv(A.T @ A)
    np.testing.assert_allclose(tr.Gamma_tilde_true, A @ AtA_inv @ np.linalg.inv(tr.Sigma_ff_daily), rtol=1e-10)
    # Eq. 5 inversion: Sigma_ff_daily A' Gamma_tilde = I_K  (so beta Sigma A' Gamma_tilde = beta)
    np.testing.assert_allclose(tr.Sigma_ff_daily @ A.T @ tr.Gamma_tilde_true, np.eye(cfg.K), atol=1e-10)
    b_mve = np.linalg.solve(tr.Sigma_ff_period, tr.mu_f_period)
    np.testing.assert_allclose(tr.impact_z_to_mve_true, A @ AtA_inv @ b_mve, rtol=1e-10)
    # impact of the shock vector A x_tau on the MVE state is b_mve' x_tau
    x = tr.x_daily.values[:50]
    np.testing.assert_allclose((x @ A.T) @ tr.impact_z_to_mve_true, x @ b_mve, rtol=1e-8)
    assert np.all(tr.Gamma_tilde_true[~tr.relevant] == 0.0) and np.all(tr.impact_z_to_mve_true[~tr.relevant] == 0.0)
    # realised Sharpe of the true MVE portfolio of true period factors is near the population value
    mve = tr.f_period.values @ b_mve
    realised = mve.mean() / mve.std(ddof=1) * np.sqrt(12.0)
    assert tr.meta["sharpe_mve_realized"] == pytest.approx(realised)
    T = len(tr.periods)
    assert abs(realised - tr.sharpe_mve_true) < 4.0 * np.sqrt((1.0 + tr.sharpe_mve_true**2 / 24.0) * 12.0 / T)


def test_systematic_r2_in_unit_interval_and_formula(baseline: SimulatedData):
    assert 0.0 < baseline.truth.systematic_r2 < 1.0
    # single class, no dispersion, constant loadings: closed form beta' Sigma beta / (beta' Sigma beta + vol^2 / 12)
    spec = AssetClassSpec("only", 1.0, 0.20, (1.0, 0.5, -0.25), beta_sd=0.0)
    cfg = replace(FAST, asset_classes=(spec,), beta_innov_sd=0.0)
    tr = simulate(cfg).truth
    beta = np.array(spec.beta_mean)
    q = beta @ tr.Sigma_ff_period @ beta
    assert tr.systematic_r2 == pytest.approx(q / (q + spec.idio_vol_annual**2 / 12.0), rel=1e-12)
    # time-varying loadings add beta_innov_sd^2/(1-ar1^2) tr(Sigma) of systematic variance
    tr2 = simulate(replace(cfg, beta_innov_sd=0.05, beta_ar1=0.9)).truth
    q2 = q + 0.05**2 / (1 - 0.9**2) * np.trace(tr.Sigma_ff_period)
    assert tr2.systematic_r2 == pytest.approx(q2 / (q2 + spec.idio_vol_annual**2 / 12.0), rel=1e-12)


# ---------------------------------------------------------------------------
# Scenarios, other periods, speed
# ---------------------------------------------------------------------------
def test_scenario_config_overrides():
    base = SimulationConfig()
    assert scenario_config("baseline") == base
    assert scenario_config("topic_null") == replace(base, signal_strength=0.0)
    assert scenario_config("null").signal_strength == 0.0
    nf = scenario_config("no_factor")
    assert nf.signal_strength == 0.0 and nf.beta_innov_sd == 0.0
    assert nf == replace(base, signal_strength=0.0, beta_innov_sd=0.0, asset_classes=nf.asset_classes)
    assert all(spec.beta_sd == 0.0 and set(spec.beta_mean) == {0.0} for spec in nf.asset_classes)
    assert [spec.name for spec in nf.asset_classes] == [spec.name for spec in base.asset_classes]
    assert set(sim.SCENARIOS) == {"baseline", "topic_null", "no_factor", "softmax", "weak", "balanced"}
    assert scenario_config("softmax").attention_model == "softmax"
    weak = scenario_config("weak")
    assert weak.signal_strength == 0.35 and weak.mve_sharpe_annual == 0.6
    bal = scenario_config("balanced")
    assert bal.unbalanced_fraction == 0.0 and bal.missing_day_fraction == 0.0
    # everything else is inherited from the base
    custom = SimulationConfig(seed=7, n_assets=150, n_topics=40, n_relevant=8, n_placebo=8, n_years=8)
    weak_custom = scenario_config("Weak", base=custom)
    assert weak_custom == replace(custom, signal_strength=0.35, mve_sharpe_annual=0.6)
    # fast variants
    fast = scenario_config("null-fast")
    assert fast == replace(SimulationConfig(), **sim.FAST_OVERRIDES, signal_strength=0.0)
    assert scenario_config("fast") == replace(SimulationConfig(), **sim.FAST_OVERRIDES)
    assert scenario_config("softmax", fast=True).n_assets == 150
    with pytest.raises(ValueError):
        scenario_config("no-such-scenario")


def test_scenario_name_override_and_inference():
    d = simulate(FAST, scenario="my-run")
    assert d.truth.meta["scenario"] == "my-run"
    assert simulate(replace(FAST, signal_strength=0.35, mve_sharpe_annual=0.6)).truth.meta["scenario"] == "weak"
    assert simulate(replace(FAST, signal_strength=0.0, attention_model="softmax")).truth.meta["scenario"] == "topic_null"
    assert simulate(scenario_config("no_factor", base=FAST)).truth.meta["scenario"] == "no_factor"
    assert simulate(replace(scenario_config("no_factor", base=FAST), beta_innov_sd=0.05)).truth.meta["scenario"] == "topic_null"


def test_weekly_periods():
    cfg = replace(FAST, period="W")
    d = simulate(cfg)
    tr = d.truth
    cal = tr.f_daily.index
    assert len(tr.periods) == cal.to_period("W").nunique()
    n_days = len(cal)
    assert tr.meta["periods_per_year"] == 52.0 and tr.meta["d_bar"] == pytest.approx(n_days / len(tr.periods))
    assert 4.5 < tr.meta["d_bar"] <= 5.0
    np.testing.assert_allclose(tr.Sigma_ff_daily, tr.Sigma_ff_period / tr.meta["d_bar"])
    np.testing.assert_allclose(tr.Sigma_ff_period, np.diag(np.array(cfg.factor_vol_annual) ** 2 / 52.0))
    assert annualized_sharpe(tr.mu_f_period, tr.Sigma_ff_period, 52.0) == pytest.approx(cfg.mve_sharpe_annual)
    sums = tr.f_daily.groupby(cal.to_period("W")).sum()
    np.testing.assert_allclose(sums.values, tr.f_period.values, atol=1e-14)
    assert tr.beta.shape[0] == len(tr.periods)


def test_factor_vol_padding_and_truncation():
    tr = simulate(replace(FAST, K=4, factor_vol_annual=(0.16, 0.08), n_relevant=5)).truth
    np.testing.assert_allclose(np.sqrt(np.diag(tr.Sigma_ff_period) * 12.0), [0.16, 0.08, 0.08, 0.08])
    tr = simulate(replace(FAST, K=2, factor_vol_annual=(0.16, 0.08, 0.06))).truth
    np.testing.assert_allclose(np.sqrt(np.diag(tr.Sigma_ff_period) * 12.0), [0.16, 0.08])
    assert tr.A.shape == (FAST.n_topics, 2) and tr.beta.shape[2] == 2


def test_fast_config_runs_quickly():
    t0 = time.perf_counter()
    d = simulate(FAST)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0
    assert d.attention.levels.shape == (3 * 252, 20)
    assert d.returns.returns.shape == (3 * 252, 60)
