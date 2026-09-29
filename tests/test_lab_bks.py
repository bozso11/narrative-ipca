"""Tests of the lab's BKS wrapper (DESIGN.md G.7.2, G.8; D52, D65, D70): narrative_ipca/exposure_lab/bks.py.

A small generic simulation (30 artificial assets, 12 generic topics of which
6 carry links, weekly periods over 2015-2025) runs end to end in about a
second; it is built once per module. The real-data smoke test is skipped when
``data/market`` is missing.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from narrative_ipca.exposure_lab import bks, reference
from narrative_ipca.exposure_lab.config import (
    BKSLabConfig,
    ExposureConfig,
    LabConfig,
    TopicSetConfig,
    UniverseConfig,
    WindowConfig,
)
from narrative_ipca.exposure_lab.dgp import simulate_lab
from narrative_ipca.exposure_lab.types import BKSLabResult
from narrative_ipca.oos import oos_factor


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _generic_cfg(lead: int = 0, n_betas: int = 1, bks_cfg: BKSLabConfig | None = None) -> LabConfig:
    """Generic universe of 30 assets, 12 generic topics (6 linked), beta_1 = 0.5 (seed 0)."""
    return LabConfig(
        universe=UniverseConfig(asset_source="generic", n_generic_assets=30, seed=0),
        topics=TopicSetConfig(manual="none", n_generic=12, generic_signal_share=0.5),
        exposure=ExposureConfig(beta_1=0.5, n_betas=n_betas, lead_days=lead, seed=0),
        bks=bks_cfg if bks_cfg is not None else BKSLabConfig(),
    )


@pytest.fixture(scope="module")
def cfg() -> LabConfig:
    return _generic_cfg()


@pytest.fixture(scope="module")
def sim(cfg):
    return simulate_lab(cfg)


@pytest.fixture(scope="module")
def panel(sim, cfg):
    return bks.build_bks_panel(sim, cfg.bks, cfg.window.shock_window)


@pytest.fixture(scope="module")
def fit(panel, cfg):
    return bks.fit_bks(panel, cfg.bks, cfg.window.train_end, train_start=cfg.window.train_start)


@pytest.fixture(scope="module")
def result(panel, fit, cfg) -> BKSLabResult:
    return bks.evaluate_bks(panel, fit, cfg.window)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_pipeline_config_fields():
    b = BKSLabConfig()
    pc = bks.bks_pipeline_config(b, shock_window=5, lead_days=0, n_assets=55)
    assert pc.data.period == "W"
    assert pc.data.attention_lag_days == 0
    assert pc.data.asset_weighting == "inverse_vol"
    assert pc.data.min_assets_per_period == 20
    assert pc.shocks.window == 5
    assert pc.covariance.xi == pytest.approx(b.xi_weekly)
    assert pc.covariance.xi == pytest.approx(0.9977, abs=5e-5)  # G.7.2: 69 months half-life
    assert pc.covariance.burn_in_periods == 52
    assert pc.covariance.min_days == 60
    assert pc.estimation.K == 3
    assert pc.estimation.lam is None
    assert pc.estimation.lam_grid.n_lambdas == 12
    assert pc.estimation.lam_grid.ratio == pytest.approx(1e-2)
    assert pc.estimation.penalize_intercept is True
    assert pc.estimation.max_iter == 300
    assert pc.tuning.criterion == "is_sharpe"
    assert pc.tuning.tolerance == pytest.approx(0.02)
    assert pc.oos.enabled is False
    assert pc.evaluation.annualization == 52.0
    assert pc.run_wrapup is False


@pytest.mark.parametrize("n_assets, expected", [(None, 20), (2, 2), (3, 2), (30, 15), (55, 20), (500, 20)])
def test_pipeline_config_min_assets(n_assets, expected):
    pc = bks.bks_pipeline_config(BKSLabConfig(), 5, 0, n_assets=n_assets)
    assert pc.data.min_assets_per_period == expected


def test_pipeline_config_rules():
    arg = bks.bks_pipeline_config(BKSLabConfig(lambda_rule="argmax"), 3, 1)
    assert arg.tuning.tolerance == 0.0 and arg.estimation.lam is None
    assert arg.shocks.window == 3 and arg.data.attention_lag_days == 1
    fixed = bks.bks_pipeline_config(BKSLabConfig(lambda_rule="fixed", lam=0.3), 5, 0)
    assert fixed.estimation.lam == pytest.approx(0.3) and fixed.tuning.tolerance == 0.0
    none_w = bks.bks_pipeline_config(BKSLabConfig(asset_weighting="none", half_life_months=12.0), 5, 0)
    assert none_w.data.asset_weighting == "none"
    assert none_w.covariance.xi == pytest.approx(0.5 ** (1.0 / 52.0))
    with pytest.raises(ValueError):
        bks.bks_pipeline_config(BKSLabConfig(), 0, 0)


# ---------------------------------------------------------------------------
# end to end on the generic simulation
# ---------------------------------------------------------------------------
def test_panel_shapes(panel, sim, cfg):
    p = panel.panel
    assert p.L == 12 and p.N == 30
    assert list(p.topics) == sim.topics.ids
    assert list(p.assets) == [str(a) for a in sim.market.assets.index]
    # weekly periods: last trading day of each week is a Friday on the weekday calendar, except the
    # truncated last week of the data (2025-12-31 is a Wednesday)
    periods = pd.DatetimeIndex(p.periods)
    assert (periods[:-1].dayofweek == 4).all()
    assert periods[-1] == pd.Timestamp(cfg.universe.end)
    # burn-in of 52 weeks (plus the inverse-vol warm-up of the returns)
    assert p.periods[0] > pd.Timestamp(cfg.universe.start) + pd.Timedelta(weeks=52)
    assert panel.meta["panel_params"]["shock_window"] == cfg.window.shock_window
    assert panel.meta["topic_labels"]["G001"] == "Generic topic 001"
    assert panel.aligned.scale is not None  # inverse_vol is the lab default (D70)


def test_end_to_end(result, fit, cfg):
    r = result
    assert isinstance(r, BKSLabResult)
    fs, fe = pd.Timestamp(cfg.window.forecast_start), cfg.window.forecast_end
    assert len(r.periods) == cfg.window.forecast_weeks  # the window starts on a Monday
    assert ((r.periods >= fs) & (r.periods <= fe)).all()
    assert r.fitted.shape == (len(r.periods), 30) and r.realized.shape == r.fitted.shape
    assert r.contrib.shape == (30, 12) and list(r.contrib.columns) == [f"G{j:03d}" for j in range(1, 13)]
    assert np.isfinite(r.r2_pooled)
    assert r.r2.notna().all()
    assert np.isfinite(r.in_sample_total_r2) and 0.0 < r.in_sample_total_r2 < 1.0
    assert r.K == 3 and r.lam == pytest.approx(fit.lam)
    assert r.path is not None and len(r.path) == 12
    assert r.lam in set(r.path["lam"])
    assert set(r.selected_topics) == set(r.gamma_norms.index[r.gamma_norms > 0])
    assert r.meta["warnings"][0] == bks.D52_NOTE
    assert "approximate" in r.meta["units"]
    assert set(r.meta["timings"]) >= {"panel", "fit", "evaluate"}
    # training weeks only (D65)
    assert fit.train_periods.max() <= pd.Timestamp(cfg.window.train_end)
    assert fit.train_periods.min() >= pd.Timestamp(cfg.window.train_start)
    assert fit.train_periods.equals(pd.DatetimeIndex(fit.fit.periods))
    assert r.periods.min() > fit.train_periods.max()


def test_identity_per_asset_week(result):
    """const + sum of topic terms == fitted value for every asset-week (panel units), and summed (return units)."""
    weekly = result.meta["contrib_weekly_panel"]
    fitted_p = result.meta["fitted_panel"].stack().rename_axis(["period", "asset_id"])
    total = weekly.sum(axis=1)
    assert len(total) == len(fitted_p) == int(result.meta["n_assets_per_period"].sum())
    np.testing.assert_allclose(total.to_numpy(), fitted_p.reindex(total.index).to_numpy(), rtol=0, atol=1e-8)
    # in return units, summed over the window
    lhs = result.const_contrib + result.contrib.sum(axis=1)
    np.testing.assert_allclose(lhs.to_numpy(), result.fitted.sum(axis=0).to_numpy(), rtol=0, atol=1e-10)


def test_oos_factor_and_r2(panel, fit, result):
    """Fitted values come from each forecast week's own cross-section with the frozen Gamma; R2 in panel units."""
    p = panel.panel
    slices = dict(p.period_slices())
    periods = pd.DatetimeIndex(p.periods)
    Gamma = fit.fit.Gamma
    for end in result.periods:
        t = int(periods.get_loc(end))
        sl = slices[t]
        f = oos_factor(p.X[sl], p.y[sl], Gamma)
        np.testing.assert_allclose(result.meta["factors"].loc[end].to_numpy(), f, rtol=1e-12, atol=1e-14)
        assets = [str(a) for a in p.assets[p.asset_idx[sl]]]
        np.testing.assert_allclose(result.meta["realized_panel"].loc[end, assets].to_numpy(), p.y[sl])
    fp, rp = result.meta["fitted_panel"], result.meta["realized_panel"]
    r2 = 1.0 - ((rp - fp) ** 2).sum(axis=0) / (rp**2).sum(axis=0)
    np.testing.assert_allclose(result.r2.to_numpy(), r2.to_numpy(), rtol=1e-12)
    pooled = 1.0 - float(((rp - fp) ** 2).sum().sum()) / float((rp**2).sum().sum())
    assert result.r2_pooled == pytest.approx(pooled, rel=1e-12)


def test_units_conversion(result):
    div = result.meta["divisor"]
    np.testing.assert_allclose(result.fitted.to_numpy(), (result.meta["fitted_panel"] * div).to_numpy())
    np.testing.assert_allclose(result.realized.to_numpy(), (result.meta["realized_panel"] * div).to_numpy())
    # the mean-divisor conversion is close to the exact weekly return (trailing 252-day vol barely moves in a week)
    exact = result.meta["realized_exact"].to_numpy()
    approx = result.realized.to_numpy()
    scale = np.nanstd(exact)
    assert np.nanmax(np.abs(approx - exact)) < 0.05 * scale


def test_selects_linked_topics(sim, result):
    """At least half of the design-linked topics are selected (selection by the in-sample Sharpe is seed-dependent, D27/D47)."""
    linked = set(sim.links.table["topic_id"])
    assert len(linked) == 6
    selected = set(result.selected_topics)
    assert len(selected & linked) >= len(linked) / 2


def test_fixed_lambda_rule(panel, fit, cfg):
    lam = fit.lam
    bcfg = replace(cfg.bks, lambda_rule="fixed", lam=lam)
    calls: list[tuple[int, int, str]] = []
    f2 = bks.fit_bks(panel, bcfg, cfg.window.train_end, progress=lambda d, t, m: calls.append((d, t, m)),
                     train_start=cfg.window.train_start)
    assert f2.tuning is None and f2.lam == pytest.approx(lam) and f2.meta["lambda_rule"] == "fixed"
    assert f2.meta["lam_max"] is None
    assert calls and calls[-1][0] == calls[-1][1] == 1
    r2 = bks.evaluate_bks(panel, f2, cfg.window)
    assert r2.path is None and r2.lam == pytest.approx(lam)
    assert np.isfinite(r2.r2_pooled) and len(r2.selected_topics) >= 1


def test_argmax_rule_not_sparser_than_tolerance(panel, fit, cfg):
    """The tolerance rule picks the largest lambda within 2% of the best Sharpe, so lambda_tol >= lambda_argmax."""
    f_arg = bks.fit_bks(panel, replace(cfg.bks, lambda_rule="argmax"), cfg.window.train_end,
                        train_start=cfg.window.train_start)
    assert f_arg.meta["tolerance"] == 0.0 and fit.meta["tolerance"] == pytest.approx(0.02)
    assert fit.lam >= f_arg.lam
    assert fit.tuning is not None and f_arg.tuning is not None
    assert fit.tuning.meta["best_criterion"] <= f_arg.tuning.meta["best_criterion"] + 1e-12


def test_lead_one_lags_attention():
    c = _generic_cfg(lead=1)
    s = simulate_lab(c)
    assert s.lead_days == 1
    pan = bks.build_bks_panel(s, c.bks, c.window.shock_window)
    assert pan.pipeline_cfg.data.attention_lag_days == 1
    assert pan.meta["lead_days"] == 1
    # day tau carries the attention of day tau - 1 (D4)
    expected = s.attention.shift(1).reindex(pan.aligned.calendar)
    pd.testing.assert_frame_equal(pan.aligned.attention, expected, check_names=False, check_freq=False)
    res = bks.evaluate_bks(pan, bks.fit_bks(pan, c.bks, c.window.train_end), c.window)
    assert np.isfinite(res.r2_pooled) and res.meta["lead_days"] == 1


def test_no_period_in_window_raises(panel, fit):
    # the forecast window lies after the data
    late = WindowConfig(forecast_start="2026-01-05", forecast_weeks=2)
    with pytest.raises(ValueError, match="no complete weekly period ends inside the forecast window"):
        bks.evaluate_bks(panel, fit, late)
    # a shortened window (Monday to Wednesday) holds no week's last trading day
    short = SimpleNamespace(forecast_start="2023-01-02", forecast_end=pd.Timestamp("2023-01-04"))
    with pytest.raises(ValueError, match="no complete weekly period ends inside the forecast window"):
        bks.evaluate_bks(panel, fit, short)


def test_forecast_overlapping_training_raises(panel, fit):
    early = WindowConfig(train_end="2020-12-31", forecast_start="2021-01-04", forecast_weeks=4)
    with pytest.raises(ValueError, match="not after the last training week"):
        bks.evaluate_bks(panel, fit, early)


def test_short_training_window_raises(panel, cfg):
    with pytest.raises(ValueError, match="weekly periods end inside the training window"):
        bks.fit_bks(panel, cfg.bks, "2016-03-31")


def test_midweek_forecast_start_warns(panel, fit):
    w = WindowConfig(forecast_start="2023-01-04", forecast_weeks=1)  # Wednesday
    r = bks.evaluate_bks(panel, fit, w)
    assert list(r.periods) == [pd.Timestamp("2023-01-06")]
    assert r.meta["period_first_day"][0] == pd.Timestamp("2023-01-02")
    assert any("before forecast_start" in m for m in r.meta["warnings"])
    assert not any("fewer than 5 trading days" in m for m in r.meta["warnings"])
    # the window's Monday and Tuesday (2023-01-09, -10) are in no evaluated week: recorded, with the span
    assert any("2023-01-09 to 2023-01-10 (2 trading day(s)) are in no evaluated week" in m for m in r.meta["warnings"])
    assert list(r.meta["window_days_not_evaluated"]) == [pd.Timestamp("2023-01-09"), pd.Timestamp("2023-01-10")]
    assert r.meta["evaluated_span"] == (pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-06"))
    # a Monday start has no uncovered day
    r2 = bks.evaluate_bks(panel, fit, WindowConfig(forecast_start="2023-01-02", forecast_weeks=2))
    assert len(r2.meta["window_days_not_evaluated"]) == 0
    assert not any("no evaluated week" in m for m in r2.meta["warnings"])


def test_truncated_last_week_warns(panel, fit):
    w = WindowConfig(forecast_start="2025-12-29", forecast_weeks=1)
    r = bks.evaluate_bks(panel, fit, w)
    assert list(r.periods) == [pd.Timestamp("2025-12-31")]
    assert any("fewer than 5 trading days: 2025-12-31 (3 days)" in m for m in r.meta["warnings"])


def test_panel_mismatch_is_recorded(panel, cfg):
    other = replace(cfg.bks, half_life_months=24.0)
    f = bks.fit_bks(panel, other, cfg.window.train_end)
    assert "xi" in f.meta["panel_mismatch"] and "half_life_months" in f.meta["panel_mismatch"]
    assert any("different settings" in m for m in f.meta["warnings"])


def test_run_bks_matches_stages_and_is_deterministic(sim, cfg, result):
    r = bks.run_bks(sim, cfg.bks, cfg.window)
    assert r.r2_pooled == result.r2_pooled
    assert r.selected_topics == result.selected_topics
    pd.testing.assert_frame_equal(r.contrib, result.contrib)
    pd.testing.assert_frame_equal(r.fitted, result.fitted)


def test_no_asset_weighting_gives_exact_units(sim, cfg):
    b = replace(cfg.bks, asset_weighting="none")
    pan = bks.build_bks_panel(sim, b, cfg.window.shock_window)
    assert pan.aligned.scale is None
    r = bks.evaluate_bks(pan, bks.fit_bks(pan, b, cfg.window.train_end), cfg.window)
    assert r.meta["units"] == "return units (exact)"
    div = r.meta["divisor"].to_numpy()
    assert np.all(div[np.isfinite(div)] == 1.0)
    np.testing.assert_allclose(r.realized.to_numpy(), r.meta["realized_exact"].to_numpy(), atol=1e-14)
    lhs = r.const_contrib + r.contrib.sum(axis=1)
    np.testing.assert_allclose(lhs.to_numpy(), r.fitted.sum(axis=0).to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------
# real data smoke
# ---------------------------------------------------------------------------
def _real_data_available() -> bool:
    try:
        return (reference.market_dir() / "asset_returns.parquet").is_file() and (
            reference.reference_dir() / "assets.csv"
        ).is_file()
    except Exception:  # pragma: no cover - a broken data folder counts as missing
        return False


@pytest.mark.skipif(not _real_data_available(), reason="data/market or data/reference missing")
def test_real_data_smoke(caplog):
    caplog.set_level(logging.WARNING)
    c = LabConfig(topics=TopicSetConfig(manual="both", n_generic=0))
    s = simulate_lab(c)
    r = bks.run_bks(s, c.bks, c.window)
    assert r.fitted.shape[1] == 55 and r.contrib.shape == (55, 20)
    assert r.gamma_norms.shape == (20,)
    assert np.isfinite(r.r2_pooled)
    present = r.realized.notna().any(axis=0)
    assert present.sum() >= 50
    assert r.contrib.loc[present].notna().all().all()
    lhs = r.const_contrib + r.contrib.sum(axis=1)
    np.testing.assert_allclose(
        lhs[present].to_numpy(), r.fitted.loc[:, present].sum(axis=0).to_numpy(), rtol=0, atol=1e-10
    )
    assert r.meta["timings"]["fit"] < 30.0


# ---------------------------------------------------------------------------
# Review fixes 2026-09-29 (D79)
# ---------------------------------------------------------------------------
def test_lambda_zero_uses_ridge_zero_in_the_oos_factor(panel, cfg):
    """At lambda 0 (plain IPCA, Gamma'Gamma = I) the OOS factor has no ridge; the R2 matches a tiny lambda."""
    res = {}
    for lam in (0.0, 1e-6):
        b = replace(cfg.bks, lambda_rule="fixed", lam=lam)
        f = bks.fit_bks(panel, b, cfg.window.train_end, train_start=cfg.window.train_start)
        res[lam] = bks.evaluate_bks(panel, f, cfg.window)
        assert res[lam].meta["oos_ridge"] == (0.0 if lam == 0.0 else 2.0)
    assert res[0.0].r2_pooled > 0.1  # was about 0 with ridge 2 on a unit-norm Gamma
    assert res[0.0].r2_pooled == pytest.approx(res[1e-6].r2_pooled, abs=0.01)


def test_shuffled_instrument_reference(panel, fit, cfg, result):
    """The reference R2 with shuffled topic instruments is below the real one, deterministic and per asset."""
    ref = result.meta["shuffled_r2_pooled"]
    assert np.isfinite(ref) and ref < result.r2_pooled
    assert result.meta["n_shuffles"] == bks.N_SHUFFLES
    assert result.meta["shuffled_r2"].index.equals(result.r2.index)
    again = bks.evaluate_bks(panel, fit, cfg.window)
    assert again.meta["shuffled_r2_pooled"] == ref


def test_k_not_below_the_number_of_assets_raises():
    c = LabConfig(
        universe=UniverseConfig(asset_source="generic", n_generic_assets=3, seed=0),
        topics=TopicSetConfig(manual="none", n_generic=12, generic_signal_share=0.5),
        exposure=ExposureConfig(beta_1=0.5, n_betas=1, seed=0),
        bks=BKSLabConfig(K=3),
    )
    pan = bks.build_bks_panel(simulate_lab(c), c.bks, c.window.shock_window)
    with pytest.raises(ValueError, match="K = 3 factors need more than 3 assets"):
        bks.fit_bks(pan, c.bks, c.window.train_end)
    assert bks.fit_bks(pan, replace(c.bks, K=2), c.window.train_end).K == 2


def test_panel_does_not_keep_the_covariance_array(panel):
    assert not hasattr(panel, "cov") and not hasattr(panel, "shock_panel")
    assert panel.meta["shapes"]["covariance_array"][1:] == (30, 12)


@pytest.mark.slow
@pytest.mark.skipif(not _real_data_available(), reason="data/market or data/reference missing")
def test_bks_500_topics_completes_on_the_listed_assets():
    """G.10 scale check (deselected by default; run with -m slow): 55 listed assets x 500 generic topics."""
    import time

    c = LabConfig(topics=TopicSetConfig(manual="none", n_generic=500, generic_signal_share=0.2))
    t0 = time.perf_counter()
    s = simulate_lab(c)
    r = bks.run_bks(s, c.bks, c.window)
    elapsed = time.perf_counter() - t0
    assert r.gamma_norms.shape == (500,) and np.isfinite(r.r2_pooled)
    assert np.isfinite(r.meta["shuffled_r2_pooled"])
    assert elapsed < 180.0, f"BKS at 500 topics took {elapsed:.0f} s"
