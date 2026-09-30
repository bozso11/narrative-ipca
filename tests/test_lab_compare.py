"""Tests of the method comparison and the BKS-implied exposures (DESIGN.md G.7, G.8; D52, D65, D74).

Files under test: ``narrative_ipca/exposure_lab/compare.py``,
``bks.implied_exposures`` and the ``bks_implied`` and ``comparison`` stages of
``session.py``.

Small generic simulations (30 artificial assets, 5 to 12 generic topics,
weekly BKS panels over 2015-2025) keep every test to a second or two. The
numerical checks of the BKS-implied exposures are:

1. units: ``beta_i = c_i Gamma`` equals ``sparse_ipca.betas`` and, with the
   fit's factors, reproduces the fit's in-sample total R2 (``Gamma`` is in the
   units of ``panel.X``);
2. algebra: with ``K = L`` and ``lambda = 0``, ``m_i = cov_i' + (Gamma_tilde')^-1 Gamma_0'``;
3. signal: positive rank correlation with ``B_true`` and sign agreement above
   one half on the strongly linked pairs (measured 2026-09-29: Spearman 0.49,
   sign agreement 0.60 on the 48 linked pairs and 0.88 on the 8 strong-tier
   pairs; weak, because the ``K = 3`` directions the fit keeps hold little of the topic signal,
   DESIGN.md G.15.1);
4. lead 1;
5. units conversion: without the constant instrument's term, the implied
   exposures of the ``"none"`` and ``"inverse_vol"`` panels agree and match
   OLS (measured max differences 0.026 and 0.029).

The training-window covariance history (D88) has its own checks: nothing
before ``train_start - w`` weekdays reaches it (perturbation), its
instruments are kernel covariances of the training window (by hand), plain
IPCA on it is close to OLS, and its cache keys hold the training window.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from narrative_ipca.exposure_lab import bks, reference
from narrative_ipca.exposure_lab.compare import (
    METHOD_LABELS,
    METHODS,
    ORACLE_NOTE,
    SUMMARY_COLUMNS,
    ComparisonResult,
    compare_methods,
    method_config,
    method_label,
)
from narrative_ipca.exposure_lab.config import (
    BKSLabConfig,
    DirectConfig,
    ExposureConfig,
    LabConfig,
    TopicSetConfig,
    UniverseConfig,
    WindowConfig,
)
from narrative_ipca.exposure_lab.dgp import observed_shocks, simulate_lab, truth_for_window
from narrative_ipca.exposure_lab.direct import fit_direct
from narrative_ipca.exposure_lab.evaluate import evaluate_window, median_finite, recovery_metrics, window_sweep
from narrative_ipca.exposure_lab.session import BKS_NOT_RUN, BKS_OFF, BKS_REFUSED, LabSession, run_lab
from narrative_ipca.exposure_lab.types import DirectFit
from narrative_ipca.sparse_ipca import betas as ipca_betas
from narrative_ipca.sparse_ipca import fitted_values


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def _cfg(
    n_topics: int = 8,
    share: float = 1.0,
    lead: int = 0,
    bks_cfg: BKSLabConfig | None = None,
    window: WindowConfig | None = None,
) -> LabConfig:
    """30 generic assets; ``n_topics`` generic topics, ``share`` of them linked; every link at beta_1 = 0.5."""
    return LabConfig(
        universe=UniverseConfig(asset_source="generic", n_generic_assets=30, seed=0),
        topics=TopicSetConfig(manual="none", n_generic=n_topics, generic_signal_share=share),
        exposure=ExposureConfig(beta_1=0.5, n_betas=1, lead_days=lead, seed=0),
        bks=bks_cfg if bks_cfg is not None else BKSLabConfig(),
        window=window if window is not None else WindowConfig(),
    )


@dataclasses.dataclass
class Route:
    cfg: LabConfig
    sim: object
    shocks: object
    truth: object
    panel: bks.BKSPanel
    fit: bks.BKSFit
    implied: DirectFit


def _route(cfg: LabConfig) -> Route:
    sim = simulate_lab(cfg)
    w = cfg.window
    shocks = observed_shocks(sim.attention, w.shock_window, w.train_start, w.train_end)
    truth = truth_for_window(sim, w.shock_window)
    panel = bks.build_bks_panel(sim, cfg.bks, w.shock_window, train_start=w.train_start, train_end=w.train_end)
    fit = bks.fit_bks(panel, cfg.bks, w.train_end, train_start=w.train_start)
    imp = bks.implied_exposures(panel, fit, sim, shocks, select_tau=cfg.direct.select_tau)
    return Route(cfg, sim, shocks, truth, panel, fit, imp)


def _last_train_rows(r: Route) -> np.ndarray:
    pnl = r.panel.panel
    t_last = pnl.periods.get_loc(r.fit.train_periods.max())
    return np.flatnonzero(pnl.t_idx == t_last)


@pytest.fixture(scope="module")
def signal_route() -> Route:
    """The signal configuration: 8 generic topics, all linked, K = 3, no asset weighting."""
    return _route(_cfg(n_topics=8, bks_cfg=BKSLabConfig(K=3, asset_weighting="none")))


@pytest.fixture(scope="module")
def plain_routes() -> dict[str, Route]:
    """Plain IPCA (K = L = 8, lambda = 0) on the same simulation, with and without asset weighting."""
    out = {}
    for wt in ("none", "inverse_vol"):
        out[wt] = _route(_cfg(n_topics=8, bks_cfg=BKSLabConfig(K=8, asset_weighting=wt, lambda_rule="fixed", lam=0.0)))
    return out


# ---------------------------------------------------------------------------
# BKS-implied exposures: numerical checks
# ---------------------------------------------------------------------------
def test_betas_reproduce_sparse_ipca_and_gamma_is_in_panel_units(signal_route):
    r = signal_route
    rows = _last_train_rows(r)
    C = r.panel.panel.X[rows]
    Gamma = r.fit.fit.Gamma
    m, beta, rank = bks.implied_topic_covariance(C, Gamma, r.panel.pipeline_cfg.evaluation.rcond)
    np.testing.assert_array_equal(beta, ipca_betas(C, Gamma))
    assert m.shape == (len(rows), r.panel.panel.L) and rank == int(r.fit.K)
    # Gamma is in the units of panel.X: X Gamma f reproduces the fit's in-sample total R2
    keep = np.asarray(r.panel.panel.periods.isin(r.fit.train_periods), dtype=bool)
    sub = r.panel.panel.subset_periods(keep)
    fv = fitted_values(sub, Gamma, r.fit.fit.F)
    total_r2 = 1.0 - np.sum((sub.y - fv) ** 2) / np.sum(sub.y**2)
    assert total_r2 == pytest.approx(r.fit.fit.total_r2, abs=1e-10)
    # topics the group lasso dropped get zero implied covariance
    dropped = ~np.asarray(r.fit.fit.selected, dtype=bool)
    assert np.all(m[:, dropped] == 0.0)
    # the rows used are those of the last training week
    assert r.implied.meta["train_period"] == r.fit.train_periods.max()
    assert r.implied.meta["stale_assets"] == [] and r.implied.meta["skipped_assets"] == []


def test_square_gamma_gives_covariance_plus_constant_term():
    """K = L, lambda = 0 (plain IPCA): m_i = cov_i' + (Gamma_tilde')^-1 Gamma_0' (BKS Eq. 5 inverted exactly)."""
    r = _route(_cfg(n_topics=5, bks_cfg=BKSLabConfig(K=5, asset_weighting="none", lambda_rule="fixed", lam=0.0)))
    assert r.fit.lam == 0.0 and r.fit.K == 5
    rows = _last_train_rows(r)
    C = r.panel.panel.X[rows]
    Gamma = r.fit.fit.Gamma
    m, _, rank = bks.implied_topic_covariance(C, Gamma, r.panel.pipeline_cfg.evaluation.rcond)
    assert rank == 5
    expected = C[:, 1:] + np.linalg.solve(Gamma[1:].T, Gamma[0])[None, :]
    scale = float(np.abs(expected).max())
    np.testing.assert_allclose(m, expected, rtol=0, atol=1e-8 * scale)


def test_implied_exposures_carry_signal(signal_route):
    """Spearman > 0 with B_true and sign agreement above one half on the strongly linked pairs (weak, see module)."""
    r = signal_route
    imp, truth = r.implied, r.truth
    rec = recovery_metrics(imp, truth, 0.05)
    B = imp.B_hat.to_numpy()
    Bt = truth.B_true.reindex(index=imp.B_hat.index, columns=imp.B_hat.columns).to_numpy()
    W = truth.W_unscaled.reindex(index=imp.B_hat.index, columns=imp.B_hat.columns).to_numpy()
    strong = np.abs(W) >= 0.5 - 1e-12  # n_betas = 1: every link is at beta_1 = 0.5
    assert strong.sum() == 48
    sign_strong = float(np.mean(np.sign(B[strong]) == np.sign(Bt[strong])))
    assert rec["spearman"] > 0.0
    assert sign_strong > 0.5
    # the DirectFit contract of the comparison
    assert imp.method == "bks_implied" and imp.B_hat.shape == (8, 30)
    assert imp.penalty.isna().all() and (imp.intercept == 0.0).all()
    pd.testing.assert_frame_equal(imp.selected, imp.B_hat.abs() >= 0.05)
    assert imp.meta["caveat"] == bks.IMPLIED_NOTE and "D52" in imp.meta["caveat"]
    assert imp.meta["K"] == 3 and imp.meta["selected_topics"] == r.fit.fit.selected_topics
    # the same training scales as the direct fit, so the oracle line is shared (D74)
    d = fit_direct(r.sim, r.shocks, DirectConfig(), r.truth)
    pd.testing.assert_series_equal(imp.ret_scale, d.ret_scale)
    pd.testing.assert_series_equal(imp.ret_mean, d.ret_mean)
    pd.testing.assert_series_equal(imp.n_train, d.n_train)
    assert imp.meta["last_return_day"] <= r.shocks.train_end
    # the forecast window evaluation applies unchanged
    ev = evaluate_window(r.sim, r.shocks, imp, r.cfg.window, r.truth)
    assert np.isfinite(ev.r2).all()


def test_implied_exposures_with_lead_one():
    r = _route(_cfg(n_topics=8, lead=1, bks_cfg=BKSLabConfig(K=3, asset_weighting="none")))
    assert r.panel.pipeline_cfg.data.attention_lag_days == 1 and r.implied.meta["lead_days"] == 1
    rec = recovery_metrics(r.implied, r.truth, 0.05)
    assert rec["spearman"] > 0.0  # 0.57 when measured
    ev = evaluate_window(r.sim, r.shocks, r.implied, r.cfg.window, r.truth)
    assert np.isfinite(ev.r2).all()


def test_units_conversion_matches_ols_and_both_weightings(plain_routes):
    """Without the constant term, 'none' and 'inverse_vol' agree and match OLS (K = L, lambda = 0)."""
    a, b = plain_routes["none"], plain_routes["inverse_vol"]
    assert a.implied.meta["divisor"].eq(1.0).all()
    assert (b.implied.meta["divisor"] < 0.05).all()  # daily vol of the artificial returns
    ols = fit_direct(a.sim, a.shocks, DirectConfig(method="ols"), a.truth).B_hat
    net_a = a.implied.B_hat - a.implied.meta["B_const"]
    net_b = b.implied.B_hat - b.implied.meta["B_const"]
    assert float((net_a - ols).abs().to_numpy().max()) < 0.05  # 0.029 when measured
    assert float((net_b - ols).abs().to_numpy().max()) < 0.05  # 0.023
    assert float((net_a - net_b).abs().to_numpy().max()) < 0.05  # 0.026
    # with the constant term the two differ more (0.39 when measured): plain IPCA keeps Gamma_0
    assert a.fit.fit.gamma_norms[0] > 0.0


def test_implied_exposures_missing_rows():
    """An asset absent in the last training week uses its latest row; one absent in training gets zeros."""
    cfg = _cfg(n_topics=8, bks_cfg=BKSLabConfig(K=3, asset_weighting="none"))
    sim = simulate_lab(cfg)
    rets = sim.market.returns.copy()
    a_gone, a_stale = rets.columns[0], rets.columns[1]
    rets.loc[:"2022-12-30", a_gone] = np.nan
    rets.loc["2022-12-26":"2022-12-30", a_stale] = np.nan  # the last training week
    sim = dataclasses.replace(sim, market=dataclasses.replace(sim.market, returns=rets))
    w = cfg.window
    shocks = observed_shocks(sim.attention, w.shock_window, w.train_start, w.train_end)
    panel = bks.build_bks_panel(sim, cfg.bks, w.shock_window)
    fit = bks.fit_bks(panel, cfg.bks, w.train_end, train_start=w.train_start)
    imp = bks.implied_exposures(panel, fit, sim, shocks)
    assert imp.meta["skipped_assets"] == [a_gone]
    assert (imp.B_hat[a_gone] == 0.0).all() and not imp.selected[a_gone].any()
    assert imp.meta["stale_assets"] == [a_stale]
    assert imp.meta["row_period"][a_stale] == pd.Timestamp("2022-12-23")
    assert pd.isna(imp.meta["row_period"][a_gone])
    assert np.isfinite(imp.B_hat.to_numpy()).all()


def test_implied_exposures_refuse_inconsistent_inputs(signal_route):
    r = signal_route
    other = observed_shocks(r.sim.attention, 3, r.cfg.window.train_start, r.cfg.window.train_end)
    with pytest.raises(ValueError, match="shock window"):
        bks.implied_exposures(r.panel, r.fit, r.sim, other)
    early = observed_shocks(r.sim.attention, 5, r.cfg.window.train_start, "2020-12-31")
    with pytest.raises(ValueError, match="D65"):
        bks.implied_exposures(r.panel, r.fit, r.sim, early)


# ---------------------------------------------------------------------------
# No data after the training end reaches BKS or its implied exposures (D65)
# ---------------------------------------------------------------------------
def _perturb_after(sim, train_end: str):
    """Returns and attention after ``train_end`` replaced by other values (the forecast data change)."""
    te = pd.Timestamp(train_end)
    rng = np.random.default_rng([7, 2])
    ret = sim.market.returns.copy()
    after_r = ret.index > te
    ret.loc[after_r] = ret.loc[after_r] * 3.0 + rng.normal(0.0, 0.01, size=(int(after_r.sum()), ret.shape[1]))
    att = sim.attention.copy()
    after_a = att.index > te
    att.loc[after_a] = att.loc[after_a] * 2.0 + np.abs(rng.normal(0.0, 1.0, size=(int(after_a.sum()), att.shape[1])))
    return dataclasses.replace(sim, market=dataclasses.replace(sim.market, returns=ret), attention=att)


def _bks_route(cfg: LabConfig, sim) -> tuple[bks.BKSPanel, bks.BKSFit, DirectFit]:
    w = cfg.window
    shocks = observed_shocks(sim.attention, w.shock_window, w.train_start, w.train_end)
    panel = bks.build_bks_panel(sim, cfg.bks, w.shock_window, train_start=w.train_start, train_end=w.train_end)
    fit = bks.fit_bks(panel, cfg.bks, w.train_end, train_start=w.train_start)
    return panel, fit, bks.implied_exposures(panel, fit, sim, shocks, select_tau=cfg.direct.select_tau)


@pytest.mark.parametrize("history", ["full", "training"])
@pytest.mark.parametrize(("lead", "train_end", "forecast_start"), [(0, "2022-12-30", "2023-01-02"),
                                                                   (1, "2022-12-28", "2022-12-29")])
def test_bks_route_ignores_data_after_the_training_end(lead, train_end, forecast_start, history):
    """Perturbing returns and attention after the cut-off leaves Gamma, the implied exposures and the
    training scales unchanged (lead 1 with a Wednesday cut-off: the boundary week), mirroring the direct
    estimator's test in test_lab_direct.py; for both covariance histories (D88)."""
    win = WindowConfig(train_start="2021-01-01", train_end=train_end, forecast_start=forecast_start)
    cfg = _cfg(n_topics=8, lead=lead, window=win, bks_cfg=BKSLabConfig(history=history))  # inverse_vol weighting
    sim = simulate_lab(cfg)
    panel, fit, imp = _bks_route(cfg, sim)
    sim2 = _perturb_after(sim, train_end)
    panel2, fit2, imp2 = _bks_route(cfg, sim2)
    # the perturbation reaches the panel after the cut-off ...
    after = np.asarray(panel.panel.periods > pd.Timestamp(train_end), dtype=bool)[panel.panel.t_idx]
    assert not np.allclose(panel.panel.y[after], panel2.panel.y[after])
    # ... and nothing the comparison scores
    np.testing.assert_array_equal(fit.fit.Gamma, fit2.fit.Gamma)
    np.testing.assert_array_equal(imp.B_hat.to_numpy(), imp2.B_hat.to_numpy())
    np.testing.assert_array_equal(imp.ret_scale.to_numpy(), imp2.ret_scale.to_numpy())
    np.testing.assert_array_equal(imp.ret_mean.to_numpy(), imp2.ret_mean.to_numpy())
    assert imp.meta["last_return_day"] <= pd.Timestamp(train_end)
    assert imp.method == bks.IMPLIED_METHODS[history]
    # lead 1: the burn-in check and the kernel-history share use the same calendar as the panel
    n, first = bks.training_weeks(win.train_start, win.train_end, cfg.bks, lead)
    assert first == panel.panel.periods[0] and n == int(fit.meta["n_train_periods"])
    assert imp.meta["kernel_share_before_train"] == pytest.approx(
        bks.kernel_history_share(win.train_start, win.train_end, cfg.bks, lead), abs=1e-12)


def test_training_weeks_include_the_burn_in(plain_routes):
    """training_weeks counts the weeks fit_bks can use without building a panel (D17: 52 burn-in weeks)."""
    for route in plain_routes.values():
        b = route.cfg.bks
        periods = route.panel.panel.periods
        n, first = bks.training_weeks("2015-01-02", "2016-12-30", b)
        assert first == periods[0]
        assert n == int(np.sum(np.asarray(periods <= pd.Timestamp("2016-12-30"), dtype=bool)))
    # the lab's data: the first usable week ends 2016-04-08 with inverse_vol weighting
    n, first = bks.training_weeks("2015-07-01", "2015-12-31", BKSLabConfig())
    assert (n, first) == (0, pd.Timestamp("2016-04-08"))
    assert bks.training_weeks("2025-01-01", "2025-06-30", BKSLabConfig())[0] == 26
    with pytest.raises(ValueError, match="only 0 weekly periods"):
        bks.fit_bks(plain_routes["inverse_vol"].panel, plain_routes["inverse_vol"].cfg.bks, "2015-12-31",
                    train_start="2015-07-01")


def test_kernel_history_share():
    """Share of the instruments' kernel weight before the training start (G.15): about 92% on the
    dashboard defaults, 0 when the window starts at the data start, and smaller for longer windows."""
    b = BKSLabConfig()
    six = bks.kernel_history_share("2025-01-01", "2025-06-30", b)
    assert 0.90 < six < 0.94  # 0.9225 when measured
    assert bks.kernel_history_share("2015-01-02", "2022-12-30", b) == 0.0
    two_years = bks.kernel_history_share("2023-07-01", "2025-06-30", b)
    assert 0.0 < two_years < six
    # hand check: weekly weights xi^(weeks back) over the days of the panel's calendar
    days = pd.bdate_range("2015-04-01", "2025-12-31")  # the inverse_vol warm-up ends 2015-04-01
    week = pd.factorize(days.to_period("W"))[0]  # calendar weeks, numbered in order
    j = int(week[days.get_loc(pd.Timestamp("2025-06-20"))])  # the week before the last training week
    wts = np.where(week <= j, b.xi_weekly ** (j - week).clip(0), 0.0)
    wts[days == pd.Timestamp("2025-06-20")] = 0.0  # the week's last day is cut (skip_days = 1)
    assert six == pytest.approx(wts[days < pd.Timestamp("2025-01-01")].sum() / wts.sum(), rel=1e-12)
    # no usable training week: NaN
    assert np.isnan(bks.kernel_history_share("2015-07-01", "2015-12-31", b))


# ---------------------------------------------------------------------------
# Training-window covariance history (D88): BKS sees the data the direct methods see
# ---------------------------------------------------------------------------
def _perturb_before(sim, first_day: str, attention_only_on: str | None = None):
    """Returns and attention before ``first_day`` replaced by other values (or, with ``attention_only_on``,
    only the attention of that one day)."""
    rng = np.random.default_rng([7, 3])
    ret = sim.market.returns.copy()
    att = sim.attention.copy()
    if attention_only_on is not None:
        day = att.index == pd.Timestamp(attention_only_on)
        att.loc[day] = att.loc[day] * 2.0 + 0.05
    else:
        before_r = ret.index < pd.Timestamp(first_day)
        ret.loc[before_r] = ret.loc[before_r] * 3.0 + rng.normal(0.0, 0.01, size=(int(before_r.sum()), ret.shape[1]))
        before_a = att.index < pd.Timestamp(first_day)
        att.loc[before_a] = att.loc[before_a] * 2.0 + np.abs(
            rng.normal(0.0, 1.0, size=(int(before_a.sum()), att.shape[1])))
    return dataclasses.replace(sim, market=dataclasses.replace(sim.market, returns=ret), attention=att)


@pytest.mark.parametrize("lead", [0, 1])
def test_training_history_ignores_data_before_the_training_window(lead):
    """D88: returns and attention before train_start - w weekdays never reach the training-history BKS route:
    the panel instruments and returns, Gamma, the implied exposures and the training scales are bit-identical.
    The full history does use them; and the attention of the w-th weekday before train_start is used."""
    win = WindowConfig(train_start="2021-01-01", train_end="2021-12-31", forecast_start="2022-01-03")
    cfg = _cfg(n_topics=8, lead=lead, window=win, bks_cfg=BKSLabConfig(history="training"))
    sim = simulate_lab(cfg)
    first = "2020-12-25"  # 5 weekdays before 2021-01-01 (Friday) on the weekday calendar
    assert len(pd.bdate_range(first, "2020-12-31")) == cfg.window.shock_window
    w = cfg.window

    def route(s):
        # the direct estimator's shocks on the training days need no attention before `first` either; computed
        # from it, they are bit-identical under the perturbation (from the whole series, pandas' rolling mean
        # carries rounding of about 1e-16 from earlier days into later ones)
        shocks = observed_shocks(s.attention.loc[first:], w.shock_window, w.train_start, w.train_end)
        panel = bks.build_bks_panel(s, cfg.bks, w.shock_window, train_start=w.train_start, train_end=w.train_end)
        fit = bks.fit_bks(panel, cfg.bks, w.train_end, train_start=w.train_start)
        return panel, fit, bks.implied_exposures(panel, fit, s, shocks)

    panel, fit, imp = route(sim)
    sim2 = _perturb_before(sim, first)
    panel2, fit2, imp2 = route(sim2)
    p, p2 = panel.panel, panel2.panel
    assert p.periods.equals(p2.periods)
    for name in ("X", "y", "t_idx", "asset_idx"):
        np.testing.assert_array_equal(getattr(p, name), getattr(p2, name))
    np.testing.assert_array_equal(fit.fit.Gamma, fit2.fit.Gamma)
    np.testing.assert_array_equal(imp.B_hat.to_numpy(), imp2.B_hat.to_numpy())
    np.testing.assert_array_equal(imp.ret_scale.to_numpy(), imp2.ret_scale.to_numpy())
    np.testing.assert_array_equal(imp.ret_mean.to_numpy(), imp2.ret_mean.to_numpy())
    assert panel.meta["first_input_day"] == pd.Timestamp(first)
    # with the lab's own shocks (from the whole attention series) the exposures agree to rounding
    imp_lab, imp_lab2 = _bks_route(cfg, sim)[2], _bks_route(cfg, sim2)[2]
    np.testing.assert_allclose(imp_lab2.B_hat.to_numpy(), imp_lab.B_hat.to_numpy(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(imp_lab.B_hat.to_numpy(), imp.B_hat.to_numpy(), rtol=1e-12, atol=1e-14)
    # the full history reads those days
    full = dataclasses.replace(cfg, bks=BKSLabConfig())
    fp, fp2 = _bks_route(full, sim)[0].panel, _bks_route(full, sim2)[0].panel
    assert not np.array_equal(fp.X, fp2.X)
    # the boundary is tight: the attention of 2020-12-25 enters the first shock (2021-01-01)
    p3 = _bks_route(cfg, _perturb_before(sim, first, attention_only_on=first))[0].panel
    assert not np.array_equal(p.X, p3.X)


def test_training_history_instruments_are_kernel_covariances_of_the_training_window():
    """The last training week's instrument row is the kernel covariance (BKS App. B.1) of the scaled returns
    and the direct estimator's shocks z, over the days from train_start to the window end of the week before."""
    from narrative_ipca.covariances import brute_force_covariance, kernel_weights, window_bounds
    from narrative_ipca.data import period_end_index

    win = WindowConfig(train_start="2022-07-01", train_end="2022-12-30", forecast_start="2023-01-02")
    r = _route(_cfg(n_topics=5, window=win, bks_cfg=BKSLabConfig(history="training")))
    pan, fit = r.panel, r.fit
    days = pd.DatetimeIndex(pan.aligned.calendar)
    assert days[0] == pd.Timestamp("2022-07-01")
    pid, ends = period_end_index(days, "W")
    last = pd.Timestamp(fit.train_periods.max())
    j = int(ends.get_loc(last)) - 1  # the return week pairs with the instruments of the week before
    wts = kernel_weights(pid, j, float(pan.pipeline_cfg.covariance.xi))
    _, stop, cut = window_bounds(pid, int(pan.pipeline_cfg.covariance.skip_days))
    wts[cut[j]:stop[j]] = 0.0
    topics = [str(t) for t in pan.panel.topics]
    z = r.shocks.z.reindex(index=days, columns=topics)  # lead 0: the panel's shocks are the direct estimator's z
    rows = np.flatnonzero(pan.panel.t_idx == int(pan.panel.periods.get_loc(last)))
    assert len(rows) == 30
    for row in rows:
        a = str(pan.panel.assets[pan.panel.asset_idx[row]])
        expected = brute_force_covariance(pan.aligned.returns[a], z, wts)
        np.testing.assert_allclose(pan.panel.X[row, 1:], expected, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("lead", [0, 1])
def test_training_history_plain_ipca_is_close_to_ols(lead):
    """K = L = 8 and lambda = 0 (plain IPCA) on the training window only: without the constant's term the
    implied exposures are close to OLS on the same window, because the kernel is nearly flat over it
    (xi^26 = 0.94). They differ because the last instrument ends a week and a day before the cut-off and
    because of the kernel weights. Measured 2026-09-30, max |difference| over the 8 x 30 exposures:
    six months 0.074 (lead 0) and 0.104 (lead 1), against exposures up to 0.59; eight years 0.029 and 0.031."""
    b = BKSLabConfig(K=8, lambda_rule="fixed", lam=0.0, history="training")
    for win, tol in ((WindowConfig(train_start="2022-07-01", train_end="2022-12-30", forecast_start="2023-01-02"), 0.15),
                     (WindowConfig(), 0.05)):
        r = _route(_cfg(n_topics=8, lead=lead, window=win, bks_cfg=b))
        assert r.fit.lam == 0.0 and r.implied.method == "bks_implied_train"
        ols = fit_direct(r.sim, r.shocks, DirectConfig(method="ols"), r.truth).B_hat
        net = r.implied.B_hat - r.implied.meta["B_const"]
        assert float((net - ols).abs().to_numpy().max()) < tol, (win.train_start, lead)
        # the unit conversion is exact: the divisor is the training standard deviation of fit_direct
        np.testing.assert_allclose(r.implied.meta["divisor"].to_numpy(), r.implied.ret_scale.to_numpy(), rtol=1e-12)
        assert r.implied.meta["kernel_share_before_train"] == 0.0


def test_training_history_keys():
    """D71, D88: the training-history panel key holds the history, its own settings, the training window, w and
    the lead; the full-history panel key holds none of the window; fit, implied and comparison keys follow."""
    k = LabSession.stage_key
    full = _cfg(n_topics=8)
    tr = dataclasses.replace(full, bks=dataclasses.replace(full.bks, history="training"))

    def win(c, **kw):
        return dataclasses.replace(c, window=dataclasses.replace(c.window, **kw))

    def bk(c, **kw):
        return dataclasses.replace(c, bks=dataclasses.replace(c.bks, **kw))

    def lead(c):
        return dataclasses.replace(c, exposure=dataclasses.replace(c.exposure, lead_days=1))

    assert k("bks_panel", tr) != k("bks_panel", full)
    # the training start (and end: the divisor) enter the training panel only
    assert k("bks_panel", win(tr, train_start="2016-01-04")) != k("bks_panel", tr)
    assert k("bks_panel", win(full, train_start="2016-01-04")) == k("bks_panel", full)
    assert k("bks_panel", win(tr, train_end="2022-12-23")) != k("bks_panel", tr)
    assert k("bks_panel", win(full, train_end="2022-12-23")) == k("bks_panel", full)
    for c in (tr, full):
        assert k("bks_panel", win(c, shock_window=3)) != k("bks_panel", c)
        assert k("bks_panel", lead(c)) != k("bks_panel", c)
        assert k("bks_panel", win(c, forecast_start="2023-03-06")) == k("bks_panel", c)
        assert k("bks_fit", win(c, forecast_start="2023-03-06")) == k("bks_fit", c)
    # each history's own settings enter its panel key only
    assert k("bks_panel", bk(tr, min_days_training=5)) != k("bks_panel", tr)
    assert k("bks_panel", bk(full, min_days_training=5)) == k("bks_panel", full)
    assert k("bks_panel", bk(tr, burn_in_weeks=26)) == k("bks_panel", tr)
    assert k("bks_panel", bk(full, burn_in_weeks=26)) != k("bks_panel", full)
    # fit, implied and comparison keys follow
    assert k("bks_fit", tr) != k("bks_fit", full) and k("bks_implied", tr) != k("bks_implied", full)
    assert k("bks_fit", win(tr, train_start="2016-01-04")) != k("bks_fit", tr)
    assert k("comparison", full, bks_token={"bks_implied": "a", "bks_implied_train": "b"}) != k(
        "comparison", full, bks_token={"bks_implied": "a", "bks_implied_train": "c"})


def test_both_bks_variants_in_one_comparison():
    """D88: the full-history and the training-window BKS-implied exposures are separate methods with separate
    fits; the comparison never starts either, and use_bks can allow one of them."""
    cfg = _cfg(n_topics=12, share=0.5)
    methods = ("elastic_net", "bks_implied", "bks_implied_train", "oracle")
    mc = method_config(cfg, "bks_implied_train")
    assert method_config(cfg, "bks_implied") is cfg and mc.bks.history == "training"
    assert mc.bks == dataclasses.replace(cfg.bks, history="training") and mc.window == cfg.window
    assert method_config(mc, "bks_implied").bks.history == "full"
    assert METHOD_LABELS["bks_implied"] == "BKS-implied (full history)"
    assert METHOD_LABELS["bks_implied_train"] == "BKS-implied (training window)"

    s = LabSession()
    res0 = s.comparison(cfg, methods=methods)
    assert not res0.summary.loc[["bks_implied", "bks_implied_train"], "available"].any()
    assert (res0.summary.loc[["bks_implied", "bks_implied_train"], "note"] == BKS_NOT_RUN).all()
    assert s.stats["bks_fit"]["misses"] == 0 and s.stats["bks_panel"]["misses"] == 0
    s.bks_fit(mc)  # the training-window variant only
    assert s.bks_ready(cfg, "bks_implied_train") and not s.bks_ready(cfg, "bks_implied")
    key1 = s.comparison_key(cfg, methods)
    res1 = s.comparison(cfg, methods=methods)
    assert res1.summary.loc["bks_implied_train", "available"] and not res1.summary.loc["bks_implied", "available"]
    s.bks_fit(cfg)
    assert s.comparison_key(cfg, methods) != key1
    res2 = s.comparison(cfg, methods=methods)
    assert res2.summary["available"].all() and list(res2.summary.index) == list(methods)
    assert list(res2.summary["label"]) == ["Elastic net", "BKS-implied (full history)",
                                           "BKS-implied (training window)", "Oracle (true exposures)"]
    f_full, f_train = res2.fits["bks_implied"], res2.fits["bks_implied_train"]
    assert (f_full.method, f_train.method) == ("bks_implied", "bks_implied_train")
    assert f_full.meta["history"] == "full" and f_train.meta["history"] == "training"
    assert res2.summary.loc["bks_implied_train", "note"] == bks.IMPLIED_NOTE
    assert not np.allclose(f_full.B_hat.to_numpy(), f_train.B_hat.to_numpy())
    assert res2.meta["same_training_scales"]  # both on fit_direct's training pairs (D74)
    assert s.method_fit(cfg, "bks_implied_train") is f_train
    # use_bks as a collection: only the allowed variant is scored
    res3 = s.comparison(cfg, methods=methods, use_bks=("bks_implied_train",))
    assert res3.summary.loc["bks_implied", "note"] == BKS_OFF and res3.summary.loc["bks_implied_train", "available"]
    assert s.comparison_key(cfg, methods, use_bks=("bks_implied_train",)) != s.comparison_key(cfg, methods)


# ---------------------------------------------------------------------------
# compare_methods
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def session_run():
    cfg = _cfg(n_topics=12, share=0.5)
    s = LabSession()
    s.bks_fit(cfg)
    methods = ("elastic_net", "ridge", "oracle", "bks_implied")
    res = s.comparison(cfg, methods=methods)
    return s, cfg, methods, res


def test_method_config_and_labels():
    cfg = LabConfig(direct=DirectConfig(method="ridge", ridge_lambda=0.3, select_tau=0.1))
    assert method_config(cfg, "ridge") is cfg
    en = method_config(cfg, "elastic_net")
    assert en.direct == DirectConfig(method="elastic_net", select_tau=0.1)
    assert en.window == cfg.window and en.bks == cfg.bks
    assert method_config(cfg, "bks_implied") is cfg
    assert method_config(cfg, "bks_implied_train").bks.history == "training"
    assert METHODS == ("elastic_net", "ridge", "ols", "bks_implied", "bks_implied_train", "oracle")
    with pytest.raises(ValueError, match="unknown method"):
        method_config(cfg, "lasso")
    assert set(METHOD_LABELS) == set(METHODS) and METHODS[-1] == "oracle"
    assert method_label("ridge") == "Ridge (GCV)"
    assert method_label("ridge", SimpleNamespace(meta={"alpha_rule": "fixed"})) == "Ridge (fixed lambda)"
    assert method_label("elastic_net", SimpleNamespace(meta={"alpha_rule": "cv"})) == "Elastic net (CV)"
    assert method_label("elastic_net", SimpleNamespace(meta={"alpha_rule": "universal"})) == "Elastic net"


def test_compare_methods_numbers_are_consistent(session_run):
    s, cfg, methods, res = session_run
    assert isinstance(res, ComparisonResult)
    assert list(res.summary.index) == ["elastic_net", "ridge", "bks_implied", "oracle"]
    assert list(res.summary.columns) == list(SUMMARY_COLUMNS)
    assert res.summary["available"].all()
    sim, shocks, truth = s.simulation(cfg), s.shocks(cfg), s.truth(cfg)
    B_or = res.fits["oracle"].B_hat
    np.testing.assert_array_equal(B_or.to_numpy(), truth.B_true.loc[B_or.index, B_or.columns].to_numpy())
    for m in res.summary.index:
        fit = res.fits[m]
        ev = evaluate_window(sim, shocks, fit, cfg.window, truth)
        sw = window_sweep(sim, shocks, fit, cfg.window, truth)
        row = res.summary.loc[m]
        pd.testing.assert_series_equal(res.r2[m], ev.r2, check_names=False)
        assert row["r2_median_window"] == pytest.approx(median_finite(ev.r2), abs=1e-14)
        assert row["r2_pooled_window"] == pytest.approx(float(sw["pooled_r2"].iloc[0]), abs=1e-12)
        assert row["r2_median_all_windows"] == pytest.approx(median_finite(sw["median_r2"]), abs=1e-14)
        rec = recovery_metrics(fit, truth, cfg.direct.select_tau)
        for c in ("n_selected", "coverage", "sign_agreement", "mcc", "spearman", "rmse"):
            assert row[c] == pytest.approx(rec[c], abs=1e-14, nan_ok=True)
        # the oracle method's R2 is the evaluation's oracle line of every method (D74)
        np.testing.assert_allclose(res.r2["oracle"].to_numpy(), ev.r2_oracle.to_numpy(), rtol=0, atol=1e-12)
        if m != "oracle":
            above = np.mean(sw["median_r2"].to_numpy() > sw["median_r2_oracle"].to_numpy())
            assert row["share_windows_above_oracle"] == pytest.approx(above)
        assert row["fit_seconds"] >= 0.0
    assert np.isnan(res.summary.loc["oracle", "share_windows_above_oracle"])
    assert res.summary.loc["oracle", "note"] == ORACLE_NOTE
    assert res.summary.loc["bks_implied", "note"] == bks.IMPLIED_NOTE
    assert res.meta["same_training_scales"] and res.meta["unavailable"] == {}
    # the long sweep frame: one row per method and window
    n_win = res.meta["n_windows"]
    assert n_win == 39 and len(res.r2_sweep) == 4 * n_win
    assert set(res.r2_sweep.columns) == {"start", "end", "method", "median_r2"}
    assert res.r2_sweep["start"].min() == pd.Timestamp(cfg.window.forecast_start)


def test_compare_methods_lists_unavailable_methods(session_run):
    s, cfg, _, res = session_run
    sim, shocks, truth = s.simulation(cfg), s.shocks(cfg), s.truth(cfg)
    fits = {"elastic_net": res.fits["elastic_net"], "oracle": res.fits["oracle"]}
    # a fit whose training data reach into the forecast window is not scored (D65)
    leaky = dataclasses.replace(res.fits["ridge"], meta={**res.fits["ridge"].meta, "last_return_day": pd.Timestamp("2023-01-03")})
    out = compare_methods(
        sim, shocks, truth, cfg.window, {**fits, "ridge": leaky},
        unavailable={"ols": "refused", "bks_implied": BKS_NOT_RUN},
    )
    assert list(out.summary.index) == ["elastic_net", "ridge", "ols", "bks_implied", "oracle"]
    assert out.summary["available"].tolist() == [True, False, False, False, True]
    assert out.summary.loc["ols", "note"] == "refused"
    assert "D65" in out.summary.loc["ridge", "note"]
    assert out.summary.loc[["ridge", "ols", "bks_implied"], "r2_median_window"].isna().all()
    assert list(out.r2.columns) == ["elastic_net", "oracle"]
    assert set(out.r2_sweep["method"]) == {"elastic_net", "oracle"}
    assert set(out.meta["unavailable"]) == {"ridge", "ols", "bks_implied"}
    # nothing available at all still gives a well-formed result
    empty = compare_methods(sim, shocks, truth, cfg.window, {}, unavailable={"ols": "refused"})
    assert empty.summary.shape == (1, len(SUMMARY_COLUMNS)) and empty.r2.shape[1] == 0 and empty.r2_sweep.empty


def test_ridge_note_names_the_gcv_lower_edge(session_run):
    """A ridge fit whose GCV chose the smallest lambda for some assets says so in its note (review
    2026-09-29: the one-month window showed ridge at -2,824% with an empty note)."""
    s, cfg, _, res = session_run
    ridge = res.fits["ridge"]
    assert ridge.meta["gcv_at_lower_edge"] == 0 and res.summary.loc["ridge", "note"] == ""
    edge = dataclasses.replace(ridge, meta={**ridge.meta, "gcv_at_lower_edge": 7})
    out = compare_methods(s.simulation(cfg), s.shocks(cfg), s.truth(cfg), cfg.window, {"ridge": edge})
    assert out.summary.loc["ridge", "note"] == (
        "GCV chose the smallest lambda of its grid for 7 of 30 assets, so their exposures are close to OLS."
    )


# ---------------------------------------------------------------------------
# session stages
# ---------------------------------------------------------------------------
def test_session_bks_implied_never_starts_a_bks_fit():
    cfg = _cfg(n_topics=12, share=0.5)
    s = LabSession()
    with pytest.raises(LookupError, match="Run BKS first"):
        s.method_fit(cfg, "bks_implied")
    res = s.comparison(cfg)
    assert not s.has("bks_fit", cfg) and s.stats["bks_fit"]["misses"] == 0
    assert res.summary.loc["bks_implied", "note"] == BKS_NOT_RUN
    assert not res.summary.loc["bks_implied", "available"]
    key_before = s.comparison_key(cfg)
    assert s.has("comparison", cfg)
    # after a BKS run the comparison key changes and the method becomes available
    s.bks_fit(cfg)
    assert s.bks_ready(cfg) and s.comparison_key(cfg) != key_before and not s.has("comparison", cfg)
    res2 = s.comparison(cfg)
    assert res2.summary.loc["bks_implied", "available"]
    # use_bks=False leaves it out even with a cached fit
    res3 = s.comparison(cfg, use_bks=False)
    assert res3.summary.loc["bks_implied", "note"] == BKS_OFF
    assert s.method_fit(cfg, "elastic_net") is s.direct(cfg)


def test_session_keys_reuse_fits_across_forecast_windows(session_run):
    s, cfg, methods, res = session_run
    misses = {st: s.stats[st]["misses"] for st in ("direct", "bks_implied", "bks_fit", "comparison")}
    assert misses["direct"] == 3 and misses["bks_implied"] == 1  # elastic_net, ridge, oracle
    assert s.comparison(cfg, methods=methods) is res
    cfg2 = dataclasses.replace(cfg, window=dataclasses.replace(cfg.window, forecast_start="2023-03-06"))
    res2 = s.comparison(cfg2, methods=methods)
    assert res2 is not res
    assert s.stats["direct"]["misses"] == misses["direct"]
    assert s.stats["bks_implied"]["misses"] == misses["bks_implied"]
    assert s.stats["comparison"]["misses"] == misses["comparison"] + 1
    for m in methods:
        assert res2.fits[m] is res.fits[m]
    # keys hold what the stages depend on (D71)
    k = LabSession.stage_key
    assert k("bks_implied", cfg2) == k("bks_implied", cfg)
    tau = dataclasses.replace(cfg, direct=dataclasses.replace(cfg.direct, select_tau=0.1))
    assert k("bks_implied", tau) != k("bks_implied", cfg)
    kk = dataclasses.replace(cfg, bks=dataclasses.replace(cfg.bks, K=2))
    assert k("bks_implied", kk) != k("bks_implied", cfg)
    assert k("comparison", cfg, methods=methods) != k("comparison", cfg)
    assert k("comparison", cfg, bks_token="a") != k("comparison", cfg, bks_token="b")
    assert k("comparison", cfg2) != k("comparison", cfg)


def test_session_lists_ols_refusal_as_unavailable():
    # 30 topics against about 44 training days: OLS needs L < n_train / 2
    w = WindowConfig(train_start="2022-11-01", train_end="2022-12-30", forecast_start="2023-01-02")
    cfg = _cfg(n_topics=30, share=0.3, window=w)
    res = LabSession().comparison(cfg)
    assert not res.summary.loc["ols", "available"]
    assert "ols refused" in res.summary.loc["ols", "note"]
    assert res.summary.loc[["elastic_net", "ridge", "oracle"], "available"].all()


def test_run_lab_with_compare():
    cfg = _cfg(n_topics=12, share=0.5)
    out = run_lab(cfg, with_bks=True, with_compare=True)
    res = out["comparison"]
    assert res.summary["available"].all() and list(res.summary.index) == list(METHODS)
    assert out["keys"]["comparison"].startswith("comparison-")
    assert "comparison" not in run_lab(cfg)
    # the other variant's fit is a stage of its own, after cfg's BKS stages: timed and keyed (D88)
    other = LabSession.stage_key("bks_fit", method_config(cfg, "bks_implied_train"))
    assert out["keys"]["bks_fit_training"] == other and out["bks_fit_training"] is not None
    assert list(out["keys"]).index("bks_fit_training") > list(out["keys"]).index("bks")
    assert not out["timings"]["market"]["cached"] and "error" not in out["timings"]["bks_fit_training"]


def test_run_lab_with_compare_when_the_other_variant_refuses():
    """A six-month window ending on a Wednesday: the full history has 25 usable weeks, the training window 23
    (review 2026-09-30: run_lab raised instead of listing the training-window variant as unavailable)."""
    w = WindowConfig(train_start="2025-01-01", train_end="2025-06-25", forecast_start="2025-06-26", forecast_weeks=4)
    cfg = _cfg(n_topics=12, share=0.5, window=w)
    assert bks.training_weeks(w.train_start, w.train_end, cfg.bks)[0] == 25
    assert bks.training_weeks(w.train_start, w.train_end, BKSLabConfig(history="training"))[0] == 23
    out = run_lab(cfg, with_bks=True, with_compare=True)
    s = out["comparison"].summary
    assert s.loc["bks_implied", "available"] and not s.loc["bks_implied_train", "available"]
    note = s.loc["bks_implied_train", "note"]
    assert note.startswith(BKS_REFUSED) and "only 23 weekly periods" in note and BKS_NOT_RUN not in note
    assert out["bks_fit_training"] is None and "only 23 weekly periods" in out["timings"]["bks_fit_training"]["error"]
    assert out["keys"]["bks_fit_training"].startswith("bks_fit-")
    assert s.drop(index="bks_implied_train")["available"].all()
    # without the caller's error text the same comparison says BKS was not run, under another key
    s2 = LabSession().comparison(cfg).summary
    assert s2.loc["bks_implied_train", "note"] == BKS_NOT_RUN
    # the training-history choice itself still raises: the user asked for that fit
    tcfg = method_config(cfg, "bks_implied_train")
    with pytest.raises(ValueError, match="only 23 weekly periods"):
        run_lab(tcfg, with_bks=True, with_compare=True)


# ---------------------------------------------------------------------------
# real data: the dashboard's six-month default window
# ---------------------------------------------------------------------------
def _real_data_available() -> bool:
    try:
        return (reference.market_dir() / "asset_returns.parquet").is_file() and (
            reference.reference_dir() / "assets.csv"
        ).is_file()
    except Exception:  # pragma: no cover - a broken data folder counts as missing
        return False


@pytest.mark.skipif(not _real_data_available(), reason="data/market or data/reference missing")
def test_real_data_six_month_training_window_runs_bks_and_the_comparison():
    """The dashboard's default window runs both BKS variants and the comparison (D84, D88). Measured
    2026-09-30, median OOS R2 in the forecast window, noise seed 0: full history 7.0% (26 weeks, lambda 0.277,
    10 topics kept), training window -16.2% (24 weeks, lambda 0.017, 20 topics kept); elastic net 16.8%, oracle
    23.2%. Seed 0 is the best of seeds 0-4 for both variants (training window -16% to -107%, full history -18%
    to 7%; the training window lower in every seed; DESIGN.md D88, G.15.1)."""
    w = WindowConfig(train_start="2025-01-01", train_end="2025-06-30", forecast_start="2025-07-01", forecast_weeks=4)
    cfg = LabConfig(window=w)  # 55 listed assets, 20 manual topics, K = 3
    s = LabSession()
    fit = s.bks_fit(cfg)
    assert bks.MIN_TRAIN_PERIODS <= fit.meta["n_train_periods"] <= 26
    assert fit.meta["last_train_period"] <= pd.Timestamp("2025-06-30")
    assert np.isfinite(s.bks(cfg).r2_pooled)
    tcfg = method_config(cfg, "bks_implied_train")
    tfit = s.bks_fit(tcfg)
    assert tfit.meta["n_train_periods"] == 24 and tfit.train_periods[0] == pd.Timestamp("2025-01-17")
    assert np.isfinite(s.bks(tcfg).r2_pooled)
    res = s.comparison(cfg)
    assert res.summary["available"].all()
    assert res.r2.shape == (55, len(METHODS))
    for m in ("bks_implied", "bks_implied_train"):
        imp = res.fits[m]
        assert imp.meta["skipped_assets"] == [] and np.isfinite(imp.B_hat.to_numpy()).all()
    assert res.fits["bks_implied"].meta["kernel_share_before_train"] > 0.9
    assert res.fits["bks_implied_train"].meta["kernel_share_before_train"] == 0.0
    assert np.isfinite(res.summary.loc["bks_implied_train", "r2_median_window"])
