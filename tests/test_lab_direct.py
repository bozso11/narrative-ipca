"""Tests of the lab's direct exposure regression and forecast-window evaluation (DESIGN.md G.7.1, G.8; D64-D69).

Most tests use a generic universe of 12 assets and 20 generic topics, half of
them linked (10 linked topics x 6 links), with the default windows
(training 2015-2022, forecast from 2023-01-02). The central checks are the
out-of-sample discipline (no return after ``train_end`` changes the fit) and
the exact attribution identity ``realised = explained + residual``.
"""

from __future__ import annotations

import dataclasses
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from narrative_ipca.exposure_lab import reference
from narrative_ipca.exposure_lab.config import (
    DirectConfig,
    ExposureConfig,
    LabConfig,
    TopicSetConfig,
    UniverseConfig,
    WindowConfig,
)
from narrative_ipca.exposure_lab.dgp import observed_shocks, simulate_lab, truth_for_window
from narrative_ipca.exposure_lab.direct import (
    MIN_TRAIN_OBS,
    RIDGE_GCV_GRID,
    _fit_ridge,
    fit_direct,
    shock_matrix,
    training_pairs,
    universal_alpha,
)
from narrative_ipca.exposure_lab.evaluate import (
    SWEEP_COLUMNS,
    evaluate_window,
    recovery_metrics,
    window_return_days,
    window_sweep,
)
from narrative_ipca.exposure_lab.types import DirectFit, MarketData, ObservedShocks, SimData, SimTruth

GENERIC = UniverseConfig(asset_source="generic", n_generic_assets=12)
TOPICS = TopicSetConfig(manual="none", n_generic=20, generic_signal_share=0.5)
STRONG = ExposureConfig(beta_1=0.6, beta_2=0.3, beta_3=0.1)
WIN12 = WindowConfig(forecast_weeks=12)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Run:
    cfg: LabConfig
    sim: SimData
    shocks: ObservedShocks
    truth: SimTruth


def _cfg(
    exposure: ExposureConfig | None = None,
    window: WindowConfig | None = None,
    universe: UniverseConfig = GENERIC,
    topics: TopicSetConfig = TOPICS,
) -> LabConfig:
    return LabConfig(
        universe=universe,
        topics=topics,
        exposure=exposure or ExposureConfig(),
        window=window or WindowConfig(),
    )


def _run(cfg: LabConfig) -> Run:
    sim = simulate_lab(cfg)
    w = cfg.window
    shocks = observed_shocks(sim.attention, w.shock_window, w.train_start, w.train_end)
    return Run(cfg=cfg, sim=sim, shocks=shocks, truth=truth_for_window(sim, w.shock_window))


def _fit(run: Run, **kwargs) -> DirectFit:
    return fit_direct(run.sim, run.shocks, DirectConfig(**kwargs), run.truth)


def _tier_coverage(run: Run, fit: DirectFit, tiers: tuple[str, ...]) -> float:
    lt = run.sim.links.table
    sub = lt[lt["tier"].isin(tiers)]
    assert len(sub) > 0
    return float(np.mean([bool(fit.selected.loc[r.topic_id, r.asset_id]) for r in sub.itertuples()]))


def _with_returns(sim: SimData, returns: pd.DataFrame) -> SimData:
    market = MarketData(returns=returns, assets=sim.market.assets, meta=dict(sim.market.meta))
    return dataclasses.replace(sim, market=market)


@pytest.fixture(scope="module")
def base() -> Run:
    return _run(_cfg())


@pytest.fixture(scope="module")
def strong() -> Run:
    return _run(_cfg(exposure=STRONG))


@pytest.fixture(scope="module")
def lead1() -> Run:
    return _run(_cfg(exposure=ExposureConfig(lead_days=1)))


@pytest.fixture(scope="module")
def base_fit(base: Run) -> DirectFit:
    return _fit(base)


# ---------------------------------------------------------------------------
# Pairing (G.6, D64, D65)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["base", "lead1"])
def test_training_pairs_respect_train_end_and_lead(which: str, request: pytest.FixtureRequest) -> None:
    run: Run = request.getfixturevalue(which)
    cal = run.sim.market.calendar
    lead = run.sim.lead_days
    p, q = training_pairs(run.sim, run.shocks)
    te, ts = run.shocks.train_end, run.shocks.train_start
    assert np.all(q - p == lead)
    assert cal[q].max() <= te
    assert cal[p].min() >= ts
    # s_hat is NaN on the first w rows, so the first pair starts at row w
    assert p[0] == run.shocks.window
    # every shock day in the training window whose return day is in it
    in_train = np.flatnonzero((cal >= ts) & (cal <= te))
    expected = [i for i in in_train if i >= run.shocks.window and i + lead <= in_train[-1]]
    assert list(p) == expected
    fit = _fit(run)
    assert fit.meta["n_pairs"] == len(p)
    assert fit.meta["last_return_day"] == cal[q[-1]]
    assert fit.meta["lead_days"] == lead
    assert (fit.n_train == len(p)).all()


def test_lead_one_drops_the_pair_whose_return_is_after_train_end(base: Run, lead1: Run) -> None:
    p0, q0 = training_pairs(base.sim, base.shocks)
    p1, q1 = training_pairs(lead1.sim, lead1.shocks)
    assert len(p1) == len(p0) - 1
    assert q1[-1] == q0[-1]  # last return day is train_end in both
    assert p1[-1] == p0[-1] - 1  # with l = 1 the last shock day is the day before


def test_lead_matters(lead1: Run) -> None:
    """Attention built with l = 1: the correct pairing recovers the links; the same-day pairing does not.

    With l = 1 the same-day shock z_t only sees the returns of day t through
    the trailing mean (-1/w of the previous designed shock), so the wrongly
    paired fit picks up flipped, small exposures and explains almost nothing
    out of sample.
    """
    right = _fit(lead1)
    wrong_sim = dataclasses.replace(lead1.sim, lead_days=0)
    wrong = fit_direct(wrong_sim, lead1.shocks, DirectConfig(), lead1.truth)
    assert _tier_coverage(lead1, right, ("strong",)) >= 0.8

    ev_right = evaluate_window(lead1.sim, lead1.shocks, right, WIN12, lead1.truth)
    ev_wrong = evaluate_window(wrong_sim, lead1.shocks, wrong, WIN12, lead1.truth)
    assert ev_right.recovery["sign_agreement"] >= 0.9
    assert np.nanmedian(ev_right.r2) > 0.1
    assert np.nanmedian(ev_wrong.r2) < 0.05

    # strong links: the wrong pairing has the opposite sign of the design
    lt = lead1.sim.links.table
    st = lt[lt["tier"] == "strong"]
    b_wrong = np.array([wrong.B_hat.loc[r.topic_id, r.asset_id] for r in st.itertuples()])
    b_right = np.array([right.B_hat.loc[r.topic_id, r.asset_id] for r in st.itertuples()])
    assert np.mean(np.sign(b_right) == st["sign"].to_numpy()) >= 0.9
    assert np.mean(np.sign(b_wrong) == st["sign"].to_numpy()) <= 0.2
    assert np.abs(b_wrong).mean() < 0.5 * np.abs(b_right).mean()


# ---------------------------------------------------------------------------
# Elastic net (D66)
# ---------------------------------------------------------------------------
def test_elastic_net_recovers_links_default_betas(base: Run, base_fit: DirectFit) -> None:
    fit = base_fit
    assert fit.method == "elastic_net"
    assert fit.meta["alpha_rule"] == "universal"
    assert fit.meta["n_convergence_warnings"] == 0
    n = int(fit.n_train.iloc[0])
    assert np.allclose(fit.penalty, universal_alpha(20, n))
    assert fit.penalty.iloc[0] == pytest.approx(np.sqrt(2 * np.log(20) / n))
    assert fit.selected.equals(fit.B_hat != 0.0)
    assert fit.B_hat.shape == (20, 12)
    assert list(fit.B_hat.index) == base.sim.topics.ids

    assert _tier_coverage(base, fit, ("strong",)) >= 0.8
    assert _tier_coverage(base, fit, ("strong", "moderate")) >= 0.7
    rec = recovery_metrics(fit, base.truth, tau=0.05)
    assert rec["n_linked"] == 60
    assert rec["sign_agreement"] >= 0.9
    assert rec["spearman"] > 0.7
    # noise topics (no links) are rarely selected
    linked_topics = set(base.sim.links.table["topic_id"])
    noise = [k for k in base.sim.topics.ids if k not in linked_topics]
    assert len(noise) == 10
    assert fit.selected.loc[noise].to_numpy().mean() < 0.1


def test_elastic_net_recovers_links_strong_betas(strong: Run) -> None:
    fit = _fit(strong)
    assert _tier_coverage(strong, fit, ("strong",)) == 1.0
    assert _tier_coverage(strong, fit, ("strong", "moderate")) >= 0.85
    rec = recovery_metrics(fit, strong.truth, tau=0.05)
    assert rec["sign_agreement"] >= 0.9
    assert rec["mcc"] >= 0.7


def test_elastic_net_fixed_penalty(base: Run) -> None:
    fit = _fit(base, penalty="fixed", alpha=10.0)
    assert fit.meta["alpha_rule"] == "fixed"
    assert np.allclose(fit.penalty, 10.0)
    assert not fit.selected.to_numpy().any()
    assert (fit.B_hat == 0.0).all().all()
    rec = recovery_metrics(fit, base.truth, tau=0.05)
    assert rec["coverage"] == 0.0 and np.isnan(rec["sign_agreement"]) and np.isnan(rec["mcc"])
    small = _fit(base, penalty="fixed", alpha=0.01)
    assert small.selected.to_numpy().sum() > _fit(base).selected.to_numpy().sum()
    with pytest.raises(ValueError, match="alpha > 0"):
        _fit(base, penalty="fixed", alpha=0.0)


def test_elastic_net_convergence_warnings_are_counted(base: Run) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no ConvergenceWarning escapes fit_direct
        fit = _fit(base, max_iter=1)
    assert fit.meta["n_convergence_warnings"] > 0


def test_elastic_net_cv(base: Run) -> None:
    fit = _fit(base, penalty="cv")
    assert fit.meta["alpha_rule"] == "cv"
    assert fit.meta["cv_folds"] == 5
    assert np.all(np.isfinite(fit.penalty)) and np.all(fit.penalty > 0)
    assert _tier_coverage(base, fit, ("strong",)) >= 0.8


# ---------------------------------------------------------------------------
# Ridge, OLS, oracle
# ---------------------------------------------------------------------------
def _train_xy(run: Run, fit: DirectFit) -> tuple[np.ndarray, np.ndarray]:
    p, q = training_pairs(run.sim, run.shocks)
    X = shock_matrix(run.shocks, run.sim.market.calendar, run.sim.topics.ids)[p]
    R = run.sim.market.returns.to_numpy()[q]
    Y = (R - fit.ret_mean.to_numpy()) / fit.ret_scale.to_numpy()
    return X, Y


def test_ridge_fixed_matches_closed_form(base: Run) -> None:
    lam = 0.05
    fit = _fit(base, method="ridge", ridge_lambda=lam)
    assert fit.meta["alpha_rule"] == "fixed"
    assert np.allclose(fit.penalty, lam)
    X, Y = _train_xy(base, fit)
    n, L = X.shape
    Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
    B = np.linalg.lstsq(Xc.T @ Xc + n * lam * np.eye(L), Xc.T @ Yc, rcond=None)[0]
    assert np.allclose(fit.B_hat.to_numpy(), B, atol=1e-10)
    icpt = Y.mean(axis=0) - X.mean(axis=0) @ B
    assert np.allclose(fit.intercept.to_numpy(), icpt, atol=1e-10)
    assert fit.selected.equals(fit.B_hat.abs() >= 0.05)
    # shrinkage grows with lambda
    big = _fit(base, method="ridge", ridge_lambda=5.0)
    assert np.abs(big.B_hat.to_numpy()).sum() < 0.5 * np.abs(fit.B_hat.to_numpy()).sum()


def test_ridge_gcv_picks_the_grid_minimum(base: Run) -> None:
    fit = _fit(base, method="ridge")
    assert fit.meta["alpha_rule"] == "gcv"
    assert np.all(np.isin(fit.penalty.to_numpy(), RIDGE_GCV_GRID))
    X, Y = _train_xy(base, fit)
    n, L = X.shape
    Xc = X - X.mean(axis=0)
    for j in (0, 5):
        y = Y[:, j] - Y[:, j].mean()
        scores = []
        for lam in RIDGE_GCV_GRID:
            A = np.linalg.inv(Xc.T @ Xc + n * lam * np.eye(L))
            H = Xc @ A @ Xc.T
            resid = y - H @ y
            scores.append((resid @ resid / n) / (1.0 - (np.trace(H) + 1.0) / n) ** 2)  # + 1: the intercept
        lam_best = RIDGE_GCV_GRID[int(np.argmin(scores))]
        assert fit.penalty.iloc[j] == pytest.approx(lam_best)
        b = np.linalg.solve(Xc.T @ Xc + n * lam_best * np.eye(L), Xc.T @ y)
        assert np.allclose(fit.B_hat.iloc[:, j].to_numpy(), b, atol=1e-10)


def test_ridge_gcv_counts_the_intercept_when_topics_fill_the_window() -> None:
    """21 days and 20 topics: the centred shocks span the centred returns, so GCV without the intercept's
    degree of freedom tends to 0 as lambda -> 0 and picks the grid's lower edge (review 2026-09-29: 47 of
    55 assets on a one-month dashboard window, median OOS R2 -2,824%)."""
    rng = np.random.default_rng([11, 3])
    n, L, m = 21, 20, 40
    X = rng.standard_normal((n, L))
    B = np.zeros((L, m))
    B[:3] = 0.4
    Y = X @ B + rng.standard_normal((n, m))
    coef, icpt, lam, at_edge = _fit_ridge(X, Y, None)
    Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
    at_lower_without = 0
    for j in range(m):
        with_icpt, without = [], []
        for g in RIDGE_GCV_GRID:
            H = Xc @ np.linalg.solve(Xc.T @ Xc + n * g * np.eye(L), Xc.T)
            r = Yc[:, j] - H @ Yc[:, j]
            with_icpt.append((r @ r / n) / (1.0 - (np.trace(H) + 1.0) / n) ** 2)
            without.append((r @ r / n) / (1.0 - np.trace(H) / n) ** 2)
        assert lam[j] == pytest.approx(RIDGE_GCV_GRID[int(np.argmin(with_icpt))])
        at_lower_without += int(np.argmin(without) == 0)
    assert at_lower_without / m > 0.8  # the old rule: mostly the lower edge
    assert np.mean(lam == RIDGE_GCV_GRID[0]) < 0.25 and np.median(lam) > 0.1
    assert at_edge == int(np.sum((lam == RIDGE_GCV_GRID[0]) | (lam == RIDGE_GCV_GRID[-1])))


def test_ridge_gcv_on_a_one_month_window() -> None:
    """The lab route on 21 training days with 20 topics: the lower-edge count is recorded and the
    out-of-sample fit stays sane (measured 2026-09-29: median OOS R2 9.6%, oracle 35.4%)."""
    win = WindowConfig(train_start="2022-12-02", train_end="2022-12-30", forecast_start="2023-01-02")
    run = _run(_cfg(window=win))
    fit = _fit(run, method="ridge")
    assert int(fit.n_train.min()) == 21 and len(fit.B_hat.index) == 20
    assert fit.meta["gcv_at_lower_edge"] == int(np.sum(fit.penalty.to_numpy() == RIDGE_GCV_GRID[0]))
    assert fit.meta["gcv_at_lower_edge"] <= 3
    ev = evaluate_window(run.sim, run.shocks, fit, win, run.truth)
    assert np.isfinite(ev.r2).all() and float(np.median(ev.r2)) > -0.25


def test_ols_matches_least_squares(base: Run) -> None:
    fit = _fit(base, method="ols")
    assert fit.meta["alpha_rule"] == "none"
    assert fit.penalty.isna().all()
    X, Y = _train_xy(base, fit)
    Z = np.column_stack([np.ones(len(X)), X])
    coef = np.linalg.lstsq(Z, Y, rcond=None)[0]
    assert np.allclose(fit.B_hat.to_numpy(), coef[1:], atol=1e-10)
    assert np.allclose(fit.intercept.to_numpy(), coef[0], atol=1e-10)
    # a tiny ridge penalty is OLS
    tiny = _fit(base, method="ridge", ridge_lambda=1e-12)
    assert np.allclose(tiny.B_hat.to_numpy(), fit.B_hat.to_numpy(), atol=1e-8)


def test_ols_refused_when_topics_exceed_half_the_training_days() -> None:
    win = WindowConfig(train_start="2022-01-03", train_end="2022-12-30", forecast_start="2023-01-02")
    run = _run(_cfg(window=win, topics=TopicSetConfig(manual="none", n_generic=130, generic_signal_share=0.2)))
    n = len(training_pairs(run.sim, run.shocks)[0])
    assert 130 >= n / 2
    with pytest.raises(ValueError, match="ols refused"):
        _fit(run, method="ols")
    # the penalised methods still run
    assert _fit(run, method="ridge").B_hat.shape == (130, 12)
    assert _fit(run).B_hat.shape == (130, 12)


def test_oracle_equals_truth(base: Run) -> None:
    fit = _fit(base, method="oracle")
    assert fit.B_hat.equals(base.truth.B_true)
    assert (fit.intercept == 0.0).all()
    assert fit.penalty.isna().all()
    assert fit.selected.equals(base.truth.B_true.abs() >= 0.05)
    ref = _fit(base)
    assert np.allclose(fit.ret_scale, ref.ret_scale) and np.allclose(fit.ret_mean, ref.ret_mean)
    rec = recovery_metrics(fit, base.truth, tau=0.05)
    assert rec["sign_agreement"] == 1.0
    assert rec["spearman"] == pytest.approx(1.0)
    assert rec["rmse"] == 0.0
    assert rec["mcc"] == 1.0
    assert rec["n_selected"] == rec["n_truly_exposed"]
    # one oracle definition (D74): the oracle method reproduces the oracle reference exactly
    ev = evaluate_window(base.sim, base.shocks, fit, WIN12, base.truth)
    np.testing.assert_array_equal(ev.r2.to_numpy(), ev.r2_oracle.to_numpy())
    np.testing.assert_array_equal(ev.contrib.to_numpy(), ev.contrib_true.to_numpy())
    np.testing.assert_array_equal(ev.var_share.to_numpy(), ev.var_share_true.to_numpy())
    np.testing.assert_array_equal(ev.fitted.to_numpy(), ev.fitted_oracle.to_numpy())
    sweep = window_sweep(base.sim, base.shocks, fit, WindowConfig(), base.truth)
    np.testing.assert_array_equal(sweep["median_r2"].to_numpy(), sweep["median_r2_oracle"].to_numpy())


# ---------------------------------------------------------------------------
# Out-of-sample discipline (D65)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["base", "lead1"])
@pytest.mark.parametrize("method", ["elastic_net", "ridge"])
def test_no_leakage_from_returns_after_train_end(which: str, method: str, request: pytest.FixtureRequest) -> None:
    run: Run = request.getfixturevalue(which)
    fit = _fit(run, method=method)
    ret = run.sim.market.returns
    after = ret.index > run.shocks.train_end
    assert after.sum() > 500  # includes the forecast window
    rng = np.random.default_rng([7, 1])
    perturbed = ret.copy()
    perturbed.loc[after] = ret.loc[after] * 3.0 + rng.normal(0.0, 0.01, size=(int(after.sum()), ret.shape[1]))
    fit2 = fit_direct(_with_returns(run.sim, perturbed), run.shocks, DirectConfig(method=method), run.truth)
    assert np.array_equal(fit.B_hat.to_numpy(), fit2.B_hat.to_numpy())
    assert np.array_equal(fit.intercept.to_numpy(), fit2.intercept.to_numpy())
    assert np.array_equal(fit.ret_mean.to_numpy(), fit2.ret_mean.to_numpy())
    assert np.array_equal(fit.ret_scale.to_numpy(), fit2.ret_scale.to_numpy())

    # the first return day after train_end alone (the l = 1 boundary pair)
    first_after = ret.index[after][0]
    edge = ret.copy()
    edge.loc[first_after] = 0.5
    fit3 = fit_direct(_with_returns(run.sim, edge), run.shocks, DirectConfig(method=method), run.truth)
    assert np.array_equal(fit.B_hat.to_numpy(), fit3.B_hat.to_numpy())

    # a change inside the training window does change the fit
    inside = (ret.index >= "2018-01-01") & (ret.index <= "2018-06-30")
    changed = ret.copy()
    changed.loc[inside] = ret.loc[inside] * 3.0
    fit4 = fit_direct(_with_returns(run.sim, changed), run.shocks, DirectConfig(method=method), run.truth)
    assert not np.allclose(fit.B_hat.to_numpy(), fit4.B_hat.to_numpy())
    assert not np.allclose(fit.ret_scale.to_numpy(), fit4.ret_scale.to_numpy())


def test_missing_returns_are_grouped_and_skipped(base: Run, base_fit: DirectFit) -> None:
    ret = base.sim.market.returns.copy()
    ids = list(ret.columns)
    ret.loc["2016-01-01":"2016-12-31", ids[1]] = np.nan
    ret.loc["2019-03-01":"2019-05-31", ids[2]] = np.nan
    ret.loc[ret.index <= "2022-12-30", ids[3]] = np.nan  # no training data at all
    sim = _with_returns(base.sim, ret)
    fit = fit_direct(sim, base.shocks, DirectConfig(), base.truth)
    assert fit.meta["n_groups"] == 4
    assert fit.meta["skipped_assets"] == [ids[3]]
    assert (fit.B_hat[ids[3]] == 0.0).all() and not fit.selected[ids[3]].any()
    assert fit.n_train[ids[3]] == 0 < MIN_TRAIN_OBS
    n_all = int(base_fit.n_train.iloc[0])
    assert fit.n_train[ids[1]] < n_all and fit.n_train[ids[2]] < n_all
    # the training mean of asset 1 uses its non-missing training pairs only
    p, q = training_pairs(sim, base.shocks)
    r1 = ret[ids[1]].to_numpy()[q]
    assert fit.ret_mean[ids[1]] == pytest.approx(np.nanmean(r1), rel=1e-12)
    assert fit.ret_scale[ids[1]] == pytest.approx(np.nanstd(r1), rel=1e-12)
    # complete assets are fitted exactly as before (independent targets)
    full = [ids[0]] + ids[4:]
    assert np.allclose(fit.B_hat[full].to_numpy(), base_fit.B_hat[full].to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------
# Evaluation (G.8)
# ---------------------------------------------------------------------------
def test_window_return_days() -> None:
    cal = pd.bdate_range("2023-01-02", "2023-03-31")
    w1 = WindowConfig(forecast_start="2023-01-02", forecast_weeks=1)
    assert list(window_return_days(cal, w1)) == list(pd.bdate_range("2023-01-02", "2023-01-06"))
    wed = WindowConfig(forecast_start="2023-01-04", forecast_weeks=2)
    days = window_return_days(cal, wed)
    assert len(days) == 10 and days[0] == pd.Timestamp("2023-01-04") and days[-1] == pd.Timestamp("2023-01-17")
    late = WindowConfig(forecast_start="2024-01-01", forecast_weeks=1)
    assert len(window_return_days(cal, late)) == 0


def test_evaluate_window_identities(base: Run, base_fit: DirectFit) -> None:
    win = WindowConfig()  # 4 weeks from 2023-01-02
    ev = evaluate_window(base.sim, base.shocks, base_fit, win, base.truth)
    days = window_return_days(base.sim.market.calendar, win)
    assert ev.n_days == 20 and ev.return_days.equals(days)
    assert ev.corr.shape == (12, 20) and list(ev.corr.index) == list(base_fit.B_hat.columns)

    ret = base.sim.market.returns.loc[days]
    assert ev.realized_daily.equals(ret)
    assert np.allclose(ev.realized, ret.sum())
    # attribution identity and its pieces
    assert np.max(np.abs(ev.realized - (ev.explained + ev.residual))) <= 1e-12
    assert np.allclose(ev.contrib.sum(axis=1), ev.explained, atol=1e-15)
    assert np.allclose(ev.fitted.sum(), ev.explained, atol=1e-12)
    assert np.allclose(ev.contrib_true.sum(axis=1), ev.explained_true, atol=1e-15)
    assert np.allclose(ev.fitted_oracle.sum(), ev.explained_true, atol=1e-12)

    # fitted = sd_train * S @ B_hat with S on the same day (l = 0)
    S = base.shocks.s_hat.loc[days, base_fit.B_hat.index].to_numpy()
    fitted = (S @ base_fit.B_hat.to_numpy()) * base_fit.ret_scale.to_numpy()
    assert np.allclose(ev.fitted.to_numpy(), fitted, atol=1e-15)
    # the oracle uses the same training return scale as the estimator (D74), not the full-sample asset_vol
    fo = (S @ base.truth.B_true.to_numpy()) * base_fit.ret_scale.to_numpy()
    assert np.allclose(ev.fitted_oracle.to_numpy(), fo, atol=1e-15)

    # uncentered R2, correlation and variance share by hand
    r = ret.to_numpy()
    r2 = 1.0 - ((r - fitted) ** 2).sum(axis=0) / (r**2).sum(axis=0)
    assert np.allclose(ev.r2.to_numpy(), r2)
    corr = ev.corr.to_numpy()
    fin = corr[np.isfinite(corr)]
    assert fin.size == corr.size and np.all((fin >= -1.0) & (fin <= 1.0))
    assert ev.corr.iloc[3, 7] == pytest.approx(np.corrcoef(r[:, 3], S[:, 7])[0, 1], abs=1e-10)
    vs = ev.var_share.to_numpy()
    assert np.allclose(vs.sum(axis=1), (fitted * r).sum(axis=0) / (r**2).sum(axis=0))
    assert ev.recovery == recovery_metrics(base_fit, base.truth, tau=0.05)


def test_evaluate_window_uses_the_lagged_shock(lead1: Run) -> None:
    fit = _fit(lead1)
    ev = evaluate_window(lead1.sim, lead1.shocks, fit, WindowConfig(), lead1.truth)
    cal = lead1.sim.market.calendar
    first = ev.return_days[0]
    shock_day = cal[cal.get_loc(first) - 1]
    assert shock_day == pd.Timestamp("2022-12-30")  # the last training day: observed, return is out of sample
    s = lead1.shocks.s_hat.loc[shock_day, fit.B_hat.index].to_numpy()
    expected = (s @ fit.B_hat.to_numpy()) * fit.ret_scale.to_numpy()
    assert np.allclose(ev.fitted.loc[first].to_numpy(), expected, atol=1e-15)


def test_evaluate_window_missing_returns(base: Run, base_fit: DirectFit) -> None:
    ret = base.sim.market.returns.copy()
    ids = list(ret.columns)
    days = window_return_days(ret.index, WindowConfig())
    ret.loc[days[:3], ids[0]] = np.nan
    ret.loc[days, ids[1]] = np.nan
    sim = _with_returns(base.sim, ret)
    ev = evaluate_window(sim, base.shocks, base_fit, WindowConfig(), base.truth)
    assert ev.fitted[ids[0]].isna().sum() == 3
    assert np.allclose(ev.realized[ids[0]], ret.loc[days, ids[0]].sum())
    assert np.isnan(ev.realized[ids[1]]) and np.isnan(ev.r2[ids[1]]) and ev.corr.loc[ids[1]].isna().all()
    ok = ev.realized.notna()
    assert np.max(np.abs(ev.realized[ok] - ev.explained[ok] - ev.residual[ok])) <= 1e-12


def test_oracle_at_least_as_good_as_estimator_on_12_weeks(base: Run, base_fit: DirectFit) -> None:
    ev = evaluate_window(base.sim, base.shocks, base_fit, WIN12, base.truth)
    assert ev.n_days == 60
    med, med_oracle = float(np.nanmedian(ev.r2)), float(np.nanmedian(ev.r2_oracle))
    assert med > 0.1
    assert med_oracle >= med - 0.05


def test_recovery_metrics_hand_case() -> None:
    topics = pd.Index(["T1", "T2"], name="topic_id")
    assets = pd.Index(["A1", "A2", "A3"], name="asset_id")

    def frame(x) -> pd.DataFrame:
        return pd.DataFrame(np.array(x, dtype=float), index=topics, columns=assets)

    W = frame([[0.35, 0.0, -0.15], [0.0, 0.05, 0.0]])  # linked: (T1,A1), (T1,A3), (T2,A2)
    Bt = frame([[0.30, 0.02, -0.12], [0.01, 0.04, 0.06]])  # exposed at tau 0.05: (T1,A1), (T1,A3), (T2,A3)
    Bh = frame([[0.25, 0.01, 0.03], [0.0, 0.0, 0.0]])  # selected: (T1,A1), (T1,A2), (T1,A3)
    ones_t, ones_a = pd.Series(1.0, index=topics), pd.Series(1.0, index=assets)
    truth = SimTruth(
        W=W, W_unscaled=W, feasibility_scale=ones_t, sigma_u=ones_t, attenuation=ones_t, B_true=Bt,
        r2_true=ones_a, S_z=pd.DataFrame(np.eye(2), index=topics, columns=topics), asset_vol=ones_a,
        R=pd.DataFrame(np.eye(3), index=assets, columns=assets), shock_window=5,
    )
    fit = DirectFit(
        B_hat=Bh, intercept=0 * ones_a, ret_mean=0 * ones_a, ret_scale=ones_a, selected=Bh != 0.0,
        penalty=ones_a, n_train=ones_a.astype(int), method="elastic_net", meta={"select_tau": 0.05},
    )
    rec = recovery_metrics(fit, truth, tau=0.05)
    assert rec["n_linked"] == 3
    assert rec["coverage"] == pytest.approx(2 / 3)
    assert rec["sign_agreement"] == pytest.approx(0.5)  # (T1,A3): +0.03 against -0.12
    # TP 2, FP 1, FN 1, TN 2
    assert rec["mcc"] == pytest.approx((2 * 2 - 1 * 1) / np.sqrt(3 * 3 * 3 * 3))
    assert rec["n_selected"] == 3 and rec["n_truly_exposed"] == 3
    assert rec["rmse"] == pytest.approx(np.sqrt(np.mean((Bh.to_numpy() - Bt.to_numpy()) ** 2)))
    assert rec["spearman"] == pytest.approx(spearmanr(Bh.to_numpy().ravel(), Bt.to_numpy().ravel()).statistic)


def test_window_sweep(base: Run, base_fit: DirectFit) -> None:
    win = WindowConfig()  # 4-week windows
    sweep = window_sweep(base.sim, base.shocks, base_fit, win, base.truth)
    assert tuple(sweep.columns) == SWEEP_COLUMNS
    last = base.sim.market.calendar[-1]
    expected_n = ((last - pd.Timestamp("2023-01-02")).days + 1) // 28
    assert len(sweep) == expected_n == 39
    assert (sweep["n_days"] == 20).all()
    assert (sweep["end"] <= last).all()
    assert (sweep["start"].diff().dropna() == pd.Timedelta(days=28)).all()
    assert (sweep["end"] - sweep["start"] == pd.Timedelta(days=27)).all()
    ev = evaluate_window(base.sim, base.shocks, base_fit, win, base.truth)
    row = sweep.iloc[0]
    assert row["median_r2"] == pytest.approx(float(np.nanmedian(ev.r2)), abs=1e-12)
    assert row["median_r2_oracle"] == pytest.approx(float(np.nanmedian(ev.r2_oracle)), abs=1e-12)
    r = ev.realized_daily.to_numpy()
    pooled = 1.0 - ((r - ev.fitted.to_numpy()) ** 2).sum() / (r**2).sum()
    assert row["pooled_r2"] == pytest.approx(pooled, abs=1e-12)
    # a later window equals evaluate_window on that window
    k = 10
    wk = dataclasses.replace(win, forecast_start=str(sweep["start"].iloc[k].date()))
    evk = evaluate_window(base.sim, base.shocks, base_fit, wk, base.truth)
    assert sweep["median_r2"].iloc[k] == pytest.approx(float(np.nanmedian(evk.r2)), abs=1e-12)
    assert np.isfinite(sweep[["median_r2", "median_r2_oracle", "pooled_r2", "pooled_r2_oracle"]].to_numpy()).all()

    capped = window_sweep(base.sim, base.shocks, base_fit, win, base.truth, max_windows=5)
    assert len(capped) == 5 and capped.equals(sweep.iloc[:5])
    one_week = window_sweep(base.sim, base.shocks, base_fit, WindowConfig(forecast_weeks=1), base.truth)
    assert len(one_week) == 150


def test_window_sweep_and_evaluation_beyond_the_data(base: Run, base_fit: DirectFit) -> None:
    late = WindowConfig(forecast_start="2026-06-01")
    sweep = window_sweep(base.sim, base.shocks, base_fit, late, base.truth)
    assert sweep.empty and tuple(sweep.columns) == SWEEP_COLUMNS
    ev = evaluate_window(base.sim, base.shocks, base_fit, late, base.truth)
    assert ev.n_days == 0 and ev.r2.isna().all() and ev.realized.isna().all()


def test_evaluation_refuses_a_window_inside_the_training_window(base: Run, base_fit: DirectFit) -> None:
    """D65 in the library API: evaluate_window and window_sweep raise instead of scoring in-sample days."""
    late = observed_shocks(base.sim.attention, 5, "2015-01-02", "2023-06-30")  # trained past the window start
    fit = fit_direct(base.sim, late, DirectConfig(), base.truth)
    assert fit.meta["last_return_day"] == pd.Timestamp("2023-06-30")
    win = WindowConfig()  # forecast from 2023-01-02
    with pytest.raises(ValueError, match="not after the training end"):
        evaluate_window(base.sim, late, fit, win, base.truth)
    with pytest.raises(ValueError, match="not after the training end"):
        window_sweep(base.sim, late, fit, win, base.truth)
    # a fit trained on later returns than its shocks' window is refused too
    odd = dataclasses.replace(base_fit, meta={**base_fit.meta, "last_return_day": pd.Timestamp("2023-01-03")})
    with pytest.raises(ValueError, match="last training return day"):
        evaluate_window(base.sim, base.shocks, odd, win, base.truth)
    # the normal case still runs
    assert evaluate_window(base.sim, base.shocks, base_fit, win, base.truth).n_days == 20


# ---------------------------------------------------------------------------
# Scale and real data
# ---------------------------------------------------------------------------
def test_elastic_net_500_topics_is_fast() -> None:
    run = _run(_cfg(universe=UniverseConfig(asset_source="generic", n_generic_assets=55),
                    topics=TopicSetConfig(manual="none", n_generic=500, generic_signal_share=0.2)))
    t0 = time.perf_counter()
    fit = _fit(run)
    ev = evaluate_window(run.sim, run.shocks, fit, WIN12, run.truth)
    sweep = window_sweep(run.sim, run.shocks, fit, WindowConfig(), run.truth)
    elapsed = time.perf_counter() - t0
    assert fit.B_hat.shape == (500, 55)
    assert elapsed < 10.0
    assert np.nanmedian(ev.r2_oracle) >= np.nanmedian(ev.r2) - 0.05
    assert len(sweep) == 39


@pytest.mark.skipif(
    not (reference.market_dir() / "asset_returns.parquet").exists(), reason="data/market not available"
)
def test_listed_real_data_smoke() -> None:
    cfg = LabConfig()  # 55 listed assets, real prices, 20 manual topics
    t0 = time.perf_counter()
    run = _run(cfg)
    fit = fit_direct(run.sim, run.shocks, cfg.direct, run.truth)
    ev = evaluate_window(run.sim, run.shocks, fit, cfg.window, run.truth)
    sweep = window_sweep(run.sim, run.shocks, fit, cfg.window, run.truth)
    elapsed = time.perf_counter() - t0
    assert fit.B_hat.shape == (20, 55)
    assert elapsed < 20.0
    assert ev.r2.notna().sum() >= 50
    ok = ev.realized.notna()
    assert np.max(np.abs(ev.realized[ok] - ev.explained[ok] - ev.residual[ok])) <= 1e-12
    assert ev.recovery["n_linked"] > 50
    assert np.isfinite(ev.recovery["coverage"])
    assert len(sweep) > 30
