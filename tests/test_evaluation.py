"""Tests for narrative_ipca.evaluation: the mathematics, not just the shapes."""

from __future__ import annotations

import sys
import types as _pytypes

import numpy as np
import pandas as pd
import pytest
from scipy import stats as _stats

import narrative_ipca
from narrative_ipca import evaluation as ev
from narrative_ipca.config import EvaluationConfig, PipelineConfig
from narrative_ipca.types import (
    IPCAPanel,
    LambdaPathPoint,
    OOSResult,
    PlaceboResult,
    ShockPanel,
    SparseIPCAResult,
    TuningResult,
    WrapUpResult,
    annualized_sharpe,
    compute_sigma_c,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_panel(T=8, N=6, L=4, K=2, seed=0, noise=0.0, drop_period=None):
    """Long-form IPCAPanel generated from a known (Gamma, F); returns (panel, Gamma, F)."""
    rng = np.random.default_rng(seed)
    Gamma = rng.normal(size=(L + 1, K))
    F = rng.normal(size=(T, K))
    rows_X, rows_y, t_idx, a_idx = [], [], [], []
    for t in range(T):
        if drop_period is not None and t == drop_period:
            continue
        for i in range(N):
            x = np.concatenate([[1.0], rng.normal(size=L)])
            y = float((x @ Gamma) @ F[t]) + noise * rng.normal()
            rows_X.append(x)
            rows_y.append(y)
            t_idx.append(t)
            a_idx.append(i)
    X = np.asarray(rows_X)
    panel = IPCAPanel(
        X=X,
        y=np.asarray(rows_y),
        t_idx=np.asarray(t_idx),
        asset_idx=np.asarray(a_idx),
        periods=pd.date_range("2001-01-31", periods=T, freq="ME"),
        assets=np.array([f"a{i}" for i in range(N)]),
        instrument_names=["const"] + [f"topic_{j}" for j in range(L)],
        sigma_c=compute_sigma_c(X),
    )
    return panel, Gamma, F


def make_fit(panel, Gamma, F, lam=0.1, selected=None, converged=True, n_iter=7):
    K = Gamma.shape[1]
    populated = np.asarray(panel.moments().n) > 0
    Fp = F[populated]
    norms = np.linalg.norm(Gamma, axis=1)
    if selected is None:
        selected = norms[1:] > 0
    return SparseIPCAResult(
        Gamma=Gamma,
        F=F,
        mu_f=Fp.mean(axis=0),
        Sigma_ff=np.cov(Fp, rowvar=False, ddof=1).reshape(K, K),
        lam=lam,
        K=K,
        objective=1.234,
        obj_path=[2.0, 1.5, 1.234],
        n_iter=n_iter,
        converged=converged,
        total_r2=ev.total_r2(panel, Gamma, F),
        pred_r2=ev.predictive_r2(panel, Gamma, Fp.mean(axis=0)),
        gamma_norms=norms,
        selected=np.asarray(selected, dtype=bool),
        instrument_names=list(panel.instrument_names),
        periods=panel.periods,
        n_obs=panel.n_obs,
        populated=populated,
    )


def make_path_point(lam, K, gamma_norms, mve=1.0):
    return LambdaPathPoint(
        lam=lam,
        K=K,
        total_r2=0.1,
        pred_r2=0.01,
        mve_sharpe=mve,
        n_selected=int(np.count_nonzero(np.asarray(gamma_norms)[1:])),
        gamma_norms=np.asarray(gamma_norms, dtype=float),
        objective=1.0,
        converged=True,
        n_iter=3,
        criterion=mve,
    )


def make_tuning(fit, lams=(0.01, 0.1, 1.0)):
    path = []
    for j, lam in enumerate(lams):
        norms = np.asarray(fit.gamma_norms, dtype=float).copy()
        norms[1 + j:] = 0.0  # progressively sparser
        path.append(make_path_point(lam, fit.K, norms, mve=1.0 - 0.1 * j))
    return TuningResult(lam=fit.lam, K=fit.K, criterion="is_sharpe", path=path, fit=fit, lam_max=max(lams) * 2)


# ---------------------------------------------------------------------------
# realized_sharpe (D34)
# ---------------------------------------------------------------------------


def test_realized_sharpe_known_series():
    x = pd.Series([0.01, 0.03, 0.02, 0.04])
    mean = 0.025
    sd = np.sqrt(((0.015**2) * 2 + (0.005**2) * 2) / 3)  # ddof = 1
    expected = mean / sd * np.sqrt(12.0)
    assert ev.realized_sharpe(x, 12.0) == pytest.approx(expected, rel=1e-12)
    # annualisation scales with sqrt
    assert ev.realized_sharpe(x, 252.0) == pytest.approx(expected * np.sqrt(252.0 / 12.0), rel=1e-12)


def test_realized_sharpe_nan_safe_and_degenerate():
    x = pd.Series([0.01, np.nan, 0.03, 0.02, np.inf, 0.04])
    assert ev.realized_sharpe(x, 12.0) == pytest.approx(ev.realized_sharpe([0.01, 0.03, 0.02, 0.04], 12.0))
    assert np.isnan(ev.realized_sharpe([0.02, 0.02, 0.02], 12.0))  # constant
    assert np.isnan(ev.realized_sharpe([0.02], 12.0))  # single obs
    assert np.isnan(ev.realized_sharpe([np.nan, np.nan], 12.0))
    assert np.isnan(ev.realized_sharpe([], 12.0))


def test_realized_sharpe_of_mve_portfolio_equals_closed_form():
    """The realised Sharpe of f' Sigma^-1 mu equals sqrt(ann mu' Sigma^-1 mu) (same ddof)."""
    rng = np.random.default_rng(3)
    F = rng.normal(size=(60, 3)) * 0.05 + np.array([0.01, 0.005, -0.002])
    mu = F.mean(axis=0)
    Sigma = np.cov(F, rowvar=False, ddof=1)
    b = np.linalg.solve(Sigma, mu)
    assert ev.realized_sharpe(F @ b, 12.0) == pytest.approx(annualized_sharpe(mu, Sigma, 12.0), rel=1e-10)


# ---------------------------------------------------------------------------
# total_r2 / predictive_r2 (BKS Section 3.2, KKPS footnote 33)
# ---------------------------------------------------------------------------


def test_total_r2_matches_brute_force():
    panel, Gamma, F = make_panel(T=7, N=5, L=3, K=2, seed=1, noise=0.5)
    ssr = 0.0
    sst = 0.0
    for n in range(panel.n_obs):
        c = panel.X[n]
        t = panel.t_idx[n]
        fitted = float(c @ Gamma @ F[t])
        ssr += (panel.y[n] - fitted) ** 2
        sst += panel.y[n] ** 2
    expected = 1.0 - ssr / sst
    assert ev.total_r2(panel, Gamma, F) == pytest.approx(expected, rel=1e-12)
    assert 0.0 < expected < 1.0


def test_total_r2_exact_fit_rotation_invariance_and_dataframe_input():
    panel, Gamma, F = make_panel(T=6, N=4, L=3, K=2, seed=2, noise=0.0)
    assert ev.total_r2(panel, Gamma, F) == pytest.approx(1.0, abs=1e-12)
    rng = np.random.default_rng(5)
    R = rng.normal(size=(2, 2))
    Gamma_r, F_r = Gamma @ R, F @ np.linalg.inv(R).T  # Gamma R, R^-1 f
    panel_noisy, Gn, Fn = make_panel(T=6, N=4, L=3, K=2, seed=2, noise=0.7)
    assert ev.total_r2(panel_noisy, Gn @ R, Fn @ np.linalg.inv(R).T) == pytest.approx(
        ev.total_r2(panel_noisy, Gn, Fn), rel=1e-10
    )
    assert ev.total_r2(panel, Gamma_r, F_r) == pytest.approx(1.0, abs=1e-10)
    Fdf = pd.DataFrame(F, index=panel.periods)
    assert ev.total_r2(panel, Gamma, Fdf) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        ev.total_r2(panel, Gamma[:-1], F)
    with pytest.raises(ValueError):
        ev.total_r2(panel, Gamma, F[:-1])


def test_predictive_r2_matches_brute_force():
    panel, Gamma, F = make_panel(T=7, N=5, L=3, K=2, seed=4, noise=0.3)
    mu = F.mean(axis=0)
    ssr = sum((panel.y[n] - float(panel.X[n] @ Gamma @ mu)) ** 2 for n in range(panel.n_obs))
    sst = float(panel.y @ panel.y)
    assert ev.predictive_r2(panel, Gamma, mu) == pytest.approx(1.0 - ssr / sst, rel=1e-12)
    # predictive R2 equals total R2 when the factors are constant at mu
    Fconst = np.tile(mu, (panel.T, 1))
    assert ev.predictive_r2(panel, Gamma, mu) == pytest.approx(ev.total_r2(panel, Gamma, Fconst), rel=1e-12)
    with pytest.raises(ValueError):
        ev.predictive_r2(panel, Gamma, mu[:-1])


# ---------------------------------------------------------------------------
# price_test_assets / grs_test (D35)
# ---------------------------------------------------------------------------


def simulate_test_assets(T=600, N=5, K=2, alpha=None, seed=0, sigma_e=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-31", periods=T, freq="ME")
    F = rng.normal(loc=[0.006, 0.002][:K], scale=0.04, size=(T, K))
    beta = rng.normal(size=(N, K))
    alpha = np.zeros(N) if alpha is None else np.asarray(alpha, dtype=float)
    E = rng.normal(scale=sigma_e, size=(T, N))
    R = alpha + F @ beta.T + E
    fdf = pd.DataFrame(F, index=idx, columns=[f"f{k+1}" for k in range(K)])
    rdf = pd.DataFrame(R, index=idx, columns=[f"p{j}" for j in range(N)])
    return rdf, fdf, alpha, beta, E


def test_price_test_assets_recovers_alpha_and_beta():
    alpha_true = np.array([0.0, 0.002, -0.003, 0.004, 0.0])
    rdf, fdf, alpha, beta, E = simulate_test_assets(T=2000, N=5, K=2, alpha=alpha_true, seed=11, sigma_e=0.005)
    res = ev.price_test_assets(rdf, fdf, t_crit=1.96, model_name="sim")
    assert res.model_name == "sim"
    # alphas in input units (decimal per period), no scaling by 100
    np.testing.assert_allclose(res.alphas.to_numpy(), alpha, atol=6e-4)
    np.testing.assert_allclose(res.betas.to_numpy(), beta, atol=1e-2)
    assert list(res.betas.columns) == ["f1", "f2"]
    assert list(res.alphas.index) == list(rdf.columns)
    # centred R2 per asset ~ 1 - sigma_e^2 / var(r_j) (T = 2000, so the sample agrees closely)
    expected_r2 = 1.0 - 0.005**2 / rdf.var(ddof=0)
    np.testing.assert_allclose(res.r2.to_numpy(), expected_r2.to_numpy(), atol=0.02)
    # asset with a large alpha and tiny noise is significant; zero-alpha assets mostly not
    assert abs(res.t_stats["p3"]) > 1.96
    assert res.frac_significant == pytest.approx(np.mean(np.abs(res.t_stats.to_numpy()) > 1.96))
    assert res.avg_abs_alpha == pytest.approx(np.mean(np.abs(res.alphas.to_numpy())))
    assert res.avg_abs_t == pytest.approx(np.mean(np.abs(res.t_stats.to_numpy())))
    assert res.grs_stat is not None and res.grs_pvalue is not None
    assert res.grs_pvalue < 0.01  # non-zero alphas are jointly rejected


def test_price_test_assets_tstat_matches_manual_ols_formula():
    rdf, fdf, *_ = simulate_test_assets(T=120, N=3, K=2, seed=5)
    res = ev.price_test_assets(rdf, fdf, t_crit=2.0)
    F = fdf.to_numpy()
    X = np.column_stack([np.ones(len(F)), F])
    for name in rdf.columns:
        y = rdf[name].to_numpy()
        b = np.linalg.solve(X.T @ X, X.T @ y)
        e = y - X @ b
        s2 = e @ e / (len(y) - X.shape[1])
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
        assert res.alphas[name] == pytest.approx(b[0], rel=1e-10, abs=1e-14)
        assert res.t_stats[name] == pytest.approx(b[0] / se, rel=1e-10)
        assert res.r2[name] == pytest.approx(1 - e @ e / np.sum((y - y.mean()) ** 2), rel=1e-10)
    # GRS on the balanced panel equals grs_test on the reported alphas and the OLS residuals
    alphas = res.alphas.to_numpy()
    resid = np.column_stack([rdf[c].to_numpy() - X @ np.linalg.lstsq(X, rdf[c].to_numpy(), rcond=None)[0] for c in rdf.columns])
    stat, p = ev.grs_test(alphas, resid, F)
    assert res.grs_stat == pytest.approx(stat, rel=1e-12)
    assert res.grs_pvalue == pytest.approx(p, rel=1e-12)


def test_price_test_assets_plain_and_newey_west_match_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rdf, fdf, *_ = simulate_test_assets(T=150, N=2, K=2, seed=8)
    X = sm.add_constant(fdf.to_numpy())
    plain = ev.price_test_assets(rdf, fdf, t_crit=1.96)
    nw = ev.price_test_assets(rdf, fdf, t_crit=1.96, nw_lags=6)
    for name in rdf.columns:
        y = rdf[name].to_numpy()
        fit_plain = sm.OLS(y, X).fit()
        fit_nw = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": False})
        assert plain.t_stats[name] == pytest.approx(fit_plain.tvalues[0], rel=1e-8)
        assert nw.t_stats[name] == pytest.approx(fit_nw.tvalues[0], rel=1e-8)
        assert nw.alphas[name] == pytest.approx(plain.alphas[name])  # point estimates unchanged
        assert plain.r2[name] == pytest.approx(fit_plain.rsquared, rel=1e-10)


def test_price_test_assets_unbalanced_panel_uses_asset_sample_and_balanced_grs():
    rdf, fdf, *_ = simulate_test_assets(T=200, N=3, K=1, seed=9)
    rdf = rdf.copy()
    rdf.iloc[:50, 0] = np.nan  # first asset starts later
    res = ev.price_test_assets(rdf, fdf, t_crit=1.96)
    # per-asset regression on its own sample
    full = ev.price_test_assets(rdf.iloc[50:], fdf, t_crit=1.96)
    assert res.alphas["p0"] == pytest.approx(full.alphas["p0"], rel=1e-12)
    only1 = ev.price_test_assets(rdf[["p1"]], fdf, t_crit=1.96)
    assert res.alphas["p1"] == pytest.approx(only1.alphas["p1"], rel=1e-12)
    # GRS uses the balanced sub-sample (rows 50:)
    assert res.grs_stat == pytest.approx(full.grs_stat, rel=1e-12)
    # an asset with too few observations is NaN, not an error
    rdf.iloc[:-2, 2] = np.nan
    res2 = ev.price_test_assets(rdf, fdf, t_crit=1.96)
    assert np.isnan(res2.alphas["p2"]) and np.isnan(res2.t_stats["p2"])
    assert res2.grs_stat is None and res2.grs_pvalue is None


def test_price_test_assets_frac_significant_respects_t_crit():
    rdf, fdf, *_ = simulate_test_assets(T=300, N=6, K=2, seed=12)
    res_low = ev.price_test_assets(rdf, fdf, t_crit=0.0)
    res_high = ev.price_test_assets(rdf, fdf, t_crit=1e6)
    assert res_low.frac_significant == 1.0
    assert res_high.frac_significant == 0.0


def _grs_ml_form(alphas, resid, F):
    """Brute-force GRS in the original ML form: ((T-N-K)/N)(1+mu'Om^-1 mu)^-1 a' S_ML^-1 a."""
    T, N = resid.shape
    K = F.shape[1]
    S = resid.T @ resid / T
    mu = F.mean(axis=0)
    Om = (F - mu).T @ (F - mu) / T
    stat = ((T - N - K) / N) * (alphas @ np.linalg.solve(S, alphas)) / (1 + mu @ np.linalg.solve(Om, mu))
    return stat, _stats.f.sf(stat, N, T - N - K)


def test_grs_equals_ml_form_and_is_scale_invariant():
    rdf, fdf, *_ = simulate_test_assets(T=100, N=4, K=2, seed=21)
    F = fdf.to_numpy()
    alphas, resid = ev._multivariate_ols(rdf.to_numpy(), F)
    stat, p = ev.grs_test(alphas, resid, F)
    stat_ref, p_ref = _grs_ml_form(alphas, resid, F)
    assert stat == pytest.approx(stat_ref, rel=1e-12)
    assert p == pytest.approx(p_ref, rel=1e-12)
    stat2, p2 = ev.grs_test(alphas * 100, resid * 100, F)
    assert stat2 == pytest.approx(stat, rel=1e-12)
    assert p2 == pytest.approx(p, rel=1e-12)
    # 1-D residuals / factors are accepted
    s1, p1 = ev.grs_test(alphas[:1], resid[:, 0], F[:, 0])
    assert s1 is not None and 0 <= p1 <= 1


def test_grs_size_under_the_null():
    """Under alpha = 0 with normal errors the GRS p-value is uniform: 5% rejections in [2%, 10%]."""
    n_sims, T, N, K = 200, 120, 8, 2
    pvals = []
    for s in range(n_sims):
        rdf, fdf, *_ = simulate_test_assets(T=T, N=N, K=K, seed=1000 + s)
        res = ev.price_test_assets(rdf, fdf, t_crit=1.96)
        assert res.grs_pvalue is not None
        pvals.append(res.grs_pvalue)
    pvals = np.asarray(pvals)
    reject = float(np.mean(pvals < 0.05))
    assert 0.02 <= reject <= 0.10, f"reject rate {reject}"
    assert 0.4 <= pvals.mean() <= 0.6
    assert _stats.kstest(pvals, "uniform").pvalue > 0.01


def test_grs_returns_none_when_t_too_small_or_singular():
    rdf, fdf, *_ = simulate_test_assets(T=12, N=8, K=2, seed=3)
    F = fdf.to_numpy()
    alphas, resid = ev._multivariate_ols(rdf.to_numpy(), F)
    stat, p = ev.grs_test(alphas, resid, F)  # T = 12 > N + K + 1 = 11: computed
    assert stat is not None and 0.0 <= p <= 1.0
    rdf, fdf, *_ = simulate_test_assets(T=11, N=8, K=2, seed=3)
    F = fdf.to_numpy()
    alphas, resid = ev._multivariate_ols(rdf.to_numpy(), F)
    assert ev.grs_test(alphas, resid, F) == (None, None)  # T <= N + K + 1
    # duplicated test asset -> singular residual covariance
    rdf, fdf, *_ = simulate_test_assets(T=100, N=3, K=1, seed=4)
    rdf["dup"] = rdf["p0"]
    res = ev.price_test_assets(rdf, fdf, t_crit=1.96)
    assert res.grs_stat is None and res.grs_pvalue is None
    assert np.isfinite(res.alphas).all()  # per-asset stats still reported
    # non-finite inputs
    F = fdf.to_numpy()
    alphas, resid = ev._multivariate_ols(rdf[["p0", "p1"]].to_numpy(), F)
    resid[0, 0] = np.nan
    assert ev.grs_test(alphas, resid, F) == (None, None)
    with pytest.raises(ValueError):
        ev.grs_test(alphas[:1], resid, F)


def test_grs_t_boundary():
    # T = N + K + 2 is the smallest sample that is computed
    N, K = 3, 1
    rdf, fdf, *_ = simulate_test_assets(T=N + K + 2, N=N, K=K, seed=6)
    F = fdf.to_numpy()
    alphas, resid = ev._multivariate_ols(rdf.to_numpy(), F)
    stat, p = ev.grs_test(alphas, resid, F)
    assert stat is not None and np.isfinite(stat) and 0.0 <= p <= 1.0
    rdf, fdf, *_ = simulate_test_assets(T=N + K + 1, N=N, K=K, seed=6)
    F = fdf.to_numpy()
    alphas, resid = ev._multivariate_ols(rdf.to_numpy(), F)
    assert ev.grs_test(alphas, resid, F) == (None, None)


def test_grs_with_one_asset_equals_alpha_t_stat_squared():
    """N = 1: the GRS F(1, T-K-1) statistic must equal the OLS alpha t-statistic squared.

    Pins the finite-sample form of BKS Section 4.1 / D35: with the unbiased residual
    covariance E'E/(T-K-1) the T/(T-K-1) factor is exactly what makes
    ((T-N-K)/N)(1 + mu' Omega^-1 mu)^-1 alpha' Sigma^-1 alpha reduce to t(alpha)^2
    (with (X'X)^-1_{00} = (1 + mu' Omega_ML^-1 mu) / T); an extra or missing
    degrees-of-freedom factor would break the identity.
    """
    for K in (1, 2):  # simulate_test_assets supports K <= 2
        rdf, fdf, *_ = simulate_test_assets(T=90, N=1, K=K, alpha=[0.003], seed=30 + K, sigma_e=0.02)
        res = ev.price_test_assets(rdf, fdf, t_crit=1.96)
        assert res.grs_stat is not None
        assert res.grs_stat == pytest.approx(res.t_stats.iloc[0] ** 2, rel=1e-10)
        # and the p-value is the two-sided t p-value with T-K-1 degrees of freedom
        p_t = 2.0 * _stats.t.sf(abs(res.t_stats.iloc[0]), 90 - K - 1)
        assert res.grs_pvalue == pytest.approx(p_t, rel=1e-10)


def test_newey_west_lag_zero_is_white():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(50), rng.normal(size=(50, 2))])
    e = rng.normal(size=50)
    V = ev.newey_west_cov(X, e, 0)
    XtX_inv = np.linalg.inv(X.T @ X)
    white = XtX_inv @ (X.T @ np.diag(e**2) @ X) @ XtX_inv
    np.testing.assert_allclose(V, white, rtol=1e-10)
    with pytest.raises(ValueError):
        ev.newey_west_cov(X, e, -1)


# ---------------------------------------------------------------------------
# factor_correlations
# ---------------------------------------------------------------------------


def test_factor_correlations_known_values_and_alignment():
    idx = pd.date_range("2000-01-31", periods=100, freq="ME")
    rng = np.random.default_rng(7)
    a = rng.normal(size=100)
    b = rng.normal(size=100)
    F = pd.DataFrame({"f1": a, "f2": b}, index=idx)
    others = pd.DataFrame(
        {"same": a[10:], "neg": -a[10:], "mix": 0.5 * a[10:] + b[10:]}, index=idx[10:]
    )  # partial overlap
    corr = ev.factor_correlations(F, others)
    assert corr.shape == (2, 3)
    assert list(corr.index) == ["f1", "f2"] and list(corr.columns) == ["same", "neg", "mix"]
    assert corr.loc["f1", "same"] == pytest.approx(1.0)
    assert corr.loc["f1", "neg"] == pytest.approx(-1.0)
    common = idx[10:]
    expected = np.corrcoef(F.loc[common, "f2"], others.loc[common, "mix"])[0, 1]
    assert corr.loc["f2", "mix"] == pytest.approx(expected, rel=1e-12)
    # pairwise-complete with NaN; too few observations -> NaN, not an error
    others2 = others.copy()
    others2.iloc[:85, 0] = np.nan
    corr2 = ev.factor_correlations(F, others2)
    assert corr2.loc["f1", "same"] == pytest.approx(1.0)
    others3 = others.copy()
    others3.iloc[:-2, 0] = np.nan
    assert np.isnan(ev.factor_correlations(F, others3).loc["f1", "same"])
    # column-name collisions between the two frames are handled
    clash = pd.DataFrame({"f1": -a}, index=idx)
    assert ev.factor_correlations(F, clash).loc["f1", "f1"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# selection helpers: Jaccard, stability, lambda_max
# ---------------------------------------------------------------------------


def test_jaccard():
    assert ev.jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert ev.jaccard([], []) == 1.0
    assert ev.jaccard(["a"], []) == 0.0
    assert ev.jaccard(["a", "a", "b"], ["b", "a"]) == 1.0


def test_selection_stability_mean_consecutive_jaccard():
    hist = pd.DataFrame(
        [[True, True, False], [True, False, False], [False, False, True]],
        columns=["t1", "t2", "t3"],
    )
    # pairs: {t1,t2} vs {t1} = 1/2 ; {t1} vs {t3} = 0
    assert ev.selection_stability(hist) == pytest.approx(0.25)
    assert np.isnan(ev.selection_stability(hist.iloc[:1]))
    assert np.isnan(ev.selection_stability(None))


def test_lambda_max_by_instrument_on_synthetic_path():
    names = ["const", "t1", "t2", "t3"]
    path = [
        make_path_point(0.01, 3, [1.0, 0.5, 0.2, 0.0]),
        make_path_point(0.10, 3, [1.0, 0.4, 0.0, 0.0]),
        make_path_point(1.00, 3, [0.9, 0.0, 0.0, 0.0]),
        make_path_point(5.00, 2, [0.9, 0.3, 0.3, 0.3]),  # different K: ignored when K=3
    ]
    lm = ev.lambda_max_by_instrument(path, names, K=3)
    assert list(lm.index) == names
    assert lm["const"] == 1.0
    assert lm["t1"] == 0.10
    assert lm["t2"] == 0.01
    assert np.isnan(lm["t3"])  # never selected on the K=3 path
    lm_all = ev.lambda_max_by_instrument(path, names)  # all K
    assert lm_all["t3"] == 5.0 and lm_all["t1"] == 5.0
    # order of the path does not matter
    lm_rev = ev.lambda_max_by_instrument(path[::-1], names, K=3)
    pd.testing.assert_series_equal(lm, lm_rev)
    # constant row omitted from gamma_norms is aligned with names[1:]
    path_l = [make_path_point(0.3, 3, [0.0, 0.0, 0.7])]
    lm2 = ev.lambda_max_by_instrument(path_l, names, K=3)
    assert np.isnan(lm2["const"]) and lm2["t3"] == 0.3
    with pytest.raises(ValueError):
        ev.lambda_max_by_instrument([make_path_point(0.3, 3, [0.0, 0.7])], names)


# ---------------------------------------------------------------------------
# placebo_test through stubbed stage modules (the stages are written by other agents)
# ---------------------------------------------------------------------------


def _install_stub(monkeypatch, name, **funcs):
    mod = _pytypes.ModuleType(f"narrative_ipca.{name}")
    for k, v in funcs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, f"narrative_ipca.{name}", mod)
    monkeypatch.setattr(narrative_ipca, name, mod, raising=False)
    return mod


def test_placebo_test_wiring_and_result_fields(monkeypatch):
    calls = {}
    days = pd.date_range("2000-01-03", periods=30, freq="B")
    z = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 3)), index=days, columns=["t1", "t2", "t3"])
    shocks = ShockPanel(z=z, window=5)
    returns = pd.DataFrame(np.random.default_rng(1).normal(size=(30, 2)), index=days, columns=["a", "b"])
    cfg = PipelineConfig()

    def append_placebos(sp, n, seed):
        calls["append"] = (sp, n, seed)
        rng = np.random.default_rng(seed)
        z2 = sp.z.copy()
        for k in range(n):
            z2[f"placebo_{k}"] = rng.normal(size=len(z2))
        mask = np.array([False] * sp.z.shape[1] + [True] * n)
        return ShockPanel(z=z2, window=sp.window), mask

    def build_covariance_panel(sp, rets, cov_cfg, period):
        calls["cov"] = (sp, rets, cov_cfg, period)
        return "COV"

    # augmented panel: const + 3 real + 2 placebo instruments
    panel, Gamma, F = make_panel(T=6, N=4, L=5, K=2, seed=3)
    panel.instrument_names[:] = ["const", "t1", "t2", "t3", "placebo_0", "placebo_1"]

    def build_panel(cov, rets, data_cfg, cov_cfg):
        calls["panel"] = (cov, rets, data_cfg, cov_cfg)
        return panel

    # tuned fit selects t1, t3 and placebo_1
    fit = make_fit(panel, Gamma, F, lam=0.2, selected=[True, False, True, False, True])
    fit.instrument_names = list(panel.instrument_names)
    path = [
        make_path_point(0.05, 2, [1, 1, 1, 1, 1, 1]),
        make_path_point(0.20, 2, [1, 1, 0, 1, 0, 1]),
        make_path_point(0.80, 2, [1, 1, 0, 0, 0, 0]),
    ]
    tr = TuningResult(lam=0.2, K=2, criterion="is_sharpe", path=path, fit=fit, lam_max=1.6)

    def tune(pnl, est_cfg, tune_cfg, eval_cfg):
        calls["tune"] = (pnl, est_cfg, tune_cfg, eval_cfg)
        return tr

    _install_stub(monkeypatch, "shocks", append_placebos=append_placebos)
    _install_stub(monkeypatch, "covariances", build_covariance_panel=build_covariance_panel)
    _install_stub(monkeypatch, "panel", build_panel=build_panel)
    _install_stub(monkeypatch, "tuning", tune=tune)

    # reference fit (no placebos) selected t1 and t2
    ref_panel, rG, rF = make_panel(T=6, N=4, L=3, K=2, seed=4)
    ref = make_fit(ref_panel, rG, rF, lam=0.15, selected=[True, True, False])
    ref.instrument_names = ["const", "t1", "t2", "t3"]

    res = ev.placebo_test(shocks, returns, cfg, ref, n=2, seed=99)

    # wiring against the DESIGN.md Part C signatures
    assert calls["append"][1:] == (2, 99)
    assert calls["cov"][1] is returns and calls["cov"][2] is cfg.covariance and calls["cov"][3] == cfg.data.period
    assert calls["cov"][0].topics == ["t1", "t2", "t3", "placebo_0", "placebo_1"]
    assert calls["panel"] == ("COV", returns, cfg.data, cfg.covariance)
    assert calls["tune"] == (panel, cfg.estimation, cfg.tuning, cfg.evaluation)

    assert isinstance(res, PlaceboResult)
    assert res.n_placebo == 2
    assert res.n_placebo_selected == 1
    assert res.n_real_selected == 2
    assert res.real_selected_before == ["t1", "t2"]
    assert res.real_selected_after == ["t1", "t3"]
    assert res.jaccard_real == pytest.approx(1 / 3)
    assert res.lam_star == 0.2
    assert list(res.lam_max_by_instrument.index) == ["t1", "t2", "t3", "placebo_0", "placebo_1"]
    assert res.lam_max_by_instrument["t1"] == 0.8
    assert res.lam_max_by_instrument["t2"] == 0.05
    assert res.lam_max_by_instrument["t3"] == 0.2
    assert res.lam_max_by_instrument["placebo_0"] == 0.05
    assert res.lam_max_by_instrument["placebo_1"] == 0.2
    with pytest.raises(ValueError):
        ev.placebo_test(shocks, returns, cfg, ref, n=0, seed=1)


# ---------------------------------------------------------------------------
# evaluate_run on hand-built result objects
# ---------------------------------------------------------------------------

BASE_KEYS = {
    "total_r2", "pred_r2", "mve_sharpe_is", "lam_star", "K", "n_selected", "n_obs", "T", "N", "L",
    "converged", "n_iter",
}
OOS_KEYS = {"oos_sharpe", "oos_n_periods", "oos_mean_n_selected", "oos_selection_stability"}


def test_evaluate_run_minimal_produces_metric_keys():
    panel, Gamma, F = make_panel(T=8, N=6, L=4, K=2, seed=10, noise=0.4, drop_period=3)
    fit = make_fit(panel, Gamma, F, lam=0.3, selected=[True, True, False, True], converged=False, n_iter=42)
    tuning = make_tuning(fit)
    rep = ev.evaluate_run(panel, tuning, fit, None, None, PipelineConfig())
    m = rep.metrics
    assert BASE_KEYS <= set(m)
    assert not (OOS_KEYS & set(m))
    assert m["total_r2"] == pytest.approx(ev.total_r2(panel, Gamma, F))
    assert m["pred_r2"] == pytest.approx(ev.predictive_r2(panel, Gamma, fit.mu_f))
    assert m["mve_sharpe_is"] == pytest.approx(annualized_sharpe(fit.mu_f, fit.Sigma_ff, 12.0))
    assert m["lam_star"] == 0.3 and m["K"] == 2 and m["n_selected"] == 3
    assert m["n_obs"] == panel.n_obs and m["T"] == 8 and m["N"] == 6 and m["L"] == 4
    assert m["n_populated_periods"] == 7
    assert m["converged"] == 0 and m["n_iter"] == 42
    assert m["lam_max"] == tuning.lam_max and m["n_path_points"] == 3
    assert isinstance(rep.lambda_path, pd.DataFrame) and len(rep.lambda_path) == 3
    assert rep.selected_topics == ["topic_0", "topic_1", "topic_3"]
    assert list(rep.gamma_norms.index) == panel.instrument_names
    np.testing.assert_allclose(rep.gamma_norms.to_numpy(), fit.gamma_norms)
    sel = rep.tables["selected"]
    assert list(sel.columns) == ["topic", "gamma_norm", "rank"]
    assert set(sel["topic"]) == set(rep.selected_topics)
    assert list(sel["rank"]) == [1, 2, 3]
    assert sel["gamma_norm"].is_monotonic_decreasing
    assert rep.pricing_tests == {} and rep.factor_correlations is None and rep.placebo is None
    # an EvaluationConfig is accepted in place of the PipelineConfig
    rep2 = ev.evaluate_run(panel, None, fit, None, None, EvaluationConfig(annualization=1.0))
    assert rep2.metrics["mve_sharpe_is"] == pytest.approx(m["mve_sharpe_is"] / np.sqrt(12.0))
    assert rep2.lambda_path is None and "lam_max" not in rep2.metrics
    assert all(isinstance(v, (int, float, np.integer, np.floating)) for v in m.values())


def test_evaluate_run_with_oos_wrapup_pricing_and_benchmarks():
    T = 60
    panel, Gamma, F = make_panel(T=T, N=8, L=4, K=2, seed=20, noise=0.4)
    fit = make_fit(panel, Gamma, F, lam=0.3)
    tuning = make_tuning(fit)
    rng = np.random.default_rng(21)
    oos_idx = panel.periods[36:]
    oos_F = pd.DataFrame(rng.normal(size=(len(oos_idx), 2)), index=oos_idx, columns=["f1", "f2"])
    mve = pd.Series(rng.normal(loc=0.1, size=len(oos_idx)), index=oos_idx)
    mve.iloc[2] = np.nan
    refits = [oos_idx[0], oos_idx[12]]
    sel_hist = pd.DataFrame([[True, True, False, False], [True, False, True, False]], index=refits, columns=panel.topics)
    oos = OOSResult(
        factors=oos_F,
        mve=mve,
        sharpe=ev.realized_sharpe(mve, 12.0),
        refit_periods=refits,
        lam_history=pd.Series([0.2, 0.4], index=refits),
        K_history=pd.Series([2, 2], index=refits),
        n_selected_history=pd.Series([2, 2], index=refits),
        gamma_norm_history=pd.DataFrame(np.ones((2, 5)), index=refits, columns=panel.instrument_names),
        selected_history=sel_hist,
        is_sharpe_history=pd.Series([1.1, 1.2], index=refits),
    )
    days = pd.date_range("2001-01-01", periods=20, freq="B")
    wrapup = WrapUpResult(
        A=pd.DataFrame(np.zeros((4, 2)), index=panel.topics),
        states=pd.DataFrame(np.zeros((20, 2)), index=days),
        x_mve=pd.Series(np.zeros(20), index=days),
        b_mve=fit.b_mve(),
        impact_z_to_x=pd.DataFrame(np.zeros((4, 2)), index=panel.topics),
        impact_z_to_mve=pd.Series(np.zeros(4), index=panel.topics),
        obs_projection={"mkt": {"b_obs": np.ones(2), "r2": 0.93}},
        rank_deficient=True,
    )
    placebo = PlaceboResult(
        n_placebo=5, n_placebo_selected=0, n_real_selected=3,
        lam_max_by_instrument=pd.Series([0.1] * 4, index=panel.topics), lam_star=0.3,
        real_selected_before=["topic_0"], real_selected_after=["topic_0", "topic_1"], jaccard_real=0.5,
    )
    test_assets = pd.DataFrame(rng.normal(size=(T, 3)), index=panel.periods, columns=["ta1", "ta2", "ta3"])
    bench = pd.DataFrame(rng.normal(size=(T, 2)), index=panel.periods, columns=["Mkt", "SMB"])

    rep = ev.evaluate_run(panel, tuning, fit, oos, wrapup, PipelineConfig(), test_assets=test_assets,
                          benchmark_factors=bench, placebo=placebo)
    m = rep.metrics
    assert BASE_KEYS | OOS_KEYS <= set(m)
    assert m["oos_sharpe"] == pytest.approx(ev.realized_sharpe(mve, 12.0))
    assert m["oos_n_periods"] == len(oos_idx) - 1
    assert m["oos_n_refits"] == 2
    assert m["oos_mean_n_selected"] == 2.0
    assert m["oos_mean_lam"] == pytest.approx(0.3)
    assert m["oos_selection_stability"] == pytest.approx(1 / 3)  # {0,1} vs {0,2}
    assert m["wrapup_rank_deficient"] == 1 and m["obs_r2_mkt"] == pytest.approx(0.93)
    assert m["placebo_n"] == 5 and m["placebo_n_selected"] == 0 and m["placebo_jaccard_real"] == 0.5
    assert rep.placebo is placebo
    assert set(rep.pricing_tests) == {"narrative_is", "narrative_oos", "benchmark"}
    for key, res in rep.pricing_tests.items():
        assert res.model_name == key
        assert list(res.alphas.index) == ["ta1", "ta2", "ta3"]
        assert m[f"pricing_{key}_avg_abs_alpha"] == pytest.approx(res.avg_abs_alpha)
        assert m[f"pricing_{key}_grs_pvalue"] == pytest.approx(res.grs_pvalue)
    # in-sample factors are the fit's factors: the pricing test is the direct call
    direct = ev.price_test_assets(test_assets, fit.factors_frame(), 1.96, "narrative_is")
    pd.testing.assert_series_equal(rep.pricing_tests["narrative_is"].alphas, direct.alphas)
    assert rep.pricing_tests["narrative_oos"].betas.shape == (3, 2)
    assert list(rep.pricing_tests["benchmark"].betas.columns) == ["Mkt", "SMB"]
    assert rep.factor_correlations.shape == (3, 2)
    assert list(rep.factor_correlations.index) == ["f1", "f2", "mve"]
    expected_mve_corr = np.corrcoef(fit.F @ fit.b_mve(), bench["Mkt"])[0, 1]
    assert rep.factor_correlations.loc["mve", "Mkt"] == pytest.approx(expected_mve_corr, rel=1e-10)
    assert set(rep.tables) == {"selected", "oos_refits", "pricing"}
    assert list(rep.tables["pricing"].index) == ["narrative_is", "narrative_oos", "benchmark"]
    assert list(rep.tables["oos_refits"].columns) == ["lam", "K", "n_selected", "is_sharpe"]
