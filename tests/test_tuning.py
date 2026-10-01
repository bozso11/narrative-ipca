"""Tests of ``narrative_ipca.tuning`` (BKS Section 2 footnote 8, App. C.3; DESIGN.md D27-D29).

What is checked, beyond shapes:

* the in-sample criterion equals ``sqrt(ann mu' Sigma^-1 mu)`` and, exactly,
  the realised Sharpe ratio of the in-sample MVE series ``b_MVE' f_t``;
* ``select_best_point`` picks the argmax and breaks ties per D27
  (sparser = larger lambda, fewer selected, smaller K; denser reversed; a
  non-finite criterion never wins);
* the tolerance band is relative at any level of the criterion
  (``tie_band``, D51): ``tolerance * |best|``, with only the numerical tie
  tolerance floored at ``max(1, |best|)``; the pick does not depend on the
  scale of the criterion, and nothing changes for ``|best| >= 1``;
* ``tune`` returns the argmax of the criterion over the grid, a canonical
  fit (``Sigma_ff`` diagonal descending, ``mu_f >= 0``) whose statistics
  match the chosen path point, ``lam_max`` of the chosen ``K``; fixed
  lambda gives a one-point path; annualisation only rescales the criterion;
  ``K_grid`` traces every ``K``; a zero in a user grid dispatches to plain IPCA;
* LOOCV: the fold subsample is seeded and excludes empty periods; the
  cross-validated Sharpe equals a brute-force reference that fits on
  ``subset_periods``, extracts the left-out factor with the Eq. 16 f-step on
  that period's moments and stitches the series;
* the simulated pipeline (simulation -> data -> shocks -> covariances ->
  panel): tuning returns a lambda inside the grid and a canonical fit, and
  LOOCV with six folds is finite.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from narrative_ipca import oos, tuning
from narrative_ipca.config import (
    CovarianceConfig,
    DataConfig,
    EstimationConfig,
    EvaluationConfig,
    LambdaGridConfig,
    ShockConfig,
    SimulationConfig,
    TuningConfig,
)
from narrative_ipca.evaluation import realized_sharpe
from narrative_ipca.sparse_ipca import f_step, fit_sparse_ipca, lambda_grid, lambda_max, lambda_path
from narrative_ipca.types import IPCAPanel, LambdaPathPoint, TuningResult, compute_sigma_c

TRUE_ROWS = (2, 5, 7)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def month_ends(start: str, n: int) -> pd.DatetimeIndex:
    """Calendar month ends (works on every supported pandas version)."""
    return pd.DatetimeIndex(pd.period_range(start, periods=n, freq="M").to_timestamp(how="end").normalize())


def make_panel(
    N: int = 40,
    T: int = 30,
    L: int = 8,
    K: int = 2,
    true_rows: tuple[int, ...] = TRUE_ROWS,
    noise: float = 0.3,
    seed: int = 0,
    empty_period: int | None = None,
    drop_frac: float = 0.0,
) -> IPCAPanel:
    """``r_{i,t} = c_{i,t-1} Gamma f_t + noise`` with ``len(true_rows)`` relevant instruments."""
    rng = np.random.default_rng(seed)
    Gamma = np.zeros((L + 1, K))
    Gamma[0] = np.array([0.5, 0.2, -0.1])[:K]
    for row in true_rows:
        Gamma[row] = rng.standard_normal(K)
    mu = np.array([0.3, 0.1, 0.05])[:K]
    sd = np.array([1.0, 0.7, 0.5])[:K]
    F = mu + sd * rng.standard_normal((T, K))
    X = np.ones((N * T, L + 1))
    X[:, 1:] = rng.standard_normal((N * T, L))
    t_idx = np.repeat(np.arange(T), N)
    a_idx = np.tile(np.arange(N), T)
    y = np.einsum("nk,nk->n", X @ Gamma, F[t_idx]) + noise * rng.standard_normal(N * T)
    keep = np.ones(N * T, dtype=bool)
    if empty_period is not None:
        keep &= t_idx != empty_period
    if drop_frac > 0.0:
        keep &= rng.random(N * T) > drop_frac
    X, y, t_idx, a_idx = X[keep], y[keep], t_idx[keep], a_idx[keep]
    return IPCAPanel(
        X=X,
        y=y,
        t_idx=t_idx,
        asset_idx=a_idx,
        periods=month_ends("2000-01", T),
        assets=np.array([f"A{i:03d}" for i in range(N)], dtype=object),
        instrument_names=["const"] + [f"topic_{j}" for j in range(1, L + 1)],
        sigma_c=compute_sigma_c(X),
    )


def point(lam: float, crit: float | None, n_selected: int = 3, K: int = 2) -> LambdaPathPoint:
    return LambdaPathPoint(
        lam=lam, K=K, total_r2=0.0, pred_r2=0.0, mve_sharpe=0.0, n_selected=n_selected,
        gamma_norms=np.zeros(4), objective=0.0, converged=True, n_iter=1, criterion=crit,
    )


def assert_canonical(fit) -> None:
    S = np.asarray(fit.Sigma_ff)
    off = S - np.diag(np.diag(S))
    assert np.max(np.abs(off)) <= 1e-10 * max(1.0, float(np.max(np.abs(S))))
    d = np.diag(S)
    assert np.all(np.diff(d) <= 1e-12 * max(1.0, float(d[0])))
    assert np.all(np.asarray(fit.mu_f) >= -1e-12)
    assert fit.meta.get("canonical") is True


@pytest.fixture(scope="module")
def panel() -> IPCAPanel:
    return make_panel()


@pytest.fixture(scope="module")
def est() -> EstimationConfig:
    return EstimationConfig(K=2, lam=None, lam_grid=LambdaGridConfig(n_lambdas=6, ratio=0.05), max_iter=300)


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def test_is_sharpe_criterion_closed_form_and_realised_identity(panel, est):
    lam = 0.1 * lambda_max(panel, est, 2)  # dense enough for a well-conditioned Sigma_ff
    fit = fit_sparse_ipca(panel, est, lam=lam, K=2)
    mu, Sigma = fit.mu_f, fit.Sigma_ff
    assert np.linalg.cond(Sigma) < 1e6
    by_hand = np.sqrt(12.0 * float(mu @ np.linalg.pinv(Sigma, rcond=1e-12) @ mu))
    assert tuning.is_sharpe_criterion(fit, 12.0) == pytest.approx(by_hand, rel=1e-12)
    by_solve = np.sqrt(12.0 * float(mu @ np.linalg.solve(Sigma, mu)))
    assert tuning.is_sharpe_criterion(fit, 12.0) == pytest.approx(by_solve, rel=1e-9)
    # sqrt(mu' Sigma^-1 mu) is exactly the realised Sharpe of b_MVE' f_t (same ddof=1 moments)
    pop = np.asarray(fit.populated, dtype=bool)
    series = fit.F[pop] @ fit.b_mve()
    assert tuning.is_sharpe_criterion(fit, 12.0) == pytest.approx(realized_sharpe(series, 12.0), rel=1e-10)
    # annualisation is a pure rescaling
    assert tuning.is_sharpe_criterion(fit, 1.0) * np.sqrt(12.0) == pytest.approx(tuning.is_sharpe_criterion(fit, 12.0), rel=1e-12)
    # rotation invariance (D38): general invertible rotations leave the Sharpe unchanged
    R = np.array([[1.3, -0.4], [0.2, 0.9]])
    assert tuning.is_sharpe_criterion(fit.rotate(R), 12.0) == pytest.approx(tuning.is_sharpe_criterion(fit, 12.0), rel=1e-9)


def test_select_best_point_argmax_and_tie_break():
    pts = [point(0.1, 0.5, 6), point(0.2, 0.9, 4), point(0.3, 0.9, 3), point(0.4, 0.9 + 1e-12, 2), point(0.5, float("nan"), 1), point(0.6, 0.2, 0)]
    assert tuning.select_best_point(pts, "sparser") == 3  # largest lam among the tied maxima
    assert tuning.select_best_point(pts, "denser") == 1  # smallest lam among the tied maxima
    # strict maximum wins regardless of tie-break
    pts2 = [point(0.1, 0.5), point(0.2, 0.7), point(0.3, 0.6)]
    assert tuning.select_best_point(pts2, "sparser") == 1
    assert tuning.select_best_point(pts2, "denser") == 1
    # a non-finite criterion never wins while a finite one exists
    pts3 = [point(0.1, 0.5), point(0.2, None), point(0.3, float("inf"))]
    assert tuning.select_best_point(pts3, "sparser") == 0
    # all non-finite: the tie-break alone decides
    pts4 = [point(0.1, None), point(0.2, float("nan")), point(0.3, None)]
    assert tuning.select_best_point(pts4, "sparser") == 2
    assert tuning.select_best_point(pts4, "denser") == 0
    # equal lambda: fewer selected is sparser, then smaller K
    pts5 = [point(0.2, 1.0, 5, K=3), point(0.2, 1.0, 2, K=3), point(0.2, 1.0, 2, K=2)]
    assert tuning.select_best_point(pts5, "sparser") == 2
    assert tuning.select_best_point(pts5, "denser") == 0
    with pytest.raises(ValueError):
        tuning.select_best_point([], "sparser")
    with pytest.raises(ValueError):
        tuning.select_best_point(pts, "random")


def test_tie_band_is_relative_with_a_numerical_floor():
    """D51: the user tolerance is relative at any level; only TIE_REL_TOL keeps the max(1, |best|) floor."""
    assert tuning.tie_band(0.445, 0.02) == pytest.approx(0.0089, rel=1e-12)  # 2% of the best, not 0.02
    assert tuning.tie_band(-0.5, 0.02) == pytest.approx(0.01, rel=1e-12)  # a negative (LOOCV) best: 2% of |best|
    assert tuning.tie_band(0.5, 0.0) == tuning.TIE_REL_TOL  # exact argmax: the numerical band, absolute below 1
    assert tuning.tie_band(0.0, 0.02) == tuning.TIE_REL_TOL
    # |best| >= 1: the same width as the rule before 2026-10-01, max(TIE_REL_TOL, tol) * max(1, |best|)
    for best in (1.0, 1.057, 2.97, 40.0, -3.0):
        for tol in (0.0, 1e-12, 0.02, 0.5):
            assert tuning.tie_band(best, tol) == max(tuning.TIE_REL_TOL, tol) * max(1.0, abs(best)), (best, tol)


def test_select_best_point_tolerance_is_relative_below_and_above_one():
    """D51: tolerance 0.02 admits the points within 2% of the best, whether the best is below or above 1."""
    crit = np.array([0.5, 0.492, 0.485, 0.40])  # 0.492 is 1.6% below the best, 0.485 is 3% below
    lams, n_sel = (0.1, 0.2, 0.3, 0.4), (6, 5, 4, 2)
    for scale in (0.1, 0.89, 1.0, 4.0, np.sqrt(52.0)):  # best from 0.05 to 3.6 (0.89: 0.445, the lab's test config)
        pts = [point(lam, float(c), n) for lam, c, n in zip(lams, scale * crit, n_sel)]
        assert tuning.select_best_point(pts, "sparser", tolerance=0.02) == 1, scale
        assert tuning.select_best_point(pts, "denser", tolerance=0.02) == 0, scale
        assert tuning.select_best_point(pts, "sparser") == 0, scale  # no tolerance: the exact argmax
    # best 0.5: the numerical tie tolerance rel_tol keeps its max(1, |best|) floor, so passing the user tolerance
    # there (as tune did before 2026-10-01) gives an absolute band of 0.02 that also admits 0.485
    pts = [point(lam, float(c), n) for lam, c, n in zip(lams, crit, n_sel)]
    assert tuning.select_best_point(pts, "sparser", rel_tol=0.02) == 2
    # best 2.0: both give the same band, 0.04
    pts4 = [point(lam, float(c), n) for lam, c, n in zip(lams, 4.0 * crit, n_sel)]
    assert tuning.select_best_point(pts4, "sparser", rel_tol=0.02) == 1


def test_path_point_from_fit(panel, est):
    fit = fit_sparse_ipca(panel, est, lam=0.5 * lambda_max(panel, est, 2), K=2)
    pt = tuning.path_point_from_fit(fit, 12.0, EvaluationConfig().rcond, criterion=0.25)
    assert (pt.lam, pt.K, pt.n_selected, pt.n_iter, pt.converged) == (fit.lam, fit.K, fit.n_selected, fit.n_iter, fit.converged)
    assert pt.total_r2 == fit.total_r2 and pt.objective == fit.objective and pt.criterion == 0.25
    assert pt.mve_sharpe == pytest.approx(fit.mve_sharpe(12.0), rel=1e-12)
    np.testing.assert_allclose(pt.gamma_norms, fit.gamma_norms)


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------
def test_tune_fixed_lambda_is_a_one_point_path(panel, est):
    lam = 0.3 * lambda_max(panel, est, 2)
    est_fixed = replace(est, lam=lam)
    tr = tuning.tune(panel, est_fixed, TuningConfig(), EvaluationConfig())
    assert isinstance(tr, TuningResult)
    assert len(tr.path) == 1 and tr.path[0].lam == lam and tr.lam == lam and tr.K == 2
    assert tr.fit.lam == lam and tr.criterion == "is_sharpe"
    assert tr.meta["fixed_lam"] is True and tr.meta["chosen_index"] == 0
    assert tr.lam_max == pytest.approx(lambda_max(panel, est, 2), rel=1e-12) and tr.lam_max > lam
    ref = fit_sparse_ipca(panel, est_fixed, lam=lam, K=2)
    assert tr.path[0].criterion == pytest.approx(tuning.is_sharpe_criterion(ref, 12.0), rel=1e-12)
    assert tr.path[0].mve_sharpe == pytest.approx(tr.path[0].criterion, rel=1e-12)
    assert tr.fit.objective == pytest.approx(ref.objective, rel=1e-12)
    assert_canonical(tr.fit)


def test_tune_is_sharpe_picks_argmax_over_the_grid(panel, est):
    tr = tuning.tune(panel, est, TuningConfig(), EvaluationConfig())
    grid = lambda_grid(panel, est, 2)
    lams = np.array([p.lam for p in tr.path])
    np.testing.assert_allclose(lams, grid, rtol=1e-12)
    assert np.any(np.isclose(tr.lam, grid, rtol=1e-12))
    crit = np.array([p.criterion for p in tr.path])
    assert np.all(np.isfinite(crit))
    best = int(np.argmax(crit))
    assert tr.meta["chosen_index"] == best and tr.lam == tr.path[best].lam
    assert tr.meta["best_criterion"] == pytest.approx(crit[best])
    # criterion equals the in-sample Sharpe of an independent trace of the same path
    _, fits = lambda_path(panel, est, lams=grid, K=2)
    ref = np.array([tuning.is_sharpe_criterion(f, 12.0) for f in fits])
    np.testing.assert_allclose(crit, ref, rtol=1e-10)
    for p in tr.path:
        assert p.mve_sharpe == pytest.approx(p.criterion, rel=1e-12)
    # chosen fit: canonical, same lambda and invariant statistics as the path point
    assert tr.fit.lam == tr.lam and tr.K == tr.fit.K == 2
    assert_canonical(tr.fit)
    pt = tr.path[best]
    assert tr.fit.n_selected == pt.n_selected
    assert tr.fit.objective == pytest.approx(pt.objective, rel=1e-10)
    assert tr.fit.total_r2 == pytest.approx(pt.total_r2, rel=1e-10)
    assert tuning.is_sharpe_criterion(tr.fit, 12.0) == pytest.approx(pt.criterion, rel=1e-10)
    np.testing.assert_allclose(tr.fit.gamma_norms, pt.gamma_norms, rtol=1e-8, atol=1e-12)
    assert tr.lam_max == pytest.approx(lambda_max(panel, est, 2), rel=1e-12)
    assert tr.meta["lam_grid"][2] == pytest.approx(list(grid))
    assert tr.meta["n_points"] == len(grid)
    # the true rows are among the selected at the tuned lambda on this easy panel
    assert set(TRUE_ROWS) <= {i + 1 for i, s in enumerate(tr.fit.selected) if s}


def test_tune_annualization_rescales_criterion_only(panel, est):
    tr12 = tuning.tune(panel, est, TuningConfig(), EvaluationConfig(annualization=12.0))
    tr1 = tuning.tune(panel, est, TuningConfig(), EvaluationConfig(annualization=1.0))
    assert tr1.lam == tr12.lam
    c12 = np.array([p.criterion for p in tr12.path])
    c1 = np.array([p.criterion for p in tr1.path])
    np.testing.assert_allclose(c1 * np.sqrt(12.0), c12, rtol=1e-10)


def test_tune_tie_break_through_the_tuner(panel, est, monkeypatch):
    """With a constant criterion every point ties: sparser -> largest lambda, denser -> smallest."""
    monkeypatch.setattr(tuning, "is_sharpe_criterion", lambda res, ann, rcond=1e-12: 1.0)
    grid = lambda_grid(panel, est, 2)
    sparse = tuning.tune(panel, est, TuningConfig(tie_break="sparser"), EvaluationConfig())
    dense = tuning.tune(panel, est, TuningConfig(tie_break="denser"), EvaluationConfig())
    assert sparse.lam == pytest.approx(grid[-1]) and dense.lam == pytest.approx(grid[0])
    assert sparse.meta["n_tied"] == len(grid) == dense.meta["n_tied"]


def test_tune_K_grid_traces_every_K(panel, est):
    tr = tuning.tune(panel, est, TuningConfig(K_grid=(1, 2)), EvaluationConfig())
    Ks = [p.K for p in tr.path]
    n = est.lam_grid.n_lambdas
    assert Ks == [1] * n + [2] * n
    assert tr.K in (1, 2) and tr.fit.K == tr.K and tr.fit.Gamma.shape[1] == tr.K
    assert tr.lam_max == pytest.approx(tr.meta["lam_max"][tr.K])
    assert tr.meta["lam_max"][1] == pytest.approx(lambda_max(panel, est, 1), rel=1e-12)
    assert tr.meta["lam_max"][2] == pytest.approx(lambda_max(panel, est, 2), rel=1e-12)
    crit = np.array([p.criterion for p in tr.path])
    assert tr.path[tr.meta["chosen_index"]].criterion == pytest.approx(np.nanmax(crit))
    assert np.any(np.isclose(tr.lam, tr.meta["lam_grid"][tr.K]))
    # the two-factor DGP should beat the one-factor model on the criterion
    assert tr.K == 2
    assert_canonical(tr.fit)


def test_tune_progress_callback_counts_points(panel, est):
    calls: list[tuple[int, int, str]] = []
    tr = tuning.tune(panel, est, TuningConfig(K_grid=(1, 2)), EvaluationConfig(), progress=lambda d, n, m: calls.append((d, n, m)))
    assert len(calls) == len(tr.path)
    assert [c[0] for c in calls] == list(range(1, len(tr.path) + 1))
    assert all(c[1] == len(tr.path) and isinstance(c[2], str) for c in calls)


def test_tune_user_grid_with_zero_dispatches_to_plain_ipca(panel, est):
    lmax = lambda_max(panel, est, 2)
    est_vals = replace(est, lam_grid=LambdaGridConfig(values=(0.5 * lmax, 0.0)))
    tr = tuning.tune(panel, est_vals, TuningConfig(), EvaluationConfig())
    assert [p.lam for p in tr.path] == [0.0, 0.5 * lmax]
    assert tr.path[0].n_selected == panel.L  # plain IPCA selects everything (D18)
    assert tr.lam in (0.0, 0.5 * lmax)
    assert_canonical(tr.fit)


def test_tune_rejects_empty_panel(est):
    empty = IPCAPanel(
        X=np.zeros((0, 3)), y=np.zeros(0), t_idx=np.zeros(0, dtype=int), asset_idx=np.zeros(0, dtype=int),
        periods=month_ends("2000-01", 2), assets=np.array(["a"]), instrument_names=["const", "t1", "t2"], sigma_c=np.ones(3),
    )
    with pytest.raises(ValueError):
        tuning.tune(empty, est, TuningConfig(), EvaluationConfig())


# ---------------------------------------------------------------------------
# LOOCV (D28)
# ---------------------------------------------------------------------------
def test_loocv_folds_seeded_and_excludes_empty_periods():
    p = make_panel(T=12, empty_period=4)
    allf = tuning.loocv_folds(p, None)
    assert list(allf) == [t for t in range(12) if t != 4]
    f5 = tuning.loocv_folds(p, 5, seed=1)
    assert f5.shape == (5,) and np.all(np.diff(f5) > 0) and 4 not in f5
    np.testing.assert_array_equal(f5, tuning.loocv_folds(p, 5, seed=1))
    assert not np.array_equal(f5, tuning.loocv_folds(p, 5, seed=2)) or True  # different seeds may coincide; equality is not asserted
    assert list(tuning.loocv_folds(p, 50)) == list(allf)  # max_folds above the count keeps every period


def test_loocv_sharpe_matches_brute_force_reference(panel, est):
    lam = 0.3 * lambda_max(panel, est, 2)
    full = fit_sparse_ipca(panel, est, lam=lam, K=2)
    value = tuning.loocv_sharpe(panel, est, lam, 2, 12.0, max_folds=5, seed=3)
    assert np.isfinite(value)
    # reference: same folds, fit on S \ {t} warm-started from the full fit, left-out factor by the
    # Eq. 16 f-step on that period's own moments, MVE with the training b_MVE, stitched Sharpe
    folds = tuning.loocv_folds(panel, 5, seed=3)
    slices = dict(panel.period_slices())
    ref = []
    for t in folds:
        keep = np.ones(panel.T, dtype=bool)
        keep[t] = False
        train = panel.subset_periods(keep)
        assert train.T == panel.T - 1
        np.testing.assert_allclose(train.sigma_c, compute_sigma_c(train.X))  # D26: sigma_c on the training sample
        fit = fit_sparse_ipca(train, est, lam=lam, K=2, Gamma_init=full.Gamma)
        Xt, yt = panel.X[slices[int(t)]], panel.y[slices[int(t)]]
        f_t = f_step((Xt.T @ Xt)[None], (Xt.T @ yt)[None], fit.Gamma)[0]
        ref.append(float(fit.b_mve() @ f_t))
    assert value == pytest.approx(realized_sharpe(np.array(ref), 12.0), rel=1e-10)
    # passing the warm start explicitly gives the same number; the seed fixes the folds
    assert tuning.loocv_sharpe(panel, est, lam, 2, 12.0, max_folds=5, seed=3, Gamma_init=full.Gamma) == pytest.approx(value, rel=1e-12)
    assert tuning.loocv_sharpe(panel, est, lam, 2, 1.0, max_folds=5, seed=3) * np.sqrt(12.0) == pytest.approx(value, rel=1e-10)


def test_loocv_sharpe_all_periods_and_progress():
    p = make_panel(T=8, N=30)
    cfg = EstimationConfig(K=2, max_iter=200)
    lam = 0.3 * lambda_max(p, cfg, 2)
    calls: list[int] = []
    v = tuning.loocv_sharpe(p, cfg, lam, 2, 12.0, progress=lambda d, n, m: calls.append(n))
    assert np.isfinite(v) and calls == [8] * 8


def test_loocv_sharpe_undefined_with_one_populated_period():
    p = make_panel(T=2, N=30, empty_period=1)
    cfg = EstimationConfig(K=2)
    assert np.isnan(tuning.loocv_sharpe(p, cfg, 0.01, 2, 12.0))


def test_loocv_sharpe_lambda_zero_uses_plain_ipca(panel):
    cfg = EstimationConfig(K=2)
    v = tuning.loocv_sharpe(panel, cfg, 0.0, 2, 12.0, max_folds=3, seed=0)
    assert np.isfinite(v)


def test_tune_loocv_criterion_matches_direct_loocv(panel):
    est3 = EstimationConfig(K=2, lam_grid=LambdaGridConfig(n_lambdas=3, ratio=0.1), max_iter=300, seed=4)
    tcfg = TuningConfig(criterion="loocv_sharpe", loocv_max_folds=4)
    tr = tuning.tune(panel, est3, tcfg, EvaluationConfig())
    assert tr.criterion == "loocv_sharpe" and tr.meta["loocv_max_folds"] == 4 and tr.meta["loocv_seed"] == 4
    crit = np.array([p.criterion for p in tr.path])
    grid = lambda_grid(panel, est3, 2)
    _, fits = lambda_path(panel, est3, lams=grid, K=2)
    # a point whose Gamma is entirely zero (penalised intercept gone too, here at lam_max) has a
    # constant LOOCV series and an undefined criterion; it can never be chosen. Every other point is finite.
    nonzero = np.array([np.any(f.gamma_norms > 0) for f in fits])
    assert nonzero[:-1].all() and np.all(np.isfinite(crit[nonzero]))
    assert np.isfinite(crit).any() and tr.lam == tr.path[int(np.nanargmax(crit))].lam
    ref = [tuning.loocv_sharpe(panel, est3, f.lam, 2, 12.0, max_folds=4, seed=4, Gamma_init=f.Gamma) for f in fits]
    np.testing.assert_allclose(crit, ref, rtol=1e-10)  # NaN == NaN under assert_allclose
    # the in-sample Sharpe is still reported on the point, distinct from the LOOCV criterion
    for p, f in zip(tr.path, fits):
        assert p.mve_sharpe == pytest.approx(tuning.is_sharpe_criterion(f, 12.0), rel=1e-10)


# ---------------------------------------------------------------------------
# simulated data end to end
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sim_panel() -> IPCAPanel:
    from narrative_ipca.covariances import build_covariance_panel
    from narrative_ipca.data import align_inputs
    from narrative_ipca.panel import build_panel
    from narrative_ipca.shocks import attention_shocks
    from narrative_ipca.simulation import simulate

    sim = simulate(SimulationConfig(seed=0, n_assets=120, n_topics=24, n_relevant=6, n_placebo=6, n_years=6))
    data_cfg = DataConfig(min_assets_per_period=20)
    aligned = align_inputs(sim.attention, sim.returns, data_cfg)
    shocks = attention_shocks(aligned.attention, ShockConfig())
    cov_cfg = CovarianceConfig()
    cov = build_covariance_panel(shocks, aligned.returns, cov_cfg, data_cfg.period)
    return build_panel(cov, aligned.returns, data_cfg, cov_cfg)


def test_tune_on_simulated_panel(sim_panel):
    est_sim = EstimationConfig(K=3, lam_grid=LambdaGridConfig(n_lambdas=6, ratio=0.05))
    tr = tuning.tune(sim_panel, est_sim, TuningConfig(), EvaluationConfig())
    grid = lambda_grid(sim_panel, est_sim, 3)
    assert grid[0] <= tr.lam <= grid[-1] and np.any(np.isclose(tr.lam, grid))
    assert tr.K == 3 and tr.fit.Gamma.shape == (sim_panel.p, 3) and tr.fit.F.shape == (sim_panel.T, 3)
    assert_canonical(tr.fit)
    crit = np.array([p.criterion for p in tr.path])
    assert np.all(np.isfinite(crit)) and tr.path[tr.meta["chosen_index"]].criterion == crit.max()
    assert 0 < tr.fit.n_selected <= sim_panel.L
    assert np.isfinite(tr.fit.mve_sharpe()) and tr.fit.mve_sharpe() == pytest.approx(crit.max(), rel=1e-10)


def test_loocv_on_simulated_panel(sim_panel):
    est_sim = EstimationConfig(K=3)
    lam = 0.2 * lambda_max(sim_panel, est_sim, 3)
    v = tuning.loocv_sharpe(sim_panel, est_sim, lam, 3, 12.0, max_folds=6, seed=0)
    assert np.isfinite(v)
    assert tuning.loocv_folds(sim_panel, 6, seed=0).shape == (6,)


# ---------------------------------------------------------------------------
# near-singular Sigma_ff along the path: the pseudo-inverse cut in the criterion is flagged (Section 2 / D43)
# ---------------------------------------------------------------------------
def test_tune_records_sigma_ff_truncation_along_the_path(panel):
    est10 = EstimationConfig(K=2, lam=None, lam_grid=LambdaGridConfig(n_lambdas=10, ratio=0.02), max_iter=300)
    tr = tuning.tune(panel, est10, TuningConfig(), EvaluationConfig())
    flags = tr.meta["sigma_ff_truncated"]
    assert len(flags) == len(tr.path)
    _, fits = lambda_path(panel, est10, lams=[p.lam for p in tr.path], K=2)
    for flag, f in zip(flags, fits):
        ev = np.linalg.eigvalsh(0.5 * (f.Sigma_ff + f.Sigma_ff.T))
        assert flag == (ev.max() <= 0.0 or ev.min() <= EvaluationConfig().rcond * ev.max())
    assert flags[0] is False  # dense end: every factor alive
    assert flags[-1] is True  # lam_max: Gamma = 0, Sigma_ff = 0
    # the truncated points sit at the sparse end of the path (a factor direction has died)
    first_trunc = flags.index(True)
    assert all(flags[first_trunc:])


def test_tune_warns_when_the_chosen_fit_has_singular_sigma_ff(caplog):
    """K=3, one relevant narrative: <= 1 narrative selected, rank(Gamma) <= 2 < K, Sigma_ff singular by construction."""
    p = make_panel(true_rows=(2,), K=1)
    base = EstimationConfig(K=3, max_iter=300)
    est = replace(base, lam=0.6 * lambda_max(p, base, 3))
    with caplog.at_level(logging.WARNING, logger="narrative_ipca.tuning"):
        tr = tuning.tune(p, est, TuningConfig(), EvaluationConfig())
    assert tr.fit.n_selected + 1 < tr.K
    assert tr.meta["sigma_ff_truncated"] == [True]
    assert any("singular Sigma_ff" in rec.message for rec in caplog.records)
    # the criterion is still the documented closed form, truncation included (nothing is rescaled)
    assert tr.path[0].criterion == pytest.approx(tr.fit.mve_sharpe(12.0, 1e-12), rel=1e-12)
    # LOOCV on the same panel warns about the truncated folds
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="narrative_ipca.tuning"):
        v = tuning.loocv_sharpe(p, est, est.lam, 3, 12.0, max_folds=4, seed=0)
    assert np.isfinite(v) or np.isnan(v)
    assert any("fold fits have a singular Sigma_ff" in rec.message for rec in caplog.records)


def test_tolerance_prefers_the_sparsest_point_within_the_band(panel, est, monkeypatch):
    """TuningConfig.tolerance: sparsest point whose criterion >= (1 - tol) * max (D27)."""
    # criterion decreasing slowly in lambda: 1.00 at the densest point, minus 0.01 per doubling of lambda
    lam_min = float(np.min(lambda_grid(panel, est, 2)))

    def fake_crit(res, ann, rcond=1e-6):
        return 1.0 - 0.01 * float(np.log2(res.lam / lam_min))

    monkeypatch.setattr(tuning, "is_sharpe_criterion", fake_crit)
    exact = tuning.tune(panel, est, TuningConfig(tolerance=0.0), EvaluationConfig())
    loose = tuning.tune(panel, est, TuningConfig(tolerance=0.05), EvaluationConfig())
    crit_max = max(p.criterion for p in exact.path)
    assert exact.lam == min(p.lam for p in exact.path)  # exact argmax = densest point
    assert loose.lam >= exact.lam
    chosen = next(p for p in loose.path if p.lam == loose.lam)
    assert chosen.criterion >= (1 - 0.05) * crit_max
    # no sparser point satisfies the band
    assert all(p.criterion < (1 - 0.05) * crit_max for p in loose.path if p.lam > loose.lam)
    assert loose.meta["tie_rel_tol"] == 0.05
    with pytest.raises(ValueError):
        TuningConfig(tolerance=1.0)


@pytest.mark.parametrize("scale", [0.445, 2.5])
def test_tune_tolerance_band_is_relative_below_and_above_one(panel, est, monkeypatch, scale):
    """D51: the band is tolerance x |best| at any level of the criterion, so the pick does not depend on its scale."""
    lam_min = float(np.min(lambda_grid(panel, est, 2)))

    def fake_crit(res, ann, rcond=1e-6):  # best = scale at the densest point, 1% lower per doubling of lambda
        return scale * (1.0 - 0.01 * float(np.log2(res.lam / lam_min)))

    monkeypatch.setattr(tuning, "is_sharpe_criterion", fake_crit)
    tr = tuning.tune(panel, est, TuningConfig(tolerance=0.02), EvaluationConfig())
    best = max(p.criterion for p in tr.path)
    assert best == pytest.approx(scale, rel=1e-12)
    assert tr.meta["tie_band"] == pytest.approx(0.02 * best, rel=1e-12)
    floor = best - 0.02 * best
    sparser = [p for p in tr.path if p.lam > tr.lam]
    assert next(p for p in tr.path if p.lam == tr.lam).criterion >= floor
    assert sparser and all(p.criterion < floor for p in sparser)
    # 2% of the best is 2 doublings of lambda at either scale: the same grid point
    assert np.log2(tr.lam / lam_min) <= 2.0 < np.log2(min(p.lam for p in sparser) / lam_min)
    if scale < 1.0:
        # the absolute band of 0.02 used before 2026-10-01 would have admitted sparser points
        assert any(p.criterion >= best - 0.02 for p in sparser)
