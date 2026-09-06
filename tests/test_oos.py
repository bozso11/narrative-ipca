"""Tests of ``narrative_ipca.oos`` (BKS Section 4.2; DESIGN.md D30-D33).

What is checked, beyond shapes:

* ``oos_factor`` equals the Eq. 16 f-step on one period's moments, the
  explicit ridge normal equations, and (ridge 0) least squares; applied to
  the training periods of a fit it reproduces the fit's own factors exactly
  (the OOS formula is the in-sample f-step with a frozen ``Gamma``);
* ``oos_schedule``: first OOS index from a date or a fraction, the push to
  ``min_train_periods`` (flagged), refit indices, and the error cases;
* ``run_oos`` with a fixed lambda: factors only on OOS periods, every
  training panel holds exactly the periods before its refit period (no
  look-ahead), ``sigma_c`` and the penalty recomputed on the training rows
  (D26/D31), factors and MVE reproduce ``oos_factor`` / ``b_MVE`` of the
  serving fit (D32/D33), the Sharpe is the realised Sharpe of the MVE series
  (D34), the histories mirror the fits, empty periods are skipped;
* re-tuning: every refit's lambda equals a standalone ``tune`` on the same
  training panel and lies in that refit's grid; ``retune_lambda=False`` tunes
  once and freezes ``(lambda, K)``;
* the simulated pipeline (simulation -> data -> shocks -> covariances ->
  panel) at a fixed lambda: OOS-only factors, no look-ahead, finite Sharpe.
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
    OOSConfig,
    PipelineConfig,
    ShockConfig,
    SimulationConfig,
    TuningConfig,
)
from narrative_ipca.evaluation import realized_sharpe
from narrative_ipca.sparse_ipca import f_step, fit_sparse_ipca, lambda_max
from narrative_ipca.types import IPCAPanel, OOSResult, compute_sigma_c

TRUE_ROWS = (2, 5, 7)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def month_ends(start: str, n: int) -> pd.DatetimeIndex:
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


def assert_diagonal(S: np.ndarray) -> None:
    S = np.asarray(S)
    assert np.max(np.abs(S - np.diag(np.diag(S)))) <= 1e-10 * max(1.0, float(np.max(np.abs(S))))


def check_oos_structure(res: OOSResult, panel: IPCAPanel, ann: float = 12.0) -> None:
    """Invariants every ``run_oos`` result must satisfy (no look-ahead, D32/D33 identities, histories)."""
    periods = pd.DatetimeIndex(panel.periods)
    first = res.meta["first_oos_index"]
    refits = res.meta["refit_indices"]
    assert refits[0] == first and res.refit_periods == [periods[c] for c in refits]
    assert res.meta["train_end"] == [periods[c - 1] for c in refits]
    assert len(res.fits) == len(refits) == res.meta["n_refits"]
    slices = dict(panel.period_slices())
    # factors only on OOS periods with observations, in order
    expected = [periods[t] for t in range(first, panel.T) if slices[t].stop > slices[t].start]
    assert list(res.factors.index) == expected and list(res.mve.index) == expected
    assert res.meta["n_skipped_empty"] == (panel.T - first) - len(expected)
    served = res.meta["refit_of_period"]
    assert list(served.index) == expected
    for j, (c, fit) in enumerate(zip(refits, res.fits)):
        # no look-ahead: the training panel is exactly the periods before the refit period
        assert pd.DatetimeIndex(fit.periods).equals(periods[:c])
        assert fit.periods.max() < res.refit_periods[j]
        train = panel.subset_periods(np.arange(panel.T) < c)
        assert fit.n_obs == train.n_obs and res.meta["n_train_periods"][j] == c
        # sigma_c of the training rows enters the penalty (D26/D31), never the full-sample one
        pen = fit.meta.get("penalties")
        if pen is not None:
            np.testing.assert_allclose(pen, fit.lam * train.n_obs * compute_sigma_c(train.X), rtol=1e-12)
        assert_diagonal(fit.Sigma_ff)
        # the block served by this refit
        block = served.index[served.to_numpy() == j]
        nxt = res.refit_periods[j + 1] if j + 1 < len(refits) else None
        for ts in block:
            assert ts >= res.refit_periods[j] and (nxt is None or ts < nxt)
            t = int(periods.get_loc(ts))
            sl = slices[t]
            f_ref = oos.oos_factor(panel.X[sl], panel.y[sl], fit.Gamma)
            np.testing.assert_allclose(res.factors.loc[ts].to_numpy()[: fit.K], f_ref, rtol=1e-12, atol=1e-14)
            assert res.mve.loc[ts] == pytest.approx(float(fit.b_mve() @ f_ref), rel=1e-12, abs=1e-14)
        # histories mirror the fit
        rp = res.refit_periods[j]
        assert res.lam_history.loc[rp] == fit.lam and res.K_history.loc[rp] == fit.K
        assert res.n_selected_history.loc[rp] == fit.n_selected
        assert res.is_sharpe_history.loc[rp] == pytest.approx(fit.mve_sharpe(ann), rel=1e-12)
        assert res.meta["sigma_ff_truncated_history"][j] == oos.sigma_ff_truncated(fit.Sigma_ff, res.meta["rcond"])
        np.testing.assert_allclose(res.gamma_norm_history.loc[rp].to_numpy(), fit.gamma_norms)
        np.testing.assert_array_equal(res.selected_history.loc[rp].to_numpy().astype(bool), fit.selected)
    assert list(res.gamma_norm_history.columns) == list(panel.instrument_names)
    assert list(res.selected_history.columns) == list(panel.topics)
    # D34: the Sharpe is the realised Sharpe of the MVE series
    assert res.sharpe == pytest.approx(realized_sharpe(res.mve, ann), rel=1e-12, nan_ok=True)


@pytest.fixture(scope="module")
def panel() -> IPCAPanel:
    return make_panel()


@pytest.fixture(scope="module")
def est_fixed(panel) -> EstimationConfig:
    base = EstimationConfig(K=2, max_iter=300)
    return replace(base, lam=0.3 * lambda_max(panel, base, 2))


# ---------------------------------------------------------------------------
# oos_factor (D32)
# ---------------------------------------------------------------------------
def test_oos_factor_equals_f_step_on_one_period():
    rng = np.random.default_rng(0)
    N, p, K = 50, 6, 3
    C = np.column_stack([np.ones(N), rng.standard_normal((N, p - 1))])
    r = rng.standard_normal(N)
    Gamma = rng.standard_normal((p, K))
    S = (C.T @ C)[None]
    V = (C.T @ r)[None]
    f = oos.oos_factor(C, r, Gamma)
    np.testing.assert_allclose(f, f_step(S, V, Gamma)[0], rtol=1e-12, atol=1e-14)
    B = C @ Gamma
    np.testing.assert_allclose(f, np.linalg.solve(B.T @ B + 2.0 * np.eye(K), B.T @ r), rtol=1e-12)
    # other ridge values follow the f-step too; ridge 0 is least squares
    np.testing.assert_allclose(oos.oos_factor(C, r, Gamma, ridge=0.5), f_step(S, V, Gamma, ridge=0.5)[0], rtol=1e-12)
    np.testing.assert_allclose(oos.oos_factor(C, r, Gamma, ridge=0.0), np.linalg.lstsq(B, r, rcond=None)[0], rtol=1e-10)
    assert oos.RIDGE == 2.0
    # one-dimensional inputs and an empty cross-section
    np.testing.assert_allclose(oos.oos_factor(C[:1], r[:1], Gamma), f_step(S * 0 + (C[:1].T @ C[:1])[None], (C[:1].T @ r[:1])[None], Gamma)[0])
    np.testing.assert_array_equal(oos.oos_factor(np.zeros((0, p)), np.zeros(0), Gamma), np.zeros(K))


def test_oos_factor_reproduces_in_sample_factors(panel, est_fixed):
    """The OOS formula with the fitted Gamma is the f-step of Eq. 16: on the training periods it returns fit.F."""
    fit = fit_sparse_ipca(panel, est_fixed, K=2)
    for t, sl in panel.period_slices():
        f = oos.oos_factor(panel.X[sl], panel.y[sl], fit.Gamma)
        np.testing.assert_allclose(f, fit.F[t], rtol=1e-10, atol=1e-13)


def test_oos_factor_validation():
    C = np.ones((5, 3))
    r = np.ones(5)
    with pytest.raises(ValueError):
        oos.oos_factor(C, r, np.ones((4, 2)))
    with pytest.raises(ValueError):
        oos.oos_factor(C, np.ones(4), np.ones((3, 2)))
    with pytest.raises(ValueError):
        oos.oos_factor(C, r, np.ones(3))
    with pytest.raises(ValueError):
        oos.oos_factor(C, r, np.ones((3, 2)), ridge=-1.0)


# ---------------------------------------------------------------------------
# schedule (D30)
# ---------------------------------------------------------------------------
def test_oos_schedule_from_date_fraction_and_min_train(caplog):
    periods = month_ends("2000-01", 40)
    first, refits, moved = oos.oos_schedule(periods, OOSConfig(first_oos_period="2002-01-15", refit_every=12, min_train_periods=2))
    assert periods[first] == pd.Timestamp("2002-01-31") and first == 24 and not moved
    assert refits == [24, 36]
    first, refits, moved = oos.oos_schedule(periods, OOSConfig(first_oos_period="2001-12-31", refit_every=5, min_train_periods=2))
    assert first == 23 and refits == [23, 28, 33, 38]
    first, refits, moved = oos.oos_schedule(periods, OOSConfig(oos_fraction=0.25, refit_every=4, min_train_periods=2))
    assert first == 30 and refits == [30, 34, 38] and not moved
    with caplog.at_level(logging.WARNING, logger="narrative_ipca.oos"):
        first, refits, moved = oos.oos_schedule(periods, OOSConfig(first_oos_period="2000-03-31", refit_every=12, min_train_periods=10))
    assert first == 10 and moved and refits == [10, 22, 34]
    assert any("min_train_periods" in rec.message for rec in caplog.records)
    with pytest.raises(ValueError):
        oos.oos_schedule(periods, OOSConfig(first_oos_period="2010-01-31", min_train_periods=2))
    with pytest.raises(ValueError):
        oos.oos_schedule(periods, OOSConfig(oos_fraction=0.1, min_train_periods=40))
    with pytest.raises(ValueError):
        oos.oos_schedule(periods[:0], OOSConfig())


def test_pad_rows_handles_unequal_K():
    out = oos._pad_rows([np.array([1.0]), np.array([2.0, 3.0]), np.array([4.0, 5.0, 6.0])], 3)
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out[2], [4.0, 5.0, 6.0])
    assert out[0, 0] == 1.0 and np.isnan(out[0, 1:]).all() and np.isnan(out[1, 2])
    assert oos._pad_rows([], 2).shape == (0, 2)


# ---------------------------------------------------------------------------
# run_oos (D30-D33)
# ---------------------------------------------------------------------------
def test_run_oos_fixed_lambda_no_retune(panel, est_fixed):
    cfg = PipelineConfig(estimation=est_fixed, oos=OOSConfig(oos_fraction=0.4, refit_every=5, min_train_periods=10, retune_lambda=False))
    res = oos.run_oos(panel, cfg)
    assert res.meta["first_oos_index"] == 18 and res.meta["refit_indices"] == [18, 23, 28]
    assert len(res.mve) == 12 and res.factors.shape == (12, 2)
    assert np.isfinite(res.sharpe)
    check_oos_structure(res, panel)
    assert res.meta["tuned_history"] == [False, False, False]
    assert np.all(np.isnan(res.meta["lam_max_history"]))
    assert np.all(res.lam_history.to_numpy() == est_fixed.lam) and np.all(res.K_history.to_numpy() == 2)
    assert all(f.lam == est_fixed.lam and f.meta.get("canonical") for f in res.fits)
    assert not res.meta["first_oos_moved"] and res.meta["retune_lambda"] is False
    # D33: b_MVE from the training moments; the OOS MVE is b' f block by block
    for j, c in enumerate(res.meta["refit_indices"]):
        b = res.fits[j].b_mve()
        rows = res.meta["refit_of_period"].to_numpy() == j
        np.testing.assert_allclose(res.mve.to_numpy()[rows], res.factors.to_numpy()[rows] @ b, rtol=1e-12, atol=1e-14)


def test_run_oos_fixed_lambda_with_retune_goes_through_tune(panel, est_fixed):
    """retune_lambda=True with a fixed lambda: every refit runs the one-point tuner (lam_max reported)."""
    cfg = PipelineConfig(estimation=est_fixed, oos=OOSConfig(oos_fraction=0.3, refit_every=6, min_train_periods=10))
    res = oos.run_oos(panel, cfg)
    check_oos_structure(res, panel)
    assert res.meta["tuned_history"] == [True, True]
    assert np.all(np.isfinite(res.meta["lam_max_history"]))
    assert np.all(res.lam_history.to_numpy() == est_fixed.lam)
    for j, c in enumerate(res.meta["refit_indices"]):
        train = panel.subset_periods(np.arange(panel.T) < c)
        assert res.meta["lam_max_history"][j] == pytest.approx(lambda_max(train, est_fixed, 2), rel=1e-12)


def test_run_oos_retunes_lambda_every_refit(panel):
    est = EstimationConfig(K=2, lam=None, lam_grid=LambdaGridConfig(n_lambdas=4, ratio=0.1), max_iter=300)
    cfg = PipelineConfig(estimation=est, oos=OOSConfig(oos_fraction=0.4, refit_every=6, min_train_periods=10))
    res = oos.run_oos(panel, cfg)
    check_oos_structure(res, panel)
    assert res.meta["refit_indices"] == [18, 24] and res.meta["tuned_history"] == [True, True]
    for j, c in enumerate(res.meta["refit_indices"]):
        train = panel.subset_periods(np.arange(panel.T) < c)
        tr = tuning.tune(train, est, cfg.tuning, cfg.evaluation)
        assert res.lam_history.iloc[j] == tr.lam and res.fits[j].lam == tr.lam
        lmax = res.meta["lam_max_history"][j]
        assert lmax == pytest.approx(tr.lam_max, rel=1e-12)
        assert est.lam_grid.ratio * lmax * (1 - 1e-9) <= tr.lam <= lmax * (1 + 1e-9)
        assert res.fits[j].objective == pytest.approx(tr.fit.objective, rel=1e-10)
        assert res.is_sharpe_history.iloc[j] == pytest.approx(tr.fit.mve_sharpe(), rel=1e-10)


def test_run_oos_tunes_once_when_not_retuning(panel):
    est = EstimationConfig(K=2, lam=None, lam_grid=LambdaGridConfig(n_lambdas=4, ratio=0.1), max_iter=300)
    cfg = PipelineConfig(estimation=est, oos=OOSConfig(oos_fraction=0.4, refit_every=4, min_train_periods=10, retune_lambda=False))
    res = oos.run_oos(panel, cfg)
    check_oos_structure(res, panel)
    assert res.meta["tuned_history"] == [True, False, False]
    assert res.lam_history.nunique() == 1 and res.K_history.nunique() == 1
    lam0 = res.lam_history.iloc[0]
    train0 = panel.subset_periods(np.arange(panel.T) < 18)
    assert lam0 == tuning.tune(train0, est, cfg.tuning, cfg.evaluation).lam
    # the frozen fits are fresh fits at lam0 on their own training windows
    for j, c in enumerate(res.meta["refit_indices"][1:], start=1):
        train = panel.subset_periods(np.arange(panel.T) < c)
        ref = fit_sparse_ipca(train, est, lam=lam0, K=2)
        assert res.fits[j].objective == pytest.approx(ref.objective, rel=1e-10)
        assert res.fits[j].n_selected == ref.n_selected


def test_run_oos_skips_empty_period_and_uses_date():
    p = make_panel(T=30, empty_period=25)
    base = EstimationConfig(K=2, max_iter=300)
    est = replace(base, lam=0.3 * lambda_max(p, base, 2))
    cfg = PipelineConfig(estimation=est, oos=OOSConfig(first_oos_period="2001-07-31", refit_every=12, min_train_periods=5, retune_lambda=False))
    res = oos.run_oos(p, cfg)
    assert res.meta["first_oos_index"] == 18 and res.meta["first_oos_period"] == pd.Timestamp("2001-07-31")
    assert res.meta["n_skipped_empty"] == 1 and pd.Timestamp("2002-02-28") not in res.factors.index
    assert len(res.mve) == 11
    check_oos_structure(res, p)


def test_run_oos_min_train_moves_first_period(panel, est_fixed, caplog):
    cfg = PipelineConfig(estimation=est_fixed, oos=OOSConfig(first_oos_period="2000-06-30", refit_every=12, min_train_periods=20, retune_lambda=False))
    with caplog.at_level(logging.WARNING, logger="narrative_ipca.oos"):
        res = oos.run_oos(panel, cfg)
    assert res.meta["first_oos_moved"] and res.meta["first_oos_index"] == 20
    assert len(res.mve) == 10
    check_oos_structure(res, panel)


def test_run_oos_progress_and_errors(panel, est_fixed):
    calls: list[tuple[int, int, str]] = []
    cfg = PipelineConfig(estimation=est_fixed, oos=OOSConfig(oos_fraction=0.3, refit_every=3, min_train_periods=10, retune_lambda=False))
    res = oos.run_oos(panel, cfg, progress=lambda j, n, m: calls.append((j, n, m)))
    n = res.meta["n_refits"]
    assert [c[0] for c in calls] == list(range(n + 1)) and all(c[1] == n for c in calls)
    with pytest.raises(ValueError):
        oos.run_oos(panel, PipelineConfig(estimation=est_fixed, oos=OOSConfig(first_oos_period="2030-01-31", min_train_periods=2)))
    empty = IPCAPanel(
        X=np.zeros((0, 3)), y=np.zeros(0), t_idx=np.zeros(0, dtype=int), asset_idx=np.zeros(0, dtype=int),
        periods=month_ends("2000-01", 2), assets=np.array(["a"]), instrument_names=["const", "t1", "t2"], sigma_c=np.ones(3),
    )
    with pytest.raises(ValueError):
        oos.run_oos(empty, PipelineConfig(estimation=est_fixed))


def test_run_oos_unbalanced_panel(est_fixed):
    p = make_panel(drop_frac=0.3, seed=5)
    cfg = PipelineConfig(estimation=est_fixed, oos=OOSConfig(oos_fraction=0.4, refit_every=5, min_train_periods=10, retune_lambda=False))
    res = oos.run_oos(p, cfg)
    assert np.isfinite(res.sharpe)
    check_oos_structure(res, p)


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


def test_run_oos_on_simulated_panel(sim_panel):
    base = EstimationConfig(K=3)
    est = replace(base, lam=0.2 * lambda_max(sim_panel, base, 3))
    cfg = PipelineConfig(
        estimation=est,
        tuning=TuningConfig(),
        oos=OOSConfig(oos_fraction=0.4, refit_every=6, min_train_periods=24),
        evaluation=EvaluationConfig(),
    )
    res = oos.run_oos(sim_panel, cfg)
    periods = pd.DatetimeIndex(sim_panel.periods)
    first = res.meta["first_oos_index"]
    assert first == sim_panel.T - int(round(0.4 * sim_panel.T)) and first >= 24
    # factors only for the OOS periods, none before the first OOS period
    assert res.factors.index.equals(periods[first:]) and res.factors.shape == (sim_panel.T - first, 3)
    assert res.factors.index.min() == periods[first] and np.all(np.isfinite(res.factors.to_numpy()))
    # no look-ahead: every serving fit was trained on periods strictly before its refit period
    for j, fit in enumerate(res.fits):
        assert fit.periods.max() < res.refit_periods[j]
        assert pd.DatetimeIndex(fit.periods).equals(periods[: res.meta["refit_indices"][j]])
    assert np.isfinite(res.sharpe)
    check_oos_structure(res, sim_panel)
    assert list(res.lam_history.round(15).unique()) == [round(est.lam, 15)]


# ---------------------------------------------------------------------------
# near-singular Sigma_ff: the pseudo-inverse truncation of b_MVE is flagged (Section 4.2 / D43)
# ---------------------------------------------------------------------------
def test_sigma_ff_truncated_flag_semantics():
    assert not oos.sigma_ff_truncated(np.eye(2), 1e-12)
    assert not oos.sigma_ff_truncated(np.diag([1.0, 1e-11]), 1e-12)  # above the cut-off: kept
    assert oos.sigma_ff_truncated(np.diag([1.0, 1e-13]), 1e-12)  # below the cut-off: dropped
    assert oos.sigma_ff_truncated(np.diag([1.0, 1e-13]), 1e-12) and not oos.sigma_ff_truncated(np.diag([1.0, 1e-13]), 1e-14)
    assert oos.sigma_ff_truncated(np.zeros((3, 3)), 1e-12)  # Gamma = 0 (lam_max point): everything dropped
    assert oos.sigma_ff_truncated(np.array([[1.0, 1.0], [1.0, 1.0]]), 1e-12)  # exactly singular
    assert not oos.sigma_ff_truncated(np.zeros((0, 0)), 1e-12)


def test_run_oos_flags_singular_sigma_ff_and_warns(caplog):
    """K=3 with a single relevant narrative: at most one narrative survives, Gamma has rank <= 2 < K,
    Sigma_ff is singular by construction, and b_MVE = pinv(Sigma_ff) mu_f drops a direction (D43)."""
    p = make_panel(true_rows=(2,), K=1)
    base = EstimationConfig(K=3, max_iter=300)
    est = replace(base, lam=0.6 * lambda_max(p, base, 3))
    cfg = PipelineConfig(estimation=est, oos=OOSConfig(oos_fraction=0.3, refit_every=6, min_train_periods=10, retune_lambda=False))
    with caplog.at_level(logging.WARNING, logger="narrative_ipca.oos"):
        res = oos.run_oos(p, cfg)
    check_oos_structure(res, p)
    assert all(f.n_selected + 1 < f.K for f in res.fits)
    assert res.meta["sigma_ff_truncated_history"] == [True] * res.meta["n_refits"]
    assert sum("singular at rcond" in rec.message for rec in caplog.records) == res.meta["n_refits"]


def test_run_oos_well_conditioned_fit_is_not_flagged(panel, caplog):
    """Dense end of the path (0.05 lam_max): both factors alive, Sigma_ff well conditioned, no flag, no warning.
    (est_fixed at 0.3 lam_max already sits in the near-dead-factor regime, cond(Sigma_ff) ~ 1e7.)"""
    base = EstimationConfig(K=2, max_iter=300)
    est_dense = replace(base, lam=0.05 * lambda_max(panel, base, 2))
    cfg = PipelineConfig(estimation=est_dense, oos=OOSConfig(oos_fraction=0.4, refit_every=5, min_train_periods=10, retune_lambda=False))
    with caplog.at_level(logging.WARNING, logger="narrative_ipca.oos"):
        res = oos.run_oos(panel, cfg)
    assert all(np.linalg.cond(f.Sigma_ff) < 1e3 for f in res.fits)
    assert res.meta["sigma_ff_truncated_history"] == [False] * res.meta["n_refits"]
    assert not any("singular at rcond" in rec.message for rec in caplog.records)
