"""Tests of the BKS trace (DESIGN.md G.16; D90): narrative_ipca/exposure_lab/trace.py and LabSession.bks_trace.

A small generic simulation (30 artificial assets, 12 generic topics of which
6 carry links, a two-year training window) is fitted under both covariance
histories, both leads, the tuned lambda and the fixed lambda 0; every
identity check of the trace must hold on each. The per-selection helpers
are checked against the panel they explain. The dashboard defaults (55
listed assets, 20 topics) reproduce the diagnosis numbers of DESIGN.md
G.15.1; that test is skipped when ``data/market`` is missing.

The review of 2026-09-30 adds edge configurations to the identity sweep
(small lambdas, every topic dropped, plain IPCA with four or five assets, no
topic signal) and regression tests for its fixes: the stationarity checks
graded on a polished copy of the fit (and still off for a ``Gamma`` the fit
did not reach), tolerances that follow the conditioning, the independent
step 8 recomputes (off for a wrong projection or pseudo-inverse), the
findings' severity (``"defect"``) and origin, the topic table, and
:class:`~narrative_ipca.exposure_lab.session.BKSNotCached`.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from narrative_ipca import tuning
from narrative_ipca.covariances import brute_force_covariance, kernel_weights, window_bounds
from narrative_ipca.data import period_end_index
from narrative_ipca.exposure_lab import bks, reference
from narrative_ipca.exposure_lab import trace as T
from narrative_ipca.wrapup import _lsq_map
from narrative_ipca.exposure_lab.config import (
    BKSLabConfig,
    ExposureConfig,
    LabConfig,
    TopicSetConfig,
    UniverseConfig,
    WindowConfig,
)
from narrative_ipca.exposure_lab.dgp import observed_shocks, simulate_lab, truth_for_window
from narrative_ipca.exposure_lab.direct import shock_matrix, training_pairs
from narrative_ipca.exposure_lab.evaluate import evaluate_window, median_finite
from narrative_ipca.exposure_lab.session import BKS_NOT_RUN, BKS_STAGES, SESSION_STAGES, BKSNotCached, LabSession

ROOT = Path(__file__).resolve().parents[1]
HAS_MARKET = (reference.market_dir() / "asset_returns.parquet").is_file()
needs_market = pytest.mark.skipif(not HAS_MARKET, reason="data/market/asset_returns.parquet is missing")

#: Two training years keep the fits fast (about 100 training weeks).
WINDOW = {"train_start": "2021-01-01", "train_end": "2022-12-30", "forecast_start": "2023-01-02", "forecast_weeks": 4}


def _generic_cfg(lead: int = 0, *, n_assets: int = 30, signal_share: float = 0.5, window: dict | None = None,
                 **bks_kw: object) -> LabConfig:
    """30 generic assets, 12 generic topics (6 linked), beta_1 = 0.5 (seed 0), a two-year training window."""
    return LabConfig(
        universe=UniverseConfig(asset_source="generic", n_generic_assets=n_assets, seed=0),
        topics=TopicSetConfig(manual="none", n_generic=12, generic_signal_share=signal_share),
        exposure=ExposureConfig(beta_1=0.5, lead_days=lead, seed=0),
        window=WindowConfig(**(WINDOW if window is None else window)),
        bks=BKSLabConfig(**bks_kw),
    )


@pytest.fixture(scope="module")
def session() -> LabSession:
    return LabSession(max_entries=20, bks_max_entries=20)


def _traced(session: LabSession, cfg: LabConfig) -> T.BKSTrace:
    session.bks_fit(cfg)
    return session.bks_trace(cfg)


@pytest.fixture(scope="module")
def cfg() -> LabConfig:
    return _generic_cfg()


@pytest.fixture(scope="module")
def tr(session, cfg) -> T.BKSTrace:
    return _traced(session, cfg)


def _identity_off(trace: T.BKSTrace) -> list[tuple[str, str, float, str]]:
    return [(c.step, c.name, c.value, c.note) for c in trace.checks if c.kind == T.IDENTITY and c.status == T.OFF]


#: The step 8 identity checks that rebuild Eq. 5 with numpy and pandas (not the production helpers).
INDEPENDENT_STEP8 = (
    "Projection = numpy pseudo-inverse",
    "Projection is orthogonal, of the fit's rank",
    "Sigma_z = numpy covariance",
    "Sigma_z pseudo-inverse is Moore-Penrose",
    "Return scale and divisor = pandas",
    "Implied sensitivities rebuilt with numpy = production",
)


def _page_texts(trace: T.BKSTrace) -> list[str]:
    """Every text of the trace the page prints: key numbers, findings, check names, relations and notes."""
    texts = list(trace.meta["key_numbers"].values())
    texts += [f["title"] for f in trace.findings] + [f["text"] for f in trace.findings]
    texts += [c.name for c in trace.checks] + [c.relation for c in trace.checks] + [c.note for c in trace.checks]
    return texts


def _assert_no_nan_text(trace: T.BKSTrace) -> None:
    bad = [t for t in _page_texts(trace) if re.search(r"\bnan\b", str(t))]  # "NaN rule" (D17) is a word, not a value
    assert bad == []


# ---------------------------------------------------------------------------
# every identity check holds
# ---------------------------------------------------------------------------
CASES = {
    "full-lead0-tuned": dict(lead=0),
    "training-lead0-tuned": dict(lead=0, history="training"),
    "full-lead1-tuned": dict(lead=1),
    "training-lead1-tuned": dict(lead=1, history="training"),
    "full-lam0": dict(lead=0, lambda_rule="fixed", lam=0.0),
    "training-lam0": dict(lead=0, history="training", lambda_rule="fixed", lam=0.0),
    "full-lead1-lam0": dict(lead=1, lambda_rule="fixed", lam=0.0),
    "training-fixed-positive": dict(lead=0, history="training", lambda_rule="fixed", lam=0.3),
    "full-K1-no-weighting": dict(lead=0, K=1, asset_weighting="none"),
    "training-argmax-K2-no-weighting": dict(lead=0, K=2, lambda_rule="argmax", asset_weighting="none",
                                            history="training"),
    # small lambdas: the stored fit's KKT residual grows like 1 / lambda (graded on the polished copy)
    "full-fixed-0.002": dict(lead=0, lambda_rule="fixed", lam=0.002),
    "training-argmax-ratio-0.001": dict(lead=0, lambda_rule="argmax", lambda_ratio=0.001, history="training"),
    # above lambda_max: every row of Gamma is dropped (Gamma = 0, F = 0)
    "full-fixed-above-lambda-max": dict(lead=0, lambda_rule="fixed", lam=50.0),
    # plain IPCA with a handful of assets: ill-conditioned weekly factor systems
    "full-lam0-4-assets": dict(lead=0, n_assets=4, lambda_rule="fixed", lam=0.0),
    "full-lam0-5-assets": dict(lead=0, n_assets=5, lambda_rule="fixed", lam=0.0),
    # no topic signal: no generic topic has a link
    "full-no-signal": dict(lead=0, signal_share=0.0),
}


@pytest.mark.parametrize("case", list(CASES))
def test_identity_checks_hold(session, case):
    c = _generic_cfg(**CASES[case])
    trace = _traced(session, c)
    assert _identity_off(trace) == []
    implied = session.bks_implied(c)
    # the step-by-step chain is the production result, to rounding
    np.testing.assert_allclose(trace.chain["B_hat"].to_numpy(), implied.B_hat.to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(trace.chain["B_const"].to_numpy(), implied.meta["B_const"].to_numpy(), rtol=0,
                               atol=1e-12)
    assert trace.variants["bks_implied"].equals(implied.B_hat.reindex(index=trace.topics, columns=trace.assets))
    # every check is fully described
    for ch in trace.checks:
        assert ch.step in T.STEPS and ch.status in (T.OK, T.OFF, T.INFO) and ch.kind in (T.IDENTITY, T.DIAGNOSTIC)
        assert ch.name and ch.relation and len(ch.note) > 20, ch
        assert ch.status == T.INFO or np.isfinite(ch.tolerance), ch
    assert trace.history == c.bks.history and trace.lead_days == c.exposure.lead_days and trace.K == c.bks.K
    # the independent step 8 recomputes exist and hold
    by_name = {ch.name: ch for ch in trace.checks}
    for name in INDEPENDENT_STEP8:
        assert by_name[name].kind == T.IDENTITY and by_name[name].status == T.OK, by_name[name]
    # page texts never print "nan" (key numbers, findings, notes); no finding comes from a defect
    _assert_no_nan_text(trace)
    assert all(f["severity"] != "defect" for f in trace.findings)


def test_lambda_zero_branches(session):
    """At lambda 0 the objective is the SSR, the ridge is 0 and KKT / scale balance do not apply (info)."""
    trace = _traced(session, _generic_cfg(lambda_rule="fixed", lam=0.0))
    by_name = {c.name: c for c in trace.checks}
    assert by_name["Objective recompute"].status == T.OK and "SSR" in by_name["Objective recompute"].relation
    for name in ("Stationarity of the group lasso (KKT)", "Penalty and ridge in balance",
                 "Chosen lambda follows the band rule", "lambda_max recompute"):
        assert by_name[name].status == T.INFO and by_name[name].note, name
    assert by_name["Ridge follows lambda"].status == T.OK and trace.meta["oos_ridge"] == 0.0
    assert trace.path is None and trace.gamma_path is None and trace.path_trace is None
    assert trace.lambda_rule == "fixed" and trace.lam == 0.0
    assert trace.kkt["ratio"].isna().all()


def test_ladder_rows_order_and_scores(session, cfg, tr):
    assert list(tr.ladder.index) == list(T.LADDER)
    assert list(tr.ladder.columns) == ["label", "spearman", "rmse", "median_r2", "d_spearman", "d_median_r2", "what"]
    assert tr.ladder["label"].tolist() == list(T.LADDER.values())
    assert tr.ladder.loc["oracle", "spearman"] == pytest.approx(1.0) and tr.ladder.loc["oracle", "rmse"] == 0.0
    # BKS-implied is scored exactly like the Compare tab
    implied = session.bks_implied(cfg)
    ev_imp = evaluate_window(session.simulation(cfg), session.shocks(cfg), implied, cfg.window, session.truth(cfg))
    assert tr.ladder.loc["bks_implied", "spearman"] == pytest.approx(ev_imp.recovery["spearman"], abs=1e-12)
    assert tr.ladder.loc["bks_implied", "median_r2"] == pytest.approx(median_finite(ev_imp.r2), abs=1e-12)
    chain_keys = [k for k in T.LADDER if k not in T.LADDER_BENCHMARKS]
    assert T.LADDER_BENCHMARKS == ("zero",) and list(T.LADDER)[-1] == "zero"
    assert list(T.LADDER).index("bks_ls") == list(T.LADDER).index("best_rank") + 1
    assert list(T.LADDER).index("bks_ls") + 1 == list(T.LADDER).index("bks_no_const")
    for col in ("spearman", "median_r2"):
        diffs = tr.ladder.loc[chain_keys, col].diff()
        np.testing.assert_allclose(tr.ladder.loc[chain_keys, f"d_{col}"].to_numpy()[1:], diffs.to_numpy()[1:])
        assert np.isnan(tr.ladder[f"d_{col}"].iloc[0]) and np.isnan(tr.ladder.loc["zero", f"d_{col}"])
    # the zero benchmark: no ranking, the RMSE of saying nothing, an OOS R2 of exactly 0
    Bt = tr.variants["oracle"].to_numpy()
    assert (tr.variants["zero"].to_numpy() == 0.0).all()
    assert np.isnan(tr.ladder.loc["zero", "spearman"])
    assert tr.ladder.loc["zero", "rmse"] == pytest.approx(float(np.sqrt(np.mean(Bt**2))), rel=1e-12)
    assert tr.ladder.loc["zero", "median_r2"] == 0.0
    # the full-history constant is penalised away here, so "no constant" equals production
    if tr.chain["m_const"].abs().max() == 0.0:
        assert tr.variants["bks_no_const"].equals(tr.variants["bks_implied"])
    for key, frame in tr.variants.items():
        assert frame.shape == (len(tr.topics), len(tr.assets)), key
        assert list(frame.index) == tr.topics and list(frame.columns) == tr.assets


def test_capture_shares(tr):
    cap = tr.capture
    assert 0.0 <= cap["kept_share"] <= cap["best_share"] <= 1.0
    assert cap["random_share"] == pytest.approx(tr.K / len(tr.topics))
    assert cap["singular_share"].sum() == pytest.approx(1.0)
    assert cap["captured"].sum() == pytest.approx(tr.chain["gamma_rank"], abs=1e-9)  # trace of the projector
    assert len(cap["principal_cosines"]) == tr.K
    assert np.all((cap["principal_cosines"] >= 0.0) & (cap["principal_cosines"] <= 1.0))
    # the least-squares reconstruction from the same K betas keeps at least what Eq. 5 keeps, at most the best K
    assert cap["kept_share"] - 1e-12 <= cap["ls_share"] <= cap["best_share"] + 1e-12
    # per asset: ||P c_i||^2 / ||c_i||^2, whose norm-weighted mean is the kept share
    pa = cap["per_asset_share"]
    assert list(pa.index) == tr.assets and pa.between(0.0, 1.0 + 1e-12).all()
    n2 = (tr.instruments**2).sum(axis=1)
    assert float((pa * n2).sum() / n2.sum()) == pytest.approx(cap["kept_share"], rel=1e-12)
    # the projection identity: m = P c + M Gamma_0'
    has = tr.instruments.notna().all(axis=1)
    m_alt = tr.instruments.loc[has] @ tr.chain["projector"].T.to_numpy() + tr.chain["m_const"].to_numpy()
    np.testing.assert_allclose(m_alt.to_numpy(), tr.chain["m"].loc[has].to_numpy(), rtol=0, atol=1e-15)


def test_tables_and_frames(tr):
    cf = tr.checks_frame()
    assert list(cf.columns) == ["step", "check", "relation", "observed", "reference", "tolerance", "status", "kind",
                                "note"]
    assert len(cf) == len(tr.checks) and set(cf["step"]) <= set(T.STEPS.values())
    assert len(tr.checks_frame("fit")) == sum(c.step == "fit" for c in tr.checks)
    with pytest.raises(KeyError):
        tr.checks_frame("nope")
    st = tr.status_frame()
    assert list(st["step"]) == [k for k in T.STEPS if k != "summary"]
    assert set(st["reading"]) <= {"as expected", "departs", "off"}
    assert (st["n_ok"] <= st["n_checks"]).all() and st["key_number"].str.len().gt(0).all()
    assert "off" not in set(st["reading"])
    for f in tr.findings:
        assert set(f) == {"step", "severity", "origin", "title", "text"} and f["step"] in T.STEPS
        assert f["severity"] in T.SEVERITIES and f["origin"] in T.ORIGINS and f["text"]
        assert (f["severity"] == "defect") == (f["origin"] == "defect")
    assert list(T.SEVERITIES) == ["defect", "departure", "note"]
    assert set(T.ORIGINS) == {"method", "implementation", "data", "defect"}
    assert any(f["title"] == "Which instruments the implied sensitivities use" for f in tr.findings)
    # shapes of the step tables
    L, N = len(tr.topics), len(tr.assets)
    assert list(tr.units.columns) == ["divisor", "ret_scale", "asset_vol", "divisor_over_ret_scale", "conversion",
                                      "skipped", "conversion_exact", "kernel_divisor", "conversion_kernel"]
    assert tr.units.shape == (N, 9) and tr.shock_table.shape == (L, 6) and tr.stability.shape == (L, 2)
    assert list(tr.weeks.index) == list(tr.forecast_periods)
    assert list(tr.weeks.columns) == (["first_day", "n_assets", "r2", "r2_shuffled"]
                                      + [f"f{k + 1}" for k in range(tr.K)] + ["r2_exact"])
    assert list(tr.sigma_table.columns) == ["train_over_population", "kernel_over_population", "train_over_kernel",
                                            "max_corr_diff"]
    assert list(tr.sigma_table.index) == tr.topics
    assert tr.gamma_std.shape == (L + 1, tr.K) and tr.kkt.shape == (L + 1, 4)
    assert list(tr.kkt.columns) == ["ratio", "active", "penalty", "ratio_polished"]
    assert list(tr.topic_table.columns) == list(T.TOPIC_TABLE_COLUMNS) and list(tr.topic_table.index) == tr.topics
    assert list(tr.gamma_std.index) == ["const"] + tr.panel_topics
    assert tr.factors_in_sample.shape == (len(tr.train_periods), tr.K)
    assert tr.z.shape[1] == L and list(tr.z.columns) == tr.panel_topics
    assert tr.instruments.shape == (N, L) and tr.population["instrument_ref"].shape == (N, L)
    assert tr.path is not None and list(tr.path.columns) == [
        "lam", "criterion", "se", "in_band", "best", "chosen", "n_selected", "total_r2", "objective", "zero_objective",
        "above_zero", "converged", "n_iter", "sigma_ff_truncated", "null_q05", "null_q50", "null_q95",
        "band_floor", "edge"]
    assert tr.path["chosen"].sum() == 1 and tr.path["best"].sum() == 1
    assert tr.path.loc[tr.path["chosen"], "lam"].item() == pytest.approx(tr.lam)
    assert tr.path.loc[tr.path["chosen"], "in_band"].item()
    assert tr.gamma_path is not None and tr.gamma_path.shape == (len(tr.path), L + 1)
    assert tr.path_trace is not None and list(tr.path_trace.columns) == [
        "lam", "n_selected", "criterion", "kept_share", "spearman", "median_r2", "gamma_rank", "gamma_sv_min"]
    chosen = tr.path_trace.loc[np.isclose(tr.path_trace["lam"], tr.lam)]
    assert chosen["spearman"].item() == pytest.approx(tr.ladder.loc["bks_implied", "spearman"], abs=1e-12)
    assert chosen["kept_share"].item() == pytest.approx(tr.capture["kept_share"], abs=1e-12)
    assert list(tr.meta["settings"].columns) == ["setting", "value"]
    assert list(tr.meta["shapes"].columns) == ["quantity", "value", "note"]
    assert tr.meta["timings"]["total"] > 0.0
    assert tr.instrument_week < tr.row_week and tr.instrument_window_end < tr.instrument_week


def test_page_text_says_topic_sensitivity(tr):
    """The page prints check names, notes, findings and labels: they never say "exposure" (G.9)."""
    texts = [c.name for c in tr.checks] + [c.note for c in tr.checks] + [c.relation for c in tr.checks]
    texts += [f["title"] for f in tr.findings] + [f["text"] for f in tr.findings]
    texts += list(T.STEPS.values()) + list(T.STEP_WHAT.values())
    texts += list(T.LADDER.values()) + list(T.LADDER_WHAT.values())
    texts += list(tr.status_frame()["key_number"]) + tr.meta["settings"].astype(str).to_numpy().ravel().tolist()
    texts += tr.meta["shapes"].astype(str).to_numpy().ravel().tolist()
    bad = [t for t in texts if re.search(r"exposure", str(t), flags=re.IGNORECASE)]
    assert bad == []


# ---------------------------------------------------------------------------
# the "should be" references of the addendum (least squares, no-signal band, exact units, days, Sigma_z)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("history", ["full", "training"])
def test_least_squares_rung_recomputes_independently(session, history):
    """bks_ls: m_i = Sigma_c Gt (Gt' Sigma_c Gt)^+ Gt' c~_i' with Sigma_c = C'C / n over the Eq. 5 rows.

    Without the constant's part, like its neighbours best_rank and bks_no_const and like ls_share; the
    training history keeps a nonzero constant row, so there the rung differs from the one with the constant.
    """
    cfg = _generic_cfg(history=history)
    tr = _traced(session, cfg)
    fit = session.bks_fit(cfg)
    has = tr.instruments.notna().all(axis=1).to_numpy()
    C = tr.instruments.to_numpy()[has]  # simulation topic order
    col = {t: j for j, t in enumerate(tr.panel_topics)}
    order = [col[t] for t in tr.topics]
    Gamma = np.asarray(fit.fit.Gamma)
    Gt = Gamma[1:][order]
    Sigma_c = C.T @ C / C.shape[0]
    M = Sigma_c @ Gt @ np.linalg.pinv(Gt.T @ Sigma_c @ Gt)
    beta = tr.chain["beta"].to_numpy()[has]  # includes the constant's part (c_i Gamma)
    np.testing.assert_allclose(beta, np.column_stack([np.ones(len(C)), C]) @ Gamma[[0] + [c + 1 for c in order]],
                               rtol=1e-12, atol=1e-15)
    m_ls = C @ Gt @ M.T  # the topic part of the betas only
    np.testing.assert_allclose(tr.chain["m_ls"].to_numpy()[has], m_ls, rtol=1e-9, atol=1e-14)
    if history == "training":  # the constant is not penalised away here: with it the rung would differ
        assert np.abs(Gamma[0]).max() > 0.0
        assert np.abs(beta @ M.T - m_ls).max() > 1e-6
    # then the production divisor, Sigma_z and standardisation (the chain's own conversion)
    conv = tr.chain["conversion"].to_numpy()[has]
    Sz_pinv = np.linalg.pinv(tr.chain["sigma_z"]["train"].to_numpy(), hermitian=True)
    zs, rs = tr.chain["z_scale"].to_numpy(), tr.chain["ret_scale"].to_numpy()[has]
    B = ((m_ls * conv[:, None]) @ Sz_pinv * (zs[None, :] / rs[:, None])).T
    np.testing.assert_allclose(tr.variants["bks_ls"].to_numpy()[:, has], B, rtol=1e-8, atol=1e-10)
    rec = C @ Gt @ M.T  # the instruments' reconstruction (no constant)
    assert tr.capture["ls_share"] == pytest.approx(float(np.sum(rec**2) / np.sum(C**2)), rel=1e-10)


def test_least_squares_map_equals_eq5_on_instruments_in_the_fit_directions():
    """When the instruments lie in the K directions of Gamma_tilde, both inversions return them exactly."""
    rng = np.random.default_rng(3)
    Gt = rng.normal(size=(9, 3))
    C = rng.normal(size=(40, 3)) @ Gt.T  # every row in col(Gt)
    M_ls, rank = T._ls_map(C, Gt, 1e-10)
    M_eq5, _ = _lsq_map(Gt, 1e-10)
    assert rank == 3
    np.testing.assert_allclose((C @ Gt) @ M_ls.T, C, atol=1e-10)
    np.testing.assert_allclose((C @ Gt) @ M_eq5.T, C, atol=1e-10)
    # generic rows: the least-squares reconstruction keeps more than the projection (Frobenius-optimal in col(C Gt))
    C2 = rng.normal(size=(40, 9))
    M2, _ = T._ls_map(C2, Gt, 1e-10)
    P = _lsq_map(Gt, 1e-10)[0] @ Gt.T
    assert np.sum((C2 @ Gt @ M2.T) ** 2) >= np.sum((C2 @ P) ** 2) - 1e-12


def test_null_sharpe_quantiles_closed_form_and_simulation():
    """Hotelling: the no-signal in-sample annualised MVE Sharpe of K factors over T weeks (defaults 0.87/2.30/4.44)."""
    np.testing.assert_allclose(T.null_sharpe_quantiles(3, 26), [0.8686, 2.3020, 4.4439], atol=5e-4)
    assert np.isnan(T.null_sharpe_quantiles(3, 3)).all() and np.isnan(T.null_sharpe_quantiles(0, 26)).all()
    rng = np.random.default_rng(11)
    K, Tw, n = 3, 26, 40_000
    R = rng.normal(size=(n, Tw, K))
    mu = R.mean(axis=1)
    D = R - mu[:, None, :]
    S = np.einsum("ntk,ntl->nkl", D, D) / (Tw - 1)
    sr = np.sqrt(52.0 * np.einsum("nk,nk->n", mu, np.linalg.solve(S, mu[:, :, None])[:, :, 0]))
    np.testing.assert_allclose(np.quantile(sr, [0.05, 0.5, 0.95]), T.null_sharpe_quantiles(K, Tw), rtol=0.03)


def test_path_band_floor_edge_and_null_columns(cfg, tr):
    path = tr.path
    crit = path["criterion"].to_numpy()
    best = float(np.nanmax(crit))
    tol = cfg.bks.tolerance
    # D51: relative to the best at any level (the generic config's best Sharpe ratio is below 1); only the
    # numerical tie tolerance keeps the max(1, |best|) floor; the tuner's tie_band gives the same width
    band = max(tol * abs(best), 1e-9 * max(1.0, abs(best)))
    assert band == tuning.tie_band(best, tol)
    assert (path["band_floor"] == best - band).all()
    np.testing.assert_array_equal(path["in_band"].to_numpy(), crit >= path["band_floor"].to_numpy())
    assert _checks(tr)["Chosen lambda follows the band rule"].status == T.OK
    np.testing.assert_allclose(path[["null_q05", "null_q50", "null_q95"]].iloc[0].to_numpy(),
                               T.null_sharpe_quantiles(tr.K, len(tr.train_periods)))
    assert path[["null_q05", "null_q50", "null_q95"]].nunique().eq(1).all()
    ends = {int(path["lam"].idxmin()), int(path["lam"].idxmax())}
    expected = [(i in ends) and bool(path["chosen"].iloc[i] or path["best"].iloc[i]) for i in range(len(path))]
    assert path["edge"].tolist() == expected
    assert tr.meta["null_sharpe_quantiles"] == tuple(path[["null_q05", "null_q50", "null_q95"]].iloc[0])
    assert tr.meta["band_floor"] == path["band_floor"].iloc[0]
    # gamma_sv_min: the smallest singular value of the standardised Gamma_tilde; zero exactly when a direction died
    pt = tr.path_trace
    assert (pt["gamma_sv_min"] >= 0.0).all()
    assert ((pt["gamma_sv_min"] > 1e-8) | (pt["gamma_rank"] < tr.K)).all()


def _findings_with_path(tr: T.BKSTrace, path: pd.DataFrame | None, *, null_q=(0.5, 1.0, 2.0),
                        cap: dict | None = None, ladder: pd.DataFrame | None = None, history: str | None = None,
                        share: float = 0.0, eff_days: float = 1.0, ref_corr: float = 1.0,
                        **extra: object) -> list[dict[str, str]]:
    return T._findings(
        [], cap=tr.capture if cap is None else cap, K=tr.K, history=tr.history if history is None else history,
        share=share, eff_days=eff_days, n_pairs=10, ladder=tr.ladder if ladder is None else ladder,
        path=path, dead=False, dead_ratio=1.0, k_eff=tr.K, ref_corr=ref_corr,
        ref_slope=1.0, instrument_week=tr.instrument_week, window_end=tr.instrument_window_end,
        train_end=tr.meta["train_end"], cal=pd.DatetimeIndex([]), conversion_exact=np.ones(3),
        conversion_kernel=np.ones(3), scaled=True, zero_obj=1e9, null_q=np.asarray(null_q), T_train=20,
        n_topics=len(tr.topics), **extra)


def test_findings_null_band_and_grid_edge(tr):
    lam = np.logspace(-2, 0, 5)
    crit = np.array([0.80, 0.79, 0.785, 0.70, 0.60])  # best at the smallest lambda, below 1
    floor = 0.80 - 0.02 * 0.80  # the 2% band is relative below 1 too (D51): 0.784
    path = pd.DataFrame({"lam": lam, "criterion": crit, "se": 1.0, "in_band": crit >= floor,
                         "best": [True, False, False, False, False], "chosen": [False, False, True, False, False],
                         "n_selected": [9, 8, 7, 5, 2], "above_zero": False})
    titles = {f["title"]: f for f in _findings_with_path(tr, path)}
    # (a) every point inside the no-signal 5-95% band replaces "within noise"
    assert "The lambda choice follows noise" in titles and "The lambda choice is within noise" not in titles
    assert "0.50-2.00" in titles["The lambda choice follows noise"]["text"]
    assert titles["The lambda choice follows noise"]["severity"] == "departure"
    # (b) the best point is the grid's first point
    edge = titles["The choice sits at the edge of the lambda grid"]
    assert edge["severity"] == "departure" and "best in-sample Sharpe ratio is at the smallest lambda" in edge["text"]
    assert "chosen lambda" not in edge["text"]
    # (c) one band rule (D51): no finding about the tolerance band, also with a best Sharpe ratio below 1
    assert not any("tolerance band" in t for t in titles)
    # outside the no-signal band: the standard-error rule applies again
    far = _findings_with_path(tr, path, null_q=(0.9, 1.0, 2.0))
    far_titles = {f["title"] for f in far}
    assert "The lambda choice is within noise" in far_titles and "The lambda choice follows noise" not in far_titles


def _exact_kernel_cov(tr: T.BKSTrace, panel, sim, asset: str) -> tuple[np.ndarray, float]:
    """Independent recompute: the raw-return kernel covariance of the Eq. 5 instrument week and the kernel divisor."""
    days = pd.DatetimeIndex(panel.aligned.calendar)
    pid, ends = period_end_index(days, "W")
    _, stop, cut = window_bounds(pid, int(tr.meta["skip_days"]))
    j = int(ends.get_loc(tr.chain["row_period"][asset])) - 1
    wts = kernel_weights(pid, j, float(tr.meta["xi"]))
    wts[cut[j]:stop[j]] = 0.0
    raw = sim.market.returns[asset].reindex(days).to_numpy(dtype=float)
    Z = tr.z[tr.topics].to_numpy(dtype=float)
    exact = brute_force_covariance(raw, Z, wts)
    ok = np.isfinite(raw) & np.isfinite(Z).all(axis=1) & (wts > 0)
    scale = (panel.aligned.scale[asset].to_numpy(dtype=float) if panel.aligned.scale is not None
             else np.ones(len(days)))
    return exact, float(wts[ok] @ scale[ok] / wts[ok].sum())


@pytest.mark.parametrize("kw", [dict(), dict(history="training"), dict(asset_weighting="none")])
def test_units_exact_conversion(session, kw):
    c = _generic_cfg(**kw)
    trace = _traced(session, c)
    panel, sim = session.bks_panel(c), session.simulation(c)
    u = trace.units
    for asset in trace.assets[:4]:
        exact, kdiv = _exact_kernel_cov(trace, panel, sim, asset)
        conv = trace.instruments.loc[asset].to_numpy() * u.loc[asset, "divisor"]
        ratio = float(conv @ exact / (exact @ exact))
        assert u.loc[asset, "conversion_exact"] == pytest.approx(ratio, rel=1e-10)
        assert u.loc[asset, "kernel_divisor"] == pytest.approx(kdiv, rel=1e-12)
        assert u.loc[asset, "conversion_kernel"] == pytest.approx(ratio * kdiv / u.loc[asset, "divisor"], rel=1e-10)
    by_name = {ch.name: ch for ch in trace.checks}
    if c.bks.history == "training" or c.bks.asset_weighting == "none":
        # the divisor is constant (or 1): the conversion is exact for every asset
        np.testing.assert_allclose(u["conversion_exact"], 1.0, rtol=0, atol=1e-10)
        ch = by_name["Implied covariances in exact return units"]
        assert ch.status == T.OK and ch.kind == T.IDENTITY and ch.step == "align"
        assert by_name["Return units are exact"].status == T.OK
    else:
        ch = by_name["Implied covariances against exact return units"]
        assert ch.status == T.INFO and ch.value == pytest.approx(float(np.max(np.abs(u["conversion_exact"] - 1))))
        assert by_name["Approximate return units"].status == T.INFO


def test_forecast_in_exact_return_units(session, cfg, tr):
    result = session.bks(cfg)
    fitted = result.fitted
    exact = result.meta["realized_exact"].reindex(index=fitted.index, columns=fitted.columns)
    ok = fitted.notna() & exact.notna()
    sse = ((exact - fitted) ** 2).where(ok).sum(axis=1)
    syy = (exact**2).where(ok).sum(axis=1)
    np.testing.assert_allclose(tr.weeks["r2_exact"].to_numpy(), (1 - sse / syy).reindex(tr.weeks.index), rtol=1e-12)
    pooled = 1 - float(sse.sum()) / float(syy.sum())
    assert tr.meta["r2_pooled_exact"] == pytest.approx(pooled, rel=1e-12)
    assert tr.meta["r2_pooled_panel"] == result.r2_pooled
    by_name = {ch.name: ch for ch in tr.checks}
    ch = by_name["Pooled OOS R2 in exact return units"]
    assert ch.status == T.INFO and ch.value == pytest.approx(pooled) and ch.reference == result.r2_pooled
    rel = ((result.realized - exact).abs() / exact.abs()).where(ok & (exact != 0))
    assert by_name["Approximate return units"].value == pytest.approx(float(np.nanmax(rel.to_numpy())), rel=1e-12)
    assert tr.meta["realized_max_rel_error"] == by_name["Approximate return units"].value


@pytest.mark.parametrize("history", ["full", "training"])
def test_training_days_against_direct_days(session, history):
    c = _generic_cfg(history=history)
    trace = _traced(session, c)
    panel, sim = session.bks_panel(c), session.simulation(c)
    days = pd.DatetimeIndex(panel.aligned.calendar)
    pid, ends = period_end_index(days, "W")
    bks_days = days[pd.Index(ends[pid]).isin(trace.train_periods)]
    first_week = days[ends[pid] == trace.train_periods.min()]
    ts = pd.Timestamp(c.window.train_start)
    shocks = session.shocks(c)
    _, q_pos = training_pairs(sim, shocks, shock_matrix(shocks, sim.market.calendar, sim.topics.ids))
    direct_days = pd.DatetimeIndex(sim.market.calendar[q_pos])
    m = trace.meta
    assert m["bks_train_days"] == len(bks_days) and m["days_before_train_start"] == int((first_week < ts).sum())
    assert m["direct_train_days"] == len(direct_days) == trace.meta["n_pairs"]
    missing = [d.date().isoformat() for d in direct_days.difference(bks_days)]
    assert m["direct_days_not_in_bks_dates"] == missing and m["direct_days_not_in_bks"] == len(missing)
    ch = next(x for x in trace.checks if x.name == "Training weeks cover the training days")
    assert ch.step == "panel" and ch.status == T.INFO and ch.value == len(bks_days)
    if history == "training":
        assert m["days_before_train_start"] == 0


def test_sigma_table_and_fit_info_checks(session, cfg, tr):
    sz = tr.chain["sigma_z"]
    d_tr, d_k, d_p = (np.diag(sz[k].to_numpy()) for k in ("train", "kernel", "population"))
    np.testing.assert_allclose(tr.sigma_table["train_over_population"], d_tr / d_p, rtol=1e-12)
    np.testing.assert_allclose(tr.sigma_table["kernel_over_population"], d_k / d_p, rtol=1e-12)
    np.testing.assert_allclose(tr.sigma_table["train_over_kernel"], d_tr / d_k, rtol=1e-12)
    c_tr = sz["train"].to_numpy() / np.sqrt(np.outer(d_tr, d_tr))
    c_k = sz["kernel"].to_numpy() / np.sqrt(np.outer(d_k, d_k))
    gap = np.abs(c_tr - c_k)
    np.fill_diagonal(gap, -1.0)
    np.testing.assert_allclose(tr.sigma_table["max_corr_diff"], gap.max(axis=1), rtol=1e-12)
    fit = session.bks_fit(cfg)
    by_name = {ch.name: ch for ch in tr.checks}
    inner = by_name["Inner group-lasso solves converged"]
    assert inner.status == T.INFO and inner.value == float(bool(fit.fit.meta["inner_all_converged"]))
    assert tr.meta["inner_iters"] == fit.fit.meta["inner_iters"]
    nxt = by_name["Next topic to enter"]
    idle = tr.kkt.loc[~tr.kkt["active"] & tr.kkt["ratio"].notna(), "ratio"]
    assert nxt.status == T.INFO and nxt.value == pytest.approx(idle.max())
    assert tr.meta["next_to_enter"] == idle.idxmax() and tr.meta["next_to_enter"] in nxt.note
    lam0 = _traced(session, _generic_cfg(lambda_rule="fixed", lam=0.0))
    by0 = {ch.name: ch for ch in lam0.checks}
    for name in ("Inner group-lasso solves converged", "Next topic to enter"):
        assert by0[name].status == T.INFO and np.isnan(by0[name].value), name
    assert lam0.meta["inner_all_converged"] is None and lam0.meta["next_to_enter"] == ""
    assert lam0.path is None and np.isfinite(lam0.meta["null_sharpe_quantiles"]).all()


# ---------------------------------------------------------------------------
# per-selection helpers
# ---------------------------------------------------------------------------
def _asset_topic(tr: T.BKSTrace) -> tuple[str, str]:
    asset = tr.assets[0]
    return asset, str(tr.variants["oracle"][asset].abs().idxmax())


def test_asset_and_topic_days(session, cfg, tr):
    panel, sim, shocks = session.bks_panel(cfg), session.simulation(cfg), session.shocks(cfg)
    asset, topic = _asset_topic(tr)
    ad = T.asset_days(tr, panel, sim, asset)
    assert list(ad.columns) == ["raw", "divisor", "scaled", "ret_scale", "asset_vol", "role"]
    assert ad.index.equals(pd.DatetimeIndex(sim.market.calendar))
    assert set(ad["role"]) <= {"before panel", "training", "forecast", "other"}
    ok = ad["scaled"].notna()
    np.testing.assert_allclose((ad["scaled"] * ad["divisor"])[ok], ad["raw"][ok], rtol=1e-12)
    assert ad.loc[ad["role"] == "before panel", "divisor"].isna().all()
    td = T.topic_days(tr, panel, sim, shocks, topic)
    assert list(td.columns) == ["attention", "z_bks", "z_direct", "z_signal", "z_news", "z_slow", "designed"]
    assert td.index.equals(pd.DatetimeIndex(panel.aligned.calendar))
    both = td["z_bks"].notna() & td["z_direct"].notna()
    np.testing.assert_allclose(td.loc[both, "z_bks"], td.loc[both, "z_direct"], rtol=0, atol=1e-15)
    parts = td[["z_signal", "z_news", "z_slow"]].sum(axis=1, skipna=False)
    ok = parts.notna() & td["z_direct"].notna()
    assert np.abs(parts[ok] - td.loc[ok, "z_direct"]).max() < 1e-6  # exact except where attention hit its floor


def test_instrument_series_and_kernel_profile(session, cfg, tr):
    panel = session.bks_panel(cfg)
    asset, topic = _asset_topic(tr)
    s = T.instrument_series(tr, panel, asset, topic)
    assert list(s.columns) == ["kernel_cov", "panel", "n_days", "return_week", "role", "population"]
    assert s.index.equals(tr.week_ends)
    kept = s["panel"].notna()
    assert kept.sum() > 0
    np.testing.assert_allclose(s.loc[kept, "kernel_cov"], s.loc[kept, "panel"], rtol=1e-10, atol=1e-16)
    assert (s.loc[kept, "n_days"] >= tr.meta["min_days"]).all()
    assert (s.loc[s["n_days"] < tr.meta["min_days"], "kernel_cov"].isna()).all()
    assert set(s["role"]) <= {"burn-in", "training", "forecast", "other"}
    assert (s["role"] == "training").sum() == len(tr.train_periods)
    frame, stats = T.kernel_profile(tr, panel, asset, topic, tr.row_week)
    assert list(frame.columns) == ["weight", "contribution", "cumulative"]
    assert frame["weight"].sum() == pytest.approx(1.0)
    assert stats["value"] == pytest.approx(stats["panel_value"], rel=1e-12)
    assert frame["cumulative"].iloc[-1] == pytest.approx(stats["value"], rel=1e-12)
    assert stats["instrument_week"] == tr.instrument_week and stats["window_end"] == tr.instrument_window_end
    assert 0.0 <= stats["share_before_train"] <= 1.0 and stats["effective_days"] > 1.0
    assert stats["n_days"] >= tr.meta["min_days"]
    with pytest.raises(ValueError):
        T.kernel_profile(tr, panel, asset, topic, tr.week_ends[0])


def test_instrument_row_and_pairing(session, cfg, tr):
    panel, sim = session.bks_panel(cfg), session.simulation(cfg)
    asset, _ = _asset_topic(tr)
    row = T.instrument_row(tr, panel, sim, asset, tr.row_week)
    assert list(row.columns) == ["panel", "brute_force", "difference", "population", "signal_part", "noise_part"]
    assert list(row.index) == tr.topics
    np.testing.assert_allclose(row["brute_force"], row["panel"], rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(row["signal_part"] + row["noise_part"], row["brute_force"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(row["panel"], tr.instruments.loc[asset], rtol=0, atol=0)
    pt = T.pairing_table(tr, panel, sim, asset)
    assert list(pt.columns) == ["return_week", "first_day", "instrument_week", "window_end", "role", "y_panel",
                                "y_recomputed", "raw_return", "n_assets"]
    assert pt["return_week"].iloc[-1] == tr.forecast_periods.max()
    assert (pt["role"] == "training").sum() == len(tr.train_periods)
    has = pt["y_panel"].notna()
    np.testing.assert_allclose(pt.loc[has, "y_panel"], pt.loc[has, "y_recomputed"], rtol=1e-12)
    assert (pt["instrument_week"] < pt["first_day"]).all() and (pt["window_end"] < pt["instrument_week"]).all()


def test_design_forecast_week_and_chain(session, cfg, tr):
    panel, fit, result = session.bks_panel(cfg), session.bks_fit(cfg), session.bks(cfg)
    dm = T.design_matrix(tr, panel, fit, tr.row_week)
    assert list(dm.columns) == ["const"] + tr.panel_topics and (dm["const"] == 1.0).all()
    assert dm.shape[0] == len(tr.assets)
    assert T.design_matrix(tr, panel, fit, pd.Timestamp("1990-01-05")).empty
    for wk in tr.forecast_periods:
        frame, stats = T.forecast_week(tr, panel, fit, result, wk)
        np.testing.assert_allclose(stats["factors"], result.meta["factors"].loc[wk].to_numpy(), rtol=0, atol=0)
        np.testing.assert_allclose(stats["factors_closed_form"], stats["factors"], rtol=1e-10, atol=1e-14)
        assert stats["foc_max"] < 1e-8 and stats["ridge"] == result.meta["oos_ridge"]
        assert stats["r2_week"] == pytest.approx(tr.weeks.loc[wk, "r2"])
        assert list(frame.columns) == ["realized", "fitted", "residual"] + [f"beta_{k + 1}" for k in range(tr.K)]
        np.testing.assert_allclose(frame["realized"] - frame["fitted"], frame["residual"])
    with pytest.raises(ValueError):
        T.forecast_week(tr, panel, fit, result, tr.row_week)
    asset, _ = _asset_topic(tr)
    ch = T.asset_chain(tr, asset, session.direct(cfg))
    assert list(ch.index) == tr.topics and ch["direct"].notna().all()
    np.testing.assert_allclose(ch["B_hat"], session.bks_implied(cfg).B_hat[asset], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ch["m"], ch["projected"] + ch["constant_part"], rtol=0, atol=1e-15)
    assert T.asset_chain(tr, asset)["direct"].isna().all()


# ---------------------------------------------------------------------------
# session stage
# ---------------------------------------------------------------------------
def test_session_stage_never_fits_and_caches():
    assert "bks_trace" in SESSION_STAGES and "bks_trace" in BKS_STAGES
    assert SESSION_STAGES.index("bks_trace") == SESSION_STAGES.index("bks_implied") + 1
    s = LabSession(max_entries=3)
    c = _generic_cfg(lambda_rule="fixed", lam=0.5)
    key = s.stage_key("bks_trace", c)
    assert key.startswith("bks_trace-") and key == s.stage_key("bks_trace", dataclasses.replace(c))
    other = dataclasses.replace(c, window=dataclasses.replace(c.window, forecast_weeks=2))
    assert s.stage_key("bks_trace", other) != key
    with pytest.raises(LookupError, match=re.escape(BKS_NOT_RUN)):
        s.bks_trace(c)
    assert not s.has("bks_fit", c) and not s.has("bks_panel", c) and not s.has("bks_trace", c)
    s.bks_fit(c)
    first = s.bks_trace(c)
    assert s.has("bks_trace", c) and s.bks_trace(c) is first
    assert s.last_timings["bks_trace"]["cached"] is True
    assert s._limits["bks_trace"] == min(3, 2)


def test_trace_leaves_cached_results_untouched(session, cfg, tr):
    """Results of cached stages are shared: the trace copies, never modifies (and never caches moments on them)."""
    panel, fit, implied = session.bks_panel(cfg), session.bks_fit(cfg), session.bks_implied(cfg)
    before = (panel.panel.X.copy(), fit.fit.Gamma.copy(), fit.fit.F.copy(), implied.B_hat.copy())
    assert panel.panel._moments is None
    trace = T.build_trace(panel, fit, session.bks(cfg), session.simulation(cfg), session.shocks(cfg),
                          session.truth(cfg), cfg.window, cfg.bks, implied)
    assert panel.panel._moments is None
    np.testing.assert_array_equal(panel.panel.X, before[0])
    np.testing.assert_array_equal(fit.fit.Gamma, before[1])
    np.testing.assert_array_equal(fit.fit.F, before[2])
    assert implied.B_hat.equals(before[3])
    assert not np.shares_memory(trace.factors_in_sample.to_numpy(), fit.fit.F)
    assert trace.ladder.equals(tr.ladder)  # deterministic


def test_path_trace_switch_and_guard(session, cfg, monkeypatch):
    args = (session.bks_panel(cfg), session.bks_fit(cfg), session.bks(cfg), session.simulation(cfg),
            session.shocks(cfg), session.truth(cfg), cfg.window, cfg.bks, session.bks_implied(cfg))
    off = T.build_trace(*args, path_trace=False)
    assert off.path_trace is None and off.path is not None
    check = next(c for c in off.checks if c.name == "Refitting the path reproduces the fit")
    assert check.status == T.INFO and "switched off" in check.note
    monkeypatch.setattr(T, "PATH_TRACE_MAX_TOPICS", 5)
    monkeypatch.setattr(T, "LAMBDA_MAX_MAX_TOPICS", 5)
    auto = T.build_trace(*args)
    assert auto.path_trace is None and "skipped" in auto.meta["path_trace_note"]
    lm = next(c for c in auto.checks if c.name == "lambda_max recompute")
    assert lm.status == T.INFO
    # above LAMBDA_MAX_MAX_TOPICS topics the polish runs one sweep when the stored fit meets both tolerances
    assert auto.meta["polish"]["cap"] == 1 and auto.meta["polish"]["kkt_stored"] <= T._KKT_TOL
    kkt = next(c for c in auto.checks if c.name == "Stationarity of the group lasso (KKT)")
    assert kkt.status == T.OK and "one more sweep" in kkt.note and _identity_off(auto) == []
    panel, fit, _, sim, shocks, truth, window, bks_cfg, _ = args
    frame = T.lambda_path_trace(panel, fit, sim, shocks, truth, window, bks_cfg)
    assert len(frame) == cfg.bks.n_lambdas
    chosen = int(fit.tuning.meta["chosen_index"])
    assert frame["lam"].iloc[chosen] == pytest.approx(fit.lam)
    assert frame["spearman"].iloc[chosen] == pytest.approx(off.ladder.loc["bks_implied", "spearman"], abs=1e-12)
    fixed = _generic_cfg(lambda_rule="fixed", lam=0.0)
    session.bks_fit(fixed)
    with pytest.raises(ValueError, match="fixed"):
        T.lambda_path_trace(session.bks_panel(fixed), session.bks_fit(fixed), session.simulation(fixed),
                            session.shocks(fixed), session.truth(fixed), fixed.window, fixed.bks)


def test_wrong_truth_window_raises(session, cfg):
    sim = session.simulation(cfg)
    with pytest.raises(ValueError, match="shock windows differ"):
        T.build_trace(session.bks_panel(cfg), session.bks_fit(cfg), session.bks(cfg), sim, session.shocks(cfg),
                      truth_for_window(sim, 3), cfg.window, cfg.bks, session.bks_implied(cfg))


@pytest.mark.parametrize("history", ["full", "training"])
def test_asset_without_training_rows(history):
    """An asset whose returns stop before the training window has no row: zero sensitivities, as in production."""
    c = _generic_cfg(history=history)
    sim = simulate_lab(c)
    returns = sim.market.returns.copy()
    gone = str(returns.columns[3])
    returns.loc[returns.index >= pd.Timestamp("2020-06-01"), returns.columns[3]] = np.nan
    sim = dataclasses.replace(sim, market=dataclasses.replace(sim.market, returns=returns))
    w = c.window
    shocks = observed_shocks(sim.attention, w.shock_window, w.train_start, w.train_end)
    truth = truth_for_window(sim, w.shock_window)
    panel = bks.build_bks_panel(sim, c.bks, w.shock_window, train_start=w.train_start, train_end=w.train_end)
    fit = bks.fit_bks(panel, c.bks, w.train_end, train_start=w.train_start)
    result = bks.evaluate_bks(panel, fit, w)
    implied = bks.implied_exposures(panel, fit, sim, shocks)
    trace = T.build_trace(panel, fit, result, sim, shocks, truth, w, c.bks, implied, path_trace=False)
    assert _identity_off(trace) == []
    assert trace.chain["skipped_assets"] == implied.meta["skipped_assets"] == [gone]
    assert (trace.chain["B_hat"][gone] == 0.0).all() and (implied.B_hat[gone] == 0.0).all()
    assert trace.instruments.loc[gone].isna().all() and bool(trace.units.loc[gone, "skipped"])
    assert (T.asset_chain(trace, gone)["B_hat"] == 0.0).all()
    assert T.pairing_table(trace, panel, sim, gone)["y_panel"].isna().all()


# ---------------------------------------------------------------------------
# review fixes 2026-09-30: stationarity on a polished fit, conditioning, independent step 8, findings
# ---------------------------------------------------------------------------
def _checks(trace: T.BKSTrace) -> dict[str, T.TraceCheck]:
    return {c.name: c for c in trace.checks}


def _titles(trace: T.BKSTrace) -> dict[str, dict[str, str]]:
    return _by_title(trace.findings)


def _by_title(findings: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {f["title"]: f for f in findings}


def _no_path_findings(tr: T.BKSTrace, **kw: object) -> dict[str, dict[str, str]]:
    """:func:`trace._findings` without a lambda path (the fixed rule), by title."""
    return _by_title(_findings_with_path(tr, None, **kw))


def _build(session: LabSession, cfg: LabConfig, **kw: object) -> T.BKSTrace:
    """build_trace on the session's cached stages, with any of them replaced by keyword."""
    session.bks_fit(cfg)
    args = dict(panel=session.bks_panel(cfg), fit=session.bks_fit(cfg), result=session.bks(cfg),
                sim=session.simulation(cfg), shocks=session.shocks(cfg), truth=session.truth(cfg), window=cfg.window,
                bks_cfg=cfg.bks, implied=session.bks_implied(cfg))
    args.update(kw)
    return T.build_trace(args["panel"], args["fit"], args["result"], args["sim"], args["shocks"], args["truth"],
                         args["window"], args["bks_cfg"], args["implied"], path_trace=False)


def test_every_topic_dropped_balance_not_applicable(session):
    """Gamma = 0 (lambda above lambda_max): the balance 0 = 0 holds (info), no nan, no Eq. 5 finding at rank 0."""
    trace = _traced(session, _generic_cfg(lambda_rule="fixed", lam=50.0))
    by = _checks(trace)
    assert trace.meta["selected_topics"] == [] and trace.chain["gamma_rank"] == 0
    bal = by["Penalty and ridge in balance"]
    assert bal.status == T.INFO and bal.kind == T.IDENTITY and bal.value == 0.0 and "every row dropped" in bal.relation
    assert by["Stationarity of the group lasso (KKT)"].status == T.OK
    assert _identity_off(trace) == []
    assert "Spearman n/a" in trace.meta["key_numbers"]["implied"]
    titles = _titles(trace)
    assert "The fit's directions keep little of the instruments" not in titles
    assert "A factor is dead" in titles
    assert "The forecast R2 does not measure topic signal" not in titles  # R2 = 0: nothing to explain
    _assert_no_nan_text(trace)


def test_stationarity_graded_on_the_polished_fit_at_small_lambda(session):
    """At lambda 0.002 the stored fit's KKT residual is above 5e-3 (it stops on 1e-8); the polished copy is not."""
    trace = _traced(session, _generic_cfg(lambda_rule="fixed", lam=0.002))
    pol = trace.meta["polish"]
    by = _checks(trace)
    assert pol["kkt_stored"] > T._KKT_TOL  # the old check was off here
    assert pol["kkt_polished"] < T._KKT_TOL and pol["balance_polished"] < T._KKT_TOL and pol["converged"]
    kkt = by["Stationarity of the group lasso (KKT)"]
    assert kkt.status == T.OK and kkt.value == pytest.approx(pol["kkt_polished"])
    assert by["Penalty and ridge in balance"].status == T.OK
    stored = by["Stationarity of the stored fit"]
    assert stored.status == T.INFO and stored.value == pytest.approx(pol["kkt_stored"])
    stop = by["The stored fit is at its stopping point"]
    assert stop.status == T.OK and stop.kind == T.IDENTITY
    # this fit stopped at its sweep cap: a diagnostic and a departure, not a defect
    conv = by["The fit converged before its sweep cap"]
    assert conv.kind == T.DIAGNOSTIC and conv.status == (T.OK if pol["stored_converged"] else T.OFF)
    assert ("The fit stopped at its sweep cap" in _titles(trace)) == (not pol["stored_converged"])
    assert _identity_off(trace) == []
    # every kept topic's ratio is 1 on the polished copy; the frame keeps both
    k = trace.kkt
    act = k["active"] & k["ratio_polished"].notna()
    np.testing.assert_allclose(k.loc[act, "ratio_polished"], 1.0, atol=T._KKT_TOL)
    assert (k.loc[act, "ratio"] - 1.0).abs().max() > (k.loc[act, "ratio_polished"] - 1.0).abs().max()
    # every topic is kept: the KKT note says so instead of printing nan
    if bool(k.loc[k["ratio"].notna(), "active"].all()):
        assert "no topic is dropped" in kkt.note


def test_stationarity_checks_catch_a_gamma_the_fit_did_not_reach(session, cfg):
    """A stored Gamma 1% off the fit's own (its factors re-solved) turns the stopping-point and KKT checks off."""
    fit = session.bks_fit(cfg)
    panel = session.bks_panel(cfg)
    pnl = panel.panel
    sub = pnl.subset_periods(np.asarray(pd.DatetimeIndex(pnl.periods).isin(fit.train_periods), dtype=bool))
    mom = sub.moments()
    G = np.asarray(fit.fit.Gamma) * 1.01
    from narrative_ipca import sparse_ipca as si

    F = si.f_step(mom.S, mom.V, G, ridge=2.0)
    wrong = dataclasses.replace(fit, fit=dataclasses.replace(fit.fit, Gamma=G, F=F))
    trace = _build(session, cfg, fit=wrong)
    by = _checks(trace)
    assert by["The stored fit is at its stopping point"].status == T.OFF
    assert by["Stationarity of the group lasso (KKT)"].status == T.OFF
    assert by["Penalty and ridge in balance"].status == T.OFF
    defects = [f for f in trace.findings if f["severity"] == "defect"]
    assert {f["title"] for f in defects} >= {"Check off: Stationarity of the group lasso (KKT)",
                                             "Check off: The stored fit is at its stopping point"}
    assert all(f["origin"] == "defect" for f in defects)
    assert trace.status_frame().set_index("step").loc["fit", "reading"] == "off"


def test_plain_ipca_with_few_assets_scales_the_tolerances(session):
    """lambda 0, K = 3, four assets: the weekly factor systems are ill-conditioned; the tolerances follow cond."""
    trace = _traced(session, _generic_cfg(n_assets=4, lambda_rule="fixed", lam=0.0))
    by = _checks(trace)
    assert trace.meta["factor_cond_max"] > 1e3
    for name in ("Objective recompute", "Factors = closed-form factor step"):
        assert by[name].status == T.OK and by[name].tolerance > 1e-10, by[name]
        assert "condition number" in by[name].note
    assert _identity_off(trace) == []
    # on the well-conditioned default generic fit the tolerances stay at 1e-10
    base = _checks(_traced(session, _generic_cfg()))
    assert base["Objective recompute"].tolerance == 1e-10
    assert base["Factors = closed-form factor step"].tolerance == 1e-10


def test_independent_step8_checks_catch_a_wrong_projection_and_pseudo_inverse(session, cfg, monkeypatch):
    """With the production map halved (in bks and the trace's own chain), the independent recomputes go off."""
    shocks, sim = session.shocks(cfg), session.simulation(cfg)
    panel, fit = session.bks_panel(cfg), session.bks_fit(cfg)
    real_map = bks._lsq_map

    def half_map(M, rcond):
        out, rank = real_map(M, rcond)
        return 0.5 * out, rank

    with monkeypatch.context() as mp:
        mp.setattr(bks, "_lsq_map", half_map)
        mp.setattr(T, "_lsq_map", half_map)
        implied = bks.implied_exposures(panel, fit, sim, shocks)
        trace = _build(session, cfg, implied=implied)
    by = _checks(trace)
    assert by["Step-by-step chain = production (consistency)"].status == T.OK  # the copy agrees with itself
    assert by["Projection = numpy pseudo-inverse"].status == T.OFF
    assert by["Projection is orthogonal, of the fit's rank"].status == T.OFF
    assert by["Implied sensitivities rebuilt with numpy = production"].status == T.OFF
    real_pinv = bks._sym_pinv

    def diag_pinv(S, rcond):
        _, rank = real_pinv(S, rcond)
        d = np.diag(np.asarray(S, dtype=float))
        return np.diag(np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)), rank

    with monkeypatch.context() as mp:
        mp.setattr(bks, "_sym_pinv", diag_pinv)
        mp.setattr(T, "_sym_pinv", diag_pinv)
        implied = bks.implied_exposures(panel, fit, sim, shocks)
        trace = _build(session, cfg, implied=implied)
    by = _checks(trace)
    assert by["Sigma_z pseudo-inverse is Moore-Penrose"].status == T.OFF
    assert by["Implied sensitivities rebuilt with numpy = production"].status == T.OFF
    # a production result that is off by a constant is caught by both the consistency and the independent check
    shifted = dataclasses.replace(session.bks_implied(cfg), B_hat=session.bks_implied(cfg).B_hat + 1e-6)
    trace = _build(session, cfg, implied=shifted)
    by = _checks(trace)
    assert by["Step-by-step chain = production (consistency)"].status == T.OFF
    assert by["Implied sensitivities rebuilt with numpy = production"].status == T.OFF
    f = _titles(trace)["Check off: Step-by-step chain = production (consistency)"]
    assert f["severity"] == "defect" and f["origin"] == "defect" and "nan" not in f["text"]


def test_independent_pieces_match_numpy(session, cfg, tr):
    """The independent pieces themselves: P a projector of Gamma's rank, Sigma_z = np.cov over the training days."""
    ind = T._independent_eq5(session.bks_panel(cfg), session.bks_fit(cfg), session.simulation(cfg),
                             session.shocks(cfg), float(session.bks_panel(cfg).pipeline_cfg.evaluation.rcond))
    P = ind["P"]
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    np.testing.assert_allclose(P @ P, P, atol=1e-12)
    assert np.trace(P) == pytest.approx(ind["rank"]) and ind["rank"] == tr.chain["gamma_rank"]
    sim, shocks = session.simulation(cfg), session.shocks(cfg)
    p_pos, _ = training_pairs(sim, shocks, shock_matrix(shocks, sim.market.calendar, sim.topics.ids))
    Z = shocks.z.reindex(index=sim.market.calendar, columns=sim.topics.ids).to_numpy()[p_pos]
    np.testing.assert_allclose(ind["Sigma_z"], np.cov(Z, rowvar=False, bias=True), rtol=1e-12)
    np.testing.assert_allclose(ind["B"], tr.variants["bks_implied"].to_numpy(), rtol=0, atol=1e-10)


def test_chosen_fit_above_the_zero_objective(tr):
    """The chosen fit worse than Gamma = 0 is a departure; other path points above it stay a note."""
    assert _checks(tr)["Chosen fit beats Gamma = 0"].status == T.OK and tr.meta["chosen_above_zero"] is False
    lam = np.logspace(-2, 0, 4)
    path = pd.DataFrame({"lam": lam, "criterion": [3.0, 2.9, 2.8, 2.7], "se": 0.01, "in_band": True,
                         "best": [True, False, False, False], "chosen": [False, False, False, True],
                         "n_selected": [9, 8, 7, 5], "above_zero": [False, False, True, True]})
    common = dict(null_q=(0.1, 0.2, 0.3))
    above = _by_title(_findings_with_path(tr, path, chosen_above=True, objective=110.0, zero_reference=100.0,
                                          lam=1.0, **common))
    worse = above["The chosen fit is worse than no fit"]
    assert worse["severity"] == "departure" and worse["origin"] == "method" and worse["step"] == "fit"
    assert "110.0" in worse["text"] and "100.0" in worse["text"]
    spurious = above["Spurious stationary points on the path"]
    assert spurious["severity"] == "note"
    assert "1 of the 3 path points other than the chosen one ends" in spurious["text"]
    below = {f["title"] for f in _findings_with_path(tr, path, **common)}
    assert "The chosen fit is worse than no fit" not in below


def test_instruments_graded_by_effective_days(session):
    """A 3-month half-life under the full history: few effective days, so the population check is info, not off."""
    trace = _traced(session, _generic_cfg(half_life_months=3.0))
    ch = _checks(trace)["Instruments track their population value"]
    eff = trace.meta["effective_days"]
    assert eff < T._MIN_EFF_DAYS and ch.status == T.INFO and f"{eff:,.0f}" in ch.note
    assert "full-sample volatility" in ch.note  # the reference divisor, named
    base = _checks(_traced(session, _generic_cfg()))["Instruments track their population value"]
    assert base.status == T.OK and base.kind == T.DIAGNOSTIC and "full-sample volatility" in base.note
    train = _checks(_traced(session, _generic_cfg(history="training")))["Instruments track their population value"]
    assert train.status == T.INFO and "training standard deviation" in train.note
    # the finding: a note quoting the effective days below the threshold, a departure above it
    for eff_days, history, severity in ((188.0, "full", "note"), (2400.0, "full", "departure"),
                                        (120.0, "training", "note")):
        f = _no_path_findings(trace, ref_corr=0.78, eff_days=eff_days, history=history)
        inst = f["Instruments are far from their population value"]
        assert inst["severity"] == severity and inst["origin"] == "data" and f"{eff_days:,.0f}" in inst["text"]
        assert "long kernel" not in inst["text"]


def test_no_topic_signal(session):
    """No true link: the instruments' reference is 0, so the check is info and no text prints nan."""
    trace = _traced(session, _generic_cfg(signal_share=0.0))
    ch = _checks(trace)["Instruments track their population value"]
    assert ch.status == T.INFO and "no topic signal" in ch.relation and np.isfinite(ch.value)
    assert "n/a (no topic signal)" in trace.meta["key_numbers"]["instruments"]
    assert trace.status_frame().set_index("step").loc["instruments", "reading"] == "as expected"
    assert (trace.topic_table["sum_abs_B_true"] == 0.0).all() and (trace.topic_table["n_links"] == 0).all()
    assert "The selected topics are not those with the largest true sensitivities" not in _titles(trace)
    _assert_no_nan_text(trace)


def test_findings_rank_of_gamma_and_least_squares_sentence(tr):
    """Finding 1 counts the directions of Gamma's topic rows; the least-squares sentence only when it holds."""
    ladder = tr.ladder.copy()
    cap = dict(tr.capture)
    cap["kept_share"], cap["ls_share"] = 0.05, 0.40
    one = _no_path_findings(tr, cap=cap, gamma_rank=1, ladder=ladder)
    text = one["The fit's directions keep little of the instruments"]["text"]
    best1 = float(cap["singular_share"].iloc[:1].sum())
    assert "The fit's 1 direction keeps 5%" in text and f"the best 1 keep {best1:.0%}" in text
    assert f"K = {tr.K}" in text and f"(random {1 / len(tr.topics):.0%})" in text
    sp = ladder["spearman"]
    assert ("still in the betas" in text) == bool(sp["bks_ls"] > sp["bks_no_const"])
    # least squares no better than the fit's directions: no sentence
    worse = ladder.copy()
    worse.loc["bks_ls", "spearman"] = float(worse.loc["bks_no_const", "spearman"]) - 0.1
    f2 = _no_path_findings(tr, cap=cap, gamma_rank=1, ladder=worse)
    assert "still in the betas" not in f2["The fit's directions keep little of the instruments"]["text"]
    cap_low = dict(cap, ls_share=0.01)
    f3 = _no_path_findings(tr, cap=cap_low, gamma_rank=1, ladder=ladder)
    assert "still in the betas" not in f3["The fit's directions keep little of the instruments"]["text"]
    # rank 0 (every topic dropped): no finding, the dead factor covers it
    f0 = set(_no_path_findings(tr, cap=cap, gamma_rank=0, ladder=ladder))
    assert "The fit's directions keep little of the instruments" not in f0


def test_sigma_z_window_departure_only_when_material(tr):
    """The Sigma_z window is a departure only when the ladder shows it: 0.05 in Spearman or 2 points of R2."""
    def run(ds: float, dr: float) -> dict[str, str]:
        ladder = tr.ladder.copy()
        ladder.loc["instruments_train", ["spearman", "median_r2"]] = [0.70, 0.10]
        ladder.loc["instruments_kernel", ["spearman", "median_r2"]] = [0.70 + ds, 0.10 + dr]
        found = _no_path_findings(tr, history="full", share=0.9, ladder=ladder)
        return found["Instruments and the shocks' covariance cover different days"]

    assert run(0.20, 0.0)["severity"] == "departure"
    assert run(0.01, 0.03)["severity"] == "departure"
    small = run(0.01, 0.005)
    assert small["severity"] == "note" and small["origin"] == "implementation" and "small here" in small["text"]


def test_shock_check_expects_one_sided_days_from_the_data(session):
    """Without asset weighting the panel starts where the lab's shocks start: 0 one-sided days expected."""
    name = "BKS shocks = direct shocks shifted by the lead"
    none = _checks(_traced(session, _generic_cfg(K=1, asset_weighting="none")))[name]
    assert none.status == T.OK and "0 day(s) have a shock on one side only (0 expected" in none.note
    w = _generic_cfg().window.shock_window
    weighted = _checks(_traced(session, _generic_cfg()))[name]
    assert weighted.status == T.OK and f"{w} day(s) have a shock on one side only ({w} expected" in weighted.note


def test_findings_origins_and_forecast_note(tr):
    """Every finding names its origin; the forecast note says the R2 does not measure topic signal."""
    origin = {f["title"]: f["origin"] for f in tr.findings}
    assert origin["Which instruments the implied sensitivities use"] == "implementation"
    assert origin["Instruments and the shocks' covariance cover different days"] == "implementation"
    fc = _titles(tr)["The forecast R2 does not measure topic signal"]
    assert fc["step"] == "forecast" and fc["severity"] == "note" and fc["origin"] == "method"
    assert "shuffled" in fc["text"] and "not topic signal" in fc["text"]
    fake = T.TraceCheck(step="implied", name="X", relation="x = y", value=1.0, reference=0.0, tolerance=0.0,
                        status=T.OFF, note="A note.", kind=T.IDENTITY)
    out = T._findings([fake], cap=tr.capture, K=tr.K, history=tr.history, share=0.0, eff_days=1.0, n_pairs=10,
                      ladder=tr.ladder, path=None, dead=False, dead_ratio=1.0,
                      k_eff=tr.K, ref_corr=1.0, ref_slope=1.0, instrument_week=tr.instrument_week,
                      window_end=tr.instrument_window_end, train_end=tr.meta["train_end"], cal=pd.DatetimeIndex([]),
                      conversion_exact=np.ones(3), conversion_kernel=np.ones(3), scaled=True, zero_obj=1e9,
                      null_q=np.asarray([0.5, 1.0, 2.0]), T_train=20)
    defect = [f for f in out if f["title"] == "Check off: X"]
    assert len(defect) == 1 and defect[0]["severity"] == "defect" and defect[0]["origin"] == "defect"
    assert out[[f["step"] for f in out].index("implied")]["severity"] == "defect"  # defects first within a step


def test_topic_table_and_selection_note(session, cfg, tr):
    """Per topic: the sum of |B_true|, its rank, the links, the selection, the row norm and where it enters."""
    tt = tr.topic_table
    truth = session.truth(cfg)
    Bt = truth.B_true.reindex(index=tr.topics, columns=tr.assets).to_numpy()
    np.testing.assert_allclose(tt["sum_abs_B_true"], np.abs(Bt).sum(axis=1), rtol=1e-12)
    assert sorted(tt["truth_rank"]) == list(range(1, len(tr.topics) + 1))
    assert tt.sort_values("truth_rank")["sum_abs_B_true"].is_monotonic_decreasing
    W = truth.W_unscaled.reindex(index=tr.topics, columns=tr.assets).fillna(0.0).to_numpy()
    np.testing.assert_array_equal(tt["n_links"], (W != 0).sum(axis=1))
    assert list(tt.index[tt["selected"]]) == [t for t in tr.topics if t in tr.meta["selected_topics"]]
    np.testing.assert_allclose(tt["gamma_norm"], np.linalg.norm(tr.gamma_std.loc[tr.topics].to_numpy(), axis=1))
    gp = tr.gamma_path
    for t in tr.topics:
        on = gp[t] > 0
        expected = float(gp.index[on].max()) if on.any() else np.nan
        assert tt.loc[t, "enter_lambda"] == pytest.approx(expected, nan_ok=True)
    assert (tt.loc[tt["selected"], "enter_lambda"] >= tr.lam - 1e-12).all()
    fixed = _traced(session, _generic_cfg(lambda_rule="fixed", lam=0.3))
    assert fixed.topic_table["enter_lambda"].isna().all()
    # the note: fewer than half of the top K' topics by truth selected (K' = number selected, at least 3)
    table = pd.DataFrame({"sum_abs_B_true": [5.0, 4.0, 3.0, 2.0, 1.0], "truth_rank": [1, 2, 3, 4, 5],
                          "n_links": 1, "selected": [False, False, True, True, True], "gamma_norm": 0.1,
                          "enter_lambda": np.nan}, index=pd.Index(["A", "B", "C", "D", "E"], name="topic_id"))
    f = _no_path_findings(tr, topic_table=table)
    note = f["The selected topics are not those with the largest true sensitivities"]
    assert note["severity"] == "note" and note["origin"] == "method" and "selects 1; it drops A, B" in note["text"]
    table["selected"] = [True, True, False, False, True]
    f = set(_no_path_findings(tr, topic_table=table))
    assert "The selected topics are not those with the largest true sensitivities" not in f


def test_page_notation_in_check_texts(tr):
    """Check texts use the page's symbols: v_i for the topic instruments, Gamma's topic rows (not c~, Gamma_tilde)."""
    by = _checks(tr)
    split = by["Implied covariance = projected instrument + constant (consistency)"]
    assert split.relation == "m_i = P v_i + M Gamma_0'"
    texts = [c.relation for c in tr.checks] + [c.note for c in tr.checks] + [f["text"] for f in tr.findings]
    assert not [t for t in texts if "Gamma_tilde" in t or "c~" in t]
    for name in ("Step-by-step chain = production (consistency)", "Kept share <= best K share (consistency)"):
        assert "consistency check" in by[name].note


def test_forecast_end_clipped_to_the_data(session):
    """A forecast window past the data end: meta forecast_end is the last scored day; the settings show both."""
    window = {"train_start": "2023-12-15", "train_end": "2025-12-12", "forecast_start": "2025-12-15",
              "forecast_weeks": 12}
    c = _generic_cfg(window=window)
    trace = _traced(session, c)
    last_day = session.simulation(c).market.calendar[-1]
    span = session.bks(c).meta["evaluated_span"]
    assert trace.meta["forecast_end"] == pd.Timestamp(span[1]) <= last_day
    assert trace.meta["forecast_end_configured"] == pd.Timestamp(c.window.forecast_end) > last_day
    row = trace.meta["settings"].set_index("setting").loc["Forecast window", "value"]
    assert row.startswith(f"2025-12-15 to {pd.Timestamp(c.window.forecast_end).date().isoformat()}")
    assert f"the data end on {last_day.date().isoformat()}" in row
    # inside the data: the configured window as it is, the end clipped only to the last scored week's last day
    inside = _traced(session, _generic_cfg())
    row = inside.meta["settings"].set_index("setting").loc["Forecast window", "value"]
    assert "scored to" not in row
    assert inside.meta["forecast_end"] <= inside.meta["forecast_end_configured"]
    assert inside.meta["forecast_end"] == pd.Timestamp(session.bks(_generic_cfg()).meta["evaluated_span"][1])


def test_small_fixes(tr):
    """The oracle's ladder text, the tuner's tie floor, the page file the docstring names."""
    assert T.LADDER_WHAT["oracle"] == ("The simulation's true topic sensitivities: the Spearman ceiling (1 by "
                                       "construction); on a few forecast weeks another row can score a higher OOS R².")
    assert "without the constant's part" in T.LADDER_WHAT["bks_ls"]
    assert T.TIE_REL_TOL is tuning.TIE_REL_TOL
    assert "dashboard/trace_page.py" in T.__doc__ and (ROOT / "dashboard" / "trace_page.py").is_file()
    assert "bks_trace_page" not in T.__doc__.split("drawn by")[0]


def test_session_raises_bks_not_cached_and_lets_defects_through(monkeypatch):
    """A cache miss is BKSNotCached (still a LookupError); a KeyError inside the trace is not dressed up as one."""
    from narrative_ipca.exposure_lab import session as session_mod

    assert "BKSNotCached" in session_mod.__all__ and issubclass(BKSNotCached, LookupError)
    s = LabSession(max_entries=3)
    c = _generic_cfg(lambda_rule="fixed", lam=0.5)
    with pytest.raises(BKSNotCached, match=re.escape(BKS_NOT_RUN)):
        s.bks_trace(c)
    with pytest.raises(BKSNotCached, match=re.escape(BKS_NOT_RUN)):
        s.bks_implied(c)
    s.bks_fit(c)

    def broken(*args, **kwargs):
        return pd.Series([1.0], index=["a"])["missing_topic"]

    monkeypatch.setattr(T, "build_trace", broken)
    with pytest.raises(KeyError) as info:
        s.bks_trace(c)
    assert not isinstance(info.value, BKSNotCached)


# ---------------------------------------------------------------------------
# dashboard defaults (DESIGN.md G.15.1)
# ---------------------------------------------------------------------------
@needs_market
def test_dashboard_defaults_reproduce_the_diagnosis():
    if str(ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(ROOT / "dashboard"))
    import _ui

    c, errors, _ = _ui.config_from_values(_ui.default_values(), (), reference.load_assets()["asset_class"])
    assert not errors
    s = LabSession()
    s.bks_fit(c)
    trace = s.bks_trace(c)
    assert _identity_off(trace) == []
    assert trace.capture["kept_share"] == pytest.approx(0.379, abs=5e-4)
    assert trace.capture["best_share"] == pytest.approx(0.914, abs=5e-4)
    assert trace.meta["kernel_share_before_train"] == pytest.approx(0.92, abs=0.005)
    sp = trace.ladder["spearman"]
    assert sp["bks_implied"] == pytest.approx(0.19, abs=0.01)
    assert sp["instruments_kernel"] == pytest.approx(0.91, abs=0.01)
    assert sp["instruments_train"] == pytest.approx(0.71, abs=0.01)
    assert sp["best_rank"] == pytest.approx(0.69, abs=0.01)
    assert sp["window_truth"] == pytest.approx(0.78, abs=0.01)
    assert sp["oracle"] == pytest.approx(1.0)
    assert trace.ladder.loc["bks_implied", "median_r2"] == pytest.approx(0.07, abs=0.005)
    titles = {f["title"] for f in trace.findings}
    assert "The fit's directions keep little of the instruments" in titles
    assert "Instruments and the shocks' covariance cover different days" in titles
    assert trace.meta["timings"]["total"] < 5.0
    # SPEC addendum A1-A4: the least-squares inversion, the zero benchmark, the no-signal band, units and days
    assert sp["bks_ls"] == pytest.approx(0.685, abs=0.005)
    assert trace.capture["ls_share"] == pytest.approx(0.89, abs=0.005)
    assert trace.ladder.loc["zero", "rmse"] == pytest.approx(0.0784, abs=5e-4)
    assert trace.ladder.loc["bks_implied", "rmse"] == pytest.approx(0.0914, abs=5e-4)
    assert trace.ladder.loc["zero", "median_r2"] == 0.0
    np.testing.assert_allclose(trace.meta["null_sharpe_quantiles"], [0.87, 2.30, 4.44], atol=0.005)
    ce = trace.units["conversion_exact"]
    assert ce.median() == pytest.approx(0.974, abs=0.001)
    assert ce.min() == pytest.approx(0.514, abs=0.001) and ce.max() == pytest.approx(1.297, abs=0.001)
    assert trace.meta["r2_pooled_exact"] == pytest.approx(0.2305, abs=5e-4)
    assert trace.meta["r2_pooled_panel"] == pytest.approx(0.2777, abs=5e-4)
    assert (trace.meta["bks_train_days"], trace.meta["days_before_train_start"],
            trace.meta["direct_days_not_in_bks"]) == (130, 2, 1)
    assert trace.meta["direct_days_not_in_bks_dates"] == ["2025-06-30"]
    assert trace.sigma_table["train_over_kernel"].median() == pytest.approx(1.02, abs=0.01)
    assert "The lambda choice follows noise" in titles
    assert "The implied sensitivities are further from the truth than zero" in titles
    assert "The unit conversion is approximate" in titles
    # review fixes 2026-09-30: the polish is cheap, the stored residual is what it was, no text prints nan
    pol = trace.meta["polish"]
    assert pol["seconds"] < 0.5 and pol["stored_ok"] and pol["kkt_stored"] == pytest.approx(8.2e-4, abs=5e-5)
    assert pol["kkt_polished"] < 1e-4
    _assert_no_nan_text(trace)
    origin = {f["title"]: (f["severity"], f["origin"]) for f in trace.findings}
    assert origin["The fit's directions keep little of the instruments"] == ("departure", "method")
    assert origin["Instruments and the shocks' covariance cover different days"] == ("departure", "implementation")
    assert origin["The lambda choice follows noise"] == ("departure", "method")
    fc = next(f for f in trace.findings if f["title"] == "The forecast R2 does not measure topic signal")
    assert "27.8%" in fc["text"] and "9.1%" in fc["text"]
    tt = trace.topic_table
    assert tt.loc["A4", "truth_rank"] == 2 and tt.loc["S2", "truth_rank"] == 3
    assert tt.loc[["A4", "S2"], "enter_lambda"].isna().all() and not tt.loc[["A4", "S2"], "selected"].any()
    # one factor: the tuner picks lambda_max, whose fit is worse than no fit (a departure, not a note)
    k1 = dataclasses.replace(c, bks=dataclasses.replace(c.bks, K=1))
    s.bks_fit(k1)
    t1 = s.bks_trace(k1)
    assert t1.meta["chosen_above_zero"] and _identity_off(t1) == []
    worse = next(f for f in t1.findings if f["title"] == "The chosen fit is worse than no fit")
    assert worse["severity"] == "departure" and "4,820.0" in worse["text"] and "4,691.6" in worse["text"]
    assert next(ch for ch in t1.checks if ch.name == "Chosen fit beats Gamma = 0").status == T.OFF
    train = dataclasses.replace(c, bks=dataclasses.replace(c.bks, history="training"))
    s.bks_fit(train)
    t2 = s.bks_trace(train)
    assert _identity_off(t2) == []
    assert t2.meta["kernel_share_before_train"] == 0.0
    assert t2.ladder.loc["bks_implied", "spearman"] == pytest.approx(0.11, abs=0.01)
    assert (t2.meta["bks_train_days"], t2.meta["direct_train_days"]) == (120, 129)
    np.testing.assert_allclose(t2.units["conversion_exact"], 1.0, atol=1e-10)
    assert "The choice sits at the edge of the lambda grid" in {f["title"] for f in t2.findings}
    # every topic is kept under the training history: the KKT note says so, and nothing prints nan
    assert len(t2.meta["selected_topics"]) == len(t2.topics)
    kkt2 = next(ch for ch in t2.checks if ch.name == "Stationarity of the group lasso (KKT)")
    assert "no topic is dropped" in kkt2.note
    _assert_no_nan_text(t2)
