"""Tests of the BKS trace (DESIGN.md G.16; D90): narrative_ipca/exposure_lab/trace.py and LabSession.bks_trace.

A small generic simulation (30 artificial assets, 12 generic topics of which
6 carry links, a two-year training window) is fitted under both covariance
histories, both leads, the tuned lambda and the fixed lambda 0; every
identity check of the trace must hold on each. The per-selection helpers
are checked against the panel they explain. The dashboard defaults (55
listed assets, 20 topics) reproduce the diagnosis numbers of DESIGN.md
G.15.1; that test is skipped when ``data/market`` is missing.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
from narrative_ipca.exposure_lab.session import BKS_NOT_RUN, BKS_STAGES, SESSION_STAGES, LabSession

ROOT = Path(__file__).resolve().parents[1]
HAS_MARKET = (reference.market_dir() / "asset_returns.parquet").is_file()
needs_market = pytest.mark.skipif(not HAS_MARKET, reason="data/market/asset_returns.parquet is missing")

#: Two training years keep the fits fast (about 100 training weeks).
WINDOW = {"train_start": "2021-01-01", "train_end": "2022-12-30", "forecast_start": "2023-01-02", "forecast_weeks": 4}


def _generic_cfg(lead: int = 0, **bks_kw: object) -> LabConfig:
    """30 generic assets, 12 generic topics (6 linked), beta_1 = 0.5 (seed 0), a two-year training window."""
    return LabConfig(
        universe=UniverseConfig(asset_source="generic", n_generic_assets=30, seed=0),
        topics=TopicSetConfig(manual="none", n_generic=12, generic_signal_share=0.5),
        exposure=ExposureConfig(beta_1=0.5, lead_days=lead, seed=0),
        window=WindowConfig(**WINDOW),
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
        assert set(f) == {"step", "severity", "title", "text"} and f["step"] in T.STEPS
        assert f["severity"] in ("departure", "note") and f["text"]
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
    assert tr.gamma_std.shape == (L + 1, tr.K) and tr.kkt.shape == (L + 1, 3)
    assert list(tr.gamma_std.index) == ["const"] + tr.panel_topics
    assert tr.factors_in_sample.shape == (len(tr.train_periods), tr.K)
    assert tr.z.shape[1] == L and list(tr.z.columns) == tr.panel_topics
    assert tr.instruments.shape == (N, L) and tr.population["instrument_ref"].shape == (N, L)
    assert tr.path is not None and list(tr.path.columns) == [
        "lam", "criterion", "se", "in_band", "best", "chosen", "n_selected", "total_r2", "objective", "zero_objective",
        "above_zero", "converged", "n_iter", "sigma_ff_truncated", "null_q05", "null_q50", "null_q95",
        "band_floor_code", "band_floor_relative", "edge"]
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
def test_least_squares_rung_recomputes_independently(session, cfg, tr):
    """bks_ls: m_i = Sigma_c Gt (Gt' Sigma_c Gt)^+ beta_i' with Sigma_c = C'C / n over the Eq. 5 rows."""
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
    m_ls = beta @ M.T
    np.testing.assert_allclose(tr.chain["m_ls"].to_numpy()[has], m_ls, rtol=1e-9, atol=1e-14)
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


def test_path_band_floors_edge_and_null_columns(cfg, tr):
    path = tr.path
    crit = path["criterion"].to_numpy()
    best = float(np.nanmax(crit))
    tol = cfg.bks.tolerance
    assert (path["band_floor_code"] == best - max(1e-9, tol) * max(1.0, abs(best))).all()
    assert (path["band_floor_relative"] == best - tol * abs(best)).all()
    np.testing.assert_array_equal(path["in_band"].to_numpy(), crit >= path["band_floor_code"].to_numpy())
    np.testing.assert_allclose(path[["null_q05", "null_q50", "null_q95"]].iloc[0].to_numpy(),
                               T.null_sharpe_quantiles(tr.K, len(tr.train_periods)))
    assert path[["null_q05", "null_q50", "null_q95"]].nunique().eq(1).all()
    ends = {int(path["lam"].idxmin()), int(path["lam"].idxmax())}
    expected = [(i in ends) and bool(path["chosen"].iloc[i] or path["best"].iloc[i]) for i in range(len(path))]
    assert path["edge"].tolist() == expected
    assert tr.meta["null_sharpe_quantiles"] == tuple(path[["null_q05", "null_q50", "null_q95"]].iloc[0])
    assert tr.meta["band_floor_code"] == path["band_floor_code"].iloc[0]
    # gamma_sv_min: the smallest singular value of the standardised Gamma_tilde; zero exactly when a direction died
    pt = tr.path_trace
    assert (pt["gamma_sv_min"] >= 0.0).all()
    assert ((pt["gamma_sv_min"] > 1e-8) | (pt["gamma_rank"] < tr.K)).all()


def _findings_with_path(tr: T.BKSTrace, path: pd.DataFrame, *, rule: str = "tolerance", tolerance: float = 0.02,
                        null_q=(0.5, 1.0, 2.0), floor_code: float, floor_rel: float) -> list[dict[str, str]]:
    return T._findings(
        [], cap=tr.capture, K=tr.K, history=tr.history, share=0.0, eff_days=1.0, n_pairs=10, ladder=tr.ladder,
        path=path, rule=rule, tolerance=tolerance, dead=False, dead_ratio=1.0, k_eff=tr.K, ref_corr=1.0,
        ref_slope=1.0, instrument_week=tr.instrument_week, window_end=tr.instrument_window_end,
        train_end=tr.meta["train_end"], cal=pd.DatetimeIndex([]), conversion_exact=np.ones(3),
        conversion_kernel=np.ones(3), scaled=True, zero_obj=1e9, null_q=np.asarray(null_q), T_train=20,
        floor_code=floor_code, floor_rel=floor_rel)


def test_findings_null_band_grid_edge_and_band_rules(tr):
    lam = np.logspace(-2, 0, 5)
    crit = np.array([0.80, 0.79, 0.785, 0.70, 0.60])  # best at the smallest lambda, below 1
    best, tol = 0.80, 0.02
    floor_code, floor_rel = best - tol * 1.0, best - tol * best  # 0.78 and 0.784: different picks
    path = pd.DataFrame({"lam": lam, "criterion": crit, "se": 1.0, "in_band": crit >= floor_code,
                         "best": [True, False, False, False, False], "chosen": [False, False, True, False, False],
                         "n_selected": [9, 8, 7, 5, 2], "above_zero": False})
    titles = {f["title"]: f for f in _findings_with_path(tr, path, floor_code=floor_code, floor_rel=floor_rel)}
    # (a) every point inside the no-signal 5-95% band replaces "within noise"
    assert "The lambda choice follows noise" in titles and "The lambda choice is within noise" not in titles
    assert "0.50-2.00" in titles["The lambda choice follows noise"]["text"]
    assert titles["The lambda choice follows noise"]["severity"] == "departure"
    # (b) the best point is the grid's first point
    edge = titles["The choice sits at the edge of the lambda grid"]
    assert edge["severity"] == "departure" and "best in-sample Sharpe ratio is at the smallest lambda" in edge["text"]
    assert "chosen lambda" not in edge["text"]
    # (c) not here: the tuner's band (floor 0.78) and the relative band (0.784) both pick the third point
    assert "The two readings of the tolerance band choose differently" not in titles
    # outside the no-signal band: the standard-error rule applies again; argmax: no band note
    far = _findings_with_path(tr, path, null_q=(0.9, 1.0, 2.0), floor_code=floor_code, floor_rel=floor_rel,
                              rule="argmax", tolerance=0.0)
    far_titles = {f["title"] for f in far}
    assert "The lambda choice is within noise" in far_titles and "The lambda choice follows noise" not in far_titles
    assert "The two readings of the tolerance band choose differently" not in far_titles


def test_findings_band_rules_differ_only_when_picks_differ(tr):
    lam = np.logspace(-2, 0, 4)
    crit = np.array([0.50, 0.49, 0.482, 0.40])
    best, tol = 0.50, 0.02
    floor_code, floor_rel = best - tol, best - tol * best  # 0.48 (picks index 2) and 0.49 (picks index 1)
    path = pd.DataFrame({"lam": lam, "criterion": crit, "se": 1.0, "in_band": crit >= floor_code,
                         "best": [True, False, False, False], "chosen": [False, False, True, False],
                         "n_selected": [9, 8, 7, 5], "above_zero": False})
    found = {f["title"]: f for f in _findings_with_path(tr, path, floor_code=floor_code, floor_rel=floor_rel)}
    band = found["The two readings of the tolerance band choose differently"]
    assert band["severity"] == "note" and "0.480" in band["text"] and "0.490" in band["text"]
    same = {f["title"] for f in _findings_with_path(tr, path, floor_code=floor_code, floor_rel=floor_code)}
    assert "The two readings of the tolerance band choose differently" not in same


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
    train = dataclasses.replace(c, bks=dataclasses.replace(c.bks, history="training"))
    s.bks_fit(train)
    t2 = s.bks_trace(train)
    assert _identity_off(t2) == []
    assert t2.meta["kernel_share_before_train"] == 0.0
    assert t2.ladder.loc["bks_implied", "spearman"] == pytest.approx(0.11, abs=0.01)
    assert (t2.meta["bks_train_days"], t2.meta["direct_train_days"]) == (120, 129)
    np.testing.assert_allclose(t2.units["conversion_exact"], 1.0, atol=1e-10)
    assert "The choice sits at the edge of the lambda grid" in {f["title"] for f in t2.findings}
