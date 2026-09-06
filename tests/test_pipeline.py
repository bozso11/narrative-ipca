"""Tests of the orchestration layer (pipeline.py, cli.py, configs/).

The stage mathematics is tested per module; here the checks are about the
*composition*: the pipeline reproduces the stage functions exactly (no hidden
state, D42), the evaluation numbers are the identities they claim to be,
observables/test assets are re-stamped correctly on the period grid, artefacts
round-trip through files, guarded steps fail softly, and the CLI's file
round trip reproduces the in-process numbers.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import narrative_ipca
from narrative_ipca import cli, pipeline
from narrative_ipca.config import (
    CovarianceConfig,
    EstimationConfig,
    EvaluationConfig,
    HarnessConfig,
    LambdaGridConfig,
    OOSConfig,
    PipelineConfig,
    SimulationConfig,
    load_config,
    save_config,
)
from narrative_ipca.covariances import build_covariance_panel
from narrative_ipca.data import align_inputs, period_returns
from narrative_ipca.evaluation import price_test_assets, realized_sharpe, total_r2
from narrative_ipca.panel import build_panel
from narrative_ipca.shocks import attention_shocks
from narrative_ipca.simulation import simulate
from narrative_ipca.tuning import tune
from narrative_ipca.types import AttentionData, PipelineResult, ReturnsData

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# fixtures: one fast end-to-end run shared by most tests
# ---------------------------------------------------------------------------
SIM_CFG = SimulationConfig(seed=0, n_assets=120, n_topics=24, n_relevant=6, n_placebo=6, n_years=6)


def fast_cfg(**overrides) -> PipelineConfig:
    cfg = PipelineConfig(
        covariance=CovarianceConfig(burn_in_periods=6),
        oos=OOSConfig(min_train_periods=24, oos_fraction=0.35, refit_every=12),
        # ratio 0.01: lam_max is the exact KKT threshold with the penalised intercept (D19/D22), on this
        # panel ~2.5x the intercept-only least-squares value; the Sharpe peak (5-6 narratives) sits near
        # 0.03 lam_max, so a 0.05 bottom would truncate the path at 2 selected narratives.
        estimation=EstimationConfig(lam_grid=LambdaGridConfig(n_lambdas=8, ratio=0.01)),
        name="pipeline-test",
    )
    return replace(cfg, **overrides) if overrides else cfg


@pytest.fixture(scope="module")
def sim():
    return simulate(SIM_CFG)


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return fast_cfg()


@pytest.fixture(scope="module")
def observables(sim) -> dict[str, pd.Series]:
    """The true first factor, stamped at calendar month ends (not last trading days)."""
    f1 = sim.truth.f_period["f1"]
    month_end = f1.index.to_period("M").to_timestamp(how="end").normalize()
    return {"f1_true": pd.Series(f1.to_numpy(), index=month_end, name="f1_true")}


@pytest.fixture(scope="module")
def test_assets(sim) -> pd.DataFrame:
    """Six simulated assets as a *daily* return panel (the pipeline must accumulate them)."""
    return sim.returns.returns.iloc[:, :6].copy()


@pytest.fixture(scope="module")
def run(sim, cfg, observables, test_assets):
    calls: list[tuple[int, int, str]] = []
    t0 = time.perf_counter()
    res = pipeline.run_pipeline(
        sim.attention, sim.returns, cfg, test_assets=test_assets, observables=observables,
        progress=lambda d, t, m: calls.append((d, t, m)),
    )
    elapsed = time.perf_counter() - t0
    return res, calls, elapsed


@pytest.fixture(scope="module")
def result(run) -> PipelineResult:
    return run[0]


def _nan_equal(a, b, rtol=1e-12) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    try:
        return bool(np.isclose(float(a), float(b), rtol=rtol, atol=0.0))
    except (TypeError, ValueError):
        return a == b


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def test_end_to_end_runs_fast_with_finite_metrics(run, cfg):
    res, _, elapsed = run
    assert elapsed < 60.0, f"fast pipeline took {elapsed:.1f}s"
    m = res.evaluation.metrics
    for key in ("total_r2", "pred_r2", "mve_sharpe_is", "oos_sharpe", "lam_star", "lam_max", "objective"):
        assert key in m and np.isfinite(m[key]), key
    assert m["K"] == cfg.estimation.K
    assert res.fit.n_selected >= cfg.estimation.K
    assert m["n_selected"] == res.fit.n_selected
    assert res.fit.converged
    assert res.wrapup is not None and res.oos is not None
    assert set(res.timings) == set(pipeline.STEPS) | {"total"}
    assert all(v >= 0.0 for v in res.timings.values())
    assert res.timings["total"] >= sum(res.timings[s] for s in pipeline.STEPS) - 1e-6
    assert res.timings["placebo"] == 0.0 and res.meta["steps"]["placebo"] == "skipped"
    for step in ("align", "shocks", "covariances", "panel", "tune", "wrapup", "oos", "evaluate"):
        assert res.meta["steps"][step] == "ok", step
    assert res.meta["config_hash"] == cfg.hash()
    assert res.meta["package_version"] == narrative_ipca.__version__
    shapes = res.meta["shapes"]
    assert shapes["panel"]["n_obs"] == res.panel.n_obs and shapes["covariances"]["L"] == 24
    assert res.meta["period_alias"] == "M" and res.fit.meta["period"] == "M"
    assert res.config is cfg
    # lambda* is a grid point of the tuned K and lies inside [ratio lam_max, lam_max]
    lams = res.tuning.meta["lam_grid"][res.tuning.K]
    assert np.any(np.isclose(res.tuning.lam, lams))
    assert cfg.estimation.lam_grid.ratio * res.tuning.lam_max * (1 - 1e-9) <= res.tuning.lam <= res.tuning.lam_max * (1 + 1e-9)


def test_progress_callback_counts_steps(run):
    _, calls, _ = run
    n = len(pipeline.STEPS)
    assert calls, "progress never called"
    assert all(t == n for _, t, _ in calls)
    done = [d for d, _, _ in calls]
    assert done == sorted(done), "step counter must be non-decreasing"
    assert calls[-1] == (n, n, "done")
    assert any(m.startswith("tune [") for _, _, m in calls) and any(m.startswith("oos [") for _, _, m in calls)


def test_pipeline_reproduces_the_stage_functions(sim, cfg, result):
    """No hidden state (D42): the stages called by hand give the same panel and the same fit."""
    aligned = align_inputs(sim.attention, sim.returns, cfg.data)
    shocks = attention_shocks(aligned.attention, cfg.shocks)
    cov = build_covariance_panel(shocks, aligned.returns, cfg.covariance, cfg.data.period, dtype=cfg.data.dtype)
    pnl = build_panel(cov, aligned.returns, cfg.data, cfg.covariance)
    pd.testing.assert_frame_equal(result.shocks.z, shocks.z)
    np.testing.assert_array_equal(result.covariances.values, cov.values)
    assert result.covariances.periods.equals(cov.periods)
    np.testing.assert_array_equal(result.panel.X, pnl.X)
    np.testing.assert_array_equal(result.panel.y, pnl.y)
    np.testing.assert_array_equal(result.panel.t_idx, pnl.t_idx)
    np.testing.assert_array_equal(result.panel.asset_idx, pnl.asset_idx)
    np.testing.assert_array_equal(result.panel.sigma_c, pnl.sigma_c)
    assert result.panel.periods.equals(pnl.periods)
    tr = tune(pnl, cfg.estimation, cfg.tuning, cfg.evaluation)
    assert tr.lam == result.tuning.lam and tr.K == result.tuning.K
    np.testing.assert_allclose(tr.fit.Gamma, result.fit.Gamma, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(tr.fit.F, result.fit.F, rtol=1e-10, atol=1e-14)
    assert tr.fit.objective == pytest.approx(result.fit.objective, rel=1e-12)
    assert [p.lam for p in tr.path] == [p.lam for p in result.tuning.path]
    assert result.tuning.fit is result.fit  # the same (period-tagged) fit object in both places


def test_evaluation_identities(result, cfg):
    """The reported numbers are what they claim: R2 on the panel, Sharpe identities, wrap-up algebra."""
    m = result.evaluation.metrics
    ann, rcond = cfg.evaluation.annualization, cfg.evaluation.rcond
    assert m["total_r2"] == pytest.approx(total_r2(result.panel, result.fit.Gamma, result.fit.F), rel=1e-12)
    assert m["total_r2"] == pytest.approx(result.fit.total_r2, rel=1e-8)
    assert m["mve_sharpe_is"] == pytest.approx(result.fit.mve_sharpe(ann, rcond), rel=1e-12)
    assert m["oos_sharpe"] == pytest.approx(realized_sharpe(result.oos.mve, ann), rel=1e-12)
    assert m["oos_sharpe"] == pytest.approx(result.oos.sharpe, rel=1e-12)
    assert m["lam_star"] == result.tuning.lam and m["lam_max"] == result.tuning.lam_max
    assert m["n_path_points"] == len(result.tuning.path) == cfg.estimation.lam_grid.n_lambdas
    # wrap-up used this fit: Eq. 11 and the A <-> Gamma_tilde round trip with this fit's Sigma_ff
    w = result.wrapup
    np.testing.assert_allclose(w.impact_z_to_mve.to_numpy(), w.impact_z_to_x.to_numpy() @ w.b_mve, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(w.b_mve, result.fit.b_mve(rcond), rtol=1e-12)
    if not w.rank_deficient:
        A = w.A.to_numpy()
        round_trip = A @ np.linalg.pinv(A.T @ A) @ np.linalg.pinv(result.fit.Sigma_ff)
        np.testing.assert_allclose(round_trip, result.fit.Gamma_tilde, rtol=1e-6, atol=1e-10)
    assert m["wrapup_rank_deficient"] == int(w.rank_deficient)
    # states live on the shock calendar and are NaN exactly where a shock is missing
    assert w.states.index.equals(result.shocks.z.index)
    missing = result.shocks.z.isna().any(axis=1).to_numpy()
    assert np.array_equal(w.states.isna().any(axis=1).to_numpy(), missing)


# ---------------------------------------------------------------------------
# observables and test assets: re-stamping on the period grid
# ---------------------------------------------------------------------------
def test_observable_restamped_by_period_label_and_projected(sim, result, observables):
    periods = result.panel.periods
    al = pipeline.align_period_frame(observables["f1_true"], periods, "M")
    assert al.index.equals(periods) and list(al.columns) == ["f1_true"]
    truth = sim.truth.f_period["f1"]
    expected = truth.set_axis(truth.index.to_period("M")).reindex(periods.to_period("M")).to_numpy()
    np.testing.assert_allclose(al["f1_true"].to_numpy(), expected, rtol=1e-12)
    assert np.isfinite(expected).all()
    # projection of the true factor on the estimated factors (BKS Section 6.1) is tight on this DGP
    proj = result.wrapup.obs_projection["f1_true"]
    assert proj["n_obs"] == int(np.asarray(result.fit.populated).sum())
    assert 0.5 < proj["r2"] <= 1.0
    assert result.evaluation.metrics["obs_r2_f1_true"] == pytest.approx(proj["r2"])
    np.testing.assert_allclose(
        result.wrapup.impact_z_to_obs["f1_true"].to_numpy(),
        result.wrapup.impact_z_to_x.to_numpy() @ proj["b_obs"], rtol=1e-10, atol=1e-14,
    )
    # the observable doubles as a benchmark factor: correlations with the in-sample factors and MVE
    corr = result.evaluation.factor_correlations
    assert corr is not None and list(corr.columns) == ["f1_true"] and "mve" in corr.index
    assert corr["f1_true"].abs().drop("mve").max() > 0.8


def test_daily_test_assets_are_accumulated_and_priced(sim, result, test_assets, cfg):
    periods = result.panel.periods
    al = pipeline.align_period_frame(test_assets, periods, "M", how="sum", what="test_assets")
    expected = period_returns(test_assets, "M", "sum").reindex(periods)
    pd.testing.assert_frame_equal(al, expected.set_axis(al.columns, axis=1), check_names=False)
    tests = result.evaluation.pricing_tests
    assert set(tests) == {"narrative_is", "narrative_oos", "benchmark"}
    pt = tests["narrative_is"]
    assert pt.betas.shape == (6, result.fit.K) and np.isfinite(pt.avg_abs_alpha)
    populated = np.asarray(result.fit.populated, dtype=bool)
    F_is = result.fit.factors_frame().loc[populated]
    ref = price_test_assets(al, F_is, cfg.evaluation.t_crit, "narrative_is")
    pd.testing.assert_series_equal(pt.alphas, ref.alphas)
    pd.testing.assert_series_equal(pt.t_stats, ref.t_stats)
    assert result.evaluation.metrics["pricing_narrative_is_avg_abs_alpha"] == pytest.approx(pt.avg_abs_alpha)
    assert "pricing" in result.evaluation.tables and set(result.evaluation.tables["pricing"].index) == set(tests)
    assert result.meta["n_test_assets"] == 6 and result.meta["n_observables"] == 1


def test_align_period_frame_edge_cases(result, caplog):
    periods = result.panel.periods
    with caplog.at_level(logging.WARNING, logger="narrative_ipca.pipeline"):
        far = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["1990-01-31", "1990-02-28"]), name="x")
        al = pipeline.align_period_frame(far, periods, "M")
    assert al.isna().all().all() and "shares no period" in caplog.text
    with pytest.raises(ValueError):
        pipeline.align_period_frame(pd.Series([1.0, 2.0], index=["a", "b"]), periods, "M")
    with pytest.raises(ValueError):
        pipeline.align_period_frame(pd.DataFrame(index=periods), periods, "M")
    # "ME" (an offset alias pandas rejects for Period objects) is accepted like "M"
    s = pd.Series(np.arange(len(periods), dtype=float), index=periods, name="s")
    pd.testing.assert_frame_equal(pipeline.align_period_frame(s, periods, "ME"), pipeline.align_period_frame(s, periods, "M"))


# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------
REQUIRED_FILES = {
    "config": "config.json", "metrics": "metrics.json", "gamma": "gamma.csv", "factors": "factors.csv",
    "lambda_path": "lambda_path.csv", "selected": "selected.csv", "states": "states.csv",
    "impact_z_to_mve": "impact_z_to_mve.csv", "A": "A.csv", "oos_factors": "oos_factors.csv",
    "oos_mve": "oos_mve.csv", "oos_history": "oos_history.csv", "timings": "timings.json",
}


def test_save_result_writes_manifest_and_round_trips(result, cfg, tmp_path):
    out = tmp_path / "artefacts"
    manifest = pipeline.save_result(result, out)
    for key, name in REQUIRED_FILES.items():
        assert key in manifest and Path(manifest[key]).name == name and Path(manifest[key]).is_file(), key
    assert "placebo" not in manifest and "covariances" not in manifest and "panel" not in manifest
    listed = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert listed["gamma"] == "gamma.csv" and set(listed) == set(manifest) - {"manifest"}
    # config.json is loadable and equal to the config that produced the run
    assert load_config(manifest["config"], PipelineConfig) == cfg
    # metrics.json equals the in-memory metrics (NaN-aware)
    metrics = json.loads(Path(manifest["metrics"]).read_text(encoding="utf-8"))
    assert set(metrics) == set(result.evaluation.metrics)
    assert all(_nan_equal(metrics[k], v) for k, v in result.evaluation.metrics.items())
    # numeric tables round-trip to full precision
    gamma = pd.read_csv(manifest["gamma"], index_col=0)
    assert list(gamma.index) == list(result.fit.instrument_names)
    np.testing.assert_allclose(gamma.to_numpy(), result.fit.Gamma, rtol=1e-12)
    factors = pd.read_csv(manifest["factors"], index_col=0, parse_dates=True)
    np.testing.assert_allclose(factors[[f"f{k+1}" for k in range(result.fit.K)]].to_numpy(), result.fit.F, rtol=1e-12)
    assert factors.index.equals(result.fit.periods) and factors["populated"].dtype == bool
    path = pd.read_csv(manifest["lambda_path"], index_col=0)
    assert len(path) == len(result.tuning.path) and np.allclose(path["lam"], [p.lam for p in result.tuning.path])
    norm_path = pd.read_csv(manifest["gamma_norm_path"], index_col=0)
    assert norm_path.shape == (len(result.tuning.path), 2 + result.panel.p)
    np.testing.assert_allclose(norm_path.iloc[-1, 2:].to_numpy(), result.tuning.path[-1].gamma_norms, rtol=1e-12)
    selected = pd.read_csv(manifest["selected"])
    assert list(selected["topic"]) == list(result.evaluation.tables["selected"]["topic"]) and "label" in selected.columns
    assert set(selected["topic"]) == set(result.fit.selected_topics)
    states = pd.read_csv(manifest["states"], index_col=0, parse_dates=True)
    assert states.shape == (len(result.wrapup.states), result.fit.K + 1) and "x_mve" in states.columns
    impact = pd.read_csv(manifest["impact_z_to_mve"], index_col=0)
    np.testing.assert_allclose(impact.iloc[:, 0].to_numpy(), result.wrapup.impact_z_to_mve.to_numpy(), rtol=1e-12)
    A = pd.read_csv(manifest["A"], index_col=0)
    np.testing.assert_allclose(A.to_numpy(), result.wrapup.A.to_numpy(), rtol=1e-12)
    mve = pd.read_csv(manifest["oos_mve"], index_col=0, parse_dates=True)
    np.testing.assert_allclose(mve["mve"].to_numpy(), result.oos.mve.to_numpy(), rtol=1e-12)
    assert mve.index.equals(result.oos.mve.index)
    hist = pd.read_csv(manifest["oos_history"], index_col=0, parse_dates=True)
    assert len(hist) == len(result.oos.refit_periods) and {"lam", "K", "n_selected", "is_sharpe", "lam_max"} <= set(hist.columns)
    timings = json.loads(Path(manifest["timings"]).read_text(encoding="utf-8"))
    assert set(timings) == set(result.timings)
    meta = json.loads(Path(manifest["meta"]).read_text(encoding="utf-8"))
    assert meta["config_hash"] == cfg.hash() and "asset_meta" not in meta and meta["steps"]["oos"] == "ok"
    assert Path(manifest["asset_meta"]).is_file() and Path(manifest["pricing_summary"]).is_file()
    assert Path(manifest["factor_correlations"]).is_file() and Path(manifest["obs_projection"]).is_file()


def test_save_panel_writes_npz_and_parquet(result, cfg, tmp_path):
    res = replace(result, config=replace(cfg, save_panel=True))
    manifest = pipeline.save_result(res, tmp_path / "with_panel")
    with np.load(manifest["covariances"], allow_pickle=False) as npz:
        np.testing.assert_array_equal(npz["values"], np.asarray(result.covariances.values))
        assert pd.DatetimeIndex(npz["periods"]).equals(result.covariances.periods)
        assert list(npz["topics"]) == [str(t) for t in result.covariances.topics]
        assert float(npz["xi"]) == result.covariances.xi
    panel_path = Path(manifest["panel"])
    long = pd.read_parquet(panel_path) if panel_path.suffix == ".parquet" else pd.read_csv(panel_path, parse_dates=["period"])
    assert len(long) == result.panel.n_obs
    np.testing.assert_allclose(long[list(result.panel.instrument_names)].to_numpy(), result.panel.X, rtol=1e-12)
    np.testing.assert_allclose(long["y"].to_numpy(), result.panel.y, rtol=1e-12)
    assert list(long["asset"]) == [str(a) for a in np.asarray(result.panel.assets)[result.panel.asset_idx]]


def test_json_default_handles_numpy_and_pandas(tmp_path):
    obj = {
        "i": np.int64(3), "f": np.float32(1.5), "arr": np.arange(3), "ts": pd.Timestamp("2020-01-31"),
        "s": pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2020-01-31", "2020-02-29"])),
        "cfg": EvaluationConfig(), "nan": float("nan"), "path": Path("x"),
    }
    text = json.dumps(obj, default=pipeline._json_default)
    back = json.loads(text)
    assert back["i"] == 3 and back["f"] == 1.5 and back["arr"] == [0, 1, 2]
    assert back["ts"].startswith("2020-01-31") and back["s"] == {"2020-01-31 00:00:00": 1.0, "2020-02-29 00:00:00": 2.0}
    assert back["cfg"]["annualization"] == 12.0 and math.isnan(back["nan"]) and back["path"] == "x"


# ---------------------------------------------------------------------------
# switches and guarded steps
# ---------------------------------------------------------------------------
def test_wrapup_and_oos_switched_off(sim, cfg, tmp_path):
    cfg2 = replace(cfg, run_wrapup=False, oos=replace(cfg.oos, enabled=False), name="no-extras")
    res = pipeline.run_pipeline(sim.attention, sim.returns, cfg2)
    assert res.wrapup is None and res.oos is None
    assert res.meta["steps"]["wrapup"] == "skipped" and res.meta["steps"]["oos"] == "skipped"
    assert res.timings["wrapup"] == 0.0 and res.timings["oos"] == 0.0
    assert "oos_sharpe" not in res.evaluation.metrics and "wrapup_rank_deficient" not in res.evaluation.metrics
    manifest = pipeline.save_result(res, tmp_path)
    assert {"config", "metrics", "gamma", "factors", "lambda_path", "selected", "timings"} <= set(manifest)
    assert not any(k.startswith("oos") for k in manifest) and "states" not in manifest and "A" not in manifest


def test_placebo_test_and_rank_deficient_wrapup(sim, cfg, tmp_path, caplog):
    """A coarse grid keeps every path point in the 2-narrative regime (K=3): the wrap-up is rank
    deficient and must not raise; the placebo test (App. C.2) runs on top."""
    cfg3 = replace(
        cfg,
        estimation=EstimationConfig(lam_grid=LambdaGridConfig(n_lambdas=8, ratio=0.1)),
        evaluation=EvaluationConfig(placebo_n=3, placebo_seed=7),
        oos=replace(cfg.oos, enabled=False),
        name="placebo",
    )
    with caplog.at_level(logging.WARNING, logger="narrative_ipca.pipeline"):
        res = pipeline.run_pipeline(sim.attention, sim.returns, cfg3)
    assert res.wrapup is not None
    assert res.wrapup.rank_deficient == (res.fit.n_selected < res.fit.K)
    if res.wrapup.rank_deficient:
        assert "rank deficient" in caplog.text
        assert res.evaluation.metrics["wrapup_rank_deficient"] == 1
    assert res.meta["steps"]["wrapup"] == "ok" and res.meta["steps"]["placebo"] == "ok"
    pl = res.evaluation.placebo
    assert pl is not None and pl.n_placebo == 3 and 0 <= pl.n_placebo_selected <= 3
    assert pl.real_selected_before == res.fit.selected_topics
    assert set(pl.lam_max_by_instrument.index) >= {"placebo_1", "placebo_2", "placebo_3"}
    assert res.evaluation.metrics["placebo_n"] == 3 and res.evaluation.metrics["placebo_n_selected"] == pl.n_placebo_selected
    assert res.timings["placebo"] > 0.0
    manifest = pipeline.save_result(res, tmp_path)
    placebo = json.loads(Path(manifest["placebo"]).read_text(encoding="utf-8"))
    assert placebo["n_placebo"] == 3 and placebo["lam_star"] == res.evaluation.placebo.lam_star
    assert set(placebo["lam_max_by_instrument"]) == set(pl.lam_max_by_instrument.index)


def test_wrapup_failure_is_guarded(sim, cfg, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise ValueError("synthetic wrap-up failure")

    monkeypatch.setattr(pipeline._wrapup, "wrap_up", boom)
    cfg4 = replace(cfg, oos=replace(cfg.oos, enabled=False))
    with caplog.at_level(logging.WARNING, logger="narrative_ipca.pipeline"):
        res = pipeline.run_pipeline(sim.attention, sim.returns, cfg4)
    assert res.wrapup is None and res.meta["steps"]["wrapup"] == "failed"
    assert "synthetic wrap-up failure" in res.meta["wrapup_error"] and "wrapup failed" in caplog.text
    assert np.isfinite(res.evaluation.metrics["total_r2"]) and res.fit.n_selected >= 1


def test_run_pipeline_rejects_wrong_config(sim):
    with pytest.raises(TypeError):
        pipeline.run_pipeline(sim.attention, sim.returns, {"data": {}})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def test_load_inputs_round_trip(sim, tmp_path):
    att = sim.attention.levels
    ret = sim.returns.returns
    att.rename_axis("date").to_parquet(tmp_path / "attention.parquet")
    att.rename_axis("date").to_csv(tmp_path / "attention.csv")
    ret.rename_axis("date").to_csv(tmp_path / "returns.csv")
    sim.returns.asset_meta.rename_axis("asset").to_csv(tmp_path / "meta.csv")
    rf = pd.Series(0.0001, index=ret.index, name="rf")
    rf.rename_axis("date").to_csv(tmp_path / "rf.csv")
    for att_name in ("attention.parquet", "attention.csv"):
        a, r = pipeline.load_inputs(tmp_path / att_name, tmp_path / "returns.csv", tmp_path / "meta.csv", tmp_path / "rf.csv")
        assert isinstance(a, AttentionData) and isinstance(r, ReturnsData)
        assert isinstance(a.levels.index, pd.DatetimeIndex) and a.levels.index.equals(att.index)
        np.testing.assert_allclose(a.levels.to_numpy(), att.to_numpy(), rtol=1e-12)
        assert list(a.levels.columns) == list(att.columns)
        assert r.returns.index.equals(ret.index)
        np.testing.assert_allclose(r.returns.to_numpy(), ret.to_numpy(), rtol=1e-12, equal_nan=True)
        assert list(r.asset_meta.index) == list(sim.returns.asset_meta.index) and "asset_class" in r.asset_meta.columns
        assert list(r.asset_meta["asset_class"]) == list(sim.returns.asset_meta["asset_class"])
        assert r.risk_free is not None and r.risk_free.index.equals(ret.index) and float(r.risk_free.iloc[0]) == 0.0001
    with pytest.raises(ValueError):
        pipeline.read_frame(tmp_path / "attention.xlsx")
    bad = pd.DataFrame({"a": [1.0]}, index=["not-a-date"])
    bad.to_csv(tmp_path / "bad.csv")
    with pytest.raises(ValueError):
        pipeline.read_frame(tmp_path / "bad.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
CLI_SIM = SimulationConfig(seed=3, n_assets=80, n_topics=16, n_relevant=4, n_placebo=4, n_years=5)


def cli_pipeline_cfg() -> PipelineConfig:
    return PipelineConfig(
        covariance=CovarianceConfig(burn_in_periods=6),
        oos=OOSConfig(min_train_periods=24, oos_fraction=0.3, refit_every=12),
        estimation=EstimationConfig(lam_grid=LambdaGridConfig(n_lambdas=4, ratio=0.1)),
        name="cli-test",
    )


def test_cli_simulate_then_run_round_trip(tmp_path, capsys):
    sim_path = tmp_path / "sim.json"
    cfg_path = tmp_path / "cfg.yaml"
    save_config(CLI_SIM, sim_path)
    save_config(cli_pipeline_cfg(), cfg_path)
    d_sim = tmp_path / "sim"
    d_run = tmp_path / "run"

    assert cli.main(["simulate", "--config", str(sim_path), "--out", str(d_sim)]) == 0
    for name in ("attention.parquet", "returns.csv", "asset_meta.csv", "truth_summary.json", "truth.npz", "simulation_config.json"):
        assert (d_sim / name).is_file(), name
    summary = json.loads((d_sim / "truth_summary.json").read_text(encoding="utf-8"))
    assert summary["n_assets"] == 80 and summary["n_topics"] == 16 and len(summary["relevant_topics"]) == 4
    assert load_config(d_sim / "simulation_config.json", SimulationConfig) == CLI_SIM

    assert cli.main([
        "run", "--config", str(cfg_path), "--attention", str(d_sim / "attention.parquet"),
        "--returns", str(d_sim / "returns.csv"), "--meta", str(d_sim / "asset_meta.csv"), "--out", str(d_run),
    ]) == 0
    out = capsys.readouterr().out
    assert "cli-test" in out and "lambda*=" in out
    assert (d_run / "metrics.json").is_file() and (d_run / "gamma.csv").is_file() and (d_run / "asset_meta.csv").is_file()
    metrics = json.loads((d_run / "metrics.json").read_text(encoding="utf-8"))
    assert load_config(d_run / "config.json", PipelineConfig) == replace(cli_pipeline_cfg(), output_dir=str(d_run))

    # the file round trip reproduces the in-process run on the same simulated data
    sim = simulate(CLI_SIM)
    ref = pipeline.run_pipeline(sim.attention, sim.returns, cli_pipeline_cfg())
    for key, value in ref.evaluation.metrics.items():
        assert _nan_equal(metrics[key], value, rtol=1e-6), key
    gamma = pd.read_csv(d_run / "gamma.csv", index_col=0)
    scale = float(np.abs(ref.fit.Gamma).max())
    np.testing.assert_allclose(gamma.to_numpy(), ref.fit.Gamma, rtol=1e-6, atol=1e-8 * scale)


def test_cli_simulate_scenario_seed_and_csv(tmp_path):
    # "null" is the accepted alias of "topic_null" (the help lists baseline | no_factor | topic_null (alias null) | ...)
    d = tmp_path / "null"
    assert cli.main(["simulate", "--config", str(_write(tmp_path, CLI_SIM)), "--scenario", "null", "--seed", "11", "--csv", "--out", str(d)]) == 0
    assert (d / "attention.csv").is_file() and not (d / "attention.parquet").exists()
    summary = json.loads((d / "truth_summary.json").read_text(encoding="utf-8"))
    assert summary["scenario"] == "null" and summary["seed"] == 11
    cfg = load_config(d / "simulation_config.json", SimulationConfig)
    assert cfg.signal_strength == 0.0 and cfg.seed == 11 and cfg.n_assets == 80
    with np.load(d / "truth.npz") as npz:
        assert npz["A"].shape == (16, 3) and not np.any(npz["A"])


def _write(tmp_path: Path, cfg) -> Path:
    p = tmp_path / "sim_cfg.json"
    save_config(cfg, p)
    return p


def test_cli_run_usage_and_data_errors(tmp_path, capsys):
    # no output directory anywhere -> usage error 2
    (tmp_path / "a.csv").write_text("date,t1\n2020-01-02,1.0\n2020-01-03,2.0\n", encoding="utf-8")
    (tmp_path / "r.csv").write_text("date,x\n2020-01-02,0.01\n2020-01-03,-0.01\n", encoding="utf-8")
    assert cli.main(["run", "--attention", str(tmp_path / "a.csv"), "--returns", str(tmp_path / "r.csv")]) == 2
    # missing input file -> 1
    assert cli.main(["run", "--attention", str(tmp_path / "missing.csv"), "--returns", str(tmp_path / "r.csv"), "--out", str(tmp_path)]) == 1
    # a data set too short for the pipeline -> ValueError -> 1, not a traceback
    assert cli.main(["run", "--attention", str(tmp_path / "a.csv"), "--returns", str(tmp_path / "r.csv"), "--out", str(tmp_path)]) == 1
    assert "error:" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_harness_guard_and_dispatch(monkeypatch, tmp_path):
    # missing module -> exit 2 with a clear message
    monkeypatch.setitem(sys.modules, "narrative_ipca.harness", None)
    monkeypatch.delattr(narrative_ipca, "harness", raising=False)
    assert cli.main(["harness", "--scenarios", "baseline,no_factor,topic_null", "--seeds", "2", "--out", str(tmp_path)]) == 2
    # stub module -> the parsed HarnessConfig reaches run_harness and write_report
    seen: dict[str, object] = {}
    stub = types.ModuleType("narrative_ipca.harness")

    def run_harness(hcfg, pipeline_cfg=None, progress=None):
        seen["hcfg"] = hcfg
        seen["pipeline_cfg"] = pipeline_cfg
        return "RESULT"

    def write_report(result, out_dir):
        seen["report"] = (result, out_dir)
        return str(Path(out_dir) / "harness.md")

    stub.run_harness = run_harness  # type: ignore[attr-defined]
    stub.write_report = write_report  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "narrative_ipca.harness", stub)
    monkeypatch.setattr(narrative_ipca, "harness", stub, raising=False)
    cfg_path = tmp_path / "p.json"
    save_config(cli_pipeline_cfg(), cfg_path)
    assert cli.main(["harness", "--scenarios", "baseline, no_factor, topic_null", "--seeds", "2", "--fast", "--config", str(cfg_path), "--out", str(tmp_path)]) == 0
    hcfg = seen["hcfg"]
    assert isinstance(hcfg, HarnessConfig)
    assert hcfg.scenarios == ("baseline", "no_factor", "topic_null") and hcfg.n_seeds == 2 and hcfg.fast and hcfg.output_dir == str(tmp_path)
    assert seen["pipeline_cfg"] == cli_pipeline_cfg()
    assert seen["report"] == ("RESULT", str(tmp_path))


# ---------------------------------------------------------------------------
# shipped configs
# ---------------------------------------------------------------------------
def test_shipped_yaml_configs_equal_the_defaults():
    assert load_config(REPO / "configs" / "default.yaml", PipelineConfig) == PipelineConfig()
    assert load_config(REPO / "configs" / "simulation_baseline.yaml", SimulationConfig) == SimulationConfig()
