"""Tests of the simulation harness (harness.py, scripts/run_simulation_study.py).

What is checked, mathematically rather than by shape:

* the rotation-invariant numerics -- principal-angle cosines against
  ``scipy.linalg.subspace_angles`` and canonical correlations against the
  eigenvalue form ``Sxx^-1 Sxy Syy^-1 Syx`` -- including invariance to
  invertible transforms and rank-deficient inputs;
* ``compare_to_truth`` on a real (tiny) pipeline run: every scalar equals an
  independent recomputation (hand-built selection counts, a row loop for the
  systematic R2, the true-MVE OOS Sharpe, hand-aligned canonical correlations);
* ``compare_to_truth`` on a synthetic perfect estimate ``Gamma_tilde_true R``,
  ``F_true R`` for random orthogonal and general invertible ``R``: all
  recovery metrics equal one;
* the metrics that separate loading recovery from recall (DESIGN.md Part E):
  ``gamma_subspace_cos`` and the impact-vector agreement over the relevant
  topics that were *selected* (a ``Gamma_hat`` equal to the truth on the
  selected rows scores one while the all-relevant-rows values do not; with
  at most ``K`` selected relevant rows the subspace comparison is vacuous
  and the metric is NaN, with ``K < n < 2K`` rows the cosines that are one
  by dimension counting are left out of the mean), the
  strong / weak split of recall by ``||A_l||`` on a hand-built ``A``, and
  ``beta_canonical_corr`` on implied betas ``c Gamma`` that reproduce the
  true betas up to an invertible transform (one; invariant to a rotation of
  ``Gamma``; ``nan`` without true loadings);
* the pass/fail logic per scenario and the threshold relaxations (check set
  v2, DESIGN.md D52): the signal scenarios get the identified targets, the
  selective-tuning checks and the two soft recall checks; ``gamma_subspace_cos``,
  ``state_canonical_corr`` and ``impact_spearman`` are reported only (in no
  check set, no pass flag, 'reported (not identified)' in the report); the
  chance-level null ``no_factor`` gets the two null checks and no placebo
  check, the report-only ``topic_null`` (alias ``null``) none;
* ``scripts/rescore_study.py`` on a hand-built ``per_run.csv``: the pass
  flags are recomputed from the stored metrics, the tables rebuilt, the old
  report replaced;
* the instrument-informativeness diagnostic ``instrument_beta_r2`` against a
  hand-built regression, on a panel that is exactly linear in the loadings
  (R2 = 1), on a baseline run (relevant topics far above chance) and under
  ``no_factor`` (zero by construction);
* ``run_harness`` in fast mode on a tiny base (baseline + no_factor, one
  seed): tables, artefacts, report; the report-only plumbing of
  ``topic_null``; the failure path; ``write_report`` on a hand-built result;
  the study script's argument plumbing.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr

from narrative_ipca import harness
from narrative_ipca.config import (
    CovarianceConfig,
    EstimationConfig,
    HarnessConfig,
    HarnessThresholds,
    LambdaGridConfig,
    OOSConfig,
    PipelineConfig,
    SimulationConfig,
)
from narrative_ipca.covariances import build_covariance_panel
from narrative_ipca.data import align_inputs
from narrative_ipca.evaluation import realized_sharpe
from narrative_ipca.grouplasso import active_backend
from narrative_ipca.pipeline import run_pipeline
from narrative_ipca.shocks import attention_shocks
from narrative_ipca.simulation import scenario_config, simulate
from narrative_ipca.types import CovariancePanel, HarnessMetrics, HarnessResult, annualized_sharpe

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# fixtures: one tiny pipeline run shared by the compare_to_truth tests
# ---------------------------------------------------------------------------
TINY = SimulationConfig(seed=0, n_assets=100, n_topics=20, n_relevant=5, n_placebo=5, n_years=4)


def tiny_pipeline_cfg() -> PipelineConfig:
    return PipelineConfig(
        covariance=CovarianceConfig(burn_in_periods=6),
        estimation=EstimationConfig(lam_grid=LambdaGridConfig(n_lambdas=6, ratio=0.05)),
        oos=OOSConfig(min_train_periods=24, oos_fraction=0.4, refit_every=12),
        name="harness-test",
    )


@pytest.fixture(scope="module")
def sim():
    return simulate(TINY, scenario="baseline")


@pytest.fixture(scope="module")
def result(sim):
    return run_pipeline(sim.attention, sim.returns, tiny_pipeline_cfg())


@pytest.fixture(scope="module")
def metrics(result, sim) -> HarnessMetrics:
    return harness.compare_to_truth(result, sim.truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)


def _r2_subsample(T_cov: int) -> list[int]:
    """The instrument-R2 subsample: every 12th period counted back from the last, widened to 5 evenly spaced ones."""
    sel = list(range(T_cov - 1, -1, -12))[::-1]
    if len(sel) < 5:
        sel = sorted(set(np.linspace(0, T_cov - 1, min(5, T_cov)).round().astype(int).tolist()))
    return sel


def _principal_components(Xc: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Scores of ``Xc`` on its principal directions with singular value > rcond * s_max.

    A rank-deficient side (an ``F`` from a ``Gamma`` with fewer than K non-zero rows has a
    direction of relative size ~1e-16, pure rounding) makes ``Sxx`` singular and the
    ``Sxx^-1`` form below meaningless; canonical correlations are invariant to this
    invertible reduction of each side to its non-degenerate span.
    """
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[s > rcond * s[0]].T


def _cca_reference(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Canonical correlations as sqrt of the eigenvalues of Sxx^-1 Sxy Syy^-1 Syx."""
    Xc = _principal_components(X - X.mean(axis=0))
    Yc = _principal_components(Y - Y.mean(axis=0))
    Sxx, Syy, Sxy = Xc.T @ Xc, Yc.T @ Yc, Xc.T @ Yc
    M = np.linalg.solve(Sxx, Sxy) @ np.linalg.solve(Syy, Sxy.T)
    ev = np.sort(np.linalg.eigvals(M).real)[::-1]
    return np.sqrt(np.clip(ev, 0.0, 1.0))


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------
def test_subspace_cosines_match_scipy_and_are_invariant():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((25, 3))
    B = rng.standard_normal((25, 3))
    cos = harness.subspace_cosines(A, B)
    ref = np.sort(np.cos(subspace_angles(A, B)))[::-1]
    np.testing.assert_allclose(cos, ref, atol=1e-12)
    assert np.all(np.diff(cos) <= 1e-12) and np.all((cos >= 0) & (cos <= 1))
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    np.testing.assert_allclose(harness.subspace_cosines(A, A @ Q), np.ones(3), atol=1e-12)
    R = rng.standard_normal((3, 3)) + 3 * np.eye(3)  # general invertible
    np.testing.assert_allclose(harness.subspace_cosines(A @ R, A), np.ones(3), atol=1e-12)
    np.testing.assert_allclose(harness.subspace_cosines(A @ R, B @ Q), cos, atol=1e-12)
    # a 2-dimensional subspace inside a 3-dimensional one: two angles, both zero
    sub = A[:, :2] @ rng.standard_normal((2, 2))
    np.testing.assert_allclose(harness.subspace_cosines(sub, A), np.ones(2), atol=1e-12)
    # rank-deficient input gives fewer cosines, never spurious ones
    A_def = np.column_stack([A[:, 0], A[:, 1], A[:, 0] + A[:, 1]])
    assert harness.subspace_cosines(A_def, B).shape == (2,)
    assert harness.subspace_cosines(np.zeros((25, 3)), B).shape == (0,)
    with pytest.raises(ValueError):
        harness.subspace_cosines(A, rng.standard_normal((10, 3)))


def test_canonical_correlations_match_eigenvalue_form_and_invariances():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 3))
    R = rng.standard_normal((3, 3)) + 2 * np.eye(3)
    Y = X @ R + 0.7 * rng.standard_normal((300, 3))
    rho = harness.canonical_correlations(X, Y)
    np.testing.assert_allclose(rho, _cca_reference(X, Y), atol=1e-10)
    assert rho.shape == (3,) and np.all(np.diff(rho) <= 1e-12)
    # exact linear relation (any invertible R, plus shifts): all ones
    np.testing.assert_allclose(harness.canonical_correlations(X, X @ R + 5.0), np.ones(3), atol=1e-10)
    # invariance to invertible transforms and shifts on either side
    S = rng.standard_normal((3, 3)) + 2 * np.eye(3)
    np.testing.assert_allclose(harness.canonical_correlations(X @ S - 1.0, Y @ R.T + 2.0), rho, atol=1e-10)
    # different dimensions: min(p, q) correlations, the first the best single-direction correlation
    rho_2 = harness.canonical_correlations(X[:, :2], Y)
    assert rho_2.shape == (2,)
    np.testing.assert_allclose(rho_2, _cca_reference(X[:, :2], Y), atol=1e-10)
    # a rank-deficient side yields fewer correlations (no spurious unit correlation)
    X_def = np.column_stack([X[:, 0], X[:, 1], X[:, 0] - X[:, 1]])
    assert harness.canonical_correlations(X_def, Y).shape == (2,)
    # one column each: |Pearson correlation|
    r = np.corrcoef(X[:, 0], Y[:, 0])[0, 1]
    np.testing.assert_allclose(harness.canonical_correlations(X[:, 0], Y[:, 0]), [abs(r)], atol=1e-12)
    assert harness.canonical_correlations(X[:1], Y[:1]).shape == (0,)
    with pytest.raises(ValueError):
        harness.canonical_correlations(X, Y[:10])


# ---------------------------------------------------------------------------
# pass/fail logic
# ---------------------------------------------------------------------------
def test_evaluate_checks_per_scenario_and_relaxations():
    thr = HarnessThresholds()
    good = {
        "selection_recall": thr.selection_recall_min, "selection_recall_strong": thr.selection_recall_strong_min,
        "selection_precision": 0.6, "beta_canonical_corr": thr.beta_canonical_corr_min, "placebo_selected": 0.0,
        "gamma_subspace_cos": 0.85, "factor_canonical_corr": 0.9, "state_canonical_corr": 0.8, "impact_spearman": 0.7,
        "oos_sharpe_ratio_to_true": 0.5, "systematic_r2_recovered": 0.5, "null_selection_lift": 2.0,
        "null_oos_sharpe_abs": 0.4,
    }
    assert thr.selection_recall_min == 0.5 and thr.selection_recall_strong_min == 0.8 and thr.beta_canonical_corr_min == 0.9
    # thresholds are inclusive; every signal check passes at the boundary
    passed = harness.evaluate_checks(good, thr, "baseline")
    assert tuple(passed) == harness.SIGNAL_CHECKS and all(passed.values())
    # check set v2 (D52): the identified targets, the selective-tuning checks and the two soft recall checks
    assert harness.CHECK_SET_VERSION == "v2"
    assert harness.SIGNAL_CHECKS == (
        "selection_recall", "selection_recall_strong", "selection_precision", "beta_canonical_corr", "placebo_selected",
        "factor_canonical_corr", "oos_sharpe_ratio_to_true", "systematic_r2_recovered",
    )
    # ... the three unidentified metrics are reported only: still in CHECKS / METRICS (thresholds can be re-enabled),
    # in no check set, never a pass flag
    assert harness.REPORT_ONLY_METRICS == ("gamma_subspace_cos", "state_canonical_corr", "impact_spearman")
    assert set(harness.REPORT_ONLY_METRICS) <= {m for _, m, _, _ in harness.CHECKS}
    assert set(harness.REPORT_ONLY_METRICS) <= set(harness.METRICS)
    for name in ("baseline", "softmax", "weak", "balanced", "no_factor", "topic_null", "custom"):
        assert not set(harness.REPORT_ONLY_METRICS) & set(harness.scenario_checks(name)), name
        assert not set(harness.REPORT_ONLY_METRICS) & set(harness.evaluate_checks(good, thr, name)), name
    bad_reported = dict(good, gamma_subspace_cos=0.0, state_canonical_corr=0.0, impact_spearman=float("nan"))
    assert all(harness.evaluate_checks(bad_reported, thr, "baseline").values())
    for metric in harness.REPORT_ONLY_METRICS:
        assert harness.EXPECTED[metric].startswith("reported only: not an identified target of the model, see DESIGN.md D52")
    for check in ("selection_recall", "selection_recall_strong"):
        assert harness.EXPECTED[check].startswith("soft check: a sparse representative may legitimately use only the strong topics")
    # no_factor is the chance-level null: the two null checks apply, the placebo check does not (a chance-level
    # selection includes placebos at rate n_placebo / L, D52)
    assert harness.NULL_CHECKS == ("null_selection_lift", "null_oos_sharpe_abs")
    assert harness.evaluate_checks(good, thr, "no_factor") == {"null_selection_lift": True, "null_oos_sharpe_abs": True}
    assert "placebo_selected" not in harness.evaluate_checks(dict(good, placebo_selected=3.0), thr, "no_factor")
    assert "n_placebo / L" in harness.EXPECTED["placebo_selected_no_factor"]
    # topic_null (alias null) is report-only: no check, vacuously all passed
    for name in ("topic_null", "null", "NULL-fast", "topic_null_fast"):
        assert harness.scenario_checks(name) == () and harness.evaluate_checks(good, thr, name) == {}, name
        assert HarnessMetrics(values=good, passed=harness.evaluate_checks(good, thr, name)).all_passed
    assert harness._canonical_scenario("null") == "topic_null" == harness._canonical_scenario("Null-fast")
    assert harness.SCENARIO_ALIASES == {"null": "topic_null"}
    assert harness.NULL_SCENARIOS == ("no_factor",) and harness.REPORT_ONLY_SCENARIOS == ("topic_null",)
    # one unit below a min / above a max fails; NaN and missing metrics fail
    bad = dict(good, selection_recall=thr.selection_recall_min - 0.01, placebo_selected=1.0, null_selection_lift=2.01)
    p = harness.evaluate_checks(bad, thr, "softmax")
    assert not p["selection_recall"] and not p["placebo_selected"] and p["selection_precision"]
    assert not harness.evaluate_checks(bad, thr, "no_factor")["null_selection_lift"]
    assert not harness.evaluate_checks(dict(good, factor_canonical_corr=float("nan")), thr, "balanced")["factor_canonical_corr"]
    missing = dict(good)
    del missing["systematic_r2_recovered"]
    assert not harness.evaluate_checks(missing, thr, "baseline")["systematic_r2_recovered"]
    # weak: recall 0.3 and strong-half recall 0.6 are enough; baseline fails at those values. Only checks that still
    # apply are relaxed (the subspace cosine is reported only, its threshold field is untouched)
    relaxed = harness.scenario_thresholds(thr, "weak")
    assert relaxed.selection_recall_min == 0.3 and relaxed.selection_recall_strong_min == 0.6
    assert relaxed.gamma_subspace_cos_min == thr.gamma_subspace_cos_min
    assert relaxed.selection_recall_min < thr.selection_recall_min
    assert relaxed.selection_recall_strong_min < thr.selection_recall_strong_min
    weakish = dict(
        good, selection_recall=relaxed.selection_recall_min + 0.05, selection_recall_strong=relaxed.selection_recall_strong_min + 0.05,
    )
    assert all(harness.evaluate_checks(weakish, thr, "weak").values())
    pb = harness.evaluate_checks(weakish, thr, "baseline")
    assert not pb["selection_recall"] and not pb["selection_recall_strong"] and pb["beta_canonical_corr"]
    assert relaxed.selection_precision_min == thr.selection_precision_min and relaxed.beta_canonical_corr_min == thr.beta_canonical_corr_min
    assert harness.SCENARIO_THRESHOLD_OVERRIDES == {"weak": {"selection_recall_min": 0.3, "selection_recall_strong_min": 0.6}}
    assert harness.scenario_thresholds(thr, "baseline") == thr
    # scenario names: case-insensitive, '-fast' suffix ignored, unknown -> signal checks
    assert harness.scenario_checks("NO_FACTOR-fast") == harness.NULL_CHECKS == harness.scenario_checks("No-Factor")
    assert harness.scenario_checks("weak_fast") == harness.SIGNAL_CHECKS
    assert harness.scenario_checks("custom") == harness.SIGNAL_CHECKS
    assert harness.scenario_thresholds(thr, "Weak-Fast").selection_recall_min == 0.3
    # every check of the table has a threshold field; the two new Part E checks sit in the signal list and in METRICS
    for _, _, _, field in harness.CHECKS:
        assert hasattr(thr, field)
    assert ("selection_recall_strong", "selection_recall_strong", ">=", "selection_recall_strong_min") in harness.CHECKS
    assert ("beta_canonical_corr", "beta_canonical_corr", ">=", "beta_canonical_corr_min") in harness.CHECKS
    assert {"selection_recall_strong", "beta_canonical_corr"} <= set(harness.SIGNAL_CHECKS)
    assert not {"selection_recall_strong", "beta_canonical_corr"} & set(harness.NULL_CHECKS)
    assert {"selection_recall_strong", "selection_recall_weak", "beta_canonical_corr", "beta_canonical_corr_mean",
            "gamma_subspace_cos_all_relevant", "impact_spearman_all_relevant"} <= set(harness.METRICS)
    assert "near 1" in harness.EXPECTED["selection_recall_strong"] and "> 0.95" in harness.EXPECTED["beta_canonical_corr"]


# ---------------------------------------------------------------------------
# compare_to_truth on a real run: every number equals an independent recomputation
# ---------------------------------------------------------------------------
def test_compare_to_truth_reproduces_hand_computations(result, sim, metrics):
    truth = sim.truth
    v = metrics.values
    fit = result.fit
    topics = list(sim.attention.topics)
    assert list(fit.instrument_names[1:]) == topics
    # selection counts
    selected = set(fit.selected_topics)
    relevant = {t for t, r in zip(topics, truth.relevant) if r}
    placebo = {t for t, p in zip(topics, truth.placebo) if p}
    hits = len(selected & relevant)
    assert v["n_selected"] == len(selected) == fit.n_selected
    assert v["selection_recall"] == pytest.approx(hits / len(relevant))
    assert v["selection_precision"] == pytest.approx(hits / len(selected))
    assert v["selection_f1"] == pytest.approx(2 * hits / (len(selected) + len(relevant)))
    assert v["placebo_selected"] == len(selected & placebo)
    assert metrics.details["selected_topics"] == fit.selected_topics
    assert set(metrics.details["selected_placebo"]) == selected & placebo
    # recall over the strong / weak half of the relevant topics by ||A_l|| (ties at the median count as strong)
    norms = np.linalg.norm(truth.A, axis=1)
    med = float(np.median(norms[truth.relevant]))
    strong = {t for t, r, n in zip(topics, truth.relevant, norms) if r and n >= med}
    weak = relevant - strong
    assert len(strong) == 3 and len(weak) == 2  # five relevant topics of distinct strength
    assert v["selection_recall_strong"] == pytest.approx(len(selected & strong) / len(strong))
    assert v["selection_recall_weak"] == pytest.approx(len(selected & weak) / len(weak))
    assert v["n_relevant_strong"] == 3 and v["n_relevant_weak"] == 2 and v["relevant_norm_median"] == pytest.approx(med)
    assert set(metrics.details["strong_topics"]) == strong and set(metrics.details["weak_topics"]) == weak
    assert set(metrics.details["selected_strong"]) == selected & strong and set(metrics.details["selected_weak"]) == selected & weak
    # Gamma subspace: hand restriction to the relevant rows that were selected, and to all relevant rows
    rel = np.asarray(truth.relevant, dtype=bool)
    sel = np.asarray([t in selected for t in topics], dtype=bool)
    K = truth.A.shape[1]
    for key, rows in (("gamma_subspace_cos", rel & sel), ("gamma_subspace_cos_all_relevant", rel)):
        G_hat_rows = fit.Gamma_tilde[rows]
        n_rows = int(np.count_nonzero(np.any(G_hat_rows != 0, axis=1)))
        n_inf = min(K, n_rows - K)  # cosines not forced to one by dimension counting (2K - n of them are)
        if n_inf >= 1 and np.linalg.matrix_rank(G_hat_rows) >= K:
            cos = np.zeros(K)
            found = np.sort(np.cos(subspace_angles(G_hat_rows, truth.Gamma_tilde_true[rows])))[::-1][:K]
            cos[: len(found)] = found
            assert v[key] == pytest.approx(float(np.mean(np.sort(cos)[:n_inf])), abs=1e-10), key
        else:
            assert math.isnan(v[key]), key
    assert v["gamma_relevant_rows_selected"] == len(selected & relevant)
    assert v["gamma_informative_cosines"] == max(0, min(K, len(selected & relevant) - K)) == metrics.details["gamma_informative_cosines"]
    assert metrics.details["gamma_informative_cosines_all_relevant"] == min(K, len(relevant) - K)
    # factors: canonical correlations on hand-aligned dates
    populated = np.asarray(fit.populated, dtype=bool)
    F_hat = fit.factors_frame().loc[populated]
    common = F_hat.index.intersection(truth.f_period.index)
    assert len(common) == populated.sum() == metrics.details["n_factor_periods"]
    rho = _cca_reference(F_hat.loc[common].to_numpy(), truth.f_period.loc[common].to_numpy())
    assert v["factor_canonical_corr"] == pytest.approx(rho[0], abs=1e-8)
    assert v["factor_canonical_corr_mean"] == pytest.approx(rho.sum() / truth.A.shape[1], abs=1e-8)
    # states: canonical correlations on common finite days
    st = result.wrapup.states
    joint = st.join(truth.x_daily, how="inner").dropna()
    rho_x = _cca_reference(joint[st.columns].to_numpy(), joint[truth.x_daily.columns].to_numpy())
    assert v["state_canonical_corr"] == pytest.approx(rho_x[0], abs=1e-8)
    assert metrics.details["n_state_days"] == len(joint) == int(st.notna().all(axis=1).sum())
    # impact vector: over the selected relevant topics (undefined below three) and over all relevant topics;
    # the estimator's impact on a non-selected topic is exactly zero, so the all-relevant value reads recall
    assert (result.wrapup.impact_z_to_mve[~fit.selected] == 0.0).all()
    true_series = pd.Series(truth.impact_z_to_mve_true, index=topics)
    imp = result.wrapup.impact_z_to_mve.reindex(sorted(relevant)).to_numpy()
    true_imp = true_series.reindex(sorted(relevant)).to_numpy()
    assert v["impact_spearman_all_relevant"] == pytest.approx(spearmanr(imp, true_imp).statistic)
    assert v["impact_sign_agreement_all_relevant"] == pytest.approx(np.mean(np.sign(imp) == np.sign(true_imp)))
    sel_rel = sorted(selected & relevant)
    assert metrics.details["n_impact_topics"] == len(sel_rel) and metrics.details["n_impact_topics_all_relevant"] == len(relevant)
    if len(sel_rel) >= 3:
        imp_s, true_s = result.wrapup.impact_z_to_mve.reindex(sel_rel).to_numpy(), true_series.reindex(sel_rel).to_numpy()
        assert v["impact_spearman"] == pytest.approx(spearmanr(imp_s, true_s).statistic)
        assert v["impact_sign_agreement"] == pytest.approx(np.mean(np.sign(imp_s) == np.sign(true_s)))
    else:
        assert math.isnan(v["impact_spearman"]) and math.isnan(v["impact_sign_agreement"])
    # Sharpe ratios
    ann = tiny_pipeline_cfg().evaluation.annualization
    assert v["mve_sharpe_is"] == pytest.approx(fit.mve_sharpe(ann, 1e-12))
    assert v["sharpe_mve_true"] == truth.sharpe_mve_true
    assert v["oos_sharpe"] == pytest.approx(result.oos.sharpe) == pytest.approx(result.evaluation.metrics["oos_sharpe"])
    b_true = np.linalg.solve(truth.Sigma_ff_period, truth.mu_f_period)
    mve_true = pd.Series(truth.f_period.to_numpy() @ b_true, index=truth.f_period.index)
    oos_idx = result.oos.mve.index
    assert oos_idx.isin(truth.periods).all() and v["n_oos_periods"] == len(oos_idx)
    assert v["oos_sharpe_true_mve"] == pytest.approx(realized_sharpe(mve_true.loc[oos_idx], ann))
    if v["oos_sharpe_true_mve"] > 0:
        assert v["oos_sharpe_ratio_to_true"] == pytest.approx(v["oos_sharpe"] / v["oos_sharpe_true_mve"])
    else:
        assert math.isnan(v["oos_sharpe_ratio_to_true"])
    assert v["null_oos_sharpe_abs"] == pytest.approx(abs(v["oos_sharpe"]))
    n_oos = len(oos_idx)
    assert v["oos_sharpe_se"] == pytest.approx(np.sqrt(ann / n_oos) * np.sqrt(1 + (v["oos_sharpe"] / np.sqrt(ann)) ** 2 / 2))
    assert v["factor_structure"] == 1.0
    # OOS selection stability is the pipeline's evaluation metric (mean Jaccard between consecutive refits)
    assert v["oos_selection_stability"] == pytest.approx(result.evaluation.metrics["oos_selection_stability"])
    # instrument R2: a hand regression of every 12th covariance cross-section (counted from the last) on [1, beta]
    cov = result.covariances
    T_cov = cov.values.shape[0]
    sel = _r2_subsample(T_cov)
    assert len(sel) == 5 and len(range(T_cov - 1, -1, -12)) < 5  # the tiny run needs the widening rule
    id_pos = {a: i for i, a in enumerate(sim.returns.assets)}
    per_pos = {ts: i for i, ts in enumerate(truth.periods)}
    K = truth.A.shape[1]
    hand, n_used, used = [], [], []
    for t in sel:
        tt = per_pos[cov.periods[t]]
        rows, C = [], []
        for i, a in enumerate(cov.assets):
            b = truth.beta[tt, id_pos[str(a)]]
            c = cov.values[t, i]
            if np.all(np.isfinite(c)) and np.all(np.isfinite(b)):
                rows.append(np.r_[1.0, b])
                C.append(c)
        if len(rows) < K + 2:  # early kernel windows have no asset with min_days observed days: skipped
            continue
        X, Y = np.array(rows), np.array(C)
        fitted = X @ np.linalg.lstsq(X, Y, rcond=None)[0]
        hand.append(1 - ((Y - fitted) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0))
        n_used.append(len(rows))
        used.append(t)
    assert 2 <= len(used) < len(sel)  # the first covariance period of the tiny run is all-NaN
    hand = np.mean(hand, axis=0)
    kinds = pd.Series(np.where(truth.relevant, "relevant", np.where(truth.placebo, "placebo", "noise")), index=topics)
    for kind in ("relevant", "noise", "placebo"):
        assert v[f"instrument_beta_r2_{kind}"] == pytest.approx(hand[(kinds == kind).to_numpy()].mean(), abs=1e-10), kind
    assert v["instrument_beta_r2_chance"] == pytest.approx(K / np.mean(n_used))
    assert v["instrument_beta_r2_n_periods"] == len(used) and v["instrument_beta_r2_mean_n_assets"] == pytest.approx(np.mean(n_used))
    assert metrics.details["instrument_beta_r2"]["n_assets"] == n_used
    assert metrics.details["instrument_beta_r2"]["periods"] == [cov.periods[t].isoformat() for t in used]
    assert len(metrics.details["instrument_beta_r2"]["per_topic"]) == len(topics)
    # systematic R2 recovered: a row loop with dictionary lookups
    panel = result.panel
    id_pos = {a: i for i, a in enumerate(sim.returns.assets)}
    per_pos = {ts: i for i, ts in enumerate(truth.periods)}
    f_true = truth.f_period.to_numpy()
    s, h, h_narr = [], [], []
    G_narr = fit.Gamma.copy()
    G_narr[0] = 0.0
    for n in range(panel.n_obs):
        t = per_pos[panel.periods[panel.t_idx[n]]]
        i = id_pos[str(panel.assets[panel.asset_idx[n]])]
        s.append(truth.beta[t, i] @ f_true[t])
        h.append((panel.X[n] @ fit.Gamma) @ fit.F[panel.t_idx[n]])
        h_narr.append((panel.X[n] @ G_narr) @ fit.F[panel.t_idx[n]])
    s, h, h_narr = np.array(s), np.array(h), np.array(h_narr)
    assert np.all(np.isfinite(s)) and metrics.details["n_systematic_rows"] == panel.n_obs
    assert v["systematic_r2_recovered"] == pytest.approx(1 - ((s - h) ** 2).sum() / (s**2).sum(), rel=1e-10)
    assert v["systematic_r2_recovered_narrative"] == pytest.approx(1 - ((s - h_narr) ** 2).sum() / (s**2).sum(), rel=1e-10)
    assert v["systematic_r2_true"] == truth.systematic_r2
    assert v["total_r2"] == pytest.approx(result.evaluation.metrics["total_r2"])
    # implied betas X_t Gamma_hat vs the true beta_t: canonical correlations per sampled return period (the instrument
    # subsample rule over the panel's return periods), the first and the mean over K averaged over the periods
    T_p = len(panel.periods)
    sel_p = _r2_subsample(T_p)
    firsts, means, n_b = [], [], []
    for t in sel_p:
        rows = np.flatnonzero(panel.t_idx == t)
        tt = per_pos[panel.periods[t]]
        B_hat = panel.X[rows] @ fit.Gamma
        B = np.array([truth.beta[tt, id_pos[str(panel.assets[panel.asset_idx[n]])]] for n in rows])
        assert np.all(np.isfinite(B))  # a panel row has an observed return, hence a loading
        rho = _cca_reference(B_hat, B)
        firsts.append(rho[0])
        means.append(rho.sum() / K)
        n_b.append(len(rows))
    assert v["beta_canonical_corr"] == pytest.approx(np.mean(firsts), abs=1e-8)
    assert v["beta_canonical_corr_mean"] == pytest.approx(np.mean(means), abs=1e-8)
    assert v["beta_canonical_corr_n_periods"] == len(sel_p) == 5 and v["beta_canonical_corr_mean_n_assets"] == pytest.approx(np.mean(n_b))
    assert metrics.details["beta_canonical_corr"]["periods"] == [panel.periods[t].isoformat() for t in sel_p]
    assert metrics.details["beta_canonical_corr"]["n_assets"] == n_b
    np.testing.assert_allclose(metrics.details["beta_canonical_corr"]["first"], firsts, atol=1e-8)
    assert 0.0 < v["beta_canonical_corr_mean"] <= v["beta_canonical_corr"] <= 1.0
    # null metrics and bookkeeping
    L = len(topics)
    assert v["null_selection_lift"] == pytest.approx(fit.n_selected / max(1, round(0.05 * L)))
    assert v["null_selection_lift_relevant"] == pytest.approx(fit.n_selected / len(relevant))
    assert metrics.details["chance_selected"] == max(1, round(0.05 * L))
    assert v["runtime_seconds"] == result.timings["total"] and v["lam_star"] == fit.lam
    assert v["L"] == L and v["K"] == fit.K == v["K_true"]
    # pass flags follow evaluate_checks with the baseline thresholds
    assert metrics.passed == harness.evaluate_checks(v, HarnessThresholds(), "baseline")
    assert tuple(metrics.passed) == harness.SIGNAL_CHECKS
    assert metrics.all_passed == all(metrics.passed.values())
    assert metrics.details["thresholds"]["selection_recall_min"] == HarnessThresholds().selection_recall_min


def test_compare_to_truth_positional_and_period_label_fallbacks(result, sim, metrics):
    """Without asset ids the assets are matched positionally (same count); the OOS periods match by label."""
    m2 = harness.compare_to_truth(result, sim.truth, HarnessThresholds(), "baseline")
    assert m2.values["systematic_r2_recovered"] == pytest.approx(metrics.values["systematic_r2_recovered"])
    # a truth stamped at calendar month ends still aligns (period-label fallback)
    truth = sim.truth
    month_end = truth.periods.to_period("M").to_timestamp(how="end").normalize()
    f_me = truth.f_period.set_axis(month_end, axis=0)
    truth_me = replace(truth, periods=pd.DatetimeIndex(month_end), f_period=f_me)
    m3 = harness.compare_to_truth(result, truth_me, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)
    for key in ("factor_canonical_corr", "oos_sharpe_true_mve", "systematic_r2_recovered", "beta_canonical_corr", "beta_canonical_corr_mean"):
        assert m3.values[key] == pytest.approx(metrics.values[key], abs=1e-10), key
    # a run without wrap-up / OOS gives NaN for the corresponding metrics and failing checks, no exception
    bare = replace(result, wrapup=None, oos=None)
    m4 = harness.compare_to_truth(bare, truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)
    for key in ("state_canonical_corr", "impact_spearman", "impact_spearman_all_relevant", "oos_sharpe", "oos_sharpe_true_mve",
                "oos_sharpe_ratio_to_true", "null_oos_sharpe_abs"):
        assert math.isnan(m4.values[key]), key
    assert m4.values["beta_canonical_corr"] == pytest.approx(metrics.values["beta_canonical_corr"])  # needs the panel only
    assert "state_canonical_corr" not in m4.passed and not m4.passed["oos_sharpe_ratio_to_true"]  # reported only (D52)
    assert m4.values["selection_recall"] == metrics.values["selection_recall"]


@pytest.mark.parametrize("kind", ["orthogonal", "invertible"])
def test_compare_to_truth_perfect_estimate_up_to_rotation(result, sim, kind):
    """Gamma_tilde_hat = Gamma_tilde_true R, f_hat = R^-1 f_true, x_hat = R^-1 x_true: every recovery metric is one.

    ``Gamma -> Gamma R``, ``f -> R^-1 f`` leaves the fitted values ``c Gamma f`` unchanged for any invertible
    ``R`` (for an orthogonal ``R`` this is ``F -> F R``); the subspace, canonical-correlation and impact
    metrics must not notice the reparametrisation (D38).
    """
    truth = sim.truth
    rng = np.random.default_rng(7 if kind == "orthogonal" else 8)
    K = truth.A.shape[1]
    if kind == "orthogonal":
        R, _ = np.linalg.qr(rng.standard_normal((K, K)))
    else:
        R = rng.standard_normal((K, K)) + 2.0 * np.eye(K)
    R_inv = np.linalg.inv(R)
    topics = list(sim.attention.topics)
    fit = result.fit
    f_true = truth.f_period.reindex(fit.periods)
    assert not f_true.isna().any().any()
    F_syn = f_true.to_numpy() @ R_inv.T  # rows (R^-1 f_t)'
    Gamma_syn = np.vstack([np.zeros((1, K)), truth.Gamma_tilde_true @ R])
    norms = np.linalg.norm(Gamma_syn, axis=1)
    mu = F_syn.mean(axis=0)
    Sigma = np.cov(F_syn, rowvar=False, ddof=1)
    fit_syn = replace(
        fit, Gamma=Gamma_syn, F=F_syn, mu_f=mu, Sigma_ff=Sigma, gamma_norms=norms, selected=norms[1:] > 0,
        populated=np.ones(len(fit.periods), dtype=bool),
    )
    states = result.wrapup.states
    x_syn = pd.DataFrame(truth.x_daily.reindex(states.index).to_numpy() @ R_inv.T, index=states.index, columns=states.columns)
    x_syn[states.isna().any(axis=1)] = np.nan  # keep the shock-window NaNs of a real wrap-up
    impact_syn = pd.Series(2.5 * truth.impact_z_to_mve_true, index=topics, name="impact_z_to_mve")
    wrap_syn = replace(result.wrapup, states=x_syn, impact_z_to_mve=impact_syn)
    b_true = np.linalg.solve(truth.Sigma_ff_period, truth.mu_f_period)
    mve_true = pd.Series(truth.f_period.to_numpy() @ b_true, index=truth.f_period.index).loc[result.oos.mve.index]
    oos_syn = replace(result.oos, mve=mve_true, sharpe=realized_sharpe(mve_true, 12.0))
    res_syn = replace(result, fit=fit_syn, wrapup=wrap_syn, oos=oos_syn)

    m = harness.compare_to_truth(res_syn, truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)
    v = m.values
    assert v["selection_recall"] == 1.0 and v["selection_precision"] == 1.0 and v["selection_f1"] == 1.0
    assert v["selection_recall_strong"] == 1.0 and v["selection_recall_weak"] == 1.0
    assert v["placebo_selected"] == 0 and v["n_selected"] == int(truth.relevant.sum())
    assert v["gamma_subspace_cos"] == pytest.approx(1.0, abs=1e-10)
    assert v["gamma_subspace_cos_all_relevant"] == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(m.details["gamma_principal_cosines"], np.ones(K), atol=1e-10)
    np.testing.assert_allclose(m.details["gamma_principal_cosines_all_relevant"], np.ones(K), atol=1e-10)
    assert v["factor_canonical_corr"] == pytest.approx(1.0, abs=1e-8)
    assert v["factor_canonical_corr_mean"] == pytest.approx(1.0, abs=1e-8)
    assert v["state_canonical_corr"] == pytest.approx(1.0, abs=1e-8)
    assert v["state_canonical_corr_mean"] == pytest.approx(1.0, abs=1e-8)
    assert v["impact_spearman"] == pytest.approx(1.0) and v["impact_sign_agreement"] == 1.0
    assert v["impact_spearman_all_relevant"] == pytest.approx(1.0) and v["impact_sign_agreement_all_relevant"] == 1.0
    assert v["oos_sharpe_true_mve"] == pytest.approx(v["oos_sharpe"]) and v["oos_sharpe_ratio_to_true"] == pytest.approx(1.0)
    # the implied betas cov Gamma_tilde_true R agree with the true betas up to the noise of the instruments (Eq. 5 holds
    # in population, not row by row); the metric is invariant to R, so it equals the value at R = I and passes
    v_id, _ = harness.beta_canonical_corr(result.panel, np.vstack([np.zeros((1, K)), truth.Gamma_tilde_true]), truth, "M", sim.returns.assets)
    assert v["beta_canonical_corr"] == pytest.approx(v_id["beta_canonical_corr"], abs=1e-8) and v["beta_canonical_corr"] > 0.9
    assert v["beta_canonical_corr_mean"] == pytest.approx(v_id["beta_canonical_corr_mean"], abs=1e-8)
    # the in-sample Sharpe of the rotated true factors is the rotation-invariant true-factor Sharpe
    assert v["mve_sharpe_is"] == pytest.approx(annualized_sharpe(f_true.mean().to_numpy(), f_true.cov().to_numpy(), 12.0), rel=1e-8)
    # the fitted values cov_hat Gamma_tilde_true f_true reproduce most of the true systematic return
    assert 0.5 < v["systematic_r2_recovered"] <= 1.0
    assert 0.5 < v["systematic_r2_recovered_narrative"] <= 1.0
    for check in harness.SIGNAL_CHECKS:
        assert m.passed[check], check
    assert set(m.passed) == set(harness.SIGNAL_CHECKS) and not set(m.passed) & set(harness.REPORT_ONLY_METRICS)
    for metric in harness.REPORT_ONLY_METRICS:  # reported only (D52), still one on a perfect estimate
        assert v[metric] == pytest.approx(1.0, abs=1e-8), metric
    assert m.all_passed


def test_compare_to_truth_rank_deficient_gamma_gives_nan_subspace(result, sim):
    """Fewer than K relevant rows selected: the subspace is not identified and the cosine is NaN."""
    truth = sim.truth
    fit = result.fit
    K = fit.K
    rel_idx = np.flatnonzero(truth.relevant)
    Gamma = np.zeros_like(fit.Gamma)
    Gamma[0] = fit.Gamma[0]
    Gamma[1 + rel_idx[: K - 1]] = truth.Gamma_tilde_true[rel_idx[: K - 1]]  # K-1 relevant rows only
    norms = np.linalg.norm(Gamma, axis=1)
    fit2 = replace(fit, Gamma=Gamma, gamma_norms=norms, selected=norms[1:] > 0)
    m = harness.compare_to_truth(replace(result, fit=fit2), truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)
    assert math.isnan(m.values["gamma_subspace_cos"]) and "gamma_subspace_cos" not in m.passed  # reported only (D52)
    assert math.isnan(m.values["gamma_subspace_cos_all_relevant"])
    assert m.values["gamma_relevant_rows_selected"] == K - 1
    assert m.values["selection_recall"] == pytest.approx((K - 1) / truth.relevant.sum())
    assert m.values["selection_precision"] == 1.0


def _fit_with_true_rows(fit, truth, rows: np.ndarray):
    """``fit`` with ``Gamma_tilde`` equal to ``Gamma_tilde_true`` on ``rows`` (truth positions) and zero elsewhere."""
    Gamma = np.zeros_like(fit.Gamma)
    Gamma[0] = fit.Gamma[0]
    Gamma[1 + np.asarray(rows)] = truth.Gamma_tilde_true[np.asarray(rows)]
    norms = np.linalg.norm(Gamma, axis=1)
    return replace(fit, Gamma=Gamma, gamma_norms=norms, selected=norms[1:] > 0)


def test_gamma_subspace_on_selected_rows_separates_loading_recovery_from_recall(result, sim):
    """Gamma_hat equal to the truth on K+1 of the relevant rows and zero elsewhere: the subspace on the selected
    rows is recovered exactly (cosine 1) while the all-relevant-rows comparison, which sees the zero rows, is below 1
    - it was measuring recall (< 1 here), not loading recovery."""
    truth = sim.truth
    fit = result.fit
    K = fit.K
    rel_idx = np.flatnonzero(truth.relevant)
    assert len(rel_idx) > K + 1
    chosen = rel_idx[: K + 1]
    fit2 = _fit_with_true_rows(fit, truth, chosen)
    m = harness.compare_to_truth(replace(result, fit=fit2), truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)
    v = m.values
    assert v["selection_recall"] == pytest.approx((K + 1) / len(rel_idx)) and v["selection_recall"] < 1.0
    assert v["selection_precision"] == 1.0 and v["gamma_relevant_rows_selected"] == K + 1
    assert v["gamma_subspace_cos"] == pytest.approx(1.0, abs=1e-10) and "gamma_subspace_cos" not in m.passed
    np.testing.assert_allclose(m.details["gamma_principal_cosines"], np.ones(K), atol=1e-10)
    assert m.details["gamma_rank_hat"] == K == m.details["gamma_rank_true"]
    rel = np.asarray(truth.relevant, dtype=bool)
    cos_all = np.sort(np.cos(subspace_angles(fit2.Gamma_tilde[rel], truth.Gamma_tilde_true[rel])))[::-1]
    n_inf_all = min(K, int(rel.sum()) - K)  # 5 relevant rows, K = 3: one cosine is one by dimension counting
    assert m.details["gamma_informative_cosines_all_relevant"] == n_inf_all == 2
    assert v["gamma_subspace_cos_all_relevant"] == pytest.approx(float(np.mean(np.sort(cos_all)[:n_inf_all])), abs=1e-10)
    assert v["gamma_subspace_cos_all_relevant"] < 1.0 - 1e-6
    np.testing.assert_allclose(m.details["gamma_principal_cosines_all_relevant"], cos_all, atol=1e-10)
    # selecting every relevant row (plus a noise row with a non-zero Gamma row) makes both values one
    noise_idx = np.flatnonzero(~truth.relevant & ~truth.placebo)[:1]
    fit3 = _fit_with_true_rows(fit, truth, rel_idx)
    Gamma3 = fit3.Gamma.copy()
    Gamma3[1 + noise_idx] = 0.3
    norms3 = np.linalg.norm(Gamma3, axis=1)
    fit3 = replace(fit3, Gamma=Gamma3, gamma_norms=norms3, selected=norms3[1:] > 0)
    v3 = harness.compare_to_truth(replace(result, fit=fit3), truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets).values
    assert v3["selection_recall"] == 1.0 and v3["selection_precision"] < 1.0
    assert v3["gamma_subspace_cos"] == pytest.approx(1.0, abs=1e-10) and v3["gamma_subspace_cos_all_relevant"] == pytest.approx(1.0, abs=1e-10)
    # exactly K relevant rows selected: any full-rank K x K block spans R^K, so every cosine is one whatever the
    # estimate - the comparison has no content and the metric is nan (reported only, D52: no pass flag either way)
    m4 = harness.compare_to_truth(replace(result, fit=_fit_with_true_rows(fit, truth, rel_idx[:K])), truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)
    assert math.isnan(m4.values["gamma_subspace_cos"]) and "gamma_subspace_cos" not in m4.passed
    assert m4.values["gamma_relevant_rows_selected"] == K and m4.details["gamma_informative_cosines"] == 0
    np.testing.assert_allclose(m4.details["gamma_principal_cosines"], np.ones(K), atol=1e-10)
    rng = np.random.default_rng(5)
    G_rand = np.zeros_like(fit.Gamma)
    G_rand[0] = fit.Gamma[0]
    G_rand[1 + rel_idx[:K]] = rng.standard_normal((K, K))
    norms_r = np.linalg.norm(G_rand, axis=1)
    m_rand = harness.compare_to_truth(replace(result, fit=replace(fit, Gamma=G_rand, gamma_norms=norms_r, selected=norms_r[1:] > 0)), truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)
    np.testing.assert_allclose(m_rand.details["gamma_principal_cosines"], np.ones(K), atol=1e-10)  # a random block too
    assert math.isnan(m_rand.values["gamma_subspace_cos"])
    # K + 1 rows with a random block: K - 1 cosines are one by dimension counting, the metric is the remaining one
    G_rand[1 + rel_idx[: K + 1]] = rng.standard_normal((K + 1, K))
    norms_r = np.linalg.norm(G_rand, axis=1)
    m5 = harness.compare_to_truth(replace(result, fit=replace(fit, Gamma=G_rand, gamma_norms=norms_r, selected=norms_r[1:] > 0)), truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)
    cos5 = np.sort(m5.details["gamma_principal_cosines"])[::-1]
    np.testing.assert_allclose(cos5[: K - 1], np.ones(K - 1), atol=1e-10)
    assert m5.details["gamma_informative_cosines"] == 1 and m5.values["gamma_subspace_cos"] == pytest.approx(cos5[-1], abs=1e-10)
    assert m5.values["gamma_subspace_cos"] < 1.0 - 1e-6 and m5.values["gamma_subspace_cos"] < np.mean(cos5) - 1e-6


def test_impact_metrics_over_selected_relevant_topics(result, sim):
    """The impact vector is compared over the relevant topics that were selected (a wrap-up gives a non-selected
    topic an impact of exactly zero): a perfect impact on three selected relevant topics scores one there, while the
    all-relevant version (five topics, two of them zero in the estimate) does not; below three the metric is NaN."""
    truth = sim.truth
    topics = list(sim.attention.topics)
    rel_idx = np.flatnonzero(truth.relevant)
    assert len(rel_idx) == 5

    def run(chosen):
        fit2 = _fit_with_true_rows(result.fit, truth, chosen)
        impact = pd.Series(0.0, index=topics, name="impact_z_to_mve")
        impact.iloc[chosen] = 2.5 * truth.impact_z_to_mve_true[chosen]
        wrap = replace(result.wrapup, impact_z_to_mve=impact)
        return harness.compare_to_truth(replace(result, fit=fit2, wrapup=wrap), truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)

    m = run(rel_idx[:3])
    v = m.values
    assert v["selection_recall"] == pytest.approx(0.6) and v["selection_precision"] == 1.0
    assert v["impact_spearman"] == pytest.approx(1.0) and v["impact_sign_agreement"] == 1.0 and "impact_spearman" not in m.passed
    assert m.details["n_impact_topics"] == 3 and m.details["n_impact_topics_all_relevant"] == 5
    a_all = np.zeros(5)
    a_all[:3] = 2.5 * truth.impact_z_to_mve_true[rel_idx[:3]]
    b_all = truth.impact_z_to_mve_true[rel_idx]
    assert v["impact_spearman_all_relevant"] == pytest.approx(spearmanr(a_all, b_all).statistic)
    assert v["impact_spearman_all_relevant"] < 1.0 - 1e-6
    assert v["impact_sign_agreement_all_relevant"] == pytest.approx(np.mean(np.sign(a_all) == np.sign(b_all)))
    assert v["impact_sign_agreement_all_relevant"] < 1.0
    # two selected relevant topics: undefined on the selected set, still defined over all relevant topics
    m2 = run(rel_idx[:2])
    assert math.isnan(m2.values["impact_spearman"]) and math.isnan(m2.values["impact_sign_agreement"])
    assert "impact_spearman" not in m2.passed and m2.details["n_impact_topics"] == 2  # reported only (D52)
    assert np.isfinite(m2.values["impact_spearman_all_relevant"]) and m2.details["n_impact_topics_all_relevant"] == 5
    # every relevant topic selected: the two versions coincide
    m5 = run(rel_idx)
    assert m5.values["impact_spearman"] == pytest.approx(1.0) == pytest.approx(m5.values["impact_spearman_all_relevant"])


def test_selection_recall_strong_and_weak_partition_on_hand_built_truth(result, sim):
    """Relevant rows of A with norms 3, 2, 1, 1, 0.1: the median is 1, ties count as strong (four strong, one weak);
    selecting the two strongest and the weakest gives recall_strong 2/4, recall_weak 1/1, recall 3/5."""
    truth = sim.truth
    topics = list(sim.attention.topics)
    rel_idx = np.flatnonzero(truth.relevant)  # ascending truth positions
    A = np.zeros_like(truth.A)
    for i, n in zip(rel_idx, (3.0, 2.0, 1.0, 1.0, 0.1)):
        A[i, 0] = n
    truth2 = replace(truth, A=A)
    chosen = rel_idx[[0, 1, 4]]
    fit2 = _fit_with_true_rows(result.fit, truth, chosen)
    m = harness.compare_to_truth(replace(result, fit=fit2), truth2, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)
    v = m.values
    assert v["selection_recall"] == pytest.approx(3 / 5) and v["selection_precision"] == 1.0
    assert v["selection_recall_strong"] == pytest.approx(2 / 4) and v["selection_recall_weak"] == pytest.approx(1.0)
    assert v["n_relevant_strong"] == 4 and v["n_relevant_weak"] == 1 and v["relevant_norm_median"] == 1.0
    assert m.details["strong_topics"] == [topics[i] for i in rel_idx[:4]] and m.details["weak_topics"] == [topics[rel_idx[4]]]
    assert m.details["selected_strong"] == [topics[i] for i in rel_idx[:2]] and m.details["selected_weak"] == [topics[rel_idx[4]]]
    assert not m.passed["selection_recall_strong"] and m.passed["selection_recall"]
    # selecting the four strong topics only: recall_strong 1 (passes), recall_weak 0, recall 0.8
    m4 = harness.compare_to_truth(replace(result, fit=_fit_with_true_rows(result.fit, truth, rel_idx[:4])), truth2, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)
    assert m4.values["selection_recall_strong"] == 1.0 and m4.values["selection_recall_weak"] == 0.0 and m4.passed["selection_recall_strong"]
    assert m4.values["selection_recall"] == pytest.approx(0.8)
    # the partition follows the truth's A, not the estimate: the real run's split is by the simulated norms
    norms = np.linalg.norm(truth.A, axis=1)
    med = np.median(norms[truth.relevant])
    v_real = harness.compare_to_truth(result, truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets).values
    assert v_real["n_relevant_strong"] == int(np.sum(norms[truth.relevant] >= med)) and v_real["relevant_norm_median"] == pytest.approx(med)
    # fewer than two relevant topics: the split is undefined
    one = np.zeros_like(truth.relevant)
    one[rel_idx[0]] = True
    v1 = harness.compare_to_truth(result, replace(truth, relevant=one), HarnessThresholds(), "baseline", asset_ids=sim.returns.assets).values
    assert math.isnan(v1["selection_recall_strong"]) and math.isnan(v1["selection_recall_weak"]) and math.isnan(v1["relevant_norm_median"])
    assert v1["n_relevant_strong"] == 0 and v1["n_relevant_weak"] == 0
    # all relevant norms equal (topic_null: A = 0): every relevant topic is strong, the weak half is empty
    v0 = harness.compare_to_truth(result, replace(truth, A=np.zeros_like(truth.A)), HarnessThresholds(), "baseline", asset_ids=sim.returns.assets).values
    assert v0["n_relevant_strong"] == 5 and v0["n_relevant_weak"] == 0 and math.isnan(v0["selection_recall_weak"])
    assert v0["selection_recall_strong"] == pytest.approx(v0["selection_recall"])


def test_beta_canonical_corr_is_one_for_implied_betas_up_to_rotation(result, sim):
    """True beta_{i,t} = c_{i,t-1} Gamma R on every panel row (R invertible): beta_canonical_corr == 1 for Gamma, and
    for Gamma Q with Q orthogonal; on the real run the metric is invariant to a rotation of the fit; without true
    loadings it is NaN."""
    truth = sim.truth
    panel, fit = result.panel, result.fit
    K = fit.K
    rng = np.random.default_rng(11)
    R = rng.standard_normal((K, K)) + 2.0 * np.eye(K)
    Q, _ = np.linalg.qr(rng.standard_normal((K, K)))
    Gamma_syn = rng.standard_normal((panel.p, K))  # dense, full rank: the implied betas span K dimensions per period
    id_pos = {a: i for i, a in enumerate(sim.returns.assets)}
    per_pos = {ts: i for i, ts in enumerate(truth.periods)}
    t_true = np.array([per_pos[panel.periods[t]] for t in panel.t_idx])
    a_true = np.array([id_pos[str(a)] for a in panel.assets[panel.asset_idx]])
    beta_syn = np.full(truth.beta.shape, np.nan)
    beta_syn[t_true, a_true, :] = (panel.X @ Gamma_syn) @ R
    truth_syn = replace(truth, beta=beta_syn)
    for G in (Gamma_syn, Gamma_syn @ Q):
        v, d = harness.beta_canonical_corr(panel, G, truth_syn, "M", sim.returns.assets)
        assert v["beta_canonical_corr"] == pytest.approx(1.0, abs=1e-8) and v["beta_canonical_corr_mean"] == pytest.approx(1.0, abs=1e-8)
        assert v["beta_canonical_corr_n_periods"] == 5 == len(d["periods"]) == len(d["first"]) == len(d["mean"])
        np.testing.assert_allclose(d["first"], np.ones(5), atol=1e-8)
        assert d["periods"] == [panel.periods[t].isoformat() for t in _r2_subsample(len(panel.periods))]
        assert d["n_assets"] == [int(np.sum(panel.t_idx == t)) for t in _r2_subsample(len(panel.periods))]
    # through compare_to_truth (the fit's Gamma replaced): the check passes
    norms = np.linalg.norm(Gamma_syn, axis=1)
    fit_syn = replace(fit, Gamma=Gamma_syn, gamma_norms=norms, selected=norms[1:] > 0)
    m = harness.compare_to_truth(replace(result, fit=fit_syn), truth_syn, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets)
    assert m.values["beta_canonical_corr"] == pytest.approx(1.0, abs=1e-8) and m.passed["beta_canonical_corr"]
    assert m.values["beta_canonical_corr_mean_n_assets"] == pytest.approx(np.mean([np.sum(panel.t_idx == t) for t in _r2_subsample(len(panel.periods))]))
    # a Gamma unrelated to the true betas does far worse than the reproducing one
    v_other, _ = harness.beta_canonical_corr(panel, rng.standard_normal((panel.p, K)), truth_syn, "M", sim.returns.assets)
    assert v_other["beta_canonical_corr"] < 0.9
    # the real run: invariant to an orthogonal rotation of the fit (Gamma -> Gamma Q, F -> F Q), the mean at most the first
    m_ref = harness.compare_to_truth(result, truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets).values
    m_rot = harness.compare_to_truth(replace(result, fit=fit.rotate(Q)), truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets).values
    assert 0.0 < m_ref["beta_canonical_corr_mean"] <= m_ref["beta_canonical_corr"] <= 1.0
    for key in ("beta_canonical_corr", "beta_canonical_corr_mean", "beta_canonical_corr_n_periods"):
        assert m_rot[key] == pytest.approx(m_ref[key], abs=1e-8), key
    # positional asset matching; zero true loadings (no_factor) give no canonical correlation
    v_pos, _ = harness.beta_canonical_corr(panel, Gamma_syn, truth_syn, "M", None)
    assert v_pos["beta_canonical_corr"] == pytest.approx(1.0, abs=1e-8)
    v0, d0 = harness.beta_canonical_corr(panel, fit.Gamma, replace(truth, beta=np.zeros_like(truth.beta)), "M", sim.returns.assets)
    assert all(math.isnan(x) for x in v0.values()) and d0["periods"] == []
    # the subsample rule shared with the instrument R2: every 12th position from the last, widened to five
    assert harness._subsample_indices(41).tolist() == _r2_subsample(41) and len(_r2_subsample(41)) == 5
    assert harness._subsample_indices(130).tolist() == list(range(9, 130, 12)) and harness._subsample_indices(0).size == 0
    assert harness._subsample_indices(3, min_periods=5).tolist() == [0, 1, 2]


# ---------------------------------------------------------------------------
# instrument informativeness
# ---------------------------------------------------------------------------
def _covariance_panel_of(sim_data, cfg: PipelineConfig) -> CovariancePanel:
    """Steps 1-3 of the pipeline only (no IPCA): the covariance instruments of a simulated data set."""
    aligned = align_inputs(sim_data.attention, sim_data.returns, cfg.data)
    shocks = attention_shocks(aligned.attention, cfg.shocks)
    return build_covariance_panel(shocks, aligned.returns, cfg.covariance, cfg.data.period)


def test_instrument_beta_r2_is_one_on_a_panel_linear_in_beta_and_chance_on_noise(sim):
    truth = sim.truth
    T, N, K = truth.beta.shape
    L = truth.A.shape[0]
    rng = np.random.default_rng(3)
    W = rng.standard_normal((K, L))
    values = np.einsum("tik,kl->til", np.nan_to_num(truth.beta), W) + rng.standard_normal(L)[None, None, :]
    values[~np.isfinite(truth.beta).all(axis=2)] = np.nan
    panel = CovariancePanel(
        values=values, periods=truth.periods, window_end=truth.periods, assets=np.asarray(sim.returns.assets),
        topics=np.asarray(sim.attention.topics), n_days=np.full((T, N), 20), xi=0.99,
    )
    v, d = harness.instrument_beta_r2(panel, truth, "M", sim.returns.assets)
    for kind in ("relevant", "noise", "placebo"):
        assert v[f"instrument_beta_r2_{kind}"] == pytest.approx(1.0, abs=1e-10), kind
    assert len(d["periods"]) == v["instrument_beta_r2_n_periods"] == len(_r2_subsample(T))
    assert d["periods"] == [truth.periods[t].isoformat() for t in _r2_subsample(T)]
    # a long panel takes every 12th period counted back from the last one
    long_T = 130
    long_values = np.repeat(values[:1], long_T, axis=0)
    long_periods = pd.bdate_range("2000-01-31", periods=long_T, freq="BME")
    long_beta = np.repeat(truth.beta[:1], long_T, axis=0)
    long_truth = replace(truth, beta=long_beta, periods=long_periods, f_period=pd.DataFrame(np.zeros((long_T, K)), index=long_periods))
    long_panel = replace(panel, values=long_values, periods=long_periods, window_end=long_periods, n_days=np.full((long_T, N), 20))
    v_long, d_long = harness.instrument_beta_r2(long_panel, long_truth, "M", sim.returns.assets)
    assert v_long["instrument_beta_r2_n_periods"] == len(range(long_T - 1, -1, -12)) == 11
    assert d_long["periods"][-1] == long_periods[-1].isoformat() and d_long["periods"][0] == long_periods[long_T - 1 - 120].isoformat()
    assert all(r2 == pytest.approx(1.0, abs=1e-10) for r2 in d["per_topic"].values())
    assert v["instrument_beta_r2_chance"] == pytest.approx(K / np.mean(d["n_assets"]))
    # pure noise instruments: R2 at the chance level K / N (expected K / (n - 1)), far below one
    noise = rng.standard_normal((T, N, L))
    noise[~np.isfinite(truth.beta).all(axis=2)] = np.nan
    v2, _ = harness.instrument_beta_r2(replace(panel, values=noise), truth, "M", sim.returns.assets)
    for kind in ("relevant", "noise", "placebo"):
        assert v2[f"instrument_beta_r2_{kind}"] < 3 * v2["instrument_beta_r2_chance"], kind
        assert v2[f"instrument_beta_r2_{kind}"] > 0.2 * v2["instrument_beta_r2_chance"], kind
    # the subsample is widened to at least five periods on a short panel; positional asset matching works
    short = replace(panel, values=values[:20], periods=truth.periods[:20], window_end=truth.periods[:20], n_days=np.full((20, N), 20))
    v3, d3 = harness.instrument_beta_r2(short, truth, "M", None)
    assert v3["instrument_beta_r2_n_periods"] == 5 and v3["instrument_beta_r2_relevant"] == pytest.approx(1.0, abs=1e-10)
    assert d3["periods"][-1] == truth.periods[19].isoformat()
    with pytest.raises(ValueError):
        harness.instrument_beta_r2(panel, truth, "M", list(sim.returns.assets)[:-1])


def test_instrument_beta_r2_baseline_far_above_chance_and_no_factor_at_chance(result, sim):
    """Baseline: relevant topics' instruments are linear in beta (Eq. 5), R2 well above chance K/N;
    no_factor: beta is identically zero, so every R2 is (numerically) zero, within a few multiples of chance."""
    v = result.evaluation.metrics  # noqa: F841 - the fixture run is the baseline data of TINY seed 0
    m = harness.compare_to_truth(result, sim.truth, HarnessThresholds(), "baseline", asset_ids=sim.returns.assets).values
    chance = m["instrument_beta_r2_chance"]
    assert chance == pytest.approx(3 / m["instrument_beta_r2_mean_n_assets"]) and 0.02 < chance < 0.05
    assert m["instrument_beta_r2_relevant"] > 10 * chance
    assert m["instrument_beta_r2_relevant"] > m["instrument_beta_r2_noise"] > 0.0
    nf = simulate(scenario_config("no_factor", base=TINY))
    assert nf.truth.meta["factor_structure"] is False
    panel = _covariance_panel_of(nf, tiny_pipeline_cfg())
    v_nf, _ = harness.instrument_beta_r2(panel, nf.truth, "M", nf.returns.assets)
    assert v_nf["instrument_beta_r2_chance"] == pytest.approx(chance, rel=0.1)
    for kind in ("relevant", "noise", "placebo"):
        assert 0.0 <= v_nf[f"instrument_beta_r2_{kind}"] <= 3 * v_nf["instrument_beta_r2_chance"], kind
        assert v_nf[f"instrument_beta_r2_{kind}"] < 1e-8, kind  # intercept-only regression on zero loadings
    # topic_null keeps the factor structure: the noise topics' instruments span beta (well above chance)
    tn = simulate(scenario_config("topic_null", base=TINY))
    v_tn, _ = harness.instrument_beta_r2(_covariance_panel_of(tn, tiny_pipeline_cfg()), tn.truth, "M", tn.returns.assets)
    assert v_tn["instrument_beta_r2_noise"] > 2 * v_tn["instrument_beta_r2_chance"]
    assert v_tn["instrument_beta_r2_noise"] < m["instrument_beta_r2_relevant"]


# ---------------------------------------------------------------------------
# run_harness
# ---------------------------------------------------------------------------
BASE = SimulationConfig(n_assets=100, n_topics=20, n_relevant=5, n_placebo=5, n_years=5)


def test_run_harness_fast_two_scenarios_writes_tables_artefacts_and_report(tmp_path):
    cfg = HarnessConfig(scenarios=("baseline", "no_factor"), n_seeds=1, fast=True, output_dir=str(tmp_path / "sim"))
    calls: list[tuple[int, int, str]] = []
    t0 = time.perf_counter()
    res = harness.run_harness(cfg, base=BASE, progress=lambda d, t, m: calls.append((d, t, m)))
    elapsed = time.perf_counter() - t0
    limit = 180.0 if active_backend() == "numpy" else 60.0  # the pure-numpy reference kernel is ~30x slower (72 s reported)
    assert elapsed < limit, f"fast harness took {elapsed:.0f}s (limit {limit:.0f}s on the {active_backend()} backend)"
    assert isinstance(res, HarnessResult)
    assert [c[0] for c in calls] == [0, 1, 2] and all(c[1] == 2 for c in calls) and calls[-1][2] == "done"
    # per-run table: one row per (scenario, seed), every metric, every pass flag, runtime
    pr = res.per_run
    assert list(pr["scenario"]) == ["baseline", "no_factor"] and list(pr["seed"]) == [0, 0]
    assert set(harness.METRICS) <= set(pr.columns)
    checks = [c for c, *_ in harness.CHECKS]
    assert all(f"pass_{c}" in pr.columns for c in checks)
    base_row = pr.iloc[0]
    null_row = pr.iloc[1]
    for c in harness.SIGNAL_CHECKS:
        assert base_row[f"pass_{c}"] in (0.0, 1.0), c
    for c in ("null_selection_lift", "null_oos_sharpe_abs"):
        assert math.isnan(base_row[f"pass_{c}"]) and null_row[f"pass_{c}"] in (0.0, 1.0)
    for c in harness.SIGNAL_CHECKS:  # no placebo check under no_factor (D52): a chance-level selection includes placebos
        assert math.isnan(null_row[f"pass_{c}"]), c
    for c in harness.REPORT_ONLY_METRICS:  # reported only (D52): the value column exists, the pass flag is NaN everywhere
        assert c in pr.columns and math.isnan(base_row[f"pass_{c}"]) and math.isnan(null_row[f"pass_{c}"]), c
    assert (pr["error"] == "").all() and (pr["runtime_seconds"] > 0).all()
    assert pr["all_passed"].dtype == bool
    # the no_factor truth has no signal: its lift uses the same chance level as the baseline
    assert null_row["null_selection_lift"] == pytest.approx(null_row["n_selected"] / max(1, round(0.05 * 20)))
    assert math.isnan(null_row["gamma_subspace_cos"])  # no true subspace under the null
    # ... and no factor structure: the true MVE is not spanned by returns, the instrument R2 is zero by construction
    assert null_row["factor_structure"] == 0.0 and base_row["factor_structure"] == 1.0
    assert math.isnan(null_row["oos_sharpe_true_mve"]) and math.isnan(null_row["oos_sharpe_ratio_to_true"])
    assert null_row["systematic_r2_true"] == 0.0 and math.isnan(null_row["systematic_r2_recovered"])
    for kind in ("relevant", "noise", "placebo"):
        assert abs(null_row[f"instrument_beta_r2_{kind}"]) < 1e-8
    assert base_row["instrument_beta_r2_relevant"] > 5 * base_row["instrument_beta_r2_chance"]
    assert 0.0 <= null_row["oos_selection_stability"] <= 1.0 and np.isfinite(null_row["oos_sharpe_se"])
    # strong-half recall and the implied-beta canonical correlation (Part E): defined for the baseline, the latter NaN
    # without true loadings; the all-relevant-rows subspace value is reported next to the selected-rows one
    assert 0.0 <= base_row["selection_recall_strong"] <= 1.0 and base_row["n_relevant_strong"] + base_row["n_relevant_weak"] == 5
    assert 0.0 < base_row["beta_canonical_corr"] <= 1.0 and base_row["beta_canonical_corr_mean"] <= base_row["beta_canonical_corr"] + 1e-12
    assert base_row["beta_canonical_corr_n_periods"] == 5
    assert math.isnan(null_row["beta_canonical_corr"]) and math.isnan(null_row["gamma_subspace_cos_all_relevant"])
    assert base_row["pass_beta_canonical_corr"] in (0.0, 1.0) and math.isnan(null_row["pass_beta_canonical_corr"])
    assert base_row["pass_selection_recall_strong"] in (0.0, 1.0) and math.isnan(null_row["pass_selection_recall_strong"])
    # summary: mean/std per scenario, std NaN with a single seed
    assert list(res.summary.index) == ["baseline", "no_factor"]
    assert ("selection_recall", "mean") in res.summary.columns and ("selection_recall", "std") in res.summary.columns
    assert res.summary.loc["baseline", ("selection_recall", "mean")] == base_row["selection_recall"]
    assert math.isnan(res.summary.loc["baseline", ("selection_recall", "std")])
    # passed: share of seeds per check, NaN where the check does not apply, plus 'all'
    assert list(res.passed.index) == ["baseline", "no_factor"] and list(res.passed.columns) == checks + ["all"]
    assert math.isnan(res.passed.loc["baseline", "null_selection_lift"]) and math.isnan(res.passed.loc["no_factor", "selection_recall"])
    assert res.passed.loc["no_factor", "null_oos_sharpe_abs"] in (0.0, 1.0)
    assert res.passed.loc["baseline", "all"] == float(base_row["all_passed"])
    # scenario configs carry the base sizes and the scenario overrides
    assert set(res.scenario_configs) == {"baseline", "no_factor"}
    nf_cfg = res.scenario_configs["no_factor"]
    assert nf_cfg.signal_strength == 0.0 and nf_cfg.n_assets == 100 and nf_cfg.beta_innov_sd == 0.0
    assert all(spec.beta_sd == 0.0 and set(spec.beta_mean) == {0.0} for spec in nf_cfg.asset_classes)
    assert res.scenario_configs["baseline"] == replace(BASE, seed=0)
    assert res.meta["checks"]["no_factor"] == list(harness.NULL_CHECKS) and res.meta["report_only"] == []
    assert res.thresholds == HarnessConfig().thresholds
    assert res.meta["errors"] == {} and res.meta["n_runs"] == 2 and res.meta["fast"] is True and res.meta["sizes"] == "base"
    assert res.meta["pipeline_config"]["estimation"]["lam_grid"]["n_lambdas"] == 12
    assert res.meta["pipeline_config"]["estimation"]["lam_grid"]["ratio"] == 1e-3
    assert res.meta["pipeline_config"]["oos"]["min_train_periods"] == 24
    assert res.meta["solver_backend"] == active_backend()
    # artefacts: metrics JSON (equal to the table), lambda path and gamma norms per run; tables in the output dir
    art = tmp_path / "sim" / "artefacts"
    for key in ("baseline_seed0", "no_factor_seed0"):
        for suffix in ("_metrics.json", "_lambda_path.csv", "_gamma_norms.csv"):
            assert (art / f"{key}{suffix}").is_file(), key + suffix
    js = json.loads((art / "baseline_seed0_metrics.json").read_text(encoding="utf-8"))
    assert js["scenario"] == "baseline" and js["seed"] == 0
    for metric in harness.METRICS:
        a, b = js["values"][metric], float(base_row[metric])
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b), metric
    assert js["passed"] == {c: bool(base_row[f"pass_{c}"]) for c in harness.SIGNAL_CHECKS}
    assert js["simulation"]["n_topics"] == 20 and "total_r2" in js["pipeline_metrics"]
    path = pd.read_csv(art / "baseline_seed0_lambda_path.csv")
    assert len(path) == 12 and {"lam", "criterion", "n_selected"} <= set(path.columns)
    norms = pd.read_csv(art / "baseline_seed0_gamma_norms.csv")
    assert len(norms) == 12 and {"const", "topic_001", "topic_020"} <= set(norms.columns)
    for name in ("per_run.csv", "summary.csv", "passed.csv"):
        assert (tmp_path / "sim" / name).is_file()
    back = pd.read_csv(tmp_path / "sim" / "per_run.csv", keep_default_na=False, na_values=[""])
    assert list(back["scenario"]) == ["baseline", "no_factor"]
    assert back.loc[0, "selection_recall"] == pytest.approx(base_row["selection_recall"])
    # report
    report = harness.write_report(res, cfg.output_dir)
    p = Path(report)
    assert p.is_file() and p.parent == tmp_path / "sim"
    assert p.name.startswith("harness_") and p.suffix == ".md" and len(p.stem) == len("harness_2026-01-01")
    text = p.read_text(encoding="utf-8")
    for section in ("## Setup", "## Summary per scenario", "## Instrument informativeness", "## Pass / fail",
                    "## Per-run table", "## Timings", "## Expected vs observed", "## Artefacts"):
        assert section in text, section
    assert "**baseline**" in text and "**no_factor**" in text and "n/a" in text
    assert ("Checks failed:" in text) or ("All checks passed." in text)
    assert "selection_recall" in text and "null_selection_lift" in text
    assert "| selection_recall_strong |" in text and "| beta_canonical_corr |" in text
    assert "recall of the strong half of the relevant topics near 1" in text
    assert "chance-level null" in text and "Baseline rows of this run for comparison" in text
    assert "| no_factor | 0.000 | 0.000 | 0.000 |" in text  # instrument R2 zero by construction
    # check set v2 (D52): the report-only metrics read 'reported (not identified)' in the signal column and n/a in the
    # null column, the placebo check is n/a for no_factor, the baseline paragraph lists the reported values
    pf = text.split("## Pass / fail")[1].split("## Per-run table")[0]
    for c in harness.REPORT_ONLY_METRICS:
        assert f"| {c} | {harness.REPORTED_TEXT} | n/a |" in pf, c
    assert f"| placebo_selected | {int(base_row['pass_placebo_selected'])}/1 (<= 0) | n/a |" in pf
    thr_table = text.split("Thresholds (check set")[1].split("## Summary")[0]
    assert "| gamma_subspace_cos | gamma_subspace_cos >= | reported only | n/a |" in thr_table
    para_base = text.split("**baseline** (1 seed).")[1].split("\n\n")[0]
    assert "Reported (not pass/fail): gamma_subspace_cos " in para_base and "state_canonical_corr" in para_base
    assert "not identified targets of the model" in para_base and "[>= 0.85" not in para_base
    assert "Reported (not pass/fail)" not in text.split("**no_factor** (1 seed).")[1].split("\n\n")[0]
    assert "Pass flags rescored" not in text  # a fresh run, not a rescoring
    # the setup states the solver backend that was active during the run (D46)
    assert f"- Solver backend: {active_backend()} (" in text.split("## Setup")[1].split("## Summary")[0]
    # no_factor: the true MVE Sharpe ratios are not attainable and are never printed as numbers
    nf = harness.NO_FACTOR_TEXT
    summary_section = text.split("## Summary per scenario")[1].split("## Instrument informativeness")[0]
    summary_rows = {line.split("|")[1].strip(): line for line in summary_section.splitlines() if line.startswith("| ")}
    assert summary_rows["sharpe_mve_true"] == f"| sharpe_mve_true | 1 | {nf} |"
    assert summary_rows["oos_sharpe_true_mve"].endswith(f"| {nf} |") and not summary_rows["oos_sharpe_true_mve"].startswith("| oos_sharpe_true_mve | nan")
    assert summary_rows["oos_sharpe"].endswith(f"| {harness._fmt(null_row['oos_sharpe'])} |")  # other cells untouched
    per_run_section = text.split("## Per-run table")[1].split("## Timings")[0]
    header = per_run_section.strip().splitlines()[0]
    assert header.startswith("| scenario | seed | recall | recall strong | precision | selected | placebo | beta cc | gamma cos |")
    nf_line = next(line for line in per_run_section.splitlines() if line.startswith("| no_factor | 0 |"))
    base_line = next(line for line in per_run_section.splitlines() if line.startswith("| baseline | 0 |"))
    assert f"| {nf} |" in nf_line and nf not in base_line
    para_nf = text.split("**no_factor** (1 seed).")[1].split("\n\n")[0]
    assert f"vs true {nf}" in para_nf and f"(true MVE {nf})" in para_nf and "vs true 1" not in para_nf
    assert "unstable across refits" in para_nf and "true MVE Sharpe ratios are " + nf in para_nf
    # the placebo count of the null is reported against its chance level n_selected x n_placebo / L (D52)
    chance_placebo = null_row["n_selected"] * null_row["n_placebo_topics"] / null_row["L"]
    assert f"placebo selected {harness._fmt(null_row['placebo_selected'])} (reported only, D52: a chance-level selection of " \
           f"{harness._fmt(null_row['n_selected'])} topics includes about {harness._fmt(chance_placebo, 2)} placebos" in para_nf
    assert "placebo_selected (failed" not in para_nf
    # a null Sharpe that fails the absolute 0.75 threshold on this short panel is read in standard-error units
    if null_row["pass_null_oos_sharpe_abs"] == 0.0 and abs(null_row["oos_sharpe"]) / null_row["oos_sharpe_se"] <= 2.0:
        assert "standard errors of zero" in para_nf and "calibrated for 90-100 OOS periods" in para_nf
    # stability is a diagnostic of estimation noise, not evidence of narrative information (D47, F.2)
    assert "not evidence that narratives carry information" in text.split("## Instrument informativeness")[1].split("## Pass")[0]
    # a second report of the same day gets a suffix instead of overwriting
    second = Path(harness.write_report(res, cfg.output_dir))
    assert second != p and second.name.endswith("_2.md") and p.is_file()


def test_run_harness_topic_null_is_report_only(monkeypatch, result, tmp_path):
    """topic_null: every pass flag NaN, all_passed vacuously true (1.0), the report explains the mechanism."""
    monkeypatch.setattr(harness, "run_pipeline", lambda attention, returns, cfg, **kw: result)
    cfg = HarnessConfig(scenarios=("baseline", "topic_null", "null"), n_seeds=1, fast=False, output_dir=str(tmp_path))
    res = harness.run_harness(cfg, tiny_pipeline_cfg(), base=TINY)
    pr = res.per_run.set_index("scenario")
    checks = [c for c, *_ in harness.CHECKS]
    for name in ("topic_null", "null"):
        row = pr.loc[name]
        assert all(math.isnan(row[f"pass_{c}"]) for c in checks), name
        assert row["all_passed"] == True and float(row["all_passed"]) == 1.0 and row["error"] == ""  # noqa: E712
        assert res.passed.loc[name, "all"] == 1.0 and res.passed.loc[name, checks].isna().all()
        assert res.meta["checks"][name] == []
        assert res.scenario_configs[name] == replace(TINY, signal_strength=0.0) and res.scenario_configs[name].beta_innov_sd > 0
    assert res.meta["report_only"] == ["topic_null", "null"]
    assert res.per_run["all_passed"].dtype == bool
    # the metrics JSON records an empty pass dict
    js = json.loads((tmp_path / "artefacts" / "topic_null_seed0_metrics.json").read_text(encoding="utf-8"))
    assert js["passed"] == {} and js["values"]["factor_structure"] == 1.0
    # report: n/a in every check row and in the per-run 'all passed' column, the mechanism paragraph, the baseline comparison
    text = Path(harness.write_report(res, tmp_path)).read_text(encoding="utf-8")
    assert "| all applicable checks | " in text and "1/1 (report-only, no checks)" in text
    assert "| null_selection_lift | n/a | n/a | n/a |" in text
    assert "| topic_null | 0 |" in text and "| n/a |" in text
    pf = text.split("## Pass / fail")[1].split("## Per-run table")[0]
    assert f"| gamma_subspace_cos | {harness.REPORTED_TEXT} | n/a | n/a |" in pf
    para = text.split("**topic_null** (1 seed).")[1].split("\n\n")[0]
    for phrase in ("Report-only", "G_{t,l}", "69 months", "spans beta", "cannot certify", "positive but degraded",
                   "selection above chance does not either", "real narratives must beat variance-matched noise",
                   "Observed:", "Baseline rows of this run for comparison (1 seed)", "instrument R2 on the true loadings",
                   "selection stability across refits", "Report-only scenario: no pass/fail checks apply.",
                   # D47 / F.2: the selected set is stable under topic_null too (persistent G_{t,l}); stability is a
                   # diagnostic of estimation noise, not evidence of narrative information
                   "stable across refits, as in baseline", "low only under no_factor",
                   "not evidence that narratives carry information", "pricing errors"):
        assert phrase in para, phrase
    assert "unstable" not in para and "the stability of the selected set across refits" not in para
    assert "not evidence that narratives carry information" in harness.EXPECTED["oos_selection_stability"]
    assert "unstable" not in harness.EXPECTED["oos_selection_stability"] and "unstable" not in harness.__doc__
    # selection lift is not a certificate of signal: topic_null itself selects far above chance with A = 0 (DESIGN F.2)
    assert "only selection lift" not in para
    assert "Checks failed" not in para and "All checks passed" not in para
    # an erroring topic_null run is still a failure
    def boom(attention, returns, cfg, **kw):
        if "topic_null" in cfg.name:
            raise RuntimeError("synthetic")
        return result

    monkeypatch.setattr(harness, "run_pipeline", boom)
    res2 = harness.run_harness(HarnessConfig(scenarios=("baseline", "topic_null"), n_seeds=1, output_dir=str(tmp_path / "err")), tiny_pipeline_cfg(), base=TINY)
    bad = res2.per_run.set_index("scenario").loc["topic_null"]
    assert bad["all_passed"] == False and bad["error"] and res2.passed.loc["topic_null", "all"] == 0.0  # noqa: E712


def test_run_harness_records_failed_runs_and_reuses_the_pipeline(monkeypatch, result, tmp_path):
    """A failing run is recorded (NaN metrics, all checks False), the others complete; all failing raises."""
    seen: list[str] = []

    def fake_pipeline(attention, returns, cfg, **kwargs):
        seen.append(cfg.name)
        if "no_factor" in cfg.name:
            raise ValueError("synthetic pipeline failure")
        return result  # the fixture run was made on exactly TINY seed 0 = this harness run's baseline data

    monkeypatch.setattr(harness, "run_pipeline", fake_pipeline)
    cfg = HarnessConfig(scenarios=("baseline", "no_factor"), n_seeds=1, fast=False, output_dir=str(tmp_path))
    res = harness.run_harness(cfg, tiny_pipeline_cfg(), base=TINY)
    assert seen == ["harness-baseline_seed0", "harness-no_factor_seed0"]
    pr = res.per_run
    assert list(pr["scenario"]) == ["baseline", "no_factor"]
    assert pr.loc[0, "error"] == "" and "synthetic pipeline failure" in pr.loc[1, "error"]
    assert math.isnan(pr.loc[1, "selection_recall"]) and pr.loc[1, "pass_null_selection_lift"] == 0.0
    assert math.isnan(pr.loc[1, "pass_selection_recall"]) and pr.loc[1, "all_passed"] == False  # noqa: E712
    assert res.meta["errors"] == {"no_factor_seed0": "ValueError: synthetic pipeline failure"}
    assert res.passed.loc["no_factor", "all"] == 0.0 and res.passed.loc["no_factor", "null_selection_lift"] == 0.0
    assert res.meta["sizes"] == "base" and res.meta["pipeline_name"] == "harness-test"
    # the baseline row equals a direct comparison of the fixture run with its truth
    sim = simulate(replace(TINY, seed=0), scenario="baseline")
    direct = harness.compare_to_truth(result, sim.truth, cfg.thresholds, "baseline", asset_ids=sim.returns.assets)
    for metric in harness.METRICS:
        a, b = float(pr.loc[0, metric]), direct.values[metric]
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b), metric
    assert (tmp_path / "artefacts" / "baseline_seed0_metrics.json").is_file()
    assert not (tmp_path / "artefacts" / "no_factor_seed0_metrics.json").exists()
    # the report lists the error
    text = Path(harness.write_report(res, tmp_path)).read_text(encoding="utf-8")
    assert "## Errors" in text and "synthetic pipeline failure" in text and "raised an error" in text

    def always_fails(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(harness, "run_pipeline", always_fails)
    with pytest.raises(RuntimeError, match="every harness run failed"):
        harness.run_harness(HarnessConfig(scenarios=("baseline",), n_seeds=1, output_dir=str(tmp_path / "all_fail")), base=TINY)
    with pytest.raises(ValueError):
        harness.run_harness(HarnessConfig(scenarios=(), n_seeds=1, output_dir=str(tmp_path)), base=TINY)
    with pytest.raises(TypeError):
        harness.run_harness(HarnessConfig(scenarios=("baseline",), n_seeds=1, output_dir=str(tmp_path)), {"K": 3}, base=TINY)  # type: ignore[arg-type]


def test_default_pipeline_config_fast_and_full():
    fast = harness.default_pipeline_config(fast=True)
    full = harness.default_pipeline_config(fast=False)
    assert fast.covariance.burn_in_periods == 6 and fast.estimation.lam_grid.n_lambdas == harness.FAST_N_LAMBDAS == 12
    assert fast.oos.min_train_periods == 24 and fast.estimation.lam_grid.ratio == harness.FAST_LAMBDA_RATIO
    # the fast grid spans the full range (a grid truncated at 0.05 lam_max missed the tuned lambda on small panels)
    assert harness.FAST_LAMBDA_RATIO == 1e-3 == LambdaGridConfig().ratio
    assert full.covariance.burn_in_periods == 12 and full.estimation.lam_grid.n_lambdas == 20
    assert full.oos.oos_fraction == 0.4 and full.oos.refit_every == 12 and full.oos.min_train_periods == 60
    assert full.estimation.lam_grid.ratio == LambdaGridConfig().ratio


# ---------------------------------------------------------------------------
# write_report on a hand-built result
# ---------------------------------------------------------------------------
def _synthetic_result() -> HarnessResult:
    checks = [c for c, *_ in harness.CHECKS]
    rows = []
    rng = np.random.default_rng(0)
    for scenario in ("baseline", "no_factor"):
        for seed in range(2):
            row = {"scenario": scenario, "seed": seed}
            for metric in harness.METRICS:
                row[metric] = float(rng.uniform(0.0, 1.0))
            row["n_selected"] = 8.0
            row["selection_recall"] = 0.9 if scenario == "baseline" else 0.2
            row["placebo_selected"] = 0.0
            row["null_selection_lift"] = 1.0 if scenario == "no_factor" else 8.0
            row["runtime_seconds"] = 3.0 + seed
            if scenario == "no_factor":
                # fails the absolute 0.75 threshold but sits within 1.5 standard errors of zero (a short OOS panel)
                row["oos_sharpe"] = 0.9
                row["null_oos_sharpe_abs"] = 0.9
                row["oos_sharpe_se"] = 0.6
                row["n_oos_periods"] = 36.0
            applicable = harness.scenario_checks(scenario)
            passed = harness.evaluate_checks(row, HarnessThresholds(), scenario)
            row.update({f"pass_{c}": (float(passed[c]) if c in applicable else float("nan")) for c in checks})
            row["all_passed"] = all(passed.values())
            row["error"] = ""
            rows.append(row)
    rows[-1]["error"] = "ValueError: synthetic"
    rows[-1]["all_passed"] = False
    per_run = pd.DataFrame(rows)
    grouped = per_run.groupby("scenario", sort=False)
    summary = grouped[list(harness.METRICS)].agg(["mean", "std"])
    passed = grouped[[f"pass_{c}" for c in checks]].mean()
    passed.columns = checks
    passed["all"] = grouped["all_passed"].mean()
    return HarnessResult(
        per_run=per_run, summary=summary, passed=passed,
        scenario_configs={"baseline": SimulationConfig(), "no_factor": scenario_config("no_factor")},
        thresholds=HarnessThresholds(),
        meta={"scenarios": ["baseline", "no_factor"], "seeds": [0, 1], "n_runs": 4, "n_failed": 1, "elapsed_seconds": 12.5,
              "errors": {"no_factor_seed1": "ValueError: synthetic"}, "fast": False, "sizes": "full",
              "simulation": {"baseline": harness._sim_summary(SimulationConfig())}, "artefact_dir": "x/artefacts",
              "output_dir": "x", "tables": {"per_run": "x/per_run.csv"}},
    )


def test_write_report_on_hand_built_result(tmp_path):
    res = _synthetic_result()
    path = Path(harness.write_report(res, tmp_path / "reports"))
    assert path.is_file() and path.parent == tmp_path / "reports"
    text = path.read_text(encoding="utf-8")
    assert "# Simulation harness report" in text and "4 run(s)" in text and "1 failed" in text
    assert "| baseline | 500 | 120 | 20 | 20 | 3 | 20 |" in text  # sizes table from the SimulationConfig defaults
    # summary cells carry mean ± std; the recall row shows the hand-set means
    assert "| selection_recall | 0.900 ± 0.000 | 0.200 ± 0.000 |" in text
    # pass/fail: recall passes 2/2 for the baseline and is n/a for the null; the lift is n/a for the baseline
    assert f"| selection_recall | 2/2 (>= {HarnessThresholds().selection_recall_min:.2f}) | n/a |" in text
    assert "| null_selection_lift | n/a | 2/2 (<= 2.00) |" in text
    assert "| placebo_selected | 2/2 (<= 0) | n/a |" in text  # no placebo check under no_factor (D52)
    assert "| selection_recall_strong | selection_recall_strong >= | 0.80 | n/a |" in text
    assert "| beta_canonical_corr | beta_canonical_corr >= | 0.90 | n/a |" in text
    # the report-only metrics (D52): 'reported only' in the thresholds table, 'reported (not identified)' in the
    # pass/fail table, values under 'Reported (not pass/fail)' in the baseline paragraph, still in the summary table
    for c in harness.REPORT_ONLY_METRICS:
        assert f"| {c} | {c} >= | reported only | n/a |" in text, c
        assert f"| {c} | {harness.REPORTED_TEXT} | n/a |" in text, c
        assert f"| {c} | " in text.split("## Summary per scenario")[1].split("## Instrument")[0], c
    assert f"| all applicable checks | " in text
    para_base = text.split("**baseline** (2 seeds).")[1].split("\n\n")[0]
    assert "Reported (not pass/fail): gamma_subspace_cos " in para_base and "DESIGN.md D52" in para_base
    assert "gamma_subspace_cos (failed" not in text and "state_canonical_corr (failed" not in text
    assert f"Check set {harness.CHECK_SET_VERSION} (DESIGN.md D52)" in text.split("## Pass / fail")[1]
    assert "`beta_canonical_corr` is the first canonical correlation" in text.split("## Summary per scenario")[1].split("## Instrument")[0]
    assert "## Errors" in text and "`no_factor_seed1`: ValueError: synthetic" in text
    assert "**no_factor** (2 seeds)" in text and "**baseline** (2 seeds)" in text
    assert "## Instrument informativeness" in text and "| baseline | " in text.split("## Instrument informativeness")[1]
    assert "Checks failed:" in text and "raised an error" in text
    assert "| baseline | 2 | 3.5 | 3.0 | 4.0 | 7.0 |" in text  # timings row
    assert f"Base thresholds: selection_recall_min={HarnessThresholds().selection_recall_min}" in text
    # without a recorded backend the setup line falls back to the backend that is active now
    assert f"- Solver backend: {active_backend()} (" in text
    # without a factor_structure column the no_factor scenario is recognised from its SimulationConfig
    assert not harness._has_factor_structure(res, "no_factor") and harness._has_factor_structure(res, "baseline")
    nf = harness.NO_FACTOR_TEXT
    summary_section = text.split("## Summary per scenario")[1].split("## Instrument informativeness")[0]
    for metric in ("sharpe_mve_true", "oos_sharpe_true_mve"):
        line = next(l for l in summary_section.splitlines() if l.startswith(f"| {metric} |"))
        assert line.endswith(f"| {nf} |") and "±" in line.split("|")[2], line  # baseline cell keeps mean ± std
    assert f"| no_factor | 0 |" in text and text.count(f"| {nf} |") >= 4  # per-run 'true OOS SR' of both null rows
    para_nf = text.split("**no_factor** (2 seeds).")[1].split("\n\n")[0]
    assert f"vs true {nf}" in para_nf and "vs true 0." not in para_nf
    # the null Sharpe of seed 0 (0.9, se 0.6) fails the absolute threshold but is within 1.5 standard errors of zero
    assert "Small-sample reading of null_oos_sharpe_abs - seed 0: the realised OOS Sharpe 0.90 fails the absolute " \
           "threshold (|SR| <= 0.75) but is within 1.5 standard errors of zero (se 0.60 at 36 OOS periods); " \
           "the 0.75 threshold is calibrated for 90-100 OOS periods." in para_nf
    assert "seed 1:" not in para_nf  # the erroring seed is not read
    assert "null_oos_sharpe_abs (failed 2/2 seeds)" in para_nf
    assert "placebo selected 0 (reported only, D52)" in para_nf and "placebo_selected (failed" not in para_nf
    # a rescored result says so in the setup section
    res.meta.update({"rescored": True, "rescored_at": "2026-09-07T10:00:00+00:00", "check_set": harness.CHECK_SET_VERSION})
    text2 = Path(harness.write_report(res, tmp_path / "reports2")).read_text(encoding="utf-8")
    assert "- Pass flags rescored on 2026-09-07 with check set v2 (DESIGN.md D52" in text2.split("## Setup")[1].split("## Summary")[0]
    # the note is only written when the standard-error reading differs from the absolute one
    thr = HarnessThresholds()
    ok = res.per_run[(res.per_run["scenario"] == "no_factor") & (res.per_run["error"] == "")]
    assert len(harness._null_sharpe_within_se(ok, thr)) == 1
    assert harness._null_sharpe_within_se(ok.assign(oos_sharpe=0.5, null_oos_sharpe_abs=0.5), thr) == []  # passes outright
    assert harness._null_sharpe_within_se(ok.assign(oos_sharpe=1.5, null_oos_sharpe_abs=1.5), thr) == []  # 2.5 se: a real failure
    assert harness._null_sharpe_within_se(ok.drop(columns=["oos_sharpe_se"]), thr) == []


# ---------------------------------------------------------------------------
# the study script
# ---------------------------------------------------------------------------
def _load_script():
    spec = importlib.util.spec_from_file_location("run_simulation_study", REPO / "scripts" / "run_simulation_study.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_study_script_plumbs_arguments_and_prints_the_table(monkeypatch, tmp_path, capsys):
    script = _load_script()
    seen: dict = {}
    res = _synthetic_result()

    def fake_run_harness(hcfg, pipeline_cfg=None, progress=None, base=None):
        seen["hcfg"], seen["pipeline_cfg"], seen["base"] = hcfg, pipeline_cfg, base
        if progress is not None:
            progress(0, 1, "x")
        return res

    def fake_write_report(result, out_dir):
        seen["report"] = (result, out_dir)
        return str(Path(out_dir) / "harness_x.md")

    monkeypatch.setattr(harness, "run_harness", fake_run_harness)
    monkeypatch.setattr(harness, "write_report", fake_write_report)
    code = script.main(["--scenarios", "baseline, no_factor", "--seeds", "2", "--fast", "--out", str(tmp_path), "-q"])
    assert code == 1  # the synthetic result has failures
    hcfg = seen["hcfg"]
    assert isinstance(hcfg, HarnessConfig) and hcfg.scenarios == ("baseline", "no_factor") and hcfg.n_seeds == 2
    assert hcfg.fast and hcfg.output_dir == str(tmp_path) and seen["pipeline_cfg"] is None and seen["base"] is None
    assert seen["report"] == (res, str(tmp_path))
    out = capsys.readouterr().out
    assert "report:" in out and "harness_x.md" in out and "selection_recall" in out and "some checks failed" in out
    assert "failed runs: no_factor_seed1" in out
    # a passing result exits 0; a pipeline config file is loaded and forwarded
    ok = _synthetic_result()
    ok.per_run["all_passed"] = True
    ok.meta["errors"] = {}
    monkeypatch.setattr(harness, "run_harness", lambda hcfg, pipeline_cfg=None, progress=None, base=None: ok)
    from narrative_ipca.config import save_config

    cfg_path = tmp_path / "p.json"
    save_config(tiny_pipeline_cfg(), cfg_path)
    seen.clear()

    def capture(hcfg, pipeline_cfg=None, progress=None, base=None):
        seen["hcfg"], seen["pipeline_cfg"] = hcfg, pipeline_cfg
        return ok

    monkeypatch.setattr(harness, "run_harness", capture)
    assert script.main(["--config", str(cfg_path), "--out", str(tmp_path), "-q"]) == 0
    assert seen["pipeline_cfg"] == tiny_pipeline_cfg()
    assert "all checks passed" in capsys.readouterr().out
    # a missing config file is a usage error (2), not a traceback
    assert script.main(["--config", str(tmp_path / "missing.json"), "-q"]) == 2
    parser = script.build_parser()
    ns = parser.parse_args(["--scenarios", "weak", "--seeds", "1"])
    assert ns.scenarios == "weak" and ns.seeds == 1 and not ns.fast and ns.out is None
    # without --scenarios the HarnessConfig default (both nulls included) is used; --help documents the two nulls
    seen.clear()
    monkeypatch.setattr(harness, "run_harness", capture)
    assert script.main(["--out", str(tmp_path), "-q"]) == 0
    assert seen["hcfg"].scenarios == HarnessConfig().scenarios
    assert {"no_factor", "topic_null"} <= set(HarnessConfig().scenarios)
    help_text = parser.format_help()
    assert "no_factor" in help_text and "topic_null" in help_text and "report-only" in help_text
    flat = " ".join(help_text.split())  # argparse wraps the help; compare on collapsed whitespace
    assert "baseline | no_factor" in flat and "topic_null (alias null" in flat and "| softmax | weak | balanced" in flat


# ---------------------------------------------------------------------------
# scripts/rescore_study.py on a hand-built per_run.csv
# ---------------------------------------------------------------------------
def _load_rescore_script():
    spec = importlib.util.spec_from_file_location("rescore_study", REPO / "scripts" / "rescore_study.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stale_per_run_rows() -> list[dict]:
    """Two scenarios x two seeds with hand-set metrics and stale (check set v1) flags: everything 0.0, all_passed False."""
    checks = [c for c, *_ in harness.CHECKS]
    good = {
        "selection_recall": 0.6, "selection_recall_strong": 0.9, "selection_precision": 0.8, "placebo_selected": 0.0,
        "beta_canonical_corr": 0.97, "factor_canonical_corr": 0.99, "oos_sharpe_ratio_to_true": 0.7, "systematic_r2_recovered": 0.9,
        # far below the (disabled) v1 thresholds: must not fail anything under check set v2
        "gamma_subspace_cos": 0.3, "state_canonical_corr": 0.4, "impact_spearman": float("nan"),
        "null_selection_lift": 9.0, "null_oos_sharpe_abs": 0.9,
    }
    variants = (
        ("baseline", 0, {}),
        ("baseline", 1, {"selection_precision": 0.2, "placebo_selected": 3.0}),
        ("no_factor", 0, {"null_selection_lift": 1.0, "null_oos_sharpe_abs": 0.2, "placebo_selected": 1.0, "n_selected": 3.0}),
        ("no_factor", 1, {"null_selection_lift": 2.5, "null_oos_sharpe_abs": 0.2, "placebo_selected": 0.0, "n_selected": 7.0}),
    )
    rows = []
    for scenario, seed, over in variants:
        row: dict = {"scenario": scenario, "seed": seed}
        row.update({metric: float("nan") for metric in harness.METRICS})
        row.update(good)
        row.update({"n_selected": 12.0, "n_placebo_topics": 5.0, "L": 20.0, "runtime_seconds": 5.0 + seed, "factor_structure": 0.0 if scenario == "no_factor" else 1.0})
        row.update(over)
        row.update({f"pass_{c}": 0.0 for c in checks})
        row["all_passed"] = False
        row["harness_seconds"] = 5.5 + seed
        row["error"] = ""
        rows.append(row)
    return rows


def test_rescore_study_recomputes_flags_from_per_run_csv(tmp_path, capsys):
    script = _load_rescore_script()
    checks = [c for c, *_ in harness.CHECKS]
    out = tmp_path / "study"
    var = out / "bks"
    art = var / "artefacts"
    art.mkdir(parents=True)
    pd.DataFrame(_stale_per_run_rows()).to_csv(var / "per_run.csv", index=False)
    stale_report = var / "harness_2026-01-01.md"
    stale_report.write_text("stale report", encoding="utf-8")
    (var / "notes.md").write_text("keep me", encoding="utf-8")
    (art / "baseline_seed0_metrics.json").write_text("{}", encoding="utf-8")
    # the artefact is the primary record: its (exact) values replace the CSV's; 7/27 needs 17 digits, which pandas'
    # default parser gets wrong by an ulp; keys that are not columns are ignored
    artefact_b1 = json.dumps({"values": {"selection_precision": 7 / 27, "not_a_column": 1.0}})
    (art / "baseline_seed1_metrics.json").write_text(artefact_b1, encoding="utf-8")
    (out / "study_run.log").write_text(
        "2026-09-06 17:05:23,131 run_full_study INFO variant bks: 4 runs on 8 workers (3 threads each)\n"
        "2026-09-06 17:05:50,000 narrative_ipca.harness INFO write_report: study\\other\\harness_2026-09-06.md\n"
        "2026-09-06 17:06:00,131 narrative_ipca.harness INFO write_report: study\\bks\\harness_2026-09-06.md\n"
        "2026-09-06 17:06:00,200 run_full_study INFO variant tol02: 4 runs on 8 workers (3 threads each)\n",
        encoding="utf-8",
    )
    # the timings of the original run come from the driver log (start line to the variant's write_report line)
    assert script.timings_from_log(out, "bks") == (datetime(2026, 9, 6, 17, 5, 23), 37.0)
    started, elapsed = script.timings_from_log(out, "tol02")
    assert started == datetime(2026, 9, 6, 17, 6, 0) and math.isnan(elapsed)  # started, no report line
    s_none, e_nan = script.timings_from_log(out, "loocv")  # variant absent from the log
    assert s_none is None and math.isnan(e_nan)
    assert math.isnan(script.timings_from_log(tmp_path / "nowhere", "bks")[1])

    assert script.main(["--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "=== variant bks: rescored with check set v2" in printed and "null_selection_lift" in printed
    assert "reported only (no pass flag): gamma_subspace_cos, state_canonical_corr, impact_spearman" in printed

    # per_run.csv: flags recomputed from the metrics with the v2 check set, metrics untouched (bit-exact, the
    # artefact's value restored where one exists), column order kept
    back = pd.read_csv(var / "per_run.csv", float_precision="round_trip")
    assert list(back["scenario"]) == ["baseline", "baseline", "no_factor", "no_factor"] and list(back["seed"]) == [0, 1, 0, 1]
    assert list(back.columns[:2]) == ["scenario", "seed"] and list(back.columns[-3:]) == ["all_passed", "harness_seconds", "error"]
    assert back.columns.get_loc("pass_selection_recall") < back.columns.get_loc("all_passed")
    assert "not_a_column" not in back.columns
    b0, b1, n0, n1 = (back.iloc[i] for i in range(4))
    assert b0["selection_recall"] == 0.6 and b1["selection_precision"] == 7 / 27 and n0["null_selection_lift"] == 1.0
    assert b0["gamma_subspace_cos"] == 0.3 and math.isnan(b0["impact_spearman"])  # values kept, ...
    for c in harness.REPORT_ONLY_METRICS:  # ... no pass flag anywhere
        assert back[f"pass_{c}"].isna().all(), c
    for c in harness.SIGNAL_CHECKS:
        assert b0[f"pass_{c}"] == 1.0, c
        assert math.isnan(n0[f"pass_{c}"]) and math.isnan(n1[f"pass_{c}"]), c
    assert b1["pass_selection_precision"] == 0.0 and b1["pass_placebo_selected"] == 0.0 and b1["pass_selection_recall"] == 1.0
    assert math.isnan(n0["pass_placebo_selected"])  # no placebo check under no_factor even with a placebo selected (D52)
    assert n0["pass_null_selection_lift"] == 1.0 and n0["pass_null_oos_sharpe_abs"] == 1.0
    assert n1["pass_null_selection_lift"] == 0.0 and n1["pass_null_oos_sharpe_abs"] == 1.0
    for row in (b0, b1):
        assert math.isnan(row["pass_null_selection_lift"]) and math.isnan(row["pass_null_oos_sharpe_abs"])
    assert list(back["all_passed"]) == [True, False, True, False] and back["all_passed"].dtype == bool
    assert back["error"].isna().all() and list(back["harness_seconds"]) == [5.5, 6.5, 5.5, 6.5]
    # passed.csv / summary.csv rebuilt as run_full_study.assemble does
    passed = pd.read_csv(var / "passed.csv", index_col=0)
    assert list(passed.index) == ["baseline", "no_factor"] and list(passed.columns) == checks + ["all"]
    assert passed.loc["baseline", "selection_recall"] == 1.0 and passed.loc["baseline", "selection_precision"] == 0.5
    assert passed.loc["baseline", "placebo_selected"] == 0.5 and math.isnan(passed.loc["no_factor", "placebo_selected"])
    assert passed.loc["no_factor", "null_selection_lift"] == 0.5 and passed.loc["no_factor", "null_oos_sharpe_abs"] == 1.0
    assert math.isnan(passed.loc["baseline", "null_selection_lift"]) and passed["gamma_subspace_cos"].isna().all()
    assert list(passed["all"]) == [0.5, 0.5]
    summary = pd.read_csv(var / "summary.csv", header=[0, 1], index_col=0)
    assert list(summary.index) == ["baseline", "no_factor"]
    assert summary.loc["baseline", ("selection_recall", "mean")] == pytest.approx(0.6)
    assert summary.loc["baseline", ("selection_precision", "std")] == pytest.approx(np.std([0.8, 7 / 27], ddof=1))
    # the old report is gone, exactly one new report of today exists, other files and the artefacts are untouched
    assert not stale_report.exists() and (var / "notes.md").read_text(encoding="utf-8") == "keep me"
    assert (art / "baseline_seed0_metrics.json").read_text(encoding="utf-8") == "{}"
    assert (art / "baseline_seed1_metrics.json").read_text(encoding="utf-8") == artefact_b1
    reports = sorted(var.glob("harness_*.md"))
    assert [p.name for p in reports] == [f"harness_{datetime.now():%Y-%m-%d}.md"]
    text = reports[0].read_text(encoding="utf-8")
    today = f"{datetime.now():%Y-%m-%d}"
    setup = text.split("## Setup")[1].split("## Summary")[0]
    assert f"rescored on {today} with check set v2" in setup and "37.0 s wall clock" in text
    assert "Pipeline config `study-bks`: K = 3, criterion = is_sharpe" in setup
    assert "| baseline | 500 | 120 | 20 | 20 | 3 | 20 |" in setup  # sizes rebuilt from scenario_config
    pf = text.split("## Pass / fail")[1].split("## Per-run table")[0]
    assert "| selection_precision | 1/2 (>= 0.60) | n/a |" in pf and "| placebo_selected | 1/2 (<= 0) | n/a |" in pf
    assert "| null_selection_lift | n/a | 1/2 (<= 2.00) |" in pf and "| all applicable checks | 1/2 | 1/2 |" in pf
    assert f"| gamma_subspace_cos | {harness.REPORTED_TEXT} | n/a |" in pf
    para_nf = text.split("**no_factor** (2 seeds).")[1].split("\n\n")[0]
    assert "placebo selected 0.500 (min 0, max 1) (reported only, D52: a chance-level selection of 5 topics includes about 1.25 placebos" in para_nf
    assert "Checks failed: null_selection_lift (failed 1/2 seeds)." in para_nf
    assert "Artefacts" in text and "baseline_seed0_metrics.json" not in text  # artefact dir named, files not listed
    # a second pass is idempotent on the tables and replaces the report again
    assert script.main(["--out", str(out), "--variants", "bks"]) == 0
    capsys.readouterr()
    again = pd.read_csv(var / "per_run.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(again, back)
    assert [p.name for p in sorted(var.glob("harness_*.md"))] == [f"harness_{today}.md"]
    # rescore_rows on its own: an erroring run keeps 0.0 on its applicable checks; a report-only scenario is
    # vacuously all_passed with NaN everywhere; an unknown variant directory falls back to the default config
    rows = _stale_per_run_rows()
    rows[0]["error"] = "ValueError: synthetic"
    rows[2]["scenario"] = "topic_null"
    script.rescore_rows(rows, HarnessThresholds())
    assert rows[0]["all_passed"] is False and rows[0]["pass_selection_recall"] == 0.0 and math.isnan(rows[0]["pass_null_selection_lift"])
    assert rows[2]["all_passed"] is True and all(math.isnan(rows[2][f"pass_{c}"]) for c in checks)
    assert script.pipeline_config_of("bks").tuning.criterion == "is_sharpe" and script.pipeline_config_of("loocv").tuning.criterion == "loocv_sharpe"
    assert script.pipeline_config_of("custom").name == "study-custom"
    assert script.find_variants(out) == ["bks"] and script.find_variants(tmp_path / "nowhere") == []
    with pytest.raises(SystemExit):
        script.main(["--out", str(tmp_path / "nowhere")])


def test_rescore_study_timings_follow_the_latest_run(tmp_path):
    """A re-run of one variant appends its log; the timings come from that run, not the first one."""
    script = _load_rescore_script()
    (tmp_path / "study_run.log").write_text(
        "2026-09-06 17:17:14,908 run_full_study INFO variant tol02: 15 runs on 8 workers (3 threads each)\n"
        "2026-09-06 17:30:00,000 narrative_ipca.harness INFO write_report: study\\tol02\\harness_2026-09-06.md\n"
        "2026-09-06 17:31:00,000 run_full_study INFO variant loocv: 15 runs on 8 workers (3 threads each)\n"
        "2026-10-02 08:35:45,960 run_full_study INFO variant tol02: 15 runs on 4 workers (3 threads each)\n"
        "2026-10-02 08:40:00,000 narrative_ipca.harness INFO write_report: study\\loocv\\harness_2026-10-02.md\n"
        "2026-10-02 09:00:30,129 narrative_ipca.harness INFO write_report: C:\\tmp\\tol02\\harness_2026-10-02.md\n"
        "2026-10-02 09:10:00,000 narrative_ipca.harness INFO write_report: study\\tol02\\harness_2026-10-03.md\n",
        encoding="utf-8",
    )
    assert script.timings_from_log(tmp_path, "tol02") == (datetime(2026, 10, 2, 8, 35, 45), 1485.0)
    started, elapsed = script.timings_from_log(tmp_path, "loocv")  # its report line follows another variant's start
    assert started == datetime(2026, 9, 6, 17, 31, 0) and elapsed == (datetime(2026, 10, 2, 8, 40) - started).total_seconds()
    with (tmp_path / "study_run.log").open("a", encoding="utf-8") as f:  # a later re-run that crashed before its report
        f.write("2026-10-03 10:00:00,000 run_full_study INFO variant tol02: 15 runs on 4 workers (3 threads each)\n")
    assert script.timings_from_log(tmp_path, "tol02") == (datetime(2026, 10, 2, 8, 35, 45), 1485.0)


def test_rescore_study_reads_metrics_exactly_and_prefers_the_artefact(tmp_path):
    """load_rows parses floats exactly; refresh_metrics_from_artefact restores the artefact's values, ignores junk."""
    script = _load_rescore_script()
    p = tmp_path / "per_run.csv"
    pd.DataFrame([{"scenario": "baseline", "seed": 0, "selection_precision": 7 / 27, "n_selected": 12.0, "error": ""}]).to_csv(p, index=False)
    (row,) = script.load_rows(p)
    assert row["selection_precision"] == 7 / 27  # 17 significant digits survive the CSV round trip
    assert row["seed"] == 0 and row["error"] == ""
    art = tmp_path / "baseline_seed0_metrics.json"
    assert script.refresh_metrics_from_artefact(row, art) == 0  # no artefact: row untouched
    art.write_text(json.dumps({"values": {"selection_precision": 0.25, "n_selected": 3, "seed": 9, "other": 1.0, "s": "x", "none": None}}), encoding="utf-8")
    assert script.refresh_metrics_from_artefact(row, art) == 2  # selection_precision and n_selected; seed is identity
    assert row["selection_precision"] == 0.25 and row["n_selected"] == 3.0 and row["seed"] == 0 and "other" not in row
    for junk in ("{}", "[1, 2]", "", "{\"values\": 5}"):
        art.write_text(junk, encoding="utf-8")
        assert script.refresh_metrics_from_artefact(row, art) == 0, junk
    assert row["selection_precision"] == 0.25

