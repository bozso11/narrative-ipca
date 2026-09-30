"""BKS trace: the lab's BKS run recomputed step by step, next to what each step should give (DESIGN.md G.16; D90).

The BKS trace page (``dashboard/bks_trace_page.py``) follows one cached BKS
run of the lab (:mod:`.bks`) from the simulated inputs to the implied topic
sensitivities. Nothing here changes a value: every quantity is either read
from the cached stages (panel, fit, forecast evaluation, implied
sensitivities) or recomputed independently from their inputs, and the two
are compared (:class:`TraceCheck`). Each step also gets a reference for what
it should give, from the simulation's population truth where one exists.

Steps (:data:`STEPS`)
---------------------
1. Inputs: simulated attention ``a``, market returns ``r`` and the true
   sensitivities ``B_true`` of :func:`.dgp.truth_for_window`.
2. Alignment and scaling (:func:`narrative_ipca.data.align_inputs`):
   attention lagged by the lead ``l``; returns divided by the divisor
   ``d_{i,tau}`` (trailing 252-day volatility under the full history, the
   asset's training standard deviation under the training history, 1
   without asset weighting).
3. Attention shocks ``z_tau = a_tau - mean(a_{tau-1}, ..., a_{tau-w})`` on
   the panel's days. The panel does not keep them; :func:`build_trace`
   recomputes them exactly (``BKSTrace.z``).
4. Instruments: the kernel covariance ``cov_{i,j}`` of the scaled return
   with ``z`` over every day up to the window end of week ``j``, day
   ``tau`` weighted ``xi^(j - week(tau))`` and the last ``skip_days`` day(s)
   of week ``j`` left out (BKS App. B.1, D11-D13).
5. Weekly panel (:func:`narrative_ipca.panel.build_panel`): rows
   ``(c_{i,t-1} = [1, cov_{i,t-1}], y_{i,t})``, ``y`` the week's sum of
   scaled daily returns.
6. Fit and lambda: Sparse IPCA (BKS Eq. 8) on the training weeks, lambda
   from the regularisation path (tolerance rule or argmax) or fixed.
7. Forecast weeks: ``f_t = (B'B + ridge I)^-1 B'y`` with ``B = C Gamma`` from
   the week's own cross-section (ridge 2, or 0 at ``lambda = 0``).
8. Implied sensitivities (BKS Eq. 5, :func:`.bks.implied_exposures`):
   ``m_i = P c~_i + M Gamma_0'``, ``b_i = Sigma_z^+ (d_i m_i)``,
   ``B_hat[k, i] = b_i[k] sd_train(z_k) / sd_train(r_i)``.

Symbols: ``i`` assets, ``k`` topics (``L`` of them), ``K`` factors;
``c~_i`` the asset's instrument row without the constant (panel units: a
covariance of the scaled return with the raw shock), taken in the last
training week; ``Gamma_tilde = Gamma[1:]`` and ``Gamma_0 = Gamma[0]``;
``M = Gamma_tilde (Gamma_tilde' Gamma_tilde)^+`` and ``P = M Gamma_tilde'``
the orthogonal projector onto the ``K`` directions of ``Gamma_tilde`` in
topic space; ``d_i`` the unit conversion (the asset's mean divisor over its
training return days); ``Sigma_z`` the covariance (``ddof = 0``) of the raw
shocks over the direct estimator's training days.

Units. Topic x asset matrices are topics x assets (``topic_id`` rows) in the
simulation's topic order (``sim.topics.ids``), except the instrument frames,
which are assets x topics like ``IPCAPanel.X``. Sensitivities (``B_hat``,
the ladder variants, ``B_true``) are in the direct estimator's standardised
units; instruments and ``m`` are in panel units (scaled return x raw shock);
``m_ret`` and ``b_raw`` in return units.

The reference ladder (:data:`LADDER`, ``BKSTrace.ladder``) scores, like the
Compare tab, a chain of sensitivity matrices from what the data allow to
what BKS delivers, so the step that loses the signal stands out.

Validity boundaries
-------------------
* Population references (``population``) are the simulation's full-sample
  moments (:func:`.dgp.truth_for_window`); the lab's listed and generic
  assets are complete, and with missing returns ``B_true`` is solved per
  observation group while these moments pool all days.
* ``instrument_ref`` (the instruments' population value in panel units) is
  ``Cov(z_k, r_i) / d_i``: exact under the training history (constant
  divisor) and without weighting; approximate under the full history, where
  the trailing volatility varies over the kernel's days (measured on the
  dashboard defaults: correlation 0.97-0.98 and slope 1.0-1.07). A short
  half-life or the training history makes the instrument a small-sample
  quantity: compare it with the window truth or its signal part instead.
* The window truth assumes the noise terms at their population variance
  (only the signal's own moments are taken over the training days).
* The split of ``z`` into signal, news and slow parts ignores the attention
  floor at 1e-6 (``sim.meta["n_clipped"]`` days are not exact).
* The "kernel" ``Sigma_z`` is taken over the kernel-weighted days of the
  last training week's instruments; stale assets (an earlier row) use it too.
* ``B_true`` is full-sample standardised while the estimators use training
  scales (D74); ``B_true_train_units`` converts it.

Cost: about 0.3 s on the dashboard defaults (55 assets x 20 topics), most
of it the per-lambda refit of the path (``path_trace``), which is skipped
above :data:`PATH_TRACE_MAX_TOPICS` topics; the ``lambda_max`` recompute is
skipped above :data:`LAMBDA_MAX_MAX_TOPICS` topics; 2 to 10 s at 500 x 500
(dense decompositions of 500 x 500 matrices, slower on a busy machine). The
per-selection helpers take milliseconds to a third of a second. Nothing is cached here;
:meth:`.session.LabSession.bks_trace` caches the result.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .. import sparse_ipca as si
from ..covariances import brute_force_covariance, kernel_weights, window_bounds
from ..data import period_end_index
from ..oos import RIDGE
from ..shocks import attention_shocks
from ..wrapup import _lsq_map, _sym_pinv
from . import bks as lab_bks
from .bks import BKSFit, BKSPanel
from .config import BKSLabConfig, WindowConfig
from .dgp import _cross_cov, _var_filtered_slow
from .direct import MIN_TRAIN_OBS, _train_moments, shock_matrix, training_pairs
from .evaluate import evaluate_window, median_finite
from .types import BKSLabResult, DirectFit, ObservedShocks, SimData, SimTruth

logger = logging.getLogger(__name__)

__all__ = [
    "STEPS",
    "STEP_WHAT",
    "OK",
    "OFF",
    "INFO",
    "IDENTITY",
    "DIAGNOSTIC",
    "LADDER",
    "LADDER_WHAT",
    "PATH_TRACE_MAX_TOPICS",
    "LAMBDA_MAX_MAX_TOPICS",
    "TraceCheck",
    "BKSTrace",
    "build_trace",
    "population_moments",
    "window_truth",
    "asset_days",
    "topic_days",
    "instrument_series",
    "kernel_profile",
    "instrument_row",
    "pairing_table",
    "design_matrix",
    "forecast_week",
    "asset_chain",
    "lambda_path_trace",
]

#: Steps of the trace, in pipeline order: key -> page label.
STEPS: dict[str, str] = {
    "summary": "Summary",
    "inputs": "1 Inputs",
    "align": "2 Alignment and scaling",
    "shocks": "3 Attention shocks",
    "instruments": "4 Instruments",
    "panel": "5 Weekly panel",
    "fit": "6 Fit and lambda",
    "forecast": "7 Forecast weeks",
    "implied": "8 Implied sensitivities",
}

#: What each step computes, in one plain-English line (the status table).
STEP_WHAT: dict[str, str] = {
    "inputs": "Simulated attention, market returns and the true topic sensitivities the run starts from.",
    "align": "Attention lagged by the lead; each return divided by its volatility divisor.",
    "shocks": "Daily attention shocks: attention minus its mean over the previous w days.",
    "instruments": "Kernel-weighted covariances of each asset's scaled return with the shocks, one per week.",
    "panel": "Weekly rows pairing last week's instruments with this week's return.",
    "fit": "Sparse IPCA on the training weeks, with lambda chosen on the path.",
    "forecast": "Each forecast week's factors from its own cross-section and the frozen training fit.",
    "implied": "The topic sensitivities the training fit implies (BKS Eq. 5).",
}

#: Check status words (tables show words, not symbols).
OK, OFF, INFO = "ok", "off", "info"

#: Check kinds: an identity must hold up to rounding (off = an implementation defect); a diagnostic
#: describes the run (off = the run departs from what the method assumes, not a bug).
IDENTITY, DIAGNOSTIC = "identity", "diagnostic"

#: Ladder variants, in chain order from what the data allow to what BKS delivers (key -> page label).
LADDER: dict[str, str] = {
    "oracle": "True sensitivities",
    "window_truth": "Best a training window allows",
    "instruments_kernel": "Instruments, shocks' covariance over the same history",
    "instruments_train": "Instruments, shocks' covariance over the training days",
    "best_rank": "Best K directions of the instruments",
    "bks_no_const": "BKS directions, no constant",
    "bks_implied": "BKS-implied (production)",
}

#: What each ladder variant is, in one plain-English sentence (the ladder table's ``what`` column).
LADDER_WHAT: dict[str, str] = {
    "oracle": "The simulation's true topic sensitivities: the ceiling.",
    "window_truth": (
        "The population sensitivity of the training window's own signal: what a perfect estimator on these "
        "days would find."
    ),
    "instruments_kernel": (
        "Each asset's instruments used as its topic covariances, divided by the shocks' covariance over the "
        "same kernel-weighted days."
    ),
    "instruments_train": (
        "The same instruments divided by the shocks' covariance over the training days, as the production "
        "step does."
    ),
    "best_rank": "The instruments kept only in their best K directions: the most any K-direction fit can keep.",
    "bks_no_const": "The instruments kept only in the K directions the BKS fit chose, without the constant's part.",
    "bks_implied": "The production BKS-implied sensitivities (BKS Eq. 5).",
}

#: Above this many topics the per-lambda refit of the path (``BKSTrace.path_trace``) is skipped (cost).
PATH_TRACE_MAX_TOPICS = 60

#: Above this many topics the ``lambda_max`` recompute is skipped (one warm start plus a bisection).
LAMBDA_MAX_MAX_TOPICS = 100

#: Plain-words name of each lambda rule (the settings table).
_RULE_LABELS: dict[str, str] = {
    "tolerance": "tolerance: the sparsest lambda within the tolerance of the best in-sample Sharpe ratio",
    "argmax": "argmax: the best in-sample Sharpe ratio",
    "fixed": "fixed lambda",
}

#: Weeks of the pairing table before the first training week.
_PAIRING_WEEKS_BEFORE = 4

#: Tolerance of the KKT ratio and the penalty-ridge balance (the ARLS stops on a relative objective change).
_KKT_TOL = 5e-3

#: Minimum eigenvalue ratio of Sigma_ff below which a factor counts as dead (the rcond of the Sharpe criterion).
_DEAD_RATIO = 1e-6

#: Instruments-vs-population correlation below which the full-history instruments depart from their reference.
_MIN_REF_CORR = 0.9


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass
class TraceCheck:
    """One check of the trace: an observed statistic against what it should be.

    Attributes
    ----------
    step:
        Key of :data:`STEPS`.
    name:
        Short plain-English name, e.g. "Scaled return x divisor = raw return".
    relation:
        What should hold, in words or a formula.
    value:
        The observed statistic (a largest difference, a ratio, a count or a
        correlation).
    reference:
        What it should be (``NaN`` when the check is a pure difference).
    tolerance:
        Threshold used for the status (``NaN`` for ``"info"``).
    status:
        ``"ok"``, ``"off"`` or ``"info"``.
    note:
        One sentence: why the check is there and what it means when off.
    kind:
        ``"identity"`` (must hold up to rounding; off points to an
        implementation defect) or ``"diagnostic"`` (describes the run; off
        means the run departs from what the method assumes).
    """

    step: str
    name: str
    relation: str
    value: float
    reference: float
    tolerance: float
    status: str
    note: str = ""
    kind: str = IDENTITY


@dataclass
class BKSTrace:
    """The lab's BKS run traced step by step (module docstring; DESIGN.md G.16).

    Frames of topic x asset matrices are topics x assets in the simulation's
    topic order (``topics``) unless noted; instrument frames are assets x
    topics. Everything is a fresh object (never a view of a cached stage).

    Attributes
    ----------
    history, lead_days, shock_window, K, lam, lambda_rule:
        Identification of the run (``lambda_rule`` ``"tolerance"``,
        ``"argmax"`` or ``"fixed"``).
    topics, assets:
        Simulation topic order (``sim.topics.ids``) and asset order.
    panel_topics:
        The panel's instrument order without ``"const"``.
    train_periods, forecast_periods:
        Week ends of the training weeks of the fit and of the evaluated
        forecast weeks.
    row_week:
        Return week whose rows the Eq. 5 step uses (the last training week).
    instrument_week:
        Week end of that row's instruments (the week before ``row_week``).
    instrument_window_end:
        Last day inside that kernel window (``skip_days`` before the week end).
    z:
        Daily BKS shocks on ``panel.aligned.calendar``, panel topic order
        (exact recompute; ``NaN`` where the panel had none).
    week_ends:
        Period ends of ``panel.aligned.calendar`` (instrument weeks).
    population:
        ``var_z`` (topics x topics), ``sd_z`` (Series), ``cov_z_r`` (topics
        x assets, return units), ``C`` (topics x assets, standardised),
        ``S_z`` (topics x topics), ``instrument_ref`` (assets x topics, panel
        units), ``instrument_divisor`` (Series ``d_i`` of the reference) and
        ``instrument_ref_note``.
    window_truth:
        Topics x assets (:func:`window_truth`).
    B_true_train_units:
        ``B_true`` converted to the estimators' training scales.
    instruments:
        Assets x topics: ``c~_i`` used by Eq. 5 (panel units; ``NaN`` rows
        for assets without a training row).
    chain:
        The Eq. 5 chain: ``beta`` (assets x f1..fK), ``m_proj``, ``m``
        (assets x topics, panel units), ``m_const`` (Series over topics),
        ``divisor`` and ``conversion`` (Series over assets; the conversion is
        0 for skipped assets), ``m_ret``, ``b_raw`` (assets x topics, return
        units), ``sigma_z`` (dict ``train`` / ``kernel`` / ``population`` ->
        topics x topics), ``B_hat``, ``B_const`` (topics x assets),
        ``ret_scale``, ``z_scale`` (Series), ``projector`` (topics x
        topics), ``gamma_rank``, ``sigma_z_rank``, ``row_period`` (Series
        over assets), ``skipped_assets``, ``stale_assets`` (lists).
    variants:
        :data:`LADDER` key -> topics x assets (standardised units).
    ladder:
        Index :data:`LADDER` keys (name ``variant``); columns ``label``,
        ``spearman``, ``rmse``, ``median_r2`` (fraction), ``d_spearman``,
        ``d_median_r2`` (change from the row above) and ``what``.
    capture:
        ``kept_share``, ``best_share``, ``random_share`` (floats),
        ``singular_share`` and ``captured`` (Series over directions
        1..min(n, L)), ``principal_cosines`` (array of ``K``; zeros when
        ``Gamma_tilde`` has fewer than ``K`` directions).
    path:
        Lambda path table, ``None`` for the fixed rule: columns ``lam``,
        ``criterion``, ``se``, ``in_band``, ``best``, ``chosen``,
        ``n_selected``, ``total_r2``, ``objective``, ``zero_objective``,
        ``above_zero``, ``converged``, ``n_iter``, ``sigma_ff_truncated``.
    gamma_path:
        Lambda (index ``lam``) x instruments: ``sigma^c_l ||Gamma_l||`` per
        path point (``None`` for the fixed rule).
    path_trace:
        Per-lambda refit (:func:`lambda_path_trace`) or ``None`` (fixed rule,
        or skipped above :data:`PATH_TRACE_MAX_TOPICS` topics).
    gamma_std:
        Instruments (``const`` + panel topics) x f1..fK: ``sigma^c_l Gamma_lk``.
    kkt:
        Instruments x ``ratio`` (``||grad_l|| / pen_l``, ``NaN`` where the
        penalty is 0), ``active`` (bool), ``penalty`` (``lambda N_S
        sigma^c_l``).
    factors_in_sample:
        Training weeks x f1..fK (``fit.fit.F``).
    units:
        Per asset: ``divisor``, ``ret_scale``, ``asset_vol``,
        ``divisor_over_ret_scale``, ``conversion`` (``u_i``, 1 is exact),
        ``skipped``.
    shock_table:
        Per topic: ``sd_train``, ``sd_population``, ``ratio``,
        ``corr_designed``, ``attenuation_true``, ``signal_share``.
    stability:
        Per topic (simulation order): ``within_over_cross`` and
        ``mean_over_sd`` of the training instruments.
    weeks:
        Per forecast week (index ``period``): ``first_day``, ``n_assets``,
        ``r2``, ``r2_shuffled``, ``f1..fK``.
    checks:
        Every :class:`TraceCheck`, in step order.
    findings:
        ``{"step", "severity" ("departure" | "note"), "title", "text"}``.
    meta:
        ``timings``, ``notes``, ``settings`` and ``shapes`` (tables for the
        page), ``key_numbers`` (step -> text), ``train_start``,
        ``train_end``, ``forecast_start``, ``forecast_end``, ``xi``,
        ``skip_days``, ``min_days``, ``burn_in_weeks``,
        ``kernel_share_before_train``, ``effective_days``, ``n_pairs``,
        ``instrument_ref_corr``, ``instrument_ref_slope``, ``path_trace_note``
        and more (see :func:`build_trace`).
    """

    history: str
    lead_days: int
    shock_window: int
    K: int
    lam: float
    lambda_rule: str
    topics: list[str]
    assets: list[str]
    panel_topics: list[str]
    train_periods: pd.DatetimeIndex
    forecast_periods: pd.DatetimeIndex
    row_week: pd.Timestamp
    instrument_week: pd.Timestamp
    instrument_window_end: pd.Timestamp
    z: pd.DataFrame
    week_ends: pd.DatetimeIndex
    population: dict[str, Any]
    window_truth: pd.DataFrame
    B_true_train_units: pd.DataFrame
    instruments: pd.DataFrame
    chain: dict[str, Any]
    variants: dict[str, pd.DataFrame]
    ladder: pd.DataFrame
    capture: dict[str, Any]
    path: pd.DataFrame | None
    gamma_path: pd.DataFrame | None
    path_trace: pd.DataFrame | None
    gamma_std: pd.DataFrame
    kkt: pd.DataFrame
    factors_in_sample: pd.DataFrame
    units: pd.DataFrame
    shock_table: pd.DataFrame
    stability: pd.DataFrame
    weeks: pd.DataFrame
    checks: list[TraceCheck]
    findings: list[dict[str, str]]
    meta: dict[str, Any] = field(default_factory=dict)

    def checks_frame(self, step: str | None = None) -> pd.DataFrame:
        """The checks (of one step, or all) as a table.

        Columns: ``step`` (the page label), ``check``, ``relation``,
        ``observed``, ``reference``, ``tolerance``, ``status``, ``kind`` and
        ``note``; one row per check in step order.
        """
        if step is not None and step not in STEPS:
            raise KeyError(f"unknown step {step!r}; known: {list(STEPS)}")
        rows = [
            {
                "step": STEPS.get(c.step, c.step),
                "check": c.name,
                "relation": c.relation,
                "observed": float(c.value),
                "reference": float(c.reference),
                "tolerance": float(c.tolerance),
                "status": c.status,
                "kind": c.kind,
                "note": c.note,
            }
            for c in self.checks
            if step is None or step == "summary" or c.step == step
        ]
        cols = ["step", "check", "relation", "observed", "reference", "tolerance", "status", "kind", "note"]
        return pd.DataFrame(rows, columns=cols)

    def status_frame(self) -> pd.DataFrame:
        """One row per step (except the summary): what it computes, its checks and the reading.

        Columns: ``step`` (key), ``label``, ``what``, ``checks`` (e.g.
        ``"5 of 5 ok"``, over the checks that are not ``"info"``), ``n_ok``,
        ``n_checks``, ``key_number`` (text with this run's numbers) and
        ``reading``: ``"off"`` when an identity check of the step is off,
        ``"departs"`` when a diagnostic check is off or a finding of severity
        ``"departure"`` belongs to the step, ``"as expected"`` otherwise.
        """
        keys = self.meta.get("key_numbers", {})
        rows = []
        for step, label in STEPS.items():
            if step == "summary":
                continue
            cs = [c for c in self.checks if c.step == step]
            graded = [c for c in cs if c.status != INFO]
            n_ok = sum(c.status == OK for c in graded)
            if any(c.status == OFF and c.kind == IDENTITY for c in cs):
                reading = "off"
            elif any(c.status == OFF for c in cs) or any(
                f["step"] == step and f["severity"] == "departure" for f in self.findings
            ):
                reading = "departs"
            else:
                reading = "as expected"
            rows.append({
                "step": step,
                "label": label,
                "what": STEP_WHAT.get(step, ""),
                "checks": f"{n_ok} of {len(graded)} ok",
                "n_ok": int(n_ok),
                "n_checks": int(len(graded)),
                "key_number": str(keys.get(step, "")),
                "reading": reading,
            })
        return pd.DataFrame(rows, columns=["step", "label", "what", "checks", "n_ok", "n_checks", "key_number",
                                           "reading"])


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _assets(sim: SimData) -> list[str]:
    return [str(a) for a in sim.market.returns.columns]


def _trailing_filter(x: np.ndarray, w: int) -> np.ndarray:
    """``x_t - mean(x_{t-1}, ..., x_{t-w})`` along axis 0 (the D9 filter); ``NaN`` for the first ``w`` rows.

    Sliding windows, so a missing value only affects the rows whose window
    holds it (independent of the pandas rolling mean the package uses).
    """
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    w = int(w)
    if x.shape[0] > w:
        win = np.lib.stride_tricks.sliding_window_view(x, w, axis=0)  # win[j] holds rows j .. j + w - 1
        out[w:] = x[w:] - win[:-1].mean(axis=-1)
    return out


def _max_diff(a: Any, b: Any, relative: bool = True) -> tuple[float, int]:
    """Largest ``|a - b|`` over cells finite in both (relative to the largest ``|b|`` there).

    Also returns the number of cells finite on one side only.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    one = int(np.sum(fa != fb))
    if not both.any():
        return float("nan"), one
    diff = float(np.max(np.abs(a[both] - b[both])))
    if not relative:
        return diff, one
    scale = float(np.max(np.abs(b[both])))
    return (diff / scale if scale > 0.0 else diff), one


def _weighted_cov(r: np.ndarray, Z: np.ndarray, wts: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """``sum_tau k_tau (r_tau - rbar)(z_tau - zbar)`` with ``k = wts / sum(wts)`` over the ``ok`` days; ``(L,)``."""
    if not ok.any():
        return np.full(Z.shape[1], np.nan)
    k = wts[ok] / wts[ok].sum()
    ro, Zo = r[ok], Z[ok]
    return ((ro - k @ ro) * k) @ (Zo - k @ Zo)


def _week_structure(days: pd.DatetimeIndex, skip_days: int) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray,
                                                                       np.ndarray, np.ndarray]:
    """Week id per day, week ends, and the ``(start, stop, cut)`` day bounds of every week (D12)."""
    pid, ends = period_end_index(pd.DatetimeIndex(days), lab_bks.PERIOD)
    start, stop, cut = window_bounds(pid, int(skip_days))
    return pid, ends, start, stop, cut


def _instrument_weights(pid: np.ndarray, j: int, xi: float, stop: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """Raw kernel weights of instrument week ``j``: ``xi^(j - week(tau))`` up to the window end (D11, D12)."""
    wts = kernel_weights(pid, int(j), float(xi))
    wts[cut[j]:stop[j]] = 0.0
    return wts


def _corr_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson correlation and least-squares slope (with intercept) of ``y`` on ``x`` over finite pairs."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0.0 or np.ptp(y[ok]) == 0.0:
        return float("nan"), float("nan")
    xo, yo = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    corr = float((xo @ yo) / np.sqrt((xo @ xo) * (yo @ yo)))
    slope = float((xo @ yo) / (xo @ xo))
    return corr, slope


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
def population_moments(sim: SimData, truth: SimTruth) -> dict[str, Any]:
    """Population moments of the observed shocks behind ``truth`` (G.5.3; :func:`.dgp.truth_for_window`).

    The internals of :func:`.dgp._population_truth` recomputed for complete
    data: ``z = kappa q + noise`` with ``q`` the trailing-mean filter of the
    designed signal, paired with the full-sample standardised return
    ``rt_{t+l}``.

    Returns
    -------
    dict
        ``var_z`` (``L x L``, raw shock units), ``sd_z`` (``L``), ``S_z``
        (``var_z`` scaled to a unit diagonal), ``cov_z_rt`` (``L x N``:
        ``Cov(z_k, rt_i)``), ``cov_z_r`` (``L x N``: ``Cov(z_k, r_i)`` in
        return units, ``cov_z_rt * asset_vol``), ``C`` (``cov_z_rt / sd_z``)
        and ``noise_diag`` (the noise variance on the diagonal); numpy
        arrays in the simulation's topic and asset order.
    """
    topics, assets = sim.topics.ids, _assets(sim)
    w, lead = int(truth.shock_window), int(sim.lead_days)
    acfg = sim.meta["attention_cfg"]
    kappa = float(acfg["kappa"])
    p = np.asarray(sim.meta["signal"], dtype=float)
    rt = np.asarray(sim.meta["rt"], dtype=float)
    obs = np.asarray(sim.meta["obs"], dtype=bool)
    n = p.shape[0]
    q = _trailing_filter(p, w)
    rows = np.arange(w, n - lead)
    qp, yp, op = q[rows], rt[rows + lead], obs[rows + lead]
    su2 = truth.sigma_u.reindex(topics).to_numpy(dtype=float) ** 2
    noise = kappa**2 * (1.0 + 1.0 / w) * su2 + _var_filtered_slow(kappa, acfg["slow_ar1"], acfg["slow_sd_ratio"], w)
    var_z = kappa**2 * np.atleast_2d(np.cov(qp, rowvar=False, ddof=0)) + np.diag(noise)
    sd_z = np.sqrt(np.diag(var_z))
    cov_z_rt = kappa * _cross_cov(qp, yp, op)
    sigma = truth.asset_vol.reindex(assets).to_numpy(dtype=float)
    return {
        "var_z": var_z,
        "sd_z": sd_z,
        "S_z": var_z / np.outer(sd_z, sd_z),
        "cov_z_rt": cov_z_rt,
        "cov_z_r": cov_z_rt * sigma[None, :],
        "C": cov_z_rt / sd_z[:, None],
        "noise_diag": noise,
    }


def window_truth(sim: SimData, shocks: ObservedShocks, truth: SimTruth) -> pd.DataFrame:
    """Population sensitivity of the training window's own signal, in the estimators' training units.

    The regression of ``_population_truth`` with the signal's moments
    ``Cov(q)`` and ``Cov(q, r)`` taken over the direct estimator's training
    pairs (:func:`.direct.training_pairs`) and the noise at its population
    variance: ``b = var_zw^-1 kappa Cov_train(q, r)``, then
    ``B[k, i] = b[k, i] sd_train(z_k) / sd_train(r_i)``. It is what a
    perfect estimator on the training days would find (on the dashboard
    defaults Spearman 0.78 with ``B_true``).

    Returns
    -------
    DataFrame
        Topics x assets (``NaN`` when the window has fewer than two pairs).
    """
    topics, assets = sim.topics.ids, _assets(sim)
    t_index, a_index = pd.Index(topics, name="topic_id"), pd.Index(assets, name="asset_id")
    w = int(shocks.window)
    acfg = sim.meta["attention_cfg"]
    kappa = float(acfg["kappa"])
    cal = sim.market.calendar
    p_pos, q_pos = training_pairs(sim, shocks, shock_matrix(shocks, cal, topics))
    if len(p_pos) < 2:
        return pd.DataFrame(np.nan, index=t_index, columns=a_index)
    Q = _trailing_filter(np.asarray(sim.meta["signal"], dtype=float), w)[p_pos]
    Rtr = np.array(sim.market.returns.to_numpy(dtype=float)[q_pos])
    obs = np.isfinite(Rtr)
    _, ret_scale = _train_moments(Rtr, obs)
    su2 = truth.sigma_u.reindex(topics).to_numpy(dtype=float) ** 2
    noise = kappa**2 * (1.0 + 1.0 / w) * su2 + _var_filtered_slow(kappa, acfg["slow_ar1"], acfg["slow_sd_ratio"], w)
    var_zw = kappa**2 * np.atleast_2d(np.cov(Q, rowvar=False, ddof=0)) + np.diag(noise)
    b_raw = np.linalg.solve(var_zw, kappa * _cross_cov(Q, Rtr, obs))
    zs = shocks.scale.reindex(topics).to_numpy(dtype=float)
    return pd.DataFrame(b_raw * zs[:, None] / ret_scale[None, :], index=t_index, columns=a_index)


def _z_components(sim: SimData, w: int, topics: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                                  pd.DataFrame]:
    """Exact split of the lab shock ``z = z_signal + z_news + z_slow`` on the simulation calendar.

    ``z_signal = kappa F(p)`` (the designed signal), ``z_news = kappa F(s - p)``
    (the news noise of the designed shock) and ``z_slow = F(g)`` (the slow
    attention component ``g = a - base - kappa s``), with ``F`` the trailing
    filter of the shocks. Exact when no attention value hit the floor.
    """
    ids = sim.topics.ids
    cols = list(ids) if topics is None else [str(t) for t in topics]
    pos = np.array([ids.index(t) for t in cols], dtype=np.int64)
    kappa = float(sim.meta["attention_cfg"]["kappa"])
    p = np.asarray(sim.meta["signal"], dtype=float)[:, pos]
    s = sim.designed_shocks.reindex(columns=cols).to_numpy(dtype=float)
    base = sim.meta["base_level"].reindex(cols).to_numpy(dtype=float)
    g = sim.attention.reindex(columns=cols).to_numpy(dtype=float) - base[None, :] - kappa * s
    cal = sim.market.calendar

    def frame(a: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(a, index=cal, columns=pd.Index(cols, name="topic_id"))

    z_signal, z_news = frame(kappa * _trailing_filter(p, w)), frame(kappa * _trailing_filter(s - p, w))
    return z_signal, z_news, frame(_trailing_filter(g, w))


def _bks_shocks(panel: BKSPanel, sim: SimData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The attention the panel's shock stage read and its shocks ``z`` (on that attention's own days).

    Full history: :func:`narrative_ipca.shocks.attention_shocks` on
    ``panel.aligned.attention``. Training history (D88): the attention from
    ``w`` weekdays before the training start, lagged by ``l`` days on the
    same grid, as :func:`.bks._training_inputs` builds it. Never the shifted
    lab ``z`` under the full history: the panel's first ``w`` days have no
    shock there.
    """
    pcfg = panel.pipeline_cfg
    if str(panel.meta.get("history", "full")) == "training":
        cal = pd.DatetimeIndex(sim.market.returns.index)
        i0 = int(cal.searchsorted(pd.Timestamp(panel.meta["train_start"]), side="left"))
        a0 = max(i0 - int(pcfg.shocks.window), 0)
        att = sim.attention.reindex(cal).iloc[a0:]
        att = att.set_axis([str(c) for c in att.columns], axis=1)
        lead = int(pcfg.data.attention_lag_days)
        att_in = att.shift(lead) if lead > 0 else att
    else:
        att_in = panel.aligned.attention
    return att_in, attention_shocks(att_in, pcfg.shocks).z


# ---------------------------------------------------------------------------
# The Eq. 5 chain
# ---------------------------------------------------------------------------
@dataclass
class _Rows:
    """Each asset's Eq. 5 instrument row (the latest training row), simulation asset order."""

    row: np.ndarray  # panel row per asset, -1 without a training row
    has: np.ndarray  # bool
    week: pd.Series  # row period per asset (NaT without a row)


def _eq5_rows(panel: BKSPanel, fit: BKSFit, assets: list[str]) -> _Rows:
    """The latest training row of every asset, as :func:`.bks.implied_exposures` step 1 picks it."""
    pnl = panel.panel
    periods = pd.DatetimeIndex(pnl.periods)
    train_pos = np.flatnonzero(np.asarray(periods.isin(fit.train_periods), dtype=bool))
    rows = np.flatnonzero(np.isin(pnl.t_idx, train_pos))
    rev = rows[::-1]  # t_idx is sorted: the first hit per asset in reverse order is its latest row
    found, first = np.unique(pnl.asset_idx[rev], return_index=True)
    panel_assets = [str(a) for a in pnl.assets]
    row_of = {panel_assets[int(a)]: int(r) for a, r in zip(found, rev[first])}
    row = np.array([row_of.get(a, -1) for a in assets], dtype=np.int64)
    has = row >= 0
    week = pd.Series(
        [periods[int(pnl.t_idx[r])] if r >= 0 else pd.NaT for r in row],
        index=pd.Index(assets, name="asset_id"), name="row_period", dtype="datetime64[ns]",
    )
    return _Rows(row=row, has=has, week=week)


@dataclass
class _Chain:
    """Numpy form of the Eq. 5 chain (simulation topic and asset order)."""

    cov: np.ndarray  # (N, L) instruments c~_i, panel units, NaN rows without a row
    beta: np.ndarray  # (N, K)
    Mmap: np.ndarray  # (L, K) in simulation topic order
    P: np.ndarray  # (L, L) projector, simulation topic order
    m_const: np.ndarray  # (L,)
    m: np.ndarray  # (N, L) production formula beta M'
    m_proj: np.ndarray  # (N, L) P c~
    divisor: np.ndarray  # (N,)
    conv: np.ndarray  # (N,) 0 for skipped assets
    zero: np.ndarray  # (N,) bool: skipped (B = 0)
    ret_scale: np.ndarray  # (N,)
    zs: np.ndarray  # (L,) shocks.scale
    Sigma_z: np.ndarray  # (L, L) training days
    Sz_pinv: np.ndarray
    sz_rank: int
    gamma_rank: int
    n_pairs: int
    n_train: np.ndarray
    p_pos: np.ndarray
    q_pos: np.ndarray

    def to_B(self, m: np.ndarray, Sz_pinv: np.ndarray | None = None) -> np.ndarray:
        """Steps 3-5 of Eq. 5 for implied covariances ``m`` (``N x L``, panel units): topics x assets."""
        S = self.Sz_pinv if Sz_pinv is None else Sz_pinv
        m0 = np.where(np.isfinite(m), m, 0.0)
        b_raw = (m0 * self.conv[:, None]) @ S
        B = (b_raw * (self.zs[None, :] / self.ret_scale[:, None])).T
        B[:, self.zero] = 0.0
        return B


def _eq5_chain(panel: BKSPanel, fit: BKSFit, sim: SimData, shocks: ObservedShocks, rows: _Rows,
               order: np.ndarray) -> _Chain:
    """Recompute :func:`.bks.implied_exposures` steps 1-5 with every intermediate kept."""
    pnl = panel.panel
    assets, topics = _assets(sim), sim.topics.ids
    N, L = len(assets), len(topics)
    rcond = float(panel.pipeline_cfg.evaluation.rcond)
    Gamma = np.asarray(fit.fit.Gamma, dtype=float)
    Mmap_p, gamma_rank = _lsq_map(Gamma[1:], rcond)  # panel topic order
    Mmap = Mmap_p[order]
    P = Mmap @ Gamma[1:][order].T
    m_const = Mmap @ Gamma[0]
    C = np.full((N, pnl.p), np.nan)
    C[rows.has] = pnl.X[rows.row[rows.has]]
    cov = C[:, 1:][:, order]
    beta = C @ Gamma
    m = beta @ Mmap.T  # the production formula (bks.implied_topic_covariance)
    m_proj = cov @ P.T
    cal = sim.market.calendar
    p_pos, q_pos = training_pairs(sim, shocks, shock_matrix(shocks, cal, topics))
    Rtr = np.array(sim.market.returns.to_numpy(dtype=float)[q_pos])
    obs = np.isfinite(Rtr)
    n_train = obs.sum(axis=0)
    _, ret_scale = _train_moments(Rtr, obs)
    if panel.aligned.scale is not None:
        sc = panel.aligned.scale.reindex(index=cal, columns=assets).to_numpy(dtype=float)[q_pos]
        ok = obs & np.isfinite(sc)
        cnt = ok.sum(axis=0)
        divisor = np.where(cnt > 0, np.where(ok, sc, 0.0).sum(axis=0) / np.maximum(cnt, 1), np.nan)
    else:
        divisor = np.ones(N)
    zero = ~rows.has | ~np.isfinite(divisor) | (n_train < MIN_TRAIN_OBS)
    conv = np.where(zero, 0.0, np.where(np.isfinite(divisor), divisor, 0.0))
    Z = shocks.z.reindex(index=cal, columns=topics).to_numpy(dtype=float)[p_pos]
    if Z.shape[0] >= 1:
        Zc = Z - Z.mean(axis=0)
        Sigma_z = (Zc.T @ Zc) / Z.shape[0]
        Sz_pinv, sz_rank = _sym_pinv(Sigma_z, rcond)
    else:
        Sigma_z, Sz_pinv, sz_rank = np.full((L, L), np.nan), np.zeros((L, L)), 0
    return _Chain(
        cov=cov, beta=beta, Mmap=Mmap, P=P, m_const=m_const, m=m, m_proj=m_proj, divisor=divisor, conv=conv,
        zero=zero, ret_scale=ret_scale, zs=shocks.scale.reindex(topics).to_numpy(dtype=float), Sigma_z=Sigma_z,
        Sz_pinv=Sz_pinv, sz_rank=int(sz_rank), gamma_rank=int(gamma_rank), n_pairs=int(len(p_pos)),
        n_train=n_train, p_pos=p_pos, q_pos=q_pos,
    )


def _instrument_svd(cov_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Singular values and right singular vectors of the (uncentred) instrument rows; ``None`` when all zero."""
    if cov_rows.size == 0 or not float(np.sum(cov_rows**2)) > 0.0:
        return None
    _, s, Vt = np.linalg.svd(cov_rows, full_matrices=False)
    return s, Vt


def _capture(cov_rows: np.ndarray, P: np.ndarray, Mmap: np.ndarray, K: int, rcond: float,
             svd: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any]:
    """How much of the instruments' squared norm the fit's directions keep, against the best ``K`` directions.

    ``svd`` is :func:`_instrument_svd` of ``cov_rows``.
    """
    L = P.shape[0]
    out: dict[str, Any] = {
        "kept_share": float("nan"), "best_share": float("nan"), "random_share": float(min(K, L) / max(L, 1)),
        "singular_share": pd.Series(dtype=float, name="singular_share"),
        "captured": pd.Series(dtype=float, name="captured"), "principal_cosines": np.zeros(int(K)),
    }
    if svd is None:
        return out
    total = float(np.sum(cov_rows**2))
    s, Vt = svd
    k = min(int(K), len(s))
    directions = pd.RangeIndex(1, len(s) + 1, name="direction")
    out["kept_share"] = float(np.sum((cov_rows @ P) ** 2) / total)
    out["best_share"] = float(np.sum(s[:k] ** 2) / np.sum(s**2))
    out["singular_share"] = pd.Series(s**2 / np.sum(s**2), index=directions, name="singular_share")
    out["captured"] = pd.Series(np.sum((Vt @ P) ** 2, axis=1), index=directions, name="captured")
    # principal angles between the fit's directions (column space of Gamma_tilde) and the best K
    U, sg, _ = np.linalg.svd(Mmap, full_matrices=False)
    r = int(np.sum(sg**2 > rcond * sg[0] ** 2)) if sg.size and sg[0] > 0.0 else 0
    cos = np.zeros(int(K))
    if r > 0 and k > 0:
        c = np.linalg.svd(U[:, :r].T @ Vt[:k].T, compute_uv=False)
        cos[: len(c)] = np.clip(c, 0.0, 1.0)
    out["principal_cosines"] = cos
    return out


# ---------------------------------------------------------------------------
# Check bookkeeping
# ---------------------------------------------------------------------------
class _Checks:
    """Collects :class:`TraceCheck` objects; the status follows ``value <= tolerance`` unless given."""

    def __init__(self) -> None:
        self.items: list[TraceCheck] = []

    def add(self, step: str, name: str, relation: str, value: float, *, reference: float = float("nan"),
            tolerance: float = float("nan"), note: str, kind: str = IDENTITY, status: str | None = None) -> TraceCheck:
        value = float(value)
        tolerance = float(tolerance)
        if status is None:
            if not np.isfinite(tolerance):
                status = INFO
            else:
                status = OK if np.isfinite(value) and value <= tolerance else OFF
        c = TraceCheck(step=step, name=name, relation=relation, value=value, reference=float(reference),
                       tolerance=tolerance if status != INFO else float("nan"), status=status, note=note, kind=kind)
        self.items.append(c)
        return c

    def info(self, step: str, name: str, relation: str, value: float, *, note: str, reference: float = float("nan"),
             kind: str = DIAGNOSTIC) -> TraceCheck:
        return self.add(step, name, relation, value, reference=reference, note=note, kind=kind, status=INFO)


def _fmt_day(x: Any) -> str:
    return "n/a" if x is None or pd.isna(x) else pd.Timestamp(x).date().isoformat()


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------
def build_trace(
    panel: BKSPanel,
    fit: BKSFit,
    result: BKSLabResult,
    sim: SimData,
    shocks: ObservedShocks,
    truth: SimTruth,
    window: WindowConfig,
    bks_cfg: BKSLabConfig,
    implied: DirectFit,
    *,
    path_trace: bool | None = None,
) -> BKSTrace:
    """Trace one cached BKS run step by step (module docstring; DESIGN.md G.16).

    Parameters
    ----------
    panel, fit, result:
        The cached BKS panel, training fit and forecast evaluation of one
        configuration (:meth:`.session.LabSession.bks_panel`, ``bks_fit``,
        ``bks``).
    sim, shocks:
        The simulation and its observed shocks (same configuration).
    truth:
        The truth for the shocks' window: ``session.truth(cfg)``, not
        ``sim.truth`` (which may belong to another ``w``).
    window:
        The lab's windows (training, forecast).
    bks_cfg:
        The BKS settings of the run (``cfg.bks``): the estimation settings
        are rebuilt from it (:func:`.bks.bks_pipeline_config`), never read
        from ``panel.pipeline_cfg`` (whose key ignores them).
    implied:
        ``session.bks_implied(cfg)``, the production implied sensitivities.
    path_trace:
        Refit the lambda path and score every point
        (:func:`lambda_path_trace`); ``None`` means on when there are at most
        :data:`PATH_TRACE_MAX_TOPICS` topics and the rule is not fixed.

    Returns
    -------
    BKSTrace
        Fresh objects only: the cached inputs are not modified.

    Raises
    ------
    ValueError
        When the inputs do not belong together (lead, shock window, topics,
        assets or instruments differ).
    """
    t_all = time.perf_counter()
    timings: dict[str, float] = {}
    tick = [time.perf_counter()]

    def lap(name: str) -> None:
        now = time.perf_counter()
        timings[name] = now - tick[0]
        tick[0] = now

    pnl, pcfg, al, res = panel.panel, panel.pipeline_cfg, panel.aligned, fit.fit
    topics, assets = sim.topics.ids, _assets(sim)
    L, N, K = len(topics), len(assets), int(fit.K)
    panel_topics = [str(t) for t in pnl.topics]
    panel_assets = [str(a) for a in pnl.assets]
    lead, w = int(sim.lead_days), int(pcfg.shocks.window)
    if int(pcfg.data.attention_lag_days) != lead:
        raise ValueError(
            f"the panel lags attention by {pcfg.data.attention_lag_days} day(s), the simulation's lead is {lead}"
        )
    if int(shocks.window) != w or int(truth.shock_window) != w:
        raise ValueError(
            f"shock windows differ: panel {w}, shocks {shocks.window}, truth {truth.shock_window} "
            "(pass session.truth(cfg), not sim.truth)"
        )
    if sorted(panel_topics) != sorted(topics) or sorted(panel_assets) != sorted(assets):
        raise ValueError("the panel's topics or assets differ from the simulation's")
    if list(res.instrument_names) != list(pnl.instrument_names):
        raise ValueError("fit and panel have different instruments")
    col = {t: j for j, t in enumerate(panel_topics)}
    order = np.array([col[t] for t in topics], dtype=np.int64)
    t_index, a_index = pd.Index(topics, name="topic_id"), pd.Index(assets, name="asset_id")
    instr_names = [str(n) for n in pnl.instrument_names]
    history = str(panel.meta.get("history", "full"))
    weighting = str(pcfg.data.asset_weighting)
    scaled = al.scale is not None
    xi, skip = float(pcfg.covariance.xi), int(pcfg.covariance.skip_days)
    min_days, burn_in = int(pcfg.covariance.min_days), int(pcfg.covariance.burn_in_periods)
    rcond = float(pcfg.evaluation.rcond)
    lam = float(fit.lam)
    rule = str(fit.meta.get("lambda_rule", bks_cfg.lambda_rule))
    ts, te = pd.Timestamp(window.train_start), pd.Timestamp(window.train_end)
    days = pd.DatetimeIndex(al.calendar)
    pid, week_ends, start, stop, cut = _week_structure(days, skip)
    periods = pd.DatetimeIndex(pnl.periods)
    tw = week_ends.get_indexer(periods)
    if np.any(tw < 1):
        raise ValueError("panel periods are not weeks of the aligned calendar with an instrument week before them")
    train_periods = pd.DatetimeIndex(fit.train_periods)
    forecast_periods = pd.DatetimeIndex(result.periods)
    row_week = pd.Timestamp(train_periods.max())
    j_row = int(week_ends.get_loc(row_week)) - 1
    instrument_week = pd.Timestamp(week_ends[j_row])
    window_end = pd.Timestamp(days[cut[j_row] - 1]) if cut[j_row] > 0 else pd.NaT
    est = lab_bks.bks_pipeline_config(bks_cfg, w, lead, n_assets=int(pnl.N)).estimation
    checks = _Checks()
    notes: list[str] = []
    lap("setup")

    # ---- shocks and references ------------------------------------------------------------------
    att_in, z_own = _bks_shocks(panel, sim)
    z = z_own.reindex(index=days, columns=panel_topics)
    Zp = z.to_numpy(dtype=float)  # panel topic order, aligned calendar
    zok = np.isfinite(Zp).all(axis=1)
    pop = population_moments(sim, truth)
    ret_scale_prod = implied.ret_scale.reindex(assets).to_numpy(dtype=float)
    asset_vol = truth.asset_vol.reindex(assets).to_numpy(dtype=float)
    if not scaled:
        d_ref, ref_note = np.ones(N), "no asset weighting: the reference is Cov(z, r) in return units (exact)"
    elif history == "training":
        d_ref = ret_scale_prod
        ref_note = ("training history: the divisor is the training standard deviation on every day, so the "
                    "reference is exact")
    else:
        d_ref = asset_vol
        ref_note = ("full history: the reference divides by the full-sample volatility while the panel divides by a "
                    "trailing one, so it is approximate")
    instrument_ref = pd.DataFrame((pop["cov_z_r"] / d_ref[None, :]).T, index=a_index, columns=t_index)
    sd_z = pop["sd_z"]
    zs = shocks.scale.reindex(topics).to_numpy(dtype=float)
    Bt = truth.B_true.reindex(index=topics, columns=assets).to_numpy(dtype=float)
    B_tu = Bt * (asset_vol / ret_scale_prod)[None, :] * (zs / sd_z)[:, None]
    wt = window_truth(sim, shocks, truth)
    population = {
        "var_z": pd.DataFrame(pop["var_z"], index=t_index, columns=t_index),
        "sd_z": pd.Series(sd_z, index=t_index, name="sd_z"),
        "S_z": pd.DataFrame(pop["S_z"], index=t_index, columns=t_index),
        "cov_z_r": pd.DataFrame(pop["cov_z_r"], index=t_index, columns=a_index),
        "C": pd.DataFrame(pop["C"], index=t_index, columns=a_index),
        "instrument_ref": instrument_ref,
        "instrument_divisor": pd.Series(d_ref, index=a_index, name="divisor"),
        "instrument_ref_note": ref_note,
    }
    lap("references")

    # ---- 1 inputs -----------------------------------------------------------------------------------
    S_ref = truth.S_z.reindex(index=topics, columns=topics).to_numpy(dtype=float)
    obs_all = np.asarray(sim.meta["obs"], dtype=bool).all(axis=0)
    with np.errstate(invalid="ignore"):
        B_from_pop = np.linalg.solve(pop["S_z"], pop["C"])
    d_S, _ = _max_diff(pop["S_z"], S_ref)
    d_B, _ = _max_diff(B_from_pop[:, obs_all], Bt[:, obs_all]) if obs_all.any() else (0.0, 0)
    checks.add(
        "inputs", "Population moments reproduce the truth", "S_z = Var(z) scaled to a unit diagonal; S_z^-1 C = B_true",
        max(d_S, d_B), tolerance=1e-8,
        note=("The trace's references (population covariances, instrument reference) are recomputed from the "
              f"simulation's signal; they reproduce S_z and B_true (assets with complete data, {int(obs_all.sum())} "
              "of them), so the references below describe the same truth. Off means the references do not match "
              "the truth."),
    )
    n_clipped = int(sim.meta.get("n_clipped", 0) or 0)
    checks.add(
        "inputs", "Attention never hit its floor", "attention > 1e-6 on every day", float(n_clipped),
        reference=0.0, tolerance=0.0, kind=DIAGNOSTIC,
        status=OK if n_clipped == 0 else INFO,
        note=("The truth and the split of the shocks into signal and noise ignore the attention floor; "
              f"{n_clipped} clipped topic-day(s) make them approximate on those days only."),
    )
    lap("inputs")

    # ---- 2 alignment and scaling ------------------------------------------------------------------
    raw = sim.market.returns.reindex(index=days)
    raw = raw.set_axis([str(c) for c in raw.columns], axis=1).reindex(columns=panel_assets)
    rec = al.returns.reindex(columns=panel_assets)
    if scaled:
        rec = rec * al.scale.reindex(columns=panel_assets)
    d_a, one_a = _max_diff(rec.to_numpy(dtype=float), raw.to_numpy(dtype=float))
    checks.add(
        "align", "Scaled return x divisor = raw return", "aligned return x divisor = simulated return, every day",
        d_a, tolerance=1e-12,
        note=(("The panel divides each daily return by its divisor; multiplying back must give the input return "
               f"(largest difference relative to the largest return; {one_a} cell(s) are missing on one side only, "
               "where no divisor exists yet). Off means the returns the panel uses are not the simulated ones.")
              if scaled else
              "Without asset weighting the panel uses the simulated returns as they are. Off means they differ."),
    )
    att_ref = sim.attention.shift(lead).reindex(index=days)
    att_ref = att_ref.set_axis([str(c) for c in att_ref.columns], axis=1)
    att_ref = att_ref.reindex(columns=[str(c) for c in al.attention.columns])
    d_att, one_att = _max_diff(al.attention.to_numpy(dtype=float), att_ref.to_numpy(dtype=float), relative=False)
    checks.add(
        "align", "Attention is lagged by the lead", f"aligned attention on day tau = attention on day tau - {lead}",
        d_att if one_att == 0 else float("inf"), tolerance=0.0,
        note=(f"The panel pairs day tau's return with the attention of {lead} trading day(s) earlier (the lead), "
              "exactly as the direct estimator pairs return and shock days. Off means the pairing is shifted."),
    )
    try:
        pred_days, _ = lab_bks._panel_days(bks_cfg, lead, None, ts, w)  # the dashboard's call: default calendar
        cal_sim = sim.market.calendar
        gap = (float(abs(int(cal_sim.get_loc(days[0])) - int(cal_sim.get_loc(pred_days[0]))))
               if len(pred_days) else float("inf"))
        pred_first = _fmt_day(pred_days[0]) if len(pred_days) else "none"
    except (KeyError, ValueError) as exc:  # pragma: no cover - defensive
        gap, pred_first = float("inf"), f"error: {exc}"
    checks.add(
        "align", "Panel starts where the week count predicts", "first aligned day = bks._panel_days prediction",
        gap, reference=0.0, tolerance=0.0,
        note=(f"The panel's first day ({_fmt_day(days[0])}) must match the day the training-week count "
              f"(bks.training_weeks, which the sidebar checks before a run on the lab's weekday calendar) assumes "
              f"({pred_first}); off means that count is wrong."),
    )
    lap("align")

    # ---- 3 attention shocks --------------------------------------------------------------------------
    z_formula = _trailing_filter(att_in.to_numpy(dtype=float), w)
    d_zf, one_zf = _max_diff(z_own.to_numpy(dtype=float), z_formula)
    checks.add(
        "shocks", "Shocks follow the trailing-mean formula", f"z_tau = a_tau - mean(a over the previous {w} days)",
        d_zf if one_zf == 0 else float("inf"), tolerance=1e-12,
        note=("The shocks recomputed by the package equal an explicit sliding-window formula on the panel's "
              "attention (largest difference relative to the largest shock; missing days must match too). Off "
              "means the shock definition differs from D9."),
    )
    z_dir = shocks.z.shift(lead).reindex(index=days)
    z_dir = z_dir.set_axis([str(c) for c in z_dir.columns], axis=1).reindex(columns=panel_topics)
    d_zd, one_zd = _max_diff(Zp, z_dir.to_numpy(dtype=float))
    one_days = int(np.sum(np.isfinite(Zp).all(axis=1) != np.isfinite(z_dir.to_numpy(dtype=float)).all(axis=1)))
    expected_one = w if history == "full" else 0
    checks.add(
        "shocks", "BKS shocks = direct shocks shifted by the lead", f"z_BKS(tau) = z(tau - {lead}) on shared days",
        d_zd, tolerance=1e-12,
        note=(f"BKS and the direct methods see the same shocks. {one_days} day(s) have a shock on one side only "
              f"({expected_one} expected: " + ("the panel recomputes the shocks on its own days, so its first "
                                               f"{w} days have none)." if history == "full" else
                                               "the training panel starts where the shocks start).")
              + " Off means BKS works on different shocks."),
    )
    corr_designed = np.full(L, np.nan)
    zl = shocks.z.reindex(columns=topics).to_numpy(dtype=float)
    ds = sim.designed_shocks.reindex(index=shocks.z.index, columns=topics).to_numpy(dtype=float)
    for k in range(L):
        ok = np.isfinite(zl[:, k]) & np.isfinite(ds[:, k])
        if ok.sum() > 2:
            corr_designed[k] = float(np.corrcoef(zl[ok, k], ds[ok, k])[0, 1])
    att_true = truth.attenuation.reindex(topics).to_numpy(dtype=float)
    gap_att = float(np.nanmax(np.abs(corr_designed - att_true))) if np.isfinite(corr_designed).any() else float("nan")
    checks.info(
        "shocks", "Shocks track the designed shocks as the truth says",
        "corr(z_k, designed shock s_k) over the whole sample = attenuation a_k", gap_att,
        note=("The observed shock is the designed shock plus the attention's slow drift, filtered; their "
              f"correlation should be the truth's attenuation (mean {np.nanmean(att_true):.2f}). The largest gap is "
              "shown; it is a sample moment, so a few hundredths are normal."),
    )
    ratio_sd = zs / sd_z
    checks.info(
        "shocks", "Training shock scale against the population", "sd_train(z_k) / sd_population(z_k)",
        float(np.nanmedian(ratio_sd)), reference=1.0,
        note=(f"The estimators standardise with the training standard deviation; across topics it is "
              f"{np.nanmin(ratio_sd):.2f} to {np.nanmax(ratio_sd):.2f} times the population value (median shown). "
              "Far from 1 means B_true and the estimates use different units (D74); B_true in training units "
              "corrects for it."),
    )
    z_sig, z_news, z_slow = _z_components(sim, w)
    zl_all = shocks.z.reindex(columns=topics)
    signal_share = np.full(L, np.nan)
    zs_np, zl_np = z_sig.to_numpy(dtype=float), zl_all.to_numpy(dtype=float)
    for k in range(L):
        ok = np.isfinite(zs_np[:, k]) & np.isfinite(zl_np[:, k])
        if ok.sum() > 2 and np.var(zl_np[ok, k]) > 0.0:
            signal_share[k] = float(np.var(zs_np[ok, k]) / np.var(zl_np[ok, k]))
    shock_table = pd.DataFrame(
        {"sd_train": zs, "sd_population": sd_z, "ratio": ratio_sd, "corr_designed": corr_designed,
         "attenuation_true": att_true, "signal_share": signal_share},
        index=t_index,
    )
    del z_sig, z_news, z_slow
    lap("shocks")

    # ---- 4 instruments ---------------------------------------------------------------------------------
    wts_row = _instrument_weights(pid, j_row, xi, stop, cut)
    R_p = al.returns.reindex(columns=panel_assets).to_numpy(dtype=float)  # panel asset order
    rows_week = np.flatnonzero(pnl.t_idx == int(periods.get_loc(row_week)))
    brute = np.full((len(rows_week), len(panel_topics)), np.nan)
    ok_z = zok & (wts_row > 0.0)
    r_ok = np.isfinite(R_p) & ok_z[:, None]
    a_rows = pnl.asset_idx[rows_week]
    full_cols = r_ok[ok_z].all(axis=0)
    fast = full_cols[a_rows]
    if fast.any():
        k = wts_row[ok_z] / wts_row[ok_z].sum()
        Zo = Zp[ok_z]
        Ro = R_p[ok_z][:, a_rows[fast]]
        brute[fast] = ((Ro - k @ Ro) * k[:, None]).T @ (Zo - k @ Zo)
    for n_i in np.flatnonzero(~fast):
        brute[n_i] = brute_force_covariance(R_p[:, a_rows[n_i]], Zp, wts_row)
    d_br, one_br = _max_diff(pnl.X[rows_week, 1:], brute)
    checks.add(
        "instruments", "Panel instruments = brute-force kernel covariance",
        f"X row = sum_tau k_tau (r - rbar)(z - zbar), week ending {_fmt_day(row_week)}, every asset",
        d_br if one_br == 0 else float("inf"), tolerance=1e-10,
        note=(f"The {len(rows_week)} instrument rows the Eq. 5 step reads (instrument week ending "
              f"{_fmt_day(instrument_week)}, window up to {_fmt_day(window_end)}) are recomputed by direct "
              "summation over every day of the kernel (largest difference relative to the largest instrument). "
              "Off means the covariance recursion or the shocks differ from their definition."),
    )
    # the NaN rule: an instrument needs min_days observed days, on every panel week and asset
    ok_days = np.isfinite(R_p) & zok[:, None]
    cum_ok = np.vstack([np.zeros((1, R_p.shape[1]), dtype=np.int64), np.cumsum(ok_days, axis=0)])
    n_rows_days = cum_ok[cut[tw[pnl.t_idx] - 1], pnl.asset_idx]
    viol_a = int(np.sum(n_rows_days < min_days))
    fin = np.isfinite(R_p)
    wk_sum = np.add.reduceat(np.where(fin, R_p, 0.0), start, axis=0)
    wk_cnt = np.add.reduceat(fin.astype(np.int64), start, axis=0)
    Y = np.where(wk_cnt > 0, wk_sum, np.nan)  # (weeks, assets): the week's sum of scaled daily returns
    present = np.zeros((len(periods), R_p.shape[1]), dtype=bool)
    present[pnl.t_idx, pnl.asset_idx] = True
    nd_weeks = cum_ok[cut[tw - 1]]
    viol_b = int(np.sum(~present & np.isfinite(Y[tw]) & (nd_weeks >= min_days)))
    checks.add(
        "instruments", "An instrument needs its minimum of days",
        f"row exists iff >= {min_days} observed days in the kernel window",
        float(viol_a + viol_b), reference=0.0, tolerance=0.0,
        note=(f"Every panel row's instrument window holds at least {min_days} days with a return and a shock, and "
              "no asset-week with a return and enough days is missing from a kept week (violations counted: "
              f"{viol_a} rows with too few days, {viol_b} missing cells). Off means the NaN rule differs from D17."),
    )
    wsum = float(wts_row.sum())
    share_all = float(wts_row[np.asarray(days < ts)].sum() / wsum) if wsum > 0 else float("nan")
    try:
        share_pred = float(lab_bks.kernel_history_share(ts, te, bks_cfg, lead, None, w))
    except ValueError:  # pragma: no cover - defensive
        share_pred = float("nan")
    checks.add(
        "instruments", "Kernel share before the training start",
        "share of the kernel weight on days before train_start",
        abs(share_all - share_pred) if np.isfinite(share_pred) else float("nan"), reference=0.0, tolerance=1e-6,
        note=(f"The Eq. 5 instruments put {share_all:.1%} of their kernel weight on days before the training start; "
              f"the no-panel estimate the dashboard shows before a run (bks.kernel_history_share) says "
              f"{share_pred:.1%}. Off means that estimate is wrong."),
    )
    k_ok = wts_row[ok_z] / wts_row[ok_z].sum() if ok_z.any() else np.zeros(0)
    eff_days = float(1.0 / np.sum(k_ok**2)) if k_ok.size else float("nan")
    rows_obj = _eq5_rows(panel, fit, assets)
    chain = _eq5_chain(panel, fit, sim, shocks, rows_obj, order)
    ref_corr, ref_slope = _corr_slope(instrument_ref.to_numpy()[rows_obj.has], chain.cov[rows_obj.has])
    if history == "full":
        checks.add(
            "instruments", "Instruments track their population value", "corr(instrument, population reference) >= 0.9",
            ref_corr, reference=1.0, tolerance=_MIN_REF_CORR, kind=DIAGNOSTIC,
            status=OK if np.isfinite(ref_corr) and ref_corr >= _MIN_REF_CORR else OFF,
            note=(f"Over every asset and topic, the Eq. 5 instruments correlate {ref_corr:.2f} with Cov(z, r) / d from "
                  "the simulation's population moments. With a long kernel they are close to noise-free (0.97-0.98 "
                  "on the defaults); below 0.9 they carry much sampling noise or the units are off."),
        )
    else:
        checks.info(
            "instruments", "Instruments track their population value", "corr(instrument, population reference)",
            ref_corr, reference=1.0,
            note=(f"Correlation {ref_corr:.2f}. Under the training history the instruments are covariances over the "
                  "training weeks only, a small sample, so they need not match the population (0.70 on the defaults)."),
        )
    checks.info(
        "instruments", "Slope of instruments on their population value", "least-squares slope, 1 = same scale",
        ref_slope, reference=1.0,
        note=(f"Slope {ref_slope:.2f} of the instruments on the reference ({ref_note}). Far from 1 means the "
              "instruments are larger or smaller than the population covariance."),
    )
    lap("instruments")

    # ---- 5 weekly panel ---------------------------------------------------------------------------------
    d_c0 = float(np.max(np.abs(pnl.X[:, 0] - 1.0))) if pnl.n_obs else float("nan")
    checks.add("panel", "Constant column = 1", "X[:, 0] = 1 on every row", d_c0, tolerance=0.0,
               note="Column 0 of the instruments is the constant; off means the panel's columns are shifted.")
    y_ref = Y[tw[pnl.t_idx], pnl.asset_idx]
    d_y, one_y = _max_diff(pnl.y, y_ref)
    checks.add(
        "panel", "Weekly return = sum of scaled daily returns", "y_{i,t} = sum over week t's days of r_tau / d_tau",
        d_y if one_y == 0 else float("inf"), tolerance=1e-12,
        note=(f"Every one of the {pnl.n_obs} panel rows' returns is recomputed from the daily scaled returns "
              "(largest difference relative to the largest weekly return). Off means rows pair the wrong week "
              "or asset."),
    )
    keep = np.asarray(periods.isin(train_periods), dtype=bool)
    sub = pnl.subset_periods(keep)
    sd_c = sub.X.std(axis=0) if sub.n_obs else np.ones(pnl.p)
    sd_c = np.where(np.isfinite(sd_c) & (sd_c > 0.0), sd_c, 1.0)
    sd_c[0] = 1.0
    sigma_c_fit = np.asarray(fit.meta.get("sigma_c", sub.sigma_c), dtype=float)
    d_sc, _ = _max_diff(sd_c, sigma_c_fit)
    checks.add(
        "panel", "Penalty weights = training standard deviations",
        "sigma^c_l = sd of instrument l over the training rows",
        d_sc, tolerance=1e-12,
        note=("The group-lasso penalty of each topic scales with its instrument's standard deviation over the "
              "training rows (D26); recomputed here with numpy. Off means the penalty weights come from other rows."),
    )
    same_sub = bool(len(sub.periods) == len(train_periods) and (pd.DatetimeIndex(sub.periods) == train_periods).all()
                    and int(sub.n_obs) == int(fit.meta.get("n_obs", sub.n_obs)))
    checks.add(
        "panel", "Training rows = the fit's", "training weeks and row count equal the fit's",
        0.0 if same_sub else 1.0, reference=0.0, tolerance=0.0,
        note=(f"The trace refits and recomputes on the panel weeks inside the training window ({len(train_periods)} "
              f"weeks, {int(sub.n_obs)} rows); they must be the fit's own. Off means the trace looks at other rows."),
    )
    try:
        n_pred, first_pred = lab_bks.training_weeks(ts, te, bks_cfg, lead, None, w)
    except ValueError:  # pragma: no cover - defensive
        n_pred, first_pred = -1, pd.NaT
    tw_gap = float(abs(int(n_pred) - len(train_periods)) + (0 if pd.Timestamp(first_pred) == periods[0] else 1))
    checks.add(
        "panel", "Training weeks = the pre-run count", "n and first usable week = bks.training_weeks",
        tw_gap, reference=0.0, tolerance=0.0,
        note=(f"Before a run the dashboard predicts {n_pred} training weeks and a first usable week ending "
              f"{_fmt_day(first_pred)}; the panel has {len(train_periods)} and {_fmt_day(periods[0])}. Off means "
              "the pre-run check misleads (or weeks were dropped for too few assets)."),
    )
    n_per_week = np.bincount(pnl.t_idx, minlength=len(periods))
    min_assets = int(pcfg.data.min_assets_per_period)
    checks.add(
        "panel", "Every week has enough assets", f"rows per kept week >= {min_assets}",
        float(np.sum(n_per_week < min_assets)), reference=0.0, tolerance=0.0,
        note=(f"The panel keeps a week only with at least {min_assets} assets (fewest here: "
              f"{int(n_per_week.min()) if len(n_per_week) else 0}). Off means a thin week entered the fit."),
    )
    # instrument stability over the training weeks (per topic, simulation order)
    Xs = sub.X[:, 1:]
    within = np.full(len(panel_topics), np.nan)
    cross = np.full(len(panel_topics), np.nan)
    if sub.n_obs:
        frame = pd.DataFrame(Xs)
        g = frame.groupby(sub.asset_idx)
        within = g.std(ddof=0).mean(axis=0).to_numpy(dtype=float)
        cross = g.mean().std(ddof=0).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        wo = within / cross
        mos = Xs.mean(axis=0) / sub.sigma_c[1:] if sub.n_obs else np.full(len(panel_topics), np.nan)
    stability = pd.DataFrame({"within_over_cross": wo[order], "mean_over_sd": mos[order]}, index=t_index)
    lap("panel")

    # ---- 6 fit and lambda ----------------------------------------------------------------------------------
    Gamma, F = np.asarray(res.Gamma, dtype=float), np.asarray(res.F, dtype=float)
    mom = sub.moments()  # on the trace's own sub-panel, never on the cached panel
    ridge = 0.0 if lam == 0.0 else float(RIDGE)
    ssr = si.ssr_from_moments(mom.S, mom.V, mom.yy, Gamma, F)
    if lam > 0.0:
        obj = si.objective_value(sub, Gamma, F, lam, sub.sigma_c, bool(est.penalize_intercept))
        rel_obj = "0.5 SSR + lambda N_S sum_l sigma_l ||Gamma_l|| + sum_t ||f_t||^2 (BKS Eq. 8)"
    else:
        obj = ssr
        rel_obj = "SSR (at lambda = 0 the fit is plain IPCA and reports the sum of squared residuals)"
    checks.add(
        "fit", "Objective recompute", rel_obj, abs(obj - float(res.objective)) / max(abs(float(res.objective)), 1e-300),
        tolerance=1e-10,
        note=(f"The fit's reported objective ({float(res.objective):,.4g}) recomputed from Gamma and F on the "
              "training rows (relative difference). Off means the fit optimised something else."),
    )
    F_cf = si.f_step(mom.S, mom.V, Gamma, ridge=ridge)
    d_F = float(np.max(np.abs(F_cf - F)) / max(float(np.max(np.abs(F))), 1e-300)) if F.size else float("nan")
    checks.add(
        "fit", "Factors = closed-form factor step",
        f"f_t = (Gamma' S_t Gamma + {ridge:g} I)^-1 Gamma' V_t (BKS Eq. 16)",
        d_F, tolerance=1e-10,
        note=("Each training week's factor is the exact least-squares (ridge) solution for the fitted Gamma "
              "(relative to the largest factor). Off means the stored factors belong to another Gamma."),
    )
    syy = float(np.sum(mom.yy))
    r2_rec = 1.0 - ssr / syy if syy > 0 else float("nan")
    checks.add(
        "fit", "In-sample R2 recompute", "total R2 = 1 - SSR / sum y^2 over the training rows",
        abs(r2_rec - float(res.total_r2)), tolerance=1e-8,
        note=f"In-sample total R2 {float(res.total_r2):.4f}; off means the reported fit quality is not this fit's.",
    )
    pop_mask = np.asarray(res.populated if res.populated is not None else mom.n > 0, dtype=bool)
    Fp = F[pop_mask]
    mu_rec = Fp.mean(axis=0) if len(Fp) else np.zeros(K)
    Sig_rec = np.atleast_2d(np.cov(Fp, rowvar=False, ddof=1)).reshape(K, K) if len(Fp) > 1 else np.zeros((K, K))
    Sig_rec = 0.5 * (Sig_rec + Sig_rec.T)
    q_mve = float(mu_rec @ np.linalg.pinv(Sig_rec, rcond=rcond) @ mu_rec)
    sr_rec = float(np.sqrt(max(q_mve, 0.0) * lab_bks.ANNUALIZATION))
    sr_fit = float(fit.meta.get("is_sharpe", np.nan))
    checks.add(
        "fit", "In-sample Sharpe ratio recompute",
        "sqrt(52 mu_f' Sigma_ff^+ mu_f) from the factors' mean and covariance",
        abs(sr_rec - sr_fit) / max(abs(sr_fit), 1e-12), tolerance=1e-10,
        note=(f"The criterion lambda is tuned on (annualised in-sample Sharpe ratio of the factors' best "
              f"combination, {sr_fit:.3f}) recomputed from the stored factors. Off means the tuning scored "
              "something else."),
    )
    Sff = np.atleast_2d(np.asarray(res.Sigma_ff, dtype=float))
    dg = np.diag(Sff)
    top = max(float(np.max(np.abs(dg))), 1e-300)
    canon = max(
        float(np.max(np.abs(Sff - np.diag(dg)))) / top,
        float(max(np.max(np.diff(dg)), 0.0)) / top if K > 1 else 0.0,
        float(max(-np.min(np.asarray(res.mu_f, dtype=float)), 0.0)) / max(1.0, float(np.linalg.norm(res.mu_f))),
    )
    checks.add(
        "fit", "Factors in canonical form", "Sigma_ff diagonal and descending, mean of each factor >= 0 (D24)",
        canon, tolerance=1e-10,
        note=("The factors are rotated so that they are uncorrelated, ordered by variance and have positive means; "
              "the labels f1..fK and the plots rely on it. Off means the rotation was not applied."),
    )
    # stationarity: KKT ratio and the penalty-ridge balance (lambda > 0 only)
    p = int(sub.p)
    Gt = si.to_standardized(Gamma, sub.sigma_c)
    active = np.linalg.norm(Gt, axis=1) > 0.0
    pen_orig = si.penalty_vector(lam, sub.n_obs, sub.sigma_c, bool(est.penalize_intercept))
    ratio = np.full(p, np.nan)
    if lam > 0.0:
        mt = si.standardized_moments(mom, sub.sigma_c)
        SG = np.matmul(mt.S, Gt)  # (T, p, K): S~_t Gamma~
        grad = (np.einsum("tpk,tk->tp", SG, F) - mt.V).T @ F  # d(0.5 SSR)/dGamma~ = sum_t (S~_t Gamma~ f_t - V~_t) f_t'
        pen_t = si.standardized_penalties(lam, sub.n_obs, p, bool(est.penalize_intercept))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(pen_t > 0.0, np.linalg.norm(grad, axis=1) / np.where(pen_t > 0.0, pen_t, 1.0), np.nan)
        act_dev = np.abs(ratio[active & np.isfinite(ratio)] - 1.0)
        ina_dev = np.maximum(ratio[~active & np.isfinite(ratio)] - 1.0, 0.0)
        kkt_value = float(max(act_dev.max() if act_dev.size else 0.0, ina_dev.max() if ina_dev.size else 0.0))
        checks.add(
            "fit", "Stationarity of the group lasso (KKT)",
            "||grad_l|| / penalty_l = 1 on kept rows, <= 1 on dropped rows", kkt_value, tolerance=_KKT_TOL,
            note=(f"At the solution each kept topic's gradient balances its penalty and a dropped topic's gradient "
                  f"stays below it; largest violation shown (the fit stops on a relative objective change, so about "
                  f"1e-4 to 2e-3 is normal; the closest dropped topic is at "
                  f"{np.nanmax(ratio[~active]) if (~active & np.isfinite(ratio)).any() else float('nan'):.3f}). Off "
                  "means the fit did not converge to a stationary point."),
        )
        bal_p = float(pen_orig @ np.linalg.norm(Gamma, axis=1))
        bal_r = 2.0 * float(np.sum(F**2))
        checks.add(
            "fit", "Penalty and ridge in balance", "sum_l pen_l ||Gamma_l|| = 2 sum_t ||f_t||^2",
            abs(bal_p / bal_r - 1.0) if bal_r > 0 else float("nan"), reference=0.0, tolerance=_KKT_TOL,
            note=("Rescaling Gamma up and the factors down leaves the fit unchanged, so at a stationary point the "
                  "penalty equals twice the factors' ridge term (relative gap shown). Off means Gamma's scale "
                  "is not at the optimum."),
        )
    else:
        checks.info("fit", "Stationarity of the group lasso (KKT)", "not applicable at lambda = 0", float("nan"),
                    kind=IDENTITY,
                    note="At lambda = 0 there is no penalty: the fit is plain IPCA with Gamma'Gamma = I.")
        checks.info("fit", "Penalty and ridge in balance", "not applicable at lambda = 0", float("nan"),
                    kind=IDENTITY,
                    note="At lambda = 0 the factor step has no ridge and Gamma'Gamma = I fixes the scale.")
    ev_ff = np.linalg.eigvalsh(0.5 * (Sff + Sff.T))
    dead_ratio = float(ev_ff.min() / ev_ff.max()) if ev_ff.size and ev_ff.max() > 0 else 0.0
    dead = not dead_ratio > _DEAD_RATIO
    k_eff = int(np.sum(ev_ff > _DEAD_RATIO * max(float(ev_ff.max()), 0.0))) if ev_ff.size else 0
    checks.add(
        "fit", "Every factor is alive", "smallest / largest eigenvalue of Sigma_ff > 1e-6", dead_ratio,
        tolerance=_DEAD_RATIO, kind=DIAGNOSTIC, status=OFF if dead else OK,
        note=(f"{k_eff} of the K = {K} factors carry variance (smallest / largest eigenvalue {dead_ratio:.1e}). "
              + ("A factor is dead: effective K is smaller, and the Sharpe criterion drops that direction."
                 if dead else "A dead factor (variance about 0) would make the effective K smaller.")),
    )
    tuning = fit.tuning
    path = gamma_path = ptrace = None
    zero_obj = 0.5 * syy
    if tuning is not None:
        pts = tuning.path
        crit = np.array([np.nan if q.criterion is None else float(q.criterion) for q in pts], dtype=float)
        lams = np.array([float(q.lam) for q in pts])
        n_sel = np.array([int(q.n_selected) for q in pts])
        Ks = np.array([int(q.K) for q in pts])
        tol_rule = max(1e-9, float(fit.meta.get("tolerance", 0.0) or 0.0))
        fin = np.isfinite(crit)
        best_i = int(np.nanargmax(crit)) if fin.any() else -1
        best_v = float(crit[best_i]) if best_i >= 0 else float("nan")
        band = tol_rule * max(1.0, abs(best_v)) if best_i >= 0 else float("nan")
        in_band = fin & (crit >= best_v - band) if best_i >= 0 else np.zeros(len(pts), dtype=bool)
        cand = np.flatnonzero(in_band)
        pick = int(max(cand, key=lambda i: (lams[i], -n_sel[i], -Ks[i]))) if cand.size else -1
        chosen = int(tuning.meta.get("chosen_index", -1))
        checks.add(
            "fit", "Chosen lambda follows the band rule",
            "the largest lambda with criterion >= best - tol x max(1, |best|)",
            0.0 if (pick == chosen and np.isclose(lams[chosen], lam, rtol=0, atol=0)) else 1.0,
            reference=0.0, tolerance=0.0,
            note=(f"Re-picked from the path: index {pick} (lambda {lams[pick] if pick >= 0 else float('nan'):.4g}) "
                  f"against the tuner's {chosen} (lambda {lam:.4g}); band width {band:.3g}. Off means the tuner chose "
                  "by another rule."),
        )
        A = lab_bks.ANNUALIZATION
        T_tr = max(int(sub.T), 1)
        se = np.sqrt(A * (1.0 + (crit / np.sqrt(A)) ** 2 / 2.0) / T_tr)
        trunc = list(tuning.meta.get("sigma_ff_truncated", [False] * len(pts)))
        objs = np.array([float(q.objective) for q in pts])
        above = objs > zero_obj
        path = pd.DataFrame({
            "lam": lams, "criterion": crit, "se": se, "in_band": in_band, "best": np.arange(len(pts)) == best_i,
            "chosen": np.arange(len(pts)) == chosen, "n_selected": n_sel,
            "total_r2": [float(q.total_r2) for q in pts], "objective": objs, "zero_objective": zero_obj,
            "above_zero": above, "converged": [bool(q.converged) for q in pts], "n_iter": [int(q.n_iter) for q in pts],
            "sigma_ff_truncated": [bool(x) for x in trunc],
        })
        norms = np.vstack([np.asarray(q.gamma_norms, dtype=float) for q in pts]) * np.asarray(sub.sigma_c)[None, :]
        gamma_path = pd.DataFrame(norms, index=pd.Index(lams, name="lam"),
                                  columns=pd.Index(instr_names, name="instrument"))
        checks.add(
            "fit", "Path stays below the all-zero objective", "objective at every lambda <= 0.5 sum y^2",
            float(above.sum()), reference=0.0, tolerance=0.0, kind=DIAGNOSTIC, status=OK if not above.any() else INFO,
            note=(f"{int(above.sum())} of {len(pts)} path point(s) end above the objective of Gamma = 0 "
                  f"({zero_obj:,.1f}): a warm-started fit at large lambda can stop at a spurious stationary point "
                  "(D22). The tuner can still pick another point."),
        )
        grid = np.asarray(tuning.meta.get("lam_grid", {}).get(K, lams), dtype=float)
        lm_t = float(tuning.lam_max)
        grid_ref = np.logspace(np.log10(float(bks_cfg.lambda_ratio) * lm_t), np.log10(lm_t), int(bks_cfg.n_lambdas))
        d_grid = (float(np.max(np.abs(np.sort(grid) - grid_ref)) / lm_t) if grid.shape == grid_ref.shape
                  else float("inf"))
        checks.add(
            "fit", "Lambda grid = log-spaced below lambda_max", "n points from ratio x lambda_max to lambda_max",
            d_grid, tolerance=1e-12,
            note=(f"{len(grid)} grid points from {grid.min():.4g} to {grid.max():.4g}; off means the path used another "
                  "grid than the settings say."),
        )
        if L <= LAMBDA_MAX_MAX_TOPICS:
            lm = float(si.lambda_max(sub, est, K))
            checks.add(
                "fit", "lambda_max recompute", "largest lambda that keeps a topic, from the warm-up factors (D22)",
                abs(lm - lm_t) / max(abs(lm_t), 1e-300), tolerance=1e-8,
                note=(f"lambda_max {lm_t:.5g} recomputed on the training rows with the run's own estimation settings "
                      "(not the panel's). Off means the grid was built from another sample or settings."),
            )
        else:
            checks.info("fit", "lambda_max recompute", "skipped above 100 topics", float("nan"), kind=IDENTITY,
                        note=f"Skipped for cost: {L} topics (one warm start and a bisection on the training rows).")
        run_trace = (L <= PATH_TRACE_MAX_TOPICS) if path_trace is None else bool(path_trace)
        if run_trace:
            ptrace, repro = _path_refit(panel, fit, sim, shocks, truth, window, sub, est, chain.cov[rows_obj.has])
            checks.add(
                "fit", "Refitting the path reproduces the fit", "the refit at the chosen lambda gives the same Gamma",
                repro, tolerance=1e-8,
                note=("The per-lambda trace refits the whole path with warm starts; at the chosen index it must "
                      "reproduce the cached Gamma (largest difference relative to the largest entry). Off means the "
                      "per-lambda numbers belong to another fit."),
            )
            path_note = f"per-lambda refit of {len(ptrace)} points"
        else:
            path_note = (f"per-lambda refit skipped: {L} topics > {PATH_TRACE_MAX_TOPICS}" if path_trace is None
                         else "per-lambda refit switched off")
            checks.info("fit", "Refitting the path reproduces the fit", "skipped", float("nan"), kind=IDENTITY,
                        note=f"Not run: {path_note}.")
    else:
        path_note = "fixed lambda: no path"
        for name in ("Chosen lambda follows the band rule", "lambda_max recompute",
                     "Refitting the path reproduces the fit"):
            checks.info("fit", name, "not applicable to the fixed rule", float("nan"), kind=IDENTITY,
                        note=f"The fixed rule fits one lambda ({lam:g}); there is no path to check.")
    gamma_std = pd.DataFrame(Gamma * np.asarray(sub.sigma_c)[:, None], index=pd.Index(instr_names, name="instrument"),
                             columns=[f"f{k + 1}" for k in range(K)])
    kkt = pd.DataFrame({"ratio": ratio, "active": active, "penalty": pen_orig},
                       index=pd.Index(instr_names, name="instrument"))
    factors_in_sample = pd.DataFrame(np.array(F, copy=True), index=pd.DatetimeIndex(sub.periods, name="period"),
                                     columns=[f"f{k + 1}" for k in range(K)])
    lap("fit")

    # ---- 7 forecast weeks -----------------------------------------------------------------------------------
    weeks, fc = _forecast_weeks(panel, fit, result, ridge)
    checks.add(
        "forecast", "Forecast factors = closed form", f"f_t = (B'B + {ridge:g} I)^-1 B'y_t, B = C Gamma, each week",
        fc["d_factor"], tolerance=1e-10,
        note=(f"Each of the {len(weeks)} forecast weeks' factors is re-solved from that week's own rows (relative to "
              "the largest factor). Off means the forecast factors were fitted differently."),
    )
    checks.add(
        "forecast", "First-order condition of the factor step", f"B'(y - B f) = {ridge:g} f",
        fc["foc"], tolerance=1e-8,
        note=("The stored factor solves its week's (ridge) least squares: the residual is orthogonal to the "
              "loadings up to the ridge term (relative to the largest B'y). Off means the factor is not optimal."),
    )
    checks.add(
        "forecast", "Fitted and realised panels recompute", "fitted = C Gamma f_t; realised = the panel's y",
        fc["d_fitted"], tolerance=1e-12,
        note="The weekly fitted and realised values the R2 uses are rebuilt from the rows; off means they differ.",
    )
    checks.add(
        "forecast", "Pooled OOS R2 recompute", "1 - sum (y - fitted)^2 / sum y^2 over the forecast asset-weeks",
        abs(fc["r2_pooled"] - float(result.r2_pooled)), tolerance=1e-12,
        note=f"Pooled OOS R2 {float(result.r2_pooled):.4f} in panel units; off means it is computed on other cells.",
    )
    checks.add(
        "forecast", "Shuffled-instrument reference recompute", "same seeded shuffles, same pooled R2",
        abs(fc["r2_shuffled_pooled"] - float(result.meta.get("shuffled_r2_pooled", np.nan))), tolerance=1e-12,
        note=(f"The reference R2 with the topic instruments shuffled across assets "
              f"({float(result.meta.get('shuffled_r2_pooled', np.nan)):.4f}) is replayed week by week, so the per-week "
              "values in the table are the evaluation's own."),
    )
    oos_ridge = float(result.meta.get("oos_ridge", np.nan))
    checks.add(
        "forecast", "Ridge follows lambda", "ridge = 0 at lambda = 0, else 2",
        abs(oos_ridge - ridge), reference=0.0, tolerance=0.0,
        note=(f"Ridge {oos_ridge:g} at lambda {lam:.4g}: plain IPCA (lambda 0) has unit-norm Gamma columns and no "
              "ridge in its factor step, so a ridge of 2 there would shrink the factors to about 0."),
    )
    lap("forecast")

    # ---- 8 implied sensitivities -----------------------------------------------------------------------------
    B_chain = chain.to_B(chain.m)
    m0_ret = chain.conv[:, None] * chain.m_const[None, :]
    B_const = ((m0_ret @ chain.Sz_pinv) * (chain.zs[None, :] / chain.ret_scale[:, None])).T
    B_const[:, chain.zero] = 0.0
    B_prod = implied.B_hat.reindex(index=topics, columns=assets).to_numpy(dtype=float)
    Bc_prod = implied.meta["B_const"].reindex(index=topics, columns=assets).to_numpy(dtype=float)
    d_chain = max(_max_diff(B_chain, B_prod, relative=False)[0], _max_diff(B_const, Bc_prod, relative=False)[0])
    checks.add(
        "implied", "Step-by-step chain = production",
        "recomputed B_hat and constant part = the production implied sensitivities",
        d_chain, tolerance=1e-12,
        note=("The trace's own Eq. 5 chain (rows, projection, unit conversion, Sigma_z, standardisation) gives the "
              "production sensitivities to rounding (largest absolute difference), so every intermediate shown "
              "belongs to them. Off means the trace or the production code deviates."),
    )
    has = rows_obj.has
    m_alt = chain.m_proj + chain.m_const[None, :]
    d_proj = _max_diff(chain.m[has], m_alt[has])[0] if has.any() else float("nan")
    checks.add(
        "implied", "Implied covariance = projected instrument + constant", "m_i = P c~_i + M Gamma_0'",
        d_proj, tolerance=1e-12,
        note=("BKS Eq. 5 keeps only the part of each asset's instruments inside the K directions of Gamma_tilde, "
              "plus the constant's share (relative difference). This is what makes the step lossy."),
    )
    cov0 = np.where(np.isfinite(chain.cov), chain.cov, 0.0)
    svd = _instrument_svd(cov0[has])
    cap = _capture(chain.cov[has], chain.P, chain.Mmap, K, rcond, svd)
    kept, best = cap["kept_share"], cap["best_share"]
    checks.add(
        "implied", "Kept share <= best K share", "||C P||^2 / ||C||^2 <= best rank-K share (the reference)",
        kept, reference=best, tolerance=1e-12,
        status=OK if np.isfinite(kept) and np.isfinite(best) and kept <= best + 1e-12 else OFF,
        note=(f"The fit's directions keep {kept:.1%} of the instruments' squared norm; the best {K} directions keep "
              f"{best:.1%} (random {cap['random_share']:.1%}). No K directions can beat the best, so off means the "
              "shares are computed wrongly."),
    )
    stale = [a for i, a in enumerate(assets) if rows_obj.has[i] and pd.Timestamp(rows_obj.week.iloc[i]) != row_week]
    checks.add(
        "implied", "Every asset uses the last training week", f"row week = {_fmt_day(row_week)} for every asset",
        float(len(stale)), reference=0.0, tolerance=0.0, kind=DIAGNOSTIC, status=OK if not stale else INFO,
        note=(f"{len(stale)} asset(s) use an earlier week's row (no row in the last training week); "
              f"{int((~rows_obj.has).sum())} asset(s) have no training row and get zero sensitivities."),
    )
    # the kernel Sigma_z of the Eq. 5 instruments' own days
    Zk = z.reindex(columns=topics).to_numpy(dtype=float)
    okk = np.isfinite(Zk).all(axis=1) & (wts_row > 0.0)
    if okk.any():
        ww = wts_row[okk] / wts_row[okk].sum()
        Dk = Zk[okk] - ww @ Zk[okk]
        Sz_kernel = (Dk * ww[:, None]).T @ Dk
    else:
        Sz_kernel = np.full((L, L), np.nan)
    Szk_pinv = _sym_pinv(Sz_kernel, rcond)[0] if np.isfinite(Sz_kernel).all() else np.zeros((L, L))
    lap("implied")

    # ---- the reference ladder --------------------------------------------------------------------------------
    Vk = np.zeros((L, 0)) if svd is None else svd[1][: min(K, L)].T  # the best K directions of the instruments
    variants_np = {
        "oracle": Bt,
        "window_truth": wt.reindex(index=topics, columns=assets).to_numpy(dtype=float),
        "instruments_kernel": chain.to_B(chain.cov, Szk_pinv),
        "instruments_train": chain.to_B(chain.cov),
        "best_rank": chain.to_B(cov0 @ Vk @ Vk.T),
        "bks_no_const": chain.to_B(chain.m_proj),
        "bks_implied": B_prod,
    }
    variants = {k: pd.DataFrame(np.array(v, copy=True), index=t_index, columns=a_index) for k, v in variants_np.items()}
    tau = float(implied.meta.get("select_tau", 0.05))
    scores = {k: _score(v, implied, sim, shocks, window, truth, tau) for k, v in variants.items()}
    ladder = pd.DataFrame(
        {"label": [LADDER[k] for k in LADDER], "spearman": [scores[k][0] for k in LADDER],
         "rmse": [scores[k][1] for k in LADDER], "median_r2": [scores[k][2] for k in LADDER]},
        index=pd.Index(list(LADDER), name="variant"),
    )
    ladder["d_spearman"] = ladder["spearman"].diff()
    ladder["d_median_r2"] = ladder["median_r2"].diff()
    ladder["what"] = [LADDER_WHAT[k] for k in LADDER]
    lap("ladder")

    # ---- step tables -------------------------------------------------------------------------------------------
    conversion = np.ones(N)
    if scaled:
        sc_al = al.scale.reindex(columns=assets).to_numpy(dtype=float)
        r_al = al.returns.reindex(columns=assets).to_numpy(dtype=float)
        for i in range(N):
            ok = np.isfinite(r_al[:, i]) & np.isfinite(sc_al[:, i]) & (sc_al[:, i] > 0) & ok_z
            conversion[i] = (chain.divisor[i] * float((wts_row[ok] / wts_row[ok].sum()) @ (1.0 / sc_al[ok, i]))
                             if ok.any() and np.isfinite(chain.divisor[i]) else np.nan)
    units = pd.DataFrame(
        {"divisor": chain.divisor, "ret_scale": chain.ret_scale, "asset_vol": asset_vol,
         "divisor_over_ret_scale": chain.divisor / chain.ret_scale, "conversion": conversion, "skipped": chain.zero},
        index=a_index,
    )
    live = ~chain.zero
    if scaled and history == "training":
        d_u = float(np.nanmax(np.abs(chain.divisor[live] / chain.ret_scale[live] - 1.0))) if live.any() else 0.0
        checks.add(
            "align", "Unit conversion is exact", "divisor = training return scale for every asset",
            d_u, tolerance=1e-12,
            note=("Under the training history the divisor is each asset's training standard deviation on every day, "
                  "so converting the implied covariances back to returns is exact. Off means the divisor is another "
                  "one."),
        )
    elif scaled:
        conv_dev = float(np.nanmax(np.abs(conversion[live] - 1.0))) if live.any() else float("nan")
        checks.info(
            "align", "Unit conversion is approximate",
            "u_i = divisor_i x kernel mean of 1 / trailing volatility, 1 = exact",
            conv_dev, reference=0.0,
            note=(f"The implied covariances are multiplied by the mean training divisor, but the instruments average "
                  f"returns scaled by a trailing volatility over the whole kernel; u ranges "
                  f"{np.nanmin(conversion[live]):.2f} to {np.nanmax(conversion[live]):.2f} across assets (largest gap "
                  "from 1 shown). It rescales each asset's sensitivities."),
        )
    else:
        checks.add("align", "Unit conversion is exact", "divisor = 1 without asset weighting",
                   float(np.max(np.abs(chain.divisor - 1.0))), tolerance=0.0,
                   note="Without asset weighting the panel is in return units and nothing is converted.")
    lap("tables")

    # ---- chain frames -------------------------------------------------------------------------------------------
    def at(x: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(np.array(x, copy=True), index=a_index, columns=t_index)

    m_ret = np.where(has[:, None], chain.m, 0.0) * chain.conv[:, None]
    chain_out: dict[str, Any] = {
        "beta": pd.DataFrame(chain.beta, index=a_index, columns=[f"f{k + 1}" for k in range(K)]),
        "m_proj": at(chain.m_proj),
        "m_const": pd.Series(chain.m_const, index=t_index, name="m_const"),
        "m": at(chain.m),
        "divisor": pd.Series(chain.divisor, index=a_index, name="divisor"),
        "conversion": pd.Series(chain.conv, index=a_index, name="conversion"),
        "m_ret": at(m_ret),
        "sigma_z": {
            "train": pd.DataFrame(chain.Sigma_z, index=t_index, columns=t_index),
            "kernel": pd.DataFrame(Sz_kernel, index=t_index, columns=t_index),
            "population": pd.DataFrame(pop["var_z"], index=t_index, columns=t_index),
        },
        "b_raw": at(m_ret @ chain.Sz_pinv),
        "B_hat": pd.DataFrame(B_chain, index=t_index, columns=a_index),
        "B_const": pd.DataFrame(B_const, index=t_index, columns=a_index),
        "ret_scale": pd.Series(chain.ret_scale, index=a_index, name="ret_scale"),
        "z_scale": pd.Series(chain.zs, index=t_index, name="z_scale"),
        "projector": pd.DataFrame(chain.P, index=t_index, columns=t_index),
        "gamma_rank": int(chain.gamma_rank),
        "sigma_z_rank": int(chain.sz_rank),
        "row_period": rows_obj.week.copy(),
        "skipped_assets": [a for a, zr in zip(assets, chain.zero) if zr],
        "stale_assets": stale,
    }

    # ---- key numbers, findings -----------------------------------------------------------------------------------
    sp = ladder["spearman"]
    key_numbers = {
        "inputs": f"{L} topics x {N} assets; training {_fmt_day(ts)} to {_fmt_day(te)}",
        "align": (f"divisor / training return scale {np.nanmin(units['divisor_over_ret_scale']):.2f} to "
                  f"{np.nanmax(units['divisor_over_ret_scale']):.2f}" if scaled else "no asset weighting"),
        "shocks": (f"attenuation {np.nanmean(att_true):.2f}; training / population sd {np.nanmin(ratio_sd):.2f} to "
                   f"{np.nanmax(ratio_sd):.2f}"),
        "instruments": f"corr with population {ref_corr:.2f}; kernel before training {share_all:.0%}",
        "panel": (f"{len(train_periods)} training weeks, {int(sub.n_obs)} rows; within / across assets "
                  f"{np.nanmedian(wo):.2f}"),
        "fit": f"lambda {lam:.4g}, {int(res.n_selected)} of {L} topics, in-sample Sharpe {sr_fit:.2f}",
        "forecast": (f"pooled OOS R2 {float(result.r2_pooled):.1%} (shuffled "
                     f"{float(result.meta.get('shuffled_r2_pooled', np.nan)):.1%})"),
        "implied": f"kept {kept:.0%} of the instruments (best {best:.0%}); Spearman {sp['bks_implied']:.2f}",
    }
    findings = _findings(
        checks.items, cap=cap, K=K, history=history, share=share_all, eff_days=eff_days, n_pairs=chain.n_pairs,
        ladder=ladder, path=path, rule=rule, tolerance=float(fit.meta.get("tolerance", 0.0) or 0.0), dead=dead,
        dead_ratio=dead_ratio, k_eff=k_eff, ref_corr=ref_corr, ref_slope=ref_slope, instrument_week=instrument_week,
        window_end=window_end, train_end=te, cal=sim.market.calendar, conversion=conversion[live],
        scaled=scaled, zero_obj=zero_obj,
    )

    # ---- settings and shapes tables for the page -------------------------------------------------------------------
    hl = float(bks_cfg.half_life_months)
    settings = pd.DataFrame(
        [
            ("History", lab_bks.HISTORY_LABELS.get(history, history)),
            ("Period", "calendar weeks (last trading day)"),
            ("Lead (days)", f"{lead}"),
            ("Shock window w (days)", f"{w}"),
            ("Asset weighting", {"inverse_vol": "divide by volatility", "none": "none"}.get(weighting, weighting)),
            ("Kernel half-life (months) -> weekly decay xi", f"{hl:g} -> {xi:.6f}"),
            ("Burn-in (weeks)", f"{burn_in}"),
            ("Fewest days per instrument", f"{min_days}"),
            ("Days left out at the week end", f"{skip}"),
            ("Fewest assets per week", f"{min_assets}"),
            ("Factors K", f"{K}"),
            ("Lambda rule", _RULE_LABELS.get(rule, rule)),
            ("Tolerance", f"{float(fit.meta.get('tolerance', 0.0) or 0.0):g}"),
            ("Lambda grid", (f"{int(bks_cfg.n_lambdas)} points, ratio {float(bks_cfg.lambda_ratio):g}"
                             if tuning is not None else f"fixed lambda {lam:g}")),
            ("Constant penalised", "yes" if bool(est.penalize_intercept) else "no"),
            ("Forecast ridge", f"{ridge:g}"),
            ("Training window", f"{_fmt_day(ts)} to {_fmt_day(te)}"),
            ("Training weeks",
             f"{len(train_periods)} ({_fmt_day(train_periods.min())} to {_fmt_day(train_periods.max())})"),
            ("Forecast window", f"{_fmt_day(window.forecast_start)} to {_fmt_day(window.forecast_end)}"),
            ("Forecast weeks", f"{len(forecast_periods)} ("
                               f"{', '.join(_fmt_day(x) for x in forecast_periods)})"),
        ],
        columns=["setting", "value"],
    )
    shapes = pd.DataFrame(
        [
            ("Simulation days", len(sim.market.calendar),
             f"{_fmt_day(sim.market.calendar[0])} to {_fmt_day(sim.market.calendar[-1])}"),
            ("Panel days", len(days), f"{_fmt_day(days[0])} to {_fmt_day(days[-1])}"),
            ("Topics", L, ""),
            ("Assets", N, ""),
            ("Instrument weeks", len(week_ends), ""),
            ("Panel return weeks", len(periods), f"{_fmt_day(periods[0])} to {_fmt_day(periods[-1])}"),
            ("Panel rows", int(pnl.n_obs), ""),
            ("Training weeks", len(train_periods), ""),
            ("Training rows", int(sub.n_obs), ""),
            ("Training return days of the direct estimator", int(chain.n_pairs), ""),
            ("Forecast weeks", len(forecast_periods), ""),
        ],
        columns=["quantity", "value", "note"],
    )
    if not scaled:
        notes.append("no asset weighting: panel units are return units")
    timings["total"] = time.perf_counter() - t_all
    meta: dict[str, Any] = {
        "timings": timings,
        "notes": notes,
        "settings": settings,
        "shapes": shapes,
        "key_numbers": key_numbers,
        "train_start": ts,
        "train_end": te,
        "forecast_start": pd.Timestamp(window.forecast_start),
        "forecast_end": pd.Timestamp(window.forecast_end),
        "xi": xi,
        "skip_days": skip,
        "min_days": min_days,
        "burn_in_weeks": burn_in,
        "asset_weighting": weighting,
        "kernel_share_before_train": share_all,
        "kernel_share_predicted": share_pred,
        "effective_days": eff_days,
        "n_pairs": int(chain.n_pairs),
        "instrument_ref_corr": ref_corr,
        "instrument_ref_slope": ref_slope,
        "path_trace_note": path_note,
        "n_clipped": n_clipped,
        "dead_factor": bool(dead),
        "effective_K": int(k_eff),
        "zero_objective": zero_obj,
        "oos_ridge": ridge,
        "selected_topics": list(res.selected_topics),
        "sizes": {"L": L, "N": N, "T_train": int(sub.T), "n_obs_train": int(sub.n_obs)},
    }
    trace = BKSTrace(
        history=history, lead_days=lead, shock_window=w, K=K, lam=lam, lambda_rule=rule, topics=list(topics),
        assets=list(assets), panel_topics=panel_topics, train_periods=train_periods, forecast_periods=forecast_periods,
        row_week=row_week, instrument_week=instrument_week, instrument_window_end=window_end,
        z=z, week_ends=week_ends, population=population, window_truth=wt,
        B_true_train_units=pd.DataFrame(B_tu, index=t_index, columns=a_index),
        instruments=at(chain.cov), chain=chain_out, variants=variants, ladder=ladder, capture=cap, path=path,
        gamma_path=gamma_path, path_trace=ptrace, gamma_std=gamma_std, kkt=kkt, factors_in_sample=factors_in_sample,
        units=units, shock_table=shock_table, stability=stability, weeks=weeks, checks=_ordered(checks.items),
        findings=findings, meta=meta,
    )
    n_off = sum(c.status == OFF and c.kind == IDENTITY for c in trace.checks)
    logger.info(
        "build_trace: %s history, K=%d lambda=%.4g, %d checks (%d identity off), %d findings, kept share %.3f "
        "(best %.3f), Spearman %.3f (%.2fs)",
        history, K, lam, len(trace.checks), n_off, len(findings), kept, best, float(sp["bks_implied"]),
        timings["total"],
    )
    return trace


def _ordered(items: list[TraceCheck]) -> list[TraceCheck]:
    """Checks sorted by step order (stable within a step)."""
    rank = {k: i for i, k in enumerate(STEPS)}
    return sorted(items, key=lambda c: rank.get(c.step, len(rank)))


def _score(B: pd.DataFrame, implied: DirectFit, sim: SimData, shocks: ObservedShocks, window: WindowConfig,
           truth: SimTruth, tau: float) -> tuple[float, float, float]:
    """Spearman with ``B_true``, RMSE and median forecast-window OOS R2 of ``B``, scored like the Compare tab."""
    fit = replace(implied, B_hat=B, selected=B.abs() >= tau)
    ev = evaluate_window(sim, shocks, fit, window, truth)
    return float(ev.recovery["spearman"]), float(ev.recovery["rmse"]), float(median_finite(ev.r2))


def _forecast_weeks(panel: BKSPanel, fit: BKSFit, result: BKSLabResult,
                    ridge: float) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per forecast week: factors re-solved, fitted values, R2 and the seeded shuffled reference replayed."""
    pnl = panel.panel
    Gamma = np.asarray(fit.fit.Gamma, dtype=float)
    K = int(Gamma.shape[1])
    periods = pd.DatetimeIndex(pnl.periods)
    ends = pd.DatetimeIndex(result.periods)
    idx = periods.get_indexer(ends)
    slices = dict(pnl.period_slices())
    factors = np.array(result.meta["factors"].to_numpy(dtype=float), copy=True)  # never a view of the cached result
    fp_st = result.meta["fitted_panel"]
    rp_st = result.meta["realized_panel"]
    col = {str(a): j for j, a in enumerate(fp_st.columns)}
    rng = np.random.default_rng([lab_bks.SHUFFLE_STREAM])
    n_sh = int(lab_bks.N_SHUFFLES)
    rows = []
    d_factor = foc = d_fit = 0.0
    sse_tot = syy_tot = sse_sh_tot = 0.0
    for wk, t in enumerate(idx):
        sl = slices.get(int(t), slice(0, 0)) if t >= 0 else slice(0, 0)
        if sl.stop <= sl.start:
            rows.append({"n_assets": 0, "r2": np.nan, "r2_shuffled": np.nan})
            continue
        C, y, ai = pnl.X[sl], pnl.y[sl], pnl.asset_idx[sl]
        B = C @ Gamma
        f = factors[wk]
        if ridge > 0.0:
            f_cf = np.linalg.solve(B.T @ B + ridge * np.eye(K), B.T @ y)
        else:
            f_cf = np.linalg.pinv(B.T @ B, rcond=1e-12, hermitian=True) @ (B.T @ y)
        d_factor = max(d_factor, float(np.max(np.abs(f_cf - f))) / max(float(np.max(np.abs(f))), 1e-300))
        By = B.T @ y
        foc = max(foc, float(np.max(np.abs(B.T @ (y - B @ f) - ridge * f))) / max(float(np.max(np.abs(By))), 1e-300))
        fitted = B @ f
        cols = [col[str(pnl.assets[a])] for a in ai]
        d_f, _ = _max_diff(fp_st.to_numpy(dtype=float)[wk, cols], fitted)
        d_r, _ = _max_diff(rp_st.to_numpy(dtype=float)[wk, cols], y)
        d_fit = max(d_fit, d_f, d_r)
        sse, syy = float(np.sum((y - fitted) ** 2)), float(np.sum(y**2))
        sse_sh = 0.0
        for _ in range(n_sh):  # the evaluation's shuffles, same generator and order
            Cs = C.copy()
            Cs[:, 1:] = C[rng.permutation(C.shape[0]), 1:]
            Bs = Cs @ Gamma
            if ridge > 0.0:
                fs = np.linalg.solve(Bs.T @ Bs + ridge * np.eye(K), Bs.T @ y)
            else:
                fs = np.linalg.pinv(Bs.T @ Bs, rcond=1e-12, hermitian=True) @ (Bs.T @ y)
            sse_sh += float(np.sum((y - Bs @ fs) ** 2)) / n_sh
        sse_tot += sse
        syy_tot += syy
        sse_sh_tot += sse_sh
        rows.append({"n_assets": int(sl.stop - sl.start), "r2": 1.0 - sse / syy if syy > 0 else np.nan,
                     "r2_shuffled": 1.0 - sse_sh / syy if syy > 0 else np.nan})
    weeks = pd.DataFrame(rows, index=pd.DatetimeIndex(ends, name="period"))
    weeks.insert(0, "first_day", pd.DatetimeIndex(result.meta.get("period_first_day", ends)))
    for k in range(K):
        weeks[f"f{k + 1}"] = factors[:, k]
    out = {
        "d_factor": d_factor, "foc": foc, "d_fitted": d_fit,
        "r2_pooled": 1.0 - sse_tot / syy_tot if syy_tot > 0 else float("nan"),
        "r2_shuffled_pooled": 1.0 - sse_sh_tot / syy_tot if syy_tot > 0 else float("nan"),
    }
    return weeks, out


def _path_refit(panel: BKSPanel, fit: BKSFit, sim: SimData, shocks: ObservedShocks, truth: SimTruth,
                window: WindowConfig, sub: Any, est: Any, cov_rows: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Refit the lambda path (warm starts, as the tuner did) and score every point's implied sensitivities.

    Returns the per-lambda table and the largest difference, relative to the
    largest entry, between the refit's chosen ``Gamma`` and the cached one.
    """
    tuning = fit.tuning
    assert tuning is not None
    K = int(fit.K)
    grid = [float(v) for v in tuning.meta.get("lam_grid", {}).get(K, [q.lam for q in tuning.path])]
    _, fits = si.lambda_path(sub, est, lams=grid, K=K)
    rcond = float(panel.pipeline_cfg.evaluation.rcond)
    tau = 0.05
    total = float(np.sum(cov_rows**2)) if cov_rows.size else 0.0
    out = []
    Gc = np.asarray(fit.fit.Gamma, dtype=float)
    repro = float("nan")
    chosen = int(tuning.meta.get("chosen_index", -1))
    topics = sim.topics.ids
    col = {str(t): j for j, t in enumerate(panel.panel.topics)}
    order = np.array([col[t] for t in topics], dtype=np.int64)
    for i, f in enumerate(fits):
        fc = si.canonicalize(f)
        if i == chosen:
            repro = float(np.max(np.abs(fc.Gamma - Gc)) / max(float(np.max(np.abs(Gc))), 1e-300))
        fit_i = replace(fit, fit=fc, lam=float(fc.lam), K=int(fc.K))
        imp = lab_bks.implied_exposures(panel, fit_i, sim, shocks, select_tau=tau)
        ev = evaluate_window(sim, shocks, imp, window, truth)
        Mm, _ = _lsq_map(np.asarray(fc.Gamma, dtype=float)[1:], rcond)
        P = Mm[order] @ np.asarray(fc.Gamma, dtype=float)[1:][order].T
        kept = float(np.sum((cov_rows @ P) ** 2) / total) if total > 0 else float("nan")
        crit = tuning.path[i].criterion if i < len(tuning.path) else None
        out.append({
            "lam": float(fc.lam), "n_selected": int(fc.n_selected),
            "criterion": float("nan") if crit is None else float(crit), "kept_share": kept,
            "spearman": float(ev.recovery["spearman"]), "median_r2": float(median_finite(ev.r2)),
            "gamma_rank": int(imp.meta["gamma_rank"]),
        })
    return pd.DataFrame(out, columns=["lam", "n_selected", "criterion", "kept_share", "spearman", "median_r2",
                                      "gamma_rank"]), repro


def lambda_path_trace(panel: BKSPanel, fit: BKSFit, sim: SimData, shocks: ObservedShocks, truth: SimTruth,
                      window: WindowConfig, bks_cfg: BKSLabConfig) -> pd.DataFrame:
    """Refit the lambda path and score the implied sensitivities of every point (DESIGN.md G.15.1 point 4).

    The path is refitted on the training rows with the run's own estimation
    settings (:func:`.bks.bks_pipeline_config` of ``bks_cfg``), warm-started
    in ascending lambda exactly as the tuner fitted it; each point is
    canonicalised and passed through :func:`.bks.implied_exposures` and
    :func:`.evaluate.evaluate_window`.

    Returns
    -------
    DataFrame
        One row per grid point: ``lam``, ``n_selected``, ``criterion`` (the
        tuner's), ``kept_share`` (share of the last training week's
        instruments' squared norm in the fit's directions), ``spearman``
        (with ``B_true``), ``median_r2`` (forecast window) and
        ``gamma_rank``.

    Raises
    ------
    ValueError
        For a fit with the fixed rule (no path).
    """
    if fit.tuning is None:
        raise ValueError("the fixed lambda rule has no path to trace")
    pnl = panel.panel
    periods = pd.DatetimeIndex(pnl.periods)
    sub = pnl.subset_periods(np.asarray(periods.isin(fit.train_periods), dtype=bool))
    lead, w = int(panel.pipeline_cfg.data.attention_lag_days), int(panel.pipeline_cfg.shocks.window)
    est = lab_bks.bks_pipeline_config(bks_cfg, w, lead, n_assets=int(pnl.N)).estimation
    assets = _assets(sim)
    rows = _eq5_rows(panel, fit, assets)
    col = {str(t): j for j, t in enumerate(pnl.topics)}
    order = np.array([col[t] for t in sim.topics.ids], dtype=np.int64)
    cov_rows = pnl.X[rows.row[rows.has]][:, 1:][:, order]
    frame, _ = _path_refit(panel, fit, sim, shocks, truth, window, sub, est, cov_rows)
    return frame


def _findings(checks: list[TraceCheck], *, cap: dict[str, Any], K: int, history: str, share: float,
              eff_days: float, n_pairs: int, ladder: pd.DataFrame, path: pd.DataFrame | None, rule: str,
              tolerance: float, dead: bool, dead_ratio: float, k_eff: int, ref_corr: float, ref_slope: float,
              instrument_week: pd.Timestamp, window_end: pd.Timestamp, train_end: pd.Timestamp,
              cal: pd.DatetimeIndex, conversion: np.ndarray, scaled: bool, zero_obj: float) -> list[dict[str, str]]:
    """The "where it departs" list (DESIGN.md G.16): plain English with this run's numbers."""
    out: list[dict[str, str]] = []

    def add(step: str, severity: str, title: str, text: str) -> None:
        out.append({"step": step, "severity": severity, "title": title, "text": text})

    kept, best, rnd = cap["kept_share"], cap["best_share"], cap["random_share"]
    if np.isfinite(kept) and np.isfinite(best) and kept < 0.5 * best:
        add("implied", "departure", "The fit's directions keep little of the instruments",
            f"The fit's {K} directions keep {kept:.0%} of the instruments' squared norm; the best {K} keep {best:.0%} "
            f"(random {rnd:.0%}). The Eq. 5 step discards the rest.")
    sp = ladder["spearman"]
    if history == "full" and np.isfinite(share) and share > 0.5:
        add("implied", "departure", "Instruments and the shocks' covariance cover different days",
            f"The instruments weigh {share:.0%} of their kernel before the training start (effective {eff_days:,.0f} "
            f"days), but the shocks' covariance Sigma_z uses the {n_pairs} training days. With Sigma_z over the same "
            f"history the instruments alone score Spearman {sp['instruments_kernel']:.2f} instead of "
            f"{sp['instruments_train']:.2f}.")
    if path is not None and len(path) > 1:
        crit = path["criterion"].to_numpy(dtype=float)
        rng_c = float(np.nanmax(crit) - np.nanmin(crit)) if np.isfinite(crit).any() else float("nan")
        se_med = float(np.nanmedian(path["se"]))
        if np.isfinite(rng_c) and np.isfinite(se_med) and rng_c < 2.0 * se_med:
            add("fit", "departure", "The lambda choice is within noise",
                f"The lambda criterion varies by {rng_c:.2f} over the grid against a standard error of about "
                f"{se_med:.2f}: the lambda choice and the selected topics are within noise.")
        best_v = float(np.nanmax(crit)) if np.isfinite(crit).any() else float("nan")
        if rule == "tolerance" and np.isfinite(best_v) and abs(best_v) < 1.0:
            band = max(1e-9, tolerance) * max(1.0, abs(best_v))
            add("fit", "note", "The tolerance band is absolute here",
                f"The tolerance band is absolute below a Sharpe ratio of 1: the best in-sample Sharpe ratio is "
                f"{best_v:.2f}, so every point within {band:.3g} of it counts as tied ({band / abs(best_v):.0%} of it, "
                f"not {tolerance:.0%}), and the sparsest of them wins.")
        above = path["above_zero"].to_numpy(dtype=bool)
        if above.any():
            chosen_above = bool(np.any(above & path["chosen"].to_numpy(dtype=bool)))
            add("fit", "note", "Spurious stationary points on the path",
                f"{int(above.sum())} of the {len(path)} path points end above the objective of the all-zero solution "
                f"({zero_obj:,.1f}): warm-started fits at large lambda that stopped at a spurious stationary point "
                f"(D22). The chosen point is {'' if chosen_above else 'not '}among them.")
    if dead:
        add("fit", "departure", "A factor is dead",
            f"At the chosen lambda the factors' covariance is singular (smallest to largest eigenvalue "
            f"{dead_ratio:.1e}): the fit effectively has {k_eff} of its K = {K} factors, and the Sharpe criterion "
            "ignores the dead one.")
    if np.isfinite(ref_corr) and ref_corr < _MIN_REF_CORR:
        add("instruments", "departure" if history == "full" else "note",
            "Instruments are far from their population value",
            f"The instruments of the week ending {_fmt_day(instrument_week)} correlate {ref_corr:.2f} with their "
            f"population value (slope {ref_slope:.2f}). "
            + ("With the full history's long kernel they should be close to 1: the instruments carry sampling noise "
               "or the units differ." if history == "full" else
               "Under the training history they are covariances over a few months of days, so sampling noise "
               "is expected; compare the sensitivities with the window truth."))
    if history == "full" and scaled and conversion.size and np.nanmax(np.abs(conversion - 1.0)) > 0.2:
        add("align", "note", "The unit conversion is approximate",
            f"The unit conversion of the implied sensitivities is approximate: the kernel-weighted 1/volatility times "
            f"the training divisor ranges {np.nanmin(conversion):.2f} to {np.nanmax(conversion):.2f} across assets "
            "(1 is exact).")
    n_after = int(np.sum((cal > window_end) & (cal <= train_end))) if not pd.isna(window_end) else 0
    add("implied", "note", "Which instruments the implied sensitivities use",
        f"The implied sensitivities use the instruments of the week ending {_fmt_day(instrument_week)} (kernel window "
        f"up to {_fmt_day(window_end)}), {n_after} trading days before the training end.")
    for c in checks:
        if c.kind == IDENTITY and c.status == OFF:
            add(c.step, "departure", f"Check off: {c.name}",
                f"Check off: {c.name}. Observed {c.value:.3g} against {c.relation} (tolerance {c.tolerance:.1g}). "
                f"{c.note} Identity checks are expected to hold; this one points to an implementation defect.")
    rank = {k: i for i, k in enumerate(STEPS)}
    return sorted(out, key=lambda f: (rank.get(f["step"], len(rank)), f["severity"] != "departure"))


# ---------------------------------------------------------------------------
# Per-selection helpers (cheap; the page calls them on every rerun)
# ---------------------------------------------------------------------------
def _role_of_week(week: Any, trace: BKSTrace) -> str:
    if pd.isna(week):
        return "other"
    if week in trace.train_periods:
        return "training"
    if week in trace.forecast_periods:
        return "forecast"
    return "other"


def asset_days(trace: BKSTrace, panel: BKSPanel, sim: SimData, asset: str) -> pd.DataFrame:
    """One asset's daily inputs: raw return, divisor, scaled return (step 2).

    Returns
    -------
    DataFrame
        Index the simulation calendar; columns ``raw`` (daily return),
        ``divisor`` (``aligned.scale``, 1 without weighting, ``NaN`` before
        the panel's days), ``scaled`` (``aligned.returns``), ``ret_scale``
        and ``asset_vol`` (constants), ``role`` (``"before panel"``,
        ``"training"`` or ``"forecast"`` for the days of those weeks,
        ``"other"``).
    """
    asset = str(asset)
    cal = sim.market.calendar
    ret = sim.market.returns
    raw = ret[ret.columns[[str(c) for c in ret.columns].index(asset)]].to_numpy(dtype=float)
    al = panel.aligned
    days = pd.DatetimeIndex(al.calendar)
    scaled = al.returns[asset].reindex(cal).to_numpy(dtype=float)
    if al.scale is not None:
        divisor = al.scale[asset].reindex(cal).to_numpy(dtype=float)
    else:
        divisor = np.where(np.asarray(cal.isin(days)), 1.0, np.nan)
    pid, ends = period_end_index(days, lab_bks.PERIOD)
    week_of_day = pd.Series(ends[pid], index=days).reindex(cal)
    role = np.where(np.asarray(cal < days[0]), "before panel", "other").astype(object)
    in_panel = np.asarray(cal >= days[0])
    role[in_panel & np.asarray(week_of_day.isin(trace.train_periods))] = "training"
    role[in_panel & np.asarray(week_of_day.isin(trace.forecast_periods))] = "forecast"
    return pd.DataFrame(
        {"raw": raw, "divisor": divisor, "scaled": scaled,
         "ret_scale": float(trace.units.loc[asset, "ret_scale"]),
         "asset_vol": float(trace.units.loc[asset, "asset_vol"]),
         "role": role},
        index=pd.DatetimeIndex(cal, name="date"),
    )


def topic_days(trace: BKSTrace, panel: BKSPanel, sim: SimData, shocks: ObservedShocks, topic: str) -> pd.DataFrame:
    """One topic's daily attention and shocks on the panel's days, split into signal and noise (step 3).

    Every column is paired with the return day (index): the attention and
    the lab shocks are shifted by the lead.

    Returns
    -------
    DataFrame
        Index the panel's days; columns ``attention`` (the level on the
        attention day), ``z_bks`` (the panel's shock), ``z_direct`` (the
        direct estimator's shock of the same attention day), ``z_signal``,
        ``z_news``, ``z_slow`` (exact split of the lab shock: designed signal,
        news noise, slow drift), ``designed`` (the designed shock).
    """
    topic = str(topic)
    days = pd.DatetimeIndex(panel.aligned.calendar)
    lead = int(trace.lead_days)

    def paired(s: pd.Series) -> np.ndarray:
        return s.shift(lead).reindex(days).to_numpy(dtype=float)

    att_col = sim.attention.columns[[str(c) for c in sim.attention.columns].index(topic)]
    z_sig, z_news, z_slow = _z_components(sim, trace.shock_window, [topic])
    ds_col = sim.designed_shocks.columns[[str(c) for c in sim.designed_shocks.columns].index(topic)]
    zd_col = shocks.z.columns[[str(c) for c in shocks.z.columns].index(topic)]
    return pd.DataFrame(
        {"attention": paired(sim.attention[att_col]), "z_bks": trace.z[topic].to_numpy(dtype=float),
         "z_direct": paired(shocks.z[zd_col]), "z_signal": paired(z_sig[topic]), "z_news": paired(z_news[topic]),
         "z_slow": paired(z_slow[topic]), "designed": paired(sim.designed_shocks[ds_col])},
        index=pd.DatetimeIndex(days, name="date"),
    )


def _asset_arrays(trace: BKSTrace, panel: BKSPanel, asset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scaled daily return of ``asset``, the panel's shocks and the all-topic shock mask (aligned calendar)."""
    r = panel.aligned.returns[str(asset)].to_numpy(dtype=float)
    Z = trace.z.to_numpy(dtype=float)
    return r, Z, np.isfinite(Z).all(axis=1)


def instrument_series(trace: BKSTrace, panel: BKSPanel, asset: str, topic: str) -> pd.DataFrame:
    """One instrument cell over every week, recomputed independently of the covariance recursion (step 4).

    For each instrument week ``j``: the normalised kernel over the days up to
    its window end with a return and a shock (every topic), and
    ``sum k r z - (sum k r)(sum k z)``, from a weeks x days kernel matrix.

    Returns
    -------
    DataFrame
        Index the instrument weeks (``week_ends``); columns ``kernel_cov``
        (``NaN`` with fewer than ``min_days`` days), ``panel`` (the value in
        ``panel.X`` at the row that uses this instrument week, ``NaN`` without
        a row), ``n_days``, ``return_week`` (the week it pairs with),
        ``role`` (``"burn-in"``, ``"training"``, ``"forecast"``, ``"other"``)
        and ``population`` (the constant population reference, panel units).
    """
    asset, topic = str(asset), str(topic)
    pnl = panel.panel
    days = pd.DatetimeIndex(panel.aligned.calendar)
    pid, ends, start, stop, cut = _week_structure(days, int(trace.meta["skip_days"]))
    r, Z, zok = _asset_arrays(trace, panel, asset)
    z = trace.z[topic].to_numpy(dtype=float)
    ok = np.isfinite(r) & zok
    r0, z0 = np.where(ok, r, 0.0), np.where(ok, z, 0.0)
    T, D = len(ends), len(days)
    dist = np.arange(T)[:, None] - pid[None, :]
    live = (dist >= 0) & (np.arange(D)[None, :] < cut[:, None]) & ok[None, :]
    Km = np.where(live, np.power(float(trace.meta["xi"]), np.clip(dist, 0, None).astype(float)), 0.0)
    tot = Km.sum(axis=1)
    n_days = live.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Kn = Km / np.where(tot > 0, tot, 1.0)[:, None]
        cov = Kn @ (r0 * z0) - (Kn @ r0) * (Kn @ z0)
    cov[(n_days < int(trace.meta["min_days"])) | (tot <= 0)] = np.nan
    panel_val = np.full(T, np.nan)
    a_i = [str(a) for a in pnl.assets].index(asset)
    t_col = 1 + [str(t) for t in pnl.topics].index(topic)
    rws = np.flatnonzero(pnl.asset_idx == a_i)
    tw = ends.get_indexer(pd.DatetimeIndex(pnl.periods)[pnl.t_idx[rws]])
    panel_val[tw - 1] = pnl.X[rws, t_col]
    ret_week = pd.DatetimeIndex(list(ends[1:]) + [pd.NaT])
    burn = int(trace.meta["burn_in_weeks"])
    role = np.array([
        "burn-in" if j < burn else _role_of_week(ret_week[j], trace) for j in range(T)
    ], dtype=object)
    return pd.DataFrame(
        {"kernel_cov": cov, "panel": panel_val, "n_days": n_days.astype(np.int64), "return_week": ret_week,
         "role": role, "population": float(trace.population["instrument_ref"].loc[asset, topic])},
        index=pd.DatetimeIndex(ends, name="instrument_week"),
    )


def _instrument_week_of(trace: BKSTrace, days: pd.DatetimeIndex, return_week: Any) -> tuple[int, np.ndarray,
                                                                                            pd.DatetimeIndex,
                                                                                            np.ndarray, np.ndarray]:
    pid, ends, start, stop, cut = _week_structure(days, int(trace.meta["skip_days"]))
    t = ends.get_indexer([pd.Timestamp(return_week)])[0]
    if t < 1:
        raise ValueError(f"{_fmt_day(return_week)} is not a return week with an instrument week before it")
    return int(t) - 1, pid, ends, stop, cut


def kernel_profile(trace: BKSTrace, panel: BKSPanel, asset: str, topic: str,
                   return_week: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """How one instrument cell is built: the kernel weight and each day's contribution (step 4).

    Returns
    -------
    (DataFrame, dict)
        Daily over the panel's days: ``weight`` (normalised kernel weight
        ``k_tau`` of the instrument paired with ``return_week``, 0 where the
        day is not used), ``contribution`` (``k_tau (r - rbar)(z - zbar)``)
        and ``cumulative`` (its running sum). The dict holds ``value`` (the
        sum of the contributions), ``panel_value`` (the panel's value, ``NaN``
        without a row), ``difference``, ``share_before_train`` (weight on
        days before the training start), ``effective_days``
        (``1 / sum k^2``), ``window_end``, ``instrument_week``,
        ``return_week`` and ``n_days``.
    """
    asset, topic = str(asset), str(topic)
    days = pd.DatetimeIndex(panel.aligned.calendar)
    j, pid, ends, stop, cut = _instrument_week_of(trace, days, return_week)
    wts = _instrument_weights(pid, j, float(trace.meta["xi"]), stop, cut)
    r, Z, zok = _asset_arrays(trace, panel, asset)
    z = trace.z[topic].to_numpy(dtype=float)
    ok = np.isfinite(r) & zok & (wts > 0.0)
    k = np.where(ok, wts, 0.0)
    k = k / k.sum() if k.sum() > 0 else k
    rbar = float(k[ok] @ r[ok]) if ok.any() else np.nan
    zbar = float(k[ok] @ z[ok]) if ok.any() else np.nan
    contrib = np.where(ok, k * (np.where(ok, r, 0.0) - rbar) * (np.where(ok, z, 0.0) - zbar), 0.0)
    frame = pd.DataFrame({"weight": k, "contribution": contrib, "cumulative": np.cumsum(contrib)},
                         index=pd.DatetimeIndex(days, name="date"))
    pnl = panel.panel
    periods = pd.DatetimeIndex(pnl.periods)
    panel_value = np.nan
    if pd.Timestamp(return_week) in periods:
        a_i = [str(a) for a in pnl.assets].index(asset)
        rws = np.flatnonzero((pnl.t_idx == periods.get_loc(pd.Timestamp(return_week))) & (pnl.asset_idx == a_i))
        if rws.size:
            panel_value = float(pnl.X[rws[0], 1 + [str(t) for t in pnl.topics].index(topic)])
    value = float(contrib.sum()) if ok.sum() >= int(trace.meta["min_days"]) else float("nan")
    stats = {
        "value": value,
        "panel_value": panel_value,
        "difference": value - panel_value,
        "share_before_train": float(k[np.asarray(days < trace.meta["train_start"])].sum()),
        "effective_days": float(1.0 / np.sum(k**2)) if ok.any() else float("nan"),
        "window_end": pd.Timestamp(days[cut[j] - 1]) if cut[j] > 0 else pd.NaT,
        "instrument_week": pd.Timestamp(ends[j]),
        "return_week": pd.Timestamp(return_week),
        "n_days": int(ok.sum()),
    }
    return frame, stats


def instrument_row(trace: BKSTrace, panel: BKSPanel, sim: SimData, asset: str, return_week: Any) -> pd.DataFrame:
    """One asset's instrument row of one return week, recomputed, against the population, split into signal and noise.

    Returns
    -------
    DataFrame
        Per topic (simulation order): ``panel`` (the ``panel.X`` value,
        ``NaN`` without a row), ``brute_force`` (the kernel covariance by
        direct summation), ``difference``, ``population`` (the reference,
        panel units), ``signal_part`` and ``noise_part`` (the same kernel
        covariance with the shock's signal part, and with its news and slow
        parts; they sum to ``brute_force`` when no attention hit its floor).
    """
    asset = str(asset)
    topics = trace.topics
    days = pd.DatetimeIndex(panel.aligned.calendar)
    j, pid, ends, stop, cut = _instrument_week_of(trace, days, return_week)
    wts = _instrument_weights(pid, j, float(trace.meta["xi"]), stop, cut)
    r, _, zok = _asset_arrays(trace, panel, asset)
    Zs = trace.z.reindex(columns=topics).to_numpy(dtype=float)
    ok = np.isfinite(r) & zok & (wts > 0.0)
    brute = _weighted_cov(r, Zs, wts, ok)
    lead = int(trace.lead_days)
    z_sig, z_news, z_slow = _z_components(sim, trace.shock_window)

    def paired(f: pd.DataFrame) -> np.ndarray:
        return f.shift(lead).reindex(index=days, columns=topics).to_numpy(dtype=float)

    sig = paired(z_sig)
    noise = paired(z_news) + paired(z_slow)
    ok_parts = ok & np.isfinite(sig).all(axis=1) & np.isfinite(noise).all(axis=1)
    pnl = panel.panel
    periods = pd.DatetimeIndex(pnl.periods)
    panel_row = np.full(len(topics), np.nan)
    if pd.Timestamp(return_week) in periods:
        a_i = [str(a) for a in pnl.assets].index(asset)
        rws = np.flatnonzero((pnl.t_idx == periods.get_loc(pd.Timestamp(return_week))) & (pnl.asset_idx == a_i))
        if rws.size:
            col = {str(t): c for c, t in enumerate(pnl.topics)}
            panel_row = pnl.X[rws[0], 1:][[col[t] for t in topics]]
    if ok.sum() < int(trace.meta["min_days"]):
        brute = np.full(len(topics), np.nan)
    return pd.DataFrame(
        {"panel": panel_row, "brute_force": brute, "difference": brute - panel_row,
         "population": trace.population["instrument_ref"].loc[asset].reindex(topics).to_numpy(dtype=float),
         "signal_part": _weighted_cov(r, sig, wts, ok_parts), "noise_part": _weighted_cov(r, noise, wts, ok_parts)},
        index=pd.Index(topics, name="topic_id"),
    )


def pairing_table(trace: BKSTrace, panel: BKSPanel, sim: SimData, asset: str) -> pd.DataFrame:
    """Which instrument week each return week pairs with, and the week's return three ways (step 5).

    One row per return week from four weeks before the first training week
    to the last forecast week.

    Returns
    -------
    DataFrame
        Columns ``return_week``, ``first_day``, ``instrument_week``,
        ``window_end`` (last day of the instrument's kernel window), ``role``
        (``"training"``, ``"forecast"``, ``"other"`` or ``"not in panel"``),
        ``y_panel`` (the panel's return, ``NaN`` without a row),
        ``y_recomputed`` (sum of the scaled daily returns over the week's
        days), ``raw_return`` (sum of the simulated raw returns over the same
        days) and ``n_assets`` (the week's rows in the panel).
    """
    asset = str(asset)
    pnl = panel.panel
    days = pd.DatetimeIndex(panel.aligned.calendar)
    pid, ends, start, stop, cut = _week_structure(days, int(trace.meta["skip_days"]))
    t0 = int(ends.get_loc(trace.train_periods.min()))
    t1 = int(ends.get_loc(trace.forecast_periods.max())) if len(trace.forecast_periods) else int(
        ends.get_loc(trace.train_periods.max()))
    periods = pd.DatetimeIndex(pnl.periods)
    counts = np.bincount(pnl.t_idx, minlength=len(periods))
    a_i = [str(a) for a in pnl.assets].index(asset)
    r = panel.aligned.returns[asset].to_numpy(dtype=float)
    ret = sim.market.returns
    raw = ret[ret.columns[[str(c) for c in ret.columns].index(asset)]].reindex(days).to_numpy(dtype=float)
    rows = []
    for t in range(max(t0 - _PAIRING_WEEKS_BEFORE, 1), t1 + 1):
        wk = pd.Timestamp(ends[t])
        dd = slice(start[t], stop[t])
        y_panel = np.nan
        n_assets = 0
        if wk in periods:
            tp = int(periods.get_loc(wk))
            n_assets = int(counts[tp])
            rws = np.flatnonzero((pnl.t_idx == tp) & (pnl.asset_idx == a_i))
            if rws.size:
                y_panel = float(pnl.y[rws[0]])
            role = _role_of_week(wk, trace)
        else:
            role = "not in panel"
        rr, rw = r[dd], raw[dd]
        rows.append({
            "return_week": wk, "first_day": pd.Timestamp(days[start[t]]), "instrument_week": pd.Timestamp(ends[t - 1]),
            "window_end": pd.Timestamp(days[cut[t - 1] - 1]) if cut[t - 1] > 0 else pd.NaT, "role": role,
            "y_panel": y_panel,
            "y_recomputed": float(np.nansum(rr)) if np.isfinite(rr).any() else np.nan,
            "raw_return": float(np.nansum(rw)) if np.isfinite(rw).any() else np.nan,
            "n_assets": n_assets,
        })
    return pd.DataFrame(rows, columns=["return_week", "first_day", "instrument_week", "window_end", "role", "y_panel",
                                       "y_recomputed", "raw_return", "n_assets"])


def design_matrix(trace: BKSTrace, panel: BKSPanel, fit: BKSFit, return_week: Any) -> pd.DataFrame:
    """The standardised design of one return week: instruments divided by the training ``sigma^c`` (step 5).

    Returns
    -------
    DataFrame
        Assets (the week's rows, panel order) x instruments (``const`` and
        the panel topics): ``X / sigma^c`` with the training ``sigma^c`` of
        the fit (``fit.meta["sigma_c"]``); empty when the week is not in the
        panel.
    """
    pnl = panel.panel
    names = [str(n) for n in pnl.instrument_names]
    periods = pd.DatetimeIndex(pnl.periods)
    wk = pd.Timestamp(return_week)
    cols = pd.Index(names, name="instrument")
    if wk not in periods:
        return pd.DataFrame(columns=cols, index=pd.Index([], name="asset_id"), dtype=float)
    rws = np.flatnonzero(pnl.t_idx == periods.get_loc(wk))
    sigma_c = np.asarray(fit.meta.get("sigma_c", pnl.sigma_c), dtype=float)
    ids = [str(pnl.assets[a]) for a in pnl.asset_idx[rws]]
    return pd.DataFrame(pnl.X[rws] / sigma_c[None, :], index=pd.Index(ids, name="asset_id"), columns=cols)


def forecast_week(trace: BKSTrace, panel: BKSPanel, fit: BKSFit, result: BKSLabResult,
                  week: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One forecast week: realised against fitted per asset, and the factor re-solved (step 7).

    Returns
    -------
    (DataFrame, dict)
        Per asset of the week (panel order): ``realized`` (panel units),
        ``fitted``, ``residual``, ``beta_1`` .. ``beta_K`` (``c Gamma``). The
        dict holds ``factors`` (the evaluation's, ``K``), ``factors_closed_form``,
        ``foc_max`` (``max |B'(y - Bf) - ridge f|``), ``ridge``, ``r2_week`` and
        ``r2_week_shuffled``.

    Raises
    ------
    ValueError
        When ``week`` is not an evaluated forecast week.
    """
    wk = pd.Timestamp(week)
    ends = pd.DatetimeIndex(result.periods)
    if wk not in ends:
        raise ValueError(f"{_fmt_day(wk)} is not an evaluated forecast week")
    pnl = panel.panel
    Gamma = np.asarray(fit.fit.Gamma, dtype=float)
    K = int(Gamma.shape[1])
    ridge = float(result.meta.get("oos_ridge", 0.0 if float(fit.lam) == 0.0 else RIDGE))
    periods = pd.DatetimeIndex(pnl.periods)
    rws = np.flatnonzero(pnl.t_idx == periods.get_loc(wk))
    C, y = pnl.X[rws], pnl.y[rws]
    B = C @ Gamma
    f = result.meta["factors"].loc[wk].to_numpy(dtype=float)
    if ridge > 0.0:
        f_cf = np.linalg.solve(B.T @ B + ridge * np.eye(K), B.T @ y)
    else:
        f_cf = np.linalg.pinv(B.T @ B, rcond=1e-12, hermitian=True) @ (B.T @ y)
    fitted = B @ f
    frame = pd.DataFrame({"realized": y, "fitted": fitted, "residual": y - fitted},
                         index=pd.Index([str(pnl.assets[a]) for a in pnl.asset_idx[rws]], name="asset_id"))
    for k in range(K):
        frame[f"beta_{k + 1}"] = B[:, k]
    syy = float(np.sum(y**2))
    row = trace.weeks.loc[wk] if wk in trace.weeks.index else None
    stats = {
        "factors": f,
        "factors_closed_form": f_cf,
        "foc_max": float(np.max(np.abs(B.T @ (y - fitted) - ridge * f))) if len(y) else float("nan"),
        "ridge": ridge,
        "r2_week": 1.0 - float(np.sum((y - fitted) ** 2)) / syy if syy > 0 else float("nan"),
        "r2_week_shuffled": float(row["r2_shuffled"]) if row is not None else float("nan"),
    }
    return frame, stats


def asset_chain(trace: BKSTrace, asset: str, direct: DirectFit | None = None) -> pd.DataFrame:
    """One asset's Eq. 5 chain per topic, next to the references (step 8).

    Returns
    -------
    DataFrame
        Per topic (simulation order): ``instrument`` (``c~_i``, panel units),
        ``projected`` (``P c~_i``), ``constant_part`` (``M Gamma_0'``), ``m``
        (their sum), ``m_ret`` (times the unit conversion, return units),
        ``b_raw`` (``Sigma_z^+ m_ret``), ``B_hat``, ``B_true``,
        ``B_true_train_units``, ``window_truth``, ``instruments_train`` and
        ``instruments_kernel`` (ladder variants) and ``direct`` (the given
        direct fit's ``B_hat``, ``NaN`` without one); sensitivities in
        standardised units.
    """
    a = str(asset)
    ch = trace.chain
    topics = trace.topics
    direct_col = (direct.B_hat.reindex(index=topics)[a].to_numpy(dtype=float)
                  if direct is not None and a in direct.B_hat.columns else np.full(len(topics), np.nan))
    return pd.DataFrame(
        {"instrument": trace.instruments.loc[a].to_numpy(dtype=float),
         "projected": ch["m_proj"].loc[a].to_numpy(dtype=float),
         "constant_part": ch["m_const"].to_numpy(dtype=float),
         "m": ch["m"].loc[a].to_numpy(dtype=float),
         "m_ret": ch["m_ret"].loc[a].to_numpy(dtype=float),
         "b_raw": ch["b_raw"].loc[a].to_numpy(dtype=float),
         "B_hat": ch["B_hat"][a].to_numpy(dtype=float),
         "B_true": trace.variants["oracle"][a].to_numpy(dtype=float),
         "B_true_train_units": trace.B_true_train_units[a].to_numpy(dtype=float),
         "window_truth": trace.window_truth[a].to_numpy(dtype=float),
         "instruments_train": trace.variants["instruments_train"][a].to_numpy(dtype=float),
         "instruments_kernel": trace.variants["instruments_kernel"][a].to_numpy(dtype=float),
         "direct": direct_col},
        index=pd.Index(topics, name="topic_id"),
    )
