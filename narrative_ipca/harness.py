"""Simulation harness: compare pipeline runs with the simulation truth (DESIGN.md Part E; D37-D40).

The harness closes the loop of the validation design: :mod:`narrative_ipca.simulation`
draws data whose structure is known (BKS Figure 1 with the attention-level
mapping of Part D), :mod:`narrative_ipca.pipeline` estimates the model without
seeing that structure, and :func:`compare_to_truth` measures how much of it
was recovered. Because ``Gamma``, ``f_t`` and ``x_tau`` are identified only
up to a rotation of the K-dimensional state space (D24, D38), every
comparison is rotation invariant:

* the *selected set* of narratives against the relevant / placebo masks
  (recall over all relevant topics and over the strong half by ``||A_l||``,
  precision, F1, placebo count);
* the *implied loadings* ``beta_{i,t} = c_{i,t-1} Gamma`` - the object IPCA
  identifies, invariant to the rotation and to the selected set - through
  the canonical correlations between ``X_t Gamma_hat`` and the true
  ``beta_t`` across the assets of sampled return periods
  (:func:`beta_canonical_corr`);
* the *column space* of ``Gamma_tilde`` on the rows that are relevant *and
  selected*, through the cosines of the principal angles (Björck & Golub
  1973) between ``col(Gamma_tilde_hat)`` and ``col(Gamma_tilde_true)``;
  non-selected rows of ``Gamma_hat`` are exactly zero, so a comparison over
  all relevant rows measures recall rather than loading recovery (that value
  is reported next to it as ``gamma_subspace_cos_all_relevant``); with
  fewer than ``2K`` such rows the cosines that are one by dimension
  counting are left out of the mean (see :func:`_gamma_subspace_cos`);
* the *span* of the factors and of the states, through canonical
  correlations (Hotelling 1936), computed here from orthonormal bases of the
  centred data matrices (the canonical correlations are the cosines of the
  principal angles between the two column spaces);
* the impact vector ``I_{z->MVE}`` (Eq. 11), itself rotation invariant, by
  Spearman correlation and sign agreement over the relevant topics that
  were selected (BKS report impact vectors for selected narratives only;
  the all-relevant value is reported as ``impact_spearman_all_relevant``);
* Sharpe ratios of the MVE portfolios in sample and out of sample, the latter
  relative to the realised OOS Sharpe of the *true* MVE portfolio;
* the share of the true systematic return ``beta_{i,t}' f_t`` that the fitted
  values ``c_{i,t-1} Gamma_hat f_hat_t`` reproduce.

Pass/fail flags come from :class:`~narrative_ipca.config.HarnessThresholds`
(D40: regression-test signals, not statistical tests). Which checks apply
depends on the scenario: the signal scenarios (``baseline``, ``softmax``,
``balanced``, ``weak``) get the recovery checks; ``no_factor`` (returns
without a common factor structure, the chance-level null) only the no-lift
checks; ``topic_null`` (alias ``null``: no topic carries information but
returns keep their priced factor structure) is *report-only* - no check
applies and ``all_passed`` is vacuously true. ``weak`` uses relaxed recall,
strong-recall and subspace thresholds (:data:`SCENARIO_THRESHOLD_OVERRIDES`).

Why ``topic_null`` is not a pass/fail null (verification finding of
2026-09-06): under that DGP the Eq. 6 kernel covariance of *any* noise topic
``l`` with asset ``i`` is ``beta_i' G_{t,l}`` plus idiosyncratic noise, where
``G_{t,l} = sum_tau w_tau f_tau z_{l,tau}`` is a common ``K``-vector that is
non-zero at order ``1/sqrt(n_eff)`` and persistent over ``t`` because the
kernel half-life is 69 months. With ``L`` noise topics the cross-section of
instruments spans ``beta``, IPCA recovers ``beta`` from pure noise topics,
the factor portfolios load on the true factors and earn the premium.
Selection above chance and a positive OOS Sharpe ratio are therefore the
*correct* behaviour of the estimator there, and the scenario is informative
in a different way: it shows that an OOS Sharpe ratio alone cannot certify
that narratives carry information. The harness quantifies the mechanism with
the diagnostic :func:`instrument_beta_r2` (cross-sectional R2 of every
topic's instrument on the true loadings, against the chance level ``K/N``).

The OOS selection stability (mean Jaccard similarity of the selected sets
of consecutive refits) is reported next to it but does not separate the two
cases: the selected set is expected to be *stable* under ``topic_null`` as
well as under ``baseline``, because the spurious instruments ``G_{t,l}``
persist with the 69-month kernel half-life and the same noise topics are
re-selected at every refit. Stability is low only under ``no_factor``, where
the instruments are pure noise. It is therefore a diagnostic of estimation
noise, not evidence that narratives carry information (D47, Part F point 2).
The signal-quality evidence is the relative placebo test (real narratives
must beat variance-matched placebos), the pricing errors, and the
``topic_null`` rows read next to the ``baseline`` rows.

:func:`run_harness` loops over scenarios x seeds, :func:`write_report` turns
the result into the markdown report of Part E. All functions are pure apart
from the artefact files they write under ``HarnessConfig.output_dir``.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _stats

from . import __version__
from .config import (
    CovarianceConfig,
    EstimationConfig,
    HarnessConfig,
    HarnessThresholds,
    LambdaGridConfig,
    OOSConfig,
    PipelineConfig,
    SimulationConfig,
    config_to_dict,
)
from .data import normalize_period_alias
from .evaluation import realized_sharpe
from .grouplasso import active_backend
from .pipeline import run_pipeline
from .simulation import scenario_config, simulate
from .types import CovariancePanel, HarnessMetrics, HarnessResult, IPCAPanel, PipelineResult, SimulationTruth, TuningResult

logger = logging.getLogger(__name__)

__all__ = [
    "compare_to_truth",
    "run_harness",
    "write_report",
    "canonical_correlations",
    "subspace_cosines",
    "evaluate_checks",
    "scenario_checks",
    "scenario_thresholds",
    "instrument_beta_r2",
    "beta_canonical_corr",
    "default_pipeline_config",
    "CHECKS",
    "SIGNAL_CHECKS",
    "NULL_CHECKS",
    "NULL_SCENARIOS",
    "REPORT_ONLY_SCENARIOS",
    "SCENARIO_ALIASES",
    "METRICS",
    "SCENARIO_THRESHOLD_OVERRIDES",
    "CHANCE_SELECTION_FRACTION",
    "FAST_LAMBDA_RATIO",
    "FAST_N_LAMBDAS",
    "NO_FACTOR_TEXT",
    "INSTRUMENT_R2_EVERY",
    "INSTRUMENT_R2_MIN_PERIODS",
]

ProgressFn = Callable[[int, int, str], None]
"""``progress(done, total, message)`` callback over harness runs (D44)."""

CHANCE_SELECTION_FRACTION: float = 0.05
"""Chance level of the null scenarios: ``round(0.05 L)`` (at least one) narratives selected by luck."""

INSTRUMENT_R2_EVERY: int = 12
"""Period step of the :func:`instrument_beta_r2` subsample (every 12th covariance period, counted back from the last)."""

INSTRUMENT_R2_MIN_PERIODS: int = 5
"""Minimum number of covariance periods in the :func:`instrument_beta_r2` subsample.

The same rule (:func:`_subsample_indices`) picks the return periods of the
:func:`beta_canonical_corr` subsample."""

SCENARIO_ALIASES: dict[str, str] = {"null": "topic_null"}
"""Accepted alternative scenario names (``null`` is the former name of ``topic_null``)."""

NULL_SCENARIOS: tuple[str, ...] = ("no_factor",)
"""Scenarios tested against the chance-level thresholds (:data:`NULL_CHECKS`)."""

REPORT_ONLY_SCENARIOS: tuple[str, ...] = ("topic_null",)
"""Scenarios with no pass/fail check: metrics are reported and discussed, ``all_passed`` is vacuously true."""

SIGNAL_SCENARIOS: tuple[str, ...] = ("baseline", "softmax", "balanced", "weak")
"""Scenarios with signal, tested with :data:`SIGNAL_CHECKS`."""

FAST_LAMBDA_RATIO: float = 1e-3
"""``LambdaGridConfig.ratio`` of the fast pipeline config: the package default.

The fast grid used to stop at ``0.05 lam_max`` to save time at the dense end
of the path. With the numba kernel (D46) the dense end is cheap, and the
truncated grid made the fast baseline unreadable (recall 0.29 over ten
seeds against 0.75-0.875 with the full ratio): on the small fast panels the
tuned ``lambda`` sits below ``0.05 lam_max``, so the grid never reached it.
"""

FAST_N_LAMBDAS: int = 12
"""``LambdaGridConfig.n_lambdas`` of the fast pipeline config (20 in full mode, 30 in the package default)."""

#: Relative singular-value cut-off used for the orthonormal bases of the comparisons.
_RCOND: float = 1e-10

METRICS: tuple[str, ...] = (
    "selection_recall",
    "selection_recall_strong",
    "selection_recall_weak",
    "selection_precision",
    "selection_f1",
    "n_selected",
    "beta_canonical_corr",
    "beta_canonical_corr_mean",
    "placebo_selected",
    "gamma_subspace_cos",
    "gamma_subspace_cos_all_relevant",
    "factor_canonical_corr",
    "factor_canonical_corr_mean",
    "state_canonical_corr",
    "impact_spearman",
    "impact_spearman_all_relevant",
    "impact_sign_agreement",
    "mve_sharpe_is",
    "sharpe_mve_true",
    "oos_sharpe",
    "oos_sharpe_true_mve",
    "oos_sharpe_ratio_to_true",
    "systematic_r2_recovered",
    "total_r2",
    "systematic_r2_true",
    "null_selection_lift",
    "null_oos_sharpe_abs",
    "instrument_beta_r2_relevant",
    "instrument_beta_r2_noise",
    "instrument_beta_r2_placebo",
    "instrument_beta_r2_chance",
    "oos_selection_stability",
    "runtime_seconds",
)
"""The Part E metrics plus the instrument-informativeness diagnostics, in report order
(``HarnessMetrics.values`` carries a few extras)."""

CHECKS: tuple[tuple[str, str, str, str], ...] = (
    # (check name, metric, comparison, HarnessThresholds field)
    ("selection_recall", "selection_recall", ">=", "selection_recall_min"),
    ("selection_recall_strong", "selection_recall_strong", ">=", "selection_recall_strong_min"),
    ("selection_precision", "selection_precision", ">=", "selection_precision_min"),
    ("beta_canonical_corr", "beta_canonical_corr", ">=", "beta_canonical_corr_min"),
    ("placebo_selected", "placebo_selected", "<=", "placebo_selected_max"),
    ("gamma_subspace_cos", "gamma_subspace_cos", ">=", "gamma_subspace_cos_min"),
    ("factor_canonical_corr", "factor_canonical_corr", ">=", "factor_canonical_corr_min"),
    ("state_canonical_corr", "state_canonical_corr", ">=", "state_canonical_corr_min"),
    ("impact_spearman", "impact_spearman", ">=", "impact_vector_spearman_min"),
    ("oos_sharpe_ratio_to_true", "oos_sharpe_ratio_to_true", ">=", "oos_sharpe_ratio_to_true_min"),
    ("systematic_r2_recovered", "systematic_r2_recovered", ">=", "systematic_r2_min"),
    ("null_selection_lift", "null_selection_lift", "<=", "null_selection_lift_max"),
    ("null_oos_sharpe_abs", "null_oos_sharpe_abs", "<=", "null_oos_sharpe_abs_max"),
)
"""Every pass/fail check: metric, direction and the threshold field (DESIGN.md Part E table)."""

_CHECK_TABLE: dict[str, tuple[str, str, str]] = {name: (metric, op, field) for name, metric, op, field in CHECKS}

SIGNAL_CHECKS: tuple[str, ...] = (
    "selection_recall",
    "selection_recall_strong",
    "selection_precision",
    "beta_canonical_corr",
    "placebo_selected",
    "gamma_subspace_cos",
    "factor_canonical_corr",
    "state_canonical_corr",
    "impact_spearman",
    "oos_sharpe_ratio_to_true",
    "systematic_r2_recovered",
)
"""Checks applied to the scenarios with signal (baseline, softmax, balanced, weak)."""

NULL_CHECKS: tuple[str, ...] = ("null_selection_lift", "null_oos_sharpe_abs", "placebo_selected")
"""Checks applied to the chance-level null ``no_factor`` (returns without a common factor structure:
the kernel covariances carry no information, so selection should sit at chance and the realised OOS
Sharpe within two standard errors of zero). They are *not* applied to ``topic_null``, where selection
above chance and a positive OOS Sharpe are the estimator's correct behaviour (module docstring)."""

SCENARIO_THRESHOLD_OVERRIDES: dict[str, dict[str, float]] = {
    "weak": {"selection_recall_min": 0.3, "selection_recall_strong_min": 0.6, "gamma_subspace_cos_min": 0.7},
}
"""Per-scenario threshold relaxations (``weak`` degrades gracefully: recall falls, precision stays)."""

EXPECTED: dict[str, str] = {
    "selection_recall": (
        "recall well below 1: the relevant rows of A are standard-normal draws, so a share of the relevant "
        "topics are weak and legitimately not selected"
    ),
    "selection_recall_strong": "recall over the relevant topics with ||A_l|| at or above the median near 1",
    "selection_precision": "precision high (irrelevant persistent topics may enter at small lambda)",
    "beta_canonical_corr": "first canonical correlation of implied vs true betas > 0.95",
    "placebo_selected": "no placebo topic selected (BKS App. C.2)",
    "gamma_subspace_cos": (
        "col(Gamma_tilde) recovered on the relevant rows that were selected (mean principal-angle cosine > 0.9)"
    ),
    "factor_canonical_corr": "first canonical correlation between F_hat and the true period factors > 0.95",
    "state_canonical_corr": "first canonical correlation between x_hat and the true daily states > 0.85",
    "impact_spearman": "Spearman correlation of I_{z->MVE} hat vs true over the selected relevant topics > 0.8",
    "oos_sharpe_ratio_to_true": "realised OOS MVE Sharpe about 0.5-0.9 of the true MVE's realised OOS Sharpe",
    "systematic_r2_recovered": "R2 of the true systematic return on the fitted values > 0.7",
    "null_selection_lift": "no_factor: selection at chance level, no lift",
    "null_oos_sharpe_abs": "no_factor: realised OOS Sharpe within two standard errors of zero",
    "instrument_beta_r2_relevant": "relevant topics' instruments are linear in beta (Eq. 5): R2 well above chance K/N",
    "instrument_beta_r2_noise": (
        "topic_null: noise topics' instruments span beta through the persistent common vector G_{t,l}, "
        "R2 well above chance K/N; no_factor: beta is zero, R2 zero by construction"
    ),
    "oos_selection_stability": (
        "stable (high Jaccard) under baseline and topic_null alike - the spurious instruments G_{t,l} persist "
        "with the 69-month kernel half-life -, low only under no_factor: a diagnostic of estimation noise, "
        "not evidence that narratives carry information"
    ),
}
"""Expected outcome per check / diagnostic (DESIGN.md Part E, 'Expected (baseline)' column)."""


# ---------------------------------------------------------------------------
# rotation-invariant numerics
# ---------------------------------------------------------------------------
def _orthonormal_basis(M: np.ndarray, rcond: float = _RCOND) -> np.ndarray:
    """Orthonormal basis of ``col(M)`` from a thin SVD; singular values ``<= rcond * s_max`` are dropped.

    Returns an ``(n, rank)`` matrix with orthonormal columns (``(n, 0)`` for
    an all-zero or empty ``M``). The SVD, rather than a QR factorisation, is
    used so that a rank-deficient ``M`` (for instance ``Gamma_tilde`` with
    fewer than ``K`` non-zero rows) gives a basis of the right dimension
    instead of numerically tiny directions.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {M.shape}")
    if M.size == 0 or not np.all(np.isfinite(M)) or not np.any(M):
        return np.zeros((M.shape[0], 0))
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    keep = s > rcond * s[0]
    return U[:, keep]


def subspace_cosines(A: np.ndarray, B: np.ndarray, rcond: float = _RCOND) -> np.ndarray:
    """Cosines of the principal angles between ``col(A)`` and ``col(B)`` (Björck & Golub 1973).

    With ``Q_A``, ``Q_B`` orthonormal bases of the two column spaces, the
    singular values of ``Q_A' Q_B`` are ``cos(theta_1) >= ... >= cos(theta_m)``,
    ``m = min(rank A, rank B)``; the first is the largest correlation between
    a direction of ``A`` and a direction of ``B``, the last the smallest such
    correlation after removing the previous pairs. All ones means the smaller
    space is contained in the larger. Invariant to ``A -> A R`` and
    ``B -> B S`` for any invertible ``R``, ``S`` (D38). Returns an empty array
    when either matrix is zero.
    """
    Qa = _orthonormal_basis(A, rcond)
    Qb = _orthonormal_basis(B, rcond)
    if Qa.shape[0] != Qb.shape[0]:
        raise ValueError(f"A and B must have the same number of rows, got {Qa.shape[0]} and {Qb.shape[0]}")
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return np.zeros(0)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def canonical_correlations(X: np.ndarray, Y: np.ndarray, rcond: float = _RCOND) -> np.ndarray:
    """Canonical correlations between the column sets ``X`` (n, p) and ``Y`` (n, q), descending.

    Hotelling's canonical correlations ``rho_1 >= ... >= rho_m`` are the
    maxima of ``corr(X a, Y b)`` over successive orthogonal pairs ``(a, b)``.
    Equivalently they are the cosines of the principal angles between the
    column spaces of the *centred* data matrices ``X - mean`` and
    ``Y - mean`` (Björck & Golub 1973), which is what this function computes
    through :func:`subspace_cosines` (one thin SVD per side, no covariance
    inversion). ``m = min(rank X, rank Y)`` after the ``rcond`` cut-off, so a
    rank-deficient side (``F`` from a ``Gamma`` with fewer than ``K``
    non-zero rows) yields fewer correlations rather than spurious ones.
    Invariant to invertible linear transforms of either side (D38).

    Rows are observations; the caller aligns them. Returns an empty array
    with fewer than two rows.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of rows, got {X.shape[0]} and {Y.shape[0]}")
    if X.shape[0] < 2:
        return np.zeros(0)
    return subspace_cosines(X - X.mean(axis=0), Y - Y.mean(axis=0), rcond)


# ---------------------------------------------------------------------------
# alignment helpers
# ---------------------------------------------------------------------------
def _period_freq(alias: str | None) -> str | None:
    """A pandas *period* frequency equivalent to the offset alias (``"ME"`` -> ``"M"``), or ``None``."""
    if not alias:
        return None
    probe = pd.DatetimeIndex(["2000-01-03"])
    for cand in normalize_period_alias(str(alias)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pd.PeriodIndex(probe, freq=cand)
                return cand
            except (ValueError, TypeError):
                continue
    return None


def _match_positions(source: pd.Index, target: pd.Index, period: str | None) -> np.ndarray:
    """Position in ``target`` of every label of ``source`` (``-1`` when absent).

    Timestamps are matched exactly first; when not every source label is
    found and ``period`` is given, both indexes are converted to period
    labels (so a month-end stamp pairs with a last-trading-day stamp, as in
    :func:`narrative_ipca.pipeline.align_period_frame`) and the label match
    is used when it is at least as complete and both label sets are unique.
    """
    src = pd.DatetimeIndex(source)
    tgt = pd.DatetimeIndex(target)
    pos = np.asarray(tgt.get_indexer(src), dtype=np.int64)
    if len(src) and len(tgt) and not np.all(pos >= 0):
        freq = _period_freq(period)
        if freq is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ps = pd.PeriodIndex(src, freq=freq)
                pt = pd.PeriodIndex(tgt, freq=freq)
            if not pt.has_duplicates and not ps.has_duplicates:
                by_label = np.asarray(pt.get_indexer(ps), dtype=np.int64)
                if np.count_nonzero(by_label >= 0) >= np.count_nonzero(pos >= 0):
                    pos = by_label
    return pos


def _align_frames(a: pd.DataFrame, b: pd.DataFrame, period: str | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Rows of ``a`` and ``b`` on their common dates with finite values: ``(A, B, n_common)``."""
    pos = _match_positions(a.index, b.index, period)
    keep = pos >= 0
    A = a.to_numpy(dtype=float)[keep]
    B = b.to_numpy(dtype=float)[pos[keep]]
    finite = np.all(np.isfinite(A), axis=1) & np.all(np.isfinite(B), axis=1)
    return A[finite], B[finite], int(finite.sum())


def _canonical_scenario(name: str) -> str:
    """``"Baseline-fast"`` -> ``"baseline"``; ``"fast"`` -> ``"baseline"``; ``"null"`` -> ``"topic_null"``."""
    key = str(name).strip().lower().replace("-", "_")
    if key == "fast":
        return "baseline"
    if key.endswith("_fast"):
        key = key[: -len("_fast")]
    return SCENARIO_ALIASES.get(key, key)


def _annualization(result: PipelineResult, truth: SimulationTruth) -> float:
    cfg = result.config
    if isinstance(cfg, PipelineConfig):
        return float(cfg.evaluation.annualization)
    if isinstance(cfg, dict):
        try:
            return float(cfg["evaluation"]["annualization"])
        except (KeyError, TypeError, ValueError):
            pass
    ann = truth.meta.get("periods_per_year") if isinstance(truth.meta, dict) else None
    return float(ann) if ann else 12.0


def _rcond(result: PipelineResult) -> float:
    cfg = result.config
    if isinstance(cfg, PipelineConfig):
        return float(cfg.evaluation.rcond)
    if isinstance(cfg, dict):
        try:
            return float(cfg["evaluation"]["rcond"])
        except (KeyError, TypeError, ValueError):
            pass
    return 1e-12


def _period_alias(result: PipelineResult) -> str | None:
    meta = result.fit.meta if isinstance(result.fit.meta, dict) else {}
    if meta.get("period"):
        return str(meta["period"])
    cfg = result.config
    if isinstance(cfg, PipelineConfig):
        return str(cfg.data.period)
    if isinstance(cfg, dict):
        try:
            return str(cfg["data"]["period"])
        except (KeyError, TypeError):
            return None
    return None


def _truth_topics(truth: SimulationTruth) -> list[str]:
    return [str(c) for c in truth.z_daily.columns]


def _truth_topic_positions(names: Sequence[Any], truth: SimulationTruth) -> np.ndarray:
    """Position in ``truth.z_daily.columns`` of every topic name (``-1`` when absent)."""
    lookup = {t: i for i, t in enumerate(_truth_topics(truth))}
    return np.asarray([lookup.get(str(n), -1) for n in names], dtype=np.int64)


def _truth_asset_positions(
    assets: Sequence[Any], truth: SimulationTruth, asset_ids: Sequence[str] | None, what: str
) -> np.ndarray | None:
    """Position in ``truth.beta``'s asset axis of every entry of ``assets`` (``-1`` when absent).

    The simulated asset ids come from ``asset_ids`` (in truth order), else
    from ``truth.meta["asset_ids"]``; without ids the assets are matched
    positionally when the counts agree and ``None`` is returned (with a
    warning naming ``what``) otherwise.
    """
    ids: list[str] | None = None
    if asset_ids is not None:
        ids = [str(a) for a in asset_ids]
    elif isinstance(truth.meta, dict) and truth.meta.get("asset_ids") is not None:
        ids = [str(a) for a in truth.meta["asset_ids"]]
    N_true = int(np.asarray(truth.beta).shape[1])
    if ids is None:
        if len(assets) != N_true:
            logger.warning("%s: no asset ids and %d assets vs %d in the truth; skipped", what, len(assets), N_true)
            return None
        logger.debug("%s: assets matched positionally (%d)", what, len(assets))
        return np.arange(len(assets), dtype=np.int64)
    if len(ids) != N_true:
        raise ValueError(f"asset_ids has {len(ids)} entries but truth.beta has {N_true} assets")
    lookup = {a: i for i, a in enumerate(ids)}
    return np.asarray([lookup.get(str(a), -1) for a in assets], dtype=np.int64)


def _populated_mask(result: PipelineResult) -> np.ndarray:
    fit = result.fit
    if fit.populated is not None:
        return np.asarray(fit.populated, dtype=bool).ravel()
    return np.any(np.asarray(fit.F) != 0.0, axis=1)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def scenario_checks(scenario: str) -> tuple[str, ...]:
    """Checks that apply to ``scenario``.

    :data:`NULL_CHECKS` for ``no_factor``; an empty tuple for ``topic_null``
    (alias ``null``; report-only, see the module docstring);
    :data:`SIGNAL_CHECKS` for ``baseline``, ``softmax``, ``balanced``,
    ``weak``. Unknown scenario names (a custom base) get the signal checks
    with a log message; ``-fast`` / ``_fast`` suffixes are ignored.
    """
    key = _canonical_scenario(scenario)
    if key in NULL_SCENARIOS:
        return NULL_CHECKS
    if key in REPORT_ONLY_SCENARIOS:
        return ()
    if key not in SIGNAL_SCENARIOS:
        logger.info("scenario %r is not a named scenario; applying the signal checks", scenario)
    return SIGNAL_CHECKS


def scenario_thresholds(thresholds: HarnessThresholds, scenario: str) -> HarnessThresholds:
    """Thresholds with the per-scenario relaxations of :data:`SCENARIO_THRESHOLD_OVERRIDES` applied."""
    overrides = SCENARIO_THRESHOLD_OVERRIDES.get(_canonical_scenario(scenario))
    return replace(thresholds, **overrides) if overrides else thresholds


def evaluate_checks(values: Mapping[str, float], thresholds: HarnessThresholds, scenario: str) -> dict[str, bool]:
    """Pass/fail flag of every check that applies to ``scenario`` (D40).

    A check passes when ``values[metric] >= threshold`` (or ``<=`` for the
    ``max`` thresholds); a missing or non-finite metric fails. The returned
    dict holds exactly the checks of :func:`scenario_checks`.
    """
    thr = scenario_thresholds(thresholds, scenario)
    out: dict[str, bool] = {}
    for check in scenario_checks(scenario):
        metric, op, field = _CHECK_TABLE[check]
        raw = values.get(metric, float("nan"))
        try:
            v = float(raw)
        except (TypeError, ValueError):
            v = float("nan")
        bound = float(getattr(thr, field))
        if not np.isfinite(v):
            out[check] = False
        elif op == ">=":
            out[check] = bool(v >= bound)
        else:
            out[check] = bool(v <= bound)
    return out


def _threshold_text(thresholds: HarnessThresholds, check: str) -> str:
    metric, op, field = _CHECK_TABLE[check]
    bound = getattr(thresholds, field)
    text = f"{int(bound)}" if float(bound).is_integer() and "placebo" in check else f"{float(bound):.2f}"
    return f"{op} {text}"


# ---------------------------------------------------------------------------
# compare_to_truth
# ---------------------------------------------------------------------------
def compare_to_truth(
    result: PipelineResult,
    truth: SimulationTruth,
    thresholds: HarnessThresholds,
    scenario: str,
    asset_ids: Sequence[str] | None = None,
) -> HarnessMetrics:
    """Rotation-invariant comparison of one pipeline run with the simulation truth (DESIGN.md Part E).

    ``values`` (all floats; ``nan`` where undefined):

    * ``selection_recall`` = |selected ∩ relevant| / |relevant|,
      ``selection_precision`` = |selected ∩ relevant| / |selected|,
      ``selection_f1``, ``n_selected``, ``placebo_selected`` (selected topics
      whose ``truth.placebo`` flag is set). Topics are matched by name
      between ``fit.instrument_names[1:]`` and ``truth.z_daily.columns``.
    * ``selection_recall_strong`` / ``selection_recall_weak``: recall over
      the relevant topics whose row norm ``||A_l||`` (``truth.A``) is at or
      above the median of the relevant rows' norms, and over the others;
      ``nan`` with fewer than two relevant topics (or an empty half).
    * ``beta_canonical_corr`` / ``beta_canonical_corr_mean``: the first and
      the mean canonical correlation between the implied loadings
      ``X_t Gamma_hat`` and the true ``beta_t`` across the assets of a
      return period, averaged over the sampled return periods of
      :func:`beta_canonical_corr` (every 12th, at least 5). This is the
      object IPCA identifies (``beta_{i,t} = c_{i,t-1} Gamma``), invariant
      to the rotation and to the selected set.
    * ``gamma_subspace_cos``: mean cosine of the ``K`` principal angles
      between ``col(Gamma_tilde_hat[rows])`` and
      ``col(Gamma_tilde_true[rows])`` (:func:`subspace_cosines`) over the
      ``n`` rows that are relevant *and selected*. With ``n < 2K`` rows the
      ``2K - n`` largest cosines are one by dimension counting (two
      ``K``-dimensional subspaces of an ``n``-space intersect in at least
      ``2K - n`` dimensions; with ``n = K`` any full-rank block scores one),
      so the mean runs over the ``min(K, n - K)`` smallest cosines
      (``gamma_informative_cosines``; the plain mean once ``n >= 2K``);
      ``nan`` when at most ``K`` relevant rows are selected or the selected
      block has rank below ``K`` (the subspace is then not estimated).
      Non-selected rows of ``Gamma_hat`` are exactly zero, so the same
      comparison over *all* relevant rows
      (``gamma_subspace_cos_all_relevant``, reported only) measures recall
      rather than loading recovery.
    * ``factor_canonical_corr`` / ``factor_canonical_corr_mean``: first and
      mean canonical correlation (:func:`canonical_correlations`) between
      ``fit.F`` over populated periods and ``truth.f_period`` on common
      dates; the mean is over ``K`` with missing pairs (rank deficiency)
      counted as zero.
    * ``state_canonical_corr`` (and ``_mean``): the same between
      ``wrapup.states`` and ``truth.x_daily`` on common finite days.
    * ``impact_spearman``: Spearman correlation of ``wrapup.impact_z_to_mve``
      and ``truth.impact_z_to_mve_true`` over the relevant topics that were
      selected (both are rotation invariant and agree up to a positive
      scale; BKS report impact vectors for selected narratives only), ``nan``
      with fewer than three such topics; ``impact_sign_agreement``: share of
      those topics with equal signs. ``impact_spearman_all_relevant`` and
      ``impact_sign_agreement_all_relevant`` are the same over all relevant
      topics (reported only).
    * ``mve_sharpe_is`` (fit), ``sharpe_mve_true`` (population),
      ``oos_sharpe`` (``result.oos.sharpe``), ``oos_sharpe_true_mve``: realised
      Sharpe over the OOS periods of the true MVE portfolio
      ``b_true' f_period`` with ``b_true = Sigma_ff_period^-1 mu_f_period``;
      ``oos_sharpe_ratio_to_true = oos_sharpe / oos_sharpe_true_mve``, ``nan``
      when the denominator is not finite or not positive (a negative
      realised true Sharpe makes the ratio meaningless).
    * ``systematic_r2_recovered``: ``1 - sum (s - s_hat)^2 / sum s^2`` over
      the panel rows with ``s_{i,t} = beta_{i,t}' f_t`` (true systematic
      period return) and ``s_hat = c_{i,t-1} Gamma_hat f_hat_t`` (in-sample
      fitted value); rows are mapped to ``truth.beta`` by asset id
      (``asset_ids`` = the simulated asset ids in truth order, else
      ``truth.meta["asset_ids"]``, else positionally when the counts agree)
      and by period date. ``systematic_r2_recovered_narrative`` is the same
      with the constant row of ``Gamma`` zeroed (narrative part only; the
      discriminating variant against the null, see
      :func:`_systematic_r2_recovered`).
    * ``total_r2``, ``systematic_r2_true`` (population, Part D step 7).
    * ``null_selection_lift`` = ``n_selected / max(1, round(0.05 L))``: the
      number of selected narratives relative to a scenario-independent chance
      level of 5 % of the ``L`` candidates (:data:`CHANCE_SELECTION_FRACTION`);
      1.0 means selection at chance, ``<= 2`` passes. The literal Part E
      ratio ``(selected / L) / (relevant / L)`` with the *designated*
      relevant count of the truth is reported as
      ``null_selection_lift_relevant``.
    * ``null_oos_sharpe_abs`` = ``|oos_sharpe|``; ``oos_sharpe_se``: the
      i.i.d. standard error of the annualised realised OOS Sharpe
      (``sqrt(ann / n_oos) sqrt(1 + SR_period^2 / 2)``, Lo 2002), so that a
      null Sharpe can be read in standard-error units.
    * ``instrument_beta_r2_relevant`` / ``_noise`` / ``_placebo`` /
      ``_chance``: the instrument-informativeness diagnostic of
      :func:`instrument_beta_r2` (cross-sectional R2 of ``cov[t, :, l]`` on
      ``[1, beta_true]`` averaged over the topics of each kind, and the chance
      level ``K_true / mean N``).
    * ``oos_selection_stability``: mean Jaccard similarity of the selected
      sets of consecutive OOS refits (``result.evaluation.metrics``; ``nan``
      when absent).
    * ``factor_structure``: 1 when some asset loads on the factors
      (``truth.meta["factor_structure"]``, default 1). Without factor
      structure (``no_factor``) the true MVE portfolio is not spanned by
      returns, so ``oos_sharpe_true_mve`` and the ratio are ``nan``.
    * ``runtime_seconds``.

    ``passed`` holds the checks of :func:`scenario_checks` evaluated with the
    (scenario-adjusted) thresholds - an empty dict for the report-only
    ``topic_null``, whose ``all_passed`` is then vacuously true; ``details``
    keeps the vectors behind the scalars (principal-angle cosines, canonical
    correlations, selected / relevant / placebo / strong / weak topic lists,
    thresholds used, sample sizes, the instrument-R2 and beta subsamples).

    Assumptions: the run was made on the data of ``truth`` (same topic ids,
    asset ids and calendar); ``result.fit.F`` rows are indexed by
    ``fit.periods``; period objects are stamped at the last trading day (or
    match ``truth.periods`` by period label).
    """
    fit = result.fit
    values: dict[str, float] = {}
    details: dict[str, Any] = {"scenario": str(scenario), "scenario_key": _canonical_scenario(scenario)}
    ann = _annualization(result, truth)
    rcond = _rcond(result)
    period = _period_alias(result)
    K_true = int(np.asarray(truth.A).shape[1])
    K_hat = int(fit.K)

    # -- selection ---------------------------------------------------------
    fit_topics = [str(n) for n in fit.instrument_names[1:]]
    truth_topics = _truth_topics(truth)
    L_true = len(truth_topics)
    pos = _truth_topic_positions(fit_topics, truth)
    in_truth = pos >= 0
    if not np.all(in_truth):
        logger.warning("compare_to_truth: %d fit instrument(s) are not topics of the truth", int((~in_truth).sum()))
    selected_fit = np.asarray(fit.selected, dtype=bool).ravel()
    relevant = np.asarray(truth.relevant, dtype=bool).ravel()
    placebo = np.asarray(truth.placebo, dtype=bool).ravel()
    selected_truth = np.zeros(L_true, dtype=bool)
    selected_truth[pos[in_truth & selected_fit]] = True
    n_selected = int(selected_fit.sum())
    n_relevant = int(relevant.sum())
    hits = int((selected_truth & relevant).sum())
    recall = hits / n_relevant if n_relevant else float("nan")
    precision = hits / n_selected if n_selected else float("nan")
    if np.isfinite(recall) and np.isfinite(precision):
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        f1 = float("nan")
    values["selection_recall"] = float(recall)
    values["selection_precision"] = float(precision)
    values["selection_f1"] = float(f1)
    values["n_selected"] = float(n_selected)
    values["n_relevant"] = float(n_relevant)
    values["n_relevant_effective"] = float(truth.meta.get("n_relevant_effective", n_relevant))
    values["n_placebo_topics"] = float(placebo.sum())
    values["placebo_selected"] = float((selected_truth & placebo).sum())
    values["L"] = float(len(fit_topics))
    values["K"] = float(K_hat)
    values["K_true"] = float(K_true)
    details["selected_topics"] = [t for t, s in zip(fit_topics, selected_fit) if s]
    details["relevant_topics"] = [t for t, r in zip(truth_topics, relevant) if r]
    details["placebo_topics"] = [t for t, p in zip(truth_topics, placebo) if p]
    details["selected_relevant"] = [t for t, s, r in zip(truth_topics, selected_truth, relevant) if s and r]
    details["selected_placebo"] = [t for t, s, p in zip(truth_topics, selected_truth, placebo) if s and p]
    details["selected_noise"] = [t for t, s, r, p in zip(truth_topics, selected_truth, relevant, placebo) if s and not r and not p]

    # -- recall over the strong / weak half of the relevant topics ------------
    strong, weak, norm_median = _strong_weak_split(truth, relevant)
    n_strong, n_weak = int(strong.sum()), int(weak.sum())
    values["selection_recall_strong"] = float((selected_truth & strong).sum() / n_strong) if n_strong else float("nan")
    values["selection_recall_weak"] = float((selected_truth & weak).sum() / n_weak) if n_weak else float("nan")
    values["n_relevant_strong"] = float(n_strong)
    values["n_relevant_weak"] = float(n_weak)
    values["relevant_norm_median"] = norm_median
    details["strong_topics"] = [t for t, s in zip(truth_topics, strong) if s]
    details["weak_topics"] = [t for t, w in zip(truth_topics, weak) if w]
    details["selected_strong"] = [t for t, s, r in zip(truth_topics, selected_truth, strong) if s and r]
    details["selected_weak"] = [t for t, s, w in zip(truth_topics, selected_truth, weak) if s and w]

    # -- Gamma_tilde column space on the relevant rows that were selected -----
    G_hat = np.zeros((L_true, K_hat))
    G_hat[pos[in_truth]] = np.asarray(fit.Gamma_tilde, dtype=float)[in_truth]
    G_true = np.asarray(truth.Gamma_tilde_true, dtype=float)
    gamma_cos, cosines, rank_hat, rank_true, n_inf = _gamma_subspace_cos(G_hat, G_true, relevant & selected_truth, K_true, "relevant and selected")
    gamma_cos_all, cosines_all, rank_hat_all, rank_true_all, n_inf_all = _gamma_subspace_cos(G_hat, G_true, relevant, K_true, "relevant")
    n_rows_hat = int(np.count_nonzero(np.any(G_hat[relevant] != 0.0, axis=1)))
    values["gamma_subspace_cos"] = gamma_cos
    values["gamma_subspace_cos_all_relevant"] = gamma_cos_all
    values["gamma_relevant_rows_selected"] = float(n_rows_hat)
    values["gamma_informative_cosines"] = float(n_inf)
    details["gamma_principal_cosines"] = cosines.tolist()
    details["gamma_rank_hat"] = rank_hat
    details["gamma_rank_true"] = rank_true
    details["gamma_informative_cosines"] = n_inf
    details["gamma_principal_cosines_all_relevant"] = cosines_all.tolist()
    details["gamma_rank_hat_all_relevant"] = rank_hat_all
    details["gamma_rank_true_all_relevant"] = rank_true_all
    details["gamma_informative_cosines_all_relevant"] = n_inf_all

    # -- factors: canonical correlations ------------------------------------
    populated = _populated_mask(result)
    F_hat = fit.factors_frame().loc[populated]
    Xf, Yf, n_f = _align_frames(F_hat, truth.f_period, period)
    rho_f = canonical_correlations(Xf, Yf) if n_f >= 3 else np.zeros(0)
    values["factor_canonical_corr"] = float(rho_f[0]) if rho_f.size else float("nan")
    values["factor_canonical_corr_mean"] = float(rho_f.sum() / K_true) if rho_f.size else float("nan")
    details["factor_canonical_corrs"] = rho_f.tolist()
    details["n_factor_periods"] = n_f
    if n_f < 3:
        logger.warning("compare_to_truth: only %d common factor period(s) between the fit and the truth", n_f)

    # -- states: canonical correlations -------------------------------------
    n_x = 0
    rho_x = np.zeros(0)
    if result.wrapup is not None:
        Xx, Yx, n_x = _align_frames(result.wrapup.states, truth.x_daily, None)
        rho_x = canonical_correlations(Xx, Yx) if n_x >= 3 else np.zeros(0)
    values["state_canonical_corr"] = float(rho_x[0]) if rho_x.size else float("nan")
    values["state_canonical_corr_mean"] = float(rho_x.sum() / K_true) if rho_x.size else float("nan")
    details["state_canonical_corrs"] = rho_x.tolist()
    details["n_state_days"] = n_x

    # -- impact vector over the selected relevant topics ---------------------
    imp_hat: pd.Series | None = None
    if result.wrapup is not None:
        imp_hat = pd.Series(result.wrapup.impact_z_to_mve).copy()
        imp_hat.index = [str(i) for i in imp_hat.index]
    spearman, sign_agree, n_imp = _impact_agreement(imp_hat, truth, truth_topics, relevant & selected_truth)
    spearman_all, sign_all, n_imp_all = _impact_agreement(imp_hat, truth, truth_topics, relevant)
    values["impact_spearman"] = spearman
    values["impact_sign_agreement"] = sign_agree
    values["impact_spearman_all_relevant"] = spearman_all
    values["impact_sign_agreement_all_relevant"] = sign_all
    details["n_impact_topics"] = n_imp
    details["n_impact_topics_all_relevant"] = n_imp_all

    # -- Sharpe ratios ------------------------------------------------------
    values["mve_sharpe_is"] = float(fit.mve_sharpe(annualization=ann, rcond=rcond))
    values["sharpe_mve_true"] = float(truth.sharpe_mve_true)
    realised = truth.meta.get("sharpe_mve_realized") if isinstance(truth.meta, dict) else None
    values["sharpe_mve_true_realised"] = float(realised) if realised is not None else float("nan")
    factor_structure = bool(truth.meta.get("factor_structure", True)) if isinstance(truth.meta, dict) else True
    values["factor_structure"] = float(factor_structure)
    oos_sharpe = float("nan")
    oos_true = float("nan")
    n_oos = 0
    if result.oos is not None:
        oos_sharpe = float(result.oos.sharpe)
        if not np.isfinite(oos_sharpe):
            oos_sharpe = float(realized_sharpe(result.oos.mve, ann))
        Sigma_p = np.atleast_2d(np.asarray(truth.Sigma_ff_period, dtype=float))
        mu_p = np.asarray(truth.mu_f_period, dtype=float).ravel()
        b_true = np.linalg.pinv(0.5 * (Sigma_p + Sigma_p.T), rcond=rcond) @ mu_p
        mve_true = truth.f_period.to_numpy(dtype=float) @ b_true
        pos_oos = _match_positions(result.oos.mve.index, truth.periods, period)
        keep = pos_oos >= 0
        n_oos = int(keep.sum())
        if n_oos < 2:
            logger.warning("compare_to_truth: %d OOS period(s) matched the truth's periods", n_oos)
        elif not factor_structure:
            logger.info("compare_to_truth: no asset loads on the factors; the true MVE OOS Sharpe is not attainable (nan)")
        else:
            oos_true = float(realized_sharpe(mve_true[pos_oos[keep]], ann))
    values["oos_sharpe"] = oos_sharpe
    values["oos_sharpe_true_mve"] = oos_true
    values["n_oos_periods"] = float(n_oos)
    if n_oos >= 2 and np.isfinite(oos_sharpe):
        values["oos_sharpe_se"] = float(np.sqrt(ann / n_oos) * np.sqrt(1.0 + (oos_sharpe / np.sqrt(ann)) ** 2 / 2.0))
    else:
        values["oos_sharpe_se"] = float("nan")
    if np.isfinite(oos_sharpe) and np.isfinite(oos_true) and oos_true > 0.0:
        values["oos_sharpe_ratio_to_true"] = oos_sharpe / oos_true
    else:
        values["oos_sharpe_ratio_to_true"] = float("nan")
        if np.isfinite(oos_true) and oos_true <= 0.0:
            logger.info("compare_to_truth: realised true OOS Sharpe %.3f <= 0; ratio undefined", oos_true)

    # -- systematic R2 recovered ---------------------------------------------
    sys_r2, n_sys = _systematic_r2_recovered(result, truth, period, asset_ids)
    sys_r2_narr, _ = _systematic_r2_recovered(result, truth, period, asset_ids, intercept=False)
    values["systematic_r2_recovered"] = sys_r2
    values["systematic_r2_recovered_narrative"] = sys_r2_narr
    values["systematic_r2_true"] = float(truth.systematic_r2)
    details["n_systematic_rows"] = n_sys
    metrics = result.evaluation.metrics if result.evaluation is not None else {}
    values["total_r2"] = float(metrics.get("total_r2", fit.total_r2))
    values["pred_r2"] = float(metrics.get("pred_r2", fit.pred_r2))
    stability = metrics.get("oos_selection_stability", float("nan"))
    try:
        values["oos_selection_stability"] = float(stability)
    except (TypeError, ValueError):
        values["oos_selection_stability"] = float("nan")

    # -- instrument informativeness -------------------------------------------
    cov_panel = getattr(result, "covariances", None)
    if cov_panel is not None:
        r2_values, r2_details = instrument_beta_r2(cov_panel, truth, period, asset_ids)
    else:
        r2_values, r2_details = _instrument_r2_nan(), {}
    values.update(r2_values)
    details["instrument_beta_r2"] = r2_details

    # -- implied loadings c Gamma_hat vs the true betas ------------------------
    beta_values, beta_details = beta_canonical_corr(result.panel, fit.Gamma, truth, period, asset_ids)
    values.update(beta_values)
    details["beta_canonical_corr"] = beta_details

    # -- null-scenario metrics ----------------------------------------------
    L_fit = len(fit_topics)
    chance = max(1, int(round(CHANCE_SELECTION_FRACTION * L_fit)))
    values["null_selection_lift"] = n_selected / chance
    values["null_selection_lift_relevant"] = n_selected / max(1, n_relevant)
    values["null_oos_sharpe_abs"] = abs(oos_sharpe) if np.isfinite(oos_sharpe) else float("nan")
    details["chance_selected"] = chance

    # -- bookkeeping --------------------------------------------------------
    values["lam_star"] = float(fit.lam)
    values["lam_max"] = float(result.tuning.lam_max) if result.tuning is not None else float("nan")
    values["converged"] = float(bool(fit.converged))
    values["wrapup_rank_deficient"] = float(result.wrapup.rank_deficient) if result.wrapup is not None else float("nan")
    values["runtime_seconds"] = float(result.timings.get("total", float("nan")))

    thr = scenario_thresholds(thresholds, scenario)
    passed = evaluate_checks(values, thresholds, scenario)
    details["checks"] = list(passed)
    details["thresholds"] = {f.name: getattr(thr, f.name) for f in fields(thr)}
    details["annualization"] = ann
    logger.info(
        "compare_to_truth[%s]: recall %.2f (strong %.2f) precision %.2f beta_cc %.3f placebo %d gamma_cos %.3f "
        "factor_cc %.3f state_cc %.3f impact_rho %.3f oos %.2f (true %.2f, se %.2f) sys_r2 %.3f lift %.2f "
        "instrument R2 rel %.3f noise %.3f chance %.3f stability %.2f -> %d/%d checks passed%s",
        scenario, values["selection_recall"], values["selection_recall_strong"], values["selection_precision"],
        values["beta_canonical_corr"], int(values["placebo_selected"]),
        values["gamma_subspace_cos"], values["factor_canonical_corr"], values["state_canonical_corr"],
        values["impact_spearman"], oos_sharpe, oos_true, values["oos_sharpe_se"], sys_r2, values["null_selection_lift"],
        values["instrument_beta_r2_relevant"], values["instrument_beta_r2_noise"], values["instrument_beta_r2_chance"],
        values["oos_selection_stability"], sum(passed.values()), len(passed),
        "" if passed else " (report-only scenario)",
    )
    return HarnessMetrics(values=values, passed=passed, details=details)


def _strong_weak_split(truth: SimulationTruth, relevant: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """``(strong, weak, median)``: relevant topics with ``||A_l|| >= median`` of the relevant rows' norms, and the rest.

    Ties at the median count as strong. With fewer than two relevant topics
    (or an ``A`` whose row count differs from the mask) both masks are empty
    and the median ``nan``.
    """
    L = relevant.shape[0]
    strong = np.zeros(L, dtype=bool)
    weak = np.zeros(L, dtype=bool)
    A = np.asarray(truth.A, dtype=float)
    if A.ndim != 2 or A.shape[0] != L or int(relevant.sum()) < 2:
        return strong, weak, float("nan")
    norms = np.linalg.norm(A, axis=1)
    median = float(np.median(norms[relevant]))
    strong = relevant & (norms >= median)
    weak = relevant & ~strong
    return strong, weak, median


def _gamma_subspace_cos(
    G_hat: np.ndarray, G_true: np.ndarray, rows: np.ndarray, K_true: int, what: str
) -> tuple[float, np.ndarray, int, int, int]:
    """Mean of the informative principal-angle cosines between ``col(G_hat[rows])`` and ``col(G_true[rows])``.

    Two ``K_true``-dimensional column spaces of a block with ``n`` rows
    share at least ``2 K_true - n`` dimensions, so with fewer than
    ``2 K_true`` rows the ``max(0, 2 K_true - n)`` largest cosines are one by
    dimension counting and say nothing about the estimate (with ``n =
    K_true`` rows *any* full-rank block scores one on every angle). The mean
    is therefore taken over the ``n_informative = min(K_true, n - K_true)``
    smallest of the ``K_true`` cosines; for ``n >= 2 K_true`` this is the
    plain mean over the ``K_true`` principal angles of DESIGN.md Part E.
    ``n`` counts the rows of the block (for the selected rows every row is
    non-zero; for the all-relevant block the zero rows of non-selected
    topics still count towards the ambient dimension).

    Returns ``(mean cosine, cosines, rank_hat, rank_true, n_informative)``;
    the mean is ``nan`` (the subspace is not estimated) when the block has
    at most ``K_true`` rows, when fewer than ``K_true`` of its rows are
    non-zero or its rank is below ``K_true``, or when the true block is
    zero. Missing pairs (a true block of rank below ``K_true``) count as
    zero in the mean.
    """
    H, Tr = G_hat[rows], G_true[rows]
    n_rows = int(np.count_nonzero(np.any(H != 0.0, axis=1)))
    rank_hat = int(_orthonormal_basis(H).shape[1])
    rank_true = int(_orthonormal_basis(Tr).shape[1])
    cosines = subspace_cosines(H, Tr) if rank_hat and rank_true else np.zeros(0)
    n_informative = max(0, min(K_true, int(H.shape[0]) - K_true))
    if n_informative < 1 or n_rows < K_true or rank_hat < K_true or rank_true < 1:
        logger.info(
            "compare_to_truth: Gamma subspace on the %s rows not identified (%d non-zero rows, rank %d, K=%d)",
            what, n_rows, rank_hat, K_true,
        )
        return float("nan"), cosines, rank_hat, rank_true, n_informative
    padded = np.zeros(K_true)
    padded[: min(cosines.size, K_true)] = cosines[:K_true]
    return float(np.sort(padded)[:n_informative].mean()), cosines, rank_hat, rank_true, n_informative


def _impact_agreement(
    imp_hat: pd.Series | None, truth: SimulationTruth, truth_topics: Sequence[str], mask: np.ndarray
) -> tuple[float, float, int]:
    """``(Spearman, sign agreement, n)`` of the estimated vs true impact vector over the topics of ``mask``.

    Both are ``nan`` with fewer than three topics carrying a finite pair
    (``n`` still counts them); the Spearman correlation is also ``nan`` when
    either side is constant.
    """
    if imp_hat is None or not mask.any():
        return float("nan"), float("nan"), 0
    topics = [t for t, m in zip(truth_topics, mask) if m]
    a = imp_hat.reindex(topics).to_numpy(dtype=float)
    b = np.asarray(truth.impact_z_to_mve_true, dtype=float).ravel()[mask]
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    a, b = a[ok], b[ok]
    spearman = float(_stats.spearmanr(a, b).statistic) if a.std() > 0.0 and b.std() > 0.0 else float("nan")
    return spearman, float(np.mean(np.sign(a) == np.sign(b))), n


def _systematic_r2_recovered(
    result: PipelineResult,
    truth: SimulationTruth,
    period: str | None,
    asset_ids: Sequence[str] | None,
    intercept: bool = True,
) -> tuple[float, int]:
    """``1 - sum (s - s_hat)^2 / sum s^2`` over panel rows, ``s = beta' f_true``, ``s_hat = c Gamma_hat f_hat``.

    Panel rows are mapped to ``truth.beta[t, i, :]`` and ``truth.f_period[t]``
    by period date (exact, else period label) and asset id. With
    ``intercept=False`` the constant row of ``Gamma`` is zeroed, so ``s_hat``
    is the narrative part ``cov_{i,t-1} Gamma_tilde_hat f_hat_t`` only: the
    intercept row alone reproduces the common component of the systematic
    return (a market-like factor) even when the narratives carry no
    information, so the narrative-only variant is the discriminating
    diagnostic against the null. Returns ``(nan, 0)`` when no row can be
    mapped.
    """
    panel = result.panel
    fit = result.fit
    beta = np.asarray(truth.beta, dtype=float)
    a_pos = _truth_asset_positions(panel.assets, truth, asset_ids, "systematic R2")
    if a_pos is None:
        return float("nan"), 0
    t_pos = _match_positions(pd.DatetimeIndex(panel.periods), pd.DatetimeIndex(truth.periods), period)
    rows_t = t_pos[panel.t_idx]
    rows_a = a_pos[panel.asset_idx]
    ok = (rows_t >= 0) & (rows_a >= 0)
    if not ok.any():
        logger.warning("systematic R2: no panel row could be mapped to the truth")
        return float("nan"), 0
    f_true = truth.f_period.to_numpy(dtype=float)
    K_true = f_true.shape[1]
    b = beta[rows_t[ok], rows_a[ok], :K_true]
    s_true = np.einsum("nk,nk->n", b, f_true[rows_t[ok]])
    Gamma = np.array(fit.Gamma, dtype=float, copy=True)
    if not intercept:
        Gamma[0] = 0.0
    B_hat = panel.X[ok] @ Gamma
    s_hat = np.einsum("nk,nk->n", B_hat, np.asarray(fit.F, dtype=float)[panel.t_idx[ok]])
    finite = np.isfinite(s_true) & np.isfinite(s_hat)
    s_true, s_hat = s_true[finite], s_hat[finite]
    n = int(finite.sum())
    denom = float(s_true @ s_true)
    if n == 0 or denom <= 0.0:
        return float("nan"), n
    resid = s_true - s_hat
    return float(1.0 - float(resid @ resid) / denom), n


# ---------------------------------------------------------------------------
# instrument informativeness
# ---------------------------------------------------------------------------
_INSTRUMENT_R2_KEYS: tuple[str, ...] = (
    "instrument_beta_r2_relevant",
    "instrument_beta_r2_noise",
    "instrument_beta_r2_placebo",
    "instrument_beta_r2_chance",
    "instrument_beta_r2_n_periods",
    "instrument_beta_r2_mean_n_assets",
)


def _instrument_r2_nan() -> dict[str, float]:
    return {k: float("nan") for k in _INSTRUMENT_R2_KEYS}


def _subsample_indices(T: int, every: int = INSTRUMENT_R2_EVERY, min_periods: int = INSTRUMENT_R2_MIN_PERIODS) -> np.ndarray:
    """Positions of the period subsample of the per-period diagnostics.

    Every ``every``-th position counted back from the last one (``T-1``,
    ``T-1-every``, ...; returned ascending), widened to ``min_periods``
    positions spread evenly over ``0 .. T-1`` when there are fewer.
    """
    T = int(T)
    if T <= 0:
        return np.zeros(0, dtype=np.int64)
    every = max(1, int(every))
    min_periods = max(1, int(min_periods))
    idx = np.arange(T - 1, -1, -every)[::-1]
    if idx.size < min_periods:
        idx = np.unique(np.linspace(0, T - 1, min(min_periods, T)).round().astype(np.int64))
    return idx.astype(np.int64)


def instrument_beta_r2(
    cov: CovariancePanel,
    truth: SimulationTruth,
    period: str | None = None,
    asset_ids: Sequence[str] | None = None,
    every: int = INSTRUMENT_R2_EVERY,
    min_periods: int = INSTRUMENT_R2_MIN_PERIODS,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Cross-sectional R2 of every topic's covariance instrument on the true loadings.

    For a subsample of covariance periods (every ``every``-th period counted
    back from the last one, widened to at least ``min_periods`` periods
    spread evenly over the panel) and for every topic ``l``, the cross-section
    ``cov[t, :, l]`` over the assets with a finite instrument and a finite
    true loading is regressed by OLS on ``[1, beta_true[t, :, :]]`` and the
    R2 is recorded; a topic's R2 is then averaged over the subsample.

    Why this is the diagnostic of the ``topic_null`` mechanism: under Eq. 5
    a relevant topic's instrument is ``beta_{i,t} Sigma_ff A_l'``, exactly
    linear in ``beta``, so its R2 is high whenever the instrument is
    precise. A noise topic's instrument is ``beta_i' G_{t,l}`` plus
    idiosyncratic noise (module docstring), *also* linear in ``beta``: its
    R2 measures how much of the instrument's cross-sectional variation is
    the persistent common vector ``G_{t,l}`` rather than idiosyncratic noise,
    i.e. how much of ``beta`` the estimator can read off pure noise topics.
    The chance level is the expected R2 of a regression on ``K`` random
    regressors, ``K_true / mean N``. Under ``no_factor`` the true loadings
    are identically zero, the regression is intercept-only and every R2 is
    exactly zero.

    Returns ``(values, details)``: ``values`` holds
    ``instrument_beta_r2_relevant`` (mean over the relevant topics; ``nan``
    when there are none), ``instrument_beta_r2_noise`` (non-relevant,
    non-placebo topics), ``instrument_beta_r2_placebo``,
    ``instrument_beta_r2_chance``, ``instrument_beta_r2_n_periods`` and
    ``instrument_beta_r2_mean_n_assets``; ``details`` holds the subsample
    dates, the cross-section sizes and the per-topic mean R2. Covariance
    periods are matched to ``truth.periods`` (exact date, else period label)
    and assets to ``truth.beta`` exactly as the systematic-R2 metric does
    (:func:`_truth_asset_positions`). Everything is ``nan`` when no period
    can be mapped.
    """
    values = _instrument_r2_nan()
    details: dict[str, Any] = {"periods": [], "n_assets": [], "per_topic": {}}
    C_all = np.asarray(cov.values, dtype=float)
    if C_all.ndim != 3 or C_all.shape[0] == 0:
        return values, details
    T_cov, N, L = C_all.shape
    a_pos = _truth_asset_positions(cov.assets, truth, asset_ids, "instrument R2")
    if a_pos is None:
        return values, details
    t_pos = _match_positions(pd.DatetimeIndex(cov.periods), pd.DatetimeIndex(truth.periods), period)
    l_pos = _truth_topic_positions(cov.topics, truth)
    beta = np.asarray(truth.beta, dtype=float)
    K_true = int(beta.shape[2])
    relevant = np.asarray(truth.relevant, dtype=bool).ravel()
    placebo = np.asarray(truth.placebo, dtype=bool).ravel()

    idx = _subsample_indices(T_cov, every, min_periods)
    idx = idx[t_pos[idx] >= 0]
    if idx.size == 0:
        logger.warning("instrument R2: no covariance period could be mapped to the truth")
        return values, details

    a_ok = a_pos >= 0
    r2_rows: list[np.ndarray] = []
    n_rows: list[int] = []
    dates: list[str] = []
    for t in idx:
        C = C_all[t]  # (N, L)
        B = np.full((N, K_true), np.nan)
        B[a_ok] = beta[t_pos[t], a_pos[a_ok], :]
        ok = np.all(np.isfinite(C), axis=1) & np.all(np.isfinite(B), axis=1)
        n = int(ok.sum())
        if n < K_true + 2:
            continue
        X = np.column_stack([np.ones(n), B[ok]])
        Y = C[ok]
        sst = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        ssr = np.sum((Y - X @ coef) ** 2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = np.where(sst > 0.0, 1.0 - ssr / sst, np.nan)
        r2_rows.append(np.clip(r2, 0.0, 1.0))
        n_rows.append(n)
        dates.append(pd.Timestamp(cov.periods[t]).isoformat())
    if not r2_rows:
        logger.warning("instrument R2: no covariance period had a usable cross-section")
        return values, details

    R2 = np.vstack(r2_rows)  # (n_periods, L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        per_topic = np.nanmean(R2, axis=0)
    in_truth = l_pos >= 0
    kind_rel = in_truth & relevant[np.clip(l_pos, 0, None)]
    kind_pl = in_truth & placebo[np.clip(l_pos, 0, None)]
    kind_noise = in_truth & ~kind_rel & ~kind_pl

    def _mean(mask: np.ndarray) -> float:
        vals = per_topic[mask]
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else float("nan")

    mean_n = float(np.mean(n_rows))
    values["instrument_beta_r2_relevant"] = _mean(kind_rel)
    values["instrument_beta_r2_noise"] = _mean(kind_noise)
    values["instrument_beta_r2_placebo"] = _mean(kind_pl)
    values["instrument_beta_r2_chance"] = float(K_true / mean_n) if mean_n > 0 else float("nan")
    values["instrument_beta_r2_n_periods"] = float(len(n_rows))
    values["instrument_beta_r2_mean_n_assets"] = mean_n
    details["periods"] = dates
    details["n_assets"] = n_rows
    details["per_topic"] = {str(name): float(v) for name, v in zip(cov.topics, per_topic)}
    return values, details


# ---------------------------------------------------------------------------
# implied loadings
# ---------------------------------------------------------------------------
_BETA_CC_KEYS: tuple[str, ...] = (
    "beta_canonical_corr",
    "beta_canonical_corr_mean",
    "beta_canonical_corr_n_periods",
    "beta_canonical_corr_mean_n_assets",
)


def _beta_cc_nan() -> dict[str, float]:
    return {k: float("nan") for k in _BETA_CC_KEYS}


def beta_canonical_corr(
    panel: IPCAPanel,
    Gamma: np.ndarray,
    truth: SimulationTruth,
    period: str | None = None,
    asset_ids: Sequence[str] | None = None,
    every: int = INSTRUMENT_R2_EVERY,
    min_periods: int = INSTRUMENT_R2_MIN_PERIODS,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Canonical correlations between the implied loadings ``X_t Gamma`` and the true ``beta_t`` over sampled periods.

    IPCA identifies the loadings ``beta_{i,t} = c_{i,t-1} Gamma`` (BKS Eq.
    7), not ``Gamma`` itself: ``Gamma`` is defined up to a rotation of the
    factor space, and which instruments carry the loadings is a matter of
    selection. The implied betas are invariant to both, so their agreement
    with the true betas is the loading-recovery metric that does not
    confound recovery with recall (DESIGN.md Part E).

    For every return period ``t`` of the subsample (:func:`_subsample_indices`
    over ``panel.periods``: every ``every``-th period counted back from the
    last, at least ``min_periods``) the rows of ``panel`` in that period give
    ``beta_hat = X_t Gamma`` (``N_t x K_hat``; the constant row of ``Gamma``
    only shifts every asset by the same vector and drops out in the
    centring); the assets are mapped to ``truth.beta[t_true, assets, :]``
    with ``t_true`` the position of the panel's *return* period in
    ``truth.periods`` (``beta_{i,t}`` applies to return period ``t``; exact
    date, else period label) and the assets by id
    (:func:`_truth_asset_positions`); rows with a non-finite true loading
    are dropped and :func:`canonical_correlations` of the two centred
    ``N_t x K`` matrices is taken. A period with fewer than ``max(K_true,
    K_hat) + 2`` usable rows, or whose true block is zero (``no_factor``),
    is skipped.

    Returns ``(values, details)``: ``values`` holds ``beta_canonical_corr``
    (mean over the periods of the first canonical correlation),
    ``beta_canonical_corr_mean`` (mean over the periods of the mean over
    ``K_true`` canonical correlations, missing pairs counted as zero),
    ``beta_canonical_corr_n_periods`` and ``beta_canonical_corr_mean_n_assets``;
    ``details`` the subsample dates, the cross-section sizes and the
    per-period first / mean correlations. Everything is ``nan`` when no
    period can be used.
    """
    values = _beta_cc_nan()
    details: dict[str, Any] = {"periods": [], "n_assets": [], "first": [], "mean": []}
    Gamma = np.asarray(Gamma, dtype=float)
    T_p = len(panel.periods)
    if T_p == 0 or Gamma.ndim != 2 or Gamma.shape[0] != panel.p:
        logger.warning("beta canonical correlation: empty panel or Gamma of shape %s for %d instruments; skipped", Gamma.shape, panel.p)
        return values, details
    a_pos = _truth_asset_positions(panel.assets, truth, asset_ids, "beta canonical correlation")
    if a_pos is None:
        return values, details
    t_pos = _match_positions(pd.DatetimeIndex(panel.periods), pd.DatetimeIndex(truth.periods), period)
    beta = np.asarray(truth.beta, dtype=float)
    K_true = int(beta.shape[2])
    K_hat = int(Gamma.shape[1])
    idx = _subsample_indices(T_p, every, min_periods)
    idx = idx[t_pos[idx] >= 0]
    if idx.size == 0:
        logger.warning("beta canonical correlation: no panel period could be mapped to the truth")
        return values, details

    t_idx = np.asarray(panel.t_idx)
    firsts: list[float] = []
    means: list[float] = []
    n_rows: list[int] = []
    dates: list[str] = []
    for t in idx:
        lo, hi = np.searchsorted(t_idx, [t, t + 1])
        if hi <= lo:
            continue
        ap = a_pos[panel.asset_idx[lo:hi]]
        ok = ap >= 0
        if not ok.any():
            continue
        B_hat = panel.X[lo:hi][ok] @ Gamma
        B = beta[t_pos[t], ap[ok], :]
        finite = np.all(np.isfinite(B), axis=1) & np.all(np.isfinite(B_hat), axis=1)
        n = int(finite.sum())
        if n < max(K_true, K_hat) + 2:
            continue
        rho = canonical_correlations(B_hat[finite], B[finite])
        if rho.size == 0:
            continue
        firsts.append(float(rho[0]))
        means.append(float(rho.sum() / K_true))
        n_rows.append(n)
        dates.append(pd.Timestamp(panel.periods[t]).isoformat())
    if not firsts:
        logger.info("beta canonical correlation: no sampled period had a usable cross-section (no true loadings?)")
        return values, details
    values["beta_canonical_corr"] = float(np.mean(firsts))
    values["beta_canonical_corr_mean"] = float(np.mean(means))
    values["beta_canonical_corr_n_periods"] = float(len(firsts))
    values["beta_canonical_corr_mean_n_assets"] = float(np.mean(n_rows))
    details["periods"] = dates
    details["n_assets"] = n_rows
    details["first"] = firsts
    details["mean"] = means
    return values, details


# ---------------------------------------------------------------------------
# run_harness
# ---------------------------------------------------------------------------
def default_pipeline_config(fast: bool = False) -> PipelineConfig:
    """The pipeline config the harness uses when none is given.

    Full mode: the package defaults with ``burn_in_periods = 12``,
    ``oos_fraction = 0.4``, ``refit_every = 12`` and ``n_lambdas = 20``.
    Fast mode: ``burn_in_periods = 6``, ``n_lambdas = FAST_N_LAMBDAS`` (12),
    ``min_train_periods = 24`` and ``ratio = FAST_LAMBDA_RATIO`` (``1e-3``,
    the package default). The fast grid spans the same range as the full
    one with fewer points: the dense end of the path is cheap with the
    numba kernel (D46), and a grid truncated at ``0.05 lam_max`` misses the
    tuned ``lambda`` on the small fast panels (see :data:`FAST_LAMBDA_RATIO`).
    """
    if fast:
        return PipelineConfig(
            covariance=CovarianceConfig(burn_in_periods=6),
            estimation=EstimationConfig(lam_grid=LambdaGridConfig(n_lambdas=FAST_N_LAMBDAS, ratio=FAST_LAMBDA_RATIO)),
            oos=OOSConfig(oos_fraction=0.4, refit_every=12, min_train_periods=24),
            name="harness-fast",
        )
    return PipelineConfig(
        covariance=CovarianceConfig(burn_in_periods=12),
        estimation=EstimationConfig(lam_grid=LambdaGridConfig(n_lambdas=20)),
        oos=OOSConfig(oos_fraction=0.4, refit_every=12),
        name="harness",
    )


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, pd.Series):
        return {str(k): v for k, v in o.items()}
    if isinstance(o, pd.DataFrame):
        return {str(k): {str(c): v for c, v in row.items()} for k, row in o.to_dict(orient="index").items()}
    if isinstance(o, pd.Index):
        return [str(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if is_dataclass(o) and not isinstance(o, type):
        return {f.name: getattr(o, f.name) for f in fields(o)}
    return str(o)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _gamma_norm_frame(tr: TuningResult, instrument_names: Sequence[str]) -> pd.DataFrame:
    """One row per path point: ``K``, ``lam`` and ``||Gamma_l||`` per instrument (BKS Figure 2)."""
    names = [str(n) for n in instrument_names]
    rows = []
    for pt in tr.path:
        norms = np.asarray(pt.gamma_norms, dtype=float).ravel()
        if norms.shape[0] == len(names):
            cols = names
        elif norms.shape[0] == len(names) - 1:
            cols = names[1:]
        else:
            cols = [f"g{i}" for i in range(norms.shape[0])]
        row: dict[str, Any] = {"K": int(pt.K), "lam": float(pt.lam), "criterion": pt.criterion}
        row.update(dict(zip(cols, norms)))
        rows.append(row)
    return pd.DataFrame(rows)


def _sim_summary(cfg: SimulationConfig) -> dict[str, Any]:
    return {
        "n_assets": cfg.n_assets,
        "n_topics": cfg.n_topics,
        "n_relevant": cfg.n_relevant,
        "n_placebo": cfg.n_placebo,
        "K": cfg.K,
        "n_years": cfg.n_years,
        "period": cfg.period,
        "signal_strength": cfg.signal_strength,
        "mve_sharpe_annual": cfg.mve_sharpe_annual,
        "attention_model": cfg.attention_model,
        "unbalanced_fraction": cfg.unbalanced_fraction,
        "missing_day_fraction": cfg.missing_day_fraction,
    }


def run_harness(
    cfg: HarnessConfig,
    pipeline_cfg: PipelineConfig | None = None,
    progress: ProgressFn | None = None,
    base: SimulationConfig | None = None,
) -> HarnessResult:
    """Run every scenario of ``cfg`` for ``cfg.n_seeds`` seeds and compare with the truth (Part E).

    For each ``(scenario, seed)``: ``sim_cfg = scenario_config(scenario, base,
    fast)`` with the seed replaced (``fast = cfg.fast`` unless ``base`` is
    given, in which case ``base`` fixes the sizes and ``cfg.fast`` only
    selects the fast pipeline defaults); ``simulate``; ``run_pipeline`` with
    ``pipeline_cfg`` (default :func:`default_pipeline_config(cfg.fast)`);
    :func:`compare_to_truth`. A run that raises is logged with its traceback,
    recorded in ``meta["errors"]`` and reported as a failed row (all checks
    False, metrics ``nan``); the harness raises only when every run failed.

    Returns a :class:`HarnessResult`:

    * ``per_run``: one row per run with ``scenario``, ``seed``, every metric,
      every pass flag as ``pass_<check>`` (1.0 / 0.0, ``nan`` when the check
      does not apply to the scenario - every flag for the report-only
      ``topic_null``), ``all_passed`` (vacuously true, i.e. ``1.0``, for a
      report-only scenario that ran without error), ``runtime_seconds`` and
      ``error``;
    * ``summary``: ``per_run.groupby("scenario")[metrics].agg(["mean", "std"])``
      (MultiIndex columns ``(metric, stat)``, scenario order preserved);
    * ``passed``: share of seeds passing each check per scenario (``nan``
      where not applicable) plus an ``all`` column (``1.0`` for a report-only
      scenario whose runs completed);
    * ``scenario_configs``: the :class:`SimulationConfig` of each scenario
      (seed 0); ``thresholds``; ``meta`` with the pipeline config, sizes,
      the solver backend (``"numba"`` or ``"numpy"``, D46), timings,
      artefact paths and errors.

    Artefacts under ``cfg.output_dir/artefacts``: per run
    ``<scenario>_seed<k>_metrics.json`` (values, pass flags, details and the
    pipeline's evaluation metrics), ``..._lambda_path.csv`` and
    ``..._gamma_norms.csv``; under ``cfg.output_dir``: ``per_run.csv``,
    ``summary.csv``, ``passed.csv``. ``progress(done, total, message)`` is
    called before every run and once at the end.
    """
    started = datetime.now(timezone.utc)
    t_all = time.perf_counter()
    out = Path(cfg.output_dir)
    art = out / "artefacts"
    art.mkdir(parents=True, exist_ok=True)
    pcfg = pipeline_cfg if pipeline_cfg is not None else default_pipeline_config(bool(cfg.fast))
    if not isinstance(pcfg, PipelineConfig):
        raise TypeError("pipeline_cfg must be a PipelineConfig")
    scenarios = [str(s) for s in cfg.scenarios]
    seeds = list(range(int(cfg.n_seeds)))
    if not scenarios or not seeds:
        raise ValueError("HarnessConfig needs at least one scenario and one seed")
    runs = [(s, k) for s in scenarios for k in seeds]
    n_runs = len(runs)
    sim_fast = bool(cfg.fast) and base is None
    logger.info(
        "run_harness: %d scenario(s) x %d seed(s) = %d run(s), fast=%s (sizes %s), pipeline %r, output %s",
        len(scenarios), len(seeds), n_runs, cfg.fast, "fast" if sim_fast else ("base" if base is not None else "full"),
        pcfg.name, out,
    )

    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    scenario_cfgs: dict[str, SimulationConfig] = {}
    artefacts: dict[str, dict[str, str]] = {}
    checks_all = [name for name, *_ in CHECKS]

    for k, (scenario, seed) in enumerate(runs):
        key = f"{scenario}_seed{seed}"
        msg = f"{scenario} seed {seed} ({k + 1}/{n_runs})"
        if progress is not None:
            progress(k, n_runs, msg)
        sim_cfg = replace(scenario_config(scenario, base=base, fast=sim_fast), seed=int(seed))
        scenario_cfgs.setdefault(scenario, sim_cfg)
        t0 = time.perf_counter()
        row: dict[str, Any] = {"scenario": scenario, "seed": int(seed)}
        applicable = scenario_checks(scenario)
        try:
            data = simulate(sim_cfg, scenario=scenario)
            res = run_pipeline(data.attention, data.returns, replace(pcfg, name=f"harness-{key}"))
            m = compare_to_truth(res, data.truth, cfg.thresholds, scenario, asset_ids=data.returns.assets)
        except Exception as exc:  # one bad run must not lose the others (logged with traceback)
            logger.error("run_harness: %s failed: %s: %s", key, type(exc).__name__, exc, exc_info=True)
            errors[key] = f"{type(exc).__name__}: {exc}"
            row.update({metric: float("nan") for metric in METRICS})
            row.update({f"pass_{c}": (0.0 if c in applicable else float("nan")) for c in checks_all})
            row["all_passed"] = False
            row["runtime_seconds"] = time.perf_counter() - t0
            row["error"] = errors[key]
            rows.append(row)
            continue
        elapsed = time.perf_counter() - t0
        row.update({metric: float(v) for metric, v in m.values.items()})
        row.update({f"pass_{c}": (float(m.passed[c]) if c in m.passed else float("nan")) for c in checks_all})
        row["all_passed"] = bool(m.all_passed)
        row["runtime_seconds"] = float(m.values.get("runtime_seconds", elapsed))
        row["harness_seconds"] = elapsed
        row["error"] = ""
        rows.append(row)

        files: dict[str, str] = {}
        p_metrics = art / f"{key}_metrics.json"
        _write_json(
            p_metrics,
            {
                "scenario": scenario,
                "seed": int(seed),
                "values": m.values,
                "passed": m.passed,
                "details": m.details,
                "pipeline_metrics": res.evaluation.metrics,
                "timings": res.timings,
                "simulation": _sim_summary(sim_cfg),
            },
        )
        files["metrics"] = str(p_metrics)
        p_path = art / f"{key}_lambda_path.csv"
        res.tuning.path_frame().rename_axis("point").to_csv(p_path)
        files["lambda_path"] = str(p_path)
        p_norms = art / f"{key}_gamma_norms.csv"
        _gamma_norm_frame(res.tuning, res.fit.instrument_names).rename_axis("point").to_csv(p_norms)
        files["gamma_norms"] = str(p_norms)
        artefacts[key] = files
        logger.info(
            "run_harness: %s done in %.1fs (pipeline %.1fs): %d/%d checks passed%s",
            key, elapsed, row["runtime_seconds"], sum(m.passed.values()), len(m.passed),
            "" if m.all_passed else " (failed: " + ", ".join(c for c, ok in m.passed.items() if not ok) + ")",
        )

    if len(errors) == n_runs:
        raise RuntimeError(f"every harness run failed: {errors}")

    per_run = pd.DataFrame(rows)
    metric_cols = [c for c in METRICS if c in per_run.columns]
    extra_cols = [
        c for c in per_run.columns
        if c not in ("scenario", "seed", "all_passed", "error", "harness_seconds")
        and not c.startswith("pass_") and c not in metric_cols
    ]
    grouped = per_run.groupby("scenario", sort=False)
    summary = grouped[metric_cols + extra_cols].agg(["mean", "std"])
    pass_cols = [f"pass_{c}" for c in checks_all]
    passed = grouped[pass_cols].mean()
    passed.columns = checks_all
    passed["all"] = grouped["all_passed"].apply(lambda s: float(np.mean(s.astype(bool))))
    elapsed_all = time.perf_counter() - t_all

    per_run.to_csv(out / "per_run.csv", index=False)
    summary.to_csv(out / "summary.csv")
    passed.to_csv(out / "passed.csv")

    meta: dict[str, Any] = {
        "package_version": __version__,
        "solver_backend": active_backend(),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_all,
        "n_runs": n_runs,
        "n_failed": len(errors),
        "scenarios": scenarios,
        "seeds": seeds,
        "fast": bool(cfg.fast),
        "sizes": "fast" if sim_fast else ("base" if base is not None else "full"),
        "pipeline_config": config_to_dict(pcfg),
        "pipeline_name": pcfg.name,
        "base_config": config_to_dict(base) if base is not None else None,
        "simulation": {s: _sim_summary(c) for s, c in scenario_cfgs.items()},
        "output_dir": str(out),
        "artefact_dir": str(art),
        "artefacts": artefacts,
        "tables": {"per_run": str(out / "per_run.csv"), "summary": str(out / "summary.csv"), "passed": str(out / "passed.csv")},
        "errors": errors,
        "checks": {s: list(scenario_checks(s)) for s in scenarios},
        "report_only": [s for s in scenarios if not scenario_checks(s)],
        "thresholds_used": {s: {f.name: getattr(scenario_thresholds(cfg.thresholds, s), f.name) for f in fields(HarnessThresholds)} for s in scenarios},
    }
    logger.info(
        "run_harness: finished %d run(s) in %.1fs, %d failed; share passing all checks: %s",
        n_runs, elapsed_all, len(errors), ", ".join(f"{s}={passed.loc[s, 'all']:.2f}" for s in passed.index),
    )
    if progress is not None:
        progress(n_runs, n_runs, "done")
    return HarnessResult(
        per_run=per_run,
        summary=summary,
        passed=passed,
        scenario_configs=dict(scenario_cfgs),
        thresholds=cfg.thresholds,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt(x: Any, digits: int = 3, keep_int: bool = True) -> str:
    """Number formatting of the report: ``nan`` for undefined, integers bare (unless ``keep_int=False``)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "nan"
    if keep_int and v.is_integer() and abs(v) < 1e6:
        return str(int(v))
    return f"{v:.{digits}f}"


def _fmt_seconds(x: Any) -> str:
    return _fmt(x, 1, keep_int=False)


def _mean_std(mean: Any, std: Any, digits: int = 3) -> str:
    m = _fmt(mean, digits)
    try:
        s = float(std)
    except (TypeError, ValueError):
        return m
    return m if not np.isfinite(s) else f"{m} ± {s:.{digits}f}"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _scenario_order(result: HarnessResult) -> list[str]:
    order = list(result.meta.get("scenarios", [])) if isinstance(result.meta, dict) else []
    seen = list(dict.fromkeys(result.per_run["scenario"].tolist())) if "scenario" in result.per_run else []
    return [s for s in order if s in seen] + [s for s in seen if s not in order]


def _thresholds_of(result: HarnessResult, scenario: str) -> HarnessThresholds:
    thr = result.thresholds if isinstance(result.thresholds, HarnessThresholds) else HarnessThresholds()
    return scenario_thresholds(thr, scenario)


def _col_mean(frame: pd.DataFrame, col: str) -> float:
    """Mean of a numeric column over the frame's rows (``nan`` when absent or empty)."""
    if col not in frame or not len(frame):
        return float("nan")
    return float(pd.to_numeric(frame[col], errors="coerce").mean())


def _span(frame: pd.DataFrame, col: str, digits: int = 3) -> str:
    """``mean`` for one row, ``mean (min a, max b)`` for several; ``nan`` when undefined."""
    if col not in frame or not len(frame):
        return "nan"
    vals = pd.to_numeric(frame[col], errors="coerce")
    if not vals.notna().any():
        return "nan"
    if len(vals) == 1:
        return _fmt(vals.iloc[0], digits)
    return f"{_fmt(vals.mean(), digits)} (min {_fmt(vals.min(), digits)}, max {_fmt(vals.max(), digits)})"


def _sharpe_with_se(frame: pd.DataFrame) -> str:
    """``oos_sharpe`` span plus its size in standard-error units (``oos_sharpe_se``, i.i.d. Lo 2002)."""
    text = _span(frame, "oos_sharpe")
    se = _col_mean(frame, "oos_sharpe_se")
    m = _col_mean(frame, "oos_sharpe")
    if np.isfinite(se) and se > 0 and np.isfinite(m):
        text += f", about {abs(m) / se:.1f} standard errors from zero (se {se:.2f})"
    return text


def _instrument_text(frame: pd.DataFrame) -> str:
    return (
        f"instrument R2 on the true loadings: relevant {_span(frame, 'instrument_beta_r2_relevant')}, "
        f"noise {_span(frame, 'instrument_beta_r2_noise')}, placebo {_span(frame, 'instrument_beta_r2_placebo')} "
        f"vs chance {_fmt(_col_mean(frame, 'instrument_beta_r2_chance'))}; selection stability across refits "
        f"{_span(frame, 'oos_selection_stability')}"
    )


def _baseline_rows(result: HarnessResult, scenario: str) -> pd.DataFrame:
    """Error-free rows of the baseline scenario(s) of the same harness run (empty when none)."""
    per_run = result.per_run
    if "scenario" not in per_run:
        return per_run.iloc[0:0]
    is_base = per_run["scenario"].map(lambda x: _canonical_scenario(x) == "baseline") & (per_run["scenario"] != scenario)
    rows = per_run[is_base]
    if "error" in rows:
        rows = rows[rows["error"].fillna("") == ""]
    return rows


NO_FACTOR_TEXT: str = "n/a (no priced factor in returns)"
"""Report cell for the true MVE Sharpe ratios of a scenario whose returns have no factor structure."""

_TRUE_SHARPE_METRICS: tuple[str, ...] = ("sharpe_mve_true", "oos_sharpe_true_mve")


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _has_factor_structure(result: HarnessResult, scenario: str) -> bool:
    """Whether some asset of ``scenario`` loads on the factors, i.e. whether the true MVE Sharpe is attainable.

    Read from ``values["factor_structure"]`` of the scenario's runs (the
    truth's ``meta["factor_structure"]``); without that column (a hand-built
    result) from the scenario's :class:`SimulationConfig` (every class with
    zero loadings and ``beta_innov_sd = 0`` means no structure), and without
    a config from the scenario name (``no_factor``).
    """
    per_run = result.per_run
    if "scenario" in per_run and "factor_structure" in per_run:
        vals = pd.to_numeric(per_run.loc[per_run["scenario"] == scenario, "factor_structure"], errors="coerce").dropna()
        if len(vals):
            return bool((vals != 0.0).any())
    cfg = result.scenario_configs.get(scenario) if isinstance(result.scenario_configs, dict) else None
    if isinstance(cfg, SimulationConfig):
        loads = any(
            float(spec.beta_sd) != 0.0 or any(float(b) != 0.0 for b in spec.beta_mean) for spec in cfg.asset_classes
        )
        return loads or float(cfg.beta_innov_sd) != 0.0
    return _canonical_scenario(scenario) != "no_factor"


def _true_sharpe_cell(metric: str, text: str, factor_structure: bool) -> str:
    """``text``, or :data:`NO_FACTOR_TEXT` for a true-MVE Sharpe of a scenario without factor structure."""
    return NO_FACTOR_TEXT if (metric in _TRUE_SHARPE_METRICS and not factor_structure) else text


def _null_sharpe_within_se(ok_runs: pd.DataFrame, thr: HarnessThresholds) -> list[str]:
    """Seeds whose |OOS Sharpe| fails the absolute ``no_factor`` threshold but sits within two standard errors of zero.

    The absolute threshold ``null_oos_sharpe_abs_max`` (0.75) is calibrated
    for 90-100 OOS periods (``2 sqrt(12 / n_oos)``); on a short fast panel
    the standard error (``oos_sharpe_se``, Lo 2002) is the informative
    yardstick, so the report says so explicitly.
    """
    bound = float(thr.null_oos_sharpe_abs_max)
    notes: list[str] = []
    if "oos_sharpe" not in ok_runs or "oos_sharpe_se" not in ok_runs:
        return notes
    for _, r in ok_runs.iterrows():
        sr, se = _num(r.get("oos_sharpe")), _num(r.get("oos_sharpe_se"))
        if not (np.isfinite(sr) and np.isfinite(se) and se > 0.0):
            continue
        k = abs(sr) / se
        if abs(sr) > bound and k <= 2.0:
            n_oos = _num(r.get("n_oos_periods"))
            at = f" at {int(n_oos)} OOS periods" if np.isfinite(n_oos) and n_oos > 0 else ""
            notes.append(
                f"seed {r.get('seed', '?')}: the realised OOS Sharpe {sr:.2f} fails the absolute threshold "
                f"(|SR| <= {bound:.2f}) but is within {k:.1f} standard errors of zero (se {se:.2f}{at}); "
                f"the {bound:.2f} threshold is calibrated for 90-100 OOS periods"
            )
    return notes


def _expected_vs_observed(result: HarnessResult, scenario: str) -> str:
    """One short paragraph per scenario generated from the numbers: expectations, observations, failures.

    ``no_factor`` is the chance-level null (its true MVE Sharpe ratios are
    printed as :data:`NO_FACTOR_TEXT`, and a Sharpe that fails the absolute
    threshold but sits within two standard errors of zero is said to);
    ``topic_null`` is report-only and its paragraph explains the mechanism
    (module docstring), states the expectation and compares the observed
    numbers with the baseline rows of the same harness run when present.
    """
    runs = result.per_run[result.per_run["scenario"] == scenario]
    ok_runs = runs[runs["error"].fillna("") == ""] if "error" in runs else runs
    n = len(runs)
    thr = _thresholds_of(result, scenario)
    checks = scenario_checks(scenario)
    key = _canonical_scenario(scenario)
    seeds = f"{n} seed{'s' if n != 1 else ''}"
    lines: list[str] = []
    if key == "no_factor":
        intro = (
            f"**{scenario}** ({seeds}). The chance-level null: no common factor structure in returns (every loading "
            "is zero, returns are pure idiosyncratic noise), so the kernel covariances carry no information. "
            "Expected: selection at chance, an OOS Sharpe within two standard errors of zero, and a selected set "
            "that is unstable across refits (the instruments are pure noise, so this is the one scenario where the "
            "stability diagnostic is low). The true-factor MVE portfolio is not spanned by returns here, so the "
            f"true MVE Sharpe ratios are {NO_FACTOR_TEXT}, the OOS ratio is undefined (nan) and the instrument R2 "
            "on the (zero) true loadings is zero by construction."
        )
    elif key == "topic_null":
        intro = (
            f"**{scenario}** ({seeds}). Report-only: no pass/fail check applies. No topic carries information "
            "(A = 0) but returns keep their priced factor structure. The kernel covariance of any noise topic l "
            "with asset i is beta_i' G_{t,l} plus idiosyncratic noise, where G_{t,l} = sum_tau w_tau f_tau z_{l,tau} "
            "is a common K-vector that is non-zero at order 1/sqrt(n_eff) and persists over t because the kernel "
            "half-life is 69 months. With L noise topics the cross-section of instruments spans beta, IPCA recovers "
            "beta from pure noise topics, and the factor portfolios load on the true factors and earn the premium. "
            "Expected: selection above chance, a positive but degraded OOS Sharpe relative to baseline, an instrument "
            "R2 for noise topics well above the chance level K/N, and a selected set that is stable across refits, "
            "as in baseline: the spurious instruments G_{t,l} persist with the 69-month kernel half-life, so the same "
            "noise topics are re-selected at every refit. Selection stability is therefore a diagnostic of estimation "
            "noise (low only under no_factor), not evidence that narratives carry information. This is also why an "
            "OOS Sharpe alone cannot certify that narratives carry information (selection above chance does not "
            "either: this scenario selects far above chance with A = 0). The signal-quality evidence is relative "
            "(DESIGN.md Part F point 2): the placebo test (real narratives must beat variance-matched noise), "
            "pricing errors, and this scenario's numbers next to the baseline's."
        )
    elif key == "weak":
        intro = (
            f"**{scenario}** ({seeds}). Expected: graceful degradation - recall falls (threshold relaxed to "
            f"{thr.selection_recall_min:.2f}, strong-half recall to {thr.selection_recall_strong_min:.2f}), precision "
            f"stays, no placebo selected, subspace cosine threshold {thr.gamma_subspace_cos_min:.2f}."
        )
    elif key == "softmax":
        intro = (
            f"**{scenario}** ({seeds}). Expected: the simplex non-linearity of the "
            "attention mapping does not break selection or recovery (baseline expectations apply)."
        )
    else:
        intro = (
            f"**{scenario}** ({seeds}). Expected: recall of the strong half of the relevant topics near 1 (overall "
            "recall well below 1: the relevant rows of A are standard-normal draws, so the weak relevant topics are "
            "legitimately left out), precision high, the implied betas c Gamma recovered (first canonical correlation "
            "> 0.95), no placebo selected, the Gamma subspace on the selected rows, factors, states and impact vector "
            "recovered, the OOS Sharpe about 0.5-0.9 of the true MVE's, systematic R2 recovered above 0.7."
        )
    lines.append(intro)
    obs: list[str] = []
    failed: list[str] = []
    for check in checks:
        metric, op, field = _CHECK_TABLE[check]
        n_pass = int(np.nansum(pd.to_numeric(runs[f"pass_{check}"], errors="coerce"))) if f"pass_{check}" in runs else 0
        bound = getattr(thr, field)
        obs.append(f"{metric} {_span(ok_runs, metric)} [{op} {_fmt(bound, 2)}: {n_pass}/{n} pass]")
        if n_pass < n:
            failed.append(f"{check} (failed {n - n_pass}/{n} seed{'s' if n != 1 else ''})")
    if key == "topic_null":
        obs = [
            f"{_span(ok_runs, 'n_selected')} narratives selected on average, {_span(ok_runs, 'null_selection_lift', 2)} x "
            f"the chance level (recall {_span(ok_runs, 'selection_recall')} of the nominal relevant set, placebo selected "
            f"{_span(ok_runs, 'placebo_selected')})",
            f"OOS Sharpe {_sharpe_with_se(ok_runs)} vs the true MVE's realised {_span(ok_runs, 'oos_sharpe_true_mve')}",
            _instrument_text(ok_runs),
        ]
    elif key == "no_factor":
        obs.append(f"OOS Sharpe {_sharpe_with_se(ok_runs)} (true MVE {NO_FACTOR_TEXT})")
        obs.append(_instrument_text(ok_runs))
    if "sharpe_mve_true" in ok_runs and len(ok_runs):
        structure = _has_factor_structure(result, scenario)
        true_is = _true_sharpe_cell("sharpe_mve_true", _fmt(_col_mean(ok_runs, "sharpe_mve_true")), structure)
        extra = (
            f"in-sample MVE Sharpe {_fmt(_col_mean(ok_runs, 'mve_sharpe_is'))} vs true {true_is}; total R2 "
            f"{_fmt(_col_mean(ok_runs, 'total_r2'))} vs population systematic R2 "
            f"{_fmt(_col_mean(ok_runs, 'systematic_r2_true'))}; "
            f"{_fmt(_col_mean(ok_runs, 'n_selected'))} narratives selected on average."
        )
        lines.append("Observed: " + "; ".join(obs) + ". Context: " + extra)
    else:
        lines.append("Observed: " + "; ".join(obs) + ".")
    if key == "no_factor":
        notes = _null_sharpe_within_se(ok_runs, thr)
        if notes:
            lines.append("Small-sample reading of null_oos_sharpe_abs - " + "; ".join(notes) + ".")
    if key in ("topic_null", "no_factor"):
        base = _baseline_rows(result, scenario)
        if len(base):
            lines.append(
                f"Baseline rows of this run for comparison ({len(base)} seed{'s' if len(base) != 1 else ''}): "
                f"{_span(base, 'n_selected')} selected, OOS Sharpe {_span(base, 'oos_sharpe')} "
                f"(true MVE {_span(base, 'oos_sharpe_true_mve')}); {_instrument_text(base)}."
            )
    n_err = int((runs["error"].fillna("") != "").sum()) if "error" in runs else 0
    if n_err:
        lines.append(f"{n_err} run(s) of this scenario raised an error (see the errors section).")
    if not checks:
        lines.append("Report-only scenario: no pass/fail checks apply.")
    elif failed:
        lines.append("Checks failed: " + ", ".join(failed) + ".")
    else:
        lines.append("All checks passed.")
    return "\n".join(lines)


def _render_report(result: HarnessResult) -> str:
    meta = result.meta if isinstance(result.meta, dict) else {}
    scenarios = _scenario_order(result)
    per_run = result.per_run
    thr_base = result.thresholds if isinstance(result.thresholds, HarnessThresholds) else HarnessThresholds()
    now = datetime.now()
    parts: list[str] = []
    parts.append("# Simulation harness report")
    parts.append("")
    n_runs = int(meta.get("n_runs", len(per_run)))
    n_failed = int(meta.get("n_failed", 0))
    parts.append(
        f"Generated {now.strftime('%Y-%m-%d %H:%M')} by narrative-ipca {meta.get('package_version', __version__)}; "
        f"{n_runs} run(s) ({len(scenarios)} scenario(s) x {len(meta.get('seeds', []))} seed(s)), "
        f"{n_failed} failed, {_fmt_seconds(meta.get('elapsed_seconds', float('nan')))} s wall clock. "
        f"Design: DESIGN.md Part E; thresholds are regression-test signals (D40), not statistical tests."
    )
    parts.append("")

    # ---- setup ----
    parts.append("## Setup")
    parts.append("")
    parts.append(f"- Scenarios: {', '.join(scenarios)}; seeds: {meta.get('seeds', [])}; sizes: {meta.get('sizes', '?')}.")
    backend = str(meta.get("solver_backend") or active_backend())
    parts.append(
        f"- Solver backend: {backend} ("
        + ("numba JIT group-lasso kernel" if backend == "numba" else "pure-numpy reference group-lasso kernel; numba not active")
        + "; D46)."
    )
    sims = meta.get("simulation", {})
    if sims:
        sim_rows = []
        for s in scenarios:
            d = sims.get(s, {})
            sim_rows.append([
                s, d.get("n_assets", ""), d.get("n_topics", ""), d.get("n_relevant", ""), d.get("n_placebo", ""),
                d.get("K", ""), d.get("n_years", ""), d.get("period", ""), _fmt(d.get("signal_strength", ""), 2),
                _fmt(d.get("mve_sharpe_annual", ""), 2), d.get("attention_model", ""),
                _fmt(d.get("unbalanced_fraction", ""), 2), _fmt(d.get("missing_day_fraction", ""), 3),
            ])
        parts.append("")
        parts.append(_md_table(
            ["scenario", "assets", "topics", "relevant", "placebo", "K", "years", "period", "signal", "true SR",
             "attention", "unbalanced", "missing"],
            sim_rows,
        ))
    pc = meta.get("pipeline_config", {})
    if pc:
        est, tun, oos_c, cov, ev = pc.get("estimation", {}), pc.get("tuning", {}), pc.get("oos", {}), pc.get("covariance", {}), pc.get("evaluation", {})
        grid = est.get("lam_grid", {})
        parts.append("")
        parts.append(
            f"- Pipeline config `{meta.get('pipeline_name', '')}`: K = {est.get('K')}, criterion = {tun.get('criterion')}, "
            f"lambda grid {grid.get('n_lambdas')} points at ratio {grid.get('ratio')}, burn-in {cov.get('burn_in_periods')} periods, "
            f"xi = {cov.get('xi')}, shock window {pc.get('shocks', {}).get('window')}; OOS fraction {oos_c.get('oos_fraction')}, "
            f"refit every {oos_c.get('refit_every')}, min train {oos_c.get('min_train_periods')}, retune = {oos_c.get('retune_lambda')}; "
            f"annualisation {ev.get('annualization')}."
        )
    parts.append("")
    parts.append("Thresholds (per scenario after the relaxations of `SCENARIO_THRESHOLD_OVERRIDES`):")
    parts.append("")
    thr_rows = []
    for check, metric, op, field in CHECKS:
        cells = [check, f"{metric} {op}"]
        for s in scenarios:
            cells.append(_fmt(getattr(_thresholds_of(result, s), field), 2) if check in scenario_checks(s) else "n/a")
        thr_rows.append(cells)
    parts.append(_md_table(["check", "metric", *scenarios], thr_rows))
    parts.append("")

    # ---- summary ----
    parts.append("## Summary per scenario (mean ± std over seeds)")
    parts.append("")
    summary = result.summary
    structure = {s: _has_factor_structure(result, s) for s in scenarios}
    sum_rows = []
    for metric in METRICS:
        if (metric, "mean") not in summary.columns:
            continue
        cells = [metric]
        for s in scenarios:
            if s in summary.index:
                text = _mean_std(summary.loc[s, (metric, "mean")], summary.loc[s, (metric, "std")])
            else:
                text = "nan"
            cells.append(_true_sharpe_cell(metric, text, structure[s]))
        sum_rows.append(cells)
    parts.append(_md_table(["metric", *scenarios], sum_rows))
    parts.append("")
    parts.append(
        "`selection_recall_strong` is recall over the relevant topics whose row norm ||A_l|| is at or above the "
        "median of the relevant rows (`selection_recall_weak`: the others). `beta_canonical_corr` is the first "
        "canonical correlation between the implied loadings c Gamma_hat and the true betas across the assets of a "
        "return period, averaged over sampled periods (the object IPCA identifies; invariant to the rotation and to "
        "the selected set). `gamma_subspace_cos` and `impact_spearman` are computed over the relevant topics that "
        "were selected: non-selected rows of Gamma_hat are exactly zero, so the all-relevant-rows values "
        "(`*_all_relevant`, reported only) measure recall rather than loading recovery. With n selected relevant "
        "rows and n < 2K, the 2K - n largest principal-angle cosines are one by dimension counting (any full-rank "
        "K x K block scores one on every angle), so `gamma_subspace_cos` averages the min(K, n - K) smallest cosines "
        "and is nan with n <= K."
    )
    parts.append("")
    if not all(structure.values()):
        parts.append(
            f"`{NO_FACTOR_TEXT}`: the scenario's returns have no common factor structure, so the true MVE portfolio "
            "is not spanned by returns and its Sharpe ratio is not attainable by any estimator."
        )
        parts.append("")

    # ---- instrument informativeness ----
    parts.append("## Instrument informativeness per scenario (mean ± std over seeds)")
    parts.append("")
    parts.append(
        "Cross-sectional R2 of the covariance instrument `cov[t, :, l]` on `[1, beta_true]` "
        f"(every {INSTRUMENT_R2_EVERY}th covariance period, at least {INSTRUMENT_R2_MIN_PERIODS}), averaged over the "
        "topics of each kind; `chance` = K / N is the expected R2 of K random regressors. Under `topic_null` the noise "
        "topics' R2 sits well above chance: their instruments span the true loadings through the persistent common "
        "vector G_{t,l}, which is why the estimator earns a positive OOS Sharpe there. `stability` is the mean Jaccard "
        "similarity of the selected sets of consecutive OOS refits; it is expected to be high under `baseline` and "
        "`topic_null` alike (the spurious instruments G_{t,l} persist with the 69-month kernel half-life) and low only "
        "under `no_factor`, so it is a diagnostic of estimation noise, not evidence that narratives carry information."
    )
    parts.append("")
    instr_rows = []
    instr_cols = [
        ("instrument_beta_r2_relevant", "relevant R2"), ("instrument_beta_r2_noise", "noise R2"),
        ("instrument_beta_r2_placebo", "placebo R2"), ("instrument_beta_r2_chance", "chance"),
        ("oos_selection_stability", "stability"),
    ]
    for s_name in scenarios:
        cells = [s_name]
        for metric, _label in instr_cols:
            if s_name in summary.index and (metric, "mean") in summary.columns:
                cells.append(_mean_std(summary.loc[s_name, (metric, "mean")], summary.loc[s_name, (metric, "std")]))
            else:
                cells.append("nan")
        instr_rows.append(cells)
    parts.append(_md_table(["scenario", *[label for _, label in instr_cols]], instr_rows))
    parts.append("")

    # ---- pass / fail ----
    parts.append("## Pass / fail (seeds passing / seeds run)")
    parts.append("")
    pf_rows = []
    counts = {s: int((per_run["scenario"] == s).sum()) for s in scenarios}
    for check, metric, op, field in CHECKS:
        cells = [check]
        for s in scenarios:
            if check not in scenario_checks(s):
                cells.append("n/a")
                continue
            col = f"pass_{check}"
            n_pass = int(np.nansum(pd.to_numeric(per_run.loc[per_run["scenario"] == s, col], errors="coerce"))) if col in per_run else 0
            cells.append(f"{n_pass}/{counts[s]} ({_threshold_text(_thresholds_of(result, s), check)})")
        pf_rows.append(cells)
    all_cells = ["all applicable checks"]
    for s in scenarios:
        n_all = int(per_run.loc[per_run["scenario"] == s, "all_passed"].astype(bool).sum()) if "all_passed" in per_run else 0
        all_cells.append(f"{n_all}/{counts[s]}" + ("" if scenario_checks(s) else " (report-only, no checks)"))
    pf_rows.append(all_cells)
    parts.append(_md_table(["check", *scenarios], pf_rows))
    parts.append("")

    # ---- per run ----
    parts.append("## Per-run table")
    parts.append("")
    cols = [
        ("scenario", "scenario"), ("seed", "seed"), ("selection_recall", "recall"), ("selection_recall_strong", "recall strong"),
        ("selection_precision", "precision"), ("n_selected", "selected"), ("placebo_selected", "placebo"),
        ("beta_canonical_corr", "beta cc"), ("gamma_subspace_cos", "gamma cos"),
        ("factor_canonical_corr", "factor cc"), ("state_canonical_corr", "state cc"), ("impact_spearman", "impact rho"),
        ("mve_sharpe_is", "IS SR"), ("oos_sharpe", "OOS SR"), ("oos_sharpe_true_mve", "true OOS SR"),
        ("oos_sharpe_ratio_to_true", "OOS ratio"), ("systematic_r2_recovered", "sys R2"), ("total_r2", "total R2"),
        ("null_selection_lift", "lift"), ("instrument_beta_r2_relevant", "rel R2"), ("instrument_beta_r2_noise", "noise R2"),
        ("instrument_beta_r2_chance", "chance"), ("oos_selection_stability", "stability"),
        ("all_passed", "all passed"), ("runtime_seconds", "seconds"),
    ]
    run_rows = []
    for _, r in per_run.iterrows():
        cells = []
        row_structure = _num(r.get("factor_structure", float("nan")))
        has_structure = bool(row_structure != 0.0) if np.isfinite(row_structure) else structure.get(str(r.get("scenario", "")), True)
        for key, _label in cols:
            v = r.get(key, "")
            if key in ("scenario", "seed"):
                cells.append(str(v))
            elif key == "all_passed":
                cells.append(("yes" if bool(v) else "no") if scenario_checks(str(r.get("scenario", ""))) else "n/a")
            elif key == "runtime_seconds":
                cells.append(_fmt_seconds(v))
            else:
                cells.append(_true_sharpe_cell(key, _fmt(v), has_structure))
        if str(r.get("error", "") or ""):
            cells[-2] = "error"
        run_rows.append(cells)
    parts.append(_md_table([label for _, label in cols], run_rows))
    parts.append("")

    # ---- timings ----
    parts.append("## Timings")
    parts.append("")
    t_rows = []
    for s in scenarios:
        rt = pd.to_numeric(per_run.loc[per_run["scenario"] == s, "runtime_seconds"], errors="coerce")
        t_rows.append([s, counts[s], _fmt_seconds(rt.mean()), _fmt_seconds(rt.min()), _fmt_seconds(rt.max()), _fmt_seconds(rt.sum())])
    parts.append(_md_table(["scenario", "runs", "mean s", "min s", "max s", "total s"], t_rows))
    parts.append("")
    parts.append(f"Harness wall clock: {_fmt_seconds(meta.get('elapsed_seconds', float('nan')))} s (pipeline seconds are per run, simulation excluded).")
    parts.append("")

    # ---- expected vs observed ----
    parts.append("## Expected vs observed")
    parts.append("")
    for s in scenarios:
        parts.append(_expected_vs_observed(result, s))
        parts.append("")

    # ---- errors ----
    errors = meta.get("errors", {})
    if errors:
        parts.append("## Errors")
        parts.append("")
        for key, msg in errors.items():
            parts.append(f"- `{key}`: {msg}")
        parts.append("")

    # ---- artefacts ----
    parts.append("## Artefacts")
    parts.append("")
    parts.append(f"- Directory: `{meta.get('artefact_dir', '')}` (per run: `<scenario>_seed<k>_metrics.json`, `_lambda_path.csv`, `_gamma_norms.csv`).")
    tables = meta.get("tables", {})
    if tables:
        parts.append("- Tables: " + ", ".join(f"`{Path(p).name}`" for p in tables.values()) + f" in `{meta.get('output_dir', '')}`.")
    parts.append("")
    parts.append("Metric definitions: `narrative_ipca.harness.compare_to_truth` (DESIGN.md Part E). "
                 f"Base thresholds: {', '.join(f'{f.name}={getattr(thr_base, f.name)}' for f in fields(thr_base))}.")
    parts.append("")
    return "\n".join(parts)


def write_report(result: HarnessResult, out_dir: str | Path) -> str:
    """Write the markdown report ``harness_<YYYY-MM-DD>.md`` into ``out_dir`` and return its path.

    Sections: setup (scenarios, seeds, sizes, solver backend, pipeline
    config, thresholds), summary table per scenario (mean ± std of every
    Part E metric; the true MVE Sharpe ratios of a scenario without factor
    structure read :data:`NO_FACTOR_TEXT`), the instrument-informativeness
    table (instrument R2 on the true loadings per topic kind vs chance, OOS
    selection stability), pass/fail table (``n/a`` for checks that do not
    apply, e.g. every check of the report-only ``topic_null``), per-run
    table, timings, an 'expected vs observed' paragraph per scenario
    generated from the numbers (stating plainly which checks failed; for
    ``topic_null`` the mechanism, the expectation and a comparison with the
    baseline rows of the same run; for ``no_factor`` a Sharpe that fails the
    absolute threshold but sits within two standard errors of zero is said
    to), errors (if any) and the artefact locations. An existing report of
    the same day is not overwritten: a ``_2``, ``_3``, ... suffix is added.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    path = out / f"harness_{stamp}.md"
    n = 2
    while path.exists():
        path = out / f"harness_{stamp}_{n}.md"
        n += 1
    path.write_text(_render_report(result), encoding="utf-8")
    logger.info("write_report: %s", path)
    return str(path)
