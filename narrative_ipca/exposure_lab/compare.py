"""Like-for-like comparison of topic-sensitivity estimation methods (DESIGN.md G.7, G.8, G.15; D52, D65, D74, D88).

Every method delivers the same object, a :class:`~narrative_ipca.exposure_lab.types.DirectFit`
with the same observed shocks and the same training scales: exposures
``B_hat`` (topics x assets) in standardised units, with returns standardised by
their training standard deviation and shocks by the training standard deviation
of ``z``. The forecast-window evaluation (:func:`.evaluate.evaluate_window`),
the window sweep (:func:`.evaluate.window_sweep`) and the recovery metrics
(:func:`.evaluate.recovery_metrics`) therefore apply unchanged, and nothing is
fitted inside the forecast window (D65).

The direct methods are fitted on the training pairs only. BKS enters in two
variants that differ only in the data its panel is built from
(``BKSLabConfig.history``, D88):

* ``bks_implied`` (full history) is not fully like-for-like: its ``Gamma``,
  the shock covariance ``Sigma_z`` and the scales use the training window,
  but its instruments are kernel covariances that weigh the whole history
  before the cut-off (half-life ``BKSLabConfig.half_life_months``). On the
  dashboard defaults about 92% of that kernel weight lies before the
  training start (:func:`.bks.kernel_history_share`,
  ``meta["kernel_share_before_train"]``).
* ``bks_implied_train`` (training window only) is the like-for-like variant:
  its panel reads returns from the training start and attention from ``w``
  weekdays before it, so it sees the data the direct methods see.

Both can be in one comparison: :func:`method_config` gives each its own BKS
configuration (the sidebar's BKS settings with the history set).

Methods (:data:`METHODS`):

1. ``elastic_net``, ``ridge``, ``ols``: the direct regression of G.7.1
   (:func:`.direct.fit_direct`). The method selected in the sidebar keeps its
   penalty settings; the other direct methods use the
   :class:`~narrative_ipca.exposure_lab.config.DirectConfig` defaults with the
   same selection threshold (:func:`method_config`).
2. ``bks_implied`` and ``bks_implied_train``: the topic exposures that the
   BKS training fit implies (:func:`.bks.implied_exposures`, BKS Eq. 5), with
   the full or the training-window covariance history. BKS identifies the
   assets' factor betas, not their split across topics (D52): the implied
   exposures keep only the part of the assets' topic covariances in the
   ``K`` directions BKS fitted to price weekly returns (little of the topic
   signal on the lab data; DESIGN.md G.15.1), and they depend on which topics
   the sparse fit kept.
3. ``oracle``: the true exposures ``B_true`` with the same training scales
   (D74): the reference every estimator is scored against, not an estimator.

Summary metrics per method (:class:`ComparisonResult`):

* recovery of ``B_true`` over all topic-asset pairs (G.8 point 5):
  ``n_selected``, ``coverage``, ``sign_agreement``, ``mcc``, ``spearman``,
  ``rmse``;
* the forecast window: ``r2_median_window`` (median over assets of the
  per-asset uncentered OOS R2) and ``r2_pooled_window`` (one R2 over all
  valid asset-days, in return units, as in the window sweep);
* the window sweep (G.8 point 6): ``r2_median_all_windows`` (median over the
  sweep windows of the cross-asset median R2) and
  ``share_windows_above_oracle`` (share of sweep windows where the method's
  cross-asset median R2 exceeds the oracle's; ``NaN`` for the oracle);
* ``fit_seconds``, ``available`` and ``note``.

Validity boundaries
-------------------
* The oracle line of each sweep uses that method's training return scale
  (D74). All methods here compute the scale on the same training pairs, so
  the oracle line is the same for every method; ``meta["same_training_scales"]``
  records whether this held.
* The BKS-implied exposures need a BKS fit on the same training window and
  covariance history; the comparison does not start one
  (:meth:`.session.LabSession.comparison` lists the method as unavailable
  instead).
* Short forecast windows (one week, five days) give noisy per-window R2; the
  sweep medians are the more stable figures.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .bks import IMPLIED_METHOD, IMPLIED_METHODS, IMPLIED_NOTE, IMPLIED_TRAIN_METHOD
from .config import DirectConfig, LabConfig, WindowConfig
from .evaluate import evaluate_window, median_finite, window_sweep
from .types import DirectFit, ObservedShocks, SimData, SimTruth, WindowEval

logger = logging.getLogger(__name__)

__all__ = [
    "METHODS",
    "DIRECT_METHODS",
    "BKS_METHODS",
    "BKS_HISTORY",
    "METHOD_LABELS",
    "SUMMARY_COLUMNS",
    "ORACLE_NOTE",
    "ComparisonResult",
    "method_config",
    "method_label",
    "compare_methods",
]

#: Methods of the comparison, in display order, the oracle last. The chart colour slots follow this
#: order (the oracle is drawn in ink and takes no slot), so a new method is added before the oracle
#: and after the existing ones: ``bks_implied_train`` (D88) keeps every earlier method's colour.
METHODS: tuple[str, ...] = ("elastic_net", "ridge", "ols", IMPLIED_METHOD, IMPLIED_TRAIN_METHOD, "oracle")

#: Methods fitted by :func:`.direct.fit_direct` (the oracle included).
DIRECT_METHODS: tuple[str, ...] = ("elastic_net", "ridge", "ols", "oracle")

#: BKS-implied method -> the covariance history of its BKS fit (``BKSLabConfig.history``, D88).
BKS_HISTORY: dict[str, str] = {m: h for h, m in IMPLIED_METHODS.items()}

#: The BKS-implied methods, in display order.
BKS_METHODS: tuple[str, ...] = tuple(m for m in METHODS if m in BKS_HISTORY)

#: Display labels of :data:`METHODS` at their default settings.
METHOD_LABELS: dict[str, str] = {
    "elastic_net": "Elastic net",
    "ridge": "Ridge (GCV)",
    "ols": "OLS",
    IMPLIED_METHOD: "BKS-implied (full history)",
    IMPLIED_TRAIN_METHOD: "BKS-implied (training window)",
    "oracle": "Oracle (true sensitivities)",
}

#: Columns of :attr:`ComparisonResult.summary`.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "label",
    "n_selected",
    "coverage",
    "sign_agreement",
    "mcc",
    "spearman",
    "rmse",
    "r2_median_window",
    "r2_pooled_window",
    "r2_median_all_windows",
    "share_windows_above_oracle",
    "fit_seconds",
    "available",
    "note",
)

#: Note on the oracle row.
ORACLE_NOTE = "Reference, not an estimator: the true sensitivities with the same training scales."

_RECOVERY_COLUMNS = ("n_selected", "coverage", "sign_agreement", "mcc", "spearman", "rmse")


@dataclass
class ComparisonResult:
    """Estimation methods scored on the same training window and forecast window (G.7, G.8).

    Attributes
    ----------
    summary:
        Indexed by ``method`` (:data:`METHODS` order, then any other method
        given) with the columns of :data:`SUMMARY_COLUMNS`; unavailable
        methods have ``available = False``, the reason in ``note`` and ``NaN``
        metrics.
    r2:
        Assets x available methods: per-asset OOS R2 in the forecast window.
    r2_sweep:
        Long frame with columns ``start``, ``end``, ``method``, ``median_r2``:
        the cross-asset median OOS R2 of every sweep window per available
        method.
    fits:
        Available method -> the :class:`DirectFit` scored.
    evals:
        Available method -> its :class:`WindowEval` in the forecast window.
    meta:
        ``methods`` (row order), ``unavailable`` (method -> reason),
        ``n_windows``, ``sweep_first_day``, ``sweep_last_day`` (``NaT``
        without a complete window), ``same_training_scales``, ``timings``.
    """

    summary: pd.DataFrame
    r2: pd.DataFrame
    r2_sweep: pd.DataFrame
    fits: dict[str, DirectFit]
    evals: dict[str, WindowEval]
    meta: dict[str, Any] = field(default_factory=dict)


def method_config(cfg: LabConfig, method: str) -> LabConfig:
    """The lab configuration that fits ``method`` (G.7).

    * The direct method selected in ``cfg.direct`` keeps every setting (``cfg``
      is returned as is, so its cached direct fit is reused).
    * Another direct method (``elastic_net``, ``ridge``, ``ols``, ``oracle``)
      gets the :class:`DirectConfig` defaults with the same ``select_tau``.
    * ``bks_implied`` and ``bks_implied_train`` use ``cfg.bks`` with the
      history set to ``"full"`` or ``"training"`` (D88); ``cfg`` is returned
      as is when it already has that history.

    Raises
    ------
    ValueError
        For a method not in :data:`METHODS`.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; known: {list(METHODS)}")
    if method in BKS_HISTORY:
        history = BKS_HISTORY[method]
        return cfg if cfg.bks.history == history else replace(cfg, bks=replace(cfg.bks, history=history))
    if cfg.direct.method == method:
        return cfg
    return replace(cfg, direct=DirectConfig(method=method, select_tau=float(cfg.direct.select_tau)))


def method_label(method: str, fit: DirectFit | None = None) -> str:
    """Display label of ``method``; names a non-default penalty rule of ``fit`` when given.

    Examples: ``"Elastic net"`` (universal penalty), ``"Elastic net (CV)"``,
    ``"Elastic net (fixed alpha)"``, ``"Ridge (GCV)"``, ``"Ridge (fixed lambda)"``.
    """
    label = METHOD_LABELS.get(method, method)
    rule = None if fit is None else fit.meta.get("alpha_rule")
    if method == "elastic_net" and rule == "cv":
        return "Elastic net (CV)"
    if method == "elastic_net" and rule == "fixed":
        return "Elastic net (fixed alpha)"
    if method == "ridge" and rule == "fixed":
        return "Ridge (fixed lambda)"
    return label


def compare_methods(
    sim: SimData,
    shocks: ObservedShocks,
    truth: SimTruth,
    window: WindowConfig,
    fits: dict[str, DirectFit],
    fit_seconds: dict[str, float] | None = None,
    unavailable: dict[str, str] | None = None,
) -> ComparisonResult:
    """Score each method's training fit in the forecast window and the window sweep (G.8).

    Parameters
    ----------
    sim, shocks, truth:
        The simulation, its observed shocks and the truth for ``shocks.window``;
        every fit must come from these shocks (same training pairs).
    window:
        Forecast window (``forecast_start``, ``forecast_weeks``).
    fits:
        Method -> :class:`DirectFit` (from :func:`.direct.fit_direct` or
        :func:`.bks.implied_exposures`).
    fit_seconds:
        Optional method -> fitting time in seconds; missing entries use
        ``fit.meta["timings"]["total"]`` when recorded.
    unavailable:
        Method -> reason, for methods without a fit (for example OLS refused
        when ``L >= n_train / 2``, or BKS not run yet). A method with a fit is
        scored even when it is also listed here.

    Returns
    -------
    ComparisonResult
        A method whose evaluation raises ``ValueError`` (for example a fit
        whose training data reach into the forecast window, D65) is listed as
        unavailable with the error text.
    """
    t0 = time.perf_counter()
    fit_seconds = dict(fit_seconds or {})
    unavailable = dict(unavailable or {})
    order = [m for m in METHODS if m in fits or m in unavailable]
    order += [m for m in list(fits) + list(unavailable) if m not in order]
    order = list(dict.fromkeys(order))

    rows: dict[str, dict[str, Any]] = {}
    r2_cols: dict[str, pd.Series] = {}
    sweeps: list[pd.DataFrame] = []
    used_fits: dict[str, DirectFit] = {}
    evals: dict[str, WindowEval] = {}
    reasons: dict[str, str] = {}
    first_day, last_day, n_windows = pd.NaT, pd.NaT, 0

    for m in order:
        fit = fits.get(m)
        if fit is None:
            reasons[m] = str(unavailable.get(m, "not fitted"))
            rows[m] = _empty_row(m, reasons[m])
            continue
        try:
            ev = evaluate_window(sim, shocks, fit, window, truth)
            sw = window_sweep(sim, shocks, fit, window, truth)
        except ValueError as exc:
            logger.warning("compare_methods: %s not scored: %s", m, exc)
            reasons[m] = str(exc)
            rows[m] = _empty_row(m, reasons[m])
            continue
        used_fits[m], evals[m] = fit, ev
        r2_cols[m] = ev.r2.rename(m)
        if len(sw):
            n_windows = max(n_windows, int(len(sw)))
            first_day = pd.Timestamp(sw["start"].iloc[0])
            last_day = pd.Timestamp(sw["end"].iloc[-1])
        sweeps.append(pd.DataFrame({
            "start": sw["start"].to_numpy(), "end": sw["end"].to_numpy(), "method": m,
            "median_r2": sw["median_r2"].to_numpy(dtype=float),
        }))
        rec = ev.recovery
        row: dict[str, Any] = {"label": method_label(m, fit)}
        row.update({c: float(rec.get(c, np.nan)) for c in _RECOVERY_COLUMNS})
        row["r2_median_window"] = median_finite(ev.r2)
        row["r2_pooled_window"] = _pooled_r2(ev)
        row["r2_median_all_windows"] = median_finite(sw["median_r2"]) if len(sw) else np.nan
        row["share_windows_above_oracle"] = np.nan if m == "oracle" else _share_above(sw)
        secs = fit_seconds.get(m)
        if secs is None:
            secs = fit.meta.get("timings", {}).get("total", np.nan) if isinstance(fit.meta, dict) else np.nan
        row["fit_seconds"] = float(secs)
        row["available"] = True
        row["note"] = _note(m, fit)
        rows[m] = row

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary = summary.reindex(columns=list(SUMMARY_COLUMNS))
    summary.index = pd.Index(list(rows), name="method")
    summary["available"] = summary["available"].astype(bool)
    summary["label"] = summary["label"].astype(str)
    summary["note"] = summary["note"].astype(str)
    for c in SUMMARY_COLUMNS:
        if c not in ("label", "available", "note"):
            summary[c] = pd.to_numeric(summary[c], errors="coerce").astype(float)

    if r2_cols:
        r2 = pd.concat(list(r2_cols.values()), axis=1)
    else:
        r2 = pd.DataFrame(index=pd.Index([str(a) for a in sim.market.returns.columns], name="asset_id"))
    r2.columns = pd.Index(list(r2_cols), name="method")
    if sweeps:
        r2_sweep = pd.concat(sweeps, ignore_index=True)
    else:
        r2_sweep = pd.DataFrame({
            "start": pd.Series(dtype="datetime64[ns]"), "end": pd.Series(dtype="datetime64[ns]"),
            "method": pd.Series(dtype=object), "median_r2": pd.Series(dtype=float),
        })

    same_scales = _same_scales(used_fits)
    if not same_scales:
        logger.warning("compare_methods: the fits' training return scales differ; the oracle lines differ by method")
    meta: dict[str, Any] = {
        "methods": list(rows),
        "unavailable": reasons,
        "n_windows": int(n_windows),
        "sweep_first_day": first_day,
        "sweep_last_day": last_day,
        "same_training_scales": same_scales,
        "forecast_start": pd.Timestamp(window.forecast_start),
        "forecast_end": pd.Timestamp(window.forecast_end),
        "timings": {"compare": time.perf_counter() - t0},
    }
    logger.info(
        "compare_methods: %d method(s) scored, %d unavailable, %d sweep window(s) (%.2fs)",
        len(used_fits), len(reasons), n_windows, meta["timings"]["compare"],
    )
    return ComparisonResult(summary=summary, r2=r2, r2_sweep=r2_sweep, fits=used_fits, evals=evals, meta=meta)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _empty_row(method: str, reason: str) -> dict[str, Any]:
    row: dict[str, Any] = {c: np.nan for c in SUMMARY_COLUMNS}
    row.update({"label": method_label(method), "available": False, "note": reason})
    return row


def _pooled_r2(ev: WindowEval) -> float:
    """One uncentered R2 over all valid asset-days of the window, in return units (as in the sweep)."""
    r = ev.realized_daily.to_numpy(dtype=float)
    f = ev.fitted.to_numpy(dtype=float)
    v = np.isfinite(r) & np.isfinite(f)
    sst = float(np.sum(np.where(v, r, 0.0) ** 2))
    if sst <= 0.0:
        return float("nan")
    return float(1.0 - np.sum(np.where(v, r - f, 0.0) ** 2) / sst)


def _share_above(sweep: pd.DataFrame) -> float:
    """Share of sweep windows where the method's median R2 exceeds the oracle's (both finite)."""
    if not len(sweep):
        return float("nan")
    a = sweep["median_r2"].to_numpy(dtype=float)
    o = sweep["median_r2_oracle"].to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(o)
    return float(np.mean(a[ok] > o[ok])) if ok.any() else float("nan")


def _note(method: str, fit: DirectFit) -> str:
    if method == "oracle":
        return ORACLE_NOTE
    if method in BKS_HISTORY:
        return str(fit.meta.get("caveat", IMPLIED_NOTE))
    meta = fit.meta if isinstance(fit.meta, dict) else {}
    skipped = meta.get("skipped_assets", [])
    parts: list[str] = []
    if skipped:
        parts.append(f"{len(skipped)} asset(s) with too few training days are not fitted (sensitivities 0).")
    low = int(meta.get("gcv_at_lower_edge", 0) or 0)
    if method == "ridge" and low:
        n_fitted = len(fit.B_hat.columns) - len(skipped)
        parts.append(
            f"GCV chose the smallest lambda of its grid for {low} of {n_fitted} assets, so their sensitivities "
            "are close to OLS."
        )
    return " ".join(parts)


def _same_scales(fits: dict[str, DirectFit]) -> bool:
    scales = [f.ret_scale for f in fits.values()]
    if len(scales) < 2:
        return True
    ref = scales[0]
    return all(
        s.index.equals(ref.index) and np.allclose(s.to_numpy(dtype=float), ref.to_numpy(dtype=float), rtol=1e-12, atol=0)
        for s in scales[1:]
    )
