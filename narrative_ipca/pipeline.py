"""Step 10: orchestration, artefact export and input loading (DESIGN.md Part A table; D42-D44).

:func:`run_pipeline` chains the nine estimation stages of Bybee, Kelly & Su
(2023) (BKS) exactly in the order of the DESIGN.md Part A table and returns a
:class:`~narrative_ipca.types.PipelineResult`:

1. :func:`narrative_ipca.data.align_inputs`            calendar alignment (Section 3.1)
2. :func:`narrative_ipca.shocks.attention_shocks`      ``z_tau = theta_tau - MA_w(theta)``
3. :func:`narrative_ipca.covariances.build_covariance_panel`  ``cov_{i,t}`` (Eq. 6)
4. :func:`narrative_ipca.panel.build_panel`            ``c_{i,t-1} -> r_{i,t}`` (Eq. 7)
5-6. :func:`narrative_ipca.tuning.tune`                Sparse IPCA (Eq. 8, 16) on the full
     sample along the ``lambda`` path, ``lambda*`` by the tuning criterion
7. :func:`narrative_ipca.wrapup.wrap_up`               ``A``, states, impact vectors (Eq. 1, 10-12)
8. :func:`narrative_ipca.oos.run_oos`                  expanding-window OOS factors (Section 4.2)
9. :func:`narrative_ipca.evaluation.evaluate_run`      R2, Sharpe, pricing tests, placebo test
   (:func:`narrative_ipca.evaluation.placebo_test`, App. C.2, when ``placebo_n > 0``)

Every stage is a pure function of its inputs and the config (D42); the
pipeline adds only bookkeeping: wall-clock timings per step, shapes, the
config hash and package version in ``result.meta``, and one ``logging.info``
line per step (D44). :func:`save_result` writes the artefacts as CSV/JSON
(parquet/npz for the bulky covariance panel on request) and returns a file
manifest; :func:`load_inputs` reads attention and return panels from
parquet or CSV.

Conventions
-----------
* Optional side products are guarded, core stages are not. The wrap-up and
  the placebo test are interpretation add-ons: if one of them fails (for
  instance an observable series that does not overlap the factor sample) a
  warning is logged, the error is recorded in ``result.meta`` and the run
  continues without that product. Failures of alignment, shocks,
  covariances, panel, tuning, OOS or evaluation propagate, because a result
  without them is not a result.
* Wrap-up rank deficiency (fewer than ``K`` narratives selected, D43) never
  raises: ``recover_A`` returns a flag, which is logged as a warning and
  carried in ``WrapUpResult.rank_deficient`` and the evaluation metrics.
* ``test_assets`` and ``observables`` are *period* excess-return series. They
  are re-stamped on the panel's period grid by period label (a month-end
  stamp pairs with the last trading day of that month); series that carry
  more than one row per period are read as daily and accumulated with
  ``DataConfig.return_aggregation`` first (D6). The observables double as
  the benchmark factor set of the evaluation report (BKS Table C.1
  correlations and a benchmark pricing test).
* ``progress(done, total, message)`` counts pipeline steps; the tuning and
  OOS loops report their own progress through the same callback with the
  current step index and a sub-message.
"""

from __future__ import annotations

import json
import logging
import platform
import time
import warnings
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import __version__
from . import covariances as _covariances
from . import data as _data
from . import evaluation as _evaluation
from . import oos as _oos
from . import panel as _panel
from . import shocks as _shocks
from . import tuning as _tuning
from . import wrapup as _wrapup
from .config import PipelineConfig, config_hash, config_to_dict
from .types import (
    AttentionData,
    EvaluationReport,
    OOSResult,
    PipelineResult,
    ReturnsData,
    ShockPanel,
    SparseIPCAResult,
    TuningResult,
    WrapUpResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "run_pipeline",
    "save_result",
    "load_inputs",
    "align_period_frame",
    "read_frame",
    "STEPS",
]

ProgressFn = Callable[[int, int, str], None]
"""``progress(done, total, message)`` callback; ``done``/``total`` count pipeline steps (D44)."""

STEPS: tuple[str, ...] = (
    "align",
    "shocks",
    "covariances",
    "panel",
    "tune",
    "wrapup",
    "oos",
    "placebo",
    "evaluate",
)
"""Pipeline steps in execution order; the keys of ``PipelineResult.timings`` (plus ``"total"``)."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
class _StepTimer:
    """Context manager recording the wall-clock seconds of one step into ``timings``."""

    def __init__(self, timings: dict[str, float], name: str) -> None:
        self.timings = timings
        self.name = name
        self.t0 = 0.0

    def __enter__(self) -> "_StepTimer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.timings[self.name] = time.perf_counter() - self.t0


def _resolve_period_alias(period: str) -> str:
    """The pandas *period* alias equivalent to ``period`` (``"ME"`` -> ``"M"``; ``"M"`` unchanged).

    Uses the candidate list of :func:`narrative_ipca.data.normalize_period_alias`
    and returns the first one ``pd.PeriodIndex`` accepts, so that the label
    stored in ``fit.meta["period"]`` works for ``DatetimeIndex.to_period``.
    """
    probe = pd.DatetimeIndex(["2000-01-03"])
    last: Exception | None = None
    for cand in _data.normalize_period_alias(period):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pd.PeriodIndex(probe, freq=cand)
                return cand
            except (ValueError, TypeError) as exc:
                last = exc
    raise ValueError(f"period {period!r} is not a valid pandas period alias") from last


def _period_labels(index: pd.DatetimeIndex, period: str) -> pd.PeriodIndex:
    """Period label of every timestamp of ``index`` (alias normalised as in ``data.py``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.PeriodIndex(pd.DatetimeIndex(index), freq=_resolve_period_alias(period))


def align_period_frame(
    frame: pd.DataFrame | pd.Series,
    periods: pd.DatetimeIndex,
    period: str,
    how: str = "sum",
    what: str = "series",
) -> pd.DataFrame:
    """Re-stamp a period series (or daily series) on the panel's period grid.

    Rows of ``frame`` are matched to ``periods`` (the last trading day of
    each estimation period, ``IPCAPanel.periods``) by *period label* under
    the alias ``period`` (D5), so that a month-end-stamped market factor
    pairs with the last-trading-day-stamped factors. When ``frame`` carries
    more than one row per period it is read as daily data and accumulated
    first with :func:`narrative_ipca.data.period_returns` (``how`` =
    ``DataConfig.return_aggregation``, D6). Periods absent from ``frame``
    are ``NaN``.

    Assumptions: the index is datetime-like (or convertible); values are
    excess returns in decimal units, so that accumulation is meaningful.

    Returns a ``(T, n_cols)`` DataFrame indexed exactly by ``periods``.
    """
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=frame.name if frame.name is not None else what)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame or Series")
    if frame.shape[1] == 0:
        raise ValueError(f"{what} has no columns")
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # pandas' "could not infer format" hint on odd indexes
                idx = pd.DatetimeIndex(pd.to_datetime(frame.index))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{what} must have a datetime-like index") from exc
        frame = frame.set_axis(idx, axis=0)
    frame = frame.astype(float).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    target = pd.DatetimeIndex(periods)
    labels = _period_labels(frame.index, period)
    if labels.has_duplicates:
        logger.info(
            "%s carries %d rows for %d %s periods; reading it as daily data and accumulating with %r",
            what, len(frame), labels.nunique(), period, how,
        )
        frame = _data.period_returns(frame, period, how)
        labels = _period_labels(frame.index, period)
    aligned = frame.set_axis(labels, axis=0).reindex(_period_labels(target, period))
    aligned.index = target
    n_found = int(aligned.notna().any(axis=1).sum())
    if n_found == 0:
        logger.warning("%s shares no period with the panel (%s .. %s)", what, target[0].date(), target[-1].date())
    elif n_found < len(target) // 2:
        logger.warning("%s covers only %d of %d panel periods", what, n_found, len(target))
    aligned.columns = [str(c) for c in aligned.columns]
    return aligned


def _restamp_observables(
    observables: dict[str, pd.Series] | None, periods: pd.DatetimeIndex, period: str, how: str
) -> tuple[dict[str, pd.Series], pd.DataFrame | None]:
    """Observables on the panel period grid: per-name Series (for the wrap-up) and one frame (benchmark factors)."""
    if not observables:
        return {}, None
    series: dict[str, pd.Series] = {}
    for name, s in observables.items():
        try:
            al = align_period_frame(s, periods, period, how, what=f"observable {name!r}")
        except (ValueError, TypeError) as exc:
            logger.warning("observable %r skipped: %s", name, exc)
            continue
        col = al.iloc[:, 0].rename(str(name))
        if col.notna().sum() == 0:
            logger.warning("observable %r has no value on the panel periods; skipped", name)
            continue
        series[str(name)] = col
    if not series:
        return {}, None
    frame = pd.DataFrame(series, index=pd.DatetimeIndex(periods))
    return series, frame


def _shapes_of_shocks(shock_panel: ShockPanel) -> dict[str, int]:
    z = shock_panel.z
    return {
        "n_days": int(z.shape[0]),
        "n_topics": int(z.shape[1]),
        "n_days_missing_shock": int(z.isna().any(axis=1).sum()),
    }


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    attention: AttentionData,
    returns: ReturnsData,
    cfg: PipelineConfig,
    test_assets: pd.DataFrame | None = None,
    observables: dict[str, pd.Series] | None = None,
    progress: ProgressFn | None = None,
) -> PipelineResult:
    """Run the BKS three-step procedure with tuning, wrap-up, OOS and evaluation (DESIGN.md Part A).

    Steps (equation references are to BKS):

    1. ``align``: :func:`narrative_ipca.data.align_inputs` puts ``theta_tau``
       and ``r_{i,tau}`` on the return calendar (Section 3.1, D3-D4).
    2. ``shocks``: ``z_tau = theta_tau - (1/w) sum_{j=1..w} theta_{tau-j}`` (D9).
    3. ``covariances``: ``cov_{i,t}`` by the kernel-weighted Eq. 6 (D11-D16);
       storage dtype ``cfg.data.dtype``.
    4. ``panel``: rows ``(i, t)`` pair ``c_{i,t-1} = [1, cov_{i,t-1}]`` with
       ``r_{i,t}`` (Eq. 7) and ``sigma^c_l`` is computed on the panel (D26).
    5-6. ``tune``: Eq. 8 along the ``lambda`` path on the full sample,
       ``lambda*`` (and ``K``) by ``cfg.tuning.criterion`` (D22, D27-D29);
       the chosen fit is canonical (D24) and carries ``meta["period"]``.
    7. ``wrapup`` (``cfg.run_wrapup``): ``A``, ``x_tau``, ``I_{z->x}``,
       ``I_{z->MVE}`` (Eq. 10-11), observable projections (footnote 17) and
       ``I_{w->MVE}`` (Eq. 12) when ``attention.phi`` is present.
    8. ``oos`` (``cfg.oos.enabled``): expanding-window OOS factors and MVE
       series (Section 4.2, D30-D33).
    9. ``placebo`` (``cfg.evaluation.placebo_n > 0``): App. C.2 on the shock
       panel (D36); ``evaluate``: the :class:`EvaluationReport` (D34-D35),
       with ``observables`` as benchmark factors.

    ``result.timings`` holds seconds per step (``0.0`` for skipped steps)
    and ``"total"``; ``result.meta`` holds shapes, the config hash, the
    package version, per-step status and any guarded-step error message.

    Assumptions: ``attention.levels`` and ``returns.returns`` satisfy the
    input contracts of :mod:`narrative_ipca.types`; ``test_assets`` and
    ``observables`` are excess returns (see the module docstring for the
    re-stamping rules).
    """
    if not isinstance(cfg, PipelineConfig):
        raise TypeError("cfg must be a PipelineConfig")
    started = datetime.now(timezone.utc)
    t_all = time.perf_counter()
    timings: dict[str, float] = {name: 0.0 for name in STEPS}
    status: dict[str, str] = {name: "skipped" for name in STEPS}
    n_steps = len(STEPS)
    meta: dict[str, Any] = {
        "package_version": __version__,
        "config_hash": config_hash(cfg),
        "config_name": cfg.name,
        "started_at": started.isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "steps": status,
        "shapes": {},
    }
    period_alias = _resolve_period_alias(cfg.data.period)
    meta["period_alias"] = period_alias

    def report(step: str, message: str) -> None:
        if progress is not None:
            progress(STEPS.index(step), n_steps, message)

    def sub_progress(step: str) -> ProgressFn:
        def cb(done: int, total: int, message: str) -> None:
            if progress is not None:
                progress(STEPS.index(step), n_steps, f"{step} [{done}/{total}] {message}")

        return cb

    # ---- 1. alignment -----------------------------------------------------
    report("align", "aligning attention and returns")
    with _StepTimer(timings, "align"):
        aligned = _data.align_inputs(attention, returns, cfg.data)
    status["align"] = "ok"
    meta["shapes"]["aligned"] = {
        "n_days": int(len(aligned.calendar)),
        "n_topics": int(aligned.attention.shape[1]),
        "n_assets": int(aligned.returns.shape[1]),
        "calendar_start": str(aligned.calendar[0].date()),
        "calendar_end": str(aligned.calendar[-1].date()),
    }
    meta["topic_labels"] = aligned.topic_labels
    meta["asset_meta"] = aligned.asset_meta
    logger.info(
        "step 1 align: %d days x %d topics x %d assets (%.2fs)",
        len(aligned.calendar), aligned.attention.shape[1], aligned.returns.shape[1], timings["align"],
    )

    # ---- 2. shocks --------------------------------------------------------
    report("shocks", "attention shocks")
    with _StepTimer(timings, "shocks"):
        shock_panel = _shocks.attention_shocks(aligned.attention, cfg.shocks)
    status["shocks"] = "ok"
    meta["shapes"]["shocks"] = _shapes_of_shocks(shock_panel)
    logger.info(
        "step 2 shocks: window=%d, %d days x %d topics, %d days with a missing shock (%.2fs)",
        shock_panel.window, *shock_panel.z.shape, meta["shapes"]["shocks"]["n_days_missing_shock"], timings["shocks"],
    )

    # ---- 3. covariances ---------------------------------------------------
    report("covariances", "kernel-weighted covariances")
    with _StepTimer(timings, "covariances"):
        cov = _covariances.build_covariance_panel(
            shock_panel, aligned.returns, cfg.covariance, cfg.data.period, dtype=cfg.data.dtype
        )
    status["covariances"] = "ok"
    T_cov, N_cov, L_cov = cov.shape
    n_valid = int(np.isfinite(np.asarray(cov.values)[:, :, 0]).sum()) if L_cov else 0
    meta["shapes"]["covariances"] = {"T": T_cov, "N": N_cov, "L": L_cov, "n_valid_asset_periods": n_valid}
    logger.info(
        "step 3 covariances: (T, N, L) = (%d, %d, %d), %d/%d asset-periods valid, dtype=%s (%.2fs)",
        T_cov, N_cov, L_cov, n_valid, T_cov * N_cov, cfg.data.dtype, timings["covariances"],
    )

    # ---- 4. panel ---------------------------------------------------------
    report("panel", "estimation panel")
    with _StepTimer(timings, "panel"):
        pnl = _panel.build_panel(cov, aligned.returns, cfg.data, cfg.covariance)
    status["panel"] = "ok"
    meta["shapes"]["panel"] = {"n_obs": pnl.n_obs, "T": pnl.T, "N": pnl.N, "p": pnl.p}
    logger.info(
        "step 4 panel: %d rows, T=%d periods (%s .. %s), N=%d assets, p=%d instruments (%.2fs)",
        pnl.n_obs, pnl.T, pnl.periods[0].date(), pnl.periods[-1].date(), pnl.N, pnl.p, timings["panel"],
    )

    # ---- 5-6. tuning on the full sample -----------------------------------
    report("tune", "sparse IPCA along the lambda path")
    with _StepTimer(timings, "tune"):
        tr: TuningResult = _tuning.tune(pnl, cfg.estimation, cfg.tuning, cfg.evaluation, progress=sub_progress("tune"))
    fit: SparseIPCAResult = replace(tr.fit, meta={**tr.fit.meta, "period": period_alias})
    tr = replace(tr, fit=fit)
    status["tune"] = "ok"
    meta["shapes"]["fit"] = {"K": fit.K, "p": int(fit.Gamma.shape[0]), "T": int(fit.F.shape[0])}
    logger.info(
        "step 5-6 tune: %s -> lambda*=%.4g K=%d, %d/%d narratives selected, IS Sharpe %.3f, total R2 %.4f, "
        "%d path points, converged=%s (%.2fs)",
        tr.criterion, tr.lam, tr.K, fit.n_selected, pnl.L, fit.mve_sharpe(cfg.evaluation.annualization, cfg.evaluation.rcond),
        fit.total_r2, len(tr.path), fit.converged, timings["tune"],
    )

    # ---- re-stamped observables / test assets ----------------------------
    obs_series, benchmark = _restamp_observables(observables, pnl.periods, cfg.data.period, cfg.data.return_aggregation)
    test_assets_aligned: pd.DataFrame | None = None
    if test_assets is not None:
        test_assets_aligned = align_period_frame(
            test_assets, pnl.periods, cfg.data.period, cfg.data.return_aggregation, what="test_assets"
        )

    # ---- 7. wrap-up -------------------------------------------------------
    wrap: WrapUpResult | None = None
    if cfg.run_wrapup:
        report("wrapup", "A, states and impact vectors")
        with _StepTimer(timings, "wrapup"):
            try:
                wrap = _wrapup.wrap_up(fit, shock_panel, cfg.evaluation, observables=obs_series or None, phi=aligned.phi)
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                logger.warning("step 7 wrapup failed and is skipped: %s", exc)
                meta["wrapup_error"] = f"{type(exc).__name__}: {exc}"
                status["wrapup"] = "failed"
        if wrap is not None:
            status["wrapup"] = "ok"
            if wrap.rank_deficient:
                logger.warning(
                    "step 7 wrapup: rank deficient (%d narratives selected for K=%d); A and the states are "
                    "minimum-norm solutions, continuing",
                    fit.n_selected, fit.K,
                )
            logger.info(
                "step 7 wrapup: A %s, states %s, %d observable(s), term vector=%s, rank_deficient=%s (%.2fs)",
                tuple(wrap.A.shape), tuple(wrap.states.shape), len(wrap.impact_z_to_obs),
                wrap.impact_w_to_mve is not None, wrap.rank_deficient, timings["wrapup"],
            )

    # ---- 8. out of sample -------------------------------------------------
    oos_res: OOSResult | None = None
    if cfg.oos.enabled:
        report("oos", "expanding-window out-of-sample")
        with _StepTimer(timings, "oos"):
            oos_res = _oos.run_oos(pnl, cfg, progress=sub_progress("oos"))
        status["oos"] = "ok"
        meta["shapes"]["oos"] = {
            "n_oos_periods": int(len(oos_res.mve)),
            "n_refits": int(len(oos_res.refit_periods)),
            "first_oos_period": str(oos_res.meta.get("first_oos_period", "")),
        }
        logger.info(
            "step 8 oos: %d OOS periods from %s, %d refit(s), realised MVE Sharpe %.3f (%.2fs)",
            len(oos_res.mve), oos_res.meta.get("first_oos_period"), len(oos_res.refit_periods), oos_res.sharpe, timings["oos"],
        )

    # ---- 9a. placebo test -------------------------------------------------
    placebo = None
    if cfg.evaluation.placebo_n > 0:
        report("placebo", f"placebo test with {cfg.evaluation.placebo_n} placebos")
        with _StepTimer(timings, "placebo"):
            try:
                placebo = _evaluation.placebo_test(
                    shock_panel, aligned.returns, cfg, fit, int(cfg.evaluation.placebo_n), int(cfg.evaluation.placebo_seed)
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("step 9 placebo test failed and is skipped: %s", exc)
                meta["placebo_error"] = f"{type(exc).__name__}: {exc}"
                status["placebo"] = "failed"
        if placebo is not None:
            status["placebo"] = "ok"
            logger.info(
                "step 9 placebo: %d/%d placebos selected at lambda*=%.4g, real set Jaccard %.3f (%.2fs)",
                placebo.n_placebo_selected, placebo.n_placebo, placebo.lam_star, placebo.jaccard_real, timings["placebo"],
            )

    # ---- 9b. evaluation ---------------------------------------------------
    report("evaluate", "evaluation report")
    with _StepTimer(timings, "evaluate"):
        report_: EvaluationReport = _evaluation.evaluate_run(
            pnl, tr, fit, oos_res, wrap, cfg,
            test_assets=test_assets_aligned, benchmark_factors=benchmark, placebo=placebo,
        )
    status["evaluate"] = "ok"
    timings["total"] = time.perf_counter() - t_all
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["n_observables"] = len(obs_series)
    meta["n_test_assets"] = 0 if test_assets_aligned is None else int(test_assets_aligned.shape[1])
    m = report_.metrics
    logger.info(
        "step 9 evaluate: total_r2=%.4f pred_r2=%.4f mve_sharpe_is=%.3f%s n_selected=%d (%.2fs); pipeline total %.2fs",
        m.get("total_r2", float("nan")), m.get("pred_r2", float("nan")), m.get("mve_sharpe_is", float("nan")),
        f" oos_sharpe={m['oos_sharpe']:.3f}" if "oos_sharpe" in m else "",
        int(m.get("n_selected", fit.n_selected)), timings["evaluate"], timings["total"],
    )
    if progress is not None:
        progress(n_steps, n_steps, "done")

    return PipelineResult(
        config=cfg,
        shocks=shock_panel,
        covariances=cov,
        panel=pnl,
        tuning=tr,
        fit=fit,
        wrapup=wrap,
        oos=oos_res,
        evaluation=report_,
        timings=timings,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------
def _json_default(o: Any) -> Any:
    """JSON fallback: numpy scalars/arrays, pandas objects, timestamps, dataclasses; ``str`` otherwise."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, pd.Timedelta):
        return str(o)
    if isinstance(o, pd.Series):
        return {str(k): v for k, v in o.items()}
    if isinstance(o, pd.DataFrame):
        return {str(k): {str(c): v for c, v in row.items()} for k, row in o.to_dict(orient="index").items()}
    if isinstance(o, (pd.Index,)):
        return [str(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(str(v) for v in o)
    if is_dataclass(o) and not isinstance(o, type):
        return {f.name: getattr(o, f.name) for f in fields(o)}
    return str(o)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _config_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config) and not isinstance(config, type):
        return config_to_dict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(f"result.config must be a PipelineConfig or a dict, got {type(config).__name__}")


def _write_table(path: Path, frame: pd.DataFrame | pd.Series, index_label: str | None = None) -> None:
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=frame.name if frame.name is not None else "value")
    frame.to_csv(path, index=True, index_label=index_label)


def _placebo_dict(pl: Any) -> dict[str, Any]:
    return {
        "n_placebo": int(pl.n_placebo),
        "n_placebo_selected": int(pl.n_placebo_selected),
        "n_real_selected": int(pl.n_real_selected),
        "lam_star": float(pl.lam_star),
        "jaccard_real": float(pl.jaccard_real),
        "real_selected_before": list(pl.real_selected_before),
        "real_selected_after": list(pl.real_selected_after),
        "lam_max_by_instrument": {str(k): v for k, v in pd.Series(pl.lam_max_by_instrument).items()},
    }


def _gamma_norm_path(tr: TuningResult, instrument_names: list[str]) -> pd.DataFrame:
    """One row per path point: ``K``, ``lam`` and ``||Gamma_l||`` per instrument (BKS Figure 2)."""
    rows = []
    for pt in tr.path:
        norms = np.asarray(pt.gamma_norms, dtype=float).ravel()
        if norms.shape[0] == len(instrument_names):
            names = instrument_names
        elif norms.shape[0] == len(instrument_names) - 1:
            names = instrument_names[1:]
        else:
            names = [f"g{i}" for i in range(norms.shape[0])]
        row: dict[str, Any] = {"K": int(pt.K), "lam": float(pt.lam)}
        row.update(dict(zip(names, norms)))
        rows.append(row)
    return pd.DataFrame(rows)


def save_result(result: PipelineResult, out_dir: str | Path) -> dict[str, str]:
    """Write the artefacts of a run into ``out_dir`` and return ``{key: path}`` (D42).

    Always written: ``config.json`` (loadable by ``config.load_config``),
    ``metrics.json``, ``gamma.csv`` (``Gamma``, rows = instruments),
    ``factors.csv`` (``f_t`` with a ``populated`` flag), ``lambda_path.csv``,
    ``gamma_norm_path.csv``, ``selected.csv``, ``shock_diagnostics.csv``,
    ``timings.json``, ``meta.json`` and ``manifest.json``.
    With a wrap-up: ``states.csv`` (``x_tau`` plus ``x_mve``),
    ``impact_z_to_mve.csv``, ``A.csv``, ``impact_z_to_x.csv``, and when
    present ``impact_w_to_mve.csv``, ``impact_z_to_obs.csv``,
    ``obs_projection.json``. With OOS: ``oos_factors.csv``, ``oos_mve.csv``,
    ``oos_history.csv`` (plus the selected/gamma-norm histories). With a
    placebo test: ``placebo.json``. With pricing tests: ``pricing_summary.csv``
    and ``pricing_<model>.csv``; with benchmark factors:
    ``factor_correlations.csv``. With ``cfg.save_panel``: ``covariances.npz``
    (the ``(T, N, L)`` array and its labels) and ``panel.parquet`` (long form;
    CSV when pyarrow is unavailable).

    Numbers go through ``pandas.to_csv`` (full float precision) and
    ``json.dumps`` with a fallback that converts numpy/pandas objects and
    otherwise stringifies (``NaN`` is written as the JSON extension ``NaN``).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    def put(key: str, filename: str) -> Path:
        path = out / filename
        manifest[key] = str(path)
        return path

    cfg_dict = _config_dict(result.config)
    _write_json(put("config", "config.json"), cfg_dict)
    _write_json(put("metrics", "metrics.json"), result.evaluation.metrics)

    fit = result.fit
    _write_table(put("gamma", "gamma.csv"), fit.gamma_frame(), index_label="instrument")
    factors = fit.factors_frame()
    populated = fit.populated if fit.populated is not None else np.any(np.asarray(fit.F) != 0.0, axis=1)
    factors["populated"] = np.asarray(populated, dtype=bool)
    _write_table(put("factors", "factors.csv"), factors, index_label="period")

    tr = result.tuning
    _write_table(put("lambda_path", "lambda_path.csv"), tr.path_frame().rename_axis("point"))
    _write_table(put("gamma_norm_path", "gamma_norm_path.csv"), _gamma_norm_path(tr, list(fit.instrument_names)).rename_axis("point"))

    selected = result.evaluation.tables.get("selected")
    if selected is None:
        selected = pd.DataFrame({"topic": fit.selected_topics})
    selected = selected.copy()
    labels = result.meta.get("topic_labels") if isinstance(result.meta, dict) else None
    if isinstance(labels, dict) and "topic" in selected.columns:
        selected["label"] = [labels.get(str(t), "") for t in selected["topic"]]
    selected.to_csv(put("selected", "selected.csv"), index=False)

    try:
        _write_table(put("shock_diagnostics", "shock_diagnostics.csv"), _shocks.shock_diagnostics(result.shocks))
    except Exception as exc:  # diagnostics only; never block the artefacts
        logger.warning("shock diagnostics not written: %s", exc)

    wrap = result.wrapup
    if wrap is not None:
        states = wrap.states.copy()
        states["x_mve"] = wrap.x_mve
        _write_table(put("states", "states.csv"), states, index_label="date")
        _write_table(put("impact_z_to_mve", "impact_z_to_mve.csv"), wrap.impact_z_to_mve.rename("impact_z_to_mve"), index_label="topic")
        _write_table(put("A", "A.csv"), wrap.A, index_label="topic")
        _write_table(put("impact_z_to_x", "impact_z_to_x.csv"), wrap.impact_z_to_x, index_label="topic")
        if wrap.impact_w_to_mve is not None:
            _write_table(put("impact_w_to_mve", "impact_w_to_mve.csv"), wrap.impact_w_to_mve, index_label="term")
        if wrap.impact_z_to_obs:
            obs = pd.DataFrame({name: s for name, s in wrap.impact_z_to_obs.items()})
            _write_table(put("impact_z_to_obs", "impact_z_to_obs.csv"), obs, index_label="topic")
        if wrap.obs_projection:
            _write_json(put("obs_projection", "obs_projection.json"), wrap.obs_projection)

    oos_res = result.oos
    if oos_res is not None:
        _write_table(put("oos_factors", "oos_factors.csv"), oos_res.factors, index_label="period")
        _write_table(put("oos_mve", "oos_mve.csv"), oos_res.mve.rename("mve"), index_label="period")
        history = pd.DataFrame(
            {
                "lam": oos_res.lam_history,
                "K": oos_res.K_history,
                "n_selected": oos_res.n_selected_history,
                "is_sharpe": oos_res.is_sharpe_history,
            }
        )
        lam_max_hist = oos_res.meta.get("lam_max_history") if isinstance(oos_res.meta, dict) else None
        if lam_max_hist is not None and len(lam_max_hist) == len(history):
            history["lam_max"] = np.asarray(lam_max_hist, dtype=float)
        train_end = oos_res.meta.get("train_end") if isinstance(oos_res.meta, dict) else None
        if train_end is not None and len(train_end) == len(history):
            history["train_end"] = [pd.Timestamp(t) for t in train_end]
        _write_table(put("oos_history", "oos_history.csv"), history, index_label="refit_period")
        _write_table(put("oos_selected_history", "oos_selected_history.csv"), oos_res.selected_history, index_label="refit_period")
        _write_table(put("oos_gamma_norm_history", "oos_gamma_norm_history.csv"), oos_res.gamma_norm_history, index_label="refit_period")

    ev = result.evaluation
    if ev.placebo is not None:
        _write_json(put("placebo", "placebo.json"), _placebo_dict(ev.placebo))
    if ev.pricing_tests:
        summary = ev.tables.get("pricing")
        if summary is None:
            summary = _evaluation.pricing_summary(ev.pricing_tests)
        _write_table(put("pricing_summary", "pricing_summary.csv"), summary, index_label="model")
        for key, pt in ev.pricing_tests.items():
            table = pd.DataFrame({"alpha": pt.alphas, "t_stat": pt.t_stats, "r2": pt.r2})
            table = table.join(pt.betas.add_prefix("beta_"))
            _write_table(put(f"pricing_{key}", f"pricing_{key}.csv"), table, index_label="asset")
    if ev.factor_correlations is not None:
        _write_table(put("factor_correlations", "factor_correlations.csv"), ev.factor_correlations, index_label="factor")

    _write_json(put("timings", "timings.json"), result.timings)
    meta_out = {k: v for k, v in (result.meta or {}).items() if k not in ("asset_meta",)}
    meta_out["config_hash"] = meta_out.get("config_hash") or (config_hash(result.config) if is_dataclass(result.config) else None)
    meta_out["n_files"] = len(manifest) + 2
    _write_json(put("meta", "meta.json"), meta_out)
    asset_meta = (result.meta or {}).get("asset_meta")
    if isinstance(asset_meta, pd.DataFrame) and len(asset_meta):
        _write_table(put("asset_meta", "asset_meta.csv"), asset_meta, index_label="asset")

    if bool(cfg_dict.get("save_panel", False)):
        cov = result.covariances
        np.savez(
            put("covariances", "covariances.npz"),
            values=np.asarray(cov.values),
            n_days=np.asarray(cov.n_days),
            periods=np.asarray(pd.DatetimeIndex(cov.periods).asi8),
            window_end=np.asarray(pd.DatetimeIndex(cov.window_end).asi8),
            assets=np.asarray(cov.assets, dtype=str),
            topics=np.asarray(cov.topics, dtype=str),
            xi=np.asarray(cov.xi),
        )
        pnl = result.panel
        long = pd.DataFrame(pnl.X, columns=list(pnl.instrument_names))
        long.insert(0, "y", pnl.y)
        long.insert(0, "asset", np.asarray(pnl.assets, dtype=object)[pnl.asset_idx])
        long.insert(0, "period", pd.DatetimeIndex(pnl.periods)[pnl.t_idx])
        try:
            long.to_parquet(put("panel", "panel.parquet"), index=False)
        except (ImportError, ValueError) as exc:
            logger.warning("panel.parquet not written (%s); writing panel.csv instead", exc)
            manifest.pop("panel", None)
            long.to_csv(put("panel", "panel.csv"), index=False)

    _write_json(out / "manifest.json", {k: Path(v).name for k, v in manifest.items()})
    manifest["manifest"] = str(out / "manifest.json")
    logger.info("save_result: %d files written to %s", len(manifest), out)
    return manifest


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def read_frame(path: str | Path, dates: bool = True) -> pd.DataFrame:
    """Read a parquet (``.parquet``/``.pq``) or CSV (``.csv``/``.txt``) table.

    With ``dates=True`` the index is parsed as dates and sorted; a table
    whose index is not datetime-like but whose first column is (or is named
    ``date``/``Date``/``period``/``index``) uses that column as the index.
    Column labels are coerced to ``str``.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    elif ext in (".csv", ".txt"):
        # round_trip: parse floats exactly (the default "high" parser can be off by one ulp,
        # which the ARLS amplifies into visible differences of near-zero Gamma entries).
        df = pd.read_csv(p, index_col=0, float_precision="round_trip")
    else:
        raise ValueError(f"unsupported input format {ext!r} for {p} (use .parquet or .csv)")
    if dates:
        if not isinstance(df.index, pd.DatetimeIndex):
            if df.shape[1] and (str(df.columns[0]).lower() in ("date", "period", "index", "day", "time")):
                df = df.set_index(df.columns[0])
            try:
                df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{p}: the index is not datetime-like") from exc
        df = df.sort_index()
        df.index.name = "date"
    df.columns = [str(c) for c in df.columns]
    return df


def load_inputs(
    attention_path: str | Path,
    returns_path: str | Path,
    meta_path: str | Path | None = None,
    risk_free_path: str | Path | None = None,
) -> tuple[AttentionData, ReturnsData]:
    """Load the two daily inputs (and optional asset metadata / risk-free rate) from parquet or CSV.

    ``attention_path``: dates x topics levels ``theta_tau`` (D1);
    ``returns_path``: dates x assets daily returns, ``NaN`` outside the
    universe (D2); ``meta_path``: a table indexed by asset id (e.g. an
    ``asset_class`` column); ``risk_free_path``: a one-column daily series
    used when ``DataConfig.return_kind == "total"``. Format is chosen by
    extension; indexes are parsed as dates.
    """
    att = read_frame(attention_path, dates=True).astype(float)
    ret = read_frame(returns_path, dates=True).astype(float)
    meta: pd.DataFrame | None = None
    if meta_path is not None:
        meta = read_frame(meta_path, dates=False)
        meta.index = pd.Index([str(i) for i in meta.index], name="asset")
    rf: pd.Series | None = None
    if risk_free_path is not None:
        rf_frame = read_frame(risk_free_path, dates=True)
        if rf_frame.shape[1] == 0:
            raise ValueError(f"{risk_free_path}: no risk-free column")
        rf = rf_frame.iloc[:, 0].astype(float).rename("risk_free")
    logger.info(
        "load_inputs: attention %s from %s; returns %s from %s%s%s",
        att.shape, attention_path, ret.shape, returns_path,
        f"; asset_meta {meta.shape} from {meta_path}" if meta is not None else "",
        f"; risk_free {len(rf)} rows from {risk_free_path}" if rf is not None else "",
    )
    return AttentionData(levels=att), ReturnsData(returns=ret, asset_meta=meta, risk_free=rf)
