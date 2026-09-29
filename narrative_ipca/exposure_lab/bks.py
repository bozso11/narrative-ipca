"""BKS Sparse IPCA run by the topic-exposure lab (DESIGN.md G.7.2, G.8, G.10; D52, D65, D70).

The lab calls the package's BKS stages directly, with weekly periods, and
does not change them (D53):

1. :func:`build_bks_panel`: ``align_inputs -> attention_shocks (window w) ->
   build_covariance_panel -> build_panel`` on the simulated attention levels
   and the market returns (treated as excess returns). The panel depends only
   on the simulation, the panel settings of :class:`BKSLabConfig`
   (half-life, asset weighting, burn-in, ``min_days``) and the shock window
   ``w``, so callers can cache it under ``LabConfig.key("bks_panel")``.
2. :func:`fit_bks`: Sparse IPCA on the weekly periods whose last trading day
   is on or before ``train_end`` (``IPCAPanel.subset_periods`` recomputes
   ``sigma^c_l`` on the training rows, D26); ``lambda`` by the tolerance
   rule (default, D51), the exact BKS argmax, or fixed.
3. :func:`evaluate_bks`: every weekly period whose last trading day falls in
   the forecast window gets its factor from its own cross-section with the
   frozen training ``Gamma`` (``oos.oos_factor``, BKS Section 4.2), and the
   fitted return is split into per-topic terms.
4. :func:`implied_exposures`: the topic exposures that the training fit
   implies (BKS Eq. 5), in the direct estimator's units and on its training
   pairs, so that the method comparison (:mod:`.compare`) scores BKS with the
   same forecast-window evaluation as the direct methods.

Two checks need no panel: :func:`training_weeks` (the training weeks a fit
can use after the burn-in, D87) and :func:`kernel_history_share` (the share
of the instruments' kernel weight before the training window, G.15).

The BKS OOS R2 is a contemporaneous factor fit: each forecast week's ``K``
factors are estimated from that week's own returns. It is therefore not
comparable to the direct estimator's R2, which fits nothing in the window
(D79). :func:`evaluate_bks` reports a reference for the part that the
per-week factor fit alone explains: the same R2 with the topic instruments
shuffled across assets within each week (``meta["shuffled_r2_pooled"]``).

Symbols: ``i`` assets, ``t`` weekly periods, ``l = 1..L`` topics (instrument
column ``l``; column 0 is the constant), ``K`` factors; ``C_{t-1}`` the
``(N_t x (L+1))`` instrument rows ``c_{i,t-1} = [1, cov_{i,t-1}]`` of the
assets observed in period ``t``; ``y_t`` their period returns; ``Gamma``
the ``((L+1) x K)`` instrument-to-loading map of the training fit; ``f_t``
the ``K`` out-of-sample factors of period ``t``.

Per-asset fitted return in forecast week ``t`` (G.7.2):
``rf_{i,t} = c_{i,t-1} Gamma f_t = Gamma_0 f_t + sum_l cov_{i,t-1,l} Gamma_l f_t``.
The per-topic terms sum exactly to the fitted value, but by D52 only
``c Gamma`` is identified: the split across topics belongs to the sparse
representative the group lasso chose.

Units. With ``asset_weighting = "inverse_vol"`` (the lab default, D7) the
panel returns are sums of daily returns divided by their trailing volatility.
Fitted and realised returns and the per-topic terms are converted back to
return units by multiplying each asset-week with the asset's mean daily
divisor over that week's days; the conversion is approximate because the
divisor varies slightly within a week. R2 values are computed in panel units.

Randomness: none. Every stage here is deterministic given its inputs (the
Sparse IPCA initialisation uses ``EstimationConfig.seed``'s fixed default).

Validity boundaries
-------------------
* The first ``burn_in_weeks`` weekly instrument periods are dropped (kernel
  warm-up, D17), so the training window must extend at least that far past
  the start of the data.
* A forecast window that starts mid-week evaluates the whole first week; its
  return includes days before ``forecast_start``. Window days in a week that
  ends after ``forecast_end`` are in no evaluated week. Both are recorded in
  ``meta["warnings"]``, and ``meta["evaluated_span"]`` gives the days BKS
  scores. The last week of the data may be truncated (the sample ends on
  Wednesday 2025-12-31); forecast weeks with fewer than five trading days
  are also recorded there.
* ``K`` must be below the number of assets: with ``K >= N`` each week's
  factors fit its returns exactly (:func:`fit_bks` raises).
* At ``lambda = 0`` the core fits plain IPCA with ``Gamma'Gamma = I`` and no
  ridge in its factor step (D18), so the out-of-sample factor uses ridge 0
  there as well; every other ``lambda`` uses the BKS ridge 2.
* Out-of-sample discipline (D65): ``Gamma`` uses training periods only; the
  instruments ``c_{i,t-1}`` of a forecast week use data up to the window end
  of week ``t-1`` (ex ante), which may lie after ``train_end``.
* The implied exposures (:func:`implied_exposures`) are a rank-``K``
  reconstruction of the assets' topic covariances: with ``K`` below the
  number of selected topics they cannot represent independent exposures to
  every topic, and when the fit keeps the constant instrument its implied
  covariance is added to every asset (``meta["B_const"]``). The kernel
  covariance behind ``c_i`` weighs the whole history before the last
  training week (half-life ``half_life_months``), while ``Sigma_z`` and the
  scales use the training window only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import (
    CovarianceConfig,
    DataConfig,
    EstimationConfig,
    EvaluationConfig,
    LambdaGridConfig,
    OOSConfig,
    PipelineConfig,
    ShockConfig,
    TuningConfig,
)
from ..covariances import build_covariance_panel, kernel_weights, window_bounds
from ..data import align_inputs, period_end_index, trailing_volatility
from ..oos import RIDGE, oos_factor
from ..panel import build_panel
from ..shocks import attention_shocks
from ..sparse_ipca import betas as ipca_betas
from ..sparse_ipca import canonicalize, fit_sparse_ipca
from ..tuning import tune
from ..types import (
    AlignedData,
    AttentionData,
    IPCAPanel,
    ReturnsData,
    SparseIPCAResult,
    TuningResult,
)
from ..wrapup import _lsq_map, _sym_pinv
from .config import DATA_END, DATA_START, BKSLabConfig, WindowConfig
from .direct import MIN_TRAIN_OBS, _train_moments, shock_matrix, training_pairs
from .types import BKSLabResult, DirectFit, ObservedShocks, SimData

logger = logging.getLogger(__name__)

__all__ = [
    "PERIOD",
    "ANNUALIZATION",
    "MIN_TRAIN_PERIODS",
    "N_SHUFFLES",
    "SHUFFLE_STREAM",
    "D52_NOTE",
    "IMPLIED_NOTE",
    "IMPLIED_METHOD",
    "BKSPanel",
    "BKSFit",
    "bks_pipeline_config",
    "training_weeks",
    "kernel_history_share",
    "build_bks_panel",
    "fit_bks",
    "evaluate_bks",
    "implied_topic_covariance",
    "implied_exposures",
    "run_bks",
]

ProgressFn = Callable[[int, int, str], None]
"""``progress(done, total, message)`` callback, passed to :func:`narrative_ipca.tuning.tune` (D44)."""

#: Estimation period of the lab's BKS runs: calendar weeks (G.7.2, D70).
PERIOD = "W"

#: Weekly periods per year, used to annualise the tuning Sharpe ratio.
ANNUALIZATION = 52.0

#: Upper bound of ``DataConfig.min_assets_per_period`` (the package default).
_MAX_MIN_ASSETS = 20

#: Fewest training periods :func:`fit_bks` accepts: about six months of weeks. A training
#: window of six calendar months holds 25 or 26 week ends, so the floor leaves a margin of one
#: to two weeks (lowered from 26 on 2026-09-29 for the dashboard's six-month default window).
MIN_TRAIN_PERIODS = 24

#: Trading days of a full week on the weekday calendar (shorter weeks are flagged).
_WEEK_DAYS = 5

#: Shuffles per forecast week of the shuffled-instrument reference R2 (D79).
N_SHUFFLES = 20

#: rng stream of the shuffled-instrument reference (D71: one stream per random component).
SHUFFLE_STREAM = 7201

#: Caveat attached to every per-topic split of the BKS fitted return (D52).
D52_NOTE = (
    "Per-topic split not identified (D52): only c Gamma is identified, so the split of the fitted "
    "return across topics belongs to the sparse representative the group lasso chose; another "
    "Gamma with the same fitted values would split it differently."
)

#: Method name of the BKS-implied exposures in the method comparison (:func:`implied_exposures`).
IMPLIED_METHOD = "bks_implied"

#: Caveat attached to the BKS-implied exposures (D52). Where the signal goes, on the dashboard
#: defaults (2026-09-29): the instruments alone, through the same unit conversion, reach Spearman
#: 0.71 with ``B_true`` (review measurement; the rank-K reconstruction equals them only at K = L);
#: the implied exposures reach 0.19 for the tuned K = 3 fit and 0.18 at K = 10 with all 20 topics
#: kept (lambda 0), so the loss comes from the rank-K projection, not from topic selection.
IMPLIED_NOTE = (
    "BKS identifies the assets' factor betas, not how they split across topics (D52). With K factors the "
    "implied exposures cannot represent independent exposures to every topic, so most of the topic signal "
    "in the BKS instruments is lost in the conversion. They also depend on which topics the sparse fit kept."
)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass
class BKSPanel:
    """The weekly BKS estimation panel of a lab simulation (G.7.2).

    Attributes
    ----------
    aligned:
        Attention and returns on the trading-day calendar
        (:class:`~narrative_ipca.types.AlignedData`); with ``inverse_vol``
        weighting ``aligned.returns`` are vol-scaled and ``aligned.scale``
        holds the daily divisor.
    panel:
        Long-form panel pairing ``c_{i,t-1}`` with the week-``t`` return
        (Eq. 7); ``panel.periods`` are the last trading days of the return
        weeks.
    pipeline_cfg:
        The :class:`~narrative_ipca.config.PipelineConfig` the panel was built
        with (see :func:`bks_pipeline_config`).
    meta:
        ``timings`` (seconds per stage), ``panel_params`` (the settings the
        panel depends on), ``shapes``, ``shock_window``, ``lead_days``,
        ``topic_labels``.

    The daily shocks and the 3-D covariance array are not kept: nothing reads
    them after ``build_panel``, and at 500 assets x 500 topics the covariance
    array alone is about 1.1 GB per cached panel.
    """

    aligned: AlignedData
    panel: IPCAPanel
    pipeline_cfg: PipelineConfig
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BKSFit:
    """Sparse IPCA fitted on the training weeks (G.7.2).

    Attributes
    ----------
    fit:
        The chosen fit in canonical form (D24); ``Gamma`` in original
        instrument units.
    tuning:
        The lambda path and choice, or ``None`` for ``lambda_rule = "fixed"``.
    lam, K:
        Penalty and number of factors of ``fit``.
    train_periods:
        Last trading day of every training week used.
    meta:
        ``timings``, ``lambda_rule``, ``tolerance``, ``n_selected``,
        ``converged``, ``n_iter``, ``is_sharpe``, ``lam_max``, training window
        bounds, ``n_obs`` and ``warnings``.
    """

    fit: SparseIPCAResult
    tuning: TuningResult | None
    lam: float
    K: int
    train_periods: pd.DatetimeIndex
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def bks_pipeline_config(
    cfg: BKSLabConfig,
    shock_window: int,
    lead_days: int,
    n_assets: int | None = None,
) -> PipelineConfig:
    """The package :class:`PipelineConfig` of a lab BKS run (G.7.2, D70).

    Parameters
    ----------
    cfg:
        Lab BKS settings.
    shock_window:
        ``w``, the trailing window of the attention shocks (D9).
    lead_days:
        ``l`` of the simulation (G.5): attention on day ``t`` relates to the
        return of day ``t + l``, so the attention is lagged by ``l`` trading
        days before pairing (``DataConfig.attention_lag_days``, D4, D64).
    n_assets:
        Number of assets ``N``; when given, ``min_assets_per_period`` is
        ``min(20, max(2, N // 2))`` so that small universes keep their weeks.
        ``None`` keeps the package default (20).

    Returns
    -------
    PipelineConfig
        Weekly periods, ``xi = cfg.xi_weekly`` (half-life matched), burn-in in
        weeks, ``lambda`` tuned by the in-sample Sharpe ratio (annualised with
        52 weeks) with tolerance ``cfg.tolerance`` for the ``"tolerance"``
        rule and ``0`` (exact argmax) otherwise, ``lam = cfg.lam`` only for
        the ``"fixed"`` rule, no OOS stage and no wrap-up.
    """
    if int(shock_window) < 1:
        raise ValueError("shock_window must be >= 1")
    data_kw: dict[str, Any] = {
        "period": PERIOD,
        "attention_lag_days": int(lead_days),
        "asset_weighting": cfg.asset_weighting,
    }
    if n_assets is not None:
        data_kw["min_assets_per_period"] = int(min(_MAX_MIN_ASSETS, max(2, int(n_assets) // 2)))
    rule = cfg.lambda_rule
    return PipelineConfig(
        data=DataConfig(**data_kw),
        shocks=ShockConfig(window=int(shock_window)),
        covariance=CovarianceConfig(
            xi=cfg.xi_weekly, burn_in_periods=int(cfg.burn_in_weeks), min_days=int(cfg.min_days)
        ),
        estimation=EstimationConfig(
            K=int(cfg.K),
            lam=float(cfg.lam) if rule == "fixed" else None,
            lam_grid=LambdaGridConfig(n_lambdas=int(cfg.n_lambdas), ratio=float(cfg.lambda_ratio)),
            penalize_intercept=bool(cfg.penalize_intercept),
            max_iter=int(cfg.max_iter),
        ),
        tuning=TuningConfig(criterion="is_sharpe", tolerance=float(cfg.tolerance) if rule == "tolerance" else 0.0),
        oos=OOSConfig(enabled=False),
        evaluation=EvaluationConfig(annualization=ANNUALIZATION),
        run_wrapup=False,
        name="exposure-lab-bks",
    )


def _panel_params(cfg: BKSLabConfig, shock_window: int, lead_days: int) -> dict[str, Any]:
    """The settings a :class:`BKSPanel` depends on (besides the simulation)."""
    return {
        "half_life_months": float(cfg.half_life_months),
        "xi": float(cfg.xi_weekly),
        "asset_weighting": str(cfg.asset_weighting),
        "burn_in_weeks": int(cfg.burn_in_weeks),
        "min_days": int(cfg.min_days),
        "shock_window": int(shock_window),
        "lead_days": int(lead_days),
    }


# ---------------------------------------------------------------------------
# Training weeks and kernel history, without building the panel
# ---------------------------------------------------------------------------
def _panel_days(cfg: BKSLabConfig, lead_days: int, calendar: pd.DatetimeIndex | None) -> pd.DatetimeIndex:
    """Trading days a BKS panel starts from: the data calendar after the warm-up of ``align_inputs``.

    ``align_inputs`` keeps the days from the first one that carries attention
    (lagged by ``l`` days) and a return; with ``inverse_vol`` weighting a
    return also needs a trailing volatility. The first such day is found with
    the package's own :func:`narrative_ipca.data.trailing_volatility` on a
    probe series, so its ``min_periods`` rule is not restated here. Assumes
    every asset has returns from the first day of ``calendar`` (true for the
    lab's listed and generic assets).
    """
    cal = pd.bdate_range(DATA_START, DATA_END) if calendar is None else pd.DatetimeIndex(calendar)
    first = min(max(int(lead_days), 0), len(cal))
    if cfg.asset_weighting == "inverse_vol" and len(cal):
        window = int(bks_pipeline_config(cfg, 1, lead_days).data.vol_window_days)
        probe = pd.DataFrame({"x": np.resize(np.array([1.0, -1.0]), len(cal))}, index=cal)
        ok = np.isfinite(trailing_volatility(probe, window)["x"].to_numpy(dtype=float))
        first = max(first, int(np.argmax(ok)) if ok.any() else len(cal))
    return cal[first:]


def training_weeks(
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    cfg: BKSLabConfig,
    lead_days: int = 0,
    calendar: pd.DatetimeIndex | None = None,
) -> tuple[int, pd.Timestamp]:
    """Training weeks a BKS fit on this window can use, and the first week the data allow (G.7.2; D17).

    ``build_panel`` drops the first ``cfg.burn_in_weeks`` weekly instrument
    periods (kernel warm-up), and the return of week ``t`` pairs with the
    instruments of week ``t - 1``. The first usable return week is therefore
    week ``burn_in_weeks + 2`` of the panel's days (:func:`_panel_days`). On
    the lab's data (from 2015-01-02, ``inverse_vol``, 52 weeks) it ends on
    2016-04-08, so a training window that starts earlier loses its first
    weeks.

    The count is the one :func:`fit_bks` checks against
    :data:`MIN_TRAIN_PERIODS` (week ends in ``[train_start, train_end]``),
    before the package drops weeks with fewer than ``min_assets_per_period``
    assets. It needs no panel, so the dashboard can check a window before a
    run.

    Parameters
    ----------
    train_start, train_end:
        Training window, inclusive.
    cfg:
        Lab BKS settings; the asset weighting and the burn-in matter.
    lead_days:
        ``l`` of the simulation (attention is lagged by ``l`` days).
    calendar:
        Trading days of the data; default the lab's weekday calendar from
        :data:`DATA_START` to :data:`DATA_END`.

    Returns
    -------
    (n_weeks, first_week)
        The usable training weeks and the last trading day of the first week
        any BKS fit can use (``NaT`` when the data are too short).
    """
    days = _panel_days(cfg, lead_days, calendar)
    _, ends = period_end_index(days, PERIOD)
    pos = int(cfg.burn_in_weeks) + 1
    if pos >= len(ends):
        return 0, pd.NaT
    usable = ends[pos:]
    ts, te = pd.Timestamp(train_start), pd.Timestamp(train_end)
    n = int(np.sum(np.asarray((usable >= ts) & (usable <= te), dtype=bool)))
    return n, pd.Timestamp(usable[0])


def _history_share(days: pd.DatetimeIndex, xi: float, skip_days: int, week_end: pd.Timestamp,
                   train_start: pd.Timestamp) -> float:
    """Kernel weight share before ``train_start`` of the instruments paired with the week ending ``week_end``."""
    pid, ends = period_end_index(days, PERIOD)
    t = int(ends.get_indexer([pd.Timestamp(week_end)])[0])
    if t < 1:
        return float("nan")
    j = t - 1  # the instruments of week t are measured at the end of week t - 1
    w = kernel_weights(pid, j, float(xi))
    _, stop, cut = window_bounds(pid, int(skip_days))
    w[cut[j]:stop[j]] = 0.0  # the window-end cut-off of week j (D12)
    total = float(w.sum())
    if total <= 0.0:
        return float("nan")
    before = np.asarray(days < pd.Timestamp(train_start), dtype=bool)
    return float(w[before].sum() / total)


def kernel_history_share(
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    cfg: BKSLabConfig,
    lead_days: int = 0,
    calendar: pd.DatetimeIndex | None = None,
) -> float:
    """Share of the BKS instruments' kernel weight on days before the training window (G.15).

    The instrument row that the BKS-implied exposures use (the last training
    week's, :func:`implied_exposures`) is a kernel covariance: day ``tau``
    gets weight ``xi^(j - t_tau)``, with ``j`` the week before the last
    training week, ``t_tau`` the week of day ``tau`` and
    ``xi = cfg.xi_weekly``, over every day from the start of the panel's
    days (BKS App. B.1; the last ``skip_days`` days of week ``j`` are left
    out, D12). This returns the share of that weight on days before
    ``train_start``, with every day counted (missing days move it slightly).
    On the dashboard defaults (training 2025-01-01 to 2025-06-30, half-life
    69 months) it is about 0.92: BKS-implied draws on history the direct
    methods never see.

    Returns ``NaN`` when no training week is usable (:func:`training_weeks`).
    Parameters as in :func:`training_weeks`.
    """
    days = _panel_days(cfg, lead_days, calendar)
    _, ends = period_end_index(days, PERIOD)
    pos = int(cfg.burn_in_weeks) + 1
    ts, te = pd.Timestamp(train_start), pd.Timestamp(train_end)
    keep = np.flatnonzero(np.asarray((ends >= ts) & (ends <= te), dtype=bool) & (np.arange(len(ends)) >= pos))
    if not len(keep):
        return float("nan")
    skip = int(bks_pipeline_config(cfg, 1, lead_days).covariance.skip_days)
    return _history_share(days, float(cfg.xi_weekly), skip, pd.Timestamp(ends[int(keep[-1])]), ts)


# ---------------------------------------------------------------------------
# Stage 1: the weekly panel
# ---------------------------------------------------------------------------
def build_bks_panel(sim: SimData, cfg: BKSLabConfig, shock_window: int) -> BKSPanel:
    """Build the weekly BKS panel from the simulated attention and the market returns (G.7.2).

    Stages (package functions, unchanged): :func:`narrative_ipca.data.align_inputs`
    (attention lagged by ``sim.lead_days``, returns treated as excess
    returns, optional inverse-volatility scaling), :func:`narrative_ipca.shocks.attention_shocks`
    (window ``shock_window``), :func:`narrative_ipca.covariances.build_covariance_panel`
    (weekly, float64) and :func:`narrative_ipca.panel.build_panel`.

    Parameters
    ----------
    sim:
        Output of :func:`narrative_ipca.exposure_lab.dgp.simulate_lab`; uses
        ``attention`` (the levels the estimators see), ``market.returns``,
        ``market.assets``, ``topics`` (names as labels) and ``lead_days``.
    cfg:
        Lab BKS settings; only the panel settings matter here (half-life,
        asset weighting, burn-in, ``min_days``).
    shock_window:
        ``w`` of the attention shocks.

    Returns
    -------
    BKSPanel
        Cacheable: it depends only on ``sim``, the panel settings and ``w``.
    """
    t_all = time.perf_counter()
    timings: dict[str, float] = {}
    returns = sim.market.returns
    n_assets = int(returns.shape[1])
    lead = int(sim.lead_days)
    pcfg = bks_pipeline_config(cfg, shock_window, lead, n_assets=n_assets)
    table = sim.topics.table
    labels = {str(k): str(v) for k, v in table["name"].items()} if "name" in table.columns else None
    attention = AttentionData(levels=sim.attention, topic_labels=labels)
    rets = ReturnsData(returns=returns, asset_meta=sim.market.assets)

    t = time.perf_counter()
    aligned = align_inputs(attention, rets, pcfg.data)
    timings["align"] = time.perf_counter() - t
    t = time.perf_counter()
    shock_panel = attention_shocks(aligned.attention, pcfg.shocks)
    timings["shocks"] = time.perf_counter() - t
    t = time.perf_counter()
    cov = build_covariance_panel(shock_panel, aligned.returns, pcfg.covariance, pcfg.data.period, dtype="float64")
    timings["covariances"] = time.perf_counter() - t
    t = time.perf_counter()
    panel = build_panel(cov, aligned.returns, pcfg.data, pcfg.covariance)
    timings["panel"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t_all
    cov_shape = tuple(int(x) for x in np.shape(getattr(cov, "values", ())))
    del cov, shock_panel  # not needed after build_panel; the covariance array dominates memory

    meta: dict[str, Any] = {
        "timings": timings,
        "panel_params": _panel_params(cfg, shock_window, lead),
        "shock_window": int(shock_window),
        "lead_days": lead,
        "topic_labels": labels,
        "shapes": {
            "n_days": int(len(aligned.calendar)),
            "n_topics": int(aligned.attention.shape[1]),
            "n_assets": n_assets,
            "T": int(panel.T),
            "n_obs": int(panel.n_obs),
            "first_period": pd.Timestamp(panel.periods[0]),
            "last_period": pd.Timestamp(panel.periods[-1]),
            "covariance_array": cov_shape,
        },
    }
    logger.info(
        "build_bks_panel: %d days x %d topics x %d assets -> %d weekly periods (%s .. %s), %d rows; "
        "w=%d lead=%d xi=%.4f weighting=%s (%.2fs: align %.2f, shocks %.2f, covariances %.2f, panel %.2f)",
        len(aligned.calendar), aligned.attention.shape[1], n_assets, panel.T, panel.periods[0].date(),
        panel.periods[-1].date(), panel.n_obs, int(shock_window), lead, cfg.xi_weekly, cfg.asset_weighting,
        timings["total"], timings["align"], timings["shocks"], timings["covariances"], timings["panel"],
    )
    return BKSPanel(aligned=aligned, panel=panel, pipeline_cfg=pcfg, meta=meta)


# ---------------------------------------------------------------------------
# Stage 2: the training fit
# ---------------------------------------------------------------------------
def fit_bks(
    panel: BKSPanel,
    cfg: BKSLabConfig,
    train_end: str | pd.Timestamp,
    progress: ProgressFn | None = None,
    *,
    train_start: str | pd.Timestamp | None = None,
) -> BKSFit:
    """Fit Sparse IPCA on the training weeks (G.7.2, D65, D70).

    The training weeks are those whose last trading day is on or before
    ``train_end`` (and on or after ``train_start`` when given). The panel is
    restricted to them with ``IPCAPanel.subset_periods``, which recomputes
    ``sigma^c_l`` (the panel standard deviation of instrument ``l`` that
    scales its penalty) on the training rows only (D26).

    ``lambda`` by ``cfg.lambda_rule``:

    * ``"tolerance"`` (default): :func:`narrative_ipca.tuning.tune` with the
      in-sample Sharpe criterion and the sparsest grid point within
      ``cfg.tolerance`` of the best Sharpe ratio (D51);
    * ``"argmax"``: the same with tolerance 0 (BKS exact argmax);
    * ``"fixed"``: one :func:`narrative_ipca.sparse_ipca.fit_sparse_ipca` at
      ``cfg.lam``, canonicalised (D24); ``tuning`` is ``None``.

    Parameters
    ----------
    panel:
        Output of :func:`build_bks_panel`.
    cfg:
        Lab BKS settings. The estimation settings (``K``, lambda rule and
        grid, intercept penalty, ``max_iter``) are taken from here; if the
        panel settings differ from those the panel was built with, the panel
        is used as is and a warning is recorded.
    train_end:
        Last day of the training window (inclusive).
    progress:
        Optional ``progress(done, total, message)`` callback.
    train_start:
        Optional first day of the training window (inclusive).

    Raises
    ------
    ValueError
        When fewer than :data:`MIN_TRAIN_PERIODS` training weeks remain, or
        when ``K`` is not below the number of assets (each forecast week's
        factors would then fit its returns exactly).
    """
    t_all = time.perf_counter()
    warnings_: list[str] = []
    n_assets = int(panel.panel.N)
    if int(cfg.K) >= n_assets:
        raise ValueError(
            f"K = {int(cfg.K)} factors need more than {int(cfg.K)} assets; this run has {n_assets}, so each "
            "forecast week's factors would fit its returns exactly. Lower K or add assets."
        )
    shock_window = int(panel.pipeline_cfg.shocks.window)
    lead = int(panel.pipeline_cfg.data.attention_lag_days)
    pcfg = bks_pipeline_config(cfg, shock_window, lead, n_assets=int(panel.panel.N))
    built_with = panel.meta.get("panel_params", {})
    wanted = _panel_params(cfg, shock_window, lead)
    mismatch = {k: (built_with.get(k), v) for k, v in wanted.items() if k in built_with and built_with.get(k) != v}
    if mismatch:
        msg = f"panel was built with different settings than cfg (built, cfg): {mismatch}; the panel is used as is"
        logger.warning("fit_bks: %s", msg)
        warnings_.append(msg)

    periods = pd.DatetimeIndex(panel.panel.periods)
    te = pd.Timestamp(train_end)
    keep = np.asarray(periods <= te, dtype=bool)
    ts = None if train_start is None else pd.Timestamp(train_start)
    if ts is not None:
        keep &= np.asarray(periods >= ts, dtype=bool)
    n_keep = int(keep.sum())
    if n_keep < MIN_TRAIN_PERIODS:
        raise ValueError(
            f"only {n_keep} weekly periods end inside the training window "
            f"[{'start' if ts is None else ts.date()}, {te.date()}] (need >= {MIN_TRAIN_PERIODS}); the panel "
            f"starts at {periods[0].date()} after a burn-in of {cfg.burn_in_weeks} weeks"
        )
    t = time.perf_counter()
    sub = panel.panel.subset_periods(keep)
    t_subset = time.perf_counter() - t

    t = time.perf_counter()
    tr: TuningResult | None
    if cfg.lambda_rule == "fixed":
        lam = float(cfg.lam)  # BKSLabConfig guarantees lam >= 0 for the fixed rule
        if progress is not None:
            progress(0, 1, f"fixed lambda={lam:.4g}")
        res = canonicalize(fit_sparse_ipca(sub, pcfg.estimation, lam=lam, K=int(cfg.K)))
        tr = None
        if progress is not None:
            progress(1, 1, f"fixed lambda={lam:.4g} selected={res.n_selected}")
    else:
        tr = tune(sub, pcfg.estimation, pcfg.tuning, pcfg.evaluation, progress=progress)
        res = tr.fit
    t_fit = time.perf_counter() - t

    if not res.converged:
        msg = f"Sparse IPCA did not converge in {res.n_iter} sweeps (lambda={res.lam:.4g}, K={res.K})"
        logger.warning("fit_bks: %s", msg)
        warnings_.append(msg)
    if res.n_selected < res.K:
        msg = f"{res.n_selected} topics selected for K={res.K}: fewer narratives than factors (D43)"
        warnings_.append(msg)
    timings = {"subset": t_subset, "fit": t_fit, "total": time.perf_counter() - t_all}
    meta: dict[str, Any] = {
        "timings": timings,
        "lambda_rule": str(cfg.lambda_rule),
        "tolerance": float(pcfg.tuning.tolerance),
        "n_selected": int(res.n_selected),
        "converged": bool(res.converged),
        "n_iter": int(res.n_iter),
        "is_sharpe": float(res.mve_sharpe(annualization=ANNUALIZATION, rcond=pcfg.evaluation.rcond)),
        "lam_max": None if tr is None else float(tr.lam_max),
        "n_path_points": 1 if tr is None else len(tr.path),
        "train_start": ts,
        "train_end": te,
        "first_train_period": pd.Timestamp(sub.periods[0]),
        "last_train_period": pd.Timestamp(sub.periods[-1]),
        "n_train_periods": int(sub.T),
        "n_obs": int(sub.n_obs),
        "sigma_c": np.asarray(sub.sigma_c, dtype=float).copy(),
        "panel_mismatch": mismatch,
        "warnings": warnings_,
    }
    logger.info(
        "fit_bks: %s rule -> lambda=%.4g K=%d, %d/%d topics selected, IS Sharpe %.3f, total R2 %.4f, "
        "%d training weeks (%s .. %s), converged=%s (%.2fs)",
        cfg.lambda_rule, res.lam, res.K, res.n_selected, sub.L, meta["is_sharpe"], res.total_r2, sub.T,
        sub.periods[0].date(), sub.periods[-1].date(), res.converged, timings["total"],
    )
    return BKSFit(
        fit=res,
        tuning=tr,
        lam=float(res.lam),
        K=int(res.K),
        train_periods=pd.DatetimeIndex(sub.periods),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Stage 3: evaluation in the forecast window
# ---------------------------------------------------------------------------
def _week_days(panel: BKSPanel, ends: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Trading-day rows of each weekly period ending at ``ends``.

    Returns ``(codes, period_ends, pos)``: the period id of every day of
    ``panel.aligned.calendar``, the last trading day of every period, and the
    position of each of ``ends`` among those periods.
    """
    codes, pend = period_end_index(panel.aligned.calendar, panel.pipeline_cfg.data.period)
    pos = pend.get_indexer(pd.DatetimeIndex(ends))
    if np.any(pos < 0):
        raise ValueError("panel periods are not periods of the aligned calendar")
    return codes, pend, pos


def _divisors(panel: BKSPanel, pos: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean daily divisor and exact return-unit sum per (week, asset).

    For week ``j`` and asset ``i`` over the days where the scaled return is
    observed: ``d_{j,i}`` the mean of ``AlignedData.scale`` (1 without
    scaling) and ``R_{j,i} = sum_tau y_{i,tau} * scale_{i,tau}``, the week's
    return in return units over the same days. ``NaN`` where no day is
    observed.
    """
    ret = panel.aligned.returns.to_numpy(dtype=float)
    scale = None if panel.aligned.scale is None else panel.aligned.scale.to_numpy(dtype=float)
    n_w, n_a = len(pos), ret.shape[1]
    div = np.full((n_w, n_a), np.nan)
    exact = np.full((n_w, n_a), np.nan)
    for w, j in enumerate(pos):
        rows = codes == int(j)
        r = ret[rows]
        s = np.ones_like(r) if scale is None else scale[rows]
        ok = np.isfinite(r) & np.isfinite(s)
        n_ok = ok.sum(axis=0)
        has = n_ok > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            div[w, has] = np.where(ok, s, 0.0).sum(axis=0)[has] / n_ok[has]
        exact[w, has] = np.where(ok, r * s, 0.0).sum(axis=0)[has]
    return div, exact


def evaluate_bks(panel: BKSPanel, fit: BKSFit, window: WindowConfig) -> BKSLabResult:
    """Out-of-sample BKS fit in the forecast window, with the per-topic split (G.7.2, G.8; D52, D67).

    Evaluated weeks: the panel periods whose last trading day lies in
    ``[window.forecast_start, window.forecast_end]``. For each such week
    ``t``, with ``C`` the instrument rows ``c_{i,t-1}`` and ``y`` the returns
    of the assets observed in week ``t``:

    * ``f_t = oos_factor(C, y, Gamma, ridge)`` (BKS Section 4.2): the factor
      from the week's own cross-section with the frozen training ``Gamma``;
      ``ridge`` is 2, or 0 when the fit has ``lambda = 0`` (plain IPCA with
      ``Gamma'Gamma = I``, D18);
    * fitted return ``C Gamma f_t``; per-topic term
      ``contrib_{i,l} = C_{i,l} (Gamma_l f_t)`` for topic columns ``l >= 1``
      and the constant's term ``C_{i,0} (Gamma_0 f_t)``; the constant term
      plus the topic terms equals the fitted return.

    Reference (D79): the same pooled and per-asset R2 with the topic
    instrument columns ``C_{.,1..L}`` shuffled across the week's assets
    (:data:`N_SHUFFLES` shuffles per week, seeded), which measures what ``K``
    per-week factors reach with instruments unrelated to the assets.

    Accumulated over the window: fitted and realised weekly returns
    (weeks x assets, ``NaN`` where the asset is absent); per-asset uncentered
    R2 ``1 - sum_t (y - fitted)^2 / sum_t y^2`` over the asset's window
    weeks and the pooled uncentered R2 over all asset-weeks (both in panel
    units); the topic terms and the constant's term summed over the window
    weeks (assets x topics).

    Units: with ``inverse_vol`` weighting, fitted, realised, topic and
    constant terms are multiplied by the asset's mean daily divisor over the
    week's days (approximate return units, see the module docstring).

    Parameters
    ----------
    panel:
        Output of :func:`build_bks_panel`.
    fit:
        Output of :func:`fit_bks` on the same panel.
    window:
        Forecast window (``forecast_start``, ``forecast_weeks``).

    Returns
    -------
    BKSLabResult
        ``meta`` holds ``units``, ``units_note``, ``warnings`` (the D52 note
        first), ``timings``, ``factors`` (weeks x K), ``fitted_panel`` and
        ``realized_panel`` (weeks x assets, panel units),
        ``contrib_weekly_panel`` (rows ``(period, asset_id)``, columns
        ``const`` and the topics, panel units), ``divisor`` (weeks x assets),
        ``realized_exact`` (weeks x assets, exact return units),
        ``period_first_day``, ``n_assets_per_period``, ``evaluated_span``
        (first day of the first week, last day of the last week),
        ``window_days_not_evaluated`` (window days in no evaluated week),
        ``oos_ridge``, ``shuffled_r2_pooled``, ``shuffled_r2`` (per asset),
        ``n_shuffles``,
        ``gamma_norms_standardized`` (``sigma^c_l ||Gamma_l||`` on the
        training rows), ``gamma_const_norm`` and the fit summary.

    Raises
    ------
    ValueError
        When no weekly period ends inside the forecast window, or when the
        forecast weeks overlap the training weeks of ``fit``.
    """
    t_all = time.perf_counter()
    pnl = panel.panel
    periods = pd.DatetimeIndex(pnl.periods)
    fs, fe = pd.Timestamp(window.forecast_start), pd.Timestamp(window.forecast_end)
    in_win = np.asarray((periods >= fs) & (periods <= fe), dtype=bool)
    if not in_win.any():
        raise ValueError(
            f"no complete weekly period ends inside the forecast window [{fs.date()}, {fe.date()}]: "
            f"the panel's weekly periods end between {periods[0].date()} and {periods[-1].date()}"
        )
    eval_idx = np.flatnonzero(in_win)
    last_train = pd.Timestamp(fit.train_periods.max())
    if periods[eval_idx[0]] <= last_train:
        raise ValueError(
            f"forecast week ending {periods[eval_idx[0]].date()} is not after the last training week "
            f"{last_train.date()}; refit with a train_end before the forecast window (D65)"
        )
    res = fit.fit
    if list(res.instrument_names) != list(pnl.instrument_names):
        raise ValueError("fit and panel have different instruments; fit_bks must be run on this panel")

    Gamma = np.asarray(res.Gamma, dtype=float)
    ridge = 0.0 if float(res.lam) == 0.0 else float(RIDGE)
    topics = [str(c) for c in pnl.topics]
    assets = [str(a) for a in pnl.assets]
    n_w, n_a, n_l, K = len(eval_idx), len(assets), len(topics), int(Gamma.shape[1])
    ends = pd.DatetimeIndex(periods[eval_idx], name="period")
    codes, pend, pos = _week_days(panel, ends)
    div, exact = _divisors(panel, pos, codes)
    scaled = panel.aligned.scale is not None
    if not scaled:
        div = np.where(np.isfinite(div), 1.0, np.nan)

    fitted_p = np.full((n_w, n_a), np.nan)
    realized_p = np.full((n_w, n_a), np.nan)
    divisor = np.full((n_w, n_a), np.nan)
    factors = np.full((n_w, K), np.nan)
    contrib = np.zeros((n_a, n_l))
    const = np.zeros(n_a)
    sse = np.zeros(n_a)
    syy = np.zeros(n_a)
    present = np.zeros(n_a, dtype=bool)
    n_per = np.zeros(n_w, dtype=np.int64)
    sse_shuf = np.zeros(n_a)
    rng = np.random.default_rng([SHUFFLE_STREAM])
    weekly_blocks: list[np.ndarray] = []
    weekly_index: list[tuple[pd.Timestamp, str]] = []
    slices = dict(pnl.period_slices())
    for w, t in enumerate(eval_idx):
        sl = slices[int(t)]
        if sl.stop <= sl.start:  # build_panel keeps only periods with >= min assets; defensive
            continue
        C, y, ai = pnl.X[sl], pnl.y[sl], pnl.asset_idx[sl]
        f = oos_factor(C, y, Gamma, ridge=ridge)
        g = Gamma @ f  # (L+1,): Gamma_l f_t per instrument
        fit_t = (C @ Gamma) @ f
        parts = C * g[None, :]  # (n_t, L+1): column 0 the constant's term, columns 1.. the topics'
        d = div[w, ai]
        d = np.where(np.isfinite(d), d, 1.0)  # an observed y always has an observed day; defensive
        factors[w] = f
        fitted_p[w, ai] = fit_t
        realized_p[w, ai] = y
        divisor[w, ai] = d
        contrib[ai] += parts[:, 1:] * d[:, None]
        const[ai] += parts[:, 0] * d
        sse[ai] += (y - fit_t) ** 2
        syy[ai] += y**2
        for _ in range(N_SHUFFLES):  # reference: topic instruments shuffled across the week's assets
            Cs = C.copy()
            Cs[:, 1:] = C[rng.permutation(C.shape[0]), 1:]
            fit_s = (Cs @ Gamma) @ oos_factor(Cs, y, Gamma, ridge=ridge)
            sse_shuf[ai] += (y - fit_s) ** 2 / N_SHUFFLES
        present[ai] = True
        n_per[w] = int(sl.stop - sl.start)
        weekly_blocks.append(parts)
        weekly_index.extend((ends[w], assets[i]) for i in ai)

    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(present & (syy > 0.0), 1.0 - sse / syy, np.nan)
        r2_shuf = np.where(present & (syy > 0.0), 1.0 - sse_shuf / syy, np.nan)
    tot_yy = float(syy.sum())
    r2_pooled = float(1.0 - sse.sum() / tot_yy) if tot_yy > 0.0 else float("nan")
    r2_pooled_shuf = float(1.0 - sse_shuf.sum() / tot_yy) if tot_yy > 0.0 else float("nan")
    contrib[~present] = np.nan
    const[~present] = np.nan

    a_index = pd.Index(assets, name="asset_id")
    t_index = pd.Index(topics, name="topic_id")
    fitted = pd.DataFrame(fitted_p * divisor, index=ends, columns=a_index)
    realized = pd.DataFrame(realized_p * divisor, index=ends, columns=a_index)
    exact_df = pd.DataFrame(np.where(np.isfinite(realized_p), exact, np.nan), index=ends, columns=a_index)
    weekly = pd.DataFrame(
        np.vstack(weekly_blocks) if weekly_blocks else np.zeros((0, n_l + 1)),
        index=pd.MultiIndex.from_tuples(weekly_index, names=["period", "asset_id"]),
        columns=pd.Index(["const"] + topics, name="instrument"),
    )

    # first trading day of each evaluated week; days of the first week before forecast_start
    cal = panel.aligned.calendar
    first_day = pd.DatetimeIndex([cal[np.flatnonzero(codes == int(j))[0]] for j in pos], name="period_first_day")
    warnings_: list[str] = [D52_NOTE]
    n_before = int(np.sum((cal[codes == int(pos[0])] < fs)))
    if n_before:
        warnings_.append(
            f"the first forecast week (ending {ends[0].date()}) starts on {first_day[0].date()}, before "
            f"forecast_start {fs.date()}: its return includes {n_before} trading day(s) before the window"
        )
    win_days = cal[np.asarray((cal >= fs) & (cal <= fe), dtype=bool)]
    not_evaluated = win_days[win_days > ends[-1]]
    if len(not_evaluated):
        warnings_.append(
            f"forecast window days {not_evaluated[0].date()} to {not_evaluated[-1].date()} "
            f"({len(not_evaluated)} trading day(s)) are in no evaluated week: their week ends after the window"
        )
    n_days_week = np.array([int(np.sum(codes == int(j))) for j in pos])
    short_weeks = [f"{ends[w].date()} ({n_days_week[w]} days)" for w in range(n_w) if n_days_week[w] < _WEEK_DAYS]
    if short_weeks:
        warnings_.append(f"forecast week(s) with fewer than {_WEEK_DAYS} trading days: {', '.join(short_weeks)}")
    warnings_.extend(str(m) for m in fit.meta.get("warnings", []))
    if not present.all():
        warnings_.append(f"{int((~present).sum())} asset(s) have no panel row in the forecast weeks (NaN)")

    if scaled:
        units = "approximate return units (vol-scaled panel values x mean daily divisor of the week)"
        units_note = (
            "inverse_vol weighting: fitted, realized, contrib and const_contrib are panel values times the "
            "asset's mean daily divisor over the week (approximate return units; exact realised weekly "
            "returns are in meta['realized_exact']); R2 values are computed in panel (vol-scaled) units"
        )
    else:
        units = "return units (exact)"
        units_note = "no asset weighting: panel units are return units (weekly sums of daily returns)"
    sigma_c = np.asarray(fit.meta.get("sigma_c", np.ones(Gamma.shape[0])), dtype=float)
    norms = np.asarray(res.gamma_norms, dtype=float)
    timings = {
        "panel": float(panel.meta.get("timings", {}).get("total", float("nan"))),
        "fit": float(fit.meta.get("timings", {}).get("total", float("nan"))),
        "evaluate": time.perf_counter() - t_all,
    }
    meta: dict[str, Any] = {
        "units": units,
        "units_note": units_note,
        "asset_weighting": str(panel.pipeline_cfg.data.asset_weighting),
        "warnings": warnings_,
        "d52_note": D52_NOTE,
        "timings": timings,
        "factors": pd.DataFrame(factors, index=ends, columns=[f"f{k + 1}" for k in range(K)]),
        "fitted_panel": pd.DataFrame(fitted_p, index=ends, columns=a_index),
        "realized_panel": pd.DataFrame(realized_p, index=ends, columns=a_index),
        "contrib_weekly_panel": weekly,
        "divisor": pd.DataFrame(divisor, index=ends, columns=a_index),
        "realized_exact": exact_df,
        "period_first_day": first_day,
        "n_assets_per_period": pd.Series(n_per, index=ends, name="n_assets"),
        "evaluated_span": (pd.Timestamp(first_day[0]), pd.Timestamp(cal[codes == int(pos[-1])][-1])),
        "window_days_not_evaluated": pd.DatetimeIndex(not_evaluated),
        "oos_ridge": ridge,
        "shuffled_r2_pooled": r2_pooled_shuf,
        "shuffled_r2": pd.Series(r2_shuf, index=a_index, name="shuffled_r2"),
        "n_shuffles": int(N_SHUFFLES),
        "gamma_norms_standardized": pd.Series(
            sigma_c[1:] * norms[1:] if sigma_c.shape == norms.shape else np.full(n_l, np.nan),
            index=t_index, name="gamma_norm_std",
        ),
        "gamma_const_norm": float(norms[0]),
        "forecast_start": fs,
        "forecast_end": fe,
        "n_weeks": int(n_w),
        "last_train_period": last_train,
        "n_train_periods": int(len(fit.train_periods)),
        "lambda_rule": fit.meta.get("lambda_rule"),
        "tolerance": fit.meta.get("tolerance"),
        "n_selected": int(res.n_selected),
        "converged": bool(res.converged),
        "is_sharpe": fit.meta.get("is_sharpe"),
        "shock_window": int(panel.pipeline_cfg.shocks.window),
        "lead_days": int(panel.pipeline_cfg.data.attention_lag_days),
    }
    logger.info(
        "evaluate_bks: %d forecast week(s) %s .. %s, %d asset-weeks, pooled OOS R2 %.4f, median asset R2 %.4f "
        "(%.2fs)",
        n_w, ends[0].date(), ends[-1].date(), int(n_per.sum()), r2_pooled,
        float(np.nanmedian(r2)) if np.isfinite(r2).any() else float("nan"), timings["evaluate"],
    )
    return BKSLabResult(
        selected_topics=list(res.selected_topics),
        gamma_norms=pd.Series(norms[1:], index=t_index, name="gamma_norm"),
        lam=float(fit.lam),
        K=int(fit.K),
        path=None if fit.tuning is None else fit.tuning.path_frame(),
        periods=ends,
        fitted=fitted,
        realized=realized,
        r2=pd.Series(r2, index=a_index, name="r2"),
        r2_pooled=r2_pooled,
        contrib=pd.DataFrame(contrib, index=a_index, columns=t_index),
        const_contrib=pd.Series(const, index=a_index, name="const_contrib"),
        in_sample_total_r2=float(res.total_r2),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Stage 4: BKS-implied topic exposures (method comparison)
# ---------------------------------------------------------------------------
def implied_topic_covariance(
    C: np.ndarray, Gamma: np.ndarray, rcond: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Loadings and model-implied topic covariances of instrument rows (BKS Eq. 5).

    Symbols: ``c_i = [1, cov_i]`` an instrument row (a row of ``C``, in the
    units of ``IPCAPanel.X``), ``Gamma`` the ``((L+1) x K)`` training map in
    the same units (row 0 the constant, ``Gamma_tilde = Gamma[1:]`` the
    topics), ``beta_i = c_i Gamma`` the asset's ``K`` factor loadings.

    The model-implied covariance of the asset's (panel-unit) daily return
    with the raw topic shocks is

    ``m_i = Gamma_tilde (Gamma_tilde' Gamma_tilde)^+ beta_i'``   (an ``L``-vector).

    In words: BKS Eq. 5 gives ``cov_i = beta_i Sigma_ff A'`` with
    ``A = Gamma_tilde (Gamma_tilde' Gamma_tilde)^-1 Sigma_ff^-1``
    (:func:`narrative_ipca.wrapup.recover_A`), so
    ``A Sigma_ff beta_i' = Gamma_tilde (Gamma_tilde' Gamma_tilde)^-1 beta_i'``:
    the low-rank, sparse-topic reconstruction of the asset's topic
    covariances. ``Sigma_ff`` cancels. Topics the group lasso dropped have zero
    rows of ``Gamma_tilde`` and get ``m = 0`` exactly. With ``K = L`` and
    ``Gamma_tilde`` invertible, ``m_i = cov_i' + (Gamma_tilde')^-1 Gamma_0'``.

    Parameters
    ----------
    C:
        ``(n, L+1)`` instrument rows, constant in column 0.
    Gamma:
        ``(L+1, K)`` map of the fit (``SparseIPCAResult.Gamma``, original
        instrument units).
    rcond:
        Pseudo-inverse cut-off (``EvaluationConfig.rcond``): a singular value
        ``s`` of ``Gamma_tilde`` is kept when ``s^2 > rcond * s_max^2``, as in
        :func:`narrative_ipca.wrapup.recover_A`.

    Returns
    -------
    (m, beta, rank):
        ``m`` ``(n, L)``, ``beta`` ``(n, K)`` (equal to
        :func:`narrative_ipca.sparse_ipca.betas`), ``rank`` the numerical
        rank of ``Gamma_tilde``.
    """
    Gamma = np.asarray(Gamma, dtype=float)
    beta = ipca_betas(C, Gamma)
    M, rank = _lsq_map(Gamma[1:], float(rcond))  # (L, K): Gamma_tilde (Gamma_tilde' Gamma_tilde)^+
    return beta @ M.T, beta, int(rank)


def implied_exposures(
    panel: BKSPanel,
    fit: BKSFit,
    sim: SimData,
    shocks: ObservedShocks,
    select_tau: float = 0.05,
) -> DirectFit:
    """Topic exposures implied by the BKS training fit, in the direct estimator's units (G.7, G.8; D52, D65).

    The result is a :class:`DirectFit` on the same training pairs and the
    same observed shocks as :func:`narrative_ipca.exposure_lab.direct.fit_direct`,
    so the forecast-window evaluation (:func:`.evaluate.evaluate_window`,
    :func:`.evaluate.window_sweep`) and the recovery metrics apply unchanged
    and nothing is fitted inside the forecast window.

    Steps, per asset ``i`` (symbols of :func:`implied_topic_covariance`):

    1. ``c_i``: the asset's panel row in the last training week of ``fit``
       (the row pairs ``c_{i,T-1}``, measured at the end of the week before,
       with that week's return); if the asset has no row that week, its
       latest training row. An asset with no training row, no divisor (step
       3) or fewer than :data:`.direct.MIN_TRAIN_OBS` training pairs (the
       rule of :func:`.direct.fit_direct`) gets zero exposures and is listed
       in ``meta["skipped_assets"]``.
    2. ``beta_i = c_i Gamma`` and ``m_i = Gamma_tilde (Gamma_tilde' Gamma_tilde)^+ beta_i'``
       (BKS Eq. 5).
    3. Units back to returns: with ``asset_weighting = "inverse_vol"`` the
       panel's daily returns were divided by a trailing volatility
       (``AlignedData.scale``), so ``m_i`` is multiplied by the asset's mean
       divisor over its training return days (an approximation: the divisor
       varies over time and the kernel covariance also weighs days before the
       training window). Factor 1 for ``"none"``.
    4. Raw exposure ``b_i = Sigma_z^+ m_i``, with ``Sigma_z`` the covariance
       (``ddof = 0``) of the raw observed shocks ``z`` over the direct
       estimator's training shock days (:func:`.direct.training_pairs`);
       pseudo-inverse with cut-off ``rcond`` on the eigenvalues.
    5. Standardised: ``B_hat[k, i] = b_i[k] * sd_train(z_k) / sd_train(r_i)``,
       with ``sd_train(z_k) = shocks.scale`` and ``sd_train(r_i)`` computed
       exactly as :func:`.direct.fit_direct` does (returned as
       ``ret_scale``, with ``ret_mean``).

    Lead: the panel lags attention by ``l`` days (``attention_lag_days``), so
    its covariances pair ``r_{t+l}`` with ``z_t``, as the direct training
    pairs do.

    Parameters
    ----------
    panel:
        Output of :func:`build_bks_panel` for ``sim`` and ``shocks.window``.
    fit:
        Output of :func:`fit_bks` on ``panel``; its training weeks must end on
        or before ``shocks.train_end``.
    sim:
        The simulation (returns, calendar, topics, lead).
    shocks:
        Observed shocks of the lab's training window (:func:`.dgp.observed_shocks`).
    select_tau:
        A pair is selected when ``|B_hat| >= select_tau`` (the dense-method
        rule of G.7.1).

    Returns
    -------
    DirectFit
        ``method = "bks_implied"``, ``intercept`` 0, ``penalty`` ``NaN``,
        ``n_train`` the direct training count. ``meta`` holds the keys of
        :func:`.direct.fit_direct` (``select_tau``, ``last_return_day``, ...)
        plus ``K``, ``lam``, ``selected_topics`` (nonzero ``Gamma`` rows),
        ``train_period`` (last training week), ``row_period`` (per asset, the
        week of the row used), ``stale_assets`` (assets that used an earlier
        row), ``divisor`` (per asset), ``B_const`` (topics x assets: the part
        of ``B_hat`` that comes from the constant instrument, ``c = [1, 0, ..., 0]``;
        the same implied covariance for every asset before the unit
        conversion), ``gamma_const_norm`` (``||Gamma_0||``, 0 when the
        penalty dropped the constant), ``gamma_rank``, ``sigma_z_rank``,
        ``rcond``, ``asset_weighting``, ``units_note``, ``caveat``
        (:data:`IMPLIED_NOTE`), ``kernel_share_before_train`` (the share of
        the instruments' kernel weight on days before the training window,
        :func:`kernel_history_share`), ``bks_train_start``, ``bks_train_end`` and
        ``timings`` (``implied``, ``bks_panel``, ``bks_fit`` and ``total``,
        the sum: the cost of the whole BKS route).

    Raises
    ------
    ValueError
        When the panel's lead or shock window differ from ``sim`` and
        ``shocks``, the fit's training weeks end after ``shocks.train_end``
        (D65), or topics, assets or instruments do not match.
    """
    t0 = time.perf_counter()
    pnl = panel.panel
    res = fit.fit
    lead = int(sim.lead_days)
    topic_ids = sim.topics.ids
    asset_ids = [str(a) for a in sim.market.returns.columns]
    n_topics, n_assets = len(topic_ids), len(asset_ids)

    # consistency with the lab's direct route
    if int(panel.pipeline_cfg.data.attention_lag_days) != lead:
        raise ValueError(
            f"the BKS panel lags attention by {panel.pipeline_cfg.data.attention_lag_days} day(s) but the "
            f"simulation's lead is {lead}"
        )
    if int(panel.pipeline_cfg.shocks.window) != int(shocks.window):
        raise ValueError(
            f"the BKS panel uses shock window {panel.pipeline_cfg.shocks.window} but the shocks use {shocks.window}"
        )
    if list(res.instrument_names) != list(pnl.instrument_names):
        raise ValueError("fit and panel have different instruments; fit_bks must be run on this panel")
    if sorted(str(t) for t in pnl.topics) != sorted(topic_ids):
        raise ValueError("the BKS panel's topics differ from the simulation's")
    if sorted(str(a) for a in pnl.assets) != sorted(asset_ids):
        raise ValueError("the BKS panel's assets differ from the simulation's")
    last_train = pd.Timestamp(fit.train_periods.max())
    if last_train > pd.Timestamp(shocks.train_end):
        raise ValueError(
            f"the BKS fit's last training week ends {last_train.date()}, after the training end "
            f"{pd.Timestamp(shocks.train_end).date()} of the shocks (D65); refit BKS on the same training window"
        )

    # 1. instrument rows: the last training week, else the asset's latest training row
    periods = pd.DatetimeIndex(pnl.periods)
    train_pos = np.flatnonzero(np.asarray(periods.isin(fit.train_periods), dtype=bool))
    rows = np.flatnonzero(np.isin(pnl.t_idx, train_pos))
    rev = rows[::-1]  # t_idx is sorted, so the first hit per asset in reverse order is its latest row
    found_assets, first = np.unique(pnl.asset_idx[rev], return_index=True)
    last_rows = rev[first]
    panel_assets = [str(a) for a in pnl.assets]
    row_of = {panel_assets[int(a)]: int(r) for a, r in zip(found_assets, last_rows)}

    # training pairs and return scales exactly as fit_direct computes them
    cal = sim.market.calendar
    S_all = shock_matrix(shocks, cal, topic_ids)
    p_pos, q_pos = training_pairs(sim, shocks, S_all)
    Rtr = sim.market.returns.to_numpy(dtype=float)[q_pos]
    obs = np.isfinite(Rtr)
    n_train = obs.sum(axis=0)
    ret_mean, ret_scale = _train_moments(Rtr, obs)

    # 2. loadings and implied topic covariances (panel topic order -> simulation topic order)
    rcond = float(panel.pipeline_cfg.evaluation.rcond)
    Gamma = np.asarray(res.Gamma, dtype=float)
    col = {str(t): j for j, t in enumerate(pnl.topics)}
    order = np.array([col[k] for k in topic_ids], dtype=np.int64)
    a_pos = {a: j for j, a in enumerate(asset_ids)}
    fitted_ids = [a for a in asset_ids if a in row_of]
    idx_rows = np.array([row_of[a] for a in fitted_ids], dtype=np.int64)
    C = pnl.X[idx_rows] if len(idx_rows) else np.zeros((0, pnl.p))
    m_panel, _, gamma_rank = implied_topic_covariance(C, Gamma, rcond)
    m = np.zeros((n_assets, n_topics))
    m[np.array([a_pos[a] for a in fitted_ids], dtype=np.int64)] = m_panel[:, order]
    # the constant instrument's share of m_i, the same for every asset: m of the row c = [1, 0, ..., 0]
    e0 = np.zeros((1, pnl.p))
    e0[0, 0] = 1.0
    m_const = implied_topic_covariance(e0, Gamma, rcond)[0][0, order]

    # 3. units back to returns: mean divisor of the vol-scaled panel returns over the training return days
    scaled = panel.aligned.scale is not None
    divisor = np.ones(n_assets)
    if scaled:
        sc = panel.aligned.scale.reindex(index=cal, columns=asset_ids).to_numpy(dtype=float)[q_pos]
        ok = obs & np.isfinite(sc)
        cnt = ok.sum(axis=0)
        divisor = np.where(cnt > 0, np.where(ok, sc, 0.0).sum(axis=0) / np.maximum(cnt, 1), np.nan)
    # assets without a training row, a divisor or MIN_TRAIN_OBS training pairs (fit_direct's rule) get 0
    skipped = [
        a for j, a in enumerate(asset_ids)
        if a not in row_of or not np.isfinite(divisor[j]) or int(n_train[j]) < MIN_TRAIN_OBS
    ]
    zero = np.array([a in set(skipped) for a in asset_ids], dtype=bool)
    conv = np.where(zero, 0.0, np.where(np.isfinite(divisor), divisor, 0.0))
    m = m * conv[:, None]
    m0 = conv[:, None] * m_const[None, :]  # (N, L): the constant's part of m, in return units

    # 4. raw exposures b_i = Sigma_z^+ m_i on the training shock days
    Z = shocks.z.reindex(index=cal, columns=topic_ids).to_numpy(dtype=float)[p_pos]
    if Z.shape[0] >= 1:
        Zc = Z - Z.mean(axis=0)
        Sigma_z = (Zc.T @ Zc) / Z.shape[0]
        Sz_pinv, sz_rank = _sym_pinv(Sigma_z, rcond)
    else:
        Sz_pinv, sz_rank = np.zeros((n_topics, n_topics)), 0
    b_raw = m @ Sz_pinv  # (N, L); Sigma_z^+ is symmetric

    # 5. standardised units of the direct estimator
    z_scale = shocks.scale.reindex(topic_ids).to_numpy(dtype=float)
    std = z_scale[None, :] / ret_scale[:, None]  # (N, L): sd_train(z_k) / sd_train(r_i)
    B = (b_raw * std).T  # (L, N)
    B_const = ((m0 @ Sz_pinv) * std).T
    B[:, zero] = 0.0
    B_const[:, zero] = 0.0
    tau = float(select_tau)
    selected = np.abs(B) >= tau
    t_implied = time.perf_counter() - t0

    t_index = pd.Index(topic_ids, name="topic_id")
    a_index = pd.Index(asset_ids, name="asset_id")
    row_period = pd.Series(
        [periods[int(pnl.t_idx[row_of[a]])] if a in row_of else pd.NaT for a in asset_ids],
        index=a_index, name="row_period", dtype="datetime64[ns]",
    )
    stale = [a for a in asset_ids if a in row_of and row_period[a] != last_train]
    last_pair_day = cal[q_pos[-1]] if len(q_pos) else pd.NaT
    last_used = last_train if pd.isna(last_pair_day) else max(pd.Timestamp(last_pair_day), last_train)
    history_share = _history_share(
        pd.DatetimeIndex(panel.aligned.calendar), float(panel.pipeline_cfg.covariance.xi),
        int(panel.pipeline_cfg.covariance.skip_days), last_train, pd.Timestamp(shocks.train_start),
    )
    if scaled:
        units_note = (
            "inverse_vol weighting: the implied covariances are in vol-scaled units and are multiplied by the "
            "asset's mean daily divisor over its training return days (approximate: the divisor varies over "
            "time, and the kernel covariance also weighs days before the training window)"
        )
    else:
        units_note = "no asset weighting: the implied covariances are in return units (no conversion)"
    t_panel = float(panel.meta.get("timings", {}).get("total", 0.0))
    t_fit = float(fit.meta.get("timings", {}).get("total", 0.0))
    if skipped:
        logger.warning(
            "implied_exposures: %d asset(s) without a training row or divisor get zero exposures: %s",
            len(skipped), ", ".join(skipped[:10]),
        )
    meta: dict[str, Any] = {
        "method": IMPLIED_METHOD,
        "alpha_rule": "none",
        "select_tau": tau,
        "sparse": False,
        "l1_ratio": np.nan,
        "n_topics": n_topics,
        "n_assets": n_assets,
        "lead_days": lead,
        "shock_window": int(shocks.window),
        "n_pairs": int(len(p_pos)),
        "first_shock_day": cal[p_pos[0]] if len(p_pos) else pd.NaT,
        "last_return_day": last_used,
        "skipped_assets": skipped,
        "n_convergence_warnings": 0,
        "K": int(fit.K),
        "lam": float(fit.lam),
        "selected_topics": list(res.selected_topics),
        "n_selected_topics": int(res.n_selected),
        "train_period": last_train,
        "row_period": row_period,
        "stale_assets": stale,
        "divisor": pd.Series(divisor, index=a_index, name="divisor"),
        "B_const": pd.DataFrame(B_const, index=t_index, columns=a_index),
        "gamma_const_norm": float(np.linalg.norm(Gamma[0])),
        "gamma_rank": int(gamma_rank),
        "sigma_z_rank": int(sz_rank),
        "rcond": rcond,
        "asset_weighting": str(panel.pipeline_cfg.data.asset_weighting),
        "units_note": units_note,
        "caveat": IMPLIED_NOTE,
        "kernel_share_before_train": history_share,
        "bks_train_start": fit.meta.get("train_start"),
        "bks_train_end": fit.meta.get("train_end"),
        "timings": {"implied": t_implied, "bks_panel": t_panel, "bks_fit": t_fit, "total": t_implied + t_panel + t_fit},
    }
    logger.info(
        "implied_exposures: K=%d lambda=%.4g, %d/%d topics in Gamma (rank %d), rows of week %s, Sigma_z rank %d/%d, "
        "%d pairs selected at tau=%.3g (%.3fs)",
        fit.K, fit.lam, res.n_selected, n_topics, gamma_rank, last_train.date(), sz_rank, n_topics,
        int(selected.sum()), tau, t_implied,
    )
    return DirectFit(
        B_hat=pd.DataFrame(B, index=t_index, columns=a_index),
        intercept=pd.Series(np.zeros(n_assets), index=a_index, name="intercept"),
        ret_mean=pd.Series(ret_mean, index=a_index, name="ret_mean"),
        ret_scale=pd.Series(ret_scale, index=a_index, name="ret_scale"),
        selected=pd.DataFrame(selected, index=t_index, columns=a_index),
        penalty=pd.Series(np.full(n_assets, np.nan), index=a_index, name="penalty"),
        n_train=pd.Series(n_train.astype(np.int64), index=a_index, name="n_train"),
        method=IMPLIED_METHOD,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def run_bks(sim: SimData, cfg: BKSLabConfig, window: WindowConfig, progress: ProgressFn | None = None) -> BKSLabResult:
    """Panel, training fit and forecast-window evaluation in one call (G.7.2).

    Equivalent to :func:`build_bks_panel` with ``window.shock_window``, then
    :func:`fit_bks` on ``[window.train_start, window.train_end]``, then
    :func:`evaluate_bks`. Callers that vary only the fit or the forecast
    window should cache the panel (and the fit) and call the stages directly.
    """
    panel = build_bks_panel(sim, cfg, int(window.shock_window))
    fit = fit_bks(panel, cfg, window.train_end, progress=progress, train_start=window.train_start)
    return evaluate_bks(panel, fit, window)
