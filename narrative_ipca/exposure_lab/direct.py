"""Direct topic-to-asset exposure regression of the topic-exposure lab (DESIGN.md G.7.1; D64-D66).

The plan's direct arm (research plan v0.3 Section 5): per asset ``n``, on the
training pairs,

    rh_{n,t+l} = alpha_n + sum_k b_{k,n} sh_{k,t} + e_{n,t+l}

where ``sh_{k,t}`` is topic ``k``'s observed standardised shock on shock day
``t`` (:func:`~narrative_ipca.exposure_lab.dgp.observed_shocks`), ``l`` the
lead in days (D64), ``rh`` the asset's return standardised by its training
mean and standard deviation, ``alpha_n`` an intercept and ``b_{k,n}`` the
exposure in standardised units.

Pairing (G.6, D65)
------------------
Shock day ``t`` at calendar position ``p`` is paired with the return day at
position ``p + l``. A pair is a **training pair** when
``train_start <= t <= train_end``, the return day is on or before
``train_end`` (no return after ``train_end`` is ever used) and every topic's
``sh`` is finite at ``t``. Per asset, pairs with a missing return are
dropped; ``ret_mean`` and ``ret_scale`` are the mean and population standard
deviation (``ddof = 0``) of the remaining training returns (a zero scale is
replaced by 1).

Methods (G.7.1)
---------------
1. ``elastic_net`` (D66): scikit-learn ``ElasticNet`` with ``l1_ratio`` and
   penalty ``alpha`` by ``penalty``: ``universal``
   ``alpha = sqrt(2 ln max(L, 2) / n)`` with ``L`` the number of topics and
   ``n`` the asset's training pairs; ``fixed`` ``alpha = DirectConfig.alpha``;
   ``cv`` ``ElasticNetCV`` with ``TimeSeriesSplit(cv_folds)`` and a 30-point
   grid per asset.
2. ``ridge``: ``b = (X'X + n lam I)^-1 X'y`` on centred data, with ``X`` the
   ``n x L`` shock matrix and ``y`` the standardised returns; ``lam`` fixed or
   chosen per asset by generalised cross-validation (GCV) over
   ``logspace(-4, 1, 30)`` from the singular value decomposition of ``X``,
   with the intercept counted in the degrees of freedom (:func:`_fit_ridge`).
3. ``ols``: least squares with intercept; refused when ``L >= n / 2``.
4. ``oracle``: ``b = B_true`` (G.5.3), intercept 0, with the same training
   return scale as the estimators. A reference, not an estimator: the
   evaluation's oracle line uses exactly these exposures and scales (D74),
   so this method reproduces it.

A pair is **selected** when ``b != 0`` (elastic net) or ``|b| >= select_tau``
(ridge, ols, oracle).

Validity boundaries
-------------------
* Assets are grouped by their set of valid training rows; each group is one
  fit with a 2-D target (the targets are independent: the elastic net and
  ridge solve per column). With complete data there is one group.
* An asset with fewer than :data:`MIN_TRAIN_OBS` training pairs is not fitted:
  ``B_hat`` is 0 and it is listed in ``meta["skipped_assets"]``.
* The estimator is deterministic; no random draws are made here (D71).
* Dense linear algebra uses ``scipy.linalg`` (Cholesky and ``gesdd``); the
  multi-threaded OpenBLAS bundled with numpy 2.4 is slow on LU and SVD of a
  few hundred columns (the same issue noted in :mod:`.dgp`).
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg as sla
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

from .config import DirectConfig
from .dgp import observation_groups as _mask_groups
from .types import DirectFit, ObservedShocks, SimData, SimTruth

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_TRAIN_OBS",
    "EN_TOL",
    "CV_N_ALPHAS",
    "RIDGE_GCV_GRID",
    "SPARSE_METHODS",
    "fit_direct",
    "training_pairs",
    "universal_alpha",
    "shock_matrix",
]

#: Minimum training pairs for an asset to be fitted.
MIN_TRAIN_OBS = 20

#: Coordinate-descent tolerance of the elastic net.
EN_TOL = 1e-4

#: Number of penalty values on the ``ElasticNetCV`` grid (``penalty = "cv"``).
CV_N_ALPHAS = 30

#: Ridge penalty grid searched by GCV when ``ridge_lambda`` is ``None``.
RIDGE_GCV_GRID: np.ndarray = np.logspace(-4, 1, 30)

#: Methods whose selection rule is ``b != 0``.
SPARSE_METHODS: frozenset[str] = frozenset({"elastic_net"})


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def fit_direct(sim: SimData, shocks: ObservedShocks, cfg: DirectConfig, truth: SimTruth) -> DirectFit:
    """Fit the direct exposure regression on the training pairs (G.7.1, D65, D66).

    Parameters
    ----------
    sim:
        Simulation output; supplies the returns ``r`` (``sim.market.returns``),
        the topic order (``sim.topics``) and the lead ``l`` (``sim.lead_days``).
    shocks:
        Observed shocks; ``s_hat`` is the regressor ``sh`` and
        ``train_start`` / ``train_end`` define the training pairs.
    cfg:
        Method, penalty rule and values, selection threshold ``select_tau``.
    truth:
        Population truth for ``shocks.window``; only the ``oracle`` method
        uses it (``B_hat = truth.B_true``).

    Returns
    -------
    DirectFit
        ``B_hat`` and ``selected`` are topics x assets; ``intercept``,
        ``ret_mean``, ``ret_scale``, ``penalty`` and ``n_train`` are per asset.
        ``meta`` holds ``method``, ``alpha_rule``, ``select_tau``,
        ``n_topics``, ``n_assets``, ``lead_days``, ``n_pairs`` (training pairs
        before dropping missing returns), ``first_shock_day``,
        ``last_return_day``, ``n_groups``, ``skipped_assets``,
        ``n_convergence_warnings`` and ``timings`` (seconds); ``gcv_grid``,
        ``gcv_at_boundary`` (assets whose lambda is at either end of the grid)
        and ``gcv_at_lower_edge`` (at the smallest lambda, so close to OLS)
        for ridge with GCV; ``cv_folds`` for ``cv``.

    Raises
    ------
    ValueError
        ``ols`` with ``L >= n / 2`` for some fitted asset; ``penalty =
        "fixed"`` with ``alpha = 0``; topics of ``shocks`` or ``truth`` not
        matching ``sim``.
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    topic_ids = sim.topics.ids
    asset_ids = [str(a) for a in sim.market.returns.columns]
    n_topics, n_assets = len(topic_ids), len(asset_ids)
    lead = int(sim.lead_days)
    method = str(cfg.method)

    # training pairs and per-asset standardisation
    t = time.perf_counter()
    S_all = shock_matrix(shocks, sim.market.calendar, topic_ids)
    p_pos, q_pos = training_pairs(sim, shocks, S_all)
    n_pairs = len(p_pos)
    if n_pairs == 0:
        logger.warning("fit_direct: no training pairs in [%s, %s]", shocks.train_start.date(), shocks.train_end.date())
    X = S_all[p_pos]
    Rtr = sim.market.returns.to_numpy(dtype=float)[q_pos]
    obs = np.isfinite(Rtr)
    n_train = obs.sum(axis=0)
    ret_mean, ret_scale = _train_moments(Rtr, obs)
    Y = np.where(obs, (Rtr - ret_mean) / ret_scale, np.nan)
    groups = _mask_groups(obs)
    timings["prepare"] = time.perf_counter() - t

    B = np.zeros((n_topics, n_assets))
    intercept = np.zeros(n_assets)
    penalty = np.full(n_assets, np.nan)
    skipped: list[str] = []
    n_conv = 0
    meta_extra: dict[str, Any] = {}

    fit_groups = []
    for rows, cols in groups:
        n_g = int(rows.sum())
        if n_g < MIN_TRAIN_OBS:
            skipped.extend(asset_ids[j] for j in cols)
            continue
        fit_groups.append((rows, cols, n_g))
    if skipped:
        logger.warning(
            "fit_direct: %d assets with fewer than %d training pairs are not fitted (B_hat = 0): %s",
            len(skipped), MIN_TRAIN_OBS, ", ".join(skipped[:10]),
        )

    if method == "ols":
        too_short = [(asset_ids[j], n_g) for rows, cols, n_g in fit_groups for j in cols if n_topics >= n_g / 2.0]
        if too_short:
            raise ValueError(
                f"ols refused: L = {n_topics} topics >= n_train / 2 for {len(too_short)} assets "
                f"(e.g. {too_short[0][0]} with n_train = {too_short[0][1]}); use elastic_net or ridge"
            )
    if method == "elastic_net" and cfg.penalty == "fixed" and float(cfg.alpha) <= 0.0:
        raise ValueError("penalty 'fixed' needs alpha > 0; use method 'ols' for an unpenalised fit")

    t = time.perf_counter()
    if method == "oracle":
        if truth.shock_window != shocks.window:
            logger.warning(
                "fit_direct: oracle truth refers to shock window %d but the shocks use %d",
                truth.shock_window, shocks.window,
            )
        Bt = truth.B_true.reindex(index=topic_ids, columns=asset_ids)
        if Bt.isna().to_numpy().any():
            raise ValueError("fit_direct: truth.B_true does not cover the simulation's topics and assets")
        B = Bt.to_numpy(dtype=float).copy()
        alpha_rule = "none"
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            for rows, cols, n_g in fit_groups:
                Xg, Yg = X[rows], Y[np.ix_(rows, cols)]
                if method == "elastic_net":
                    if cfg.penalty == "cv":
                        b, c, pen = _fit_enet_cv(Xg, Yg, cfg)
                    else:
                        alpha = universal_alpha(n_topics, n_g) if cfg.penalty == "universal" else float(cfg.alpha)
                        b, c = _fit_enet(Xg, Yg, alpha, cfg)
                        pen = np.full(len(cols), alpha)
                elif method == "ridge":
                    b, c, pen, at_edge = _fit_ridge(Xg, Yg, cfg.ridge_lambda)
                    meta_extra["gcv_at_boundary"] = meta_extra.get("gcv_at_boundary", 0) + at_edge
                elif method == "ols":
                    b, c = _fit_ols(Xg, Yg)
                    pen = np.full(len(cols), np.nan)
                else:  # pragma: no cover - DirectConfig validates the method
                    raise ValueError(f"unknown method {method!r}")
                B[:, cols] = b
                intercept[cols] = c
                penalty[cols] = pen
        n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        for w in caught:
            if not issubclass(w.category, ConvergenceWarning):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        if n_conv:
            logger.warning("fit_direct: %d elastic-net fits did not converge (max_iter = %d)", n_conv, cfg.max_iter)
        if method == "elastic_net":
            alpha_rule = str(cfg.penalty)
        elif method == "ridge":
            alpha_rule = "gcv" if cfg.ridge_lambda is None else "fixed"
            if cfg.ridge_lambda is None:
                meta_extra["gcv_grid"] = RIDGE_GCV_GRID.copy()
                meta_extra["gcv_at_lower_edge"] = int(np.sum(penalty == RIDGE_GCV_GRID[0]))
        else:
            alpha_rule = "none"
        if method == "elastic_net" and cfg.penalty == "cv":
            meta_extra["cv_folds"] = int(cfg.cv_folds)
    timings["fit"] = time.perf_counter() - t

    tau = float(cfg.select_tau)
    selected = (B != 0.0) if method in SPARSE_METHODS else (np.abs(B) >= tau)
    t_index = pd.Index(topic_ids, name="topic_id")
    a_index = pd.Index(asset_ids, name="asset_id")
    timings["total"] = time.perf_counter() - t_start
    cal = sim.market.calendar
    meta: dict[str, Any] = {
        "method": method,
        "alpha_rule": alpha_rule,
        "select_tau": tau,
        "sparse": method in SPARSE_METHODS,
        "l1_ratio": float(cfg.l1_ratio) if method == "elastic_net" else np.nan,
        "n_topics": n_topics,
        "n_assets": n_assets,
        "lead_days": lead,
        "shock_window": int(shocks.window),
        "n_pairs": int(n_pairs),
        "first_shock_day": cal[p_pos[0]] if n_pairs else pd.NaT,
        "last_return_day": cal[q_pos[-1]] if n_pairs else pd.NaT,
        "n_groups": len(groups),
        "skipped_assets": skipped,
        "n_convergence_warnings": int(n_conv),
        "timings": timings,
        **meta_extra,
    }
    logger.info(
        "fit_direct: %s (%s) on %d pairs x %d topics x %d assets, lead %d, %d groups, %d selected (%.2fs)",
        method, alpha_rule, n_pairs, n_topics, n_assets, lead, len(groups), int(selected.sum()), timings["total"],
    )
    return DirectFit(
        B_hat=pd.DataFrame(B, index=t_index, columns=a_index),
        intercept=pd.Series(intercept, index=a_index, name="intercept"),
        ret_mean=pd.Series(ret_mean, index=a_index, name="ret_mean"),
        ret_scale=pd.Series(ret_scale, index=a_index, name="ret_scale"),
        selected=pd.DataFrame(selected, index=t_index, columns=a_index),
        penalty=pd.Series(penalty, index=a_index, name="penalty"),
        n_train=pd.Series(n_train.astype(np.int64), index=a_index, name="n_train"),
        method=method,
        meta=meta,
    )


def universal_alpha(n_topics: int, n_obs: int) -> float:
    """Universal penalty ``alpha = sqrt(2 ln max(L, 2) / n)`` (G.7.1, D66).

    ``L`` is the number of topics and ``n`` the number of training pairs. With
    unit-variance regressors and noise it keeps the expected number of noise
    topics entering the lasso near zero and grows slowly with ``L``.
    """
    return float(np.sqrt(2.0 * np.log(max(int(n_topics), 2)) / max(int(n_obs), 1)))


def shock_matrix(shocks: ObservedShocks, calendar: pd.DatetimeIndex, topic_ids: list[str]) -> np.ndarray:
    """``s_hat`` as a ``(n_days, L)`` array on ``calendar`` with columns in ``topic_ids`` order.

    Raises ``ValueError`` when ``s_hat`` lacks one of the topics. Days of the
    calendar missing from ``s_hat`` are ``NaN``.
    """
    s = shocks.s_hat
    missing = [k for k in topic_ids if k not in s.columns]
    if missing:
        raise ValueError(f"shocks.s_hat lacks {len(missing)} topics of the simulation, e.g. {missing[:5]}")
    if not s.index.equals(calendar):
        s = s.reindex(index=calendar)
    if list(s.columns) != list(topic_ids):
        s = s.reindex(columns=topic_ids)
    return s.to_numpy(dtype=float)


def training_pairs(
    sim: SimData, shocks: ObservedShocks, S: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Calendar positions ``(p, p + l)`` of the training pairs (G.6, D65).

    Shock day ``t = calendar[p]`` is kept when ``train_start <= t <=
    train_end``, the return day ``calendar[p + l]`` is on or before
    ``train_end`` and every topic's ``s_hat`` is finite at ``t``. Missing
    returns are handled per asset by :func:`fit_direct`.

    Parameters
    ----------
    sim, shocks:
        As in :func:`fit_direct`.
    S:
        Optional ``s_hat`` array from :func:`shock_matrix` (avoids a rebuild).

    Returns
    -------
    (shock_positions, return_positions):
        Integer arrays of equal length, increasing.
    """
    cal = sim.market.calendar
    lead = int(sim.lead_days)
    if S is None:
        S = shock_matrix(shocks, cal, sim.topics.ids)
    n_days = len(cal)
    p = np.arange(max(n_days - lead, 0))
    q = p + lead
    ts, te = pd.Timestamp(shocks.train_start), pd.Timestamp(shocks.train_end)
    shock_days, return_days = cal[p], cal[q]
    keep = (
        np.asarray(shock_days >= ts)
        & np.asarray(shock_days <= te)
        & np.asarray(return_days <= te)
        & np.isfinite(S[p]).all(axis=1)
    )
    return p[keep], q[keep]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _train_moments(R: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and population std over observed rows; scale 0 or undefined -> 1 (mean undefined -> NaN)."""
    n = obs.sum(axis=0)
    R0 = np.where(obs, R, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, R0.sum(axis=0) / n, np.nan)
        dev = np.where(obs, R - mean, 0.0)
        sd = np.where(n > 0, np.sqrt((dev**2).sum(axis=0) / n), np.nan)
    scale = np.where(np.isfinite(sd) & (sd > 0.0), sd, 1.0)
    n_fixed = int(((n > 0) & ~(np.isfinite(sd) & (sd > 0.0))).sum())
    if n_fixed:
        logger.warning("fit_direct: %d assets with zero training volatility; ret_scale set to 1", n_fixed)
    return mean, scale


def _fit_enet(X: np.ndarray, Y: np.ndarray, alpha: float, cfg: DirectConfig) -> tuple[np.ndarray, np.ndarray]:
    """``ElasticNet`` with a 2-D target (independent per column): ``(coef L x m, intercept m)``."""
    m = Y.shape[1]
    model = ElasticNet(
        alpha=float(alpha), l1_ratio=float(cfg.l1_ratio), fit_intercept=True,
        max_iter=int(cfg.max_iter), tol=EN_TOL,
    )
    model.fit(X, Y)
    coef = np.asarray(model.coef_, dtype=float).reshape(m, X.shape[1]).T
    icpt = np.asarray(model.intercept_, dtype=float).reshape(m)
    return coef, icpt


def _fit_enet_cv(X: np.ndarray, Y: np.ndarray, cfg: DirectConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``ElasticNetCV`` per column with ``TimeSeriesSplit(cv_folds)`` and a :data:`CV_N_ALPHAS`-point grid."""
    m = Y.shape[1]
    coef = np.zeros((X.shape[1], m))
    icpt = np.zeros(m)
    pen = np.full(m, np.nan)
    for j in range(m):
        model = ElasticNetCV(
            l1_ratio=float(cfg.l1_ratio), alphas=CV_N_ALPHAS, cv=TimeSeriesSplit(n_splits=int(cfg.cv_folds)),
            fit_intercept=True, max_iter=int(cfg.max_iter), tol=EN_TOL,
        )
        model.fit(X, Y[:, j])
        coef[:, j] = model.coef_
        icpt[j] = float(model.intercept_)
        pen[j] = float(model.alpha_)
    return coef, icpt, pen


def _centre(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xbar, ybar = X.mean(axis=0), Y.mean(axis=0)
    return X - xbar, Y - ybar, xbar, ybar


def _fit_ridge(
    X: np.ndarray, Y: np.ndarray, lam: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Ridge on centred data: ``b = (X'X + n lam I)^-1 X'y``; ``lam`` per column by GCV when ``None``.

    GCV with the thin SVD ``X = U diag(d) V'`` (``d`` the singular values):
    for penalty ``lam`` the fitted values shrink the component along ``U_i``
    by ``d_i^2 / (d_i^2 + n lam)``, the effective degrees of freedom are
    ``df = 1 + sum_i d_i^2 / (d_i^2 + n lam)`` (the 1 counts the
    intercept), and ``GCV(lam) = (RSS(lam) / n) / (1 - df / n)^2`` with
    ``RSS`` the residual sum of squares; a grid point with ``df >= n`` gets
    ``GCV = inf``, and an asset with no finite GCV gets the largest ``lam``.
    Without the intercept's degree of freedom, a window with ``L >= n - 1``
    topics (20 topics on a one-month window of 21 days) interpolates the
    centred returns as ``lam -> 0``, GCV tends to 0 there, and the grid's
    lower edge was chosen for every asset (review of 2026-09-29: median OOS
    R2 below -1,000%). Returns ``(coef L x m, intercept m, lam m, n at grid
    edge)``.
    """
    n, L = X.shape
    m = Y.shape[1]
    Xc, Yc, xbar, ybar = _centre(X, Y)
    if lam is not None:
        lam = float(lam)
        if lam > 0.0:
            G = Xc.T @ Xc
            G[np.diag_indices_from(G)] += n * lam
            coef = sla.cho_solve(sla.cho_factor(G, lower=True), Xc.T @ Yc)
        else:
            coef = sla.lstsq(Xc, Yc, lapack_driver="gelsd")[0]
        return coef, ybar - xbar @ coef, np.full(m, lam), 0
    U, d, Vt = sla.svd(Xc, full_matrices=False, lapack_driver="gesdd")
    d2 = d**2
    Uty = U.T @ Yc  # (r, m)
    rss_perp = (Yc**2).sum(axis=0) - (Uty**2).sum(axis=0)  # part of y outside the column space of X
    nl = n * RIDGE_GCV_GRID[:, None]  # (G, 1)
    shrink_resid = nl / (d2[None, :] + nl)  # (G, r): share of each component left in the residual
    rss = np.maximum(rss_perp[None, :], 0.0) + (shrink_resid**2) @ (Uty**2)  # (G, m)
    df = 1.0 + (d2[None, :] / (d2[None, :] + nl)).sum(axis=1)  # (G,): the intercept counts one
    ok = df < n
    denom = np.where(ok, (1.0 - df / n) ** 2, 1.0)
    gcv = np.where(ok[:, None], (rss / n) / denom[:, None], np.inf)
    gcv = np.where(np.isfinite(gcv), gcv, np.inf)  # np.argmin would pick a NaN
    best = np.argmin(gcv, axis=0)  # (m,)
    none_finite = ~np.isfinite(gcv[best, np.arange(m)])
    best = np.where(none_finite, len(RIDGE_GCV_GRID) - 1, best)  # no finite GCV: the most shrinkage
    lam_best = RIDGE_GCV_GRID[best]
    factors = d[None, :] / (d2[None, :] + n * lam_best[:, None])  # (m, r)
    coef = Vt.T @ (factors.T * Uty)
    at_edge = int(np.sum((best == 0) | (best == len(RIDGE_GCV_GRID) - 1)))
    return coef, ybar - xbar @ coef, lam_best, at_edge


def _fit_ols(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with intercept (centred data, ``gelsd``)."""
    Xc, Yc, xbar, ybar = _centre(X, Y)
    coef = sla.lstsq(Xc, Yc, lapack_driver="gelsd")[0]
    return coef, ybar - xbar @ coef
