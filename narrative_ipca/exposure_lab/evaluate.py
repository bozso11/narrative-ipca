"""Out-of-sample evaluation of the direct estimator in the forecast window (DESIGN.md G.8; D65, D67-D69).

Symbols (G.8): ``H`` the return days of the forecast window, ``l`` the lead in
days, ``r_{n,t+l}`` asset ``n``'s daily return, ``sh_{k,t}`` topic ``k``'s
observed standardised shock on the matching shock day ``t`` (calendar
position of the return day minus ``l``), ``b_{k,n}`` the frozen training
exposure (``DirectFit.B_hat``) and ``sd_train(r_n)`` the training standard
deviation of the return (``DirectFit.ret_scale``).

* Topic-explained return (no intercept):
  ``rhat_{n,t+l} = sd_train(r_n) * sum_k b_{k,n} sh_{k,t}``. The oracle is
  the same formula with ``b = B_true`` and the same training scales
  ``sd_train(r_n)`` and ``sd_train(z_k)`` (D74): no forecast-window data
  enters it, and the direct method ``oracle`` reproduces it exactly.
* A cell ``(t, n)`` is **valid** when the return is finite and every topic's
  shock on the matching shock day is finite. Every sum below runs over asset
  ``n``'s valid days.
* OOS R2 (D67): ``R2_n = 1 - sum (r - rhat)^2 / sum r^2`` (uncentered; can be
  negative; ``NaN`` when ``sum r^2 = 0`` or no valid day).
* OOS correlation: Pearson correlation of ``r_n`` and ``sh_k`` (``NaN`` with
  fewer than 3 valid days or a constant series).
* Contribution (D68): ``c_{k,n} = sd_train(r_n) b_{k,n} sum_t sh_{k,t}``;
  realised move ``sum_t r``; explained move ``sum_k c_{k,n}``; residual
  ``realised - explained``. The identity holds exactly. Because each shock is
  attention minus its trailing mean, the window sum of the shocks depends
  mostly on the attention level at the window's edges and largely cancels
  (D76), so the contribution understates the topics' day-to-day role.
* Variance share (the dashboard's default view, D76):
  ``v_{k,n} = sd_train(r_n) b_{k,n} sum_t sh_{k,t} r_{n,t+l} / sum_t r^2``.
* Recovery (G.8 point 5, D63): training fit against the truth over all
  topic-asset pairs.

Validity boundaries
-------------------
* The estimates, the shock scale and the return scale come from the training
  window only (D65); nothing here re-fits. :func:`evaluate_window` and
  :func:`window_sweep` raise ``ValueError`` when the forecast window starts
  on or before the training end of the shocks or of the fit.
* ``B_true`` is defined with full-sample standardisation (G.5.3); the oracle
  and the recovery metrics treat it as an exposure in the estimator's
  training-standardised units (D74). The two standardisations differ by a
  few percent per asset and topic.
* Assets with no valid day in the window get ``NaN`` for every per-asset
  quantity.
* Pooled R2 in :func:`window_sweep` sums squared returns across assets in
  return units (the IPCA total-R2 convention), so high-volatility assets
  weigh more; the median over assets does not.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import WindowConfig
from .types import DirectFit, ObservedShocks, SimData, SimTruth, WindowEval

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_CORR_DAYS",
    "SWEEP_COLUMNS",
    "window_return_days",
    "evaluate_window",
    "recovery_metrics",
    "window_sweep",
    "median_finite",
]

#: Minimum valid days for an OOS correlation (G.8 point 2).
MIN_CORR_DAYS = 3

#: Columns of :func:`window_sweep`.
SWEEP_COLUMNS: tuple[str, ...] = (
    "start", "end", "n_days", "median_r2", "median_r2_oracle", "pooled_r2", "pooled_r2_oracle",
)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def window_return_days(calendar: pd.DatetimeIndex, window: WindowConfig) -> pd.DatetimeIndex:
    """Return days of the forecast window: calendar days in ``[forecast_start, forecast_end]`` (G.6).

    ``forecast_end = forecast_start + 7 * forecast_weeks - 1`` calendar days
    (:attr:`WindowConfig.forecast_end`).
    """
    cal = pd.DatetimeIndex(calendar)
    fs, fe = pd.Timestamp(window.forecast_start), window.forecast_end
    return cal[np.asarray((cal >= fs) & (cal <= fe))]


def evaluate_window(
    sim: SimData, shocks: ObservedShocks, fit: DirectFit, window: WindowConfig, truth: SimTruth
) -> WindowEval:
    """Evaluate the frozen training fit on the forecast window (G.8 points 1-5; D67, D68).

    Parameters
    ----------
    sim:
        Simulation output (returns, calendar, lead ``l``).
    shocks:
        Observed shocks with the training-window scale; ``s_hat`` on the
        matching shock days is the regressor.
    fit:
        The direct fit (:func:`~narrative_ipca.exposure_lab.direct.fit_direct`).
    window:
        Supplies ``forecast_start`` and ``forecast_weeks``.
    truth:
        Population truth for the shock window of ``shocks`` (``B_true`` gives
        the oracle, with the fit's training return scale; D74).

    Returns
    -------
    WindowEval
        Asset x topic frames ``corr``, ``contrib``, ``contrib_true``,
        ``var_share``, ``var_share_true``; per-asset ``r2``, ``r2_oracle``,
        ``realized``, ``explained``, ``explained_true``, ``residual``; the
        daily ``fitted``, ``fitted_oracle`` (``NaN`` on invalid cells) and
        ``realized_daily`` frames indexed by the return days; ``recovery``
        from :func:`recovery_metrics` with ``tau = fit.meta["select_tau"]``.

    Raises
    ------
    ValueError
        When the forecast window is not after the training window (D65).
    """
    days = window_return_days(sim.market.calendar, window)
    _check_out_of_sample(shocks, fit, window, days)
    if len(days) == 0:
        logger.warning(
            "evaluate_window: no return day in [%s, %s]; the evaluation is empty",
            pd.Timestamp(window.forecast_start).date(), window.forecast_end.date(),
        )
    wa = _window_arrays(sim, shocks, fit, truth, days)
    O = wa.valid.astype(float)
    n_valid = O.sum(axis=0)
    has = n_valid > 0
    r0 = np.where(wa.valid, wa.r, 0.0)
    S0 = wa.S0

    sum_s = O.T @ S0  # (N, L): sum of sh_k over asset n's valid days
    sum_rs = r0.T @ S0  # (N, L): sum of r_n sh_k
    sum_r2 = (r0**2).sum(axis=0)
    realized = r0.sum(axis=0)

    load = wa.scale[:, None] * wa.B.T  # (N, L): sd_train(r_n) * b_{k,n}
    load_true = wa.scale[:, None] * wa.B_true.T  # the oracle uses the same training scale (D74)
    contrib = load * sum_s
    contrib_true = load_true * sum_s
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.where(sum_r2 > 0.0, sum_r2, np.nan)[:, None]
        var_share = load * sum_rs / denom
        var_share_true = load_true * sum_rs / denom
    explained = contrib.sum(axis=1)
    explained_true = contrib_true.sum(axis=1)

    r2 = _r2(wa.r, wa.fitted, wa.valid)
    r2_oracle = _r2(wa.r, wa.fitted_oracle, wa.valid)
    corr = _masked_corr(wa.r, S0, wa.valid)

    # assets without a valid day: every per-asset quantity is undefined
    for arr in (realized, explained, explained_true):
        arr[~has] = np.nan
    for arr in (contrib, contrib_true, var_share, var_share_true):
        arr[~has, :] = np.nan
    residual = realized - explained

    a_index, t_index = wa.assets, wa.topics

    def per_asset(x: np.ndarray, name: str) -> pd.Series:
        return pd.Series(x, index=a_index, name=name)

    def asset_topic(x: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(x, index=a_index, columns=t_index)

    def daily(x: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(x, index=days, columns=a_index)

    return WindowEval(
        return_days=days,
        corr=asset_topic(corr),
        r2=per_asset(r2, "r2"),
        r2_oracle=per_asset(r2_oracle, "r2_oracle"),
        contrib=asset_topic(contrib),
        contrib_true=asset_topic(contrib_true),
        var_share=asset_topic(var_share),
        var_share_true=asset_topic(var_share_true),
        realized=per_asset(realized, "realized"),
        explained=per_asset(explained, "explained"),
        explained_true=per_asset(explained_true, "explained_true"),
        residual=per_asset(residual, "residual"),
        fitted=daily(np.where(wa.valid, wa.fitted, np.nan)),
        fitted_oracle=daily(np.where(wa.valid, wa.fitted_oracle, np.nan)),
        realized_daily=daily(wa.r),
        recovery=recovery_metrics(fit, truth, tau=float(fit.meta.get("select_tau", 0.05))),
        n_days=int(len(days)),
    )


def recovery_metrics(fit: DirectFit, truth: SimTruth, tau: float) -> dict[str, float]:
    """Recovery of the true exposures by the training fit, over all topic-asset pairs (G.8 point 5, D63).

    Symbols: ``linked`` the design-linked pairs (``W_unscaled != 0``),
    ``selected`` the fit's selection (``fit.selected``), ``exposed`` the truly
    exposed pairs (``|B_true| >= tau``).

    Returns
    -------
    dict
        ``n_linked``; ``coverage`` the share of linked pairs that are
        selected; ``sign_agreement`` the share of linked and selected pairs
        with ``sign(B_hat) == sign(B_true)`` (``NaN`` if there are none);
        ``mcc`` the Matthews correlation of ``selected`` against ``exposed``
        over all pairs (``NaN`` when a margin is empty); ``spearman`` the rank
        correlation of ``B_hat`` with ``B_true`` over all pairs; ``rmse`` the
        root mean squared difference ``B_hat - B_true``; ``n_selected``;
        ``n_truly_exposed``.
    """
    topics, assets = fit.B_hat.index, fit.B_hat.columns
    B = fit.B_hat.to_numpy(dtype=float)
    Bt = truth.B_true.reindex(index=topics, columns=assets).to_numpy(dtype=float)
    W = truth.W_unscaled.reindex(index=topics, columns=assets).fillna(0.0).to_numpy(dtype=float)
    sel = fit.selected.reindex(index=topics, columns=assets).fillna(False).to_numpy(dtype=bool)
    linked = W != 0.0
    exposed = np.abs(Bt) >= float(tau)

    n_linked = int(linked.sum())
    coverage = float((linked & sel).sum() / n_linked) if n_linked else np.nan
    ls = linked & sel
    sign_agreement = float((np.sign(B[ls]) == np.sign(Bt[ls])).mean()) if ls.any() else np.nan

    tp = float((sel & exposed).sum())
    fp = float((sel & ~exposed).sum())
    fn = float((~sel & exposed).sum())
    tn = float((~sel & ~exposed).sum())
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = float((tp * tn - fp * fn) / np.sqrt(den)) if den > 0 else np.nan

    finite = np.isfinite(B) & np.isfinite(Bt)
    x, y = B[finite].ravel(), Bt[finite].ravel()
    spearman = np.nan
    if x.size >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spearman = float(spearmanr(x, y).statistic)
    rmse = float(np.sqrt(np.mean((x - y) ** 2))) if x.size else np.nan
    return {
        "n_linked": float(n_linked),
        "coverage": coverage,
        "sign_agreement": sign_agreement,
        "mcc": mcc,
        "spearman": spearman,
        "rmse": rmse,
        "n_selected": float(sel.sum()),
        "n_truly_exposed": float(exposed.sum()),
    }


def window_sweep(
    sim: SimData,
    shocks: ObservedShocks,
    fit: DirectFit,
    window: WindowConfig,
    truth: SimTruth,
    max_windows: int = 150,
) -> pd.DataFrame:
    """OOS R2 over consecutive non-overlapping windows with the training fit frozen (G.8 point 6).

    Window ``i = 0, 1, ...`` covers the calendar days
    ``[forecast_start + i * 7 * weeks, forecast_start + (i + 1) * 7 * weeks - 1]``
    (``weeks = window.forecast_weeks``); windows are kept while their last
    calendar day is on or before the last day of the data, at most
    ``max_windows``. Window 0 is the forecast window of :func:`evaluate_window`.

    Returns
    -------
    DataFrame
        One row per window with ``start``, ``end`` (last calendar day,
        inclusive), ``n_days`` (return days), ``median_r2`` and
        ``median_r2_oracle`` (median over assets of the per-asset OOS R2),
        ``pooled_r2`` and ``pooled_r2_oracle`` (one R2 over all valid
        asset-days, in return units). Empty (with these columns) when no
        complete window fits.

    Raises
    ------
    ValueError
        When the forecast window is not after the training window (D65).
    """
    cal = sim.market.calendar
    length = 7 * int(window.forecast_weeks)
    fs = pd.Timestamp(window.forecast_start)
    _check_out_of_sample(shocks, fit, window, window_return_days(cal, window))
    n_win = 0
    if len(cal):
        n_win = max(0, ((cal[-1] - fs).days + 1) // length)
    n_win = min(n_win, int(max_windows))
    if n_win == 0:
        logger.warning("window_sweep: no complete %d-week window from %s", window.forecast_weeks, fs.date())
        dtypes = {"start": "datetime64[ns]", "end": "datetime64[ns]", "n_days": "int64"}
        return pd.DataFrame({c: pd.Series(dtype=dtypes.get(c, "float64")) for c in SWEEP_COLUMNS})

    last = fs + pd.Timedelta(days=n_win * length - 1)
    days = cal[np.asarray((cal >= fs) & (cal <= last))]
    wa = _window_arrays(sim, shocks, fit, truth, days)
    wid = np.asarray((days - fs).days // length, dtype=np.int64)

    starts = fs + pd.to_timedelta(np.arange(n_win) * length, unit="D")
    out = {
        "start": starts,
        "end": starts + pd.Timedelta(days=length - 1),
        "n_days": np.bincount(wid, minlength=n_win).astype(np.int64),
    }
    for suffix, fitted in (("", wa.fitted), ("_oracle", wa.fitted_oracle)):
        valid = wa.valid & np.isfinite(fitted)
        err2 = np.where(valid, (wa.r - fitted) ** 2, 0.0)
        tot2 = np.where(valid, wa.r**2, 0.0)
        cnt = valid.astype(np.int64)
        ssr = np.zeros((n_win, wa.r.shape[1]))
        sst = np.zeros_like(ssr)
        nv = np.zeros(ssr.shape, dtype=np.int64)
        np.add.at(ssr, wid, err2)
        np.add.at(sst, wid, tot2)
        np.add.at(nv, wid, cnt)
        ok = (nv > 0) & (sst > 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.where(ok, 1.0 - ssr / np.where(ok, sst, 1.0), np.nan)
            pooled_sst = np.where(ok, sst, 0.0).sum(axis=1)
            pooled = np.where(pooled_sst > 0.0, 1.0 - np.where(ok, ssr, 0.0).sum(axis=1) / pooled_sst, np.nan)
        out[f"median_r2{suffix}"] = _row_nanmedian(r2)
        out[f"pooled_r2{suffix}"] = pooled
    frame = pd.DataFrame(out)
    return frame.loc[:, list(SWEEP_COLUMNS)]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
@dataclass
class _WindowArrays:
    """Aligned arrays of a set of return days (rows) for the evaluation.

    ``S`` / ``S0`` the shocks on the matching shock days (``S0`` with invalid
    rows set to 0), ``r`` the returns, ``valid`` the valid cells, ``fitted``
    and ``fitted_oracle`` the topic-explained returns (unmasked), ``B`` and
    ``B_true`` (topics x assets), ``scale`` the training return scale per
    asset (used by both, D74).
    """

    topics: pd.Index
    assets: pd.Index
    S0: np.ndarray
    r: np.ndarray
    valid: np.ndarray
    fitted: np.ndarray
    fitted_oracle: np.ndarray
    B: np.ndarray
    B_true: np.ndarray
    scale: np.ndarray


def _window_arrays(
    sim: SimData, shocks: ObservedShocks, fit: DirectFit, truth: SimTruth, days: pd.DatetimeIndex
) -> _WindowArrays:
    """Pair each return day with its shock day (position minus ``l``) and build the aligned arrays."""
    cal = sim.market.calendar
    topics, assets = fit.B_hat.index, fit.B_hat.columns
    lead = int(sim.lead_days)
    q = cal.get_indexer(days)
    if (q < 0).any():
        raise ValueError("evaluate: some return days are not on the simulation calendar")
    p = q - lead
    s = shocks.s_hat
    if not s.index.equals(cal):
        s = s.reindex(index=cal)
    if list(s.columns) != list(topics):
        s = s.reindex(columns=topics)
    S_full = s.to_numpy(dtype=float)
    n_topics = len(topics)
    S = np.full((len(q), n_topics), np.nan)
    has_shock = p >= 0
    S[has_shock] = S_full[p[has_shock]]
    ret = sim.market.returns
    if list(ret.columns) != list(assets):
        ret = ret.reindex(columns=assets)
    r = ret.to_numpy(dtype=float)[q] if len(q) else np.empty((0, len(assets)))
    row_ok = np.isfinite(S).all(axis=1)
    valid = np.isfinite(r) & row_ok[:, None]
    S0 = np.where(row_ok[:, None], S, 0.0)

    B = fit.B_hat.to_numpy(dtype=float)
    scale = fit.ret_scale.reindex(assets).to_numpy(dtype=float)
    B_true = truth.B_true.reindex(index=topics, columns=assets).to_numpy(dtype=float)
    fitted = (S0 @ B) * scale[None, :]
    fitted_oracle = (S0 @ B_true) * scale[None, :]  # D74: the fit's training scale, not asset_vol
    return _WindowArrays(
        topics=topics, assets=assets, S0=S0, r=r, valid=valid, fitted=fitted, fitted_oracle=fitted_oracle,
        B=B, B_true=B_true, scale=scale,
    )


def _check_out_of_sample(
    shocks: ObservedShocks, fit: DirectFit, window: WindowConfig, days: pd.DatetimeIndex
) -> None:
    """Raise ``ValueError`` unless the forecast window lies after the training window (D65).

    Checks that ``forecast_start`` is after the shocks' ``train_end`` and that
    the first return day of the window is after the fit's last training
    return day (``fit.meta["last_return_day"]``, when recorded).
    """
    fs, te = pd.Timestamp(window.forecast_start), pd.Timestamp(shocks.train_end)
    if fs <= te:
        raise ValueError(
            f"forecast_start {fs.date()} is not after the training end {te.date()} of the shocks; "
            "refit with a train_end before the forecast window (D65)"
        )
    last = fit.meta.get("last_return_day") if isinstance(fit.meta, dict) else None
    if last is not None and not pd.isna(last) and len(days) and pd.Timestamp(last) >= days[0]:
        raise ValueError(
            f"the fit's last training return day {pd.Timestamp(last).date()} is not before the first forecast "
            f"return day {days[0].date()}; refit with a train_end before the forecast window (D65)"
        )


def _r2(r: np.ndarray, fitted: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-column uncentered R2 ``1 - sum (r - f)^2 / sum r^2`` over valid cells (``NaN`` if undefined)."""
    v = valid & np.isfinite(fitted)
    ssr = np.where(v, (r - fitted) ** 2, 0.0).sum(axis=0)
    sst = np.where(v, r**2, 0.0).sum(axis=0)
    ok = v.any(axis=0) & (sst > 0.0)
    out = np.full(r.shape[1], np.nan)
    out[ok] = 1.0 - ssr[ok] / sst[ok]
    return out


def _masked_corr(r: np.ndarray, S0: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pearson correlation per (asset, topic) over the asset's valid days: ``(N, L)``.

    Moments per asset ``n`` and topic ``k`` over the valid days of ``n``: with
    ``m`` the day count, ``cov = sum r s / m - (sum r / m)(sum s / m)`` and
    the variances likewise; ``NaN`` when ``m < 3`` or a variance is (near) 0.
    """
    O = valid.astype(float)
    r0 = np.where(valid, r, 0.0)
    m = O.sum(axis=0)[:, None]  # (N, 1)
    s_r = r0.sum(axis=0)[:, None]
    s_rr = (r0**2).sum(axis=0)[:, None]
    s_s = O.T @ S0
    s_ss = O.T @ (S0**2)
    s_rs = r0.T @ S0
    with np.errstate(invalid="ignore", divide="ignore"):
        mr, ms = s_r / m, s_s / m
        cov = s_rs / m - mr * ms
        vr = s_rr / m - mr**2
        vs = s_ss / m - ms**2
        corr = cov / np.sqrt(vr * vs)
    bad = (m < MIN_CORR_DAYS) | ~(vr > 1e-14 * np.maximum(s_rr / np.maximum(m, 1), 1e-300)) | ~(vs > 1e-12)
    corr = np.where(bad, np.nan, corr)
    return np.clip(corr, -1.0, 1.0)


def median_finite(values: pd.Series | np.ndarray) -> float:
    """Median over the finite values (``NaN`` if there is none); used for the headline medians."""
    a = pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def _row_nanmedian(x: np.ndarray) -> np.ndarray:
    """Median of each row over finite values (``NaN`` for a row without one), without all-NaN warnings."""
    out = np.full(x.shape[0], np.nan)
    for i, row in enumerate(x):
        f = row[np.isfinite(row)]
        if f.size:
            out[i] = float(np.median(f))
    return out
