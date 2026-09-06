"""Step 4: the estimation panel ``c_{i,t-1} -> r_{i,t}`` (BKS Eq. 7, App. B.1; DESIGN.md D8, D17, D26).

BKS Eq. 7 reads ``r_{i,t+1} = c_{i,t} Gamma f_{t+1} + e_{i,t+1}`` with
``c_{i,t} = [1, cov_{i,t}]``: the instruments measured at the close of the
window of period ``t`` explain the return earned over period ``t+1``. This
module performs exactly that pairing, in the long form expected by
:class:`~narrative_ipca.types.IPCAPanel` (one row per observed asset-period),
and nothing else: no standardisation, winsorisation or demeaning.

Validity boundaries
-------------------
* Rows need a finite return *and* a fully finite instrument vector; an
  asset-period failing either is simply absent (the IPCA objective sums over
  observed ``(i, t)``).
* The pairing is by position in the sequence of periods of the return
  calendar: the instrument period ``t-1`` is the period immediately before
  the return period ``t`` in that sequence.
* ``sigma_c`` is the population std over the rows of *this* panel (D26) and
  must be recomputed for sub-samples (``IPCAPanel.subset_periods`` does).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import CovarianceConfig, DataConfig
from .data import period_returns
from .types import CovariancePanel, IPCAPanel, compute_sigma_c

logger = logging.getLogger(__name__)

__all__ = ["build_panel", "split_periods", "panel_summary"]


def build_panel(
    cov: CovariancePanel,
    returns: pd.DataFrame,
    cfg: DataConfig,
    cov_cfg: CovarianceConfig,
) -> IPCAPanel:
    """Pair lagged instruments with period returns (Eq. 7) into an :class:`IPCAPanel`.

    For every instrument period ``j`` of ``cov`` (position in
    ``cov.periods``) the return period is the next period of the return
    calendar; rows are ``X = [1, cov.values[j, i, :]]`` and
    ``y = r_{i, j+1}`` with ``r`` from :func:`narrative_ipca.data.period_returns`
    (``cfg.period``, ``cfg.return_aggregation``).

    Filters, in order:

    1. the first ``cov_cfg.burn_in_periods`` instrument periods are dropped
       (kernel warm-up, D17);
    2. rows with a non-finite ``y`` or any non-finite instrument are dropped;
    3. return periods with fewer than ``cfg.min_assets_per_period`` surviving
       rows are dropped (D8).

    ``panel.periods`` holds the last trading day of each kept *return*
    period; ``t_idx`` indexes into it and is sorted; ``sigma_c`` is
    :func:`~narrative_ipca.types.compute_sigma_c` of the assembled ``X``.

    Parameters
    ----------
    cov:
        Output of :func:`narrative_ipca.covariances.build_covariance_panel`.
    returns:
        The same daily excess returns the covariances were built from
        (``AlignedData.returns``); its calendar defines the periods.
    """
    if not isinstance(returns, pd.DataFrame) or not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be a DataFrame with a DatetimeIndex")
    values = np.asarray(cov.values)
    T_cov, N, L = values.shape
    assets = np.asarray(cov.assets, dtype=object)
    topics = [str(c) for c in np.asarray(cov.topics)]

    pr = period_returns(returns, cfg.period, cfg.return_aggregation)
    pr.columns = [str(c) for c in pr.columns]
    pr = pr.reindex(columns=[str(a) for a in assets])
    ret_periods = pd.DatetimeIndex(pr.index)
    pos = ret_periods.get_indexer(pd.DatetimeIndex(cov.periods))
    if np.any(pos < 0):
        missing = pd.DatetimeIndex(cov.periods)[pos < 0]
        raise ValueError(
            f"{len(missing)} covariance periods are not periods of the return calendar "
            f"(first: {missing[0].date()}); pass the daily returns the covariances were built from"
        )
    Y = pr.to_numpy(dtype=float)

    burn_in = int(cov_cfg.burn_in_periods)
    min_assets = int(cfg.min_assets_per_period)

    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    a_blocks: list[np.ndarray] = []
    t_blocks: list[np.ndarray] = []
    kept_periods: list[pd.Timestamp] = []
    n_dropped_min = 0
    n_dropped_rows = 0

    for j in range(burn_in, T_cov):
        rp = int(pos[j]) + 1  # position of the return period following instrument period j
        if rp >= len(ret_periods):
            break
        Xj = values[j].astype(float, copy=False)  # (N, L)
        yj = Y[rp]  # (N,)
        ok = np.isfinite(yj) & np.isfinite(Xj).all(axis=1)
        n_ok = int(ok.sum())
        n_dropped_rows += int(np.isfinite(yj).sum()) - n_ok
        if n_ok < min_assets:
            n_dropped_min += 1
            logger.debug(
                "build_panel: dropping return period %s (%d assets < min_assets_per_period=%d)",
                ret_periods[rp].date(),
                n_ok,
                min_assets,
            )
            continue
        t_new = len(kept_periods)
        idx = np.flatnonzero(ok)
        block = np.empty((n_ok, L + 1), dtype=float)
        block[:, 0] = 1.0
        block[:, 1:] = Xj[idx]
        X_blocks.append(block)
        y_blocks.append(yj[idx])
        a_blocks.append(idx.astype(np.int64))
        t_blocks.append(np.full(n_ok, t_new, dtype=np.int64))
        kept_periods.append(pd.Timestamp(ret_periods[rp]))

    if not kept_periods:
        raise ValueError(
            f"no usable period: {T_cov} instrument periods, burn_in={burn_in}, "
            f"min_assets_per_period={min_assets}, {n_dropped_min} periods below the asset minimum"
        )

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    t_idx = np.concatenate(t_blocks)
    asset_idx = np.concatenate(a_blocks)
    periods = pd.DatetimeIndex(kept_periods)
    sigma_c = compute_sigma_c(X)

    logger.info(
        "build_panel: %d rows, %d return periods (%s .. %s), %d assets, %d instruments; burn_in=%d dropped, "
        "%d periods below min_assets=%d, %d asset-periods with a return but no instrument",
        X.shape[0],
        len(periods),
        periods[0].date(),
        periods[-1].date(),
        N,
        L + 1,
        burn_in,
        n_dropped_min,
        min_assets,
        n_dropped_rows,
    )
    return IPCAPanel(
        X=X,
        y=y,
        t_idx=t_idx,
        asset_idx=asset_idx,
        periods=periods,
        assets=assets,
        instrument_names=["const"] + topics,
        sigma_c=sigma_c,
    )


def split_periods(panel: IPCAPanel, first_oos: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks ``(train, test)`` over ``panel.periods``: test = return periods ``>= first_oos``."""
    first_oos = pd.Timestamp(first_oos)
    periods = pd.DatetimeIndex(panel.periods)
    test = np.asarray(periods >= first_oos, dtype=bool)
    return ~test, test


def panel_summary(panel: IPCAPanel) -> pd.DataFrame:
    """Per return period: number of assets and the cross-sectional mean / std (``ddof=1``) of ``y``.

    A sanity screen before estimation (a collapsing ``n_assets`` or a period
    whose ``std_y`` is an order of magnitude off points at a data problem).
    """
    rows = []
    for t, sl in panel.period_slices():
        yt = panel.y[sl]
        n = int(sl.stop - sl.start)
        rows.append(
            {
                "n_assets": n,
                "mean_y": float(yt.mean()) if n else np.nan,
                "std_y": float(yt.std(ddof=1)) if n > 1 else np.nan,
            }
        )
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(panel.periods, name="period"))
    out["n_assets"] = out["n_assets"].astype(int)
    return out
