"""Step 3: kernel-weighted asset/narrative covariances ``cov_{i,t}`` (BKS Eq. 6, App. B.1; DESIGN.md D11-D16).

BKS Eq. 6 with the kernel of Appendix B.1::

    cov_{i,t} = sum_tau k(tau;t) r_{i,tau} z_tau'  -  (sum_tau k(tau;t) r_{i,tau}) (sum_tau k(tau;t) z_tau')

    k(tau;t) = xi^(t - t_tau) / sum_tau' xi^(t - t_tau')      for days tau inside the window of t, 0 otherwise

where ``t_tau`` is the period (month) of day ``tau``. Three points the code pins down:

1. **The decay steps per period, not per day** (D11): every day of period
   ``t`` has raw weight ``xi^0 = 1``, every day of period ``t-1`` weight ``xi``.
2. **The window of period ``t`` closes ``skip_days`` trading days before the
   last day of ``t``** (D12, BKS footnote 9): with ``skip_days = 1`` it ends on
   the second-to-last trading day, so ``cov_{i,t}`` is known before the
   period-``t+1`` return accrues.
3. **Weights renormalise over observed asset-days** (D13, D14): the sum in
   the denominator of ``k`` runs over the days on which asset ``i`` has a
   return *and* the full shock vector ``z_tau`` is observed. Days with a
   missing shock are excluded for every asset.

Implementation (D16): because the raw weight is constant within a period,
``cov_{i,t}`` is a function of per-period sums of ``r z'``, ``r``, ``z`` (over
the days asset ``i`` is observed) and observed-day counts. The stage makes one
pass over the daily data, accumulating an exponentially weighted recursion
across periods; the current period's partial sums (up to the window end) are
handled separately. Cost ``O(D N L)`` in days, memory ``O(T N L)`` for the
output only.

Validity boundaries
-------------------
* Period ids are positional (see :func:`narrative_ipca.data.period_end_index`):
  a period absent from the calendar counts as one decay step.
* Available-case weighting is unbiased only when missingness is unrelated to
  the shocks; ``CovarianceConfig.min_days`` guards against thin windows.
* No shrinkage: the group lasso of Eq. 8 disciplines noisy instruments
  downstream, not this stage.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import CovarianceConfig
from .data import period_end_index
from .types import CovariancePanel, ShockPanel

logger = logging.getLogger(__name__)

__all__ = [
    "kernel_weights",
    "window_bounds",
    "brute_force_covariance",
    "build_covariance_panel",
]


def kernel_weights(period_id: np.ndarray, t: int, xi: float, lookback: int | None = None) -> np.ndarray:
    """Raw (unnormalised) kernel ``xi^(t - t_tau)`` of BKS App. B.1 for every day.

    Reference implementation used by the tests. Days in periods ``> t`` get
    weight 0; with ``lookback = m`` only periods ``t - m + 1 .. t`` keep a
    positive weight. The window-end cutoff (``skip_days``) is *not* applied
    here: callers zero the tail of period ``t`` themselves (see
    :func:`window_bounds`). Normalisation over observed days is done by
    :func:`brute_force_covariance`.
    """
    if not (0.0 < xi <= 1.0):
        raise ValueError(f"xi must lie in (0, 1], got {xi!r}")
    if lookback is not None and int(lookback) < 1:
        raise ValueError("lookback must be >= 1 or None")
    pid = np.asarray(period_id, dtype=np.int64)
    if pid.ndim != 1:
        raise ValueError(f"period_id must be 1-D, got shape {pid.shape}")
    dist = int(t) - pid
    keep = dist >= 0
    if lookback is not None:
        keep &= dist < int(lookback)
    return np.where(keep, np.power(float(xi), np.clip(dist, 0, None).astype(float)), 0.0)


def window_bounds(period_id: np.ndarray, skip_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Day positions ``(start, stop, cut)`` of every period.

    Period ``t`` occupies rows ``start[t]:stop[t]``; the rows of period ``t``
    inside its own window are ``start[t]:cut[t]`` with
    ``cut[t] = max(start[t], stop[t] - skip_days)`` (D12). A period with at
    most ``skip_days`` trading days contributes none of its own days.
    """
    pid = np.asarray(period_id, dtype=np.int64)
    T = int(pid.max()) + 1 if pid.size else 0
    ids = np.arange(T)
    start = np.searchsorted(pid, ids, side="left")
    stop = np.searchsorted(pid, ids, side="right")
    cut = np.maximum(start, stop - int(skip_days))
    return start, stop, cut


def brute_force_covariance(r: pd.Series, z: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Eq. 6 for one asset by direct summation over days (``O(days x L)`` reference).

    ``r``, ``z`` and ``weights`` are row-aligned (same length). Days with a
    missing return, a missing shock in any topic, or zero weight are dropped;
    the remaining raw weights are normalised to sum to one; the result is
    ``sum k r z' - (sum k r)(sum k z')``. Returns an all-``NaN`` vector when
    no day survives. The caller enforces ``min_days``.
    """
    rv = np.asarray(r, dtype=float).ravel()
    Z = np.asarray(z, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if Z.ndim != 2 or rv.shape[0] != Z.shape[0] or w.shape[0] != Z.shape[0]:
        raise ValueError("r, z and weights must have the same number of rows")
    ok = np.isfinite(rv) & np.isfinite(Z).all(axis=1) & (w > 0.0)
    if not ok.any():
        return np.full(Z.shape[1], np.nan)
    w = w[ok] / w[ok].sum()
    rv = rv[ok]
    Z = Z[ok]
    return (w * rv) @ Z - (w @ rv) * (w @ Z)


def _block_sums(
    Rf: np.ndarray, Mk: np.ndarray, Rok: np.ndarray, Zf: np.ndarray, s: int, e: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-asset sums over rows ``s:e``: ``sum r z'`` (N,L), ``sum r`` (N,), ``sum_{i observed} z`` (N,L),
    observed-day count as float (N,) and as int (N,)."""
    Rb, Mb, Zb = Rf[s:e], Mk[s:e], Zf[s:e]
    rz = Rb.T @ Zb
    r = Rb.sum(axis=0)
    zz = Mb.T @ Zb
    wcount = Mb.sum(axis=0)
    ncount = Rok[s:e].sum(axis=0)
    return rz, r, zz, wcount, ncount


def build_covariance_panel(
    shocks: ShockPanel,
    returns: pd.DataFrame,
    cfg: CovarianceConfig,
    period: str,
    dtype: str | np.dtype = "float64",
) -> CovariancePanel:
    """Kernel-weighted covariance instruments ``cov_{i,t}`` for every asset and period (Eq. 6, D11-D16).

    Parameters
    ----------
    shocks:
        :class:`~narrative_ipca.types.ShockPanel`; ``z`` is re-indexed to
        ``returns.index``. Days with any missing shock are excluded.
    returns:
        ``(n_days, N)`` daily excess returns on the trading-day calendar
        (``AlignedData.returns``); ``NaN`` = not observed.
    cfg:
        :class:`~narrative_ipca.config.CovarianceConfig` (``xi``,
        ``skip_days``, ``lookback_periods``, ``min_days``). ``burn_in_periods``
        is *not* applied here (the panel stage drops those periods).
    period:
        Pandas period alias defining ``t`` (``DataConfig.period``).
    dtype:
        Storage dtype of ``values`` (``DataConfig.dtype``); all arithmetic is
        float64.

    Returns
    -------
    :class:`~narrative_ipca.types.CovariancePanel` with ``values`` of shape
    ``(T, N, L)``, ``NaN`` where the asset has fewer than ``cfg.min_days``
    observed days inside the window; ``n_days`` ``(T, N)`` observed-day
    counts; ``window_end`` the last calendar day inside each window.

    Algorithm
    ---------
    With ``F_m`` the per-period full sums and ``P_t`` the partial sums of
    period ``t`` up to the window end, the window sums for period ``t`` are
    ``W_t = sum_{m<t} xi^(t-m) F_m + P_t`` (restricted to ``m > t - lookback``
    when truncated). The closed-period part is carried as the recursion
    ``H_t = xi H_{t-1} + F_t`` (minus ``xi^lookback F_{t-lookback}`` when
    truncated, with an exact rebuild every ``lookback`` periods to stop
    round-off drift). ``cov_{i,t}`` is then ``W_rz / W_n - (W_r / W_n)(W_z / W_n)``
    with ``W_n`` the weighted count of observed days.
    """
    if not isinstance(returns, pd.DataFrame) or not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be a DataFrame with a DatetimeIndex")
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("returns index must be sorted and unique")
    z = shocks.z
    if not isinstance(z, pd.DataFrame):
        raise TypeError("shocks.z must be a DataFrame")

    calendar = pd.DatetimeIndex(returns.index)
    Z = z.reindex(calendar).to_numpy(dtype=float)  # (D, L)
    R = returns.to_numpy(dtype=float)  # (D, N)
    D, N = R.shape
    L = Z.shape[1]
    if D == 0 or N == 0 or L == 0:
        raise ValueError(f"empty inputs: returns {R.shape}, shocks {Z.shape}")

    xi = float(cfg.xi)
    skip = int(cfg.skip_days)
    lb = None if cfg.lookback_periods is None else int(cfg.lookback_periods)
    min_days = int(cfg.min_days)

    pid, period_ends = period_end_index(calendar, period)
    T = len(period_ends)
    start, stop, cut = window_bounds(pid, skip)

    zok = np.isfinite(Z).all(axis=1)
    n_zmiss = int((~zok).sum())
    Zf = np.where(zok[:, None], Z, 0.0)
    Rok = np.isfinite(R) & zok[:, None]
    Rf = np.where(Rok, R, 0.0)
    Mk = Rok.astype(float)

    values = np.full((T, N, L), np.nan, dtype=np.dtype(dtype))
    n_days = np.zeros((T, N), dtype=np.int64)
    window_end: list[pd.Timestamp] = []

    # closed-period accumulators H_* = sum_{m <= t-1, m in window} xi^(t-1-m) F_m
    H_rz = np.zeros((N, L))
    H_r = np.zeros(N)
    H_z = np.zeros((N, L))
    H_w = np.zeros(N)
    H_n = np.zeros(N, dtype=np.int64)
    xi_lb = xi**lb if lb is not None else 0.0

    log_every = max(1, T // 10)
    for t in range(T):
        s, e, c = int(start[t]), int(stop[t]), int(cut[t])

        # age the closed accumulators to period t; drop the period leaving a truncated window
        G_rz, G_r, G_z, G_w, G_n = xi * H_rz, xi * H_r, xi * H_z, xi * H_w, H_n.copy()
        if lb is not None and t - lb >= 0:
            m = t - lb
            F = _block_sums(Rf, Mk, Rok, Zf, int(start[m]), int(stop[m]))
            G_rz -= xi_lb * F[0]
            G_r -= xi_lb * F[1]
            G_z -= xi_lb * F[2]
            G_w -= xi_lb * F[3]
            G_n -= F[4]

        # partial sums of period t (days inside the window) and its tail (excluded days)
        P_rz, P_r, P_z, P_w, P_n = _block_sums(Rf, Mk, Rok, Zf, s, c)
        W_rz = G_rz + P_rz
        W_r = G_r + P_r
        W_z = G_z + P_z
        W_w = G_w + P_w
        W_n = G_n + P_n

        valid = (W_n >= min_days) & (W_w > 0.0)
        if valid.any():
            denom = np.where(valid, W_w, 1.0)
            mean_r = W_r / denom
            mean_z = W_z / denom[:, None]
            cov = W_rz / denom[:, None] - mean_r[:, None] * mean_z
            values[t, valid, :] = cov[valid]
        n_days[t] = W_n
        if c > s:
            window_end.append(calendar[c - 1])
        elif t > 0:
            window_end.append(calendar[int(stop[t - 1]) - 1])
        else:
            window_end.append(pd.NaT)

        # close period t
        T_rz, T_r, T_z, T_w, T_n = _block_sums(Rf, Mk, Rok, Zf, c, e)
        H_rz = W_rz + T_rz
        H_r = W_r + T_r
        H_z = W_z + T_z
        H_w = W_w + T_w
        H_n = W_n + T_n

        # exact rebuild of the truncated accumulators every lb periods (removes round-off drift)
        if lb is not None and lb > 1 and t >= lb - 1 and (t + 1) % lb == 0:
            H_rz[...] = 0.0
            H_r[...] = 0.0
            H_z[...] = 0.0
            H_w[...] = 0.0
            H_n[...] = 0
            for j in range(lb):
                m = t - j
                F = _block_sums(Rf, Mk, Rok, Zf, int(start[m]), int(stop[m]))
                wj = xi**j
                H_rz += wj * F[0]
                H_r += wj * F[1]
                H_z += wj * F[2]
                H_w += wj * F[3]
                H_n += F[4]
        elif lb == 1:
            H_rz, H_r, H_z, H_w, H_n = T_rz + P_rz, T_r + P_r, T_z + P_z, T_w + P_w, T_n + P_n

        if (t + 1) % log_every == 0 or t == T - 1:
            logger.debug("covariances: period %d/%d (%s), %d valid assets", t + 1, T, period_ends[t].date(), int(valid.sum()))

    n_valid = int(np.isfinite(values[:, :, 0]).sum())
    logger.info(
        "build_covariance_panel: T=%d periods (%s), N=%d assets, L=%d topics; xi=%.4f skip_days=%d lookback=%s "
        "min_days=%d; %d/%d asset-periods valid; %d days with missing shocks excluded",
        T,
        period,
        N,
        L,
        xi,
        skip,
        lb,
        min_days,
        n_valid,
        T * N,
        n_zmiss,
    )
    return CovariancePanel(
        values=values,
        periods=period_ends,
        window_end=pd.DatetimeIndex(window_end),
        assets=np.asarray([str(c) for c in returns.columns], dtype=object),
        topics=np.asarray([str(c) for c in z.columns], dtype=object),
        n_days=n_days,
        xi=xi,
    )
