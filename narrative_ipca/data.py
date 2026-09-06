"""Step 1: input validation and calendar alignment (BKS Section 3.1; DESIGN.md D1-D8).

The module turns the two raw inputs, daily attention levels ``theta_tau``
(:class:`~narrative_ipca.types.AttentionData`) and daily asset returns
(:class:`~narrative_ipca.types.ReturnsData`), into an
:class:`~narrative_ipca.types.AlignedData` object on a single trading-day
calendar. It also holds the period helpers shared by the covariance and panel
stages (which trading days form period ``t`` and what its last day is) and the
BKS document-to-day aggregation of attention.

Conventions
-----------
* The calendar is the return calendar (D3). Attention observed on non-trading
  days is dropped or folded into the next trading day.
* ``theta_tau`` is assumed synchronised with ``r_tau`` (D4);
  ``DataConfig.attention_lag_days = k`` moves attention *forward* on the
  trading-day grid, so that day ``tau`` is paired with ``theta_{tau-k}``
  (``DataFrame.shift(k)``). This never introduces look-ahead.
* Period ids are positions in the sequence of periods that actually occur in
  the calendar (0, 1, 2, ...). With a continuous daily calendar these coincide
  with calendar month differences; a period entirely absent from the data
  counts as one step for the kernel of the covariance stage.

Validity boundaries
-------------------
* Everything here is available-case: no return, attention or risk-free value
  is imputed. A missing attention row stays ``NaN`` and propagates into the
  shocks of the following ``ShockConfig.window`` days.
* ``inverse_vol`` scaling uses only days strictly before ``tau`` (ex ante) but
  the resulting factors are portfolios of vol-scaled positions (D7).
"""

from __future__ import annotations

import logging
import re
import warnings

import numpy as np
import pandas as pd

from .config import DataConfig
from .types import AlignedData, AttentionData, ReturnsData

logger = logging.getLogger(__name__)

__all__ = [
    "align_inputs",
    "aggregate_documents",
    "period_end_index",
    "period_returns",
    "trailing_volatility",
    "normalize_period_alias",
]

# Offset aliases that pandas does not accept as *period* frequencies, mapped to
# the period alias with the same span. ``"M"`` itself is accepted by every
# pandas version we support; the map only widens what the user may write.
_PERIOD_ALIAS = {
    "ME": "M",
    "BME": "M",
    "MS": "M",
    "BMS": "M",
    "QE": "Q",
    "BQE": "Q",
    "QS": "Q",
    "BQS": "Q",
    "YE": "Y",
    "BYE": "Y",
    "YS": "Y",
    "BYS": "Y",
    "A": "Y",
    "AS": "Y",
    "BA": "Y",
    "BAS": "Y",
}


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------
def normalize_period_alias(period: str) -> list[str]:
    """Candidate pandas *period* aliases for a user-supplied frequency string.

    Returns ``[period]`` plus, when ``period`` is an end/start-anchored offset
    alias such as ``"ME"`` or ``"QE"`` (valid for resampling but not for
    ``pd.PeriodIndex``), the equivalent period alias (``"M"``, ``"Q"``).
    Multipliers and anchors are preserved (``"2ME"`` -> ``"2M"``,
    ``"W-FRI"`` unchanged).
    """
    if not isinstance(period, str) or not period:
        raise ValueError(f"period must be a non-empty pandas offset alias, got {period!r}")
    candidates = [period]
    head, sep, anchor = period.partition("-")
    m = re.match(r"^(\d*)([A-Za-z]+)$", head)
    if m and m.group(2).upper() in _PERIOD_ALIAS:
        candidates.append(m.group(1) + _PERIOD_ALIAS[m.group(2).upper()] + sep + anchor)
    return candidates


def _to_period_index(calendar: pd.DatetimeIndex, period: str) -> pd.PeriodIndex:
    """``pd.PeriodIndex(calendar, freq=period)``, silently accepting ``"M"`` and ``"ME"`` alike."""
    last_err: Exception | None = None
    for cand in normalize_period_alias(period):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return pd.PeriodIndex(calendar, freq=cand)
            except (ValueError, TypeError) as exc:  # invalid alias, try the next candidate
                last_err = exc
    raise ValueError(f"period {period!r} is not a valid pandas period alias") from last_err


def period_end_index(calendar: pd.DatetimeIndex, period: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Period id of every trading day and the last trading day of every period.

    Implements the BKS timing convention "``t_tau`` is the month of day
    ``tau``, ``tau_t`` is the last day of month ``t``" (App. B.1) for a generic
    pandas period alias ``period`` (``"M"``, ``"W"``, ``"Q"``, ...).

    Parameters
    ----------
    calendar:
        Sorted, unique ``DatetimeIndex`` of trading days.
    period:
        Pandas period/offset alias; ``"M"`` and ``"ME"`` are both accepted.

    Returns
    -------
    period_id:
        ``(n_days,)`` int64, ``0 .. T-1``, non-decreasing; the position of the
        day's period among the periods present in ``calendar``.
    period_ends:
        ``(T,)`` ``DatetimeIndex``, the last trading day of each period.

    Assumptions
    -----------
    Ids are positional: a period with no trading day in ``calendar`` does not
    get an id, so consecutive ids may span more than one calendar period.
    """
    calendar = pd.DatetimeIndex(calendar)
    if len(calendar) == 0:
        return np.zeros(0, dtype=np.int64), pd.DatetimeIndex([])
    if not calendar.is_monotonic_increasing:
        raise ValueError("calendar must be sorted ascending")
    if calendar.has_duplicates:
        raise ValueError("calendar has duplicate dates")
    pidx = _to_period_index(calendar, period)
    codes, _uniques = pd.factorize(pidx)  # order of appearance == chronological (calendar sorted)
    codes = np.asarray(codes, dtype=np.int64)
    if np.any(codes < 0):
        raise ValueError("calendar contains NaT")
    last_pos = np.flatnonzero(np.concatenate([codes[1:] != codes[:-1], [True]]))
    return codes, pd.DatetimeIndex(calendar[last_pos])


def period_returns(returns: pd.DataFrame, period: str, how: str) -> pd.DataFrame:
    """Accumulate daily excess returns to the period return ``r_{i,t}`` of BKS Eq. 7.

    ``how="sum"``: ``r_{i,t} = sum_{tau in t} r_{i,tau}`` (linear accumulation,
    D6). ``how="compound"``: ``r_{i,t} = prod_{tau in t} (1 + r_{i,tau}) - 1``.
    Both run over the *observed* days of the period only (available case);
    the result is ``NaN`` when the asset has no observed day in the period.

    Parameters
    ----------
    returns:
        ``(n_days, N)`` daily returns on a sorted ``DatetimeIndex``; ``NaN``
        marks unobserved asset-days.
    period:
        Pandas period alias (see :func:`period_end_index`).
    how:
        ``"sum"`` or ``"compound"``.

    Returns
    -------
    ``(T, N)`` DataFrame indexed by the last trading day of each period.
    """
    if how not in ("sum", "compound"):
        raise ValueError(f"how must be 'sum' or 'compound', got {how!r}")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    codes, ends = period_end_index(returns.index, period)
    r = returns.astype(float)
    if how == "sum":
        out = r.groupby(codes).sum(min_count=1)
    else:
        out = (1.0 + r).groupby(codes).prod(min_count=1) - 1.0
    # groupby sorts the integer codes, which are already chronological
    out = out.reindex(np.arange(len(ends)))
    out.index = ends
    return out


# ---------------------------------------------------------------------------
# Document-level aggregation (D1)
# ---------------------------------------------------------------------------
def _positional(values, index: pd.Index, name: str, n: int) -> np.ndarray:
    """Return ``values`` as a length-``n`` array aligned to the documents.

    A ``Series`` whose index equals ``index`` is aligned by label; any other
    length-``n`` sequence is taken positionally.
    """
    if isinstance(values, pd.Series):
        if values.index.equals(index):
            arr = values.to_numpy()
        elif len(values) == n:
            arr = values.to_numpy()
        else:
            raise ValueError(f"{name} has length {len(values)}, expected {n}")
    else:
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(f"{name} must be 1-D of length {n}, got shape {arr.shape}")
    return arr


def aggregate_documents(
    theta_docs: pd.DataFrame,
    dates,
    weights=None,
) -> pd.DataFrame:
    """BKS daily attention from document-level topic distributions.

    Implements ``theta_tau = (sum_{m in tau} N_m theta_m) / (sum_{m in tau} N_m)``
    (BKS Section 1.1), the term-count-weighted mean of the article attention
    vectors of day ``tau``. With ``weights=None`` every document has weight 1
    (plain daily mean).

    Parameters
    ----------
    theta_docs:
        ``(M, L)`` DataFrame, one row per document, one column per topic.
    dates:
        Length-``M`` datetime-like sequence (or ``Series``) with the day of
        each document; intra-day timestamps are normalised to midnight.
    weights:
        Optional length-``M`` non-negative weights ``N_m`` (term counts).

    Returns
    -------
    ``(n_days, L)`` DataFrame on a sorted ``DatetimeIndex``. A topic whose
    documents on a day are all ``NaN`` (or all zero-weight) is ``NaN`` that
    day; per-topic ``NaN`` cells are excluded from both sums.
    """
    if not isinstance(theta_docs, pd.DataFrame):
        raise TypeError("theta_docs must be a DataFrame")
    M, L = theta_docs.shape
    if M == 0 or L == 0:
        raise ValueError(f"theta_docs is empty, shape {theta_docs.shape}")
    theta = theta_docs.to_numpy(dtype=float)
    d = pd.DatetimeIndex(pd.to_datetime(_positional(dates, theta_docs.index, "dates", M))).normalize()
    if d.hasnans:
        raise ValueError("dates contains NaT")
    if weights is None:
        w = np.ones(M, dtype=float)
    else:
        w = np.asarray(_positional(weights, theta_docs.index, "weights", M), dtype=float)
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and non-negative")
    mask = np.isfinite(theta)
    num = np.where(mask, theta, 0.0) * w[:, None]
    den = mask.astype(float) * w[:, None]
    num_g = pd.DataFrame(num, index=d).groupby(level=0).sum()
    den_g = pd.DataFrame(den, index=d).groupby(level=0).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num_g.to_numpy() / den_g.to_numpy()
    out[den_g.to_numpy() <= 0.0] = np.nan
    result = pd.DataFrame(out, index=pd.DatetimeIndex(num_g.index), columns=theta_docs.columns)
    result = result.sort_index()
    logger.info("aggregate_documents: %d documents -> %d days x %d topics", M, result.shape[0], L)
    return result


# ---------------------------------------------------------------------------
# Alignment (D2-D4, D7)
# ---------------------------------------------------------------------------
def trailing_volatility(returns: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Ex ante trailing volatility used by ``asset_weighting="inverse_vol"`` (D7).

    ``scale_{i,tau} = std(r_{i,tau-window} .. r_{i,tau-1})`` (sample std,
    ``ddof=1``, over the observed days of the window), i.e. the rolling
    standard deviation shifted by one day so that day ``tau`` uses days
    strictly before ``tau``. Requires ``min_periods`` observed days
    (default ``max(window // 4, 2)``); the value is ``NaN`` otherwise and
    wherever the std is zero.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    mp = max(window // 4, 2) if min_periods is None else max(int(min_periods), 2)
    vol = returns.astype(float).rolling(window=int(window), min_periods=mp).std(ddof=1).shift(1)
    return vol.where(vol > 0.0)


def _align_risk_free(risk_free: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free rate on the return calendar (last known value carried forward)."""
    if not isinstance(risk_free, pd.Series) or not isinstance(risk_free.index, pd.DatetimeIndex):
        raise TypeError("risk_free must be a Series with a DatetimeIndex")
    rf = risk_free.astype(float).sort_index()
    rf = rf[~rf.index.duplicated(keep="last")]
    exact = rf.reindex(calendar)
    n_missing = int(exact.isna().sum())
    if n_missing:
        logger.warning(
            "risk_free is missing on %d of %d trading days; carrying the last known value forward",
            n_missing,
            len(calendar),
        )
        rf = rf.reindex(rf.index.union(calendar)).ffill().reindex(calendar)
        still = int(rf.isna().sum())
        if still:
            logger.warning("risk_free unavailable on the first %d trading days; excess returns set to NaN there", still)
        return rf
    return exact


def _reindex_attention(att: pd.DataFrame, calendar: pd.DatetimeIndex, policy: str) -> pd.DataFrame:
    """Put attention on the trading-day grid per ``non_trading_day_policy`` (D3).

    ``drop``: keep trading-day rows only. ``fold_mean`` / ``fold_sum``: every
    attention row is assigned to the first trading day on or after its date
    and rows sharing a trading day are averaged / summed (a trading day's own
    row is included in its group). Rows dated after the last trading day have
    no target and are discarded.
    """
    if policy == "drop":
        n_off = int((~att.index.isin(calendar)).sum())
        if n_off:
            logger.info("dropping %d attention rows on non-trading days", n_off)
        return att.reindex(calendar)
    if policy not in ("fold_mean", "fold_sum"):
        raise ValueError(f"unknown non_trading_day_policy {policy!r}")
    pos = calendar.searchsorted(att.index, side="left")
    valid = pos < len(calendar)
    n_after = int((~valid).sum())
    if n_after:
        logger.info("dropping %d attention rows dated after the last trading day", n_after)
    sub = att.iloc[np.flatnonzero(valid)]
    target = calendar[pos[valid]]
    grouped = sub.groupby(target)
    folded = grouped.mean() if policy == "fold_mean" else grouped.sum(min_count=1)
    return folded.reindex(calendar)


def _validate(attention: AttentionData, returns: ReturnsData, cfg: DataConfig) -> None:
    if not hasattr(attention, "levels") or not isinstance(attention.levels, pd.DataFrame):
        raise TypeError("attention must be an AttentionData with a DataFrame 'levels'")
    if not hasattr(returns, "returns") or not isinstance(returns.returns, pd.DataFrame):
        raise TypeError("returns must be a ReturnsData with a DataFrame 'returns'")
    if not isinstance(attention.levels.index, pd.DatetimeIndex) or not isinstance(returns.returns.index, pd.DatetimeIndex):
        raise TypeError("attention.levels and returns.returns must have DatetimeIndex")
    if cfg.return_kind == "total" and getattr(returns, "risk_free", None) is None:
        raise ValueError("return_kind='total' requires ReturnsData.risk_free")
    if len(attention.levels.index.intersection(returns.returns.index)) == 0 and cfg.non_trading_day_policy == "drop":
        raise ValueError("attention and returns share no trading day")


def align_inputs(attention: AttentionData, returns: ReturnsData, cfg: DataConfig) -> AlignedData:
    """Put attention levels and daily excess returns on the return calendar (Step 1).

    Procedure (DESIGN.md Part C, ``data.align_inputs``):

    1. validate the inputs;
    2. daily excess returns ``r_{i,tau}``: the input as is when
       ``return_kind == "excess"``, else ``total_{i,tau} - rf_tau`` with the
       risk-free rate carried forward to every trading day;
    3. re-index attention to the trading days per ``non_trading_day_policy``
       (D3);
    4. shift attention forward by ``attention_lag_days`` rows (D4), so that
       trading day ``tau`` carries ``theta_{tau - lag}``;
    5. ``asset_weighting == "inverse_vol"``: divide ``r_{i,tau}`` by the
       trailing volatility of days ``< tau`` (:func:`trailing_volatility`);
       the divisor is returned as ``AlignedData.scale`` (D7);
    6. drop leading and trailing days until at least one attention value and
       at least one asset return are present.

    Column labels are coerced to ``str``. Nothing is imputed.
    """
    _validate(attention, returns, cfg)

    calendar = pd.DatetimeIndex(returns.returns.index)
    rets = returns.returns.astype(float).copy()
    rets.columns = [str(c) for c in rets.columns]

    # 2. excess returns
    if cfg.return_kind == "total":
        rf = _align_risk_free(returns.risk_free, calendar)
        rets = rets.sub(rf.to_numpy(), axis=0)

    # 3. attention on the trading-day grid
    att = attention.levels.astype(float).copy()
    att.columns = [str(c) for c in att.columns]
    att = _reindex_attention(att, calendar, cfg.non_trading_day_policy)

    # 4. lag
    if cfg.attention_lag_days > 0:
        att = att.shift(int(cfg.attention_lag_days))

    # 5. inverse-vol scaling (ex ante)
    scale: pd.DataFrame | None = None
    if cfg.asset_weighting == "inverse_vol":
        scale = trailing_volatility(rets, cfg.vol_window_days)
        rets = rets / scale
    elif cfg.asset_weighting != "none":
        raise ValueError(f"unknown asset_weighting {cfg.asset_weighting!r}")

    # 6. trim
    has_att = att.notna().any(axis=1).to_numpy()
    has_ret = rets.notna().any(axis=1).to_numpy()
    both = has_att & has_ret
    if not both.any():
        raise ValueError("no trading day carries both attention and at least one asset return")
    first = int(np.argmax(both))
    last = len(both) - 1 - int(np.argmax(both[::-1]))
    sl = slice(first, last + 1)
    att = att.iloc[sl]
    rets = rets.iloc[sl]
    if scale is not None:
        scale = scale.iloc[sl]
    calendar = pd.DatetimeIndex(rets.index)

    logger.info(
        "align_inputs: %d trading days (%s .. %s), %d topics, %d assets; dropped %d leading / %d trailing days; "
        "policy=%s lag=%d weighting=%s",
        len(calendar),
        calendar[0].date(),
        calendar[-1].date(),
        att.shape[1],
        rets.shape[1],
        first,
        len(both) - 1 - last,
        cfg.non_trading_day_policy,
        cfg.attention_lag_days,
        cfg.asset_weighting,
    )
    return AlignedData(
        attention=att,
        returns=rets,
        calendar=calendar,
        scale=scale,
        asset_meta=getattr(returns, "asset_meta", None),
        topic_labels=getattr(attention, "topic_labels", None),
        phi=getattr(attention, "phi", None),
    )
