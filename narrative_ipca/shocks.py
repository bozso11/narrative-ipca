"""Step 2: attention shocks ``z_tau`` (BKS Section 3.1, App. C.5; DESIGN.md D9-D10a).

BKS define the narrative shock as the deviation of today's attention from a
short trailing moving average,

    z_tau = theta_tau - (1/w) sum_{j=1..w} theta_{tau-j},        w = 5,

where the average runs over the ``w`` trading days *strictly before* ``tau``
(rows of the aligned attention frame, not calendar days). This keeps
``z_tau`` measurable at the close of day ``tau``, which is what makes the
covariance instruments of Eq. 6 ex ante.

Validity boundaries
-------------------
* The first ``w`` rows are ``NaN`` (incomplete trailing window); any ``NaN``
  inside the trailing window also produces a ``NaN`` shock (no imputation).
* Standardisation divides by the full-sample standard deviation and is
  therefore not out-of-sample safe; it is off by default (D10).
* Placebos (App. C.2) are i.i.d. normal and match only the variance of a
  real topic, not its autocorrelation or tail behaviour.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import ShockConfig
from .types import ShockPanel

logger = logging.getLogger(__name__)

__all__ = ["attention_shocks", "shock_diagnostics", "append_placebos", "PLACEBO_PATTERN"]

PLACEBO_PATTERN = re.compile(r"^placebo_\d+$")


def attention_shocks(attention: pd.DataFrame, cfg: ShockConfig) -> ShockPanel:
    """Attention shocks ``z_tau = theta_tau - mean(theta_{tau-1}, ..., theta_{tau-w})`` (D9).

    Implemented as ``theta - theta.rolling(w, min_periods=w).mean().shift(1)``:
    the ``shift(1)`` enforces "strictly prior rows". With
    ``cfg.standardize`` each column is divided by its full-sample standard
    deviation (``ddof=1``, ``NaN`` skipped); columns with zero or undefined
    std keep scale 1. The divisor is returned in ``ShockPanel.scale``.

    Parameters
    ----------
    attention:
        ``(n_days, L)`` attention levels on the trading-day grid
        (``AlignedData.attention``), sorted ``DatetimeIndex``.
    cfg:
        :class:`~narrative_ipca.config.ShockConfig`.

    Returns
    -------
    :class:`~narrative_ipca.types.ShockPanel` with ``z`` of the same shape
    and index as ``attention``; the first ``w`` rows are ``NaN``.
    """
    if not isinstance(attention, pd.DataFrame):
        raise TypeError("attention must be a DataFrame")
    if not isinstance(attention.index, pd.DatetimeIndex) or not attention.index.is_monotonic_increasing:
        raise ValueError("attention must have a sorted DatetimeIndex")
    w = int(cfg.window)
    if w < 1:
        raise ValueError("window must be >= 1")
    if attention.shape[0] <= w:
        raise ValueError(f"attention has {attention.shape[0]} rows; more than window={w} rows are needed")
    theta = attention.astype(float)
    trailing = theta.rolling(window=w, min_periods=w).mean().shift(1)
    z = theta - trailing

    scale: pd.Series | None = None
    if cfg.standardize:
        sd = z.std(axis=0, ddof=1)
        scale = sd.where(np.isfinite(sd) & (sd > 0.0), 1.0).astype(float)
        z = z.div(scale, axis=1)
        logger.info("attention_shocks: standardised %d topics by their full-sample std", z.shape[1])

    n_nan = int(z.isna().any(axis=1).sum())
    logger.info("attention_shocks: window=%d, %d days x %d topics, %d days with a missing shock", w, *z.shape, n_nan)
    return ShockPanel(z=z, window=w, scale=scale)


def shock_diagnostics(shocks: ShockPanel) -> pd.DataFrame:
    """Per-topic summary of the shocks: count, mean, std, skew, kurtosis, lag-1 autocorrelation, share of zero days.

    ``std`` uses ``ddof=1``; ``ac1`` is the correlation of ``z_tau`` with
    ``z_{tau-1}`` over the days where both are observed (``NaN`` for a
    constant column); ``share_zero`` is the fraction of observed days with
    ``z == 0`` exactly (attention that did not move, e.g. days without
    articles). Diagnostics only; nothing downstream depends on them.
    """
    z = shocks.z.astype(float)
    n_obs = z.notna().sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ac1 = pd.Series({c: _autocorr1(z[c]) for c in z.columns}, dtype=float)
        share_zero = (z == 0.0).sum(axis=0) / n_obs.replace(0, np.nan)
    out = pd.DataFrame(
        {
            "n_obs": n_obs.astype(int),
            "mean": z.mean(axis=0),
            "std": z.std(axis=0, ddof=1),
            "skew": z.skew(axis=0),
            "kurt": z.kurt(axis=0),
            "ac1": ac1.reindex(z.columns),
            "share_zero": share_zero,
        }
    )
    out.index.name = "topic"
    return out


def _autocorr1(s: pd.Series) -> float:
    x = s.to_numpy(dtype=float)
    a, b = x[1:], x[:-1]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok], b[ok]
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def append_placebos(shocks: ShockPanel, n: int, seed: int) -> tuple[ShockPanel, np.ndarray]:
    """Append ``n`` i.i.d. normal placebo narratives (BKS App. C.2; D36).

    Each placebo ``z_{l,tau}`` is an i.i.d. ``N(0, s^2)`` sequence whose
    variance ``s^2`` equals the sample variance (``ddof=1``) of a real topic
    drawn uniformly at random (with replacement) among the real topics with a
    finite, positive variance. Rows on which any real shock is ``NaN`` are
    ``NaN`` for the placebos too, so the set of usable days is unchanged.
    Columns are named ``placebo_1, placebo_2, ...`` continuing any numbering
    already present; existing placebo columns are never used as references.

    Returns
    -------
    (panel, mask):
        The new :class:`~narrative_ipca.types.ShockPanel` and a boolean
        ``(L + n,)`` mask that is True for every placebo column.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    z = shocks.z
    is_placebo_existing = np.array([bool(PLACEBO_PATTERN.match(str(c))) for c in z.columns], dtype=bool)
    if n == 0:
        return replace(shocks, z=z.copy()), is_placebo_existing
    real_cols = [c for c, p in zip(z.columns, is_placebo_existing) if not p]
    if not real_cols:
        raise ValueError("no real topics to match placebo variances to")
    var = z[real_cols].var(axis=0, ddof=1).to_numpy(dtype=float)
    candidates = np.flatnonzero(np.isfinite(var) & (var > 0.0))
    if candidates.size == 0:
        raise ValueError("no real topic has a finite positive shock variance")

    rng = np.random.default_rng(seed)
    picks = candidates[rng.integers(0, candidates.size, size=n)]
    draws = rng.standard_normal((z.shape[0], n)) * np.sqrt(var[picks])[None, :]
    nan_rows = z.isna().any(axis=1).to_numpy()
    draws[nan_rows, :] = np.nan

    start = int(is_placebo_existing.sum()) + 1
    names = [f"placebo_{k}" for k in range(start, start + n)]
    placebo = pd.DataFrame(draws, index=z.index, columns=names)
    new_z = pd.concat([z, placebo], axis=1)

    scale = shocks.scale
    if scale is not None:
        scale = pd.concat([scale, pd.Series(1.0, index=names)])
    mask = np.concatenate([is_placebo_existing, np.ones(n, dtype=bool)])
    logger.info(
        "append_placebos: %d placebos (seed=%d) matched to real topics %s",
        n,
        seed,
        [str(real_cols[i]) for i in picks],
    )
    return ShockPanel(z=new_z, window=shocks.window, scale=scale), mask
