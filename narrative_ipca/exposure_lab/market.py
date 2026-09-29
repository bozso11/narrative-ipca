"""Market data of the topic-exposure lab: calendar, real and artificial returns (DESIGN.md G.2, G.2.1, G.2.2, G.13; D54-D56, D71).

Symbols: ``t`` a trading day of the weekday calendar, ``n`` an asset.

* **Listed assets** (G.2 point 1): the 55 assets of ``data/reference/assets.csv``.
  An asset "A v B" has the daily simple return ``r_A - r_B`` (D54); the
  returns are precomputed in ``data/market/asset_returns.parquet`` by
  ``scripts/fetch_market_data.py`` (D55).
* **Artificial returns** (G.2.2, D56), used for generic assets, for the
  ``price_source = "artificial"`` mode and for a listed asset whose leg failed:

      r_{n,t} = lambda_n' F_t + e_{n,t}

  where ``F_t`` holds three latent daily factors (risk-on, US dollar, rates),
  each a GARCH(1,1) process with ``alpha = 0.08``, ``beta = 0.90`` and
  ``omega`` set so that the unconditional annualised volatility is 10%
  (252 days), Gaussian innovations; ``lambda_n`` is the asset's loading
  vector, its class loading (Equity ``(0.6, -0.1, -0.1)``, FX
  ``(0.3, -0.6, 0.0)``, Fixed income ``(-0.2, 0.1, 0.5)``) plus ``N(0, 0.2^2)``
  per element; ``e_{n,t}`` is Student-t (5 degrees of freedom) noise scaled
  to unit variance times ``sigma_idio_n``. The asset's annualised volatility
  target is the class target (Equity 10%, FX 9%, Fixed income 5%) times a
  multiplier ``U(0.6, 1.5)``; with ``Sigma_F`` the unconditional daily factor
  covariance and ``target`` the daily target volatility,
  ``sigma_idio^2 = max(target^2 - lambda' Sigma_F lambda, 0.1 target^2)``.

Randomness (D71): every draw comes from ``np.random.default_rng([seed, STREAM, ...])``
with a fixed stream integer per component. The factors use one stream; the
loadings and multiplier, and the idiosyncratic noise, use one stream each,
further keyed by a stable hash of the ``asset_id``, so an asset's artificial
series does not depend on which other assets are in the universe (the
artificial fill of a failed listed asset equals its series in the
all-artificial mode).

Validity boundaries
-------------------
* When an asset's factor variance ``lambda' Sigma_F lambda`` exceeds 90% of
  its target variance (common for Fixed income, whose factor variance is
  about 6.5% annualised against a 3-7.5% target), the loading vector is
  shrunk to exactly 90% so that the floor ``0.1 target^2`` of
  ``sigma_idio^2`` holds with equality and the total volatility equals the
  target, as G.2.2 requires. The shrink factor is reported by
  :func:`artificial_parameters` (``loading_scale``).
* The GARCH factors start at their unconditional variance (no burn-in);
  the factor paths depend on the calendar length and start day, so two
  calendars give different (equally valid) draws.
* Real returns are read as stored; days outside the file's range are
  ``NaN``. A listed asset counts as failed (and is filled with artificial
  returns by :func:`build_market`) when a leg or the asset has status
  ``failed`` in ``manifest.json``, or its column is missing or all-``NaN`` in
  the requested window. ``partial`` legs keep their ``NaN`` days.
"""

from __future__ import annotations

import functools
import json
import logging
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MAX_GENERIC_ASSETS, UniverseConfig
from .reference import ASSET_CLASSES, _file_key, data_dir, load_assets, load_legs, market_dir
from .types import MarketData

logger = logging.getLogger(__name__)

__all__ = [
    "FACTOR_NAMES",
    "CLASS_LOADINGS",
    "CLASS_VOL_TARGET",
    "ASSET_TABLE_COLUMNS",
    "weekday_calendar",
    "generic_asset_table",
    "garch_factors",
    "artificial_parameters",
    "artificial_returns",
    "load_real_returns",
    "build_market",
]

# ---------------------------------------------------------------------------
# Model constants (G.2.2)
# ---------------------------------------------------------------------------
#: Latent market factors of the artificial model, in loading order.
FACTOR_NAMES: tuple[str, str, str] = ("risk_on", "usd", "rates")
#: Class loadings on (risk-on, US dollar, rates).
CLASS_LOADINGS: dict[str, tuple[float, float, float]] = {
    "Equity": (0.6, -0.1, -0.1),
    "FX": (0.3, -0.6, 0.0),
    "Fixed income": (-0.2, 0.1, 0.5),
}
#: Class annualised volatility targets.
CLASS_VOL_TARGET: dict[str, float] = {"Equity": 0.10, "FX": 0.09, "Fixed income": 0.05}
#: Class probabilities of generic assets.
GENERIC_CLASS_PROBS: dict[str, float] = {"Equity": 0.5, "FX": 0.3, "Fixed income": 0.2}

DAYS_PER_YEAR = 252
FACTOR_ANNUAL_VOL = 0.10
GARCH_ALPHA = 0.08
GARCH_BETA = 0.90
LOADING_SD = 0.2
VOL_MULTIPLIER_RANGE = (0.6, 1.5)
IDIO_DF = 5.0
IDIO_FLOOR_SHARE = 0.1

#: Fixed rng stream per random component (D71).
STREAM_FACTORS = 5601
STREAM_LOADINGS = 5602
STREAM_IDIO = 5603
STREAM_GENERIC_TABLE = 5604

#: Columns of ``MarketData.assets`` built here, in display order.
ASSET_TABLE_COLUMNS = [
    "order", "name", "asset_class", "sub_class", "long_leg", "short_leg", "source",
    "long_index", "long_proxy", "short_index", "short_proxy",
]
_DISPLAY_COLUMNS = ["long_index", "long_proxy", "short_index", "short_proxy"]

ASSET_RETURNS_FILE = "asset_returns.parquet"
MANIFEST_FILE = "manifest.json"
_MISSING_MARKET_HINT = (
    "Run scripts/fetch_market_data.py to create the market data store, "
    "or use price_source='artificial'."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_seed(seed: int) -> int:
    s = int(seed)
    if s < 0:
        raise ValueError("seed must be a non-negative integer")
    return s


def _asset_key(asset_id: str) -> int:
    """Stable 32-bit key of an asset id (CRC-32), used to key per-asset rng streams."""
    return int(zlib.crc32(str(asset_id).encode("utf-8")))


@functools.lru_cache(maxsize=8)
def _read_parquet_cached(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    logger.debug("reading %s (mtime_ns=%d, size=%d)", path, mtime_ns, size)
    return pd.read_parquet(path)


@functools.lru_cache(maxsize=8)
def _manifest_cached(path: str, mtime_ns: int, size: int) -> tuple[tuple, tuple, str]:
    """Leg statuses, asset entries and run timestamp of ``manifest.json`` (immutable)."""
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    legs = m.get("legs", {}) or {}
    assets = m.get("assets", {}) or {}
    leg_status = tuple((str(k), str((v or {}).get("status", ""))) for k, v in legs.items())
    asset_info = tuple(
        (str(k), str((v or {}).get("status", "")), str((v or {}).get("long_leg", "")),
         str((v or {}).get("short_leg", "")))
        for k, v in assets.items()
    )
    return leg_status, asset_info, str(m.get("run_timestamp_utc", ""))


def _read_manifest(mdir: Path) -> tuple[dict[str, str], dict[str, tuple[str, str, str]], str, str | None]:
    """``(leg_status, asset_info, timestamp, path)``; empty when the manifest is missing."""
    path = mdir / MANIFEST_FILE
    if not path.is_file():
        logger.warning("load_real_returns: %s not found; failed legs are detected from the data only", path)
        return {}, {}, "", None
    leg_status, asset_info, ts = _manifest_cached(*_file_key(path))
    return dict(leg_status), {a: (s, lo, sh) for a, s, lo, sh in asset_info}, ts, str(path.resolve())


# ---------------------------------------------------------------------------
# Calendar and generic assets
# ---------------------------------------------------------------------------
def weekday_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """All weekdays (Monday to Friday) from ``start`` to ``end`` inclusive (G.2.1).

    Parameters
    ----------
    start, end:
        First and last calendar day (anything ``pd.Timestamp`` accepts);
        times of day are dropped.

    Returns
    -------
    pandas.DatetimeIndex
        Named ``"date"``; holidays are not removed (the market data mark
        them as stale days with return 0).
    """
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    if s > e:
        raise ValueError(f"start {s.date()} is after end {e.date()}")
    cal = pd.bdate_range(s, e, name="date")
    if len(cal) == 0:
        raise ValueError(f"no weekday between {s.date()} and {e.date()}")
    return cal


def generic_asset_table(n: int, seed: int) -> pd.DataFrame:
    """``n`` generic assets ``G_ASSET_001, ...`` with a random asset class (G.2 point 2).

    The class of each asset is drawn independently with probabilities Equity
    0.5, FX 0.3, Fixed income 0.2 from the stream ``[seed, 5604]``; the draw
    of asset ``i`` does not depend on ``n`` (the table for ``n`` is a prefix of
    the table for any larger ``n``).

    Parameters
    ----------
    n:
        Number of assets (1 to 500).
    seed:
        Non-negative seed.

    Returns
    -------
    pandas.DataFrame
        Index ``asset_id``; columns ``order`` (1..n), ``name`` (for example
        "Generic asset 001 (Equity)"), ``long_leg`` and ``short_leg`` (``""``),
        ``asset_class``, ``sub_class`` (``"generic"``) and ``source``
        (``"generic"``).
    """
    n = int(n)
    if not 1 <= n <= MAX_GENERIC_ASSETS:
        raise ValueError(f"n must be in [1, {MAX_GENERIC_ASSETS}]")
    rng = np.random.default_rng([_check_seed(seed), STREAM_GENERIC_TABLE])
    classes = list(GENERIC_CLASS_PROBS)
    cdf = np.cumsum(list(GENERIC_CLASS_PROBS.values()))
    idx = np.minimum(np.searchsorted(cdf, rng.random(n), side="right"), len(classes) - 1)
    asset_class = [classes[i] for i in idx]
    width = max(3, len(str(n)))
    ids = [f"G_ASSET_{i:0{width}d}" for i in range(1, n + 1)]
    table = pd.DataFrame(
        {
            "order": np.arange(1, n + 1, dtype=np.int64),
            "name": [f"Generic asset {i:0{width}d} ({c})" for i, c in zip(range(1, n + 1), asset_class)],
            "long_leg": "",
            "short_leg": "",
            "asset_class": asset_class,
            "sub_class": "generic",
            "source": "generic",
        },
        index=pd.Index(ids, name="asset_id"),
    )
    return table


# ---------------------------------------------------------------------------
# Artificial returns (G.2.2)
# ---------------------------------------------------------------------------
def _factor_daily_var() -> float:
    return FACTOR_ANNUAL_VOL**2 / DAYS_PER_YEAR


def garch_factors(calendar: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """The three latent GARCH(1,1) factors ``F_t`` of G.2.2.

    Each factor follows ``F_t = sqrt(h_t) z_t`` with ``z_t`` i.i.d. standard
    normal and conditional variance ``h_t = omega + alpha F_{t-1}^2 + beta h_{t-1}``,
    ``alpha = 0.08``, ``beta = 0.90``, ``omega = v (1 - alpha - beta)`` where
    ``v = 0.10^2 / 252`` is the unconditional daily variance (10% annualised);
    ``h`` starts at ``v``. The factors are independent of each other.

    Parameters
    ----------
    calendar:
        Trading days (rows of the result).
    seed:
        Non-negative seed; the stream is ``[seed, 5601]``.

    Returns
    -------
    pandas.DataFrame
        ``(len(calendar), 3)`` with columns :data:`FACTOR_NAMES`.
    """
    T = len(calendar)
    rng = np.random.default_rng([_check_seed(seed), STREAM_FACTORS])
    z = rng.standard_normal((T, len(FACTOR_NAMES)))
    v = _factor_daily_var()
    omega = v * (1.0 - GARCH_ALPHA - GARCH_BETA)
    f = np.empty_like(z)
    h = np.full(len(FACTOR_NAMES), v)
    for t in range(T):
        f[t] = np.sqrt(h) * z[t]
        h = omega + GARCH_ALPHA * f[t] ** 2 + GARCH_BETA * h
    return pd.DataFrame(f, index=pd.DatetimeIndex(calendar, name="date"), columns=list(FACTOR_NAMES))


def _check_asset_table(assets: pd.DataFrame) -> None:
    if "asset_class" not in assets.columns:
        raise ValueError("assets needs an 'asset_class' column")
    if assets.index.has_duplicates:
        raise ValueError("assets index (asset_id) has duplicates")
    bad = sorted(set(map(str, assets["asset_class"])) - set(ASSET_CLASSES))
    if bad:
        raise ValueError(f"unknown asset_class {bad}; allowed {list(ASSET_CLASSES)}")


def artificial_parameters(assets: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Per-asset parameters of the artificial model (G.2.2).

    For asset ``n`` the stream ``[seed, 5602, crc32(asset_id)]`` draws the
    three loading deviations ``N(0, 0.2^2)`` and then the volatility
    multiplier ``U(0.6, 1.5)``.

    Parameters
    ----------
    assets:
        Index ``asset_id``; column ``asset_class`` (Equity | FX | Fixed income).
    seed:
        Non-negative seed.

    Returns
    -------
    pandas.DataFrame
        Index ``asset_id``; columns ``asset_class``, ``lam_risk_on``,
        ``lam_usd``, ``lam_rates`` (loadings ``lambda_n`` after any shrink),
        ``vol_multiplier``, ``target_vol`` (annualised target),
        ``loading_scale`` (1, or the shrink applied so that the factor
        variance is at most 90% of the target variance), ``factor_vol`` and
        ``idio_vol`` (annualised volatilities of ``lambda' F`` and ``e``) and
        ``sigma_idio`` (daily standard deviation of ``e``).
    """
    _check_asset_table(assets)
    seed = _check_seed(seed)
    fvar_d = _factor_daily_var()
    rows = []
    for asset_id, cls in zip(assets.index, assets["asset_class"]):
        cls = str(cls)
        rng = np.random.default_rng([seed, STREAM_LOADINGS, _asset_key(asset_id)])
        lam = np.asarray(CLASS_LOADINGS[cls], dtype=float) + rng.normal(0.0, LOADING_SD, size=3)
        mult = float(rng.uniform(*VOL_MULTIPLIER_RANGE))
        target_ann = CLASS_VOL_TARGET[cls] * mult
        target_var_d = target_ann**2 / DAYS_PER_YEAR
        factor_var_d = float(lam @ lam) * fvar_d  # lambda' Sigma_F lambda, Sigma_F = v I
        cap = (1.0 - IDIO_FLOOR_SHARE) * target_var_d
        scale = 1.0
        if factor_var_d > cap:
            scale = float(np.sqrt(cap / factor_var_d))
            lam = lam * scale
            factor_var_d = cap
        idio_var_d = max(target_var_d - factor_var_d, IDIO_FLOOR_SHARE * target_var_d)
        rows.append(
            {
                "asset_class": cls,
                "lam_risk_on": lam[0],
                "lam_usd": lam[1],
                "lam_rates": lam[2],
                "vol_multiplier": mult,
                "target_vol": target_ann,
                "loading_scale": scale,
                "factor_vol": float(np.sqrt(factor_var_d * DAYS_PER_YEAR)),
                "idio_vol": float(np.sqrt(idio_var_d * DAYS_PER_YEAR)),
                "sigma_idio": float(np.sqrt(idio_var_d)),
            }
        )
    out = pd.DataFrame(rows, index=pd.Index(assets.index, name="asset_id"))
    n_shrunk = int((out["loading_scale"] < 1.0).sum()) if len(out) else 0
    if n_shrunk:
        logger.debug("artificial_parameters: loadings shrunk for %d of %d assets (factor variance > 90%% of target)",
                     n_shrunk, len(out))
    return out


def artificial_returns(assets: pd.DataFrame, calendar: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Artificial daily returns ``r_{n,t} = lambda_n' F_t + e_{n,t}`` (G.2.2, D56).

    Parameters
    ----------
    assets:
        Index ``asset_id``; column ``asset_class`` (Equity | FX | Fixed income).
    calendar:
        Trading days (rows of the result).
    seed:
        Non-negative seed; the same seed, assets and calendar give the same
        series. Factors come from :func:`garch_factors`, loadings from
        :func:`artificial_parameters` and the noise ``e_{n,t}`` (Student-t
        with 5 degrees of freedom, scaled to unit variance, times
        ``sigma_idio_n``) from the stream ``[seed, 5603, crc32(asset_id)]``.

    Returns
    -------
    pandas.DataFrame
        ``(len(calendar), N)`` daily simple returns in decimals, index
        ``date``, columns ``assets.index``; no missing values.
    """
    cal = pd.DatetimeIndex(calendar, name="date")
    params = artificial_parameters(assets, seed)
    seed = _check_seed(seed)
    T, N = len(cal), len(params)
    F = garch_factors(cal, seed).to_numpy()
    lam = params[["lam_risk_on", "lam_usd", "lam_rates"]].to_numpy(dtype=float)
    r = F @ lam.T if N else np.empty((T, 0))
    t_scale = np.sqrt(IDIO_DF / (IDIO_DF - 2.0))  # sd of a Student-t(5) draw
    for j, asset_id in enumerate(params.index):
        rng = np.random.default_rng([seed, STREAM_IDIO, _asset_key(asset_id)])
        r[:, j] += rng.standard_t(IDIO_DF, size=T) / t_scale * params["sigma_idio"].iat[j]
    return pd.DataFrame(r, index=cal, columns=pd.Index(params.index, name="asset_id"))


# ---------------------------------------------------------------------------
# Real returns (G.2.1, D55)
# ---------------------------------------------------------------------------
def load_real_returns(asset_ids: list[str], start: str, end: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Real daily returns of listed assets from ``data/market/asset_returns.parquet`` (G.2.1, D54-D55).

    Parameters
    ----------
    asset_ids:
        Listed asset ids; the result has these columns in this order.
    start, end:
        First and last day; rows are :func:`weekday_calendar` ``(start, end)``.

    Returns
    -------
    returns:
        ``(n_days, N)`` daily simple returns (``r_long - r_short``); ``NaN``
        where not observed (outside the file's range, missing columns,
        ``partial`` legs). Infinite values are set to ``NaN``.
    meta:
        ``failed_assets`` (list, in ``asset_ids`` order: a leg or the asset has
        status ``failed`` in ``manifest.json``, or the column is missing or
        all-``NaN`` in the window), ``failed_reasons`` (asset -> reason),
        ``failed_legs``, ``coverage`` (asset -> share of days with a finite
        return), ``data_dir``, ``market_dir``, ``source_file``, ``manifest``
        (path or ``None``) and ``manifest_timestamp``.

    Raises
    ------
    FileNotFoundError
        If the parquet file is missing; the message says how to create it or
        to use the artificial price source.
    """
    ids = [str(a) for a in asset_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("asset_ids contains duplicates")
    mdir = market_dir()
    path = mdir / ASSET_RETURNS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"market data not found: {path}. {_MISSING_MARKET_HINT}")
    raw = _read_parquet_cached(*_file_key(path))
    idx = pd.DatetimeIndex(pd.to_datetime(raw.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    raw = raw.set_axis(idx.normalize(), axis=0)
    if raw.index.has_duplicates:
        raise ValueError(f"{path}: duplicate dates in the index")
    raw = raw.sort_index()
    raw.columns = [str(c) for c in raw.columns]

    cal = weekday_calendar(start, end)
    returns = raw.reindex(index=cal, columns=ids).astype(float)
    returns.index.name = "date"
    returns.columns.name = "asset_id"
    arr = returns.to_numpy()
    n_inf = int(np.isinf(arr).sum())
    if n_inf:
        logger.warning("load_real_returns: %d infinite returns set to NaN", n_inf)
        returns = returns.mask(np.isinf(arr))
    if len(raw.index) and (cal[0] < raw.index[0] or cal[-1] > raw.index[-1]):
        logger.warning("load_real_returns: window %s..%s extends beyond the data (%s..%s); those days are NaN",
                       cal[0].date(), cal[-1].date(), raw.index[0].date(), raw.index[-1].date())

    leg_status, asset_info, ts, manifest_path = _read_manifest(mdir)
    try:
        ref = load_assets()
        ref_legs = {a: (ref.at[a, "long_leg"], ref.at[a, "short_leg"]) for a in ref.index}
    except FileNotFoundError:
        ref_legs = {}

    finite = np.isfinite(returns.to_numpy())
    coverage = {a: float(finite[:, j].mean()) for j, a in enumerate(ids)}
    reasons: dict[str, str] = {}
    failed_legs: list[str] = []
    for j, a in enumerate(ids):
        legs = ref_legs.get(a) or (asset_info[a][1:] if a in asset_info else ())
        bad_legs = [leg for leg in legs if leg and leg_status.get(leg) == "failed"]
        for leg in bad_legs:
            if leg not in failed_legs:
                failed_legs.append(leg)
        if bad_legs:
            reasons[a] = f"leg {', '.join(bad_legs)} failed"
        elif a in asset_info and asset_info[a][0] == "failed":
            reasons[a] = "asset status failed"
        elif a not in raw.columns:
            reasons[a] = "missing column"
        elif not finite[:, j].any():
            reasons[a] = "no data in the window"
    failed = [a for a in ids if a in reasons]
    if failed:
        logger.warning("load_real_returns: %d failed assets: %s", len(failed), reasons)
    meta = {
        "failed_assets": failed,
        "failed_reasons": reasons,
        "failed_legs": failed_legs,
        "coverage": coverage,
        "data_dir": str(data_dir()),
        "market_dir": str(mdir),
        "source_file": str(path.resolve()),
        "manifest": manifest_path,
        "manifest_timestamp": ts,
    }
    return returns, meta


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
def _listed_asset_table(listed_assets: tuple[str, ...] | None) -> pd.DataFrame:
    """Reference assets (optionally a subset, in reference order) with leg display columns."""
    ref = load_assets()
    if listed_assets is not None:
        wanted = [str(a) for a in listed_assets]
        unknown = [a for a in wanted if a not in ref.index]
        if unknown:
            raise ValueError(f"unknown listed asset ids {unknown}; known ids are in data/reference/assets.csv")
        ref = ref.loc[ref.index.isin(wanted)]
    legs = load_legs()
    used = pd.unique(pd.concat([ref["long_leg"], ref["short_leg"]]).to_numpy())
    unknown_legs = [leg for leg in used if leg not in legs.index]
    if unknown_legs:
        raise ValueError(f"assets.csv refers to legs missing from legs.csv: {unknown_legs}")
    is_cash = legs.index.to_series() == "cash"
    if "leg_type" in legs.columns:
        is_cash = is_cash | (legs["leg_type"] == "cash")
    index = legs["benchmark_index"].where(~is_cash, "")
    proxy = legs["proxy_ticker"].where(~is_cash, "")
    out = ref.copy()
    out["long_index"] = index.reindex(out["long_leg"]).to_numpy()
    out["long_proxy"] = proxy.reindex(out["long_leg"]).to_numpy()
    out["short_index"] = index.reindex(out["short_leg"]).to_numpy()
    out["short_proxy"] = proxy.reindex(out["short_leg"]).to_numpy()
    return out


def build_market(cfg: UniverseConfig) -> MarketData:
    """Asset table and daily returns of the universe (G.2, D54-D56).

    * ``asset_source = "listed"``: the reference assets (all, or the subset
      ``cfg.listed_assets`` in reference order), with the display columns
      ``long_index``, ``long_proxy``, ``short_index``, ``short_proxy`` from
      ``legs.csv`` (empty for the cash leg). With ``price_source = "real"`` the
      returns come from :func:`load_real_returns`, and every failed asset is
      replaced by :func:`artificial_returns` (seed ``cfg.seed``) and flagged
      ``source = "artificial"``; the others have ``source = "real"``. With
      ``price_source = "artificial"`` every asset is artificial.
    * ``asset_source = "generic"``: :func:`generic_asset_table` with
      ``cfg.n_generic_assets`` assets and artificial returns
      (``source = "generic"``, display columns empty).

    Parameters
    ----------
    cfg:
        :class:`~narrative_ipca.exposure_lab.config.UniverseConfig`.

    Returns
    -------
    MarketData
        ``assets`` with columns :data:`ASSET_TABLE_COLUMNS` (index
        ``asset_id``), ``returns`` on :func:`weekday_calendar` ``(cfg.start,
        cfg.end)`` with columns equal to ``assets.index``, and ``meta`` with
        ``asset_source``, ``price_source``, ``failed_assets``,
        ``failed_reasons``, ``coverage`` (asset -> share of finite days in the
        returned series), ``n_days``, ``n_assets``, ``start``, ``end``,
        ``data_dir``, ``seed`` and, for real prices, ``coverage_real`` and the
        manifest provenance.

    Raises
    ------
    ValueError
        On unknown listed asset ids.
    FileNotFoundError
        If real prices are requested and the market data store is missing.
    """
    cal = weekday_calendar(cfg.start, cfg.end)
    seed = _check_seed(cfg.seed)
    extra: dict[str, Any] = {}
    failed: list[str] = []
    reasons: dict[str, str] = {}
    if cfg.asset_source == "generic":
        assets = generic_asset_table(cfg.n_generic_assets, seed)
        for c in _DISPLAY_COLUMNS:
            assets[c] = ""
        returns = artificial_returns(assets, cal, seed)
        price_source = "artificial"
    else:
        assets = _listed_asset_table(cfg.listed_assets)
        price_source = cfg.price_source
        if price_source == "artificial":
            assets["source"] = "artificial"
            returns = artificial_returns(assets, cal, seed)
        else:
            returns, rmeta = load_real_returns(list(assets.index), cfg.start, cfg.end)
            failed = list(rmeta["failed_assets"])
            reasons = dict(rmeta["failed_reasons"])
            source = pd.Series("real", index=assets.index, dtype="str")
            if failed:
                fill = artificial_returns(assets.loc[failed], cal, seed)
                for a in failed:
                    returns[a] = fill[a].to_numpy()
                source[failed] = "artificial"
                logger.warning("build_market: filled %d failed listed assets with artificial returns: %s",
                               len(failed), failed)
            assets["source"] = source
            extra = {
                "coverage_real": rmeta["coverage"],
                "failed_legs": rmeta["failed_legs"],
                "manifest": rmeta["manifest"],
                "manifest_timestamp": rmeta["manifest_timestamp"],
                "source_file": rmeta["source_file"],
            }

    assets = assets[ASSET_TABLE_COLUMNS].copy()
    assets.index.name = "asset_id"
    returns = returns[list(assets.index)].copy()
    returns.columns = pd.Index(assets.index, name="asset_id")
    returns.index = pd.DatetimeIndex(returns.index, name="date")

    finite = np.isfinite(returns.to_numpy())
    meta: dict[str, Any] = {
        "asset_source": cfg.asset_source,
        "price_source": price_source,
        "failed_assets": failed,
        "failed_reasons": reasons,
        "coverage": {a: float(finite[:, j].mean()) for j, a in enumerate(assets.index)},
        "n_days": int(len(cal)),
        "n_assets": int(len(assets)),
        "start": cal[0].date().isoformat(),
        "end": cal[-1].date().isoformat(),
        "data_dir": str(data_dir()),
        "seed": seed,
        "sources": {str(k): int(v) for k, v in assets["source"].value_counts().items()},
        **extra,
    }
    logger.info("build_market: %s assets (%s prices), %d days %s..%s, sources %s",
                len(assets), price_source, len(cal), meta["start"], meta["end"], meta["sources"])
    return MarketData(returns=returns, assets=assets, meta=meta)
