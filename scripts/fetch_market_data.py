#!/usr/bin/env python
"""Stage daily market data for the research dashboard: 55 assets built from long/short legs.

Inputs (registries in data/reference/ unless overridden)
    legs.csv    one row per leg: proxy ticker, source, fallbacks, currencies, conventions.
    assets.csv  one row per asset: long_leg and short_leg (leg_ids), in dashboard order.
    No ticker is hard-coded in this script; edit the registries to change proxies.

Outputs (in --out)
    raw/<ticker-or-series>.csv  each series as downloaded (date, close, adj_close if any).
    leg_levels.parquet          weekday calendar x leg_id, level in the leg's return convention.
    leg_stale.parquet           True where a level was forward-filled (no fresh observation).
    leg_returns.parquet/.csv    simple daily leg returns.
    asset_returns.parquet/.csv  55 columns (asset_id): long-leg return minus short-leg return.
    manifest.json               per-leg source actually used, coverage, status, diagnostics.

Conventions
    Calendar     All weekdays (Mon-Fri), no holiday calendar. Levels run from --first-level
                 (2014-12-31); returns from --first-return (2015-01-02) to --last-date
                 (2025-12-31), inclusive.
    Forward-fill Leg levels are forward-filled over weekdays without a fresh observation for at
                 most --ffill-limit (5) consecutive weekdays. pandas limit semantics: in a longer
                 gap the first 5 weekdays are filled and the rest stay NaN. leg_stale marks
                 every filled cell. Returns on stale days are therefore 0.
    Returns      r_t = L_t / L_{t-1} - 1 on the weekday calendar (L = leg level).
                 Asset return = r_long - r_short. The 'cash' leg has level 1 and return 0, so
                 outright assets (Global Duration, Global Equity, Global Credit, USD) and every
                 'XXX v USD' pair are long vs cash.
    Price field  tr_or_price 'TR' uses Yahoo 'Adj Close' (downloaded with auto_adjust=False,
                 which gives both close and adj_close; 'Adj Close' equals 'Close' under
                 auto_adjust=True, i.e. dividends and splits reinvested). 'price' and 'spot'
                 use 'Close'. FRED series have a single value column.
    FX legs      Spot only, no carry. Level = USD price of one unit of XXX, so the return of
                 'XXX v USD' is the % change of that price. invert=True for Yahoo XXX=X quotes
                 (XXX per USD). The default FX fallback is FRED H.10 (noon New York).
    FX dates     Yahoo 'XXX=X' daily bars hold the snapshot taken at the start of the labelled
                 day (about 00:00 London, i.e. the end of the previous New York day). Evidence:
                 same-day correlation with US-listed currency ETFs (FXE, FXY, FXF) is ~0.05, and
                 ~0.95 when the Yahoo bar is moved one weekday earlier; the SNB floor removal
                 (2015-01-15) and the BoJ YCC change (2022-12-20) show up on the next day's bar.
                 Every Yahoo '=X' observation is therefore re-labelled to the previous weekday
                 (disable with --no-fx-redate). DX-Y.NYB and FRED series are dated correctly.
    FX cleaning  Yahoo FX legs: (1) a print identical to the previous one is a vendor
                 carry-forward and is dropped (repeated_prints_dropped); (2) isolated bad ticks
                 (one print far from both neighbours that reverses the next day; threshold
                 max(6 robust sigmas, 2%); see remove_bad_ticks) are dropped (bad_ticks_removed). Dropped prints become
                 ordinary missing days: forward-filled up to 5 weekdays and flagged stale, and a
                 longer run counts as a gap > 5 weekdays (FRED fallback). --no-fx-clean disables.
    FX quality   When a FRED H.10 fallback exists, Yahoo is compared with it; if the weekly-return
                 correlation is below --fx-min-weekly-corr (0.8) Yahoo counts as failed and FRED
                 is used (manifest 'quality_gate').
    Currencies   quote_ccy 'GBp' (pence) is divided by 100. When a non-FX leg's quote currency
                 differs from its return_ccy, the level is converted on the proxy's own trading
                 days, before the calendar forward-fill:
                     level_ret = level_quote * usd_per(quote_ccy) / usd_per(return_ccy)
                 where usd_per(XXX) is the level of leg fx_<xxx> (USD per 1 XXX), or with
                 --conversion-fx fred the FRED H.10 series of that leg.
                 Example: gov_jp_7_10 = XJSE.DE (EUR) x EURUSD / (USD per JPY) = JPY-local level.
    100x jumps   Daily level jumps of ~100x or ~1/100 on Yahoo series (pence/pound mix-ups on
                 LSE listings, decimal slips on FX) are rescaled to the series' majority unit
                 and counted in the manifest (fixes_100x). raw/ keeps the unfixed data.
    Fallbacks    Candidates are the proxy, then fallback_tickers in order. The first candidate
                 with no missing level in the delivered window (after forward-fill) that passes
                 the FX quality gate is used; if none qualifies, the candidate with the most
                 valid levels is used and the leg is 'partial' if it has gaps. For FX this means
                 FRED H.10 replaces Yahoo when Yahoo fails or has a gap longer than 5 weekdays.
                 Whole-series switch, no splicing.
                 Token grammar:  [yahoo:|fred:]TICKER[@QUOTE_CCY][/inv]
                 e.g. 'fred:DEXSZUS/inv' (CHF per USD, inverted), 'yahoo:ISF.L@GBp'.
    Constructed  source 'constructed' (fx_em_basket): equal-weight, daily-rebalanced basket.
                 Basket return = mean of the available component leg returns on that day
                 (requires at least --min-basket-frac of the components); level rebased to
                 100 on --first-level. A basket day is stale when no component is fresh.
    Failures     A ticker that fails is recorded (manifest 'downloads') and never stops the run.
                 A leg with no usable candidate is left NaN and marked 'failed'; nothing is
                 synthesised here (artificial fill belongs to the dashboard code).

Usage
    python fetch_market_data.py                      # full download + build into this folder
    python fetch_market_data.py --out D:/staging --start 2014-12-01 --end 2026-01-05
    python fetch_market_data.py --offline            # rebuild from raw/*.csv, no network

Requires pandas (3.x compatible), numpy, yfinance (1.x), pyarrow.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import platform
import random
import re
import sys
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REFERENCE_DIR = REPO / "data" / "reference"
MARKET_DIR = REPO / "data" / "market"
LOG = logging.getLogger("fetch_market_data")

TOKEN_RE = re.compile(r"^(?:(?P<src>yahoo|fred):)?(?P<tkr>[^@/]+?)(?:@(?P<ccy>[A-Za-z]{3}))?(?:/(?P<inv>inv))?$")
FX_LEG_RE = re.compile(r"^fx_([a-z]{3})$")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}&coed={end}"


# --------------------------------------------------------------------------------------------
# registries
# --------------------------------------------------------------------------------------------
@dataclass
class Candidate:
    source: str          # yahoo | fred
    ticker: str
    quote_ccy: str
    invert: bool
    is_fallback: bool
    token: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.ticker)


def as_bool(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes", "y")


def load_registries(legs_path: Path, assets_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    legs = pd.read_csv(legs_path, dtype=str, keep_default_na=False)
    assets = pd.read_csv(assets_path, dtype=str, keep_default_na=False)
    need_l = {"leg_id", "proxy_ticker", "source", "fallback_tickers", "quote_ccy", "return_ccy",
              "tr_or_price", "invert", "leg_type", "components"}
    need_a = {"order", "asset_id", "name", "long_leg", "short_leg"}
    miss = (need_l - set(legs.columns)) | (need_a - set(assets.columns))
    if miss:
        raise SystemExit(f"registry columns missing: {sorted(miss)}")
    if legs["leg_id"].duplicated().any():
        raise SystemExit(f"duplicate leg_id: {legs.loc[legs['leg_id'].duplicated(), 'leg_id'].tolist()}")
    if assets["asset_id"].duplicated().any():
        raise SystemExit(f"duplicate asset_id: {assets.loc[assets['asset_id'].duplicated(), 'asset_id'].tolist()}")
    assets = assets.assign(order=assets["order"].astype(int)).sort_values("order").reset_index(drop=True)
    return legs, assets


def parse_token(tok: str, leg: pd.Series) -> Candidate:
    m = TOKEN_RE.match(tok.strip())
    if not m:
        raise ValueError(f"bad fallback token {tok!r}")
    src = m.group("src") or "yahoo"
    ccy = m.group("ccy")
    if ccy is None:
        ccy = leg["quote_ccy"] if src == "yahoo" else ("USD" if leg["leg_type"] == "fx" else leg["quote_ccy"])
    elif ccy.lower() == "gbp" and ccy != "GBP":
        ccy = "GBp"
    else:
        ccy = ccy.upper()
    return Candidate(src, m.group("tkr").strip(), ccy, m.group("inv") is not None, True, tok.strip())


def leg_candidates(leg: pd.Series) -> list[Candidate]:
    out = []
    if leg["source"] in ("yahoo", "fred") and leg["proxy_ticker"]:
        out.append(Candidate(leg["source"], leg["proxy_ticker"], leg["quote_ccy"], as_bool(leg["invert"]),
                             False, f"{leg['source']}:{leg['proxy_ticker']}"))
    for tok in [t for t in leg["fallback_tickers"].split(";") if t.strip()]:
        out.append(parse_token(tok, leg))
    return out


# --------------------------------------------------------------------------------------------
# downloading
# --------------------------------------------------------------------------------------------
def safe_name(ticker: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", ticker)


def backoff_sleep(attempt: int, base: float = 1.5, cap: float = 60.0) -> None:
    time.sleep(min(cap, base * (2 ** attempt)) + random.uniform(0, 1))


class RawStore:
    """Downloads (or, offline, reads back) raw series; keeps a per-ticker download log."""

    def __init__(self, raw_dir: Path, start: str, end: str, offline: bool, max_retries: int, batch_size: int):
        self.raw_dir = raw_dir
        self.start, self.end = start, end
        self.offline = offline
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.data: dict[tuple[str, str], pd.DataFrame | None] = {}
        self.log: dict[str, dict[str, dict]] = {"yahoo": {}, "fred": {}}
        raw_dir.mkdir(parents=True, exist_ok=True)

    # ---- public -------------------------------------------------------------------------
    def ensure(self, keys: list[tuple[str, str]]) -> None:
        todo = [k for k in dict.fromkeys(keys) if k not in self.data]
        if not todo:
            return
        if self.offline:
            for k in todo:
                self._read_raw(k)
            return
        ytk = [t for s, t in todo if s == "yahoo"]
        ftk = [t for s, t in todo if s == "fred"]
        if ytk:
            self._yahoo(ytk)
        for sid in ftk:
            self._fred(sid)

    def get(self, key: tuple[str, str]) -> pd.DataFrame | None:
        self.ensure([key])
        return self.data.get(key)

    # ---- raw files ------------------------------------------------------------------------
    def _path(self, key):
        return self.raw_dir / f"{safe_name(key[1])}.csv"

    def _write_raw(self, key, df: pd.DataFrame) -> None:
        out = df.copy()
        out.index = out.index.strftime("%Y-%m-%d")
        out.index.name = "date"
        out.to_csv(self._path(key))

    def _read_raw(self, key) -> None:
        p = self._path(key)
        if not p.exists():
            self.data[key] = None
            self.log[key[0]][key[1]] = {"status": "failed", "error": "offline: raw file not found"}
            return
        # round_trip: the offline rebuild reads back exactly the floats the online run wrote
        df = pd.read_csv(p, parse_dates=["date"], index_col="date", float_precision="round_trip")
        self.data[key] = df
        self.log[key[0]][key[1]] = self._summary(df, "ok (offline raw)", 0, None)

    @staticmethod
    def _summary(df, status, attempts, err):
        c = df["close"].dropna() if df is not None and "close" in df else pd.Series(dtype=float)
        return {"status": status, "n_rows": int(len(c)),
                "first": str(c.index.min().date()) if len(c) else None,
                "last": str(c.index.max().date()) if len(c) else None,
                "attempts": attempts, "error": err}

    # ---- Yahoo ----------------------------------------------------------------------------
    @staticmethod
    def _yf_errors() -> dict:
        try:
            import yfinance.shared as shared
            return dict(getattr(shared, "_ERRORS", {}) or {})
        except Exception as e:
            LOG.debug("yfinance error table not available: %s", e)
            return {}

    def _yf_call(self, tickers: list[str]) -> tuple[pd.DataFrame | None, str | None]:
        import yfinance as yf
        try:
            df = yf.download(tickers, start=self.start, end=self.end, auto_adjust=False, actions=False,
                             group_by="ticker", threads=True, progress=False, repair=False, timeout=30,
                             multi_level_index=True)
            return df, None
        except Exception as e:  # rate limit, network, parsing
            return None, f"{type(e).__name__}: {str(e)[:200]}"

    @staticmethod
    def _extract(df: pd.DataFrame | None, t: str) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        try:
            if isinstance(df.columns, pd.MultiIndex):
                lv0 = df.columns.get_level_values(0)
                sub = df[t] if t in set(lv0) else None
            else:
                sub = df
            if sub is None or "Close" not in sub.columns:
                return None
            out = pd.DataFrame({"close": pd.to_numeric(sub["Close"], errors="coerce")})
            if "Adj Close" in sub.columns:
                out["adj_close"] = pd.to_numeric(sub["Adj Close"], errors="coerce")
            idx = pd.DatetimeIndex(out.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            out.index = idx.normalize()
            out = out[out["close"].notna()]
            out = out[~out.index.duplicated(keep="last")].sort_index()
            return out if len(out) else None
        except Exception as e:
            LOG.debug("could not extract %s from the Yahoo frame: %s", t, e)
            return None

    def _yahoo(self, tickers: list[str]) -> None:
        got: dict[str, pd.DataFrame] = {}
        attempts: dict[str, int] = {t: 0 for t in tickers}
        errors: dict[str, str | None] = {t: None for t in tickers}
        # batched pass, retrying whole batches on exceptions
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            for attempt in range(self.max_retries):
                LOG.info("yahoo batch %d-%d (%d tickers), attempt %d", i, i + len(batch) - 1, len(batch), attempt + 1)
                df, err = self._yf_call(batch)
                for t in batch:
                    attempts[t] += 1
                if err is None:
                    yerr = self._yf_errors()
                    for t in batch:
                        x = self._extract(df, t)
                        if x is not None:
                            got[t] = x
                        else:
                            errors[t] = str(yerr.get(t, "no data returned"))[:200]
                    break
                for t in batch:
                    errors[t] = err
                backoff_sleep(attempt)
        # individual retries for whatever is still missing
        for t in [t for t in tickers if t not in got]:
            for attempt in range(self.max_retries):
                backoff_sleep(attempt)
                LOG.info("yahoo retry %s, attempt %d", t, attempt + 1)
                df, err = self._yf_call([t])
                attempts[t] += 1
                x = self._extract(df, t) if err is None else None
                if x is not None:
                    got[t] = x
                    errors[t] = None
                    break
                errors[t] = err or str(self._yf_errors().get(t, "no data returned"))[:200]
        for t in tickers:
            key = ("yahoo", t)
            x = got.get(t)
            self.data[key] = x
            if x is not None:
                self._write_raw(key, x)
                self.log["yahoo"][t] = self._summary(x, "ok", attempts[t], None)
            else:
                LOG.warning("yahoo %s failed: %s", t, errors[t])
                self.log["yahoo"][t] = self._summary(None, "failed", attempts[t], errors[t])

    # ---- FRED -----------------------------------------------------------------------------
    def _fred(self, sid: str) -> None:
        key = ("fred", sid)
        url = FRED_URL.format(sid=sid, start=self.start, end=self.end)
        err = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research data staging)"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    text = r.read().decode("utf-8")
                raw = pd.read_csv(io.StringIO(text), na_values=[".", ""])
                dcol = raw.columns[0]
                vcol = sid if sid in raw.columns else raw.columns[1]
                df = pd.DataFrame({"close": pd.to_numeric(raw[vcol], errors="coerce").to_numpy()},
                                  index=pd.DatetimeIndex(pd.to_datetime(raw[dcol]), name="date"))
                df = df[~df.index.duplicated(keep="last")].sort_index()
                if df["close"].notna().sum() == 0:
                    raise ValueError("series has no values")
                self.data[key] = df
                self._write_raw(key, df)
                self.log["fred"][sid] = self._summary(df, "ok", attempt + 1, None)
                return
            except urllib.error.HTTPError as e:
                err = f"HTTPError {e.code}"
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
            backoff_sleep(attempt)
        LOG.warning("fred %s failed: %s", sid, err)
        self.data[key] = None
        self.log["fred"][sid] = self._summary(None, "failed", self.max_retries, err)


# --------------------------------------------------------------------------------------------
# series helpers
# --------------------------------------------------------------------------------------------
def fix_100x(s: pd.Series) -> tuple[pd.Series, int]:
    """Rescale segments that sit 100x (or 1/100x) off the series' majority unit."""
    if len(s) < 3:
        return s, 0
    d = np.log10(s).diff()
    k = np.round(d / 2.0)
    jump = (k != 0) & ((d - 2.0 * k).abs() < 0.35)
    if not jump.any():
        return s, 0
    cum = k.where(jump, 0.0).fillna(0.0).cumsum()
    mode = cum.mode().iloc[0]
    shift = cum - mode
    fixed = s / (100.0 ** shift)
    return fixed, int((shift != 0).sum())


def max_run(mask: pd.Series) -> int:
    best = cur = 0
    for v in mask.to_numpy():
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def remove_bad_ticks(s: pd.Series, k_sigma: float, floor: float, max_pass: int = 6) -> tuple[pd.Series, list[str]]:
    """Drop isolated one-observation outliers from an FX level series (fresh observations only).

    An observation is a bad tick when, in log levels x, the move in (d1 = x_t - x_{t-1}) and the
    move out (d2 = x_{t+1} - x_t) both exceed thr, have opposite signs, and x_t sits more than thr
    away from the midpoint of its two neighbours. thr = max(k_sigma * sigma, floor), sigma = robust
    (MAD) std of daily log changes over a centred 125-observation window. Only the worst candidate
    in each cluster is dropped per pass, so a good print between two bad ones survives.
    """
    removed: list[str] = []
    x = np.log(s)
    for _ in range(max_pass):
        d1 = x.diff()
        d2 = x.shift(-1) - x
        mid = x - (x.shift(1) + x.shift(-1)) / 2.0
        sigma = 1.4826 * d1.abs().rolling(125, center=True, min_periods=20).median()
        thr = np.maximum(k_sigma * sigma.bfill().ffill(), floor)
        cand = (d1.abs() > thr) & (d2.abs() > thr) & (np.sign(d1) != np.sign(d2)) & (mid.abs() > thr)
        score = (mid.abs() / thr).where(cand, 0.0).fillna(0.0)
        peak = cand & (score >= score.shift(1, fill_value=0.0)) & (score >= score.shift(-1, fill_value=0.0))
        if not peak.any():
            break
        removed += [str(d.date()) for d in x.index[peak.to_numpy()]]
        x = x[~peak]
    return np.exp(x), sorted(removed)


def detect_spikes(r: pd.Series, n_show: int = 5) -> dict:
    """Informational: one-day moves that reverse the next day (|r_t|,|r_t+1| > thr, opposite signs,
    net < half). Real in stress periods (e.g. March 2020); a large count on a quiet series means noise."""
    x = r.dropna()
    x = x[x != 0]
    if len(x) < 50:
        return {"n": 0, "threshold": None, "dates": []}
    sigma = 1.4826 * (x - x.median()).abs().median()
    thr = max(6.0 * sigma, 0.015)
    nxt = r.shift(-1)
    m = (r.abs() > thr) & (nxt.abs() > thr) & (np.sign(r) != np.sign(nxt)) & ((r + nxt).abs() < 0.5 * r.abs())
    ds = r.index[m.fillna(False).to_numpy()]
    return {"n": int(len(ds)), "threshold": round(float(thr), 4),
            "dates": [f"{d.date()} ({r[d]:+.2%} then {nxt[d]:+.2%})" for d in ds[:n_show]]}


def compare_to_fred(y: pd.Series, f: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp) -> dict:
    """Yahoo vs FRED on common fresh dates (both in USD per unit)."""
    y = y[(y.index >= lo) & (y.index <= hi)]
    f = f[(f.index >= lo) & (f.index <= hi)]
    common = y.index.intersection(f.index)
    if len(common) < 50:
        return {"n_common": int(len(common))}
    ly, lf = np.log(y[common]), np.log(f[common])
    dev = (ly - lf)
    dev = dev - dev.rolling(21, center=True, min_periods=5).median()
    bad = dev.abs() > 0.03
    wy = ly.resample("W-FRI").last().diff()
    wf = lf.resample("W-FRI").last().diff()
    ry, rf = ly.diff(), lf.diff()
    lag = {str(k): round(float(ry.shift(-k).corr(rf)), 3) for k in (-1, 0, 1)}
    return {
        "n_common": int(len(common)),
        "corr_daily_ret_by_lag": lag,   # corr(yahoo r_{t+k}, fred r_t); peak at k=0 means dates line up
        "corr_daily_ret": round(float(ly.diff().corr(lf.diff())), 3),
        "corr_weekly_ret": round(float(wy.corr(wf)), 3),
        "_corr_weekly_exact": float(wy.corr(wf)),   # unrounded, for the quality gate (popped before the manifest)
        "median_abs_level_gap_pct": round(float((ly - lf).abs().median() * 100), 3),
        "n_days_level_disagree_gt3pct": int(bad.sum()),
        "disagree_dates": [str(d.date()) for d in dev.index[bad.to_numpy()][:8]],
    }


# --------------------------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------------------------
class Builder:
    def __init__(self, args, legs: pd.DataFrame, store: RawStore):
        self.args = args
        self.legs = legs
        self.store = store
        self.cal = pd.bdate_range(args.start, args.last_date, freq="B", name="date")
        self.lo = pd.Timestamp(args.first_level)
        self.hi = pd.Timestamp(args.last_date)
        self.win = (self.cal >= self.lo) & (self.cal <= self.hi)
        self.levels: dict[str, pd.Series] = {}
        self.stale: dict[str, pd.Series] = {}
        self.obs: dict[str, pd.Series] = {}      # fresh observations used (return convention)
        self.manifest: dict[str, dict] = {}
        self._conv_cache: dict[str, pd.Series] = {}

    # ---- basic ops ------------------------------------------------------------------------
    def align(self, obs: pd.Series) -> tuple[pd.Series, pd.Series]:
        o = obs[~obs.index.duplicated(keep="last")].reindex(self.cal)
        fresh = o.notna()
        lv = o.ffill(limit=self.args.ffill_limit)
        stale = lv.notna() & ~fresh
        return lv, stale

    def usd_per(self, ccy: str) -> pd.Series:
        """USD per 1 unit of ccy on the calendar (forward-filled like any leg).

        --conversion-fx legs: the fx_<ccy> leg level (default).
        --conversion-fx fred: the FRED H.10 fallback of leg fx_<ccy> (noon New York fixing).
        """
        if ccy == "USD":
            return pd.Series(1.0, index=self.cal)
        lid = f"fx_{ccy.lower()}"
        if self.args.conversion_fx == "fred":
            if lid not in self._conv_cache:
                row = self.legs.loc[self.legs["leg_id"] == lid]
                if row.empty:
                    raise RuntimeError(f"currency conversion needs leg {lid} in the registry")
                leg = row.iloc[0]
                fc = [c for c in leg_candidates(leg) if c.source == "fred"]
                obs = self.candidate_obs(fc[0], leg)[0] if fc else None
                if obs is None:
                    raise RuntimeError(f"no FRED series for {lid}")
                self._conv_cache[lid] = self.align(obs)[0]
            return self._conv_cache[lid]
        if lid not in self.levels or self.levels[lid].notna().sum() == 0:
            raise RuntimeError(f"currency conversion needs leg {lid}, which is unavailable")
        return self.levels[lid]

    def coverage(self, lv: pd.Series, stale: pd.Series) -> dict:
        w_lv, w_st = lv[self.win], stale[self.win]
        valid = w_lv.notna()
        return {
            "first_valid": str(w_lv.first_valid_index().date()) if valid.any() else None,
            "last_valid": str(w_lv.last_valid_index().date()) if valid.any() else None,
            "n_valid": int(valid.sum()),
            "n_stale": int(w_st.sum()),
            "n_missing": int((~valid).sum()),
        }

    # ---- candidate -> observations ----------------------------------------------------------
    def candidate_obs(self, cand: Candidate, leg: pd.Series) -> tuple[pd.Series | None, dict]:
        info: dict = {"token": cand.token}
        df = self.store.get(cand.key)
        if df is None or df.empty:
            info["error"] = "no data"
            return None, info
        want_tr = leg["tr_or_price"].upper().startswith("TR")
        if want_tr and "adj_close" in df.columns and df["adj_close"].notna().any():
            s = df["adj_close"].astype(float)
            info["price_field"] = "adj_close"
            ratio = (df["adj_close"] / df["close"]).dropna()
            ratio = ratio[(ratio.index >= self.lo) & (ratio.index <= self.hi)]
            if len(ratio) > 10:
                info["div_adjustment_detected"] = bool(ratio.max() / ratio.min() - 1 > 1e-3)
        else:
            s = df["close"].astype(float)
            info["price_field"] = "close"
            if want_tr:
                info["warning"] = "TR requested but no adj_close; used close"
        s = s.dropna()
        s = s[s > 0]
        if cand.source == "yahoo":
            s, nfix = fix_100x(s)
            info["fixes_100x"] = nfix
        if cand.source == "yahoo" and cand.ticker.endswith("=X") and not self.args.no_fx_redate:
            # Yahoo FX daily bars carry the snapshot taken at the START of the labelled day
            # (about 00:00 London = end of the previous New York day). Re-label to the previous weekday.
            s.index = s.index - pd.offsets.BDay(1)
            info["redated_minus_1_weekday"] = True
        n_we = int((s.index.dayofweek >= 5).sum())
        if n_we:
            info["n_weekend_obs_dropped"] = n_we
        s = s[s.index.dayofweek < 5]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        q = cand.quote_ccy
        if q == "GBp":
            s = s / 100.0
            q = "GBP"
        if cand.invert:
            s = 1.0 / s
            info["inverted"] = True
        if cand.source == "yahoo" and leg["leg_type"] == "fx" and not self.args.no_fx_clean:
            # an FX print identical to the previous one is a vendor carry-forward, not a fresh quote
            sw_all = s[(s.index >= self.lo) & (s.index <= self.hi)]
            rep = s.diff() == 0
            info["repeated_prints_dropped"] = {"n": int(rep[sw_all.index].sum()),
                                               "longest_run": max_run(rep[sw_all.index])}
            s = s[~rep]
            s, bad = remove_bad_ticks(s, self.args.fx_bad_tick_sigma, self.args.fx_bad_tick_floor)
            bad_w = [d for d in bad if self.args.first_level <= d <= self.args.last_date]
            info["bad_ticks_removed"] = {"n": len(bad_w), "dates": bad_w}
        if leg["leg_type"] not in ("fx", "cash") and q not in ("index", leg["return_ccy"]):
            fac = (self.usd_per(q) / self.usd_per(leg["return_ccy"])).reindex(s.index)
            s = (s * fac).dropna()
            info["currency_conversion"] = (f"{q}->{leg['return_ccy']} via usd_per({q})/usd_per({leg['return_ccy']}), "
                                           f"FX source: {self.args.conversion_fx}")
        if s.empty:
            info["error"] = "empty after cleaning/conversion"
            return None, info
        # stuck prices: identical consecutive observations inside the window
        sw = s[(s.index >= self.lo) & (s.index <= self.hi)]
        info["max_identical_run"] = max_run(sw.diff() == 0) + 1 if len(sw) > 1 else 0
        return s, info

    # ---- leg builders -------------------------------------------------------------------------
    def build_downloaded(self, leg: pd.Series) -> None:
        lid = leg["leg_id"]
        cands = leg_candidates(leg)
        tried = []
        chosen = None
        best = None
        for i, cand in enumerate(cands):
            try:
                obs, info = self.candidate_obs(cand, leg)
            except Exception as e:
                obs, info = None, {"token": cand.token, "error": f"{type(e).__name__}: {e}"}
            if obs is not None:
                lv, st = self.align(obs)
                cov = self.coverage(lv, st)
                info.update({k: cov[k] for k in ("n_valid", "n_missing")})
                info["max_gap_weekdays"] = max_run(~obs.reindex(self.cal).notna()[self.win])
                quality_ok = True
                # FX quality gate: Yahoo vs FRED H.10 weekly-return correlation (needs the FRED series)
                if (cand.source == "yahoo" and leg["leg_type"] == "fx" and not self.args.no_fred_check):
                    fc = [c for c in cands if c.source == "fred"]
                    f_obs = None
                    if fc:
                        try:
                            f_obs = self.candidate_obs(fc[0], leg)[0]
                        except Exception as e:
                            LOG.debug("FRED check series %s for %s failed: %s", fc[0].token, lid, e)
                            f_obs = None
                    if f_obs is not None:
                        cmp = compare_to_fred(obs, f_obs, self.lo, self.hi)
                        # QA 2026-09-29: gate on the unrounded correlation (the rounded value let
                        # 0.7995 <= corr < 0.8 pass a 'below 0.8' gate)
                        wc = cmp.pop("_corr_weekly_exact", None)
                        info["yahoo_vs_fred"] = {"fred_series": fc[0].ticker, **cmp}
                        if wc is not None and wc < self.args.fx_min_weekly_corr:
                            quality_ok = False
                            info["quality_gate"] = (f"failed: weekly-return corr with FRED {wc:.4f} < "
                                                    f"{self.args.fx_min_weekly_corr}")
                rec = (cand, obs, lv, st, cov, info)
                if best is None or cov["n_valid"] > best[4]["n_valid"]:
                    best = rec
                if cov["n_missing"] == 0 and quality_ok and chosen is None:
                    chosen = rec
            tried.append(info)
            if chosen is not None:
                break
        rec = chosen or best
        m = {"leg_type": leg["leg_type"], "candidates_tried": tried}
        if rec is None:
            self.levels[lid] = pd.Series(np.nan, index=self.cal)
            self.stale[lid] = pd.Series(False, index=self.cal)
            m.update({"source": None, "ticker": None, "fallback_used": False, "status": "failed",
                      **self.coverage(self.levels[lid], self.stale[lid])})
        else:
            cand, obs, lv, st, cov, info = rec
            self.levels[lid], self.stale[lid], self.obs[lid] = lv, st, obs
            status = "ok" if cov["n_missing"] == 0 else ("partial" if cov["n_valid"] else "failed")
            m.update({"source": cand.source, "ticker": cand.ticker, "fallback_used": cand.is_fallback,
                      "status": status, **cov})
            m["diagnostics"] = {k: v for k, v in info.items() if k not in ("token", "n_valid", "n_missing")}
        self.manifest[lid] = m

    def build_constructed(self, leg: pd.Series) -> None:
        lid = leg["leg_id"]
        comps = [c for c in leg["components"].split(";") if c.strip()]
        avail = [c for c in comps if c in self.levels and self.levels[c].notna().any()]
        m = {"leg_type": leg["leg_type"], "components": comps,
             "components_missing": [c for c in comps if c not in avail]}
        min_n = int(np.ceil(self.args.min_basket_frac * len(comps)))
        if len(avail) < min_n:
            self.levels[lid] = pd.Series(np.nan, index=self.cal)
            self.stale[lid] = pd.Series(False, index=self.cal)
            m.update({"source": "constructed", "ticker": None, "fallback_used": False, "status": "failed",
                      "error": f"only {len(avail)} of {len(comps)} components available (need {min_n})",
                      **self.coverage(self.levels[lid], self.stale[lid])})
            self.manifest[lid] = m
            return
        L = pd.DataFrame({c: self.levels[c] for c in avail})
        S = pd.DataFrame({c: self.stale[c] for c in avail})
        R = L / L.shift(1) - 1.0
        cnt = R.notna().sum(axis=1)
        r = R.mean(axis=1, skipna=True).where(cnt >= min_n)
        lvl = (1.0 + r.fillna(0.0)).cumprod()
        ok_row = (cnt >= min_n) | (lvl.index == lvl.index[0])
        lvl = lvl.where(ok_row)
        base = lvl.loc[self.lo] if pd.notna(lvl.get(self.lo, np.nan)) else lvl[lvl.index >= self.lo].dropna().iloc[0]
        lvl = 100.0 * lvl / base
        fresh_any = (L.notna() & ~S).any(axis=1)
        st = lvl.notna() & ~fresh_any
        self.levels[lid], self.stale[lid] = lvl, st
        cov = self.coverage(lvl, st)
        status = "ok" if cov["n_missing"] == 0 else ("partial" if cov["n_valid"] else "failed")
        cw = cnt[self.win]
        m.update({"source": "constructed", "ticker": None, "fallback_used": False, "status": status, **cov,
                  "diagnostics": {"method": "equal-weight daily-rebalanced mean of component simple returns",
                                  "min_components_required": min_n,
                                  "min_components_on_a_day": int(cw.iloc[1:].min()),
                                  "n_days_fewer_than_all": int((cw.iloc[1:] < len(comps)).sum())}})
        self.manifest[lid] = m

    def build_cash(self, leg: pd.Series) -> None:
        lid = leg["leg_id"]
        self.levels[lid] = pd.Series(1.0, index=self.cal)
        self.stale[lid] = pd.Series(False, index=self.cal)
        self.manifest[lid] = {"leg_type": "cash", "source": "none", "ticker": None, "fallback_used": False,
                              "status": "ok", **self.coverage(self.levels[lid], self.stale[lid]),
                              "diagnostics": {"note": "constant level 1, return 0"}}

    # ---- orchestration ------------------------------------------------------------------------
    def run(self) -> None:
        legs = self.legs
        # prefetch all primaries (and FRED counterparts of FX legs for the cross-check)
        keys = []
        for _, leg in legs.iterrows():
            cands = leg_candidates(leg)
            if cands:
                keys.append(cands[0].key)
            if leg["leg_type"] == "fx" and not self.args.no_fred_check:
                keys += [c.key for c in cands[1:] if c.source == "fred"]
        self.store.ensure(keys)
        # FX first (other legs convert with them), then everything downloaded, then constructed/cash
        def stage(i):
            if legs.at[i, "source"] in ("constructed", "none"):
                return 2
            return 0 if legs.at[i, "leg_type"] == "fx" else 1

        for i in sorted(legs.index, key=lambda i: (stage(i), i)):
            leg = legs.loc[i]
            lid = leg["leg_id"]
            try:
                if leg["source"] == "constructed":
                    self.build_constructed(leg)
                elif leg["source"] == "none" or leg["leg_type"] == "cash":
                    self.build_cash(leg)
                else:
                    self.build_downloaded(leg)
            except Exception as e:  # never let one leg kill the run
                LOG.exception("leg %s failed", lid)
                self.levels[lid] = pd.Series(np.nan, index=self.cal)
                self.stale[lid] = pd.Series(False, index=self.cal)
                self.manifest[lid] = {"leg_type": leg["leg_type"], "source": None, "ticker": None,
                                      "fallback_used": False, "status": "failed",
                                      "error": f"{type(e).__name__}: {e}",
                                      **self.coverage(self.levels[lid], self.stale[lid])}
            LOG.info("leg %-16s %-8s %s", lid, self.manifest[lid]["status"], self.manifest[lid].get("ticker"))


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def yahoo_meta(tickers: list[str]) -> dict[str, dict]:
    import yfinance as yf

    def one(t):
        for attempt in range(3):
            try:
                tk = yf.Ticker(t)
                tk.history(period="5d", auto_adjust=False)
                md = tk.history_metadata or {}
                return t, {"currency": md.get("currency"), "long_name": md.get("longName") or md.get("shortName"),
                           "exchange": md.get("fullExchangeName") or md.get("exchangeName")}
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:120]}"
                backoff_sleep(attempt)
        return t, {"error": err}

    with ThreadPoolExecutor(6) as ex:
        return dict(ex.map(one, tickers))


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("pandas", "numpy", "yfinance", "pyarrow", "curl_cffi"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:
            LOG.debug("version of %s not available: %s", mod, e)
            out[mod] = None
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(MARKET_DIR), help="output folder (default: data/market)")
    p.add_argument("--start", default="2014-12-01", help="download start (default 2014-12-01)")
    p.add_argument("--end", default="2026-01-05", help="download end, exclusive for Yahoo (default 2026-01-05)")
    p.add_argument("--first-level", default="2014-12-31", help="first delivered level date")
    p.add_argument("--first-return", default="2015-01-02", help="first delivered return date")
    p.add_argument("--last-date", default="2025-12-31", help="last delivered level/return date")
    p.add_argument("--ffill-limit", type=int, default=5, help="max consecutive weekdays to forward-fill a level")
    p.add_argument("--legs", default=None, help="legs registry (default: data/reference/legs.csv)")
    p.add_argument("--assets", default=None, help="assets registry (default: data/reference/assets.csv)")
    p.add_argument("--batch-size", type=int, default=20, help="tickers per yf.download call")
    p.add_argument("--max-retries", type=int, default=5, help="attempts per download (exponential backoff)")
    p.add_argument("--min-basket-frac", type=float, default=2 / 3, help="min share of basket components per day")
    p.add_argument("--offline", action="store_true", help="rebuild from <out>/raw/*.csv without network")
    p.add_argument("--no-fred-check", action="store_true",
                   help="skip the Yahoo vs FRED FX cross-check and quality gate")
    p.add_argument("--fx-min-weekly-corr", type=float, default=0.8,
                   help="FX quality gate: use FRED when Yahoo's weekly-return corr with FRED is below this")
    p.add_argument("--no-fx-redate", action="store_true",
                   help="keep Yahoo '=X' bar dates as delivered (default re-labels them one weekday earlier)")
    p.add_argument("--no-fx-clean", action="store_true", help="skip FX bad-tick removal")
    # QA 2026-09-29: default lowered from 8 to 6. At 8 sigma, confirmed bad ticks survived (COP +10%/-10% on
    # 2015-08-10/11, NOK 2018-12-31, CAD 2020-12-31, IDR 2025-06, five THB Fridays in 2016); every extra print
    # removed at 6 sigma disagrees with FRED H.10 (or, for COP/IDR, fully reverses a 3-10% move the next day).
    p.add_argument("--fx-bad-tick-sigma", type=float, default=6.0,
                   help="bad-tick threshold in robust sigmas (default 6; 8 reproduces the pre-QA build)")
    p.add_argument("--fx-bad-tick-floor", type=float, default=0.02, help="bad-tick threshold floor (log move)")
    # Owner-side default 2026-09-29 (TBC): FRED noon fixings are closer to the Xetra close than Yahoo's
    # midnight-London snapshot, so the JP 7-10y leg carries less closing-time noise (5.5% vs 7.6% vol).
    p.add_argument("--conversion-fx", choices=("legs", "fred"), default="fred",
                   help="FX used to convert non-USD proxies: the fx_ legs (Yahoo) or their FRED H.10 series")
    p.add_argument("--no-meta", action="store_true", help="skip the Yahoo currency/name metadata check")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    # only yfinance's own deprecation noise; pandas and numpy warnings of this script stay visible
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"yfinance(\.|$)")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    legs, assets = load_registries(Path(args.legs or REFERENCE_DIR / "legs.csv"), Path(args.assets or REFERENCE_DIR / "assets.csv"))
    t0 = time.time()
    if not args.offline:
        import yfinance as yf
        cache = out / ".cache" / "yfinance"
        cache.mkdir(parents=True, exist_ok=True)
        try:
            yf.set_tz_cache_location(str(cache))
        except Exception as e:
            LOG.debug("yfinance cache location not set: %s", e)

    store = RawStore(out / "raw", args.start, args.end, args.offline, args.max_retries, args.batch_size)
    b = Builder(args, legs, store)
    b.run()

    # ---- frames -------------------------------------------------------------------------------
    leg_ids = legs["leg_id"].tolist()
    L = pd.DataFrame({lid: b.levels[lid] for lid in leg_ids}, index=b.cal)
    S = pd.DataFrame({lid: b.stale[lid] for lid in leg_ids}, index=b.cal).astype(bool)
    R = L / L.shift(1) - 1.0
    lo, r0, hi = pd.Timestamp(args.first_level), pd.Timestamp(args.first_return), pd.Timestamp(args.last_date)
    L_out = L.loc[lo:hi]
    S_out = S.loc[lo:hi]
    R_out = R.loc[r0:hi]

    A = {}
    asset_man = {}
    for _, a in assets.iterrows():
        aid, lg, sh = a["asset_id"], a["long_leg"], a["short_leg"]
        err = [x for x in (lg, sh) if x not in R_out.columns]
        if err:
            A[aid] = pd.Series(np.nan, index=R_out.index)
            asset_man[aid] = {"name": a["name"], "status": "failed", "error": f"unknown leg(s) {err}"}
            continue
        A[aid] = R_out[lg] - R_out[sh]
        v = A[aid].notna()
        vol = float(A[aid].std() * np.sqrt(261)) if v.sum() > 20 else None
        asset_man[aid] = {"order": int(a["order"]), "name": a["name"], "long_leg": lg, "short_leg": sh,
                          "n_valid": int(v.sum()), "n_missing": int((~v).sum()),
                          "first_valid": str(A[aid].first_valid_index().date()) if v.any() else None,
                          "last_valid": str(A[aid].last_valid_index().date()) if v.any() else None,
                          "ann_vol": round(vol, 4) if vol is not None else None,
                          "status": "ok" if v.all() else ("partial" if v.any() else "failed")}
    AR = pd.DataFrame(A, index=R_out.index)
    if len(assets) != 55:
        LOG.warning("assets registry has %d rows (expected 55)", len(assets))

    # per-leg return diagnostics (window)
    for lid in leg_ids:
        m = b.manifest[lid]
        r = R_out[lid]
        if r.notna().sum() > 20 and m.get("leg_type") != "cash":
            d = m.setdefault("diagnostics", {})
            ia = r.abs().idxmax()
            d["ann_vol"] = round(float(r.std() * np.sqrt(261)), 4)
            d["max_abs_return"] = f"{r[ia]:+.2%} on {ia.date()}"
            d["spikes"] = detect_spikes(r)

    # ---- write --------------------------------------------------------------------------------
    for df in (L_out, S_out, R_out, AR):
        df.index.name = "date"
    L_out.to_parquet(out / "leg_levels.parquet")
    S_out.to_parquet(out / "leg_stale.parquet")
    R_out.to_parquet(out / "leg_returns.parquet")
    AR.to_parquet(out / "asset_returns.parquet")
    R_out.to_csv(out / "leg_returns.csv", date_format="%Y-%m-%d")
    AR.to_csv(out / "asset_returns.csv", date_format="%Y-%m-%d")

    if not args.offline and not args.no_meta:
        used = sorted({m["ticker"] for m in b.manifest.values() if m.get("source") == "yahoo" and m.get("ticker")})
        meta = yahoo_meta(used)
        for lid in leg_ids:
            m = b.manifest[lid]
            if m.get("source") == "yahoo" and m.get("ticker") in meta:
                md = meta[m["ticker"]]
                leg = legs.set_index("leg_id").loc[lid]
                exp = leg["quote_ccy"] if not m.get("fallback_used") else None
                md = dict(md)
                if exp and md.get("currency") and exp not in ("index",) and md["currency"] != exp:
                    md["currency_mismatch"] = f"registry {exp} vs Yahoo {md['currency']}"
                m.setdefault("diagnostics", {})["yahoo_meta"] = md

    statuses = {lid: b.manifest[lid]["status"] for lid in leg_ids}
    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 1),
        "script": Path(__file__).name,
        "args": vars(args),
        "package_versions": package_versions(),
        "calendar": {"type": "all weekdays Mon-Fri", "levels_from": str(L_out.index[0].date()),
                     "levels_to": str(L_out.index[-1].date()), "n_level_rows": int(len(L_out)),
                     "returns_from": str(R_out.index[0].date()), "returns_to": str(R_out.index[-1].date()),
                     "n_return_rows": int(len(R_out)), "ffill_limit_weekdays": args.ffill_limit},
        "field_definitions": {
            "n_valid": "non-NaN levels in the level window (fresh + forward-filled)",
            "n_stale": "forward-filled levels in the level window",
            "n_missing": "NaN levels in the level window",
            "status": "ok = no NaN level; partial = some NaN; failed = no data",
        },
        "summary": {
            "n_legs": len(leg_ids), "n_assets": len(assets),
            "ok": [k for k, v in statuses.items() if v == "ok"],
            "partial": [k for k, v in statuses.items() if v == "partial"],
            "failed": [k for k, v in statuses.items() if v == "failed"],
            "fallback_used": [k for k in leg_ids if b.manifest[k].get("fallback_used")],
            "failed_downloads": {s: [t for t, v in d.items() if v["status"] == "failed"] for s, d in store.log.items()},
        },
        "legs": {lid: b.manifest[lid] for lid in leg_ids},
        "assets": asset_man,
        "downloads": store.log,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    LOG.info("done in %.0fs: %d ok, %d partial, %d failed legs", time.time() - t0,
             len(manifest["summary"]["ok"]), len(manifest["summary"]["partial"]), len(manifest["summary"]["failed"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
