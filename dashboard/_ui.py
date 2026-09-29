"""Pure helpers of the topic-exposure lab dashboard (DESIGN.md G.9; D66-D70).

No Streamlit imports: everything here maps widget values and lab results to
configurations, tables and figures, so it can be tested without a running
app. ``dashboard/app.py`` owns the widgets.

Conventions: asset x topic frames have assets as rows (the exposure-table
layout of G.9); the lab's topic x asset frames are transposed here.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from narrative_ipca.exposure_lab import charts
from narrative_ipca.exposure_lab.config import (
    DATA_END,
    DATA_START,
    TIERS,
    BKSLabConfig,
    DirectConfig,
    ExposureConfig,
    LabConfig,
    TopicSetConfig,
    UniverseConfig,
    WindowConfig,
)
from narrative_ipca.exposure_lab.evaluate import median_finite
from narrative_ipca.exposure_lab.reference import ASSET_CLASSES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Option lists and defaults
# ---------------------------------------------------------------------------
TOPIC_SETS: tuple[str, ...] = ("both", "sector", "economic", "none")
TOPIC_SET_LABELS: dict[str, str] = {
    "both": "Both (20 topics)",
    "sector": "Sector S1-S11 (11)",
    "economic": "Economic A1-A6, B1-B3 (9)",
    "none": "None",
}
MIN_GENERIC_TOPICS_ALONE = 10
SHOCK_WINDOWS: tuple[int, ...] = (1, 3, 5, 20)
NOISE_DFS: tuple[float, ...] = (5.0, 3.0, 10.0, 0.0)
METHODS: tuple[str, ...] = ("elastic_net", "ridge", "ols", "oracle")
METHOD_LABELS: dict[str, str] = {
    "elastic_net": "Elastic net (plan baseline)",
    "ridge": "Ridge",
    "ols": "OLS",
    "oracle": "Oracle (true exposures)",
}
PENALTY_LABELS: dict[str, str] = {
    "universal": "Universal: sqrt(2 ln L / n)",
    "fixed": "Fixed alpha",
    "cv": "Time-series cross-validation (slower)",
}
LAMBDA_RULE_LABELS: dict[str, str] = {
    "tolerance": "Tolerance rule (sparsest within x% of best Sharpe)",
    "argmax": "BKS argmax (best in-sample Sharpe)",
    "fixed": "Fixed lambda",
}
LAMBDA_RATIOS: tuple[float, ...] = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)
HEATMAP_COLORS: dict[str, str] = {
    "Red and blue": "diverging",
    "Red and black (as the desk example)": "example",
}

#: Elastic-net cross-validation cost per topic and asset in seconds (measured
#: 2026-09-29 on 55 assets: 25 s at 520 topics, 0.5 s at 20 topics).
CV_SECONDS_PER_TOPIC_ASSET = 25.0 / (520 * 55)

METRICS: tuple[str, ...] = (
    "OOS correlation",
    "Estimated exposure",
    "True exposure",
    "Design value (W)",
    "OOS contribution (% points)",
)
EXPOSURE_METRICS = ("Estimated exposure", "True exposure", "Design value (W)")
EXPOSURE_UNITS: tuple[str, ...] = ("Standardised", "% per 1 sd shock")
ROW_ORDERS: tuple[str, ...] = ("List order", "Asset class", "OOS R²")
GROUP_ORDER: tuple[str, ...] = ("Sector", "Macro", "Micro", "Generic")
DEFAULT_CONTRIB_ASSET = "ENERGY_v_WEQ"
HEATMAP_DEFAULT_MAX_ROWS = 60

def default_values() -> dict[str, Any]:
    """Default value of every sidebar control, keyed by widget key (``sb_`` prefix).

    The values reproduce ``LabConfig()`` (DESIGN.md G.9): listed real assets,
    20 manual topics, three betas 0.35 / 0.15 / 0.05, training 2015-01-02 to
    2022-12-30, forecast 2023-01-02 for 4 weeks, ``w = 5``, elastic net with
    the universal penalty.
    """
    c = LabConfig()
    u, t, e, w, d, b = c.universe, c.topics, c.exposure, c.window, c.direct, c.bks
    return {
        "sb_asset_source": u.asset_source,
        "sb_asset_classes": list(ASSET_CLASSES),
        "sb_drop_assets": [],
        "sb_price_source": u.price_source,
        "sb_n_generic_assets": int(u.n_generic_assets),
        "sb_universe_seed": int(u.seed),
        "sb_manual": t.manual,
        "sb_n_generic_topics": int(t.n_generic),
        "sb_signal_share": float(t.generic_signal_share),
        "sb_n_betas": int(e.n_betas),
        "sb_beta_1": float(e.beta_1),
        "sb_beta_2": float(e.beta_2),
        "sb_beta_3": float(e.beta_3),
        "sb_lead": int(e.lead_days),
        "sb_noise_df": float(e.noise_df),
        "sb_exposure_seed": int(e.seed),
        "sb_noise_seed": int(e.noise_seed),
        "sb_train_start": dt.date.fromisoformat(w.train_start),
        "sb_train_end": dt.date.fromisoformat(w.train_end),
        "sb_forecast_start": dt.date.fromisoformat(w.forecast_start),
        "sb_forecast_weeks": int(w.forecast_weeks),
        "sb_shock_window": int(w.shock_window),
        "sb_method": d.method,
        "sb_penalty": d.penalty,
        "sb_alpha": float(d.alpha),
        "sb_l1_ratio": float(d.l1_ratio),
        "sb_ridge_gcv": d.ridge_lambda is None,
        "sb_ridge_lambda": 0.1,
        "sb_select_tau": float(d.select_tau),
        "sb_bks_K": int(b.K),
        "sb_bks_half_life": float(b.half_life_months),
        "sb_bks_rule": b.lambda_rule,
        "sb_bks_tolerance": float(b.tolerance),
        "sb_bks_lam": 0.05,
        "sb_bks_n_lambdas": int(b.n_lambdas),
        "sb_bks_ratio": float(b.lambda_ratio),
        "sb_bks_pen_int": bool(b.penalize_intercept),
    }


# ---------------------------------------------------------------------------
# Configuration from widget values
# ---------------------------------------------------------------------------
def _iso(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def listed_subset(
    asset_classes: pd.Series, classes: list[str] | tuple[str, ...], drop: list[str] | tuple[str, ...]
) -> tuple[str, ...] | None:
    """Listed assets kept by the class filter and the drop list, in reference order.

    Parameters
    ----------
    asset_classes:
        ``asset_class`` by listed ``asset_id`` in reference order.
    classes:
        Asset classes to keep.
    drop:
        Asset ids to leave out.

    Returns
    -------
    tuple or None
        ``None`` when every listed asset is kept (the config default).
    """
    keep = [str(a) for a, c in asset_classes.items() if c in set(classes) and str(a) not in set(drop)]
    if len(keep) == len(asset_classes):
        return None
    return tuple(keep)


def config_from_values(
    v: dict[str, Any],
    overrides: tuple[tuple[str, str, str, int], ...] = (),
    asset_classes: pd.Series | None = None,
) -> tuple[LabConfig | None, list[str], list[str]]:
    """Build the :class:`LabConfig` of the sidebar values, or say why not.

    Parameters
    ----------
    v:
        Sidebar values keyed by widget key (see :func:`default_values`).
    overrides:
        Session link edits ``(topic_id, asset_id, tier, sign)``.
    asset_classes:
        ``asset_class`` by listed asset id (for the listed subset); ``None``
        keeps all listed assets.

    Returns
    -------
    (cfg, errors, notes)
        ``cfg`` is ``None`` when ``errors`` is not empty. ``errors`` are
        plain sentences for ``st.error``; ``notes`` are warnings that do not
        stop the run.
    """
    errors: list[str] = []
    notes: list[str] = []

    # windows first: the most common invalid combinations, in plain words
    ts, te = pd.Timestamp(v["sb_train_start"]), pd.Timestamp(v["sb_train_end"])
    fs = pd.Timestamp(v["sb_forecast_start"])
    weeks = int(v["sb_forecast_weeks"])
    d0, d1 = pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)
    if ts < d0 or te > d1:
        errors.append(f"The training window must lie within the data, {DATA_START} to {DATA_END}.")
    if not ts < te:
        errors.append(f"Training start ({ts.date()}) must be before training end ({te.date()}).")
    if not fs > te:
        errors.append(f"Forecast start ({fs.date()}) must be after training end ({te.date()}).")
    if fs > d1:
        errors.append(f"Forecast start ({fs.date()}) is after the last day of the data ({DATA_END}).")
    elif fs + pd.Timedelta(days=7 * weeks - 1) > d1:
        n_days = len(pd.bdate_range(fs, d1))
        notes.append(
            f"The forecast window runs past the last day of the data ({DATA_END}); it has {n_days} return day(s)."
        )
    if ts < te and len(pd.bdate_range(ts, te)) < WindowConfig().min_train_days:
        errors.append(f"The training window needs at least {WindowConfig().min_train_days} weekdays (about one year).")
    if v["sb_manual"] == "none" and int(v["sb_n_generic_topics"]) < MIN_GENERIC_TOPICS_ALONE:
        errors.append(f"With no manual topic set, choose at least {MIN_GENERIC_TOPICS_ALONE} generic topics.")
    if errors:
        return None, errors, notes

    listed = None
    if v["sb_asset_source"] == "listed" and asset_classes is not None:
        listed = listed_subset(asset_classes, v["sb_asset_classes"], v["sb_drop_assets"])
        if listed is not None and len(listed) < 2:
            return None, ["Keep at least 2 listed assets (asset classes and dropped assets in Universe)."], notes

    parts: dict[str, Any] = {}
    builders = {
        "Universe": lambda: UniverseConfig(
            asset_source=v["sb_asset_source"],
            listed_assets=listed,
            price_source=v["sb_price_source"],
            n_generic_assets=int(v["sb_n_generic_assets"]),
            seed=int(v["sb_universe_seed"]),
        ),
        "Topics": lambda: TopicSetConfig(
            manual=v["sb_manual"],
            n_generic=int(v["sb_n_generic_topics"]),
            generic_signal_share=float(v["sb_signal_share"]),
        ),
        "Exposures": lambda: ExposureConfig(
            n_betas=int(v["sb_n_betas"]),
            beta_1=float(v["sb_beta_1"]),
            beta_2=float(v["sb_beta_2"]),
            beta_3=float(v["sb_beta_3"]),
            lead_days=int(v["sb_lead"]),
            noise_df=float(v["sb_noise_df"]),
            link_overrides=tuple(tuple(o) for o in overrides),  # type: ignore[misc]
            seed=int(v["sb_exposure_seed"]),
            noise_seed=int(v.get("sb_noise_seed", 0)),
        ),
        "Windows": lambda: WindowConfig(
            train_start=_iso(ts),
            train_end=_iso(te),
            forecast_start=_iso(fs),
            forecast_weeks=weeks,
            shock_window=int(v["sb_shock_window"]),
        ),
        "Direct estimator": lambda: DirectConfig(
            method=v["sb_method"],
            penalty=v["sb_penalty"],
            alpha=float(v["sb_alpha"]),
            l1_ratio=float(v["sb_l1_ratio"]),
            ridge_lambda=None if bool(v["sb_ridge_gcv"]) else float(v["sb_ridge_lambda"]),
            select_tau=float(v["sb_select_tau"]),
        ),
        "BKS model": lambda: BKSLabConfig(
            K=int(v["sb_bks_K"]),
            half_life_months=float(v["sb_bks_half_life"]),
            lambda_rule=v["sb_bks_rule"],
            tolerance=float(v["sb_bks_tolerance"]),
            lam=float(v["sb_bks_lam"]) if v["sb_bks_rule"] == "fixed" else None,
            n_lambdas=int(v["sb_bks_n_lambdas"]),
            lambda_ratio=float(v["sb_bks_ratio"]),
            penalize_intercept=bool(v["sb_bks_pen_int"]),
        ),
    }
    for group, build in builders.items():
        try:
            parts[group] = build()
        except (ValueError, TypeError) as exc:
            errors.append(f"{group}: {exc}.")
    if errors:
        return None, errors, notes
    cfg = LabConfig(
        universe=parts["Universe"],
        topics=parts["Topics"],
        exposure=parts["Exposures"],
        window=parts["Windows"],
        direct=parts["Direct estimator"],
        bks=parts["BKS model"],
    )
    return cfg, errors, notes


# ---------------------------------------------------------------------------
# Labels and formatting
# ---------------------------------------------------------------------------
def topic_labels(topics: pd.DataFrame) -> dict[str, str]:
    """Display label per topic id: "S1 Energy" for manual topics, the id for generic ones."""
    out: dict[str, str] = {}
    for tid, row in topics.iterrows():
        tid = str(tid)
        out[tid] = tid if str(row.get("group", "")) == "Generic" else f"{tid} {row.get('name', '')}".strip()
    return out


def asset_labels(assets: pd.DataFrame) -> dict[str, str]:
    """Display name per asset id."""
    if "name" not in assets.columns:
        return {str(a): str(a) for a in assets.index}
    return {str(a): str(n) for a, n in assets["name"].items()}


def fmt_pct(x: Any, decimals: int = 1, signed: bool = False) -> str:
    """``0.1234`` -> ``"12.3%"``; missing -> ``"n/a"``."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(f):
        return "n/a"
    return f"{100 * f:+.{decimals}f}%" if signed else f"{100 * f:.{decimals}f}%"


def fmt_pts(x: Any, decimals: int = 2) -> str:
    """Decimal return -> signed percentage points, ``0.0123`` -> ``"+1.23 pp"``."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(f):
        return "n/a"
    return f"{100 * f:+.{decimals}f} pp"


def fmt_num(x: Any, decimals: int = 2) -> str:
    """Plain number with ``decimals`` places; missing -> ``"n/a"``."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{f:.{decimals}f}" if math.isfinite(f) else "n/a"


def share_positive(s: pd.Series) -> float:
    """Share of finite values above zero (``NaN`` if none)."""
    a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    return float((a > 0).mean()) if a.size else float("nan")


# ---------------------------------------------------------------------------
# Exposure table (G.9 tab 2, D69)
# ---------------------------------------------------------------------------
def exposure_values(metric: str, units: str, ev: Any, fit: Any, truth: Any) -> tuple[pd.DataFrame, str]:
    """Asset x topic values of the chosen cell metric, in display units.

    Parameters
    ----------
    metric:
        One of :data:`METRICS`.
    units:
        One of :data:`EXPOSURE_UNITS` (exposure metrics only): standardised
        units, or percent per one-standard-deviation shock (``b * sd * 100``
        with the asset's training volatility ``fit.ret_scale`` for the
        estimate, the truth and the design alike, so the three compare like
        with like; D74).
    ev, fit, truth:
        :class:`WindowEval`, :class:`DirectFit`, :class:`SimTruth`.

    Returns
    -------
    (values, value_label)
    """
    pct = units == EXPOSURE_UNITS[1]
    if metric == "OOS correlation":
        return ev.corr.copy(), "OOS correlation"
    if metric == "OOS contribution (% points)":
        return ev.contrib * 100.0, "OOS contribution (% points)"
    if metric == "Estimated exposure":
        vals = (fit.B_hat_pct if pct else fit.B_hat).T
        return vals.copy(), "Estimated exposure" + (" (% per 1 sd shock)" if pct else " (standardised)")
    if metric == "True exposure":
        vals = truth.B_true.mul(fit.ret_scale * 100.0, axis=1) if pct else truth.B_true
        return vals.T.copy(), "True exposure" + (" (% per 1 sd shock)" if pct else " (standardised)")
    if metric == "Design value (W)":
        vals = truth.W.mul(fit.ret_scale * 100.0, axis=1) if pct else truth.W
        return vals.T.copy(), "Design value W" + (" (% per 1 sd shock)" if pct else " (standardised)")
    raise ValueError(f"unknown metric {metric!r}")


def blank_mask(
    values: pd.DataFrame, selected: pd.DataFrame | None, blank_unselected: bool, threshold: float
) -> pd.DataFrame:
    """Cells to blank: pairs the estimator did not select and/or ``|value| < threshold`` (D69).

    ``selected`` is asset x topic (``True`` = selected); missing pairs count
    as not selected.
    """
    mask = pd.DataFrame(False, index=values.index, columns=values.columns)
    if blank_unselected and selected is not None:
        sel = selected.astype(bool).reindex(index=values.index, columns=values.columns, fill_value=False)
        mask = mask | ~sel
    if float(threshold) > 0:
        mask = mask | (values.abs() < float(threshold))
    return mask


def order_rows(assets: pd.DataFrame, r2: pd.Series | None, mode: str) -> list[str]:
    """Row order of the exposure table: list order, asset class (then list order) or OOS R2 (descending)."""
    ids = [str(a) for a in assets.index]
    if mode == "Asset class" and "asset_class" in assets.columns:
        rank = {c: i for i, c in enumerate(ASSET_CLASSES)}
        pos = {a: i for i, a in enumerate(ids)}
        return sorted(ids, key=lambda a: (rank.get(str(assets.loc[a, "asset_class"]), 99), pos[a]))
    if mode == "OOS R²" and r2 is not None:
        s = pd.to_numeric(r2.reindex(ids), errors="coerce")
        pos = {a: i for i, a in enumerate(ids)}
        return sorted(ids, key=lambda a: (not np.isfinite(s[a]), -s[a] if np.isfinite(s[a]) else 0.0, pos[a]))
    return ids


def order_columns(values: pd.DataFrame, blank: pd.DataFrame | None, topics: pd.DataFrame) -> list[str]:
    """Column order: manual topics in ontology order, then generic topics by mean ``|value|`` (blanks as 0)."""
    ids = [str(t) for t in values.columns]
    group = topics["group"].reindex(ids).astype(str) if "group" in topics.columns else pd.Series("", index=ids)
    order = topics["order"].reindex(ids) if "order" in topics.columns else pd.Series(range(len(ids)), index=ids)
    manual = sorted([t for t in ids if group[t] != "Generic"], key=lambda t: float(order[t]))
    generic = [t for t in ids if group[t] == "Generic"]
    if generic:
        shown = values[generic].where(~blank[generic], 0.0) if blank is not None else values[generic]
        score = shown.abs().fillna(0.0).mean(axis=0)
        generic = sorted(generic, key=lambda t: (-float(score[t]), float(order[t])))
    return manual + generic


def apply_long_short(values: pd.DataFrame, views: pd.Series | None) -> tuple[pd.DataFrame, pd.Series | None]:
    """Flip the rows of short-view assets and return the ``L``/``S`` row prefix (G.9 tab 2)."""
    if views is None:
        return values, None
    v = views.reindex(values.index).fillna("L").astype(str)
    sign = np.where(v.to_numpy() == "S", -1.0, 1.0)
    return values.mul(sign, axis=0), v


def exposure_table(
    metric: str,
    units: str,
    ev: Any,
    fit: Any,
    truth: Any,
    assets: pd.DataFrame,
    topics: pd.DataFrame,
    *,
    blank_unselected: bool = True,
    threshold: float = 0.0,
    row_mode: str = "List order",
    views: pd.Series | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Everything the exposure heatmap needs, in display order.

    Returns
    -------
    dict
        ``values`` (asset x topic, signs flipped for short views), ``blank``
        (bool, same shape), ``prefix`` (``L``/``S`` per row or ``None``),
        ``value_label``, ``subtitle`` (the blank rule in words),
        ``n_rows_total``.
    """
    vals, label = exposure_values(metric, units, ev, fit, truth)
    rows = order_rows(assets, ev.r2, row_mode)
    vals = vals.reindex(index=rows)
    selected = fit.selected.T if fit is not None else None
    vals, prefix = apply_long_short(vals, views)
    blank = blank_mask(vals, selected, blank_unselected, threshold)
    cols = order_columns(vals, blank, topics)
    vals, blank = vals[cols], blank[cols]
    n_total = len(vals)
    if max_rows is not None and n_total > int(max_rows):
        vals, blank = vals.iloc[: int(max_rows)], blank.iloc[: int(max_rows)]
        prefix = prefix.iloc[: int(max_rows)] if prefix is not None else None
    rules = []
    if blank_unselected:
        rules.append("pairs the estimator did not select")
    if float(threshold) > 0:
        rules.append(f"|value| below {float(threshold):g}")
    subtitle = "Blank: " + " and ".join(rules) + ". " if rules else "No blank rule. "
    subtitle += "AVERAGE counts blanks as zero."
    if n_total > len(vals):
        subtitle += f" Showing the first {len(vals)} of {n_total} assets."
    return {
        "values": vals,
        "blank": blank,
        "prefix": prefix,
        "value_label": label,
        "subtitle": subtitle,
        "n_rows_total": n_total,
    }


# ---------------------------------------------------------------------------
# Topic contributions (G.9 tab 3)
# ---------------------------------------------------------------------------
def rollup_by_group(values: pd.Series, topics: pd.DataFrame) -> pd.Series:
    """Sum per-topic values by topic group (Sector, Macro, Micro, Generic), in that order."""
    group = topics["group"].reindex(values.index).fillna("Other").astype(str)
    summed = values.groupby(group).sum(min_count=1)
    order = [g for g in GROUP_ORDER if g in summed.index] + [g for g in summed.index if g not in GROUP_ORDER]
    return summed.reindex(order)


def default_asset(asset_ids: list[str]) -> str:
    """The listed Energy spread when present, else the first asset."""
    return DEFAULT_CONTRIB_ASSET if DEFAULT_CONTRIB_ASSET in asset_ids else asset_ids[0]


# ---------------------------------------------------------------------------
# Lists tab: reference assets and link edits (G.9 tab 5)
# ---------------------------------------------------------------------------
def reference_asset_table(
    ref_assets: pd.DataFrame, legs: pd.DataFrame, run_assets: pd.DataFrame | None
) -> pd.DataFrame:
    """The listed assets with legs, index, proxy and the data source of the current run.

    Parameters
    ----------
    ref_assets:
        ``reference.load_assets()``.
    legs:
        ``reference.load_legs()``.
    run_assets:
        ``MarketData.assets`` of the run (supplies ``source``); ``None`` or a
        generic universe marks every asset "not in run".
    """
    is_cash = legs.index.to_series() == "cash"
    if "leg_type" in legs.columns:
        is_cash = is_cash | (legs["leg_type"] == "cash")
    index = legs["benchmark_index"].where(~is_cash, "")
    proxy = legs["proxy_ticker"].where(~is_cash, "")
    leg_name = legs["name"] if "name" in legs.columns else pd.Series(legs.index, index=legs.index)
    src = pd.Series("not in run", index=ref_assets.index, dtype=object)
    if run_assets is not None and "source" in run_assets.columns:
        common = [a for a in ref_assets.index if a in run_assets.index]
        src.loc[common] = run_assets.loc[common, "source"].astype(str).to_numpy()
    out = pd.DataFrame(
        {
            "#": ref_assets["order"].to_numpy(),
            "Asset": ref_assets["name"].to_numpy(),
            "Class": ref_assets["asset_class"].to_numpy(),
            "Sub-class": ref_assets["sub_class"].to_numpy(),
            "Long leg": leg_name.reindex(ref_assets["long_leg"]).fillna("").to_numpy(),
            "Long index": index.reindex(ref_assets["long_leg"]).fillna("").to_numpy(),
            "Long proxy": proxy.reindex(ref_assets["long_leg"]).fillna("").to_numpy(),
            "Short leg": leg_name.reindex(ref_assets["short_leg"]).fillna("").to_numpy(),
            "Short index": index.reindex(ref_assets["short_leg"]).fillna("").to_numpy(),
            "Short proxy": proxy.reindex(ref_assets["short_leg"]).fillna("").to_numpy(),
            "Data source": src.to_numpy(),
        },
        index=pd.Index([str(a) for a in ref_assets.index], name="asset_id"),
    )
    return out


def link_edit_frame(links: pd.DataFrame, topic_label: dict[str, str], asset_label: dict[str, str]) -> pd.DataFrame:
    """The run's links as the editable table of the Lists tab (tier and sign editable)."""
    t = links.reset_index(drop=True)
    return pd.DataFrame(
        {
            "topic_id": t["topic_id"].astype(str).to_numpy(),
            "topic": [topic_label.get(str(x), str(x)) for x in t["topic_id"]],
            "asset_id": t["asset_id"].astype(str).to_numpy(),
            "asset": [asset_label.get(str(x), str(x)) for x in t["asset_id"]],
            "tier": t["tier"].astype(str).to_numpy(),
            "sign": t["sign"].astype(np.int64).to_numpy(),
            "mechanism": t["mechanism"].astype(str).to_numpy(),
            "origin": t["origin"].astype(str).to_numpy(),
        }
    )


def overrides_from_edits(original: pd.DataFrame, edited: pd.DataFrame) -> list[tuple[str, str, str, int]]:
    """Session edits ``(topic_id, asset_id, tier, sign)`` for rows whose tier or sign changed.

    Rows are matched by ``(topic_id, asset_id)``; ``tier == "none"`` removes
    the link (G.4, :class:`ExposureConfig.link_overrides`).
    """
    key = ["topic_id", "asset_id"]
    a = original.set_index(key)[["tier", "sign"]]
    b = edited.set_index(key)[["tier", "sign"]].reindex(a.index)
    out: list[tuple[str, str, str, int]] = []
    for (t, s), row in b.iterrows():
        tier = str(row["tier"]) if pd.notna(row["tier"]) else str(a.loc[(t, s), "tier"])
        try:
            sign = int(row["sign"])
        except (TypeError, ValueError):
            sign = int(a.loc[(t, s), "sign"])
        sign = 1 if sign >= 0 else -1
        if tier not in TIERS + ("none",):
            continue
        if tier != str(a.loc[(t, s), "tier"]) or sign != int(a.loc[(t, s), "sign"]):
            out.append((str(t), str(s), tier, sign))
    return out


def merge_overrides(
    existing: dict[tuple[str, str], tuple[str, str, str, int]], new: list[tuple[str, str, str, int]]
) -> dict[tuple[str, str], tuple[str, str, str, int]]:
    """Add edits to the session's overrides; a later edit of the same pair replaces the earlier one."""
    out = dict(existing)
    for t, a, tier, sign in new:
        out[(str(t), str(a))] = (str(t), str(a), str(tier), int(sign))
    return out


# ---------------------------------------------------------------------------
# Small figures that the chart module does not cover
# ---------------------------------------------------------------------------
def r2_compare_figure(r2_bks: pd.Series, r2_direct: pd.Series, labels: dict[str, str]) -> go.Figure:
    """OOS R2 per asset: BKS (weekly) against the direct estimator (daily), one dot pair per asset.

    Not the same measure (D79): BKS fits K factors to each forecast week's own
    returns, the direct estimator fits nothing in the window. The title and
    the legend say so.
    """
    ids = [a for a in r2_direct.index if a in r2_bks.index]
    if not ids:
        return go.Figure()
    frame = pd.DataFrame({"bks": r2_bks.reindex(ids), "direct": r2_direct.reindex(ids)})
    frame = frame.sort_values("direct", ascending=False, na_position="last")
    y = [labels.get(a, a) for a in frame.index]
    clip = -1.0
    fig = go.Figure()
    for col, name, color in (("direct", "Direct estimator (daily)", charts.CATEGORICAL[0]),
                             ("bks", "BKS (weekly, factors fitted per week)", charts.CATEGORICAL[1])):
        x = frame[col].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=np.maximum(x, clip),
                y=y,
                mode="markers",
                name=name,
                marker={"color": color, "size": 8, "symbol": "circle" if col == "direct" else "diamond"},
                customdata=x,
                hovertemplate="%{y}<br>" + name + ": %{customdata:.1%}<extra></extra>",
            )
        )
    height = 130 + 16 * len(y)
    fig.update_layout(
        title={"text": "Out-of-sample R² per asset: BKS vs direct estimator (not the same measure)",
               "font": {"size": 15, "color": charts.INK}},
        height=height,
        margin={"l": 10, "r": 20, "t": 70, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=charts.SURFACE,
        font={"family": charts.FONT_FAMILY, "color": charts.INK, "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0},
        hovermode="closest",
    )
    fig.update_xaxes(tickformat=".0%", gridcolor=charts.GRIDLINE, zeroline=True, zerolinecolor=charts.BASELINE,
                     title={"text": f"Uncentered OOS R² (values below {clip:.0%} drawn at {clip:.0%})"})
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont={"size": 11}, automargin=True)
    return fig


# ---------------------------------------------------------------------------
# Notes shown next to the results
# ---------------------------------------------------------------------------
def feasibility_summary(sim: Any, exposure_cfg: ExposureConfig, labels: dict[str, str] | None = None,
                        max_listed: int = 5) -> dict[str, Any]:
    """What feasibility scaling (D61) did to the links, in plain numbers (M3).

    Symbols: ``q_k = W_k V W_k'`` is the share of topic ``k``'s shock
    variance its links would explain with the values as set (``V`` the
    covariance of the standardised returns); a topic with ``q_k`` above the
    cap (0.95) has its links multiplied by ``sqrt(0.95 / q_k)``.

    Returns
    -------
    dict
        ``n_scaled`` topics and ``n_topics``; ``share_links_scaled`` (share of
        links on scaled topics); ``max_beta_1`` (the largest beta 1 that needs
        no scaling at the current beta ratios, ``inf`` without links);
        ``rows`` (up to ``max_listed`` strongest cuts as ``(label, factor,
        largest value set, largest value used)``); ``text`` (one plain
        sentence per line, empty when nothing was scaled).
    """
    labels = labels or {}
    tr = sim.truth
    scale = tr.feasibility_scale.astype(float)
    q = pd.Series(sim.meta.get("feasibility_q", pd.Series(dtype=float)), dtype=float)
    cap = float(sim.meta.get("attention_cfg", {}).get("feasibility_cap", 0.95))
    linked = tr.W_unscaled != 0
    n_links = int(linked.to_numpy().sum())
    scaled = scale[scale < 1.0]
    n_scaled_links = int(linked.loc[scaled.index].to_numpy().sum()) if len(scaled) else 0
    q_max = float(q.max()) if len(q) and np.isfinite(q.to_numpy()).any() else 0.0
    max_beta_1 = float(exposure_cfg.beta_1) * math.sqrt(cap / q_max) if q_max > 0 else float("inf")
    rows = []
    for tid in scaled.sort_values().index[:max_listed]:
        w_set = float(tr.W_unscaled.loc[tid].abs().max())
        w_used = float(tr.W.loc[tid].abs().max())
        rows.append((labels.get(str(tid), str(tid)), float(scaled[tid]), w_set, w_used))
    text = ""
    if len(scaled):
        share = n_scaled_links / n_links if n_links else float("nan")
        cuts = "; ".join(f"{lab} x{f:.2f} (largest link {a:.2f} -> {b:.2f})" for lab, f, a, b in rows)
        more = f"; and {len(scaled) - len(rows)} more" if len(scaled) > len(rows) else ""
        text = (
            f"Feasibility scaling shrank the links of {len(scaled)} of {len(scale)} topics ({share:.0%} of all links): "
            f"with the values as set, their linked assets would explain more than {cap:.0%} of the topic's shock "
            f"variance.\n\nLargest cuts: {cuts}{more}.\n\nAt the current beta ratios, beta 1 up to "
            f"{min(max_beta_1, 0.95):.2f} needs no scaling."
        )
    return {
        "n_scaled": int(len(scaled)),
        "n_topics": int(len(scale)),
        "share_links_scaled": (n_scaled_links / n_links) if n_links else float("nan"),
        "max_beta_1": max_beta_1,
        "rows": rows,
        "text": text,
    }


def linked_asset_note(ev: Any, truth: Any) -> str:
    """A caption when some assets have no design link: the medians over the linked assets only (m9).

    Returns ``""`` when every asset has at least one link.
    """
    linked = (truth.W_unscaled != 0).any(axis=0)
    linked = linked.reindex(ev.r2.index).fillna(False).astype(bool)
    n, n_all = int(linked.sum()), int(len(linked))
    if n == n_all:
        return ""
    if n == 0:
        return f"No asset is linked to a topic ({n_all} assets): every OOS R² is noise around zero."
    est, orc = median_finite(ev.r2[linked]), median_finite(ev.r2_oracle[linked])
    return (
        f"{n} of {n_all} assets are linked to a topic; the medians above include the {n_all - n} unlinked ones. "
        f"Over the {n} linked assets the median OOS R² is {fmt_pct(est)} (estimator) and {fmt_pct(orc)} (oracle)."
    )


def cv_seconds_estimate(n_topics: int, n_assets: int) -> float:
    """Rough run time of the elastic-net cross-validation in seconds (m7): linear in topics x assets."""
    return float(CV_SECONDS_PER_TOPIC_ASSET * max(int(n_topics), 1) * max(int(n_assets), 1))


def sweep_caption(sweep: pd.DataFrame, weeks: int, n_eval_days: int, max_windows: int = 150) -> str:
    """Caption of the window sweep built from the sweep itself (OOS-4)."""
    if sweep is None or len(sweep) == 0:
        return (
            f"No complete {weeks}-week window fits between the forecast start and the end of the data "
            f"({DATA_END}); the forecast window above has {n_eval_days} return day(s)."
        )
    first, last = pd.Timestamp(sweep["start"].iloc[0]).date(), pd.Timestamp(sweep["end"].iloc[-1]).date()
    capped = " (the cap)" if len(sweep) >= int(max_windows) else ""
    return (
        f"{len(sweep)} consecutive {weeks}-week windows{capped} from {first} to {last}, training fit frozen; "
        f"at most {max_windows} windows, and an incomplete last window is dropped. Pooled R² sums squared "
        "returns across assets, so volatile assets weigh more."
    )


#: rng stream of the display subsample of :func:`subsample_pairs` (D71: one stream per component).
SCATTER_STREAM = 7101


def subsample_pairs(
    estimate: pd.DataFrame, linked: pd.DataFrame, max_points: int = 20000, seed: int = 0
) -> tuple[pd.DataFrame, int]:
    """Keep every linked pair and a seeded random subset of the others, for a readable scatter.

    Dropped pairs are set to ``NaN`` in the returned copy of ``estimate``
    (the scatter skips them). Draws come from
    ``default_rng([seed, SCATTER_STREAM])``.

    Returns
    -------
    (estimate, n_total)
        The thinned estimate and the number of pairs before thinning.
    """
    n_total = int(estimate.size)
    if n_total <= int(max_points):
        return estimate, n_total
    link = linked.astype(bool).reindex(index=estimate.index, columns=estimate.columns, fill_value=False).to_numpy()
    n_keep_other = max(0, int(max_points) - int(link.sum()))
    other = np.flatnonzero(~link.ravel())
    rng = np.random.default_rng([int(seed), SCATTER_STREAM])
    keep = np.zeros(n_total, dtype=bool)
    keep[np.flatnonzero(link.ravel())] = True
    if n_keep_other and other.size:
        keep[rng.choice(other, size=min(n_keep_other, other.size), replace=False)] = True
    arr = estimate.to_numpy(dtype=float).copy().ravel()
    arr[~keep] = np.nan
    return pd.DataFrame(arr.reshape(estimate.shape), index=estimate.index, columns=estimate.columns), n_total


def timings_frame(timings: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Stage timings as a small table: stage, seconds, served from cache."""
    rows = [
        {"stage": k, "seconds": round(float(v.get("seconds", float("nan"))), 3), "cached": bool(v.get("cached", False))}
        for k, v in timings.items()
    ]
    return pd.DataFrame(rows, columns=["stage", "seconds", "cached"])
