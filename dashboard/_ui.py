"""Pure helpers of the topic-exposure lab dashboard (DESIGN.md G.9, G.15; D66-D70, D83-D85, D88).

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
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from narrative_ipca.exposure_lab import bks as lab_bks
from narrative_ipca.exposure_lab import charts
from narrative_ipca.exposure_lab import compare as lab_compare
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
from narrative_ipca.config import ShockConfig
from narrative_ipca.exposure_lab.evaluate import median_finite
from narrative_ipca.exposure_lab.reference import ASSET_CLASSES
from narrative_ipca.shocks import attention_shocks

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
#: Options of the BKS model's "Covariance history" radio (D88), in display order.
BKS_HISTORIES: tuple[str, ...] = ("full", "training")
BKS_HISTORY_LABELS: dict[str, str] = dict(lab_bks.HISTORY_LABELS)
HEATMAP_COLORS: dict[str, str] = {
    "Red and blue": "diverging",
    "Red and black (as the desk example)": "example",
}

#: Elastic-net cross-validation cost per topic and asset in seconds (measured
#: 2026-09-29 on 55 assets: 25 s at 520 topics, 0.5 s at 20 topics).
CV_SECONDS_PER_TOPIC_ASSET = 25.0 / (520 * 55)

#: Training windows shorter than this many weekdays get a noise note (D81).
SHORT_TRAINING_DAYS = 250

# Time windows of the dashboard (owner request 2026-09-29). These are dashboard
# defaults only: the library's WindowConfig defaults stay as they are for the
# scripts and tests.
#: Default training end (cut-off) of the dashboard.
DEFAULT_TRAIN_END = "2025-06-30"
#: Default training length in calendar months.
DEFAULT_TRAIN_MONTHS = 6
#: Default forecast start and length of the dashboard.
DEFAULT_FORECAST_START = "2025-07-01"
DEFAULT_FORECAST_WEEKS = 4
#: Training lengths offered, in calendar months (one month to 10 years).
TRAIN_MONTHS: tuple[int, ...] = (1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120)


def train_months_label(months: int) -> str:
    """Training length in words: 1 -> "1 month", 12 -> "1 year", 18 -> "18 months", 24 -> "2 years"."""
    m = int(months)
    if m == 12:
        return "1 year"
    if m >= 24 and m % 12 == 0:
        return f"{m // 12} years"
    return "1 month" if m == 1 else f"{m} months"


def plural(n: int, word: str, words: str | None = None) -> str:
    """``"1 window"``, ``"2 windows"``: the count with the word in the right number."""
    return f"{int(n)} {word if int(n) == 1 else (words or word + 's')}"


def first_shock_day(shock_window: int) -> dt.date:
    """First day of the data with an observed topic shock: the shock needs ``w`` earlier days (D9).

    Found with the package's :func:`narrative_ipca.shocks.attention_shocks` on
    a probe series, so its rule is not restated here.
    """
    w = int(shock_window)
    cal = pd.bdate_range(DATA_START, periods=w + 5)
    probe = pd.DataFrame({"x": np.arange(len(cal), dtype=float) ** 2}, index=cal)
    first = attention_shocks(probe, ShockConfig(window=w, standardize=False)).z["x"].first_valid_index()
    return pd.Timestamp(first).date()


def shock_days(train_start: Any, train_end: Any, shock_window: int) -> int:
    """Weekdays of the training window that carry a topic shock (all of them away from the data start)."""
    ts = max(pd.Timestamp(train_start), pd.Timestamp(first_shock_day(shock_window)))
    te = pd.Timestamp(train_end)
    return len(pd.bdate_range(ts, te)) if ts <= te else 0


def earliest_train_end(shock_window: int, min_days: int | None = None) -> dt.date:
    """Earliest cut-off whose training window can hold ``min_days`` shock days (default 21, D81)."""
    min_days = WindowConfig().min_train_days if min_days is None else int(min_days)
    return pd.bdate_range(first_shock_day(shock_window), periods=min_days)[-1].date()


def training_window(train_end: Any, months: int, min_days: int | None = None) -> dict[str, Any]:
    """Training window that reaches ``months`` calendar months back from the cut-off.

    The start is ``(cut-off + 1 day) - months``, so a month-end cut-off gives
    whole calendar months (2025-06-30 and 6 months: 2025-01-01 to
    2025-06-30). Two corrections follow, each stated in ``text``:

    1. When the window has fewer than ``min_days`` weekdays (a February
       month has 20), the start moves back until it has ``min_days``
       (default ``WindowConfig().min_train_days``, 21; D81).
    2. When the start falls before the first day of the data
       (:data:`DATA_START`), it is clipped to that day.

    Parameters
    ----------
    train_end:
        Training end (cut-off), inclusive.
    months:
        Training length in calendar months.
    min_days:
        Fewest weekdays; ``None`` uses ``WindowConfig().min_train_days``.

    Returns
    -------
    dict
        ``start`` and ``end`` (``datetime.date``), ``n_days`` (weekdays in
        the window), ``nominal_start`` (before the corrections),
        ``extended`` and ``clipped`` (bool), ``note`` (the correction in
        words, ``""`` when none) and ``text`` (the caption).
    """
    min_days = WindowConfig().min_train_days if min_days is None else int(min_days)
    te = pd.Timestamp(train_end).normalize()
    nominal = te + pd.Timedelta(days=1) - pd.DateOffset(months=int(months))
    ts = nominal
    extended = clipped = False
    if len(pd.bdate_range(ts, te)) < min_days:
        earliest = pd.bdate_range(end=te, periods=min_days)[0]
        if earliest < ts:
            ts, extended = earliest, True
    d0 = pd.Timestamp(DATA_START)
    if ts < d0:
        ts, clipped = d0, True
    n = len(pd.bdate_range(ts, te)) if ts <= te else 0
    note = ""
    if clipped:
        note = (f"{train_months_label(months)} would start on {nominal.date()}, before the data; the window starts "
                f"on the first day of the data, {DATA_START}.")
    elif extended:
        note = f"The start moved back from {nominal.date()} to reach the minimum of {min_days} weekdays."
    text = f"Training window: {ts.date()} to {te.date()} ({plural(n, 'weekday')})." + (f" {note}" if note else "")
    return {
        "start": ts.date(),
        "end": te.date(),
        "n_days": n,
        "nominal_start": nominal.date(),
        "extended": extended,
        "clipped": clipped,
        "note": note,
        "text": text,
    }


def bks_training_check(
    train_start: Any, train_end: Any, bks_cfg: BKSLabConfig | None = None, lead_days: int = 0,
    shock_window: int = 5,
) -> dict[str, Any]:
    """Whether BKS can run on this training window, and why not in plain words (G.7.2; D17, D88).

    BKS needs :data:`narrative_ipca.exposure_lab.bks.MIN_TRAIN_PERIODS`
    training weeks that it can use
    (:func:`narrative_ipca.exposure_lab.bks.training_weeks`). With the full
    history these are the weeks that end inside the window and after the
    burn-in at the start of the data (on the lab's data the first usable week
    ends on 2016-04-08). With the training window only, the first one or two
    weeks of the window are lost while the instruments collect their first
    days; the reason advises a longer window, or a later cut-off when the
    first usable week is already the earliest the data allow (a window that
    starts near 2015-01-02, where the first shocks need ``w`` earlier days).

    Returns
    -------
    dict
        ``weeks`` (Fridays in the window), ``n_weeks`` (weeks BKS can use),
        ``first_week`` (the first week any BKS fit can use), ``can_run`` and
        ``reason`` (``""`` when BKS can run).
    """
    b = BKSLabConfig() if bks_cfg is None else bks_cfg
    ts, te = pd.Timestamp(train_start), pd.Timestamp(train_end)
    weeks = len(pd.date_range(ts, te, freq="W-FRI")) if ts <= te else 0
    n, first = lab_bks.training_weeks(ts, te, b, lead_days, shock_window=int(shock_window))
    min_weeks = int(lab_bks.MIN_TRAIN_PERIODS)
    reason = ""
    if n < min_weeks and weeks < min_weeks:
        reason = f"BKS needs at least {min_weeks} training weeks; the training window has {weeks}."
    elif n < min_weeks and b.history == "training":
        # a longer window helps only when its start can move the first usable week earlier; near the data
        # start the first week is already the earliest the data allow
        earliest = lab_bks.training_weeks(DATA_START, te, b, lead_days, shock_window=int(shock_window))[1]
        at_data_start = pd.isna(first) or (not pd.isna(earliest) and pd.Timestamp(first) <= pd.Timestamp(earliest))
        advice = (
            f"Near the start of the data a longer window does not help (the first usable week ends on "
            f"{pd.Timestamp(first).date()}). Move the cut-off later." if at_data_start and not pd.isna(first)
            else "Move the cut-off later." if at_data_start else "Lengthen the training window."
        )
        reason = (
            f"BKS needs at least {min_weeks} training weeks and can use {n} of this window's {weeks}. With the "
            f"training window only, its instruments start at the training start and need {b.min_days_training} "
            f"days of data, so the first weeks of the window are lost. {advice}"
        )
    elif n < min_weeks:
        start = "" if pd.isna(first) else f", so its first usable week ends on {pd.Timestamp(first).date()}"
        reason = (
            f"BKS needs at least {min_weeks} training weeks and can use {n} of this window's {weeks}. It skips the "
            f"first {int(b.burn_in_weeks)} weeks of the data to warm up its instruments{start}. Move the cut-off "
            "later."
        )
    return {"weeks": weeks, "n_weeks": n, "first_week": first, "can_run": n >= min_weeks, "reason": reason}


def short_training_note(
    train_start: Any, train_end: Any, n_topics: int, bks_cfg: BKSLabConfig | None = None, lead_days: int = 0,
    shock_window: int = 5,
) -> str | None:
    """Plain-words note on a training window shorter than about a year (D81), or ``None``.

    States the standard error of one exposure (about ``1/sqrt(n)`` in
    standardised units, ``n`` the training weekdays), the default elastic-net
    penalty ``sqrt(2 ln L / n)`` (``L`` the number of topics), and whether BKS
    can run (:func:`bks_training_check`, with ``bks_cfg`` default
    ``BKSLabConfig()``).
    """
    ts, te = pd.Timestamp(train_start), pd.Timestamp(train_end)
    n = len(pd.bdate_range(ts, te))
    if n <= 0 or n >= SHORT_TRAINING_DAYS:
        return None
    check = bks_training_check(ts, te, bks_cfg, lead_days, shock_window)
    penalty = math.sqrt(2.0 * math.log(max(int(n_topics), 2)) / n)
    bks = "BKS can run." if check["can_run"] else check["reason"]
    return (
        f"Short training window ({plural(n, 'weekday')}, {plural(check['weeks'], 'week')}):\n\n"
        f"- one exposure's standard error is about 1/sqrt(n) = {1.0 / math.sqrt(n):.2f} (a strong link is about 0.32);\n"
        f"- the default elastic-net penalty sqrt(2 ln L / n) = {penalty:.2f} sets most exposures to zero;\n"
        f"- {bks}"
    )

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

    The time windows are the dashboard's own defaults (owner request
    2026-09-29): training 6 months to the cut-off 2025-06-30 (2025-01-01 to
    2025-06-30), forecast 2025-07-01 for 4 weeks. Everything else reproduces
    ``LabConfig()`` (DESIGN.md G.9): listed real assets, 20 manual topics,
    three betas 0.35 / 0.15 / 0.05, ``w = 5``, elastic net with the universal
    penalty.
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
        "sb_train_end": dt.date.fromisoformat(DEFAULT_TRAIN_END),
        "sb_train_months": DEFAULT_TRAIN_MONTHS,
        "sb_forecast_start": dt.date.fromisoformat(DEFAULT_FORECAST_START),
        "sb_forecast_weeks": DEFAULT_FORECAST_WEEKS,
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
        "sb_bks_history": b.history,
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
        Sidebar values keyed by widget key (see :func:`default_values`). The
        training window comes from the cut-off ``sb_train_end`` and the
        length ``sb_train_months`` (:func:`training_window`).
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
    if v.get("sb_train_end") is None or v.get("sb_forecast_start") is None:
        return None, ["Choose a training end (cut-off) and a forecast start."], notes
    tw = training_window(v["sb_train_end"], int(v["sb_train_months"]))
    ts, te = pd.Timestamp(tw["start"]), pd.Timestamp(tw["end"])
    fs = pd.Timestamp(v["sb_forecast_start"])
    weeks = int(v["sb_forecast_weeks"])
    d0, d1 = pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)
    if te < d0 or te > d1:
        errors.append(f"The training end (cut-off, {te.date()}) must lie within the data, {DATA_START} to {DATA_END}.")
    if not fs > te:
        errors.append(f"Forecast start ({fs.date()}) must be after training end ({te.date()}).")
    if fs > d1:
        errors.append(f"Forecast start ({fs.date()}) is after the last day of the data ({DATA_END}).")
    elif fs + pd.Timedelta(days=7 * weeks - 1) > d1:
        n_days = len(pd.bdate_range(fs, d1))
        notes.append(
            f"The forecast window runs past the last day of the data ({DATA_END}); it has {n_days} return day(s)."
        )
    min_days = WindowConfig().min_train_days
    w = int(v["sb_shock_window"])
    if d0 <= te <= d1 and tw["n_days"] < min_days:
        errors.append(
            f"The training window {ts.date()} to {te.date()} has {plural(tw['n_days'], 'weekday')}; it needs at "
            f"least {min_days} (about one month). Move the cut-off to {earliest_train_end(w, min_days)} or later."
        )
    elif d0 <= te <= d1 and shock_days(ts, te, w) < min_days:
        errors.append(
            f"The training window {ts.date()} to {te.date()} has {plural(shock_days(ts, te, w), 'weekday')} with a "
            f"topic shock; it needs at least {min_days}. The shock needs {plural(w, 'earlier day')}, so the data's "
            f"first shock is on {first_shock_day(w)}. Move the cut-off to {earliest_train_end(w, min_days)} or "
            "later."
        )
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
            history=v.get("sb_bks_history", "full"),
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
# Shared settings on the Real data page (G.14)
# ---------------------------------------------------------------------------
#: The line under the "Settings in use" table of the Real data page.
SIMULATION_ONLY_NOTE = (
    "Simulation-only settings (price source, generic assets and topics, betas and link seeds) do not apply to "
    "real data."
)


def settings_in_use(
    v: dict[str, Any],
    asset_classes: pd.Series | None = None,
    asset_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """The sidebar settings that will apply to real data, as a three-column table.

    Groups, in order: time windows (cut-off, length, the resulting training
    window, forecast start and length, shock window), direct estimator
    (method and its penalty settings), BKS model (``K``, half-life and weekly
    ``xi``, lambda rule, grid, covariance history) and asset selection (listed assets, classes,
    left-out assets). Simulation-only settings are left out
    (:data:`SIMULATION_ONLY_NOTE`).

    Parameters
    ----------
    v:
        Sidebar values keyed by widget key (see :func:`default_values`).
    asset_classes:
        ``asset_class`` by listed asset id, for the count of listed assets;
        ``None`` omits the count.
    asset_names:
        Display name by asset id, for the left-out assets.

    Returns
    -------
    pd.DataFrame
        Columns ``Group``, ``Setting``, ``Value`` (all text).
    """
    names = asset_names or {}
    rows: list[tuple[str, str, str]] = []

    g = "Time windows"
    months = int(v["sb_train_months"])
    rows.append((g, "Training end (cut-off)", _iso(v["sb_train_end"])))
    rows.append((g, "Training length", train_months_label(months)))
    tw = training_window(v["sb_train_end"], months)
    rows.append((g, "Training window", f"{tw['start']} to {tw['end']} ({tw['n_days']} weekdays)"
                 + (f". {tw['note']}" if tw["note"] else "")))
    weeks = int(v["sb_forecast_weeks"])
    fs = pd.Timestamp(v["sb_forecast_start"])
    rows.append((g, "Forecast start", _iso(fs)))
    rows.append((g, "Forecast length", f"{weeks} week{'s' if weeks != 1 else ''} "
                 f"({fs.date()} to {(fs + pd.Timedelta(days=7 * weeks - 1)).date()})"))
    rows.append((g, "Shock window w", f"{int(v['sb_shock_window'])} days"))

    g = "Direct estimator"
    method = str(v["sb_method"])
    label = METHOD_LABELS.get(method, method)
    if method == "oracle":
        label += "; not available on real data (there is no truth)"
    rows.append((g, "Method", label))
    if method == "elastic_net":
        penalty = str(v["sb_penalty"])
        rows.append((g, "Penalty rule", PENALTY_LABELS.get(penalty, penalty)))
        if penalty == "fixed":
            rows.append((g, "Penalty alpha", f"{float(v['sb_alpha']):.3f}"))
        rows.append((g, "L1 ratio", f"{float(v['sb_l1_ratio']):.2f}"))
    elif method == "ridge":
        ridge = ("chosen by generalised cross-validation" if bool(v["sb_ridge_gcv"])
                 else f"{float(v['sb_ridge_lambda']):.4f}")
        rows.append((g, "Ridge lambda", ridge))
    rows.append((g, "Selection threshold tau", f"{float(v['sb_select_tau']):.2f}"))

    g = "BKS model"
    hl = float(v["sb_bks_half_life"])
    rule = str(v["sb_bks_rule"])
    rows.append((g, "Factors K", str(int(v["sb_bks_K"]))))
    rows.append((g, "Kernel half-life", f"{hl:g} months (weekly xi = {BKSLabConfig(half_life_months=hl).xi_weekly:.4f})"))
    rule_text = LAMBDA_RULE_LABELS.get(rule, rule)
    if rule == "tolerance":
        rule_text += f", x = {float(v['sb_bks_tolerance']):.1%}"
    elif rule == "fixed":
        rule_text += f", lambda = {float(v['sb_bks_lam']):.4g}"
    rows.append((g, "Lambda rule", rule_text))
    rows.append((g, "Lambda grid", f"{int(v['sb_bks_n_lambdas'])} points, smallest / largest "
                 f"{float(v['sb_bks_ratio']):g}"))
    rows.append((g, "Penalise the intercept", "yes" if bool(v["sb_bks_pen_int"]) else "no"))
    history = str(v.get("sb_bks_history", "full"))
    rows.append((g, "Covariance history", BKS_HISTORY_LABELS.get(history, history)))

    g = "Asset selection"
    classes = list(v.get("sb_asset_classes") or [])
    dropped = [str(a) for a in (v.get("sb_drop_assets") or [])]
    if asset_classes is not None:
        kept = listed_subset(asset_classes, classes, dropped)
        n_kept = len(asset_classes) if kept is None else len(kept)
        listed = f"{n_kept} of {len(asset_classes)}"
    else:
        listed = "all listed assets after the filters below"
    if v.get("sb_asset_source") == "generic":
        listed += " (the simulation uses generic assets; real data uses the listed assets)"
    rows.append((g, "Listed assets", listed))
    all_classes = set(classes) >= set(ASSET_CLASSES)
    rows.append((g, "Asset classes", "all" if all_classes else (", ".join(classes) or "none")))
    rows.append((g, "Left-out assets", ", ".join(names.get(a, a) for a in dropped) or "none"))
    return pd.DataFrame(rows, columns=["Group", "Setting", "Value"])


# ---------------------------------------------------------------------------
# Compare methods tab (G.9 tab 4, G.15)
# ---------------------------------------------------------------------------
#: Methods the Compare methods tab shows by default (OLS is offered too): both BKS variants (D88).
COMPARE_DEFAULT_METHODS: tuple[str, ...] = (
    "elastic_net", "ridge", lab_bks.IMPLIED_METHOD, lab_bks.IMPLIED_TRAIN_METHOD, "oracle",
)

#: Columns of the comparison table: (summary column, heading, kind). Kinds: ``pct`` (share shown in
#: percent), ``num2`` and ``num3`` (decimals), ``int``, ``sec`` (seconds) and ``text``.
#: The out-of-sample columns come first, so they stay on screen at laptop width.
COMPARISON_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("label", "Method", "text"),
    ("r2_median_window", "Median OOS R², this window", "pct"),
    ("r2_median_all_windows", "Median OOS R², all windows", "pct"),
    ("share_windows_above_oracle", "Windows above the oracle", "pct"),
    ("n_selected", "Selected pairs", "int"),
    ("coverage", "Coverage", "pct"),
    ("sign_agreement", "Sign agreement", "pct"),
    ("mcc", "MCC", "num2"),
    ("spearman", "Spearman vs truth", "num2"),
    ("rmse", "RMSE vs truth", "num3"),
    ("fit_seconds", "Fit time (s)", "sec"),
    ("note", "Note", "text"),
)


def method_option_label(method: str, direct: DirectConfig | None = None) -> str:
    """Label of a method in the Compare methods controls.

    The sidebar's direct method keeps its own settings in the comparison
    (:func:`narrative_ipca.exposure_lab.compare.method_config`), so its label
    names a non-default penalty rule ("Elastic net (CV)", "Ridge (fixed
    lambda)"); every other method has its default label.
    """
    label = lab_compare.METHOD_LABELS.get(method, method)
    if direct is None or direct.method != method:
        return label
    if method == "elastic_net" and direct.penalty == "cv":
        return "Elastic net (CV)"
    if method == "elastic_net" and direct.penalty == "fixed":
        return "Elastic net (fixed alpha)"
    if method == "ridge" and direct.ridge_lambda is not None:
        return "Ridge (fixed lambda)"
    return label


def comparison_table(summary: pd.DataFrame, notes: dict[str, str] | None = None) -> pd.DataFrame:
    """The comparison summary as the table of the Compare methods tab.

    Parameters
    ----------
    summary:
        :attr:`narrative_ipca.exposure_lab.compare.ComparisonResult.summary`
        (one row per method, oracle last).
    notes:
        Method -> note that replaces the summary's note (for example the
        dashboard's own reason why BKS-implied is not available).

    Returns
    -------
    pd.DataFrame
        The columns of :data:`COMPARISON_COLUMNS` under their headings, in
        the summary's row order, indexed by method. Shares are in percent
        (``pct`` columns times 100); unavailable methods have missing numbers
        and their reason in ``Note`` (:func:`unavailable_note`).
    """
    out = pd.DataFrame(index=pd.Index([str(m) for m in summary.index], name="method"))
    for col, head, kind in COMPARISON_COLUMNS:
        s = summary[col] if col in summary.columns else pd.Series(np.nan, index=summary.index)
        s = s.set_axis(out.index)
        if kind == "text":
            out[head] = s.fillna("").astype(str)
        else:
            vals = pd.to_numeric(s, errors="coerce").astype(float)
            out[head] = vals * 100.0 if kind == "pct" else vals
    if "available" in summary.columns:
        avail = summary["available"].set_axis(out.index).astype(bool)
        out.loc[~avail, "Note"] = [unavailable_note(n) for n in out.loc[~avail, "Note"]]
    for m, text in (notes or {}).items():
        if str(m) in out.index:
            out.loc[str(m), "Note"] = str(text)
    return out


_OLS_REFUSED = re.compile(r"ols refused: L = (\d+) topics >= n_train / 2")


def unavailable_note(note: str) -> str:
    """A method's reason for being unavailable, in plain words (the OLS refusal of ``fit_direct`` reworded)."""
    m = _OLS_REFUSED.search(str(note))
    if m:
        return (f"Not fitted: OLS needs fewer topics than half the training days, and {m.group(1)} topics is too "
                "many for this training window. Use elastic net or ridge.")
    return str(note)


def comparison_column_formats() -> dict[str, str]:
    """Number format per heading of :func:`comparison_table` (printf style, for ``st.column_config``)."""
    fmt = {"pct": "%.1f%%", "num2": "%.2f", "num3": "%.3f", "int": "%d", "sec": "%.3f"}
    return {head: fmt[kind] for _, head, kind in COMPARISON_COLUMNS if kind in fmt}


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
