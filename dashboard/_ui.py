"""Pure helpers of the topic-sensitivity lab dashboard (DESIGN.md G.9, G.15, G.16; D66-D70, D83-D85, D88, D90).

No Streamlit imports: everything here maps widget values and lab results to
configurations, tables and figures, so it can be tested without a running
app. ``dashboard/app.py`` owns the widgets.

Conventions: asset x topic frames have assets as rows (the correlation-table
layout of G.9); the lab's topic x asset frames are transposed here.

Naming: the text shown to the user says "topic sensitivity"
(:data:`SENSITIVITY_DEFINITION`); in code, "exposure" (``exposure_table``,
``EXPOSURE_METRICS``, ``ExposureConfig``, ``B_hat``) means topic sensitivity.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
from collections.abc import Sequence
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
    "oracle": "Oracle (true sensitivities)",
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


#: Above this many topics the BKS tab and the BKS trace page warn about the run time before their Run BKS buttons.
BKS_RUNTIME_TOPICS = 100


def bks_runtime_warning(n_topics: int) -> str:
    """The run-time warning shown before a Run BKS button above :data:`BKS_RUNTIME_TOPICS` topics (DESIGN.md
    G.7.2), or "" up to that many."""
    if int(n_topics) <= BKS_RUNTIME_TOPICS:
        return ""
    return (f"{int(n_topics)} topics: a BKS run takes from about half a minute to several minutes (measured on 55 "
            "assets: 1.5 s at 100 topics, 36 s at 500 topics with 12 grid points). Keep the grid coarse.")


def short_training_note(
    train_start: Any, train_end: Any, n_topics: int, bks_cfg: BKSLabConfig | None = None, lead_days: int = 0,
    shock_window: int = 5,
) -> str | None:
    """Plain-words note on a training window shorter than about a year (D81), or ``None``.

    States the standard error of one sensitivity (about ``1/sqrt(n)`` in
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
        f"- one sensitivity's standard error is about 1/sqrt(n) = {1.0 / math.sqrt(n):.2f} "
        "(a strong link is about 0.32);\n"
        f"- the default elastic-net penalty sqrt(2 ln L / n) = {penalty:.2f} sets most sensitivities to zero;\n"
        f"- {bks}"
    )

#: Plain definition of the topic sensitivity, shown on the Real data page (owner decision 2026-09-30). The
#: simulation page's Data and method tab has the full version (``TERMINOLOGY`` in ``dashboard/app.py``).
SENSITIVITY_DEFINITION = (
    "Topic sensitivity: the expected return response of an asset to a one-standard-deviation attention shock in a "
    "topic, with the other topics' shocks held fixed. It says how the asset's return moves with news attention, "
    "not how much of the asset a portfolio holds."
)

#: Cell metrics of the correlation table. The three sensitivity metrics (:data:`EXPOSURE_METRICS`) are the
#: estimated sensitivity ``B_hat``, the true sensitivity ``B_true`` and the set sensitivity ``W``.
METRICS: tuple[str, ...] = (
    "OOS correlation",
    "Estimated sensitivity",
    "True sensitivity",
    "Set sensitivity (W)",
    "OOS contribution (% points)",
)
EXPOSURE_METRICS = ("Estimated sensitivity", "True sensitivity", "Set sensitivity (W)")
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
        "Sensitivities": lambda: ExposureConfig(
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
        exposure=parts["Sensitivities"],
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
    "Simulation-only settings (price source, generic assets and topics, set sensitivities (betas) and link "
    "seeds) do not apply to real data."
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
# Correlation table (G.9 tab 2, D69; "exposure" in the code names)
# ---------------------------------------------------------------------------
def exposure_values(metric: str, units: str, ev: Any, fit: Any, truth: Any) -> tuple[pd.DataFrame, str]:
    """Asset x topic values of the chosen cell metric, in display units.

    Parameters
    ----------
    metric:
        One of :data:`METRICS`.
    units:
        One of :data:`EXPOSURE_UNITS` (sensitivity metrics only):
        standardised units, or percent per one-standard-deviation shock
        (``b * sd * 100`` with the asset's training volatility
        ``fit.ret_scale`` for the estimated, true and set sensitivity alike,
        so the three compare like with like; D74).
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
    if metric == "Estimated sensitivity":
        vals = (fit.B_hat_pct if pct else fit.B_hat).T
        return vals.copy(), "Estimated sensitivity" + (" (% per 1 sd shock)" if pct else " (standardised)")
    if metric == "True sensitivity":
        vals = truth.B_true.mul(fit.ret_scale * 100.0, axis=1) if pct else truth.B_true
        return vals.T.copy(), "True sensitivity" + (" (% per 1 sd shock)" if pct else " (standardised)")
    if metric == "Set sensitivity (W)":
        vals = truth.W.mul(fit.ret_scale * 100.0, axis=1) if pct else truth.W
        return vals.T.copy(), "Set sensitivity W" + (" (% per 1 sd shock)" if pct else " (standardised)")
    raise ValueError(f"unknown metric {metric!r}")


def blank_rule(metric: str, tau: float) -> dict[str, str]:
    """The blank rule of a cell metric in words: the checkbox ``label`` and the chart ``subtitle`` phrase.

    The rule follows the view (owner request 2026-09-30, D69 amendment): the
    true sensitivity blanks the pairs with ``|B_true| < tau`` (standardised
    units, whatever the unit shown), the set sensitivity the pairs with no
    link, and the other metrics the pairs the estimator did not select.
    """
    if metric == "True sensitivity":
        return {"label": f"Blank pairs below the true-sensitivity threshold ({float(tau):g}, standardised)",
                "subtitle": f"pairs with |true sensitivity| below {float(tau):g} (standardised)"}
    if metric == "Set sensitivity (W)":
        return {"label": "Blank pairs with no link", "subtitle": "pairs with no link"}
    return {"label": "Blank pairs the estimator did not select", "subtitle": "pairs the estimator did not select"}


def blank_keep(metric: str, fit: Any, truth: Any, tau: float) -> pd.DataFrame | None:
    """Asset x topic pairs the view's blank rule keeps (``True`` = shown), see :func:`blank_rule`.

    ``None`` when the rule has nothing to test (no fit for the selection rule).
    """
    if metric == "True sensitivity":
        return (truth.B_true.abs() >= float(tau)).T
    if metric == "Set sensitivity (W)":
        return (truth.W != 0).T
    return fit.selected.T if fit is not None else None


def blank_mask(values: pd.DataFrame, keep: pd.DataFrame | None, apply_rule: bool, threshold: float) -> pd.DataFrame:
    """Cells to blank: pairs the view's rule does not keep and/or ``|value| < threshold`` (D69).

    ``keep`` is asset x topic (``True`` = shown, :func:`blank_keep`); missing
    pairs count as not kept. ``threshold`` applies to the values in the units
    shown.
    """
    mask = pd.DataFrame(False, index=values.index, columns=values.columns)
    if apply_rule and keep is not None:
        kept = keep.astype(bool).reindex(index=values.index, columns=values.columns, fill_value=False)
        mask = mask | ~kept
    if float(threshold) > 0:
        mask = mask | (values.abs() < float(threshold))
    return mask


def order_rows(assets: pd.DataFrame, r2: pd.Series | None, mode: str) -> list[str]:
    """Row order of the correlation table: list order, asset class (then list order) or OOS R2 (descending)."""
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
    blank_rule_on: bool = True,
    threshold: float = 0.0,
    row_mode: str = "List order",
    views: pd.Series | None = None,
    max_rows: int | None = None,
    tau: float | None = None,
) -> dict[str, Any]:
    """Everything the sensitivity heatmap of the Correlation table tab needs, in display order.

    ``blank_rule_on`` applies the view's blank rule (:func:`blank_rule`);
    ``tau`` is the run's selection threshold (default: the fit's
    ``select_tau``), which the true-sensitivity view tests in standardised
    units.

    Returns
    -------
    dict
        ``values`` (asset x topic, signs flipped for short views), ``blank``
        (bool, same shape), ``prefix`` (``L``/``S`` per row or ``None``),
        ``value_label``, ``subtitle`` (the blank rule in words),
        ``n_rows_total``.
    """
    if tau is None:
        tau = float((fit.meta if fit is not None else {}).get("select_tau", DirectConfig().select_tau))
    vals, label = exposure_values(metric, units, ev, fit, truth)
    rows = order_rows(assets, ev.r2, row_mode)
    vals = vals.reindex(index=rows)
    keep = blank_keep(metric, fit, truth, tau)
    vals, prefix = apply_long_short(vals, views)
    blank = blank_mask(vals, keep, blank_rule_on, threshold)
    cols = order_columns(vals, blank, topics)
    vals, blank = vals[cols], blank[cols]
    n_total = len(vals)
    if max_rows is not None and n_total > int(max_rows):
        vals, blank = vals.iloc[: int(max_rows)], blank.iloc[: int(max_rows)]
        prefix = prefix.iloc[: int(max_rows)] if prefix is not None else None
    rules = []
    if blank_rule_on and keep is not None:
        rules.append(blank_rule(metric, tau)["subtitle"])
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
    capped = f" (the cap of {max_windows})" if len(sweep) >= int(max_windows) else ""
    # the rest (the cap, the dropped last window, the pooled R2) is in the chart's "How to read" caption
    windows = plural(len(sweep), f"consecutive {weeks}-week window")
    return f"{windows}{capped} from {first} to {last}, training fit frozen."


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


# ---------------------------------------------------------------------------
# "How to read" captions (owner request 2026-09-30)
# ---------------------------------------------------------------------------
# One caption under each chart, table or row of tiles: a lead line, then one bullet per item, and each bullet
# ends with one example sentence. The examples use static, illustrative numbers, one consistent set for the
# whole dashboard: Energy Global v World EQ with a training volatility of about 1.1% a day, its strong Energy
# link set at 0.35 with a true sensitivity of 0.35 and an estimate of 0.32 standardised, shocks summing to +2.0
# (a +0.70 pp contribution), a realised move of +1.20 pp, 20 return days in the 4-week default window, 55
# assets and 20 topics (1,100 pairs). They were checked against the code on the dashboard defaults (DESIGN.md
# G.9); an example is never computed from the current run.
Bullets = Sequence[tuple[str, str]]


def how_to_read(lead: str, bullets: Bullets) -> str:
    """A "How to read" caption: ``lead``, a blank line, then ``- text Example: example`` per bullet.

    ``bullets`` holds ``(text, example)`` pairs; the example is given without
    the "Example:" prefix, which this function adds. Raises ``ValueError``
    when there is no bullet or a bullet lacks its text or its example.
    """
    if not bullets:
        raise ValueError("a 'How to read' caption needs at least one bullet")
    lines = []
    for text, example in bullets:
        text, example = str(text).strip(), str(example).strip()
        if not text or not example:
            raise ValueError("every bullet needs a text and an example")
        lines.append(f"- {text} Example: {example}")
    return f"{lead}\n\n" + "\n".join(lines)


# --- Overview -----------------------------------------------------------------
HOW_OVERVIEW_TILES: tuple[str, Bullets] = ("How to read the tiles:", (
    ("Median OOS R², estimator: per asset, 1 minus the squared forecast errors over the squared returns, summed "
     "over the forecast window's return days; then the median over assets. The forecast is the frozen training "
     "sensitivities times the window's shocks times the training volatility, with no intercept. 'Assets with "
     "positive OOS R²' is the share above 0. With the elastic net, an asset with no selected topic gets a zero "
     "forecast and scores exactly 0.",
     "for an asset at 17%, the squared forecast errors over the 20 return days add up to 83% of its squared "
     "returns; 45 of 55 assets above zero show as 82% on the 'Assets with positive OOS R²' tile."),
    ("Median OOS R², oracle: the same, with the simulation's true sensitivities in place of the estimates and "
     "the same training volatility; nothing is fitted on the window. The gap to the estimator's median is what "
     "estimation costs. In a short window the oracle is not a ceiling for every asset.",
     "an oracle median of 23% against 17% for the estimator means about 6 points of R² are lost to estimation "
     "error."),
    ("Median population R²: the share of each asset's daily return variance that the topics' observed shocks "
     "explain in the simulation, over the whole simulated history rather than the forecast window; median over "
     "assets. A single window can land above or below it.",
     "an asset with a population R² of 23% can show an oracle R² of 34% in a 20-day window that followed the "
     "topics more closely than usual."),
    ("Coverage: the share of the pairs with a link in the link map that the estimator selected (elastic net: "
     "estimate not zero; the other methods: |estimate| at least tau). Sign agreement: among the linked pairs it "
     "selected, the share whose estimate has the sign of the true sensitivity. Both compare the training fit "
     "with the link map (sign agreement also with the sign of the true sensitivity); the forecast window plays "
     "no part.",
     "50 of 95 links selected gives a coverage of 53%; if one of those 50 had the wrong sign, sign agreement "
     "would be 98%."),
    ("MCC: the correlation between two yes/no labels over all topic-asset pairs: selected by the estimator, and "
     "truly sensitive (|true sensitivity| at least tau, 0.05 by default). 1 is a perfect match and 0 is no "
     "better than chance. Truly sensitive pairs include spillovers with no link, so selecting exactly the linked "
     "pairs does not score 1.",
     "of 1,100 pairs, 350 are truly sensitive; an estimator that selects 200 pairs, 140 of them truly "
     "sensitive, scores an MCC of 0.39."),
    ("Spearman: the rank correlation of the estimated with the true sensitivities over all topic-asset pairs, "
     "the estimator's zeros included. It asks whether larger true sensitivities get larger estimates, whatever "
     "their size.",
     "an estimator that halved every true sensitivity would still score a Spearman of 1, because only the order "
     "counts."),
))

HOW_LINKED_NOTE: tuple[str, Bullets] = ("How to read this note:", (
    ("It appears when some assets have no link in the link map. The tiles' medians include those assets; the "
     "note repeats the two medians over the linked assets only.",
     "with 43 of 55 assets linked, the estimator's median OOS R² is 17% over all 55 and 21% over the 43 linked "
     "ones."),
    ("An asset without a link can still move with the topics through correlated assets that have links, so its "
     "R² need not be zero.",
     "an unlinked currency pair that moves with linked ones can reach an OOS R² of 41%."),
))

HOW_SWEEP: tuple[str, Bullets] = ("How to read this chart:", (
    ("Each point is one forecast window of the chosen length, placed back to back from the forecast start; the "
     "sensitivities stay those fitted on the training window. The first point is the forecast window of the "
     "tiles.",
     "with 4-week windows from 2025-07-01, the first point covers 2025-07-01 to 2025-07-28 and repeats the "
     "tiles' median of 17%, and the second covers 2025-07-29 to 2025-08-25 with the same January to June fit."),
    ("Solid lines: the median over assets of the per-asset OOS R², blue for the estimator and orange for the "
     "oracle. The oracle line is not a ceiling in every window.",
     "a blue point at 0.06 means half the assets had an OOS R² above 6% in that window."),
    ("Dotted lines: the pooled R², one R² over all assets and days: 1 minus the total squared forecast error "
     "over the total squared return, in return units. Volatile assets weigh more than in the median.",
     "an asset whose daily moves in a window are twice as large as another's counts four times as much in the "
     "pooled R²."),
    ("At most 150 windows are shown, and an incomplete last window is dropped.",
     "with data to 2025-12-31, six 4-week windows fit from 2025-07-01, the last ending on 2025-12-15; the next "
     "would end on 2026-01-12 and is dropped."),
))

HOW_SCATTER: tuple[str, Bullets] = ("How to read this chart:", (
    ("One point per topic-asset pair: across, the true sensitivity; up, the estimated sensitivity from the "
     "training window; both in standardised units. Points on the diagonal are estimated exactly.",
     "a point at true 0.30 and estimate 0.10 has the right sign but a third of the true size."),
    ("Blue points ('Linked in the design') have a link in the link map, grey points none. A grey point away "
     "from zero across is a spillover: the topic's attention is built from assets that move with this one.",
     "a grey point at true 0.11 can be the Materials topic on Energy Global v World EQ, which has no Materials "
     "link but moves with Materials Global v World EQ."),
    ("Points on the horizontal zero line are pairs the estimator did not select: the elastic net sets them to "
     "exactly zero. Ridge and OLS give no exact zeros.",
     "if the elastic net keeps 200 of 1,100 pairs, the other 900 points sit on the zero line."),
    ("Above 20,000 pairs the chart keeps every linked pair and a fixed random sample of the others, and a "
     "caption says so.",
     "520 topics on 55 assets make 28,600 pairs, so all linked pairs are drawn and the others are sampled down "
     "to 20,000 points in total."),
))


def how_r2_bars(who: str, overview: bool = True) -> str:
    """The OOS R² bar chart (Overview, and the inspected method on the Compare methods tab); ``who`` names the bars."""
    tile = " They are the values behind the first tile." if overview else ""
    return how_to_read("How to read this chart:", (
        (f"Blue bars ({who}): the OOS R² per asset on the forecast window's return days, highest on top.{tile} "
         "R² is uncentered over those days, topics only, no intercept; the axis is a plain share (0.20 is 20%), "
         "and a negative bar means the forecast did worse than predicting zero every day.",
         "a bar at 0.20 means the squared forecast errors add up to 80% of the squared returns; a bar at -0.10 "
         "means they are 10% larger than the squared returns."),
        ("Orange circles: the oracle, the same R² with the true sensitivities and the same training volatility. "
         "The distance from bar to circle is what estimation costs for that asset; in a short window a bar can "
         "also pass its circle by chance.",
         "a circle at 0.34 next to a bar at 0.02 means the true sensitivities would have explained about a third "
         "of the squared returns and the estimates almost none."),
        ("Black ticks: the population R² of the simulation, the share of the asset's return variance the topics "
         "explain over the whole simulated history. It does not depend on the window; the circle moves around it "
         "from window to window.",
         "a tick at 0.23 with a circle at 0.34 means this window followed the topics more closely than the asset "
         "does on average."),
        ("Values below -1 are drawn at -1 and counted in the note under the title (bars, circles and ticks "
         "together); the hover shows the actual value. With more than 100 assets, the 100 with the highest OOS "
         "R² are shown.",
         "an oracle R² of -1.46 sits at -1 on the axis, adds one to the count in the note, and the hover shows "
         "-1.460."),
    ))


# --- Correlation table ----------------------------------------------------------
#: One bullet per cell metric; the caption shows the chosen metric's bullet (and the unit bullet for the three
#: sensitivity metrics), so the text follows the view as the blank rule does.
CELL_METRIC_BULLETS: dict[str, tuple[str, str]] = {
    "OOS correlation": (
        "OOS correlation: the Pearson correlation, over the forecast window's return days, of the asset's daily "
        "return with the topic's shock on the matching day (the same day, or the previous weekday with the "
        "next-day lead). It uses no fitted sensitivity, looks at one topic at a time and needs at least 3 days.",
        "0.38 over 20 days means days with larger shocks in the topic tended to be days with higher returns for "
        "the asset; a correlated topic without a link can show a similar value."),
    "Estimated sensitivity": (
        "Estimated sensitivity: the direct estimator's coefficient from the training window, frozen: the "
        "expected return response to a one-standard-deviation shock in the topic, with the other topics' shocks "
        "held fixed. In standardised units it is counted in the asset's training standard deviations of daily "
        "return.",
        "0.32 standardised means a one-sd shock in the topic comes with a return 0.32 training standard "
        "deviations higher, the other topics unchanged."),
    "True sensitivity": (
        "True sensitivity: the population value the simulation produces on the observed shocks. It includes "
        "spillovers through correlated assets, so most pairs without a link are not zero.",
        "Energy Global v World EQ can show a true sensitivity of 0.35 to the Energy topic, its strong link, and "
        "0.11 to the Materials topic, which has no link to it."),
    "Set sensitivity (W)": (
        "Set sensitivity (W): the value set with the link map and the betas, the link's sign times its tier's "
        "beta (0.35 strong, 0.15 moderate, 0.05 weak by default), after any feasibility scaling of the topic; "
        "zero where there is no link.",
        "a strong negative link shows -0.35, or -0.28 if the topic was scaled down by a factor of 0.8 for "
        "feasibility."),
    "OOS contribution (% points)": (
        "OOS contribution (% points): the frozen training sensitivity times the sum of the topic's shocks over "
        "the window's return days, times the asset's training volatility, in percentage points of return. Over "
        "all topics, these plus the part not explained add up to the realised move (Topic contributions tab).",
        "a sensitivity of 0.32, a training volatility of 1.1% and shocks summing to +2.0 over 20 days give "
        "0.32 × 1.1% × 2.0 = +0.70 pp."),
}

#: The unit bullet of the three sensitivity metrics.
UNIT_BULLET: tuple[str, str] = (
    "% per 1 sd shock: the standardised value times the asset's daily return volatility over the training "
    "window, times 100. Estimated, true and set sensitivity all use this same volatility, so the three compare "
    "like with like; the unit choice applies to these three metrics only.",
    "0.32 standardised on an asset with 1.1% training volatility shows as 0.32 × 1.1 = 0.35% per 1 sd shock.")


def how_cell_metric(metric: str) -> str:
    """The Correlation table's metric caption: the chosen metric's bullet, plus the unit bullet if it has units."""
    bullets = [CELL_METRIC_BULLETS[metric]]
    if metric in EXPOSURE_METRICS:
        bullets.append(UNIT_BULLET)
    return how_to_read("How to read the cell metric:", bullets)


HOW_TABLE: tuple[str, Bullets] = ("How to read the table:", (
    ("Blank cells follow the view. The Estimated sensitivity, OOS correlation and OOS contribution views blank "
     "the pairs the estimator did not select (elastic net: a zero estimate; other methods: |estimate| below tau). "
     "The True view blanks pairs with |true sensitivity| below tau (0.05 by default, standardised in both units), "
     "the Set view pairs with no link. The checkbox turns the rule off.",
     "the Materials topic on Energy Global v World EQ, with a true sensitivity of 0.11, no link and no "
     "selection, shows 0.11 in the True view and is blank in the Set and Estimated views."),
    ("The '|value| below' slider blanks, on top, every cell whose absolute value is below it, in the units "
     "shown (correlation, standardised, % or pp).",
     "at 0.10 in the % view, a selected estimate of 0.08% is blanked as well."),
    ("AVERAGE row: the mean of each column over the displayed rows, with blank cells counted as zero. It mixes "
     "how often a topic shows up with how large the shown values are.",
     "if 7 of 55 assets show an Energy correlation averaging 0.21, the AVERAGE cell is 7 × 0.21 / 55 = 0.03."),
    ("Long/short view: rows you mark S have every value's sign flipped and the label prefixed S (L otherwise), "
     "so the cell reads for the position rather than the asset. The AVERAGE uses the flipped values.",
     "a short position in an asset with a correlation of +0.38 to a topic is labelled 'S' and shows -0.38: "
     "positive shocks in that topic went with losses on the position."),
    ("Colours are symmetric around zero: red for negative, blue (or grey to black) for positive. OOS "
     "correlation always runs from -1 to 1; the other views end at the largest absolute value shown, so shades "
     "are not comparable across views.",
     "if the largest estimated sensitivity shown is 0.38, a 0.19 cell gets the colour a 0.50 correlation gets "
     "in the correlation view."),
    ("Rows: list order, asset class (Equity, FX, Fixed income, then list order) or the estimator's OOS R² in "
     "the window, highest first. Columns: manual topics in ontology order, then generic topics by mean absolute "
     "value shown (blanks as zero), cut at Maximum columns.",
     "with 20 manual and 100 generic topics and Maximum columns at 40, the table shows the 20 manual topics and "
     "the 20 generic topics with the largest average absolute value."),
))


# --- Topic contributions ----------------------------------------------------------
#: The model block the owner asked to extend (2026-09-30); the other blocks follow its style.
HOW_TWO_VIEWS: tuple[str, Bullets] = ("How to read the two views:", (
    ("Variance share (default): each topic's share of the window's day-to-day return variation. It uses every "
     "day of the window.",
     "a variance share of 10% for a topic is what you get when its explained return is one tenth of the asset's "
     "return on every day of the window (+0.10% on a +1.0% day, -0.05% on a -0.5% day)."),
    ("Return attribution: each topic's sensitivity, estimated on the training window and frozen, times the sum "
     "of its standardised shocks over the window, times the asset's training volatility. The contributions and "
     "the residual add up exactly to the realised move.",
     "with a training volatility of 1.1% a day, a sensitivity of 0.32 and shocks summing to +2.0, a topic "
     "contributes 1.1% × 0.32 × 2.0 = +0.70 pp; if all topics together contribute +0.54 pp to a +1.20 pp move, "
     "the residual is +0.66 pp."),
    ("Summed shocks largely cancel over a window: each shock is attention minus its trailing mean, so their sum "
     "depends mostly on the attention level at the window's edges. The return attribution therefore understates "
     "the topics' role, and the explained line in the cumulative chart drifts back towards zero.",
     "with w = 5 and the same-day lead, a topic whose attention is 0.20 on the five days before the window and "
     "0.20 on its last five days has shocks summing to zero, so its contribution is 0.00 pp however much its "
     "attention moved in between."),
    ("Diamonds use the true sensitivities of the simulation.",
     "with the same shocks and training volatility, a bar at +0.70 pp next to a diamond at +0.77 pp (1.1% × "
     "0.35 × 2.0) means the estimated sensitivity is about 0.9 times the true one (0.32 against 0.35)."),
))

_HOW_TILE_R2: tuple[str, str] = (
    "OOS R², estimator / oracle: 1 minus the sum of squared daily forecast errors divided by the sum of squared "
    "daily returns over the window, with the topic-explained return as the forecast. The oracle uses the true "
    "sensitivities with the same training volatility. The R² is negative when the forecast does worse than "
    "forecasting zero.",
    "squared daily returns adding up to 25 and squared errors adding up to 22 give 1 - 22/25 = 12%; errors "
    "adding up to 28 give -12%.")

HOW_TILES_ATTRIBUTION: tuple[str, Bullets] = ("How to read the tiles:", (
    ("Realised move: the sum of the asset's daily returns over the window's return days, in percentage points "
     "(pp). It is a sum, not a compounded return.",
     "two days of +1.00% show as +2.00 pp, not the compounded +2.01%."),
    ("Explained by topics: the sum of all topic contributions. It is where the blue line of the cumulative chart "
     "ends.",
     "contributions of +0.70, -0.15 and -0.01 pp give +0.54 pp."),
    ("Not explained by topics: the realised move minus the explained move.",
     "+1.20 pp realised and +0.54 pp explained leave +0.66 pp."),
    _HOW_TILE_R2,
))

HOW_TILES_SHARE: tuple[str, Bullets] = ("How to read the tiles:", (
    ("Share explained by topics: the sum of the topic shares, which is the sum of all bars including Other "
     "topics. The 'Not explained by topics' bar is 100% minus it.",
     "shares of +6%, +3% and -1% sum to 8%, and the 'Not explained by topics' bar shows 92%."),
    ("True share (simulation): the same sum with the true sensitivities. The diamonds sum to it.",
     "a true share of 30% against 8% estimated means the true sensitivities, applied to the same shocks, would "
     "co-move with 30% of the window's variation."),
    ("Largest topic: the topic with the largest share in absolute value, shown with its sign; 'none' when every "
     "share is zero. The tooltip gives its name.",
     "with S1 at +6%, S10 at +3% and S5 at -1% the tile shows 'S1 · 6.0%'; had S5 been at -7%, it would show "
     "'S5 · -7.0%'."),
    ("The share explained is not the OOS R². The R² also subtracts the size of the explained returns: R² = 2 × "
     "share explained - (sum of squared explained returns / sum of squared returns).",
     "a share explained of 8% with squared explained returns worth 4% of the squared returns gives an R² of "
     "2 × 8% - 4% = 12%."),
    _HOW_TILE_R2,
))

HOW_CONTRIB_BARS: tuple[str, Bullets] = ("How to read the bar chart:", (
    ("Topics are sorted by the size of their bar, largest on top. 'Topics shown' sets how many get their own "
     "bar, and the rest are summed into one 'Other topics (n)' bar. Topics with a zero bar (with the elastic net, "
     "the topics it did not select) are ordered by their diamond.",
     "with 20 topics and 15 shown, the last topic bar is 'Other topics (5)', the sum of the five smallest."),
    ("The lower panel has its own scale. The grey bar is the part not explained by topics. The black bar is the "
     "realised move (Return attribution) or the total variation of 100% (Variance share). The topic bars, "
     "Other topics and the grey bar add up to the black bar.",
     "topic bars summing to +0.54 pp and a grey bar of +0.66 pp add up to the black bar of +1.20 pp; in the "
     "Variance share view, 8% and 92% add up to 100%."),
    ("Blue bars are positive and red bars negative. A red bar in Return attribution pulled the move down. In "
     "Variance share, a red bar means the topic's explained return moved against the asset's daily returns.",
     "a sensitivity of 0.07 with shocks summing to -2.0 gives 1.1% × 0.07 × (-2.0) = -0.15 pp, a red bar."),
    ("The diamond on the grey row is the realised move minus the sum of the true contributions (Variance share: "
     "100% minus the true share).",
     "true contributions summing to +0.90 pp on a +1.20 pp move put that diamond at +0.30 pp; in Variance share, "
     "a true share of 30% puts it at 70%."),
))

HOW_CUMULATIVE: tuple[str, Bullets] = ("How to read the cumulative chart:", (
    ("Each line is a running sum of daily returns from the start of the window, in percent (not compounded).",
     "daily returns of +1.0%, -0.5% and +0.3% put a line at +1.0%, +0.5% and +0.8%."),
    ("Black is the realised return and blue the return explained by topics with the estimated sensitivities. "
     "Orange is the same with the true sensitivities. The black line ends at the Realised move tile and the blue "
     "line at Explained by topics. The orange line ends at the sum of the true contributions, the diamonds in "
     "Return attribution.",
     "with a realised move of +1.20 pp, +0.54 pp explained and true contributions of +0.90 pp, the lines end at "
     "+1.20%, +0.54% and +0.90%."),
    ("The explained lines follow the summed shocks: a topic's part of a line returns to zero once its attention "
     "has been back at its level from before the window for w days.",
     "with w = 5 and the same-day lead, attention at 0.20, raised to 0.24 on days 5 to 9 of the window and back "
     "at 0.20 from day 10, moves that topic's part of the blue line away from zero from day 5 and puts it back "
     "at exactly zero from day 14."),
))

HOW_ATTENTION: tuple[str, Bullets] = ("How to read the attention chart:", (
    ("The chart shows up to three topics with the largest contributions to the move in absolute value (Return "
     "attribution), whichever view is chosen. Topics with a zero contribution are left out.",
     "with contributions of +0.70, -0.15 and -0.01 pp and zero for all other topics, it shows those three, even "
     "if the Variance share view ranks them differently."),
    ("Top panel: the simulated attention level, averaged per week (weeks ending Friday). Base levels are drawn "
     "between 0.15 and 0.35 and only changes feed the shocks, so the level itself says nothing about how much a "
     "topic matters.",
     "one topic sitting flat near 0.33 and another flat near 0.22 both have shocks of zero."),
    ("Bottom panel: the daily observed shock. It is attention minus its average over the previous w days, "
     "divided by the standard deviation of that difference on the training window.",
     "for a topic whose difference has a training standard deviation of 0.02, attention 0.04 above its average "
     "of the previous five days gives a shock of +2.0."),
    ("The sum of a topic's shocks over the shaded forecast window (with the next-day lead, the shocks of the "
     "weekday before each shaded day), times its sensitivity and the training volatility, is its bar in Return "
     "attribution.",
     "with the same-day lead, shocks summing to +2.0 over the shaded days, a sensitivity of 0.32 and a training "
     "volatility of 1.1% give the +0.70 pp bar."),
    ("The shaded band is the forecast window and the vertical line the training end. The chart runs from 26 "
     "weeks before the window to 8 weeks after it.",
     "for the window 2025-07-01 to 2025-07-28 the chart covers 2024-12-31 to 2025-09-22."),
))

HOW_ROLLUP: tuple[str, Bullets] = ("How to read the roll-up:", (
    ("With 'Roll up by topic group' ticked, each bar is the sum of one group's topics: Sector (S1-S11), Macro "
     "(A1-A6), Micro (B1-B3) and Generic (G001, ...). Only groups in the run appear.",
     "Sector topics at +0.70 and -0.15 pp and a Macro topic at -0.01 pp give a Sector bar of +0.55 pp and a "
     "Macro bar of -0.01 pp."),
    ("The diamonds are summed the same way. The grey and black bars and the tiles do not change.",
     "a +1.20 pp realised move with +0.66 pp not explained shows the same two bars with or without the "
     "roll-up."),
    ("Topics of opposite sign offset each other inside a group, so a small group bar can hide large topic bars.",
     "Macro topics at +0.30 and -0.28 pp give a Macro bar of +0.02 pp."),
))


# --- Compare methods --------------------------------------------------------------
#: Examples of the Compare methods tab's lead caption, by bullet; ``app.compare_tab`` writes the texts, which
#: carry the run's own figures (the kernel-weight share, the BKS tab's R²).
COMPARE_LEAD_EXAMPLES: dict[str, str] = {
    "direct": "with training from 2025-01-01 to 2025-06-30 and a 4-week forecast from 2025-07-01, the elastic "
              "net sees no return after 2025-06-30 and is scored on the 20 return days from 2025-07-01 to "
              "2025-07-28.",
    "bks": "on the defaults both variants use K = 3 factors and the 2% tolerance rule for lambda; only the "
           "history and the return scaling differ.",
    "full": "with the default half-life of 69 months, a day 69 months before the cut-off counts half as much as "
            "a day just before it.",
    "training": "with training from 2025-01-01, it reads no return before 2025-01-01, and attention only from "
                "the 5 weekdays before, to form the first shocks.",
    "bks_tab": "the BKS tab can show 28% while BKS-implied (full history) scores 7% here; only the 7% comes from "
               "sensitivities frozen at the training end.",
    "oracle": "if the oracle scores 23% and a method 17% on the same days, the method reaches about three "
              "quarters of what the true sensitivities achieve.",
}


def how_compare_table(windows: str) -> str:
    """The Compare methods table; ``windows`` names the run's consecutive windows ("the 6 consecutive ...")."""
    return how_to_read("How to read the table:", (
        ("Median OOS R², this window: each asset's out-of-sample R² on the forecast window's return days, with "
         "the sensitivities frozen at the training end, then the median over assets. 0% is what predicting a "
         "zero return every day would score.",
         "with 55 assets, a median of 17% means 27 assets score above it and 27 below; an asset at 17% has "
         "squared gaps adding up to 83% of its squared returns."),
        (f"Median OOS R², all windows: the same median for each of {windows} from the forecast start to the end "
         "of the data, then the median over those windows. The first window is the forecast window above, and "
         "every training fit stays frozen.",
         "six windows with medians of 17%, 18%, 6%, 6%, 2% and 7% give 6.5%, halfway between the two middle "
         "values, 6% and 7%."),
        ("Windows above the oracle: the share of those windows where the method's median R² is higher than the "
         "oracle's. The oracle is not the best fit in every window, so a method can beat it by chance; the "
         "oracle's own row shows a dash.",
         "a method above the oracle in 1 of 6 windows shows 16.7%."),
        ("Selected pairs counts the pairs a method keeps: a non-zero estimate for the elastic net, an estimate of "
         "at least tau in absolute value for the other methods. Coverage and Sign agreement use the pairs linked "
         "in the link map, as on the Overview; MCC and Spearman vs truth use all topic-asset pairs. The tiles "
         "under Inspect one method explain them.",
         "20 topics and 55 assets make 1,100 pairs, and the oracle row's Selected pairs, about 350, is the "
         "number of pairs whose true sensitivity is at least 0.05 in absolute value."),
        ("RMSE vs truth: the root mean squared gap between estimated and true sensitivities over all pairs, in "
         "standardised units (return standard deviations per 1 standard deviation of the topic shock).",
         "an RMSE of 0.05 on an asset with 1.1% daily volatility is a typical miss of about 0.055% of daily "
         "return per 1 sd topic shock (0.05 × 1.1%)."),
        ("Fit time: seconds to fit on the training window; for the BKS-implied rows, the BKS panel and fit plus "
         "the conversion. The oracle row is last, as the reference. A dash marks a figure that does not apply.",
         "the elastic net fits in about 0.01 s and BKS-implied (full history) in about 0.3 s, almost all of it "
         "the BKS panel and fit."),
    ))


def how_compare_dots(has_oracle: bool) -> str:
    """The Compare tab's OOS R² dot plot; the rows follow the oracle when it is chosen, else the first method."""
    bullets: list[tuple[str, str]] = [
        ("One row per asset and one marker per method: the method's out-of-sample R² on the forecast window's "
         "return days, with its sensitivities frozen at the training end.",
         "a marker at 20% means that over the window's return days the squared gaps between the realised and "
         "the topic-explained returns add up to 80% of the squared returns."),
    ]
    if has_oracle:
        bullets += [
            ("The black tick is the oracle: the true sensitivities with the same training volatility. Rows are "
             "sorted by it, highest on top. The gap between a marker and the tick is what estimating the "
             "sensitivities cost on that asset.",
             "on a row with the tick at 34% and the elastic-net marker at 2%, the true sensitivities explain 34% "
             "of the asset's squared returns in this window and the estimates 2%."),
            ("A marker right of the tick beats the oracle on that asset. In a short window this happens by "
             "chance.",
             "an elastic-net marker at 41% next to a tick at 20% means the estimates happened to fit this asset's "
             "window better than the true sensitivities."),
        ]
    else:
        bullets.append(
            ("Without the oracle, rows are sorted by the R² of the first chosen method in the list's fixed order "
             "(elastic net, ridge, OLS, then BKS-implied), highest on top.",
             "with ridge and the elastic net chosen, rows follow the elastic net, so an asset at 41% for the "
             "elastic net sits above one at 17%."))
    bullets += [
        ("Below 0% the topic-explained returns did worse than predicting a zero return every day. Values below "
         "-50% are drawn at -50% and counted under the title; the hover shows the value.",
         "an R² of -140% (squared gaps 2.4 times the squared returns) is drawn at -50%."),
        ("The median of each method's markers is the table's Median OOS R², this window.",
         "an elastic-net median of 17% here is the 17% in the table's this-window column and the first point of "
         "the elastic-net line in the chart of consecutive windows."),
    ]
    return how_to_read("How to read this chart:", bullets)


def how_compare_sweep(span: str, has_oracle: bool) -> str:
    """The Compare tab's window chart; ``span`` describes the run's windows (" (... from X to Y here)") or ""."""
    bullets: list[tuple[str, str]] = [
        ("Each point is one window of the forecast length: the median over assets of the method's out-of-sample "
         "R² in that window, plotted at the window's start. The sensitivities stay frozen at the training end in "
         "every window; nothing is refitted.",
         "a point at 6% for the window starting 2025-08-26 means about half the assets had an R² above 6% over "
         "its 20 return days, with sensitivities fitted on data to 2025-06-30."),
        (f"The windows follow each other without overlap from the forecast start{span}, and only windows that "
         "end within the data count. The first point is the forecast window, so it equals the table's "
         "this-window column.",
         "with a forecast start of 2025-07-01, 4-week windows and data to 2025-12-31, six windows fit, the last "
         "one ending 2025-12-15."),
    ]
    if has_oracle:
        bullets.append(
            ("The dashed black line is the oracle, the true sensitivities: it shows what they reach in each window, "
             "so compare a line with it rather than with zero. The table's all-windows column is the median of a "
             "line's points, and Windows above the oracle is the share of points above the dashed line.",
             "in a window where the oracle's median is 6%, a method at 7% is above it, and one such window out of "
             "six gives Windows above the oracle 16.7%."))
    bullets.append(
        ("Values below -100% are drawn at -100% and counted in the subtitle; the hover shows the value.",
         "a window median of -130% is drawn at -100%, and the hover shows -130.0%."))
    return how_to_read("How to read this chart:", bullets)


HOW_COMPARE_TILES: tuple[str, Bullets] = ("How to read the tiles:", (
    ("Coverage: the share of the pairs linked in the link map that the method selected. A pair is selected when "
     "the elastic net keeps it (a non-zero estimate) or, for the other methods, when its estimate is at least "
     "tau (0.05 by default) in absolute value.",
     "50 of 95 linked pairs selected gives 53%; even the oracle stays below 100% when some linked pairs have a "
     "true sensitivity under 0.05."),
    ("Sign agreement: among the linked pairs the method selected, the share whose estimate has the same sign as "
     "the true sensitivity.",
     "if 50 linked pairs are selected and 49 of them carry the true sign, sign agreement is 98%."),
    ("MCC (Matthews correlation): how well 'selected' matches 'truly sensitive' (a true sensitivity of at least "
     "tau in absolute value) over all pairs, linked or not. 1 is a perfect match and 0 is no better than "
     "chance; the oracle row of the table scores 1 by construction.",
     "of 1,100 pairs, 350 are truly sensitive; a method that selects 200 pairs, 140 of them truly sensitive, "
     "scores an MCC of 0.39."),
    ("Spearman: the rank correlation of the estimated with the true sensitivities over all pairs, using every "
     "pair's estimate, selected or not (zero where the elastic net dropped the pair).",
     "a Spearman of 0.55 means the estimates order the 1,100 pairs only partly as the truth does; the oracle row "
     "scores 1.00."),
    ("The tiles repeat this method's row of the table, with percentages rounded to whole numbers.",
     "a coverage of 52.6% in the table shows as 53% on the tile."),
))

HOW_COMPARE_SCATTER: tuple[str, Bullets] = ("How to read this chart:", (
    ("Each point is one topic-asset pair: across, its true sensitivity; up, the method's estimate from the "
     "training window. Both are in standardised units: return standard deviations per 1 standard deviation of "
     "the topic shock.",
     "a point at (0.35, 0.32) is a pair with a true sensitivity of 0.35 that the method puts at 0.32; on an "
     "asset with a training volatility of 1.1%, that is 0.35% of return per 1 sd shock instead of about 0.39%."),
    ("The diagonal marks estimate = truth. Points between the diagonal and the horizontal zero line are shrunk "
     "towards zero; points on the zero line are pairs the method set to zero.",
     "an elastic net that keeps 200 of the 1,100 pairs puts the other 900 on the zero line."),
    ("Blue points ('Linked in the design') are the pairs the link map links directly; grey points are all other "
     "pairs. A grey pair can still have a true sensitivity, because asset returns and topic shocks are "
     "correlated with each other.",
     "with 95 linked pairs out of 1,100, about 350 pairs have a true sensitivity of at least 0.05, so many grey "
     "points sit away from zero on the horizontal axis."),
    ("With more than 20,000 pairs the chart shows every linked pair and a fixed random sample of the others; the "
     "caption then gives the total.",
     "520 topics on 55 assets make 28,600 pairs, so all linked pairs are drawn and the others are sampled down "
     "to 20,000 points in total."),
))


def how_bks_implied(half_life_months: float) -> str:
    """The BKS-implied note of the inspect section (only when a BKS-implied method is inspected)."""
    return how_to_read("How to read the BKS-implied note:", (
        ("K factors: BKS reduces each asset's topic covariances to K factor betas, and the implied sensitivities "
         "are rebuilt from those K betas only. When fewer than K directions carry weight, the note says how many "
         "are used.",
         "with K = 3 and 20 topics, an asset's 20 implied sensitivities all come from its 3 factor betas, so any "
         "part of its topic covariances outside the 3 fitted directions is lost."),
        ("Topics kept: the topics the sparse fit kept. A dropped topic gets no implied covariance with the asset, "
         "but it can still get a sensitivity, because the conversion to sensitivities allows for correlated "
         "topic shocks.",
         "with 10 of 20 topics kept, the other 10 have zero implied covariance, yet a dropped topic can still end "
         "up with a sensitivity of 0.1 or more."),
        ("lambda: the penalty of the sparse fit, chosen by the sidebar's rule; the larger it is, the more topics "
         "it drops.",
         "lambda = 0.28 with 10 of 20 topics kept means the penalty dropped half the topics."),
        (f"Covariance history: the full history weighs every day before the cut-off, with weights that halve "
         f"every {float(half_life_months):g} months (the kernel half-life); the training window uses only the "
         "training window. For the full history the note gives the share of that weight on days before the "
         "training start.",
         "with training from 2025-01-01 to 2025-06-30, about 92% of the full-history weight lies before "
         "2025-01-01, on data the direct methods never see; the training-window note shows no share, because it "
         "is 0."),
    ))


# --- BKS --------------------------------------------------------------------------
HOW_BKS_TILES: tuple[str, Bullets] = ("How to read the tiles:", (
    ("Chosen lambda: the penalty of the training fit, picked on the lambda path below. A larger lambda sets more "
     "topics' Gamma rows to zero.",
     "a chosen lambda of 0.28 on a grid from 0.034 to 3.4 keeps 10 topics; the smallest grid value keeps 16 and "
     "the largest keeps 1."),
    ("Factors K: the number of weekly factors. Each asset's K factor loadings are its instruments (a constant and "
     "one covariance per topic) times Gamma, and each forecast week's K factor values are fitted to that week's "
     "returns.",
     "with K = 3 and 55 assets, each forecast week fits 3 numbers to 55 weekly returns; K must stay below the "
     "number of assets."),
    ("Selected topics: the topics whose Gamma row is not zero at the chosen lambda. Only they enter the fitted "
     "return.",
     "10 of 20 means the other 10 topics have a Gamma row of exactly zero and show 0.00 pp in the per-topic "
     "split below."),
    ("In-sample total R²: the training fit scored on its own weeks, with Gamma and the factors both fitted to "
     "them: 1 minus the squared misses over the sum of squared weekly returns, over all training asset-weeks, in "
     "volatility-scaled units.",
     "45% over 26 training weeks × 55 assets (1,430 asset-weeks) means the squared misses add up to 55% of the "
     "summed squared returns."),
    ("Pooled OOS R² (weekly): the same ratio over all asset-weeks of the forecast weeks, with Gamma frozen at the "
     "cut-off and each week's K factors fitted to that week's returns. Pooled means one ratio over all "
     "asset-weeks; the volatility scaling makes each asset count about equally.",
     "28% over 4 weeks × 55 assets (220 asset-weeks) means the squared misses add up to 72% of the summed "
     "squared weekly returns."),
    ("Same, instruments shuffled: the pooled OOS R² with each asset given another asset's topic instruments at "
     "random within each week (20 shuffles, averaged), factors still fitted to each week. The gap to the pooled "
     "OOS R² is what the real instruments add; it does not show topic signal, because noise topics' instruments "
     "also carry the assets' betas.",
     "28% against 9% shuffled is a gap of 19 points; with all betas set to 0 (no topic signal) the pair still "
     "reads about 25% and 8%."),
))

HOW_BKS_GAMMA: tuple[str, Bullets] = ("How to read the Gamma chart:", (
    ("Each bar is the length of the topic's Gamma row (the map from the topic's instrument to the K factor "
     "loadings) times the standard deviation of that instrument over the training asset-weeks. It says how far "
     "a one-standard-deviation move in the instrument shifts an asset's factor loadings.",
     "A3 Monetary Policy & Liquidity at 0.24 against S11 Real Estate at 0.026 means the same "
     "one-standard-deviation move shifts the loadings about 9 times as far."),
    ("The scaling puts the topics on the scale the lasso penalty uses. Raw row norms are large and not "
     "comparable across topics, because the instruments are small covariances with different spreads.",
     "a raw row norm of 80 on an instrument with a standard deviation of 0.003 gives a bar of 0.24."),
    ("Blue bars are the selected topics. The other topics have a Gamma row of zero and no visible bar.",
     "with 10 of 20 topics selected, the chart lists 20 topics and 10 of them have a blue bar."),
    ("A long bar does not show that the topic moves the assets: only each asset's loadings are pinned down, not "
     "the individual rows, and noise topics' instruments carry the assets' betas too.",
     "with all betas set to 0 (no topic signal), BKS still selects 11 of 20 topics and draws 11 blue bars."),
))

HOW_BKS_PATH: tuple[str, Bullets] = ("How to read the lambda path:", (
    ("The x-axis is log10 of lambda, over a grid from the largest value down to the grid ratio times it (1/100 "
     "and 12 points by default). The top panel counts the topics with a non-zero Gamma row at each point.",
     "at log10(lambda) = 0.53 (lambda = 3.4) 1 topic survives; at -1.47 (lambda = 0.034) 16 do."),
    ("The bottom panel is the in-sample Sharpe ratio of the best mix of the K factors over the training weeks, "
     "annualised with 52 weeks.",
     "a Sharpe ratio of 3.0 means that mix earned on average 0.42 of its weekly standard deviation per week "
     "(3.0 / √52)."),
    ("The vertical line marks the chosen lambda. The tolerance rule (2% by default) takes the largest lambda "
     "whose Sharpe ratio is within the tolerance of the peak, so it prefers fewer topics; the argmax rule takes "
     "the peak itself.",
     "with a peak of 3.00 at lambda 0.12 (13 topics), the 2% rule accepts every grid point with a Sharpe ratio "
     "of at least 2.94 and picks the largest of their lambdas, 0.28 (10 topics, Sharpe 2.95); the argmax rule "
     "would keep 0.12."),
    ("A high Sharpe ratio does not show that the chosen topics carry news information: noise topics' "
     "instruments inherit the assets' betas, so the factors earn their premium whichever topics carry them.",
     "with all betas set to 0 (no topic signal), the path still peaks at a Sharpe ratio of about 2.7 and keeps "
     "11 topics."),
))

HOW_BKS_R2: tuple[str, Bullets] = ("How to read the R² comparison:", (
    ("One row per asset, sorted by the direct estimator's R², highest on top. Circles: the direct estimator on "
     "the window's daily returns, with sensitivities frozen at the cut-off and nothing fitted in the window. "
     "Diamonds: BKS on the forecast weeks' returns, with K factors fitted to each week's returns.",
     "Energy Global v World EQ shows a circle at 2% (20 daily returns, 1 to 28 July) and a diamond at 15% (4 "
     "weekly returns, 30 June to 25 July)."),
    ("Both are uncentered: 1 minus the squared misses over the sum of squared returns, so below zero means the "
     "fit missed by more than the returns themselves. Values below -100% are drawn at -100%; the hover shows the "
     "actual value.",
     "an asset whose squared weekly misses are 2.6 times its squared weekly returns scores -160% and is drawn "
     "at -100%."),
    ("The medians in the caption under the tiles are the middle asset of each set of dots. They are not "
     "comparable: BKS refits its factors every week, so it scores above zero even without topic signal.",
     "medians of 28% (BKS) against 17% (direct) do not mean BKS forecasts better; with all betas set to 0 they "
     "read about 25% against 0%."),
))

HOW_BKS_SPLIT: tuple[str, Bullets] = ("How to read the per-topic split:", (
    ("Each topic bar is the topic's part of the asset's fitted return, summed over the forecast weeks. Each week "
     "it is the asset's instrument for the topic (measured before the week) times the topic's Gamma row times "
     "that week's factor values.",
     "S7 Financials at -0.14 pp means that term lowered Energy Global v World EQ's fitted return by 0.14 "
     "percentage points over the four weeks."),
    ("The topic bars, the constant's bar and the 'Other topics' bar add up to the fitted return. Adding 'Not "
     "explained by BKS' gives the realised move exactly.",
     "bars summing to -0.21 pp plus -0.59 pp not explained give the realised -0.80 pp."),
    ("Only selected topics have non-zero bars. The constant's bar is zero when its Gamma row is zero, as in the "
     "default full-history run; under the training-window history the penalised constant can stay non-zero.",
     "with 10 of 20 topics selected under the full history, the other 10 topics and the constant all read "
     "0.00 pp; under the training-window history the constant of Energy Global v World EQ reads +0.11 pp."),
    ("The split is not unique. Only each asset's loadings are pinned down, and in the model the topic "
     "instruments move along only K common directions, so another Gamma with the same fitted values would split "
     "them differently.",
     "the -0.14 pp shown for S7 Financials could sit partly on other topics in an equally good fit with the same "
     "-0.21 pp fitted return."),
    ("The realised move covers the BKS forecast weeks, which follow calendar weeks rather than the window's "
     "days. Under the full history its units are approximate: each week's volatility-scaled value times the "
     "asset's average daily volatility that week.",
     "for Energy Global v World EQ the weeks run 30 June to 25 July and give -0.80 pp (-0.81 pp from the daily "
     "returns), while the Topic contributions tab shows about +1.2 pp for 1 to 28 July."),
))


def how_bks_history(half_life_months: float) -> str:
    """The two covariance histories of BKS (shown under the BKS tab's lead caption)."""
    hl = f"{float(half_life_months):g}"
    return how_to_read("How to read the covariance history:", (
        (f"Each instrument is a covariance between the asset's daily (volatility-scaled) returns and the topic's "
         f"attention shocks, measured before the return week, with day weights that halve every {hl} months "
         "(the kernel half-life).",
         "at the default half-life of 69 months, a day 69 months earlier counts half as much as a day in the week "
         "just before the return week."),
        ("Full history before the cut-off (default): every day since the start of the data (April 2015, after "
         "the volatility warm-up), a 52-week burn-in, at least 60 days per instrument, and each daily return "
         "divided by its trailing 252-day volatility.",
         "for the last training week of the window 1 January to 30 June 2025, about 92% of the instruments' "
         "weight falls on days before 1 January, which the direct methods never see."),
        ("Training window only: returns from the training start, attention from w weekdays earlier (the shock "
         "window), no burn-in, at least 3 days per instrument, and each return divided by the asset's training "
         "standard deviation. BKS then sees what the direct methods see, but its first instruments rest on a few "
         "days.",
         "on the window 1 January to 30 June 2025 the first usable week ends 17 January with instruments over 7 "
         "days, and 24 of the 26 week ends are kept."),
        ("The history also sets the units of the fitted and realised returns: approximate under the full history "
         "(the week's average trailing volatility), exact under the training window (one training standard "
         "deviation).",
         "Energy Global v World EQ's four weeks sum to -0.80 pp under the full history and -0.81 pp under the "
         "training window, which matches its daily returns."),
    ))


# --- Lists ------------------------------------------------------------------------
HOW_ASSETS_TABLE: tuple[str, Bullets] = (
    "How to read the assets table (listed assets in the order of the source image):", (
        ("Each asset 'A v B' is long leg A and short leg B. Its daily return is the return of A minus the return "
         "of B.",
         "on a day when Energy Global rises 1.5% and World EQ 0.5%, Energy Global v World EQ returns +1.0%."),
        ("Outrights and 'XXX v USD' pairs are long against USD cash, a leg with return 0 and no index or proxy.",
         "Global Equity is MSCI ACWI (proxy ACWI) against USD cash, so its daily return is the ACWI leg's "
         "return."),
        ("Data source: 'real' uses daily prices from data/market. 'artificial' uses model returns (the Artificial "
         "price source, or a leg that failed to load). 'not in run' means the asset is not in this run.",
         "leaving out CHF v USD in the sidebar marks its row 'not in run', and switching the price source to "
         "Artificial marks the other 54 rows 'artificial'."),
    ))

HOW_GENERIC_ASSETS: tuple[str, Bullets] = ("How to read the generic assets table:", (
    ("One row per generic asset of this run, G_ASSET_001 onwards. The name repeats the asset class, drawn at "
     "random for each asset: Equity with probability 0.5, FX 0.3 and Fixed income 0.2.",
     "with 55 generic assets, on average 27.5 are Equity, 16.5 FX and 11 Fixed income, each named like "
     "'Generic asset 002 (Equity)'."),
    ("Sub class and source read 'generic' on every row: a generic asset has no legs, index or proxy, and its "
     "returns are artificial.",
     "G_ASSET_001 reads 'generic' in both columns, where Energy Global v World EQ has two legs in the assets "
     "table above."),
))

HOW_TOPICS_TABLE: tuple[str, Bullets] = ("How to read the manual topics table:", (
    ("The table lists the 20 manual topics of the report (Tables 1-2), with names and scope as printed. S1-S11 "
     "are in group Sector (Sector Ontology). A1-A6 are in group Macro and B1-B3 in group Micro (Global "
     "Multi-Asset Hierarchy).",
     "S1 is Energy in group Sector, and A2 is Inflation Expectations in group Macro."),
    ("The table always lists all 20. The sidebar's 'Manual topics' setting decides which of them enter the run.",
     "with 'Sector S1-S11 (11)' only S1-S11 run, and the link map below has no A or B rows."),
    ("The group is what the roll-up on the Topic contributions tab sums by. Generic topics G001, G002, ... form "
     "the group Generic, and the note below the table says how many of them carry random links.",
     "with 50 generic topics and a link share of 0.20, the note says 10 of them carry random links."),
))

HOW_LINK_MAP: tuple[str, Bullets] = (
    "How to read the link map (one row per linked pair; the default map is illustrative, not a research claim):", (
        ("A pair without a row has a set sensitivity of 0. Origin says where a link comes from: default (the "
         "authored map), random (seeded draws for generic topics) or override (a session edit).",
         "the default run has 95 links, all of origin default, 8 of them for S1 Energy."),
        ("Sign: the direction the asset moves when attention to the topic rises. For 'A v B' it is the direction "
         "of long A, short B.",
         "S1 Energy on Japan v World EQ has sign -1 (Japan imports most of its energy), so rising energy "
         "attention goes with Japan lagging the world index."),
        ("Tier sets the size through the sidebar's set sensitivities (betas): the set sensitivity W is the sign "
         "times the tier's beta. With two betas, moderate and weak links share beta 2. With one beta, every link "
         "has beta 1.",
         "with the default betas 0.35, 0.15 and 0.05, 'strong, +1' gives W = +0.35 and 'weak, -1' gives W = "
         "-0.05; with one beta both become 0.35 in size."),
        ("In standardised units, a topic linked to one asset with set sensitivity W has a designed shock "
         "correlated about W with the asset's return. The observed shock's correlation is about 0.90 times that "
         "at w = 5.",
         "a strong link of 0.35 gives an observed shock correlated about 0.35 × 0.90 ≈ 0.32 with the return, and "
         "the topic explains about 0.35² ≈ 12% of the asset's variance before that attenuation."),
        ("Feasibility scaling: when a topic's links would give its signal part a variance above 0.95, the "
         "topic's whole row is scaled down by one factor so that the variance equals 0.95.",
         "a topic whose signal variance would be 1.90 is scaled by the square root of 0.95 / 1.90, about 0.71, "
         "so its strong links of 0.35 become about 0.25."),
        ("To edit, change Tier (none removes the link) or Sign and press Apply edits. Edited rows show Origin "
         "override and Mechanism 'Session edit'. Edits last for this browser session, and Reset to default "
         "removes them.",
         "setting S1 Energy on CAD v USD to none and pressing Apply edits removes that row and sets its set "
         "sensitivity to 0."),
    ))


# --- Real data page ---------------------------------------------------------------
HOW_REAL_SETTINGS: tuple[str, Bullets] = ("How to read the settings table:", (
    ("One row per sidebar setting that will apply to real data, in four groups: time windows, direct estimator, "
     "BKS model and asset selection.",
     "on the defaults, Training window reads '2025-01-01 to 2025-06-30 (129 weekdays)' and Forecast length "
     "'4 weeks (2025-07-01 to 2025-07-28)'."),
    ("The direct estimator rows follow the chosen method: the elastic net lists its penalty rule and L1 ratio, "
     "ridge its lambda, and every method the selection threshold tau.",
     "with ridge chosen, one Ridge lambda row ('chosen by generalised cross-validation' by default) replaces "
     "the Penalty rule and L1 ratio rows."),
    ("Settings that only shape the simulation are left out; the note below the table lists them.",
     "changing the set sensitivities (betas) or the price source leaves this table unchanged."),
))

HOW_REAL_CONTRACT: tuple[str, Bullets] = ("How to read the data contract:", (
    ("One row per input file in data/real/: what it holds and its columns. The contract is still to be "
     "confirmed (TBC).",
     "attention.parquet has a date index and one column per topic_id, so 20 topics make 20 columns."),
    ("sensitivities.parquet has one row per estimation date (as_of), topic and asset: the sensitivity in % "
     "return per one-sd attention shock, its uncertainty as a standard deviation, its source, the window it was "
     "estimated on and its coverage.",
     "a row dated 2025-12-31 for S1 and ENERGY_v_WEQ with sensitivity 0.40, uncertainty 0.05 and coverage full "
     "reads: a +1 sd shock in S1 goes with +0.40% on that asset, give or take 0.05 (one standard deviation)."),
))

HOW_REAL_STATUS: tuple[str, Bullets] = ("How to read the status table:", (
    ("One row per file of the data contract. Found says whether the file is in data/real/.",
     "with only sensitivities.parquet in place the rows read no, no, yes, and the caption says 1 of 3 input "
     "files found."),
    ("Rows is the file's row count. sensitivities.parquet has one row per estimation date, topic and asset.",
     "20 topics, 55 assets and 12 month-end dates give 13,200 rows."),
    ("Dates is the range of as_of in sensitivities.parquet, or of the date index in attention.parquet. "
     "topics.csv has none. A file that cannot be read shows 'unreadable' and the error type.",
     "estimates dated from 2025-01-31 to 2025-12-31 show '2025-01-31 to 2025-12-31'."),
))

HOW_REAL_PREVIEW: tuple[str, Bullets] = ("How to read the preview:", (
    ("Each cell is the estimated topic sensitivity of the asset (row) to the topic (column) at the chosen "
     "estimation date, which is the latest by default. The unit is % return per one-standard-deviation (sd) "
     "attention shock, with the other topics held fixed.",
     "a cell of 0.35 means a +1 sd attention shock in the topic goes with an expected return of +0.35% for the "
     "asset, and a -2 sd shock with -0.70%."),
    ("Blank cells have coverage 'none' or no estimate in the file.",
     "a pair with sensitivity 0.20 and coverage 'none' is blank."),
    ("The AVERAGE row averages each column over the rows shown, counting blanks as zero.",
     "0.40, a blank and -0.10 average to (0.40 + 0 - 0.10) / 3 = 0.10."),
    ("Colours run from red (negative) through grey to blue (positive), scaled to the largest absolute value "
     "shown. Blank cells do not set the scale.",
     "if the largest absolute value shown is 0.40, cells of -0.40 and +0.40 get the darkest red and blue, and a "
     "cell of 0.20 a medium blue."),
    ("At most 40 topics are shown, in alphabetical order of topic_id, and the title then says how many there "
     "are. Rows follow data/reference/assets.csv, with other assets at the end.",
     "a file with 60 topics shows the first 40 and the title ends '(showing 40 of 60 topics)'."),
))


# --- BKS trace page (D90) ---------------------------------------------------------
# The examples are the dashboard defaults' BKS run (full history, seed 0), traced on the page: Energy Global v
# World EQ with S1 Energy, lambda 0.2773 with 10 of 20 topics, 26 training weeks, 4 forecast weeks, 55 assets x
# 20 topics, 92% of the kernel weight before 2025-01-01. They were checked against the trace of that run.
def _trace_tiles() -> tuple[str, Bullets]:
    """:data:`HOW_BKS_TILES` for the trace page, whose tiles sit above the steps: the lambda path and the Gamma
    rows it points to are in step 6, and the page has no per-topic split."""
    lead, bullets = HOW_BKS_TILES
    out = list(bullets)
    text, example = out[0]
    out[0] = (text.replace("on the lambda path below", "on the lambda path of step 6 (Fit and lambda)"), example)
    out[2] = (out[2][0], "10 of 20 means the other 10 topics have a Gamma row of exactly zero (step 6, the Gamma "
                         "heatmap).")
    return lead, tuple(out)


HOW_TRACE_TILES: tuple[str, Bullets] = _trace_tiles()

HOW_TRACE_STATUS: tuple[str, Bullets] = ("How to read the status table:", (
    ("One row per step: what it computes, how many of its graded checks hold ('info' checks are not counted) and "
     "this run's key numbers.",
     "on the defaults, 6 Fit and lambda reads '15 of 15 ok' and 'lambda 0.2773, 10 of 20 topics, in-sample Sharpe "
     "2.91'."),
    ("Reading: 'as expected' when every check holds and no departure is found; 'departs' when a diagnostic check "
     "is off or a finding marks a departure from its reference, which comes from the method, an implementation "
     "choice or the data; 'off' when an identity check fails, which points to a defect in the code.",
     "on the defaults, 6 Fit and lambda and 8 Implied sensitivities read 'departs', the other six steps 'as "
     "expected', and no step reads 'off'."),
))

HOW_TRACE_LADDER: tuple[str, Bullets] = ("How to read the ladder:", (
    ("Rows run from what the data allow (top) to what BKS delivers: the true sensitivities, the best a "
     "training-window estimator could find, the instruments used directly as topic covariances (with the shocks' "
     "covariance over the same kernel-weighted days, then over the training days as production does), their best "
     "K directions, the fit's K loadings inverted by least squares (their topic part, without the constant's, "
     "like the next row), the fit's own K directions, and the production BKS-implied result.",
     "on the defaults the Spearman values read 1.00, 0.78, 0.91, 0.71, 0.69, 0.68, 0.19 and 0.19 from top to "
     "bottom."),
    ("The drop from one row to the next is what that step costs; the largest drop is where the signal is lost.",
     "from 'BKS betas, least-squares inversion' (0.68) to 'BKS directions, no constant' (0.19) the Spearman falls "
     "by 0.49: the same 3 loadings keep the ranking when inverted by least squares and lose it in the projection "
     "BKS Eq. 5 uses."),
    ("Left panel: the Spearman rank correlation with the true sensitivities over all topic-asset pairs. Right "
     "panel: the median over assets of the OOS R² on the forecast window's return days, with the sensitivities "
     "frozen at the cut-off, as in the Compare methods tab. The table below gives the rows' full names.",
     "BKS-implied reads 0.19 and 7.0%, where the true sensitivities read 1.00 and 23.2%."),
    ("The ladder need not fall at every row: the full-history instruments see about ten years of kernel-weighted "
     "days, more than any estimator of the training window, so they can beat the window's own best.",
     "'Best a training window allows' scores 0.78 and the instruments with the same-history shocks' covariance "
     "0.91."),
    ("The median OOS R² over a few forecast weeks is noisy, so a row can beat the true sensitivities on it; the "
     "Spearman column is the steadier guide.",
     "'Best a training window allows' reads 25.8% and the instruments with the same-history shocks' covariance "
     "25.1%, against 23.2% for the true sensitivities."),
    ("Blue: the production BKS-implied row. Black: the true sensitivities and the two benchmarks, which are not "
     "steps of the chain: the sidebar's direct method and 'All zero' (every sensitivity 0, so its median OOS R² is "
     "exactly 0 and it has no Spearman). Grey: the steps in between. The hover gives the change from the row "
     "above.",
     "the elastic net benchmark reads 0.55 and 16.8%, and 'All zero' 0.0%."),
))

HOW_TRACE_LADDER_TABLE: tuple[str, Bullets] = ("How to read the ladder table:", (
    ("The chart's rows with their numbers. The change columns are this row minus the row above, the cost of that "
     "step; they are empty for the first row and for the benchmarks.",
     "'BKS directions, no constant' shows a change in Spearman of -0.494 (0.191 minus 0.685) and in R² of -10.9 "
     "points (7.0% minus 17.9%)."),
    ("RMSE: the root mean squared difference from the true sensitivities over all pairs, in standardised units. A "
     "row with a larger RMSE than 'All zero' misses the truth by more than the sensitivities themselves.",
     "BKS-implied has an RMSE of 0.0914, against 0.0784 for all zero and 0.0547 for the elastic net."),
    ("Median OOS R² is in percent and its change in percentage points; 'What it is' says how each row is built.",
     "the true sensitivities score 23.2%, so BKS-implied at 7.0% keeps less than a third of what the truth "
     "explains."),
))

HOW_TRACE_FINDINGS: tuple[str, Bullets] = ("How to read the findings:", (
    ("Defects come first: an identity check that is off, named 'Check off' with its numbers. A defect points to a "
     "coding error, in the code or in the trace.",
     "the defaults have none."),
    ("Departures follow: places where this run's result is not what it should be, with the reference it is "
     "measured against and this run's numbers. None comes from a coding error; the word in brackets says where it "
     "comes from: the method (BKS itself), an implementation choice (how this lab builds a step; its row of the "
     "ladder or its step shows the alternative) or the data (the sample of this run).",
     "on the defaults, step 8 lists 'Departure (method)', the fit's directions keep 38% of the instruments' squared "
     "norm against 91% for the best 3 directions, and 'Departure (implementation choice)', the instruments and the "
     "shocks' covariance cover different days."),
    ("Notes follow: facts that explain a number or limit a reference, without a departure; they name their origin "
     "too.",
     "a step 8 note says the implied sensitivities use the instruments of the week ending 2025-06-20, 7 trading "
     "days before the training end."),
))

HOW_TRACE_CHECKS: tuple[str, Bullets] = ("How to read the checks:", (
    ("One row per check: its status and kind, the observed value, the reference, the tolerance, what should "
     "hold and a note. Most observed values are the largest difference between the run and an independent "
     "recomputation, relative to the largest value.",
     "'Panel instruments = brute-force kernel covariance' observes 1.5e-15 against a tolerance of 1e-10: ok."),
    ("Status: 'ok' when the 'Should hold' rule holds, 'off' when it does not, 'info' shown for context and never "
     "graded. For a difference the rule is at most the tolerance; for the correlation and factor checks it is at "
     "least the threshold in the Tolerance column; the kept share must be at most the best share, the Reference.",
     "'Instruments track their population value' is ok at 0.98 against a threshold of 0.9; 'Unit conversion is "
     "approximate' is info: 0.41, the largest gap of the conversion from 1."),
    ("A name ending in '(consistency)' marks an identity between two of the trace's own results; the other "
     "identities recompute a result from the inputs, independently of the package code.",
     "'Step-by-step chain = production (consistency)' observes 0 on the defaults, and 'Implied sensitivities "
     "rebuilt with numpy = production' 2.7e-15."),
    ("Kind: an identity must hold up to rounding or the solver's stopping tolerance, so 'off' points to a code "
     "defect; a diagnostic describes the run, so 'off' means the run departs from what the method assumes.",
     "'Every factor is alive' is a diagnostic: the smallest over the largest factor variance is 0.37 on the "
     "defaults and would be off below 1e-6."),
    ("Values near 1e-16 are the rounding error of the arithmetic, not differences.",
     "'Objective recompute' observes 1.4e-16: the reported and the recomputed objective agree to about 16 "
     "digits."),
))

HOW_TRACE_SETTINGS: tuple[str, Bullets] = ("How to read the settings table:", (
    ("One row per setting the BKS run uses, as the steps below apply them; they come from the sidebar.",
     "'Kernel half-life (months) -> weekly decay xi' reads '69 -> 0.997684': each week further back weighs 0.23% "
     "less."),
    ("Training weeks are the return weeks ending inside the training window; forecast weeks are calendar weeks, "
     "so their days need not match the forecast window's days.",
     "'Training weeks' reads '26 (2025-01-03 to 2025-06-27)', and the first forecast week, ending 2025-07-04, "
     "starts on Monday 2025-06-30, the day before the forecast window."),
    ("'Forecast ridge' is the ridge of each forecast week's factor fit: 2, or 0 at lambda 0.",
     "a fixed lambda of 0 shows a forecast ridge of 0."),
))

HOW_TRACE_SHAPES: tuple[str, Bullets] = ("How to read the sizes table:", (
    ("Counts of days, weeks and rows at each stage, with their date ranges.",
     "55 assets in each of the 26 training weeks give 1,430 training rows."),
    ("The panel's days start after the volatility warm-up and its return weeks after the burn-in, so the first "
     "year of the full history feeds instruments only.",
     "the simulation starts on 2015-01-02, the panel's days on 2015-04-01 and its return weeks on 2016-04-08."),
    ("'Training return days of the direct estimator' are the days whose shocks give Sigma_z in step 8.",
     "129 days, 2025-01-01 to 2025-06-30."),
))

HOW_TRACE_INPUTS: tuple[str, Bullets] = ("How to read the inputs chart:", (
    ("Top: the simulated attention of the chosen topic, every day; bottom: the chosen asset's daily return. BKS "
     "and the direct methods start from these data only.",
     "S1 Energy's attention stays between 0.08 and 0.31 around a mean of 0.20."),
    ("Shaded: the training window and the forecast window up to the end of the last forecast week BKS scores. The "
     "days before the training start are history only the full-history BKS uses.",
     "on the defaults, training runs from 2025-01-01 to 2025-06-30 and the forecast shading from 2025-07-01 to "
     "2025-07-25, the end of the fourth forecast week (the window's last day, 2025-07-28, starts a week BKS does "
     "not score); the ten years before 2025 enter only the full-history instruments."),
    ("The topic follows the asset, its topic with the largest true sensitivity, until you pick one.",
     "for Energy Global v World EQ the topic is S1 Energy, with a true sensitivity of 0.35."),
))

HOW_TRACE_TRUTH: tuple[str, Bullets] = ("How to read the truth chart:", (
    ("Per topic, three reference values for the chosen asset: the set sensitivity W from the link map, the true "
     "sensitivity B_true the simulation produces, and B_true in the training scales the estimators use.",
     "Energy Global v World EQ has W = 0.35 and B_true = 0.35 for S1 Energy, 0.36 in training units."),
    ("B_true includes spillovers: a topic without a link can have a true sensitivity, because asset returns and "
     "topic shocks are correlated.",
     "S7 Financials has no link to Energy Global v World EQ (W = 0) and still a true sensitivity of 0.11."),
    ("Training units: B_true times the asset's full-sample over its training volatility, times the shock's "
     "training over its full-sample standard deviation. The implied sensitivities of step 8 are in these units.",
     "A6 has B_true = 0.105 and 0.117 in training units on Energy Global v World EQ."),
    ("With more than 30 topics, the 30 with the largest values are shown.",
     "with 520 topics the chart keeps 30 and says so in its subtitle."),
))

HOW_TRACE_DIVISOR: tuple[str, Bullets] = ("How to read the divisor chart:", (
    ("Top, solid: the divisor of each daily return. Under the full history it is the trailing 252-day volatility "
     "known the day before; under the training history the asset's training standard deviation, a flat line. "
     "Dashed: the training return scale that standardises the sensitivities; dotted: the full-sample volatility, "
     "the unit of B_true.",
     "for Energy Global v World EQ the divisor is 0.99% on 2025-01-02 and 1.11% on 2025-06-27, against a "
     "training scale of 1.14% and a full-sample volatility of 1.23%."),
    ("Bottom: the scaled return, the return divided by the divisor; with a good divisor its standard deviation is "
     "near 1.",
     "Energy Global v World EQ's scaled returns have a standard deviation of 1.10 over the training window."),
    ("Step 8 converts back with one number per asset, its mean training divisor (the mean of d over its training "
     "return days), while the instruments mix about ten years of divisors; the units table below measures the "
     "gap.",
     "a mean training divisor of 1.04% against a kernel-weighted mean divisor of 1.22% for Energy Global v World "
     "EQ."),
))

HOW_TRACE_UNITS: tuple[str, Bullets] = ("How to read the units table:", (
    ("One row per asset, the chosen asset first. Mean training divisor: the mean of the daily divisor d over the "
     "asset's training return days, which step 8 multiplies by; training return scale and full-sample volatility "
     "as in the chart.",
     "Energy Global v World EQ: a mean training divisor of 0.0104, training scale 0.0114 and full-sample "
     "volatility 0.0123."),
    ("Mean training divisor / training scale is below 1 when the trailing volatility over the training days was "
     "lower than the training standard deviation.",
     "0.91 for Energy Global v World EQ, and 0.54 to 1.10 across the 55 assets."),
    ("Conversion u: the mean training divisor times the kernel-weighted mean of 1 over d on the instrument's days. "
     "At 1 the conversion would be exact.",
     "0.95 for Energy Global v World EQ, and 0.64 to 1.41 across assets."),
    ("Exact-unit ratio: the instruments times the mean training divisor over the same kernel covariance computed "
     "with raw returns (a least-squares ratio over the asset's topics); 1 is exact, as under the training history. "
     "Kernel mean divisor: d averaged with the kernel's weights; the last column is the exact-unit ratio with it "
     "in place of the mean training divisor.",
     "0.78 for Energy Global v World EQ (median 0.97, range 0.51 to 1.30 across assets); its kernel mean divisor "
     "0.0122 would give 0.92 (0.92 to 1.09 across assets)."),
    ("'Zero (no training row)' marks assets without a row in the training weeks; they get zero sensitivities.",
     "no asset on the defaults."),
))

HOW_TRACE_SHOCKS: tuple[str, Bullets] = ("How to read the shocks chart:", (
    ("Top: the shock the BKS panel uses and, dotted, the direct estimator's shock of the same attention day. They "
     "should lie on top of each other; the check 'BKS shocks = direct shocks shifted by the lead' measures the "
     "gap.",
     "S1 Energy's shock on 2025-06-27 is -0.0209 on both lines."),
    ("Bottom: the same shock split into its designed signal part, which the returns load on, and its noise part "
     "(news noise plus the slow drift of attention). The two add up to the shock.",
     "on 2025-06-27, -0.0209 splits into a signal part of -0.0055 and a noise part of -0.0155."),
    ("The chart runs from three months before the training start to the forecast end; 'Show all days' shows the "
     "whole history.",
     "on the defaults, from 2024-10-01 to 2025-07-25, the end of the last forecast week."),
))

HOW_TRACE_SHOCK_TABLE: tuple[str, Bullets] = ("How to read the shocks table:", (
    ("Training sd and population sd: the shock's standard deviation over the training days, which the estimators "
     "standardise with, and over the whole simulation, which B_true uses. A ratio away from 1 changes the units "
     "of the sensitivities.",
     "S5's training sd is 1.41 times its population sd (0.031 against 0.022)."),
    ("Correlation with the designed shock against the truth's attenuation: the two should agree to a few "
     "hundredths.",
     "S1 Energy: 0.902 against 0.901."),
    ("Signal share: the share of the shock's variance that is designed signal; the rest is noise.",
     "29% for S1 Energy and 6% for A6."),
))

HOW_TRACE_INSTRUMENT: tuple[str, Bullets] = ("How to read the instrument chart:", (
    ("One point per instrument week: the kernel covariance of the asset's scaled return with the topic's shock, "
     "recomputed for every week (line), the panel's value (dots, which must sit on the line) and the population "
     "reference (dashed).",
     "Energy Global v World EQ and S1 Energy in the week ending 2025-06-20: 0.00881 on the line and in the panel, "
     "against a reference of 0.00881."),
    ("With a half-life of 69 months each new week adds little weight, so the line moves slowly.",
     "from the week ending 2025-06-13 to the week ending 2025-07-18 the value moves from 0.00884 to 0.00875."),
    ("Shaded: the instrument weeks that feed the training rows and the forecast rows, one week before their "
     "return weeks. The dashed vertical line marks the instrument of the chosen return week.",
     "the training rows use the instrument weeks ending 2024-12-27 to 2025-06-20."),
    ("The line starts once 60 days are available; the dots start after the 52-week burn-in.",
     "the line starts at the week ending 2015-07-03 (62 days), the dots at the week ending 2016-04-01."),
))

HOW_TRACE_KERNEL: tuple[str, Bullets] = ("How to read the kernel chart:", (
    ("Top: the weight of each day in the chosen week's instrument, xi to the power of the weeks back, normalised to "
     "sum to 1. The last day of the instrument week is left out, and later days weigh nothing.",
     "for the week ending 2025-06-20 the last day used is 2025-06-19, and it weighs 3.4 times as much as the first "
     "day, 2015-04-08."),
    ("Bottom: the running sum of each day's contribution, the weight times the return's and the shock's "
     "deviations from their weighted means. It ends at the instrument; the dashed line is the panel's value.",
     "for Energy Global v World EQ and S1 Energy the sum ends at 0.00881, the panel's value."),
    ("The subtitle gives the share of the weight before the training start and the effective number of days (1 "
     "over the sum of the squared weights).",
     "92% of the weight lies before 2025-01-01, and the 2,662 days count as 2,369 effective days."),
))

HOW_TRACE_INSTR_TRUTH: tuple[str, Bullets] = ("How to read the instrument scatter:", (
    ("One point per asset and topic: across, the population reference (the simulation's covariance of the shock "
     "with the return, divided by the asset's full-sample volatility under the full history, by its training "
     "standard deviation under the training history; the axis title names it); up, the instrument the implied "
     "sensitivities use. Orange: the chosen asset.",
     "Energy Global v World EQ's S1 Energy point sits at (0.00881, 0.00881)."),
    ("The subtitle gives the least-squares slope and the correlation; on the diagonal an instrument equals its "
     "population value.",
     "on the defaults the slope is 1.07 and the correlation 0.98."),
    ("Under the full history the panel divides by a trailing volatility, not by the reference's full-sample one, "
     "so the match is close but not exact. Under the training history both divide by the training standard "
     "deviation, but the instruments are covariances over a few months and scatter more.",
     "the training-history run of the defaults correlates 0.70."),
))

HOW_TRACE_INSTR_ROW: tuple[str, Bullets] = ("How to read the instrument row:", (
    ("One row per topic for the chosen asset and return week: the panel's value, the same covariance recomputed by "
     "direct summation over the days, and their difference, which should be rounding only.",
     "S1 Energy on Energy Global v World EQ: 0.00881 in both, a difference of 3e-17."),
    ("Population reference: what the instrument should be with unlimited data. Signal part and noise part: the "
     "covariance with the shock's designed signal and with its noise; they add up to the recomputed value.",
     "S1 Energy: reference 0.00881, signal part 0.00891 and noise part -0.00010."),
))

HOW_TRACE_PAIRING: tuple[str, Bullets] = ("How to read the pairing table:", (
    ("One row per return week, from four weeks before the first training week to the last forecast week. Each "
     "return week's row uses the instruments of the week before, whose kernel window ends one day before that "
     "week's end.",
     "the return week ending 2025-06-27 starts on 2025-06-23 and uses the instrument week ending 2025-06-20, with "
     "a kernel window up to 2025-06-19."),
    ("Weekly return (panel): the week's sum of scaled daily returns in the panel; Recomputed: the same sum from the "
     "daily data; Raw return: the sum of the unscaled daily returns, in percent.",
     "Energy Global v World EQ in the week ending 2025-06-27: -5.78 in the panel and recomputed, -6.33% raw."),
    ("Role: training, forecast, other (in the panel but not used) or not in panel; 'Assets that week' counts the "
     "week's rows.",
     "the four weeks before 2025-01-03 read 'other', with 55 assets each."),
))

HOW_TRACE_DESIGN: tuple[str, Bullets] = ("How to read the design heatmap:", (
    ("One row per asset of the chosen return week and one column per instrument: the constant and each topic's "
     "instrument, divided by its standard deviation over the training rows. The fit multiplies these rows by "
     "Gamma.",
     "Energy Global v World EQ's S1 Energy cell reads 3.16: its instrument is 3.16 training standard deviations."),
    ("Blue is positive and red negative, on a scale symmetric around zero; the outlined row is the chosen asset.",
     "in the week ending 2025-06-27 the topic cells run from -4.3 to 6.0."),
))

HOW_TRACE_STABILITY: tuple[str, Bullets] = ("How to read the stability chart:", (
    ("Left: how much each topic's instrument changes over the training weeks (the mean over assets of its "
     "standard deviation over the weeks), over how much it differs across assets. Near 0 the fit sees the same "
     "cross-section every week.",
     "0.03 to 0.08 on the defaults: over 26 weeks the instruments move by less than a tenth of their spread "
     "across assets."),
    ("Right: the instrument's mean over the training rows divided by its standard deviation. Far from 0, much of "
     "the instrument is a level shared by all assets, like the constant.",
     "S2 at 0.88 and A3 at -0.62."),
))

HOW_TRACE_PATH: tuple[str, Bullets] = ("How to read the lambda path and its noise:", (
    ("Top panel: the in-sample Sharpe ratio at each grid lambda, with a band of one standard error. Filled points "
     "are inside the tolerance band; the ring marks the best point, the diamond the chosen one, and the dashed "
     "horizontal line the lowest Sharpe ratio inside the band. A dotted line marks the band's relative reading "
     "(the tolerance times the best value) when it differs; the two differ below a best Sharpe ratio of 1.",
     "on the defaults the Sharpe ratio runs from 2.35 to 2.97 with a standard error of about 1.47; both readings "
     "put the band floor at 2.91, and the chosen lambda 0.277 keeps 10 topics."),
    ("Grey band 'No priced signal' with its dashed median: the 5% to 95% range of the in-sample Sharpe ratio that "
     "K factors reach over the training weeks when no factor is priced. A path inside it cannot tell the lambdas "
     "apart.",
     "with 3 factors over 26 weeks the range is 0.87 to 4.44 around a median of 2.30, and the whole path lies "
     "inside it."),
    ("Middle panel: the number of topics selected. Bottom panel: the implied sensitivities refitted at each lambda: "
     "their Spearman with the true sensitivities, the share of the instruments the fit's directions keep and "
     "their median OOS R² (as fractions).",
     "at the chosen lambda: Spearman 0.19, kept share 0.38 and median OOS R² 0.07."),
    ("The x-axis is logarithmic, from the grid ratio times lambda max up to lambda max.",
     "12 points from 0.034 to 3.42."),
))

HOW_TRACE_COEF_PATH: tuple[str, Bullets] = ("How to read the Gamma path:", (
    ("One line per instrument: the length of its Gamma row times its training standard deviation, at each grid "
     "lambda. A line at zero is a dropped instrument.",
     "A3 falls from 0.97 at the smallest lambda to 0.24 at the chosen one."),
    ("Coloured: the instruments largest at the chosen lambda (at most 8); dark grey: the other selected ones; "
     "light grey: the rest. The dashed vertical line marks the chosen lambda.",
     "at lambda 0.277 the 10 selected topics have a line above zero and the constant is at zero."),
    ("The topics the path keeps need not be the topics that move the assets most; the topic chart and table below "
     "set them side by side.",
     "A4 Financial Conditions & Market Stress, the topic with the most links (10) and the second-largest true "
     "sensitivities, and S2 Materials (third-largest) never enter at any grid lambda; the last survivor at lambda "
     "3.42 is A5 China Macro, the topic with the largest."),
))

HOW_TRACE_GAMMA: tuple[str, Bullets] = ("How to read the Gamma heatmap:", (
    ("Rows: the constant and each topic's instrument; columns: the K factors. Each cell is Gamma times the "
     "instrument's training standard deviation: how far a one-standard-deviation move of that instrument shifts "
     "that factor loading.",
     "A1 has 0.214 in factor 1: an asset whose A1 instrument is one standard deviation higher has a factor 1 "
     "loading 0.214 higher."),
    ("Rows of zeros are the dropped instruments; bold rows are the selected topics.",
     "10 of 20 topic rows are not zero at lambda 0.2773, and the constant's row is zero under the full history."),
))

HOW_TRACE_TOPICS: tuple[str, Bullets] = ("How to read the topic chart:", (
    ("Per topic, the largest true sensitivities first: its share of the true sensitivities (the sum over assets of "
     "their absolute values, over the same sum for all topics) and its share of the standardised Gamma row norms at "
     "the chosen lambda. A topic that matters for the assets has a long first bar; a topic the fit keeps has a "
     "second bar.",
     "A5 China Macro has the largest share of the true sensitivities, 11.5%, and 18.9% of the Gamma row norms."),
    ("The group lasso keeps the instruments that help fit the weekly returns, not the topics with the largest true "
     "sensitivities, so a long first bar can come without a second.",
     "A4 Financial Conditions & Market Stress and S2 Materials, second and third by true sensitivity, have no "
     "Gamma bar, while S6, S11 and B2, 10th, 15th and 17th, are kept."),
))

HOW_TRACE_TOPIC_TABLE: tuple[str, Bullets] = ("How to read the topic table:", (
    ("One row per topic, ranked by the sum over assets of its absolute true sensitivities: that sum, its rank, its "
     "links in the link map, whether the fit selected it, its standardised Gamma row norm (the row's length in the "
     "heatmap above) and the largest grid lambda at which it is selected.",
     "A5 China Macro: 6.22, rank 1, 8 links, selected, entering at lambda 3.42, the largest of the grid."),
    ("'Enters at lambda' is empty for a topic that is selected at no grid lambda, and for every topic under a "
     "fixed lambda, which has no path.",
     "A4 Financial Conditions & Market Stress, rank 2 with the most links (10), is never selected, so its "
     "'Enters at lambda' cell is empty."),
))

HOW_TRACE_KKT: tuple[str, Bullets] = ("How to read the optimum check:", (
    ("For each instrument, the length of the fit's gradient over its penalty, on the polished copy of the fit that "
     "the check grades (the stored fit run on to a much tighter stopping rule). At the optimum a kept topic sits at "
     "exactly 1 and a dropped one at 1 or below; the dashed line marks 1.",
     "the 10 kept topics lie within 0.00001 of 1 and the dropped ones at 0.94 or below."),
    ("The stored fit stops earlier, when a sweep changes the objective by less than 1e-8 (relative), so its own "
     "ratios miss 1 by up to about 0.001; the info check 'Stationarity of the stored fit' reports how far.",
     "the stored fit's kept topics lie between 0.9992 and 1.0001, A5 China Macro at 0.9992."),
    ("A dropped topic close to 1 would enter the fit at a slightly smaller lambda; the subtitle names the "
     "closest.",
     "A6 at 0.94 is the next to enter."),
))

HOW_TRACE_FACTORS: tuple[str, Bullets] = ("How to read the factor chart:", (
    ("Top: the K factor values of each training week; bottom: their running sums, where a steady drift is a mean "
     "the Sharpe criterion rewards.",
     "factor 2 has a mean of 0.61 and a standard deviation of 1.91 over the 26 weeks, an annualised Sharpe ratio "
     "of 2.31 on its own."),
    ("The factors are in canonical form: uncorrelated, factor 1 with the largest variance, every mean at least 0.",
     "their standard deviations are 2.01, 1.91 and 1.23."),
))

HOW_TRACE_PATH_TABLE: tuple[str, Bullets] = ("How to read the path table:", (
    ("One row per grid lambda: the Sharpe ratio, its standard error, the topics selected, the in-sample R², the "
     "objective against the objective of Gamma = 0, and the solver's sweeps.",
     "the chosen row, lambda 0.2773, has a Sharpe ratio of 2.910, 10 topics and an in-sample R² of 44.7%; the "
     "best row, lambda 0.1200, has 2.967 with 13 topics."),
    ("Flags: Chosen, Best, In band (inside the tolerance band), Grid edge (the point is the first or last of the "
     "grid), Above Gamma = 0 (a fit that stopped above the all-zero objective, a spurious stationary point), "
     "Converged and Rank cut (the Sharpe criterion dropped a factor direction).",
     "lambda 3.419 is flagged above Gamma = 0: an objective of 4,821.3 against 4,691.6."),
    ("The null columns repeat the Sharpe range reachable with no priced factor. The last columns refit the "
     "implied sensitivities at each lambda: kept share, Spearman, median OOS R², and the rank and smallest "
     "standardised singular value of Gamma's topic rows (near 0, the fit uses fewer than K directions).",
     "the null range 0.87 to 4.44 holds every Sharpe ratio of the path; from lambda 0.64 Gamma's rank drops to 2 "
     "and its smallest singular value to about 0."),
))

HOW_TRACE_WEEK_R2: tuple[str, Bullets] = ("How to read the weekly R² chart:", (
    ("One group of bars per forecast week: the R² over all the week's assets in panel units, and the same with "
     "the topic instruments shuffled across assets.",
     "the week ending 2025-07-04 reads 20.2% against 6.3% shuffled."),
    ("Each week's K factors are fitted to that week's returns, so both bars are above zero even without topic "
     "signal. The shuffled bar is a reference for the instruments' cross-sectional structure, not for topic "
     "signal: noise topics' instruments also carry the assets' betas, so the gap appears without any. Step 8 and "
     "the Summary's ladder score the topics.",
     "the pooled R² is 27.8% against 9.1% shuffled on the defaults; with every set sensitivity at 0 the same run "
     "reads 25.0% against 7.7%."),
    ("Third bar: the same forecasts in return units against the exact weekly returns (sums of the raw daily "
     "returns). Under the full history the return units are approximate, so it differs from the first bar; the "
     "line under the caption gives both pooled values.",
     "21.8% for the week ending 2025-07-04; pooled 23.0% against 27.8% in panel units."),
))

HOW_TRACE_WEEK_SCATTER: tuple[str, Bullets] = ("How to read the week scatter:", (
    ("One point per asset of the chosen forecast week: across, the fitted weekly return; up, the realised one, "
     "both in panel units (sums of scaled daily returns). Orange: the chosen asset.",
     "Energy Global v World EQ in the week ending 2025-07-04: fitted -0.07, realised 0.48."),
    ("The week's R² is 1 minus the squared misses over the squared realised returns; the subtitle gives it with "
     "the shuffled reference.",
     "20.2% for the week ending 2025-07-04, shuffled 6.3%."),
))

HOW_TRACE_OOS_FACTORS: tuple[str, Bullets] = ("How to read the forecast factors:", (
    ("Each forecast week's K factor values, fitted to that week's own returns with the loadings frozen; they are "
     "fitted, not forecast.",
     "factor 3 is -1.21 in the week ending 2025-07-11."),
    ("Compare their size with the training factors of step 6. Both use the same ridge of 2, which shrinks each "
     "set by about the same factor; the forecast factors are smaller because the forecast weeks' returns are "
     "smaller.",
     "the forecast values stay within -1.24 and 0.83, while the training factors have standard deviations of 1.2 "
     "to 2.0; the weekly returns' standard deviation across assets is about 1.2 in the forecast weeks against 2.2 "
     "in training."),
))

HOW_TRACE_WEEK_TABLE: tuple[str, Bullets] = ("How to read the week table:", (
    ("One row per asset of the chosen forecast week, the chosen asset first: the realised and fitted weekly "
     "returns in panel units and their difference.",
     "Energy Global v World EQ in the week ending 2025-07-04: realised 0.477, fitted -0.067, residual 0.544."),
    ("Loadings 1 to K: the asset's instruments times Gamma; the fitted return is the loadings times the week's "
     "factors.",
     "0.179 × 0.688 + 0.228 × (-0.411) - 0.240 × 0.402 = -0.067."),
))

HOW_TRACE_CHAIN: tuple[str, Bullets] = ("How to read the chain chart:", (
    ("Per topic for the chosen asset: the true sensitivity, the instruments used directly as topic covariances "
     "with the shocks' covariance over the same history (the best practical row of the ladder), the fit's K "
     "betas turned back into topic covariances by least squares instead of the Eq. 5 projection, the "
     "BKS-implied sensitivity and the sidebar's direct method.",
     "Energy Global v World EQ and S1 Energy: true 0.35, instruments alone 0.32, least-squares inversion 0.07, "
     "BKS-implied -0.03."),
    ("A topic the fit dropped lies outside the fit's K directions, so the projection sets its covariance to zero; "
     "its implied sensitivity then comes only from its correlation with other topics' shocks.",
     "S1 Energy is not among the 10 selected topics: its projected covariance with Energy Global v World EQ is 0 "
     "and its implied sensitivity -0.03."),
))

HOW_TRACE_SHARES: tuple[str, Bullets] = ("How to read the shares table:", (
    ("The share of the instruments' squared norm, over all assets, that K directions in topic space keep: the "
     "fit's own directions (what BKS Eq. 5 keeps), the best K directions, a least-squares reconstruction from the "
     "same K loadings, and K random directions.",
     "37.9% for the fit's directions, 91.4% for the best 3, 89.1% by least squares and 15.0% for 3 random "
     "directions of 20."),
    ("The last row is the chosen asset's own share for the fit's directions.",
     "15.4% for Energy Global v World EQ, whose largest instrument, S1 Energy, is a topic the fit dropped."),
))

HOW_TRACE_CAPTURE: tuple[str, Bullets] = ("How to read the direction chart:", (
    ("Top: the principal directions of the instruments across assets, largest first, and each one's share of "
     "their squared norm; the first K together give the best K share.",
     "55.8%, 27.5% and 8.0% for the first three directions, 91.4% together."),
    ("Bottom: how much of each direction lies inside the fit's K directions; 100% would be all of it.",
     "31%, 53% and 65% of the first three: the fit's directions miss much of where the instruments vary."),
))

HOW_TRACE_IMPLIED_SCATTER: tuple[str, Bullets] = ("How to read the implied scatter:", (
    ("One point per topic-asset pair: across, the true sensitivity; up, the BKS-implied one, in standardised "
     "units. Orange: the chosen asset's topics.",
     "Energy Global v World EQ's S1 Energy point sits at (0.35, -0.03)."),
    ("The subtitle gives the least-squares slope and the correlation over all pairs; on the diagonal the implied "
     "sensitivity equals the truth.",
     "on the defaults the slope is 0.39 and the correlation 0.36."),
))

HOW_TRACE_SIGMA: tuple[str, Bullets] = ("How to read the Sigma_z chart:", (
    ("Per topic: the shock's variance over the training days and over the days the instruments weigh (their "
     "kernel), each divided by the population variance. Step 8 divides by the training-day covariance, while the "
     "instruments average the kernel's days.",
     "S5's variance is 1.98 times the population value over the training days, but 1.07 times over the kernel's "
     "days."),
    ("The dashed line at 1 is the population. With a long kernel the kernel bar stays near 1, while the training "
     "bar moves with six months of data.",
     "S9 reads 0.70 over the training days and 1.02 over the kernel."),
))

HOW_TRACE_SIGMA_TABLE: tuple[str, Bullets] = ("How to read the Sigma_z table:", (
    ("Per topic: the ratios of the chart (training and kernel variance over the population) and training over "
     "kernel, the mismatch step 8 carries.",
     "S5: 1.98, 1.07 and 1.85."),
    ("The last column is the largest difference between the training-day and the kernel correlation of this "
     "topic's shock with another topic's shock: Sigma_z differs off its diagonal too.",
     "A1 reaches 0.30."),
))

HOW_TRACE_CHAIN_TABLE: tuple[str, Bullets] = ("How to read the chain table:", (
    ("The columns follow BKS Eq. 5 for the chosen asset: its instrument (panel units), the part inside the fit's K "
     "directions, the constant's part, their sum m, the least-squares covariance from the same betas for "
     "comparison, m in return units, the raw b after the shocks' covariance, the BKS-implied sensitivity and the "
     "least-squares inversion's.",
     "S7 Financials on Energy Global v World EQ: instrument 0.0027, projected 0.0019, m 0.0019, least-squares "
     "covariance 0.0030, BKS-implied 0.060 and least-squares inversion 0.122."),
    ("Then the references in standardised units: the true sensitivity (also in training units), the window "
     "truth, the instruments alone with the training-day and with the same-history Sigma_z, and the direct "
     "method.",
     "S7 Financials: true 0.107, window truth -0.001, instruments alone 0.104 and 0.048."),
))
