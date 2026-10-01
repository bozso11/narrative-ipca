"""Streamlit dashboard of the topic-sensitivity lab (DESIGN.md G.9, G.10, G.15, G.16; D66-D70, D80, D83-D85, D88, D90).

Terminology: the **topic sensitivity** of asset ``n`` to topic ``k`` is the
expected return response of asset ``n`` to a one-standard-deviation attention
shock in topic ``k``, with the other topics' shocks held fixed (the
coefficient ``b_kn`` in the regression of the asset's return on all topics'
attention shocks at once). It is not a position size or dollar exposure. The
page text says "sensitivity" (renamed 2026-09-30); in code, "exposure" means
topic sensitivity: ``exposure_tab``, the ``ex_`` widget keys,
``ExposureConfig``, ``B_hat``, the package ``narrative_ipca.exposure_lab`` and
the module ``real_exposures`` keep the older word.

Run from the repository root::

    .venv/Scripts/python.exe -m streamlit run dashboard/app.py

Three pages in the top navigation (D82, D90): **Simulation lab** (this
module's ``main``), **BKS trace** (``bks_trace_page`` here, the steps in
``trace_page.py``: the BKS run of the current settings traced step by step
with a reference next to each result) and **Real data**
(``real_exposures.py``, a placeholder until the research pipeline delivers
topic sensitivities). :data:`PAGES` holds the page objects so that the BKS
and Compare methods tabs can link to the trace (``st.page_link``).

The sidebar is shared by all pages (owner request 2026-09-29): the
entrypoint at the bottom of this module draws it and validates its values
before the chosen page runs, and keeps the result in :data:`_RUN` (page
callables take no arguments). The Real data page lists the settings that will
apply to real data.

The sidebar sets a :class:`~narrative_ipca.exposure_lab.config.LabConfig`;
one :class:`~narrative_ipca.exposure_lab.session.LabSession` per server
process memoises every stage by its config key (G.10, D71), so a control
change re-runs only the stages that depend on it. BKS runs only on request
(the "Run BKS" buttons); its result is kept in ``st.session_state`` with its
config key and flagged as stale when the settings change. A cached BKS fit is
reused automatically only when this browser session requested it.

The Compare methods tab (G.15, D83) reads the session's ``comparison``
stage: the direct fits and the BKS-implied sensitivities are cached by their
training keys, so changing only the forecast window re-scores them without
refitting. The tab never starts a BKS fit; it offers a Run BKS button that
fits every selected BKS variant (full history, training window only; D88)
this browser session has not fitted yet. The sidebar's "Covariance history"
radio chooses the variant of the BKS tab, the BKS trace and the sidebar's
Run BKS.

Run BKS works on the Simulation lab and BKS trace pages: both handle the
requests with :func:`_bks_sync` (a request is never left for a later visit to
the other page) and write the same ``bks_store``, so the BKS tab and the trace
always show the same run (D90).

Widget state: every control keeps the user's own value in
``st.session_state["_values"]`` (copied back by an ``on_change`` callback),
so values survive while a control is hidden and come back when a range that
had clamped them widens again. The value in effect in a run (after clamping)
is in ``st.session_state["_effective"]``. Sidebar keys start with ``sb_``.

Every chart, results or reference table and row of tiles has a "How to read"
caption whose bullets end with a static example (owner request 2026-09-30,
G.9): under it, or directly above the link map's editor (the BKS trace page's
tiles: in a collapsed expander right under them). Diagnostics (stage
timings) and input editors (the long/short view, the session's link edits)
have none. The texts are in ``_ui`` (``how_to_read`` and the ``HOW_*``
constants).

The BKS run makes no Streamlit call while it computes: a widget change
during a long fit would otherwise raise Streamlit's rerun exception inside
the fit and discard it. The request flag is cleared only when the run ends,
so an interrupted run resumes from the cached stages on the next rerun.
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import _ui  # noqa: E402
import real_exposures  # noqa: E402
import trace_page  # noqa: E402
from narrative_ipca.exposure_lab import charts, reference  # noqa: E402
from narrative_ipca.exposure_lab import bks as lab_bks  # noqa: E402
from narrative_ipca.exposure_lab import compare as lab_compare  # noqa: E402
from narrative_ipca.exposure_lab.bks import D52_NOTE, IMPLIED_METHOD, IMPLIED_NOTE  # noqa: E402
from narrative_ipca.exposure_lab.config import (  # noqa: E402
    DATA_END,
    DATA_START,
    N_MANUAL,
    AttentionConfig,
    BKSLabConfig,
    LabConfig,
)
from narrative_ipca.exposure_lab.dgp import attenuation  # noqa: E402
from narrative_ipca.exposure_lab.session import BKS_NOT_RUN, BKS_OFF, BKSNotCached, LabSession  # noqa: E402

logger = logging.getLogger("dashboard.app")

st.set_page_config(page_title="Topic-sensitivity lab", layout="wide")

_VALUES = "_values"
_EFFECTIVE = "_effective"
_D0, _D1 = dt.date.fromisoformat(DATA_START), dt.date.fromisoformat(DATA_END)

#: URL path of the Real data page (the sidebar's Run BKS button is disabled there).
REAL_DATA_URL = "real-data"

#: URL path of the BKS trace page (D90).
BKS_TRACE_URL = "bks-trace"

#: The ``StreamlitPage`` objects of this run by name (``simulation``, ``trace``, ``real``), set by the entrypoint
#: before the page runs, so that pages can link to each other with ``st.page_link`` (D90).
PAGES: dict[str, Any] = {}

#: Result of the shared sidebar for this script run, set by the entrypoint before the page runs:
#: ``values`` (in effect), ``cfg`` (or ``None``), ``errors``, ``notes``, ``train_window``,
#: ``feas_slot`` and the reference tables. Streamlit executes the script in a fresh module
#: namespace on every run, so the holder is per run.
_RUN: dict[str, Any] = {}

#: Tabs of the simulation page, in order (G.9).
TABS: tuple[str, ...] = (
    "Overview", "Correlation table", "Topic contributions", "Compare methods", "BKS", "Lists", "Data and method",
)

#: Defaults of the controls in the main area (tabs).
MAIN_DEFAULTS: dict[str, Any] = {
    "ex_metric": _ui.METRICS[0],
    "ex_units": _ui.EXPOSURE_UNITS[0],
    "ex_blank_rule": True,
    "ex_threshold": 0.0,
    "ex_max_cols": 40,
    "ex_row_order": _ui.ROW_ORDERS[0],
    "ex_ls_view": False,
    "ex_max_rows": _ui.HEATMAP_DEFAULT_MAX_ROWS,
    "ex_show_text": True,
    "ex_colors": next(iter(_ui.HEATMAP_COLORS)),
    "tc_asset": _ui.DEFAULT_CONTRIB_ASSET,
    "tc_view": "Variance share",
    "tc_rollup": False,
    "tc_top_n": 15,
    "bks_asset": _ui.DEFAULT_CONTRIB_ASSET,
    "cm_methods": list(_ui.COMPARE_DEFAULT_METHODS),
    "cm_inspect": None,  # None: follow the sidebar's direct method until the user picks one
    # BKS trace page (D90); the asset is the BKS tab's "bks_asset", shared by both pages
    "tr_step": "summary",
    "tr_topic": None,  # None: follow the asset (its topic with the largest true sensitivity)
    "tr_week": None,  # None: follow the run (the return week whose rows the implied sensitivities use)
    "tr_fweek": None,  # None: follow the run (the first forecast week)
    "tr_shock_all": False,
}

D47_NOTE = (
    "D47: the covariance instruments of noise topics inherit the betas of the assets they co-move with, so "
    "the in-sample Sharpe ratio that chooses lambda cannot certify that a selected topic carries narrative "
    "information."
)

#: Data and method tab: the definition of the topic sensitivity (owner decision 2026-09-30). The only page
#: text that says "exposure", to explain the old name.
TERMINOLOGY = (
    "**Topic sensitivity** $b_{k,n}$ (the matrix $B$, topics x assets): the expected return response of asset $n$ "
    "to a one-standard-deviation attention shock in topic $k$, with the other topics' shocks held fixed. It is the "
    "coefficient in the regression of the asset's return on all topics' attention shocks at once (step 6 below). "
    "Its units are % per one-standard-deviation shock, or standardised units (the return divided by its standard "
    "deviation), in which case $b^2$ is about the share of variance the topic explains. It says how an asset's "
    "return moves with news attention, not how much of the asset a portfolio holds. The lab shows three versions: "
    "the set sensitivity $W$ (set with the link map and the betas), the true sensitivity $B_{true}$ (the "
    "population value the simulation produces, including spillovers) and the estimated sensitivity $\\hat B$ "
    "(each method's training-window estimate). Formerly called 'exposure', a word easily read as a dollar "
    "exposure; the code keeps the old name."
)

#: Compare tab: the pointer to the reasons (DESIGN.md G.15.1).
WHY_BKS_LOWER_CAPTION = (
    "Why the BKS-implied rows score lower than the direct methods: see the Data and method tab (DESIGN.md G.15.1)."
)

#: Data and method tab: the reasons, measured on the dashboard defaults (DESIGN.md G.15.1; mean over noise seeds 0-2).
WHY_BKS_LOWER = (
    "The loss is in the step that turns the BKS fit back into topic sensitivities (BKS Eq. 5). Measured on the "
    "dashboard defaults, mean over noise seeds 0-2:\n\n"
    "1. **The directions BKS keeps.** The implied sensitivities keep only the part of each asset's topic "
    "covariances that lies in the K directions BKS fitted to explain weekly returns. With K = 3 those directions "
    "hold 3% (lambda 0) to 38% (tuned) of the instruments' variation; the best three directions would hold 92%. "
    "This step costs about 12 points of median OOS R² and 0.6 of Spearman correlation with the true "
    "sensitivities.\n"
    "2. **Not the number of factors.** The best three directions of the same covariances score 10.7%, against "
    "11.0% with all 20.\n"
    "3. **Not the extra history, the units or the constant.** The full-history instruments alone score 11.0%, "
    "against -2.4% for OLS on the training window. The unit conversion gains a few points. The tuned fit's "
    "constant is about zero; it costs about 12 points only when K equals the number of topics.\n"
    "4. **The training-window variant** goes through the same step from weaker instruments (about as good as "
    "OLS) and scores lower still: -36% on average.\n\n"
    "BKS is built to find the few factors that price the assets and the topics behind them, and it recovers "
    "the factor betas well (D52). The direct methods estimate the sensitivity of each asset to each topic."
)


# ---------------------------------------------------------------------------
# State and widgets
# ---------------------------------------------------------------------------
@st.cache_resource
def get_session() -> LabSession:
    """One lab session per server process (stage results shared across browser sessions)."""
    return LabSession(max_entries=6)


def _init_state() -> None:
    defaults = {**_ui.default_values(), **MAIN_DEFAULTS}
    if _VALUES not in st.session_state:
        st.session_state[_VALUES] = defaults
    else:  # controls added while this browser session was open start at their defaults
        for k, v in defaults.items():
            st.session_state[_VALUES].setdefault(k, v)
    st.session_state[_EFFECTIVE] = {}
    st.session_state.setdefault("link_overrides", {})
    st.session_state.setdefault("ls_views", {})
    st.session_state.setdefault("bks_fit_keys", set())


def _sync(key: str) -> None:
    st.session_state[_VALUES][key] = st.session_state[key]


def _reset_settings() -> None:
    st.session_state[_VALUES] = {**_ui.default_values(), **MAIN_DEFAULTS}
    st.session_state["link_overrides"] = {}
    st.session_state["ls_views"] = {}


def _request_bks() -> None:
    st.session_state["bks_requested"] = True


def _request_compare_bks(methods: tuple[str, ...]) -> None:
    """The Compare tab's Run BKS: fit these BKS-implied variants on the next run (D88)."""
    st.session_state["bks_compare_requested"] = tuple(methods)


def control(kind: str, label: str, key: str, container: Any = None, *, fallback: Any = None, **kw: Any) -> Any:
    """A widget whose value persists in ``st.session_state[_VALUES]`` while hidden or re-ranged.

    The stored value is the user's own choice. The widget shows it clamped to
    the current range or options (``fallback``, else the first option, when it
    is no longer offered), but the stored choice is not overwritten by the
    clamp, so it comes back when the range allows it again. The ``on_change``
    callback copies a user change into the stored values; the value in effect
    this run is recorded in ``st.session_state[_EFFECTIVE]`` and returned.
    """
    container = st if container is None else container
    values = st.session_state[_VALUES]
    value = values.get(key, fallback)
    if kind in ("slider", "number_input", "date_input"):
        lo, hi = kw.get("min_value"), kw.get("max_value")
        if value is None:
            value = lo
        if lo is not None and value < lo:
            value = lo
        if hi is not None and value > hi:
            value = hi
    elif kind in ("selectbox", "radio", "select_slider"):
        options = list(kw["options"])
        if value not in options:
            value = fallback if fallback in options else options[0]
    elif kind == "multiselect":
        options = set(kw["options"])
        value = [x for x in (value or []) if x in options]
    elif kind in ("checkbox", "toggle"):
        value = bool(value)
    st.session_state[key] = value
    out = getattr(container, kind)(label, key=key, on_change=_sync, args=(key,), **kw)
    st.session_state[_EFFECTIVE][key] = out
    if key not in values:
        values[key] = out
    return out


def show_chart(fig: Any, key: str) -> None:
    """Plotly figure at container width with the chart module's own colours (``theme=None``)."""
    st.plotly_chart(fig, theme=None, width="stretch", key=key)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar(ref_assets: pd.DataFrame | None, on_simulation: bool = True) -> dict[str, Any]:
    """All sidebar controls, shared by every page.

    Returns the values in effect, the training window they give
    (:func:`_ui.training_window`) and the placeholder for the feasibility note
    that the simulation page fills after its run. With ``on_simulation=False``
    (the Real data page) the Run BKS button is disabled, because BKS runs on
    the Simulation lab and BKS trace pages only (D90).
    """
    sb = st.sidebar
    sb.header("Settings")
    values = st.session_state[_VALUES]

    with sb.expander("Universe", expanded=True):
        n_listed = len(ref_assets) if ref_assets is not None else 55
        src = control(
            "radio", "Assets", "sb_asset_source", options=["listed", "generic"], horizontal=True,
            format_func=lambda s: f"Listed ({n_listed})" if s == "listed" else "Generic",
        )
        if src == "listed":
            control(
                "radio", "Price source", "sb_price_source", options=["real", "artificial"], horizontal=True,
                format_func=lambda s: "Real prices" if s == "real" else "Artificial (model returns)",
                help="Real: daily returns from data/market (a failed leg falls back to model returns). "
                "Artificial: every asset from the three-factor GARCH model (DESIGN.md G.2.2).",
            )
            classes = control("multiselect", "Asset classes", "sb_asset_classes", options=list(_ui.ASSET_CLASSES))
            dropped: list[str] = []
            if ref_assets is not None:
                names = ref_assets["name"].to_dict()
                dropped = control(
                    "multiselect", "Leave out assets", "sb_drop_assets", options=list(ref_assets.index),
                    format_func=lambda a: names.get(a, a),
                )
                n_assets = int(sum(1 for a, c in ref_assets["asset_class"].items() if c in classes and a not in dropped))
            else:
                n_assets = n_listed
        else:
            n_assets = control("slider", "Number of generic assets", "sb_n_generic_assets", min_value=2,
                               max_value=500, step=1)
            st.caption("Generic assets use model returns (three-factor GARCH model).")
        control("number_input", "Return generator seed", "sb_universe_seed", min_value=0, max_value=1_000_000, step=1,
                help="Seed of the model returns (artificial and generic assets).")

    with sb.expander("Topics", expanded=True):
        manual = control(
            "selectbox", "Manual topics (report Tables 1-2)", "sb_manual", options=list(_ui.TOPIC_SETS),
            format_func=lambda s: _ui.TOPIC_SET_LABELS[s],
        )
        lo = _ui.MIN_GENERIC_TOPICS_ALONE if manual == "none" else 0
        n_gen = control("slider", "Generic topics", "sb_n_generic_topics", min_value=lo, max_value=500, step=1,
                        help="Unnamed topics G001, G002, ...; 10-500 when no manual set is chosen.")
        if n_gen > 0:
            control("slider", "Share of generic topics with links", "sb_signal_share", min_value=0.0, max_value=1.0,
                    step=0.05, help="The other generic topics are pure noise (placebos).")
        n_manual = N_MANUAL[manual]
        n_topics = n_manual + n_gen
        st.caption(f"L = {n_manual} manual + {n_gen} generic = {n_topics} topics.")

    with sb.expander("Sensitivities", expanded=True):
        n_betas = control(
            "radio", "Number of set sensitivities (betas)", "sb_n_betas", options=[1, 2, 3], horizontal=True,
            help="1: one value for every link. 2: strong links, and moderate plus weak links. 3: one value per tier.",
        )
        names = {1: ["all links"], 2: ["strong links", "moderate and weak links"],
                 3: ["strong links", "moderate links", "weak links"]}[n_betas]
        betas = []
        for i, name in enumerate(names, start=1):
            betas.append(control("slider", f"Set sensitivity {i}: {name}", f"sb_beta_{i}", min_value=0.0,
                                 max_value=0.95, step=0.01))
        w = int(values.get("sb_shock_window", 5))
        acfg = AttentionConfig()
        att = attenuation(acfg.kappa, acfg.slow_ar1, acfg.slow_sd_ratio, w)
        implied = ", ".join(f"beta {i} = {b:.2f} about {_ui.fmt_pct(b * b)}" for i, b in enumerate(betas, 1))
        st.caption(
            "Standardised units, before feasibility scaling:\n\n"
            "- a topic linked to one asset with set sensitivity beta has shocks correlated about beta with its "
            "return;\n"
            f"- it then explains about beta² of the asset's variance ({implied});\n"
            f"- the observed shock is attenuated by about {att:.2f} at w = {w}."
        )
        feas_slot = st.empty()
        control(
            "radio", "Lead", "sb_lead", options=[0, 1], horizontal=True,
            format_func=lambda x: "Same day" if x == 0 else "Next day",
            help="Same day: attention on day t relates to returns on day t (risk reading). "
            "Next day: attention on day t relates to returns on day t+1 (signal reading).",
        )
        control(
            "selectbox", "Topic noise tails", "sb_noise_df", options=list(_ui.NOISE_DFS),
            format_func=lambda d: "Gaussian" if d == 0 else f"Student-t, {d:g} degrees of freedom",
        )
        control("number_input", "Link seed", "sb_exposure_seed", min_value=0, max_value=1_000_000, step=1,
                help="Redraws the random links (generic topics, or manual topics on generic assets). The default "
                "link map of the listed assets does not change.")
        control("number_input", "Noise seed", "sb_noise_seed", min_value=0, max_value=1_000_000, step=1,
                help="Redraws the topics' news noise and slow attention component with the links held fixed.")
        n_edits = len(st.session_state["link_overrides"])
        if n_edits:
            st.caption(f"{n_edits} session link edit(s) active (Lists tab).")

    train_window = None
    with sb.expander("Time windows", expanded=True):
        t_min = _ui.earliest_train_end(int(values.get("sb_shock_window", 5) or 5))
        t_end = control("date_input", "Training end (cut-off)", "sb_train_end", min_value=t_min, max_value=_D1,
                        format="YYYY-MM-DD", help="Last day of the training window. The earliest cut-off leaves "
                        "one month of days with a topic shock.")
        months = control(
            "select_slider", "Training length", "sb_train_months", options=list(_ui.TRAIN_MONTHS),
            format_func=_ui.train_months_label, fallback=_ui.DEFAULT_TRAIN_MONTHS,
            help="How far the training window reaches back from the cut-off, in calendar months.",
        )
        if t_end:
            train_window = _ui.training_window(t_end, months)
            st.caption(train_window["text"])
            short_note = _ui.short_training_note(
                train_window["start"], train_window["end"], n_topics,
                bks_cfg=BKSLabConfig(history=values.get("sb_bks_history", "full")),
                lead_days=int(values.get("sb_lead", 0) or 0), shock_window=int(values.get("sb_shock_window", 5) or 5),
            )
            if short_note:
                st.caption(short_note)
        control("date_input", "Forecast start", "sb_forecast_start", min_value=_D0, max_value=_D1,
                format="YYYY-MM-DD")
        control("slider", "Forecast length (weeks)", "sb_forecast_weeks", min_value=1, max_value=12, step=1)
        control("selectbox", "Shock window w (days)", "sb_shock_window", options=list(_ui.SHOCK_WINDOWS),
                help="The shock is attention minus its mean over the previous w trading days.")

    with sb.expander("Direct estimator", expanded=False):
        method = control("selectbox", "Method", "sb_method", options=list(_ui.METHODS),
                         format_func=lambda m: _ui.METHOD_LABELS[m])
        if method == "elastic_net":
            penalty = control("selectbox", "Penalty rule", "sb_penalty", options=["universal", "fixed", "cv"],
                              format_func=lambda p: _ui.PENALTY_LABELS[p])
            if penalty == "fixed":
                control("number_input", "Penalty alpha", "sb_alpha", min_value=0.001, max_value=5.0, step=0.005,
                        format="%.3f")
            if penalty == "cv":
                est = _ui.cv_seconds_estimate(n_topics, n_assets)
                if est > 5.0:
                    st.warning(
                        f"Cross-validation fits each asset separately: about {est:.0f} s for {n_topics} topics and "
                        f"{n_assets} assets, repeated after every change to the universe, topics, sensitivities "
                        "or training window."
                    )
            control("slider", "L1 ratio", "sb_l1_ratio", min_value=0.05, max_value=1.0, step=0.05)
        elif method == "ridge":
            gcv = control("checkbox", "Choose lambda by generalised cross-validation", "sb_ridge_gcv")
            if not gcv:
                control("number_input", "Ridge lambda", "sb_ridge_lambda", min_value=0.0, max_value=100.0,
                        step=0.01, format="%.4f")
        control("slider", "Selection threshold tau", "sb_select_tau", min_value=0.0, max_value=0.3, step=0.01,
                help="Dense methods select a pair when |b| >= tau; recovery counts a pair as truly sensitive "
                "when its true sensitivity is at least tau in absolute value.")

    with sb.expander("BKS model", expanded=False):
        history = control(
            "radio", "Covariance history", "sb_bks_history", options=list(_ui.BKS_HISTORIES),
            format_func=lambda h: _ui.BKS_HISTORY_LABELS[h],
            help="Full history: the instruments are kernel covariances over every day before the cut-off, as in "
            "BKS. Training window only: the instruments and return scales use the training window alone, so BKS "
            "sees the data the direct methods see. Drives the BKS tab and this Run BKS button; the Compare "
            "methods tab shows both.",
        )
        control("slider", "Factors K", "sb_bks_K", min_value=1, max_value=6, step=1,
                help="Must be below the number of assets.")
        hl = control("number_input", "Kernel half-life (months)", "sb_bks_half_life", min_value=3.0, max_value=240.0,
                     step=1.0, format="%.0f")
        st.caption(f"Weekly decay xi = {BKSLabConfig(half_life_months=float(hl)).xi_weekly:.4f}.")
        rule = control("radio", "Lambda rule", "sb_bks_rule", options=["tolerance", "argmax", "fixed"],
                       format_func=lambda r: _ui.LAMBDA_RULE_LABELS[r])
        if rule == "tolerance":
            control("slider", "Tolerance", "sb_bks_tolerance", min_value=0.0, max_value=0.1, step=0.005,
                    format="%.3f", help="Sparsest grid point within this share of the best in-sample Sharpe ratio.")
        if rule == "fixed":
            control("number_input", "Lambda", "sb_bks_lam", min_value=0.0, max_value=100.0, step=0.001,
                    format="%.4f", help="0 fits plain IPCA (no penalty).")
            store = st.session_state.get("bks_store")
            if store is not None:
                st.caption(f"lambda_max of the last run: {store.get('lam_max', float('nan')):.4g}.")
        control("slider", "Grid points", "sb_bks_n_lambdas", min_value=4, max_value=30, step=1)
        control("select_slider", "Grid ratio (smallest / largest lambda)", "sb_bks_ratio",
                options=list(_ui.LAMBDA_RATIOS), format_func=lambda r: f"{r:g}")
        control("checkbox", "Penalise the intercept", "sb_bks_pen_int")
        # the same check as the BKS tab's button: a window BKS cannot use disables the button (D84, D87)
        blocked = train_window is not None and not _ui.bks_training_check(
            train_window["start"], train_window["end"], BKSLabConfig(history=history),
            int(values.get("sb_lead", 0) or 0), int(values.get("sb_shock_window", 5) or 5),
        )["can_run"]
        st.button("Run BKS", key="sb_run_bks", type="primary", on_click=_request_bks, width="stretch",
                  disabled=not on_simulation or blocked,
                  help="BKS runs on the Simulation lab and BKS trace pages." if not on_simulation
                  else "Change the training window first." if blocked else None)
    sb.button("Reset all settings", key="sb_reset", on_click=_reset_settings)
    return {"values": {**values, **st.session_state[_EFFECTIVE]}, "train_window": train_window,
            "feas_slot": feas_slot}


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def overview_tab(ctx: dict[str, Any]) -> None:
    ev, truth, fit, sim, sweep = ctx["ev"], ctx["truth"], ctx["fit"], ctx["sim"], ctx["sweep"]
    rec = ev.recovery
    tiles = [
        ("Median OOS R², estimator", _ui.fmt_pct(_ui.median_finite(ev.r2)),
         "Median over assets of 1 - sum (r - rhat)² / sum r² in the forecast window (uncentered, topics only)."),
        ("Median OOS R², oracle", _ui.fmt_pct(_ui.median_finite(ev.r2_oracle)),
         "The same with the true sensitivities and the estimator's training scales: what the estimator would "
         "reach if it recovered the true sensitivities exactly. No forecast-window data enters it."),
        ("Median population R²", _ui.fmt_pct(_ui.median_finite(truth.r2_true)),
         "Share of each asset's variance the topics explain in the simulation (G.5.3)."),
        ("Assets with positive OOS R²", _ui.fmt_pct(_ui.share_positive(ev.r2), 0),
         "Share of assets whose topic-explained return beats the zero forecast in the window."),
        ("Coverage", _ui.fmt_pct(rec.get("coverage"), 0), "Share of design-linked pairs the estimator selected."),
        ("Sign agreement", _ui.fmt_pct(rec.get("sign_agreement"), 0),
         "Share of linked and selected pairs whose estimated sign equals the true sign."),
        ("MCC", _ui.fmt_num(rec.get("mcc")),
         "Matthews correlation of 'selected' against 'truly sensitive' (true sensitivity at least tau in "
         "absolute value), over all pairs."),
        ("Spearman", _ui.fmt_num(rec.get("spearman")),
         "Rank correlation of estimated and true sensitivities, all pairs."),
    ]
    for row in (tiles[:4], tiles[4:]):
        for col, (label, value, help_) in zip(st.columns(4), row):
            col.metric(label, value, help=help_, border=True)
    st.caption(_ui.how_to_read(*_ui.HOW_OVERVIEW_TILES))

    note = _ui.linked_asset_note(ev, truth)
    if note:
        st.caption(note)
        st.caption(_ui.how_to_read(*_ui.HOW_LINKED_NOTE))
    feas = ctx["feasibility"]
    if feas["text"]:
        st.warning(feas["text"])
    clipped = float(sim.meta.get("clipped_share", 0.0) or 0.0)
    if clipped > 1e-3:
        st.warning(f"Attention was clipped at 1e-6 on {clipped:.2%} of topic-days; the truth ignores the clipping.")
    elif clipped > 0:
        st.caption(f"Attention clipped at 1e-6 on {clipped:.4%} of topic-days (negligible).")
    a_labels = ctx["a_labels"]
    failed = [a_labels.get(a, a) for a in ctx["market"].meta.get("failed_assets", [])]
    if failed:
        st.info(f"{len(failed)} listed asset(s) use model returns because a leg failed: {', '.join(failed)}.")
    skipped = [a_labels.get(a, a) for a in fit.meta.get("skipped_assets", [])]
    if skipped:
        st.warning(f"{len(skipped)} asset(s) have too few training days and were not fitted: {', '.join(skipped)}.")
    n_conv = int(fit.meta.get("n_convergence_warnings", 0) or 0)
    if n_conv:
        st.caption(f"The elastic net did not fully converge for {n_conv} asset group(s).")
    if fit.meta.get("gcv_at_boundary"):
        low = int(fit.meta.get("gcv_at_lower_edge", 0) or 0)
        high = int(fit.meta["gcv_at_boundary"]) - low
        parts = []
        if low:
            parts.append(f"the smallest lambda of its grid for {_ui.plural(low, 'asset')} (close to OLS)")
        if high:
            parts.append(f"the largest for {_ui.plural(high, 'asset')} (sensitivities shrunk towards zero)")
        st.caption(f"Ridge GCV chose {' and '.join(parts)}.")

    c1, c2 = st.columns([1.05, 1])
    with c1:
        r2, r2o, r2t = ev.r2, ev.r2_oracle, truth.r2_true
        if len(r2) > 100:
            top = r2.sort_values(ascending=False, na_position="last").index[:100]
            r2, r2o, r2t = r2.reindex(top), r2o.reindex(top), r2t.reindex(top)
            st.caption(f"Showing the 100 assets with the highest OOS R² of {len(ev.r2)}.")
        show_chart(charts.r2_bars(r2, r2o, r2t, labels=ctx["a_labels"]), "fig_r2")
        st.caption(_ui.how_r2_bars("estimator"))
    with c2:
        weeks = ctx["cfg"].window.forecast_weeks
        caption = _ui.sweep_caption(sweep, weeks, ev.n_days)
        show_chart(charts.window_sweep_chart(sweep, empty_message=f"No complete {weeks}-week window in the data"),
                   "fig_sweep")
        st.caption(caption)
        st.caption(_ui.how_to_read(*_ui.HOW_SWEEP))
        linked = truth.W_unscaled != 0
        est, n_total = _ui.subsample_pairs(fit.B_hat, linked, max_points=20000)
        show_chart(charts.exposure_scatter(est, truth.B_true, linked=linked), "fig_scatter")
        if est.size and n_total > int(np.isfinite(est.to_numpy()).sum()):
            st.caption(f"Showing all linked pairs and a seeded sample of the others ({n_total} pairs in total).")
        st.caption(_ui.how_to_read(*_ui.HOW_SCATTER))

    with st.expander("Stage timings of this page"):
        st.dataframe(_ui.timings_frame(ctx["timings"]), hide_index=True, width="content")


def exposure_tab(ctx: dict[str, Any]) -> None:
    ev, fit, truth, sim, cfg = ctx["ev"], ctx["fit"], ctx["truth"], ctx["sim"], ctx["cfg"]
    assets = ctx["market"].assets
    days = ev.return_days
    if len(days):
        st.markdown(
            f"**Forecast window** {days[0].date()} to {days[-1].date()} · **{ev.n_days} return days** · "
            f"sensitivities fitted on {cfg.window.train_start} to {cfg.window.train_end} (out of sample)."
        )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric = control("selectbox", "Cell metric", "ex_metric", options=list(_ui.METRICS))
        units = _ui.EXPOSURE_UNITS[0]
        if metric in _ui.EXPOSURE_METRICS:
            units = control("radio", "Units", "ex_units", options=list(_ui.EXPOSURE_UNITS), horizontal=True)
    tau = float(cfg.direct.select_tau)
    with c2:
        # the blank rule follows the view (owner request 2026-09-30, D69 amendment)
        blank_on = control("checkbox", _ui.blank_rule(metric, tau)["label"], "ex_blank_rule")
        threshold = control("slider", "Blank cells with |value| below", "ex_threshold", min_value=0.0, max_value=1.0,
                            step=0.01)
    with c3:
        max_cols = control("slider", "Maximum columns", "ex_max_cols", min_value=10, max_value=100, step=1)
        row_mode = control("selectbox", "Row order", "ex_row_order", options=list(_ui.ROW_ORDERS))
    with c4:
        show_text = control("checkbox", "Show values in cells", "ex_show_text")
        colors = control("selectbox", "Colours", "ex_colors", options=list(_ui.HEATMAP_COLORS))
        ls_view = control("checkbox", "Long/short view", "ex_ls_view",
                          help="Flip the sign of the rows you mark S and prefix the row label with L or S.")
        max_rows = None
        if len(assets) > _ui.HEATMAP_DEFAULT_MAX_ROWS:
            max_rows = control("slider", "Maximum rows", "ex_max_rows", min_value=10, max_value=len(assets), step=1)

    views = None
    if ls_view:
        stored = st.session_state["ls_views"]
        ids = [str(a) for a in assets.index]
        base = pd.DataFrame(
            {"Asset": [ctx["a_labels"].get(a, a) for a in ids], "View": [stored.get(a, "L") for a in ids]}, index=ids
        )
        with st.expander("Long/short view per asset", expanded=True):
            edited = st.data_editor(
                base,
                key=f"ex_ls_editor_{cfg.key('market')}",
                hide_index=True,
                height=min(400, 36 + 35 * len(ids)),
                column_config={
                    "Asset": st.column_config.TextColumn("Asset", disabled=True),
                    "View": st.column_config.SelectboxColumn("View", options=["L", "S"], required=True),
                },
            )
        for a, v in edited["View"].items():
            stored[str(a)] = str(v) if v in ("L", "S") else "L"
        views = pd.Series({a: stored.get(a, "L") for a in ids})

    tbl = _ui.exposure_table(
        metric, units, ev, fit, truth, assets, sim.topics.table,
        blank_rule_on=bool(blank_on), threshold=float(threshold), row_mode=row_mode, views=views,
        max_rows=max_rows, tau=tau,
    )
    fig = charts.exposure_heatmap(
        tbl["values"],
        blank=tbl["blank"],
        value_label=tbl["value_label"],
        row_labels=ctx["a_labels"],
        col_labels=ctx["t_labels"],
        row_prefix=tbl["prefix"],
        average_row=True,
        max_cols=int(max_cols),
        show_text=None if show_text else False,
        subtitle=tbl["subtitle"],
        colorscale=_ui.HEATMAP_COLORS[colors],
        row_title="Asset (key view)" if ls_view else "Asset",
        col_title="Topics",
    )
    show_chart(fig, "fig_exposure")
    st.caption(_ui.how_cell_metric(metric))
    st.caption(_ui.how_to_read(*_ui.HOW_TABLE))
    csv = tbl["values"].rename(index=ctx["a_labels"]).to_csv().encode("utf-8")
    st.download_button("Download the table (CSV)", data=csv, file_name="sensitivity_table.csv", mime="text/csv",
                       key="ex_download", on_click="ignore")


def contributions_tab(ctx: dict[str, Any]) -> None:
    ev, sim = ctx["ev"], ctx["sim"]
    ids = [str(a) for a in ctx["market"].assets.index]
    a_labels, t_labels = ctx["a_labels"], ctx["t_labels"]
    c1, c2, c3 = st.columns([1.4, 1.2, 1])
    with c1:
        asset = control("selectbox", "Asset", "tc_asset", options=ids, format_func=lambda a: a_labels.get(a, a),
                        fallback=_ui.default_asset(ids))
    with c2:
        view = control("radio", "View", "tc_view", options=["Variance share", "Return attribution"], horizontal=True,
                       help="Variance share: each topic's share of the window's day-to-day return variation. "
                       "Return attribution: each topic's part of the window's cumulative move.")
        rollup = control("checkbox", "Roll up by topic group (Sector / Macro / Micro / Generic)", "tc_rollup")
    with c3:
        top_n = control("slider", "Topics shown", "tc_top_n", min_value=5, max_value=30, step=1)

    name = a_labels.get(asset, asset)
    days = ev.return_days
    period = f"{days[0].date()} to {days[-1].date()}" if len(days) else "no return days"
    m = st.columns(4)
    if view == "Return attribution":
        m[0].metric("Realised move", _ui.fmt_pts(ev.realized.get(asset)), border=True,
                    help="Sum of the asset's daily returns over the window.")
        m[1].metric("Explained by topics", _ui.fmt_pts(ev.explained.get(asset)), border=True,
                    help="Sum of the topic contributions (estimated sensitivities).")
        m[2].metric("Not explained by topics", _ui.fmt_pts(ev.residual.get(asset)), border=True,
                    help="Realised move minus the explained move.")
    else:
        shares = ev.var_share.loc[asset].astype(float)
        shares_true = ev.var_share_true.loc[asset].astype(float)
        # "none" when every share is zero (the elastic net selected no topic for this asset), "n/a" without data
        top = shares.abs().idxmax() if (shares.abs() > 0).any() else None
        largest = f"{top} · {_ui.fmt_pct(shares.get(top))}" if top is not None else (
            "none" if shares.notna().any() else "n/a")
        m[0].metric("Share explained by topics", _ui.fmt_pct(shares.sum(min_count=1)), border=True,
                    help="Share of the window's return variation that co-moves with the topics; the bars sum to it.")
        m[1].metric("True share (simulation)", _ui.fmt_pct(shares_true.sum(min_count=1)), border=True,
                    help="The same share with the simulation's true sensitivities.")
        m[2].metric("Largest topic", largest, border=True,
                    help=f"{t_labels.get(top, top)}" if top is not None else None)
    m[3].metric("OOS R², estimator / oracle",
                f"{_ui.fmt_pct(ev.r2.get(asset))} / {_ui.fmt_pct(ev.r2_oracle.get(asset))}", border=True)
    st.caption(_ui.how_to_read(*(_ui.HOW_TILES_ATTRIBUTION if view == "Return attribution" else _ui.HOW_TILES_SHARE)))

    if view == "Return attribution":
        contrib, true = ev.contrib.loc[asset], ev.contrib_true.loc[asset]
        realized, residual = float(ev.realized.get(asset)), float(ev.residual.get(asset))
        title = f"Topic contributions to {name}, {period}"
        subtitle = "Percentage points of return; topics on top, the unexplained part and the realised move below."
        chart_kw = {"units": "pp", "realized_label": "Realised move", "axis_title": "Contribution over the window (pp)"}
    else:
        contrib, true = ev.var_share.loc[asset], ev.var_share_true.loc[asset]
        realized = 1.0 if np.isfinite(contrib.to_numpy(dtype=float)).any() else float("nan")
        residual = realized - float(contrib.sum())
        title = f"Share of {name}'s return variation by topic, {period}"
        subtitle = "Percent of the window's uncentered variation (sum of r²) that co-moves with each topic."
        chart_kw = {"units": "%", "realized_label": "Total variation (100%)",
                    "axis_title": "Share of the window's return variation (%)"}
    labels: Any = t_labels
    if rollup:
        contrib = _ui.rollup_by_group(contrib, sim.topics.table)
        true = _ui.rollup_by_group(true, sim.topics.table)
        labels = None
    fig = charts.contribution_bars(contrib, true_contrib=true, realized=realized, residual=residual,
                                   top_n=int(top_n), labels=labels, title=title, subtitle=subtitle, **chart_kw)
    left, right = st.columns([1.15, 1])
    with left:
        show_chart(fig, "fig_contrib")
        st.caption(_ui.how_to_read(*_ui.HOW_CONTRIB_BARS))
    with right:
        show_chart(
            charts.cumulative_explained(
                ev.realized_daily[asset], ev.fitted[asset], ev.fitted_oracle[asset],
                title=f"Cumulative realised vs topic-explained return: {name}",
            ),
            "fig_cumulative",
        )
        st.caption(_ui.how_to_read(*_ui.HOW_CUMULATIVE))
    st.caption(_ui.how_to_read(*_ui.HOW_TWO_VIEWS))
    if rollup:
        st.caption(_ui.how_to_read(*_ui.HOW_ROLLUP))
    with st.expander("Why this method"):
        st.markdown(
            "The lab uses the out-of-sample linear regression proposed for this question: sensitivities fitted on "
            "the training window, frozen, and applied to the window's shocks.\n\n"
            "1. **A regression inside the window** is rejected. It is in-sample for the window and has 5 to 60 "
            "daily observations against 9 to 520 topics, so it is either not identified or fits noise.\n"
            "2. **One-topic-at-a-time regressions** double count correlated topics and do not add up to the move.\n"
            "3. **Shapley or LMG decompositions** of R² are order-free. They answer a variance question rather than "
            "what moved the asset, and they are combinatorial, so 500 topics need sampling.\n"
            "4. **The variance share** (the default view) is the per-day alternative: it shows which topics "
            "co-move with the returns through the window. Use the return attribution for the window's net move, "
            "keeping in mind that summed shocks cancel.\n"
            "5. **The BKS per-topic split** is not identified (D52); the BKS tab shows it with that caveat."
        )

    with st.expander("Simulated attention of the largest contributors"):
        c = ev.contrib.loc[asset].abs().sort_values(ascending=False)
        top = [t for t in c.index[:3] if np.isfinite(c[t]) and c[t] > 0]
        shocks = ctx["shocks"].s_hat
        if len(days) and not top:  # e.g. the elastic net selected no topic for this asset
            st.info("No topic contributes to this asset's move in the window (every estimated sensitivity of the "
                    "asset is zero), so there is no attention to show."
                    if np.isfinite(c.to_numpy(dtype=float)).any() else "No contributions to show for this asset.")
        elif len(days):
            lo, hi = days[0] - pd.Timedelta(weeks=26), days[-1] + pd.Timedelta(weeks=8)
            att = sim.attention.loc[lo:hi, top]
            sh = shocks.loc[lo:hi, top]
            show_chart(
                charts.attention_chart(att, sh, window=(days[0], days[-1]), train_end=pd.Timestamp(
                    ctx["cfg"].window.train_end), labels=t_labels),
                "fig_attention",
            )
            st.caption(_ui.how_to_read(*_ui.HOW_ATTENTION))


def _bks_implied_reason(ctx: dict[str, Any], lib_reason: str, method: str = IMPLIED_METHOD) -> tuple[str, bool]:
    """Why a BKS-implied variant is not available, and whether a BKS run can help (D88)."""
    cfg = lab_compare.method_config(ctx["cfg"], method)
    w = cfg.window
    check = _ui.bks_training_check(w.train_start, w.train_end, cfg.bks, cfg.exposure.lead_days, w.shock_window)
    if not check["can_run"]:
        return check["reason"], False
    session = ctx["session"]
    fit_err = st.session_state.get("bks_fit_errors", {}).get(session.stage_key("bks_fit", cfg))
    if fit_err:
        return f"BKS could not run with these settings: {fit_err}", True
    err = st.session_state.get("bks_error")
    if err and err[0] == session.stage_key("bks", cfg):
        return f"BKS could not run with these settings: {err[1]}", True
    if lib_reason and lib_reason not in (BKS_NOT_RUN, BKS_OFF):  # a BKS fit that does not match these settings
        return lib_reason, True
    return ("BKS has not been run on the current settings (training window, BKS model and covariance history). "
            "Press Run BKS; the comparison never starts a BKS fit on its own."), True


def _bks_unavailable_box(ctx: dict[str, Any], lib_reasons: dict[str, str], key: str) -> dict[str, str]:
    """Info box with the reasons the BKS-implied variants are not available, and one Run BKS button.

    The button fits every listed variant that can run (D88). Returns method -> reason.
    """
    reasons: dict[str, str] = {}
    runnable: list[str] = []
    lines = []
    for m, lib_reason in lib_reasons.items():
        reason, can_run = _bks_implied_reason(ctx, lib_reason, m)
        reasons[m] = reason
        lines.append(f"- {lab_compare.METHOD_LABELS.get(m, m)}: {reason}")
        if can_run:
            runnable.append(m)
    st.info("BKS-implied is not available.\n\n" + "\n".join(lines))
    st.button("Run BKS", key=key, type="primary", on_click=_request_compare_bks, args=(tuple(runnable),),
              disabled=not runnable, help=None if runnable else "Change the training window first.")
    return reasons


def follow_control(label: str, key: str, options: list[Any], default: Any, format_func: Any = str,
                   container: Any = None, **kw: Any) -> Any:
    """A selectbox that follows ``default`` until the user picks an option (stored ``None`` means "follow").

    The user's choice is kept in ``st.session_state[_VALUES]`` like :func:`control`; a stored value that is no
    longer among ``options`` falls back to ``default`` (else the first option) without being overwritten.
    """
    container = st if container is None else container
    stored = st.session_state[_VALUES].get(key)
    value = stored if stored in options else (default if default in options else options[0])
    st.session_state[key] = value
    out = container.selectbox(label, options, key=key, format_func=format_func, on_change=_sync, args=(key,), **kw)
    st.session_state[_EFFECTIVE][key] = out
    return out


def _inspect_control(options: list[str], default: str, format_func: Any) -> str:
    """"Method to inspect": follows the sidebar's direct method until the user picks another one."""
    return follow_control("Method to inspect", "cm_inspect", options, default, format_func)


def _switch_to_trace(history: str) -> None:
    """Compare tab: trace a BKS-implied variant whose covariance history the sidebar does not show (D88, D90).

    Sets the sidebar's "Covariance history" to ``history`` and opens the BKS trace page. The variant's fit key is
    already in this browser session's fit keys (the Compare tab ran it), so the trace uses the cached fit; if the
    fit has been evicted since (two fits are kept), the trace page says so and its Run BKS fits it again. The
    Compare tab says so under the button when the fit is no longer cached.
    """
    st.session_state[_VALUES]["sb_bks_history"] = history
    st.switch_page(PAGES["trace"])


def trace_link(label: str) -> None:
    """Link to the BKS trace page (D90); nothing when the page is not registered in this run."""
    if "trace" in PAGES:
        st.page_link(PAGES["trace"], label=label, icon=":material/troubleshoot:")


def compare_tab(ctx: dict[str, Any]) -> None:
    """Compare methods (G.15, D83): all methods on the same forecast days, sensitivities frozen at the cut-off."""
    cfg, session, truth = ctx["cfg"], ctx["session"], ctx["truth"]
    a_labels = ctx["a_labels"]
    w = cfg.window
    full_bks = lab_compare.method_config(cfg, IMPLIED_METHOD).bks
    share = lab_bks.kernel_history_share(w.train_start, w.train_end, full_bks, cfg.exposure.lead_days,
                                         shock_window=w.shock_window)
    history = (f" On this window about {share:.0%} of that weight lies before the training start."
               if np.isfinite(share) else "")
    store = st.session_state.get("bks_store")
    bks_r2 = ""
    if store is not None and store["key"] == ctx["bks_key"]:
        bks_r2 = f" (here {_ui.fmt_pct(store['result'].r2_pooled)})"
    ex = _ui.COMPARE_LEAD_EXAMPLES
    st.caption(_ui.how_to_read(
        "How to read the comparison:", (
            ("Every method is scored on the same forecast days, with its sensitivities frozen at the training end. "
             "The direct methods are fitted on the training window only.", ex["direct"]),
            ("BKS enters through its implied sensitivities: the topic-asset covariances that the BKS fit implies "
             "for each asset, turned into sensitivities with the training covariance of the topic shocks. Both BKS "
             "variants use the sidebar's BKS model and differ in the covariance history and the return scaling.",
             ex["bks"]),
            ("BKS-implied (full history) sees more past data than the direct methods. Its Gamma and scales use the "
             "training window, but its instruments weigh all days before the cut-off (half-life "
             f"{cfg.bks.half_life_months:g} months).{history}", ex["full"]),
            ("BKS-implied (training window) sees only the training window, as the direct methods do: its "
             "instruments and return scales start at the training start. It is the like-for-like BKS figure.",
             ex["training"]),
            (f"The BKS tab's OOS R²{bks_r2} fits K factors to each forecast week's own returns, so it cannot be set "
             "next to the direct methods. The BKS-implied rows here are the comparable figures.", ex["bks_tab"]),
            ("The oracle uses the true sensitivities of the simulation. It is the reference, not an estimator.",
             ex["oracle"]),
        ),
    ))
    options = list(lab_compare.METHODS)

    def label_of(m: str) -> str:
        return _ui.method_option_label(m, cfg.direct)

    chosen = control("multiselect", "Methods", "cm_methods", options=options, format_func=label_of)
    st.caption(
        "The sidebar's direct method keeps its settings; the other direct methods use their defaults (elastic net "
        "with the universal penalty, ridge with lambda chosen by generalised cross-validation). All methods share "
        f"the threshold tau = {cfg.direct.select_tau:g}: the elastic net selects every non-zero estimate, the other "
        "methods every estimate of at least tau in absolute value, and tau also marks the truly sensitive pairs."
    )
    if not chosen:
        st.info("Choose at least one method.")
        return
    methods = tuple(m for m in options if m in chosen)
    # D80: a BKS fit is reused only when this browser session requested it for these settings (per variant, D88)
    use_bks = tuple(
        m for m in lab_compare.BKS_METHODS
        if session.stage_key("bks_fit", lab_compare.method_config(cfg, m)) in st.session_state["bks_fit_keys"]
    )
    with st.spinner("Comparing methods ..."):
        res = session.comparison(cfg, methods=methods, use_bks=use_bks)

    notes: dict[str, str] = {}
    missing = {m: str(res.meta.get("unavailable", {}).get(m, "")) for m in methods
               if m in lab_compare.BKS_METHODS and m not in res.fits}
    if missing:
        notes.update(_bks_unavailable_box(ctx, missing, "cm_run_bks"))

    table = _ui.comparison_table(res.summary, notes)
    column_config: dict[str, Any] = {
        head: st.column_config.NumberColumn(head, format=fmt) for head, fmt in _ui.comparison_column_formats().items()
    }
    column_config["Note"] = st.column_config.TextColumn("Note", width="medium")
    st.dataframe(table, hide_index=True, column_config=column_config, width="stretch", placeholder="–")
    weeks = cfg.window.forecast_weeks
    n_win = int(res.meta.get("n_windows", 0))
    windows = (f"the {_ui.plural(n_win, f'consecutive {weeks}-week window')}" if n_win
               else f"the consecutive {weeks}-week windows (none fits here)")
    st.caption(_ui.how_compare_table(windows))
    if any(m in lab_compare.BKS_METHODS for m in methods):
        st.caption(WHY_BKS_LOWER_CAPTION)

    labels = {str(m): str(res.summary.loc[m, "label"]) for m in res.summary.index}
    if not res.fits:
        st.info("None of the chosen methods is available; the table gives the reasons.")
        return
    c1, c2 = st.columns([1.05, 1])
    with c1:
        r2 = res.r2
        ref = "oracle" if "oracle" in r2.columns else None
        if len(r2) > 100 and r2.shape[1]:
            key = r2[ref] if ref else r2.iloc[:, 0]
            r2 = r2.reindex(key.sort_values(ascending=False, na_position="last").index[:100])
            st.caption(f"Showing the 100 assets with the highest OOS R² of the {'oracle' if ref else 'first method'} "
                       f"({len(res.r2)} assets).")
        show_chart(
            charts.method_r2_dots(r2, labels=a_labels, method_labels=labels, clip=-0.5, slots=lab_compare.METHODS,
                                  title="Out-of-sample R² per asset by method, this window"),
            "fig_cm_r2",
        )
        st.caption(_ui.how_compare_dots(ref is not None))
    with c2:
        show_chart(
            charts.method_sweep_lines(res.r2_sweep, method_labels=labels, slots=lab_compare.METHODS,
                                      empty_message=f"No complete {weeks}-week window in the data"),
            "fig_cm_sweep",
        )
        first, last = res.meta.get("sweep_first_day"), res.meta.get("sweep_last_day")
        span = ""
        if n_win and first is not None and not pd.isna(first):
            span = (f" ({_ui.plural(n_win, f'consecutive {weeks}-week window')} here, from "
                    f"{pd.Timestamp(first).date()} to {pd.Timestamp(last).date()})")
        else:
            st.caption(f"No complete {weeks}-week window fits between the forecast start and the end of the data.")
        st.caption(_ui.how_compare_sweep(span, ref is not None))

    st.subheader("Inspect one method")
    inspect_options = [m for m in options if m != "oracle"]  # the oracle is the reference in every inspect chart
    inspect = _inspect_control(inspect_options, cfg.direct.method, label_of)
    one = res if inspect in res.evals else session.comparison(cfg, methods=(inspect,), use_bks=use_bks)
    name = str(one.summary.loc[inspect, "label"]) if inspect in one.summary.index else label_of(inspect)
    if inspect not in one.evals:
        reason = str(one.meta.get("unavailable", {}).get(inspect, "not fitted"))
        if inspect in lab_compare.BKS_METHODS:
            if inspect in methods:
                st.info(f"{name} is not available; see the note above.")
            else:
                _bks_unavailable_box(ctx, {inspect: reason}, "cm_run_bks_inspect")
        else:
            st.info(f"{name} is not available. {_ui.unavailable_note(reason)}")
        return
    fit, ev = one.fits[inspect], one.evals[inspect]
    rec = ev.recovery
    m = st.columns(4)
    m[0].metric("Coverage", _ui.fmt_pct(rec.get("coverage"), 0), border=True,
                help="Share of design-linked pairs the method selected.")
    m[1].metric("Sign agreement", _ui.fmt_pct(rec.get("sign_agreement"), 0), border=True,
                help="Share of linked and selected pairs whose estimated sign equals the true sign.")
    m[2].metric("MCC", _ui.fmt_num(rec.get("mcc")), border=True,
                help="Matthews correlation of 'selected' against 'truly sensitive' (true sensitivity at least tau "
                "in absolute value), over all pairs.")
    m[3].metric("Spearman", _ui.fmt_num(rec.get("spearman")), border=True,
                help="Rank correlation of estimated and true sensitivities, all pairs.")
    st.caption(_ui.how_to_read(*_ui.HOW_COMPARE_TILES))
    if inspect in lab_compare.BKS_METHODS:
        meta = fit.meta
        n_topics = len(fit.B_hat.index)
        K, rank = meta.get("K"), meta.get("gamma_rank")
        used = (f" (only {rank} factor directions are used; the others are numerically zero)"
                if rank is not None and K is not None and int(rank) < int(K) else "")
        hist = str(meta.get("history", lab_compare.BKS_HISTORY[inspect]))
        share = meta.get("kernel_share_before_train")
        share_text = (f"; {float(share):.0%} of the instruments' kernel weight lies before the training start"
                      if share is not None and np.isfinite(float(share)) and float(share) > 0 else "")
        st.caption(
            f"{IMPLIED_NOTE} This fit: K = {K} factors{used}, {meta.get('n_selected_topics')} of {n_topics} "
            f"topics kept, lambda = {float(meta.get('lam', float('nan'))):.3g}; covariance history: "
            f"{_ui.BKS_HISTORY_LABELS.get(hist, hist).lower()}{share_text}."
        )
        st.caption(_ui.how_bks_implied(cfg.bks.half_life_months))
        inspect_history = lab_compare.BKS_HISTORY[inspect]
        if inspect_history == cfg.bks.history:
            trace_link("Trace the BKS-implied sensitivities step by step")
        elif "trace" in PAGES:
            st.button(
                "Trace the BKS-implied sensitivities step by step", key="cm_trace", icon=":material/troubleshoot:",
                on_click=_switch_to_trace, args=(inspect_history,),
                help="Opens the BKS trace page with the sidebar's covariance history set to "
                f"{_ui.BKS_HISTORY_LABELS[inspect_history].lower()}.",
            )
        mcfg = lab_compare.method_config(cfg, inspect)
        if "trace" in PAGES and not (session.has("bks_panel", mcfg) and session.has("bks_fit", mcfg)):
            # the comparison still holds this variant's scores, but its fit left the two-entry fit cache
            st.caption("The BKS fit of this variant is no longer in the cache (it keeps the two most recent fits), "
                       "so the trace page will ask for Run BKS, which fits it again.")
    skipped = [a_labels.get(a, a) for a in fit.meta.get("skipped_assets", [])]
    if skipped:
        st.caption(f"{len(skipped)} asset(s) with too few training days get zero sensitivities: "
                   f"{', '.join(skipped)}.")

    c1, c2 = st.columns(2)
    with c1:
        linked = truth.W_unscaled != 0
        est, n_total = _ui.subsample_pairs(fit.B_hat, linked, max_points=20000)
        show_chart(charts.exposure_scatter(est, truth.B_true, linked=linked,
                                           title=f"Estimated vs true sensitivity: {name}"), "fig_cm_scatter")
        if est.size and n_total > int(np.isfinite(est.to_numpy()).sum()):
            st.caption(f"Showing all linked pairs and a seeded sample of the others ({n_total} pairs in total).")
        st.caption(_ui.how_to_read(*_ui.HOW_COMPARE_SCATTER))
    with c2:
        r2m, r2o, r2t = ev.r2, ev.r2_oracle, truth.r2_true
        if len(r2m) > 100:
            top = r2m.sort_values(ascending=False, na_position="last").index[:100]
            r2m, r2o, r2t = r2m.reindex(top), r2o.reindex(top), r2t.reindex(top)
            st.caption(f"Showing the 100 assets with the highest OOS R² of {len(ev.r2)}.")
        show_chart(charts.r2_bars(r2m, r2o, r2t, labels=a_labels, name=name,
                                  title=f"Out-of-sample R² per asset: {name}"), "fig_cm_r2_inspect")
        st.caption(_ui.how_r2_bars(name, overview=False))


def _bks_runtime_warning(n_topics: int) -> None:
    """The run-time warning above 100 topics, before the Run BKS buttons of the BKS tab and the BKS trace page."""
    text = _ui.bks_runtime_warning(n_topics)
    if text:
        st.warning(text)


def bks_tiles(res: Any, caption_expander: bool = False, caption: tuple[str, Any] = _ui.HOW_BKS_TILES) -> None:
    """The six tiles of a BKS run and their "How to read" caption (BKS tab and BKS trace page).

    ``caption_expander``: the caption goes in a collapsed "How to read the tiles" expander right under the tiles
    (the trace page, where the tiles sit above the step selector on every step; G.9's exception). ``caption``: the
    ``(lead, bullets)`` of the caption; the trace page passes ``_ui.HOW_TRACE_TILES``, whose pointers name its
    step 6.
    """
    m = st.columns(6)
    m[0].metric("Chosen lambda", f"{res.lam:.4g}", border=True)
    m[1].metric("Factors K", f"{res.K}", border=True)
    m[2].metric("Selected topics", f"{len(res.selected_topics)} of {len(res.gamma_norms)}", border=True)
    m[3].metric("In-sample total R²", _ui.fmt_pct(res.in_sample_total_r2), border=True)
    m[4].metric("Pooled OOS R² (weekly)", _ui.fmt_pct(res.r2_pooled), border=True,
                help="Uncentered R² over all asset-weeks of the window, with each week's K factors fitted to that "
                "week's returns.")
    m[5].metric("Same, instruments shuffled", _ui.fmt_pct(res.meta.get("shuffled_r2_pooled")), border=True,
                help="Reference: the topic instruments shuffled across assets within each week (20 shuffles). The "
                "gap to the pooled OOS R² is what the instruments add beyond K freely fitted weekly factors.")
    if caption_expander:
        with st.expander("How to read the tiles"):
            st.caption(_ui.how_to_read(*caption))
    else:
        st.caption(_ui.how_to_read(*caption))


def bks_tab(ctx: dict[str, Any]) -> None:
    cfg = ctx["cfg"]
    b = cfg.bks
    L = len(ctx["sim"].topics.table)
    rule = {"tolerance": f"the {b.tolerance:.1%} tolerance rule", "argmax": "the BKS argmax",
            "fixed": f"a fixed lambda of {b.lam}"}[b.lambda_rule]
    if b.history == "training":
        history = (
            f"- Covariance history: training window only. The instruments start at {cfg.window.train_start}, and "
            "returns are divided by their training standard deviation, so BKS sees the data the direct methods "
            f"see. The first training weeks' instruments cover only a few days (at least {b.min_days_training}).\n"
        )
        weighting = "inverse-volatility asset weighting (training standard deviation)"
    else:
        history = (
            "- Covariance history: full history before the cut-off. The instruments weigh every day since the start "
            f"of the data (the first {b.burn_in_weeks} weeks are a burn-in).\n"
        )
        weighting = "inverse-volatility asset weighting"
    st.caption(
        "Weekly BKS fit on the training weeks, evaluated on the forecast weeks.\n\n"
        f"- Kernel half-life {b.half_life_months:g} months (xi = {b.xi_weekly:.4f}); K = {b.K}; lambda by {rule} on "
        f"a {b.n_lambdas}-point grid; {weighting}.\n"
        f"{history}"
        f"- Fitted on the weeks ending on or before {cfg.window.train_end}; the training Gamma is then frozen.\n"
        "- Each forecast week's K factors are fitted to that week's own returns, so the BKS OOS R² is a "
        "contemporaneous factor fit. The direct estimator fits nothing in the window, so the two R² are not "
        "comparable. The shuffled-instrument reference shows what K weekly factors reach with instruments "
        "unrelated to the assets.\n"
        "- With no topic signal BKS still scores well above zero, because noise topics' instruments inherit the "
        "assets' betas (D47)."
    )
    st.caption(_ui.how_bks_history(b.half_life_months))
    check = _ui.bks_training_check(cfg.window.train_start, cfg.window.train_end, cfg.bks, cfg.exposure.lead_days,
                                   cfg.window.shock_window)
    if not check["can_run"]:
        st.warning(f"{check['reason']} The direct estimator runs on windows down to one month.")
    _bks_runtime_warning(L)
    st.button("Run BKS", key="bks_run_tab", type="primary", on_click=_request_bks, disabled=not check["can_run"],
              help=None if check["can_run"] else "Change the training window first.")
    trace_link("Trace this BKS run step by step")
    err = st.session_state.get("bks_error")
    if err and err[0] == ctx["bks_key"]:
        st.error(f"BKS could not run with these settings: {err[1]}")
    store = st.session_state.get("bks_store")
    if store is None:
        st.info("Press Run BKS (here or in the sidebar) to fit BKS on the current settings. The default settings "
                "take about a second.")
        return
    if store["key"] != ctx["bks_key"]:
        st.warning("Settings changed since this BKS run. The results below are from the earlier settings; press "
                   "Run BKS to update.")
    res = store["result"]
    t_labels = store["t_labels"]
    a_labels = store["a_labels"]
    direct_r2 = ctx["ev"].r2  # always the current direct fit, never a copy stored with the BKS run
    bks_tiles(res)
    span = res.meta.get("evaluated_span")
    days = ctx["ev"].return_days
    span_text = f"BKS scores {span[0].date()} to {span[1].date()}" if span else "BKS span n/a"
    direct_text = f"the direct estimator {days[0].date()} to {days[-1].date()}" if len(days) else "no direct days"
    st.caption(
        f"Median OOS R² per asset: BKS {_ui.fmt_pct(_ui.median_finite(res.r2))} (weekly, per-week factors); "
        f"direct estimator {_ui.fmt_pct(_ui.median_finite(direct_r2))} (daily, current settings). Not the same "
        f"measure. {span_text}, {direct_text}. Run time {store['seconds']:.1f} s; {len(res.periods)} forecast "
        f"week(s); units: {res.meta.get('units', 'return units')}."
    )

    c1, c2 = st.columns(2)
    with c1:
        norms = res.meta.get("gamma_norms_standardized")
        if norms is None or not np.isfinite(norms.to_numpy(dtype=float)).any():
            norms = res.gamma_norms
        show_chart(
            charts.gamma_norm_bars(norms, res.selected_topics, labels=t_labels,
                                   title="BKS Gamma row norms by topic (standardised)"),
            "fig_bks_gamma",
        )
        st.caption(_ui.how_to_read(*_ui.HOW_BKS_GAMMA))
    with c2:
        show_chart(charts.lambda_path_chart(res.path, res.lam, criterion_label="In-sample Sharpe ratio (annualised)"),
                   "fig_bks_path")
        st.caption(_ui.how_to_read(*_ui.HOW_BKS_PATH))
    show_chart(_ui.r2_compare_figure(res.r2, direct_r2, a_labels), "fig_bks_r2")
    st.caption(_ui.how_to_read(*_ui.HOW_BKS_R2))

    ids = [str(a) for a in res.contrib.index]
    asset = control("selectbox", "Asset for the per-topic split", "bks_asset", options=ids,
                    format_func=lambda a: a_labels.get(a, a), fallback=_ui.default_asset(ids))
    split = pd.concat([pd.Series({"const": float(res.const_contrib.get(asset))}), res.contrib.loc[asset]])
    realized = float(res.realized[asset].sum(min_count=1))
    fitted = float(res.fitted[asset].sum(min_count=1))
    fig = charts.contribution_bars(
        split, realized=realized, residual=realized - fitted, units="pp",
        labels={"const": "Constant instrument", **t_labels}, residual_label="Not explained by BKS",
        axis_title="Part of the fitted return over the forecast weeks (pp)",
        title=f"BKS per-topic split of the fitted return: {a_labels.get(asset, asset)}",
        subtitle="Not identified (D52): the split belongs to the sparse representative the lasso chose.",
    )
    show_chart(fig, "fig_bks_split")
    st.caption(_ui.how_to_read(*_ui.HOW_BKS_SPLIT))
    st.caption(D52_NOTE)
    st.caption(D47_NOTE)
    for w in res.meta.get("warnings", [])[1:]:
        st.caption(f"Note: {w}")


def lists_tab(ctx: dict[str, Any]) -> None:
    market, sim, cfg = ctx["market"], ctx["sim"], ctx["cfg"]
    st.subheader("Assets")
    ref_assets, legs = ctx["ref_assets"], ctx["legs"]
    if ref_assets is not None and legs is not None:
        run_assets = market.assets if cfg.universe.asset_source == "listed" else None
        table = _ui.reference_asset_table(ref_assets, legs, run_assets)
        st.dataframe(table, hide_index=True, height=35 * (len(table) + 1) + 3)
        st.caption(_ui.how_to_read(*_ui.HOW_ASSETS_TABLE))
    else:
        st.warning("The reference files in data/reference could not be read.")
    if cfg.universe.asset_source == "generic":
        st.markdown(f"**Generic assets of this run** ({len(market.assets)}, artificial returns)")
        st.dataframe(market.assets[["name", "asset_class", "sub_class", "source"]], height=300)
        st.caption(_ui.how_to_read(*_ui.HOW_GENERIC_ASSETS))

    st.subheader("Manual topics")
    topics_ref = ctx["topics_ref"]
    if topics_ref is not None:
        tt = topics_ref.reset_index()[["topic_id", "group", "name", "scope"]]
        tt.columns = ["ID", "Group", "Name", "Scope"]
        st.dataframe(tt, hide_index=True, height=35 * (len(tt) + 1) + 3,
                     column_config={"Scope": st.column_config.TextColumn("Scope", width="large")})
        st.caption(_ui.how_to_read(*_ui.HOW_TOPICS_TABLE))
    n_gen = cfg.topics.n_generic
    if n_gen:
        linked = sim.links.table.loc[sim.links.table["origin"] == "random", "topic_id"].nunique()
        st.caption(f"This run also has {n_gen} generic topics G001-G{n_gen:03d} (group Generic); {linked} of them "
                   "carry random links.")

    st.subheader("Link map of this run")
    st.caption(_ui.how_to_read(*_ui.HOW_LINK_MAP))
    frame = _ui.link_edit_frame(sim.links.table, ctx["t_labels"], ctx["a_labels"])
    edited = st.data_editor(
        frame,
        key=f"lists_link_editor_{cfg.key('simulation')}",
        hide_index=True,
        height=min(600, 35 * (len(frame) + 1) + 3),
        disabled=["topic_id", "topic", "asset_id", "asset", "mechanism", "origin"],
        column_config={
            "topic_id": None,
            "topic": st.column_config.TextColumn("Topic"),
            "asset_id": None,
            "asset": st.column_config.TextColumn("Asset"),
            "tier": st.column_config.SelectboxColumn("Tier", options=["strong", "moderate", "weak", "none"],
                                                     required=True),
            "sign": st.column_config.SelectboxColumn("Sign", options=[1, -1], required=True),
            "mechanism": st.column_config.TextColumn("Mechanism", width="large"),
            "origin": st.column_config.TextColumn("Origin"),
        },
    )
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Apply edits", key="lists_apply", type="primary"):
        new = _ui.overrides_from_edits(frame, edited)
        if new:
            st.session_state["link_overrides"] = _ui.merge_overrides(st.session_state["link_overrides"], new)
            st.rerun()
        else:
            st.info("No change to apply.")
    if c2.button("Reset to default", key="lists_reset"):
        st.session_state["link_overrides"] = {}
        st.rerun()

    with st.expander("Add a link"):
        tids = [str(t) for t in sim.topics.table.index]
        aids = [str(a) for a in market.assets.index]
        a1, a2, a3, a4, a5 = st.columns([2, 2, 1.2, 1, 1])
        t_new = a1.selectbox("Topic", tids, format_func=lambda t: ctx["t_labels"].get(t, t), key="lists_add_topic")
        a_new = a2.selectbox("Asset", aids, format_func=lambda a: ctx["a_labels"].get(a, a), key="lists_add_asset")
        tier_new = a3.selectbox("Tier", ["strong", "moderate", "weak"], key="lists_add_tier")
        sign_new = a4.selectbox("Sign", [1, -1], key="lists_add_sign")
        if a5.button("Add link", key="lists_add"):
            st.session_state["link_overrides"] = _ui.merge_overrides(
                st.session_state["link_overrides"], [(t_new, a_new, tier_new, int(sign_new))]
            )
            st.rerun()
    overrides = st.session_state["link_overrides"]
    if overrides:
        st.markdown(f"**Active session edits** ({len(overrides)})")
        st.dataframe(pd.DataFrame(list(overrides.values()), columns=["topic_id", "asset_id", "tier", "sign"]),
                     hide_index=True)


def method_tab(ctx: dict[str, Any]) -> None:
    st.subheader("Method")
    st.markdown(TERMINOLOGY)
    st.markdown(
        "Asset returns are real (or artificial); topic attention is simulated from them with a known link "
        "structure, so the lab can compare estimates with the truth (DESIGN.md G.5-G.8)."
    )
    st.markdown(
        "1. **Returns.** $r_{n,t}$ is asset $n$'s daily return on day $t$. The standardised return "
        "$\\tilde r_{n,t} = (r_{n,t} - \\mu_n)/\\sigma_n$ uses the full-sample mean $\\mu_n$ and standard deviation "
        "$\\sigma_n$ and is clipped at $\\pm 8$.\n"
        "2. **Designed shocks.** Topic $k$'s designed attention shock on day $t$ is the sum of its linked assets' "
        "standardised returns $l$ days later (the lead $l$ is 0 or 1), weighted by the set sensitivities, plus news "
        "noise $u_{k,t}$ with unit variance. $W_k$ is row $k$ of the design matrix $W$ of set sensitivities (sign "
        "times beta on linked pairs, 0 elsewhere), and "
        "$\\sigma_{u,k}^2 = 1 - W_k R W_k'$ makes the shock's variance 1 ($R$ the return correlation matrix):"
    )
    st.latex(r"s_{k,t} = W_k\,\tilde r_{t+l} + \sigma_{u,k}\,u_{k,t}")
    st.markdown(
        "3. **Attention.** The level is a base level $m_k$ between 0.15 and 0.35, plus a slow AR(1) component "
        "$g_{k,t}$, plus the designed shock scaled by $\\kappa = 0.02$:"
    )
    st.latex(r"a_{k,t} = m_k + g_{k,t} + \kappa\, s_{k,t}")
    st.markdown(
        "4. **Observed shocks.** The estimators see only $a$. The shock is today's attention minus its mean over "
        "the previous $w$ trading days (D9), divided by its standard deviation on the training window:"
    )
    st.latex(r"z_{k,t} = a_{k,t} - \frac{1}{w}\sum_{j=1}^{w} a_{k,t-j}, \qquad "
             r"sh_{k,t} = z_{k,t} / \mathrm{sd}_{train}(z_k)")
    st.markdown(
        "5. **Truth.** $B_{true} = S_z^{-1} C$ are the population regression coefficients of the standardised return "
        "on the observed shocks ($S_z$ the shocks' correlation matrix, $C$ their covariance with the returns). They "
        "include spillovers through correlated assets, so $B_{true}$ is not sparse even though $W$ is.\n"
        "6. **Direct regression.** Per asset on the training days, with $\\hat r$ the return standardised by its "
        "training mean and standard deviation, $\\alpha_n$ an intercept and $b_{k,n}$ the topic sensitivity; elastic "
        "net with the universal penalty $\\sqrt{2 \\ln L / n}$ by default ($L$ topics, $n$ training days):"
    )
    st.latex(r"\hat r_{n,t+l} = \alpha_n + \sum_k b_{k,n}\, sh_{k,t} + e_{n,t+l}")
    st.markdown(
        "7. **Evaluation.** Over the forecast window's return days $H$, the topic-explained return is "
        "$\\hat r^{top}_{n,t+l} = \\mathrm{sd}_{train}(r_n) \\sum_k b_{k,n}\\, sh_{k,t}$. The OOS R² is uncentered; "
        "topic $k$'s contribution $c_{k,n}$ uses the frozen sensitivity and the realised shocks, and the "
        "contributions plus the residual add up to the realised move:"
    )
    st.latex(r"R^2_n = 1 - \frac{\sum_H (r - \hat r^{top})^2}{\sum_H r^2}, \qquad "
             r"c_{k,n} = \mathrm{sd}_{train}(r_n)\, b_{k,n} \sum_H sh_{k,t}")
    st.markdown(
        "8. **BKS Sparse IPCA** runs on weekly periods with the package stages unchanged; its per-topic split of "
        "the fitted return is not identified (D52).\n"
        "9. **Method comparison.** The Compare methods tab scores every method on the same forecast days, with its "
        "sensitivities frozen at the training end. BKS enters through its implied sensitivities: the topic "
        "covariances that each asset's BKS factor betas imply, turned into sensitivities with the training "
        "covariance of the topic shocks (DESIGN.md G.15). It enters twice: with the full covariance history, as in "
        "BKS, and with the training window only, which sees the data the direct methods see (D88)."
    )

    st.subheader("Why BKS-implied scores lower (G.15.1)")
    st.markdown(WHY_BKS_LOWER)

    st.subheader("Limitations (G.12)")
    st.markdown(
        "1. **Simulated attention.** Attention is built from the returns it is meant to explain, so the lab shows "
        "estimator behaviour and the size of explainable variation under stated assumptions, not the information "
        "content of real news.\n"
        "2. **Illustrative link map.** The default links and signs are a plausible story, not estimates; real "
        "attention is unsigned, and the sign convention is a modelling assumption per topic.\n"
        "3. **The truth is a full-sample summary.** It uses the full-sample moments of the filtered noise-free "
        "signal, so it keeps the serial correlation of daily returns (lag-1 autocorrelations down to -0.5 for "
        "spreads whose legs close at different times). It does not model volatility clustering or correlations "
        "that change over time, and it ignores the clipping of attention at 1e-6.\n"
        "4. **Non-synchronous closes.** Legs close in New York, London, Frankfurt, Zurich and Tokyo, so daily spread "
        "returns mix closing times. Weekly BKS periods reduce this.\n"
        "5. **Spot FX and bucket mismatches.** FX legs exclude carry; the UK, Italy and Japan 7-10y legs use "
        "all-maturity government bond funds (TBC in data/market/README.md).\n"
        "6. **Short windows.** A one-week window has five daily observations; its OOS correlations and R² are "
        "dominated by noise. The window sweep shows the distribution across windows.\n"
        "7. **BKS identification.** D47 and D52 apply: noise topics' instruments inherit betas, and the per-topic "
        "split of BKS fitted returns is not identified.\n"
        "8. **BKS-implied sensitivities.** They keep only the part of each asset's topic covariances in the K "
        "directions BKS fitted to explain weekly returns, which on the lab data carry little of the topic signal "
        "(see above). With the full history their instruments also weigh the days before the training window, so "
        "that variant is not strictly like for like; the training-window variant is, at the cost of noisy "
        "instruments in its first training weeks."
    )

    st.subheader("Market data")
    meta = ctx["market"].meta
    parts = [f"price source: {meta.get('price_source')}", f"{meta.get('n_days')} weekdays {meta.get('start')} to "
             f"{meta.get('end')}"]
    if meta.get("manifest_timestamp"):
        parts.append(f"manifest of {meta['manifest_timestamp']}")
    failed = meta.get("failed_assets") or []
    parts.append(f"{len(failed)} failed asset(s)" + (f": {', '.join(failed)}" if failed else ""))
    st.caption("This run: " + "; ".join(parts) + ".")
    readme = reference.market_dir() / "README.md"
    if readme.is_file():
        with st.expander("data/market/README.md: sources, conventions, TBC items, QA", expanded=True):
            st.markdown(readme.read_text(encoding="utf-8"))
    else:
        st.info(f"No README found at {readme}.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def _load_reference() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, str | None]:
    try:
        return reference.load_assets(), reference.load_legs(), reference.load_topics(), None
    except (FileNotFoundError, ValueError) as exc:
        return None, None, None, str(exc)


def _run_bks(session: LabSession, cfg: LabConfig, ctx: dict[str, Any]) -> None:
    """Run BKS for ``cfg``; store the result or the error in session state.

    Only the spinner is drawn: no Streamlit call happens during the
    computation, so a widget change cannot interrupt the fit (a rerun then
    finds the fit in the cache). The request flag is cleared when the run
    ends, not before.

    The fit key joins this browser session's fit keys (D80) once the fit
    exists, so a refused fit never reads as a cached one that was evicted. A
    refusal is stored under the ``bks`` key (``bks_error``) and, when no fit
    exists, under the fit key as well (``bks_fit_errors``, which the Compare
    tab's Run BKS also writes), so every page finds it.
    """
    key = ctx["bks_key"]
    fit_key = session.stage_key("bks_fit", cfg)
    fit_errors = st.session_state.setdefault("bks_fit_errors", {})
    t0 = time.perf_counter()
    try:
        with st.spinner("Running BKS Sparse IPCA ...", show_time=True):
            session.bks_panel(cfg)
            session.bks_fit(cfg)
            st.session_state["bks_fit_keys"].add(fit_key)
            fit_errors.pop(fit_key, None)
            res = session.bks(cfg)
    except Exception as exc:  # ValueError: the settings; others: numerical failures. Show the reason, keep the page
        if isinstance(exc, ValueError):
            reason = str(exc)
        else:
            logger.exception("BKS run failed")
            reason = f"{type(exc).__name__}: {exc}"
        st.session_state["bks_error"] = (key, reason)
        if not session.has("bks_fit", cfg):
            fit_errors[fit_key] = reason
        st.session_state.pop("bks_requested", None)
        return
    _store_bks(res, ctx, time.perf_counter() - t0, session.peek("bks_fit", cfg))
    st.session_state.pop("bks_requested", None)


def _run_bks_variants(session: LabSession, cfg: LabConfig, methods: tuple[str, ...]) -> None:
    """The Compare tab's Run BKS: panel and fit of each requested BKS-implied variant (D88).

    A variant whose fit is cached already (by any browser session) costs
    nothing; each fitted variant's fit key joins this browser session's keys,
    so the comparison reuses it (D80). As in :func:`_run_bks`, only the spinner
    is drawn while the fits compute; errors are stored per fit key
    (``bks_fit_errors``), and a refused variant's key does not join.
    """
    keys = {m: session.stage_key("bks_fit", lab_compare.method_config(cfg, m)) for m in methods}
    errors: dict[str, str] = {}
    with st.spinner("Running BKS Sparse IPCA ...", show_time=True):
        for m in methods:
            mcfg = lab_compare.method_config(cfg, m)
            try:
                session.bks_panel(mcfg)
                session.bks_fit(mcfg)
            except ValueError as exc:
                errors[keys[m]] = str(exc)
            except Exception as exc:  # numerical failures: show the reason, keep the page
                logger.exception("BKS run failed (%s)", m)
                errors[keys[m]] = f"{type(exc).__name__}: {exc}"
    st.session_state["bks_fit_keys"].update(k for k in keys.values() if k not in errors)
    stored = st.session_state.setdefault("bks_fit_errors", {})
    for k in keys.values():
        stored.pop(k, None)
    stored.update(errors)
    st.session_state.pop("bks_compare_requested", None)


def _bks_sync(session: LabSession, cfg: LabConfig, ctx: dict[str, Any]) -> None:
    """BKS on request, shared by the Simulation lab and BKS trace pages (D80, D90).

    Runs the Compare tab's and the Run BKS buttons' requests (whichever page they were pressed on, so that a
    request is never left for a later visit to the other page), and re-evaluates cheaply when a fit this browser
    session requested is cached but the stored result is for other settings. ``ctx`` needs ``cfg``, ``bks_key``,
    ``t_labels`` and ``a_labels``.
    """
    if st.session_state.get("bks_compare_requested"):
        _run_bks_variants(session, cfg, tuple(st.session_state["bks_compare_requested"]))
    if st.session_state.get("bks_requested", False):
        _run_bks(session, cfg, ctx)
        return
    store = st.session_state.get("bks_store")
    mine = session.stage_key("bks_fit", cfg) in st.session_state["bks_fit_keys"]
    if mine and (store is None or store["key"] != ctx["bks_key"]) and session.has("bks_fit", cfg) and \
            session.has("bks_panel", cfg):
        try:
            t = time.perf_counter()
            _store_bks(session.bks(cfg), ctx, time.perf_counter() - t, session.peek("bks_fit", cfg))
        except ValueError as exc:
            st.session_state["bks_error"] = (ctx["bks_key"], str(exc))


def _store_bks(res: Any, ctx: dict[str, Any], seconds: float, fit: Any = None) -> None:
    raw = fit.meta.get("lam_max") if fit is not None else None
    lam_max = float(raw) if raw is not None else float("nan")
    st.session_state["bks_store"] = {
        "lam_max": lam_max,
        "key": ctx["bks_key"],
        "result": res,
        "seconds": float(seconds),
        "t_labels": dict(ctx["t_labels"]),
        "a_labels": dict(ctx["a_labels"]),
    }
    st.session_state["bks_error"] = None


def shared_settings(on_simulation: bool) -> dict[str, Any]:
    """Draw the shared sidebar and turn its values into a lab config (every page).

    Returns the sidebar's values in effect, ``cfg`` (``None`` when the values
    are invalid), ``errors`` and ``notes`` of :func:`_ui.config_from_values`,
    the training window, the feasibility placeholder, the reference tables
    and the settings the Real data page shows (``settings``).
    """
    _init_state()
    ref_assets, legs, topics_ref, ref_error = _load_reference()
    sb = sidebar(ref_assets, on_simulation=on_simulation)
    values = sb["values"]
    overrides = tuple(st.session_state["link_overrides"].values())
    classes = ref_assets["asset_class"] if ref_assets is not None else None
    cfg, errors, notes = _ui.config_from_values(values, overrides, classes)
    names = ref_assets["name"].to_dict() if ref_assets is not None else {}
    return {
        "values": values,
        "cfg": cfg,
        "errors": errors,
        "notes": notes,
        "train_window": sb["train_window"],
        "feas_slot": sb["feas_slot"],
        "ref_assets": ref_assets,
        "legs": legs,
        "topics_ref": topics_ref,
        "ref_error": ref_error,
        "settings": {"values": values, "cfg": cfg, "errors": errors, "notes": notes, "asset_classes": classes,
                     "asset_names": names},
    }


def main() -> None:
    run = _RUN
    values = run["values"]
    ref_assets, legs, topics_ref, ref_error = run["ref_assets"], run["legs"], run["topics_ref"], run["ref_error"]
    sb = {"feas_slot": run["feas_slot"]}

    st.title("Topic-sensitivity lab")
    st.caption(
        "Topic attention is simulated from asset returns with a known link structure. The direct sensitivity "
        "regression and BKS Sparse IPCA are fitted on the training window and evaluated out of sample. Every "
        "number is a property of the simulation settings, not evidence about real news."
    )
    if ref_error and values["sb_asset_source"] == "listed":
        st.error(f"The reference data could not be read: {ref_error}")
        st.stop()

    cfg, errors, notes = run["cfg"], run["errors"], run["notes"]
    if errors:
        for e in errors:
            st.error(e)
        st.stop()
    assert cfg is not None

    session = get_session()
    timings: dict[str, dict[str, Any]] = {}
    t0 = time.perf_counter()
    try:
        with st.spinner("Running the lab ..."):
            for stage in ("market", "simulation", "truth", "shocks", "direct", "evaluation", "sweep"):
                getattr(session, stage)(cfg)
                timings[stage] = dict(session.last_timings.get(stage, {}))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        st.error(f"The lab could not run with these settings: {exc}")
        st.stop()
    timings["total"] = {"seconds": time.perf_counter() - t0, "cached": False}

    market, sim, truth = session.market(cfg), session.simulation(cfg), session.truth(cfg)
    ctx: dict[str, Any] = {
        "cfg": cfg,
        "market": market,
        "sim": sim,
        "truth": truth,
        "shocks": session.shocks(cfg),
        "fit": session.direct(cfg),
        "ev": session.evaluation(cfg),
        "sweep": session.sweep(cfg),
        "timings": timings,
        "a_labels": _ui.asset_labels(market.assets),
        "t_labels": _ui.topic_labels(sim.topics.table),
        "feasibility": _ui.feasibility_summary(sim, cfg.exposure, _ui.topic_labels(sim.topics.table)),
        "ref_assets": ref_assets,
        "legs": legs,
        "topics_ref": topics_ref,
        "bks_key": session.stage_key("bks", cfg),
        "session": session,
    }

    feas = ctx["feasibility"]
    if feas["n_scaled"]:
        sb["feas_slot"].caption(
            f"In this run {feas['n_scaled']} of {feas['n_topics']} topics were scaled down for feasibility; at the "
            f"current ratios beta 1 up to {min(feas['max_beta_1'], 0.95):.2f} needs no scaling (details on the "
            "Overview)."
        )
    elif np.isfinite(feas["max_beta_1"]) and feas["max_beta_1"] < 0.95:
        sb["feas_slot"].caption(
            f"No feasibility scaling in this run; at the current ratios beta 1 up to {feas['max_beta_1']:.2f} "
            "needs none."
        )

    _bks_sync(session, cfg, ctx)

    ev = ctx["ev"]
    sources = ", ".join(f"{v} {k}" for k, v in market.meta.get("sources", {}).items())
    n_manual = int((sim.topics.table["group"] != "Generic").sum())
    days = ev.return_days
    window = f"{days[0].date()} to {days[-1].date()}" if len(days) else "no return days"
    n_train = len(pd.bdate_range(cfg.window.train_start, cfg.window.train_end))
    st.markdown(
        f"**{len(market.assets)} assets** ({sources}) · **{len(sim.topics.table)} topics** ({n_manual} manual, "
        f"{len(sim.topics.table) - n_manual} generic) · **{len(sim.links.table)} links** · training "
        f"{cfg.window.train_start} to {cfg.window.train_end} ({n_train} weekdays) · forecast {window} "
        f"(**{ev.n_days} return days**) · "
        f"w = {cfg.window.shock_window} · lead {'next day' if cfg.exposure.lead_days else 'same day'} · "
        f"{_ui.METHOD_LABELS[cfg.direct.method].split(' (')[0].lower()}"
    )
    for n in notes:
        st.warning(n)
    if ev.n_days == 0:
        st.error("The forecast window has no return day in the data. Move the forecast start earlier.")
        st.stop()

    tabs = st.tabs(list(TABS))
    with tabs[0]:
        overview_tab(ctx)
    with tabs[1]:
        exposure_tab(ctx)
    with tabs[2]:
        contributions_tab(ctx)
    with tabs[3]:
        compare_tab(ctx)
    with tabs[4]:
        bks_tab(ctx)
    with tabs[5]:
        lists_tab(ctx)
    with tabs[6]:
        method_tab(ctx)


def real_data_page() -> None:
    real_exposures.render(_RUN.get("settings"))


#: Lead caption of the BKS trace page (D90).
TRACE_LEAD = (
    "The BKS run of the current settings, traced step by step: the inputs, what each stage computes and what "
    "comes out, with an independent reference next to each result (a recomputation by the formula, an identity "
    "that must hold, or the simulation's true value). Nothing here changes the run: the settings are the "
    "sidebar's, and a BKS run here is the same run the Simulation lab's BKS tab shows."
)


def _bks_refusal(bks_key: str, fit_key: str, cached: bool) -> str | None:
    """The reason BKS refused the current settings, if it did: the error of a run under this ``bks`` key (Run BKS
    on any page), else, while no fit is cached, the refusal stored under the fit key (``_run_bks`` or the Compare
    tab's Run BKS)."""
    err = st.session_state.get("bks_error")
    if err and err[0] == bks_key:
        return str(err[1])
    if not cached:
        return st.session_state.get("bks_fit_errors", {}).get(fit_key)
    return None


def bks_trace_page() -> None:
    """The BKS trace page (D90): the current settings' BKS run, step by step, with references.

    Runs the cheap lab stages (cached after the Simulation lab page), handles the Run BKS requests of this page
    and the sidebar (:func:`_bks_sync`), and traces only a fit this browser session requested for the current
    settings (D80): it never starts a fit by itself. A refused run shows its reason (:func:`_bks_refusal`) and
    nothing else; the eviction note is only for a fit of this browser session that left the cache. The trace
    itself (:meth:`LabSession.bks_trace`) is one pure call inside one spinner; only ``BKSNotCached`` reads as
    an eviction, any other error is logged and shown. The steps are drawn by :func:`trace_page.render`.
    """
    run = _RUN
    values = run["values"]
    st.title("BKS trace")
    st.caption(TRACE_LEAD)
    st.page_link(PAGES["simulation"], label="Back to the Simulation lab", icon=":material/arrow_back:")
    if run["ref_error"] and values["sb_asset_source"] == "listed":
        st.error(f"The reference data could not be read: {run['ref_error']}")
        st.stop()
    cfg, errors = run["cfg"], run["errors"]
    if errors:
        for e in errors:
            st.error(e)
        st.stop()
    assert cfg is not None

    session = get_session()
    try:
        with st.spinner("Running the lab ..."):
            for stage in ("market", "simulation", "truth", "shocks", "direct", "evaluation"):
                getattr(session, stage)(cfg)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        st.error(f"The lab could not run with these settings: {exc}")
        st.stop()
    market, sim = session.market(cfg), session.simulation(cfg)
    ctx: dict[str, Any] = {
        "cfg": cfg,
        "session": session,
        "market": market,
        "sim": sim,
        "truth": session.truth(cfg),
        "shocks": session.shocks(cfg),
        "fit": session.direct(cfg),
        "ev": session.evaluation(cfg),
        "a_labels": _ui.asset_labels(market.assets),
        "t_labels": _ui.topic_labels(sim.topics.table),
        "bks_key": session.stage_key("bks", cfg),
    }
    _bks_sync(session, cfg, ctx)

    w = cfg.window
    check = _ui.bks_training_check(w.train_start, w.train_end, cfg.bks, cfg.exposure.lead_days, w.shock_window)
    if not check["can_run"]:
        st.warning(f"{check['reason']} The direct estimator runs on windows down to one month.")
    _bks_runtime_warning(len(sim.topics.table))
    st.button("Run BKS", key="tr_run_bks", type="primary", on_click=_request_bks, disabled=not check["can_run"],
              help=None if check["can_run"] else "Change the training window first.")
    fit_key = session.stage_key("bks_fit", cfg)
    cached = session.has("bks_panel", cfg) and session.has("bks_fit", cfg)
    reason = _bks_refusal(ctx["bks_key"], fit_key, cached)
    if reason:  # a refused run (this page, the sidebar, the BKS tab or the Compare tab): its reason, no trace
        st.error(f"BKS could not run with these settings: {reason}")
        return
    mine = fit_key in st.session_state["bks_fit_keys"]
    if not (mine and cached):
        if mine:
            st.info("The BKS fit for these settings is no longer in the cache (it keeps the two most recent fits). "
                    "Press Run BKS to fit it again; the trace then shows each step.")
        elif check["can_run"]:
            st.info("Press Run BKS (here or in the sidebar) to fit BKS on the current settings; the trace then shows "
                    "each step. The default settings take about a second.")
        return
    try:
        with st.spinner("Tracing BKS ...", show_time=True):
            trace = session.bks_trace(cfg)
        # the trace evaluated the cached fit under the bks key; peek so that nothing here can start a fit
        res = session.peek("bks", cfg)
        store = st.session_state.get("bks_store")
        if res is None and store is not None and store["key"] == ctx["bks_key"]:
            res = store["result"]
        if res is None:
            raise BKSNotCached("the evaluated BKS run is no longer cached")
    except BKSNotCached:  # evicted between the check above and the trace; any other error is a defect, logged below
        st.info("The BKS fit for these settings is no longer in the cache. Press Run BKS to fit it again.")
        return
    except Exception as exc:  # show the reason, keep the page
        logger.exception("BKS trace failed")
        st.error(f"The BKS trace could not be built: {type(exc).__name__}: {exc}")
        return
    ctx.update(trace=trace, res=res, panel=session.peek("bks_panel", cfg), bks_fit=session.peek("bks_fit", cfg))
    trace_page.render(ctx, {"control": control, "follow_control": follow_control, "show_chart": show_chart,
                            "bks_tiles": bks_tiles})


# Top level (D82, D90): the simulation lab, the BKS trace and the Real data placeholder are separate pages. The
# sidebar is shared: it is drawn here, before the chosen page runs, and its result goes to the pages through _RUN.
# PAGES holds the page objects so that the pages can link to each other.
PAGES.update(
    simulation=st.Page(main, title="Simulation lab", icon=":material/science:", url_path="simulation", default=True),
    trace=st.Page(bks_trace_page, title="BKS trace", icon=":material/troubleshoot:", url_path=BKS_TRACE_URL),
    real=st.Page(real_data_page, title="Real data", icon=":material/insights:", url_path=REAL_DATA_URL),
)
navigation = st.navigation(list(PAGES.values()), position="top")
_RUN.update(shared_settings(on_simulation=navigation.url_path != REAL_DATA_URL))
navigation.run()
