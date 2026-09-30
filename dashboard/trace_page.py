"""Steps of the BKS trace page (DESIGN.md G.16, D90): the current BKS run, step by step, with references.

``render(ctx, ui)`` is called by ``app.bks_trace_page`` once a BKS fit of the current settings is cached and
traced (:meth:`narrative_ipca.exposure_lab.session.LabSession.bks_trace`). This module cannot import ``app.py``
(that would re-run the script), so the widgets and helpers it needs come in ``ui``: ``control``,
``follow_control``, ``show_chart`` and ``bks_tiles``.

``ctx`` holds ``cfg``, ``session``, ``market``, ``sim``, ``truth``, ``shocks``, ``fit`` (the sidebar's direct
fit), ``ev`` (its evaluation), ``a_labels``, ``t_labels``, ``bks_key``, ``trace`` (``BKSTrace``), ``res``
(``BKSLabResult``), ``panel`` (``BKSPanel``) and ``bks_fit`` (``BKSFit``).

Layout: a run summary line, the BKS tiles, the step selector and the focus controls (asset, topic and, in the
steps that use one, a week), then the selected step only. Steps are lazy: only the selected step calls its
per-selection helpers of :mod:`narrative_ipca.exposure_lab.trace`; the summary reads the cached trace only.
Each step has a "What happens here" caption (inputs, computation with its formula, outputs, the package
function), its charts and tables each followed by a "How to read" caption (``_ui.HOW_TRACE_*``), its checks,
and a line saying what feeds the next step. The page text says "topic sensitivity", never "exposure".

Optional trace fields (added with the suspect hunts; SPEC addendum): the ladder rows ``bks_ls`` and ``zero``,
the path's null band and band floors, ``units.conversion_exact`` / ``kernel_divisor``, ``capture.ls_share`` /
``per_asset_share``, ``BKSTrace.sigma_table`` and the panel-day counts in ``meta``. The page uses each when
present and leaves it out otherwise.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import _ui
from narrative_ipca.exposure_lab import charts
from narrative_ipca.exposure_lab import trace as T
from narrative_ipca.exposure_lab.evaluate import median_finite

#: Help text of the status table's "Reading" column.
READING_HELP = ("as expected: every check holds and nothing departs; departs: the method departs from its "
                "reference; off: an identity check failed (a code defect).")

#: Steps whose charts use the focus week (``tr_week``) and the forecast week (``tr_fweek``).
WEEK_STEPS = ("instruments", "panel")
FWEEK_STEPS = ("forecast",)

#: Largest number of topics or directions drawn as bars before the charts keep only the largest.
MAX_BARS = 30

#: Shorter row labels of the ladder chart (the ladder table keeps the trace's full labels), so that the two panels
#: keep their width in a narrow window.
LADDER_CHART_LABELS: dict[str, str] = {
    "instruments_kernel": "Instruments, same-history Sigma_z",
    "instruments_train": "Instruments, training-day Sigma_z",
    "best_rank": "Best K directions",
}

#: What feeds the next step (the link line at the end of each step).
NEXT: dict[str, str] = {
    "summary": "Start at step 1 to follow the run from its inputs, or open the step the table marks 'departs'.",
    "inputs": "Feeds step 2: the daily attention and returns are paired by the lead and the returns are scaled.",
    "align": ("Feeds steps 3 to 5: the lagged attention becomes the shocks (step 3); the scaled returns enter the "
              "instruments (step 4) and the weekly returns (step 5); the unit conversion returns in step 8."),
    "shocks": ("Feeds step 4: the shocks are the second factor of every instrument covariance; step 8 also uses "
               "their covariance over the training days (Sigma_z)."),
    "instruments": "Feeds step 5: each week's instruments become the rows of the following return week.",
    "panel": "Feeds step 6: the training weeks' rows are what Sparse IPCA fits.",
    "fit": ("Feeds steps 7 and 8: the frozen Gamma turns each asset's instruments into K loadings, for the forecast "
            "weeks (step 7) and for the implied sensitivities (step 8)."),
    "forecast": ("Step 8 does not use the forecast weeks: the implied sensitivities come from the training fit "
                 "alone and are scored on the forecast window like the direct methods."),
    "implied": ("The implied sensitivities are what the Compare methods tab scores as BKS-implied; the Summary's "
                "ladder puts them between their references."),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _day(x: Any) -> str:
    """ISO date of a timestamp, or "n/a"."""
    try:
        ts = pd.Timestamp(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def _f(x: Any, fmt: str = ".2f") -> str:
    """A number with ``fmt``, or "n/a"."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return format(v, fmt) if math.isfinite(v) else "n/a"


def _how(name: str) -> None:
    """The "How to read" caption ``_ui.<name>``."""
    st.caption(_ui.how_to_read(*getattr(_ui, name)))


def _what(lines: list[tuple[str, str]]) -> None:
    """The step's "What happens here" caption: inputs, computation, outputs, code."""
    st.caption("What happens here:\n\n" + "\n".join(f"- {head}: {text}" for head, text in lines))


def _next(step: str) -> None:
    st.caption(f"Next: {NEXT[step]}")


def _heading(text: str) -> None:
    st.markdown(f"**{text}**")


def _num(label: str, fmt: str = "%.3g", help: str | None = None) -> Any:
    return st.column_config.NumberColumn(label, format=fmt, help=help)


def _txt(label: str, width: str | None = None, help: str | None = None) -> Any:
    return st.column_config.TextColumn(label, width=width, help=help)


#: Cell tints of the status words that need attention (tables): a departure, and a check that is off.
TINTS: dict[str, str] = {"departs": "background-color: rgba(232, 105, 47, 0.18)",
                         "off": "background-color: rgba(227, 73, 72, 0.24)"}


def _table(frame: pd.DataFrame, config: dict[str, Any] | None = None, height: int | None = None,
           tint: str | None = None) -> None:
    """A read-only table at full width (index hidden), with number formats per column; missing values show "–".

    ``tint``: a column whose words "departs" and "off" get a coloured background.
    """
    kw: dict[str, Any] = {}
    if height is not None:
        kw["height"] = int(height)
    data: Any = frame
    if tint is not None and tint in frame.columns:
        try:
            data = frame.style.map(lambda v: TINTS.get(str(v), ""), subset=[tint])
        except (ImportError, AttributeError):  # pragma: no cover - Styler needs jinja2 (a Streamlit dependency)
            data = frame
    st.dataframe(data, hide_index=True, width="stretch", column_config=config or {}, placeholder="–", **kw)


def _rows_height(n: int, cap: int = 560) -> int:
    """Height that shows ``n`` rows without scrolling, up to ``cap`` pixels."""
    return min(35 * (int(n) + 1) + 3, cap)


def _pct(s: Any) -> Any:
    """Fractions to percent numbers (for ``%.1f%%`` columns)."""
    return s * 100.0


def _accepts(func: Any, name: str) -> bool:
    """Whether ``func`` takes a keyword argument ``name`` (optional chart features)."""
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False


def _week_label(trace: T.BKSTrace, week: str) -> str:
    ts = pd.Timestamp(week)
    if ts == trace.row_week:
        return f"{week} (last training week, used in step 8)"
    if ts in trace.train_periods:
        return f"{week} (training)"
    if ts in trace.forecast_periods:
        return f"{week} (forecast)"
    return week


def _instrument_week_of(trace: T.BKSTrace, return_week: pd.Timestamp) -> pd.Timestamp | None:
    """The instrument week whose covariances form the rows of ``return_week`` (the week before it)."""
    ends = trace.week_ends
    if return_week not in ends:
        return None
    j = int(ends.get_loc(return_week))
    return pd.Timestamp(ends[j - 1]) if j >= 1 else None


def _topic_label(t_labels: dict[str, str]) -> dict[str, str]:
    """Topic labels plus the constant instrument."""
    return {"const": "Constant", **t_labels}


def _default_topic(ctx: dict[str, Any], asset: str, topics: list[str]) -> str:
    """The asset's topic with the largest absolute true sensitivity (the topic control follows it)."""
    B = ctx["truth"].B_true
    if asset in B.columns:
        col = B[asset].reindex(topics).abs()
        if col.notna().any():
            return str(col.idxmax())
    return topics[0]


# ---------------------------------------------------------------------------
# Top of the page
# ---------------------------------------------------------------------------
def run_summary(trace: T.BKSTrace, res: Any, cfg: Any) -> str:
    """One markdown line naming the traced run: history, K, lambda, topics, weeks, the Eq. 5 week, w, lead."""
    rule = {"tolerance": "tolerance rule", "argmax": "argmax rule", "fixed": "fixed"}.get(trace.lambda_rule,
                                                                                       trace.lambda_rule)
    tp, fp = trace.train_periods, trace.forecast_periods
    train = f"{len(tp)} training weeks ({_day(tp.min())} to {_day(tp.max())})" if len(tp) else "no training weeks"
    fc = f"{len(fp)} forecast weeks ({_day(fp.min())} to {_day(fp.max())})" if len(fp) else "no forecast weeks"
    history = _ui.BKS_HISTORY_LABELS.get(trace.history, trace.history)
    n_sel = len(trace.meta.get("selected_topics", res.selected_topics))
    return (
        f"**{history}** · K = {trace.K} · lambda {trace.lam:.4g} ({rule}) · **{n_sel} of {len(trace.topics)} topics** "
        f"selected · {train} · {fc} · implied sensitivities from the instruments of the week ending "
        f"{_day(trace.instrument_week)} · {len(trace.assets)} assets · w = {trace.shock_window} · lead "
        f"{'next day' if trace.lead_days else 'same day'}"
    )


def render(ctx: dict[str, Any], ui: dict[str, Any]) -> None:
    """Draw the trace page below the Run BKS button (see the module docstring)."""
    trace: T.BKSTrace = ctx["trace"]
    st.markdown(run_summary(trace, ctx["res"], ctx["cfg"]))
    ui["bks_tiles"](ctx["res"], caption_expander=True)

    step = ui["control"]("radio", "Step", "tr_step", options=list(T.STEPS), horizontal=True,
                         format_func=lambda s: T.STEPS.get(s, s))
    focus = _focus_controls(ctx, ui, trace, step)
    STEP_RENDERERS[step](ctx, ui, focus)


def _focus_controls(ctx: dict[str, Any], ui: dict[str, Any], trace: T.BKSTrace, step: str) -> dict[str, Any]:
    """Asset, topic and (when the step uses one) week; returns the focus."""
    a_labels, t_labels = ctx["a_labels"], ctx["t_labels"]
    assets, topics = list(trace.assets), list(trace.topics)
    c1, c2, c3 = st.columns(3)
    asset = ui["control"]("selectbox", "Asset", "bks_asset", container=c1, options=assets,
                          format_func=lambda a: a_labels.get(a, a), fallback=_ui.default_asset(assets),
                          help="Shared with the BKS tab's asset. The per-asset charts and tables of each step "
                          "follow it; the asset is highlighted where every asset is shown.")
    topic = ui["follow_control"]("Topic", "tr_topic", topics, _default_topic(ctx, asset, topics),
                                 format_func=lambda t: t_labels.get(t, t), container=c2,
                                 help="Follows the asset (its topic with the largest true sensitivity) until "
                                 "you pick one. Used by steps 1, 3 and 4.")
    focus: dict[str, Any] = {"asset": asset, "topic": topic, "week": None, "fweek": None}
    if step in WEEK_STEPS:
        weeks = sorted(set(trace.train_periods) | set(trace.forecast_periods))
        options = [_day(w) for w in weeks]
        if options:
            chosen = ui["follow_control"]("Return week", "tr_week", options, _day(trace.row_week),
                                          format_func=lambda w: _week_label(trace, w), container=c3,
                                          help="Follows the last training week (whose rows the implied "
                                          "sensitivities use) until you pick one.")
            focus["week"] = pd.Timestamp(chosen)
    elif step in FWEEK_STEPS:
        options = [_day(w) for w in trace.forecast_periods]
        if options:
            chosen = ui["follow_control"]("Forecast week", "tr_fweek", options, options[0], container=c3,
                                          help="Follows the first forecast week until you pick one.")
            focus["fweek"] = pd.Timestamp(chosen)
    return focus


def _checks(trace: T.BKSTrace, step: str) -> None:
    """The step's checks as a table, with its "How to read" caption."""
    _heading("Checks at this step")
    frame = trace.checks_frame(step)
    if frame.empty:
        st.caption("No checks at this step.")
        return
    _checks_table(frame, with_step=step == "summary")
    _how("HOW_TRACE_CHECKS")


def _checks_table(frame: pd.DataFrame, with_step: bool) -> None:
    cols = (["step"] if with_step else []) + ["check", "status", "kind", "observed", "reference", "tolerance",
                                                "relation", "note"]
    names = {"step": "Step", "check": "Check", "status": "Status", "kind": "Kind", "observed": "Observed",
             "reference": "Reference", "tolerance": "Tolerance", "relation": "Should hold", "note": "Note"}
    out = frame[cols].rename(columns=names)
    config = {
        "Step": _txt("Step", width=170), "Check": _txt("Check", width=300), "Status": _txt("Status", width=60),
        "Kind": _txt("Kind", width=85), "Observed": _num("Observed"), "Reference": _num("Reference"),
        "Tolerance": _num("Tolerance"), "Should hold": _txt("Should hold", width=320),
        "Note": _txt("Note", width=640),
    }
    _table(out, config, height=_rows_height(len(out)), tint="Status")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def ladder_with_direct(trace: T.BKSTrace, ctx: dict[str, Any]) -> pd.DataFrame:
    """The trace's ladder plus the sidebar's direct method as a benchmark row (before the all-zero row, if any).

    The direct row is scored like the others (``ev.recovery`` and the median OOS R² of ``ev.r2``); its
    change columns are empty, because it is not a step of the chain.
    """
    lad = trace.ladder.copy()
    cfg, ev = ctx["cfg"], ctx["ev"]
    method = _ui.method_option_label(cfg.direct.method, cfg.direct)
    rec = getattr(ev, "recovery", {}) or {}
    row = pd.DataFrame(
        {"label": [f"Direct method: {method} (benchmark)"], "spearman": [float(rec.get("spearman", np.nan))],
         "rmse": [float(rec.get("rmse", np.nan))], "median_r2": [float(median_finite(ev.r2))],
         "d_spearman": [np.nan], "d_median_r2": [np.nan],
         "what": ["The sidebar's direct estimator on the same training days, scored the same way (Compare "
                  "methods tab)."]},
        index=pd.Index(["direct"], name=lad.index.name),
    )
    keys = list(lad.index)
    at = keys.index("zero") if "zero" in keys else len(keys)
    return pd.concat([lad.iloc[:at], row, lad.iloc[at:]])


def _summary(ctx: dict[str, Any], ui: dict[str, Any], focus: dict[str, Any]) -> None:
    trace: T.BKSTrace = ctx["trace"]
    st.subheader(T.STEPS["summary"])
    _what([
        ("Inputs", "the cached BKS run of the current settings (panel, training fit, forecast evaluation, implied "
                   "sensitivities) and the simulation's truth."),
        ("Computation", "every step recomputed independently of the package code and compared with the run (the "
                        "checks), and a ladder of sensitivity matrices scored like the Compare methods tab, from "
                        "the true sensitivities down to what BKS delivers."),
        ("Outputs", "where the run departs from what it should be, step by step, with this run's numbers."),
        ("Code", "build_trace of the lab's trace module, cached by LabSession.bks_trace (the lab package's "
                 "session module)."),
    ])
    _answer(trace)

    _heading("Status by step")
    status = trace.status_frame()
    out = status[["label", "reading", "checks", "key_number", "what"]].rename(columns={
        "label": "Step", "reading": "Reading", "checks": "Checks", "key_number": "This run",
        "what": "What it computes"})
    _table(out, {"Step": _txt("Step", width=175), "Reading": _txt("Reading", width=95, help=READING_HELP),
                 "Checks": _txt("Checks", width=90), "This run": _txt("This run", width=470),
                 "What it computes": _txt("What it computes", width=560)}, height=_rows_height(len(out)),
           tint="Reading")
    _how("HOW_TRACE_STATUS")

    lad = ladder_with_direct(trace, ctx)
    refs = [k for k in ("oracle", "direct", "zero") if k in lad.index]
    lad["chart_label"] = [LADDER_CHART_LABELS.get(str(k), str(v)) for k, v in lad["label"].items()]
    ui["show_chart"](
        charts.ladder_chart(lad, highlight="bks_implied", reference=refs, label_col="chart_label",
                            metrics=(("spearman", "Spearman"), ("median_r2", "Median OOS R²")),
                            title="Where the signal is lost: the reference ladder",
                            subtitle="Each row scored like the Compare methods tab: Spearman over all topic-asset "
                                     "pairs, median OOS R² over the forecast window's return days"),
        "fig_tr_ladder",
    )
    _how("HOW_TRACE_LADDER")
    table = pd.DataFrame({
        "Variant": lad["label"].astype(str),
        "Spearman": lad["spearman"].astype(float),
        "Change in Spearman": lad["d_spearman"].astype(float),
        "RMSE": lad["rmse"].astype(float),
        "Median OOS R² (%)": _pct(lad["median_r2"].astype(float)),
        "Change in R² (points)": _pct(lad["d_median_r2"].astype(float)),
        "What it is": lad["what"].astype(str),
    })
    _table(table, {"Variant": _txt("Variant", width=340), "Spearman": _num("Spearman", "%.3f"),
                   "Change in Spearman": _num("Change in Spearman", "%+.3f"), "RMSE": _num("RMSE", "%.4f"),
                   "Median OOS R² (%)": _num("Median OOS R² (%)", "%.1f"),
                   "Change in R² (points)": _num("Change in R² (points)", "%+.1f"),
                   "What it is": _txt("What it is", width=620)}, height=_rows_height(len(table)))
    _how("HOW_TRACE_LADDER_TABLE")

    _heading("Findings")
    _findings(trace)
    _how("HOW_TRACE_FINDINGS")

    _heading("All checks")
    frame = trace.checks_frame()
    _checks_table(frame, with_step=True)
    _how("HOW_TRACE_CHECKS")
    _next("summary")


def _answer(trace: T.BKSTrace) -> None:
    """The lead answer: do the identity checks hold, and where does BKS depart from what it should be."""
    ident = [c for c in trace.checks if c.kind == T.IDENTITY and c.status != T.INFO]
    off = [c for c in ident if c.status == T.OFF]
    departures = [f for f in trace.findings if f["severity"] == "departure"]
    with st.container(border=True):
        if off:
            names = "; ".join(f"{T.STEPS.get(c.step, c.step)}: {c.name}" for c in off)
            st.warning(f"{len(off)} of {len(ident)} identity checks are off: {names}. An identity check recomputes "
                       "a result by its formula; off points to a defect in the code or in the trace.")
        else:
            st.markdown(
                f"**The code does what the formulas say.** All {len(ident)} identity checks hold: each recomputes a "
                "result of the run independently (or tests an identity the method must satisfy) and finds the same "
                "value to rounding."
            )
        if departures:
            steps = []
            for f in departures:
                label = T.STEPS.get(f["step"], f["step"])
                steps.append(f"- {label}: {f['title']}.")
            st.markdown("**Where BKS departs from what it should be:**\n\n" + "\n".join(steps))
            st.markdown(
                "These departures are properties of the method, not coding errors: each is shown with its "
                "reference (what it should be) in the ladder below and in its step."
            )
        else:
            st.markdown("**No departure found:** every step matches its reference within the thresholds of the "
                        "findings.")


def _findings(trace: T.BKSTrace) -> None:
    """Findings as bullets: departures first, then notes, each with its step."""
    items = sorted(trace.findings, key=lambda f: f["severity"] != "departure")
    if not items:
        st.markdown("No finding for this run.")
        return
    lines = []
    for f in items:
        kind = "Departure" if f["severity"] == "departure" else "Note"
        lines.append(f"- **{kind}, {T.STEPS.get(f['step'], f['step'])}: {f['title']}.** {f['text']}")
    st.markdown("\n".join(lines))


# ---------------------------------------------------------------------------
# 1 Inputs
# ---------------------------------------------------------------------------
def _windows(trace: T.BKSTrace) -> list[tuple[Any, Any, str]]:
    m = trace.meta
    return [(m["train_start"], m["train_end"], "Training window"),
            (m["forecast_start"], m["forecast_end"], "Forecast window")]


def _inputs(ctx: dict[str, Any], ui: dict[str, Any], focus: dict[str, Any]) -> None:
    trace: T.BKSTrace = ctx["trace"]
    sim, truth = ctx["sim"], ctx["truth"]
    a_labels, t_labels = ctx["a_labels"], ctx["t_labels"]
    asset, topic = focus["asset"], focus["topic"]
    st.subheader(T.STEPS["inputs"])
    _what([
        ("Inputs", "the simulation's daily attention per topic, the market's daily returns per asset, and the "
                   "truth: the set sensitivities W of the link map and the true sensitivities B_true."),
        ("Computation", "none in this step. B_true is the population regression of each asset's standardised "
                        "return on all topics' standardised shocks at once; the settings below apply to every "
                        "later step."),
        ("Outputs", f"attention a ({len(sim.attention)} days x {len(trace.topics)} topics), returns r "
                    f"({len(trace.assets)} assets) and B_true (topics x assets)."),
        ("Code", "simulate_lab and truth_for_window of the lab's dgp module; the BKS settings from "
                 "bks_pipeline_config of the lab's bks module."),
    ])
    c1, c2 = st.columns(2)
    with c1:
        _heading("Settings in effect")
        settings = trace.meta["settings"].rename(columns={"setting": "Setting", "value": "Value"})
        _table(settings, {"Setting": _txt("Setting", width="medium"), "Value": _txt("Value", width="large")},
               height=_rows_height(len(settings), cap=740))
        _how("HOW_TRACE_SETTINGS")
    with c2:
        _heading("Sizes")
        shapes = trace.meta["shapes"].rename(columns={"quantity": "Quantity", "value": "Value", "note": "Note"})
        _table(shapes, {"Quantity": _txt("Quantity", width="medium"), "Value": _num("Value", "%,d"),
                        "Note": _txt("Note", width="medium")}, height=_rows_height(len(shapes)))
        _how("HOW_TRACE_SHAPES")

    att_col = [c for c in sim.attention.columns if str(c) == topic]
    ret_col = [c for c in sim.market.returns.columns if str(c) == asset]
    panels = []
    if att_col:
        panels.append({"series": {f"Attention: {t_labels.get(topic, topic)}": sim.attention[att_col[0]]},
                       "y_title": "Attention level", "title": f"Attention to {t_labels.get(topic, topic)}"})
    if ret_col:
        panels.append({"series": {f"Daily return: {a_labels.get(asset, asset)}": sim.market.returns[ret_col[0]]},
                       "y_title": "Daily return", "tickformat": ".1%", "zero_line": True,
                       "title": f"Daily return of {a_labels.get(asset, asset)}"})
    ui["show_chart"](
        charts.line_panels(panels, shade=_windows(trace), title="The inputs: attention and returns",
                           subtitle="Simulated attention of the chosen topic and the market return of the chosen "
                                    "asset, every day of the data; shaded: the training and forecast windows"),
        "fig_tr_inputs",
    )
    _how("HOW_TRACE_INPUTS")

    frame = pd.DataFrame({
        "W": truth.W[asset].reindex(trace.topics) if asset in truth.W.columns else np.nan,
        "B_true": trace.variants["oracle"][asset],
        "B_train": trace.B_true_train_units[asset],
    }, index=pd.Index(trace.topics))
    ui["show_chart"](
        charts.grouped_bars(
            frame, labels=t_labels, top_n=MAX_BARS if len(frame) > MAX_BARS else None,
            series_labels={"W": "Set sensitivity W (link map)", "B_true": "True sensitivity B_true",
                           "B_train": "True sensitivity in training units"},
            axis_title="Standardised sensitivity",
            title=f"What the estimate should be: {a_labels.get(asset, asset)}",
            subtitle="Per topic: the link map's set value, the true sensitivity, and the same in the training "
                     "scales the estimators use"),
        "fig_tr_truth",
    )
    _how("HOW_TRACE_TRUTH")
    _checks(trace, "inputs")
    _next("inputs")


# ---------------------------------------------------------------------------
# 2 Alignment and scaling
# ---------------------------------------------------------------------------
def _align(ctx: dict[str, Any], ui: dict[str, Any], focus: dict[str, Any]) -> None:
    trace: T.BKSTrace = ctx["trace"]
    a_labels = ctx["a_labels"]
    asset = focus["asset"]
    weighting = trace.meta.get("asset_weighting", "inverse_vol")
    days = trace.z.index
    if weighting == "none":
        divisor = "1 (no asset weighting: the panel works in return units)"
    elif trace.history == "training":
        divisor = "the asset's training standard deviation, the same number on every day"
    else:
        divisor = "the asset's trailing 252-day volatility, known the day before (ex ante)"
    st.subheader(T.STEPS["align"])
    _what([
        ("Inputs", "the daily attention and returns of step 1."),
        ("Computation", f"the return of day tau is paired with the attention of day tau - {trace.lead_days} (the "
                        f"lead); each return is divided by its divisor d: {divisor}. The implied sensitivities "
                        "later multiply back by one number per asset, the mean divisor over its training return "
                        "days, so the kernel's mix of divisors makes that conversion approximate (u below)."),
        ("Outputs", f"aligned attention and scaled returns r / d on the panel's days ({_day(days[0])} to "
                    f"{_day(days[-1])}, {len(days):,} days); the unit table."),
        ("Code", "narrative_ipca.data.align_inputs, called by build_bks_panel of the lab's bks module."),
    ])
    frame = T.asset_days(trace, ctx["panel"], ctx["sim"], asset)
    styles = {"Training return scale": {"dash": "dash"}, "Full-sample volatility": {"dash": "dot"}}
    panels = [
        {"series": {"Divisor d (the panel's)": frame["divisor"], "Training return scale": frame["ret_scale"],
                    "Full-sample volatility": frame["asset_vol"]},
         "styles": styles, "y_title": "Daily volatility" if weighting != "none" else "Divisor",
         "tickformat": ".2%" if weighting != "none" else None, "title": "Divisor and its references"},
        {"series": {"Scaled return r / d": frame["scaled"]}, "y_title": "Scaled return", "zero_line": True,
         "title": "Scaled daily return"},
    ]
    if weighting == "none":
        panels[0].pop("tickformat")
    ui["show_chart"](
        charts.line_panels(panels, shade=_windows(trace), title=f"Divisor and scaled return: "
                           f"{a_labels.get(asset, asset)}",
                           subtitle="The divisor each daily return is divided by, against the training return "
                                    "scale and the full-sample volatility; shaded: training and forecast windows"),
        "fig_tr_divisor",
    )
    _how("HOW_TRACE_DIVISOR")

    _heading("Units per asset")
    units = trace.units.copy()
    order = [asset] + [a for a in units.index if a != asset] if asset in units.index else list(units.index)
    units = units.loc[order]
    table = pd.DataFrame({"Asset": [a_labels.get(a, a) for a in units.index]})
    cols = [("divisor", "Mean divisor d", "%.4f"), ("ret_scale", "Training return scale", "%.4f"),
            ("asset_vol", "Full-sample volatility", "%.4f"), ("divisor_over_ret_scale", "d / training scale", "%.3f"),
            ("conversion", "Conversion u (1 = exact)", "%.3f"),
            ("conversion_exact", "Exact-unit ratio (1 = exact)", "%.3f"),
            ("kernel_divisor", "Kernel mean divisor", "%.4f"),
            ("conversion_kernel", "Exact-unit ratio with the kernel divisor", "%.3f")]
    config: dict[str, Any] = {"Asset": _txt("Asset", width="medium")}
    for col, name, fmt in cols:
        if col in units.columns:
            table[name] = units[col].to_numpy(dtype=float)
            config[name] = _num(name, fmt)
    if "skipped" in units.columns:
        table["Zero (no training row)"] = np.where(units["skipped"].to_numpy(dtype=bool), "yes", "no")
    _table(table, config)
    _how("HOW_TRACE_UNITS")
    _checks(trace, "align")
    _next("align")


# ---------------------------------------------------------------------------
# 3 Attention shocks
# ---------------------------------------------------------------------------
def _shocks(ctx: dict[str, Any], ui: dict[str, Any], focus: dict[str, Any]) -> None:
    trace: T.BKSTrace = ctx["trace"]
    t_labels = ctx["t_labels"]
    topic = focus["topic"]
    w = trace.shock_window
    st.subheader(T.STEPS["shocks"])
    first = ("under the full history the panel recomputes them on its own days, so its first "
             f"{w} days have none; after that they equal the direct estimator's shocks shifted by the lead"
             if trace.history == "full" else
             "under the training history they start where the direct estimator's shocks start and equal them, "
             "shifted by the lead")
    _what([
        ("Inputs", "the aligned attention of step 2 (attention of the day the return reacts to)."),
        ("Computation", f"z_tau = a_tau - mean(a over the previous {w} days); {first}. The simulation also "
                        "knows each shock's parts: the designed signal the returns load on, the news noise, and "
                        "the slow drift of attention."),
        ("Outputs", "daily shocks z per topic in raw attention units (the panel does not keep them; the trace "
                    "recomputes them)."),
        ("Code", "narrative_ipca.shocks.attention_shocks."),
    ])
    frame = T.topic_days(trace, ctx["panel"], ctx["sim"], ctx["shocks"], topic)
    show_all = ui["control"]("checkbox", "Show all days", "tr_shock_all",
                             help="Off: from three months before the training start to the forecast end.")
    if not show_all:
        lo = pd.Timestamp(trace.meta["train_start"]) - pd.DateOffset(months=3)
        hi = pd.Timestamp(trace.meta["forecast_end"])
        frame = frame.loc[(frame.index >= lo) & (frame.index <= hi)]
    label = t_labels.get(topic, topic)
    panels = [
        {"series": {"BKS shock (panel)": frame["z_bks"], "Direct estimator's shock": frame["z_direct"]},
         "styles": {"Direct estimator's shock": {"dash": "dot"}}, "y_title": "Shock", "zero_line": True,
         "title": "BKS and direct shocks (should coincide)"},
        {"series": {"Signal part": frame["z_signal"], "Noise part (news + slow drift)": frame["z_news"] +
                    frame["z_slow"]}, "y_title": "Shock part", "zero_line": True,
         "title": "The shock split into its signal and noise parts"},
    ]
    ui["show_chart"](
        charts.line_panels(panels, shade=_windows(trace), title=f"Attention shocks: {label}",
                           subtitle="Paired with the return day; shaded: training and forecast windows"),
        "fig_tr_shocks",
    )
    _how("HOW_TRACE_SHOCKS")

    _heading("Shocks per topic")
    st_ = trace.shock_table
    table = pd.DataFrame({
        "Topic": [t_labels.get(t, t) for t in st_.index],
        "Training sd": st_["sd_train"].to_numpy(dtype=float),
        "Population sd": st_["sd_population"].to_numpy(dtype=float),
        "Training / population": st_["ratio"].to_numpy(dtype=float),
        "Correlation with the designed shock": st_["corr_designed"].to_numpy(dtype=float),
        "Attenuation (truth)": st_["attenuation_true"].to_numpy(dtype=float),
        "Signal share of the variance (%)": _pct(st_["signal_share"].to_numpy(dtype=float)),
    })
    _table(table, {"Topic": _txt("Topic", width="medium"), "Training sd": _num("Training sd", "%.4f"),
                   "Population sd": _num("Population sd", "%.4f"),
                   "Training / population": _num("Training / population", "%.3f"),
                   "Correlation with the designed shock": _num("Correlation with the designed shock", "%.3f"),
                   "Attenuation (truth)": _num("Attenuation (truth)", "%.3f"),
                   "Signal share of the variance (%)": _num("Signal share of the variance (%)", "%.1f")})
    _how("HOW_TRACE_SHOCK_TABLE")
    _checks(trace, "shocks")
    _next("shocks")


# ---------------------------------------------------------------------------
# 4 Instruments
# ---------------------------------------------------------------------------
def _instruments(ctx: dict[str, Any], ui: dict[str, Any], focus: dict[str, Any]) -> None:
    trace: T.BKSTrace = ctx["trace"]
    panel = ctx["panel"]
    a_labels, t_labels = ctx["a_labels"], ctx["t_labels"]
    asset, topic, week = focus["asset"], focus["topic"], focus["week"] or trace.row_week
    m = trace.meta
    hl = float(ctx["cfg"].bks.half_life_months)
    st.subheader(T.STEPS["instruments"])
    _what([
        ("Inputs", "the scaled daily returns (step 2) and the shocks (step 3)."),
        ("Computation", f"for week j, asset i and topic k: cov = sum over days tau of k_tau (r_tau - rbar)"
                        f"(z_tau - zbar), over every day up to week j's end minus the last {m['skip_days']} "
                        f"day(s); day weights k_tau proportional to xi^(j - week of tau), xi = {m['xi']:.6f} a week "
                        f"(half-life {hl:g} months), normalised over the days with a return and every shock; no "
                        f"value with fewer than {m['min_days']} such days"
                        + (f" or in the first {m['burn_in_weeks']} weeks (burn-in)" if m.get("burn_in_weeks")
                           else "") + "."),
        ("Outputs", f"one instrument per asset, topic and week ({len(trace.week_ends):,} weeks), in panel units "
                    "(scaled return x raw shock); the reference is the simulation's population covariance "
                    "Cov(z, r) divided by the asset's divisor."),
        ("Code", "narrative_ipca.covariances.build_covariance_panel."),
    ])
    a_name, t_name = a_labels.get(asset, asset), t_labels.get(topic, topic)
    series = T.instrument_series(trace, panel, asset, topic)
    iw = _instrument_week_of(trace, week)
    shade = []
    for role, label in (("training", "Instruments of the training weeks"),
                        ("forecast", "Instruments of the forecast weeks")):
        idx = series.index[series["role"].to_numpy() == role]
        if len(idx):
            shade.append((idx.min(), idx.max(), label))
    panels = [{
        "series": {"Kernel covariance (recomputed)": series["kernel_cov"], "Panel value": series["panel"],
                   "Population reference": series["population"]},
        "styles": {"Panel value": {"mode": "markers"}, "Population reference": {"dash": "dash"}},
        "y_title": "Instrument (panel units)", "zero_line": True,
    }]
    ui["show_chart"](
        charts.line_panels(panels, shade=shade, markers=[(iw, "Chosen week's instrument")] if iw is not None else (),
                           x_title="Instrument week (end)",
                           title=f"Instrument over the weeks: {a_name} and {t_name}",
                           subtitle="The covariance recomputed for every week, the panel's value and the "
                                    "population reference"),
        "fig_tr_instrument",
    )
    _how("HOW_TRACE_INSTRUMENT")

    try:
        prof, stats = T.kernel_profile(trace, panel, asset, topic, week)
    except ValueError as exc:
        st.info(f"No kernel profile for the week ending {_day(week)}: {exc}")
        prof, stats = None, {}
    if prof is not None:
        panel_value = stats.get("panel_value", np.nan)
        cum = {"Cumulative contribution": prof["cumulative"]}
        styles = {"Kernel weight": {"fill": "tozeroy"}}
        if np.isfinite(panel_value):
            cum["Panel value"] = pd.Series(panel_value, index=prof.index)
            styles["Panel value"] = {"dash": "dash"}
        used = prof.index[prof["weight"].to_numpy() > 0]
        view = prof.loc[prof.index >= used.min()] if len(used) else prof
        cum = {k: v.reindex(view.index) for k, v in cum.items()}
        ui["show_chart"](
            charts.line_panels(
                [{"series": {"Kernel weight": view["weight"]}, "styles": styles, "y_title": "Day weight",
                  "tickformat": ".3%", "title": "Weight of each day"},
                 {"series": cum, "styles": styles, "y_title": "Covariance so far", "zero_line": True,
                  "title": "Running sum of the day contributions"}],
                markers=[(m["train_start"], "Training start"), (stats.get("window_end"), "Window end")],
                title=(f"How one instrument is built: week ending {_day(stats.get('instrument_week'))} (rows of "
                       f"{_day(week)})"),
                subtitle=(f"{a_name} and {t_name}; {stats.get('n_days', 0):,} days (effective "
                          f"{_f(stats.get('effective_days'), ',.0f')}), "
                          f"{_f(100 * stats.get('share_before_train', np.nan), '.0f')}% of the weight before the "
                          "training start")),
            "fig_tr_kernel",
        )
        _how("HOW_TRACE_KERNEL")

    x = trace.population["instrument_ref"].stack()
    y = trace.instruments.stack()
    ui["show_chart"](
        charts.identity_scatter(
            x, y, labels={**a_labels, **t_labels}, highlight=asset, highlight_label=a_name,
            point_label="Other assets", x_title="Population reference Cov(z, r) / d (panel units)",
            y_title="Instrument (panel units)",
            title="Instruments against their population value",
            subtitle=f"Every asset and topic, instrument week ending {_day(trace.instrument_week)}"),
        "fig_tr_instr_truth",
    )
    _how("HOW_TRACE_INSTR_TRUTH")

    _heading(f"Instrument row: {a_name}, return week ending {_day(week)}")
    try:
        row = T.instrument_row(trace, panel, ctx["sim"], asset, week)
    except ValueError as exc:
        st.info(f"No instrument row for the week ending {_day(week)}: {exc}")
        row = None
    if row is not None:
        table = pd.DataFrame({
            "Topic": [t_labels.get(t, t) for t in row.index],
            "Panel value": row["panel"].to_numpy(dtype=float),
            "Recomputed": row["brute_force"].to_numpy(dtype=float),
            "Difference": row["difference"].to_numpy(dtype=float),
            "Population reference": row["population"].to_numpy(dtype=float),
            "Signal part": row["signal_part"].to_numpy(dtype=float),
            "Noise part": row["noise_part"].to_numpy(dtype=float),
        })
        _table(table, {"Topic": _txt("Topic", width="medium"), **{c: _num(c, "%.3g") for c in table.columns[1:]}})
        _how("HOW_TRACE_INSTR_ROW")
    _checks(trace, "instruments")
    _next("instruments")


# ---------------------------------------------------------------------------
# 5 Weekly panel
# ---------------------------------------------------------------------------
def _panel(ctx: dict[str, Any], ui: dict[str, Any], focus: dict[str, Any]) -> None:
    trace: T.BKSTrace = ctx["trace"]
    panel = ctx["panel"]
    a_labels, t_labels = ctx["a_labels"], ctx["t_labels"]
    asset, week = focus["asset"], focus["week"] or trace.row_week
    sizes = trace.meta.get("sizes", {})
    data_cfg = getattr(getattr(panel, "pipeline_cfg", None), "data", None)
    min_assets = int(getattr(data_cfg, "min_assets_per_period", 0) or 0)
    st.subheader(T.STEPS["panel"])
    _what([
        ("Inputs", "the weekly instruments (step 4) and the scaled daily returns (step 2)."),
        ("Computation", "the row of asset i in return week t is c_(i,t-1) = [1, the instruments of week t-1], "
                        "and y_(i,t) = the sum of week t's scaled daily returns; a week enters with enough "
                        "assets"
                        + (f" (at least {min_assets})" if min_assets else "")
                        + ". The fit uses the training weeks only and divides each instrument by its standard "
                          "deviation over the training rows (sigma^c, the penalty's scale)."),
        ("Outputs", f"{len(trace.train_periods)} training weeks with {sizes.get('n_obs_train', 'n/a')} rows; the "
                    "forecast weeks' rows for step 7."),
        ("Code", "narrative_ipca.panel.build_panel; IPCAPanel.subset_periods for the training weeks."),
    ])
    m = trace.meta
    extra = [(k, v) for k, v in (("bks_train_days", "BKS training return days"),
                                 ("days_before_train_start", "Days before the training start inside the first "
                                                             "training week"),
                                 ("direct_days_not_in_bks", "Direct training days in no BKS training week"))
             if k in m]
    if extra:
        dates = m.get("direct_days_not_in_bks_dates") or []
        st.caption("Training days of the two routes: " + "; ".join(
            f"{label} {_count(m[k])}" for k, label in extra)
            + (f" ({', '.join(str(d) for d in dates[:5])}{', ...' if len(dates) > 5 else ''})" if dates else "")
            + f". The direct estimator uses {m.get('n_pairs')} training return days; BKS uses the days of its "
            "training weeks, which start on the Monday of the first week.")

    _heading(f"Which instruments each return week uses: {a_labels.get(asset, asset)}")
    pairs = T.pairing_table(trace, panel, ctx["sim"], asset)
    table = pd.DataFrame({
        "Return week": [_day(x) for x in pairs["return_week"]],
        "First day": [_day(x) for x in pairs["first_day"]],
        "Instrument week": [_day(x) for x in pairs["instrument_week"]],
        "Kernel window ends": [_day(x) for x in pairs["window_end"]],
        "Role": pairs["role"].astype(str),
        "Weekly return (panel)": pairs["y_panel"].to_numpy(dtype=float),
        "Recomputed": pairs["y_recomputed"].to_numpy(dtype=float),
        "Raw return (sum)": _pct(pairs["raw_return"].to_numpy(dtype=float)),
        "Assets that week": pairs["n_assets"].to_numpy(dtype=np.int64),
    })
    _table(table, {"Return week": _txt("Return week"), "First day": _txt("First day"),
                   "Instrument week": _txt("Instrument week"), "Kernel window ends": _txt("Kernel window ends"),
                   "Role": _txt("Role", width="small"),
                   "Weekly return (panel)": _num("Weekly return (panel)", "%.4f"),
                   "Recomputed": _num("Recomputed", "%.4f"), "Raw return (sum)": _num("Raw return (sum, %)", "%.2f"),
                   "Assets that week": _num("Assets that week", "%d")}, height=_rows_height(len(table), cap=420))
    _how("HOW_TRACE_PAIRING")

    design = T.design_matrix(trace, panel, ctx["bks_fit"], week)
    ui["show_chart"](
        charts.matrix_heatmap(
            design, row_labels=a_labels, col_labels=_topic_label(t_labels), value_label="Instrument / training sd",
            highlight_rows=[asset] if asset in design.index else (), row_title="Asset", col_title="Instrument",
            title=f"The design of the return week ending {_day(week)}",
            subtitle="Each asset's row: the constant and last week's instruments, divided by their training "
                     "standard deviation (what the fit sees)"),
        "fig_tr_design",
    )
    _how("HOW_TRACE_DESIGN")

    ui["show_chart"](
        charts.grouped_bars(
            trace.stability, labels=t_labels, separate=True, top_n=MAX_BARS if len(trace.stability) > MAX_BARS
            else None,
            series_labels={"within_over_cross": "Change over the weeks / spread across assets",
                           "mean_over_sd": "Mean / standard deviation"},
            title="How much the training instruments move",
            subtitle="Per topic, over the training rows"),
        "fig_tr_stability",
    )
    _how("HOW_TRACE_STABILITY")
    _checks(trace, "panel")
    _next("panel")


def _count(x: Any) -> str:
    """A count, or a list of dates, as short text."""
    if isinstance(x, (list, tuple, pd.Index, np.ndarray)):
        items = [(_day(v) if isinstance(v, (pd.Timestamp, np.datetime64)) else str(v)) for v in x]
        return f"{len(items)}" + (f" ({', '.join(items[:5])}{', ...' if len(items) > 5 else ''})" if items else "")
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------------------
# 6 Fit and lambda
# ---------------------------------------------------------------------------
def _fit(ctx: dict[str, Any], ui: dict[str, Any], focus: dict[str, Any]) -> None:
    trace: T.BKSTrace = ctx["trace"]
    t_labels = ctx["t_labels"]
    b = ctx["cfg"].bks
    st.subheader(T.STEPS["fit"])
    rule = {"tolerance": f"the tolerance rule ({float(b.tolerance):.0%}: the largest lambda whose Sharpe ratio is "
                         "within the tolerance of the best)",
            "argmax": "the argmax rule (the best Sharpe ratio)",
            "fixed": f"none (fixed lambda {trace.lam:g})"}.get(trace.lambda_rule, trace.lambda_rule)
    _what([
        ("Inputs", "the training rows of step 5 (instruments c and weekly returns y)."),
        ("Computation", "Sparse IPCA (BKS Eq. 8): minimise 0.5 sum (y - c Gamma f_t)^2 + lambda N_S sum_l "
                        "sigma^c_l ||Gamma_l|| + sum_t ||f_t||^2 over Gamma (instruments x K) and the weekly "
                        "factors f_t, alternating a factor step and a group-lasso step; a topic whose Gamma row is "
                        f"zero is dropped. lambda comes from a {int(b.n_lambdas)}-point grid by {rule}, scored by the "
                        "annualised in-sample Sharpe ratio of the factors' best mix."),
        ("Outputs", f"Gamma, the {len(trace.factors_in_sample)} training factors, the selected topics and the "
                    "lambda path."),
        ("Code", "narrative_ipca.tuning.tune and narrative_ipca.sparse_ipca.fit_sparse_ipca (via "
                 "fit_bks of the lab's bks module)."),
    ])
    path = trace.path
    selected = list(trace.meta.get("selected_topics", []))
    if path is None:
        st.caption(f"Fixed lambda ({trace.lam:g}): there is no lambda path to show; the charts below describe the "
                   "one fit.")
    else:
        _path_chart(trace, ui, b)
        norms = trace.gamma_path
        if norms is not None:
            ui["show_chart"](
                charts.coefficient_path_chart(
                    norms, selected=selected, labels=_topic_label(t_labels), lam_star=trace.lam,
                    title="Gamma row norms along the lambda path", subtitle="Standardised row norm per instrument"),
                "fig_tr_coef_path",
            )
            _how("HOW_TRACE_COEF_PATH")

    ui["show_chart"](
        charts.matrix_heatmap(
            trace.gamma_std, row_labels=_topic_label(t_labels), value_label="Standardised Gamma",
            highlight_rows=selected, row_title="Instrument", col_title="Factor",
            title="Gamma at the chosen lambda (standardised)",
            subtitle="sigma^c times Gamma: how far a one-standard-deviation move of each instrument shifts each "
                     "factor loading; bold: selected topics"),
        "fig_tr_gamma",
    )
    _how("HOW_TRACE_GAMMA")

    kkt = trace.kkt
    ratio = kkt["ratio"].astype(float)
    nxt = trace.meta.get("next_to_enter")
    if np.isfinite(ratio.to_numpy()).any():
        active = kkt["active"].to_numpy(dtype=bool)
        frame = pd.DataFrame({"kept": np.where(active, ratio, np.nan), "dropped": np.where(~active, ratio, np.nan)},
                             index=kkt.index)
        ui["show_chart"](
            charts.grouped_bars(
                frame, labels=_topic_label(t_labels), reference=(1.0, "Penalty"),
                top_n=MAX_BARS if len(frame) > MAX_BARS else None,
                series_labels={"kept": "Kept (should be 1)", "dropped": "Dropped (should be at most 1)"},
                axis_title="Gradient / penalty", title="Is the fit at its optimum? The group-lasso condition",
                subtitle="Per instrument: the length of the fit's gradient over the penalty"
                         + (f"; next to enter: {_topic_label(t_labels).get(nxt, nxt)} "
                            f"({_f(trace.meta.get('next_to_enter_ratio'), '.3f')})" if nxt else "")),
            "fig_tr_kkt",
        )
        _how("HOW_TRACE_KKT")
    else:
        st.caption("At lambda = 0 there is no penalty, so there is no gradient-over-penalty condition to show.")

    F = trace.factors_in_sample
    names = {c: f"Factor {c[1:]}" for c in F.columns}
    Fr = F.rename(columns=names)
    ui["show_chart"](
        charts.line_panels(
            [{"series": {c: Fr[c] for c in Fr.columns}, "y_title": "Factor value", "zero_line": True,
              "title": "Weekly factors"},
             {"series": {c: Fr[c].cumsum() for c in Fr.columns}, "y_title": "Cumulative", "zero_line": True,
              "title": "Cumulative sum (the drift is the factor's mean)"}],
            title="The training factors", subtitle=f"{len(F)} training weeks; the lambda criterion is the Sharpe "
                                                   "ratio of their best mix"),
        "fig_tr_factors",
    )
    _how("HOW_TRACE_FACTORS")

    if path is not None:
        _heading("The lambda path")
        _path_table(trace)
        _how("HOW_TRACE_PATH_TABLE")
    _checks(trace, "fit")
    _next("fit")


def _band_floors(trace: T.BKSTrace, tolerance: float) -> tuple[float | None, float | None]:
    """The tuner's band floor and the relative-rule floor (addendum), from the path table or recomputed."""
    path = trace.path
    if path is None or trace.lambda_rule != "tolerance":
        return None, None
    crit = path["criterion"].to_numpy(dtype=float)
    if not np.isfinite(crit).any():
        return None, None
    best = float(np.nanmax(crit))
    code = (float(path["band_floor_code"].iloc[0]) if "band_floor_code" in path.columns
            else best - max(1e-9, tolerance) * max(1.0, abs(best)))
    rel = float(path["band_floor_relative"].iloc[0]) if "band_floor_relative" in path.columns else None
    return code, rel


def _path_chart(trace: T.BKSTrace, ui: dict[str, Any], b: Any) -> None:
    path = trace.path
    code, rel = _band_floors(trace, float(b.tolerance))
    kw: dict[str, Any] = {}
    if all(c in path.columns for c in ("null_q05", "null_q50", "null_q95")) and _accepts(charts.lambda_trace_chart,
                                                                                         "null_band"):
        kw["null_band"] = tuple(float(path[c].iloc[0]) for c in ("null_q05", "null_q50", "null_q95"))
    if rel is not None and code is not None and not np.isclose(rel, code) and _accepts(charts.lambda_trace_chart,
                                                                                       "band_floor_alt"):
        kw["band_floor_alt"] = rel
    extra = None
    labels = {"spearman": "Spearman with true sensitivities", "kept_share": "Share of the instruments kept",
              "median_r2": "Median OOS R² of the implied sensitivities"}
    if trace.path_trace is not None and len(trace.path_trace):
        extra = trace.path_trace.set_index("lam")
    ui["show_chart"](
        charts.lambda_trace_chart(
            path, lam_star=trace.lam, band_floor=code, extra=extra, extra_labels=labels if extra is not None else None,
            extra_title="The implied sensitivities refitted at each lambda",
            title="The lambda path and its noise",
            subtitle="Sharpe ratio with one standard error"
                     + (" and the no-signal band" if "null_band" in kw else "")
                     + "; topics selected"
                     + ("; the implied sensitivities refitted at each lambda" if extra is not None else ""),
            **kw),
        "fig_tr_path",
    )
    _how("HOW_TRACE_PATH")
    note = trace.meta.get("path_trace_note")
    if extra is None and note:
        st.caption(f"The recovery panel is not shown: {note}.")


def _path_table(trace: T.BKSTrace) -> None:
    p = trace.path
    table = pd.DataFrame({"lambda": p["lam"].to_numpy(dtype=float)})
    cols = [("criterion", "Sharpe ratio", "%.3f"), ("se", "Standard error", "%.3f"),
            ("n_selected", "Topics", "%d"), ("total_r2", "In-sample R² (%)", "%.1f"),
            ("objective", "Objective", "%,.1f"), ("zero_objective", "Objective of Gamma = 0", "%,.1f"),
            ("n_iter", "Sweeps", "%d")]
    config: dict[str, Any] = {"lambda": _num("lambda", "%.4g")}
    flags = [("chosen", "Chosen"), ("best", "Best"), ("in_band", "In band"), ("edge", "Grid edge"),
             ("above_zero", "Above Gamma = 0"), ("converged", "Converged"), ("sigma_ff_truncated", "Rank cut")]
    for col, name, fmt in cols:
        if col in p.columns:
            vals = p[col].to_numpy(dtype=float)
            table[name] = vals * 100.0 if col == "total_r2" else vals
            config[name] = _num(name, fmt)
    for col, name in flags:
        if col in p.columns:
            table[name] = np.where(p[col].to_numpy(dtype=bool), "yes", "")
            config[name] = _txt(name, width="small")
    for col, name in (("null_q05", "Null 5%"), ("null_q50", "Null median"), ("null_q95", "Null 95%")):
        if col in p.columns:
            table[name] = p[col].to_numpy(dtype=float)
            config[name] = _num(name, "%.2f")
    if trace.path_trace is not None and len(trace.path_trace):
        pt = trace.path_trace.set_index("lam")
        for col, name, scale, fmt in (("kept_share", "Kept share (%)", 100.0, "%.1f"),
                                      ("spearman", "Spearman (implied)", 1.0, "%.3f"),
                                      ("median_r2", "Median OOS R² (implied, %)", 100.0, "%.1f"),
                                      ("gamma_rank", "Gamma rank", 1.0, "%d"),
                                      ("gamma_sv_min", "Gamma's smallest singular value", 1.0, "%.3g")):
            if col in pt.columns:
                vals = pt[col].reindex(p["lam"].to_numpy(dtype=float)).to_numpy(dtype=float) * scale
                table[name] = vals
                config[name] = _num(name, fmt)
    _table(table, config, height=_rows_height(len(table)))


# ---------------------------------------------------------------------------
# 7 Forecast weeks
# ---------------------------------------------------------------------------
def _forecast(ctx: dict[str, Any], ui: dict[str, Any], focus: dict[str, Any]) -> None:
    trace: T.BKSTrace = ctx["trace"]
    a_labels = ctx["a_labels"]
    asset, fweek = focus["asset"], focus["fweek"]
    ridge = trace.meta.get("oos_ridge", np.nan)
    st.subheader(T.STEPS["forecast"])
    _what([
        ("Inputs", "the frozen training Gamma (step 6) and the forecast weeks' rows (step 5)."),
        ("Computation", f"each forecast week t: loadings B = C Gamma from the week's instrument rows C, and the "
                        f"factor f_t = (B'B + {_f(ridge, 'g')} I)^-1 B'y_t fitted to that week's own returns "
                        "(ridge 2; 0 at lambda = 0); fitted = B f_t; the pooled R² is 1 - sum (y - fitted)^2 / "
                        "sum y^2 over all forecast asset-weeks. The reference shuffles the topic instruments "
                        "across assets within each week."),
        ("Outputs", "the tiles' pooled OOS R² and its shuffled reference, per week and per asset."),
        ("Code", "evaluate_bks of the lab's bks module, with narrative_ipca.oos.oos_factor."),
    ])
    weeks = trace.weeks
    if weeks.empty:
        st.info("This run has no forecast week.")
        _checks(trace, "forecast")
        _next("forecast")
        return
    r2_cols = [c for c in ("r2", "r2_shuffled", "r2_exact") if c in weeks.columns]
    ui["show_chart"](
        charts.grouped_bars(
            weeks[r2_cols], orientation="v", percent=True,
            series_labels={"r2": "OOS R² (panel units)", "r2_shuffled": "Instruments shuffled (reference)",
                           "r2_exact": "OOS R² in exact return units"},
            axis_title="R² of the week", title="R² per forecast week",
            subtitle="All assets of each week; factors fitted to the week's own returns"),
        "fig_tr_week_r2",
    )
    _how("HOW_TRACE_WEEK_R2")
    pooled = trace.meta.get("r2_pooled_panel"), trace.meta.get("r2_pooled_exact")
    if pooled[1] is not None:
        st.caption(f"Pooled over the {len(weeks)} forecast weeks: {_ui.fmt_pct(pooled[0])} in panel units (the "
                   f"tiles' number), {_ui.fmt_pct(pooled[1])} in exact return units.")
    fw = fweek if fweek is not None else pd.Timestamp(trace.forecast_periods[0])
    try:
        frame, stats = T.forecast_week(trace, ctx["panel"], ctx["bks_fit"], ctx["res"], fw)
    except ValueError as exc:
        st.info(f"No forecast week {_day(fw)}: {exc}")
        frame, stats = None, {}
    if frame is not None:
        a_name = a_labels.get(asset, asset)
        ui["show_chart"](
            charts.identity_scatter(
                frame["fitted"], frame["realized"], labels=a_labels, highlight=asset if asset in frame.index
                else None, highlight_label=a_name, point_label="Other assets",
                x_title="Fitted weekly return (panel units)", y_title="Realised weekly return (panel units)",
                title=f"Realised against fitted, week ending {_day(fw)}",
                subtitle=f"R² of the week {_ui.fmt_pct(stats.get('r2_week'))} (shuffled "
                         f"{_ui.fmt_pct(stats.get('r2_week_shuffled'))})"),
            "fig_tr_week_scatter",
        )
        _how("HOW_TRACE_WEEK_SCATTER")
    fcols = [c for c in weeks.columns if c.startswith("f") and c[1:].isdigit()]
    ui["show_chart"](
        charts.grouped_bars(
            weeks[fcols], orientation="v", series_labels={c: f"Factor {c[1:]}" for c in fcols},
            axis_title="Factor value", title="Factors of each forecast week",
            subtitle="Fitted to each week's own returns; compare their size with the training factors (step 6)"),
        "fig_tr_oos_factors",
    )
    _how("HOW_TRACE_OOS_FACTORS")
    if frame is not None:
        _heading(f"Per asset, week ending {_day(fw)}")
        order = [asset] + [a for a in frame.index if a != asset] if asset in frame.index else list(frame.index)
        fr = frame.loc[order]
        table = pd.DataFrame({"Asset": [a_labels.get(a, a) for a in fr.index],
                              "Realised": fr["realized"].to_numpy(dtype=float),
                              "Fitted": fr["fitted"].to_numpy(dtype=float),
                              "Residual": fr["residual"].to_numpy(dtype=float)})
        config: dict[str, Any] = {"Asset": _txt("Asset", width="medium")}
        for c in ("Realised", "Fitted", "Residual"):
            config[c] = _num(c, "%.3f")
        for c in fr.columns:
            if c.startswith("beta_"):
                name = f"Loading {c.split('_')[1]}"
                table[name] = fr[c].to_numpy(dtype=float)
                config[name] = _num(name, "%.3f")
        _table(table, config)
        _how("HOW_TRACE_WEEK_TABLE")
    _checks(trace, "forecast")
    _next("forecast")


# ---------------------------------------------------------------------------
# 8 Implied sensitivities
# ---------------------------------------------------------------------------
def _implied(ctx: dict[str, Any], ui: dict[str, Any], focus: dict[str, Any]) -> None:
    trace: T.BKSTrace = ctx["trace"]
    a_labels, t_labels = ctx["a_labels"], ctx["t_labels"]
    asset = focus["asset"]
    a_name = a_labels.get(asset, asset)
    n_pairs = trace.meta.get("n_pairs")
    st.subheader(T.STEPS["implied"])
    _what([
        ("Inputs", f"the fit's Gamma (step 6) and each asset's instrument row c_i = [1, v_i] of the week ending "
                   f"{_day(trace.instrument_week)} (the rows of the last training week, step 5; v_i are its topic "
                   "instruments)."),
        ("Computation", "BKS Eq. 5: the K loadings beta_i = c_i Gamma are turned back into topic covariances "
                        "m_i = P v_i + M Gamma_0', with P the projector onto the K directions of Gamma's topic "
                        "rows (the part of v_i outside them is lost) and M Gamma_0' the constant's share; m_i is "
                        "converted to return units by the asset's mean training divisor; b_i = Sigma_z^+ m_i with "
                        f"Sigma_z the shocks' covariance over the {n_pairs} training days; B_hat = b sd(z) / sd(r) "
                        "in standardised units."),
        ("Outputs", "the BKS-implied topic sensitivities B_hat (topics x assets), scored in the Compare methods "
                    "tab and in the Summary's ladder."),
        ("Code", "the Eq. 5 step of the lab's bks module: implied_topic_covariance for m_i, then the conversion "
                 "to sensitivities."),
    ])
    chain = T.asset_chain(trace, asset, direct=ctx.get("fit"))
    if "bks_ls" in trace.variants and asset in trace.variants["bks_ls"].columns:
        chain["bks_ls"] = trace.variants["bks_ls"][asset].reindex(chain.index).to_numpy(dtype=float)
    m_ls = trace.chain.get("m_ls")
    if isinstance(m_ls, pd.DataFrame) and asset in m_ls.index:
        chain.insert(list(chain.columns).index("m") + 1, "m_ls",
                     m_ls.loc[asset].reindex(chain.index).to_numpy(dtype=float))
    method = _ui.method_option_label(ctx["cfg"].direct.method, ctx["cfg"].direct)
    series = {"B_true": "True sensitivity", "instruments_kernel": "Instruments alone, same-history Sigma_z",
              "bks_ls": "Same betas, least-squares inversion", "B_hat": "BKS-implied",
              "direct": f"Direct: {method}"}
    cols = [c for c in series if c in chain.columns and np.isfinite(chain[c].to_numpy(dtype=float)).any()]
    ui["show_chart"](
        charts.grouped_bars(
            chain[cols], labels=t_labels, series_labels=series, top_n=MAX_BARS if len(chain) > MAX_BARS else None,
            axis_title="Standardised sensitivity", title=f"Implied against true sensitivities: {a_name}",
            subtitle="Per topic: the truth, the instruments alone, the fit's betas inverted by least squares, "
                     "BKS-implied and the sidebar's direct method"),
        "fig_tr_chain",
    )
    _how("HOW_TRACE_CHAIN")

    _heading("How much of the instruments the K directions keep")
    _shares_table(trace, asset, a_name)
    _how("HOW_TRACE_SHARES")

    cap = trace.capture
    frame = pd.DataFrame({"singular_share": cap["singular_share"], "captured": cap["captured"]})
    frame = frame.head(MAX_BARS)
    frame.index = [f"{int(d)}" for d in frame.index]
    ui["show_chart"](
        charts.grouped_bars(
            frame, orientation="v", separate=True, percent=True,
            series_labels={"singular_share": "Share of the instruments' squared norm",
                           "captured": "Share of the direction inside the fit's K directions"},
            title="Where the instruments vary, and what the fit's directions catch",
            subtitle=f"Along the x axis: the principal directions of the instruments across assets, 1 the largest "
                     f"(K = {trace.K})"),
        "fig_tr_capture",
    )
    _how("HOW_TRACE_CAPTURE")

    x = trace.variants["oracle"].stack()
    y = trace.variants["bks_implied"].stack()
    ui["show_chart"](
        charts.identity_scatter(
            x, y, labels={**a_labels, **t_labels}, highlight=asset, highlight_label=a_name, point_label="Other assets",
            x_title="True sensitivity", y_title="BKS-implied sensitivity",
            title="BKS-implied against true sensitivities", subtitle="Every topic-asset pair, standardised units"),
        "fig_tr_implied_scatter",
    )
    _how("HOW_TRACE_IMPLIED_SCATTER")

    sig = _sigma_ratios(trace)
    ui["show_chart"](
        charts.grouped_bars(
            sig[["train", "kernel"]], labels=t_labels, reference=(1.0, "Population"),
            top_n=MAX_BARS if len(sig) > MAX_BARS else None,
            series_labels={"train": "Training days / population", "kernel": "Instruments' kernel days / population"},
            axis_title="Variance ratio", title="The shocks' variance behind Sigma_z",
            subtitle="Per topic: the shock variance over the training days and over the instruments' kernel "
                     "days, each over the population variance"),
        "fig_tr_sigma",
    )
    _how("HOW_TRACE_SIGMA")
    table = getattr(trace, "sigma_table", None)
    if isinstance(table, pd.DataFrame) and not table.empty:
        _heading("Sigma_z by history")
        out = table.copy()
        out.insert(0, "Topic", [t_labels.get(str(t), str(t)) for t in out.index])
        config: dict[str, Any] = {"Topic": _txt("Topic", width="medium")}
        for c in out.columns[1:]:
            if pd.api.types.is_numeric_dtype(out[c]):
                config[c] = _num(_sigma_col(c), "%.3f")
        _table(out, config)
        _how("HOW_TRACE_SIGMA_TABLE")

    _heading(f"The Eq. 5 chain for {a_name}")
    names = {"instrument": "Instrument v", "projected": "Projected P v", "constant_part": "Constant part",
             "m": "Implied covariance m", "m_ls": "Least-squares covariance", "m_ret": "m in return units",
             "b_raw": "Raw b", "B_hat": "BKS-implied", "bks_ls": "Least-squares inversion",
             "B_true": "True", "B_true_train_units": "True, training units", "window_truth": "Window truth",
             "instruments_train": "Instruments alone", "instruments_kernel": "Instruments, same-history Sigma_z",
             "direct": f"Direct: {method}"}
    order = [c for c in chain.columns if c != "bks_ls"]
    if "bks_ls" in chain.columns:
        order.insert(order.index("B_hat") + 1, "bks_ls")
    out = chain[order].rename(columns=names)
    out.insert(0, "Topic", [t_labels.get(t, t) for t in chain.index])
    config = {"Topic": _txt("Topic", width="medium")}
    for c in out.columns[1:]:
        config[c] = _num(c, "%.3g")
    _table(out, config)
    _how("HOW_TRACE_CHAIN_TABLE")
    _checks(trace, "implied")
    _next("implied")


def _sigma_col(name: str) -> str:
    return {"train_over_population": "Training / population", "kernel_over_population": "Kernel / population",
            "train_over_kernel": "Training / kernel",
            "max_corr_diff": "Largest correlation difference, training vs kernel"}.get(name, name.replace("_", " "))


def _sigma_ratios(trace: T.BKSTrace) -> pd.DataFrame:
    """Diagonal of Sigma_z (training, kernel) over the population variance, per topic."""
    sz = trace.chain["sigma_z"]
    pop = np.diag(sz["population"].to_numpy(dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({"train": np.diag(sz["train"].to_numpy(dtype=float)) / pop,
                             "kernel": np.diag(sz["kernel"].to_numpy(dtype=float)) / pop},
                            index=pd.Index(trace.topics))


def _shares_table(trace: T.BKSTrace, asset: str, a_name: str) -> None:
    """Shares of the instruments' squared norm kept: the fit's K directions, the best K, least squares, random."""
    cap = trace.capture
    K = trace.K
    rows = [("The fit's K directions (BKS Eq. 5, orthogonal projection)", cap.get("kept_share")),
            (f"The best {K} directions of the instruments (the most any {K} can keep)", cap.get("best_share"))]
    if "ls_share" in cap:
        rows.append(("Least-squares reconstruction from the same K loadings", cap.get("ls_share")))
    rows.append((f"{K} random directions (on average)", cap.get("random_share")))
    per = cap.get("per_asset_share")
    if isinstance(per, pd.Series) and asset in per.index:
        rows.append((f"{a_name}: the fit's K directions", per.get(asset)))
    table = pd.DataFrame({"Directions": [r[0] for r in rows],
                          "Share kept (%)": [100.0 * float(r[1]) if r[1] is not None else np.nan for r in rows]})
    _table(table, {"Directions": _txt("Directions", width=330), "Share kept (%)": _num("Share kept (%)", "%.1f")},
           height=_rows_height(len(table)))


#: Step key -> renderer.
STEP_RENDERERS: dict[str, Any] = {
    "summary": _summary,
    "inputs": _inputs,
    "align": _align,
    "shocks": _shocks,
    "instruments": _instruments,
    "panel": _panel,
    "fit": _fit,
    "forecast": _forecast,
    "implied": _implied,
}
