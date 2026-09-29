"""Tests of the lab session, the dashboard and the run script (DESIGN.md G.9, G.10, G.13; D71).

* :mod:`narrative_ipca.exposure_lab.session`: stage memoisation by config key.
* ``dashboard/_ui.py``: config from widget values, exposure-table layout,
  link edits (pure helpers, no Streamlit).
* ``dashboard/app.py``: driven headless with ``streamlit.testing.v1.AppTest``
  on the default config, after widget changes, with invalid dates, and with
  a BKS run on a small generic universe.
* ``scripts/run_lab.py``: files written for a small generic config.

The default page needs ``data/market`` and ``data/reference``; those tests are
skipped when the market data store is missing.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib.util
import json
import math
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from narrative_ipca.exposure_lab import reference
from narrative_ipca.exposure_lab.config import (
    STAGES,
    ExposureConfig,
    LabConfig,
    TopicSetConfig,
    UniverseConfig,
    WindowConfig,
)
from narrative_ipca.exposure_lab.session import RUN_LAB_KEYS, LabSession, run_lab
from narrative_ipca.exposure_lab.types import BKSLabResult, DirectFit, MarketData, SimData, WindowEval

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"
if str(ROOT / "dashboard") not in sys.path:
    sys.path.insert(0, str(ROOT / "dashboard"))

import _ui  # noqa: E402

HAS_MARKET = (reference.market_dir() / "asset_returns.parquet").is_file()
needs_market = pytest.mark.skipif(not HAS_MARKET, reason="data/market/asset_returns.parquet is missing")
TIMEOUT = 240


def _generic_cfg(**window: object) -> LabConfig:
    """30 generic assets, 12 generic topics (6 linked): fast, needs no data files."""
    return LabConfig(
        universe=UniverseConfig(asset_source="generic", n_generic_assets=30, seed=0),
        topics=TopicSetConfig(manual="none", n_generic=12, generic_signal_share=0.5),
        exposure=ExposureConfig(beta_1=0.5, seed=0),
        window=WindowConfig(**window) if window else WindowConfig(),
    )


# ---------------------------------------------------------------------------
# session.py
# ---------------------------------------------------------------------------
@needs_market
def test_run_lab_default_returns_all_keys():
    out = run_lab(LabConfig())
    assert set(RUN_LAB_KEYS) <= set(out)
    assert isinstance(out["market"], MarketData)
    assert isinstance(out["simulation"], SimData)
    assert isinstance(out["direct"], DirectFit)
    assert isinstance(out["evaluation"], WindowEval)
    ev = out["evaluation"]
    assert ev.n_days == 20
    assert ev.corr.shape == (55, 20)
    assert np.isfinite(ev.r2).sum() == 55
    assert set(out["timings"]) >= {"market", "simulation", "evaluation", "total"}
    assert out["keys"]["evaluation"] == LabConfig().key("evaluation")


def test_run_lab_with_bks_generic():
    out = run_lab(_generic_cfg(), with_bks=True)
    assert {"bks_panel", "bks_fit", "bks"} <= set(out)
    res = out["bks"]
    assert isinstance(res, BKSLabResult)
    assert res.r2.index.equals(out["market"].assets.index)
    assert len(res.periods) >= 3


def test_session_memoises_by_stage_key():
    s = LabSession()
    cfg = _generic_cfg()
    ev1 = s.evaluation(cfg)
    assert s.evaluation(cfg) is ev1
    assert all(s.stats[st]["misses"] == 1 for st in ("market", "simulation", "truth", "shocks", "direct", "evaluation"))

    # a new forecast window re-runs only the evaluation
    cfg2 = dataclasses.replace(cfg, window=dataclasses.replace(cfg.window, forecast_start="2023-03-06"))
    ev2 = s.evaluation(cfg2)
    assert ev2 is not ev1
    assert s.stats["evaluation"]["misses"] == 2
    assert s.stats["direct"]["misses"] == 1 and s.stats["simulation"]["misses"] == 1

    # a new shock window re-runs truth, shocks, fit and evaluation but not the simulation
    cfg3 = dataclasses.replace(cfg, window=dataclasses.replace(cfg.window, shock_window=3))
    s.evaluation(cfg3)
    assert s.stats["simulation"]["misses"] == 1
    assert s.stats["truth"]["misses"] == 2 and s.stats["shocks"]["misses"] == 2
    assert s.truth(cfg3).shock_window == 3 and s.truth(cfg).shock_window == 5

    # a new exposure seed re-simulates on the cached market
    cfg4 = dataclasses.replace(cfg, exposure=dataclasses.replace(cfg.exposure, seed=1))
    s.simulation(cfg4)
    assert s.stats["market"]["misses"] == 1 and s.stats["simulation"]["misses"] == 2
    assert s.has("simulation", cfg4) and s.peek("bks", cfg) is None


def test_session_computes_a_key_once_and_keeps_timings_per_thread():
    """Two threads asking for the same key compute it once (per-key lock); timings stay per thread (F10)."""
    s = LabSession()
    cfg = _generic_cfg()
    barrier = threading.Barrier(2)
    results: dict[str, object] = {}
    timings: dict[str, dict] = {}

    def work(name: str) -> None:
        barrier.wait()
        results[name] = s.simulation(cfg)
        timings[name] = dict(s.last_timings)

    threads = [threading.Thread(target=work, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert results["a"] is results["b"]
    assert s.stats["simulation"] == {"hits": 1, "misses": 1}
    assert sorted(t["simulation"]["cached"] for t in timings.values()) == [False, True]
    assert s.last_timings == {}  # the main thread made no call
    # the BKS stages keep fewer results
    assert s._limits["bks_fit"] == 2 and s._limits["evaluation"] == 6
    assert LabSession(bks_max_entries=1)._limits["bks_panel"] == 1


def test_session_lru_and_errors():
    s = LabSession(max_entries=2)
    cfgs = [_generic_cfg(forecast_start=d) for d in ("2023-01-02", "2023-02-06", "2023-03-06")]
    for c in cfgs:
        s.evaluation(c)
    assert not s.has("evaluation", cfgs[0]) and s.has("evaluation", cfgs[2])
    with pytest.raises(KeyError):
        LabSession.stage_key("nope", cfgs[0])
    with pytest.raises(ValueError):
        LabSession(max_entries=0)


# ---------------------------------------------------------------------------
# dashboard/_ui.py
# ---------------------------------------------------------------------------
def test_config_from_values_default_matches_labconfig():
    cfg, errors, notes = _ui.config_from_values(_ui.default_values())
    assert errors == [] and notes == []
    assert cfg == LabConfig()


def test_config_round_trip_normalises_lists_and_numbers():
    """D77: JSON lists and int-valued YAML numbers give the same config, hash and stage keys."""
    cfg = dataclasses.replace(
        _generic_cfg(), exposure=ExposureConfig(beta_1=0.5, link_overrides=(("G001", "G_ASSET_001", "strong", -1),))
    )
    rt = LabConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert rt == cfg and hash(rt) == hash(cfg) and rt.hash() == cfg.hash()
    assert rt.exposure.link_overrides == (("G001", "G_ASSET_001", "strong", -1),)
    d = cfg.to_dict()
    d["exposure"]["beta_3"] = 0
    d["exposure"]["noise_df"] = 5
    d["bks"]["half_life_months"] = 69
    d["window"]["train_end"] = dt.date(2022, 12, 30)  # YAML reads an unquoted date as a date
    want = dataclasses.replace(cfg, exposure=dataclasses.replace(cfg.exposure, beta_3=0.0))
    got = LabConfig.from_dict(d)
    assert got == want and hash(got) == hash(want) and got.hash() == want.hash()
    assert all(got.key(stage) == want.key(stage) for stage in STAGES)
    assert isinstance(got.exposure.beta_3, float) and got.window.train_end == "2022-12-30"
    with pytest.raises(ValueError, match="integer"):
        TopicSetConfig(manual="none", n_generic=2.5)


def test_config_from_values_reports_invalid_windows():
    v = _ui.default_values()
    v["sb_forecast_start"] = dt.date(2022, 6, 1)
    cfg, errors, _ = _ui.config_from_values(v)
    assert cfg is None and any("must be after training end" in e for e in errors)

    v = _ui.default_values()
    v["sb_train_start"], v["sb_train_end"] = dt.date(2020, 1, 1), dt.date(2020, 3, 1)
    cfg, errors, _ = _ui.config_from_values(v)
    assert cfg is None and any("250 weekdays" in e for e in errors)

    v = _ui.default_values()
    v["sb_forecast_start"], v["sb_forecast_weeks"] = dt.date(2025, 12, 15), 4
    cfg, errors, notes = _ui.config_from_values(v)
    assert cfg is not None and errors == [] and "past the last day" in notes[0]


def test_config_from_values_subset_and_overrides():
    classes = pd.Series({"A": "Equity", "B": "FX", "C": "Fixed income", "D": "Equity"})
    assert _ui.listed_subset(classes, list(_ui.ASSET_CLASSES), []) is None
    assert _ui.listed_subset(classes, ["Equity"], ["D"]) == ("A",)
    v = _ui.default_values()
    v["sb_asset_classes"], v["sb_drop_assets"] = ["Equity"], ["D"]
    cfg, errors, _ = _ui.config_from_values(v, overrides=(("S1", "A", "weak", -1),), asset_classes=classes)
    assert cfg is None and "at least 2" in errors[0]
    v["sb_asset_classes"], v["sb_drop_assets"] = ["Equity", "FX"], []
    cfg, errors, _ = _ui.config_from_values(v, overrides=(("S1", "A", "weak", -1),), asset_classes=classes)
    assert cfg.universe.listed_assets == ("A", "B", "D")
    assert cfg.exposure.link_overrides == (("S1", "A", "weak", -1),)


def test_exposure_table_blanks_orders_and_flips():
    out = run_lab(_generic_cfg())
    ev, fit, truth, sim = out["evaluation"], out["direct"], out["truth"], out["simulation"]
    assets, topics = sim.market.assets, sim.topics.table
    views = pd.Series("L", index=assets.index)
    views.iloc[0] = "S"
    tbl = _ui.exposure_table("OOS correlation", "Standardised", ev, fit, truth, assets, topics,
                             blank_unselected=True, threshold=0.0, views=views)
    vals, blank = tbl["values"], tbl["blank"]
    first = assets.index[0]
    assert np.allclose(vals.loc[first].to_numpy(), -ev.corr.loc[first, vals.columns].to_numpy(), equal_nan=True)
    assert tbl["prefix"].loc[first] == "S"
    assert blank.equals(~fit.selected.T.loc[vals.index, vals.columns])
    # generic columns sorted by mean |value| with blanks as zero
    score = vals.where(~blank, 0.0).abs().mean(axis=0).to_numpy()
    assert np.all(np.diff(score) <= 1e-12)
    # threshold rule and row order by OOS R2
    tbl2 = _ui.exposure_table("True exposure", "% per 1 sd shock", ev, fit, truth, assets, topics,
                              blank_unselected=False, threshold=0.05, row_mode="OOS R²", max_rows=10)
    assert len(tbl2["values"]) == 10 and tbl2["n_rows_total"] == 30
    assert (tbl2["values"].abs() < 0.05).equals(tbl2["blank"])
    r2 = ev.r2.reindex(tbl2["values"].index).to_numpy()
    assert np.all(np.diff(r2[np.isfinite(r2)]) <= 0)
    assert "Showing the first 10 of 30" in tbl2["subtitle"]


def test_feasibility_summary_and_linked_asset_note():
    s = LabSession()
    cfg = dataclasses.replace(_generic_cfg(), exposure=ExposureConfig(n_betas=1, beta_1=0.9, seed=0))
    sim = s.simulation(cfg)
    fs = _ui.feasibility_summary(sim, cfg.exposure)
    assert fs["n_scaled"] == len(sim.meta["feasibility_scaled_topics"]) > 0
    assert "Feasibility scaling shrank the links" in fs["text"] and "needs no scaling" in fs["text"]
    lab, factor, w_set, w_used = fs["rows"][0]
    assert w_set == pytest.approx(0.9) and w_used == pytest.approx(0.9 * factor)
    # the reported beta 1 is the exact threshold at the current ratios
    b = math.floor(fs["max_beta_1"] * 100) / 100
    below = dataclasses.replace(cfg, exposure=dataclasses.replace(cfg.exposure, beta_1=b))
    above = dataclasses.replace(cfg, exposure=dataclasses.replace(cfg.exposure, beta_1=round(b + 0.02, 2)))
    assert s.simulation(below).meta["feasibility_scaled_topics"] == []
    assert s.simulation(above).meta["feasibility_scaled_topics"] != []
    assert _ui.feasibility_summary(s.simulation(below), below.exposure)["text"] == ""
    # 10 generic topics with 2 linked: most assets have no link, and the note says so
    few = dataclasses.replace(cfg, topics=TopicSetConfig(manual="none", n_generic=10, generic_signal_share=0.2))
    note = _ui.linked_asset_note(s.evaluation(few), s.truth(few))
    assert "of 30 assets are linked to a topic" in note and "linked assets the median OOS R²" in note
    # every asset linked (4 assets, 6 linked topics with 6 links each): no note
    tiny = dataclasses.replace(_generic_cfg(), universe=UniverseConfig(asset_source="generic", n_generic_assets=4))
    assert _ui.linked_asset_note(s.evaluation(tiny), s.truth(tiny)) == ""


def test_sweep_caption_and_cv_estimate():
    s = LabSession()
    cfg = _generic_cfg()
    cap = _ui.sweep_caption(s.sweep(cfg), 4, 20)
    assert cap.startswith("39 consecutive 4-week windows from 2023-01-02 to 2025-12-28")
    late = _generic_cfg(forecast_start="2025-12-15")
    assert s.sweep(late).empty
    assert "No complete 4-week window" in _ui.sweep_caption(s.sweep(late), 4, 13)
    assert _ui.cv_seconds_estimate(520, 55) == pytest.approx(25.0)
    assert _ui.cv_seconds_estimate(20, 55) < 5.0


def test_link_edits_to_overrides():
    links = pd.DataFrame({
        "topic_id": ["S1", "S1", "A4"], "asset_id": ["X", "Y", "X"], "tier": ["strong", "weak", "moderate"],
        "sign": [1, -1, 1], "mechanism": ["m", "m", "m"], "origin": ["default"] * 3,
    })
    frame = _ui.link_edit_frame(links, {"S1": "S1 Energy"}, {"X": "Asset X"})
    assert list(frame["topic"]) == ["S1 Energy", "S1 Energy", "A4"]
    edited = frame.copy()
    edited.loc[1, "tier"] = "none"
    edited.loc[2, "sign"] = -1
    new = _ui.overrides_from_edits(frame, edited)
    assert new == [("S1", "Y", "none", -1), ("A4", "X", "moderate", -1)]
    merged = _ui.merge_overrides({("S1", "Y"): ("S1", "Y", "strong", 1)}, new)
    assert merged[("S1", "Y")] == ("S1", "Y", "none", -1) and len(merged) == 2
    assert _ui.overrides_from_edits(frame, frame.copy()) == []


def test_rollup_and_subsample():
    topics = pd.DataFrame({"group": ["Sector", "Macro", "Generic", "Sector"]}, index=["S1", "A1", "G001", "S2"])
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=["S1", "A1", "G001", "S2"])
    r = _ui.rollup_by_group(s, topics)
    assert list(r.index) == ["Sector", "Macro", "Generic"] and r["Sector"] == 5.0
    est = pd.DataFrame(np.ones((200, 150)))
    linked = pd.DataFrame(False, index=est.index, columns=est.columns)
    linked.iloc[:5, :5] = True
    thin, n = _ui.subsample_pairs(est, linked, max_points=1000)
    kept = np.isfinite(thin.to_numpy())
    assert n == 30000 and kept.sum() == 1000 and kept[:5, :5].all()
    thin2, _ = _ui.subsample_pairs(est, linked, max_points=1000)
    assert np.array_equal(kept, np.isfinite(thin2.to_numpy()))


# ---------------------------------------------------------------------------
# dashboard/app.py (AppTest)
# ---------------------------------------------------------------------------
def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=TIMEOUT)


def _assert_clean(at) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@needs_market
def test_app_default_page_renders():
    at = _app().run()
    _assert_clean(at)
    assert at.title[0].value == "Topic-exposure lab"
    assert [t.label for t in at.tabs] == ["Overview", "Exposure table", "Topic contributions", "BKS", "Lists",
                                         "Data and method"]
    labels = [m.label for m in at.metric]
    for label in ("Median OOS R², estimator", "Median OOS R², oracle", "Median population R²",
                  "Assets with positive OOS R²", "Coverage", "Sign agreement", "MCC", "Spearman",
                  "Share explained by topics", "True share (simulation)", "Largest topic"):
        assert label in labels
    assert len(at.get("plotly_chart")) >= 7
    assert at.tabs[1].get("plotly_chart"), "the exposure table tab has no chart"
    assert any("20 return days" in m.value for m in at.tabs[1].markdown)
    # lists: 55 assets and the 20 manual topics
    frames = [d.value for d in at.tabs[4].dataframe]
    assert any(len(f) == 55 and "Long proxy" in f.columns for f in frames)
    assert any(len(f) == 20 and "Scope" in f.columns for f in frames)
    assert at.tabs[3].info, "BKS tab should ask for a run"
    # the contributions tab opens on the variance share (D76); the heatmap has a colour option
    assert at.radio(key="tc_view").value == "Variance share"
    assert at.selectbox(key="ex_colors").value == next(iter(_ui.HEATMAP_COLORS))
    # no session-state warnings from widgets
    assert not [w for w in at.warning if "Session State" in str(w.value)]
    # the tiles follow the view: return attribution shows the move in percentage points
    at.radio(key="tc_view").set_value("Return attribution").run()
    _assert_clean(at)
    labels = [m.label for m in at.metric]
    for label in ("Realised move", "Explained by topics", "Not explained by topics"):
        assert label in labels
    assert "Share explained by topics" not in labels


@needs_market
def test_app_widget_changes_rerun_cleanly():
    at = _app().run()
    _assert_clean(at)
    at.radio(key="sb_n_betas").set_value(1).run()
    _assert_clean(at)
    assert "sb_beta_1" in [s.key for s in at.slider] and "sb_beta_2" not in [s.key for s in at.slider]
    at.slider(key="sb_beta_1").set_value(0.5).run()
    _assert_clean(at)
    at.selectbox(key="sb_manual").set_value("sector").run()
    _assert_clean(at)
    at.slider(key="sb_n_generic_topics").set_value(50).run()
    _assert_clean(at)
    assert any("61 topics" in m.value for m in at.markdown)
    at.slider(key="sb_forecast_weeks").set_value(12).run()
    _assert_clean(at)
    assert any("60 return days" in m.value for m in at.markdown)
    at.radio(key="sb_asset_source").set_value("generic").run()
    at.slider(key="sb_n_generic_assets").set_value(30).run()
    _assert_clean(at)
    assert any("30 assets" in m.value for m in at.markdown)
    assert at.tabs[1].get("plotly_chart")
    # hidden controls keep their values: back to three betas restores beta_2 and beta_3
    at.radio(key="sb_n_betas").set_value(3).run()
    _assert_clean(at)
    assert at.slider(key="sb_beta_2").value == pytest.approx(0.15)
    assert at.slider(key="sb_beta_1").value == pytest.approx(0.5)
    # exposure table options
    at.selectbox(key="ex_metric").set_value("OOS contribution (% points)").run()
    at.checkbox(key="ex_ls_view").check().run()
    at.selectbox(key="ex_row_order").set_value("OOS R²").run()
    _assert_clean(at)


@needs_market
def test_app_generic_topics_come_back_after_none():
    """m8: the clamp to 10 generic topics under 'None' does not overwrite the user's own value."""
    at = _app().run()
    assert any("L = 20 manual + 0 generic = 20 topics" in c.value for c in at.sidebar.caption)
    at.selectbox(key="sb_manual").set_value("none").run()
    _assert_clean(at)
    assert at.slider(key="sb_n_generic_topics").value == 10
    at.selectbox(key="sb_manual").set_value("both").run()
    _assert_clean(at)
    assert at.slider(key="sb_n_generic_topics").value == 0
    assert any("L = 20 manual + 0 generic = 20 topics" in c.value for c in at.sidebar.caption)


@needs_market
def test_app_link_edits_add_and_reset():
    at = _app().run()
    assert any("95 links" in m.value for m in at.markdown)
    at.selectbox(key="lists_add_topic").set_value("S1")
    at.selectbox(key="lists_add_asset").set_value("GLOBAL_DURATION")
    at.selectbox(key="lists_add_sign").set_value(-1)
    at.button(key="lists_add").click().run()
    _assert_clean(at)
    assert any("96 links" in m.value for m in at.markdown)
    assert at.session_state["link_overrides"] == {("S1", "GLOBAL_DURATION"): ("S1", "GLOBAL_DURATION", "strong", -1)}
    assert any("1 session link edit(s) active" in c.value for c in at.sidebar.caption)
    at.button(key="lists_reset").click().run()
    _assert_clean(at)
    assert any("95 links" in m.value for m in at.markdown)


@needs_market
def test_app_invalid_dates_show_error_not_traceback():
    at = _app().run()
    at.date_input(key="sb_forecast_start").set_value(dt.date(2022, 6, 1)).run()
    assert not at.exception
    assert any("must be after training end" in e.value for e in at.error)
    at.date_input(key="sb_forecast_start").set_value(dt.date(2023, 1, 2)).run()
    _assert_clean(at)


def test_app_bks_run_on_small_generic_config():
    at = _app().run()
    at.radio(key="sb_asset_source").set_value("generic").run()
    at.slider(key="sb_n_generic_assets").set_value(30).run()
    at.selectbox(key="sb_manual").set_value("none").run()
    at.slider(key="sb_n_generic_topics").set_value(12).run()
    at.slider(key="sb_signal_share").set_value(0.5).run()
    _assert_clean(at)
    assert any("12 topics" in m.value for m in at.markdown)
    at.button(key="sb_run_bks").click().run()
    _assert_clean(at)
    bks = at.tabs[3]
    assert "Chosen lambda" in [m.label for m in bks.metric]
    assert len(bks.get("plotly_chart")) >= 4
    assert not bks.warning, [w.value for w in bks.warning]
    # a new forecast window re-evaluates the cached fit: not stale
    at.slider(key="sb_forecast_weeks").set_value(8).run()
    _assert_clean(at)
    assert not [w for w in at.tabs[3].warning if "Settings changed" in w.value]
    # a new K makes the stored result stale until the next run
    at.slider(key="sb_bks_K").set_value(2).run()
    assert any("Settings changed" in w.value for w in at.tabs[3].warning)
    at.button(key="bks_run_tab").click().run()
    _assert_clean(at)
    assert not [w for w in at.tabs[3].warning if "Settings changed" in w.value]
    assert at.tabs[3].metric[1].value == "2"
    assert "bks_requested" not in at.session_state  # cleared once the run ended
    # F2: the BKS tab compares with the current direct fit, not a copy stored with the BKS run
    at.selectbox(key="sb_method").set_value("ridge").run()
    _assert_clean(at)
    est = next(m.value for m in at.metric if m.label == "Median OOS R², estimator")
    assert any(f"direct estimator {est} (daily, current settings)" in c.value for c in at.tabs[3].caption)
    # the shuffled-instrument reference sits next to the pooled OOS R2
    labels = [m.label for m in at.tabs[3].metric]
    assert "Same, instruments shuffled" in labels and "Median OOS R², BKS / direct" not in labels
    # F1: a fixed lambda of 0 no longer shows an OOS R2 of zero
    at.radio(key="sb_bks_rule").set_value("fixed").run()
    at.number_input(key="sb_bks_lam").set_value(0.0).run()
    at.button(key="sb_run_bks").click().run()
    _assert_clean(at)
    pooled = next(m.value for m in at.tabs[3].metric if m.label == "Pooled OOS R² (weekly)")
    assert pooled not in ("0.0%", "-0.0%", "n/a")
    # F10: another browser session whose BKS fit settings match (the fit is in the shared cache; only the
    # forecast length differs) does not adopt this session's run; it asks for its own
    other = _app().run()
    other.radio(key="sb_asset_source").set_value("generic").run()
    other.slider(key="sb_n_generic_assets").set_value(30).run()
    other.selectbox(key="sb_manual").set_value("none").run()
    other.slider(key="sb_n_generic_topics").set_value(12).run()
    other.slider(key="sb_signal_share").set_value(0.5).run()
    other.slider(key="sb_bks_K").set_value(2).run()
    other.radio(key="sb_bks_rule").set_value("fixed").run()
    other.number_input(key="sb_bks_lam").set_value(0.0).run()
    _assert_clean(other)
    assert other.tabs[3].info and "Chosen lambda" not in [m.label for m in other.tabs[3].metric]
    other.button(key="bks_run_tab").click().run()  # its own run reuses the cached fit
    _assert_clean(other)
    assert "Chosen lambda" in [m.label for m in other.tabs[3].metric]
    assert next(m.value for m in other.tabs[3].metric if m.label == "Pooled OOS R² (weekly)") not in ("0.0%", "n/a")


# ---------------------------------------------------------------------------
# scripts/run_lab.py
# ---------------------------------------------------------------------------
def _run_lab_script():
    spec = importlib.util.spec_from_file_location("run_lab_script", ROOT / "scripts" / "run_lab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_lab_script_writes_outputs(tmp_path, capsys):
    cfg_file = tmp_path / "lab.json"
    d = _generic_cfg().to_dict()
    d["exposure"]["link_overrides"] = [["G001", "G_ASSET_001", "strong", -1]]
    cfg_file.write_text(json.dumps(d), encoding="utf-8")
    out_dir = tmp_path / "out"
    mod = _run_lab_script()
    assert mod.main(["--config", str(cfg_file), "--out", str(out_dir), "--bks", "--quiet"]) == 0
    for name in ("exposure_corr.csv", "exposures.csv", "r2.csv", "contributions.csv", "recovery.json", "sweep.csv",
                 "summary.json", "bks_r2.csv", "bks_contrib.csv"):
        assert (out_dir / name).is_file(), name
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_assets"] == 30 and summary["n_topics"] == 12
    assert summary["config"]["exposure"]["link_overrides"] == [["G001", "G_ASSET_001", "strong", -1]]
    corr = pd.read_csv(out_dir / "exposure_corr.csv", index_col=0)
    assert corr.shape == (30, 12)
    contrib = pd.read_csv(out_dir / "contributions.csv")
    assert len(contrib) == 30 * 12
    assert "median OOS R2" in capsys.readouterr().out
    cfg = LabConfig.from_dict(d)  # the configs normalise lists themselves (D77); no helper in the script
    assert cfg.exposure.link_overrides == (("G001", "G_ASSET_001", "strong", -1),)
    assert not hasattr(mod, "normalise_config")
