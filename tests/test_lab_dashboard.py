"""Tests of the lab session, the dashboard and the run script (DESIGN.md G.9, G.10, G.13; D71).

* :mod:`narrative_ipca.exposure_lab.session`: stage memoisation by config key.
* ``dashboard/_ui.py``: config from widget values, the training window from
  its cut-off and length, the "Settings in use" table of the Real data page,
  exposure-table layout and the blank rule that follows the view (D69), the
  "How to read" texts and guide labels, link edits (pure helpers, no
  Streamlit).
* ``dashboard/app.py``: driven headless with ``streamlit.testing.v1.AppTest``
  on the default config, after widget changes, with invalid dates, with a
  BKS run on a small generic universe, in the Compare methods tab before and
  after a BKS run (G.15), on the Real data page with the shared sidebar and on
  the BKS trace page (G.16, D90: every step, a refused run, an evicted fit,
  defects kept apart from departures); a "How to read" guide with examples
  next to every chart, and explanations in two elements only (G.9, D91).
* ``scripts/run_lab.py``: files written for a small generic config.

The default page needs ``data/market`` and ``data/reference``; those tests are
skipped when the market data store is missing.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import importlib.util
import json
import math
import re
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from narrative_ipca.exposure_lab import reference
from narrative_ipca.exposure_lab.bks import IMPLIED_NOTE
from narrative_ipca.exposure_lab.config import (
    STAGES,
    DirectConfig,
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
def test_config_from_values_default_uses_dashboard_windows():
    """The dashboard's own time windows (owner request 2026-09-29); everything else is ``LabConfig()``."""
    v = _ui.default_values()
    assert v["sb_train_end"] == dt.date(2025, 6, 30) and v["sb_train_months"] == 6
    assert v["sb_forecast_start"] == dt.date(2025, 7, 1) and v["sb_forecast_weeks"] == 4
    assert "sb_train_start" not in v
    cfg, errors, notes = _ui.config_from_values(v)
    assert errors == [] and notes == []
    window = WindowConfig(train_start="2025-01-01", train_end="2025-06-30", forecast_start="2025-07-01",
                          forecast_weeks=4)
    assert cfg.window == window
    assert cfg == dataclasses.replace(LabConfig(), window=window)
    assert LabConfig().window.train_end == "2022-12-30"  # the library default stays for scripts and tests


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

    # a cut-off 12 weekdays after the data start: the window is clipped and too short
    v = _ui.default_values()
    v["sb_train_end"], v["sb_train_months"] = dt.date(2015, 1, 20), 1
    cfg, errors, _ = _ui.config_from_values(v)
    assert cfg is None and any("has 13 weekdays" in e and "at least 21" in e and "one month" in e for e in errors)

    # one month is accepted (D81)
    v = _ui.default_values()
    v["sb_train_end"], v["sb_train_months"], v["sb_forecast_start"] = dt.date(2022, 12, 30), 1, dt.date(2023, 1, 2)
    cfg, errors, _ = _ui.config_from_values(v)
    assert cfg is not None and not errors
    assert (cfg.window.train_start, cfg.window.train_end) == ("2022-11-30", "2022-12-30")

    v = _ui.default_values()
    v["sb_train_end"] = None
    cfg, errors, _ = _ui.config_from_values(v)
    assert cfg is None and "Choose a training end" in errors[0]

    # the first day of the data as cut-off: the real reason (too short), not "outside the data"
    v = _ui.default_values()
    v["sb_train_end"], v["sb_train_months"], v["sb_forecast_start"] = dt.date(2015, 1, 2), 1, dt.date(2015, 1, 5)
    cfg, errors, _ = _ui.config_from_values(v)
    assert cfg is None and errors == [
        "The training window 2015-01-02 to 2015-01-02 has 1 weekday; it needs at least 21 (about one month). "
        "Move the cut-off to 2015-02-06 or later."
    ]
    # 21 weekdays at the data start, but the first w = 5 have no shock: 16 shock days are too few
    v["sb_train_end"], v["sb_forecast_start"] = dt.date(2015, 1, 30), dt.date(2015, 2, 2)
    cfg, errors, _ = _ui.config_from_values(v)
    assert cfg is None and len(errors) == 1
    assert "has 16 weekdays with a topic shock" in errors[0] and "first shock is on 2015-01-09" in errors[0]
    v["sb_train_end"], v["sb_forecast_start"] = _ui.earliest_train_end(5), dt.date(2015, 2, 9)
    cfg, errors, _ = _ui.config_from_values(v)
    assert cfg is not None and not errors
    assert _ui.shock_days(cfg.window.train_start, cfg.window.train_end, 5) == 21
    assert [_ui.earliest_train_end(w) for w in (1, 5, 20)] == [dt.date(2015, 2, 2), dt.date(2015, 2, 6),
                                                                dt.date(2015, 2, 27)]
    assert _ui.first_shock_day(20) == dt.date(2015, 1, 30)
    assert (_ui.plural(1, "weekday"), _ui.plural(2, "weekday"), _ui.plural(1, "consecutive 4-week window")) == (
        "1 weekday", "2 weekdays", "1 consecutive 4-week window")

    v = _ui.default_values()
    v["sb_forecast_start"], v["sb_forecast_weeks"] = dt.date(2025, 12, 15), 4
    cfg, errors, notes = _ui.config_from_values(v)
    assert cfg is not None and errors == [] and "past the last day" in notes[0]


def test_training_window_from_cut_off_and_length():
    """Training start = (cut-off + 1 day) - length; extended to 21 weekdays; clipped at the data start."""
    assert [_ui.train_months_label(m) for m in _ui.TRAIN_MONTHS] == [
        "1 month", "2 months", "3 months", "4 months", "6 months", "9 months", "1 year", "18 months", "2 years",
        "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10 years",
    ]
    tw = _ui.training_window(dt.date(2025, 6, 30), 6)
    assert (tw["start"], tw["end"], tw["n_days"]) == (dt.date(2025, 1, 1), dt.date(2025, 6, 30), 129)
    assert not tw["extended"] and not tw["clipped"] and tw["note"] == ""
    assert tw["text"] == "Training window: 2025-01-01 to 2025-06-30 (129 weekdays)."

    # February has 20 weekdays: the start moves back one weekday to reach 21
    feb = _ui.training_window("2025-02-28", 1)
    assert feb["nominal_start"] == dt.date(2025, 2, 1) and feb["extended"] and not feb["clipped"]
    assert (feb["start"], feb["n_days"]) == (dt.date(2025, 1, 31), 21)
    assert "moved back from 2025-02-01" in feb["text"] and "21 weekdays" in feb["text"]
    v = _ui.default_values()
    v["sb_train_end"], v["sb_train_months"], v["sb_forecast_start"] = dt.date(2025, 2, 28), 1, dt.date(2025, 3, 3)
    cfg, errors, _ = _ui.config_from_values(v)
    assert not errors and (cfg.window.train_start, cfg.window.train_end) == ("2025-01-31", "2025-02-28")

    # 10 years back from 2020-06-30 starts before the data: clipped to 2015-01-02
    ten = _ui.training_window("2020-06-30", 120)
    assert ten["nominal_start"] == dt.date(2010, 7, 1) and ten["clipped"]
    assert (ten["start"], ten["n_days"]) == (dt.date(2015, 1, 2), len(pd.bdate_range("2015-01-02", "2020-06-30")))
    assert "10 years would start on 2010-07-01, before the data" in ten["text"] and "2015-01-02" in ten["text"]
    v["sb_train_end"], v["sb_train_months"], v["sb_forecast_start"] = dt.date(2020, 6, 30), 120, dt.date(2020, 7, 1)
    cfg, errors, _ = _ui.config_from_values(v)
    assert not errors and (cfg.window.train_start, cfg.window.train_end) == ("2015-01-02", "2020-06-30")

    # a mid-month cut-off: one year back to the day after
    assert _ui.training_window("2024-03-15", 12)["start"] == dt.date(2023, 3, 16)


def _settings_dict(table: pd.DataFrame) -> dict[str, str]:
    return dict(zip(table["Setting"], table["Value"]))


def test_settings_in_use_table():
    """The Real data page's table: time windows, direct estimator, BKS model and asset selection only."""
    ref = reference.load_assets()
    names = ref["name"].to_dict()
    v = _ui.default_values()
    t = _ui.settings_in_use(v, ref["asset_class"], names)
    assert list(t.columns) == ["Group", "Setting", "Value"]
    assert list(dict.fromkeys(t["Group"])) == ["Time windows", "Direct estimator", "BKS model", "Asset selection"]
    got = _settings_dict(t)
    assert got["Training end (cut-off)"] == "2025-06-30" and got["Training length"] == "6 months"
    assert got["Training window"] == "2025-01-01 to 2025-06-30 (129 weekdays)"
    assert got["Forecast start"] == "2025-07-01" and got["Forecast length"] == "4 weeks (2025-07-01 to 2025-07-28)"
    assert got["Shock window w"] == "5 days"
    assert got["Method"] == _ui.METHOD_LABELS["elastic_net"] and got["Penalty rule"] == _ui.PENALTY_LABELS["universal"]
    assert "Penalty alpha" not in got and "L1 ratio" in got
    assert got["Factors K"] == "3" and "weekly xi = " in got["Kernel half-life"]
    assert got["Lambda rule"].startswith("Tolerance rule") and got["Lambda grid"].startswith("12 points")
    assert got["Covariance history"] == "Full history before the cut-off"
    assert got["Listed assets"] == f"{len(ref)} of {len(ref)}"
    assert got["Asset classes"] == "all" and got["Left-out assets"] == "none"
    # simulation-only settings are not listed
    assert not t["Setting"].str.contains("Price source|seed|Beta|Generic", case=False).any()
    assert all(isinstance(x, str) for x in t["Value"])

    # a fixed penalty, a February window, a subset of assets and a generic simulation universe
    first = str(ref.index[0])
    v.update(sb_penalty="fixed", sb_alpha=0.1, sb_train_end=dt.date(2025, 2, 28), sb_train_months=1,
             sb_asset_classes=["Equity"], sb_drop_assets=[first], sb_asset_source="generic", sb_bks_rule="fixed",
             sb_bks_lam=0.0)
    got = _settings_dict(_ui.settings_in_use(v, ref["asset_class"], names))
    assert got["Penalty alpha"] == "0.100" and "moved back from 2025-02-01" in got["Training window"]
    n_eq = int(((ref["asset_class"] == "Equity") & (ref.index != first)).sum())
    assert got["Listed assets"].startswith(f"{n_eq} of {len(ref)}")
    assert "real data uses the listed assets" in got["Listed assets"]
    assert got["Asset classes"] == "Equity" and got["Left-out assets"] == names[first]
    assert got["Lambda rule"].endswith("lambda = 0")
    v.update(sb_bks_history="training")
    assert _settings_dict(_ui.settings_in_use(v))["Covariance history"] == "Training window only"
    # ridge and the oracle
    v.update(sb_method="ridge", sb_ridge_gcv=True)
    got = _settings_dict(_ui.settings_in_use(v))
    assert got["Ridge lambda"] == "chosen by generalised cross-validation" and "Penalty rule" not in got
    assert got["Listed assets"].startswith("all listed assets")
    v.update(sb_method="oracle")
    assert "not available on real data" in _settings_dict(_ui.settings_in_use(v))["Method"]


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
                             blank_rule_on=True, threshold=0.0, views=views)
    vals, blank = tbl["values"], tbl["blank"]
    first = assets.index[0]
    assert np.allclose(vals.loc[first].to_numpy(), -ev.corr.loc[first, vals.columns].to_numpy(), equal_nan=True)
    assert tbl["prefix"].loc[first] == "S"
    assert blank.equals(~fit.selected.T.loc[vals.index, vals.columns])
    # generic columns sorted by mean |value| with blanks as zero
    score = vals.where(~blank, 0.0).abs().mean(axis=0).to_numpy()
    assert np.all(np.diff(score) <= 1e-12)
    # threshold rule and row order by OOS R2
    tbl2 = _ui.exposure_table("True sensitivity", "% per 1 sd shock", ev, fit, truth, assets, topics,
                              blank_rule_on=False, threshold=0.05, row_mode="OOS R²", max_rows=10)
    assert len(tbl2["values"]) == 10 and tbl2["n_rows_total"] == 30
    assert (tbl2["values"].abs() < 0.05).equals(tbl2["blank"])
    r2 = ev.r2.reindex(tbl2["values"].index).to_numpy()
    assert np.all(np.diff(r2[np.isfinite(r2)]) <= 0)
    assert "Showing the first 10 of 30" in tbl2["subtitle"]


def test_blank_rule_follows_the_view():
    """Owner request 2026-09-30 (D69 amended): the selection rule for the estimated sensitivity, correlation and
    contribution; |B_true| < tau (standardised, in both units) for the true sensitivity; no link for W."""
    out = run_lab(_generic_cfg())
    ev, fit, truth, sim = out["evaluation"], out["direct"], out["truth"], out["simulation"]
    assets, topics = sim.market.assets, sim.topics.table
    tau = float(fit.meta["select_tau"])
    selected = fit.selected.T
    for metric in ("OOS correlation", "Estimated sensitivity", "OOS contribution (% points)"):
        t = _ui.exposure_table(metric, "Standardised", ev, fit, truth, assets, topics)
        assert t["blank"].equals(~selected.loc[t["blank"].index, t["blank"].columns]), metric
        assert "Blank: pairs the estimator did not select." in t["subtitle"]
        assert _ui.blank_rule(metric, tau)["label"] == "Blank pairs the estimator did not select"
    # the true sensitivity: the threshold is tested in standardised units whatever the unit shown
    sensitive = (truth.B_true.abs() >= tau).T
    for units in _ui.EXPOSURE_UNITS:
        t = _ui.exposure_table("True sensitivity", units, ev, fit, truth, assets, topics)
        shown = ~t["blank"]
        assert shown.equals(sensitive.loc[shown.index, shown.columns]), units
        missed = shown & ~selected.loc[shown.index, shown.columns]
        assert int(missed.to_numpy().sum()) > 0  # the True view shows pairs the estimator missed
        assert f"|true sensitivity| below {tau:g} (standardised)" in t["subtitle"]
    label = _ui.blank_rule("True sensitivity", 0.05)["label"]
    assert label == "Blank pairs below the true-sensitivity threshold (0.05, standardised)"
    # the set sensitivity: exactly the linked pairs, whatever the estimator selected
    linked = (truth.W_unscaled != 0).T
    t = _ui.exposure_table("Set sensitivity (W)", "% per 1 sd shock", ev, fit, truth, assets, topics)
    assert (~t["blank"]).equals(linked.loc[t["blank"].index, t["blank"].columns])
    assert "Blank: pairs with no link." in t["subtitle"]
    assert _ui.blank_rule("Set sensitivity (W)", tau)["label"] == "Blank pairs with no link"
    # the |value| threshold applies on top, in the units shown; the rule can be switched off
    t = _ui.exposure_table("Set sensitivity (W)", "Standardised", ev, fit, truth, assets, topics, threshold=0.2)
    assert t["blank"].equals(~linked.loc[t["blank"].index, t["blank"].columns] | (t["values"].abs() < 0.2))
    t = _ui.exposure_table("True sensitivity", "Standardised", ev, fit, truth, assets, topics, blank_rule_on=False)
    assert not t["blank"].to_numpy().any() and t["subtitle"].startswith("No blank rule.")
    # a run's own tau
    t = _ui.exposure_table("True sensitivity", "Standardised", ev, fit, truth, assets, topics, tau=0.1)
    assert (~t["blank"]).equals((truth.B_true.abs() >= 0.1).T.loc[t["blank"].index, t["blank"].columns])


def test_how_to_read_captions():
    """Owner request 2026-09-30: lead, then bullets that each end with one "Example:" sentence."""
    text = _ui.how_to_read("How to read this:", [("First item.", "one."), ("Second item.", "two.")])
    assert text == "How to read this:\n\n- First item. Example: one.\n- Second item. Example: two."
    with pytest.raises(ValueError):
        _ui.how_to_read("How to read this:", [])
    with pytest.raises(ValueError):
        _ui.how_to_read("How to read this:", [("Text.", "")])
    # the guide toggle (D91): the lead without its colon is the label, the bullets are the body
    assert _ui.guide_label("How to read this") == ":blue[:material/help_outline: How to read this]"
    assert _ui.guide_parts(text) == (_ui.guide_label("How to read this"),
                                     "- First item. Example: one.\n- Second item. Example: two.")
    for bad in ("How to read this:\n- First item. Example: one.",  # no blank line after the lead
                "Read this:\n\n- First item. Example: one.",
                "How to read x (y):\n\n- First item. Example: one.",
                "How to read this:\n\nFirst item. Example: one."):
        with pytest.raises(ValueError):
            _ui.guide_parts(bad)
    blocks = [_ui.how_to_read(*getattr(_ui, n)) for n in dir(_ui) if n.startswith("HOW_")]
    blocks += [_ui.how_cell_metric(m) for m in _ui.METRICS]
    blocks += [_ui.how_r2_bars("estimator"), _ui.how_compare_table("the 6 consecutive 4-week windows"),
               _ui.how_compare_dots(True), _ui.how_compare_dots(False), _ui.how_compare_sweep("", True),
               _ui.how_bks_implied(69.0), _ui.how_bks_history(69.0)]
    for b in blocks:
        lead, _, body = b.partition("\n\n")
        assert lead.startswith("How to read") and lead.endswith(":"), lead
        assert re.fullmatch(r"How to read [^():]+:", lead) and len(lead) <= 46, lead  # a short, static label
        _ui.guide_parts(b)
        lines = body.split("\n")
        assert lines and all(ln.startswith("- ") and ln.count(" Example: ") == 1 for ln in lines), lead
        assert not any(c in b for c in "$*`~§"), lead  # no markdown or LaTeX surprises in st.caption
        assert "exposure" not in b.lower() and not re.search(r"\bD\d\d\b", b), lead  # plain words, no D-codes
    # the model block keeps its four views, each with an example that follows the code (G.8 point 4)
    two = _ui.how_to_read(*_ui.HOW_TWO_VIEWS)
    assert two.count("\n- ") == 4 and "one tenth of the asset's return on every day" in two
    # ... and keeps the owner's wording: the bullets only gained their examples
    texts = [text for text, _ in _ui.HOW_TWO_VIEWS[1]]
    assert texts[0] == ("Variance share (default): each topic's share of the window's day-to-day return "
                        "variation. It uses every day of the window.")
    assert texts[3] == "Diamonds use the true sensitivities of the simulation."
    # the metric caption follows the view: the chosen metric, and the unit bullet for the sensitivities only
    assert "% per 1 sd shock:" in _ui.how_cell_metric("True sensitivity")
    assert "% per 1 sd shock:" not in _ui.how_cell_metric("OOS correlation")
    # the link map's caveat is a visible line, not part of its guide
    assert "illustrative" not in _ui.how_to_read(*_ui.HOW_LINK_MAP) and "illustrative" in _ui.LINK_MAP_CAVEAT


@needs_market
def test_how_to_read_examples_hold_on_defaults():
    """DESIGN.md G.9 "How to read" rule 1: the shared example set holds on the dashboard defaults."""
    cfg, _, _ = _ui.config_from_values(_ui.default_values(), (), reference.load_assets()["asset_class"])
    s = LabSession()
    truth, fit, ev = s.truth(cfg), s.direct(cfg), s.evaluation(cfg)
    a, k = _ui.DEFAULT_CONTRIB_ASSET, "S1"

    def pp(x: float) -> float:
        return round(float(x) * 100, 2)

    assert s.simulation(cfg).lead_days == 0 and ev.n_days == 20
    assert round(float(fit.ret_scale[a]) * 100, 2) == 1.14
    assert truth.W_unscaled.loc[k, a] == 0.35 and round(float(truth.B_true.loc[k, a]), 2) == 0.35
    assert round(float(fit.B_hat.loc[k, a]), 2) == 0.03
    assert round(float(s.shocks(cfg).s_hat.loc[ev.return_days, k].sum()), 1) == 1.8
    assert pp(ev.contrib_true.loc[a, k]) == 0.72
    assert {t: pp(v) for t, v in ev.contrib.loc[a].items() if v != 0} == {"S1": 0.06, "S5": -0.15, "S10": -0.03}
    assert [pp(x[a]) for x in (ev.realized, ev.explained, ev.residual, ev.explained_true)] == [1.19, -0.13, 1.31, 0.69]
    assert pp(ev.fitted[a].sum()) == -0.13 and pp(ev.fitted_oracle[a].sum()) == 0.69  # the cumulative lines' ends
    rolled = _ui.rollup_by_group(ev.contrib.loc[a], s.simulation(cfg).topics.table)
    assert {g: pp(v) for g, v in rolled.items()} == {"Sector": -0.13, "Macro": 0.0, "Micro": 0.0}
    shares = ev.var_share.loc[a]
    assert {t: pp(v) for t, v in shares.items() if v != 0} == {"S1": 1.40, "S5": -1.01, "S10": 1.04}
    assert list(shares[["S1", "S5", "S10"]].abs().sort_values(ascending=False).index) == ["S1", "S10", "S5"]
    assert (pp(shares.sum()), pp(1 - shares.sum()), _ui.fmt_pct(shares.sum())) == (1.43, 98.57, "1.4%")
    true_share = ev.var_share_true.loc[a].sum()
    assert (pp(true_share), pp(1 - true_share), _ui.fmt_pct(true_share)) == (29.21, 70.79, "29.2%")
    explained_sq = float((ev.fitted[a] ** 2).sum() / (ev.realized_daily[a] ** 2).sum())
    assert pp(explained_sq) == 0.87 and (_ui.fmt_pct(ev.r2[a]), _ui.fmt_pct(ev.r2_oracle[a])) == ("2.0%", "34.0%")
    m = "MATERIALS_v_WEQ"  # the Compare methods scatter's well-estimated pair
    assert (round(float(truth.B_true.loc["S2", m]), 2), round(float(fit.B_hat.loc["S2", m]), 2)) == (0.35, 0.32)
    # the captions quote these figures
    assert "+0.06 pp next to a diamond at +0.72 pp" in _ui.how_to_read(*_ui.HOW_TWO_VIEWS)
    assert "(0.35, 0.03)" in _ui.how_to_read(*_ui.HOW_COMPARE_SCATTER)
    assert "0.03 × 1.1% × 1.8 = +0.06 pp" in _ui.how_cell_metric("OOS contribution (% points)")


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
    assert at.title[0].value == "Topic-sensitivity lab"
    assert [t.label for t in at.tabs] == ["Overview", "Correlation table", "Topic contributions", "Compare methods",
                                         "BKS", "Lists", "Data and method"]
    labels = [m.label for m in at.metric]
    for label in ("Median OOS R², estimator", "Median OOS R², oracle", "Median population R²",
                  "Assets with positive OOS R²", "Coverage", "Sign agreement", "MCC", "Spearman",
                  "Share explained by topics", "True share (simulation)", "Largest topic"):
        assert label in labels
    assert len(at.get("plotly_chart")) >= 7
    assert at.tabs[1].get("plotly_chart"), "the correlation table tab has no chart"
    assert any("20 return days" in m.value for m in at.tabs[1].markdown)  # 2025-07-01 to 2025-07-28
    assert any("training 2025-01-01 to 2025-06-30 (129 weekdays)" in m.value for m in at.markdown)
    # the Time windows expander: cut-off, length and the resulting window, with the short-window note
    assert "Time windows" in [e.label for e in at.sidebar.get("expander")]
    assert at.date_input(key="sb_train_end").value == dt.date(2025, 6, 30)
    assert at.select_slider(key="sb_train_months").value == 6
    assert at.date_input(key="sb_forecast_start").value == dt.date(2025, 7, 1)
    captions = [c.value for c in at.sidebar.caption]
    assert "Training window: 2025-01-01 to 2025-06-30 (129 weekdays)." in captions
    assert any(c.startswith("Short training window (129 weekdays") for c in captions)
    assert not at.button(key="sb_run_bks").disabled
    # lists: 55 assets and the 20 manual topics
    frames = [d.value for d in at.tabs[5].dataframe]
    assert any(len(f) == 55 and "Long proxy" in f.columns for f in frames)
    assert any(len(f) == 20 and "Scope" in f.columns for f in frames)
    assert at.tabs[4].info, "BKS tab should ask for a run"
    # Compare methods (G.15): the default methods, oracle last; both BKS variants (D88), not available before a
    # BKS run
    cm = at.tabs[3]
    table = cm.dataframe[0].value
    assert list(table["Method"]) == ["Elastic net", "Ridge (GCV)", "BKS-implied (full history)",
                                     "BKS-implied (training window)", "Oracle (true sensitivities)"]
    assert list(table.index) == ["elastic_net", "ridge", "bks_implied", "bks_implied_train", "oracle"]
    r2_col = "Median OOS R², this window"
    assert np.isfinite(table.loc[["elastic_net", "ridge", "oracle"], r2_col].astype(float)).all()
    for m in ("bks_implied", "bks_implied_train"):
        assert math.isnan(table.loc[m, r2_col]) and "Run BKS" in table.loc[m, "Note"]
    info = [i.value for i in cm.info if i.value.startswith("BKS-implied is not available")]
    assert len(info) == 1 and "BKS-implied (full history):" in info[0] and "BKS-implied (training window):" in info[0]
    assert not at.button(key="cm_run_bks").disabled
    assert at.radio(key="sb_bks_history").value == "full"  # the sidebar's BKS variant (D88)
    assert len(cm.get("plotly_chart")) == 4  # dots, sweep, and the inspected method's scatter and R2 bars
    assert at.selectbox(key="cm_inspect").value == "elastic_net"  # the sidebar's direct method
    assert [m.label for m in cm.metric] == ["Coverage", "Sign agreement", "MCC", "Spearman"]
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


def _visible_texts(at) -> list[tuple[str, str]]:
    """Every text a user can see on the page, as ``(kind, text)``: titles, headings, captions, markdown and
    LaTeX, alerts, metrics, tab and expander labels, widget labels, options and help texts, table headers and
    text cells, and the Plotly figures (titles, axis titles, legend and hover labels, as the figure JSON)."""
    out: list[tuple[str, str]] = []
    for kind in ("title", "header", "subheader", "caption", "markdown", "latex", "info", "warning", "error",
                 "success"):
        out += [(kind, str(e.value)) for e in getattr(at, kind)]
    for m in at.metric:
        out += [("metric", str(m.label)), ("metric", str(m.value)), ("metric help", str(m.help or ""))]
    out += [("tab", str(t.label)) for t in at.tabs]
    out += [("expander", str(e.label)) for e in at.expander]
    for kind in ("button", "download_button", "checkbox", "slider", "number_input", "date_input", "selectbox",
                 "radio", "select_slider", "multiselect"):
        for w in getattr(at, kind):
            out += [(kind, str(w.label)), (f"{kind} help", str(getattr(w, "help", "") or ""))]
            out += [(f"{kind} option", str(o)) for o in (getattr(w, "options", None) or [])]
    for d in at.dataframe:
        frame = d.value
        out += [("table header", str(c)) for c in frame.columns]
        for col in frame.columns:
            if frame[col].dtype == object or pd.api.types.is_string_dtype(frame[col]):
                out += [("table cell", str(v)) for v in frame[col].dropna()]
    out += [("chart", el.proto.spec) for el in at.get("plotly_chart")]
    return out


@needs_market
def test_app_default_page_says_sensitivity_not_exposure():
    """Owner decision 2026-09-30: the page says "topic sensitivity", never "exposure", except in the text that
    explains the old name. The rendered data/market/README.md is a verbatim import and is left out."""
    at = _app().run()
    _assert_clean(at)
    readme = (reference.market_dir() / "README.md").read_text(encoding="utf-8")
    texts = [(k, t) for k, t in _visible_texts(at) if t != readme]
    assert len(texts) > 300  # the walk reaches every tab, the sidebar and the charts
    explains = [t for _, t in texts if "formerly called 'exposure'" in t.lower()]
    assert len(explains) == 1 and "Topic sensitivity" in explains[0]  # the Data and method tab's definition
    found = [(k, t[:120]) for k, t in texts if "exposure" in t.lower() and t not in explains]
    assert not found, found
    # the new names are on the page
    assert at.title[0].value == "Topic-sensitivity lab"
    assert "Sensitivities" in [e.label for e in at.sidebar.get("expander")]
    assert at.radio(key="sb_n_betas").label == "Number of set sensitivities (betas)"
    assert at.slider(key="sb_beta_1").label == "Set sensitivity 1: strong links"
    assert list(at.selectbox(key="ex_metric").options) == [
        "OOS correlation", "Estimated sensitivity", "True sensitivity", "Set sensitivity (W)",
        "OOS contribution (% points)",
    ]
    # every cell metric of the correlation table: no "exposure" in its chart or captions
    for metric in _ui.METRICS[1:]:
        at.selectbox(key="ex_metric").set_value(metric).run()
        _assert_clean(at)
        tab = at.tabs[1]
        shown = [c.value for c in tab.caption] + [el.proto.spec for el in tab.get("plotly_chart")]
        assert not [t[:120] for t in shown if "exposure" in t.lower()], metric
    at.selectbox(key="ex_metric").set_value("Set sensitivity (W)").run()
    at.radio(key="ex_units").set_value(_ui.EXPOSURE_UNITS[1]).run()
    _assert_clean(at)
    shown = [c.value for c in at.tabs[1].caption] + [el.proto.spec for el in at.tabs[1].get("plotly_chart")]
    assert not [t[:120] for t in shown if "exposure" in t.lower()]
    assert any("Set sensitivity W (% per 1 sd shock)" in t for t in shown)


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
    # a longer training window from the same cut-off
    at.select_slider(key="sb_train_months").set_value(24).run()
    _assert_clean(at)
    assert any("training 2023-07-01 to 2025-06-30" in m.value for m in at.markdown)
    assert "Training window: 2023-07-01 to 2025-06-30 (521 weekdays)." in [c.value for c in at.sidebar.caption]
    assert not any(c.value.startswith("Short training window") for c in at.sidebar.caption)
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
    # correlation table options
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
    at.date_input(key="sb_forecast_start").set_value(dt.date(2025, 7, 1)).run()
    _assert_clean(at)
    # a February cut-off with one month: the window is extended to 21 weekdays and the caption says so
    at.select_slider(key="sb_train_months").set_value(1)
    at.date_input(key="sb_train_end").set_value(dt.date(2025, 2, 28))
    at.date_input(key="sb_forecast_start").set_value(dt.date(2025, 3, 3)).run()
    _assert_clean(at)
    assert any("The start moved back from 2025-02-01" in c.value for c in at.sidebar.caption)
    assert any("training 2025-01-31 to 2025-02-28 (21 weekdays)" in m.value for m in at.markdown)
    # reset restores the dashboard defaults
    at.button(key="sb_reset").click().run()
    _assert_clean(at)
    assert at.date_input(key="sb_train_end").value == dt.date(2025, 6, 30)
    assert at.select_slider(key="sb_train_months").value == 6


def test_app_bks_run_on_small_generic_config():
    at = _app().run()
    at.radio(key="sb_asset_source").set_value("generic").run()
    at.slider(key="sb_n_generic_assets").set_value(30).run()
    at.selectbox(key="sb_manual").set_value("none").run()
    at.slider(key="sb_n_generic_topics").set_value(12).run()
    at.slider(key="sb_signal_share").set_value(0.5).run()
    _assert_clean(at)
    assert any("12 topics" in m.value for m in at.markdown)
    assert _has_guide(at.tabs[5], _ui.how_to_read(*_ui.HOW_GENERIC_ASSETS))  # Lists tab
    _assert_guides_placed(at.tabs[5])
    at.button(key="sb_run_bks").click().run()
    _assert_clean(at)
    bks = at.tabs[4]
    assert "Chosen lambda" in [m.label for m in bks.metric]
    assert len(bks.get("plotly_chart")) >= 4
    assert not bks.warning, [w.value for w in bks.warning]
    # every BKS chart and the tiles have their "How to read" guide with examples (owner request 2026-09-30)
    assert _assert_how_to_read(bks) == {"fig_bks_gamma", "fig_bks_path", "fig_bks_r2", "fig_bks_split"}
    assert _has_guide(bks, _ui.how_to_read(*_ui.HOW_BKS_TILES))
    assert _has_guide(bks, _ui.how_bks_history(69.0))
    _assert_two_elements(at)
    # a new forecast window re-evaluates the cached fit: not stale
    at.slider(key="sb_forecast_weeks").set_value(8).run()
    _assert_clean(at)
    assert not [w for w in at.tabs[4].warning if "Settings changed" in w.value]
    # a new K makes the stored result stale until the next run
    at.slider(key="sb_bks_K").set_value(2).run()
    assert any("Settings changed" in w.value for w in at.tabs[4].warning)
    at.button(key="bks_run_tab").click().run()
    _assert_clean(at)
    assert not [w for w in at.tabs[4].warning if "Settings changed" in w.value]
    assert at.tabs[4].metric[1].value == "2"
    assert "bks_requested" not in at.session_state  # cleared once the run ended
    # F2: the BKS tab compares with the current direct fit, not a copy stored with the BKS run
    at.selectbox(key="sb_method").set_value("ridge").run()
    _assert_clean(at)
    est = next(m.value for m in at.metric if m.label == "Median OOS R², estimator")
    assert any(f"direct estimator {est} (daily, current settings)" in c.value for c in at.tabs[4].caption)
    # the shuffled-instrument reference sits next to the pooled OOS R2
    labels = [m.label for m in at.tabs[4].metric]
    assert "Same, instruments shuffled" in labels and "Median OOS R², BKS / direct" not in labels
    # F1: a fixed lambda of 0 no longer shows an OOS R2 of zero
    at.radio(key="sb_bks_rule").set_value("fixed").run()
    at.number_input(key="sb_bks_lam").set_value(0.0).run()
    at.button(key="sb_run_bks").click().run()
    _assert_clean(at)
    pooled = next(m.value for m in at.tabs[4].metric if m.label == "Pooled OOS R² (weekly)")
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
    assert other.tabs[4].info and "Chosen lambda" not in [m.label for m in other.tabs[4].metric]
    other.button(key="bks_run_tab").click().run()  # its own run reuses the cached fit
    _assert_clean(other)
    assert "Chosen lambda" in [m.label for m in other.tabs[4].metric]
    assert next(m.value for m in other.tabs[4].metric if m.label == "Pooled OOS R² (weekly)") not in ("0.0%", "n/a")


def _generic_sidebar(at) -> None:
    """Set the sidebar to 30 generic assets and 12 generic topics (6 linked): fast, needs no data files."""
    at.radio(key="sb_asset_source").set_value("generic").run()
    at.slider(key="sb_n_generic_assets").set_value(30).run()
    at.selectbox(key="sb_manual").set_value("none").run()
    at.slider(key="sb_n_generic_topics").set_value(12).run()
    at.slider(key="sb_signal_share").set_value(0.5).run()


def _trace_names(chart) -> list[str]:
    return [t.get("name") for t in json.loads(chart.proto.spec)["data"]]


#: The name of the first "How to read" guide under each Plotly chart, by chart key (owner requests 2026-09-30 and
#: 2026-10-02, D91): the label without the help icon, which is the text's lead without its colon.
HOW_TO_READ_LEADS: dict[str, str] = {
    "fig_r2": "How to read this chart",
    "fig_sweep": "How to read this chart",
    "fig_scatter": "How to read this chart",
    "fig_exposure": "How to read the cell metric",
    "fig_contrib": "How to read the bar chart",
    "fig_cumulative": "How to read the cumulative chart",
    "fig_attention": "How to read the attention chart",
    "fig_cm_r2": "How to read this chart",
    "fig_cm_sweep": "How to read this chart",
    "fig_cm_scatter": "How to read this chart",
    "fig_cm_r2_inspect": "How to read this chart",
    "fig_bks_gamma": "How to read the Gamma chart",
    "fig_bks_path": "How to read the lambda path",
    "fig_bks_r2": "How to read the R² comparison",
    "fig_bks_split": "How to read the per-topic split",
    "real_heatmap": "How to read the preview",
    # the BKS trace page (D90)
    "fig_tr_ladder": "How to read the ladder",
    "fig_tr_inputs": "How to read the inputs chart",
    "fig_tr_truth": "How to read the truth chart",
    "fig_tr_divisor": "How to read the divisor chart",
    "fig_tr_shocks": "How to read the shocks chart",
    "fig_tr_instrument": "How to read the instrument chart",
    "fig_tr_kernel": "How to read the kernel chart",
    "fig_tr_instr_truth": "How to read the instrument scatter",
    "fig_tr_design": "How to read the design heatmap",
    "fig_tr_stability": "How to read the stability chart",
    "fig_tr_path": "How to read the lambda path and its noise",
    "fig_tr_coef_path": "How to read the Gamma path",
    "fig_tr_gamma": "How to read the Gamma heatmap",
    "fig_tr_topics": "How to read the topic chart",
    "fig_tr_kkt": "How to read the optimum check",
    "fig_tr_factors": "How to read the factor chart",
    "fig_tr_week_r2": "How to read the weekly R² chart",
    "fig_tr_week_scatter": "How to read the week scatter",
    "fig_tr_oos_factors": "How to read the forecast factors",
    "fig_tr_chain": "How to read the chain chart",
    "fig_tr_capture": "How to read the direction chart",
    "fig_tr_implied_scatter": "How to read the implied scatter",
    "fig_tr_sigma": "How to read the Sigma_z chart",
}

#: The label of a guide toggle (``_ui.guide_label``); group 1 is its name.
GUIDE_RE = re.compile(r"^:blue\[:material/help_outline: (.+)\]$")


def _guide_name(node) -> str | None:
    """The plain name of a guide toggle ("How to read the tiles"), else None."""
    if node.type != "expander":
        return None
    m = GUIDE_RE.match(str(node.label))
    return m.group(1) if m else None


def _guides(block) -> list[tuple[str, str, object]]:
    """Every guide toggle under ``block`` in tree order, as ``(name, body, node)``; the body is its captions
    joined by newlines. Collapsed bodies run, so AppTest sees them."""
    out: list[tuple[str, str, object]] = []

    def walk(node) -> None:
        for i in sorted(node.children):
            child = node.children[i]
            name = _guide_name(child)
            if name is not None:
                out.append((name, "\n".join(str(c.value) for c in child.caption), child))
            if getattr(child, "children", None) is not None:
                walk(child)

    walk(block)
    return out


def _guide_names(block) -> list[str]:
    return [name for name, _, _ in _guides(block)]


def _chart_guides(block) -> dict[str, list[tuple[str, str]]]:
    """Every Plotly chart under ``block`` by key, with the ``(name, body)`` of the guide toggles that follow it
    in its own container, up to the next chart there (the guides "next to" the chart)."""
    out: dict[str, list[tuple[str, str]]] = {}

    def walk(node) -> None:
        current = None
        for i in sorted(node.children):
            child = node.children[i]
            name = _guide_name(child)
            if child.type == "plotly_chart":
                current = str(child.proto.id).rsplit("-", 1)[-1]
                out[current] = []
            elif name is not None and current is not None:
                out[current].append((name, "\n".join(str(c.value) for c in child.caption)))
            if getattr(child, "children", None) is not None:
                walk(child)

    walk(block)
    return out


def _loose_how_to_read(block) -> list[str]:
    """Captions under ``block`` that start with "How to read" (a guide's body starts with "- "): must be none."""
    return [str(c.value)[:50] for c in block.caption if str(c.value).startswith("How to read")]


def _has_guide(block, text: str) -> bool:
    """Whether a guide toggle under ``block`` shows the ``how_to_read`` text ``text``: its label and bullets."""
    label, body = _ui.guide_parts(text)
    return any(str(node.label) == label and b == body for _, b, node in _guides(block))


#: Where the guides that do not explain a chart sit (G.9 placement), by guide name: the kinds the element they
#: explain may have (see :func:`_kind`). That element is the nearest earlier one in the guide's container once
#: the visible data notes (captions) are skipped; "first" means there is none, and "guide:<name>" is a guide of
#: the same element stacked above. A note guide explains the caption right before it: "caption:<text>" means
#: that caption contains the text. A chart guide (:data:`HOW_TO_READ_LEADS`) follows a chart, and every other
#: guide follows a table.
GUIDE_AFTER: dict[str, set[str]] = {
    "How to read the tiles": {"tiles"},
    "How to read the table": {"dataframe", "guide:How to read the cell metric"},
    "How to read the roll-up": {"guide:How to read the bar chart"},
    "How to read the two views": {"columns"},
    "Why this method": {"guide:How to read the two views"},
    "How to read the comparison": {"first"},
    "How to read the findings": {"markdown"},
    "How to read this note": {"caption:linked to a topic"},
    "How to read the BKS-implied note": {f"caption:{IMPLIED_NOTE}"},
    "How to read the covariance history": {"caption:Weekly BKS fit"},
    "How to read the link map": {f"caption:{_ui.LINK_MAP_CAVEAT}"},
}


def _kind(node) -> str:
    """The kind of an element for :data:`GUIDE_AFTER`: "chart", "dataframe" (a table or data editor), "tiles" (a
    row of metrics), "columns" (a row without metrics), "guide:<name>", "caption:<text>" or the element type."""
    name = _guide_name(node)
    if name is not None:
        return f"guide:{name}"
    if node.type == "plotly_chart":
        return "chart"
    if node.type in ("dataframe", "arrow_data_frame"):
        return "dataframe"
    if node.type == "flex_container":
        return "tiles" if node.metric else "columns"
    if node.type == "caption":
        return f"caption:{node.value}"
    return str(node.type)


def _guide_sites(block) -> list[tuple[str, str, str, str]]:
    """Every guide toggle under ``block`` in tree order as ``(name, before, explained, after)``: the kinds of the
    element right before it, of the nearest earlier element that is not a caption ("first" if none) and of the
    element right after it ("last" if none), all in the guide's own container."""
    out: list[tuple[str, str, str, str]] = []

    def walk(node) -> None:
        kids = [node.children[i] for i in sorted(node.children)]
        for j, child in enumerate(kids):
            name = _guide_name(child)
            if name is not None:
                earlier = [_kind(k) for k in kids[:j]]
                explained = next((k for k in reversed(earlier) if not k.startswith("caption:")), "first")
                after = _kind(kids[j + 1]) if j + 1 < len(kids) else "last"
                out.append((name, earlier[-1] if earlier else "first", explained, after))
            if getattr(child, "children", None) is not None:
                walk(child)

    walk(block)
    return out


def _assert_guides_placed(block) -> None:
    """G.9 placement: each guide under ``block`` sits right under the element it explains, after that element's
    data notes; the link map's guide sits directly above the editor."""
    charts = set(HOW_TO_READ_LEADS.values())
    for name, before, explained, after in _guide_sites(block):
        want = GUIDE_AFTER.get(name, {"chart"} if name in charts else {"dataframe"})
        notes = [w.removeprefix("caption:") for w in want if w.startswith("caption:")]
        if notes:
            assert before.startswith("caption:") and any(n in before for n in notes), (name, before[:80])
        else:
            assert explained in want, (name, explained[:80])
        if name == "How to read the link map":
            assert after == "dataframe", after[:80]


def _chart_guide_text(key: str) -> str | None:
    """The fixed "How to read" text of the chart ``key``; None where the text follows the view or the run (the
    cell metric, the Compare methods dots, sweep and inspected method)."""
    for prefix, stem in (("fig_bks_", "HOW_BKS_"), ("fig_tr_", "HOW_TRACE_")):
        if key.startswith(prefix):
            return _ui.how_to_read(*getattr(_ui, stem + key.removeprefix(prefix).upper()))
    fixed = {"fig_sweep": _ui.HOW_SWEEP, "fig_scatter": _ui.HOW_SCATTER, "fig_contrib": _ui.HOW_CONTRIB_BARS,
             "fig_cumulative": _ui.HOW_CUMULATIVE, "fig_attention": _ui.HOW_ATTENTION,
             "fig_cm_scatter": _ui.HOW_COMPARE_SCATTER, "real_heatmap": _ui.HOW_REAL_PREVIEW}
    if key == "fig_r2":
        return _ui.how_r2_bars("estimator")
    return _ui.how_to_read(*fixed[key]) if key in fixed else None


def _assert_how_to_read(block) -> set[str]:
    """Each Plotly chart under ``block`` has its "How to read" guide next to it, with the chart's own text where
    that text is fixed, and every bullet of that guide ends with an example; no "How to read" text is a loose
    caption, every guide is a collapsed compact toggle (D91) and sits where G.9 puts it. Returns the chart
    keys."""
    assert _loose_how_to_read(block) == []
    found = _chart_guides(block)
    for key, guides in found.items():
        assert guides and guides[0][0] == HOW_TO_READ_LEADS[key], (key, [n for n, _ in guides])
        bullets = guides[0][1].split("\n")
        assert bullets and all(b.startswith("- ") and " Example: " in b for b in bullets), key
        text = _chart_guide_text(key)
        assert text is None or guides[0][1] == _ui.guide_parts(text)[1], key
    for name, _, node in _guides(block):
        assert node.proto.type == 1 and node.proto.expanded is False, name  # compact, collapsed on load
    _assert_guides_placed(block)
    return set(found)


def _heatmap_cells_shown(block) -> int:
    """Number of non-blank cells of the Correlation table's heatmap (the AVERAGE row left out)."""
    spec = json.loads(block.get("plotly_chart")[0].proto.spec)
    z = next(t for t in spec["data"] if t.get("type") == "heatmap" and t.get("name") != "AVERAGE")["z"]
    if isinstance(z, dict):  # Plotly >= 6 sends arrays as typed base64 data
        arr = np.frombuffer(base64.b64decode(z["bdata"]), dtype=np.dtype(z["dtype"]))
    else:
        arr = np.array(z, dtype=float)
    return int(np.isfinite(arr.astype(float)).sum())


@needs_market
def test_app_how_to_read_next_to_every_chart():
    """Owner requests 2026-09-30 and 2026-10-02 (D91): a "How to read" guide whose bullets end with an example
    sits next to every chart, table and row of tiles, for every cell metric and unit and both contribution
    views; the blank rule of the Correlation table and its label follow the view."""
    at = _app().run()
    _assert_clean(at)
    assert _assert_how_to_read(at.main) == {
        "fig_r2", "fig_sweep", "fig_scatter", "fig_exposure", "fig_contrib", "fig_cumulative", "fig_attention",
        "fig_cm_r2", "fig_cm_sweep", "fig_cm_scatter", "fig_cm_r2_inspect",
    }
    assert _has_guide(at.tabs[0], _ui.how_to_read(*_ui.HOW_OVERVIEW_TILES))
    assert "How to read this note" in _guide_names(at.tabs[0])  # 43 of 55 assets are linked at the defaults
    assert _has_guide(at.tabs[2], _ui.how_to_read(*_ui.HOW_TILES_SHARE))
    assert _has_guide(at.tabs[2], _ui.how_to_read(*_ui.HOW_TWO_VIEWS))
    name, body, _ = _guides(at.tabs[3])[0]
    assert name == "How to read the comparison"
    assert body.startswith("- Every method is scored") and body.count(" Example: ") == 6
    for text in (_ui.how_to_read(*_ui.HOW_COMPARE_TILES), _ui.how_compare_table("the 6 consecutive 4-week windows")):
        assert _has_guide(at.tabs[3], text)
    assert "How to read the covariance history" in _guide_names(at.tabs[4])  # before a BKS run too
    lists = _guide_names(at.tabs[5])
    for lead in (_ui.HOW_ASSETS_TABLE[0], _ui.HOW_TOPICS_TABLE[0], _ui.HOW_LINK_MAP[0]):
        assert lead[:-1] in lists
    assert _ui.LINK_MAP_CAVEAT in [c.value for c in at.tabs[5].caption]

    # the Correlation table: every cell metric in both units; the blank rule and its label follow the view
    cfg, _, _ = _ui.config_from_values(_ui.default_values(), (), reference.load_assets()["asset_class"])
    s = LabSession()
    truth, fit = s.truth(cfg), s.direct(cfg)
    n_selected = int(fit.selected.to_numpy().sum())
    n_sensitive = int((truth.B_true.abs() >= cfg.direct.select_tau).to_numpy().sum())
    n_linked = int((truth.W_unscaled != 0).to_numpy().sum())
    assert n_sensitive > n_selected > n_linked == 95
    want = {"True sensitivity": n_sensitive, "Set sensitivity (W)": n_linked}
    for metric in _ui.METRICS:
        at.selectbox(key="ex_metric").set_value(metric).run()
        for units in (_ui.EXPOSURE_UNITS if metric in _ui.EXPOSURE_METRICS else (None,)):
            if units is not None:
                at.radio(key="ex_units").set_value(units).run()
            _assert_clean(at)
            tab = at.tabs[1]
            assert _assert_how_to_read(tab) == {"fig_exposure"}
            found = _chart_guides(tab)["fig_exposure"]
            assert [n for n, _ in found][:2] == ["How to read the cell metric", "How to read the table"], metric
            assert found[0][1] == _ui.guide_parts(_ui.how_cell_metric(metric))[1], metric
            assert found[1][1] == _ui.guide_parts(_ui.how_to_read(*_ui.HOW_TABLE))[1], metric
            assert at.checkbox(key="ex_blank_rule").label == _ui.blank_rule(metric, 0.05)["label"]
            assert f"Blank: {_ui.blank_rule(metric, 0.05)['subtitle']}" in tab.get("plotly_chart")[0].proto.spec
            assert _heatmap_cells_shown(tab) == want.get(metric, n_selected), (metric, units)
    at.selectbox(key="ex_metric").set_value("Set sensitivity (W)").run()
    at.checkbox(key="ex_blank_rule").uncheck().run()  # off: every pair shows, zeros included
    _assert_clean(at)
    assert _heatmap_cells_shown(at.tabs[1]) == 1100 and "No blank rule." in at.tabs[1].get("plotly_chart")[0].proto.spec
    at.checkbox(key="ex_blank_rule").check().run()

    # an asset the elastic net gives no topic: a note instead of the attention chart, and no largest topic
    empty = [str(a) for a in fit.B_hat.columns if not fit.selected[a].any()]
    if empty:
        at.selectbox(key="tc_asset").set_value(empty[0]).run()
        _assert_clean(at)
        tab = at.tabs[2]
        assert "fig_attention" not in _chart_guides(tab)
        assert any(i.value.startswith("No topic contributes") for i in tab.info)
        assert next(m for m in tab.metric if m.label == "Largest topic").value == "none"
        at.selectbox(key="tc_asset").set_value(_ui.DEFAULT_CONTRIB_ASSET).run()

    # the contributions tab's other view and the roll-up
    at.radio(key="tc_view").set_value("Return attribution").run()
    at.checkbox(key="tc_rollup").check().run()
    _assert_clean(at)
    assert _assert_how_to_read(at.tabs[2]) == {"fig_contrib", "fig_cumulative", "fig_attention"}
    assert _has_guide(at.tabs[2], _ui.how_to_read(*_ui.HOW_TILES_ATTRIBUTION))
    assert _has_guide(at.tabs[2], _ui.how_to_read(*_ui.HOW_ROLLUP))
    # the roll-up changes only the bar chart, so its guide sits under the bar chart's guide
    assert "How to read the roll-up" in [n for n, _ in _chart_guides(at.tabs[2])["fig_contrib"]]
    # Compare methods without the oracle: the rows follow the first method, and the guides say so
    at.multiselect(key="cm_methods").unselect("oracle").run()
    _assert_clean(at)
    assert _assert_how_to_read(at.tabs[3]) == {"fig_cm_r2", "fig_cm_sweep", "fig_cm_scatter", "fig_cm_r2_inspect"}
    assert _ui.guide_parts(_ui.how_compare_dots(False))[1] in [b for _, b in _chart_guides(at.tabs[3])["fig_cm_r2"]]


def test_app_compare_methods_with_a_bks_run():
    """G.15, D83: BKS-implied joins the comparison after this session's BKS run and stays when only the
    forecast window changes; the inspect control follows the sidebar's method; OLS on request."""
    at = _app().run()
    _generic_sidebar(at)
    _assert_clean(at)
    r2_col = "Median OOS R², this window"
    table = at.tabs[3].dataframe[0].value
    assert math.isnan(table.loc["bks_implied", r2_col]) and math.isnan(table.loc["bks_implied_train", r2_col])
    assert not any(n.startswith("BKS-implied") for n in _trace_names(at.tabs[3].get("plotly_chart")[0]))
    # "Method to inspect" follows the sidebar's direct method until the user picks one
    at.selectbox(key="sb_method").set_value("ridge").run()
    _assert_clean(at)
    assert at.selectbox(key="cm_inspect").value == "ridge"

    at.button(key="cm_run_bks").click().run()  # the tab's own Run BKS button fits both variants (D88)
    _assert_clean(at)
    assert "bks_compare_requested" not in at.session_state  # cleared once the run ended
    cm = at.tabs[3]
    assert not cm.info
    table = cm.dataframe[0].value
    for m in ("bks_implied", "bks_implied_train"):
        row = table.loc[m]
        assert np.isfinite(float(row[r2_col])) and np.isfinite(float(row["Spearman vs truth"]))
        assert row["Note"] == IMPLIED_NOTE
    implied = table.loc["bks_implied"]
    assert table.loc["bks_implied", r2_col] != table.loc["bks_implied_train", r2_col]
    assert list(table.index)[-1] == "oracle"
    dots, sweep = cm.get("plotly_chart")[:2]
    for label in ("BKS-implied (full history)", "BKS-implied (training window)"):
        assert label in _trace_names(dots) and label in _trace_names(sweep)
    # the lead guide: the full history's extra data, the like-for-like variant and the BKS tab's own R2
    lead = next(b for n, b, _ in _guides(cm) if n == "How to read the comparison")
    assert "weigh all days before the cut-off" in lead and "lies before the training start" in lead
    assert "BKS-implied (training window) sees only the training window" in lead
    assert "differ in the covariance history and the return scaling" in lead
    assert "The BKS tab's OOS R² (here " in lead
    # the pointer to the reasons BKS-implied scores lower (G.15.1), and the item it points to
    assert any(c.value.startswith("Why the BKS-implied rows score lower") and "Data and method" in c.value
               for c in cm.caption)
    assert "Why BKS-implied scores lower (G.15.1)" in [h.value for h in at.tabs[6].subheader]
    assert any("The directions BKS keeps" in m.value and "Not the number of factors" in m.value
               for m in at.tabs[6].markdown)
    # the oracle is the reference in the inspect charts, not an option
    inspect_options = at.selectbox(key="cm_inspect").options  # display labels
    assert {"BKS-implied (full history)", "BKS-implied (training window)"} <= set(inspect_options)
    assert not any(o.startswith("Oracle") for o in inspect_options)
    assert "Chosen lambda" in [m.label for m in at.tabs[4].metric]  # the BKS tab has the same run
    assert any("Covariance history: full history" in c.value for c in at.tabs[4].caption)
    # the sidebar's radio switches the BKS tab to the training-window fit the Compare run made: not stale
    at.radio(key="sb_bks_history").set_value("training").run()
    _assert_clean(at)
    assert "Chosen lambda" in [m.label for m in at.tabs[4].metric]
    assert not [w for w in at.tabs[4].warning if "Settings changed" in w.value]
    assert any("Covariance history: training window only" in c.value for c in at.tabs[4].caption)
    at.radio(key="sb_bks_history").set_value("full").run()
    _assert_clean(at)

    # a new forecast window re-scores the cached fits: BKS-implied stays, with the same fit time
    at.slider(key="sb_forecast_weeks").set_value(8).run()
    _assert_clean(at)
    after = at.tabs[3].dataframe[0].value.loc["bks_implied"]
    assert np.isfinite(float(after[r2_col])) and not at.tabs[3].info
    assert after[r2_col] != implied[r2_col] and after["Fit time (s)"] == implied["Fit time (s)"]

    # inspecting BKS-implied shows the caveat with the fit's K, kept topics and covariance history
    at.selectbox(key="cm_inspect").set_value("bks_implied").run()
    _assert_clean(at)
    assert any(c.value.startswith(IMPLIED_NOTE) and "K = 3 factors" in c.value and "full history" in c.value
               for c in at.tabs[3].caption)
    assert [m.label for m in at.tabs[3].metric] == ["Coverage", "Sign agreement", "MCC", "Spearman"]
    assert _has_guide(at.tabs[3], _ui.how_bks_implied(69.0))  # the note's "How to read"
    assert _assert_how_to_read(at.tabs[3]) == {"fig_cm_r2", "fig_cm_sweep", "fig_cm_scatter", "fig_cm_r2_inspect"}
    _assert_two_elements(at)
    at.selectbox(key="cm_inspect").set_value("bks_implied_train").run()
    _assert_clean(at)
    assert any(c.value.startswith(IMPLIED_NOTE) and "training window only" in c.value for c in at.tabs[3].caption)
    at.selectbox(key="cm_inspect").set_value("bks_implied").run()
    at.selectbox(key="sb_method").set_value("elastic_net").run()
    assert at.selectbox(key="cm_inspect").value == "bks_implied"  # the user's own choice stays

    # without a BKS-implied method the pointer caption is not shown
    at.multiselect(key="cm_methods").unselect("bks_implied").unselect("bks_implied_train").run()
    _assert_clean(at)
    assert not any(c.value.startswith("Why the BKS-implied rows score lower") for c in at.tabs[3].caption)
    at.multiselect(key="cm_methods").select("bks_implied").select("bks_implied_train").run()
    _assert_clean(at)

    # OLS on request, in method order
    at.multiselect(key="cm_methods").select("ols").run()
    _assert_clean(at)
    assert list(at.tabs[3].dataframe[0].value.index) == ["elastic_net", "ridge", "ols", "bks_implied",
                                                         "bks_implied_train", "oracle"]

    # a new K: the BKS fits of these settings have not been run, so both variants are unavailable again
    at.slider(key="sb_bks_K").set_value(2).run()
    _assert_clean(at)
    assert any(i.value.startswith("BKS-implied is not available") for i in at.tabs[3].info)
    assert math.isnan(at.tabs[3].dataframe[0].value.loc["bks_implied", r2_col])
    assert math.isnan(at.tabs[3].dataframe[0].value.loc["bks_implied_train", r2_col])
    # the sidebar's Run BKS fits only the sidebar's variant (full history)
    at.button(key="sb_run_bks").click().run()
    _assert_clean(at)
    table = at.tabs[3].dataframe[0].value
    assert np.isfinite(float(table.loc["bks_implied", r2_col])) and math.isnan(table.loc["bks_implied_train", r2_col])
    info = [i.value for i in at.tabs[3].info if i.value.startswith("BKS-implied is not available")]
    assert len(info) == 1 and "training window" in info[0] and "full history" not in info[0]


def test_app_sidebar_run_bks_follows_the_training_check():
    """The sidebar's Run BKS is disabled when BKS cannot use the training window, as the BKS tab's is (review
    2026-09-30: it stayed enabled and pressing it showed a technical error)."""
    at = _app().run()
    _generic_sidebar(at)
    _assert_clean(at)
    assert not at.button(key="sb_run_bks").disabled
    at.select_slider(key="sb_train_months").set_value(1).run()
    _assert_clean(at)
    assert at.button(key="sb_run_bks").disabled and at.button(key="bks_run_tab").disabled
    at.radio(key="sb_bks_history").set_value("training").run()
    _assert_clean(at)
    assert at.button(key="sb_run_bks").disabled and at.button(key="bks_run_tab").disabled
    at.select_slider(key="sb_train_months").set_value(6).run()  # 24 usable weeks with the training window only
    _assert_clean(at)
    assert not at.button(key="sb_run_bks").disabled and not at.button(key="bks_run_tab").disabled


def test_comparison_table_labels_and_notes():
    """The pure helpers of the Compare methods tab: table columns and units, plain notes, option labels."""
    s = LabSession()
    # a one-month window with 12 topics: OLS is refused (L >= n_train / 2)
    cfg = _generic_cfg(train_start="2022-12-01", train_end="2022-12-30", forecast_start="2023-01-02")
    res = s.comparison(cfg, methods=("elastic_net", "ols", "bks_implied", "bks_implied_train", "oracle"),
                       use_bks=False)
    t = _ui.comparison_table(res.summary, {"bks_implied": "Run BKS first."})
    assert list(t.columns) == [head for _, head, _ in _ui.COMPARISON_COLUMNS]
    assert list(t.columns[:4]) == ["Method", "Median OOS R², this window", "Median OOS R², all windows",
                                   "Windows above the oracle"]  # on screen at laptop width
    assert list(t.columns[-2:]) == ["Fit time (s)", "Note"]
    assert list(t.index) == ["elastic_net", "ols", "bks_implied", "bks_implied_train", "oracle"]
    assert t.loc["elastic_net", "Coverage"] == pytest.approx(100.0 * res.summary.loc["elastic_net", "coverage"])
    assert t.loc["elastic_net", "RMSE vs truth"] == pytest.approx(res.summary.loc["elastic_net", "rmse"])
    note = t.loc["ols", "Note"]
    assert note.startswith("Not fitted: OLS needs fewer topics") and "12 topics" in note
    assert "elastic_net" not in note  # no code names
    assert math.isnan(t.loc["ols", "Median OOS R², this window"])
    assert t.loc["bks_implied", "Note"] == "Run BKS first."
    assert math.isnan(t.loc["oracle", "Windows above the oracle"])
    fmts = _ui.comparison_column_formats()
    assert fmts["Coverage"] == "%.1f%%" and fmts["MCC"] == "%.2f" and "Method" not in fmts and "Note" not in fmts
    assert _ui.unavailable_note("BKS has not been run.") == "BKS has not been run."
    assert _ui.method_option_label("elastic_net", DirectConfig(penalty="cv")) == "Elastic net (CV)"
    assert _ui.method_option_label("ridge", DirectConfig(method="ridge", ridge_lambda=0.1)) == "Ridge (fixed lambda)"
    assert _ui.method_option_label("ridge", DirectConfig(penalty="cv")) == "Ridge (GCV)"  # not the sidebar's method
    assert t.loc["bks_implied_train", "Note"] == "BKS has not been run in this browser session. Run BKS first."
    assert _ui.method_option_label("bks_implied") == "BKS-implied (full history)"
    assert _ui.method_option_label("bks_implied_train") == "BKS-implied (training window)"
    assert _ui.COMPARE_DEFAULT_METHODS == ("elastic_net", "ridge", "bks_implied", "bks_implied_train", "oracle")


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
    for name in ("oos_corr.csv", "sensitivities.csv", "r2.csv", "contributions.csv", "recovery.json", "sweep.csv",
                 "summary.json", "bks_r2.csv", "bks_contrib.csv"):
        assert (out_dir / name).is_file(), name
    assert not list(out_dir.glob("*exposure*"))  # renamed 2026-09-30
    pairs = pd.read_csv(out_dir / "sensitivities.csv")
    assert list(pairs.columns) == ["topic_id", "asset_id", "b_hat", "b_true", "w_set", "selected"]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_assets"] == 30 and summary["n_topics"] == 12
    assert summary["config"]["exposure"]["link_overrides"] == [["G001", "G_ASSET_001", "strong", -1]]
    corr = pd.read_csv(out_dir / "oos_corr.csv", index_col=0)
    assert corr.shape == (30, 12)
    contrib = pd.read_csv(out_dir / "contributions.csv")
    assert len(contrib) == 30 * 12
    printed = capsys.readouterr().out
    assert "median OOS R2" in printed and printed.startswith("Topic-sensitivity lab:")
    cfg = LabConfig.from_dict(d)  # the configs normalise lists themselves (D77); no helper in the script
    assert cfg.exposure.link_overrides == (("G001", "G_ASSET_001", "strong", -1),)
    assert not hasattr(mod, "normalise_config")


# ---------------------------------------------------------------------------
# One-month training floor (D81)
# ---------------------------------------------------------------------------
def test_short_training_note():
    from narrative_ipca.exposure_lab.bks import MIN_TRAIN_PERIODS

    assert not hasattr(_ui, "BKS_MIN_TRAIN_WEEKS")  # read from the BKS module, not duplicated
    assert _ui.short_training_note("2015-01-02", "2022-12-30", 20) is None
    note = _ui.short_training_note("2022-12-01", "2022-12-30", 20)
    assert note is not None and "22 weekdays" in note and f"BKS needs at least {MIN_TRAIN_PERIODS}" in note
    assert "0.21" in note  # 1/sqrt(22)
    note = _ui.short_training_note("2022-01-03", "2022-09-30", 20)
    assert note is not None and "BKS can run" in note
    # the dashboard default: 129 weekdays over 26 weeks
    note = _ui.short_training_note("2025-01-01", "2025-06-30", 20)
    assert "129 weekdays, 26 weeks" in note and ("BKS can run" in note) == (26 >= MIN_TRAIN_PERIODS)


def test_bks_training_check_counts_the_burn_in():
    """BKS skips its first 52 weeks of data (D17): a window near the data start has fewer usable weeks
    than Fridays, and the reason says so (review 2026-09-29: the note said "BKS can run" and the fit
    then refused)."""
    from narrative_ipca.exposure_lab.bks import MIN_TRAIN_PERIODS

    ok = _ui.bks_training_check("2025-01-01", "2025-06-30")
    assert ok["can_run"] and ok["reason"] == "" and ok["weeks"] == ok["n_weeks"] == 26
    short = _ui.bks_training_check("2025-06-01", "2025-06-30")
    assert not short["can_run"]
    assert short["reason"] == f"BKS needs at least {MIN_TRAIN_PERIODS} training weeks; the training window has 4."
    early = _ui.bks_training_check("2015-07-01", "2015-12-31")
    assert (early["weeks"], early["n_weeks"], early["can_run"]) == (26, 0, False)
    assert "skips the first 52 weeks" in early["reason"] and "2016-04-08" in early["reason"]
    note = _ui.short_training_note("2015-07-01", "2015-12-31", 20)
    assert "BKS can run" not in note and "2016-04-08" in note
    # ten years back from 2016-06-30 (clipped to the data start): 78 Fridays, 12 usable weeks
    ten = _ui.bks_training_check("2015-01-02", "2016-06-30")
    assert (ten["weeks"], ten["n_weeks"], ten["can_run"]) == (78, 12, False)


def test_bks_training_check_with_the_training_window_only():
    """D88: the training-window variant has no burn-in; it loses the window's first week or two instead."""
    from narrative_ipca.exposure_lab.config import BKSLabConfig

    tr = BKSLabConfig(history="training")
    ok = _ui.bks_training_check("2025-01-01", "2025-06-30", tr)
    assert ok["can_run"] and (ok["weeks"], ok["n_weeks"]) == (26, 24)
    assert ok["first_week"] == pd.Timestamp("2025-01-17")
    # near the data start it runs where the full history cannot (no 52-week burn-in)
    assert _ui.bks_training_check("2015-07-01", "2015-12-31", tr)["can_run"]
    # 24 Fridays: the full history uses all 24, the training window loses the first
    assert _ui.bks_training_check("2025-01-06", "2025-06-20")["can_run"]
    short = _ui.bks_training_check("2025-01-06", "2025-06-20", tr)
    assert (short["weeks"], short["n_weeks"], short["can_run"]) == (24, 23, False)
    assert "training window only" in short["reason"] and "Lengthen the training window" in short["reason"]
    # near the data start a longer window cannot help (the start is clamped to 2015-01-02): move the cut-off
    for months in (6, 12, 120):
        tw = _ui.training_window("2015-06-30", months)
        clamped = _ui.bks_training_check(tw["start"], tw["end"], tr)
        assert (clamped["n_weeks"], clamped["can_run"]) == (23, False)
        assert "Move the cut-off later" in clamped["reason"] and "Lengthen" not in clamped["reason"]
        assert "2015-01-23" in clamped["reason"]
    assert _ui.bks_training_check("2015-01-02", "2015-07-10", tr)["can_run"]  # a later cut-off runs
    note = _ui.short_training_note("2025-01-06", "2025-06-20", 20, tr)
    assert note is not None and "training window only" in note
    # the sidebar value drives the config
    v = _ui.default_values()
    assert v["sb_bks_history"] == "full"
    v["sb_bks_history"] = "training"
    cfg, errors, _ = _ui.config_from_values(v)
    assert not errors and cfg.bks.history == "training"


def test_one_month_training_runs_direct_and_bks_refuses():
    from narrative_ipca.exposure_lab.bks import run_bks

    cfg = _generic_cfg(train_start="2022-12-01", train_end="2022-12-30", forecast_start="2023-01-02")
    out = run_lab(cfg)
    assert int(out["direct"].n_train.min()) >= 20
    assert np.isfinite(out["evaluation"].r2.to_numpy(dtype=float)).all()
    with pytest.raises(ValueError, match="training"):
        run_bks(out["simulation"], cfg.bks, cfg.window)


# ---------------------------------------------------------------------------
# Top navigation and the Real data placeholder (G.14, D82)
# ---------------------------------------------------------------------------
import real_exposures  # noqa: E402


def test_real_exposures_status_and_loading(tmp_path):
    status = real_exposures.data_status(tmp_path)
    assert list(status["Found"]) == ["no", "no", "no"]
    assert real_exposures.load_exposures(tmp_path) == (None, [])

    rows = [
        ("2025-12-31", "S1", "ENERGY_v_WEQ", 0.40, 0.05, "blended", "2023-01-02/2025-12-31", "full", "v1"),
        ("2025-12-31", "S1", "NOK_v_USD", 0.12, 0.04, "regression", "2023-01-02/2025-12-31", "partial", "v1"),
        ("2025-12-31", "A4", "ENERGY_v_WEQ", -0.05, 0.03, "llm", "2023-01-02/2025-12-31", "none", "v1"),
    ]
    assert real_exposures.SENSITIVITY_FILE == "sensitivities.parquet"
    assert "sensitivity" in real_exposures.EXPOSURE_COLUMNS and "exposure" not in real_exposures.EXPOSURE_COLUMNS
    df = pd.DataFrame(rows, columns=list(real_exposures.EXPOSURE_COLUMNS))
    df.to_parquet(tmp_path / "sensitivities.parquet")
    loaded, problems = real_exposures.load_exposures(tmp_path)
    assert problems == [] and len(loaded) == 3
    status = real_exposures.data_status(tmp_path)
    assert status.loc[status["File"].str.endswith("sensitivities.parquet"), "Found"].item() == "yes"
    values, blank = real_exposures.exposure_table(loaded, "2025-12-31", ["NOK_v_USD", "ENERGY_v_WEQ"])
    assert list(values.index) == ["NOK_v_USD", "ENERGY_v_WEQ"]
    assert values.loc["ENERGY_v_WEQ", "S1"] == pytest.approx(0.40)
    assert bool(blank.loc["ENERGY_v_WEQ", "A4"])  # coverage none is blank
    assert bool(blank.loc["NOK_v_USD", "A4"])  # no estimate is blank

    df.loc[0, "source"] = "guess"
    df.to_parquet(tmp_path / "sensitivities.parquet")
    _, problems = real_exposures.load_exposures(tmp_path)
    assert any("unknown source" in p for p in problems)


def _real_page_app(settings: str = "None"):
    """AppTest of the Real data page alone; ``settings`` is Python source evaluated in the script."""
    from streamlit.testing.v1 import AppTest

    src = (
        "import sys, datetime as dt\n"
        f"sys.path.insert(0, {str(ROOT / 'dashboard')!r})\n"
        "import _ui\n"
        "import real_exposures\n"
        f"real_exposures.render({settings})\n"
    )
    return AppTest.from_string(src, default_timeout=TIMEOUT)


def test_real_exposures_page_empty_and_with_file(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "real").mkdir(parents=True)
    ref = reference.reference_dir()
    (data / "reference").mkdir()
    for f in ("assets.csv", "legs.csv", "topics.csv", "link_map.csv"):
        (data / "reference" / f).write_bytes((ref / f).read_bytes())
    monkeypatch.setenv("NARRATIVE_IPCA_DATA_DIR", str(data))
    reference.clear_cache()

    at = _real_page_app().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Real data"
    assert any("Placeholder" in i.value for i in at.info)
    assert not at.get("plotly_chart")
    assert "Settings in use" not in [h.value for h in at.subheader]  # no settings passed

    rows = [("2025-12-31", "S1", "ENERGY_v_WEQ", 0.40, 0.05, "blended", "2023-01-02/2025-12-31", "full", "v1")]
    pd.DataFrame(rows, columns=list(real_exposures.EXPOSURE_COLUMNS)).to_parquet(
        data / "real" / "sensitivities.parquet")
    at = _real_page_app().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("plotly_chart")) == 1
    assert _assert_how_to_read(at.main) == {"real_heatmap"}  # owner request 2026-09-30
    assert _has_guide(at.main, _ui.how_to_read(*_ui.HOW_REAL_STATUS))
    assert _has_guide(at.main, _ui.how_to_read(*_ui.HOW_REAL_CONTRACT))
    # the page says "topic sensitivity" and defines it; no "exposure" anywhere on it (2026-09-30)
    assert _ui.SENSITIVITY_DEFINITION in [c.value for c in at.caption]
    texts = [(k, t.replace(str(tmp_path), "<tmp>")) for k, t in _visible_texts(at)]  # the test's own path
    found = [(k, t[:120]) for k, t in texts if "exposure" in t.lower()]
    assert not found, found
    reference.clear_cache()


def test_real_data_page_lists_settings_and_warns_on_invalid_ones():
    """render(settings): the "Settings in use" table; settings errors are a warning, not an error."""
    settings = (
        "{'values': {**_ui.default_values(), 'sb_forecast_start': dt.date(2025, 6, 2)}, "
        "'errors': ['Forecast start (2025-06-02) must be after training end (2025-06-30).'], "
        "'asset_names': {}}"
    )
    at = _real_page_app(settings).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.title[0].value == "Real data"
    heads = [h.value for h in at.subheader]
    assert heads.index("Settings in use") < heads.index("Status")  # near the top
    table = at.dataframe[0].value
    assert list(table.columns) == ["Group", "Setting", "Value"]
    got = _settings_dict(table)
    assert got["Training window"] == "2025-01-01 to 2025-06-30 (129 weekdays)"
    assert got["Forecast start"] == "2025-06-02"
    assert any("not valid" in w.value and "must be after training end" in w.value for w in at.warning)
    assert _ui.SIMULATION_ONLY_NOTE in [c.value for c in at.caption]
    assert _has_guide(at.main, _ui.how_to_read(*_ui.HOW_REAL_SETTINGS))  # owner request 2026-09-30
    assert _loose_how_to_read(at.main) == []
    _assert_guides_placed(at.main)


@needs_market
def test_app_has_top_navigation_with_three_pages():
    """D82, D90: Simulation lab (default), BKS trace and Real data in the top navigation; the BKS tab links to
    the trace before a run too."""
    at = _app().run()
    _assert_clean(at)
    assert at.title[0].value == "Topic-sensitivity lab"  # the default page is the simulation lab
    src = APP.read_text(encoding="utf-8")
    assert 'title="Simulation lab"' in src and 'title="Real data"' in src and 'position="top"' in src
    assert 'REAL_DATA_URL = "real-data"' in src and "url_path=REAL_DATA_URL" in src
    assert 'title="BKS trace"' in src and 'BKS_TRACE_URL = "bks-trace"' in src and "url_path=BKS_TRACE_URL" in src
    pages = getattr(at, "_registered_pages", None)
    if pages:  # registration order is the navigation order
        assert [info.get("page_name") for info in pages.values()] == ["Simulation lab", "BKS trace", "Real data"]
    links = _page_links(at.tabs[4])
    assert ("Trace this BKS run step by step", "bks-trace") in links


def _page_links(block) -> list[tuple[str, str]]:
    """``(label, url path)`` of every ``st.page_link`` under ``block`` (AppTest has no typed page_link element)."""
    return [(str(e.proto.label), str(e.proto.page)) for e in block.get("page_link")]


def _open_page(at, url_path: str) -> None:
    """Point a run AppTest at a callable page by its URL path (callable pages have no file for ``switch_page``).

    The default page (Simulation lab) is registered with an empty URL path. Without ``_registered_pages``
    (Streamlit 1.62) the hash is the one ``st.Page`` gives a callable page, ``calc_hash(url_path)``; an empty
    hash opens the default page.
    """
    if not hasattr(at, "_page_hash"):
        pytest.skip("this Streamlit version's AppTest cannot open a callable page")
    pages = getattr(at, "_registered_pages", None)
    if pages is None:
        from streamlit.util import calc_hash

        at._page_hash = calc_hash(url_path) if url_path else ""
        return
    hashes = [h for h, info in pages.items() if info.get("url_pathname") == url_path]
    if not hashes:
        pytest.skip("this Streamlit version's AppTest cannot open a callable page")
    at._page_hash = hashes[0]


def _open_real_data_page(at) -> None:
    """Point a run AppTest at the Real data page."""
    _open_page(at, "real-data")


@needs_market
def test_app_real_data_page_shares_the_sidebar():
    """The sidebar is drawn on the Real data page too; its settings show there, invalid ones as a warning."""
    at = _app().run()
    _assert_clean(at)
    _open_real_data_page(at)
    at.run()
    _assert_two_elements(at)
    assert at.title[0].value == "Real data"
    assert "Time windows" in [e.label for e in at.sidebar.get("expander")]
    assert at.button(key="sb_run_bks").disabled  # BKS runs on the Simulation lab and BKS trace pages only
    got = _settings_dict(at.dataframe[0].value)
    assert got["Training window"] == "2025-01-01 to 2025-06-30 (129 weekdays)"
    assert got["Listed assets"] == "55 of 55"
    # a change in the shared sidebar shows on the page; an invalid one is a warning, not an error
    at.select_slider(key="sb_train_months").set_value(12).run()
    _assert_clean(at)
    got = _settings_dict(at.dataframe[0].value)
    assert got["Training length"] == "1 year" and got["Training window"].startswith("2024-07-01 to 2025-06-30")
    at.date_input(key="sb_forecast_start").set_value(dt.date(2025, 6, 2)).run()
    _assert_clean(at)
    assert any("must be after training end" in w.value for w in at.warning)


# ---------------------------------------------------------------------------
# The BKS trace page (D90)
# ---------------------------------------------------------------------------
#: Steps of the trace page in order, as the step radio labels them (``trace.STEPS``).
TRACE_STEPS: dict[str, str] = {
    "summary": "Summary", "inputs": "1 Inputs", "align": "2 Alignment and scaling", "shocks": "3 Attention shocks",
    "instruments": "4 Instruments", "panel": "5 Weekly panel", "fit": "6 Fit and lambda",
    "forecast": "7 Forecast weeks", "implied": "8 Implied sensitivities",
}

#: The charts of each step of the trace page (tuned lambda), by chart key.
TRACE_STEP_CHARTS: dict[str, set[str]] = {
    "summary": {"fig_tr_ladder"},
    "inputs": {"fig_tr_inputs", "fig_tr_truth"},
    "align": {"fig_tr_divisor"},
    "shocks": {"fig_tr_shocks"},
    "instruments": {"fig_tr_instrument", "fig_tr_kernel", "fig_tr_instr_truth"},
    "panel": {"fig_tr_design", "fig_tr_stability"},
    "fit": {"fig_tr_path", "fig_tr_coef_path", "fig_tr_gamma", "fig_tr_topics", "fig_tr_kkt", "fig_tr_factors"},
    "forecast": {"fig_tr_week_r2", "fig_tr_week_scatter", "fig_tr_oos_factors"},
    "implied": {"fig_tr_chain", "fig_tr_capture", "fig_tr_implied_scatter", "fig_tr_sigma"},
}

#: The "How to read" guides (``_ui.HOW_TRACE_*``) each step of the trace page shows.
TRACE_STEP_GUIDES: dict[str, tuple[str, ...]] = {
    "summary": ("HOW_TRACE_STATUS", "HOW_TRACE_LADDER", "HOW_TRACE_LADDER_TABLE", "HOW_TRACE_FINDINGS",
                "HOW_TRACE_CHECKS"),
    "inputs": ("HOW_TRACE_SETTINGS", "HOW_TRACE_SHAPES", "HOW_TRACE_INPUTS", "HOW_TRACE_TRUTH", "HOW_TRACE_CHECKS"),
    "align": ("HOW_TRACE_DIVISOR", "HOW_TRACE_UNITS", "HOW_TRACE_CHECKS"),
    "shocks": ("HOW_TRACE_SHOCKS", "HOW_TRACE_SHOCK_TABLE", "HOW_TRACE_CHECKS"),
    "instruments": ("HOW_TRACE_INSTRUMENT", "HOW_TRACE_KERNEL", "HOW_TRACE_INSTR_TRUTH", "HOW_TRACE_INSTR_ROW",
                    "HOW_TRACE_CHECKS"),
    "panel": ("HOW_TRACE_PAIRING", "HOW_TRACE_DESIGN", "HOW_TRACE_STABILITY", "HOW_TRACE_CHECKS"),
    "fit": ("HOW_TRACE_PATH", "HOW_TRACE_COEF_PATH", "HOW_TRACE_GAMMA", "HOW_TRACE_TOPICS", "HOW_TRACE_TOPIC_TABLE",
            "HOW_TRACE_KKT", "HOW_TRACE_FACTORS", "HOW_TRACE_PATH_TABLE", "HOW_TRACE_CHECKS"),
    "forecast": ("HOW_TRACE_WEEK_R2", "HOW_TRACE_WEEK_SCATTER", "HOW_TRACE_OOS_FACTORS", "HOW_TRACE_WEEK_TABLE",
                 "HOW_TRACE_CHECKS"),
    "implied": ("HOW_TRACE_CHAIN", "HOW_TRACE_SHARES", "HOW_TRACE_CAPTURE", "HOW_TRACE_IMPLIED_SCATTER",
                "HOW_TRACE_SIGMA", "HOW_TRACE_SIGMA_TABLE", "HOW_TRACE_CHAIN_TABLE", "HOW_TRACE_CHECKS"),
}


def _trace_page(at) -> None:
    """Point a run AppTest at the BKS trace page and run it."""
    _open_page(at, "bks-trace")
    at.run()


def _step(at, step: str) -> None:
    """Show one step of the trace page; it must render cleanly under its own subheader."""
    at.radio(key="tr_step").set_value(step).run()
    _assert_clean(at)
    assert TRACE_STEPS[step] in [s.value for s in at.main.subheader], step


def test_trace_step_guides_cover_every_trace_guide():
    """Every HOW_TRACE_* text of _ui belongs to a step of the trace page, or is the tiles' guide above the steps
    (and nothing else is listed); the tiles' guide points to step 6, not to charts "below"."""
    listed = {n for names in TRACE_STEP_GUIDES.values() for n in names}
    assert listed | {"HOW_TRACE_TILES"} == {n for n in dir(_ui) if n.startswith("HOW_TRACE_")}
    tiles = _ui.how_to_read(*_ui.HOW_TRACE_TILES)
    assert "path below" not in tiles and "per-topic split" not in tiles and tiles.count("step 6") == 2
    assert [t for t, _ in _ui.HOW_TRACE_TILES[1]][1:] == [t for t, _ in _ui.HOW_BKS_TILES[1]][1:]
    assert set(TRACE_STEP_CHARTS) == set(TRACE_STEPS) == set(TRACE_STEP_GUIDES)
    assert {k for keys in TRACE_STEP_CHARTS.values() for k in keys} == {
        k for k in HOW_TO_READ_LEADS if k.startswith("fig_tr_")}
    from narrative_ipca.exposure_lab import trace as lab_trace

    assert lab_trace.STEPS == TRACE_STEPS


def test_app_trace_page_before_a_run():
    """D90: before a run the trace page asks for one and never fits by itself; its Run BKS and the sidebar's are
    enabled, and it links back to the Simulation lab."""
    at = _app().run()
    _generic_sidebar(at)
    _trace_page(at)
    _assert_clean(at)
    assert at.title[0].value == "BKS trace"
    assert any(i.value.startswith("Press Run BKS (here or in the sidebar)") for i in at.info)
    assert not at.button(key="tr_run_bks").disabled and not at.button(key="sb_run_bks").disabled
    assert "Back to the Simulation lab" in [label for label, _ in _page_links(at.main)]
    assert "Chosen lambda" not in [m.label for m in at.metric]  # no fit was started
    assert not at.get("plotly_chart") and "tr_step" not in [r.key for r in at.radio]
    assert "bks_requested" not in at.session_state


def test_app_trace_page_traces_the_run_step_by_step():
    """D90, D91: after the trace page's Run BKS every step renders with its charts, each chart with its "How to
    read" guide, every step's HOW_TRACE_* guides, a "What happens here" caption and a link to the next step; the
    page says "topic sensitivity", never "exposure"; the BKS tab then shows the same run."""
    at = _app().run()
    _generic_sidebar(at)
    _trace_page(at)
    at.button(key="tr_run_bks").click().run()
    _assert_clean(at)
    assert "bks_requested" not in at.session_state  # the trace page handled the request itself
    assert "Chosen lambda" in [m.label for m in at.metric]
    assert _has_guide(at.main, _ui.how_to_read(*_ui.HOW_TRACE_TILES))
    assert not _has_guide(at.main, _ui.how_to_read(*_ui.HOW_BKS_TILES))
    assert at.radio(key="tr_step").value == "summary"
    assert list(at.radio(key="tr_step").options) == list(TRACE_STEPS.values())
    assert any(m.value.startswith("**Full history before the cut-off** · K = 3 · lambda") for m in at.markdown)

    texts: list[tuple[str, str]] = []
    for step, charts in TRACE_STEP_CHARTS.items():
        _step(at, step)
        assert _assert_how_to_read(at.main) == charts, step
        caps = [c.value for c in at.main.caption]
        for name in TRACE_STEP_GUIDES[step]:
            assert _has_guide(at.main, _ui.how_to_read(*getattr(_ui, name))), (step, name)
        assert sum(c.startswith("What happens here:\n\n- Inputs: ") for c in caps) == 1, step
        assert sum(c.startswith("Next: ") for c in caps) == 1, step
        assert at.main.dataframe, step  # every step has at least its checks table
        texts += _visible_texts(at)
    found = [(k, t[:120]) for k, t in texts if "exposure" in t.lower()]
    assert not found, found

    # the summary leads with the answer: the identity checks hold, and where BKS departs
    _step(at, "summary")
    status = at.main.dataframe[0].value
    assert list(status["Step"]) == list(TRACE_STEPS.values())[1:]
    assert set(status["Reading"]) <= {"as expected", "departs", "off"} and "off" not in set(status["Reading"])
    assert any(m.value.startswith("**The code does what the formulas say.** All ") for m in at.main.markdown)
    ladder = next(d.value for d in at.main.dataframe if "Variant" in d.value.columns)
    assert "BKS-implied (production)" in list(ladder["Variant"])
    assert "Direct method: Elastic net (benchmark)" in list(ladder["Variant"])
    checks = at.main.dataframe[-1].value
    assert list(checks.columns[:3]) == ["Step", "Check", "Status"]
    assert set(checks["Status"]) <= {"ok", "off", "info"} and set(checks["Kind"]) <= {"identity", "diagnostic"}

    # focus controls: the topic follows the asset's largest true sensitivity until the user picks one
    values = at.session_state["_values"]
    cfg, _, _ = _ui.config_from_values(values, (), None)
    B = LabSession().truth(cfg).B_true
    _step(at, "inputs")
    asset = at.selectbox(key="bks_asset").value
    assert at.selectbox(key="tr_topic").value == str(B[asset].abs().idxmax())
    at.selectbox(key="bks_asset").set_value(str(B.columns[-1])).run()
    _assert_clean(at)
    assert at.selectbox(key="tr_topic").value == str(B[str(B.columns[-1])].abs().idxmax())
    picked = str(B.index[-1])
    at.selectbox(key="tr_topic").set_value(picked).run()
    at.selectbox(key="bks_asset").set_value(asset).run()
    _assert_clean(at)
    assert at.selectbox(key="tr_topic").value == picked  # the user's choice stays
    # the week controls: the return week follows the Eq. 5 rows' week, the forecast week the first one
    _step(at, "instruments")
    week = at.selectbox(key="tr_week")
    assert "(last training week" in week.format_func(week.value)
    _step(at, "forecast")
    assert at.selectbox(key="tr_fweek").value == at.selectbox(key="tr_fweek").options[0]
    _step(at, "shocks")
    at.checkbox(key="tr_shock_all").check().run()
    _assert_clean(at)

    # back on the Simulation lab the BKS tab shows the same run, not a stale one
    _open_page(at, "")
    at.run()
    _assert_clean(at)
    bks = at.tabs[4]
    assert "Chosen lambda" in [m.label for m in bks.metric]
    assert not [w for w in bks.warning if "Settings changed" in w.value]


def test_app_trace_page_training_history_and_lambda_zero():
    """D88, D90: a training-history run and a fixed lambda of 0 trace cleanly through every step; the fixed rule
    has no lambda path, lambda 0 no optimum check."""
    at = _app().run()
    _generic_sidebar(at)
    _trace_page(at)
    at.radio(key="sb_bks_history").set_value("training").run()
    _assert_clean(at)
    assert any(i.value.startswith("Press Run BKS") for i in at.info)  # a new history needs its own fit
    at.button(key="tr_run_bks").click().run()
    _assert_clean(at)
    assert any(m.value.startswith("**Training window only**") for m in at.markdown)
    for step in TRACE_STEPS:
        _step(at, step)
        assert set(_assert_how_to_read(at.main)) == TRACE_STEP_CHARTS[step], step

    at.radio(key="sb_bks_rule").set_value("fixed").run()
    at.number_input(key="sb_bks_lam").set_value(0.0).run()
    at.button(key="tr_run_bks").click().run()
    _assert_clean(at)
    for step in TRACE_STEPS:
        _step(at, step)
        _assert_how_to_read(at.main)
    _step(at, "fit")
    shown = set(_chart_guides(at.main))
    assert not shown & {"fig_tr_path", "fig_tr_coef_path", "fig_tr_kkt"}
    assert {"fig_tr_gamma", "fig_tr_factors"} <= shown
    caps = [c.value for c in at.main.caption]
    assert any(c.startswith("Fixed lambda (0): there is no lambda path") for c in caps)
    assert any(c.startswith("At lambda = 0 there is no penalty") for c in caps)


def test_app_compare_tab_links_to_the_trace():
    """D88, D90: the Compare tab's inspect section links to the trace; a BKS-implied variant with another
    covariance history than the sidebar's opens the trace with that history, reusing the Compare run's fit."""
    at = _app().run()
    _generic_sidebar(at)
    at.button(key="cm_run_bks").click().run()
    _assert_clean(at)
    at.selectbox(key="cm_inspect").set_value("bks_implied").run()
    _assert_clean(at)
    assert ("Trace the BKS-implied sensitivities step by step", "bks-trace") in _page_links(at.tabs[3])
    at.selectbox(key="cm_inspect").set_value("bks_implied_train").run()
    _assert_clean(at)
    assert not _page_links(at.tabs[3])
    at.button(key="cm_trace").click().run()
    _assert_clean(at)
    assert at.title[0].value == "BKS trace"
    assert at.radio(key="sb_bks_history").value == "training"
    assert "Chosen lambda" in [m.label for m in at.metric]  # traced from the Compare run's fit, no new run
    assert any(m.value.startswith("**Training window only**") for m in at.markdown)
    assert at.radio(key="tr_step").value == "summary"


def test_app_compare_tab_says_when_the_traced_fit_left_the_cache():
    """D80, D90 (review 2026-09-30): the Compare tab keeps offering the trace of a BKS-implied variant whose fit
    has left the two-entry fit cache (the comparison still holds its scores), and says under it that the trace
    page will ask for a refit; the trace page then says the fit left the cache."""
    note = "The BKS fit of this variant is no longer in the cache"
    at = _app().run()
    _generic_sidebar(at)
    at.button(key="cm_run_bks").click().run()
    at.selectbox(key="cm_inspect").set_value("bks_implied").run()
    _assert_clean(at)
    assert not [c for c in at.tabs[3].caption if c.value.startswith(note)]
    at.slider(key="sb_bks_K").set_value(4).run()
    at.button(key="cm_run_bks").click().run()  # two more fits push both K = 3 fits out of the cache
    at.slider(key="sb_bks_K").set_value(3).run()
    _assert_clean(at)
    assert ("Trace the BKS-implied sensitivities step by step", "bks-trace") in _page_links(at.tabs[3])
    assert [c for c in at.tabs[3].caption if c.value.startswith(note)]
    _trace_page(at)
    _assert_clean(at)
    assert any("is no longer in the cache" in i.value for i in at.info)


def test_app_trace_page_shows_a_refusal_not_an_eviction():
    """D80, D90 (review 2026-09-30): when BKS refuses the settings (K = 3 factors on 3 assets), the trace page shows
    the reason, whether the refusal came from its own Run BKS or from the Compare tab's, and never the note that a
    cached fit left the cache; a refused fit does not join this browser session's fit keys."""
    reason = "factors need more than 3 assets"

    def assert_refusal(at) -> None:
        assert not at.exception, [e.value for e in at.exception]
        assert at.error and all(reason in e.value for e in at.error), [e.value for e in at.error]
        assert not [i.value for i in at.info if "no longer in the cache" in i.value]
        assert "Chosen lambda" not in [m.label for m in at.metric] and "tr_step" not in [r.key for r in at.radio]

    at = _app().run()
    _generic_sidebar(at)
    at.slider(key="sb_n_generic_assets").set_value(3).run()
    _trace_page(at)
    for _ in range(2):  # pressing Run BKS again gives the same reason, not the eviction note
        at.button(key="tr_run_bks").click().run()
        assert_refusal(at)
    assert not at.session_state["bks_fit_keys"]
    at.slider(key="sb_forecast_weeks").set_value(3).run()  # a new bks key, the same fit key: still the reason
    assert_refusal(at)

    at = _app().run()  # the Compare tab's Run BKS refuses both variants; the trace page shows why
    _generic_sidebar(at)
    at.slider(key="sb_n_generic_assets").set_value(3).run()
    at.button(key="cm_run_bks").click().run()
    assert not at.exception
    assert _fit_errors(at) and not at.session_state["bks_fit_keys"]
    _trace_page(at)
    assert_refusal(at)


def test_app_compare_tab_gives_a_refusal_reason():
    """D80, D88 (review 2026-09-30): after the sidebar's Run BKS refuses the settings (K = 3 factors on 3 assets), the
    Compare tab gives the reason for the variant with the sidebar's history, also after a change that keeps the fit
    key (forecast weeks), not "BKS has not been run"; a refused fit does not join this browser session's fit keys."""
    reason = "factors need more than 3 assets"
    at = _app().run()
    _generic_sidebar(at)
    at.slider(key="sb_n_generic_assets").set_value(3).run()
    at.button(key="sb_run_bks").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.session_state["bks_fit_keys"] and any(reason in v for v in _fit_errors(at).values())
    at.slider(key="sb_forecast_weeks").set_value(3).run()  # a new bks key, the same fit key
    assert not at.exception, [e.value for e in at.exception]
    cm = at.tabs[3]
    box = next(i.value for i in cm.info if i.value.startswith("BKS-implied is not available"))
    full = next(line for line in box.splitlines() if line.startswith("- BKS-implied (full history):"))
    assert "BKS could not run with these settings:" in full and reason in full, full
    assert reason in cm.dataframe[0].value.loc["bks_implied", "Note"]


def _fit_errors(at) -> dict[str, str]:
    """The refusals stored per BKS fit key (``bks_fit_errors``)."""
    return dict(at.session_state["bks_fit_errors"]) if "bks_fit_errors" in at.session_state else {}


def _answer_script(trace, dashboard: str) -> None:
    """AppTest script: the trace page's answer box and findings list for ``trace``."""
    import sys

    if dashboard not in sys.path:
        sys.path.insert(0, dashboard)
    import trace_page

    trace_page._answer(trace)
    trace_page._findings(trace)


def test_trace_answer_keeps_defects_apart_from_departures():
    """G.16, D90 (review 2026-09-30): an identity check that is off is a defect. The answer box names it in its
    warning only, never among the departures, which it groups by origin (none is a coding error); the findings
    list labels it "Defect" and puts it first, and shows each departure's and note's origin. Without a defect
    the box says the checks agree to rounding or to the solver's stopping tolerance."""
    from streamlit.testing.v1 import AppTest

    import trace_page
    from narrative_ipca.exposure_lab import trace as lab_trace

    cfg = _generic_cfg(train_start="2021-01-01", train_end="2022-12-30", forecast_start="2023-01-02")
    s = LabSession()
    s.bks_fit(cfg)
    implied = s.bks_implied(cfg)
    args = (s.bks_panel(cfg), s.bks_fit(cfg), s.bks(cfg), s.simulation(cfg), s.shocks(cfg), s.truth(cfg), cfg.window,
            cfg.bks)
    good = lab_trace.build_trace(*args, implied, path_trace=False)
    # a production result that disagrees with its formula: the chain's identity check goes off
    bad = lab_trace.build_trace(*args, dataclasses.replace(implied, B_hat=implied.B_hat + 1e-6), path_trace=False)
    off = [c.name for c in bad.checks if c.kind == lab_trace.IDENTITY and c.status == lab_trace.OFF]
    assert off and not [c for c in good.checks if c.kind == lab_trace.IDENTITY and c.status == lab_trace.OFF]
    assert {f["severity"] for f in bad.findings if f["origin"] == "defect"} == {"defect"}
    assert len([f for f in bad.findings if f["severity"] == "defect"]) == len(off)

    dashboard = str(ROOT / "dashboard")
    for trace in (good, bad):
        at = AppTest.from_function(_answer_script, default_timeout=60, args=(trace, dashboard)).run()
        assert not at.exception, [e.value for e in at.exception]
        md = [m.value for m in at.markdown]
        box = next((m for m in md if m.startswith("**Where BKS departs")), "")
        findings = md[-1]
        defects = [f for f in trace.findings if f["severity"] == "defect"]
        departures = [f for f in trace.findings if f["severity"] == "departure"]
        if defects:
            assert len(at.warning) == 1 and all(name in at.warning[0].value for name in off)
            assert not any(m.startswith("**The code does what the formulas say.**") for m in md)
            assert findings.startswith("- **Defect, ")
        else:
            assert not at.warning
            assert md[0].startswith("**The code does what the formulas say.**")
            assert "to rounding or to the solver's stopping tolerance" in md[0]
            assert "**Defect, " not in findings
        assert "Check off" not in box and not any(f["title"] in box for f in defects)
        assert "properties of the method, not coding errors" not in "\n".join(md)
        if departures:
            assert box.startswith("**Where BKS departs from what it should be.** None of these is a coding error.")
            for f in departures:
                assert f"{f['title']}." in box
                assert f"**Departure ({trace_page.ORIGIN_WORDS[f['origin']]}), " in findings
                assert trace_page.ORIGIN_GROUPS[f["origin"]] in box
        else:
            assert any(m.startswith("**No departure found:**") for m in md)
        for f in trace.findings:
            if f["severity"] == "note":
                assert f"**Note ({trace_page.ORIGIN_WORDS[f['origin']]}), " in findings


def test_bks_runtime_warning():
    """G.9: the run-time warning above 100 topics (the BKS tab and the trace page show it before Run BKS)."""
    assert _ui.bks_runtime_warning(20) == _ui.bks_runtime_warning(100) == ""
    text = _ui.bks_runtime_warning(150)
    assert text.startswith("150 topics: a BKS run takes") and "Keep the grid coarse." in text
    src = APP.read_text(encoding="utf-8")
    body = src[src.index("def bks_trace_page"):]
    assert body.index("_bks_runtime_warning(") < body.index('key="tr_run_bks"')
    tab = src[src.index("def bks_tab"):src.index("def lists_tab")]
    assert tab.index("_bks_runtime_warning(") < tab.index('key="bks_run_tab"')


@needs_market
def test_app_trace_page_on_the_defaults():
    """D90, G.15.1: on the dashboard defaults the summary answers where BKS departs: every identity check holds;
    the ladder puts BKS-implied (Spearman about 0.19) below the instruments alone (about 0.91) and the direct
    method; the findings name the lost directions and the lambda choice."""
    at = _app().run()
    _trace_page(at)
    at.button(key="tr_run_bks").click().run()
    _assert_clean(at)
    assert _assert_how_to_read(at.main) == {"fig_tr_ladder"}
    spec = next(el.proto.spec for el in at.get("plotly_chart") if str(el.proto.id).endswith("fig_tr_ladder"))
    assert "BKS-implied (production)" in spec and "Direct method: Elastic net (benchmark)" in spec
    ladder = next(d.value for d in at.main.dataframe if "Variant" in d.value.columns).set_index("Variant")
    assert round(float(ladder.loc["BKS-implied (production)", "Spearman"]), 2) == 0.19
    assert round(float(ladder.loc["Instruments, shocks' covariance over the same history", "Spearman"]), 2) == 0.91
    assert float(ladder.loc["Direct method: Elastic net (benchmark)", "Spearman"]) > 0.5
    assert list(ladder.index).index("Direct method: Elastic net (benchmark)") > list(ladder.index).index(
        "BKS-implied (production)")
    assert any(m.value.startswith("**The code does what the formulas say.** All 42 ") for m in at.main.markdown)
    # the answer box groups the four departures by origin: three of the method, the Sigma_z window of this
    # implementation; none is called a coding error, and no defect is listed
    box = next(m.value for m in at.main.markdown if m.value.startswith("**Where BKS departs"))
    method, _, impl = box.partition("Choices of this implementation")
    assert "None of these is a coding error." in box and "Check off" not in box and "Defect" not in box
    assert "Properties of the method (BKS itself):" in method and impl
    for title in ("6 Fit and lambda: The lambda choice follows noise.",
                  "8 Implied sensitivities: The fit's directions keep little of the instruments.",
                  "8 Implied sensitivities: The implied sensitivities are further from the truth than zero."):
        assert title in method, title
    assert "8 Implied sensitivities: Instruments and the shocks' covariance cover different days." in impl
    findings = "\n".join(m.value for m in at.main.markdown)
    assert ("**Departure (method), 8 Implied sensitivities: The fit's directions keep little of the instruments.**"
            in findings)
    assert ("**Departure (implementation choice), 8 Implied sensitivities: Instruments and the shocks' covariance "
            "cover different days.**" in findings)
    assert "Departure (method), 6 Fit and lambda" in findings and "Note (implementation choice), 8 Implied" in findings
    assert "**Defect, " not in findings
    assert findings.index("**Departure (") < findings.index("**Note (")
    status = at.main.dataframe[0].value.set_index("Step")
    assert status.loc["8 Implied sensitivities", "Reading"] == "departs"
    assert status.loc["5 Weekly panel", "Reading"] == "as expected"
    # the steps render on the listed universe too (55 assets, 20 manual topics)
    for step in ("instruments", "fit", "implied"):
        _step(at, step)
        assert _assert_how_to_read(at.main) == TRACE_STEP_CHARTS[step], step
        if step == "instruments":  # the reference's divisor is named per history
            spec = next(el.proto.spec for el in at.main.get("plotly_chart")
                        if str(el.proto.id).endswith("fig_tr_instr_truth"))
            assert "full-sample volatility (panel units)" in spec  # Plotly's JSON escapes the slash
        if step == "fit":  # which topics should matter: A5 kept, A4 and S2 (ranks 2 and 3) never selected
            topics = next(d.value for d in at.main.dataframe if "Rank" in d.value.columns)
            assert list(topics["Rank"])[:3] == [1, 2, 3] and list(topics["Selected"])[:3] == ["yes", "no", "no"]
            assert topics["Enters at lambda"].iloc[1:3].isna().all()
    assert at.selectbox(key="bks_asset").value == _ui.DEFAULT_CONTRIB_ASSET


# ---------------------------------------------------------------------------
# Explanation elements (G.9, D91)
# ---------------------------------------------------------------------------
#: Labels of the bordered expanders: they hold content or tools, never an explanation (D91).
CONTENT_EXPANDERS: set[str] = {
    "Universe", "Topics", "Sensitivities", "Time windows", "Direct estimator", "BKS model",
    "Stage timings of this page", "Long/short view per asset", "Simulated attention of the largest contributors",
    "Add a link", "data/market/README.md: sources, conventions, TBC items, QA",
}

#: Largest "?" tooltip (``help=``), in characters; a longer explanation is a guide toggle (G.9).
MAX_HELP = 450


def _assert_two_elements(at) -> None:
    """D91: explanations are "?" tooltips of at most :data:`MAX_HELP` characters and collapsed compact guide
    toggles placed as G.9 says; no "How to read" text is a loose caption, nothing is a popover, dialog or status
    block, and every other expander holds content."""
    _assert_clean(at)  # a duplicate guide key would raise here
    assert _loose_how_to_read(at.main) == [] and _loose_how_to_read(at.sidebar) == []
    assert not at.get("popover") and not at.get("dialog") and not at.status
    others = {str(e.label) for e in at.expander if _guide_name(e) is None}
    assert others <= CONTENT_EXPANDERS, others - CONTENT_EXPANDERS
    for name, _, node in _guides(at.main):
        assert node.proto.type == 1 and node.proto.expanded is False, name  # compact, collapsed on load
    _assert_guides_placed(at.main)
    helps = [str(m.help or "") for m in at.metric]
    for kind in ("button", "download_button", "checkbox", "toggle", "slider", "number_input", "date_input",
                 "selectbox", "radio", "select_slider", "multiselect"):
        helps += [str(getattr(w, "help", "") or "") for w in getattr(at, kind)]
    longest = max(helps, key=len, default="")
    assert len(longest) <= MAX_HELP, longest[:80]


@needs_market
def test_app_explanations_use_two_elements():
    """Owner request 2026-10-02 (D91): explanations use two elements only, the "?" tooltip and the collapsed
    "How to read" guide toggle, on the Simulation lab (with the other contributions view and the roll-up) and on
    every step of the BKS trace page."""
    at = _app().run()
    at.radio(key="tc_view").set_value("Return attribution").run()
    at.checkbox(key="tc_rollup").check().run()
    _assert_two_elements(at)
    assert "Why this method" in _guide_names(at.tabs[2])
    # the Compare methods settings note is the Methods tooltip, not a caption
    assert "All methods share the threshold tau = 0.05" in at.multiselect(key="cm_methods").help
    assert not [c.value for c in at.caption if "keeps its settings" in c.value]
    # the market data README is content in a bordered expander, collapsed on load (owner decision 2026-10-02)
    readme = [e for e in at.expander if str(e.label).startswith("data/market/README.md")]
    assert readme and not any(e.proto.expanded for e in readme) and all(e.proto.type == 0 for e in readme)

    _generic_sidebar(at)
    _trace_page(at)
    at.button(key="tr_run_bks").click().run()
    _assert_clean(at)
    assert "How to read the tiles" in _guide_names(at.main)
    for step in TRACE_STEPS:
        _step(at, step)
        _assert_two_elements(at)
