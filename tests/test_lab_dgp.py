"""Tests of the lab's topics, link map and data-generating process (DESIGN.md G.3-G.5; D57-D64, D71).

The Monte Carlo truth check is the central test: on ~80,000 days of i.i.d.
correlated Gaussian returns, the OLS regression of the standardised returns
on the observed standardised shocks must reproduce the population truth
``B_true`` and ``r2_true`` of G.5.3.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from narrative_ipca.exposure_lab import reference
from narrative_ipca.exposure_lab.config import (
    ExposureConfig,
    LabConfig,
    RandomLinkConfig,
    TopicSetConfig,
    WindowConfig,
)
from narrative_ipca.exposure_lab.dgp import (
    attenuation,
    observed_shocks,
    simulate_from_links,
    simulate_lab,
    topic_noise,
    truth_for_window,
)
from narrative_ipca.exposure_lab.dgp import _pairwise_corr
from narrative_ipca.exposure_lab.links import (
    LINK_COLUMNS,
    build_link_map,
    build_topic_table,
    design_matrix,
)
from narrative_ipca.exposure_lab.types import LinkMap, MarketData, SimData

LINKS_LOGGER = "narrative_ipca.exposure_lab.links"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
def _asset_table(ids: list[str], source: str = "generic") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ids,
            "asset_class": "Equity",
            "sub_class": "generic",
            "long_leg": "",
            "short_leg": "",
            "source": source,
            "order": np.arange(1, len(ids) + 1),
        },
        index=pd.Index(ids, name="asset_id"),
    )


def _generic_ids(n: int) -> list[str]:
    return [f"G_ASSET_{i:03d}" for i in range(1, n + 1)]


def _gaussian_market(n_days: int, corr: float, seed: int, n_assets: int = 6, start: str = "1800-01-01") -> MarketData:
    """i.i.d. Gaussian returns with equal pairwise correlation ``corr`` and vols 0.5%-1.5% daily."""
    # The Mon-Fri days of pd.bdate_range(start, periods=n_days), built with numpy: 80,000 business days span
    # about 307 years, and under pandas 2 the business-day offset arithmetic goes through a nanosecond
    # Timedelta, which overflows beyond about 292 years (pandas 3 works in microseconds).
    days = np.busday_offset(np.datetime64(start, "D"), np.arange(n_days), roll="forward")
    cal = pd.DatetimeIndex(days.astype("datetime64[ns]"))
    C = np.full((n_assets, n_assets), corr)
    np.fill_diagonal(C, 1.0)
    vols = np.linspace(0.005, 0.015, n_assets)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_days, n_assets)) @ np.linalg.cholesky(C).T * vols + 2e-4
    ids = _generic_ids(n_assets)
    returns = pd.DataFrame(X, index=cal, columns=pd.Index(ids, name="asset_id"))
    return MarketData(returns=returns, assets=_asset_table(ids))


def _full_window(market: MarketData, w: int) -> WindowConfig:
    cal = market.calendar
    return WindowConfig(
        train_start=str(cal[0].date()),
        train_end=str(cal[-1].date()),
        forecast_start=str((cal[-1] + pd.Timedelta(days=3)).date()),
        shock_window=w,
    )


@pytest.fixture(scope="module")
def listed_assets() -> pd.DataFrame:
    a = reference.load_assets().copy()
    a["source"] = "real"
    return a


@pytest.fixture(scope="module")
def link_csv() -> pd.DataFrame:
    return pd.read_csv(reference.reference_dir() / "link_map.csv")


@pytest.fixture(scope="module")
def mc_market() -> MarketData:
    return _gaussian_market(80_000, 0.5, seed=12345)


# Hand-set design for the Monte Carlo check: three linked topics with mixed
# tiers and signs, one pure-noise topic (G004).
_IDS6 = _generic_ids(6)
MC_OVERRIDES = (
    ("G001", _IDS6[0], "strong", 1),
    ("G001", _IDS6[1], "moderate", -1),
    ("G001", _IDS6[4], "weak", 1),
    ("G002", _IDS6[2], "strong", -1),
    ("G002", _IDS6[3], "moderate", 1),
    ("G002", _IDS6[5], "weak", 1),
    ("G003", _IDS6[5], "strong", 1),
    ("G003", _IDS6[0], "weak", -1),
)


def _mc_config(market: MarketData, w: int, lead: int, **exposure) -> LabConfig:
    ex = dict(n_betas=3, beta_1=0.5, beta_2=0.3, beta_3=0.1, lead_days=lead, link_overrides=MC_OVERRIDES, seed=7,
              noise_seed=7)
    ex.update(exposure)
    return LabConfig(
        topics=TopicSetConfig(manual="none", n_generic=4, generic_signal_share=0.0),
        exposure=ExposureConfig(**ex),
        window=_full_window(market, w),
    )


# ---------------------------------------------------------------------------
# Topics (G.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "manual, n_manual, first, last",
    [("sector", 11, "S1", "S11"), ("economic", 9, "A1", "B3"), ("both", 20, "S1", "B3"), ("none", 0, None, None)],
)
def test_topic_table_sets(manual, n_manual, first, last):
    tt = build_topic_table(TopicSetConfig(manual=manual, n_generic=12))
    t = tt.table
    assert list(t.columns) == ["name", "group", "ontology", "scope", "order"]
    assert t.index.name == "topic_id"
    assert len(t) == n_manual + 12
    assert t["order"].tolist() == list(range(1, len(t) + 1))
    if n_manual:
        assert tt.ids[0] == first and tt.ids[n_manual - 1] == last
        assert set(t["ontology"].iloc[:n_manual]) <= {"sector", "economic"}
        assert (t["scope"].iloc[:n_manual] != "").all()
    gen = t.iloc[n_manual:]
    assert gen.index.tolist() == [f"G{j:03d}" for j in range(1, 13)]
    assert gen.loc["G007", "name"] == "Generic topic 007"
    assert set(gen["group"]) == {"Generic"} and set(gen["ontology"]) == {"generic"} and set(gen["scope"]) == {""}


def test_topic_table_manual_only():
    t = build_topic_table(TopicSetConfig(manual="economic", n_generic=0)).table
    assert t.index.tolist() == ["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3"]
    assert t.loc["A1", "group"] == "Macro" and t.loc["B1", "group"] == "Micro"


# ---------------------------------------------------------------------------
# Link map (G.4)
# ---------------------------------------------------------------------------
def test_default_links_manual_x_listed(listed_assets, link_csv):
    tc = TopicSetConfig(manual="both")
    ec = ExposureConfig()
    tt = build_topic_table(tc)
    lm = build_link_map(tt, listed_assets, tc, ec).table
    assert tuple(lm.columns) == LINK_COLUMNS
    assert len(lm) == len(link_csv)
    assert set(lm["origin"]) == {"default"}
    got = {(r.topic_id, r.asset_id): (r.tier, int(r.sign), r.mechanism) for r in lm.itertuples()}
    want = {(r.topic_id, r.asset_id): (r.tier, int(r.sign), r.mechanism) for r in link_csv.itertuples()}
    assert got == want
    # sorted by topic order, then asset order
    tpos = {t: i for i, t in enumerate(tt.ids)}
    apos = {a: i for i, a in enumerate(listed_assets.index)}
    keys = [(tpos[t], apos[a]) for t, a in zip(lm["topic_id"], lm["asset_id"])]
    assert keys == sorted(keys)
    assert lm["sign"].dtype == np.int64


def test_default_links_restricted_to_run(listed_assets, link_csv):
    subset = ["ENERGY_v_WEQ", "NOK_v_USD", "GLOBAL_DURATION", "CHF_v_USD"]
    assets = listed_assets.loc[subset]
    tc = TopicSetConfig(manual="sector")
    lm = build_link_map(build_topic_table(tc), assets, tc, ExposureConfig()).table
    want = link_csv[link_csv["topic_id"].str.startswith("S") & link_csv["asset_id"].isin(subset)]
    assert len(lm) == len(want) > 0
    assert set(lm["asset_id"]) <= set(subset)
    assert set(lm["origin"]) == {"default"}  # listed universe: no random links for manual topics


@pytest.mark.parametrize(
    "n_betas, expected",
    [(1, (0.4, 0.4, 0.4)), (2, (0.4, 0.2, 0.2)), (3, (0.4, 0.2, 0.1))],
)
def test_n_betas_merge(listed_assets, n_betas, expected):
    ec = ExposureConfig(n_betas=n_betas, beta_1=0.4, beta_2=0.2, beta_3=0.1)
    assert tuple(ec.tier_values()[t] for t in ("strong", "moderate", "weak")) == expected
    tc = TopicSetConfig(manual="both")
    tt = build_topic_table(tc)
    links = build_link_map(tt, listed_assets, tc, ec)
    W = design_matrix(links, tt, listed_assets, ec)
    assert W.shape == (20, 55)
    assert W.index.tolist() == tt.ids and W.columns.tolist() == listed_assets.index.tolist()
    lm = links.table
    assert int((W != 0).to_numpy().sum()) == len(lm)
    for tier, value in zip(("strong", "moderate", "weak"), expected):
        rows = lm[lm["tier"] == tier]
        vals = np.array([W.loc[t, a] for t, a in zip(rows["topic_id"], rows["asset_id"])])
        np.testing.assert_allclose(vals, rows["sign"].to_numpy() * value)


def test_overrides_add_modify_remove_ignore(listed_assets, caplog):
    tc = TopicSetConfig(manual="sector")
    base = build_link_map(build_topic_table(tc), listed_assets, tc, ExposureConfig()).table
    assert ((base["topic_id"] == "S1") & (base["asset_id"] == "ENERGY_v_WEQ")).any()
    assert ((base["topic_id"] == "S1") & (base["asset_id"] == "NOK_v_USD")).any()
    assert not ((base["topic_id"] == "S1") & (base["asset_id"] == "CHF_v_USD")).any()
    ov = (
        ("S1", "CHF_v_USD", "moderate", -1),  # add
        ("S1", "ENERGY_v_WEQ", "weak", -1),  # modify
        ("S1", "NOK_v_USD", "none", 1),  # remove
        ("A1", "CHF_v_USD", "strong", 1),  # topic not in this run
        ("S2", "NOT_AN_ASSET", "strong", 1),  # asset not in this run
    )
    ec = ExposureConfig(link_overrides=ov)
    with caplog.at_level(logging.INFO, logger=LINKS_LOGGER):
        lm = build_link_map(build_topic_table(tc), listed_assets, tc, ec).table
    ignored = [r for r in caplog.records if "ignored: pair not in this run" in r.getMessage()]
    assert len(ignored) == 2
    lm_i = lm.set_index(["topic_id", "asset_id"])
    assert tuple(lm_i.loc[("S1", "CHF_v_USD"), ["tier", "sign", "origin", "mechanism"]]) == (
        "moderate", -1, "override", "Session edit",
    )
    assert tuple(lm_i.loc[("S1", "ENERGY_v_WEQ"), ["tier", "sign", "origin"]]) == ("weak", -1, "override")
    assert ("S1", "NOK_v_USD") not in lm_i.index
    assert len(lm) == len(base)  # +1 added, -1 removed
    W = design_matrix(LinkMap(lm), build_topic_table(tc), listed_assets, ec)
    assert W.loc["S1", "CHF_v_USD"] == pytest.approx(-0.15)
    assert W.loc["S1", "ENERGY_v_WEQ"] == pytest.approx(-0.05)
    assert W.loc["S1", "NOK_v_USD"] == 0.0


def _check_random_topic_links(lm: pd.DataFrame, topic: str, n_assets: int, counts=(1, 2, 3)):
    rows = lm[lm["topic_id"] == topic]
    n_total = min(sum(counts), n_assets)
    assert len(rows) == n_total
    assert rows["asset_id"].is_unique
    assert set(rows["origin"]) == {"random"} and set(rows["mechanism"]) == {"Random link (seeded)"}
    assert set(rows["sign"]) <= {-1, 1}
    tiers = ["strong"] * counts[0] + ["moderate"] * counts[1] + ["weak"] * counts[2]
    assert sorted(rows["tier"]) == sorted(tiers[:n_total])


def test_random_map_generic_universe_counts():
    assets = _asset_table(_generic_ids(10))
    tc = TopicSetConfig(manual="sector", n_generic=10, generic_signal_share=0.3)
    tt = build_topic_table(tc)
    lm = build_link_map(tt, assets, tc, ExposureConfig(seed=3)).table
    linked = lm["topic_id"].unique().tolist()
    manual = [t for t in tt.ids if t.startswith("S")]
    generic_linked = [t for t in linked if t.startswith("G")]
    assert set(manual) <= set(linked)  # generic universe: every manual topic gets random links
    assert len(generic_linked) == 3  # round(0.3 * 10)
    for t in linked:
        _check_random_topic_links(lm, t, 10)
    # both signs occur over the whole map
    assert set(lm["sign"]) == {-1, 1}


def test_random_map_capped_at_n_assets():
    assets = _asset_table(_generic_ids(4))
    tc = TopicSetConfig(manual="none", n_generic=5, generic_signal_share=1.0)
    lm = build_link_map(build_topic_table(tc), assets, tc, ExposureConfig()).table
    assert lm["topic_id"].nunique() == 5
    for t in lm["topic_id"].unique():
        _check_random_topic_links(lm, t, 4)  # 1 strong, 2 moderate, 1 weak
    rc = RandomLinkConfig(n_strong=2, n_moderate=0, n_weak=1)
    lm2 = build_link_map(build_topic_table(tc), assets, tc, ExposureConfig(random_links=rc)).table
    for t in lm2["topic_id"].unique():
        _check_random_topic_links(lm2, t, 4, counts=(2, 0, 1))


def test_generic_only_run_needs_no_reference_files(tmp_path, monkeypatch):
    monkeypatch.setenv(reference.DATA_DIR_ENV, str(tmp_path))  # empty data folder
    assets = _asset_table(_generic_ids(6))
    tc = TopicSetConfig(manual="none", n_generic=4, generic_signal_share=0.5)
    lm = build_link_map(build_topic_table(tc), assets, tc, ExposureConfig()).table
    assert lm["topic_id"].nunique() == 2
    with pytest.raises(FileNotFoundError):
        build_topic_table(TopicSetConfig(manual="sector"))


def test_random_map_rounding_half_up():
    assets = _asset_table(_generic_ids(8))
    tc = TopicSetConfig(manual="none", n_generic=5, generic_signal_share=0.5)
    lm = build_link_map(build_topic_table(tc), assets, tc, ExposureConfig()).table
    assert lm["topic_id"].nunique() == 3  # 2.5 -> 3


def _generic_rows(lm: pd.DataFrame) -> pd.DataFrame:
    return lm[lm["topic_id"].str.startswith("G")].reset_index(drop=True)


def test_random_map_determinism_and_stability(listed_assets):
    assets = _asset_table(_generic_ids(12))
    ec = ExposureConfig(seed=11)
    maps = {}
    for manual in ("none", "sector", "economic", "both"):
        tc = TopicSetConfig(manual=manual, n_generic=20, generic_signal_share=0.25)
        maps[manual] = build_link_map(build_topic_table(tc), assets, tc, ec).table
    # determinism
    tc = TopicSetConfig(manual="sector", n_generic=20, generic_signal_share=0.25)
    pd.testing.assert_frame_equal(maps["sector"], build_link_map(build_topic_table(tc), assets, tc, ec).table)
    # generic draws do not change with the manual set
    ref_rows = _generic_rows(maps["none"])
    assert ref_rows["topic_id"].nunique() == 5
    for manual in ("sector", "economic", "both"):
        pd.testing.assert_frame_equal(_generic_rows(maps[manual]), ref_rows)
    # nor do manual topics' draws (S1 in 'sector' and in 'both')
    s1 = lambda m: m[m["topic_id"] == "S1"].reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(s1(maps["sector"]), s1(maps["both"]))
    # generic topics in the listed universe: same topics selected, links to listed assets
    tc = TopicSetConfig(manual="both", n_generic=20, generic_signal_share=0.25)
    lm_listed = build_link_map(build_topic_table(tc), listed_assets, tc, ec).table
    assert set(lm_listed.loc[lm_listed["origin"] == "random", "topic_id"]) == set(ref_rows["topic_id"])
    assert set(lm_listed.loc[lm_listed["topic_id"].str.match(r"^[SAB]\d"), "origin"]) == {"default"}
    # a different seed redraws
    other = build_link_map(build_topic_table(tc), assets, tc, ExposureConfig(seed=12)).table
    assert not _generic_rows(other).equals(ref_rows)
    # raising the share only adds linked topics
    tc_hi = TopicSetConfig(manual="none", n_generic=20, generic_signal_share=0.6)
    hi = build_link_map(build_topic_table(tc_hi), assets, tc_hi, ec).table
    assert set(ref_rows["topic_id"]) < set(hi["topic_id"])
    for t in ref_rows["topic_id"].unique():
        pd.testing.assert_frame_equal(
            hi[hi["topic_id"] == t].reset_index(drop=True), ref_rows[ref_rows["topic_id"] == t].reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Data-generating process (G.5)
# ---------------------------------------------------------------------------
def test_attenuation_values():
    # a ~ 1 / sqrt(1 + 1/w), slightly lower because of the slow component
    a1 = attenuation(0.02, 0.995, 0.1, 1)
    a5 = attenuation(0.02, 0.995, 0.1, 5)
    assert 0.70 < a1 < 1 / np.sqrt(2)
    assert 0.89 < a5 < 1 / np.sqrt(1.2)
    assert attenuation(0.02, 0.995, 0.0, 5) == pytest.approx(1 / np.sqrt(1.2))
    # brute-force Var(D_g) for w = 3
    phi, sd, w = 0.9, 0.3, 3
    gam = lambda h: sd**2 * phi**abs(h) / (1 - phi**2)  # noqa: E731
    wts = np.r_[1.0, -np.ones(w) / w]
    var_dg = sum(wts[i] * wts[j] * gam(i - j) for i in range(w + 1) for j in range(w + 1))
    v = 1.0 * (1 + 1 / w) + var_dg
    assert attenuation(1.0, phi, sd, w) == pytest.approx(1 / np.sqrt(v), rel=1e-12)


@pytest.mark.parametrize("w", [1, 5])
@pytest.mark.parametrize("lead", [0, 1])
def test_monte_carlo_truth(mc_market, w, lead):
    """OLS on ~80,000 days reproduces B_true and r2_true (G.5.3)."""
    cfg = _mc_config(mc_market, w, lead)
    sim = simulate_lab(cfg, mc_market)
    tr = sim.truth
    assert tr.shock_window == w and sim.lead_days == lead
    assert (tr.feasibility_scale == 1.0).all()
    # the design is what the overrides set
    assert sim.links.table["origin"].eq("override").all() and len(sim.links.table) == len(MC_OVERRIDES)
    assert tr.W.loc["G001", _IDS6[1]] == pytest.approx(-0.3)
    assert (tr.W.loc["G004"] == 0).all()

    sh = observed_shocks(sim.attention, w, mc_market.calendar[0], mc_market.calendar[-1])
    r = mc_market.returns
    y = ((r - r.mean()) / r.std(ddof=0)).shift(-lead)
    ok = (sh.s_hat.notna().all(axis=1) & y.notna().all(axis=1)).to_numpy()
    X = np.column_stack([np.ones(ok.sum()), sh.s_hat.to_numpy()[ok]])
    Y = y.to_numpy()[ok]
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    B_ols = coef[1:]
    resid = Y - X @ coef
    r2_ols = 1.0 - resid.var(axis=0) / Y.var(axis=0)

    np.testing.assert_array_less(np.abs(B_ols - tr.B_true.to_numpy()), 0.02)
    np.testing.assert_array_less(np.abs(r2_ols - tr.r2_true.to_numpy()), 0.02)
    assert tr.r2_true.min() > 0.02  # every asset carries some signal
    # the noise topic has no true exposure; spillovers make B_true dense elsewhere
    assert np.abs(tr.B_true.loc["G004"]).max() < 1e-12
    assert abs(tr.B_true.loc["G001", _IDS6[2]]) > 0.03  # unlinked pair, correlated asset

    # Var(s) = 1 and corr(z, s) = attenuation
    s = sim.designed_shocks
    np.testing.assert_allclose(s.var(ddof=0).to_numpy(), 1.0, atol=0.03)
    for k in s.columns:
        c = np.corrcoef(sh.z[k].to_numpy()[ok], s[k].to_numpy()[ok])[0, 1]
        assert c == pytest.approx(tr.attenuation[k], abs=0.02)
    # corr of observed shocks matches S_z
    S_emp = np.corrcoef(sh.s_hat.to_numpy()[ok], rowvar=False)
    np.testing.assert_allclose(S_emp, tr.S_z.to_numpy(), atol=0.02)


def _ma1_market(n_days: int, theta: float, seed: int, n_assets: int = 6) -> MarketData:
    """Correlated returns where every other asset is MA(1) with coefficient ``theta``.

    Mimics spreads whose legs close at different times (lag-1 autocorrelation
    down to -0.5 in the real data, e.g. Quality v World EQ).
    """
    base = _gaussian_market(n_days + 1, 0.5, seed, n_assets)
    e = base.returns.to_numpy()
    X = e[1:].copy()
    X[:, ::2] = e[1:, ::2] + theta * e[:-1, ::2]
    returns = pd.DataFrame(X, index=base.returns.index[1:], columns=base.returns.columns)
    return MarketData(returns=returns, assets=base.assets)


@pytest.mark.parametrize("w", [1, 5])
@pytest.mark.parametrize("lead", [0, 1])
def test_monte_carlo_truth_autocorrelated_returns(w, lead):
    """With serially correlated returns the truth still matches OLS on long data (G.5.3).

    The closed form first written in DESIGN.md G.5.3 assumes serially
    uncorrelated returns and misses here by more than the tolerance; the
    truth computed from the filtered signal does not.
    """
    market = _ma1_market(80_000, -0.8, seed=321)
    assert market.returns.iloc[:, 0].autocorr(1) < -0.4
    cfg = _mc_config(market, w, lead)
    sim = simulate_lab(cfg, market)
    tr = sim.truth

    sh = observed_shocks(sim.attention, w, market.calendar[0], market.calendar[-1])
    r = market.returns
    y = ((r - r.mean()) / r.std(ddof=0)).shift(-lead)
    ok = (sh.s_hat.notna().all(axis=1) & y.notna().all(axis=1)).to_numpy()
    X = np.column_stack([np.ones(ok.sum()), sh.s_hat.to_numpy()[ok]])
    Y = y.to_numpy()[ok]
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    B_ols = coef[1:]
    r2_ols = 1.0 - (Y - X @ coef).var(axis=0) / Y.var(axis=0)
    np.testing.assert_array_less(np.abs(B_ols - tr.B_true.to_numpy()), 0.02)
    np.testing.assert_array_less(np.abs(r2_ols - tr.r2_true.to_numpy()), 0.02)

    # the i.i.d. closed form misses on the autocorrelated assets
    acfg = sim.meta["attention_cfg"]
    a = attenuation(acfg["kappa"], acfg["slow_ar1"], acfg["slow_sd_ratio"], w)
    Wn = tr.W.to_numpy()
    sig_s = Wn @ sim.meta["V"].to_numpy() @ Wn.T + np.diag(tr.sigma_u.to_numpy() ** 2)
    S_cf = (1.0 + 1.0 / w) * a * a * sig_s
    np.fill_diagonal(S_cf, 1.0)
    B_cf = np.linalg.solve(S_cf, a * (Wn @ sim.meta["M"].to_numpy()))
    assert np.abs(B_cf - B_ols).max() > 0.03


def test_monte_carlo_hand_built_links(mc_market):
    """simulate_from_links with a hand-built LinkMap gives the same simulation as the override route."""
    cfg = _mc_config(mc_market, 5, 0)
    sim_a = simulate_lab(cfg, mc_market)
    tt = build_topic_table(cfg.topics)
    rows = [(t, a, tier, s, "hand", "override") for t, a, tier, s in MC_OVERRIDES]
    lm = LinkMap(pd.DataFrame(rows, columns=list(LINK_COLUMNS)))
    sim_b = simulate_from_links(cfg, mc_market, tt, lm)
    pd.testing.assert_frame_equal(sim_a.attention, sim_b.attention)
    pd.testing.assert_frame_equal(sim_a.truth.B_true, sim_b.truth.B_true)


def test_truth_for_window_consistency(mc_market):
    cfg = _mc_config(mc_market, 5, 0)
    sim = simulate_lab(cfg, mc_market)
    t5 = truth_for_window(sim, 5)
    pd.testing.assert_frame_equal(t5.B_true, sim.truth.B_true)
    pd.testing.assert_series_equal(t5.r2_true, sim.truth.r2_true)
    t1 = truth_for_window(sim, 1)
    assert t1.shock_window == 1
    assert (t1.attenuation < t5.attenuation).all()
    assert (t1.r2_true < t5.r2_true).all()  # more attenuation, less explained
    # identities: S_z unit diagonal, r2 = diag(C' S_z^-1 C), sigma_u^2 = 1 - W V W'
    np.testing.assert_allclose(np.diag(t1.S_z.to_numpy()), 1.0)
    W, V = sim.truth.W.to_numpy(), sim.meta["V"].to_numpy()
    np.testing.assert_allclose(sim.truth.sigma_u.to_numpy() ** 2, 1.0 - np.einsum("kn,nm,km->k", W, V, W))


def test_feasibility_scaling():
    market = _gaussian_market(4000, 0.95, seed=5)
    ids = market.assets.index.tolist()
    ov = tuple(("G001", a, "strong", 1) for a in ids) + (("G002", ids[0], "moderate", 1),)
    cfg = LabConfig(
        topics=TopicSetConfig(manual="none", n_generic=2, generic_signal_share=0.0),
        exposure=ExposureConfig(n_betas=1, beta_1=0.9, link_overrides=ov),
        window=_full_window(market, 5),
    )
    sim = simulate_lab(cfg, market)
    tr, cap = sim.truth, cfg.attention.feasibility_cap
    V = sim.meta["V"].to_numpy()
    Wu, W = tr.W_unscaled.to_numpy(), tr.W.to_numpy()
    q_unscaled = np.einsum("kn,nm,km->k", Wu, V, Wu)
    q = np.einsum("kn,nm,km->k", W, V, W)
    assert q_unscaled[0] > 20 * cap
    assert q[0] == pytest.approx(cap) and np.all(q <= cap + 1e-12)
    assert tr.feasibility_scale["G001"] == pytest.approx(np.sqrt(cap / q_unscaled[0]))
    assert tr.feasibility_scale["G002"] == 1.0
    np.testing.assert_allclose(W[0], Wu[0] * tr.feasibility_scale["G001"])
    assert sim.meta["feasibility_scaled_topics"] == ["G001"]
    assert tr.sigma_u["G001"] ** 2 == pytest.approx(1.0 - cap)
    assert sim.meta["feasibility_q"]["G001"] == pytest.approx(q_unscaled[0])


def _rt_c(sim: SimData) -> np.ndarray:
    r = sim.market.returns
    rt = (r - sim.meta["mu"]) / sim.meta["sigma"]
    clip = sim.meta["attention_cfg"]["clip_sd"]
    return rt.clip(-clip, clip).fillna(0.0).to_numpy()


def _recover_noise(sim: SimData) -> np.ndarray:
    x = _rt_c(sim)
    lead = sim.lead_days
    x_lead = np.zeros_like(x)
    x_lead[: len(x) - lead] = x[lead:]
    return (sim.designed_shocks.to_numpy() - x_lead @ sim.truth.W.to_numpy().T) / sim.truth.sigma_u.to_numpy()


def test_determinism_and_stream_separation():
    market = _gaussian_market(3000, 0.5, seed=9)
    cfg = _mc_config(market, 5, 1)
    a = simulate_lab(cfg, market)
    b = simulate_lab(cfg, market)
    pd.testing.assert_frame_equal(a.attention, b.attention)
    pd.testing.assert_frame_equal(a.designed_shocks, b.designed_shocks)
    # only the exposure values change: noise u, base level m and slow component g stay
    cfg2 = _mc_config(market, 5, 1, beta_1=0.7, beta_2=0.1)
    c = simulate_lab(cfg2, market)
    assert not np.allclose(a.truth.W.to_numpy(), c.truth.W.to_numpy())
    u_a, u_c = _recover_noise(a), _recover_noise(c)
    np.testing.assert_allclose(u_a, u_c, atol=1e-9)
    np.testing.assert_allclose(u_a, topic_noise(a.topics.ids, len(market.calendar), 5.0, 7), atol=1e-9)
    pd.testing.assert_series_equal(a.meta["base_level"], c.meta["base_level"])
    kappa = cfg.attention.kappa
    unclipped = (a.attention.to_numpy() > 1e-6) & (c.attention.to_numpy() > 1e-6)
    slow_a = a.attention.to_numpy() - kappa * a.designed_shocks.to_numpy()
    slow_c = c.attention.to_numpy() - kappa * c.designed_shocks.to_numpy()
    np.testing.assert_allclose(slow_a[unclipped], slow_c[unclipped], atol=1e-12)
    # per-topic noise streams: a topic's noise does not depend on the other topics
    u1 = topic_noise(["G001", "G002"], 500, 5.0, 7)
    u2 = topic_noise(["S1", "G002", "A3", "G001"], 500, 5.0, 7)
    np.testing.assert_array_equal(u1[:, 0], u2[:, 3])
    np.testing.assert_array_equal(u1[:, 1], u2[:, 1])
    assert not np.allclose(topic_noise(["G001"], 500, 5.0, 8), u1[:, :1])
    g = topic_noise(["G001"], 200_000, 0.0, 1)
    assert g.std() == pytest.approx(1.0, abs=0.01)
    t4 = topic_noise(["G001"], 200_000, 4.5, 1)
    assert t4.std() == pytest.approx(1.0, abs=0.03)


def test_attention_levels_scale():
    market = _gaussian_market(5000, 0.5, seed=4)
    cfg = _mc_config(market, 5, 0)
    sim = simulate_lab(cfg, market)
    a = sim.attention
    assert a.shape == (5000, 4) and not a.isna().any().any()
    assert (a.to_numpy() >= 1e-6).all()
    base = sim.meta["base_level"]
    assert ((base >= 0.15) & (base <= 0.35)).all()
    # levels in the report's range, daily changes of a few hundredths
    assert 0.1 < float(a.mean().mean()) < 0.4
    assert 0.01 < float(a.diff().std().mean()) < 0.05
    assert sim.meta["clipped_share"] < 1e-3
    assert sim.meta["n_links"] == len(MC_OVERRIDES)
    assert sim.meta["seeds"]["noise"] == [7, 202] and sim.meta["seeds"]["attention"] == [7, 303]


def test_observed_shocks_training_scale():
    market = _gaussian_market(2000, 0.3, seed=2)
    cfg = _mc_config(market, 5, 0)
    sim = simulate_lab(cfg, market)
    cal = market.calendar
    ts, te = cal[300], cal[1200]
    sh = observed_shocks(sim.attention, 5, str(ts.date()), str(te.date()))
    assert sh.window == 5 and sh.train_start == ts and sh.train_end == te
    assert sh.z.iloc[:5].isna().all().all() and sh.z.iloc[5:].notna().all().all()
    expected = sh.z.loc[ts:te].std(ddof=0)
    np.testing.assert_allclose(sh.scale.to_numpy(), expected.to_numpy())
    np.testing.assert_allclose(sh.s_hat.loc[ts:te].std(ddof=0).to_numpy(), 1.0)
    manual_z = sim.attention - sim.attention.shift(1).rolling(5).mean()
    np.testing.assert_allclose(sh.z.to_numpy()[5:], manual_z.to_numpy()[5:], atol=1e-14)
    # a constant topic gets scale 1
    att = sim.attention.copy()
    att["G004"] = 0.2
    sh2 = observed_shocks(att, 1, cal[0], cal[-1])
    assert sh2.scale["G004"] == 1.0


def test_moments_with_missing_and_clipped_days():
    market = _gaussian_market(3000, 0.4, seed=8)
    r = market.returns.copy()
    r.iloc[100:400, 1] = np.nan
    r.iloc[50, 0] = 0.5  # a > 8 sd day, clipped in the construction
    m2 = MarketData(returns=r, assets=market.assets)
    sim = simulate_lab(_mc_config(m2, 5, 0), m2)
    # R is the pandas pairwise correlation
    np.testing.assert_allclose(sim.truth.R.to_numpy(), r.corr().to_numpy(), atol=1e-10)
    # V is the covariance of the clipped, filled standardised returns; M differs from V only via clip/missing
    x = _rt_c(sim)
    np.testing.assert_allclose(sim.meta["V"].to_numpy(), np.cov(x, rowvar=False, ddof=0), atol=1e-12)
    rt = ((r - sim.meta["mu"]) / sim.meta["sigma"]).to_numpy()
    obs = np.isfinite(rt)
    for i, j in [(0, 1), (1, 2), (2, 0), (0, 0)]:
        o = obs[:, j]
        xi, yj = x[o, i], rt[o, j]
        want = np.mean(xi * yj) - xi.mean() * yj.mean()
        assert sim.meta["M"].iloc[i, j] == pytest.approx(want, abs=1e-12)
    assert sim.meta["M"].iloc[0, 0] > sim.meta["V"].iloc[0, 0]  # unclipped covariance is larger
    np.testing.assert_allclose(sim.truth.asset_vol.to_numpy(), r.std(ddof=0).to_numpy())
    assert not sim.attention.isna().any().any()


def test_pairwise_corr_matches_pandas():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((300, 5))
    z[rng.random((300, 5)) < 0.2] = np.nan
    z[:, 4] = np.nan
    z[:3, 4] = [1.0, 2.0, 0.5]
    obs = np.isfinite(z)
    got = _pairwise_corr(z, obs)
    want = pd.DataFrame(z).corr().to_numpy()
    np.testing.assert_allclose(got[:4, :4], want[:4, :4], atol=1e-12)


def test_real_data_smoke(listed_assets):
    path = reference.market_dir() / "asset_returns.parquet"
    if not path.is_file():
        pytest.skip("data/market not available")
    pytest.importorskip("pyarrow")
    r = pd.read_parquet(path)
    r.index = pd.DatetimeIndex(r.index)
    r = r[listed_assets.index.tolist()]
    r.columns.name = "asset_id"
    market = MarketData(returns=r, assets=listed_assets)
    cfg = LabConfig()
    sim = simulate_lab(cfg, market)
    n_days = len(r)
    assert sim.attention.shape == (n_days, 20) and sim.designed_shocks.shape == (n_days, 20)
    assert sim.attention.columns.tolist() == sim.topics.ids
    assert not sim.attention.isna().any().any()
    tr = sim.truth
    assert tr.B_true.shape == (20, 55) and tr.S_z.shape == (20, 20) and tr.R.shape == (55, 55)
    assert tr.r2_true.between(0.0, 1.0).all() and tr.r2_true.max() > 0.05
    assert np.isfinite(tr.B_true.to_numpy()).all()
    assert sim.meta["n_links"] == len(sim.links.table) > 50
    assert sim.meta["clipped_share"] < 1e-3
    sh = observed_shocks(sim.attention, cfg.window.shock_window, cfg.window.train_start, cfg.window.train_end)
    assert np.isfinite(sh.s_hat.iloc[cfg.window.shock_window:].to_numpy()).all()


# ---------------------------------------------------------------------------
# Review fixes 2026-09-29: missing returns (D78), separate noise seed (D71)
# ---------------------------------------------------------------------------
def test_monte_carlo_truth_with_a_missing_block(mc_market, monkeypatch):
    """An asset missing 60% of its days gets the truth of its observed days (D78); OLS on long data agrees.

    The all-days solve used before the fix misses this asset by more than the tolerance.
    """
    r = mc_market.returns.copy()
    r.iloc[:48_000, 0] = np.nan
    market = MarketData(returns=r, assets=mc_market.assets)
    cfg = _mc_config(market, 5, 0)
    sim = simulate_lab(cfg, market)
    tr = sim.truth
    sh = observed_shocks(sim.attention, 5, market.calendar[0], market.calendar[-1])
    y = (r - r.mean()) / r.std(ddof=0)
    S = sh.s_hat.to_numpy()
    for j in range(r.shape[1]):
        ok = np.isfinite(S).all(axis=1) & np.isfinite(y.iloc[:, j].to_numpy())
        X = np.column_stack([np.ones(ok.sum()), S[ok]])
        Y = y.iloc[:, j].to_numpy()[ok]
        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        r2 = 1.0 - (Y - X @ coef).var() / Y.var()
        np.testing.assert_array_less(np.abs(coef[1:] - tr.B_true.iloc[:, j].to_numpy()), 0.015)
        assert abs(r2 - tr.r2_true.iloc[j]) < 0.01
    # the all-days solve (one observation group for every asset) is off for the gappy asset only
    import narrative_ipca.exposure_lab.dgp as dgp

    monkeypatch.setattr(dgp, "observation_groups",
                        lambda obs: [(np.ones(obs.shape[0], dtype=bool), list(range(obs.shape[1])))])
    old = truth_for_window(sim, 5)
    assert np.abs(old.B_true.iloc[:, 0] - tr.B_true.iloc[:, 0]).max() > 0.03
    np.testing.assert_allclose(old.B_true.iloc[:, 1:].to_numpy(), tr.B_true.iloc[:, 1:].to_numpy(), atol=1e-12)


def _generic_link_cfg(seed: int, noise_seed: int) -> LabConfig:
    return LabConfig(
        topics=TopicSetConfig(manual="none", n_generic=6, generic_signal_share=1.0),
        exposure=ExposureConfig(seed=seed, noise_seed=noise_seed),
    )


def test_link_seed_and_noise_seed_are_separate():
    """D71: the link seed redraws only the random links, the noise seed only the noise and attention parts."""
    market = _gaussian_market(1500, 0.3, seed=3)
    base = simulate_lab(_generic_link_cfg(0, 0), market)
    new_links = simulate_lab(_generic_link_cfg(1, 0), market)
    new_noise = simulate_lab(_generic_link_cfg(0, 1), market)
    # links: only the link seed matters
    assert not base.links.table.equals(new_links.links.table)
    pd.testing.assert_frame_equal(base.links.table, new_noise.links.table)
    # noise and attention components: only the noise seed matters
    np.testing.assert_allclose(_recover_noise(base), _recover_noise(new_links), atol=1e-9)
    assert not np.allclose(_recover_noise(base), _recover_noise(new_noise))
    pd.testing.assert_series_equal(base.meta["base_level"], new_links.meta["base_level"])
    assert base.meta["seeds"]["links"] == [0, 101] and new_links.meta["seeds"]["links"] == [1, 101]
    assert new_noise.meta["seeds"]["noise"] == [1, 202] and new_noise.meta["seeds"]["attention"] == [1, 303]
    # the truth depends on the links, not on the noise draw
    pd.testing.assert_frame_equal(base.truth.B_true, new_noise.truth.B_true)
