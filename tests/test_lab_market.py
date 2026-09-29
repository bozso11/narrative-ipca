"""Tests for narrative_ipca.exposure_lab.reference and .market (DESIGN.md G.2, G.2.1, G.2.2; D54-D56)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from narrative_ipca.exposure_lab import market, reference
from narrative_ipca.exposure_lab.config import TIERS, UniverseConfig
from narrative_ipca.exposure_lab.market import (
    artificial_parameters,
    artificial_returns,
    build_market,
    generic_asset_table,
    load_real_returns,
    weekday_calendar,
)

REPO_REFERENCE = Path(reference.__file__).resolve().parents[2] / "data" / "reference"
REPO_MARKET = Path(reference.__file__).resolve().parents[2] / "data" / "market"
HAS_REAL_DATA = (REPO_MARKET / "asset_returns.parquet").is_file()
needs_real = pytest.mark.skipif(not HAS_REAL_DATA, reason="data/market/asset_returns.parquet not present")


@pytest.fixture(autouse=True)
def _repo_data_dir(monkeypatch):
    """Run every test against the repository data folder unless a test overrides it."""
    monkeypatch.delenv(reference.DATA_DIR_ENV, raising=False)


def _tmp_data_dir(tmp_path: Path) -> Path:
    """A data folder with copies of the repository reference CSVs and an empty market folder."""
    root = tmp_path / "data"
    (root / "reference").mkdir(parents=True)
    (root / "market").mkdir()
    for f in REPO_REFERENCE.glob("*.csv"):
        shutil.copy(f, root / "reference" / f.name)
    return root


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
def test_weekday_calendar_full_sample():
    cal = weekday_calendar("2015-01-02", "2025-12-31")
    assert len(cal) == 2869
    assert cal.name == "date"
    assert (cal.dayofweek < 5).all()
    assert cal[0] == pd.Timestamp("2015-01-02") and cal[-1] == pd.Timestamp("2025-12-31")
    assert cal.is_monotonic_increasing and not cal.has_duplicates


def test_weekday_calendar_edges():
    cal = weekday_calendar("2024-01-06", "2024-01-14")  # Sat .. Sun
    assert list(cal.strftime("%a")) == ["Mon", "Tue", "Wed", "Thu", "Fri"]
    with pytest.raises(ValueError):
        weekday_calendar("2024-01-10", "2024-01-09")
    with pytest.raises(ValueError):
        weekday_calendar("2024-01-06", "2024-01-07")  # weekend only


# ---------------------------------------------------------------------------
# Reference loaders
# ---------------------------------------------------------------------------
def test_data_dir_default_and_env_override(monkeypatch, tmp_path):
    assert reference.data_dir() == REPO_REFERENCE.parent
    monkeypatch.setenv(reference.DATA_DIR_ENV, str(tmp_path))
    assert reference.data_dir() == tmp_path.resolve()
    assert reference.reference_dir() == tmp_path.resolve() / "reference"
    assert reference.market_dir() == tmp_path.resolve() / "market"


def test_reference_loaders_shapes_and_order():
    assets = reference.load_assets()
    assert len(assets) == 55 and assets.index.name == "asset_id"
    assert list(assets.columns) == ["order", "name", "long_leg", "short_leg", "asset_class", "sub_class"]
    assert assets["order"].is_monotonic_increasing
    assert assets.index[0] == "REAL_ESTATE_v_WEQ" and assets.index[-1] == "USD"
    assert set(assets["asset_class"]) == set(reference.ASSET_CLASSES)

    legs = reference.load_legs()
    assert legs.index.name == "leg_id" and "cash" in legs.index
    assert set(assets["long_leg"]) | set(assets["short_leg"]) <= set(legs.index)

    topics = reference.load_topics()
    assert len(topics) == 20 and topics.index.name == "topic_id"
    assert list(topics.columns) == ["order", "ontology", "group", "name", "scope"]
    assert list(topics.index[:2]) == ["S1", "S2"]
    assert set(topics["group"]) == {"Sector", "Macro", "Micro"}

    links = reference.load_default_links()
    assert list(links.columns) == ["topic_id", "asset_id", "tier", "sign", "mechanism"]
    assert set(links["tier"]) <= set(TIERS)
    assert set(links["sign"]) <= {1, -1}
    assert links["topic_id"].isin(topics.index).all() and links["asset_id"].isin(assets.index).all()
    assert not links.duplicated(["topic_id", "asset_id"]).any()


def test_reference_loaders_return_copies():
    a = reference.load_assets()
    a.loc["USD", "name"] = "changed"
    a["extra"] = 1
    b = reference.load_assets()
    assert b.loc["USD", "name"] == "USD" and "extra" not in b.columns


def test_default_links_validation_lists_all_problems(monkeypatch, tmp_path):
    root = _tmp_data_dir(tmp_path)
    bad = pd.DataFrame(
        {
            "topic_id": ["S1", "S1", "ZZ9", "S2", "S3"],
            "asset_id": ["ENERGY_v_WEQ", "ENERGY_v_WEQ", "USD", "NOPE", "USD"],
            "tier": ["strong", "weak", "strong", "weak", "huge"],
            "sign": [1, -1, 1, 1, 2],
            "mechanism": ["a", "b", "c", "d", "e"],
        }
    )
    bad.to_csv(root / "reference" / "link_map.csv", index=False)
    monkeypatch.setenv(reference.DATA_DIR_ENV, str(root))
    with pytest.raises(ValueError) as exc:
        reference.load_default_links()
    msg = str(exc.value)
    for fragment in ("ZZ9", "NOPE", "huge", "sign must be +1 or -1", "duplicate (topic_id, asset_id)"):
        assert fragment in msg


# ---------------------------------------------------------------------------
# Generic assets
# ---------------------------------------------------------------------------
def test_generic_asset_table_determinism_and_classes():
    a = generic_asset_table(500, seed=7)
    b = generic_asset_table(500, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(generic_asset_table(500, seed=8))
    assert list(a.index[:2]) == ["G_ASSET_001", "G_ASSET_002"] and a.index[-1] == "G_ASSET_500"
    assert list(a["order"]) == list(range(1, 501))
    assert set(a["asset_class"]) == {"Equity", "FX", "Fixed income"}
    shares = a["asset_class"].value_counts(normalize=True)
    assert 0.42 < shares["Equity"] < 0.58 and 0.22 < shares["FX"] < 0.38 and 0.13 < shares["Fixed income"] < 0.27
    assert a.loc["G_ASSET_001", "name"] == f"Generic asset 001 ({a.loc['G_ASSET_001', 'asset_class']})"
    assert (a["sub_class"] == "generic").all() and (a["source"] == "generic").all()
    assert (a["long_leg"] == "").all() and (a["short_leg"] == "").all()
    # prefix property: a smaller universe is the head of a larger one
    pd.testing.assert_frame_equal(generic_asset_table(20, seed=7), a.iloc[:20])


# ---------------------------------------------------------------------------
# Artificial returns (G.2.2)
# ---------------------------------------------------------------------------
def test_artificial_returns_determinism_and_subset_invariance():
    assets = generic_asset_table(30, seed=1)
    cal = weekday_calendar("2020-01-01", "2021-12-31")
    r1 = artificial_returns(assets, cal, seed=3)
    r2 = artificial_returns(assets, cal, seed=3)
    pd.testing.assert_frame_equal(r1, r2)
    assert r1.shape == (len(cal), 30) and list(r1.columns) == list(assets.index)
    assert r1.index.equals(cal) and np.isfinite(r1.to_numpy()).all()
    assert not np.allclose(r1.to_numpy(), artificial_returns(assets, cal, seed=4).to_numpy())
    # an asset's series does not depend on the other assets in the universe
    sub = assets.iloc[[5, 2]]
    pd.testing.assert_frame_equal(artificial_returns(sub, cal, seed=3), r1[list(sub.index)])


def test_artificial_vols_match_targets_on_long_calendar():
    assets = pd.concat([reference.load_assets(), generic_asset_table(30, seed=2)])
    cal = weekday_calendar("1985-01-01", "2024-12-31")  # about 10,400 days
    r = artificial_returns(assets, cal, seed=11)
    params = artificial_parameters(assets, seed=11)
    vol = r.std() * np.sqrt(252)
    ratio = vol / params["target_vol"]
    assert ratio.between(0.7, 1.3).all(), ratio.describe()
    assert abs(ratio.median() - 1.0) < 0.1
    # class targets times the U(0.6, 1.5) multiplier
    for cls, tgt in market.CLASS_VOL_TARGET.items():
        sel = params["asset_class"] == cls
        assert params.loc[sel, "target_vol"].between(0.6 * tgt, 1.5 * tgt).all()
    # population identity: factor variance + idiosyncratic variance = target variance
    total = np.sqrt(params["factor_vol"] ** 2 + params["idio_vol"] ** 2)
    np.testing.assert_allclose(total, params["target_vol"], rtol=1e-12)
    assert (params["idio_vol"] ** 2 >= 0.1 * params["target_vol"] ** 2 - 1e-15).all()
    # the factors have about 10% annualised volatility
    fvol = market.garch_factors(cal, seed=11).std() * np.sqrt(252)
    assert fvol.between(0.085, 0.115).all(), fvol


def test_artificial_equity_cross_correlation_positive():
    assets = generic_asset_table(200, seed=5)
    eq = assets[assets["asset_class"] == "Equity"].iloc[:40]
    cal = weekday_calendar("2010-01-01", "2019-12-31")
    corr = artificial_returns(eq, cal, seed=5).corr().to_numpy()
    off = corr[~np.eye(len(eq), dtype=bool)]
    assert off.mean() > 0.2
    assert (off > 0).mean() > 0.95


def test_artificial_rejects_unknown_class():
    bad = pd.DataFrame({"asset_class": ["Equity", "Commodity"]}, index=["a", "b"])
    with pytest.raises(ValueError, match="Commodity"):
        artificial_returns(bad, weekday_calendar("2020-01-01", "2020-02-01"), seed=0)


# ---------------------------------------------------------------------------
# build_market
# ---------------------------------------------------------------------------
@needs_real
def test_build_market_listed_real():
    md = build_market(UniverseConfig())
    ref = reference.load_assets()
    assert list(md.assets.index) == list(ref.index)
    assert list(md.returns.columns) == list(ref.index) and md.returns.shape[1] == 55
    assert list(md.assets.columns) == market.ASSET_TABLE_COLUMNS
    assert md.returns.index.equals(weekday_calendar("2015-01-02", "2025-12-31"))
    cov = pd.Series(md.meta["coverage"])
    assert (cov > 0.95).all(), cov[cov <= 0.95]
    assert md.meta["price_source"] == "real" and md.meta["n_days"] == 2869
    assert set(md.assets.loc[md.meta["failed_assets"], "source"]) <= {"artificial"}
    real = [a for a in md.assets.index if a not in md.meta["failed_assets"]]
    assert (md.assets.loc[real, "source"] == "real").all()
    raw = pd.read_parquet(REPO_MARKET / "asset_returns.parquet")
    np.testing.assert_allclose(md.returns[real].to_numpy(), raw.reindex(md.returns.index)[real].to_numpy())
    # display columns from legs.csv; empty for the cash leg
    assert md.assets.loc["CHF_v_USD", "short_index"] == "" and md.assets.loc["CHF_v_USD", "short_proxy"] == ""
    assert md.assets.loc["CHF_v_USD", "long_proxy"] != ""
    assert md.assets.loc["REAL_ESTATE_v_WEQ", "short_proxy"] == reference.load_legs().at["eq_world", "proxy_ticker"]


def test_build_market_subset_keeps_reference_order():
    cfg = UniverseConfig(listed_assets=("USD", "CHF_v_USD", "REAL_ESTATE_v_WEQ"), price_source="artificial",
                         start="2020-01-01", end="2020-12-31")
    md = build_market(cfg)
    assert list(md.assets.index) == ["REAL_ESTATE_v_WEQ", "CHF_v_USD", "USD"]
    assert list(md.returns.columns) == list(md.assets.index)
    assert list(md.assets["order"]) == [1, 3, 55]
    # the subset's artificial series equal the full universe's
    full = build_market(UniverseConfig(price_source="artificial", start="2020-01-01", end="2020-12-31"))
    pd.testing.assert_frame_equal(md.returns, full.returns[list(md.assets.index)])
    with pytest.raises(ValueError, match="NOT_AN_ASSET"):
        build_market(UniverseConfig(listed_assets=("USD", "NOT_AN_ASSET"), price_source="artificial"))


def test_build_market_artificial_mode():
    md = build_market(UniverseConfig(price_source="artificial", seed=3))
    assert md.returns.shape == (2869, 55)
    assert (md.assets["source"] == "artificial").all()
    assert np.isfinite(md.returns.to_numpy()).all()
    assert md.meta["failed_assets"] == [] and md.meta["price_source"] == "artificial"
    pd.testing.assert_frame_equal(md.returns, build_market(UniverseConfig(price_source="artificial", seed=3)).returns)


def test_build_market_generic():
    md = build_market(UniverseConfig(asset_source="generic", n_generic_assets=12, seed=4,
                                     start="2021-01-01", end="2021-06-30"))
    assert list(md.assets.index) == [f"G_ASSET_{i:03d}" for i in range(1, 13)]
    assert list(md.assets.columns) == market.ASSET_TABLE_COLUMNS
    assert (md.assets["source"] == "generic").all() and (md.assets["long_index"] == "").all()
    assert np.isfinite(md.returns.to_numpy()).all()
    assert md.meta["asset_source"] == "generic" and md.meta["n_assets"] == 12


def _write_market(root: Path, cal: pd.DatetimeIndex, *, failed_legs=(), nan_assets=(), drop_assets=(),
                  manifest: bool = True) -> pd.DataFrame:
    assets = reference.load_assets()
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(rng.normal(0, 0.005, (len(cal), len(assets))), index=cal, columns=list(assets.index))
    for a in nan_assets:
        raw[a] = np.nan
    raw = raw.drop(columns=list(drop_assets))
    raw.index.name = "date"
    raw.to_parquet(root / "market" / "asset_returns.parquet")
    if manifest:
        legs = reference.load_legs()
        m = {
            "run_timestamp_utc": "2026-09-29T00:00:00+00:00",
            "legs": {leg: {"status": "failed" if leg in failed_legs else "ok"} for leg in legs.index},
            "assets": {a: {"long_leg": assets.at[a, "long_leg"], "short_leg": assets.at[a, "short_leg"],
                           "status": "ok"} for a in assets.index},
        }
        (root / "market" / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return raw


def test_build_market_failed_leg_fallback(monkeypatch, tmp_path):
    root = _tmp_data_dir(tmp_path)
    monkeypatch.setenv(reference.DATA_DIR_ENV, str(root))
    cal = weekday_calendar("2020-01-01", "2020-06-30")
    raw = _write_market(root, cal, failed_legs=("fx_chf", "dur_global"), nan_assets=("CHF_v_USD",),
                        drop_assets=("USD",))
    md = build_market(UniverseConfig(start="2020-01-01", end="2020-06-30", seed=9))

    expected = ["GLOBAL_DURATION", "CHF_v_USD", "UK_7_10Y_v_GDUR", "IT_7_10Y_v_GDUR", "US_7_10Y_v_GDUR",
                "CH_7_10Y_v_GDUR", "DE_7_10Y_v_GDUR", "JP_7_10Y_v_GDUR", "USD"]
    assert md.meta["failed_assets"] == expected  # reference order
    assert md.meta["failed_reasons"]["USD"] == "missing column"
    assert "dur_global" in md.meta["failed_reasons"]["UK_7_10Y_v_GDUR"]
    assert set(md.meta["failed_legs"]) == {"fx_chf", "dur_global"}
    assert (md.assets.loc[expected, "source"] == "artificial").all()
    ok = [a for a in md.assets.index if a not in expected]
    assert (md.assets.loc[ok, "source"] == "real").all()
    assert md.meta["data_dir"] == str(root.resolve())

    # failed assets carry their artificial series (the same as in the artificial mode)
    fill = artificial_returns(md.assets.loc[expected], cal, seed=9)
    pd.testing.assert_frame_equal(md.returns[expected], fill, check_freq=False, check_names=False)
    art = build_market(UniverseConfig(price_source="artificial", start="2020-01-01", end="2020-06-30", seed=9))
    pd.testing.assert_frame_equal(md.returns[expected], art.returns[expected])
    # the others keep the stored returns
    np.testing.assert_allclose(md.returns[ok].to_numpy(), raw[ok].to_numpy())
    assert np.isfinite(md.returns.to_numpy()).all()
    assert pd.Series(md.meta["coverage"]).eq(1.0).all()
    assert md.meta["coverage_real"]["CHF_v_USD"] == 0.0


def test_load_real_returns_without_manifest_and_beyond_data(monkeypatch, tmp_path):
    root = _tmp_data_dir(tmp_path)
    monkeypatch.setenv(reference.DATA_DIR_ENV, str(root))
    cal = weekday_calendar("2020-01-01", "2020-03-31")
    _write_market(root, cal, nan_assets=("SEK_v_USD",), manifest=False)
    r, meta = load_real_returns(["USD", "SEK_v_USD", "CHF_v_USD"], "2019-12-02", "2020-03-31")
    assert list(r.columns) == ["USD", "SEK_v_USD", "CHF_v_USD"]
    assert r.index.equals(weekday_calendar("2019-12-02", "2020-03-31"))
    assert r.loc[:"2019-12-31"].isna().all().all()
    assert meta["failed_assets"] == ["SEK_v_USD"] and meta["manifest"] is None
    share = len(cal) / len(r)
    assert meta["coverage"]["USD"] == pytest.approx(share)


def test_missing_parquet_error_message(monkeypatch, tmp_path):
    root = _tmp_data_dir(tmp_path)
    monkeypatch.setenv(reference.DATA_DIR_ENV, str(root))
    with pytest.raises(FileNotFoundError) as exc:
        build_market(UniverseConfig())
    msg = str(exc.value)
    assert "scripts/fetch_market_data.py" in msg and "price_source='artificial'" in msg
    # the artificial mode needs no market data
    md = build_market(UniverseConfig(price_source="artificial", start="2020-01-01", end="2020-02-28"))
    assert md.returns.shape[1] == 55


def test_offline_raw_round_trip_is_exact(tmp_path):
    """scripts/fetch_market_data.py: a raw series written by the online run reads back bit for bit (offline rebuild)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "fetch_market_data_script", Path(reference.__file__).resolve().parents[2] / "scripts" / "fetch_market_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # the script's dataclasses look their module up
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    rng = np.random.default_rng([20260929, 9110])
    idx = pd.bdate_range("2020-01-01", periods=400)
    df = pd.DataFrame({"close": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))}, index=idx)
    df["adj_close"] = df["close"] * (1.0 + rng.normal(0, 1e-3, 400)) / 3.0
    store = mod.RawStore(tmp_path / "raw", "2020-01-01", "2021-12-31", True, 1, 1)
    key = ("yahoo", "TEST.X")
    store._write_raw(key, df)
    store._read_raw(key)
    back = store.data[key]
    assert np.array_equal(back.to_numpy(), df.to_numpy())  # exact: float_precision="round_trip"
    assert list(back.index) == list(idx)
