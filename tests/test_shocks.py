"""Tests for narrative_ipca.shocks: z_tau construction, diagnostics and placebo appending."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from narrative_ipca.config import ShockConfig
from narrative_ipca.shocks import append_placebos, attention_shocks, shock_diagnostics
from narrative_ipca.types import ShockPanel


def _attention(n_days: int = 60, L: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(rng.random((n_days, L)), index=cal, columns=[f"t{l}" for l in range(L)])


# ---------------------------------------------------------------------------
# attention_shocks
# ---------------------------------------------------------------------------
def test_shocks_hand_example():
    cal = pd.bdate_range("2020-01-01", periods=7)
    att = pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0]}, index=cal)
    sp = attention_shocks(att, ShockConfig(window=5))
    z = sp.z["a"].to_numpy()
    assert np.isnan(z[:5]).all()
    assert z[5] == pytest.approx(1.0)  # 2 - mean(1,1,1,1,1)
    assert z[6] == pytest.approx(-0.2)  # 1 - mean(1,1,1,1,2)
    assert sp.window == 5 and sp.scale is None
    assert sp.z.index.equals(cal)


@pytest.mark.parametrize("w", [1, 3, 5, 20])
def test_shocks_equal_brute_force_trailing_mean(w):
    att = _attention(n_days=80, L=4, seed=w)
    sp = attention_shocks(att, ShockConfig(window=w))
    theta = att.to_numpy()
    for tau in range(att.shape[0]):
        if tau < w:
            assert np.isnan(sp.z.iloc[tau]).all()
        else:
            expected = theta[tau] - theta[tau - w : tau].mean(axis=0)
            np.testing.assert_allclose(sp.z.iloc[tau].to_numpy(), expected, atol=1e-14)


def test_shocks_window_one_is_daily_difference():
    att = _attention()
    sp = attention_shocks(att, ShockConfig(window=1))
    pd.testing.assert_frame_equal(sp.z, att.diff(), check_freq=False)


def test_shocks_nan_propagates_through_the_trailing_window():
    att = _attention(n_days=40, L=2)
    att.iloc[15, 0] = np.nan
    sp = attention_shocks(att, ShockConfig(window=5))
    z0 = sp.z.iloc[:, 0].to_numpy()
    assert np.isnan(z0[15:21]).all()  # the day itself and the 5 days whose window contains it
    assert np.isfinite(z0[21:]).all() and np.isfinite(z0[5:15]).all()
    assert np.isfinite(sp.z.iloc[5:, 1]).all()  # other topic untouched


def test_shocks_standardize_records_scale():
    att = _attention(n_days=200, L=3)
    raw = attention_shocks(att, ShockConfig(window=5))
    std = attention_shocks(att, ShockConfig(window=5, standardize=True))
    assert std.scale is not None and list(std.scale.index) == list(att.columns)
    np.testing.assert_allclose(std.scale.to_numpy(), raw.z.std(ddof=1).to_numpy())
    np.testing.assert_allclose(std.z.std(ddof=1).to_numpy(), 1.0)
    pd.testing.assert_frame_equal(std.z * std.scale, raw.z, check_freq=False)
    # constant topic: scale stays 1 instead of dividing by zero
    att2 = att.copy()
    att2["t0"] = 0.3
    std2 = attention_shocks(att2, ShockConfig(window=5, standardize=True))
    assert std2.scale["t0"] == 1.0 and (std2.z["t0"].dropna() == 0).all()


def test_shocks_rejects_short_or_unsorted_input():
    att = _attention(n_days=5)
    with pytest.raises(ValueError):
        attention_shocks(att, ShockConfig(window=5))
    with pytest.raises(ValueError):
        attention_shocks(_attention().iloc[::-1], ShockConfig())


# ---------------------------------------------------------------------------
# shock_diagnostics
# ---------------------------------------------------------------------------
def test_diagnostics_values():
    cal = pd.bdate_range("2020-01-01", periods=12)
    alt = np.array([1.0, -1.0] * 6)
    z = pd.DataFrame({"alt": alt, "zero": np.zeros(12), "mixed": [0.0, 1.0, 0.0, 2.0] * 3}, index=cal)
    z.iloc[:2] = np.nan
    d = shock_diagnostics(ShockPanel(z=z, window=2))
    assert list(d.index) == ["alt", "zero", "mixed"]
    assert set(["n_obs", "std", "ac1", "share_zero"]).issubset(d.columns)
    assert (d["n_obs"] == 10).all()
    assert d.loc["alt", "ac1"] == pytest.approx(-1.0)
    assert d.loc["alt", "std"] == pytest.approx(pd.Series(alt[2:]).std(ddof=1))
    assert d.loc["alt", "share_zero"] == 0.0
    assert np.isnan(d.loc["zero", "ac1"]) and d.loc["zero", "share_zero"] == 1.0
    assert d.loc["mixed", "share_zero"] == pytest.approx(5 / 10)


# ---------------------------------------------------------------------------
# append_placebos
# ---------------------------------------------------------------------------
def test_placebos_shape_names_mask_and_nan_alignment():
    att = _attention(n_days=300, L=3)
    sp = attention_shocks(att, ShockConfig())
    sp2, mask = append_placebos(sp, 4, seed=1)
    assert sp2.z.shape == (300, 7)
    assert list(sp2.z.columns) == ["t0", "t1", "t2", "placebo_1", "placebo_2", "placebo_3", "placebo_4"]
    np.testing.assert_array_equal(mask, [False, False, False, True, True, True, True])
    pd.testing.assert_frame_equal(sp2.z.iloc[:, :3], sp.z, check_freq=False)  # real columns untouched
    nan_rows = sp.z.isna().any(axis=1)
    assert sp2.z.loc[nan_rows, mask].isna().all().all()
    assert sp2.z.loc[~nan_rows, mask].notna().all().all()
    assert sp2.window == sp.window and sp2.scale is None


def test_placebos_match_a_real_topic_variance():
    rng = np.random.default_rng(5)
    cal = pd.bdate_range("2020-01-01", periods=4000)
    # three real topics with clearly separated variances
    z = pd.DataFrame(rng.standard_normal((4000, 3)) * np.array([0.1, 1.0, 10.0]), index=cal, columns=["a", "b", "c"])
    z.iloc[:5] = np.nan
    sp = ShockPanel(z=z, window=5)
    sp2, mask = append_placebos(sp, 6, seed=2)
    real_var = z.var(ddof=1).to_numpy()
    for col in sp2.z.columns[mask]:
        v = sp2.z[col].var(ddof=1)
        rel = np.abs(v / real_var - 1.0)
        assert rel.min() < 0.1, f"{col}: variance {v} matches no real topic {real_var}"
    # placebos are i.i.d.: negligible serial correlation and correlation with the real topics
    pl = sp2.z.loc[:, mask].dropna()
    assert np.abs(pl.apply(lambda s: s.autocorr(1))).max() < 0.06
    assert np.abs(sp2.z.dropna().corr().loc[mask, ~mask].to_numpy()).max() < 0.06


def test_placebos_reproducible_and_seed_dependent():
    sp = attention_shocks(_attention(n_days=100), ShockConfig())
    a, _ = append_placebos(sp, 2, seed=7)
    b, _ = append_placebos(sp, 2, seed=7)
    c, _ = append_placebos(sp, 2, seed=8)
    pd.testing.assert_frame_equal(a.z, b.z, check_freq=False)
    assert not np.allclose(a.z["placebo_1"].dropna(), c.z["placebo_1"].dropna())


def test_placebos_numbering_continues_and_scale_extends():
    sp = attention_shocks(_attention(n_days=100), ShockConfig(standardize=True))
    first, mask1 = append_placebos(sp, 2, seed=1)
    second, mask2 = append_placebos(first, 1, seed=2)
    assert list(second.z.columns[-3:]) == ["placebo_1", "placebo_2", "placebo_3"]
    np.testing.assert_array_equal(mask2, [False, False, False, True, True, True])
    assert second.scale is not None and list(second.scale.index) == list(second.z.columns)
    assert (second.scale.iloc[-3:] == 1.0).all()
    # existing placebos are not used as references: matched variances come from the real topics
    real_var = sp.z.var(ddof=1).to_numpy()
    v = second.z["placebo_3"].var(ddof=1)
    assert np.abs(v / real_var - 1.0).min() < 0.5


def test_placebos_zero_and_errors():
    sp = attention_shocks(_attention(n_days=50), ShockConfig())
    same, mask = append_placebos(sp, 0, seed=0)
    pd.testing.assert_frame_equal(same.z, sp.z, check_freq=False)
    assert not mask.any() and len(mask) == 3
    with pytest.raises(ValueError):
        append_placebos(sp, -1, seed=0)
    flat = ShockPanel(z=pd.DataFrame(np.zeros((50, 2)), index=sp.z.index, columns=["a", "b"]), window=5)
    with pytest.raises(ValueError):
        append_placebos(flat, 1, seed=0)
