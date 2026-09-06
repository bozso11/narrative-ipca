"""Tests for narrative_ipca.data: period helpers, document aggregation and calendar alignment."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from narrative_ipca.config import DataConfig
from narrative_ipca.data import (
    aggregate_documents,
    align_inputs,
    normalize_period_alias,
    period_end_index,
    period_returns,
    trailing_volatility,
)
from narrative_ipca.types import AttentionData, ReturnsData


# ---------------------------------------------------------------------------
# period helpers
# ---------------------------------------------------------------------------
def test_period_end_index_monthly():
    cal = pd.bdate_range("2020-01-01", "2020-04-15")
    ids, ends = period_end_index(cal, "M")
    assert ids.dtype == np.int64 and ids.shape == (len(cal),)
    expected_ids = (cal.month - 1).to_numpy()
    np.testing.assert_array_equal(ids, expected_ids)
    assert list(ends) == [pd.Timestamp(d) for d in ["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-15"]]
    # every period end is the last calendar day of its period
    for t, end in enumerate(ends):
        assert end == cal[ids == t][-1]


def test_period_end_index_accepts_offset_alias_silently():
    cal = pd.bdate_range("2020-01-01", periods=80)
    ids_m, ends_m = period_end_index(cal, "M")
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        ids_me, ends_me = period_end_index(cal, "ME")
        ids_a, ends_a = period_end_index(cal, "A")  # deprecated alias, must stay silent
    assert not log, [str(w.message) for w in log]
    np.testing.assert_array_equal(ids_m, ids_me)
    assert ends_m.equals(ends_me)
    assert len(ends_a) == 1
    assert normalize_period_alias("ME") == ["ME", "M"]
    assert normalize_period_alias("2QE") == ["2QE", "2Q"]
    assert normalize_period_alias("W-FRI") == ["W-FRI"]
    with pytest.raises(ValueError):
        period_end_index(cal, "not-a-freq")


def test_period_end_index_weekly_and_gaps():
    cal = pd.bdate_range("2020-01-06", periods=15)  # three full Mon-Fri weeks
    ids, ends = period_end_index(cal, "W")
    np.testing.assert_array_equal(ids, np.repeat([0, 1, 2], 5))
    assert all(e.dayofweek == 4 for e in ends)
    # a missing whole period counts as one step (positional ids)
    cal2 = cal.append(pd.bdate_range("2020-02-10", periods=5))
    ids2, ends2 = period_end_index(cal2, "W")
    assert ids2.max() == 3 and len(ends2) == 4


def test_period_end_index_rejects_unsorted_or_duplicates():
    cal = pd.bdate_range("2020-01-01", periods=10)
    with pytest.raises(ValueError):
        period_end_index(cal[::-1], "M")
    with pytest.raises(ValueError):
        period_end_index(cal.append(cal[:1]).sort_values(), "M")
    ids, ends = period_end_index(pd.DatetimeIndex([]), "M")
    assert ids.shape == (0,) and len(ends) == 0


def test_period_returns_sum_and_compound():
    cal = pd.bdate_range("2020-01-01", "2020-02-29")
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.standard_normal((len(cal), 3)) * 0.01, index=cal, columns=list("abc"))
    r.loc[cal.month == 2, "b"] = np.nan  # b unobserved in February
    r.iloc[3, 0] = np.nan  # one missing day for a in January
    s = period_returns(r, "M", "sum")
    c = period_returns(r, "M", "compound")
    jan, feb = cal.month == 1, cal.month == 2
    assert list(s.index) == [cal[jan][-1], cal[feb][-1]]
    assert s.loc[s.index[0], "a"] == pytest.approx(np.nansum(r.loc[jan, "a"]))
    assert c.loc[c.index[0], "a"] == pytest.approx(np.prod(1 + r.loc[jan, "a"].dropna()) - 1)
    assert np.isnan(s.loc[s.index[1], "b"]) and np.isnan(c.loc[c.index[1], "b"])
    assert s.loc[s.index[1], "c"] == pytest.approx(r.loc[feb, "c"].sum())
    with pytest.raises(ValueError):
        period_returns(r, "M", "median")


# ---------------------------------------------------------------------------
# aggregate_documents
# ---------------------------------------------------------------------------
def test_aggregate_documents_term_weighted_mean():
    theta = pd.DataFrame(
        [[0.8, 0.2], [0.2, 0.8], [0.5, 0.5], [0.1, 0.9]],
        columns=["t1", "t2"],
    )
    dates = pd.Series(pd.to_datetime(["2020-01-02 09:00", "2020-01-02 15:00", "2020-01-03 00:00", "2020-01-03 12:00"]))
    weights = pd.Series([100.0, 300.0, 1.0, 3.0])
    out = aggregate_documents(theta, dates, weights)
    assert list(out.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    # day 1: (100*0.8 + 300*0.2)/400 = 0.35 ; day 2: (1*0.5 + 3*0.1)/4 = 0.2
    np.testing.assert_allclose(out.loc["2020-01-02"].to_numpy(), [0.35, 0.65])
    np.testing.assert_allclose(out.loc["2020-01-03"].to_numpy(), [0.2, 0.8])
    # unit weights = plain mean, rows stay on the simplex
    plain = aggregate_documents(theta, dates, None)
    np.testing.assert_allclose(plain.loc["2020-01-02"].to_numpy(), [0.5, 0.5])
    np.testing.assert_allclose(plain.sum(axis=1).to_numpy(), 1.0)


def test_aggregate_documents_nan_cells_and_zero_weights():
    theta = pd.DataFrame([[np.nan, 0.2], [0.4, 0.8], [0.6, np.nan]], columns=["t1", "t2"])
    dates = ["2020-01-02", "2020-01-02", "2020-01-05"]
    out = aggregate_documents(theta, dates, [1.0, 1.0, 0.0])
    np.testing.assert_allclose(out.loc["2020-01-02"].to_numpy(), [0.4, 0.5])
    assert np.isnan(out.loc["2020-01-05"]).all()  # zero total weight
    with pytest.raises(ValueError):
        aggregate_documents(theta, dates, [1.0, -1.0, 1.0])
    with pytest.raises(ValueError):
        aggregate_documents(theta, dates[:2], None)


# ---------------------------------------------------------------------------
# align_inputs
# ---------------------------------------------------------------------------
def _inputs(n_days: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2020-01-06", periods=n_days)  # trading days
    alldays = pd.date_range(cal[0] - pd.Timedelta(days=2), cal[-1] + pd.Timedelta(days=2), freq="D")
    att = pd.DataFrame(rng.random((len(alldays), 2)), index=alldays, columns=["x", "y"])
    ret = pd.DataFrame(rng.standard_normal((n_days, 3)) * 0.01, index=cal, columns=["A", "B", "C"])
    return att, ret, cal, alldays


def test_align_drop_policy_uses_return_calendar():
    att, ret, cal, _ = _inputs()
    out = align_inputs(AttentionData(att), ReturnsData(ret), DataConfig())
    assert out.calendar.equals(cal)
    assert out.attention.index.equals(cal) and out.returns.index.equals(cal)
    pd.testing.assert_frame_equal(out.attention, att.loc[cal], check_names=False, check_freq=False)
    pd.testing.assert_frame_equal(out.returns, ret, check_names=False, check_freq=False)
    assert out.scale is None
    assert out.topics == ["x", "y"] and out.assets == ["A", "B", "C"]


def test_align_fold_policies():
    att, ret, cal, _ = _inputs()
    mean_out = align_inputs(AttentionData(att), ReturnsData(ret), DataConfig(non_trading_day_policy="fold_mean"))
    sum_out = align_inputs(AttentionData(att), ReturnsData(ret), DataConfig(non_trading_day_policy="fold_sum"))
    monday = cal[5]  # second Monday: folds Sat + Sun + Mon
    block = att.loc[monday - pd.Timedelta(days=2) : monday]
    assert len(block) == 3
    np.testing.assert_allclose(mean_out.attention.loc[monday].to_numpy(), block.mean().to_numpy())
    np.testing.assert_allclose(sum_out.attention.loc[monday].to_numpy(), block.sum().to_numpy())
    tuesday = cal[6]
    np.testing.assert_allclose(mean_out.attention.loc[tuesday].to_numpy(), att.loc[tuesday].to_numpy())
    # the two days before the first trading day fold into it; the two after the last are dropped
    first_block = att.loc[: cal[0]]
    assert len(first_block) == 3
    np.testing.assert_allclose(sum_out.attention.loc[cal[0]].to_numpy(), first_block.sum().to_numpy())
    assert sum_out.attention.index[-1] == cal[-1]


def test_align_attention_lag_shifts_forward():
    att, ret, cal, _ = _inputs()
    lag = align_inputs(AttentionData(att), ReturnsData(ret), DataConfig(attention_lag_days=2))
    base = att.loc[cal]
    # day tau carries theta_{tau-2}; the first two rows have no attention and are trimmed
    assert lag.calendar[0] == cal[2]
    np.testing.assert_allclose(lag.attention.to_numpy(), base.to_numpy()[:-2])
    assert lag.returns.index.equals(lag.attention.index)


def test_align_total_returns_subtract_risk_free_with_ffill():
    att, ret, cal, _ = _inputs()
    rf = pd.Series(0.0001 * np.arange(len(cal)), index=cal)
    rf_sparse = rf.drop(cal[[3, 4, 10]])  # missing on some trading days -> last known value
    out = align_inputs(AttentionData(att), ReturnsData(ret, risk_free=rf_sparse), DataConfig(return_kind="total"))
    expected = ret.sub(rf_sparse.reindex(cal).ffill(), axis=0)
    pd.testing.assert_frame_equal(out.returns, expected, check_names=False, check_freq=False)
    with pytest.raises(ValueError):
        align_inputs(AttentionData(att), ReturnsData(ret), DataConfig(return_kind="total"))


def test_trailing_volatility_is_strictly_ex_ante():
    rng = np.random.default_rng(3)
    r = pd.DataFrame(rng.standard_normal((40, 2)), index=pd.bdate_range("2020-01-01", periods=40), columns=["A", "B"])
    r.iloc[7, 0] = np.nan
    vol = trailing_volatility(r, window=10)
    for tau in range(len(r)):
        window = r.iloc[max(0, tau - 10) : tau, 0].dropna()
        if len(window) >= 2:
            assert vol.iloc[tau, 0] == pytest.approx(window.std(ddof=1))
        else:
            assert np.isnan(vol.iloc[tau, 0])
    # changing today's return must not change today's scale
    r2 = r.copy()
    r2.iloc[20, 1] += 100.0
    vol2 = trailing_volatility(r2, window=10)
    assert vol2.iloc[20, 1] == vol.iloc[20, 1]
    assert vol2.iloc[21, 1] != vol.iloc[21, 1]


def test_align_inverse_vol_scaling():
    att, ret, cal, _ = _inputs(n_days=60)
    cfg = DataConfig(asset_weighting="inverse_vol", vol_window_days=20)
    out = align_inputs(AttentionData(att), ReturnsData(ret), cfg)
    assert out.scale is not None
    # leading rows without a scale (fewer than 20 // 4 = 5 prior days) are trimmed
    assert out.calendar[0] == cal[5]
    scale_ref = trailing_volatility(ret, 20).loc[out.calendar]
    pd.testing.assert_frame_equal(out.scale, scale_ref, check_names=False, check_freq=False)
    pd.testing.assert_frame_equal(out.returns, (ret / trailing_volatility(ret, 20)).loc[out.calendar], check_names=False, check_freq=False)
    assert out.attention.index.equals(out.calendar)


def test_align_trims_leading_and_trailing_days_without_data():
    att, ret, cal, _ = _inputs()
    ret2 = ret.copy()
    ret2.iloc[:4] = np.nan  # no asset observed on the first four days
    att2 = att.copy()
    att2.loc[att2.index > cal[-3]] = np.nan  # attention ends early
    out = align_inputs(AttentionData(att2), ReturnsData(ret2), DataConfig())
    assert out.calendar[0] == cal[4]
    assert out.calendar[-1] == cal[-3]
    # a gap in the middle is kept as NaN, not dropped
    ret3 = ret.copy()
    ret3.iloc[10] = np.nan
    out3 = align_inputs(AttentionData(att), ReturnsData(ret3), DataConfig())
    assert len(out3.calendar) == len(cal) and out3.returns.iloc[10].isna().all()


def test_align_passes_metadata_and_coerces_labels():
    att, ret, cal, _ = _inputs()
    ret.columns = [1, 2, 3]
    meta = pd.DataFrame({"asset_class": ["eq", "eq", "fx"]}, index=[1, 2, 3])
    out = align_inputs(AttentionData(att, topic_labels={"x": "X"}), ReturnsData(ret, asset_meta=meta), DataConfig())
    assert out.assets == ["1", "2", "3"]
    assert out.asset_meta is meta and out.topic_labels == {"x": "X"}


def test_align_rejects_disjoint_calendars():
    att, ret, cal, _ = _inputs()
    att_far = att.copy()
    att_far.index = att_far.index + pd.Timedelta(days=3650)
    with pytest.raises(ValueError):
        align_inputs(AttentionData(att_far), ReturnsData(ret), DataConfig())
