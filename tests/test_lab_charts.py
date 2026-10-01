"""Tests for the lab's Plotly figure builders (DESIGN.md G.9, G.13; D69)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from narrative_ipca.exposure_lab import charts
from narrative_ipca.types import LambdaPathPoint, TuningResult

SEED = 20260929
STREAM = 9109  # test-data stream of this file (D71: one stream per component)


def _rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng([SEED, STREAM, offset])


def _exposure_frame(n_assets: int = 6, n_topics: int = 8, offset: int = 0) -> pd.DataFrame:
    rng = _rng(offset)
    return pd.DataFrame(
        rng.uniform(-0.9, 0.9, (n_assets, n_topics)),
        index=pd.Index([f"A{i}" for i in range(n_assets)], name="asset_id"),
        columns=pd.Index([f"T{j:02d}" for j in range(n_topics)], name="topic_id"),
    )


def _path_frame(criterion: bool = True) -> pd.DataFrame:
    """A lambda path with the real ``TuningResult.path_frame`` columns."""
    points = []
    for K in (2, 3):
        for i, lam in enumerate(np.logspace(-3, 0, 6)):
            points.append(
                LambdaPathPoint(
                    lam=float(lam),
                    K=K,
                    total_r2=0.1,
                    pred_r2=0.01,
                    mve_sharpe=1.0 + 0.1 * i,
                    n_selected=12 - 2 * i,
                    gamma_norms=np.zeros(3),
                    objective=1.0,
                    converged=True,
                    n_iter=10,
                    criterion=(1.0 - 0.05 * (i - 2) ** 2) if criterion else None,
                )
            )
    return TuningResult.path_frame(SimpleNamespace(path=points))


METHODS = ["elastic_net", "ridge", "bks_implied", "oracle"]


def _method_r2(n_assets: int = 7, methods: list[str] | None = None, offset: int = 10) -> pd.DataFrame:
    """Assets x methods OOS R2, the layout of ``ComparisonResult.r2``."""
    methods = METHODS if methods is None else methods
    rng = _rng(offset)
    return pd.DataFrame(
        rng.uniform(-0.4, 0.5, (n_assets, len(methods))),
        index=pd.Index([f"A{i}" for i in range(n_assets)], name="asset_id"),
        columns=pd.Index(methods, name="method"),
    )


def _method_sweep(methods, n_windows: int = 6, offset: int = 11) -> pd.DataFrame:
    """Long sweep frame (``start``, ``end``, ``method``, ``median_r2``), the layout of ``ComparisonResult.r2_sweep``."""
    rng = _rng(offset)
    starts = pd.date_range("2025-07-01", periods=n_windows, freq="28D")
    return pd.concat(
        [pd.DataFrame({"start": starts, "end": starts + pd.Timedelta(days=27), "method": m,
                       "median_r2": rng.normal(0.1, 0.1, n_windows)}) for m in methods],
        ignore_index=True,
    )


def _trace_path(n: int = 8) -> pd.DataFrame:
    """A lambda path in the layout of the BKS trace's path table (DESIGN G.16)."""
    lam = np.logspace(-2, 0.5, n)
    crit = 2.9 - 0.15 * (np.arange(n) - 3) ** 2 / 4
    best = crit.max()
    floor = best - 0.02 * max(1.0, abs(best))
    chosen = int(np.flatnonzero(crit >= floor).max())  # largest lambda in the band
    return pd.DataFrame({
        "lam": lam,
        "criterion": crit,
        "se": np.full(n, 1.4),
        "in_band": crit >= floor,
        "best": crit == best,
        "chosen": np.arange(n) == chosen,
        "n_selected": np.linspace(16, 1, n).round(),
    })


def _gamma_path(n_lam: int = 8, n_inst: int = 14, offset: int = 20) -> pd.DataFrame:
    """Lambda x instruments standardised Gamma row norms, shrinking with lambda (DESIGN G.16)."""
    rng = _rng(offset)
    lam = np.logspace(-2, 0.5, n_lam)
    start = rng.uniform(0.05, 1.0, n_inst)
    decay = rng.uniform(0.5, 3.0, n_inst)
    vals = np.maximum(start[None, :] - decay[None, :] * lam[:, None] * 0.3, 0.0)
    return pd.DataFrame(vals, index=pd.Index(lam, name="lam"), columns=[f"T{j:02d}" for j in range(n_inst)])


def _ladder() -> pd.DataFrame:
    """The reference ladder in its chain order, plus the page's direct benchmark row (DESIGN G.16)."""
    keys = ["oracle", "window_truth", "instruments_kernel", "instruments_train", "best_rank", "bks_no_const",
            "bks_implied", "direct"]
    frame = pd.DataFrame(
        {"label": [k.replace("_", " ").capitalize() for k in keys],
         "spearman": [1.0, 0.95, 0.91, 0.55, 0.5, 0.21, 0.19, 0.62],
         "median_r2": [0.081, 0.07, 0.05, 0.03, 0.028, 0.004, 0.003, 0.04]},
        index=keys,
    )
    frame["d_spearman"] = frame["spearman"].diff()
    frame["d_median_r2"] = frame["median_r2"].diff()
    return frame


def _subtitle(fig: go.Figure) -> str:
    """The subtitle line (or the only title line when there is no title)."""
    if charts._HAS_SUBTITLE and fig.layout.title.subtitle.text:
        return fig.layout.title.subtitle.text
    return fig.layout.title.text or ""


def _normal_figures() -> dict[str, go.Figure]:
    rng = _rng(1)
    values = _exposure_frame()
    topics = [f"T{j:02d}" for j in range(20)]
    contrib = pd.Series(rng.normal(0.0, 0.01, 20), index=topics)
    days = pd.bdate_range("2023-01-02", periods=10)
    realized = pd.Series(rng.normal(0.0, 0.01, 10), index=days)
    long_days = pd.bdate_range("2021-01-01", "2023-06-30")
    sweep = pd.DataFrame(
        {
            "start": pd.date_range("2023-01-02", periods=12, freq="W-MON"),
            "median_r2": rng.normal(0.0, 0.1, 12),
            "median_r2_oracle": rng.normal(0.1, 0.1, 12),
            "pooled_r2": rng.normal(0.0, 0.1, 12),
            "pooled_r2_oracle": rng.normal(0.1, 0.1, 12),
        }
    )
    B = pd.DataFrame(rng.normal(0.0, 0.2, (8, 6)), index=topics[:8], columns=values.index)
    r2_methods = _method_r2()
    level = pd.Series(0.2 + rng.normal(0.0, 0.01, len(long_days)), index=long_days)
    ret = pd.Series(rng.normal(0.0, 0.01, len(long_days)), index=long_days)
    train = (pd.Timestamp("2022-07-01"), pd.Timestamp("2022-12-30"))
    fcast = (pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-27"))
    pairs = B.stack()
    return {
        "line_panels": charts.line_panels(
            [{"series": {"Attention": level}, "y_title": "Attention level"},
             {"series": {"Raw return": ret}, "y_title": "Return", "zero_line": True, "tickformat": ".1%"}],
            shade=[(train[0], train[1], "Training window"), (fcast[0], fcast[1], "Forecast window")],
            markers=[(pd.Timestamp("2022-12-23"), "Instrument week")],
            title="Inputs",
        ),
        "matrix_heatmap": charts.matrix_heatmap(values, value_label="Design value", highlight_rows=["A2"]),
        "ladder_chart": charts.ladder_chart(_ladder(), highlight="bks_implied", reference="oracle"),
        "identity_scatter": charts.identity_scatter(pairs, pairs * 0.8 + 0.01, highlight="A1",
                                                    x_title="True", y_title="Estimated"),
        "grouped_bars": charts.grouped_bars(B[["A0", "A1"]], reference=(0.1, "Reference")),
        "lambda_trace_chart": charts.lambda_trace_chart(
            _trace_path(), band_floor=2.84, null_band=(0.87, 2.30, 4.44),
            extra=pd.DataFrame({"spearman": np.linspace(0.3, 0.1, 8)}, index=_trace_path()["lam"])),
        "coefficient_path_chart": charts.coefficient_path_chart(_gamma_path(), selected=["T01", "T02"],
                                                                lam_star=float(_trace_path()["lam"].iloc[3])),

        "method_r2_dots": charts.method_r2_dots(r2_methods),
        "method_sweep_lines": charts.method_sweep_lines(_method_sweep(r2_methods.columns)),
        "exposure_heatmap": charts.exposure_heatmap(values, blank=values.abs() < 0.2),
        "contribution_bars": charts.contribution_bars(
            contrib, true_contrib=contrib * 0.8, realized=0.01, residual=0.01 - contrib.sum(), top_n=8
        ),
        "r2_bars": charts.r2_bars(
            pd.Series(rng.uniform(-0.5, 0.5, 6), index=values.index),
            pd.Series(rng.uniform(-0.5, 0.5, 6), index=values.index),
            pd.Series(0.2, index=values.index),
        ),
        "cumulative_explained": charts.cumulative_explained(realized, realized * 0.5, realized * 0.8),
        "window_sweep_chart": charts.window_sweep_chart(sweep),
        "attention_chart": charts.attention_chart(
            pd.Series(0.2 + rng.normal(0.0, 0.01, len(long_days)), index=long_days, name="S1"),
            pd.Series(rng.normal(0.0, 1.0, len(long_days)), index=long_days),
            window=(pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-27")),
            train_end=pd.Timestamp("2022-12-30"),
        ),
        "gamma_norm_bars": charts.gamma_norm_bars(pd.Series(rng.uniform(0, 1, 12), index=topics[:12]), topics[:3]),
        "lambda_path_chart": charts.lambda_path_chart(_path_frame(), 0.01),
        "exposure_scatter": charts.exposure_scatter(B + rng.normal(0.0, 0.05, B.shape), B, linked=B.abs() > 0.2),
    }


def _empty_figures(kind: str) -> dict[str, go.Figure]:
    """Every builder on empty (``kind="empty"``), all-NaN (``"nan"``) or ``None`` input."""
    idx = pd.Index(["A0", "A1"])
    if kind == "empty":
        frame, series = pd.DataFrame(), pd.Series(dtype=float)
    elif kind == "nan":
        frame = pd.DataFrame(np.nan, index=idx, columns=["T0", "T1"])
        series = pd.Series(np.nan, index=idx)
    else:
        frame, series = None, None
    sweep = frame if kind != "nan" else pd.DataFrame({"start": pd.date_range("2023-01-02", periods=2), "median_r2": np.nan})
    path = frame if kind != "nan" else pd.DataFrame({"K": [3], "lam": [np.nan], "n_selected": [np.nan]})
    long_sweep = frame if kind != "nan" else sweep.assign(method="ridge")
    ladder = frame if kind != "nan" else pd.DataFrame({"label": ["a", "b"], "spearman": np.nan}, index=idx)
    trace_path = path if kind != "nan" else pd.DataFrame({"lam": [0.1, 1.0], "criterion": np.nan,
                                                          "n_selected": np.nan})
    panels = None if kind == "none" else [{"series": {"x": series}, "y_title": "y"}]
    return {
        "line_panels": charts.line_panels(panels),
        "matrix_heatmap": charts.matrix_heatmap(frame),
        "ladder_chart": charts.ladder_chart(ladder),
        "identity_scatter": charts.identity_scatter(series, series, x_title="x", y_title="y"),
        "grouped_bars": charts.grouped_bars(frame),
        "lambda_trace_chart": charts.lambda_trace_chart(trace_path),
        "coefficient_path_chart": charts.coefficient_path_chart(frame),
        "method_r2_dots": charts.method_r2_dots(frame),
        "method_sweep_lines": charts.method_sweep_lines(long_sweep),
        "exposure_heatmap": charts.exposure_heatmap(frame),
        "contribution_bars": charts.contribution_bars(series, realized=0.01, residual=0.01),
        "r2_bars": charts.r2_bars(series, series),
        "cumulative_explained": charts.cumulative_explained(series, series),
        "window_sweep_chart": charts.window_sweep_chart(sweep),
        "attention_chart": charts.attention_chart(series, series),
        "gamma_norm_bars": charts.gamma_norm_bars(series, []),
        "lambda_path_chart": charts.lambda_path_chart(path, None),
        "exposure_scatter": charts.exposure_scatter(frame, frame),
    }


@pytest.fixture(scope="module")
def figures() -> dict[str, go.Figure]:
    return _normal_figures()


# ---------------------------------------------------------------------------
# All builders
# ---------------------------------------------------------------------------
def test_builders_return_figures_on_normal_input(figures: dict[str, go.Figure]) -> None:
    assert len(figures) == 18
    for name, fig in figures.items():
        assert isinstance(fig, go.Figure), name
        assert len(fig.data) > 0, name
        fig.to_json()  # serialisable (dates, numpy arrays, NaN)


@pytest.mark.parametrize("kind", ["empty", "nan", "none"])
def test_builders_return_annotated_empty_figure(kind: str) -> None:
    for name, fig in _empty_figures(kind).items():
        assert isinstance(fig, go.Figure), (kind, name)
        assert len(fig.data) == 0, (kind, name)
        assert len(fig.layout.annotations) == 1, (kind, name)
        fig.to_json()


def test_no_figure_has_a_secondary_y_axis(figures: dict[str, go.Figure]) -> None:
    for name, fig in figures.items():
        layout = fig.layout.to_plotly_json()
        for key, axis in layout.items():
            if key.startswith("yaxis"):
                assert "overlaying" not in axis, (name, key)


def test_every_data_mark_has_hover(figures: dict[str, go.Figure]) -> None:
    for name, fig in figures.items():
        for trace in fig.data:
            if trace.hoverinfo == "skip":
                assert trace.type == "scatter" and trace.mode == "text", (name, trace.name)
            else:
                assert trace.hovertemplate, (name, trace.name)


def test_shared_visual_tokens(figures: dict[str, go.Figure]) -> None:
    for name, fig in figures.items():
        assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)", name
        assert fig.layout.plot_bgcolor == charts.SURFACE, name
        assert fig.layout.font.color == charts.INK, name
        assert list(fig.layout.colorway) == list(charts.CATEGORICAL), name
        # legend whenever two or more series are drawn
        n_series = sum(1 for t in fig.data if t.showlegend is not False and t.hoverinfo != "skip")
        if name not in ("exposure_heatmap",) and n_series >= 2:
            assert fig.layout.showlegend, name


# ---------------------------------------------------------------------------
# Exposure heatmap
# ---------------------------------------------------------------------------
def _trace(fig: go.Figure, name: str):
    matches = [t for t in fig.data if t.name == name]
    assert matches, f"no trace named {name!r}"
    return matches[0]


def test_heatmap_blank_cells_are_nan_and_average_counts_blanks_as_zero() -> None:
    values = _exposure_frame(5, 7, offset=2)
    values.iloc[0, 1] = np.nan  # a missing value is blank too
    blank = values.abs() < 0.3
    fig = charts.exposure_heatmap(values, blank=blank, value_label="Estimated sensitivity")
    z = np.array(_trace(fig, "Estimated sensitivity").z, dtype=float)
    expected_blank = blank.to_numpy() | values.isna().to_numpy()
    assert z.shape == values.shape
    assert np.isnan(z[expected_blank]).all()
    np.testing.assert_allclose(z[~expected_blank], values.to_numpy()[~expected_blank])

    avg = np.array(_trace(fig, "AVERAGE").z, dtype=float).ravel()
    expected_avg = values.where(~expected_blank).fillna(0.0).mean(axis=0).to_numpy()
    np.testing.assert_allclose(avg, expected_avg)
    assert "AVERAGE" in fig.layout.yaxis2.ticktext[0]
    assert fig.layout.yaxis2.domain[1] < fig.layout.yaxis.domain[0]  # separated by a gap

    # text only on displayed cells; white on strong cells, ink otherwise
    text = _trace(fig, "Estimated sensitivity values")
    assert len(text.text) == int((~expected_blank).sum())
    zmax = fig.layout.coloraxis.cmax
    for x, y, colour in zip(text.x, text.y, text.textfont.color):
        v = z[int(y), int(x)]
        assert colour == (charts.WHITE if abs(v) > charts.STRONG_CELL_SHARE * zmax else charts.INK)


def test_heatmap_layout_rows_top_down_labels_on_top() -> None:
    values = _exposure_frame(4, 5)
    fig = charts.exposure_heatmap(values, row_labels={"A0": "EM v World EQ"}, col_labels={"T00": "Energy"})
    assert fig.layout.yaxis.range[0] > fig.layout.yaxis.range[1]  # first row on top
    assert fig.layout.xaxis.side == "top" and fig.layout.xaxis.tickangle == -90
    assert fig.layout.yaxis.ticktext[0] == "EM v World EQ"
    assert fig.layout.xaxis.ticktext[0] == "Energy"
    heat = _trace(fig, "OOS correlation")
    assert heat.xgap == charts.CELL_GAP_PX and heat.ygap == charts.CELL_GAP_PX
    assert heat.hoverongaps is False
    # correlation scale is fixed at +-1 and symmetric
    assert fig.layout.coloraxis.cmin == -1.0 and fig.layout.coloraxis.cmax == 1.0


def test_heatmap_zmax_default_for_non_correlation_values() -> None:
    values = _exposure_frame(3, 3) * 0.1
    fig = charts.exposure_heatmap(values, value_label="True sensitivity")
    assert fig.layout.coloraxis.cmax == pytest.approx(values.abs().to_numpy().max())
    assert fig.layout.coloraxis.cmin == pytest.approx(-values.abs().to_numpy().max())


def test_heatmap_max_cols_truncation_note() -> None:
    values = _exposure_frame(3, 50)
    fig = charts.exposure_heatmap(values, max_cols=40)
    z = np.array(_trace(fig, "OOS correlation").z, dtype=float)
    assert z.shape == (3, 40)
    np.testing.assert_allclose(z, values.iloc[:, :40].to_numpy())
    assert "showing 40 of 50 topics" in fig.layout.title.text
    assert len(_trace(fig, "AVERAGE").z[0]) == 40


def test_heatmap_row_prefixes_and_no_average_row() -> None:
    values = _exposure_frame(4, 3)
    prefix = pd.Series(["L", "S", "L", "S"], index=values.index)
    fig = charts.exposure_heatmap(values, row_prefix=prefix, row_labels={"A0": "EM v World EQ"}, average_row=False)
    ticks = list(fig.layout.yaxis.ticktext)
    assert ticks[0] == "L EM v World EQ"
    assert ticks[1] == "S A1"
    assert not [t for t in fig.data if t.name == "AVERAGE"]


def test_heatmap_auto_text_off_for_large_tables() -> None:
    values = _exposure_frame(80, 10)
    fig = charts.exposure_heatmap(values)
    assert all(t.type == "heatmap" for t in fig.data)


def test_heatmap_example_colours_axis_titles_and_label_lengths() -> None:
    values = _exposure_frame(3, 3)
    long_row = "Information Technology Global v World EQ"  # 40 characters: kept whole
    long_col = "B3 Capital Structure, Financing & Capital Allocation"
    fig = charts.exposure_heatmap(values, row_labels={"A0": long_row}, col_labels={"T00": long_col},
                                  colorscale="example", row_title="Asset (key view)")
    scale = [tuple(c) for c in fig.layout.coloraxis.colorscale]
    assert scale == [tuple(c) for c in charts.EXAMPLE_COLORSCALE]
    assert scale[2][1] == charts.WHITE and scale[-1][1] == "#111111"  # white at zero, black for large positive
    assert fig.layout.yaxis.ticktext[0] == long_row
    assert fig.layout.xaxis.ticktext[0] == long_col[: charts.COL_LABEL_CHARS - 1] + "…"
    assert fig.layout.yaxis.title.text == "Asset (key view)" and fig.layout.xaxis.title.text == "Topics"
    default = charts.exposure_heatmap(values)
    assert [tuple(c) for c in default.layout.coloraxis.colorscale] == [tuple(c) for c in charts.DIVERGING_COLORSCALE]
    with pytest.raises(ValueError):
        charts.exposure_heatmap(values, colorscale="rainbow")


def test_window_sweep_empty_message() -> None:
    fig = charts.window_sweep_chart(pd.DataFrame(), empty_message="No complete 4-week window in the data")
    assert fig.layout.annotations[0].text == "No complete 4-week window in the data"


# ---------------------------------------------------------------------------
# Contribution bars
# ---------------------------------------------------------------------------
def _bars_by_row(fig: go.Figure) -> list[tuple[float, float, str]]:
    """``(y, x, tick label)`` of every bar, top row first; the bottom panel's rows follow the topic panel's."""
    panels = {"y": fig.layout.yaxis, "y2": fig.layout.yaxis2}
    offset = {"y": 0.0, "y2": float(len(fig.layout.yaxis.tickvals))}
    rows = []
    for t in fig.data:
        if t.type != "bar":
            continue
        axis = t.yaxis or "y"
        ticks = dict(zip(panels[axis].tickvals, panels[axis].ticktext))
        rows += [(float(y) + offset[axis], float(x), ticks[y]) for x, y in zip(t.x, t.y)]
    return sorted(rows)


def test_contribution_bars_order_largest_on_top_and_other_topics() -> None:
    rng = _rng(3)
    topics = [f"T{j:02d}" for j in range(20)]
    contrib = pd.Series(rng.normal(0.0, 0.01, 20), index=topics)
    realized = 0.015
    residual = realized - contrib.sum()
    fig = charts.contribution_bars(contrib, realized=realized, residual=residual, top_n=6)

    assert fig.layout.yaxis.range[0] > fig.layout.yaxis.range[1]  # reversed: smallest y is the top row
    rows = _bars_by_row(fig)
    topic_rows = rows[:6]
    abs_x = [abs(x) for _, x, _ in topic_rows]
    assert abs_x == sorted(abs_x, reverse=True)
    expected_top = contrib.abs().sort_values(ascending=False).index[:6]
    assert [label for _, _, label in topic_rows] == list(expected_top)

    other = rows[6]
    assert other[2] == "Other topics (14)"
    rest = contrib.drop(expected_top)
    assert other[1] == pytest.approx(rest.sum() * 100.0)

    labels = [label for _, _, label in rows]
    assert labels[-2] == "Not explained by topics"
    assert "Realised move" in labels[-1]
    assert rows[-2][1] == pytest.approx(residual * 100.0)
    assert rows[-1][1] == pytest.approx(realized * 100.0)
    # the pieces add up to the realised move
    assert sum(x for _, x, _ in rows[:-1]) == pytest.approx(realized * 100.0)
    names = {t.name for t in fig.data}
    assert {"Not explained by topics", "Realised move"} <= names
    assert "True contribution (simulation)" not in names
    # residual and realised move sit in their own panel with their own x-axis (D76)
    assert fig.layout.yaxis2.domain[1] < fig.layout.yaxis.domain[0]
    bottom = [t for t in fig.data if t.type == "bar" and t.yaxis == "y2"]
    assert {t.name for t in bottom} == {"Not explained by topics", "Realised move"}
    top_x = [x for _, x, _ in rows[:-2]]
    assert fig.layout.xaxis.range[1] < 2 * max(abs(v) for v in top_x) + 1e-9  # topic scale ignores the big rows
    assert "(pp)" in fig.layout.xaxis.title.text and " pp" in [t for t in fig.data if t.type == "bar"][0].text[0]


def test_contribution_bars_true_markers_units_and_colours() -> None:
    contrib = pd.Series({"S1": 0.004, "S2": -0.002, "A1": 0.0})
    true = pd.Series({"S1": 0.003, "S2": -0.001, "A1": 0.0005})
    fig = charts.contribution_bars(
        contrib, true_contrib=true, realized=0.01, residual=0.008, units="", labels={"S1": "Energy"}
    )
    diamonds = _trace(fig, "True contribution (simulation)")
    assert diamonds.marker.symbol == "diamond" and diamonds.marker.color == charts.INK
    rows = _bars_by_row(fig)
    assert rows[0][2] == "Energy" and rows[0][1] == pytest.approx(0.004)  # raw units
    assert _trace(fig, "Topic contribution, positive").marker.color == charts.POSITIVE
    assert _trace(fig, "Topic contribution, negative").marker.color == charts.NEGATIVE
    assert _trace(fig, "Realised move").marker.color == charts.INK
    # ties at zero estimate are ordered by the true contribution; no "Other" bar when all fit
    assert not any(label.startswith("Other topics") for _, _, label in rows)


def test_contribution_bars_variance_share_labels() -> None:
    share = pd.Series({"S1": 0.19, "S2": 0.05, "A1": -0.01})
    fig = charts.contribution_bars(share, realized=1.0, residual=1.0 - share.sum(), units="%",
                                   realized_label="Total variation (100%)",
                                   axis_title="Share of the window's return variation (%)")
    rows = _bars_by_row(fig)
    assert rows[-1][2] == "<b>Total variation (100%)</b>" and rows[-1][1] == pytest.approx(100.0)
    assert fig.layout.xaxis.title.text == "Share of the window's return variation (%)"
    assert fig.layout.xaxis.ticksuffix == "%"
    assert fig.layout.xaxis.range[1] < 40.0  # the 100% row does not set the topic scale


# ---------------------------------------------------------------------------
# R2 bars
# ---------------------------------------------------------------------------
def test_r2_bars_sorted_descending_with_clipping_note() -> None:
    rng = _rng(4)
    assets = [f"A{i}" for i in range(8)]
    r2 = pd.Series(rng.uniform(-0.8, 0.6, 8), index=assets)
    oracle = r2 + 0.1
    fig = charts.r2_bars(r2, oracle)
    bars = _trace(fig, "Estimator")
    order = np.argsort(np.asarray(bars.y, dtype=float))
    xs = np.asarray(bars.x, dtype=float)[order]
    assert list(xs) == sorted(xs, reverse=True)
    assert fig.layout.yaxis.range[0] > fig.layout.yaxis.range[1]
    assert _trace(fig, "Oracle (true sensitivities)").marker.color == charts.ORANGE

    r2_low = r2.copy()
    r2_low.iloc[0] = -3.0
    fig_low = charts.r2_bars(r2_low, oracle, pd.Series(0.2, index=assets))
    assert np.nanmin(np.asarray(_trace(fig_low, "Estimator").x, dtype=float)) == -1.0
    subtitle = fig_low.layout.title.subtitle.text if charts._HAS_SUBTITLE else fig_low.layout.title.text
    assert "below -1" in subtitle
    assert _trace(fig_low, "Population truth").marker.color == charts.INK
    # the bars can carry the method's name (Compare methods tab)
    named = charts.r2_bars(r2, oracle, name="BKS-implied")
    assert _trace(named, "BKS-implied").type == "bar" and "BKS-implied R²" in named.data[0].hovertext[0]


# ---------------------------------------------------------------------------
# Other builders
# ---------------------------------------------------------------------------
def test_cumulative_explained_values_in_percent() -> None:
    days = pd.bdate_range("2023-01-02", periods=5)
    realized = pd.Series([0.01, -0.02, 0.005, 0.0, 0.01], index=days)
    fitted = realized * 0.5
    fig = charts.cumulative_explained(realized, fitted)
    np.testing.assert_allclose(_trace(fig, "Realised").y, realized.cumsum() * 100.0)
    np.testing.assert_allclose(_trace(fig, "Explained by topics (estimator)").y, fitted.cumsum() * 100.0)
    assert _trace(fig, "Realised").line.color == charts.INK
    assert fig.layout.showlegend
    assert len(fig.layout.annotations) >= 1  # end labels


def test_window_sweep_partial_columns() -> None:
    sweep = pd.DataFrame({"start": pd.date_range("2023-01-02", periods=3, freq="W-MON"), "median_r2": [0.1, -0.2, 0.05]})
    fig = charts.window_sweep_chart(sweep)
    assert len(fig.data) == 1
    assert not fig.layout.showlegend


def test_attention_chart_single_topic_and_several_topics() -> None:
    days = pd.bdate_range("2022-01-03", "2023-03-31")
    rng = _rng(5)
    levels = pd.Series(0.2 + rng.normal(0, 0.01, len(days)), index=days)
    shocks = pd.Series(rng.normal(0, 1, len(days)), index=days)
    fig = charts.attention_chart(levels, shocks, window=(days[-40], days[-20]), train_end=days[-60])
    assert len(fig.data) == 2
    assert fig.data[0].line.color == fig.data[1].line.color
    assert not fig.layout.showlegend
    assert len(fig.data[0].x) < len(days)  # weekly averages
    assert len(fig.layout.shapes) == 4  # window shading and training-end line in both panels

    lv = pd.DataFrame({k: levels + i * 0.01 for i, k in enumerate(["S1", "S2", "S3"])})
    fig3 = charts.attention_chart(lv, lv * 0 + shocks.to_frame().to_numpy())
    assert len(fig3.data) == 6
    assert fig3.layout.showlegend
    assert [t.line.color for t in fig3.data[::2]] == list(charts.CATEGORICAL[:3])


def test_gamma_norm_bars_sorted_and_truncated() -> None:
    rng = _rng(6)
    norms = pd.Series(rng.uniform(0, 1, 40), index=[f"T{j:02d}" for j in range(40)])
    fig = charts.gamma_norm_bars(norms, ["T01", "T02"], top_n=10)
    rows = sorted((float(y), float(x)) for t in fig.data for x, y in zip(t.x, t.y))
    xs = [x for _, x in rows]
    assert len(xs) == 10 and xs == sorted(xs, reverse=True)
    assert "showing top 10 of 40 topics" in fig.layout.title.text
    for t in fig.data:
        assert t.marker.color == (charts.BLUE if t.name.startswith("Selected") else charts.INK_MUTED)


def test_lambda_path_chart_real_columns_and_fallback() -> None:
    fig = charts.lambda_path_chart(_path_frame(), 0.01)
    assert len(fig.data) == 4  # two K values x two panels
    assert fig.layout.showlegend
    assert len(fig.layout.shapes) == 2
    x = np.asarray(fig.data[0].x, dtype=float)
    np.testing.assert_allclose(x, np.log10(np.logspace(-3, 0, 6)))

    fig_fallback = charts.lambda_path_chart(_path_frame(criterion=False), None)
    titles = [a.text for a in fig_fallback.layout.annotations]
    assert "In-sample MVE Sharpe ratio" in titles
    assert not fig_fallback.layout.shapes


def test_exposure_scatter_linked_split_and_webgl_for_many_points() -> None:
    rng = _rng(7)
    B = pd.DataFrame(rng.normal(0, 0.2, (10, 6)), index=[f"T{j}" for j in range(10)], columns=[f"A{i}" for i in range(6)])
    linked = B.abs() > 0.25
    fig = charts.exposure_scatter(B * 0.9, B, linked=linked)
    n_linked = len(_trace(fig, "Linked in the design").x)
    assert n_linked == int(linked.to_numpy().sum())
    assert n_linked + len(_trace(fig, "Not linked in the design").x) == B.size
    assert fig.layout.shapes[0].type == "line"  # 45-degree line

    big = pd.DataFrame(rng.normal(0, 0.2, (100, 55)))
    fig_big = charts.exposure_scatter(big, big)
    assert fig_big.data[0].type == "scattergl"
    assert not fig_big.layout.showlegend


# ---------------------------------------------------------------------------
# Method comparison charts (G.15)
# ---------------------------------------------------------------------------
def test_method_styles_fixed_slots_reference_in_ink_never_cycled() -> None:
    styles = charts.method_styles(["a", "oracle", "b", "c"])
    assert [styles[m]["color"] for m in ("a", "b", "c")] == list(charts.CATEGORICAL[:3])
    assert [styles[m]["symbol"] for m in ("a", "b", "c")] == list(charts.METHOD_SYMBOLS[:3])
    assert styles["oracle"]["color"] == charts.INK and styles["oracle"]["reference"]
    assert styles["oracle"]["symbol"] == charts.REFERENCE_SYMBOL
    assert len(set(charts.METHOD_SYMBOLS)) == len(charts.METHOD_SYMBOLS) == len(charts.CATEGORICAL)
    # nine methods besides the reference: the ninth gets no slot (not drawn) instead of reusing blue
    many = charts.method_styles([f"m{i}" for i in range(9)] + ["oracle"])
    assert "m8" not in many and len(many) == 9
    assert charts.method_styles(["a", "b"], reference=None)["a"]["color"] == charts.CATEGORICAL[0]


def test_method_styles_slots_fix_each_method_colour_across_selections() -> None:
    """With the full method list as slots, adding OLS or dropping ridge does not recolour BKS-implied
    (review 2026-09-29: without slots it turned from green to amber when OLS was added)."""
    slots = ("elastic_net", "ridge", "ols", "bks_implied", "oracle")
    a = charts.method_styles(["elastic_net", "ridge", "bks_implied", "oracle"], slots=slots)
    b = charts.method_styles(["elastic_net", "ridge", "ols", "bks_implied", "oracle"], slots=slots)
    c = charts.method_styles(["bks_implied", "elastic_net"], slots=slots)
    assert a["bks_implied"] == b["bks_implied"] == c["bks_implied"]
    assert a["bks_implied"]["color"] == charts.CATEGORICAL[3] and b["ols"]["color"] == charts.CATEGORICAL[2]
    assert set(c) == {"bks_implied", "elastic_net"}  # only the methods asked for
    assert b["oracle"]["reference"] and b["oracle"]["color"] == charts.INK
    # a method outside the slots follows them
    assert charts.method_styles(["new", "ridge"], slots=slots)["new"]["color"] == charts.CATEGORICAL[4]
    # both charts take the slots
    r2 = _method_r2(methods=["bks_implied", "elastic_net"])
    dots = charts.method_r2_dots(r2, slots=slots)
    assert {t.name: t.marker.color for t in dots.data}["bks_implied"] == charts.CATEGORICAL[3]
    sweep = charts.method_sweep_lines(_method_sweep(["bks_implied", "elastic_net"]), slots=slots)
    assert {t.name: t.line.color for t in sweep.data}["bks_implied"] == charts.CATEGORICAL[3]


def test_method_colours_stay_with_the_training_window_bks_method() -> None:
    """D88: bks_implied_train is appended to compare.METHODS before the oracle, so the earlier methods keep
    their colour slots; it takes the fifth, whatever the selection, and nothing cycles."""
    from narrative_ipca.exposure_lab.compare import METHODS as ALL

    old = ("elastic_net", "ridge", "ols", "bks_implied", "oracle")  # the slots before D88
    assert [m for m in ALL if m != "bks_implied_train"] == list(old) and ALL[-1] == "oracle"
    new = charts.method_styles(ALL, slots=ALL)
    before = charts.method_styles(old, slots=old)
    for m in old:
        assert new[m] == before[m], m
    assert new["bks_implied_train"]["color"] == charts.CATEGORICAL[4]
    assert new["bks_implied_train"]["symbol"] == charts.METHOD_SYMBOLS[4]
    assert len({new[m]["color"] for m in ALL}) == len(ALL)  # every method its own colour, the oracle in ink
    for picked in (["bks_implied_train", "oracle"], ["elastic_net", "bks_implied", "bks_implied_train"],
                   ["ridge", "bks_implied_train", "ols"]):
        styles = charts.method_styles(picked, slots=ALL)
        assert all(styles[m] == new[m] for m in picked)
    # both charts draw the two BKS variants in their slots
    r2 = _method_r2(methods=["bks_implied_train", "bks_implied", "oracle"])
    dots = charts.method_r2_dots(r2, slots=ALL)
    colours = {t.name: t.marker.color for t in dots.data}
    assert colours["bks_implied"] == charts.CATEGORICAL[3] and colours["bks_implied_train"] == charts.CATEGORICAL[4]
    sweep = charts.method_sweep_lines(_method_sweep(["bks_implied_train", "elastic_net"]), slots=ALL)
    assert {t.name: t.line.color for t in sweep.data}["bks_implied_train"] == charts.CATEGORICAL[4]


def test_method_r2_dots_sorted_by_oracle_colours_symbols_and_hover() -> None:
    r2 = _method_r2()
    labels = {"elastic_net": "Elastic net", "ridge": "Ridge (GCV)", "bks_implied": "BKS-implied",
              "oracle": "Oracle (true sensitivities)"}
    fig = charts.method_r2_dots(r2, labels={"A0": "Energy v World EQ"}, method_labels=labels)
    # one marker series per method, the oracle last (on top) as an ink tick; legend shown
    assert [t.name for t in fig.data] == ["Elastic net", "Ridge (GCV)", "BKS-implied",
                                          "Oracle (true sensitivities)"]
    assert all(t.mode == "markers" for t in fig.data)
    assert [t.marker.color for t in fig.data[:3]] == list(charts.CATEGORICAL[:3])
    assert [t.marker.symbol for t in fig.data[:3]] == list(charts.METHOD_SYMBOLS[:3])
    oracle = fig.data[3]
    assert oracle.marker.color == charts.INK and oracle.marker.symbol == charts.REFERENCE_SYMBOL
    assert fig.layout.showlegend
    # assets sorted by the oracle's R2, highest on top (y position 0 is the top row)
    expected = r2["oracle"].sort_values(ascending=False).index
    ticks = list(fig.layout.yaxis.ticktext)
    assert ticks == [("Energy v World EQ" if a == "A0" else a) for a in expected]
    assert fig.layout.yaxis.range[0] > fig.layout.yaxis.range[1]
    np.testing.assert_allclose(np.asarray(oracle.x, dtype=float), r2.loc[expected, "oracle"].to_numpy())
    np.testing.assert_allclose(np.asarray(fig.data[0].x, dtype=float), r2.loc[expected, "elastic_net"].to_numpy())
    assert "Elastic net: R²" in fig.data[0].hovertext[0]
    assert fig.layout.xaxis.tickformat == ".0%"


def test_method_r2_dots_clipping_missing_reference_and_missing_values() -> None:
    r2 = _method_r2(methods=["ridge", "elastic_net"])
    r2.iloc[2, 0] = -4.0
    r2.iloc[3, 1] = np.nan
    fig = charts.method_r2_dots(r2)  # no oracle column: sorted by the first column, all in colour
    assert [t.name for t in fig.data] == ["ridge", "elastic_net"]
    assert [t.marker.color for t in fig.data] == list(charts.CATEGORICAL[:2])
    xs = np.asarray(fig.data[0].x, dtype=float)
    assert np.nanmin(xs) == -1.0  # drawn at the clip
    finite = xs[np.isfinite(xs)]
    assert list(finite) == sorted(finite, reverse=True)
    subtitle = fig.layout.title.subtitle.text if charts._HAS_SUBTITLE else fig.layout.title.text
    assert "1 value below -100%" in subtitle
    assert "-400.0%" in " ".join(fig.data[0].hovertext)  # the hover keeps the value
    # a single method: no legend
    assert not charts.method_r2_dots(r2[["ridge"]]).layout.showlegend


def test_method_sweep_lines_colours_match_the_dots_and_reference_is_dashed_ink() -> None:
    long = _method_sweep(METHODS)
    shuffled = long.sample(frac=1.0, random_state=0)  # row order must not matter within a method
    fig = charts.method_sweep_lines(shuffled.sort_values("method", key=lambda s: s.map(METHODS.index), kind="stable"),
                                    method_labels={"oracle": "Oracle"})
    assert [t.name for t in fig.data] == ["elastic_net", "ridge", "bks_implied", "Oracle"]
    dots = charts.method_r2_dots(_method_r2())
    assert [t.line.color for t in fig.data] == [t.marker.color for t in dots.data]
    assert [t.line.dash for t in fig.data] == ["solid", "solid", "solid", "dash"]
    assert fig.data[3].line.color == charts.INK
    for t in fig.data:  # sorted by window start
        x = pd.to_datetime(pd.Series(t.x))
        assert x.is_monotonic_increasing and len(x) == 6
    part = long[long["method"] == "ridge"].sort_values("start")
    np.testing.assert_allclose(np.asarray(fig.data[1].y, dtype=float), part["median_r2"].to_numpy())
    assert fig.layout.showlegend and fig.layout.hovermode == "x unified"
    assert fig.layout.yaxis.tickformat == ".0%"
    # one method: no legend; missing columns or no finite value: the empty figure with the message
    assert not charts.method_sweep_lines(_method_sweep(["ridge"])).layout.showlegend
    empty = charts.method_sweep_lines(long.drop(columns="method"), empty_message="No complete 4-week window")
    assert len(empty.data) == 0 and empty.layout.annotations[0].text == "No complete 4-week window"


def test_method_sweep_lines_clip_a_failing_method() -> None:
    """One method far below -100% is drawn at the clip, so the other lines keep their scale; the hover
    keeps the value (review 2026-09-29: a -4,653% ridge line flattened the oracle)."""
    long = _method_sweep(["ridge", "oracle"])
    long.loc[long["method"] == "ridge", "median_r2"] = -46.5
    fig = charts.method_sweep_lines(long)
    ridge = next(t for t in fig.data if t.name == "ridge")
    np.testing.assert_allclose(np.asarray(ridge.y, dtype=float), -1.0)
    np.testing.assert_allclose(np.asarray(ridge.customdata, dtype=float), -46.5)
    assert "customdata" in ridge.hovertemplate
    assert fig.layout.yaxis.range[0] == pytest.approx(-1.05)
    subtitle = fig.layout.title.subtitle.text if charts._HAS_SUBTITLE else fig.layout.title.text
    assert "6 values below -100% drawn at -100%" in subtitle
    # nothing below the clip: no note, and the range follows the data
    plain = charts.method_sweep_lines(_method_sweep(["ridge", "oracle"]))
    text = plain.layout.title.subtitle.text if charts._HAS_SUBTITLE else plain.layout.title.text
    assert "below" not in text and plain.layout.yaxis.range[0] > -1.05


# ---------------------------------------------------------------------------
# BKS trace page builders (G.16; D90)
# ---------------------------------------------------------------------------
def test_registries_cover_every_builder() -> None:
    builders = {name for name in charts.__all__
                if callable(getattr(charts, name)) and name[0].islower() and name != "method_styles"}
    assert set(_normal_figures()) == builders
    assert set(_empty_figures("none")) == builders


def _shapes(fig: go.Figure, kind: str) -> list:
    return [s for s in fig.layout.shapes if s.type == kind]


def test_line_panels_shared_colours_legend_shading_and_markers() -> None:
    days = pd.bdate_range("2024-01-01", "2024-12-31")
    rng = _rng(21)
    a = pd.Series(rng.normal(0, 1, len(days)), index=days)
    b = a.shift(1)  # first value missing: a gap, not a zero
    c = pd.Series(rng.normal(0, 1, len(days)), index=days)
    fig = charts.line_panels(
        [{"series": {"BKS shock": a, "Direct shock": b}, "y_title": "Shock", "zero_line": True,
          "title": "Shocks", "styles": {"Direct shock": {"dash": "dot"}}},
         {"series": {"Signal part": c, "BKS shock": a}, "y_title": "Parts", "tickformat": ".0%",
          "title": "Parts"}],
        shade=[(days[100], days[200], "Training window"), (days[240], days[-1], "Forecast window")],
        markers=[(days[150], "Instrument week"), (days[180], "Window end")],
        x_title="Day",
    )
    assert len(fig.data) == 4
    colours = {t.name: t.line.color for t in fig.data}
    assert colours == {"BKS shock": charts.CATEGORICAL[0], "Direct shock": charts.CATEGORICAL[1],
                       "Signal part": charts.CATEGORICAL[2]}
    bks = [t for t in fig.data if t.name == "BKS shock"]
    assert [t.showlegend for t in bks] == [True, False] and bks[0].line.color == bks[1].line.color
    assert _trace(fig, "Direct shock").line.dash == "dot"
    assert np.isnan(np.asarray(_trace(fig, "Direct shock").y, dtype=float)[0])  # NaN kept as a gap
    assert fig.layout.showlegend and fig.layout.hovermode == "x unified"
    # two windows x two panels, two markers x two panels, dashed
    rects, lines = _shapes(fig, "rect"), _shapes(fig, "line")
    assert len(rects) == 4 and len(lines) == 4
    assert {s.yref for s in rects} == {"y domain", "y2 domain"}
    assert rects[0].opacity != rects[2].opacity  # adjacent windows differ in tone
    assert all(s.line.dash == "dash" for s in lines)
    notes = [a.text for a in fig.layout.annotations]
    assert {"Shocks", "Parts", "Training window", "Forecast window", "Instrument week", "Window end"} <= set(notes)
    # subplot titles left-aligned; stacked panels with their own y axes, shared x
    titles = [a for a in fig.layout.annotations if a.text in ("Shocks", "Parts")]
    assert all(a.x == 0.0 and a.xanchor == "left" for a in titles)
    assert fig.layout.yaxis2.tickformat == ".0%" and fig.layout.yaxis.zeroline
    assert fig.layout.xaxis2.title.text == "Day"
    assert fig.layout.xaxis.matches == "x2" and fig.layout.xaxis.showticklabels is False  # shared x, labels below
    # a note near the right edge ends at its x; notes that would overlap go to separate rows, others share one
    by_text = {a.text: a for a in fig.layout.annotations}
    assert by_text["Forecast window"].xanchor == "right" and by_text["Training window"].xanchor == "left"
    assert by_text["Forecast window"].x == "2024-12-31"  # the window's end
    assert by_text["Training window"].yshift == by_text["Forecast window"].yshift == -2
    assert by_text["Instrument week"].yshift != by_text["Window end"].yshift


def test_line_panels_log_axis_webgl_and_limits() -> None:
    x = np.logspace(-3, 1, 50)
    s = pd.Series(np.sqrt(x), index=x)
    s.loc[0.0] = 1.0  # not drawable on a log axis: dropped
    fig = charts.line_panels([{"series": {"Norm": s}, "y_title": "Norm", "styles": {"Norm": {"fill": "tozeroy"}}}],
                             markers=[(0.1, "λ*")], x_log=True)
    assert fig.layout.xaxis.type == "log"
    assert len(fig.data[0].x) == 50 and np.all(np.asarray(fig.data[0].x) > 0)
    assert fig.data[0].fill == "tozeroy" and fig.data[0].fillcolor.startswith("rgba(")
    assert not fig.layout.showlegend
    # on a log axis the shape sits at x, the annotation at log10(x)
    assert _shapes(fig, "line")[0].x0 == pytest.approx(0.1)
    assert fig.layout.annotations[0].x == pytest.approx(-1.0)

    days = pd.date_range("2010-01-01", periods=4000, freq="D")
    long = pd.Series(np.arange(4000.0), index=days)
    many = charts.line_panels([{"series": {f"s{i}": long for i in range(10)}, "y_title": "y"}] * 4)
    assert all(t.type == "scattergl" for t in many.data)
    assert len({t.yaxis for t in many.data}) == charts.MAX_LINE_PANELS  # panels past three are dropped
    colours = [t.line.color for t in many.data[:10]]
    assert colours[:8] == list(charts.CATEGORICAL) and colours[8:] == [charts.INK_MUTED] * 2  # never cycled
    # a panel without data is dropped; the others stay
    one = charts.line_panels([{"series": {"a": pd.Series([np.nan, np.nan])}, "y_title": "a"},
                              {"series": {"b": pd.Series([1.0, 2.0])}, "y_title": "b"}])
    assert len(one.data) == 1 and one.layout.yaxis.title.text == "b"


def test_matrix_heatmap_truncation_highlight_and_text() -> None:
    rng = _rng(22)
    big = pd.DataFrame(rng.normal(0, 1, (500, 120)), index=[f"A{i:03d}" for i in range(500)],
                       columns=[f"T{j:03d}" for j in range(120)])
    fig = charts.matrix_heatmap(big, value_label="Standardised instrument", highlight_rows=["A300"])
    heat = _trace(fig, "Standardised instrument")
    z = np.array(heat.z, dtype=float)
    assert z.shape == (80, 45)
    ticks = list(fig.layout.yaxis.ticktext)
    assert ticks[-1] == "<b>A300</b>" and ticks[0] == "A000" and "A079" not in ticks  # the highlight kept
    np.testing.assert_allclose(z[-1], big.loc["A300"].iloc[:45].to_numpy())
    assert "showing 80 of 500 rows" in _subtitle(fig) and "45 of 120 columns" in _subtitle(fig)
    assert fig.layout.title.text is not None  # the note shows without a figure title
    outline = _shapes(fig, "rect")
    assert len(outline) == 1 and outline[0].y0 == pytest.approx(78.5)
    assert fig.layout.coloraxis.cmin == pytest.approx(-fig.layout.coloraxis.cmax)
    assert all(t.type == "heatmap" for t in fig.data)  # no cell text above 400 cells

    small = pd.DataFrame([[0.12, -0.034], [np.nan, 0.2]], index=["S1", "S2"], columns=["f1", "f2"])
    fig_small = charts.matrix_heatmap(small, value_label="Gamma", title="Gamma", subtitle="standardised")
    text = _trace(fig_small, "Gamma values")
    assert list(text.text) == ["0.12", "-0.03", "0.20"]  # NaN cell blank, decimals from zmax 0.2
    assert fig_small.layout.coloraxis.cmax == pytest.approx(0.2)
    assert _subtitle(fig_small) == "standardised"
    assert "Gamma: 0.12" in _trace(fig_small, "Gamma").hovertext[0][0]


def test_ladder_chart_order_colours_and_formats() -> None:
    lad = _ladder()
    fig = charts.ladder_chart(lad, highlight="bks_implied", reference=["oracle"])
    assert [t.type for t in fig.data] == ["bar", "bar"]
    spear, r2 = fig.data
    np.testing.assert_allclose(np.asarray(spear.x, dtype=float), lad["spearman"].to_numpy())
    assert list(spear.y) == list(range(len(lad)))  # chain order, first on top
    assert fig.layout.yaxis.range[0] > fig.layout.yaxis.range[1]
    colours = list(spear.marker.color)
    assert colours[0] == charts.INK and colours[6] == charts.BLUE
    assert set(colours[1:6] + colours[7:]) == {charts.INK_MUTED}
    assert list(fig.layout.yaxis.ticktext)[6] == "<b>Bks implied</b>"
    assert spear.text[1] == "0.95" and r2.text[0] == "8.1%"  # percent for R2 metrics
    assert fig.layout.xaxis2.tickformat == ".0%" and not fig.layout.xaxis.tickformat
    assert "Change from the row above: -0.05" in spear.hovertext[1]
    assert "Change from the row above" not in spear.hovertext[0]
    assert fig.layout.xaxis.range[0] == 0.0  # all values positive: bars start at the axis
    assert not fig.layout.showlegend and all(t.showlegend is False for t in fig.data)
    assert fig.layout.yaxis2.matches == "y"  # panels side by side share the rows
    titles = [a.text for a in fig.layout.annotations]
    assert titles == ["Spearman with true sensitivities", "Median OOS R²"]
    assert fig.layout.annotations[1].x == pytest.approx(fig.layout.xaxis2.domain[0])
    # a missing metric column is skipped
    only = charts.ladder_chart(lad.drop(columns="median_r2"))
    assert len(only.data) == 1


def test_identity_scatter_statistics_highlight_and_sampling() -> None:
    rng = _rng(23)
    idx = pd.MultiIndex.from_product([[f"T{j}" for j in range(10)], [f"A{i}" for i in range(6)]],
                                     names=["topic", "asset"])
    x = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    y = 0.5 * x + rng.normal(0, 0.3, len(idx))
    y.iloc[0] = np.nan  # dropped pair
    fig = charts.identity_scatter(x, y, highlight="A2", labels={"A2": "EM v World EQ"},
                                  x_title="Population", y_title="Panel", subtitle="Eq. 5 week")
    ok = y.notna()
    slope, _ = np.polyfit(x[ok], y[ok], 1)
    corr = np.corrcoef(x[ok], y[ok])[0, 1]
    sub = _subtitle(fig)
    assert sub.startswith("Eq. 5 week; 59 points")
    assert f"least-squares slope {slope:.2f}" in sub and f"correlation {corr:.2f}" in sub
    others, high, line = fig.data
    assert high.name == "Highlighted" and high.marker.color == charts.ORANGE and len(high.x) == 10
    assert others.marker.color == charts.INK_MUTED and len(others.x) == 49
    assert "T1 · EM v World EQ" in list(high.customdata)  # tuple keys join the names of their parts
    assert line.name == "Least-squares line" and line.line.dash == "dash"
    np.testing.assert_allclose(np.diff(line.y) / np.diff(line.x), slope)
    assert fig.layout.yaxis.scaleanchor == "x" and list(fig.layout.xaxis.range) == list(fig.layout.yaxis.range)
    assert _shapes(fig, "line")[0].x0 == fig.layout.xaxis.range[0]  # 45-degree line
    assert fig.layout.showlegend

    plain = charts.identity_scatter(x, y, x_title="x", y_title="y", fit_line=False, identity_line=False)
    assert len(plain.data) == 1 and plain.data[0].marker.color == charts.BLUE
    assert not plain.layout.shapes and plain.layout.yaxis.scaleanchor is None

    n = 30_000
    big_x = pd.Series(rng.normal(0, 1, n))
    big = charts.identity_scatter(big_x, big_x * 2.0, highlight=[0, 1, 2], x_title="x", y_title="y",
                                  max_points=5_000)
    assert all(t.type == "scattergl" for t in big.data if t.mode == "markers")
    assert sum(len(t.x) for t in big.data if t.mode == "markers") == 5_000
    assert len(_trace(big, "Highlighted").x) == 3
    assert "showing 5,000 of 30,000 points" in _subtitle(big) and "slope 2.00" in _subtitle(big)


def test_grouped_bars_reference_top_n_sort_and_orientation() -> None:
    rng = _rng(24)
    frame = pd.DataFrame({"B_true": rng.normal(0, 0.2, 40), "B_hat": rng.normal(0, 0.1, 40)},
                         index=[f"T{j:02d}" for j in range(40)])
    fig = charts.grouped_bars(frame, series_labels={"B_true": "True", "B_hat": "BKS-implied"},
                              colors={"B_true": charts.INK}, top_n=10, reference=(0.1, "Threshold"),
                              axis_title="Sensitivity")
    true, est = fig.data
    assert (true.name, est.name) == ("True", "BKS-implied")
    assert true.marker.color == charts.INK and est.marker.color == charts.CATEGORICAL[1]
    kept = frame.abs().max(axis=1).nlargest(10).index
    expected = [t for t in frame.index if t in kept]  # original order
    assert list(fig.layout.yaxis.ticktext) == expected
    assert "showing 10 of 40 rows" in _subtitle(fig)
    assert fig.layout.showlegend and fig.layout.barmode == "group"
    ref = _shapes(fig, "line")
    assert len(ref) == 1 and ref[0].x0 == 0.1 and ref[0].line.dash == "dash"
    assert "Threshold" in [a.text for a in fig.layout.annotations]
    assert fig.layout.xaxis.title.text == "Sensitivity"
    assert true.text is not None and len(true.text) == 10  # 20 bars: values printed

    single = charts.grouped_bars(frame["B_true"], sort_by="B_true")
    assert not single.layout.showlegend
    xs = np.asarray(single.data[0].x, dtype=float)
    assert list(xs) == sorted(xs, reverse=True) and single.data[0].text is None  # 40 bars: no text

    many = charts.grouped_bars(pd.Series(np.arange(500.0)))
    assert len(many.data[0].x) == charts.MAX_BAR_ROWS
    assert f"showing {charts.MAX_BAR_ROWS} of 500 rows" in _subtitle(many)

    weeks = pd.DataFrame({"r2": [0.25, 0.31], "r2_shuffled": [0.24, 0.22]},
                         index=pd.to_datetime(["2025-07-04", "2025-07-11"]))
    v = charts.grouped_bars(weeks, orientation="v", percent=True)
    assert v.data[0].orientation == "v" and list(v.layout.xaxis.ticktext) == ["2025-07-04", "2025-07-11"]
    assert v.layout.yaxis.tickformat == ".0%" and v.data[0].text[0] == "25.0%"
    assert v.layout.yaxis.range[0] == 0.0

    sep = charts.grouped_bars(frame.iloc[:5].assign(B_hat=frame["B_hat"].iloc[:5] * 100), separate=True)
    assert not sep.layout.showlegend and [a.text for a in sep.layout.annotations] == ["B_true", "B_hat"]
    assert sep.data[1].xaxis == "x2" and sep.layout.xaxis2.range[1] > 5 * sep.layout.xaxis.range[1]
    with pytest.raises(ValueError):
        charts.grouped_bars(frame, orientation="x")


def test_lambda_trace_chart_band_markers_and_panels() -> None:
    path = _trace_path()
    fig = charts.lambda_trace_chart(path, band_floor=2.84)
    upper = _trace(fig, "Criterion + 1 standard error")
    lower = _trace(fig, "± 1 standard error")
    np.testing.assert_allclose(np.asarray(upper.y, dtype=float), path["criterion"] + 1.4)
    np.testing.assert_allclose(np.asarray(lower.y, dtype=float), path["criterion"] - 1.4)
    assert lower.fill == "tonexty" and upper.showlegend is False
    assert list(fig.data).index(upper) + 1 == list(fig.data).index(lower)  # the band fills to the upper edge
    crit = _trace(fig, "In-sample Sharpe ratio (annualised)")
    np.testing.assert_allclose(np.asarray(crit.x, dtype=float), path["lam"])
    fills = list(crit.marker.color)
    assert [c == charts.BLUE for c in fills] == path["in_band"].tolist()  # open markers outside the band
    best, chosen = _trace(fig, "Best criterion"), _trace(fig, "Chosen λ")
    assert best.x[0] == pytest.approx(path.loc[path["best"], "lam"].iloc[0])
    assert chosen.x[0] == pytest.approx(path.loc[path["chosen"], "lam"].iloc[0])
    assert chosen.marker.color == charts.ORANGE
    assert fig.layout.xaxis.type == "log" and fig.layout.xaxis2.type == "log"
    assert _trace(fig, "Topics selected").line.shape == "hvh"
    lines = _shapes(fig, "line")
    floor = [s for s in lines if s.xref == "x domain"]
    assert len(floor) == 1 and floor[0].y0 == 2.84 and floor[0].line.dash == "dash"
    vertical = [s for s in lines if s.xref in ("x", "x2")]
    assert len(vertical) == 2 and vertical[0].x0 == pytest.approx(chosen.x[0])  # λ* in both panels
    assert "yaxis3" not in fig.layout.to_plotly_json()  # two panels without extra

    extra = pd.DataFrame({"lam": path["lam"], "spearman": np.linspace(0.3, 0.1, 8),
                          "kept_share": np.linspace(0.6, 0.2, 8), "gamma_rank": 3.0})
    fig3 = charts.lambda_trace_chart(path, lam_star=float(path["lam"].iloc[2]), extra=extra,
                                     extra_labels={"spearman": "Spearman", "kept_share": "Kept share"})
    names = [t.name for t in fig3.data if t.yaxis == "y3"]
    assert names == ["Spearman", "Kept share"]  # the labels choose the columns
    assert [t.line.color for t in fig3.data if t.yaxis == "y3"] == list(charts.CATEGORICAL[2:4])
    assert _trace(fig3, "Chosen λ").x[0] == pytest.approx(path["lam"].iloc[2])
    assert fig3.layout.xaxis3.title.text == "λ (log scale)" and fig3.layout.height > fig.layout.height
    # without se, flags or a floor: the criterion and the selected topics only
    bare = charts.lambda_trace_chart(path[["lam", "criterion", "n_selected"]])
    assert [t.name for t in bare.data] == ["In-sample Sharpe ratio (annualised)", "Topics selected"]
    assert not bare.layout.shapes


def test_lambda_trace_chart_null_band_and_floor() -> None:
    """The no-signal band (5-95% filled, dashed median, SPEC addendum A2) beneath the one tolerance floor (D51)."""
    path = _trace_path()
    lam = path["lam"].to_numpy()
    fig = charts.lambda_trace_chart(path, band_floor=2.84, null_band=(0.87, 2.30, 4.44))
    upper = _trace(fig, "No priced signal: 95% quantile")
    lower = _trace(fig, "No priced signal: 5-95%")
    median = _trace(fig, "No priced signal: median")
    assert list(upper.y) == [4.44, 4.44] and list(lower.y) == [0.87, 0.87] and list(median.y) == [2.30, 2.30]
    assert list(lower.x) == [lam.min(), lam.max()] and list(median.x) == [lam.min(), lam.max()]
    assert upper.showlegend is False and lower.fill == "tonexty"
    assert lower.fillcolor == charts._rgba(charts.INK_MUTED, 0.16)
    assert median.line.dash == "dash" and median.line.color == charts.INK_MUTED
    order = [t.name for t in fig.data]
    assert order.index("No priced signal: 95% quantile") + 1 == order.index("No priced signal: 5-95%")
    assert order.index("No priced signal: median") < order.index("Criterion + 1 standard error")  # beneath the data
    assert all(t.yaxis == "y" for t in (upper, lower, median)) and all(t.hovertemplate for t in fig.data)
    floors = [s for s in _shapes(fig, "line") if s.xref == "x domain"]
    assert [s.y0 for s in floors] == [2.84] and floors[0].line.dash == "dash"
    assert [a.text for a in fig.layout.annotations if "floor" in (a.text or "")] == ["Tolerance band floor 2.84"]
    # without a null band nothing of it is drawn
    no_null = charts.lambda_trace_chart(path, band_floor=2.84)
    assert not any((t.name or "").startswith("No priced signal") for t in no_null.data)
    # malformed quantiles (non-finite or unordered) draw no band
    for bad in ((np.nan, 2.3, 4.4), (4.4, 2.3, 0.9), (1.0, 2.0), None):
        fig_bad = charts.lambda_trace_chart(path, null_band=bad)
        assert not any((t.name or "").startswith("No priced signal") for t in fig_bad.data), bad
    # the band never shows without a criterion
    no_crit = charts.lambda_trace_chart(path[["lam", "n_selected"]], null_band=(0.87, 2.30, 4.44))
    assert not any((t.name or "").startswith("No priced signal") for t in no_crit.data)


def test_coefficient_path_chart_colours_by_norm_at_the_chosen_lambda() -> None:
    norms = _gamma_path()
    lam_star = float(norms.index[3])
    fig = charts.coefficient_path_chart(norms, selected=["T01", "T02"], lam_star=lam_star, max_colored=4,
                                        labels={"T00": "Energy"})
    coloured = [t for t in fig.data if t.showlegend is not False]
    order = norms.iloc[3].sort_values(ascending=False, kind="stable")
    expected = [n for n in order.index if order[n] > 0][:4]
    assert [t.name.removesuffix(" (not selected)") for t in coloured] == [
        "Energy" if n == "T00" else n for n in expected]
    assert [t.line.color for t in coloured] == list(charts.CATEGORICAL[:4])
    grey = [t for t in fig.data if t.showlegend is False]
    assert grey and all(t.line.color in (charts.INK_MUTED, charts.INK_SECONDARY) for t in grey)
    n_grey_lines = sum(int(np.isnan(np.asarray(t.y, dtype=float)).sum()) for t in grey)
    assert n_grey_lines == norms.shape[1] - 4  # every other instrument is one gap-separated line
    assert fig.layout.xaxis.type == "log"
    vline = _shapes(fig, "line")
    assert len(vline) == 1 and vline[0].x0 == pytest.approx(lam_star) and vline[0].line.dash == "dash"
    assert "4 of 14 instruments in colour (largest at the chosen λ)" in _subtitle(fig)
    # without λ*: ranked by the largest norm anywhere on the path; unselected coloured lines are named so
    overall = charts.coefficient_path_chart(norms, selected=["T01"], max_colored=20)
    assert len([t for t in overall.data if t.showlegend is not False]) == len(charts.CATEGORICAL)
    assert "largest on the path" in _subtitle(overall)
    assert any(t.name.endswith("(not selected)") for t in overall.data)
    # no selection given: no selection note, no "(not selected)" names
    plain = charts.coefficient_path_chart(norms)
    assert "selected" not in _subtitle(plain) and not any("selected" in (t.name or "") for t in plain.data)


def test_highlight_keys_accept_lists_tuples_and_multiindex_keys() -> None:
    values = _exposure_frame(4, 3)
    for rows in (("A1", "A3"), ["A1", "A3"], pd.Index(["A1", "A3"])):
        fig = charts.matrix_heatmap(values, highlight_rows=rows)
        assert [t.startswith("<b>") for t in fig.layout.yaxis.ticktext] == [False, True, False, True]
    assert not _shapes(charts.matrix_heatmap(values), "rect")  # the default () highlights nothing
    idx = pd.MultiIndex.from_product([["T1", "T2"], ["A1", "A2"]])
    s = pd.Series([0.1, 0.2, 0.3, 0.4], index=idx)
    full = charts.identity_scatter(s, s, highlight=[("T1", "A2")], x_title="x", y_title="y")
    assert len(_trace(full, "Highlighted").x) == 1
    level = charts.identity_scatter(s, s, highlight="A2", x_title="x", y_title="y")
    assert len(_trace(level, "Highlighted").x) == 2
