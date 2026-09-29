"""Plotly figure builders for the topic-exposure lab (DESIGN.md G.9, G.13; D52, D67-D70).

Pure functions from pandas objects to ``plotly.graph_objects.Figure``. No
Streamlit imports: the dashboard (``dashboard/app.py``) only displays what
these functions return. Every builder returns an annotated empty figure
instead of raising when its input is ``None``, empty or entirely missing.

Visual system (light theme, one set of tokens for every chart)
---------------------------------------------------------------
* Ink: primary ``#0b0b0b``, secondary ``#52514e``, muted ``#898781``; all text
  uses ink colours, never a series colour.
* Gridlines ``#e1e0d9`` (hairline, solid), baseline and zero lines
  ``#c3c2b7``, chart surface ``#fcfcfb``, transparent paper.
* Categorical slots in fixed order (:data:`CATEGORICAL`), never cycled past
  eight.
* Diverging scale for signed values (exposures, correlations,
  contributions): red for negative, the neutral gray ``#f0efec`` at zero,
  blue for positive (:data:`DIVERGING_COLORSCALE`), always symmetric around 0.
  The exposure table can instead use the colours of the owner's desk example
  (:data:`EXAMPLE_COLORSCALE`: red for negative, white at zero, grey to black
  for positive).
* Thin marks, no dual y-axes (two measures go into stacked subplots), a
  legend whenever two or more series are drawn, hover on every data mark.

Displaying in Streamlit: pass ``theme=None`` to ``st.plotly_chart`` so the
Streamlit theme does not override these tokens.

Validity boundaries
-------------------
* Charts show what they are given. Sign flips (long/short view), blank
  rules, column ordering and unit conversions other than the percent display
  of :func:`contribution_bars` are the caller's job.
* :func:`contribution_bars` can also show the BKS per-topic split, which is not
  identified (D52); the caller states that caveat through ``subtitle``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

__all__ = [
    "INK",
    "INK_SECONDARY",
    "INK_MUTED",
    "GRIDLINE",
    "BASELINE",
    "SURFACE",
    "NEUTRAL_MID",
    "FONT_FAMILY",
    "CATEGORICAL",
    "POSITIVE",
    "NEGATIVE",
    "DIVERGING_COLORSCALE",
    "EXAMPLE_COLORSCALE",
    "COLORSCALES",
    "exposure_heatmap",
    "contribution_bars",
    "r2_bars",
    "cumulative_explained",
    "window_sweep_chart",
    "attention_chart",
    "gamma_norm_bars",
    "lambda_path_chart",
    "exposure_scatter",
]

# ---------------------------------------------------------------------------
# Visual tokens
# ---------------------------------------------------------------------------
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
NEUTRAL_MID = "#f0efec"
WHITE = "#ffffff"
FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

#: Categorical slots in fixed order: blue, orange, aqua, yellow, magenta, green, violet, red.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
BLUE = CATEGORICAL[0]
ORANGE = CATEGORICAL[1]
#: Poles of the diverging encoding for signed values (bars coloured by sign).
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
#: Diverging colour scale: dark red, red, neutral gray midpoint, blue, dark blue.
DIVERGING_COLORSCALE: tuple[tuple[float, str], ...] = (
    (0.0, "#8f2323"),
    (0.25, "#e34948"),
    (0.5, NEUTRAL_MID),
    (0.75, "#2a78d6"),
    (1.0, "#104281"),
)

#: Colour scale of the owner's desk example: dark red, red, white midpoint, grey, black.
EXAMPLE_COLORSCALE: tuple[tuple[float, str], ...] = (
    (0.0, "#8f2323"),
    (0.25, "#e34948"),
    (0.5, WHITE),
    (0.75, "#8a8a8a"),
    (1.0, "#111111"),
)

#: Colour scales of :func:`exposure_heatmap` by name.
COLORSCALES: dict[str, tuple[tuple[float, str], ...]] = {
    "diverging": DIVERGING_COLORSCALE,
    "example": EXAMPLE_COLORSCALE,
}

#: Label lengths before truncation (the hover always shows the full name).
ROW_LABEL_CHARS = 45
COL_LABEL_CHARS = 40

#: Heatmap cells with ``|value| > STRONG_CELL_SHARE * zmax`` get white text.
STRONG_CELL_SHARE = 0.55
#: Heatmap gap between cells in pixels (the surface shows through).
CELL_GAP_PX = 2

# Plotly >= 5.23 draws a subtitle under the title (layout.title.subtitle); older
# versions get the subtitle as a second title line.
_HAS_SUBTITLE = hasattr(go.layout.title, "Subtitle")

# Header geometry in pixels (title, optional subtitle line, optional legend row).
_TITLE_TOP_PX = 10
_TITLE_PX = 22
_SUBTITLE_PX = 22
_LEGEND_PX = 26


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _esc(text: Any) -> str:
    """Text safe for Plotly's pseudo-HTML (``<`` and ``>`` escaped)."""
    return str(text).replace("<", "&lt;").replace(">", "&gt;")


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _label(mapping: Mapping[Any, Any] | pd.Series | None, key: Any) -> str:
    """Display name of ``key`` from ``mapping`` (dict or Series), falling back to ``str(key)``."""
    text = None
    if mapping is not None:
        text = mapping.get(key)
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return str(key)
    return str(text)


def _fmt(value: float, decimals: int, *, scale: float = 1.0, suffix: str = "", sign: bool = False) -> str:
    """Number formatted with ``decimals`` places (no negative zero); empty for a missing value."""
    if value is None or not np.isfinite(value):
        return ""
    x = round(float(value) * scale, decimals) + 0.0
    return f"{x:{'+' if sign else ''}.{decimals}f}{suffix}"


def _to_float(value: Any) -> float:
    """``value`` as a float; ``NaN`` when missing or not numeric."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _float_series(obj: Any) -> pd.Series:
    """``obj`` as a float Series; non-numeric and infinite entries become ``NaN``.

    A one-column DataFrame is squeezed; a wider DataFrame gives an empty
    Series (the builder then shows an empty figure).
    """
    if obj is None:
        return pd.Series(dtype=float)
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            logger.warning("charts: expected a Series, got a DataFrame with %d columns", obj.shape[1])
            return pd.Series(dtype=float)
        obj = obj.iloc[:, 0]
    s = obj if isinstance(obj, pd.Series) else pd.Series(obj)
    if s.size == 0:
        return s.astype(float)
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return s.where(np.isfinite(s.to_numpy()))


def _float_frame(obj: Any) -> pd.DataFrame:
    """``obj`` as a float DataFrame; non-numeric and infinite entries become ``NaN``."""
    if obj is None:
        return pd.DataFrame(dtype=float)
    df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    if df.size == 0:
        return df.astype(float)
    if all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes):
        out = df.astype(float)
    else:
        out = df.apply(pd.to_numeric, errors="coerce").astype(float)
    arr = out.to_numpy()
    return pd.DataFrame(np.where(np.isfinite(arr), arr, np.nan), index=out.index, columns=out.columns)


def _bool_frame(obj: Any, index: pd.Index, columns: pd.Index) -> np.ndarray:
    """Boolean array of ``obj`` aligned to ``(index, columns)``; missing entries are ``False``.

    A NumPy array of the right shape is taken positionally.
    """
    if isinstance(obj, np.ndarray) and obj.shape == (len(index), len(columns)):
        return np.where(pd.isna(obj), False, obj).astype(bool)
    df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    arr = df.reindex(index=index, columns=columns).to_numpy(dtype=object)
    missing = pd.isna(arr)
    return np.where(missing, False, arr).astype(bool)


def _no_data(arr: np.ndarray | pd.Series | pd.DataFrame) -> bool:
    a = np.asarray(arr, dtype=float)
    return a.size == 0 or not np.isfinite(a).any()


def _date_str(value: Any) -> str:
    """A timestamp as a Plotly date string (``YYYY-MM-DD`` or with the time when not midnight)."""
    ts = pd.Timestamp(value)
    if ts == ts.normalize():
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _header_px(title: str | None, subtitle: str | None = None, legend: bool = False) -> int:
    """Top margin in pixels needed for the title, the subtitle line and a legend row."""
    y = _TITLE_TOP_PX
    if title:
        y += _TITLE_PX
        if subtitle:
            y += _SUBTITLE_PX
    if legend:
        y += _LEGEND_PX
    return max(16, y + 8)


def _title_text(title: str | None, subtitle: str | None) -> str | None:
    """Title text; the subtitle is appended as a second line only when Plotly has no ``title.subtitle``."""
    if not title:
        return None
    text = _esc(title)
    if subtitle and not _HAS_SUBTITLE:
        text += f"<br><span style='font-size:12px;color:{INK_SECONDARY}'>{_esc(subtitle)}</span>"
    return text


def _base_layout(
    title: str | None,
    height: int,
    *,
    subtitle: str | None = None,
    legend: bool = False,
    extra_top: int = 0,
    margin: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Layout shared by every figure: tokens, fonts, hover label, header geometry.

    Parameters
    ----------
    title:
        Figure title (left-aligned at the top of the figure); ``None`` for none.
    height:
        Figure height in pixels; the width is left to the container.
    subtitle:
        Optional second title line in secondary ink (notes, caveats).
    legend:
        ``True`` shows a horizontal legend row under the title.
    extra_top:
        Pixels added to the top margin below the header (for example for
        rotated column labels on top of a heatmap).
    margin:
        Overrides of the ``l``, ``r``, ``b`` margins.

    Returns
    -------
    dict
        Keyword arguments for ``Figure.update_layout``.
    """
    height = int(max(120, height))
    header = _header_px(title, subtitle, legend)
    m = {"l": 16, "r": 24, "t": header + int(extra_top), "b": 40, "pad": 4}
    if margin:
        m.update(margin)
    layout: dict[str, Any] = {
        "template": "none",
        "height": height,
        "autosize": True,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": SURFACE,
        "font": {"family": FONT_FAMILY, "size": 12, "color": INK},
        "colorway": list(CATEGORICAL),
        "hoverlabel": {
            "bgcolor": WHITE,
            "bordercolor": GRIDLINE,
            "align": "left",
            "font": {"family": FONT_FAMILY, "size": 12, "color": INK},
        },
        "hovermode": "closest",
        "showlegend": bool(legend),
        "margin": m,
        "barcornerradius": 4,
    }
    text = _title_text(title, subtitle)
    if text:
        layout["title"] = {
            "text": text,
            "x": 0.0,
            "xref": "container",
            "xanchor": "left",
            "y": 1.0 - _TITLE_TOP_PX / height,
            "yref": "container",
            "yanchor": "top",
            "pad": {"l": 12},
            "font": {"family": FONT_FAMILY, "size": 15, "color": INK},
        }
        if subtitle and _HAS_SUBTITLE:
            layout["title"]["subtitle"] = {
                "text": _esc(subtitle),
                "font": {"family": FONT_FAMILY, "size": 12, "color": INK_SECONDARY},
            }
    if legend:
        legend_top = _TITLE_TOP_PX + (_TITLE_PX if title else 0) + (_SUBTITLE_PX if title and subtitle else 0) + 2
        layout["legend"] = {
            "orientation": "h",
            "xref": "container",
            "x": 0.0,
            "xanchor": "left",
            "yref": "container",
            "y": 1.0 - legend_top / height,
            "yanchor": "top",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"family": FONT_FAMILY, "size": 11, "color": INK_SECONDARY},
            "itemsizing": "constant",
        }
    return layout


def _style_axes(fig: go.Figure) -> None:
    """Recessive axes on every subplot: hairline solid grid, no axis lines, secondary-ink ticks."""
    axis = {
        "showgrid": True,
        "gridcolor": GRIDLINE,
        "gridwidth": 1,
        "griddash": "solid",
        "zeroline": False,
        "showline": False,
        "linecolor": BASELINE,
        "ticks": "",
        "automargin": True,
        "tickfont": {"family": FONT_FAMILY, "size": 11, "color": INK_SECONDARY},
        "title": {"font": {"family": FONT_FAMILY, "size": 12, "color": INK_SECONDARY}},
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)


_ZERO_LINE = {"zeroline": True, "zerolinecolor": BASELINE, "zerolinewidth": 1}


def _empty_figure(message: str, *, title: str | None = None, height: int = 240) -> go.Figure:
    """A figure with no traces and ``message`` centred in muted ink.

    Parameters
    ----------
    message:
        Why there is nothing to show (for example "No assets in the window").
    title:
        Optional figure title, so the empty state keeps the chart's name.
    height:
        Figure height in pixels.
    """
    fig = go.Figure()
    fig.update_layout(**_base_layout(title, height))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        text=_esc(message),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"family": FONT_FAMILY, "size": 13, "color": INK_MUTED},
    )
    return fig


def _cell_text_trace(z: np.ndarray, zmax: float, *, font_size: int, name: str) -> go.Scatter | None:
    """Text marks for the finite cells of ``z``: two decimals, white on strong cells, ink otherwise."""
    rows, cols = np.nonzero(np.isfinite(z))
    if rows.size == 0:
        return None
    vals = z[rows, cols]
    colors = np.where(np.abs(vals) > STRONG_CELL_SHARE * zmax, WHITE, INK)
    return go.Scatter(
        x=cols,
        y=rows,
        mode="text",
        text=[_fmt(v, 2) for v in vals],
        textfont={"family": FONT_FAMILY, "size": font_size, "color": colors.tolist()},
        hoverinfo="skip",
        showlegend=False,
        name=name,
    )


def _numeric_range(values: Iterable[float], *, pad_share: float, include_zero: bool = True) -> tuple[float, float]:
    """``(lo, hi)`` covering the finite ``values`` (and zero) with ``pad_share`` of the span added each side."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if include_zero:
        arr = np.append(arr, 0.0)
    if arr.size == 0:
        return -1.0, 1.0
    lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    return lo - pad_share * span, hi + pad_share * span


# ---------------------------------------------------------------------------
# 1. Exposure table (G.9 tab 2; D69)
# ---------------------------------------------------------------------------
def exposure_heatmap(
    values: pd.DataFrame,
    *,
    blank: pd.DataFrame | None = None,
    value_label: str = "OOS correlation",
    row_labels: Mapping[Any, Any] | pd.Series | None = None,
    col_labels: Mapping[Any, Any] | pd.Series | None = None,
    row_prefix: pd.Series | Mapping[Any, Any] | None = None,
    average_row: bool = True,
    max_cols: int | None = 40,
    zmax: float | None = None,
    show_text: bool | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    row_height: int | None = None,
    colorscale: str = "diverging",
    row_title: str | None = "Asset",
    col_title: str | None = "Topics",
) -> go.Figure:
    """Asset x topic exposure table in the layout of the owner's desk example (G.9 tab 2).

    Assets are rows (first row on top), topics are columns with their labels
    on top rotated by -90 degrees. Cells are separated by a 2 px surface gap;
    blank cells show the empty surface (no colour, no text). An AVERAGE row
    below a small gap averages each column over the displayed rows with blank
    cells counted as zero (D69).

    Parameters
    ----------
    values:
        ``(N, L)`` frame: assets as rows, topics as columns, in display order.
        The cell metric is named by ``value_label`` (OOS correlation, exposure,
        design value, contribution).
    blank:
        Boolean frame aligned by labels to ``values``; ``True`` blanks a cell
        (for example "not selected by the estimator"). Cells whose value is
        ``NaN`` are blank as well.
    value_label:
        Name of the cell metric, shown in the colour bar and the hover.
    row_labels, col_labels:
        Display names by asset id and by topic id (dict or Series).
    row_prefix:
        Per asset ``"L"`` or ``"S"`` prefixed to the row label (long/short
        view). The caller flips the signs; this only labels the rows.
    average_row:
        Append the AVERAGE row.
    max_cols:
        Keep the first ``max_cols`` columns (the caller orders them); the title
        then notes "showing X of Y topics". ``None`` keeps every column.
    zmax:
        Colour scale runs from ``-zmax`` to ``zmax`` (midpoint 0). Default: 1
        when ``value_label`` contains "correlation", else the largest
        displayed ``|value|``.
    show_text:
        Print the values (2 decimals) in the cells. Default: when at most 45
        columns and 70 rows are displayed.
    title, subtitle:
        Title (default "<value_label> by asset and topic") and an optional
        second line (for example the blank rule in words).
    row_height:
        Pixels per row (default 24, or 14 above 70 rows).
    colorscale:
        ``"diverging"`` (red, gray, blue; default) or ``"example"`` (red,
        white, grey to black, as in the owner's example); see
        :data:`COLORSCALES`.
    row_title, col_title:
        Axis titles for the rows (left) and the columns (above the labels);
        ``None`` for none. Row labels are cut at :data:`ROW_LABEL_CHARS` and
        column labels at :data:`COL_LABEL_CHARS` characters.

    Returns
    -------
    go.Figure
        Heatmap of the displayed rows (trace name ``value_label``), the AVERAGE
        heatmap (trace name ``"AVERAGE"``, second subplot) and text marks.
    """
    base_title = title if title is not None else f"{value_label} by asset and topic"
    if colorscale not in COLORSCALES:
        raise ValueError(f"colorscale must be one of {sorted(COLORSCALES)}")
    vals = _float_frame(values)
    if vals.size == 0 or _no_data(vals):
        return _empty_figure(f"No {value_label} values to show", title=base_title)

    n_total = vals.shape[1]
    title_text = base_title
    if max_cols is not None and int(max_cols) > 0 and n_total > int(max_cols):
        vals = vals.iloc[:, : int(max_cols)]
        title_text = f"{base_title} (showing {int(max_cols)} of {n_total} topics)"
        logger.info("exposure_heatmap: showing %d of %d columns", int(max_cols), n_total)

    arr = vals.to_numpy()
    mask = ~np.isfinite(arr)
    if blank is not None:
        mask |= _bool_frame(blank, vals.index, vals.columns)
    z = np.where(mask, np.nan, arr)
    n_rows, n_cols = z.shape

    if zmax is None or not np.isfinite(zmax) or zmax <= 0:
        if "correlation" in value_label.lower():
            zmax = 1.0
        else:
            ref = z if np.isfinite(z).any() else arr
            m = float(np.nanmax(np.abs(ref))) if np.isfinite(ref).any() else float("nan")
            zmax = m if np.isfinite(m) and m > 0 else 1.0
    zmax = float(zmax)
    avg = np.where(np.isfinite(z), z, 0.0).mean(axis=0) if n_rows else np.full(n_cols, np.nan)

    if show_text is None:
        show_text = n_cols <= 45 and n_rows <= 70
    if row_height is None:
        row_height = 24 if n_rows <= 70 else 14
    font_size = 10 if n_cols <= 30 else 9

    def row_name(asset: Any) -> str:
        name = _label(row_labels, asset)
        if row_prefix is not None:
            p = row_prefix.get(asset)
            if p is not None and not pd.isna(p) and str(p).strip():
                name = f"{str(p).strip()} {name}"
        return name

    row_full = [row_name(a) for a in vals.index]
    col_full = [_label(col_labels, c) for c in vals.columns]
    row_ticks = [_esc(_truncate(t, ROW_LABEL_CHARS)) for t in row_full]
    col_ticks = [_esc(_truncate(t, COL_LABEL_CHARS)) for t in col_full]
    hover = [
        [f"{_esc(r)}<br>{_esc(c)}<br>{_esc(value_label)}: {_fmt(z[i, j], 3)}" for j, c in enumerate(col_full)]
        for i, r in enumerate(row_full)
    ]

    gap_px = 10
    plot_h = row_height * n_rows + ((row_height + gap_px) if average_row else 0)
    label_px = int(6.3 * max((len(t) for t in col_ticks), default=4)) + 14 + (20 if col_title else 0)
    header = _header_px(title_text, subtitle)
    bottom = 16
    height = max(200, header + label_px + plot_h + bottom)

    if average_row:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[n_rows, 1],
            vertical_spacing=gap_px / max(plot_h, 1),
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    fig.update_layout(**_base_layout(title_text, height, subtitle=subtitle, extra_top=label_px, margin={"b": bottom}))
    _style_axes(fig)

    x_pos = list(range(n_cols))
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x_pos,
            y=list(range(n_rows)),
            coloraxis="coloraxis",
            xgap=CELL_GAP_PX,
            ygap=CELL_GAP_PX,
            hoverongaps=False,
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            name=value_label,
        ),
        row=1,
        col=1,
    )
    if show_text:
        text_trace = _cell_text_trace(z, zmax, font_size=font_size, name=f"{value_label} values")
        if text_trace is not None:
            fig.add_trace(text_trace, row=1, col=1)

    if average_row:
        avg_hover = [
            [
                f"AVERAGE over {n_rows} rows (blanks as 0)<br>{_esc(c)}<br>{_esc(value_label)}: {_fmt(avg[j], 3)}"
                for j, c in enumerate(col_full)
            ]
        ]
        fig.add_trace(
            go.Heatmap(
                z=avg[None, :],
                x=x_pos,
                y=[0],
                coloraxis="coloraxis",
                xgap=CELL_GAP_PX,
                ygap=CELL_GAP_PX,
                hoverongaps=False,
                hovertext=avg_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                name="AVERAGE",
            ),
            row=2,
            col=1,
        )
        if show_text:
            avg_text = _cell_text_trace(avg[None, :], zmax, font_size=font_size, name="AVERAGE values")
            if avg_text is not None:
                fig.add_trace(avg_text, row=2, col=1)

    x_common = {
        "tickmode": "array",
        "tickvals": x_pos,
        "range": [-0.5, n_cols - 0.5],
        "showgrid": False,
    }
    y_common = {"tickmode": "array", "showgrid": False, "tickfont": {"size": 11, "color": INK}}
    fig.update_xaxes(
        **x_common,
        ticktext=col_ticks,
        side="top",
        tickangle=-90,
        showticklabels=True,
        tickfont={"size": 11, "color": INK},
        title={"text": _esc(col_title) if col_title else None, "standoff": 6},
        row=1,
        col=1,
    )
    fig.update_yaxes(
        **y_common, tickvals=list(range(n_rows)), ticktext=row_ticks, range=[n_rows - 0.5, -0.5],
        title={"text": _esc(row_title) if row_title else None, "standoff": 6},
        row=1, col=1,
    )
    if average_row:
        fig.update_xaxes(**x_common, showticklabels=False, row=2, col=1)
        fig.update_yaxes(**y_common, tickvals=[0], ticktext=["<b>AVERAGE</b>"], range=[0.5, -0.5], row=2, col=1)

    fig.update_layout(
        coloraxis={
            "colorscale": [list(p) for p in COLORSCALES[colorscale]],
            "cmin": -zmax,
            "cmax": zmax,
            "colorbar": {
                "title": {"text": _esc(value_label), "side": "top", "font": {"size": 11, "color": INK_SECONDARY}},
                "thickness": 10,
                "lenmode": "pixels",
                "len": int(max(80, min(220, plot_h))),
                "y": 1.0,
                "yanchor": "top",
                "x": 1.01,
                "xanchor": "left",
                "outlinewidth": 0,
                "tickfont": {"size": 10, "color": INK_SECONDARY},
            },
        },
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Topic contributions (G.8 points 3-4; G.9 tab 3; D68)
# ---------------------------------------------------------------------------
def contribution_bars(
    contrib: pd.Series,
    *,
    true_contrib: pd.Series | None = None,
    realized: float,
    residual: float,
    top_n: int = 15,
    units: str = "pp",
    labels: Mapping[Any, Any] | pd.Series | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    realized_label: str = "Realised move",
    residual_label: str = "Not explained by topics",
    axis_title: str | None = None,
) -> go.Figure:
    """Horizontal bars of the topics' contributions to one asset's move over the window (G.8 points 3-4).

    Two stacked panels, each with its own x-axis (D76). The top panel holds
    the topics, sorted by absolute contribution, largest on top (ties, such
    as unselected topics with zero contribution, by the absolute true
    contribution); topics beyond ``top_n`` are summed into one
    "Other topics (n)" bar. The bottom panel holds the neutral-gray residual
    bar and the ink realised-move bar. Separate scales keep the topic bars
    readable when the residual and the realised move are much larger. The
    contributions, the other-topics bar and the residual add up to the
    realised move.

    Parameters
    ----------
    contrib:
        Contribution per topic id, ``c_{k,n}`` of G.8 point 3 (decimal
        return points) or a variance share (point 4).
    true_contrib:
        The same computed with the true exposures ``B_true``; drawn as ink
        diamonds ("True contribution (simulation)"). The diamond on the
        residual row is the realised move minus the sum of the true
        contributions.
    realized:
        Realised move over the window (same units as ``contrib``); for a
        variance share, 1 (the whole variation).
    residual:
        Part of the realised move not explained by the topics.
    top_n:
        Number of topics drawn individually.
    units:
        ``"pp"`` (default) displays values times 100 as percentage points
        of return; ``"%"`` displays values times 100 with a percent sign (a
        share); any other string displays the raw values followed by it.
    labels:
        Display names by topic id.
    title, subtitle:
        Title (default "Topic contributions to the realised move") and an
        optional second line (for example the D52 caveat for BKS).
    realized_label, residual_label:
        Names of the two bottom rows (for a variance share, for example
        "Total variation (100%)").
    axis_title:
        Title of the topic panel's x-axis (default "Contribution over the
        window (<units>)").

    Returns
    -------
    go.Figure
        Bars on numeric y positions (0 = top row of each panel, y axes
        reversed) with the row names as tick labels; ``yaxis`` / ``xaxis``
        for the topics and ``yaxis2`` / ``xaxis2`` for the residual and the
        realised move.
    """
    title = title if title is not None else "Topic contributions to the realised move"
    c = _float_series(contrib).dropna()
    if c.empty:
        return _empty_figure("No topic contributions to show", title=title)

    unit = units.strip()
    scale = 100.0 if unit in ("%", "pp") else 1.0
    decimals = 2 if unit in ("%", "pp") else 3
    suffix = "%" if unit == "%" else (" pp" if unit == "pp" else units)

    def fmt(v: float) -> str:
        return _fmt(v, decimals, scale=scale, suffix=suffix)

    t = _float_series(true_contrib).reindex(c.index) if true_contrib is not None else None
    abs_c = np.abs(c.to_numpy())
    tie = np.nan_to_num(np.abs(t.to_numpy()), nan=0.0) if t is not None else np.zeros(len(c))
    order = np.lexsort((-tie, -abs_c))
    k = max(1, int(top_n))
    top_idx, rest_idx = order[:k], order[k:]

    # rows: (y, kind, value, true value, tick label, hover label)
    rows: list[tuple[float, str, float, float, str, str]] = []
    for pos, i in enumerate(top_idx):
        topic = c.index[i]
        name = _label(labels, topic)
        tv = float(t.iloc[i]) if t is not None else float("nan")
        rows.append((float(pos), "topic", float(c.iloc[i]), tv, _esc(_truncate(name, 38)), _esc(name)))
    if rest_idx.size:
        n_other = int(rest_idx.size)
        other_true = float(t.iloc[rest_idx].sum(min_count=1)) if t is not None else float("nan")
        label = f"Other topics ({n_other})"
        rows.append((float(len(top_idx)), "other", float(c.iloc[rest_idx].sum()), other_true, label,
                     f"{label}, summed"))
    realized_v, residual_v = _to_float(realized), _to_float(residual)
    true_resid = realized_v - float(t.sum(min_count=1)) if t is not None else float("nan")
    bottom = [
        (0.0, "residual", residual_v, true_resid, _esc(residual_label), _esc(residual_label)),
        (1.0, "realized", realized_v, float("nan"), f"<b>{_esc(realized_label)}</b>", _esc(realized_label)),
    ]

    n_top = len(rows)
    row_px = 26
    top_h, bottom_h, gap = row_px * (n_top + 0.4), row_px * 2.4, 64.0
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, row_heights=[top_h, bottom_h],
        vertical_spacing=gap / (top_h + gap + bottom_h),
    )
    height = _header_px(title, subtitle, legend=True) + int(top_h + gap + bottom_h) + 56
    fig.update_layout(**_base_layout(title, height, subtitle=subtitle, legend=True), barmode="overlay")
    _style_axes(fig)

    bar_common = {
        "orientation": "h",
        "width": 0.68,
        "textposition": "outside",
        "textfont": {"family": FONT_FAMILY, "size": 11, "color": INK_SECONDARY},
        "cliponaxis": False,
        "constraintext": "none",
        "hovertemplate": "%{hovertext}<extra></extra>",
    }

    def add_bars(sel: list[tuple[float, str, float, float, str, str]], name: str, color: str, panel: int) -> None:
        if not sel:
            return
        fig.add_trace(
            go.Bar(
                x=[r[2] * scale for r in sel],
                y=[r[0] for r in sel],
                text=[fmt(r[2]) for r in sel],
                hovertext=[f"{r[5]}<br>Contribution: {fmt(r[2])}" for r in sel],
                marker={"color": color, "opacity": [0.55 if r[1] == "other" else 1.0 for r in sel], "line": {"width": 0}},
                name=name,
                **bar_common,
            ),
            row=panel,
            col=1,
        )

    add_bars([r for r in rows if not (r[2] < 0)], "Topic contribution, positive", POSITIVE, 1)
    add_bars([r for r in rows if r[2] < 0], "Topic contribution, negative", NEGATIVE, 1)
    add_bars([bottom[0]], residual_label, INK_MUTED, 2)
    add_bars([bottom[1]], realized_label, INK, 2)

    if t is not None:
        for panel, sel in ((1, rows), (2, bottom)):
            diamonds = [r for r in sel if np.isfinite(r[3])]
            if not diamonds:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[r[3] * scale for r in diamonds],
                    y=[r[0] for r in diamonds],
                    mode="markers",
                    marker={"symbol": "diamond", "size": 10, "color": INK, "line": {"width": 1.5, "color": SURFACE}},
                    name="True contribution (simulation)",
                    legendgroup="true",
                    showlegend=panel == 1,
                    hovertext=[f"{r[5]}<br>True contribution: {fmt(r[3])}" for r in diamonds],
                    hovertemplate="%{hovertext}<extra></extra>",
                ),
                row=panel,
                col=1,
            )

    tick_suffix = "%" if unit == "%" else ""
    unit_note = f" ({unit})" if unit else ""
    for panel, sel, axis_text in (
        (1, rows, axis_title if axis_title is not None else f"Contribution over the window{unit_note}"),
        (2, bottom, f"{realized_label} and the unexplained part, own scale{unit_note}"),
    ):
        xs = [r[2] * scale for r in sel] + [r[3] * scale for r in sel]
        lo, hi = _numeric_range(xs, pad_share=0.16)
        fig.update_xaxes(range=[lo, hi], ticksuffix=tick_suffix, title={"text": _esc(axis_text)}, **_ZERO_LINE,
                         row=panel, col=1)
        y_max = sel[-1][0]
        fig.update_yaxes(
            tickmode="array",
            tickvals=[r[0] for r in sel],
            ticktext=[r[4] for r in sel],
            range=[y_max + 0.6, -0.6],
            showgrid=False,
            tickfont={"size": 11, "color": INK},
            row=panel,
            col=1,
        )
    return fig


# ---------------------------------------------------------------------------
# 3. OOS R2 per asset (G.8 point 1; G.9 tab 1; D67)
# ---------------------------------------------------------------------------
def r2_bars(
    r2: pd.Series,
    r2_oracle: pd.Series | None = None,
    r2_true: pd.Series | None = None,
    *,
    labels: Mapping[Any, Any] | pd.Series | None = None,
    title: str | None = None,
    clip: float = -1.0,
) -> go.Figure:
    """OOS R2 per asset: estimator bars, oracle markers, population-truth ticks (G.8 point 1).

    Bars are sorted by the estimator's R2, largest on top. Values below
    ``clip`` are drawn at ``clip`` and counted in a note under the title;
    the hover shows the unclipped value.

    Parameters
    ----------
    r2:
        Per asset uncentered OOS R2 of the estimator (blue bars).
    r2_oracle:
        Per asset OOS R2 of the oracle, exposures ``B_true`` (orange markers).
    r2_true:
        Per asset population share of variance explained, ``R2_true`` of
        G.5.3 (small ink ticks).
    labels:
        Display names by asset id.
    title:
        Title (default "Out-of-sample R² per asset").
    clip:
        Lower display limit of the x axis.
    """
    title = title if title is not None else "Out-of-sample R² per asset"
    est = _float_series(r2)
    ora = _float_series(r2_oracle) if r2_oracle is not None else None
    tru = _float_series(r2_true) if r2_true is not None else None
    if _no_data(est) and (ora is None or _no_data(ora)):
        return _empty_figure("No R² values to show", title=title)

    assets = est.index if not est.empty else ora.index  # type: ignore[union-attr]
    e = est.reindex(assets)
    o = ora.reindex(assets) if ora is not None else pd.Series(np.nan, index=assets)
    tr = tru.reindex(assets) if tru is not None else pd.Series(np.nan, index=assets)
    ev = e.to_numpy()
    ov = o.to_numpy()
    key_e = np.where(np.isfinite(ev), -ev, np.inf)
    key_o = np.where(np.isfinite(ov), -ov, np.inf)
    order = np.lexsort((key_o, key_e))
    assets = assets[order]
    e, o, tr = e.iloc[order], o.iloc[order], tr.iloc[order]
    pos = np.arange(len(assets), dtype=float)
    names = [_label(labels, a) for a in assets]

    all_vals = np.concatenate([e.to_numpy(), o.to_numpy(), tr.to_numpy()])
    n_clipped = int(np.sum(np.isfinite(all_vals) & (all_vals < clip)))
    subtitle = (
        f"{n_clipped} value{'s' if n_clipped != 1 else ''} below {clip:g} drawn at {clip:g} (hover shows the value)"
        if n_clipped
        else None
    )

    def clipped(s: pd.Series) -> np.ndarray:
        v = s.to_numpy()
        return np.where(np.isfinite(v), np.maximum(v, clip), np.nan)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=clipped(e),
            y=pos,
            orientation="h",
            width=0.62,
            marker={"color": BLUE, "line": {"width": 0}},
            name="Estimator",
            hovertext=[f"{_esc(n)}<br>Estimator R²: {_fmt(v, 3)}" for n, v in zip(names, e.to_numpy())],
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    n_series = 1
    if np.isfinite(o.to_numpy()).any():
        n_series += 1
        fig.add_trace(
            go.Scatter(
                x=clipped(o),
                y=pos,
                mode="markers",
                marker={"symbol": "circle", "size": 9, "color": ORANGE, "line": {"width": 1.5, "color": SURFACE}},
                name="Oracle (true exposures)",
                hovertext=[f"{_esc(n)}<br>Oracle R²: {_fmt(v, 3)}" for n, v in zip(names, o.to_numpy())],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )
    if np.isfinite(tr.to_numpy()).any():
        n_series += 1
        fig.add_trace(
            go.Scatter(
                x=clipped(tr),
                y=pos,
                mode="markers",
                marker={"symbol": "line-ns-open", "size": 14, "color": INK, "line": {"width": 2, "color": INK}},
                name="Population truth",
                hovertext=[f"{_esc(n)}<br>Population R²: {_fmt(v, 3)}" for n, v in zip(names, tr.to_numpy())],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    legend = n_series >= 2
    height = _header_px(title, subtitle, legend) + 22 * len(assets) + 56
    fig.update_layout(**_base_layout(title, height, subtitle=subtitle, legend=legend), barmode="overlay")
    _style_axes(fig)
    shown = np.concatenate([clipped(e), clipped(o), clipped(tr)])
    lo, hi = _numeric_range(shown, pad_share=0.05)
    fig.update_xaxes(
        range=[max(lo, clip - 0.05), hi],
        title={"text": "Out-of-sample R² (share of return variation explained)"},
        **_ZERO_LINE,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=pos.tolist(),
        ticktext=[_esc(_truncate(n, 34)) for n in names],
        range=[len(assets) - 0.4, -0.6],
        showgrid=False,
        tickfont={"size": 11, "color": INK},
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Cumulative realised vs explained return (G.9 tab 3)
# ---------------------------------------------------------------------------
def cumulative_explained(
    realized_daily: pd.Series,
    fitted: pd.Series | None = None,
    fitted_oracle: pd.Series | None = None,
    *,
    title: str | None = None,
) -> go.Figure:
    """Cumulative realised return against the topic-explained return through the window, in percent.

    Parameters
    ----------
    realized_daily:
        One asset's realised daily returns ``r_{n,t+l}`` over the window
        (decimals, DatetimeIndex); drawn in ink.
    fitted:
        The estimator's topic-explained daily returns ``rhat_{n,t+l}`` (G.8);
        drawn in blue. Aligned to ``realized_daily``'s index.
    fitted_oracle:
        The same with the true exposures; drawn in orange.
    title:
        Title (default "Cumulative realised vs topic-explained return").

    Notes
    -----
    Missing days count as zero in the cumulative sums. End labels are
    dropped where they would overlap a label already placed (realised first).
    """
    title = title if title is not None else "Cumulative realised vs topic-explained return"
    r = _float_series(realized_daily)
    if _no_data(r):
        return _empty_figure("No realised returns in the window", title=title)
    idx = r.index
    series: list[tuple[str, pd.Series, str]] = [("Realised", r, INK)]
    if fitted is not None:
        f = _float_series(fitted).reindex(idx)
        if not _no_data(f):
            series.append(("Explained by topics (estimator)", f, BLUE))
    if fitted_oracle is not None:
        fo = _float_series(fitted_oracle).reindex(idx)
        if not _no_data(fo):
            series.append(("Explained by topics (oracle)", fo, ORANGE))

    mode = "lines+markers" if len(idx) <= 30 else "lines"
    legend = len(series) >= 2
    fig = go.Figure()
    ends: list[tuple[float, Any, str]] = []
    for name, s, color in series:
        cum = s.fillna(0.0).cumsum() * 100.0
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=cum.to_numpy(),
                mode=mode,
                line={"color": color, "width": 2},
                marker={"size": 8, "color": color, "line": {"width": 1.5, "color": SURFACE}},
                name=name,
                hovertemplate=f"{_esc(name)}: %{{y:.2f}}%<extra></extra>",
            )
        )
        ends.append((float(cum.iloc[-1]), idx[-1], name))

    height = 380
    fig.update_layout(**{**_base_layout(title, height, legend=legend, margin={"r": 72}), "hovermode": "x unified"})
    _style_axes(fig)
    fig.update_yaxes(ticksuffix="%", title={"text": "Cumulative return (%)"}, **_ZERO_LINE)
    fig.update_xaxes(showspikes=True, spikecolor=BASELINE, spikethickness=1, spikedash="solid", spikemode="across")
    if isinstance(idx, pd.DatetimeIndex) and len(idx) and bool((idx.dayofweek < 5).all()):
        fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])  # weekday calendar: no weekend gaps

    all_y = np.concatenate([np.asarray(tr.y, dtype=float) for tr in fig.data])
    span = float(np.nanmax(all_y) - np.nanmin(all_y)) if np.isfinite(all_y).any() else 0.0
    min_sep = 0.05 * span if span > 0 else 0.0
    placed: list[float] = []
    for y_end, x_end, _name in ends:
        if not np.isfinite(y_end) or any(abs(y_end - p) < min_sep for p in placed):
            continue
        placed.append(y_end)
        fig.add_annotation(
            x=x_end,
            y=y_end,
            text=_fmt(y_end, 2, suffix="%", sign=True),
            showarrow=False,
            xanchor="left",
            xshift=8,
            font={"family": FONT_FAMILY, "size": 11, "color": INK_SECONDARY},
        )
    return fig


# ---------------------------------------------------------------------------
# 5. Window sweep (G.8 point 6; G.9 tab 1)
# ---------------------------------------------------------------------------
_SWEEP_SERIES: tuple[tuple[str, str, str, str], ...] = (
    ("median_r2", "Median over assets, estimator", BLUE, "solid"),
    ("median_r2_oracle", "Median over assets, oracle", ORANGE, "solid"),
    ("pooled_r2", "Pooled, estimator", BLUE, "dot"),
    ("pooled_r2_oracle", "Pooled, oracle", ORANGE, "dot"),
)


def window_sweep_chart(
    sweep: pd.DataFrame, *, title: str | None = None, empty_message: str = "No forecast windows to show"
) -> go.Figure:
    """OOS R2 over consecutive non-overlapping forecast windows with the training fit frozen (G.8 point 6).

    Colour follows the model (estimator blue, oracle orange); the line style
    follows the measure (median over assets solid, pooled over asset-days
    dotted).

    Parameters
    ----------
    sweep:
        One row per window with ``start`` (window start date) and any of
        ``median_r2``, ``median_r2_oracle``, ``pooled_r2``,
        ``pooled_r2_oracle``. Without ``start`` the index is used.
    title:
        Title (default "Out-of-sample R² across forecast windows").
    empty_message:
        Text of the empty figure (for example why no complete window fits).
    """
    title = title if title is not None else "Out-of-sample R² across forecast windows"
    if sweep is None or not isinstance(sweep, pd.DataFrame) or sweep.empty:
        return _empty_figure(empty_message, title=title)
    cols = [s for s in _SWEEP_SERIES if s[0] in sweep.columns]
    data = _float_frame(sweep[[s[0] for s in cols]]) if cols else pd.DataFrame()
    if data.empty or _no_data(data):
        return _empty_figure(empty_message, title=title)

    x_raw = sweep["start"] if "start" in sweep.columns else pd.Series(sweep.index, index=sweep.index)
    x = pd.to_datetime(x_raw, errors="coerce")
    if x.isna().all():
        x = x_raw
    order = np.argsort(np.asarray(x), kind="stable")
    x = pd.Series(np.asarray(x)[order])
    data = data.iloc[order]
    mode = "lines+markers" if len(data) <= 40 else "lines"

    fig = go.Figure()
    for col, name, color, dash in cols:
        y = data[col].to_numpy()
        if not np.isfinite(y).any():
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode=mode,
                line={"color": color, "width": 2, "dash": dash},
                marker={"size": 8, "color": color, "line": {"width": 1.5, "color": SURFACE}},
                name=name,
                hovertemplate=f"{_esc(name)}: %{{y:.3f}}<extra></extra>",
            )
        )
    legend = len(fig.data) >= 2
    subtitle = f"{len(data)} window{'s' if len(data) != 1 else ''}; training fit frozen"
    fig.update_layout(**{**_base_layout(title, 400, subtitle=subtitle, legend=legend), "hovermode": "x unified"})
    _style_axes(fig)
    fig.update_yaxes(title={"text": "Out-of-sample R²"}, **_ZERO_LINE)
    fig.update_xaxes(title={"text": "Window start"}, showspikes=True, spikecolor=BASELINE, spikethickness=1,
                     spikedash="solid", spikemode="across")
    return fig


# ---------------------------------------------------------------------------
# 6. Attention levels and observed shocks (G.5.2-G.5.3; D62)
# ---------------------------------------------------------------------------
def _as_frame(obj: Any) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame(dtype=float)
    if isinstance(obj, pd.Series):
        return _float_frame(obj.to_frame(name=obj.name if obj.name is not None else "topic"))
    return _float_frame(obj)


def attention_chart(
    levels: pd.Series | pd.DataFrame,
    shocks: pd.Series | pd.DataFrame,
    *,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    train_end: pd.Timestamp | None = None,
    labels: Mapping[Any, Any] | pd.Series | None = None,
    x_range: tuple[Any, Any] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Weekly average attention level (top) and daily observed shock (bottom) sharing the date axis.

    Parameters
    ----------
    levels:
        Daily attention levels ``a_{k,t}`` (G.5.2) of one topic (Series) or
        of up to eight topics (DataFrame, one colour slot each); averaged
        per week ending Friday, as in the report's relative-attention views.
    shocks:
        Daily observed standardised shocks ``sh_{k,t}`` (G.5.3), same shape.
    window:
        ``(start, end)`` of the forecast window, shaded in both panels.
    train_end:
        Last training day, marked by a vertical line.
    labels:
        Display names by topic id (used for the legend and hover).
    x_range:
        Optional ``(start, end)`` of the visible date range.
    title:
        Title (default "Topic attention and observed shocks").
    """
    title = title if title is not None else "Topic attention and observed shocks"
    if isinstance(levels, pd.Series) and isinstance(shocks, pd.Series):
        # One topic: both panels show the same series name.
        common = levels.name if levels.name is not None else shocks.name
        common = common if common is not None else "topic"
        levels, shocks = levels.rename(common), shocks.rename(common)
    lv = _as_frame(levels)
    sh = _as_frame(shocks)
    if (lv.empty or _no_data(lv)) and (sh.empty or _no_data(sh)):
        return _empty_figure("No attention series to show", title=title)
    cols = list(dict.fromkeys(list(lv.columns) + list(sh.columns)))
    if len(cols) > len(CATEGORICAL):
        logger.warning("attention_chart: %d topics given; showing the first %d", len(cols), len(CATEGORICAL))
        cols = cols[: len(CATEGORICAL)]
    if isinstance(lv.index, pd.DatetimeIndex) and not lv.empty:
        lv = lv.sort_index().resample("W-FRI").mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Weekly average attention level", "Daily observed shock (standardised)"),
    )
    fig.update_annotations(x=0.0, xanchor="left", font={"family": FONT_FAMILY, "size": 12, "color": INK_SECONDARY})
    legend = len(cols) >= 2
    for i, col in enumerate(cols):
        color = CATEGORICAL[i]
        name = _label(labels, col)
        if col in lv.columns and not _no_data(lv[col]):
            fig.add_trace(
                go.Scatter(
                    x=lv.index,
                    y=lv[col].to_numpy(),
                    mode="lines",
                    line={"color": color, "width": 2 if len(cols) == 1 else 1.5},
                    name=name,
                    legendgroup=str(col),
                    showlegend=legend,
                    hovertemplate=f"{_esc(name)}<br>Week to %{{x|%Y-%m-%d}}<br>Average attention: %{{y:.4f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if col in sh.columns and not _no_data(sh[col]):
            fig.add_trace(
                go.Scatter(
                    x=sh.index,
                    y=sh[col].to_numpy(),
                    mode="lines",
                    line={"color": color, "width": 1},
                    name=name,
                    legendgroup=str(col),
                    showlegend=False,
                    hovertemplate=f"{_esc(name)}<br>%{{x|%Y-%m-%d}}<br>Shock: %{{y:.2f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

    height = 480
    fig.update_layout(**_base_layout(title, height, legend=legend, extra_top=22))
    _style_axes(fig)
    fig.update_yaxes(title={"text": "Attention level"}, row=1, col=1)
    fig.update_yaxes(title={"text": "Shock (sd units)"}, row=2, col=1, **_ZERO_LINE)
    if x_range is not None:
        fig.update_xaxes(range=[_date_str(x_range[0]), _date_str(x_range[1])])

    note_font = {"family": FONT_FAMILY, "size": 10, "color": INK_SECONDARY}
    if window is not None:
        x0, x1 = _date_str(window[0]), _date_str(window[1])
        for xref, yref in (("x", "y domain"), ("x2", "y2 domain")):
            fig.add_shape(
                type="rect", xref=xref, yref=yref, x0=x0, x1=x1, y0=0, y1=1,
                fillcolor=GRIDLINE, opacity=0.8, line={"width": 0}, layer="below",
            )
        fig.add_annotation(
            x=x0, xref="x", y=0.0, yref="y domain", text="Forecast window", showarrow=False,
            xanchor="right", yanchor="bottom", xshift=-4, font=note_font,
        )
    if train_end is not None:
        xt = _date_str(train_end)
        for xref, yref in (("x", "y domain"), ("x2", "y2 domain")):
            fig.add_shape(
                type="line", xref=xref, yref=yref, x0=xt, x1=xt, y0=0, y1=1,
                line={"color": INK_SECONDARY, "width": 1},
            )
        fig.add_annotation(
            x=xt, xref="x", y=1.0, yref="y domain", text="Training end", showarrow=False,
            xanchor="right", yanchor="top", xshift=-4, font=note_font,
        )
    return fig


# ---------------------------------------------------------------------------
# 7. BKS Gamma row norms (G.7.2; G.9 tab 4)
# ---------------------------------------------------------------------------
def gamma_norm_bars(
    norms: pd.Series,
    selected: Iterable[Any] = (),
    *,
    labels: Mapping[Any, Any] | pd.Series | None = None,
    top_n: int = 30,
    title: str | None = None,
) -> go.Figure:
    """Row norms of the BKS ``Gamma`` per topic, largest on top; selected topics blue, others muted.

    Parameters
    ----------
    norms:
        Row norm of ``Gamma`` per topic id (``const`` excluded).
    selected:
        Topic ids with a nonzero ``Gamma`` row at the chosen lambda.
    labels:
        Display names by topic id.
    top_n:
        Number of topics drawn; the title notes "showing top X of Y topics".
    title:
        Title (default "BKS Gamma row norms by topic").
    """
    base_title = title if title is not None else "BKS Gamma row norms by topic"
    n = _float_series(norms).dropna()
    if n.empty:
        return _empty_figure("No Gamma row norms to show", title=base_title)
    sel = {str(s) for s in (selected or ())}
    order = np.argsort(-n.to_numpy(), kind="stable")
    k = max(1, int(top_n))
    title_text = base_title if len(n) <= k else f"{base_title} (showing top {k} of {len(n)} topics)"
    n = n.iloc[order[:k]]
    pos = np.arange(len(n), dtype=float)
    is_sel = np.array([str(t) in sel for t in n.index])
    names = [_label(labels, t) for t in n.index]

    fig = go.Figure()
    for flag, name, color in ((True, "Selected (nonzero Gamma row)", BLUE), (False, "Not selected", INK_MUTED)):
        m = is_sel == flag
        if not m.any():
            continue
        fig.add_trace(
            go.Bar(
                x=n.to_numpy()[m],
                y=pos[m],
                orientation="h",
                width=0.62,
                marker={"color": color, "line": {"width": 0}},
                name=name,
                hovertext=[
                    f"{_esc(nm)}<br>Gamma row norm: {v:.4g}<br>{'Selected' if flag else 'Not selected'}"
                    for nm, v in zip(np.array(names, dtype=object)[m], n.to_numpy()[m])
                ],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )
    legend = len(fig.data) >= 2
    height = _header_px(title_text, None, legend) + 22 * len(n) + 56
    fig.update_layout(**_base_layout(title_text, height, legend=legend), barmode="overlay")
    _style_axes(fig)
    fig.update_xaxes(title={"text": "Gamma row norm"}, rangemode="tozero", **_ZERO_LINE)
    fig.update_yaxes(
        tickmode="array",
        tickvals=pos.tolist(),
        ticktext=[_esc(_truncate(nm, 34)) for nm in names],
        range=[len(n) - 0.4, -0.6],
        showgrid=False,
        tickfont={"size": 11, "color": INK},
    )
    return fig


# ---------------------------------------------------------------------------
# 8. BKS lambda path (G.7.2; D51, D70)
# ---------------------------------------------------------------------------
def lambda_path_chart(
    path: pd.DataFrame | None,
    lam_star: float | None = None,
    *,
    criterion_label: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Number of selected topics (top) and the tuning criterion (bottom) against ``log10(lambda)``.

    Parameters
    ----------
    path:
        ``TuningResult.path_frame()``: columns ``K``, ``lam``, ``n_selected``,
        ``criterion`` (value of the tuning criterion, ``None`` when not
        computed), ``mve_sharpe``, ``total_r2``, ... One line per ``K``.
        When ``criterion`` is entirely missing, ``mve_sharpe`` is drawn.
    lam_star:
        Chosen penalty ``lambda*``, marked by a vertical line in both panels.
    criterion_label:
        Name of the criterion panel (default "Tuning criterion").
    title:
        Title (default "BKS lambda path").
    """
    title = title if title is not None else "BKS lambda path"
    if path is None or not isinstance(path, pd.DataFrame) or path.empty or "lam" not in path.columns:
        return _empty_figure("No lambda path to show", title=title)
    lam = _float_series(path["lam"]).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        logl = np.where(np.isfinite(lam) & (lam > 0), np.log10(np.where(lam > 0, lam, 1.0)), np.nan)
    if not np.isfinite(logl).any():
        return _empty_figure("No positive lambda values on the path", title=title)

    crit_col, crit_name = None, None
    if "criterion" in path.columns and not _no_data(_float_series(path["criterion"])):
        crit_col, crit_name = "criterion", criterion_label or "Tuning criterion"
    elif "mve_sharpe" in path.columns and not _no_data(_float_series(path["mve_sharpe"])):
        crit_col, crit_name = "mve_sharpe", criterion_label or "In-sample MVE Sharpe ratio"
    n_sel = _float_series(path["n_selected"]).to_numpy() if "n_selected" in path.columns else np.full(len(path), np.nan)
    crit = _float_series(path[crit_col]).to_numpy() if crit_col else np.full(len(path), np.nan)
    ks = path["K"].to_numpy() if "K" in path.columns else np.zeros(len(path), dtype=int)
    k_values = list(dict.fromkeys(ks[np.isfinite(logl)].tolist()))
    try:
        k_values = sorted(k_values)
    except TypeError:
        pass
    if len(k_values) > len(CATEGORICAL):
        logger.warning("lambda_path_chart: %d values of K; showing the first %d", len(k_values), len(CATEGORICAL))
        k_values = k_values[: len(CATEGORICAL)]
    multi = len(k_values) >= 2

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=("Topics selected", crit_name or "Tuning criterion (not available)"),
    )
    fig.update_annotations(x=0.0, xanchor="left", font={"family": FONT_FAMILY, "size": 12, "color": INK_SECONDARY})
    for i, kv in enumerate(k_values):
        m = (ks == kv) & np.isfinite(logl)
        o = np.argsort(logl[m])
        x = logl[m][o]
        lam_k = lam[m][o]
        color = CATEGORICAL[i] if multi else BLUE
        name = f"K = {kv}" if "K" in path.columns else "Path"
        common = {
            "mode": "lines+markers",
            "line": {"color": color, "width": 2},
            "marker": {"size": 8, "color": color, "line": {"width": 1.5, "color": SURFACE}},
            "name": name,
            "legendgroup": name,
            "customdata": lam_k,
        }
        fig.add_trace(
            go.Scatter(
                x=x, y=n_sel[m][o], showlegend=multi,
                hovertemplate=f"{name}<br>λ = %{{customdata:.3g}}<br>Topics selected: %{{y:.0f}}<extra></extra>",
                **common,
            ),
            row=1,
            col=1,
        )
        if crit_col:
            fig.add_trace(
                go.Scatter(
                    x=x, y=crit[m][o], showlegend=False,
                    hovertemplate=f"{name}<br>λ = %{{customdata:.3g}}<br>{_esc(crit_name)}: %{{y:.4f}}<extra></extra>",
                    **common,
                ),
                row=2,
                col=1,
            )

    fig.update_layout(**_base_layout(title, 480, legend=multi, extra_top=22))
    _style_axes(fig)
    fig.update_yaxes(title={"text": "Topics selected"}, rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title={"text": "Criterion"}, row=2, col=1)
    fig.update_xaxes(title={"text": "log10(λ)"}, row=2, col=1)
    if lam_star is not None and np.isfinite(lam_star) and lam_star > 0:
        xs = float(np.log10(lam_star))
        for xref, yref in (("x", "y domain"), ("x2", "y2 domain")):
            fig.add_shape(
                type="line", xref=xref, yref=yref, x0=xs, x1=xs, y0=0, y1=1,
                line={"color": INK_SECONDARY, "width": 1},
            )
        fig.add_annotation(
            x=xs, xref="x", y=1.0, yref="y domain", text=f"λ* = {lam_star:.3g}", showarrow=False,
            xanchor="left", yanchor="top", xshift=4,
            font={"family": FONT_FAMILY, "size": 10, "color": INK_SECONDARY},
        )
    return fig


# ---------------------------------------------------------------------------
# 9. Estimated vs true exposure (G.8 point 5; D63)
# ---------------------------------------------------------------------------
def exposure_scatter(
    estimate: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    linked: pd.DataFrame | None = None,
    title: str | None = None,
) -> go.Figure:
    """Estimated against true exposure for every topic-asset pair, with the 45-degree line.

    Parameters
    ----------
    estimate:
        Estimated exposures ``B_hat`` (topics x assets, standardised units).
    truth:
        True exposures ``B_true`` (same orientation); pairs are matched by
        labels and flattened.
    linked:
        Optional boolean frame (same orientation), ``True`` where the pair is
        linked in the design ``W``; linked pairs are blue, the others muted.
    title:
        Title (default "Estimated vs true exposure").
    """
    title = title if title is not None else "Estimated vs true exposure"
    est = _float_frame(estimate)
    tru = _float_frame(truth)
    if est.size == 0 or tru.size == 0:
        return _empty_figure("No exposures to compare", title=title)
    rows = est.index.intersection(tru.index, sort=False)
    cols = est.columns.intersection(tru.columns, sort=False)
    e = est.reindex(index=rows, columns=cols).to_numpy()
    t = tru.reindex(index=rows, columns=cols).to_numpy()
    ok = np.isfinite(e) & np.isfinite(t)
    if not ok.any():
        return _empty_figure("No exposures to compare", title=title)

    ri, ci = np.nonzero(ok)
    xv, yv = t[ri, ci], e[ri, ci]
    row_name = str(est.index.name or "Topic")
    col_name = str(est.columns.name or "Asset")
    custom = np.column_stack(
        [np.asarray(rows.astype(str), dtype=object)[ri], np.asarray(cols.astype(str), dtype=object)[ci]]
    )
    link = _bool_frame(linked, rows, cols)[ri, ci] if linked is not None else None
    n_pts = int(ri.size)
    trace_cls = go.Scattergl if n_pts > 3000 else go.Scatter
    size = 8 if n_pts <= 2000 else 5
    hover = (
        f"{_esc(row_name)}: %{{customdata[0]}}<br>{_esc(col_name)}: %{{customdata[1]}}"
        "<br>True: %{x:.3f}<br>Estimate: %{y:.3f}<extra></extra>"
    )
    groups: list[tuple[np.ndarray, str, str, float]]
    if link is None:
        groups = [(np.ones(n_pts, dtype=bool), "Topic-asset pairs", BLUE, 0.75)]
    else:
        groups = [(~link, "Not linked in the design", INK_MUTED, 0.45), (link, "Linked in the design", BLUE, 0.85)]
    fig = go.Figure()
    for m, name, color, alpha in groups:
        if not m.any():
            continue
        marker: dict[str, Any] = {"size": size, "color": color, "opacity": alpha}
        if trace_cls is go.Scatter:
            marker["line"] = {"width": 1, "color": SURFACE}
        fig.add_trace(
            trace_cls(x=xv[m], y=yv[m], mode="markers", marker=marker, name=name, customdata=custom[m],
                      hovertemplate=hover)
        )
    legend = len(fig.data) >= 2
    lo, hi = _numeric_range(np.concatenate([xv, yv]), pad_share=0.05)
    fig.add_shape(type="line", xref="x", yref="y", x0=lo, y0=lo, x1=hi, y1=hi, line={"color": INK_MUTED, "width": 1},
                  layer="below")
    fig.add_annotation(x=hi, y=hi, text="estimate = truth", showarrow=False, xanchor="right", yanchor="bottom",
                       font={"family": FONT_FAMILY, "size": 10, "color": INK_SECONDARY})
    subtitle = f"{n_pts} topic-asset pairs"
    fig.update_layout(**_base_layout(title, 480, subtitle=subtitle, legend=legend))
    _style_axes(fig)
    fig.update_xaxes(range=[lo, hi], title={"text": "True exposure (standardised units)"}, **_ZERO_LINE)
    fig.update_yaxes(range=[lo, hi], title={"text": "Estimated exposure"}, scaleanchor="x", scaleratio=1, **_ZERO_LINE)
    return fig
