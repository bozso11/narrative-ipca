"""Plotly figure builders for the topic-sensitivity lab (DESIGN.md G.9, G.13, G.16; D52, D67-D70, D90).

Pure functions from pandas objects to ``plotly.graph_objects.Figure``. No
Streamlit imports: the dashboard (``dashboard/app.py``) only displays what
these functions return. Every builder returns an annotated empty figure
instead of raising when its input is ``None``, empty or entirely missing.

Section 11 holds the generic builders of the BKS trace page (G.16, D90):
stacked line panels over a date or numeric axis (:func:`line_panels`), a
signed matrix (:func:`matrix_heatmap`), the reference ladder
(:func:`ladder_chart`), a scatter against the 45-degree line
(:func:`identity_scatter`), grouped bars (:func:`grouped_bars`), the lambda
path with its noise band (:func:`lambda_trace_chart`) and the Gamma row norms
along the path (:func:`coefficient_path_chart`).

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
* Size limits of the trace builders keep a 500 x 500 run displayable, and
  each is noted in the subtitle: :func:`matrix_heatmap` shows at most
  ``max_rows`` x ``max_cols`` cells, :func:`grouped_bars` at most
  :data:`MAX_BAR_ROWS` rows unless ``top_n`` is given, and
  :func:`identity_scatter` draws at most :data:`MAX_SCATTER_POINTS` points
  (its slope and correlation use every point).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
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
    "METHOD_SYMBOLS",
    "REFERENCE_SYMBOL",
    "method_styles",
    "method_r2_dots",
    "method_sweep_lines",
    "MAX_BAR_ROWS",
    "MAX_SCATTER_POINTS",
    "line_panels",
    "matrix_heatmap",
    "ladder_chart",
    "identity_scatter",
    "grouped_bars",
    "lambda_trace_chart",
    "coefficient_path_chart",
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
    elif subtitle:
        y += _SUBTITLE_PX  # a subtitle without a title takes the title's place (notes of the trace builders)
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
        Optional second title line in secondary ink (notes, caveats); without
        a title it is drawn alone in the title's place.
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
    if not text and subtitle:
        # no title: the subtitle alone, in the subtitle's style, where the title would be
        layout["title"] = {
            "text": _esc(subtitle),
            "x": 0.0,
            "xref": "container",
            "xanchor": "left",
            "y": 1.0 - _TITLE_TOP_PX / height,
            "yref": "container",
            "yanchor": "top",
            "pad": {"l": 12},
            "font": {"family": FONT_FAMILY, "size": 12, "color": INK_SECONDARY},
        }
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
        legend_top = _TITLE_TOP_PX + (_TITLE_PX if title else 0) + (_SUBTITLE_PX if subtitle else 0) + 2
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


def _cell_text_trace(
    z: np.ndarray, zmax: float, *, font_size: int, name: str, decimals: int = 2
) -> go.Scatter | None:
    """Text marks for the finite cells of ``z``: ``decimals`` places, white on strong cells, ink otherwise."""
    rows, cols = np.nonzero(np.isfinite(z))
    if rows.size == 0:
        return None
    vals = z[rows, cols]
    colors = np.where(np.abs(vals) > STRONG_CELL_SHARE * zmax, WHITE, INK)
    return go.Scatter(
        x=cols,
        y=rows,
        mode="text",
        text=[_fmt(v, decimals) for v in vals],
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
    name: str = "Estimator",
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
    name:
        Legend and hover name of the bars (for example the method's label on
        the Compare methods tab).
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
            name=_esc(name),
            hovertext=[f"{_esc(n)}<br>{_esc(name)} R²: {_fmt(v, 3)}" for n, v in zip(names, e.to_numpy())],
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
                name="Oracle (true sensitivities)",
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
        Title (default "Estimated vs true sensitivity").
    """
    title = title if title is not None else "Estimated vs true sensitivity"
    est = _float_frame(estimate)
    tru = _float_frame(truth)
    if est.size == 0 or tru.size == 0:
        return _empty_figure("No sensitivities to compare", title=title)
    rows = est.index.intersection(tru.index, sort=False)
    cols = est.columns.intersection(tru.columns, sort=False)
    e = est.reindex(index=rows, columns=cols).to_numpy()
    t = tru.reindex(index=rows, columns=cols).to_numpy()
    ok = np.isfinite(e) & np.isfinite(t)
    if not ok.any():
        return _empty_figure("No sensitivities to compare", title=title)

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
    fig.update_xaxes(range=[lo, hi], title={"text": "True sensitivity (standardised units)"}, **_ZERO_LINE)
    fig.update_yaxes(range=[lo, hi], title={"text": "Estimated sensitivity"}, scaleanchor="x", scaleratio=1,
                     **_ZERO_LINE)
    return fig


# ---------------------------------------------------------------------------
# 10. Method comparison (G.15; G.9 tab 4)
# ---------------------------------------------------------------------------
#: Marker symbols of the method charts in slot order: the secondary encoding next to the colour,
#: because four or five methods can sit close together on one asset.
METHOD_SYMBOLS: tuple[str, ...] = (
    "circle",
    "diamond",
    "square",
    "triangle-up",
    "x",
    "triangle-down",
    "star",
    "hexagon",
)
#: Marker of the reference method (the oracle) in :func:`method_r2_dots`: an ink tick.
REFERENCE_SYMBOL = "line-ns-open"


def method_styles(
    methods: Iterable[Any], reference: Any = "oracle", slots: Iterable[Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Colour, marker symbol and line dash per method, shared by the method charts.

    Methods other than ``reference`` take the categorical slots
    (:data:`CATEGORICAL`, :data:`METHOD_SYMBOLS`) in the order of ``slots``
    (then any method not in it, in the order given), or in the order given
    when ``slots`` is ``None``. Passing the full method list as ``slots``
    (the dashboard passes :data:`.compare.METHODS`) gives every method a
    fixed colour and symbol, whatever the selection. The slots are never
    cycled, so methods past the eighth get no style and are not drawn. The
    reference method is drawn in ink (tick marker, dashed line).

    Returns
    -------
    dict
        ``str(method) -> {"color", "symbol", "dash", "reference"}`` for the
        methods in ``methods`` only.
    """
    ref = None if reference is None else str(reference)
    wanted = list(dict.fromkeys(str(x) for x in methods))
    order = list(dict.fromkeys([str(x) for x in (slots or ())] + wanted))
    out: dict[str, dict[str, Any]] = {}
    slot = 0
    for m in order:
        if m == ref:
            out[m] = {"color": INK, "symbol": REFERENCE_SYMBOL, "dash": "dash", "reference": True}
            continue
        if slot >= len(CATEGORICAL):
            if m in wanted:
                logger.warning("method charts: more than %d methods; %r is not drawn", len(CATEGORICAL), m)
            continue
        out[m] = {"color": CATEGORICAL[slot], "symbol": METHOD_SYMBOLS[slot], "dash": "solid", "reference": False}
        slot += 1
    return {m: out[m] for m in wanted if m in out}


def method_r2_dots(
    r2: pd.DataFrame,
    *,
    labels: Mapping[Any, Any] | pd.Series | None = None,
    method_labels: Mapping[Any, Any] | None = None,
    reference: Any = "oracle",
    title: str | None = None,
    clip: float = -1.0,
    slots: Iterable[Any] | None = None,
) -> go.Figure:
    """OOS R2 per asset for several methods in one forecast window: a dot plot (G.15).

    Assets are rows, sorted by the reference method's R2 with the largest on
    top (by the first column when the reference is absent). Each method is
    one marker series in its slot of :func:`method_styles`; the reference
    (the oracle) is an ink tick drawn on top. Values below ``clip`` are drawn
    at ``clip`` and counted in a note under the title; the hover shows the
    unclipped value.

    Parameters
    ----------
    r2:
        Assets x methods: per-asset uncentered OOS R2
        (:attr:`.compare.ComparisonResult.r2`). Column order sets the colour
        slots unless ``slots`` is given.
    labels:
        Display names by asset id.
    method_labels:
        Display names by method (legend and hover); default the method id.
    reference:
        The reference method's column; ``None`` for no reference.
    title:
        Title (default "Out-of-sample R² per asset by method").
    clip:
        Lower display limit of the x axis.
    slots:
        Full method list that fixes each method's colour slot
        (:func:`method_styles`).
    """
    title = title if title is not None else "Out-of-sample R² per asset by method"
    data = _float_frame(r2)
    if data.size == 0 or _no_data(data):
        return _empty_figure("No R² values to show", title=title)
    data.columns = pd.Index([str(c) for c in data.columns])
    styles = method_styles(data.columns, reference, slots)
    cols = [c for c in data.columns if c in styles]
    ref = str(reference) if reference is not None and str(reference) in cols else None
    key = data[ref if ref is not None else cols[0]].to_numpy()
    order = np.lexsort((np.arange(len(key)), np.where(np.isfinite(key), -key, np.inf)))
    data = data.iloc[order]
    assets = data.index
    pos = np.arange(len(assets), dtype=float)
    names = [_label(labels, a) for a in assets]
    mnames = {c: _label(method_labels, c) for c in cols}

    vals = data[cols].to_numpy()
    n_clipped = int(np.sum(np.isfinite(vals) & (vals < clip)))
    subtitle = (
        f"{n_clipped} value{'s' if n_clipped != 1 else ''} below {clip:.0%} drawn at {clip:.0%} (hover shows the value)"
        if n_clipped
        else None
    )

    fig = go.Figure()
    shown: list[np.ndarray] = []
    draw = [c for c in cols if c != ref] + ([ref] if ref is not None else [])  # the reference on top
    for c in draw:
        v = data[c].to_numpy()
        if not np.isfinite(v).any():
            continue
        x = np.where(np.isfinite(v), np.maximum(v, clip), np.nan)
        shown.append(x)
        style = styles[c]
        if style["reference"]:
            marker = {"symbol": style["symbol"], "size": 16, "color": INK, "line": {"width": 2, "color": INK}}
        else:
            marker = {"symbol": style["symbol"], "size": 9, "color": style["color"],
                      "line": {"width": 1, "color": SURFACE}}
        fig.add_trace(
            go.Scatter(
                x=x,
                y=pos,
                mode="markers",
                marker=marker,
                name=_esc(mnames[c]),
                hovertext=[f"{_esc(n)}<br>{_esc(mnames[c])}: R² {_fmt(val, 1, scale=100.0, suffix='%')}"
                           for n, val in zip(names, v)],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )
    if not fig.data:
        return _empty_figure("No R² values to show", title=title)

    legend = len(fig.data) >= 2
    height = _header_px(title, subtitle, legend) + 22 * len(assets) + 56
    fig.update_layout(**_base_layout(title, height, subtitle=subtitle, legend=legend))
    _style_axes(fig)
    lo, hi = _numeric_range(np.concatenate(shown), pad_share=0.05)
    fig.update_xaxes(
        range=[max(lo, clip - 0.05), hi],
        tickformat=".0%",
        title={"text": "Out-of-sample R² (share of return variation explained)"},
        **_ZERO_LINE,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=pos.tolist(),
        ticktext=[_esc(_truncate(n, 34)) for n in names],
        range=[len(assets) - 0.4, -0.6],
        tickfont={"size": 11, "color": INK},
    )
    return fig


def method_sweep_lines(
    r2_sweep_long: pd.DataFrame,
    *,
    method_labels: Mapping[Any, Any] | None = None,
    reference: Any = "oracle",
    title: str | None = None,
    empty_message: str = "No forecast windows to show",
    clip: float = -1.0,
    slots: Iterable[Any] | None = None,
) -> go.Figure:
    """Median OOS R2 over consecutive forecast windows, one line per method (G.8 point 6; G.15).

    Colours and marker symbols follow :func:`method_styles` (in the order of
    ``slots``, else the order the methods first appear in the frame); the
    reference (the oracle) is a dashed ink line. One y axis: every line is
    the same measure. Values below ``clip`` are drawn at ``clip`` and counted
    in the subtitle, as in :func:`method_r2_dots`, so one method that fails
    on a short window does not flatten the others; the hover shows the
    unclipped value.

    Parameters
    ----------
    r2_sweep_long:
        Long frame with columns ``start`` (window start), ``method`` and
        ``median_r2`` (the cross-asset median OOS R2 of the window), as
        :attr:`.compare.ComparisonResult.r2_sweep`.
    method_labels:
        Display names by method; default the method id.
    reference:
        The reference method; ``None`` for no reference.
    title:
        Title (default "Median out-of-sample R² across forecast windows").
    empty_message:
        Text of the empty figure (for example why no complete window fits).
    clip:
        Lower display limit of the y axis.
    slots:
        Full method list that fixes each method's colour slot
        (:func:`method_styles`).
    """
    title = title if title is not None else "Median out-of-sample R² across forecast windows"
    need = {"start", "method", "median_r2"}
    if (r2_sweep_long is None or not isinstance(r2_sweep_long, pd.DataFrame) or r2_sweep_long.empty
            or not need <= set(r2_sweep_long.columns)):
        return _empty_figure(empty_message, title=title)
    frame = pd.DataFrame({
        "start": pd.to_datetime(r2_sweep_long["start"], errors="coerce").to_numpy(),
        "method": r2_sweep_long["method"].astype(str).to_numpy(),
        "median_r2": _float_series(r2_sweep_long["median_r2"]).to_numpy(),
    })
    if _no_data(frame["median_r2"]):
        return _empty_figure(empty_message, title=title)
    methods = list(dict.fromkeys(frame["method"]))
    styles = method_styles(methods, reference, slots)
    ref = str(reference) if reference is not None else None
    draw = [m for m in methods if m in styles and m != ref] + ([ref] if ref in styles else [])
    n_windows = int(frame["start"].nunique())
    mode = "lines+markers" if n_windows <= 40 else "lines"

    fig = go.Figure()
    shown: list[np.ndarray] = []
    n_clipped = 0
    for m in draw:
        part = frame[frame["method"] == m].sort_values("start", kind="stable")
        y = part["median_r2"].to_numpy()
        if not np.isfinite(y).any():
            continue
        n_clipped += int(np.sum(np.isfinite(y) & (y < clip)))
        y_draw = np.where(np.isfinite(y), np.maximum(y, clip), np.nan)
        shown.append(y_draw)
        style = styles[m]
        name = _esc(_label(method_labels, m))
        fig.add_trace(
            go.Scatter(
                x=part["start"],
                y=y_draw,
                customdata=y,
                mode=mode,
                line={"color": style["color"], "width": 2, "dash": style["dash"]},
                marker={"size": 8, "symbol": "circle" if style["reference"] else style["symbol"],
                        "color": style["color"], "line": {"width": 1.5, "color": SURFACE}},
                name=name,
                hovertemplate=f"{name}: %{{customdata:.1%}}<extra></extra>",
            )
        )
    if not fig.data:
        return _empty_figure(empty_message, title=title)
    legend = len(fig.data) >= 2
    subtitle = f"{n_windows} window{'s' if n_windows != 1 else ''}; training fits frozen"
    if n_clipped:
        subtitle += (f"; {n_clipped} value{'s' if n_clipped != 1 else ''} below {clip:.0%} drawn at {clip:.0%} "
                     "(hover shows the value)")
    fig.update_layout(**{**_base_layout(title, 400, subtitle=subtitle, legend=legend), "hovermode": "x unified"})
    _style_axes(fig)
    lo, hi = _numeric_range(np.concatenate(shown), pad_share=0.05)
    fig.update_yaxes(tickformat=".0%", title={"text": "Median out-of-sample R² over assets"},
                     range=[max(lo, clip - 0.05), hi], **_ZERO_LINE)
    fig.update_xaxes(title={"text": "Window start"}, showspikes=True, spikecolor=BASELINE, spikethickness=1,
                     spikedash="solid", spikemode="across")
    return fig


# ---------------------------------------------------------------------------
# 11. BKS trace page (G.16; D90)
# ---------------------------------------------------------------------------
#: Most bar rows of :func:`grouped_bars` when the caller gives no ``top_n`` (the largest are kept).
MAX_BAR_ROWS = 120
#: Most points drawn by :func:`identity_scatter`; its slope and correlation use every point.
MAX_SCATTER_POINTS = 20_000
#: Most stacked panels of :func:`line_panels`.
MAX_LINE_PANELS = 3
#: Row label length of :func:`ladder_chart` (its labels are whole phrases; the hover shows the full name).
LADDER_LABEL_CHARS = 60

_DASHES = ("solid", "dash", "dot", "dashdot", "longdash")
_LINE_MODES = ("lines", "markers", "lines+markers")
#: Opacity of the shaded windows of :func:`line_panels`, alternating so adjacent windows differ.
_SHADE_OPACITY = (0.5, 1.0)
#: Plot height in pixels per panel of :func:`line_panels`, by number of panels.
_PANEL_PX = {1: 300, 2: 190, 3: 150}
_SUBPLOT_TITLE_PX = 22
_NOTE_FONT = {"family": FONT_FAMILY, "size": 10, "color": INK_SECONDARY}
_NOTE_BG = "rgba(252,252,251,0.85)"  # SURFACE, slightly transparent, behind notes drawn over data
_NOTE_ROW_PX = 15
# Widths assumed when estimating wrapped legend rows and overlapping notes (the container sets the real width).
_LEGEND_WIDTH_PX = 760
_LEGEND_ROW_PX = 20
_PLOT_WIDTH_PX = 700


def _rgba(color: str, alpha: float) -> str:
    """A ``#rrggbb`` colour as ``rgba(r,g,b,alpha)``; other colour strings are returned unchanged."""
    c = str(color).lstrip("#")
    if len(c) != 6:
        return str(color)
    try:
        r, g, b = (int(c[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return str(color)
    return f"rgba({r},{g},{b},{alpha:g})"


def _x_position(value: Any) -> Any:
    """Shape x position: a Plotly date string for dates, a float for numbers, strings unchanged."""
    if isinstance(value, (pd.Timestamp, np.datetime64, date)):
        return _date_str(value)
    if isinstance(value, str):
        return value
    return _to_float(value)


def _annotation_x(value: Any, log_axis: bool) -> Any:
    """Annotation x position; on a log axis Plotly places annotations at ``log10(x)`` (shapes at ``x``)."""
    if log_axis:
        v = _to_float(value)
        return math.log10(v) if v > 0 else float("nan")
    return _x_position(value)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, pd.Timestamp, np.datetime64, date)):
        return pd.isna(value)
    return not np.isfinite(_to_float(value))


def _category_name(mapping: Mapping[Any, Any] | pd.Series | None, key: Any) -> str:
    """Display name of a category; timestamps without a mapped name read as dates."""
    if mapping is not None:
        text = mapping.get(key)
        if text is not None and not (isinstance(text, float) and math.isnan(text)):
            return str(text)
    if isinstance(key, (pd.Timestamp, np.datetime64, date)):
        return _date_str(key)
    return str(key)


def _point_name(labels: Mapping[Any, Any] | pd.Series | None, key: Any) -> str:
    """Name of a point; a tuple key (MultiIndex) joins the names of its parts with a middle dot."""
    if isinstance(key, tuple):
        if labels is not None:
            text = labels.get(key)
            if text is not None and not (isinstance(text, float) and math.isnan(text)):
                return str(text)
        return " · ".join(_category_name(labels, part) for part in key)
    return _category_name(labels, key)


def _key_set(keys: Any) -> set[str]:
    """``keys`` as a set of strings: one key (a string or a scalar) or an iterable of keys (list, tuple,
    set, Index); ``None`` for none. A single MultiIndex key (a tuple) goes in a list: ``[("T1", "A2")]``."""
    if keys is None:
        return set()
    if isinstance(keys, (str, bytes)) or not isinstance(keys, Iterable):
        return {str(keys)}
    return {str(k) for k in keys}


def _in_keys(key: Any, keys: set[str]) -> bool:
    """``key`` is in ``keys``; a tuple key (MultiIndex) also matches when any of its parts is."""
    if str(key) in keys:
        return True
    return isinstance(key, tuple) and any(str(part) in keys for part in key)


def _with_note(subtitle: str | None, note: str | None) -> str | None:
    """``subtitle`` and ``note`` joined by a semicolon (either may be empty)."""
    parts = [p for p in (subtitle, note) if p]
    return "; ".join(parts) if parts else None


def _fmt_sig(value: float, digits: int = 3) -> str:
    """``value`` with ``digits`` significant digits (trailing zeros kept, thousands grouped); empty when missing."""
    if value is None or not np.isfinite(value):
        return ""
    v = float(value) + 0.0
    if abs(v) >= 10 ** digits:
        return f"{v:,.0f}"
    return f"{v:#.{digits}g}".rstrip(".")


def _auto_decimals(max_abs: float) -> int:
    """Decimals for value labels: 2 from about 0.1 up, then one more per decade, at most 4."""
    if not np.isfinite(max_abs) or max_abs <= 0:
        return 2
    return int(np.clip(1 - math.floor(math.log10(max_abs)), 2, 4))


def _legend_extra_px(fig: go.Figure) -> int:
    """Pixels for the legend rows past the first, estimated from the legend names at an assumed width."""
    rows, used = 1, 0.0
    for t in fig.data:
        if t.showlegend is False or not t.name:
            continue
        w = 46.0 + 6.2 * len(str(t.name))
        if used and used + w > _LEGEND_WIDTH_PX:
            rows += 1
            used = 0.0
        used += w
    return (rows - 1) * _LEGEND_ROW_PX


def _note_rows(spans: Sequence[tuple[float, float]]) -> list[int]:
    """Row of each note (0 = top) so that notes in one row do not overlap; spans are shares of the plot width."""
    rows: list[list[tuple[float, float]]] = []
    out: list[int] = []
    for a, b in spans:
        for r, taken in enumerate(rows):
            if all(b <= c or a >= d for c, d in taken):
                taken.append((a, b))
                out.append(r)
                break
        else:
            rows.append([(a, b)])
            out.append(len(rows) - 1)
    return out


def _bar_range(values: Iterable[float], *, pad_share: float) -> tuple[float, float]:
    """Value-axis range of bars: :func:`_numeric_range`, starting at 0 when every value has one sign."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    lo, hi = _numeric_range(arr, pad_share=pad_share)
    if arr.size and arr.min() >= 0:
        lo = 0.0
    if arr.size and arr.max() <= 0:
        hi = 0.0
    return lo, hi


def _panel_refs(i: int) -> tuple[str, str]:
    """``(xref, yref)`` axis names of subplot ``i`` (1-based) of ``make_subplots``."""
    return ("x", "y") if i == 1 else (f"x{i}", f"y{i}")


def _align_subplot_titles(fig: go.Figure) -> None:
    """Left-align the subplot titles of ``make_subplots`` at the left edge of their panel, in secondary ink.

    Call it right after ``make_subplots``, before any other annotation.
    """
    layout = fig.layout.to_plotly_json()
    domains = [tuple(v["domain"]) for k, v in layout.items() if k.startswith("xaxis") and "domain" in v]
    for ann in fig.layout.annotations:
        x = float(ann.x)
        start = next((d0 for d0, d1 in domains if abs((d0 + d1) / 2.0 - x) < 1e-6), 0.0)
        ann.update(x=start, xanchor="left", font={"family": FONT_FAMILY, "size": 12, "color": INK_SECONDARY})


def _positive_index(s: pd.Series) -> pd.Series:
    """Rows of ``s`` whose index is a positive number (for a log x axis)."""
    pos = pd.to_numeric(pd.Series(s.index, index=s.index), errors="coerce").to_numpy(dtype=float)
    return s[np.isfinite(pos) & (pos > 0)]


def _sorted_index(s: pd.Series) -> pd.Series:
    if s.index.is_monotonic_increasing:
        return s
    try:
        return s.sort_index(kind="stable")
    except TypeError:
        return s


def _lambda_frame(obj: Any) -> pd.DataFrame:
    """Float frame indexed by a positive ``lam`` (from a ``lam`` column or the index), sorted ascending."""
    if obj is None or not isinstance(obj, (pd.DataFrame, pd.Series)) or len(obj) == 0:
        return pd.DataFrame(dtype=float)
    df = obj.to_frame() if isinstance(obj, pd.Series) else obj
    if "lam" in df.columns:
        lam = pd.to_numeric(df["lam"], errors="coerce").to_numpy(dtype=float)
        df = df.drop(columns="lam")
    else:
        lam = pd.to_numeric(pd.Series(df.index), errors="coerce").to_numpy(dtype=float)
    num = _float_frame(df)
    num.index = pd.Index(lam, name="lam")
    num = num[np.isfinite(lam) & (lam > 0)]
    return num.sort_index(kind="stable")


def _null_quantiles(band: Any) -> tuple[float, float, float] | None:
    """``(q05, q50, q95)`` as floats when all three are finite and ordered; ``None`` otherwise."""
    if band is None:
        return None
    try:
        vals = [_to_float(v) for v in band]
    except TypeError:
        return None
    if len(vals) != 3 or not all(np.isfinite(vals)) or not vals[0] <= vals[1] <= vals[2]:
        return None
    return vals[0], vals[1], vals[2]


def _nearest_lambda(lams: np.ndarray, lam: float | None, *, rel_tol: float = 0.05) -> int | None:
    """Position of the grid value nearest ``lam`` in log space; ``None`` when none is within ``rel_tol``."""
    if lam is None or lams.size == 0:
        return None
    v = _to_float(lam)
    if not (np.isfinite(v) and v > 0):
        return None
    gap = np.abs(np.log(lams) - math.log(v))
    i = int(np.argmin(gap))
    return i if gap[i] <= math.log1p(rel_tol) else None


def _vertical_lines(
    fig: go.Figure, x: Any, n_panels: int, *, dash: str = "dash", color: str = INK_SECONDARY
) -> None:
    """A vertical line at ``x`` across every panel (subplots ``1..n_panels``)."""
    xv = _x_position(x)
    for i in range(1, n_panels + 1):
        xref, yref = _panel_refs(i)
        fig.add_shape(
            type="line", xref=xref, yref=f"{yref} domain", x0=xv, x1=xv, y0=0, y1=1,
            line={"color": color, "width": 1, "dash": dash},
        )


def _top_note(
    fig: go.Figure, x: Any, text: str, *, log_axis: bool = False, row: int = 0, right: bool = False
) -> None:
    """A small secondary-ink note at the top of the first panel in note row ``row``.

    It starts at ``x`` (``right=False``) or ends there (``right=True``, for
    notes near the right edge of the plot).
    """
    xa = _annotation_x(x, log_axis)
    if _missing(xa):
        return
    fig.add_annotation(
        x=xa, xref="x", y=1.0, yref="y domain", text=_esc(text), showarrow=False,
        xanchor="right" if right else "left", yanchor="top", xshift=-4 if right else 4,
        yshift=-2 - _NOTE_ROW_PX * row, font=_NOTE_FONT, bgcolor=_NOTE_BG,
    )


def _x_number(value: Any, *, is_date: bool, log_axis: bool) -> float:
    """``value`` on a linear scale of the x axis (nanoseconds for dates, ``log10`` on a log axis); NaN if unknown."""
    try:
        if is_date:
            return float(pd.Timestamp(value).value)
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if log_axis:
        return math.log10(v) if v > 0 else float("nan")
    return v


def line_panels(
    panels: Sequence[Mapping[str, Any]] | None,
    *,
    shade: Iterable[tuple[Any, Any, str]] = (),
    markers: Iterable[tuple[Any, str]] = (),
    x_title: str | None = None,
    x_log: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """One to three stacked line panels sharing the x axis, with shaded windows and dashed markers (G.16).

    The generic time-series view of the BKS trace page: daily attention and
    returns, the divisor, the shocks, an instrument over the weeks, a kernel
    profile, the fitted factors. Each panel has its own y axis (no secondary
    axis); the x axis is shared, a date axis when any series has a
    DatetimeIndex, else numeric (or logarithmic with ``x_log``). Missing
    values break the line (gaps, not interpolation).

    Parameters
    ----------
    panels:
        Top to bottom, at most :data:`MAX_LINE_PANELS`; each a mapping with

        * ``"series"``: ``Mapping[name, pd.Series]``, drawn at each Series'
          own index (the name is the legend and hover name);
        * ``"y_title"``: the panel's y axis title;
        * ``"styles"`` (optional): ``Mapping[name, dict]`` with any of
          ``color``, ``dash`` (``"solid"``, ``"dash"``, ``"dot"``),
          ``mode`` (``"lines"``, ``"markers"``, ``"lines+markers"``;
          default lines with markers up to 40 points), ``width``,
          ``fill`` (``"tozeroy"``) and ``opacity``;
        * ``"zero_line"`` (optional): draw the zero line;
        * ``"title"`` (optional): a subplot title, left-aligned above the panel;
        * ``"tickformat"`` / ``"hoverformat"`` (optional): d3 formats of the
          y values, for example ``".1%"`` for shares in percent (the hover
          then defaults to ``".2%"``, else ``".4g"``).

        Series without a finite value are skipped and panels left without a
        series are dropped. Colours follow :data:`CATEGORICAL` in order of
        first appearance across panels; the same name keeps its colour (and
        one legend entry) in every panel. Names past the eighth slot without
        an explicit colour are drawn in muted ink (never cycled).
    shade:
        ``(start, end, label)`` windows (for example training and forecast)
        drawn as light rectangles on every panel, the label at the top of
        the first panel. Adjacent windows alternate two tones.
    markers:
        ``(x, label)`` vertical dashed lines on every panel (for example the
        instrument week or the training start), labelled at the top of the
        first panel. Window and marker labels that would overlap (at an
        assumed plot width of 700 px) go to separate rows; a label that would
        run past the right edge ends at its x instead of starting there.
    x_title:
        Title of the bottom x axis.
    x_log:
        Logarithmic x axis; points with a non-positive x are dropped.
    title, subtitle:
        Figure title (default none) and a second line.
    height:
        Figure height in pixels (default from the number of panels).

    Returns
    -------
    go.Figure
        One trace per series and panel (``Scattergl`` above 3,000 points),
        unified hover along x.
    """
    items = list(panels or [])
    if len(items) > MAX_LINE_PANELS:
        logger.warning("line_panels: %d panels given; showing the first %d", len(items), MAX_LINE_PANELS)
        items = items[:MAX_LINE_PANELS]
    prepared: list[tuple[Mapping[str, Any], list[tuple[str, dict[str, Any], pd.Series]]]] = []
    for panel in items:
        series = panel.get("series")
        if series is None:
            series = {}
        elif isinstance(series, pd.Series):
            series = {series.name if series.name is not None else "value": series}
        styles = panel.get("styles") or {}
        kept: list[tuple[str, dict[str, Any], pd.Series]] = []
        for key, obj in dict(series).items():
            s = _float_series(obj)
            if not isinstance(s.index, pd.DatetimeIndex) and s.index.inferred_type in ("date", "datetime64"):
                s = s.set_axis(pd.DatetimeIndex(s.index))  # datetime.date objects: a date axis all the same
            if x_log and not s.empty:
                s = _positive_index(s)
            if _no_data(s):
                continue
            style = styles.get(key, styles.get(str(key)))
            kept.append((str(key), dict(style or {}), _sorted_index(s)))
        if kept:
            prepared.append((panel, kept))
    if not prepared:
        return _empty_figure("No series to show", title=title)

    names = list(dict.fromkeys(name for _, kept in prepared for name, _, _ in kept))
    explicit: dict[str, str] = {}
    for _, kept in prepared:
        for name, style, _ in kept:
            c = style.get("color")
            if c and name not in explicit:
                explicit[name] = str(c)
    colors: dict[str, str] = {}
    slot = 0
    for name in names:
        if name in explicit:
            colors[name] = explicit[name]
        elif slot < len(CATEGORICAL):
            colors[name] = CATEGORICAL[slot]
            slot += 1
        else:
            logger.warning("line_panels: more than %d series without a colour; %r is drawn muted",
                           len(CATEGORICAL), name)
            colors[name] = INK_MUTED
    legend = len(names) >= 2
    is_date = any(isinstance(s.index, pd.DatetimeIndex) for _, kept in prepared for _, _, s in kept)

    n = len(prepared)
    titles = [str(panel.get("title") or "") for panel, _ in prepared]
    has_titles = any(titles)
    gap_px = 52 if has_titles else 30
    plot_px = n * _PANEL_PX[n] + (n - 1) * gap_px
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=gap_px / plot_px if n > 1 else 0.0,
        subplot_titles=[_esc(t) for t in titles] if has_titles else None,
    )
    if has_titles:
        _align_subplot_titles(fig)

    seen: set[str] = set()
    for i, (panel, kept) in enumerate(prepared, start=1):
        tickformat = str(panel.get("tickformat") or "")
        fmt = str(panel.get("hoverformat") or (".2%" if "%" in tickformat else ".4g"))
        for name, style, s in kept:
            color = str(style.get("color") or colors[name])
            n_pts = len(s)
            mode = style.get("mode") if style.get("mode") in _LINE_MODES else (
                "lines+markers" if n_pts <= 40 else "lines")
            dash = style.get("dash") if style.get("dash") in _DASHES else "solid"
            width = _to_float(style.get("width", 1.5 if n_pts <= 1000 else 1.0))
            opacity = _to_float(style.get("opacity", 1.0))
            trace_cls = go.Scattergl if n_pts > 3000 else go.Scatter
            marker: dict[str, Any] = {"size": 6, "color": color}
            if trace_cls is go.Scatter:
                marker["line"] = {"width": 1, "color": SURFACE}
            extra: dict[str, Any] = {}
            if style.get("fill") == "tozeroy":
                extra = {"fill": "tozeroy", "fillcolor": _rgba(color, 0.18)}
            x = s.index if isinstance(s.index, pd.DatetimeIndex) else s.index.to_numpy()
            fig.add_trace(
                trace_cls(
                    x=x,
                    y=s.to_numpy(),
                    mode=mode,
                    line={"color": color, "width": width if np.isfinite(width) else 1.5, "dash": dash},
                    marker=marker,
                    opacity=opacity if np.isfinite(opacity) else 1.0,
                    name=_esc(_truncate(name, 40)),
                    legendgroup=name,
                    showlegend=legend and name not in seen,
                    hovertemplate=f"{_esc(name)}: %{{y:{fmt}}}<extra></extra>",
                    **extra,
                ),
                row=i,
                col=1,
            )
            seen.add(name)

    extra_top = (_SUBPLOT_TITLE_PX if titles[0] else 0) + (_legend_extra_px(fig) if legend else 0)
    if height is None:
        height = _header_px(title, subtitle, legend) + extra_top + plot_px + 72
    fig.update_layout(**{**_base_layout(title, height, subtitle=subtitle, legend=legend, extra_top=extra_top),
                         "hovermode": "x unified"})
    _style_axes(fig)
    for i, (panel, _) in enumerate(prepared, start=1):
        y_axis: dict[str, Any] = {"title": {"text": _esc(panel.get("y_title") or "")}}
        if panel.get("zero_line"):
            y_axis.update(_ZERO_LINE)
        if panel.get("tickformat"):
            y_axis["tickformat"] = str(panel["tickformat"])
        fig.update_yaxes(**y_axis, row=i, col=1)
    fig.update_xaxes(showspikes=True, spikecolor=BASELINE, spikethickness=1, spikedash="solid", spikemode="across",
                     hoverformat="%Y-%m-%d" if is_date else ".4g")
    if x_log:
        fig.update_xaxes(type="log")
    if x_title:
        fig.update_xaxes(title={"text": _esc(x_title)}, row=n, col=1)

    # Windows and markers on every panel; their labels in rows at the top of the first panel, placed so
    # that labels in one row do not overlap (at an assumed plot width) and none runs past the right edge.
    windows = [(tuple(item) + (None, None, None))[:3] for item in (shade or ())]
    windows = [w for w in windows if not (_missing(w[0]) or _missing(w[1]))]
    lines = [(tuple(item) + (None, None))[:2] for item in (markers or ())]
    lines = [m for m in lines if not _missing(m[0])]
    xs = [_x_number(v, is_date=is_date, log_axis=x_log) for _, kept in prepared for _, _, s in kept
          for v in (s.index[0], s.index[-1])]
    xs += [_x_number(v, is_date=is_date, log_axis=x_log) for w in windows for v in w[:2]]
    xs += [_x_number(m[0], is_date=is_date, log_axis=x_log) for m in lines]
    finite = [v for v in xs if np.isfinite(v)]
    lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
    span = hi - lo if hi > lo else 1.0

    def share(v: Any) -> float:
        return (_x_number(v, is_date=is_date, log_axis=x_log) - lo) / span

    notes: list[tuple[Any, str, bool, tuple[float, float]]] = []  # (x, label, ends at x, span)

    def add_note(start: Any, end: Any, label: str) -> None:
        """Queue a label starting at ``start``, or ending at ``end`` when it would run past the right edge."""
        w = (6.0 * len(label) + 12) / _PLOT_WIDTH_PX
        a = share(start)
        if not np.isfinite(a):
            notes.append((start, label, False, (0.0, 1.0)))
        elif a + w > 1.0:
            b = share(end)
            notes.append((end, label, True, (b - w, b)))
        else:
            notes.append((start, label, False, (a, a + w)))

    for j, (start, end, label) in enumerate(windows):
        for i in range(1, n + 1):
            xref, yref = _panel_refs(i)
            fig.add_shape(
                type="rect", xref=xref, yref=f"{yref} domain", x0=_x_position(start), x1=_x_position(end),
                y0=0, y1=1, fillcolor=GRIDLINE, opacity=_SHADE_OPACITY[j % 2], line={"width": 0}, layer="below",
            )
        if label:
            add_note(start, end, str(label))
    for x, label in lines:
        _vertical_lines(fig, x, n)
        if label:
            add_note(x, x, str(label))
    for (x, label, right, _), row in zip(notes, _note_rows([nt[3] for nt in notes])):
        _top_note(fig, x, label, log_axis=x_log, row=row, right=right)
    return fig


def matrix_heatmap(
    values: pd.DataFrame,
    *,
    row_labels: Mapping[Any, Any] | pd.Series | None = None,
    col_labels: Mapping[Any, Any] | pd.Series | None = None,
    value_label: str = "value",
    zmax: float | None = None,
    show_text: bool | None = None,
    max_rows: int | None = 80,
    max_cols: int | None = 45,
    title: str | None = None,
    subtitle: str | None = None,
    row_title: str | None = None,
    col_title: str | None = None,
    highlight_rows: Iterable[Any] = (),
    decimals: int | None = None,
) -> go.Figure:
    """A signed matrix as a heatmap: rows top-down, column labels on top, highlighted rows outlined (G.16).

    Used on the trace page for the design matrix of one return week (assets
    x instruments) and the standardised ``Gamma`` (instruments x factors).
    Colours use :data:`DIVERGING_COLORSCALE`, symmetric around 0; missing
    cells are blank.

    Parameters
    ----------
    values:
        Any rows x columns frame of signed numbers, in display order.
    row_labels, col_labels:
        Display names by row key and by column key (dict or Series); row
        labels are cut at :data:`ROW_LABEL_CHARS` and column labels at
        :data:`COL_LABEL_CHARS` characters (the hover shows the full name).
    value_label:
        Name of the cell value (colour bar and hover).
    zmax:
        The colour scale runs from ``-zmax`` to ``zmax``; default the
        largest displayed ``|value|``.
    show_text:
        Print the values in the cells; default when at most 400 cells are
        displayed.
    max_rows, max_cols:
        Keep the first ``max_rows`` rows and ``max_cols`` columns (``None``
        keeps all); highlighted rows beyond the limit replace the last kept
        rows. The subtitle notes "showing X of Y rows" and "X of Y columns".
    title, subtitle:
        Figure title (default none) and a second line; the truncation note
        is appended to the subtitle.
    row_title, col_title:
        Axis titles for the rows (left) and the columns (above the labels).
    highlight_rows:
        Row keys drawn with a bold label and an ink outline (for example the
        selected asset).
    decimals:
        Decimals of the cell text; default from the size of ``zmax`` (2 from
        about 0.1 up, at most 4). The hover shows four significant digits.

    Returns
    -------
    go.Figure
        One heatmap trace named ``value_label`` (``coloraxis``) and, with
        cell text, a text trace named ``"<value_label> values"``.
    """
    vals = _float_frame(values)
    if vals.size == 0 or _no_data(vals):
        return _empty_figure(f"No {value_label} values to show", title=title)

    n_rows_total, n_cols_total = vals.shape
    keys = _key_set(highlight_rows)
    is_high = np.array([_in_keys(k, keys) for k in vals.index], dtype=bool)
    keep = np.ones(n_rows_total, dtype=bool)
    notes: list[str] = []
    if max_rows is not None and int(max_rows) > 0 and n_rows_total > int(max_rows):
        k = int(max_rows)
        keep[:] = False
        keep[:k] = True
        missing = np.flatnonzero(is_high & ~keep)
        if missing.size:
            droppable = np.flatnonzero(keep & ~is_high)[::-1][: missing.size]
            keep[droppable] = False
            keep[missing[: droppable.size]] = True
        notes.append(f"showing {int(keep.sum())} of {n_rows_total} rows"
                     + (" (the highlighted rows included)" if missing.size else ""))
        logger.info("matrix_heatmap: showing %d of %d rows", int(keep.sum()), n_rows_total)
    n_cols = n_cols_total
    if max_cols is not None and int(max_cols) > 0 and n_cols_total > int(max_cols):
        n_cols = int(max_cols)
        notes.append(f"showing {n_cols} of {n_cols_total} columns")
        logger.info("matrix_heatmap: showing %d of %d columns", n_cols, n_cols_total)
    view = vals.iloc[np.flatnonzero(keep), :n_cols]
    is_high = is_high[keep]
    z = view.to_numpy()
    n_rows = z.shape[0]
    subtitle_text = _with_note(subtitle, ", ".join(notes) if notes else None)

    if zmax is None or not np.isfinite(zmax) or zmax <= 0:
        m = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else float("nan")
        zmax = m if np.isfinite(m) and m > 0 else 1.0
    zmax = float(zmax)
    if show_text is None:
        show_text = z.size <= 400
    decimals = _auto_decimals(zmax) if decimals is None else int(decimals)
    row_h = 24 if show_text else int(max(12, min(22, 720 // max(n_rows, 1))))
    tick_size = 11 if row_h >= 16 else 9

    row_full = [_category_name(row_labels, r) for r in view.index]
    col_full = [_category_name(col_labels, c) for c in view.columns]
    row_ticks = [
        f"<b>{_esc(_truncate(t, ROW_LABEL_CHARS))}</b>" if h else _esc(_truncate(t, ROW_LABEL_CHARS))
        for t, h in zip(row_full, is_high)
    ]
    col_ticks = [_esc(_truncate(t, COL_LABEL_CHARS)) for t in col_full]
    hover = [
        [f"{_esc(r)}<br>{_esc(c)}<br>{_esc(value_label)}: {z[i, j]:.4g}" for j, c in enumerate(col_full)]
        for i, r in enumerate(row_full)
    ]
    max_len = max((len(t) for t in col_ticks), default=4)
    rotate = max_len * 6.5 + 8 > 640.0 / max(n_cols, 1)
    label_px = (int(6.3 * max_len) + 14 if rotate else 22) + (20 if col_title else 0)
    plot_h = row_h * n_rows
    bottom = 16
    header = _header_px(title, subtitle_text)
    height = max(200, header + label_px + plot_h + bottom)

    fig = go.Figure()
    fig.update_layout(**_base_layout(title, height, subtitle=subtitle_text, extra_top=label_px, margin={"b": bottom}))
    _style_axes(fig)
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=list(range(n_cols)),
            y=list(range(n_rows)),
            coloraxis="coloraxis",
            xgap=CELL_GAP_PX if n_cols <= 60 else 1,
            ygap=CELL_GAP_PX if row_h >= 14 else 1,
            hoverongaps=False,
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            name=value_label,
        )
    )
    if show_text:
        text_trace = _cell_text_trace(z, zmax, font_size=10 if n_cols <= 30 else 9, name=f"{value_label} values",
                                      decimals=decimals)
        if text_trace is not None:
            fig.add_trace(text_trace)
    for i in np.flatnonzero(is_high):
        fig.add_shape(type="rect", xref="x", yref="y", x0=-0.5, x1=n_cols - 0.5, y0=i - 0.5, y1=i + 0.5,
                      line={"color": INK, "width": 1.5}, fillcolor="rgba(0,0,0,0)")

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_cols)),
        ticktext=col_ticks,
        range=[-0.5, n_cols - 0.5],
        showgrid=False,
        side="top",
        tickangle=-90 if rotate else 0,
        tickfont={"size": 11, "color": INK},
        title={"text": _esc(col_title) if col_title else None, "standoff": 6},
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_rows)),
        ticktext=row_ticks,
        range=[n_rows - 0.5, -0.5],
        showgrid=False,
        tickfont={"size": tick_size, "color": INK},
        title={"text": _esc(row_title) if row_title else None, "standoff": 6},
    )
    fig.update_layout(
        coloraxis={
            "colorscale": [list(p) for p in DIVERGING_COLORSCALE],
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


def ladder_chart(
    ladder: pd.DataFrame | None,
    *,
    metrics: Sequence[tuple[str, str]] = (
        ("spearman", "Spearman with true sensitivities"),
        ("median_r2", "Median OOS R²"),
    ),
    label_col: str = "label",
    highlight: Any = None,
    reference: Any = None,
    title: str | None = None,
    subtitle: str | None = None,
) -> go.Figure:
    """The reference ladder: one row per variant, one horizontal-bar panel per metric side by side (G.16).

    Rows keep the order of ``ladder`` (top to bottom, from what is
    achievable to what BKS delivers), so the drop between two rows is the
    signal lost at that step. The highlighted variant is blue, the reference
    variants (for example the true sensitivities) ink, the others muted.

    Parameters
    ----------
    ladder:
        One row per variant (index: variant key), with the metric columns,
        an optional ``label_col`` (display name) and optional
        ``d_<metric>`` columns (change from the row above, shown in the
        hover).
    metrics:
        ``(column, panel title)`` pairs, one panel each, left to right.
        Columns missing or without a finite value are skipped. Metrics whose
        key ends with ``"r2"`` are shown in percent.
    label_col:
        Column with the row names (cut at :data:`LADDER_LABEL_CHARS`
        characters); default the index keys.
    highlight:
        Index key drawn in blue with a bold label (for example
        ``"bks_implied"``).
    reference:
        Index key or keys drawn in ink (for example ``"oracle"``).
    title, subtitle:
        Figure title (default "Reference ladder") and a second line.

    Returns
    -------
    go.Figure
        One bar trace per metric (``xaxis``, ``xaxis2``, ...), sharing the
        y axis; no legend (the colours are roles, named in the caption).
    """
    title = title if title is not None else "Reference ladder"
    if ladder is None or not isinstance(ladder, pd.DataFrame) or ladder.empty:
        return _empty_figure("No ladder values to show", title=title)
    shown = [(str(key), str(name)) for key, name in metrics
             if key in ladder.columns and not _no_data(_float_series(ladder[key]))]
    if not shown:
        return _empty_figure("No ladder values to show", title=title)

    keys = list(ladder.index)
    names_map = ladder[label_col] if label_col in ladder.columns else None
    names = [_category_name(names_map, k) for k in keys]
    refs = _key_set(reference)
    high = None if highlight is None else str(highlight)
    colors = [BLUE if str(k) == high else INK if str(k) in refs else INK_MUTED for k in keys]
    pos = np.arange(len(keys), dtype=float)

    fig = make_subplots(rows=1, cols=len(shown), shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=[_esc(name) for _, name in shown])
    _align_subplot_titles(fig)
    for j, (key, name) in enumerate(shown, start=1):
        v = _float_series(ladder[key]).to_numpy()
        d_col = f"d_{key}"
        d = _float_series(ladder[d_col]).to_numpy() if d_col in ladder.columns else np.full(len(v), np.nan)
        pct = key.lower().endswith("r2")
        dec = _auto_decimals(float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 1.0)

        def fmt(x: float, sign: bool = False, pct: bool = pct, dec: int = dec) -> str:
            return _fmt(x, 1, scale=100.0, suffix="%", sign=sign) if pct else _fmt(x, dec, sign=sign)

        hover = []
        for nm, x, dx in zip(names, v, d):
            text = f"{_esc(nm)}<br>{_esc(name)}: {fmt(x) or 'not available'}"
            if np.isfinite(dx):
                text += f"<br>Change from the row above: {fmt(dx, sign=True)}"
            hover.append(text)
        fig.add_trace(
            go.Bar(
                x=v,
                y=pos,
                orientation="h",
                width=0.62,
                marker={"color": colors, "line": {"width": 0}},
                text=[fmt(x) for x in v],
                textposition="outside",
                textfont={"family": FONT_FAMILY, "size": 11, "color": INK_SECONDARY},
                cliponaxis=False,
                constraintext="none",
                name=_esc(name),
                showlegend=False,
                hovertext=hover,
                hovertemplate="%{hovertext}<extra></extra>",
            ),
            row=1,
            col=j,
        )
        lo, hi = _bar_range(v, pad_share=0.28)
        fig.update_xaxes(range=[lo, hi], tickformat=".0%" if pct else None, **_ZERO_LINE, row=1, col=j)

    height = _header_px(title, subtitle) + _SUBPLOT_TITLE_PX + 30 * len(keys) + 56
    fig.update_layout(**_base_layout(title, height, subtitle=subtitle, extra_top=_SUBPLOT_TITLE_PX))
    _style_axes(fig)
    fig.update_yaxes(
        tickmode="array",
        tickvals=pos.tolist(),
        ticktext=[f"<b>{_esc(_truncate(nm, LADDER_LABEL_CHARS))}</b>" if str(k) == high
                  else _esc(_truncate(nm, LADDER_LABEL_CHARS)) for k, nm in zip(keys, names)],
        range=[len(keys) - 0.4, -0.6],
        showgrid=False,
        tickfont={"size": 11, "color": INK},
        row=1,
        col=1,
    )
    return fig


def identity_scatter(
    x: pd.Series,
    y: pd.Series,
    *,
    labels: Mapping[Any, Any] | pd.Series | None = None,
    highlight: Any = None,
    x_title: str,
    y_title: str,
    title: str | None = None,
    subtitle: str | None = None,
    fit_line: bool = True,
    identity_line: bool = True,
    highlight_label: str = "Highlighted",
    point_label: str = "Points",
    max_points: int | None = MAX_SCATTER_POINTS,
) -> go.Figure:
    """``y`` against ``x`` with the 45-degree line and the least-squares line; slope and correlation noted (G.16).

    The trace page's "is it what it should be" view: instruments against
    their population reference, realised against fitted returns of one
    week, estimated against true sensitivities. Points on the 45-degree line
    agree; the least-squares slope says by how much ``y`` is scaled
    relative to ``x`` and the correlation how well they line up.

    Parameters
    ----------
    x, y:
        Series aligned by index (``y`` is reindexed to ``x`` when the indexes
        differ); pairs with a missing value are dropped. A MultiIndex (for
        example topic-asset pairs) is fine.
    labels:
        Display names by index key for the hover; a tuple key without its
        own name joins the names of its parts (so one mapping of asset and
        topic names serves topic-asset pairs).
    highlight:
        Index key or keys (list or tuple of keys) drawn in orange on top, the
        others muted. With a MultiIndex, a value highlights every point that
        has it on any level (for example one asset among topic-asset pairs);
        a full key is a tuple inside a list, ``[("T1", "A2")]``.
    x_title, y_title:
        Axis titles (also the hover names of the values).
    title, subtitle:
        Figure title (default none) and a second line; the count, the
        least-squares slope of ``y`` on ``x`` and the Pearson correlation
        are appended to it.
    fit_line, identity_line:
        Draw the least-squares line (a dashed trace) and the 45-degree line
        (a shape). With the 45-degree line both axes share one range and
        scale, so a slope below one is visible as points below the line.
    highlight_label, point_label:
        Legend names of the highlighted and the other points.
    max_points:
        Most points drawn (default :data:`MAX_SCATTER_POINTS`; ``None``
        draws all): the highlighted points and an evenly spaced sample of
        the rest, noted in the subtitle. The statistics use every pair.

    Returns
    -------
    go.Figure
        Marker traces (``Scattergl`` above 3,000 points) with the point
        names in ``customdata``, and the least-squares line.

    Validity boundaries
    -------------------
    The slope and correlation are descriptive over the given pairs (no
    standard errors); with fewer than three pairs or no spread in ``x``
    they are left out.
    """
    xs = _float_series(x)
    ys = _float_series(y)
    if xs.empty or ys.empty:
        return _empty_figure("No pairs to compare", title=title)
    if not xs.index.equals(ys.index):
        try:
            ys = ys.reindex(xs.index)
        except ValueError:
            if len(ys) != len(xs):
                logger.warning("identity_scatter: x and y cannot be aligned (duplicate labels, different lengths)")
                return _empty_figure("No pairs to compare", title=title)
            ys = pd.Series(ys.to_numpy(), index=xs.index)
    xv, yv = xs.to_numpy(), ys.to_numpy()
    ok = np.isfinite(xv) & np.isfinite(yv)
    if not ok.any():
        return _empty_figure("No pairs to compare", title=title)
    keys = np.asarray(xs.index, dtype=object)[ok]
    xv, yv = xv[ok], yv[ok]
    n_pts = int(xv.size)

    slope = intercept = corr = float("nan")
    if n_pts >= 3:
        dx, dy = xv - xv.mean(), yv - yv.mean()
        sxx, syy = float(dx @ dx), float(dy @ dy)
        if sxx > 0:
            slope = float(dx @ dy) / sxx
            intercept = float(yv.mean() - slope * xv.mean())
            if syy > 0:
                corr = float(dx @ dy) / math.sqrt(sxx * syy)

    high_keys = _key_set(highlight)
    is_high = np.array([_in_keys(k, high_keys) for k in keys], dtype=bool) if high_keys else np.zeros(n_pts, bool)
    drawn = np.ones(n_pts, dtype=bool)
    sample_note = None
    if max_points is not None and int(max_points) > 0 and n_pts > int(max_points):
        others = np.flatnonzero(~is_high)
        room = min(max(int(max_points) - int(is_high.sum()), 0), others.size)
        take = others[np.unique(np.linspace(0, others.size - 1, room).round().astype(int))] if room else others[:0]
        drawn[:] = is_high
        drawn[take] = True
        sample_note = (f"showing {int(drawn.sum()):,} of {n_pts:,} points (evenly spaced sample); "
                       "the slope and correlation use all")
        logger.info("identity_scatter: drawing %d of %d points", int(drawn.sum()), n_pts)
    stats = f"{n_pts:,} points"
    if np.isfinite(slope):
        stats += f"; least-squares slope {slope:.2f}"
    if np.isfinite(corr):
        stats += f", correlation {corr:.2f}"
    subtitle_text = _with_note(_with_note(subtitle, stats), sample_note)

    names = np.array([_esc(_point_name(labels, k)) for k in keys], dtype=object)
    n_drawn = int(drawn.sum())
    trace_cls = go.Scattergl if n_drawn > 3000 else go.Scatter
    size = 8 if n_drawn <= 2000 else 5
    hover = (f"%{{customdata}}<br>{_esc(x_title)}: %{{x:.4g}}<br>{_esc(y_title)}: %{{y:.4g}}"
             "<extra></extra>")
    has_high = bool(is_high.any())
    groups = [(drawn & ~is_high, point_label, INK_MUTED if has_high else BLUE, 0.55 if has_high else 0.75, size),
              (drawn & is_high, highlight_label, ORANGE, 1.0, size + 2)]
    fig = go.Figure()
    for m, name, color, alpha, sz in groups:
        if not m.any():
            continue
        marker: dict[str, Any] = {"size": sz, "color": color, "opacity": alpha}
        if trace_cls is go.Scatter:
            marker["line"] = {"width": 1, "color": SURFACE}
        fig.add_trace(trace_cls(x=xv[m], y=yv[m], mode="markers", marker=marker, name=_esc(name),
                                customdata=names[m], hovertemplate=hover))

    shown_vals = np.concatenate([xv, yv])
    lo, hi = _numeric_range(shown_vals, pad_share=0.05, include_zero=False)
    x_lo, x_hi = (lo, hi) if identity_line else _numeric_range(xv, pad_share=0.05, include_zero=False)
    if fit_line and np.isfinite(slope):
        fig.add_trace(
            go.Scatter(
                x=[x_lo, x_hi],
                y=[intercept + slope * x_lo, intercept + slope * x_hi],
                mode="lines",
                line={"color": INK_SECONDARY, "width": 1.5, "dash": "dash"},
                name="Least-squares line",
                hovertemplate=(f"Least-squares line<br>slope {slope:.3f}, intercept {intercept:.3g}"
                               "<extra></extra>"),
            )
        )
    if identity_line:
        fig.add_shape(type="line", xref="x", yref="y", x0=lo, y0=lo, x1=hi, y1=hi,
                      line={"color": INK_MUTED, "width": 1}, layer="below")
        fig.add_annotation(x=hi, y=hi, text="y = x", showarrow=False, xanchor="right", yanchor="bottom",
                           font=_NOTE_FONT)
    legend = len(fig.data) >= 2
    extra_top = _legend_extra_px(fig) if legend else 0
    fig.update_layout(**_base_layout(title, 480 + extra_top, subtitle=subtitle_text, legend=legend,
                                     extra_top=extra_top))
    _style_axes(fig)
    fig.update_xaxes(title={"text": _esc(x_title)}, **_ZERO_LINE)
    fig.update_yaxes(title={"text": _esc(y_title)}, **_ZERO_LINE)
    if identity_line:
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(range=[lo, hi], scaleanchor="x", scaleratio=1)
    return fig


def grouped_bars(
    frame: pd.DataFrame | pd.Series | None,
    *,
    labels: Mapping[Any, Any] | pd.Series | None = None,
    series_labels: Mapping[Any, Any] | None = None,
    colors: Mapping[Any, str] | Sequence[str] | None = None,
    axis_title: str | None = None,
    reference: tuple[float, str] | None = None,
    orientation: str = "h",
    top_n: int | None = None,
    sort_by: Any = None,
    title: str | None = None,
    subtitle: str | None = None,
    percent: bool = False,
    separate: bool = False,
) -> go.Figure:
    """Bars per category, one bar per series side by side (grouped), or one panel per series (G.16).

    Rows of ``frame`` are the categories (topics, forecast weeks,
    instruments, directions), columns the series (for example the true and
    the estimated sensitivity of one asset per topic).

    Parameters
    ----------
    frame:
        Categories x series (a Series is one series). Series without a
        finite value are dropped; at most eight series are drawn (one
        categorical slot each, never cycled).
    labels:
        Display names by category (dates read as ``YYYY-MM-DD``).
    series_labels:
        Display names by column (legend, hover, panel titles).
    colors:
        Colour by column (mapping) or in column order (sequence); default
        :data:`CATEGORICAL` in column order.
    axis_title:
        Title of the value axis.
    reference:
        ``(value, label)`` (or a bare value): a dashed line at ``value`` on
        the value axis of every panel (for example 1.0 for the KKT ratio),
        labelled at the top.
    orientation:
        ``"h"`` (categories as rows, first on top) or ``"v"`` (categories
        along x, first on the left).
    top_n:
        Keep the ``top_n`` categories with the largest ``max |value|`` over
        the series, in their original order (or ``sort_by``); the subtitle
        notes the rest. ``None`` keeps every category up to
        :data:`MAX_BAR_ROWS`.
    sort_by:
        Column whose values order the categories, largest first; default
        the order of ``frame``.
    title, subtitle:
        Figure title (default none) and a second line.
    percent:
        Values are shares: axis ticks, bar text and hover in percent.
    separate:
        One panel per series with its own value axis (for series on
        different scales): side by side for ``"h"``, stacked for ``"v"``;
        the panel titles name the series and there is no legend.

    Returns
    -------
    go.Figure
        One bar trace per series (value text on the bars when at most 30
        bars are drawn); a legend with two or more grouped series.
    """
    if orientation not in ("h", "v"):
        raise ValueError("orientation must be 'h' or 'v'")
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=frame.name if frame.name is not None else "value")
    data = _float_frame(frame)
    if data.size == 0 or _no_data(data):
        return _empty_figure("No values to show", title=title)
    data = data.iloc[:, [j for j in range(data.shape[1]) if not _no_data(data.iloc[:, j])]]
    if data.shape[1] > len(CATEGORICAL):
        logger.warning("grouped_bars: %d series given; showing the first %d", data.shape[1], len(CATEGORICAL))
        data = data.iloc[:, : len(CATEGORICAL)]
    cols = list(data.columns)

    n_total = len(data)
    limit = int(top_n) if top_n is not None and int(top_n) > 0 else MAX_BAR_ROWS
    note = None
    if n_total > limit:
        arr = data.to_numpy()
        size = np.where(np.isfinite(arr), np.abs(arr), -1.0).max(axis=1)
        keep = np.sort(np.argsort(-size, kind="stable")[:limit])
        data = data.iloc[keep]
        note = f"showing {limit} of {n_total} rows (largest absolute values)"
        logger.info("grouped_bars: showing %d of %d rows", limit, n_total)
    if sort_by is not None and sort_by in data.columns:
        key = data[sort_by].to_numpy()
        data = data.iloc[np.lexsort((np.arange(len(key)), np.where(np.isfinite(key), -key, np.inf)))]
    subtitle_text = _with_note(subtitle, note)

    cats = list(data.index)
    names = [_category_name(labels, c) for c in cats]
    pos = np.arange(len(cats), dtype=float)
    if isinstance(colors, Mapping):
        palette = [str(colors.get(c) or CATEGORICAL[i]) for i, c in enumerate(cols)]
    elif colors is not None:
        seq = [str(c) for c in colors]
        palette = [seq[i] if i < len(seq) else CATEGORICAL[i] for i in range(len(cols))]
    else:
        palette = list(CATEGORICAL[: len(cols)])
    snames = [_label(series_labels, c) for c in cols]
    n_bars = int(np.isfinite(data.to_numpy()).sum())
    show_text = n_bars <= 30

    def fmt(v: float) -> str:
        if not np.isfinite(v):
            return ""
        return _fmt(v, 1, scale=100.0, suffix="%") if percent else _fmt_sig(v, 3)

    horizontal = orientation == "h"
    n_series = len(cols)
    if separate and n_series >= 2:
        if horizontal:
            fig = make_subplots(rows=1, cols=n_series, shared_yaxes=True, horizontal_spacing=0.06,
                                subplot_titles=[_esc(s) for s in snames])
        else:
            fig = make_subplots(rows=n_series, cols=1, shared_xaxes=True, vertical_spacing=0.3 / n_series,
                                subplot_titles=[_esc(s) for s in snames])
        _align_subplot_titles(fig)
        n_panels = n_series
    else:
        separate = False
        fig = go.Figure()
        n_panels = 1
    legend = n_series >= 2 and not separate

    for j, (sname, color) in enumerate(zip(snames, palette), start=1):
        v = data.iloc[:, j - 1].to_numpy()
        hover = [f"{_esc(nm)}<br>{_esc(sname)}: {_fmt(x, 2, scale=100.0, suffix='%') if percent else _fmt_sig(x, 4)}"
                 if np.isfinite(x) else f"{_esc(nm)}<br>{_esc(sname)}: not available" for nm, x in zip(names, v)]
        bar = go.Bar(
            x=v if horizontal else pos,
            y=pos if horizontal else v,
            orientation=orientation,
            marker={"color": color, "line": {"width": 0}},
            name=_esc(_truncate(sname, 40)),
            showlegend=legend,
            text=[fmt(x) for x in v] if show_text else None,
            textposition="outside" if show_text else None,
            textfont={"family": FONT_FAMILY, "size": 10, "color": INK_SECONDARY},
            cliponaxis=False,
            constraintext="none",
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
        )
        if separate:
            fig.add_trace(bar, row=1 if horizontal else j, col=j if horizontal else 1)
        else:
            fig.add_trace(bar)

    if reference is not None and not isinstance(reference, (tuple, list)):
        reference = (reference, "")
    ref_value = _to_float(reference[0]) if reference else float("nan")
    top = (_SUBPLOT_TITLE_PX if separate else 0) + (_legend_extra_px(fig) if legend else 0)
    tick_names = [_esc(_truncate(nm, 34 if horizontal else 24)) for nm in names]
    if horizontal:
        row_px = 22 if (n_series == 1 or separate) else 8 + 10 * n_series
        height = _header_px(title, subtitle_text, legend) + top + row_px * len(cats) + 56
    else:
        height = _header_px(title, subtitle_text, legend) + top + (380 if n_panels == 1 else 220 * n_panels)
    fig.update_layout(**_base_layout(title, height, subtitle=subtitle_text, legend=legend, extra_top=top),
                      barmode="group", bargap=0.25)
    _style_axes(fig)
    # the value axis of each panel: its own range (every series together when grouped)
    for j in range(1, n_panels + 1):
        vals = data.iloc[:, j - 1].to_numpy() if separate else data.to_numpy().ravel()
        lo, hi = _bar_range(np.append(vals, ref_value), pad_share=0.18 if show_text else 0.05)
        value_axis: dict[str, Any] = {"range": [lo, hi], **_ZERO_LINE}
        if percent:
            value_axis["tickformat"] = ".0%"
        if axis_title:
            value_axis["title"] = {"text": _esc(axis_title)}
        where = ({"row": 1, "col": j} if horizontal else {"row": j, "col": 1}) if separate else {}
        (fig.update_xaxes if horizontal else fig.update_yaxes)(**value_axis, **where)
    if horizontal:
        fig.update_yaxes(tickmode="array", tickvals=pos.tolist(), ticktext=tick_names,
                         range=[len(cats) - 0.4, -0.6], showgrid=False, tickfont={"size": 11, "color": INK})
    else:
        fig.update_xaxes(tickmode="array", tickvals=pos.tolist(), ticktext=tick_names,
                         range=[-0.6, len(cats) - 0.4], showgrid=False, tickfont={"size": 11, "color": INK})

    if np.isfinite(ref_value):
        label = str(reference[1]) if len(reference) > 1 and reference[1] else ""
        for i in range(1, n_panels + 1):
            xref, yref = _panel_refs(i)
            if horizontal:
                fig.add_shape(type="line", xref=xref, yref=f"{yref} domain", x0=ref_value, x1=ref_value, y0=0, y1=1,
                              line={"color": INK_SECONDARY, "width": 1, "dash": "dash"})
            else:
                fig.add_shape(type="line", xref=f"{xref} domain", yref=yref, x0=0, x1=1, y0=ref_value, y1=ref_value,
                              line={"color": INK_SECONDARY, "width": 1, "dash": "dash"})
        if label:
            if horizontal:
                fig.add_annotation(x=ref_value, xref="x", y=1.0, yref="y domain", text=_esc(label), showarrow=False,
                                   xanchor="left", yanchor="bottom", xshift=4, font=_NOTE_FONT)
            else:
                fig.add_annotation(x=1.0, xref="x domain", y=ref_value, yref="y", text=_esc(label), showarrow=False,
                                   xanchor="right", yanchor="bottom", font=_NOTE_FONT, bgcolor=_NOTE_BG)
    return fig


def lambda_trace_chart(
    path: pd.DataFrame | None,
    *,
    lam_star: float | None = None,
    lam_best: float | None = None,
    band_floor: float | None = None,
    criterion_label: str = "In-sample Sharpe ratio (annualised)",
    extra: pd.DataFrame | None = None,
    extra_labels: Mapping[Any, Any] | None = None,
    extra_title: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    null_band: tuple[float, float, float] | Sequence[float] | None = None,
    band_floor_alt: float | None = None,
) -> go.Figure:
    """The lambda path with its noise band: criterion, topics selected and, optionally, recovery (G.16; D51, D70).

    Panel 1: the tuning criterion against lambda (log axis) with a band of
    plus or minus one standard error, the tolerance threshold as a dashed
    horizontal line, the best point (ink ring) and the chosen point (orange
    diamond); points inside the tolerance band are filled, the others open.
    Optionally, beneath it, the range the criterion takes with no priced
    signal (a light grey band with a dashed median line) and a second
    tolerance threshold. Panel 2: the number of selected topics (steps).
    Panel 3 (with ``extra``): recovery measures of the fit at each lambda. A
    dashed vertical line marks the chosen lambda in every panel.

    Parameters
    ----------
    path:
        One row per grid point of one ``K`` with ``lam`` and any of
        ``criterion``, ``se`` (standard error of the criterion),
        ``n_selected``, ``in_band`` (bool), ``best`` (bool), ``chosen``
        (bool): the trace's path table (``BKSTrace.path``). Points with a
        non-positive lambda are dropped.
    lam_star, lam_best:
        Chosen and best lambda; default the rows flagged ``chosen`` and
        ``best``.
    band_floor:
        Lowest criterion inside the tolerance band (dashed line); ``None``
        for none.
    criterion_label:
        Name of the criterion (panel title and hover).
    extra:
        Recovery along the path, indexed by lambda (or with a ``lam``
        column): the columns named in ``extra_labels`` are drawn, else every
        numeric column except ``n_selected``, ``criterion`` and ``se``
        (they are in the panels above). At most six columns.
    extra_labels:
        Display names by ``extra`` column; its keys also choose the columns.
    extra_title:
        Title of panel 3 (default "Recovery along the path").
    title, subtitle:
        Figure title (default "BKS lambda path") and a second line.
    null_band:
        ``(q05, q50, q95)``: quantiles of the criterion when no factor is
        priced (the trace's ``null_q05``, ``null_q50``, ``null_q95``). Drawn
        in panel 1 as a light grey band from ``q05`` to ``q95`` across the
        grid (traces "No priced signal: 95% quantile", without a legend
        entry, then "No priced signal: 5-95%", filled to it) and a dashed
        line at ``q50`` ("No priced signal: median"). ``None`` or non-finite
        quantiles draw nothing.
    band_floor_alt:
        A second tolerance threshold (the trace's ``band_floor_relative``),
        drawn as a dotted line labelled "Relative band floor" only when it
        differs from ``band_floor``; the higher line gets its label above
        it, the lower one below.

    Returns
    -------
    go.Figure
        Stacked subplots with log x axes; traces for the no-signal band
        (with ``null_band``: upper edge, filled lower edge, median), the
        standard-error band (upper edge, then the filled lower edge), the
        criterion, the best and the chosen point, the selected topics and
        the ``extra`` lines.

    Validity boundaries
    -------------------
    The band is the standard error of each point on its own, as given. It
    is not a test of the difference between two grid points: neighbouring
    fits share most of their topics and weeks, so their criteria move
    together and their difference is less noisy than one band suggests.
    """
    title = title if title is not None else "BKS lambda path"
    frame = _lambda_frame(path)
    if frame.empty:
        return _empty_figure("No lambda path to show", title=title)
    lams = frame.index.to_numpy(dtype=float)
    crit = frame["criterion"].to_numpy() if "criterion" in frame.columns else np.full(len(frame), np.nan)
    n_sel = frame["n_selected"].to_numpy() if "n_selected" in frame.columns else np.full(len(frame), np.nan)
    if _no_data(crit) and _no_data(n_sel):
        return _empty_figure("No lambda path to show", title=title)
    se = frame["se"].to_numpy() if "se" in frame.columns else np.full(len(frame), np.nan)
    in_band = frame["in_band"].to_numpy() if "in_band" in frame.columns else np.full(len(frame), np.nan)

    def flagged(col: str) -> float | None:
        if col not in frame.columns:
            return None
        hits = np.flatnonzero(frame[col].to_numpy() == 1.0)
        return float(lams[hits[0]]) if hits.size else None

    lam_star = lam_star if lam_star is not None else flagged("chosen")
    lam_best = lam_best if lam_best is not None else flagged("best")

    ex = _lambda_frame(extra) if extra is not None else pd.DataFrame()
    if not ex.empty:
        wanted = [c for c in extra_labels if c in ex.columns] if extra_labels else [
            c for c in ex.columns if c not in ("n_selected", "criterion", "se")]
        wanted = [c for c in wanted if not _no_data(ex[c])]
        if len(wanted) > len(CATEGORICAL) - 2:
            logger.warning("lambda_trace_chart: %d extra columns; showing the first %d", len(wanted),
                           len(CATEGORICAL) - 2)
            wanted = wanted[: len(CATEGORICAL) - 2]
        ex = ex[wanted]
    n_panels = 3 if not ex.empty and ex.shape[1] else 2
    panel_titles = [criterion_label, "Topics selected"] + ([extra_title or "Recovery along the path"]
                                                           if n_panels == 3 else [])
    fig = make_subplots(rows=n_panels, cols=1, shared_xaxes=True, vertical_spacing=0.36 / n_panels,
                        subplot_titles=[_esc(t) for t in panel_titles])
    _align_subplot_titles(fig)
    crit_name = _esc(criterion_label)
    status = np.where(in_band == 1.0, "inside the tolerance band",
                      np.where(in_band == 0.0, "outside the tolerance band", ""))

    null = _null_quantiles(null_band)
    if null is not None and np.isfinite(crit).any():
        q05, q50, q95 = null
        span = [float(lams.min()), float(lams.max())]
        fig.add_trace(
            go.Scatter(x=span, y=[q95, q95], mode="lines", line={"width": 0, "color": INK_MUTED}, showlegend=False,
                       name="No priced signal: 95% quantile", legendgroup="null",
                       hovertemplate="No priced signal: 95% quantile %{y:.2f}<extra></extra>"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=span, y=[q05, q05], mode="lines", line={"width": 0, "color": INK_MUTED}, fill="tonexty",
                       fillcolor=_rgba(INK_MUTED, 0.16), name="No priced signal: 5-95%", legendgroup="null",
                       hovertemplate="No priced signal: 5% quantile %{y:.2f}<extra></extra>"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=span, y=[q50, q50], mode="lines", line={"width": 1.5, "color": INK_MUTED, "dash": "dash"},
                       name="No priced signal: median", legendgroup="null",
                       hovertemplate="No priced signal: median %{y:.2f}<extra></extra>"),
            row=1, col=1,
        )

    if np.isfinite(se).any() and np.isfinite(crit).any():
        upper, lower = crit + se, crit - se
        fig.add_trace(
            go.Scatter(x=lams, y=upper, mode="lines", line={"width": 0, "color": BLUE}, showlegend=False,
                       name="Criterion + 1 standard error", legendgroup="se",
                       hovertemplate="λ = %{x:.3g}<br>Criterion + 1 standard error: %{y:.3f}<extra></extra>"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=lams, y=lower, mode="lines", line={"width": 0, "color": BLUE}, fill="tonexty",
                       fillcolor=_rgba(BLUE, 0.14), name="± 1 standard error", legendgroup="se",
                       hovertemplate="λ = %{x:.3g}<br>Criterion - 1 standard error: %{y:.3f}<extra></extra>"),
            row=1, col=1,
        )
    if np.isfinite(crit).any():
        fill = [SURFACE if b == 0.0 else BLUE for b in in_band]
        notes = [
            "<br>".join(p for p in (f"Standard error: {_fmt(s, 2)}" if np.isfinite(s) else "", st) if p)
            for s, st in zip(se, status)
        ]
        fig.add_trace(
            go.Scatter(
                x=lams, y=crit, mode="lines+markers", line={"color": BLUE, "width": 2},
                marker={"size": 8, "color": fill, "line": {"width": 1.5, "color": BLUE}},
                name=crit_name, customdata=notes,
                hovertemplate=f"λ = %{{x:.3g}}<br>{crit_name}: %{{y:.3f}}<br>%{{customdata}}<extra></extra>",
            ),
            row=1, col=1,
        )
        for lam, name, marker in (
            (lam_best, "Best criterion", {"symbol": "circle-open", "size": 16, "color": INK, "line": {"width": 2}}),
            (lam_star, "Chosen λ", {"symbol": "diamond", "size": 12, "color": ORANGE,
                                     "line": {"width": 1.5, "color": SURFACE}}),
        ):
            i = _nearest_lambda(lams, lam)
            if i is None or not np.isfinite(crit[i]):
                continue
            fig.add_trace(
                go.Scatter(x=[lams[i]], y=[crit[i]], mode="markers", marker=marker, name=name,
                           hovertemplate=f"{name}<br>λ = %{{x:.3g}}<br>{crit_name}: %{{y:.3f}}<extra></extra>"),
                row=1, col=1,
            )
    if np.isfinite(n_sel).any():
        fig.add_trace(
            go.Scatter(x=lams, y=n_sel, mode="lines+markers",
                       line={"color": INK_SECONDARY, "width": 1.5, "shape": "hvh"},
                       marker={"size": 6, "color": INK_SECONDARY}, name="Topics selected", showlegend=False,
                       hovertemplate="λ = %{x:.3g}<br>Topics selected: %{y:.0f}<extra></extra>"),
            row=2, col=1,
        )
    if n_panels == 3:
        for j, col in enumerate(ex.columns):
            color = CATEGORICAL[2 + j]
            name = _esc(_label(extra_labels, col))
            fig.add_trace(
                go.Scatter(x=ex.index.to_numpy(dtype=float), y=ex[col].to_numpy(), mode="lines+markers",
                           line={"color": color, "width": 2}, marker={"size": 6, "color": color,
                                                                      "line": {"width": 1, "color": SURFACE}},
                           name=name, hovertemplate=f"{name}<br>λ = %{{x:.3g}}<br>%{{y:.3f}}<extra></extra>"),
                row=3, col=1,
            )

    legend = sum(1 for t in fig.data if t.showlegend is not False) >= 2
    extra_top = _SUBPLOT_TITLE_PX + (_legend_extra_px(fig) if legend else 0)
    height = _header_px(title, subtitle, legend) + extra_top + (430 if n_panels == 2 else 590)
    fig.update_layout(**_base_layout(title, height, subtitle=subtitle, legend=legend, extra_top=extra_top))
    _style_axes(fig)
    fig.update_xaxes(type="log", showspikes=True, spikecolor=BASELINE, spikethickness=1, spikedash="solid",
                     spikemode="across")
    fig.update_xaxes(title={"text": "λ (log scale)"}, row=n_panels, col=1)
    fig.update_yaxes(title={"text": "Criterion"}, row=1, col=1)
    fig.update_yaxes(title={"text": "Topics"}, rangemode="tozero", row=2, col=1)
    if n_panels == 3:
        fig.update_yaxes(title={"text": "Value"}, **_ZERO_LINE, row=3, col=1)

    bf = _to_float(band_floor) if band_floor is not None else float("nan")
    alt = _to_float(band_floor_alt) if band_floor_alt is not None else float("nan")
    if np.isfinite(alt) and np.isfinite(bf) and math.isclose(alt, bf, rel_tol=1e-9, abs_tol=1e-12):
        alt = float("nan")  # the same threshold: one line
    floors = [(v, f"{label} {v:.3g}", dash, color) for v, label, dash, color in (
        (bf, "Tolerance band floor", "dash", INK_SECONDARY), (alt, "Relative band floor", "dot", INK_MUTED))
        if np.isfinite(v)]
    top = max(v for v, *_ in floors) if len(floors) == 2 else None
    for v, text, dash, color in floors:
        fig.add_shape(type="line", xref="x domain", yref="y", x0=0, x1=1, y0=v, y1=v,
                      line={"color": color, "width": 1, "dash": dash})
        below = top is not None and v < top  # two lines: the lower one gets its label underneath
        fig.add_annotation(x=1.0, xref="x domain", y=v, yref="y", text=text, showarrow=False, xanchor="right",
                           yanchor="top" if below else "bottom", xshift=-4, font=_NOTE_FONT, bgcolor=_NOTE_BG)
    if lam_star is not None and np.isfinite(_to_float(lam_star)) and _to_float(lam_star) > 0:
        _vertical_lines(fig, _to_float(lam_star), n_panels)
        _top_note(fig, _to_float(lam_star), f"λ* = {_to_float(lam_star):.3g}", log_axis=True)
    return fig


def coefficient_path_chart(
    norms: pd.DataFrame | None,
    *,
    selected: Iterable[Any] = (),
    labels: Mapping[Any, Any] | pd.Series | None = None,
    lam_star: float | None = None,
    max_colored: int = 8,
    title: str | None = None,
    subtitle: str | None = None,
) -> go.Figure:
    """Standardised ``Gamma`` row norms per instrument along the lambda path (log x) (G.16; G.7.2).

    The ``max_colored`` instruments with the largest norm at the chosen
    lambda (else the largest anywhere on the path) are drawn in
    :data:`CATEGORICAL` colours with legend entries; the others are thin
    lines without legend entries: secondary ink when selected at the chosen
    lambda, muted otherwise. A dashed vertical line marks ``lam_star``.

    Parameters
    ----------
    norms:
        Lambda x instruments (index lambda, or a ``lam`` column): the row
        norm of ``Gamma`` times the training standard deviation of the
        instrument, for the fit at each lambda.
    selected:
        Instruments with a nonzero ``Gamma`` row at the chosen lambda.
    labels:
        Display names by instrument.
    lam_star:
        Chosen lambda.
    max_colored:
        Instruments drawn in colour (at most eight; never cycled).
    title, subtitle:
        Figure title (default "Gamma row norms along the lambda path") and a
        second line; a note on the colours is appended to it.

    Returns
    -------
    go.Figure
        One trace per coloured instrument and at most two traces for the
        others (their lines joined with gaps; the hover names each line).
    """
    title = title if title is not None else "Gamma row norms along the lambda path"
    frame = _lambda_frame(norms)
    if frame.empty or _no_data(frame):
        return _empty_figure("No Gamma row norms along the path to show", title=title)
    frame = frame.iloc[:, [j for j in range(frame.shape[1]) if not _no_data(frame.iloc[:, j])]]
    lams = frame.index.to_numpy(dtype=float)
    arr = frame.to_numpy()
    i_star = _nearest_lambda(lams, lam_star)
    rank_by = arr[i_star] if i_star is not None else np.nanmax(arr, axis=0)
    rank_by = np.nan_to_num(rank_by, nan=0.0)
    k = int(np.clip(int(max_colored), 0, len(CATEGORICAL)))
    order = [c for c in np.argsort(-rank_by, kind="stable") if rank_by[c] > 0][:k]
    sel = _key_set(selected)
    cols = list(frame.columns)
    names = [_category_name(labels, c) for c in cols]
    mode = "lines+markers" if len(lams) <= 40 else "lines"

    fig = go.Figure()
    grey = [c for c in range(len(cols)) if c not in set(order)]
    for group, name, color, width, opacity in (
        ([c for c in grey if str(cols[c]) in sel], "Selected at λ*", INK_SECONDARY, 1.2, 0.9),
        ([c for c in grey if str(cols[c]) not in sel], "Other instruments", INK_MUTED, 1.0, 0.6),
    ):
        if not group:
            continue
        xs: list[float] = []
        ys: list[float] = []
        cd: list[str] = []
        for c in group:
            xs += lams.tolist() + [float("nan")]
            ys += arr[:, c].tolist() + [float("nan")]
            cd += [_esc(names[c])] * len(lams) + [""]
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", line={"color": color, "width": width}, opacity=opacity, name=name,
                       showlegend=False, customdata=cd,
                       hovertemplate="%{customdata}<br>λ = %{x:.3g}<br>Row norm: %{y:.4g}<extra></extra>")
        )
    for slot, c in enumerate(order):
        color = CATEGORICAL[slot]
        name = names[c]
        suffix = " (not selected)" if sel and str(cols[c]) not in sel else ""
        fig.add_trace(
            go.Scatter(
                x=lams, y=arr[:, c], mode=mode, line={"color": color, "width": 2},
                marker={"size": 6, "color": color, "line": {"width": 1, "color": SURFACE}},
                name=_esc(_truncate(name, 34)) + suffix,
                hovertemplate=f"{_esc(name)}{suffix}<br>λ = %{{x:.3g}}<br>Row norm: %{{y:.4g}}<extra></extra>",
            )
        )
    where = "at the chosen λ" if i_star is not None else "on the path"
    note = f"{len(order)} of {len(cols)} instruments in colour (largest {where})"
    if sel:
        note += f"; {sum(1 for c in cols if str(c) in sel)} selected at the chosen λ"
    subtitle_text = _with_note(subtitle, note)
    legend = sum(1 for t in fig.data if t.showlegend is not False) >= 2
    extra_top = _legend_extra_px(fig) if legend else 0
    height = _header_px(title, subtitle_text, legend) + extra_top + 400
    fig.update_layout(**_base_layout(title, height, subtitle=subtitle_text, legend=legend, extra_top=extra_top))
    _style_axes(fig)
    fig.update_xaxes(type="log", title={"text": "λ (log scale)"})
    fig.update_yaxes(title={"text": "Standardised Gamma row norm"}, rangemode="tozero", **_ZERO_LINE)
    if lam_star is not None and np.isfinite(_to_float(lam_star)) and _to_float(lam_star) > 0:
        _vertical_lines(fig, _to_float(lam_star), 1)
        _top_note(fig, _to_float(lam_star), f"λ* = {_to_float(lam_star):.3g}", log_axis=True)
    return fig
