"""Real data page: placeholder for the research pipeline's estimates (DESIGN.md G.14, D82).

Nothing here is simulated. The page lists the settings of the shared sidebar
that will apply to real data, checks ``data/real/`` for the files of the data
contract (TBC), explains what it will show, and previews the sensitivity table
when ``sensitivities.parquet`` is present. The pure helpers
(:func:`data_status`, :func:`load_exposures`, :func:`exposure_table`) have no
Streamlit dependency; :func:`render` draws the page.

Naming: the page says "topic sensitivity" (the expected return response of an
asset to a one-standard-deviation attention shock in a topic, with the other
topics' shocks held fixed; :data:`_ui.SENSITIVITY_DEFINITION`). In code,
"exposure" means topic sensitivity: the module keeps its name
``real_exposures`` (the page was called "Real exposures" until 2026-09-29),
and ``EXPOSURE_COLUMNS``, :func:`load_exposures` and :func:`exposure_table`
keep theirs. The contract's file and value column were renamed from
``exposures.parquet`` and ``exposure`` to ``sensitivities.parquet`` and
``sensitivity`` on 2026-09-30.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import _ui
from narrative_ipca.exposure_lab import charts, reference

#: Allowed values of the categorical columns (research plan interface fields).
SOURCES: tuple[str, ...] = ("structural", "regression", "llm", "blended")
COVERAGES: tuple[str, ...] = ("full", "partial", "none")

#: File name of the topic sensitivities in ``data/real/`` (one row per as_of x topic x asset).
SENSITIVITY_FILE = "sensitivities.parquet"

#: Required columns of :data:`SENSITIVITY_FILE`; ``sensitivity`` holds the topic sensitivity.
EXPOSURE_COLUMNS: tuple[str, ...] = (
    "as_of", "topic_id", "asset_id", "sensitivity", "uncertainty", "source", "effective_window", "coverage", "version",
)
#: Required columns of ``topics.csv``.
TOPIC_COLUMNS: tuple[str, ...] = ("topic_id", "name", "taxonomy_class", "origin", "model_version")

#: The data contract (TBC), in display order.
CONTRACT: tuple[dict[str, str], ...] = (
    {
        "file": "data/real/topics.csv",
        "content": "One row per topic.",
        "columns": "topic_id, name, taxonomy_class, origin (fastopic | manual), model_version",
    },
    {
        "file": "data/real/attention.parquet",
        "content": "Daily attention level per topic (FASTopic or manual topics on the news archive).",
        "columns": "date index; one column per topic_id",
    },
    {
        "file": f"data/real/{SENSITIVITY_FILE}",
        "content": "Estimated topic sensitivity of each asset to each topic, per estimation date.",
        "columns": "as_of, topic_id, asset_id, sensitivity (% return per one-sd attention shock), uncertainty (sd), "
                   "source (structural | regression | llm | blended), effective_window, coverage "
                   "(full | partial | none), version",
    },
)


def real_dir() -> Path:
    """``data/real`` under the lab's data folder (``NARRATIVE_IPCA_DATA_DIR`` overrides the data folder)."""
    return reference.data_dir() / "real"


def data_status(root: Path | None = None) -> pd.DataFrame:
    """One row per contract file: whether it exists, its row count and date range where it has one."""
    root = real_dir() if root is None else Path(root)
    rows = []
    for item in CONTRACT:
        path = root / Path(item["file"]).name
        found = path.exists()
        n_rows: Any = ""
        span = ""
        if found:
            try:
                df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
                n_rows = len(df)
                if "as_of" in df.columns and len(df):
                    d = pd.to_datetime(df["as_of"])
                    span = f"{d.min().date()} to {d.max().date()}"
                elif isinstance(df.index, pd.DatetimeIndex) and len(df):
                    span = f"{df.index.min().date()} to {df.index.max().date()}"
            except Exception as exc:  # noqa: BLE001 - reported on the page, not raised
                span = f"unreadable: {type(exc).__name__}"
        rows.append({"File": item["file"], "Found": "yes" if found else "no", "Rows": n_rows, "Dates": span,
                     "Content": item["content"]})
    return pd.DataFrame(rows)


def load_exposures(root: Path | None = None) -> tuple[pd.DataFrame | None, list[str]]:
    """Read and check ``sensitivities.parquet``; returns ``(frame or None, problems)``."""
    root = real_dir() if root is None else Path(root)
    path = root / SENSITIVITY_FILE
    if not path.exists():
        return None, []
    df = pd.read_parquet(path)
    problems = [f"missing column '{c}'" for c in EXPOSURE_COLUMNS if c not in df.columns]
    if problems:
        return None, problems
    df = df.copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    bad_source = sorted(set(df["source"].astype(str)) - set(SOURCES))
    if bad_source:
        problems.append(f"unknown source values: {', '.join(bad_source)}")
    bad_cov = sorted(set(df["coverage"].astype(str)) - set(COVERAGES))
    if bad_cov:
        problems.append(f"unknown coverage values: {', '.join(bad_cov)}")
    dup = df.duplicated(["as_of", "topic_id", "asset_id"]).sum()
    if dup:
        problems.append(f"{int(dup)} duplicate (as_of, topic_id, asset_id) rows")
    return df, problems


def exposure_table(df: pd.DataFrame, as_of: Any, asset_order: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asset x topic sensitivities at ``as_of`` and the blank mask (``coverage == "none"`` or missing)."""
    day = df[df["as_of"] == pd.Timestamp(as_of)]
    values = day.pivot_table(index="asset_id", columns="topic_id", values="sensitivity", aggfunc="first")
    cov = day.pivot_table(index="asset_id", columns="topic_id", values="coverage", aggfunc="first")
    if asset_order:
        order = [a for a in asset_order if a in values.index] + [a for a in values.index if a not in asset_order]
        values, cov = values.reindex(order), cov.reindex(order)
    blank = values.isna() | cov.reindex_like(values).eq("none")
    return values, blank


def render_settings(settings: dict[str, Any]) -> None:
    """Draw the "Settings in use" section: the shared sidebar's settings that apply to real data.

    Invalid settings do not stop the page: their errors are shown as one
    warning.
    """
    import streamlit as st

    st.subheader("Settings in use")
    st.caption("The sidebar is shared with the simulation lab. These settings will apply to real data:")
    values = settings.get("values")
    if values:
        try:
            table = _ui.settings_in_use(values, settings.get("asset_classes"), settings.get("asset_names"))
        except (KeyError, TypeError, ValueError) as exc:
            st.warning(f"The settings could not be listed: {exc}")
        else:
            st.dataframe(table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3)
    errors = list(settings.get("errors") or [])
    if errors:
        st.warning("The current settings are not valid, so they would not run:\n\n"
                   + "\n".join(f"- {e}" for e in errors))
    st.caption(_ui.SIMULATION_ONLY_NOTE)


def render(settings: dict[str, Any] | None = None) -> None:
    """Draw the Real data page.

    Parameters
    ----------
    settings:
        The shared sidebar's result, from ``dashboard/app.py``: ``values``
        (sidebar values in effect, keyed by widget key), ``errors`` (settings
        validation errors), and optionally ``asset_classes`` and
        ``asset_names`` of the listed assets. ``None`` leaves out the
        "Settings in use" section.
    """
    import streamlit as st

    st.title("Real data")
    st.info(
        "Placeholder. This page will show the topic sensitivities that the research pipeline estimates on real "
        "news and real returns. Nothing on it is simulated, and there is no truth or oracle to compare against."
    )
    st.caption(_ui.SENSITIVITY_DEFINITION)

    if settings is not None:
        render_settings(settings)

    st.subheader("Status")
    status = data_status()
    st.dataframe(status, hide_index=True, width="stretch")
    n_found = int((status["Found"] == "yes").sum())
    st.caption(f"{n_found} of {len(status)} input files found in {real_dir()}.")

    st.subheader("What this page will show")
    st.markdown(
        "1. **Sensitivity table**: estimated topic sensitivity of each asset to each topic (% return per one-sd "
        "attention shock) with its uncertainty, in the layout of the simulation's correlation table, for a chosen "
        "estimation date.\n"
        "2. **Topic contributions**: for a chosen asset and period, the frozen sensitivities times the realised "
        "attention shocks, ranked with the largest on top, plus the part not explained by topics.\n"
        "3. **Explained variation over time**: out-of-sample R² per asset on consecutive windows.\n"
        "4. **Provenance**: each sensitivity's source, estimation window and coverage."
    )

    st.subheader("What it needs (data contract, TBC)")
    st.dataframe(pd.DataFrame(CONTRACT).rename(columns=str.capitalize), hide_index=True, width="stretch")
    st.caption(
        "Assets and returns come from data/reference/assets.csv and data/market/; asset_id must match. The "
        "column names follow the interface fields of the research plan (sensitivity, uncertainty, source, "
        "effective window, coverage)."
    )

    st.subheader("How it differs from the simulation lab")
    st.markdown(
        "- There is no truth and no oracle: sensitivities are judged on realised returns and on the golden set.\n"
        "- Sensitivities come from the plan's estimation (B = M·L + S), not from the lab's direct regression.\n"
        "- Attention shocks and contributions use the same definitions as the lab, so the views are comparable."
    )

    df, problems = load_exposures()
    if df is None and not problems:
        return
    st.subheader("Preview")
    for p in problems:
        st.error(f"{SENSITIVITY_FILE}: {p}")
    if df is None or df.empty:
        return
    dates = sorted(df["as_of"].unique())
    as_of = st.selectbox("Estimation date", dates, index=len(dates) - 1,
                         format_func=lambda d: str(pd.Timestamp(d).date()), key="real_as_of")
    try:
        assets = reference.load_assets()
        a_labels = assets["name"].to_dict()
        order = list(assets.index)
    except Exception:  # noqa: BLE001 - reference data is optional for the preview
        a_labels, order = {}, None
    values, blank = exposure_table(df, as_of, order)
    fig = charts.exposure_heatmap(
        values, blank=blank, value_label="Sensitivity (% per one-sd shock)", row_labels=a_labels,
        title=f"Estimated topic sensitivities at {pd.Timestamp(as_of).date()}",
        subtitle="Blank: coverage none or no estimate.",
    )
    st.plotly_chart(fig, width="stretch", theme=None, key="real_heatmap")
