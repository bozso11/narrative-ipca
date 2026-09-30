"""Reference data of the topic-sensitivity lab: locations and loaders (DESIGN.md G.2, G.3, G.4, G.13; D53, D55, D57, D58).

Files (all under :func:`data_dir`):

* ``reference/assets.csv``: the 55 listed assets in the order of the source
  image (G.2). Each asset "A v B" is long leg ``A`` and short leg ``B``;
  outrights and "XXX v USD" pairs are long against cash (D75).
* ``reference/legs.csv``: one row per leg with its benchmark index and the
  tradeable proxy (G.2.1, D54).
* ``reference/topics.csv``: the 20 manual topics of the report (G.3, D57).
* ``reference/link_map.csv``: the authored default link map for manual topics
  x listed assets (G.4 point 1, D58).
* ``market/``: the market data store written by
  ``scripts/fetch_market_data.py`` (D55); read by :mod:`.market`.

Reads are cached per resolved path (and file modification time, so a
regenerated file is re-read); every loader returns a fresh copy, so callers
may modify what they get.

Validity boundaries
-------------------
* The environment variable ``NARRATIVE_IPCA_DATA_DIR`` replaces the repository
  ``data/`` folder (used by tests and by alternative data stores); it is read
  on every call, not at import.
* :func:`load_default_links` validates ids against the reference topics and
  assets of the same data folder, not against a run's (possibly smaller)
  topic or asset set.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TIERS

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_DIR_ENV",
    "ASSET_CLASSES",
    "data_dir",
    "reference_dir",
    "market_dir",
    "load_assets",
    "load_legs",
    "load_topics",
    "load_default_links",
    "clear_cache",
]

#: Environment variable that overrides the data folder.
DATA_DIR_ENV = "NARRATIVE_IPCA_DATA_DIR"

#: Asset classes of the lab (G.2).
ASSET_CLASSES: tuple[str, str, str] = ("Equity", "FX", "Fixed income")

_ASSET_COLUMNS = ["order", "name", "long_leg", "short_leg", "asset_class", "sub_class"]
_TOPIC_COLUMNS = ["order", "ontology", "group", "name", "scope"]
_LINK_COLUMNS = ["topic_id", "asset_id", "tier", "sign", "mechanism"]
_LEG_REQUIRED = ["leg_id", "name", "benchmark_index", "proxy_ticker"]


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
def data_dir() -> Path:
    """The lab's data folder: ``<repo>/data``, or ``$NARRATIVE_IPCA_DATA_DIR`` when set.

    Returns
    -------
    Path
        Absolute path of the folder holding ``reference/`` and ``market/``.
        It is not checked for existence; the loaders raise
        ``FileNotFoundError`` when a file is missing.
    """
    override = os.environ.get(DATA_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data"


def reference_dir() -> Path:
    """``data_dir() / "reference"``: assets, legs, topics and the default link map."""
    return data_dir() / "reference"


def market_dir() -> Path:
    """``data_dir() / "market"``: the market data store of D55."""
    return data_dir() / "market"


# ---------------------------------------------------------------------------
# Cached file reads
# ---------------------------------------------------------------------------
def _file_key(path: Path) -> tuple[str, int, int]:
    """Cache key of a file: resolved path, modification time (ns) and size."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"file not found: {resolved}")
    st = resolved.stat()
    return str(resolved), int(st.st_mtime_ns), int(st.st_size)


@functools.lru_cache(maxsize=64)
def _read_csv_cached(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    # ``keep_default_na=False`` keeps text such as "none", "n/a" or "NA"
    # verbatim and reads empty cells as "" (numeric columns are coerced by
    # the loaders, which report cells that do not parse).
    logger.debug("reading %s (mtime_ns=%d, size=%d)", path, mtime_ns, size)
    return pd.read_csv(path, keep_default_na=False)


def _read_csv(path: Path) -> pd.DataFrame:
    return _read_csv_cached(*_file_key(path)).copy()


def clear_cache() -> None:
    """Drop all cached reference reads (they are re-read on next use)."""
    _read_csv_cached.cache_clear()


def _require_columns(df: pd.DataFrame, required: list[str], what: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{what}: missing columns {missing}; found {list(df.columns)}")


def _as_text(s: pd.Series) -> pd.Series:
    """Text column with missing values as ``""`` and surrounding blanks stripped."""
    return s.astype(object).where(s.notna(), "").astype(str).str.strip()


def _integer_column(s: pd.Series, what: str, problems: list[str]) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    bad = num.isna() | (np.floor(num) != num)
    if bad.any():
        problems.append(f"{what}: non-integer values {sorted(map(str, s[bad].unique()))[:10]}")
        num = num.fillna(0)
    return num.astype(np.int64)


def _raise_if(problems: list[str], what: str) -> None:
    if problems:
        raise ValueError(f"{what} failed validation:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_assets() -> pd.DataFrame:
    """The listed assets of ``reference/assets.csv`` in image order (G.2).

    Returns
    -------
    pandas.DataFrame
        Index ``asset_id`` sorted by ``order`` (the position in the source
        image, 1-based); columns ``order`` (int), ``name``, ``long_leg`` and
        ``short_leg`` (leg ids of ``legs.csv``; the short leg is ``cash`` for
        outrights and "XXX v USD" pairs), ``asset_class`` (one of
        :data:`ASSET_CLASSES`) and ``sub_class``.

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    ValueError
        If columns are missing, ids or orders are duplicated, or an asset
        class is unknown.
    """
    path = reference_dir() / "assets.csv"
    df = _read_csv(path)
    _require_columns(df, ["asset_id"] + _ASSET_COLUMNS, str(path))
    problems: list[str] = []
    df["asset_id"] = _as_text(df["asset_id"])
    for c in ("name", "long_leg", "short_leg", "asset_class", "sub_class"):
        df[c] = _as_text(df[c])
    df["order"] = _integer_column(df["order"], "order", problems)
    dup = df["asset_id"][df["asset_id"].duplicated()].unique().tolist()
    if dup:
        problems.append(f"duplicate asset_id {dup}")
    if (df["asset_id"] == "").any():
        problems.append("empty asset_id")
    dup_order = df["order"][df["order"].duplicated()].unique().tolist()
    if dup_order:
        problems.append(f"duplicate order {dup_order}")
    bad_class = sorted(set(df["asset_class"]) - set(ASSET_CLASSES))
    if bad_class:
        problems.append(f"unknown asset_class {bad_class}; allowed {list(ASSET_CLASSES)}")
    for c in ("long_leg", "short_leg"):
        empty = df.loc[df[c] == "", "asset_id"].tolist()
        if empty:
            problems.append(f"empty {c} for {empty}")
    _raise_if(problems, str(path))
    out = df.sort_values("order", kind="stable").set_index("asset_id")[_ASSET_COLUMNS]
    out.index.name = "asset_id"
    return out


def load_legs() -> pd.DataFrame:
    """The legs of ``reference/legs.csv`` (G.2.1, D54).

    Returns
    -------
    pandas.DataFrame
        Index ``leg_id`` in file order; all columns of the file (at least
        ``name``, ``benchmark_index``, ``proxy_ticker``). Text cells are
        strings, with empty cells as ``""``.

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    ValueError
        If required columns are missing or ``leg_id`` is duplicated.
    """
    path = reference_dir() / "legs.csv"
    df = _read_csv(path)
    _require_columns(df, _LEG_REQUIRED, str(path))
    df["leg_id"] = _as_text(df["leg_id"])
    for c in df.columns:
        if c != "leg_id" and df[c].dtype.kind not in "biuf":
            df[c] = _as_text(df[c])
    dup = df["leg_id"][df["leg_id"].duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"{path}: duplicate leg_id {dup}")
    return df.set_index("leg_id")


def load_topics() -> pd.DataFrame:
    """The 20 manual topics of ``reference/topics.csv`` (G.3, D57).

    Returns
    -------
    pandas.DataFrame
        Index ``topic_id`` sorted by ``order``; columns ``order`` (int),
        ``ontology`` (sector | economic), ``group`` (Sector | Macro | Micro),
        ``name`` and ``scope`` (as printed in the report).

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    ValueError
        If columns are missing or ids or orders are duplicated.
    """
    path = reference_dir() / "topics.csv"
    df = _read_csv(path)
    _require_columns(df, ["topic_id"] + _TOPIC_COLUMNS, str(path))
    problems: list[str] = []
    df["topic_id"] = _as_text(df["topic_id"])
    for c in ("ontology", "group", "name", "scope"):
        df[c] = _as_text(df[c])
    df["order"] = _integer_column(df["order"], "order", problems)
    dup = df["topic_id"][df["topic_id"].duplicated()].unique().tolist()
    if dup:
        problems.append(f"duplicate topic_id {dup}")
    dup_order = df["order"][df["order"].duplicated()].unique().tolist()
    if dup_order:
        problems.append(f"duplicate order {dup_order}")
    _raise_if(problems, str(path))
    out = df.sort_values("order", kind="stable").set_index("topic_id")[_TOPIC_COLUMNS]
    out.index.name = "topic_id"
    return out


def load_default_links() -> pd.DataFrame:
    """The default link map of ``reference/link_map.csv``, validated (G.4 point 1, D58).

    Each row links topic ``topic_id`` to asset ``asset_id`` with a ``tier``
    (strength bucket, one of :data:`~narrative_ipca.exposure_lab.config.TIERS`),
    a ``sign`` (+1 or -1: the direction the asset moves when attention to the
    topic rises) and a one-sentence ``mechanism``.

    Returns
    -------
    pandas.DataFrame
        Columns ``topic_id``, ``asset_id``, ``tier``, ``sign`` (int64) and
        ``mechanism``, in file order with a fresh ``RangeIndex``.

    Raises
    ------
    FileNotFoundError
        If the link map, the topics or the assets file is missing.
    ValueError
        Listing every problem found: unknown topic or asset ids, a tier not
        in ``TIERS``, a sign not in {+1, -1}, duplicate ``(topic_id,
        asset_id)`` pairs, missing columns.
    """
    path = reference_dir() / "link_map.csv"
    df = _read_csv(path)
    _require_columns(df, _LINK_COLUMNS, str(path))
    topics = load_topics()
    assets = load_assets()
    problems: list[str] = []
    for c in ("topic_id", "asset_id", "tier", "mechanism"):
        df[c] = _as_text(df[c])
    df["tier"] = df["tier"].str.lower()

    rows = df.index.to_numpy() + 2  # 1-based file line numbers (header is line 1)
    unknown_topic = ~df["topic_id"].isin(topics.index)
    if unknown_topic.any():
        problems.append(
            f"unknown topic_id {sorted(df.loc[unknown_topic, 'topic_id'].unique())} "
            f"(lines {rows[unknown_topic.to_numpy()].tolist()})"
        )
    unknown_asset = ~df["asset_id"].isin(assets.index)
    if unknown_asset.any():
        problems.append(
            f"unknown asset_id {sorted(df.loc[unknown_asset, 'asset_id'].unique())} "
            f"(lines {rows[unknown_asset.to_numpy()].tolist()})"
        )
    bad_tier = ~df["tier"].isin(TIERS)
    if bad_tier.any():
        problems.append(
            f"tier must be one of {list(TIERS)}; found {sorted(df.loc[bad_tier, 'tier'].unique())} "
            f"(lines {rows[bad_tier.to_numpy()].tolist()})"
        )
    sign = pd.to_numeric(df["sign"], errors="coerce")
    bad_sign = ~sign.isin([1, -1])
    if bad_sign.any():
        problems.append(
            f"sign must be +1 or -1; found {sorted(map(str, df.loc[bad_sign, 'sign'].unique()))} "
            f"(lines {rows[bad_sign.to_numpy()].tolist()})"
        )
    dup = df.duplicated(subset=["topic_id", "asset_id"], keep=False)
    if dup.any():
        pairs = sorted({(t, a) for t, a in zip(df.loc[dup, "topic_id"], df.loc[dup, "asset_id"])})
        problems.append(f"duplicate (topic_id, asset_id) pairs {pairs} (lines {rows[dup.to_numpy()].tolist()})")
    _raise_if(problems, str(path))

    empty_mech = df.loc[df["mechanism"] == "", ["topic_id", "asset_id"]]
    if len(empty_mech):
        logger.warning("load_default_links: %d links without a mechanism sentence", len(empty_mech))
    out = df[_LINK_COLUMNS].copy()
    out["sign"] = sign.astype(np.int64)
    return out.reset_index(drop=True)
