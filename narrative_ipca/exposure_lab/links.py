"""Topics, link map and design matrix of the topic-sensitivity lab (DESIGN.md G.3, G.4; D57-D59, D71).

Three steps, each a pure function:

1. :func:`build_topic_table`: the run's topics, i.e. the chosen manual set of
   the report (G.3 point 1) followed by ``n_generic`` generic topics
   ``G001, G002, ...`` (G.3 point 2). No randomness.
2. :func:`build_link_map`: the ``(topic, asset, tier, sign, mechanism)`` rows
   of G.4: the authored default map for manual topics x listed assets, the
   seeded random map for every other combination, then the session edits.
3. :func:`design_matrix`: ``W_unscaled`` (``L x N``, topics as rows) with
   ``W_{k,n} = sign_{k,n} * beta(tier_{k,n})`` for linked pairs and 0
   elsewhere (G.4), before the feasibility scaling of G.5.1 (done in
   :mod:`.dgp`).

Randomness (D71)
----------------
Every random draw of the link map comes from a per-topic sub-stream
``np.random.default_rng([seed, LINK_STREAM, *code points of topic_id])``
(:func:`topic_rng`), with ``seed = ExposureConfig.seed`` and
``LINK_STREAM = 101``. Each topic's generator first draws a selection
priority ``U(0, 1)`` and then its links. Consequences:

* a topic's random links depend only on the seed, its id, the asset universe
  and the link counts, not on which other topics are in the run; changing the
  manual set leaves every generic topic's draws unchanged;
* the linked generic topics are the ``round(generic_signal_share * n_generic)``
  with the smallest priority (chosen without replacement), so raising the
  share only adds linked topics;
* the topic noise has its own seed (``ExposureConfig.noise_seed``), so the
  links and the noise can be redrawn separately.

Validity boundaries
-------------------
* "Listed asset" means an ``asset_id`` of ``reference/assets.csv``. Manual
  topics get random links only when none of the run's assets is listed (the
  generic universe); with a listed subset they keep their default links to
  the assets present and may end up with no link at all.
* ``round`` is half-up (``0.5 * 5`` topics gives 3), not Python's banker's
  rounding.
* The default map's signs are written for a listed asset "A v B" as long
  leg A and short leg B (D75).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .config import TIERS, ExposureConfig, RandomLinkConfig, TopicSetConfig
from .types import LinkMap, TopicTable

logger = logging.getLogger(__name__)

__all__ = [
    "LINK_STREAM",
    "LINK_COLUMNS",
    "TOPIC_COLUMNS",
    "RANDOM_MECHANISM",
    "OVERRIDE_MECHANISM",
    "generic_topic_id",
    "topic_rng",
    "build_topic_table",
    "build_link_map",
    "design_matrix",
]

#: Stream id of the random link map (D71); the other lab streams are in :mod:`.dgp`.
LINK_STREAM = 101

#: Columns of :attr:`LinkMap.table` (G.4).
LINK_COLUMNS: tuple[str, ...] = ("topic_id", "asset_id", "tier", "sign", "mechanism", "origin")

#: Columns of :attr:`TopicTable.table` (G.3).
TOPIC_COLUMNS: tuple[str, ...] = ("name", "group", "ontology", "scope", "order")

#: Mechanism text of seeded random links and of session edits.
RANDOM_MECHANISM = "Random link (seeded)"
OVERRIDE_MECHANISM = "Session edit"

_MANUAL_ONTOLOGY = {"sector": ("sector",), "economic": ("economic",), "both": ("sector", "economic"), "none": ()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generic_topic_id(j: int) -> str:
    """Id of the ``j``-th generic topic (1-based): ``G001, G002, ...`` (G.3 point 2)."""
    return f"G{int(j):03d}"


def topic_rng(seed: int, stream: int, topic_id: str) -> np.random.Generator:
    """Per-topic random generator ``default_rng([seed, stream, *code points of topic_id])`` (D71).

    Parameters
    ----------
    seed:
        The component seed (``ExposureConfig.seed`` for links,
        ``ExposureConfig.noise_seed`` for noise and attention); must be >= 0.
    stream:
        Fixed integer per random component (links 101, noise 202, attention
        303), so components never share a stream.
    topic_id:
        The topic's id; its Unicode code points extend the seed sequence, so
        a topic's draws do not depend on the other topics of the run.
    """
    if int(seed) < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return np.random.default_rng([int(seed), int(stream), *(ord(c) for c in str(topic_id))])


def _half_up(x: float) -> int:
    """``round`` half-up, robust to float noise such as ``0.2 * 25 = 5.000000000000001``."""
    return int(math.floor(round(float(x), 9) + 0.5))


def _reference():
    # Imported lazily so this module imports even when the reference data or
    # its loader is unavailable (G.13).
    from narrative_ipca.exposure_lab import reference

    return reference


def _empty_links() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "topic_id": pd.Series([], dtype="str"),
            "asset_id": pd.Series([], dtype="str"),
            "tier": pd.Series([], dtype="str"),
            "sign": pd.Series([], dtype=np.int64),
            "mechanism": pd.Series([], dtype="str"),
            "origin": pd.Series([], dtype="str"),
        }
    )


# ---------------------------------------------------------------------------
# Topics (G.3)
# ---------------------------------------------------------------------------
def build_topic_table(cfg: TopicSetConfig) -> TopicTable:
    """The run's topics: the manual set, then ``cfg.n_generic`` generic topics (G.3, D57).

    Parameters
    ----------
    cfg:
        :class:`~narrative_ipca.exposure_lab.config.TopicSetConfig`. ``manual``
        selects the report topics by ontology (``sector``: S1-S11,
        ``economic``: A1-A6 and B1-B3, ``both``: all 20, ``none``: none);
        ``n_generic`` adds topics ``G001 ... G{n_generic}``.

    Returns
    -------
    TopicTable
        ``table`` with index ``topic_id`` in display order (manual topics in
        report order, then generic topics) and columns ``name``, ``group``
        (Sector | Macro | Micro | Generic), ``ontology`` (sector | economic |
        generic), ``scope`` (``""`` for generic topics) and ``order``
        (1..L, the position in the run).
    """
    frames: list[pd.DataFrame] = []
    ontologies = _MANUAL_ONTOLOGY[cfg.manual]
    if ontologies:
        ref = _reference().load_topics()
        manual = ref.loc[ref["ontology"].isin(ontologies)].sort_values("order", kind="stable")
        if len(manual) != cfg.n_manual:
            logger.warning(
                "build_topic_table: manual set %r has %d topics in topics.csv, expected %d",
                cfg.manual, len(manual), cfg.n_manual,
            )
        frames.append(manual[["name", "group", "ontology", "scope"]].astype(str))
    n = int(cfg.n_generic)
    if n > 0:
        ids = [generic_topic_id(j) for j in range(1, n + 1)]
        frames.append(
            pd.DataFrame(
                {
                    "name": [f"Generic topic {j:03d}" for j in range(1, n + 1)],
                    "group": "Generic",
                    "ontology": "generic",
                    "scope": "",
                },
                index=pd.Index(ids, name="topic_id"),
            ).astype(str)
        )
    table = pd.concat(frames) if len(frames) > 1 else frames[0].copy()
    table.index = pd.Index([str(i) for i in table.index], name="topic_id")
    table["order"] = np.arange(1, len(table) + 1, dtype=np.int64)
    table = table[list(TOPIC_COLUMNS)]
    logger.info("build_topic_table: %d manual (%s) + %d generic topics", len(table) - n, cfg.manual, n)
    return TopicTable(table=table)


# ---------------------------------------------------------------------------
# Link map (G.4)
# ---------------------------------------------------------------------------
def _random_links(rng: np.random.Generator, n_assets: int, counts: RandomLinkConfig) -> list[tuple[int, str, int]]:
    """Draw one topic's random links: ``(asset position, tier, sign)`` (G.4 point 2).

    ``n_strong`` strong, ``n_moderate`` moderate and ``n_weak`` weak links to
    distinct assets drawn without replacement; when there are fewer assets
    than links, the tiers are filled strong first. Signs are +1 or -1 with
    probability 1/2 each.
    """
    tiers = ["strong"] * int(counts.n_strong) + ["moderate"] * int(counts.n_moderate) + ["weak"] * int(counts.n_weak)
    n_total = min(len(tiers), int(n_assets))
    if n_total <= 0:
        return []
    picks = rng.choice(int(n_assets), size=n_total, replace=False)
    signs = np.where(rng.random(n_total) < 0.5, 1, -1)
    return [(int(p), tiers[i], int(s)) for i, (p, s) in enumerate(zip(picks, signs))]


def build_link_map(
    topics: TopicTable,
    assets: pd.DataFrame,
    topic_cfg: TopicSetConfig,
    exposure_cfg: ExposureConfig,
) -> LinkMap:
    """The run's topic-asset links: default map, seeded random map, session edits (G.4, D58).

    Parameters
    ----------
    topics:
        The run's topics (:func:`build_topic_table`).
    assets:
        Asset table with index ``asset_id`` in display order (for example
        ``MarketData.assets``).
    topic_cfg:
        Supplies ``generic_signal_share``: the share of generic topics that
        carry links.
    exposure_cfg:
        Supplies ``random_links`` (links per linked topic), ``seed`` (stream
        ``[seed, 101]``) and ``link_overrides`` (session edits
        ``(topic_id, asset_id, tier, sign)``; tier ``"none"`` removes a link).

    Returns
    -------
    LinkMap
        ``table`` with columns ``topic_id``, ``asset_id``, ``tier``, ``sign``
        (int64, +1 or -1), ``mechanism`` and ``origin`` (default | random |
        override), one row per linked pair, sorted by topic order and then
        asset order.

    Notes
    -----
    1. Default rows: ``reference/link_map.csv`` restricted to the run's
       topics and assets (``origin = "default"``).
    2. Random rows (``origin = "random"``): every manual topic when none of
       the run's assets is listed, and ``round(generic_signal_share *
       n_generic)`` generic topics; draws per topic as in the module
       docstring.
    3. Overrides in order; an override naming a topic or asset outside the
       run is ignored with a log message.
    """
    topic_ids = topics.ids
    asset_ids = [str(a) for a in assets.index]
    topic_pos = {t: i for i, t in enumerate(topic_ids)}
    asset_pos = {a: i for i, a in enumerate(asset_ids)}
    ontology = topics.table["ontology"].astype(str)
    rows: dict[tuple[str, str], dict[str, object]] = {}

    # 1) default map for manual topics x listed assets (reference files are
    #    read only when the run has manual topics)
    manual_ids = [t for t in topic_ids if ontology.loc[t] != "generic"]
    n_default = 0
    manual_random = False
    if manual_ids:
        ref = _reference()
        defaults = ref.load_default_links()
        keep = defaults["topic_id"].isin(topic_pos) & defaults["asset_id"].isin(asset_pos)
        for rec in defaults.loc[keep].itertuples(index=False):
            rows[(str(rec.topic_id), str(rec.asset_id))] = {
                "tier": str(rec.tier), "sign": int(rec.sign), "mechanism": str(rec.mechanism), "origin": "default",
            }
        n_default = int(keep.sum())
        listed = set(map(str, ref.load_assets().index))
        manual_random = not any(a in listed for a in asset_ids)  # generic universe

    # 2) seeded random map
    generic_ids = [t for t in topic_ids if ontology.loc[t] == "generic"]
    seed = int(exposure_cfg.seed)
    draws: dict[str, tuple[float, list[tuple[int, str, int]]]] = {}
    candidates = generic_ids + (manual_ids if manual_random else [])
    for tid in candidates:
        rng = topic_rng(seed, LINK_STREAM, tid)
        priority = float(rng.random())
        draws[tid] = (priority, _random_links(rng, len(asset_ids), exposure_cfg.random_links))
    n_signal = _half_up(float(topic_cfg.generic_signal_share) * len(generic_ids))
    if len(generic_ids) != int(topic_cfg.n_generic):
        logger.warning(
            "build_link_map: %d generic topics in the table but n_generic=%d; using the table",
            len(generic_ids), int(topic_cfg.n_generic),
        )
    ranked = sorted(generic_ids, key=lambda t: (draws[t][0], topic_pos[t]))
    linked_random = set(ranked[:n_signal]) | (set(manual_ids) if manual_random else set())
    n_random = 0
    for tid in topic_ids:
        if tid not in linked_random:
            continue
        for p, tier, sign in draws[tid][1]:
            rows[(tid, asset_ids[p])] = {"tier": tier, "sign": sign, "mechanism": RANDOM_MECHANISM, "origin": "random"}
            n_random += 1

    # 3) session edits
    n_override = n_removed = n_ignored = 0
    for t, a, tier, sign in exposure_cfg.link_overrides:
        t, a, tier = str(t), str(a), str(tier)
        if t not in topic_pos or a not in asset_pos:
            n_ignored += 1
            logger.info("build_link_map: override (%s, %s, %s, %s) ignored: pair not in this run", t, a, tier, sign)
            continue
        if tier == "none":
            if rows.pop((t, a), None) is not None:
                n_removed += 1
            else:
                logger.info("build_link_map: override removes (%s, %s), which is not linked", t, a)
            continue
        rows[(t, a)] = {"tier": tier, "sign": int(sign), "mechanism": OVERRIDE_MECHANISM, "origin": "override"}
        n_override += 1

    if rows:
        keys = sorted(rows, key=lambda ta: (topic_pos[ta[0]], asset_pos[ta[1]]))
        table = pd.DataFrame(
            {
                "topic_id": [k[0] for k in keys],
                "asset_id": [k[1] for k in keys],
                "tier": [rows[k]["tier"] for k in keys],
                "sign": np.array([rows[k]["sign"] for k in keys], dtype=np.int64),
                "mechanism": [rows[k]["mechanism"] for k in keys],
                "origin": [rows[k]["origin"] for k in keys],
            }
        )
    else:
        table = _empty_links()
    logger.info(
        "build_link_map: %d links (%d default, %d random over %d topics, %d overrides set, %d removed, %d ignored); "
        "manual topics use the %s map",
        len(table), n_default, n_random, len(linked_random), n_override, n_removed, n_ignored,
        "random" if manual_random else "default",
    )
    return LinkMap(table=table)


# ---------------------------------------------------------------------------
# Design matrix (G.4)
# ---------------------------------------------------------------------------
def design_matrix(
    links: LinkMap,
    topics: TopicTable,
    assets: pd.DataFrame,
    exposure_cfg: ExposureConfig,
) -> pd.DataFrame:
    """Design matrix ``W_unscaled`` (``L x N``): ``sign * beta(tier)`` on linked pairs, 0 elsewhere (G.4, D59).

    Parameters
    ----------
    links:
        The run's links (:func:`build_link_map`).
    topics, assets:
        Row order (``topics.ids``) and column order (``assets.index``).
    exposure_cfg:
        ``tier_values()`` maps each tier to its exposure value after merging
        by ``n_betas`` (G.4 table), in standardised (correlation) units.

    Returns
    -------
    pandas.DataFrame
        Float matrix with index ``topic_id`` and columns ``asset_id``. Links
        naming a topic or asset outside the run are dropped with a warning;
        for a duplicated pair the last row wins.
    """
    topic_ids = topics.ids
    asset_ids = [str(a) for a in assets.index]
    values = exposure_cfg.tier_values()
    W = np.zeros((len(topic_ids), len(asset_ids)), dtype=float)
    t = links.table
    if len(t):
        bad_tier = sorted(set(map(str, t["tier"])) - set(TIERS))
        if bad_tier:
            raise ValueError(f"design_matrix: unknown tiers {bad_tier}; allowed {list(TIERS)}")
        ti = t["topic_id"].astype(str).map({k: i for i, k in enumerate(topic_ids)})
        ai = t["asset_id"].astype(str).map({k: i for i, k in enumerate(asset_ids)})
        keep = (ti.notna() & ai.notna()).to_numpy()
        if not keep.all():
            logger.warning("design_matrix: %d links name topics or assets outside the run; dropped", int((~keep).sum()))
        dup = t[["topic_id", "asset_id"]].astype(str).duplicated(keep="last").to_numpy()
        if (dup & keep).any():
            logger.warning("design_matrix: duplicated (topic, asset) pairs; the last row wins")
        keep = keep & ~dup
        val = t["sign"].to_numpy(dtype=float) * t["tier"].astype(str).map(values).to_numpy(dtype=float)
        W[ti.to_numpy()[keep].astype(int), ai.to_numpy()[keep].astype(int)] = val[keep]
    return pd.DataFrame(W, index=pd.Index(topic_ids, name="topic_id"), columns=pd.Index(asset_ids, name="asset_id"))
