"""Configuration of the topic-sensitivity lab (DESIGN.md Part G).

One frozen dataclass per stage, collected in :class:`LabConfig`. Every stage
of the lab is a pure function of its inputs and of the sub-configuration it
depends on, so :meth:`LabConfig.key` gives a cache key per stage (D71).

Symbols follow DESIGN.md Part G: ``k`` topics, ``n`` assets, ``t`` trading
days; ``W`` the design matrix of set sensitivities (G.4); ``l`` the lead in
days (G.5); ``w`` the shock window (D9). In code, "exposure" means topic
sensitivity (see :mod:`narrative_ipca.exposure_lab`).

Every config normalises its values on construction (D77): numbers become
``float`` or ``int`` as annotated, dates become ISO strings and lists become
tuples. A config read from JSON or YAML (where ``0`` is an ``int`` and a pair
is a list) therefore equals, hashes and keys like the one built in code.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, fields
from numbers import Integral, Real
from typing import Any, Literal

import pandas as pd

from narrative_ipca.config import config_from_dict, config_hash, config_to_dict

__all__ = [
    "DATA_START",
    "DATA_END",
    "TIERS",
    "N_MANUAL",
    "UniverseConfig",
    "TopicSetConfig",
    "RandomLinkConfig",
    "ExposureConfig",
    "AttentionConfig",
    "WindowConfig",
    "DirectConfig",
    "BKSLabConfig",
    "LabConfig",
    "STAGES",
]

#: First and last return day of the lab calendar (weekdays; G.2.1).
DATA_START = "2015-01-02"
DATA_END = "2025-12-31"

#: Link tiers, strongest first (G.4).
TIERS: tuple[str, str, str] = ("strong", "moderate", "weak")

#: Max counts (G.2, G.3).
MAX_GENERIC_ASSETS = 500
MAX_GENERIC_TOPICS = 500

#: Number of topics in each manual set of the report (G.3 point 1).
N_MANUAL: dict[str, int] = {"none": 0, "sector": 11, "economic": 9, "both": 20}


# ---------------------------------------------------------------------------
# Value normalisation (D77)
# ---------------------------------------------------------------------------
def _as_int(name: str, v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, Integral):
        return int(v)
    if isinstance(v, Real):
        if float(v).is_integer():
            return int(v)
        raise ValueError(f"{name} must be an integer, got {v!r}")
    return v


def _as_float(v: Any) -> Any:
    if isinstance(v, Real) and not isinstance(v, bool):
        return float(v)
    return v


def _as_iso(v: Any) -> Any:
    if isinstance(v, (dt.date, dt.datetime, pd.Timestamp)):
        return pd.Timestamp(v).date().isoformat()
    return v


def _as_override(ov: Any) -> Any:
    if isinstance(ov, (list, tuple)) and len(ov) == 4:
        t, a, tier, sign = ov
        return (str(t), str(a), str(tier), _as_int("override sign", sign))
    return tuple(ov) if isinstance(ov, list) else ov


def _normalise(obj: Any) -> None:
    """Coerce the fields of a frozen config dataclass to their annotated types (D77).

    Handles the annotations used in this module: ``int``, ``float``,
    ``float | None``, ``bool``, ``str`` (dates to ISO strings),
    ``tuple[str, ...] | None`` and the link-override tuples. Other fields
    (``Literal`` values, nested configs) are left as they are; the
    ``__post_init__`` checks then validate the result.
    """
    for f in fields(obj):
        v = getattr(obj, f.name)
        ann = str(f.type).replace(" ", "")
        new = v
        if ann == "int":
            new = _as_int(f.name, v)
        elif ann in ("float", "float|None"):
            new = None if v is None else _as_float(v)
        elif ann == "bool":
            new = bool(v) if isinstance(v, (bool, Integral)) else v
        elif ann == "str":
            new = _as_iso(v)
        elif ann == "tuple[str,...]|None":
            new = None if v is None else tuple(str(x) for x in v)
        elif ann == "tuple[tuple[str,str,str,int],...]":
            new = tuple(_as_override(o) for o in v)
        if new is not v:
            object.__setattr__(obj, f.name, new)


# ---------------------------------------------------------------------------
# Universe (G.2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UniverseConfig:
    """Which assets and which price source.

    Attributes
    ----------
    asset_source:
        ``"listed"`` uses the 55 assets of ``data/reference/assets.csv``;
        ``"generic"`` uses ``n_generic_assets`` synthetic assets (G.2.2).
    listed_assets:
        Subset of listed ``asset_id`` values, kept in reference (image)
        order. ``None`` means all listed assets.
    price_source:
        ``"real"`` reads ``data/market/`` (a listed asset whose leg failed is
        filled with artificial returns and flagged); ``"artificial"``
        generates every listed asset (G.2.2). Ignored for generic assets.
    n_generic_assets:
        Number of generic assets (2-500).
    start, end:
        First and last return day (ISO dates on the weekday calendar).
    seed:
        Seed of the artificial and generic return generator.
    """

    asset_source: Literal["listed", "generic"] = "listed"
    listed_assets: tuple[str, ...] | None = None
    price_source: Literal["real", "artificial"] = "real"
    n_generic_assets: int = 55
    start: str = DATA_START
    end: str = DATA_END
    seed: int = 0

    def __post_init__(self) -> None:
        _normalise(self)
        if self.asset_source not in ("listed", "generic"):
            raise ValueError("asset_source must be 'listed' or 'generic'")
        if self.price_source not in ("real", "artificial"):
            raise ValueError("price_source must be 'real' or 'artificial'")
        if not 2 <= int(self.n_generic_assets) <= MAX_GENERIC_ASSETS:
            raise ValueError(f"n_generic_assets must be in [2, {MAX_GENERIC_ASSETS}]")
        if self.listed_assets is not None:
            if len(self.listed_assets) < 2:
                raise ValueError("listed_assets needs at least 2 assets")
            if len(set(self.listed_assets)) != len(self.listed_assets):
                raise ValueError("listed_assets contains duplicates")
        if pd.Timestamp(self.start) >= pd.Timestamp(self.end):
            raise ValueError("start must be before end")


# ---------------------------------------------------------------------------
# Topics (G.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TopicSetConfig:
    """Which topics.

    Attributes
    ----------
    manual:
        Manual topic set from the report: ``"none"``, ``"sector"`` (S1-S11),
        ``"economic"`` (A1-A6, B1-B3) or ``"both"`` (20 topics).
    n_generic:
        Number of generic topics ``G001...`` (0-500).
    generic_signal_share:
        Share of generic topics that carry links (G.4); the rest are noise.
    """

    manual: Literal["none", "sector", "economic", "both"] = "both"
    n_generic: int = 0
    generic_signal_share: float = 0.2

    def __post_init__(self) -> None:
        _normalise(self)
        if self.manual not in N_MANUAL:
            raise ValueError("manual must be one of none, sector, economic, both")
        if not 0 <= int(self.n_generic) <= MAX_GENERIC_TOPICS:
            raise ValueError(f"n_generic must be in [0, {MAX_GENERIC_TOPICS}]")
        if not 0.0 <= float(self.generic_signal_share) <= 1.0:
            raise ValueError("generic_signal_share must be in [0, 1]")
        if self.manual == "none" and int(self.n_generic) < 1:
            raise ValueError("at least one topic is needed: choose a manual set or n_generic >= 1")

    @property
    def n_manual(self) -> int:
        return N_MANUAL[self.manual]

    @property
    def n_topics(self) -> int:
        return self.n_manual + int(self.n_generic)


# ---------------------------------------------------------------------------
# Links and exposure values (G.4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RandomLinkConfig:
    """Links per linked topic in the seeded random map (G.4 point 2)."""

    n_strong: int = 1
    n_moderate: int = 2
    n_weak: int = 3

    def __post_init__(self) -> None:
        _normalise(self)
        for name in ("n_strong", "n_moderate", "n_weak"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.n_strong + self.n_moderate + self.n_weak < 1:
            raise ValueError("a linked topic needs at least one link")


@dataclass(frozen=True)
class ExposureConfig:
    """Set sensitivities ("betas") and the link structure (G.4, G.5.1).

    Naming: in code, "exposure" means topic sensitivity, the expected return
    response of an asset to a one-standard-deviation attention shock in a
    topic, with the other topics' shocks held fixed (not a position size or
    dollar exposure). The class, the field ``LabConfig.exposure`` and the
    stage-key part ``"exposure"`` keep the older name. The values set here
    are the set sensitivities: each link's ``sign * beta(tier)`` gives the
    design matrix ``W`` (:func:`.links.design_matrix`).

    Attributes
    ----------
    n_betas:
        Number of distinct values: 1 (all tiers use ``beta_1``), 2 (strong
        ``beta_1``, moderate and weak ``beta_2``) or 3 (one per tier).
    beta_1, beta_2, beta_3:
        Values in standardised (correlation) units (D59). Defaults are the
        plan's strong / moderate / weak buckets.
    lead_days:
        ``l`` in G.5: 0 = same day (risk reading), 1 = attention on day ``t``
        relates to returns on day ``t+1`` (signal-grade reading).
    noise_df:
        Degrees of freedom of the Student-t topic noise ``u`` (unit
        variance); ``0`` means Gaussian.
    link_overrides:
        Session edits on top of the default map, as tuples
        ``(topic_id, asset_id, tier, sign)``; ``tier == "none"`` removes the
        link. Applied after the default and random maps.
    random_links:
        Counts for the seeded random map.
    seed:
        Seed of the random link map (stream 101). It changes only random
        links (generic topics, or manual topics on generic assets).
    noise_seed:
        Seed of the topic news noise ``u`` (stream 202) and of the attention
        components ``m`` and ``g`` (stream 303). Changing it redraws the
        noise with the links held fixed (D71).
    """

    n_betas: int = 3
    beta_1: float = 0.35
    beta_2: float = 0.15
    beta_3: float = 0.05
    lead_days: int = 0
    noise_df: float = 5.0
    link_overrides: tuple[tuple[str, str, str, int], ...] = ()
    random_links: RandomLinkConfig = field(default_factory=RandomLinkConfig)
    seed: int = 0
    noise_seed: int = 0

    def __post_init__(self) -> None:
        _normalise(self)
        for name in ("seed", "noise_seed"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        if int(self.n_betas) not in (1, 2, 3):
            raise ValueError("n_betas must be 1, 2 or 3")
        for name in ("beta_1", "beta_2", "beta_3"):
            v = float(getattr(self, name))
            if not 0.0 <= v <= 0.95:
                raise ValueError(f"{name} must be in [0, 0.95] (standardised units)")
        if int(self.lead_days) not in (0, 1):
            raise ValueError("lead_days must be 0 or 1")
        if not (float(self.noise_df) == 0.0 or float(self.noise_df) > 2.0):
            raise ValueError("noise_df must be 0 (Gaussian) or > 2")
        for ov in self.link_overrides:
            if len(ov) != 4:
                raise ValueError("link_overrides entries are (topic_id, asset_id, tier, sign)")
            if ov[2] not in TIERS + ("none",):
                raise ValueError(f"override tier must be one of {TIERS + ('none',)}")
            if int(ov[3]) not in (-1, 1):
                raise ValueError("override sign must be +1 or -1")

    def tier_values(self) -> dict[str, float]:
        """Exposure value per tier after merging by ``n_betas`` (G.4 table)."""
        b1, b2, b3 = float(self.beta_1), float(self.beta_2), float(self.beta_3)
        if int(self.n_betas) == 1:
            return {"strong": b1, "moderate": b1, "weak": b1}
        if int(self.n_betas) == 2:
            return {"strong": b1, "moderate": b2, "weak": b2}
        return {"strong": b1, "moderate": b2, "weak": b3}


# ---------------------------------------------------------------------------
# Attention levels (G.5.2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AttentionConfig:
    """Simulated attention levels ``a = m + g + kappa * s`` (G.5.2)."""

    level_low: float = 0.15
    level_high: float = 0.35
    kappa: float = 0.02
    slow_ar1: float = 0.995
    slow_sd_ratio: float = 0.1
    clip_sd: float = 8.0
    feasibility_cap: float = 0.95

    def __post_init__(self) -> None:
        _normalise(self)
        if not 0.0 < self.level_low < self.level_high:
            raise ValueError("need 0 < level_low < level_high")
        if self.kappa <= 0:
            raise ValueError("kappa must be > 0")
        if not 0.0 <= self.slow_ar1 < 1.0:
            raise ValueError("slow_ar1 must be in [0, 1)")
        if self.slow_sd_ratio < 0:
            raise ValueError("slow_sd_ratio must be >= 0")
        if self.clip_sd <= 0:
            raise ValueError("clip_sd must be > 0")
        if not 0.0 < self.feasibility_cap < 1.0:
            raise ValueError("feasibility_cap must be in (0, 1)")


# ---------------------------------------------------------------------------
# Windows (G.6)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowConfig:
    """Training and forecast windows, and the shock window ``w``.

    The forecast window holds the weekdays in
    ``[forecast_start, forecast_start + 7 * forecast_weeks days)``.
    ``min_train_days`` is the shortest training window allowed: 21 weekdays,
    about one month (D81). Short windows are allowed but noisy: an exposure's
    standard error is about ``1 / sqrt(n)`` in standardised units. BKS needs
    at least :data:`narrative_ipca.exposure_lab.bks.MIN_TRAIN_PERIODS` training
    weeks on its own.
    """

    train_start: str = DATA_START
    train_end: str = "2022-12-30"
    forecast_start: str = "2023-01-02"
    forecast_weeks: int = 4
    shock_window: int = 5
    min_train_days: int = 21

    def __post_init__(self) -> None:
        _normalise(self)
        ts, te, fs = (pd.Timestamp(self.train_start), pd.Timestamp(self.train_end),
                      pd.Timestamp(self.forecast_start))
        if not ts < te:
            raise ValueError("train_start must be before train_end")
        if not te < fs:
            raise ValueError("forecast_start must be after train_end")
        if not 1 <= int(self.forecast_weeks) <= 12:
            raise ValueError("forecast_weeks must be between 1 and 12")
        if int(self.shock_window) < 1:
            raise ValueError("shock_window must be >= 1")
        if len(pd.bdate_range(ts, te)) < int(self.min_train_days):
            raise ValueError(f"the training window needs at least {self.min_train_days} weekdays")

    @property
    def forecast_end(self) -> pd.Timestamp:
        """Last calendar day inside the forecast window (inclusive)."""
        return pd.Timestamp(self.forecast_start) + pd.Timedelta(days=7 * int(self.forecast_weeks) - 1)


# ---------------------------------------------------------------------------
# Estimators (G.7)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DirectConfig:
    """Direct exposure regression (G.7.1)."""

    method: Literal["elastic_net", "ridge", "ols", "oracle"] = "elastic_net"
    penalty: Literal["universal", "fixed", "cv"] = "universal"
    alpha: float = 0.05
    l1_ratio: float = 0.9
    ridge_lambda: float | None = None
    select_tau: float = 0.05
    max_iter: int = 5000
    cv_folds: int = 5

    def __post_init__(self) -> None:
        _normalise(self)
        if self.method not in ("elastic_net", "ridge", "ols", "oracle"):
            raise ValueError("unknown method")
        if self.penalty not in ("universal", "fixed", "cv"):
            raise ValueError("unknown penalty rule")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if not 0.0 < self.l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be in (0, 1]")
        if self.ridge_lambda is not None and self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be >= 0")
        if self.select_tau < 0:
            raise ValueError("select_tau must be >= 0")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")


@dataclass(frozen=True)
class BKSLabConfig:
    """BKS Sparse IPCA as run by the lab (G.7.2, D70, D88).

    Attributes (the less obvious ones)
    ----------------------------------
    history:
        Which data the BKS panel is built from (D88):

        * ``"full"`` (default): every day from the start of the data. The
          instruments of a training week are kernel covariances that weigh the
          whole history before it, and ``inverse_vol`` divides each return by
          its trailing 252-day volatility. The first ``burn_in_weeks`` weekly
          instrument periods are dropped (D17), and an instrument needs
          ``min_days`` observed days.
        * ``"training"``: only the training window. Returns from
          ``train_start`` on and attention from ``w`` weekdays before
          ``train_start`` on, so the first shock falls on ``train_start`` and
          BKS sees the data the direct methods see. ``inverse_vol`` divides by
          the asset's training standard deviation (population, over the
          training return days; frozen for the forecast weeks), because a
          trailing volatility would need about 63 days of warm-up inside the
          window. The burn-in and the day minimum are
          ``burn_in_weeks_training`` and ``min_days_training``.
    burn_in_weeks_training, min_days_training:
        The training variant's burn-in (weekly instrument periods dropped at
        the start) and fewest observed days per instrument. The defaults, 0
        and 3, keep 24 of the 26 week ends of the dashboard's default window
        (2025-01-01 to 2025-06-30), so the fit meets
        :data:`narrative_ipca.exposure_lab.bks.MIN_TRAIN_PERIODS`; 3 is the
        largest minimum that keeps every six-month window with a month-end
        cut-off from 2016 to 2025 at 24 weeks for both leads (5, one trading
        week, would refuse 3 same-day and 12 next-day windows of the 119).
        The trade-off: the first training weeks' instruments are covariances
        over a few days to a few weeks of data, far noisier than the full
        variant's (at least 52 weeks).
    """

    K: int = 3
    half_life_months: float = 69.0
    lambda_rule: Literal["tolerance", "argmax", "fixed"] = "tolerance"
    tolerance: float = 0.02
    lam: float | None = None
    n_lambdas: int = 12
    lambda_ratio: float = 1e-2
    penalize_intercept: bool = True
    asset_weighting: Literal["none", "inverse_vol"] = "inverse_vol"
    burn_in_weeks: int = 52
    min_days: int = 60
    max_iter: int = 300
    history: Literal["full", "training"] = "full"
    burn_in_weeks_training: int = 0
    min_days_training: int = 3

    def __post_init__(self) -> None:
        _normalise(self)
        if not 1 <= int(self.K) <= 10:
            raise ValueError("K must be in [1, 10]")
        if self.half_life_months <= 0:
            raise ValueError("half_life_months must be > 0")
        if self.lambda_rule not in ("tolerance", "argmax", "fixed"):
            raise ValueError("unknown lambda_rule")
        if self.lambda_rule == "fixed" and (self.lam is None or self.lam < 0):
            raise ValueError("lambda_rule='fixed' needs lam >= 0")
        if not 0.0 <= self.tolerance < 1.0:
            raise ValueError("tolerance must be in [0, 1)")
        if int(self.n_lambdas) < 2:
            raise ValueError("n_lambdas must be >= 2")
        if not 0.0 < self.lambda_ratio < 1.0:
            raise ValueError("lambda_ratio must be in (0, 1)")
        if self.history not in ("full", "training"):
            raise ValueError("history must be 'full' or 'training'")
        if int(self.burn_in_weeks) < 0 or int(self.burn_in_weeks_training) < 0:
            raise ValueError("burn_in_weeks and burn_in_weeks_training must be >= 0")
        if int(self.min_days) < 2 or int(self.min_days_training) < 2:
            raise ValueError("min_days and min_days_training must be >= 2")

    @property
    def xi_weekly(self) -> float:
        """Weekly kernel decay with the given half-life: ``0.5 ** (1 / weeks)``."""
        return float(0.5 ** (1.0 / (self.half_life_months * 52.0 / 12.0)))

    @property
    def panel_burn_in_weeks(self) -> int:
        """Burn-in of the panel this config builds: ``burn_in_weeks`` or, under ``"training"``, ``burn_in_weeks_training``."""
        return int(self.burn_in_weeks_training if self.history == "training" else self.burn_in_weeks)

    @property
    def panel_min_days(self) -> int:
        """Fewest observed days per instrument: ``min_days`` or, under ``"training"``, ``min_days_training``."""
        return int(self.min_days_training if self.history == "training" else self.min_days)


# ---------------------------------------------------------------------------
# The whole lab
# ---------------------------------------------------------------------------
#: Stage names and the sub-configurations each depends on (D71).
STAGES: dict[str, tuple[str, ...]] = {
    "market": ("universe",),
    "simulation": ("universe", "topics", "exposure", "attention"),
    "shocks": ("universe", "topics", "exposure", "attention", "window_train"),
    "direct": ("universe", "topics", "exposure", "attention", "window_train", "direct"),
    "evaluation": ("universe", "topics", "exposure", "attention", "window", "direct"),
    "bks_panel": ("universe", "topics", "exposure", "attention", "window_shock", "bks_panel"),
    "bks_fit": ("universe", "topics", "exposure", "attention", "window_train", "bks"),
    "bks_evaluation": ("universe", "topics", "exposure", "attention", "window", "bks"),
}


@dataclass(frozen=True)
class LabConfig:
    """Everything that defines one lab run."""

    universe: UniverseConfig = field(default_factory=UniverseConfig)
    topics: TopicSetConfig = field(default_factory=TopicSetConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    direct: DirectConfig = field(default_factory=DirectConfig)
    bks: BKSLabConfig = field(default_factory=BKSLabConfig)

    def to_dict(self) -> dict[str, Any]:
        return config_to_dict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabConfig":
        return config_from_dict(cls, d)

    def hash(self) -> str:
        return config_hash(self)

    def _part(self, name: str) -> Any:
        w = self.window
        if name == "window_train":
            return {"train_start": w.train_start, "train_end": w.train_end, "shock_window": w.shock_window}
        if name == "window_shock":
            # the training-history panel starts at train_start and scales by the training std (D88)
            if self.bks.history == "training":
                return {"shock_window": w.shock_window, "train_start": w.train_start, "train_end": w.train_end}
            return {"shock_window": w.shock_window}
        if name == "bks_panel":
            b = self.bks
            return {"history": b.history, "half_life_months": b.half_life_months, "asset_weighting": b.asset_weighting,
                    "burn_in_weeks": b.panel_burn_in_weeks, "min_days": b.panel_min_days}
        return getattr(self, name)

    def key(self, stage: str) -> str:
        """Cache key of ``stage``: a hash of the sub-configurations it depends on (D71)."""
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}; known: {sorted(STAGES)}")
        parts = {name: config_to_dict(self._part(name)) for name in STAGES[stage]}
        return f"{stage}-{config_hash(parts)}"
