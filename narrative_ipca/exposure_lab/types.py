"""Typed containers passed between the lab stages (DESIGN.md Part G).

Conventions: returns are daily simple returns in decimals on the weekday
calendar; topic x asset matrices are ``DataFrame`` with topics as rows
(index ``topic_id``) and assets as columns (``asset_id``); sensitivities are
in standardised units unless a name ends in ``_pct`` (percent per
one-standard-deviation shock).

Naming: in code, "exposure" means topic sensitivity, the expected return
response of an asset to a one-standard-deviation attention shock in a topic,
with the other topics' shocks held fixed (not a position size or dollar
exposure; see :mod:`narrative_ipca.exposure_lab`). ``SimTruth.B_true`` is the
true sensitivity (the population value of the simulation) and
``DirectFit.B_hat`` the estimated sensitivity (a method's training-window
estimate); ``SimTruth.W`` holds the set sensitivities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

__all__ = [
    "MarketData",
    "TopicTable",
    "LinkMap",
    "SimTruth",
    "SimData",
    "ObservedShocks",
    "DirectFit",
    "WindowEval",
    "BKSLabResult",
]


@dataclass
class MarketData:
    """Asset returns and the asset table (G.2).

    Attributes
    ----------
    returns:
        ``(n_days, N)`` daily simple returns; ``NaN`` where not observed.
    assets:
        Index ``asset_id`` in display order; columns ``name``, ``asset_class``
        (Equity | FX | Fixed income), ``sub_class``, ``long_leg``,
        ``short_leg``, ``source`` (real | artificial | generic), ``order``.
    meta:
        Provenance: price source, failed legs, data folder, notes.
    """

    returns: pd.DataFrame
    assets: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if list(self.returns.columns) != list(self.assets.index):
            raise ValueError("returns columns must equal assets.index (same order)")
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            raise ValueError("returns must have a DatetimeIndex")

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.returns.index)


@dataclass
class TopicTable:
    """Topics of the run (G.3).

    ``table``: index ``topic_id`` in display order; columns ``name``,
    ``group`` (Sector | Macro | Micro | Generic), ``ontology``
    (sector | economic | generic), ``scope``, ``order``.
    """

    table: pd.DataFrame

    @property
    def ids(self) -> list[str]:
        return [str(i) for i in self.table.index]


@dataclass
class LinkMap:
    """Topic-asset links (G.4).

    ``table`` columns: ``topic_id``, ``asset_id``, ``tier``
    (strong | moderate | weak), ``sign`` (+1 | -1), ``mechanism``,
    ``origin`` (default | random | override). One row per linked pair.
    """

    table: pd.DataFrame


@dataclass
class SimTruth:
    """Ground truth of the simulation (G.5).

    Attributes
    ----------
    W:
        Design matrix after feasibility scaling (topics x assets).
    W_unscaled:
        Design matrix as set by the user (sign x tier value).
    feasibility_scale:
        Per topic multiplier applied to ``W_unscaled`` (1 = unchanged).
    sigma_u:
        Per topic standard deviation of the news noise ``u``.
    attenuation:
        Per topic ``a_k`` (G.5.3) for the shock window used.
    B_true:
        True sensitivity: population regression coefficients on the observed standardised shocks
        (topics x assets, standardised units).
    r2_true:
        Per asset population share of variance explained by the topics.
    S_z:
        Correlation matrix of the observed standardised shocks (topics x topics).
    asset_vol:
        Full-sample daily standard deviation of each asset's return (the
        unit ``B_true`` is defined in). The OOS oracle does not use it: it
        applies ``B_true`` with the fit's training volatility (D74).
    R:
        Full-sample correlation matrix of asset returns.
    shock_window:
        ``w`` the truth refers to.
    """

    W: pd.DataFrame
    W_unscaled: pd.DataFrame
    feasibility_scale: pd.Series
    sigma_u: pd.Series
    attenuation: pd.Series
    B_true: pd.DataFrame
    r2_true: pd.Series
    S_z: pd.DataFrame
    asset_vol: pd.Series
    R: pd.DataFrame
    shock_window: int


@dataclass
class SimData:
    """Output of the simulation stage (G.5).

    Attributes
    ----------
    market, topics, links:
        Inputs of the run.
    attention:
        ``(n_days, L)`` simulated attention levels ``a`` (what estimators see).
    designed_shocks:
        ``(n_days, L)`` designed shocks ``s`` (hidden from the estimators).
    truth:
        :class:`SimTruth` for the default shock window of the config.
    lead_days:
        ``l`` used in the construction.
    meta:
        Clipped share, feasibility warnings, seeds, timings.
    """

    market: MarketData
    topics: TopicTable
    links: LinkMap
    attention: pd.DataFrame
    designed_shocks: pd.DataFrame
    truth: SimTruth
    lead_days: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservedShocks:
    """Observed shocks (D9) and their training-window standardisation (G.5.3).

    ``z`` raw shocks; ``s_hat = z / scale`` with ``scale`` the training-window
    standard deviation per topic; the first ``window`` rows are ``NaN``.
    """

    z: pd.DataFrame
    s_hat: pd.DataFrame
    scale: pd.Series
    window: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp


@dataclass
class DirectFit:
    """Direct sensitivity regression fitted on the training window (G.7.1).

    Attributes
    ----------
    B_hat:
        Estimated sensitivities in standardised units (topics x assets).
    intercept:
        Per asset intercept in standardised units (not used for the
        explained return).
    ret_mean, ret_scale:
        Training-window mean and standard deviation of each asset's return.
    selected:
        Boolean (topics x assets): ``B_hat != 0`` for sparse methods,
        ``|B_hat| >= select_tau`` otherwise.
    penalty:
        Per asset penalty actually used (``NaN`` where not applicable).
    n_train:
        Per asset number of training observations.
    method:
        Method name.
    meta:
        Convergence flags, timings, notes.
    """

    B_hat: pd.DataFrame
    intercept: pd.Series
    ret_mean: pd.Series
    ret_scale: pd.Series
    selected: pd.DataFrame
    penalty: pd.Series
    n_train: pd.Series
    method: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def B_hat_pct(self) -> pd.DataFrame:
        """Estimated sensitivities in percent per one-standard-deviation shock."""
        return self.B_hat.mul(self.ret_scale * 100.0, axis=1)


@dataclass
class WindowEval:
    """Out-of-sample evaluation in the forecast window (G.8).

    All asset x topic frames have assets as rows and topics as columns (the
    correlation-table layout).

    Attributes
    ----------
    return_days:
        Return days in the window (``t + l``).
    corr:
        OOS correlation of returns with observed standardised shocks.
    r2, r2_oracle:
        Per asset uncentered OOS R2 of the estimator and of the oracle.
    contrib, contrib_true:
        Return attribution per topic over the window (decimal return points).
    var_share, var_share_true:
        Variance share per topic (G.8 point 4).
    realized, explained, explained_true:
        Per asset realised move and topic-explained move over the window.
    residual:
        ``realized - explained`` per asset.
    fitted, fitted_oracle:
        ``(days, N)`` topic-explained daily returns.
    realized_daily:
        ``(days, N)`` realised daily returns in the window.
    recovery:
        Recovery metrics of the training fit against the truth (G.8 point 5).
    n_days:
        Number of return days in the window.
    """

    return_days: pd.DatetimeIndex
    corr: pd.DataFrame
    r2: pd.Series
    r2_oracle: pd.Series
    contrib: pd.DataFrame
    contrib_true: pd.DataFrame
    var_share: pd.DataFrame
    var_share_true: pd.DataFrame
    realized: pd.Series
    explained: pd.Series
    explained_true: pd.Series
    residual: pd.Series
    fitted: pd.DataFrame
    fitted_oracle: pd.DataFrame
    realized_daily: pd.DataFrame
    recovery: dict[str, float]
    n_days: int


@dataclass
class BKSLabResult:
    """BKS Sparse IPCA run by the lab (G.7.2).

    Attributes
    ----------
    selected_topics:
        Topics with a nonzero ``Gamma`` row.
    gamma_norms:
        Row norms of ``Gamma`` per topic (``const`` excluded).
    lam, K:
        Chosen penalty and number of factors.
    path:
        The lambda path table (``TuningResult.path_frame()``) or ``None``.
    periods:
        Weekly periods (last trading day) in the forecast window.
    fitted, realized:
        ``(weeks, N)`` fitted and realised weekly returns in the window.
    r2:
        Per asset uncentered OOS R2 over the window weeks.
    r2_pooled:
        Pooled uncentered OOS R2 over all asset-weeks in the window.
    contrib:
        Assets x topics per-topic split of the fitted return summed over the
        window (not identified, D52).
    const_contrib:
        Per asset contribution of the constant instrument.
    in_sample_total_r2:
        Pooled total R2 of the training fit.
    meta:
        Timings, warnings, convergence, config used.
    """

    selected_topics: list[str]
    gamma_norms: pd.Series
    lam: float
    K: int
    path: pd.DataFrame | None
    periods: pd.DatetimeIndex
    fitted: pd.DataFrame
    realized: pd.DataFrame
    r2: pd.Series
    r2_pooled: float
    contrib: pd.DataFrame
    const_contrib: pd.Series
    in_sample_total_r2: float
    meta: dict[str, Any] = field(default_factory=dict)
