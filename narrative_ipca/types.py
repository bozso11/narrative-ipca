"""Data containers passed between the pipeline stages.

Every stage consumes one of these and produces the next. All containers are
plain dataclasses of numpy arrays / pandas objects, deliberately without
behaviour beyond shape validation and a few cheap derived quantities, so that
the numerics live in the stage modules and can be tested in isolation.

Shape conventions (BKS notation)
--------------------------------
* ``tau`` indexes trading days, ``t`` indexes estimation periods (months).
* ``L`` topics (narratives), ``N`` assets, ``K`` latent factors,
  ``p = L + 1`` instruments (column 0 is the constant).
* Daily objects are ``pandas.DataFrame`` with a ``DatetimeIndex`` and one
  column per topic / asset. Period objects use the period's *last trading
  day* as the timestamp.
* The estimation panel is stored in long form: one row per observed
  asset-period ``(i, t)`` pairing the lagged instruments ``c_{i,t-1}`` with
  the period return ``r_{i,t}`` (BKS Eq. 7).
* ``Gamma`` is ``(p, K)`` with row 0 the constant; ``Gamma_tilde``
  (``Gamma[1:]``) is the ``(L, K)`` narrative block. ``vect(Gamma)`` stacks
  the rows, coordinate ``l * K + k`` holds ``Gamma[l, k]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterator

import numpy as np
import pandas as pd

__all__ = [
    "AttentionData",
    "ReturnsData",
    "AlignedData",
    "ShockPanel",
    "CovariancePanel",
    "PanelMoments",
    "IPCAPanel",
    "SparseIPCAResult",
    "LambdaPathPoint",
    "TuningResult",
    "WrapUpResult",
    "OOSResult",
    "PricingTestResult",
    "PlaceboResult",
    "EvaluationReport",
    "PipelineResult",
    "SimulationTruth",
    "SimulatedData",
    "HarnessMetrics",
    "HarnessResult",
    "mve_weights",
    "annualized_sharpe",
    "compute_sigma_c",
    "DEFAULT_RCOND",
]

DEFAULT_RCOND: float = 1e-6
"""Relative pseudo-inverse cut-off for ``Sigma_ff`` / ``A'A``; equals ``config.EvaluationConfig.rcond``'s default (DESIGN.md D43)."""


# ---------------------------------------------------------------------------
# Small shared numerics used by several containers
# ---------------------------------------------------------------------------
def mve_weights(mu: np.ndarray, Sigma: np.ndarray, rcond: float = DEFAULT_RCOND) -> np.ndarray:
    """``b_MVE = mu' Sigma^-1`` as a 1-D vector, via pseudo-inverse (BKS Section 6.1)."""
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    Sigma = 0.5 * (Sigma + Sigma.T)
    return np.linalg.pinv(Sigma, rcond=rcond) @ mu


def annualized_sharpe(mu: np.ndarray, Sigma: np.ndarray, annualization: float = 12.0, rcond: float = DEFAULT_RCOND) -> float:
    """``sqrt(ann * mu' Sigma^-1 mu)``, the MVE Sharpe ratio of a factor set (BKS Section 3.2)."""
    mu = np.asarray(mu, dtype=float).ravel()
    q = float(mu @ mve_weights(mu, Sigma, rcond=rcond))
    return float(np.sqrt(max(q, 0.0) * annualization))


# ---------------------------------------------------------------------------
# Step 1: inputs
# ---------------------------------------------------------------------------
@dataclass
class AttentionData:
    """Daily narrative attention levels ``theta_tau`` (input, D1).

    Attributes
    ----------
    levels:
        ``(n_days, L)`` DataFrame, DatetimeIndex (sorted, unique), one column
        per topic id (str). Values are attention levels, e.g. the
        term-weighted daily mean of FASTopic document-topic distributions.
        Rows may but need not sum to one.
    topic_labels:
        Optional human-readable label per topic id.
    phi:
        Optional ``(L, V)`` topic-term matrix (rows sum to one) for term-level
        interpretation (BKS Eq. 12). FASTopic exposes this as its topic-word
        distribution.
    """

    levels: pd.DataFrame
    topic_labels: dict[str, str] | None = None
    phi: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        _check_daily_frame(self.levels, "attention.levels")
        if self.phi is not None:
            missing = set(self.levels.columns) - set(self.phi.index)
            if missing:
                raise ValueError(f"phi is missing rows for topics: {sorted(missing)[:5]}...")

    @property
    def topics(self) -> list[str]:
        return [str(c) for c in self.levels.columns]


@dataclass
class ReturnsData:
    """Daily asset returns (input, D2).

    Attributes
    ----------
    returns:
        ``(n_days, N)`` DataFrame, DatetimeIndex, one column per asset id.
        ``NaN`` marks days on which the asset is not in the universe or has
        no return. Excess or total returns according to
        ``DataConfig.return_kind``.
    risk_free:
        Optional daily risk-free rate series (same calendar) used when
        ``return_kind == "total"``.
    asset_meta:
        Optional per-asset metadata indexed by asset id (e.g. an
        ``asset_class`` column). Carried through to the reports; not used by
        the estimator.
    """

    returns: pd.DataFrame
    risk_free: pd.Series | None = None
    asset_meta: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        _check_daily_frame(self.returns, "returns.returns")

    @property
    def assets(self) -> list[str]:
        return [str(c) for c in self.returns.columns]


@dataclass
class AlignedData:
    """Attention and returns on a common trading-day calendar (output of Step 1).

    Attributes
    ----------
    attention:
        ``(n_days, L)`` levels on the trading-day grid, lag applied.
    returns:
        ``(n_days, N)`` daily excess returns on the same grid (``NaN`` when
        the asset is not observed). If ``asset_weighting != "none"`` these are
        the *scaled* returns and ``scale`` holds the divisor.
    calendar:
        The trading-day grid (equal to ``returns.index``).
    scale:
        ``(n_days, N)`` per-asset-day divisor applied to returns, or ``None``.
    asset_meta:
        Passed through from :class:`ReturnsData`.
    """

    attention: pd.DataFrame
    returns: pd.DataFrame
    calendar: pd.DatetimeIndex
    scale: pd.DataFrame | None = None
    asset_meta: pd.DataFrame | None = None
    topic_labels: dict[str, str] | None = None
    phi: pd.DataFrame | None = None

    @property
    def topics(self) -> list[str]:
        return [str(c) for c in self.attention.columns]

    @property
    def assets(self) -> list[str]:
        return [str(c) for c in self.returns.columns]


# ---------------------------------------------------------------------------
# Step 2: shocks
# ---------------------------------------------------------------------------
@dataclass
class ShockPanel:
    """Daily attention shocks ``z_tau`` (BKS Section 3.1).

    ``z`` is ``(n_days, L)``; the first ``window`` rows are ``NaN`` because the
    trailing mean is not available. ``scale`` holds the per-topic divisor when
    ``ShockConfig.standardize`` is on (``None`` otherwise).
    """

    z: pd.DataFrame
    window: int
    scale: pd.Series | None = None

    @property
    def topics(self) -> list[str]:
        return [str(c) for c in self.z.columns]


# ---------------------------------------------------------------------------
# Step 3: covariance instruments
# ---------------------------------------------------------------------------
@dataclass
class CovariancePanel:
    """Kernel-weighted asset/narrative covariances ``cov_{i,t}`` (BKS Eq. 6).

    Attributes
    ----------
    values:
        ``(T, N, L)`` array. ``values[t, i, :]`` is ``cov_{i,t}``, the
        instrument known at the close of the window of period ``t`` (which
        ends ``skip_days`` before the period's last trading day). ``NaN``
        where the asset-period could not be estimated.
    periods:
        ``(T,)`` DatetimeIndex: last trading day of period ``t``.
    window_end:
        ``(T,)`` DatetimeIndex: last day included in the kernel window of
        period ``t`` (``<= periods``).
    assets, topics:
        Column labels for axes 1 and 2.
    n_days:
        ``(T, N)`` number of observed asset-days inside the window.
    xi:
        Kernel decay used.
    """

    values: np.ndarray
    periods: pd.DatetimeIndex
    window_end: pd.DatetimeIndex
    assets: np.ndarray
    topics: np.ndarray
    n_days: np.ndarray
    xi: float

    def __post_init__(self) -> None:
        v = np.asarray(self.values)
        if v.ndim != 3:
            raise ValueError(f"values must be 3-D (T, N, L), got shape {v.shape}")
        T, N, L = v.shape
        if len(self.periods) != T or len(self.window_end) != T:
            raise ValueError("periods/window_end length must equal values.shape[0]")
        if len(self.assets) != N or len(self.topics) != L:
            raise ValueError("assets/topics length must match values shape")
        if np.asarray(self.n_days).shape != (T, N):
            raise ValueError(f"n_days must have shape ({T}, {N})")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in np.asarray(self.values).shape)  # type: ignore[return-value]

    def frame(self, t: int) -> pd.DataFrame:
        """``(N, L)`` DataFrame of period ``t``'s instruments."""
        return pd.DataFrame(np.asarray(self.values)[t], index=self.assets, columns=self.topics)


# ---------------------------------------------------------------------------
# Step 4: the estimation panel (BKS Eq. 7)
# ---------------------------------------------------------------------------
@dataclass
class PanelMoments:
    """Per-period sufficient statistics of an :class:`IPCAPanel`.

    ``S[t] = C_{t-1}' C_{t-1}`` (``(p, p)``), ``V[t] = C_{t-1}' r_t`` (``(p,)``),
    ``yy[t] = r_t' r_t`` and ``n[t]`` the number of assets observed in period
    ``t``. The whole Sparse IPCA estimator touches the data only through these
    (BKS Eq. 16 and the Gram form of the Gamma-step), so they are computed
    once per panel and cached.
    """

    S: np.ndarray
    V: np.ndarray
    yy: np.ndarray
    n: np.ndarray

    @property
    def T(self) -> int:
        return int(self.S.shape[0])

    @property
    def p(self) -> int:
        return int(self.S.shape[1])


@dataclass
class IPCAPanel:
    """Long-form panel pairing ``c_{i,t-1}`` with ``r_{i,t}`` (BKS Eq. 7).

    Attributes
    ----------
    X:
        ``(n_obs, p)`` instruments; column 0 is the constant 1.
    y:
        ``(n_obs,)`` period-``t`` excess returns.
    t_idx:
        ``(n_obs,)`` int, index into ``periods`` of the *return* period,
        sorted ascending (rows of one period are contiguous).
    asset_idx:
        ``(n_obs,)`` int, index into ``assets``.
    periods:
        ``(T,)`` DatetimeIndex, last trading day of the return period ``t``.
    assets:
        ``(N,)`` asset ids.
    instrument_names:
        Length ``p``; ``["const", topic_1, ..., topic_L]``.
    sigma_c:
        ``(p,)`` panel standard deviation of each instrument over this panel
        (``sigma_c[0] = 1``), the ``sigma^c_l`` scaling of BKS Eq. 8. Must be
        recomputed whenever the training sample changes (D26).
    """

    X: np.ndarray
    y: np.ndarray
    t_idx: np.ndarray
    asset_idx: np.ndarray
    periods: pd.DatetimeIndex
    assets: np.ndarray
    instrument_names: list[str]
    sigma_c: np.ndarray
    _moments: PanelMoments | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.X = np.ascontiguousarray(np.asarray(self.X, dtype=float))
        self.y = np.ascontiguousarray(np.asarray(self.y, dtype=float).ravel())
        self.t_idx = np.ascontiguousarray(np.asarray(self.t_idx, dtype=np.int64).ravel())
        self.asset_idx = np.ascontiguousarray(np.asarray(self.asset_idx, dtype=np.int64).ravel())
        self.sigma_c = np.asarray(self.sigma_c, dtype=float).ravel()
        n, p = self.X.shape if self.X.ndim == 2 else (0, 0)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        if self.y.shape != (n,) or self.t_idx.shape != (n,) or self.asset_idx.shape != (n,):
            raise ValueError("y, t_idx and asset_idx must all have length X.shape[0]")
        if len(self.instrument_names) != p:
            raise ValueError(f"instrument_names has length {len(self.instrument_names)}, expected {p}")
        if self.sigma_c.shape != (p,):
            raise ValueError(f"sigma_c must have shape ({p},), got {self.sigma_c.shape}")
        if n and np.any(np.diff(self.t_idx) < 0):
            raise ValueError("t_idx must be sorted ascending")
        if n and (self.t_idx.min() < 0 or self.t_idx.max() >= len(self.periods)):
            raise ValueError("t_idx out of range of periods")
        if not np.all(np.isfinite(self.X)) or not np.all(np.isfinite(self.y)):
            raise ValueError("X and y must be finite (drop unobserved rows before building the panel)")

    # -- sizes ---------------------------------------------------------------
    @property
    def n_obs(self) -> int:
        return int(self.X.shape[0])

    @property
    def p(self) -> int:
        return int(self.X.shape[1])

    @property
    def L(self) -> int:
        return self.p - 1

    @property
    def T(self) -> int:
        return int(len(self.periods))

    @property
    def N(self) -> int:
        return int(len(self.assets))

    @property
    def topics(self) -> list[str]:
        return list(self.instrument_names[1:])

    # -- iteration -----------------------------------------------------------
    def period_slices(self) -> Iterator[tuple[int, slice]]:
        """Yield ``(t, slice)`` for every period (empty slices for empty periods)."""
        bounds = np.searchsorted(self.t_idx, np.arange(self.T + 1))
        for t in range(self.T):
            yield t, slice(int(bounds[t]), int(bounds[t + 1]))

    def moments(self) -> PanelMoments:
        """Cached per-period sufficient statistics."""
        if self._moments is None:
            S = np.zeros((self.T, self.p, self.p))
            V = np.zeros((self.T, self.p))
            yy = np.zeros(self.T)
            n = np.zeros(self.T, dtype=np.int64)
            for t, sl in self.period_slices():
                if sl.stop <= sl.start:
                    continue
                Xt = self.X[sl]
                yt = self.y[sl]
                S[t] = Xt.T @ Xt
                V[t] = Xt.T @ yt
                yy[t] = float(yt @ yt)
                n[t] = sl.stop - sl.start
            self._moments = PanelMoments(S=S, V=V, yy=yy, n=n)
        return self._moments

    # -- sub-sampling --------------------------------------------------------
    def subset_periods(self, keep: np.ndarray, recompute_sigma_c: bool = True) -> "IPCAPanel":
        """Restrict to periods where ``keep[t]`` is True (boolean mask of length T).

        Period indices are re-numbered to the kept periods. ``sigma_c`` is
        recomputed on the subset by default, as BKS define it on the training
        sample ``S``.
        """
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (self.T,):
            raise ValueError(f"keep must have shape ({self.T},)")
        row_mask = keep[self.t_idx]
        new_index = np.cumsum(keep) - 1
        X = self.X[row_mask]
        sigma_c = compute_sigma_c(X) if recompute_sigma_c else self.sigma_c.copy()
        return IPCAPanel(
            X=X,
            y=self.y[row_mask],
            t_idx=new_index[self.t_idx[row_mask]],
            asset_idx=self.asset_idx[row_mask],
            periods=self.periods[keep],
            assets=self.assets,
            instrument_names=list(self.instrument_names),
            sigma_c=sigma_c,
        )

    def with_instruments(self, X_new: np.ndarray, names: list[str]) -> "IPCAPanel":
        """Return a copy with a different instrument matrix (e.g. placebo columns appended)."""
        return IPCAPanel(
            X=X_new,
            y=self.y.copy(),
            t_idx=self.t_idx.copy(),
            asset_idx=self.asset_idx.copy(),
            periods=self.periods,
            assets=self.assets,
            instrument_names=list(names),
            sigma_c=compute_sigma_c(np.asarray(X_new, dtype=float)),
        )


def compute_sigma_c(X: np.ndarray) -> np.ndarray:
    """``sigma^c_l``: population std of each instrument column; column 0 fixed at 1 (BKS App. B.1)."""
    X = np.asarray(X, dtype=float)
    s = X.std(axis=0, ddof=0) if X.shape[0] > 0 else np.ones(X.shape[1])
    s = np.where(np.isfinite(s) & (s > 0), s, 1.0)
    s[0] = 1.0
    return s


# ---------------------------------------------------------------------------
# Step 5: Sparse IPCA estimate
# ---------------------------------------------------------------------------
@dataclass
class SparseIPCAResult:
    """One Sparse IPCA fit (BKS Eq. 8) at a fixed ``(K, lambda)``.

    Attributes
    ----------
    Gamma:
        ``(p, K)`` estimated instrument-to-loading map; row 0 is the constant.
    F:
        ``(T, K)`` estimated factors ``f_t`` (period-``t`` mimicking-portfolio
        returns). Periods without observations carry zeros and are excluded
        from ``mu_f``/``Sigma_ff``.
    mu_f, Sigma_ff:
        Sample mean and covariance (ddof=1) of ``f_t`` over populated periods.
    lam, K:
        Hyper-parameters of the fit.
    objective, obj_path:
        Final value and per-sweep history of the Eq. 8 objective.
    n_iter, converged:
        ARLS iteration count and convergence flag.
    total_r2, pred_r2:
        ``1 - SSR / sum r^2`` with ``f_t`` (total) and with ``mu_f``
        (predictive), BKS Section 3.2.
    gamma_norms:
        ``(p,)`` row norms ``||Gamma_l||_2``.
    selected:
        ``(L,)`` boolean, ``gamma_norms[1:] > 0``.
    instrument_names, periods:
        Labels.
    n_obs:
        Panel size ``N_S`` used in the penalty scaling.
    """

    Gamma: np.ndarray
    F: np.ndarray
    mu_f: np.ndarray
    Sigma_ff: np.ndarray
    lam: float
    K: int
    objective: float
    obj_path: list[float]
    n_iter: int
    converged: bool
    total_r2: float
    pred_r2: float
    gamma_norms: np.ndarray
    selected: np.ndarray
    instrument_names: list[str]
    periods: pd.DatetimeIndex
    n_obs: int
    populated: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def Gamma_tilde(self) -> np.ndarray:
        """``(L, K)`` narrative block of ``Gamma`` (rows 1..L)."""
        return self.Gamma[1:]

    @property
    def n_selected(self) -> int:
        return int(np.count_nonzero(self.selected))

    @property
    def selected_topics(self) -> list[str]:
        return [n for n, s in zip(self.instrument_names[1:], self.selected) if s]

    def b_mve(self, rcond: float = DEFAULT_RCOND) -> np.ndarray:
        """``b_MVE = mu_f' Sigma_ff^-1`` (BKS Section 6.1)."""
        return mve_weights(self.mu_f, self.Sigma_ff, rcond=rcond)

    def mve_sharpe(self, annualization: float = 12.0, rcond: float = DEFAULT_RCOND) -> float:
        """Annualised in-sample MVE Sharpe ratio ``sqrt(ann * mu' Sigma^-1 mu)``."""
        return annualized_sharpe(self.mu_f, self.Sigma_ff, annualization, rcond)

    def factors_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.F, index=self.periods, columns=[f"f{k+1}" for k in range(self.K)])

    def gamma_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.Gamma, index=self.instrument_names, columns=[f"f{k+1}" for k in range(self.K)])

    def rotate(self, R: np.ndarray) -> "SparseIPCAResult":
        """Apply ``Gamma -> Gamma R``, ``f_t -> R^-1 f_t`` (model-invariant reparametrisation)."""
        R = np.asarray(R, dtype=float)
        Rinv = np.linalg.inv(R)
        F_new = self.F @ Rinv.T
        return replace(
            self,
            Gamma=self.Gamma @ R,
            F=F_new,
            mu_f=Rinv @ self.mu_f,
            Sigma_ff=Rinv @ self.Sigma_ff @ Rinv.T,
            gamma_norms=np.linalg.norm(self.Gamma @ R, axis=1),
        )


@dataclass
class LambdaPathPoint:
    """One point of the regularisation path (BKS Figure 2)."""

    lam: float
    K: int
    total_r2: float
    pred_r2: float
    mve_sharpe: float
    n_selected: int
    gamma_norms: np.ndarray
    objective: float
    converged: bool
    n_iter: int
    criterion: float | None = None


@dataclass
class TuningResult:
    """Outcome of the lambda (and K) tuning."""

    lam: float
    K: int
    criterion: str
    path: list[LambdaPathPoint]
    fit: SparseIPCAResult
    lam_max: float
    meta: dict[str, Any] = field(default_factory=dict)

    def path_frame(self) -> pd.DataFrame:
        rows = [
            {
                "K": p.K,
                "lam": p.lam,
                "total_r2": p.total_r2,
                "pred_r2": p.pred_r2,
                "mve_sharpe": p.mve_sharpe,
                "n_selected": p.n_selected,
                "criterion": p.criterion,
                "objective": p.objective,
                "converged": p.converged,
                "n_iter": p.n_iter,
            }
            for p in self.path
        ]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7: wrap-up and interpretation (BKS step 3, Section 6.1)
# ---------------------------------------------------------------------------
@dataclass
class WrapUpResult:
    """Structural objects recovered from a fit (BKS Eq. 1, 5, 10-12).

    Attributes
    ----------
    A:
        ``(L, K)`` narrative-to-state loading matrix
        ``A = Gamma_tilde (Gamma_tilde' Gamma_tilde)^-1 Sigma_ff^-1``.
    states:
        ``(n_days, K)`` DataFrame of latent states ``x_tau = (A'A)^-1 A' z_tau``.
    x_mve:
        Daily MVE state ``b_MVE x_tau`` (the univariate pricing kernel proxy).
    b_mve:
        ``(K,)`` MVE combination weights ``mu_f' Sigma_ff^-1``.
    impact_z_to_x:
        ``(L, K)`` DataFrame ``I_{z->x} = A (A'A)^-1`` (BKS Eq. 10).
    impact_z_to_mve:
        ``(L,)`` Series ``I_{z->MVE} = I_{z->x} b_MVE`` (Eq. 11).
    impact_z_to_obs:
        Optional dict name -> ``(L,)`` Series for observable targets projected
        on the factors (e.g. the market factor, BKS ``I_{z->Mkt}``), together
        with the projection weights ``b_obs`` and R2 in ``obs_projection``.
    impact_w_to_mve:
        Optional ``(V,)`` term-level impact vector (Eq. 12) when ``phi`` is
        available.
    """

    A: pd.DataFrame
    states: pd.DataFrame
    x_mve: pd.Series
    b_mve: np.ndarray
    impact_z_to_x: pd.DataFrame
    impact_z_to_mve: pd.Series
    impact_z_to_obs: dict[str, pd.Series] = field(default_factory=dict)
    obs_projection: dict[str, dict[str, Any]] = field(default_factory=dict)
    impact_w_to_mve: pd.Series | None = None
    rank_deficient: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 8: out-of-sample
# ---------------------------------------------------------------------------
@dataclass
class OOSResult:
    """Expanding-window out-of-sample factors and MVE portfolio (BKS Section 4.2)."""

    factors: pd.DataFrame
    mve: pd.Series
    sharpe: float
    refit_periods: list[pd.Timestamp]
    lam_history: pd.Series
    K_history: pd.Series
    n_selected_history: pd.Series
    gamma_norm_history: pd.DataFrame
    selected_history: pd.DataFrame
    is_sharpe_history: pd.Series
    fits: list[SparseIPCAResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 9: evaluation
# ---------------------------------------------------------------------------
@dataclass
class PricingTestResult:
    """Time-series pricing errors of test assets on a factor set (BKS Table 1)."""

    alphas: pd.Series
    t_stats: pd.Series
    betas: pd.DataFrame
    r2: pd.Series
    avg_abs_alpha: float
    avg_abs_t: float
    frac_significant: float
    grs_stat: float | None
    grs_pvalue: float | None
    model_name: str = ""


@dataclass
class PlaceboResult:
    """Placebo narrative selection test (BKS Appendix C.2)."""

    n_placebo: int
    n_placebo_selected: int
    n_real_selected: int
    lam_max_by_instrument: pd.Series
    lam_star: float
    real_selected_before: list[str]
    real_selected_after: list[str]
    jaccard_real: float


@dataclass
class EvaluationReport:
    """All evaluation outputs of a run, plus a flat ``metrics`` dict for dashboards."""

    metrics: dict[str, float]
    lambda_path: pd.DataFrame | None = None
    selected_topics: list[str] = field(default_factory=list)
    gamma_norms: pd.Series | None = None
    pricing_tests: dict[str, PricingTestResult] = field(default_factory=dict)
    placebo: PlaceboResult | None = None
    factor_correlations: pd.DataFrame | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Everything a production run produces."""

    config: Any
    shocks: ShockPanel
    covariances: CovariancePanel
    panel: IPCAPanel
    tuning: TuningResult
    fit: SparseIPCAResult
    wrapup: WrapUpResult | None
    oos: OOSResult | None
    evaluation: EvaluationReport
    timings: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulation ground truth (DESIGN.md Part D)
# ---------------------------------------------------------------------------
@dataclass
class SimulationTruth:
    """Known characteristics of a simulated data set.

    All daily objects are on the simulated trading-day calendar; period
    objects are accumulated over the simulated periods.

    Attributes
    ----------
    A:
        ``(L, K)`` true narrative-to-state loadings, zero rows for irrelevant
        topics (identified up to the rotation convention of the DGP).
    relevant, placebo:
        ``(L,)`` boolean masks; ``placebo`` is a subset of ``~relevant``.
    f_daily, f_period:
        True tradable factors, daily and accumulated per period.
    x_daily:
        True states ``f + nu``.
    z_daily:
        True topic shocks ``A x + eta`` (before the attention-level mapping).
    beta:
        ``(T, N, K)`` true loadings per period (``NaN`` when not in universe).
    mu_f_period, Sigma_ff_period:
        Population mean and covariance of the period factors.
    sharpe_mve_true:
        Annualised population MVE Sharpe ratio ``sqrt(ann * mu' Sigma^-1 mu)``.
    Gamma_tilde_true:
        ``(L, K)`` ``A (A'A)^-1 Sigma_ff_daily^-1``: the instrument-to-loading
        map implied by BKS Eq. 5 in daily-covariance units.
    impact_z_to_mve_true:
        ``(L,)`` true ``I_{z->MVE}``.
    asset_class:
        ``(N,)`` labels.
    systematic_r2:
        Population share of period-return variance explained by the factors,
        averaged over assets.
    """

    A: np.ndarray
    relevant: np.ndarray
    placebo: np.ndarray
    f_daily: pd.DataFrame
    f_period: pd.DataFrame
    x_daily: pd.DataFrame
    z_daily: pd.DataFrame
    beta: np.ndarray
    periods: pd.DatetimeIndex
    mu_f_period: np.ndarray
    Sigma_ff_period: np.ndarray
    Sigma_ff_daily: np.ndarray
    sharpe_mve_true: float
    Gamma_tilde_true: np.ndarray
    impact_z_to_mve_true: np.ndarray
    asset_class: np.ndarray
    systematic_r2: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatedData:
    """A simulated input pair plus its ground truth."""

    attention: AttentionData
    returns: ReturnsData
    truth: SimulationTruth
    config: Any


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
@dataclass
class HarnessMetrics:
    """Comparison of one pipeline run against the simulation truth."""

    values: dict[str, float]
    passed: dict[str, bool]
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        """True when every applicable check passed; vacuously True for report-only scenarios (no checks)."""
        return all(self.passed.values())


@dataclass
class HarnessResult:
    """All scenarios x seeds of a harness run."""

    per_run: pd.DataFrame
    summary: pd.DataFrame
    passed: pd.DataFrame
    scenario_configs: dict[str, Any]
    thresholds: Any
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _check_daily_frame(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted ascending")
    if df.index.has_duplicates:
        raise ValueError(f"{name} index has duplicate dates")
    if df.shape[1] == 0:
        raise ValueError(f"{name} has no columns")
    if len(set(map(str, df.columns))) != df.shape[1]:
        raise ValueError(f"{name} has duplicate column labels")
