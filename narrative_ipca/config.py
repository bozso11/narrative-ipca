"""Configuration objects, one per methodology step (Step 0 of the pipeline).

Every knob of the pipeline lives here so that a run is fully described by one
``PipelineConfig`` and can be reproduced from a JSON/YAML file. The dataclasses
are frozen; use :func:`dataclasses.replace` to derive variants.

Naming follows Bybee, Kelly & Su (2023), "Narrative Asset Pricing" (BKS):

* ``theta_tau`` daily narrative attention levels (input),
* ``z_tau``     attention shocks (Section 3.1),
* ``cov_{i,t}`` kernel-weighted asset/narrative covariances (Eq. 6),
* ``c_{i,t}``   instruments ``[1, cov_{i,t}]`` (Eq. 7),
* ``Gamma``     the (L+1) x K instrument-to-loading map (Eq. 7-8),
* ``f_t``       the K narrative factors (mimicking portfolios),
* ``lambda``    the group-lasso penalty (Eq. 8),
* ``K``         number of latent factors / state variables.

See DESIGN.md for the assumptions behind every default (decision register D1-D45).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "DataConfig",
    "ShockConfig",
    "CovarianceConfig",
    "LambdaGridConfig",
    "EstimationConfig",
    "TuningConfig",
    "OOSConfig",
    "EvaluationConfig",
    "PipelineConfig",
    "AssetClassSpec",
    "SimulationConfig",
    "HarnessThresholds",
    "HarnessConfig",
    "config_to_dict",
    "config_from_dict",
    "load_config",
    "save_config",
    "config_hash",
]


# ---------------------------------------------------------------------------
# Step 1: input data alignment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    """How the raw attention and return inputs are aligned (DESIGN.md D1-D8).

    Attributes
    ----------
    period:
        Pandas offset alias defining the estimation period ``t`` over which
        returns are accumulated and at which instruments are refreshed.
        BKS use the calendar month (``"M"``). ``"W"`` gives weekly periods.
    attention_lag_days:
        Shift attention forward by this many *trading days* before pairing it
        with returns (day ``tau`` carries ``theta_{tau-k}``). ``0`` is the BKS
        convention: ``theta_tau`` already reflects the information of trading
        day ``tau`` (BKS stamp the edition published the morning of ``tau+1``
        on day ``tau``; do that re-stamping upstream, it is not a positive
        lag). ``k >= 1`` is the lagged robustness variant of DESIGN.md Part F
        point 5: covariances of today's returns with attention published
        ``k`` days earlier. Negative lags (pulling attention earlier) would
        introduce look-ahead and are not allowed.
    non_trading_day_policy:
        What to do with attention observed on days that are not in the return
        calendar (weekends, holidays). ``"drop"`` ignores them; ``"fold_mean"``
        averages them into the next trading day; ``"fold_sum"`` sums them.
    return_kind:
        ``"excess"`` means the return panel is already in excess of the
        risk-free rate; ``"total"`` means a risk-free series must be supplied
        to :func:`narrative_ipca.data.align_inputs` and is subtracted daily.
    return_aggregation:
        How daily excess returns are accumulated to the period return
        ``r_{i,t}`` (Eq. 7): ``"sum"`` (linear accumulation, the BKS reading of
        returns as innovations) or ``"compound"`` (``prod(1 + r) - 1``).
    asset_weighting:
        Weighting of assets in the pooled least-squares objective. ``"none"``
        is BKS. ``"inverse_vol"`` scales every asset's daily returns by its
        trailing ``vol_window_days`` volatility (ex ante), which puts low- and
        high-volatility asset classes on a comparable footing in a
        multi-asset universe. The model is scale-equivariant per asset, so
        this changes only the relative weight of assets in the fit (D7).
    vol_window_days:
        Trailing window for ``asset_weighting="inverse_vol"``.
    min_assets_per_period:
        Periods with fewer observed assets are dropped from the panel.
    dtype:
        Storage dtype of the covariance panel (``"float32"`` halves memory for
        large universes; the estimator itself always works in float64).
    """

    period: str = "M"
    attention_lag_days: int = 0
    non_trading_day_policy: Literal["drop", "fold_mean", "fold_sum"] = "drop"
    return_kind: Literal["excess", "total"] = "excess"
    return_aggregation: Literal["sum", "compound"] = "sum"
    asset_weighting: Literal["none", "inverse_vol"] = "none"
    vol_window_days: int = 252
    min_assets_per_period: int = 20
    dtype: Literal["float64", "float32"] = "float64"

    def __post_init__(self) -> None:
        if self.attention_lag_days < 0:
            raise ValueError("attention_lag_days must be >= 0")
        if self.vol_window_days < 5:
            raise ValueError("vol_window_days must be >= 5")
        if self.min_assets_per_period < 1:
            raise ValueError("min_assets_per_period must be >= 1")


# ---------------------------------------------------------------------------
# Step 2: attention shocks (BKS Section 3.1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShockConfig:
    """Attention shock construction ``z_tau = theta_tau - MA_w(theta)`` (D9-D10).

    Attributes
    ----------
    window:
        ``w`` in ``z_tau = theta_tau - (1/w) sum_{j=1..w} theta_{tau-j}``.
        BKS use ``5``; ``1`` gives the daily difference, ``3`` and ``20`` are
        the robustness variants of BKS Appendix C.5.
    standardize:
        Divide each topic's shocks by its full-sample standard deviation.
        Display convenience only: it is not out-of-sample safe and the
        estimator is invariant to it up to the ``sigma^c_l`` rescaling in
        Eq. 8. Default off.
    """

    window: int = 5
    standardize: bool = False

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")


# ---------------------------------------------------------------------------
# Step 3: kernel-weighted covariances (BKS Eq. 6, Appendix B.1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CovarianceConfig:
    """Kernel ``kappa(tau; t) = xi^(t - t_tau) / sum(...)`` settings (D11-D16).

    Attributes
    ----------
    xi:
        Per-*period* decay of the exponential kernel. BKS: ``0.99`` per month,
        i.e. a half-life of about 69 months. Note the decay steps by period,
        not by day: every day of period ``t`` has raw weight 1, every day of
        period ``t-1`` has weight ``xi``, and so on.
    skip_days:
        The estimation window for ``cov_{i,t}`` closes this many trading days
        before the last day of period ``t`` (BKS footnote 9: "up to the second
        last day", i.e. ``skip_days = 1``), so the instrument is in the
        information set before the period-``t+1`` return starts accruing.
    burn_in_periods:
        Periods dropped at the start of the sample while the kernel warms up
        (BKS: 12 months).
    lookback_periods:
        Truncate the kernel to this many periods back. ``None`` uses the full
        history, as BKS do.
    min_days:
        Minimum number of observed asset-days inside the (possibly truncated)
        window for ``cov_{i,t}`` to be computed; otherwise it is missing.
    """

    xi: float = 0.99
    skip_days: int = 1
    burn_in_periods: int = 12
    lookback_periods: int | None = None
    min_days: int = 60

    def __post_init__(self) -> None:
        if not (0.0 < self.xi <= 1.0):
            raise ValueError("xi must lie in (0, 1]")
        if self.skip_days < 0:
            raise ValueError("skip_days must be >= 0")
        if self.burn_in_periods < 0:
            raise ValueError("burn_in_periods must be >= 0")
        if self.lookback_periods is not None and self.lookback_periods < 1:
            raise ValueError("lookback_periods must be >= 1 or None")
        if self.min_days < 2:
            raise ValueError("min_days must be >= 2")


# ---------------------------------------------------------------------------
# Step 4-5: Sparse IPCA (BKS Eq. 8, Appendix B.2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LambdaGridConfig:
    """Regularisation path grid (D22).

    ``values`` overrides the automatic grid. Otherwise the grid is
    ``n_lambdas`` log-spaced points from ``ratio * lam_max`` to ``lam_max``,
    where ``lam_max`` is the data-dependent smallest penalty at which every
    narrative row of ``Gamma`` is zero (see ``sparse_ipca.lambda_max``).
    """

    n_lambdas: int = 30
    ratio: float = 1e-3
    values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.n_lambdas < 1:
            raise ValueError("n_lambdas must be >= 1")
        if not (0.0 < self.ratio < 1.0):
            raise ValueError("ratio must lie in (0, 1)")
        if self.values is not None and any(v < 0 for v in self.values):
            raise ValueError("lambda values must be >= 0")


@dataclass(frozen=True)
class EstimationConfig:
    """Sparse IPCA hyper-parameters (D17-D26).

    Attributes
    ----------
    K:
        Number of latent factors / state variables. BKS main specification: 3.
    lam:
        Group-lasso penalty ``lambda`` in Eq. 8. ``None`` means "tune it"
        (see :class:`TuningConfig`); ``0.0`` means unregularised IPCA, which is
        estimated by the plain alternating least squares of Kelly, Korsaye,
        Pruitt & Su (2026), Algorithm 1, because Eq. 8 has no scale
        identification at ``lambda = 0`` (D18).
    lam_grid:
        Grid used when ``lam`` is tuned or a path is traced.
    penalize_intercept:
        Whether row 0 of ``Gamma`` (the constant instrument) is penalised.
        BKS Eq. 8 sums the penalty over ``l = 0..L`` with ``sigma^c_0 = 1``,
        so the default is ``True``.
    max_iter, tol:
        Outer ARLS loop: stop when the relative change of the Eq. 8 objective
        falls below ``tol`` or after ``max_iter`` sweeps.
    inner_max_iter, inner_tol:
        Group-lasso (Gamma-step) solver limits per sweep.
    n_warmup:
        Unpenalised ARLS sweeps run from the SVD initialisation before the
        penalty is switched on (brings ``Gamma`` onto the right scale so the
        first penalised sweep does not zero everything).
    seed:
        Seed for the (tiny) jitter added to the deterministic initialisation.
    """

    K: int = 3
    lam: float | None = None
    lam_grid: LambdaGridConfig = field(default_factory=LambdaGridConfig)
    penalize_intercept: bool = True
    max_iter: int = 500
    tol: float = 1e-8
    inner_max_iter: int = 200
    inner_tol: float = 1e-10
    n_warmup: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError("K must be >= 1")
        if self.lam is not None and self.lam < 0:
            raise ValueError("lam must be >= 0 or None")
        if self.max_iter < 1 or self.inner_max_iter < 1:
            raise ValueError("max_iter and inner_max_iter must be >= 1")
        if self.tol <= 0 or self.inner_tol <= 0:
            raise ValueError("tol and inner_tol must be > 0")
        if self.n_warmup < 0:
            raise ValueError("n_warmup must be >= 0")


# ---------------------------------------------------------------------------
# Step 6: hyper-parameter tuning (BKS Section 2, footnote 8, Appendix C.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TuningConfig:
    """How ``lambda`` (and optionally ``K``) are chosen (D27-D29).

    Attributes
    ----------
    criterion:
        ``"is_sharpe"``: BKS main scheme, maximise the in-sample annualised
        MVE Sharpe ratio ``sqrt(ann * mu_f' Sigma_ff^-1 mu_f)`` over the
        lambda grid. ``"loocv_sharpe"``: BKS Appendix C.3, leave-one-period-out
        cross-validated Sharpe ratio of the stitched MVE returns.
    K_grid:
        When given, ``K`` is tuned jointly with ``lambda`` over this grid
        (BKS Appendix C.3, "joint lambda, K tuning"). ``None`` keeps ``K``
        fixed at ``EstimationConfig.K``.
    tie_break:
        Among grid points with equal criterion, prefer the sparser (larger
        lambda) or denser solution.
    tolerance:
        Relative tolerance below the maximum criterion within which points
        count as tied, so that with ``tie_break="sparser"`` the sparsest
        point whose criterion is at least ``(1 - tolerance) * max`` wins
        (``0.02`` = within 2% of the best Sharpe ratio; for ``|max| < 1`` the
        tolerance is absolute). ``0.0`` is the BKS rule (exact argmax). The
        option exists because the in-sample Sharpe surface can be flat over a
        wide range of selected-narrative counts (DESIGN.md D27, study of
        2026-09-06), in which case the exact argmax admits many noise
        narratives that add nothing to the criterion.
    loocv_max_folds:
        Subsample at most this many left-out periods for the LOOCV criterion
        (speed). ``None`` uses every period.
    """

    criterion: Literal["is_sharpe", "loocv_sharpe"] = "is_sharpe"
    K_grid: tuple[int, ...] | None = None
    tie_break: Literal["sparser", "denser"] = "sparser"
    tolerance: float = 0.0
    loocv_max_folds: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.tolerance < 1.0):
            raise ValueError("tolerance must lie in [0, 1)")
        if self.K_grid is not None and (len(self.K_grid) == 0 or any(k < 1 for k in self.K_grid)):
            raise ValueError("K_grid must be a non-empty tuple of ints >= 1")
        if self.loocv_max_folds is not None and self.loocv_max_folds < 2:
            raise ValueError("loocv_max_folds must be >= 2 or None")


# ---------------------------------------------------------------------------
# Step 8: out-of-sample evaluation (BKS Section 4.2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OOSConfig:
    """Expanding-window out-of-sample scheme (D30-D33).

    Attributes
    ----------
    enabled:
        Run the OOS stage at all.
    first_oos_period:
        ISO date: the first period whose return is evaluated out of sample is
        the first period whose *last trading day* is on or after this date,
        so give the first calendar day of the intended period
        (``"2000-01-01"`` for January 2000). ``None`` uses ``oos_fraction``.
    oos_fraction:
        When ``first_oos_period`` is ``None``, the last ``oos_fraction`` of
        the panel's periods are evaluated out of sample.
    refit_every:
        Refit (and retune) the model every this many periods; the frozen
        ``Gamma``, ``mu_f``, ``Sigma_ff`` are applied to the following
        ``refit_every`` periods (BKS: once every December, 12 months).
    retune_lambda:
        Re-run the tuning at every refit (BKS) or keep the first tuned value.
    min_train_periods:
        Minimum number of training periods before the first OOS period.
    """

    enabled: bool = True
    first_oos_period: str | None = None
    oos_fraction: float = 0.4
    refit_every: int = 12
    retune_lambda: bool = True
    min_train_periods: int = 60

    def __post_init__(self) -> None:
        if not (0.0 < self.oos_fraction < 1.0):
            raise ValueError("oos_fraction must lie in (0, 1)")
        if self.refit_every < 1:
            raise ValueError("refit_every must be >= 1")
        if self.min_train_periods < 2:
            raise ValueError("min_train_periods must be >= 2")


# ---------------------------------------------------------------------------
# Step 9: evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation settings (D34-D36).

    Attributes
    ----------
    annualization:
        Periods per year used to annualise Sharpe ratios (12 for monthly).
    rcond:
        Relative cut-off for every pseudo-inverse of ``Sigma_ff`` and ``A'A``
        (eigen-directions below ``rcond * lambda_max`` are dropped). ``1e-6``
        drops a factor whose standard deviation is below one thousandth of the
        largest factor's; such a direction is a numerically dead factor (it
        appears when fewer narratives than ``K`` are selected) whose
        ``mu^2 / sigma^2`` would otherwise inject noise into the MVE Sharpe
        criterion and the MVE weights (D43, verification finding of
        2026-09-06). Legitimate small factors are orders of magnitude above
        this cut.
    t_crit:
        Critical value for counting significant pricing errors.
    placebo_n:
        Number of i.i.d. placebo narratives appended for the selection
        robustness test of BKS Appendix C.2 (``0`` skips the test). Each
        placebo matches the time-series variance of a randomly chosen real
        narrative's shocks.
    placebo_seed:
        Seed for the placebo generator.
    """

    annualization: float = 12.0
    rcond: float = 1e-6
    t_crit: float = 1.96
    placebo_n: int = 0
    placebo_seed: int = 12345

    def __post_init__(self) -> None:
        if self.annualization <= 0:
            raise ValueError("annualization must be > 0")
        if self.placebo_n < 0:
            raise ValueError("placebo_n must be >= 0")


# ---------------------------------------------------------------------------
# The composed pipeline configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineConfig:
    """Everything needed to reproduce one production run."""

    data: DataConfig = field(default_factory=DataConfig)
    shocks: ShockConfig = field(default_factory=ShockConfig)
    covariance: CovarianceConfig = field(default_factory=CovarianceConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    oos: OOSConfig = field(default_factory=OOSConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    run_wrapup: bool = True
    output_dir: str | None = None
    save_panel: bool = False
    name: str = "narrative-ipca-run"

    def to_dict(self) -> dict[str, Any]:
        return config_to_dict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        return config_from_dict(cls, d)

    def hash(self) -> str:
        return config_hash(self)


# ---------------------------------------------------------------------------
# Simulation (DESIGN.md Part D)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AssetClassSpec:
    """One asset class in the simulated multi-asset universe.

    Attributes
    ----------
    name:
        Label carried into ``asset_meta["asset_class"]``.
    share:
        Fraction of ``n_assets`` in this class (shares are renormalised).
    idio_vol_annual:
        Annualised idiosyncratic volatility of the class's assets.
    beta_mean:
        Mean of the class's long-run loadings on the K factors (length K;
        shorter tuples are zero-padded, longer ones truncated).
    beta_sd:
        Cross-sectional standard deviation of the long-run loadings.
    """

    name: str
    share: float
    idio_vol_annual: float
    beta_mean: tuple[float, ...]
    beta_sd: float = 0.3


@dataclass(frozen=True)
class SimulationConfig:
    """Data-generating process with known ground truth (DESIGN.md Part D, D37-D40).

    The DGP follows BKS Figure 1: K state variables ``x_tau = f_tau + nu_tau``
    (``f`` tradable, ``nu`` orthogonal to returns), L topic shocks
    ``z_tau = A x_tau + eta_tau`` with a row-sparse ``A`` (``n_relevant``
    nonzero rows, ``n_placebo`` pure-noise rows, the rest persistent noise),
    daily attention levels built from the shocks, and asset returns
    ``r_{i,tau} = beta_{i,t(tau)} f_tau + eps_{i,tau}`` with slowly
    time-varying loadings and class-specific idiosyncratic volatility.
    """

    seed: int = 0
    n_assets: int = 500
    n_topics: int = 120
    n_relevant: int = 20
    n_placebo: int = 20
    K: int = 3
    n_years: int = 20
    days_per_year: int = 252
    period: str = "M"
    # --- factor / state process ---
    mve_sharpe_annual: float = 1.0
    factor_vol_annual: tuple[float, ...] = (0.16, 0.08, 0.06)
    factor_ar1: float = 0.0
    nontradable_share: float = 0.5
    # --- topic / attention process ---
    signal_strength: float = 1.0
    topic_noise_vol: float = 1.0
    topic_noise_ar1: float = 0.0
    attention_model: Literal["additive", "softmax"] = "additive"
    attention_persistence: float = 0.98
    attention_slow_vol: float = 0.05
    attention_level_mean: float = 1.0
    attention_level_dispersion: float = 0.5
    # --- asset / return process ---
    asset_classes: tuple[AssetClassSpec, ...] = (
        AssetClassSpec("equity", 0.6, 0.30, (1.0, 0.3, 0.0), 0.4),
        AssetClassSpec("credit", 0.2, 0.08, (0.3, -0.2, 0.2), 0.15),
        AssetClassSpec("rates", 0.1, 0.06, (-0.2, 0.0, 0.4), 0.15),
        AssetClassSpec("commodity", 0.1, 0.25, (0.2, 0.5, -0.3), 0.3),
    )
    beta_ar1: float = 0.9
    beta_innov_sd: float = 0.05
    unbalanced_fraction: float = 0.3
    missing_day_fraction: float = 0.01
    fat_tails_df: float | None = None

    def __post_init__(self) -> None:
        if self.n_relevant + self.n_placebo > self.n_topics:
            raise ValueError("n_relevant + n_placebo must be <= n_topics")
        if self.K < 1 or self.n_assets < 2 or self.n_topics < 1:
            raise ValueError("K >= 1, n_assets >= 2, n_topics >= 1 required")
        if self.n_relevant < self.K and self.signal_strength > 0:
            raise ValueError("n_relevant must be >= K for the states to be identifiable")
        if not (0.0 <= self.unbalanced_fraction <= 1.0):
            raise ValueError("unbalanced_fraction must lie in [0, 1]")
        if not (0.0 <= self.missing_day_fraction < 1.0):
            raise ValueError("missing_day_fraction must lie in [0, 1)")
        if len(self.asset_classes) == 0:
            raise ValueError("at least one asset class is required")


# ---------------------------------------------------------------------------
# Harness (DESIGN.md Part E)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HarnessThresholds:
    """Pass/fail thresholds for the simulation harness (best guesses, D40).

    Every threshold is a documented judgement call; see DESIGN.md Part E for
    the reasoning and for which scenario each applies to. Selection recall is
    judged twice: over all relevant topics (``selection_recall_min``, 0.5,
    because the simulated relevant topics have heterogeneous strengths and
    the weak ones are legitimately left out) and over the strong half
    (``selection_recall_strong_min``, 0.8). Loading recovery is judged at the
    level of the implied betas ``c Gamma`` (``beta_canonical_corr_min``),
    which is what the estimator identifies; the loading-subspace cosine and
    the impact-vector correlation are computed on the selected relevant rows
    only (DESIGN.md Part E). The two ``null_*``
    thresholds apply to the ``no_factor`` scenario only (returns without a
    common factor structure): ``null_selection_lift`` is the number of
    selected narratives relative to a 5% chance level (``round(0.05 L)``),
    and ``null_oos_sharpe_abs_max = 0.75`` is about two standard errors of an
    annualised Sharpe ratio estimated from 90-100 monthly out-of-sample
    returns (``2 * sqrt(12 / n_oos)``).
    """

    selection_recall_min: float = 0.50
    selection_recall_strong_min: float = 0.80
    selection_precision_min: float = 0.60
    beta_canonical_corr_min: float = 0.90
    placebo_selected_max: int = 0
    gamma_subspace_cos_min: float = 0.85
    factor_canonical_corr_min: float = 0.90
    state_canonical_corr_min: float = 0.80
    impact_vector_spearman_min: float = 0.70
    oos_sharpe_ratio_to_true_min: float = 0.50
    systematic_r2_min: float = 0.50
    null_selection_lift_max: float = 2.0
    null_oos_sharpe_abs_max: float = 0.75


@dataclass(frozen=True)
class HarnessConfig:
    """Which simulation scenarios to run and with how many seeds."""

    n_seeds: int = 3
    scenarios: tuple[str, ...] = ("baseline", "no_factor", "topic_null", "softmax", "weak")
    thresholds: HarnessThresholds = field(default_factory=HarnessThresholds)
    output_dir: str = "reports/simulation"
    fast: bool = False


# ---------------------------------------------------------------------------
# (De)serialisation helpers
# ---------------------------------------------------------------------------
def config_to_dict(cfg: Any) -> Any:
    """Recursively convert dataclass configs to JSON-serialisable dicts."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, (list, tuple)):
        return [config_to_dict(v) for v in cfg]
    if isinstance(cfg, dict):
        return {str(k): config_to_dict(v) for k, v in cfg.items()}
    return cfg


def _coerce(field_type: Any, value: Any) -> Any:
    """Best-effort coercion of a JSON value to the annotated field type."""
    if value is None:
        return None
    origin = getattr(field_type, "__origin__", None)
    # dataclass field
    if isinstance(field_type, type) and is_dataclass(field_type):
        return config_from_dict(field_type, value) if isinstance(value, dict) else value
    if isinstance(field_type, str):
        # postponed annotations: resolve the few names we use
        name = field_type.split("|")[0].strip()
        resolved = globals().get(name)
        if isinstance(resolved, type) and is_dataclass(resolved) and isinstance(value, dict):
            return config_from_dict(resolved, value)
        if name.startswith("tuple"):
            return _coerce_tuple(name, value)
        return value
    if origin is tuple:
        args = getattr(field_type, "__args__", ())
        inner = args[0] if args else None
        return tuple(_coerce(inner, v) if inner is not None else v for v in value)
    if origin is not None and str(origin) in ("typing.Union", "<class 'types.UnionType'>"):
        for a in field_type.__args__:
            if a is type(None):
                continue
            try:
                return _coerce(a, value)
            except Exception:  # pragma: no cover - fall through to next member
                continue
        return value
    return value


def _coerce_tuple(name: str, value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if "AssetClassSpec" in name:
            return tuple(config_from_dict(AssetClassSpec, v) if isinstance(v, dict) else v for v in value)
        return tuple(value)
    return value


def config_from_dict(cls: type, d: dict[str, Any]) -> Any:
    """Build dataclass ``cls`` from a (possibly nested) dict, ignoring unknown keys."""
    if not (isinstance(cls, type) and is_dataclass(cls)):
        raise TypeError(f"{cls!r} is not a dataclass")
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in d.items():
        if key not in known:
            continue
        f = known[key]
        ftype = f.type
        # Resolve dataclass-typed fields from their default factories when
        # annotations are strings (from __future__ import annotations).
        default_obj = None
        if f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
            default_obj = f.default_factory()  # type: ignore[misc]
        if default_obj is not None and is_dataclass(default_obj) and isinstance(value, dict):
            kwargs[key] = config_from_dict(type(default_obj), value)
        else:
            kwargs[key] = _coerce(ftype, value)
    return cls(**kwargs)


def load_config(path: str | Path, cls: type = PipelineConfig) -> Any:
    """Load a JSON or YAML config file into ``cls`` (YAML needs PyYAML)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # optional dependency

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return config_from_dict(cls, data)


def save_config(cfg: Any, path: str | Path) -> None:
    """Write a config to JSON or YAML (by extension)."""
    path = Path(path)
    data = config_to_dict(cfg)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # optional dependency

        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def config_hash(cfg: Any) -> str:
    """Stable 12-hex-digit hash of a config, for naming output folders."""
    blob = json.dumps(config_to_dict(cfg), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
