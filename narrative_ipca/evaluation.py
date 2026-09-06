"""Evaluation of a narrative-IPCA run (DESIGN.md step 9, decisions D34-D36).

The module collects the yardsticks BKS use to judge a fit:

* model fit on the estimation panel -- Total and Predictive R2 (BKS Section
  3.2 item 1; KKPS footnote 33),
* mean-variance efficiency -- the in-sample MVE Sharpe ratio of the factors
  and the realised Sharpe ratio of an out-of-sample MVE series (BKS Section
  4.2, D34),
* cross-sectional pricing -- time-series alphas of test assets with the
  Gibbons-Ross-Shanken joint test (BKS Table 1, D35),
* correlations with benchmark factors (BKS Table C.1),
* the placebo-narrative selection test (BKS Appendix C.2, D36),

and assembles them into an :class:`~narrative_ipca.types.EvaluationReport`.

Units and conventions
---------------------
* Returns are in the units of the inputs (decimal per period). Alphas are
  reported in those units (D35); nothing is multiplied by 100.
* Sharpe ratios are annualised with ``annualization`` periods per year
  (``mean / std * sqrt(annualization)``, ``ddof = 1``, D34).
* Every function is pure and never raises on a degenerate but well-formed
  input: undefined statistics come back as ``nan`` (or ``None`` for the GRS
  pair), with a log message, so that a report can still be written.
* The pricing regressions are implemented directly in numpy (D41: statsmodels
  is not required). Newey-West standard errors are available through
  ``nw_lags`` and match ``statsmodels`` ``cov_type="HAC"`` without the
  small-sample correction.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _stats

from .config import EvaluationConfig, PipelineConfig
from .types import (
    EvaluationReport,
    IPCAPanel,
    LambdaPathPoint,
    OOSResult,
    PlaceboResult,
    PricingTestResult,
    ShockPanel,
    SparseIPCAResult,
    TuningResult,
    WrapUpResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "realized_sharpe",
    "total_r2",
    "predictive_r2",
    "price_test_assets",
    "grs_test",
    "newey_west_cov",
    "factor_correlations",
    "jaccard",
    "selection_stability",
    "lambda_max_by_instrument",
    "pricing_summary",
    "placebo_test",
    "evaluate_run",
]

#: Smallest-to-largest eigenvalue ratio below which a covariance matrix is
#: treated as singular by :func:`grs_test`. The GRS statistic needs a true
#: inverse: a pseudo-inverse would silently drop the part of ``alpha`` living
#: in the null space and the ``F(N, T-N-K)`` reference would no longer apply.
GRS_SINGULAR_RATIO: float = 1e-10


# ---------------------------------------------------------------------------
# Sharpe ratios and R2 (D34, BKS Section 3.2)
# ---------------------------------------------------------------------------
def realized_sharpe(x: pd.Series | np.ndarray | Sequence[float], annualization: float) -> float:
    """Annualised realised Sharpe ratio ``mean(x) / std(x) * sqrt(annualization)`` (D34).

    Implements the Sharpe ratio BKS report for realised out-of-sample MVE
    returns (Section 4.2, Table 2): the sample mean over the sample standard
    deviation (``ddof = 1``), scaled by the square root of the number of
    periods per year.

    Assumptions: ``x`` holds per-period *excess* returns; non-finite entries
    are dropped before the moments are taken (NaN-safe). Returns ``nan`` when
    fewer than two finite observations remain or the series is constant.
    """
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        logger.debug("realized_sharpe: fewer than two finite observations")
        return float("nan")
    sd = float(arr.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        logger.debug("realized_sharpe: zero or non-finite standard deviation")
        return float("nan")
    return float(arr.mean() / sd * np.sqrt(float(annualization)))


def _check_gamma(panel: IPCAPanel, Gamma: np.ndarray) -> np.ndarray:
    G = np.asarray(Gamma, dtype=float)
    if G.ndim != 2 or G.shape[0] != panel.p:
        raise ValueError(f"Gamma must have shape ({panel.p}, K), got {G.shape}")
    return G


def _uncentred_r2(y: np.ndarray, fitted: np.ndarray) -> float:
    """``1 - sum (y - fitted)^2 / sum y^2`` (the BKS/KKPS R2 without demeaning)."""
    denom = float(y @ y)
    if not np.isfinite(denom) or denom <= 0.0:
        logger.debug("R2 undefined: sum of squared returns is zero")
        return float("nan")
    resid = y - fitted
    return float(1.0 - float(resid @ resid) / denom)


def total_r2(panel: IPCAPanel, Gamma: np.ndarray, F: np.ndarray | pd.DataFrame) -> float:
    """Total R2 of a fit, BKS Section 3.2 item 1 (KKPS footnote 33).

    ``Total R2 = 1 - sum_{i,t} (r_{i,t} - c_{i,t-1} Gamma f_t)^2 / sum_{i,t} r_{i,t}^2``.

    The panel is in long form: row ``n`` holds ``c_{i,t-1}`` in ``panel.X[n]``
    and ``r_{i,t}`` in ``panel.y[n]`` with ``t = panel.t_idx[n]`` indexing the
    rows of ``F`` (``(T, K)``, aligned with ``panel.periods``). The R2 is
    uncentred (raw sum of squares in the denominator) and can be negative for
    a poor fit. Invariant to invertible rotations ``Gamma -> Gamma R``,
    ``f -> R^-1 f`` (D38).
    """
    G = _check_gamma(panel, Gamma)
    Fm = np.asarray(F, dtype=float)
    if Fm.ndim == 1:
        Fm = Fm[:, None]
    if Fm.shape != (panel.T, G.shape[1]):
        raise ValueError(f"F must have shape ({panel.T}, {G.shape[1]}), got {Fm.shape}")
    B = panel.X @ G  # (n_obs, K): beta_{i,t-1} = c_{i,t-1} Gamma
    fitted = np.einsum("nk,nk->n", B, Fm[panel.t_idx])
    return _uncentred_r2(panel.y, fitted)


def predictive_r2(panel: IPCAPanel, Gamma: np.ndarray, mu_f: np.ndarray) -> float:
    """Predictive R2, KKPS footnote 33 (``pred_r2`` of ``SparseIPCAResult``).

    ``Predictive R2 = 1 - sum_{i,t} (r_{i,t} - c_{i,t-1} Gamma mu_f)^2 / sum_{i,t} r_{i,t}^2``,

    i.e. the Total R2 with the factor realisation ``f_t`` replaced by the
    factor premium ``mu_f`` (the in-sample mean of ``f_t``): the share of
    realised return variation explained by the model's expected returns.
    Uncentred, as :func:`total_r2`.
    """
    G = _check_gamma(panel, Gamma)
    mu = np.asarray(mu_f, dtype=float).ravel()
    if mu.shape != (G.shape[1],):
        raise ValueError(f"mu_f must have shape ({G.shape[1]},), got {mu.shape}")
    fitted = panel.X @ (G @ mu)
    return _uncentred_r2(panel.y, fitted)


# ---------------------------------------------------------------------------
# Time-series regressions and pricing tests (D35, BKS Table 1)
# ---------------------------------------------------------------------------
def newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West (1987) HAC covariance of OLS coefficients with a Bartlett kernel.

    ``V = (X'X)^-1 S (X'X)^-1`` with
    ``S = sum_t u_t u_t' + sum_{j=1..lags} (1 - j/(lags+1)) sum_{t>j} (u_t u_{t-j}' + u_{t-j} u_t')``
    and scores ``u_t = x_t e_t``. No small-sample correction is applied, so the
    result equals ``statsmodels`` ``cov_type="HAC"`` with ``use_correction=False``.
    ``lags = 0`` gives the White heteroskedasticity-robust covariance.
    """
    X = np.asarray(X, dtype=float)
    e = np.asarray(resid, dtype=float).ravel()
    if X.shape[0] != e.shape[0]:
        raise ValueError("X and resid must have the same number of rows")
    if lags < 0:
        raise ValueError("lags must be >= 0")
    U = X * e[:, None]
    S = U.T @ U
    for j in range(1, int(lags) + 1):
        if j >= U.shape[0]:
            break
        w = 1.0 - j / (lags + 1.0)
        Gj = U[j:].T @ U[:-j]
        S += w * (Gj + Gj.T)
    XtX_inv = np.linalg.pinv(X.T @ X)
    return XtX_inv @ S @ XtX_inv


def _ols_with_intercept(y: np.ndarray, F: np.ndarray, nw_lags: int | None) -> dict[str, Any]:
    """OLS of ``y_t = alpha + beta' f_t + e_t``; plain or Newey-West standard errors.

    Plain standard errors use ``sigma^2 (X'X)^-1`` with ``sigma^2 = e'e / (T - K - 1)``.
    R2 is the centred time-series R2 ``1 - RSS / TSS``.
    """
    T = y.shape[0]
    X = np.column_stack([np.ones(T), F])
    k = X.shape[1]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = T - k
    if dof > 0:
        if nw_lags is None:
            sigma2 = float(resid @ resid) / dof
            cov = sigma2 * np.linalg.pinv(X.T @ X)
        else:
            cov = newey_west_cov(X, resid, int(nw_lags))
        var = np.clip(np.diag(cov), 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(var > 0, coef / np.sqrt(var), np.nan)
    else:
        t = np.full(k, np.nan)
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - float(resid @ resid) / tss) if tss > 0 else float("nan")
    return {"coef": coef, "t": t, "resid": resid, "r2": r2, "n": T}


def _multivariate_ols(Y: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS of every column of ``Y`` on ``[1, F]``; returns (alphas, residual matrix)."""
    X = np.column_stack([np.ones(F.shape[0]), F])
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return B[0].copy(), Y - X @ B


def grs_test(
    alphas: np.ndarray, residuals: np.ndarray, factors: np.ndarray
) -> tuple[float, float] | tuple[None, None]:
    """Gibbons-Ross-Shanken (1989) joint test that all ``N`` alphas are zero (D35).

    With ``T`` periods, ``N`` test assets and ``K`` factors, the finite-sample
    statistic implemented is

    ``F = (T / (T-K-1)) * ((T-N-K) / N) * (1 + mu_f' Omega^-1 mu_f)^-1 * alpha' Sigma_e^-1 alpha
       ~ F(N, T-N-K)`` under the null,

    where ``Sigma_e = E'E / (T-K-1)`` is the *unbiased* residual covariance
    (``K+1`` regressors per regression), ``mu_f`` the factor sample mean and
    ``Omega = (F-mu)'(F-mu) / T`` the maximum-likelihood factor covariance.
    Substituting ``Sigma_e = (T / (T-K-1)) Sigma_ML`` shows this is exactly the
    original GRS form ``((T-N-K)/N)(1 + mu'Omega^-1 mu)^-1 alpha' Sigma_ML^-1 alpha``
    with ML residual covariance; the ``T/(T-K-1)`` factor is what makes the
    unbiased-covariance version exact (Hotelling ``T^2`` with ``T-K-1``
    Wishart degrees of freedom). The statistic is invariant to a common
    rescaling of ``alphas`` and ``residuals``.

    Assumptions: residuals come from per-asset OLS regressions with an
    intercept on exactly these ``factors`` over the same ``T`` periods, and
    errors are i.i.d. normal (the finite-sample distribution needs this).

    Returns ``(None, None)`` when ``T <= N + K + 1``, when any input is
    non-finite, or when ``Sigma_e`` (or ``Omega``) is singular by the
    :data:`GRS_SINGULAR_RATIO` criterion.
    """
    a = np.asarray(alphas, dtype=float).ravel()
    E = np.asarray(residuals, dtype=float)
    Fm = np.asarray(factors, dtype=float)
    if E.ndim == 1:
        E = E[:, None]
    if Fm.ndim == 1:
        Fm = Fm[:, None]
    if E.ndim != 2 or Fm.ndim != 2:
        raise ValueError("residuals must be (T, N) and factors (T, K)")
    T, N = E.shape
    if a.shape != (N,):
        raise ValueError(f"alphas must have shape ({N},), got {a.shape}")
    if Fm.shape[0] != T:
        raise ValueError(f"factors must have {T} rows, got {Fm.shape[0]}")
    K = Fm.shape[1]
    if N < 1 or T <= N + K + 1:
        logger.info("GRS not computed: need T > N + K + 1 (T=%d, N=%d, K=%d)", T, N, K)
        return None, None
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(E)) and np.all(np.isfinite(Fm))):
        logger.info("GRS not computed: non-finite inputs")
        return None, None

    Sigma_e = E.T @ E / (T - K - 1)
    mu = Fm.mean(axis=0)
    Fc = Fm - mu
    Omega = Fc.T @ Fc / T
    for name, M in (("residual covariance", Sigma_e), ("factor covariance", Omega)):
        w = np.linalg.eigvalsh(0.5 * (M + M.T))
        if not np.isfinite(w[-1]) or w[-1] <= 0.0 or w[0] <= GRS_SINGULAR_RATIO * w[-1]:
            logger.info("GRS not computed: %s is singular", name)
            return None, None

    sr2 = float(mu @ np.linalg.solve(Omega, mu))
    quad = float(a @ np.linalg.solve(Sigma_e, a))
    stat = (T / (T - K - 1.0)) * ((T - N - K) / N) * quad / (1.0 + sr2)
    if not np.isfinite(stat) or stat < 0.0:
        return None, None
    pvalue = float(_stats.f.sf(stat, N, T - N - K))
    return float(stat), pvalue


def price_test_assets(
    test_assets: pd.DataFrame,
    factors: pd.DataFrame,
    t_crit: float = 1.96,
    model_name: str = "",
    nw_lags: int | None = None,
) -> PricingTestResult:
    """Time-series pricing errors of test assets on a factor set (BKS Table 1, D35).

    For every test asset ``j`` the regression ``r_{j,t} = alpha_j + beta_j' f_t + e_{j,t}``
    is estimated by OLS over the periods where the asset and all factors are
    observed. Reported per asset: ``alpha_j`` in the return units of the
    inputs, its t-statistic, ``beta_j``, and the centred R2. Summary
    statistics follow BKS Table 1: the cross-sectional average ``|alpha|``,
    average ``|t|``, the fraction with ``|t| > t_crit`` and the GRS statistic
    of :func:`grs_test`.

    Standard errors are plain OLS (``sigma^2 (X'X)^-1``) unless ``nw_lags``
    is given, in which case Newey-West HAC errors with that many Bartlett
    lags are used (:func:`newey_west_cov`). Only the t-statistics change.

    The GRS test needs one common sample, so it is computed on the balanced
    sub-sample of periods where *every* test asset and factor is observed
    (identical to the per-asset samples when the test-asset panel is
    balanced, the usual case for sorted portfolios). It is ``None`` when that
    sub-sample has ``T <= N + K + 1`` periods.
    """
    if isinstance(test_assets, pd.Series):
        test_assets = test_assets.to_frame()
    if isinstance(factors, pd.Series):
        factors = factors.to_frame()
    if test_assets.shape[1] == 0 or factors.shape[1] == 0:
        raise ValueError("test_assets and factors must each have at least one column")

    common = test_assets.index.intersection(factors.index)
    fa = factors.loc[common].astype(float)
    ta = test_assets.loc[common].astype(float)
    fmask = np.isfinite(fa.to_numpy()).all(axis=1)
    fa = fa.loc[fmask]
    ta = ta.loc[fmask]
    Fm = fa.to_numpy()
    K = Fm.shape[1]
    names = list(ta.columns)
    fnames = list(fa.columns)
    N = len(names)

    alphas = np.full(N, np.nan)
    tstats = np.full(N, np.nan)
    betas = np.full((N, K), np.nan)
    r2s = np.full(N, np.nan)
    Y = ta.to_numpy()
    for j, name in enumerate(names):
        y = Y[:, j]
        ok = np.isfinite(y)
        n_ok = int(ok.sum())
        if n_ok < K + 2:
            logger.warning("price_test_assets[%s]: asset %r has %d usable periods (< K + 2); skipped",
                           model_name, name, n_ok)
            continue
        res = _ols_with_intercept(y[ok], Fm[ok], nw_lags)
        alphas[j] = res["coef"][0]
        tstats[j] = res["t"][0]
        betas[j] = res["coef"][1:]
        r2s[j] = res["r2"]

    grs_stat: float | None = None
    grs_p: float | None = None
    bal = np.isfinite(Y).all(axis=1)
    Tb = int(bal.sum())
    if N >= 1 and Tb > N + K + 1:
        a_b, E_b = _multivariate_ols(Y[bal], Fm[bal])
        grs_stat, grs_p = grs_test(a_b, E_b, Fm[bal])
    else:
        logger.info("price_test_assets[%s]: GRS skipped (balanced T=%d, N=%d, K=%d)", model_name, Tb, N, K)

    finite_t = tstats[np.isfinite(tstats)]
    finite_a = alphas[np.isfinite(alphas)]
    return PricingTestResult(
        alphas=pd.Series(alphas, index=names, name="alpha"),
        t_stats=pd.Series(tstats, index=names, name="t_stat"),
        betas=pd.DataFrame(betas, index=names, columns=fnames),
        r2=pd.Series(r2s, index=names, name="r2"),
        avg_abs_alpha=float(np.mean(np.abs(finite_a))) if finite_a.size else float("nan"),
        avg_abs_t=float(np.mean(np.abs(finite_t))) if finite_t.size else float("nan"),
        frac_significant=float(np.mean(np.abs(finite_t) > t_crit)) if finite_t.size else float("nan"),
        grs_stat=grs_stat,
        grs_pvalue=grs_p,
        model_name=model_name,
    )


def pricing_summary(tests: dict[str, PricingTestResult]) -> pd.DataFrame:
    """One row per model with the BKS Table 1 summary statistics."""
    rows = []
    for key, res in tests.items():
        rows.append(
            {
                "model": key,
                "n_test_assets": int(len(res.alphas)),
                "n_factors": int(res.betas.shape[1]),
                "avg_abs_alpha": res.avg_abs_alpha,
                "avg_abs_t": res.avg_abs_t,
                "frac_significant": res.frac_significant,
                "grs_stat": np.nan if res.grs_stat is None else res.grs_stat,
                "grs_pvalue": np.nan if res.grs_pvalue is None else res.grs_pvalue,
            }
        )
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Factor correlations (BKS Table C.1)
# ---------------------------------------------------------------------------
def factor_correlations(F: pd.DataFrame, others: pd.DataFrame, min_obs: int = 3) -> pd.DataFrame:
    """Pearson correlations ``corr(F_k, other_j)`` on the common, pairwise-complete sample.

    Returns a ``(K_F, K_others)`` DataFrame indexed by the columns of ``F``.
    Entries with fewer than ``min_obs`` jointly finite observations are ``nan``.
    """
    if isinstance(F, pd.Series):
        F = F.to_frame()
    if isinstance(others, pd.Series):
        others = others.to_frame()
    common = F.index.intersection(others.index)
    A = F.loc[common].to_numpy(dtype=float)
    B = others.loc[common].to_numpy(dtype=float)
    out = np.full((A.shape[1], B.shape[1]), np.nan)
    if len(common) < min_obs:
        logger.warning("factor_correlations: only %d common periods", len(common))
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            m = np.isfinite(A[:, i]) & np.isfinite(B[:, j])
            if m.sum() < min_obs:
                continue
            a, b = A[m, i], B[m, j]
            if a.std() <= 0 or b.std() <= 0:
                continue
            out[i, j] = float(np.corrcoef(a, b)[0, 1])
    return pd.DataFrame(out, index=list(F.columns), columns=list(others.columns))


# ---------------------------------------------------------------------------
# Selection-set helpers (placebo test, OOS stability)
# ---------------------------------------------------------------------------
def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    """Jaccard similarity ``|A ∩ B| / |A ∪ B|`` of two sets; ``1.0`` when both are empty."""
    A, B = set(a), set(b)
    union = A | B
    if not union:
        return 1.0
    return float(len(A & B) / len(union))


def selection_stability(selected_history: pd.DataFrame) -> float:
    """Mean Jaccard similarity between the selected sets of consecutive refits.

    ``selected_history`` is a boolean ``(n_refits, L)`` DataFrame (one row per
    refit, one column per narrative). ``nan`` with fewer than two refits.
    """
    if selected_history is None or selected_history.shape[0] < 2:
        return float("nan")
    M = selected_history.to_numpy().astype(bool)
    cols = np.asarray(selected_history.columns)
    sims = [jaccard(cols[M[r - 1]], cols[M[r]]) for r in range(1, M.shape[0])]
    return float(np.mean(sims))


def lambda_max_by_instrument(
    path: Sequence[LambdaPathPoint],
    instrument_names: Sequence[str],
    K: int | None = None,
    tol: float = 0.0,
) -> pd.Series:
    """Per-instrument ``lambda_max``: the largest ``lambda`` on the path at which the row is selected.

    BKS Appendix C.2 (Figure C.2) rank instruments by ``max_lambda_l``, the
    maximum penalty at which ``||Gamma_l||_2 > 0`` still holds along the
    regularisation path. Path points with a different ``K`` are ignored when
    ``K`` is given. Instruments never selected on the path get ``nan``.

    ``gamma_norms`` of a path point may have length ``p = L + 1`` (constant
    row included, aligned with ``instrument_names``) or ``L`` (constant row
    omitted, aligned with ``instrument_names[1:]``).
    """
    names = list(instrument_names)
    out = np.full(len(names), np.nan)
    for pt in path:
        if K is not None and int(pt.K) != int(K):
            continue
        norms = np.asarray(pt.gamma_norms, dtype=float).ravel()
        if norms.shape[0] == len(names):
            sl = slice(None)
        elif norms.shape[0] == len(names) - 1:
            sl = slice(1, None)
        else:
            raise ValueError(
                f"gamma_norms of length {norms.shape[0]} cannot be aligned with {len(names)} instruments"
            )
        cur = out[sl]
        sel = norms > tol
        out[sl] = np.where(sel & (np.isnan(cur) | (pt.lam > cur)), pt.lam, cur)
    return pd.Series(out, index=names, name="lam_max")


# ---------------------------------------------------------------------------
# Placebo test (BKS Appendix C.2, D36)
# ---------------------------------------------------------------------------
def placebo_test(
    shocks: ShockPanel,
    returns: pd.DataFrame,
    cfg: PipelineConfig,
    reference_fit: SparseIPCAResult,
    n: int,
    seed: int,
) -> PlaceboResult:
    """Placebo-narrative selection test of BKS Appendix C.2 (D36).

    ``n`` i.i.d. normal placebo shock series, each with the time-series
    variance of a randomly chosen real narrative, are appended to the shock
    panel; the pipeline from Eq. 6 onward (covariances -> panel -> tuning) is
    re-run on the augmented instrument set; the report states how many
    placebos are selected at the tuned ``lambda*``, the per-instrument
    ``lambda_max`` along the tuned path, and how much the real selected set
    moved (``jaccard_real`` = Jaccard similarity of the real selected sets
    before and after adding the placebos; ``reference_fit`` is the fit
    without placebos).

    ``returns`` is the daily excess-return panel on the shock calendar
    (``AlignedData.returns``). The stage modules are imported lazily so that
    this module stays importable on its own.
    """
    from . import covariances as _covariances
    from . import panel as _panel
    from . import shocks as _shocks
    from . import tuning as _tuning

    if n < 1:
        raise ValueError("n (number of placebo narratives) must be >= 1")

    augmented, mask = _shocks.append_placebos(shocks, n, seed)
    mask = np.asarray(mask, dtype=bool).ravel()
    topics = list(augmented.topics)
    if mask.shape[0] != len(topics):
        raise ValueError("append_placebos returned a mask that does not match the augmented topics")
    placebo_names = [t for t, m in zip(topics, mask) if m]
    real_names = [t for t, m in zip(topics, mask) if not m]
    logger.info("placebo_test: %d real + %d placebo narratives, seed=%d", len(real_names), len(placebo_names), seed)

    cov = _covariances.build_covariance_panel(augmented, returns, cfg.covariance, cfg.data.period)
    pnl = _panel.build_panel(cov, returns, cfg.data, cfg.covariance)
    tr = _tuning.tune(pnl, cfg.estimation, cfg.tuning, cfg.evaluation)
    fit = tr.fit

    placebo_set = set(placebo_names)
    selected_after = list(fit.selected_topics)
    n_placebo_selected = sum(1 for t in selected_after if t in placebo_set)
    real_after = [t for t in selected_after if t not in placebo_set]
    real_before = list(reference_fit.selected_topics)

    lam_max_all = lambda_max_by_instrument(tr.path, list(fit.instrument_names), K=tr.K)
    topic_set = set(topics)
    lam_max = lam_max_all.loc[[nm for nm in fit.instrument_names if nm in topic_set]]

    result = PlaceboResult(
        n_placebo=len(placebo_names),
        n_placebo_selected=int(n_placebo_selected),
        n_real_selected=int(len(real_after)),
        lam_max_by_instrument=lam_max,
        lam_star=float(tr.lam),
        real_selected_before=real_before,
        real_selected_after=real_after,
        jaccard_real=jaccard(real_before, real_after),
    )
    logger.info(
        "placebo_test: %d/%d placebos selected at lambda*=%.4g; real selected %d -> %d (Jaccard %.3f)",
        result.n_placebo_selected, result.n_placebo, result.lam_star, len(real_before), len(real_after),
        result.jaccard_real,
    )
    return result


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _eval_cfg(cfg: PipelineConfig | EvaluationConfig) -> EvaluationConfig:
    return cfg.evaluation if hasattr(cfg, "evaluation") else cfg  # type: ignore[return-value]


def _populated_mask(panel: IPCAPanel | None, fit: SparseIPCAResult) -> np.ndarray:
    """Periods that carry observations (their ``f_t`` is an estimate, not a zero placeholder)."""
    T = int(fit.F.shape[0])
    if panel is not None and panel.T == T:
        return np.asarray(panel.moments().n) > 0
    if fit.populated is not None:
        return np.asarray(fit.populated, dtype=bool).ravel()
    return np.ones(T, dtype=bool)


def _mve_series(fit: SparseIPCAResult, populated: np.ndarray, rcond: float) -> pd.Series:
    """In-sample MVE factor ``f^MVE_t = b_MVE' f_t`` over populated periods."""
    b = fit.b_mve(rcond=rcond)
    F = fit.factors_frame().loc[populated]
    return pd.Series(F.to_numpy() @ b, index=F.index, name="mve")


def _selected_table(fit: SparseIPCAResult) -> pd.DataFrame:
    names = list(fit.instrument_names[1:])
    norms = np.asarray(fit.gamma_norms, dtype=float).ravel()[1:]
    sel = np.asarray(fit.selected, dtype=bool).ravel()
    df = pd.DataFrame({"topic": names, "gamma_norm": norms})[sel]
    df = df.sort_values("gamma_norm", ascending=False, kind="stable").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def evaluate_run(
    panel: IPCAPanel,
    tuning: TuningResult | None,
    fit: SparseIPCAResult,
    oos: OOSResult | None,
    wrapup: WrapUpResult | None,
    cfg: PipelineConfig | EvaluationConfig,
    test_assets: pd.DataFrame | None = None,
    benchmark_factors: pd.DataFrame | None = None,
    placebo: PlaceboResult | None = None,
) -> EvaluationReport:
    """Assemble the :class:`EvaluationReport` of one run (DESIGN.md step 9).

    ``metrics`` always holds ``total_r2``, ``pred_r2`` (recomputed on
    ``panel`` from ``fit``), ``mve_sharpe_is`` (``sqrt(ann * mu' Sigma^-1 mu)``),
    ``lam_star``, ``K``, ``n_selected``, ``frac_selected``, ``n_obs``, ``T``,
    ``N``, ``L``, ``converged``, ``n_iter``, ``objective``; with ``tuning``:
    ``lam_max``, ``n_path_points``; with ``oos``: ``oos_sharpe`` (realised,
    D34), ``oos_n_periods``, ``oos_n_refits``, ``oos_mean_n_selected``,
    ``oos_mean_lam``, ``oos_selection_stability`` (mean Jaccard between
    consecutive refits' selected sets); with ``wrapup``:
    ``wrapup_rank_deficient`` and ``obs_r2_<name>`` per projected observable;
    with ``placebo``: ``placebo_n``, ``placebo_n_selected``,
    ``placebo_jaccard_real``; with ``test_assets``: the Table 1 summary of
    every pricing test as ``pricing_<model>_<stat>``.

    ``pricing_tests`` holds ``"narrative_is"`` (in-sample factors over
    populated periods), ``"narrative_oos"`` (when ``oos`` is given) and
    ``"benchmark"`` (when ``benchmark_factors`` is given), all against
    ``test_assets``. ``factor_correlations`` correlates the in-sample factors
    and their MVE combination with ``benchmark_factors``. ``tables`` holds
    ``"selected"`` (topic, gamma_norm, rank), ``"pricing"`` and, with ``oos``,
    ``"oos_refits"``.

    Boolean flags are stored as ``0/1`` so that ``metrics`` stays numeric.
    """
    ecfg = _eval_cfg(cfg)
    ann, rcond, t_crit = float(ecfg.annualization), float(ecfg.rcond), float(ecfg.t_crit)
    metrics: dict[str, float] = {}

    populated = _populated_mask(panel, fit)
    metrics["total_r2"] = total_r2(panel, fit.Gamma, fit.F)
    metrics["pred_r2"] = predictive_r2(panel, fit.Gamma, fit.mu_f)
    metrics["mve_sharpe_is"] = float(fit.mve_sharpe(annualization=ann, rcond=rcond))
    metrics["lam_star"] = float(tuning.lam) if tuning is not None else float(fit.lam)
    metrics["K"] = int(fit.K)
    metrics["n_selected"] = int(fit.n_selected)
    metrics["frac_selected"] = float(fit.n_selected / len(fit.selected)) if len(fit.selected) else float("nan")
    metrics["n_obs"] = int(panel.n_obs)
    metrics["T"] = int(panel.T)
    metrics["N"] = int(panel.N)
    metrics["L"] = int(panel.L)
    metrics["n_populated_periods"] = int(populated.sum())
    metrics["converged"] = int(bool(fit.converged))
    metrics["n_iter"] = int(fit.n_iter)
    metrics["objective"] = float(fit.objective)
    if abs(metrics["total_r2"] - float(fit.total_r2)) > 1e-6 and np.isfinite(fit.total_r2):
        logger.warning("evaluate_run: fit.total_r2=%.6f differs from recomputed %.6f", fit.total_r2, metrics["total_r2"])

    lambda_path = None
    if tuning is not None:
        metrics["lam_max"] = float(tuning.lam_max)
        metrics["n_path_points"] = int(len(tuning.path))
        lambda_path = tuning.path_frame()

    tables: dict[str, pd.DataFrame] = {"selected": _selected_table(fit)}

    if oos is not None:
        oos_sharpe = realized_sharpe(oos.mve, ann)
        metrics["oos_sharpe"] = oos_sharpe
        if np.isfinite(oos_sharpe) and np.isfinite(oos.sharpe) and abs(oos_sharpe - float(oos.sharpe)) > 1e-8:
            logger.warning("evaluate_run: oos.sharpe=%.6f differs from realized_sharpe %.6f", oos.sharpe, oos_sharpe)
        mve_arr = np.asarray(oos.mve, dtype=float)
        metrics["oos_n_periods"] = int(np.isfinite(mve_arr).sum())
        metrics["oos_n_refits"] = int(len(oos.refit_periods))
        nsel = pd.Series(oos.n_selected_history, dtype=float)
        metrics["oos_mean_n_selected"] = float(nsel.mean()) if len(nsel) else float("nan")
        lam_hist = pd.Series(oos.lam_history, dtype=float)
        metrics["oos_mean_lam"] = float(lam_hist.mean()) if len(lam_hist) else float("nan")
        metrics["oos_selection_stability"] = selection_stability(oos.selected_history)
        tables["oos_refits"] = pd.DataFrame(
            {
                "lam": oos.lam_history,
                "K": oos.K_history,
                "n_selected": oos.n_selected_history,
                "is_sharpe": oos.is_sharpe_history,
            }
        )

    if wrapup is not None:
        metrics["wrapup_rank_deficient"] = int(bool(wrapup.rank_deficient))
        for name, info in (wrapup.obs_projection or {}).items():
            if isinstance(info, dict):
                r2 = info.get("r2", info.get("R2", info.get("rsquared")))
                if r2 is not None:
                    metrics[f"obs_r2_{name}"] = float(r2)

    if placebo is not None:
        metrics["placebo_n"] = int(placebo.n_placebo)
        metrics["placebo_n_selected"] = int(placebo.n_placebo_selected)
        metrics["placebo_n_real_selected"] = int(placebo.n_real_selected)
        metrics["placebo_jaccard_real"] = float(placebo.jaccard_real)

    F_is = fit.factors_frame().loc[populated]
    pricing: dict[str, PricingTestResult] = {}
    if test_assets is not None:
        pricing["narrative_is"] = price_test_assets(test_assets, F_is, t_crit, "narrative_is")
        if oos is not None and oos.factors is not None and len(oos.factors):
            pricing["narrative_oos"] = price_test_assets(test_assets, oos.factors, t_crit, "narrative_oos")
        if benchmark_factors is not None:
            pricing["benchmark"] = price_test_assets(test_assets, benchmark_factors, t_crit, "benchmark")
        for key, res in pricing.items():
            metrics[f"pricing_{key}_avg_abs_alpha"] = res.avg_abs_alpha
            metrics[f"pricing_{key}_avg_abs_t"] = res.avg_abs_t
            metrics[f"pricing_{key}_frac_significant"] = res.frac_significant
            metrics[f"pricing_{key}_grs_stat"] = np.nan if res.grs_stat is None else res.grs_stat
            metrics[f"pricing_{key}_grs_pvalue"] = np.nan if res.grs_pvalue is None else res.grs_pvalue
        tables["pricing"] = pricing_summary(pricing)

    corr = None
    if benchmark_factors is not None:
        F_with_mve = F_is.copy()
        F_with_mve["mve"] = _mve_series(fit, populated, rcond)
        corr = factor_correlations(F_with_mve, benchmark_factors)

    logger.info(
        "evaluate_run: total_r2=%.4f pred_r2=%.4f mve_sharpe_is=%.3f n_selected=%d lam*=%.4g%s",
        metrics["total_r2"], metrics["pred_r2"], metrics["mve_sharpe_is"], metrics["n_selected"],
        metrics["lam_star"], f" oos_sharpe={metrics['oos_sharpe']:.3f}" if oos is not None else "",
    )
    return EvaluationReport(
        metrics=metrics,
        lambda_path=lambda_path,
        selected_topics=list(fit.selected_topics),
        gamma_norms=pd.Series(np.asarray(fit.gamma_norms, dtype=float).ravel(), index=list(fit.instrument_names), name="gamma_norm"),
        pricing_tests=pricing,
        placebo=placebo,
        factor_correlations=corr,
        tables=tables,
    )
