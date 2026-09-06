"""Sparse IPCA (BKS Eq. 8) and plain IPCA (KKPS Alg. 1) on panel moments.

The estimator touches the data only through the per-period sufficient
statistics of :class:`narrative_ipca.types.PanelMoments`:
``S_t = C_{t-1}' C_{t-1}``, ``V_t = C_{t-1}' r_t``, ``yy_t = r_t' r_t`` and
``n_t``. Everything below is written against those.

Objective (BKS Eq. 8, DESIGN.md A.2)
------------------------------------
    min_{Gamma, {f_t}}  0.5 sum_{i,t in S} (r_{i,t} - c_{i,t-1} Gamma f_t)^2
                      + lambda N_S sum_{l=0..L} sigma^c_l ||Gamma_l||_2
                      + sum_{t in S} ||f_t||_2^2

Alternating regularised least squares (App. B.2):

* f-step (Eq. 16): ``f_t = (Gamma' S_t Gamma + 2 I_K)^-1 Gamma' V_t``.
* Gamma-step: group lasso on ``vect(Gamma)`` with the Gram pair
  ``G = sum_t S_t kron (f_t f_t')``, ``b = sum_t V_t kron f_t``
  (coordinate ``l*K + k`` holds ``Gamma[l, k]``), solved by
  :func:`narrative_ipca.grouplasso.group_lasso_gram`.

Both steps are exact or majorisation-descent block minimisations, so the
objective path is non-increasing (D21, asserted by the tests).

Numerical reparametrisation: standardised instrument coordinates
----------------------------------------------------------------
The covariance instruments have scales of ``1e-7 .. 1e-5`` next to the
constant column of ``1``, so on a full-size panel the Gamma-step Gram ``G``
written in the units of ``c`` has a condition number of order ``1e15``
(verification finding, 2026-09-06): the inner group lasso then stalls above
its absolute tolerance (``Gamma`` entries of order ``1e5`` sit on a rounding
floor above ``inner_tol``), drifts along near-null directions, and the fit
is not invariant to the units of the attention series. The solver therefore
runs in *standardised* coordinates. With ``sigma = panel.sigma_c``
(``sigma_0 = 1``) and ``D = diag(1/sigma)``:

    X~ = X D,    Gamma~ = diag(sigma) Gamma,    S~_t = D S_t D,    V~_t = D V_t,

so that ``X Gamma = X~ Gamma~`` (same fitted values),
``Gamma' S_t Gamma = Gamma~' S~_t Gamma~`` and ``Gamma' V_t = Gamma~' V~_t``
(the f-step is unchanged), and the penalty is uniform across rows:

    lambda N_S sum_l sigma_l ||Gamma_l||_2 = lambda N_S sum_l ||Gamma~_l||_2.

Eq. 8 is therefore the same function of the same unknowns written in other
coordinates, not a different estimator: every objective value, fitted
value, factor, row norm ``sigma_l ||Gamma_l||`` and selected set is
identical, and the result is reported as ``Gamma = Gamma~ / sigma`` in the
original units, so BKS App. B.1's point that the regressors are *not*
standardised (so that ``Gamma`` keeps its magnitude) is respected. What
does change is the starting point: the SVD initialisation of D20 is taken
in the standardised coordinates, which makes it (and ``lambda_max``, which
is conditional on the warm-up factors, D22) invariant to the units of the
inputs. :func:`objective_value` keeps evaluating Eq. 8 in the original units
and agrees with the reported ``objective`` to rounding.

Scale identity used by the tests: at any block-stationary point of Eq. 8 the
scaling ``Gamma -> a Gamma, f -> f/a`` leaves the SSR unchanged, and
stationarity along that ray gives ``lambda N_S sum_l sigma_l ||Gamma_l|| =
2 sum_t ||f_t||^2`` (penalty term equals twice the ridge term).

At ``lambda = 0`` Eq. 8 has no scale identification (D18); that case is the
plain IPCA ALS of KKPS Algorithm 1 with the ``Theta_Y`` normalisation
(``Gamma'Gamma = I_K``, ``(1/T) sum_t f_t f_t'`` diagonal descending, first
non-zero entry of each ``Gamma`` column positive), see :func:`fit_ipca`.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, Sequence

import numpy as np

from .config import EstimationConfig
from .grouplasso import group_lasso_gram
from .types import IPCAPanel, LambdaPathPoint, PanelMoments, SparseIPCAResult

logger = logging.getLogger(__name__)

__all__ = [
    "fit_sparse_ipca",
    "fit_ipca",
    "f_step",
    "gram_from_moments",
    "objective_value",
    "ssr_from_moments",
    "penalty_vector",
    "standardized_penalties",
    "standardized_moments",
    "to_standardized",
    "from_standardized",
    "initial_gamma",
    "lambda_max",
    "lambda_grid",
    "lambda_path",
    "canonicalize",
    "fitted_values",
    "betas",
]

ProgressFn = Callable[[int, int, LambdaPathPoint], None]

#: Relative cut-off of every pseudo-inverse in this module (D43).
_RCOND = 1e-12
#: Relative size of the seeded jitter added to the SVD initialisation (D20).
_JITTER = 1e-3
#: Ridge of the f-step inherited from the ``sum_t ||f_t||^2`` term of Eq. 8.
_RIDGE = 2.0


# ---------------------------------------------------------------------------
# elementary maps
# ---------------------------------------------------------------------------
def betas(X: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    """Loadings ``beta_{i,t} = c_{i,t} Gamma`` (BKS Eq. 5/7) for the rows of ``X``.

    ``X`` is ``(n, p)`` with the constant in column 0, ``Gamma`` is ``(p, K)``;
    returns ``(n, K)``.
    """
    X = np.asarray(X, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    if X.ndim != 2 or Gamma.ndim != 2 or X.shape[1] != Gamma.shape[0]:
        raise ValueError(f"shape mismatch: X {X.shape}, Gamma {Gamma.shape}")
    return X @ Gamma


def fitted_values(panel: IPCAPanel, Gamma: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Model fit ``c_{i,t-1} Gamma f_t`` for every row of the long-form panel (Eq. 7).

    Long-form counterpart of :func:`ssr_from_moments`; provided for callers
    that hold ``panel.X`` and need per-observation residuals.
    """
    Gamma = np.asarray(Gamma, dtype=float)
    F = np.asarray(F, dtype=float)
    _check_shapes(panel.p, panel.T, Gamma, F)
    B = betas(panel.X, Gamma)
    return np.einsum("nk,nk->n", B, F[panel.t_idx])


def f_step(S: np.ndarray, V: np.ndarray, Gamma: np.ndarray, ridge: float = _RIDGE) -> np.ndarray:
    """Factor step for all periods at once, BKS Eq. 16.

    ``f_t = (Gamma' S_t Gamma + ridge I_K)^-1 Gamma' V_t`` is the minimiser of
    ``0.5 ||r_t - C_{t-1} Gamma f||^2 + (ridge/2) ||f||^2`` per period; with
    ``ridge = 2`` this is the ``sum_t ||f_t||^2`` term of Eq. 8 (App. B.2).
    ``ridge = 0`` gives the plain IPCA F-step of KKPS Alg. 1 via a
    pseudo-inverse (singular when a period has fewer than K assets).

    Assumptions: ``S`` is ``(T, p, p)`` symmetric PSD, ``V`` is ``(T, p)``,
    ``Gamma`` is ``(p, K)``. Empty periods (``S_t = 0``, ``V_t = 0``) get
    ``f_t = 0`` in both branches. Returns ``(T, K)``. The step is invariant
    to the standardisation of the module docstring: ``(S, V, Gamma)`` and
    ``(S~, V~, Gamma~)`` give the same ``F``.
    """
    F, _, _ = _f_step_parts(S, V, Gamma, ridge)
    return F


def _f_step_parts(S: np.ndarray, V: np.ndarray, Gamma: np.ndarray, ridge: float = _RIDGE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`f_step` plus the per-period projections it is built from.

    Returns ``(F, A, rhs)`` with ``A_t = sym(Gamma' S_t Gamma)`` (``(T, K, K)``,
    without the ridge) and ``rhs_t = Gamma' V_t`` (``(T, K)``). They are the
    sufficient statistics of the SSR at ``(Gamma, F)``,
    ``sum_t [yy_t - 2 f_t' rhs_t + f_t' A_t f_t]`` (:func:`_ssr_from_parts`),
    so the ARLS loop evaluates the objective in ``O(T K^2)`` from what the
    f-step already computed instead of a second ``O(T p^2)`` pass.

    ``S_t Gamma`` for all ``t`` is one GEMM on the ``(T p, p)`` view of ``S``.
    """
    S = np.asarray(S, dtype=float)
    V = np.asarray(V, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    if S.ndim != 3 or V.ndim != 2 or Gamma.ndim != 2:
        raise ValueError(f"bad ranks: S {S.shape}, V {V.shape}, Gamma {Gamma.shape}")
    T, p = V.shape
    if S.shape != (T, p, p) or Gamma.shape[0] != p:
        raise ValueError(f"shape mismatch: S {S.shape}, V {V.shape}, Gamma {Gamma.shape}")
    K = int(Gamma.shape[1])
    if T == 0:
        return np.zeros((0, K)), np.zeros((0, K, K)), np.zeros((0, K))
    SG = (S.reshape(T * p, p) @ Gamma).reshape(T, p, K)  # S_t Gamma for all t, one GEMM
    A = np.einsum("lk,tlj->tkj", Gamma, SG)  # Gamma' S_t Gamma, (T, K, K)
    A = 0.5 * (A + np.transpose(A, (0, 2, 1)))
    rhs = V @ Gamma  # Gamma' V_t, (T, K)
    if ridge > 0.0:
        M = A + float(ridge) * np.eye(K)
        try:
            F = np.linalg.solve(M, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:  # pragma: no cover - M is PD by construction
            F = np.einsum("tkj,tj->tk", np.linalg.pinv(M, rcond=_RCOND, hermitian=True), rhs)
    else:
        F = np.einsum("tkj,tj->tk", np.linalg.pinv(A, rcond=_RCOND, hermitian=True), rhs)
    if not np.all(np.isfinite(F)):  # pragma: no cover - defensive
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(F), A, rhs


def _ssr_from_parts(sum_yy: float, A: np.ndarray, rhs: np.ndarray, F: np.ndarray) -> float:
    """SSR ``sum_yy - 2 sum_t f_t' rhs_t + sum_t f_t' A_t f_t`` from the f-step projections.

    Algebraically identical to :func:`ssr_from_moments` (substitute
    ``rhs_t = Gamma' V_t`` and ``A_t = Gamma' S_t Gamma``); clipped at zero
    against rounding.
    """
    cross = float(np.einsum("tk,tk->", F, rhs))
    quad = float(np.einsum("tk,tkj,tj->", F, A, F))
    return max(float(sum_yy) - 2.0 * cross + quad, 0.0)


def gram_from_moments(S: np.ndarray, V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gram pair of the Gamma-step (DESIGN.md A.2, BKS App. B.2).

    ``G = sum_t S_t kron (f_t f_t')`` and ``b = sum_t V_t kron f_t`` for the
    coordinate map ``vect(Gamma)_{l*K + k} = Gamma[l, k]``, i.e. the
    normal equations of the regression of ``r_{i,t}`` on ``c_{i,t-1} kron f_t``
    pooled over the panel. ``G[(l,k), (m,j)] = sum_t S_t[l,m] f_t[k] f_t[j]``.

    Returns ``G`` of shape ``(pK, pK)`` (symmetric PSD) and ``b`` of shape ``(pK,)``.
    Called with the standardised moments (:func:`standardized_moments`) it
    returns the Gram pair of the standardised problem, ``G~ = (D kron I) G (D kron I)``
    and ``b~ = (D kron I) b``.
    """
    S = np.asarray(S, dtype=float)
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=float)
    T, p = V.shape
    K = int(F.shape[1])
    if S.shape != (T, p, p) or F.shape != (T, K):
        raise ValueError(f"shape mismatch: S {S.shape}, V {V.shape}, F {F.shape}")
    FF = np.einsum("tk,tj->tkj", F, F).reshape(T, K * K)  # rows vect(f_t f_t')
    # sum_t S_t (x) f_t f_t' as one GEMM over t: (p p, T) @ (T, K K) -> (p, p, K, K), then
    # interleave to (l, k, m, j). The transposed (T, p p) view of S goes to BLAS as-is
    # (no transposed copy of S per call).
    G4 = S.reshape(T, p * p).T @ FF
    G = np.ascontiguousarray(G4.reshape(p, p, K, K).transpose(0, 2, 1, 3).reshape(p * K, p * K))
    G = 0.5 * (G + G.T)
    b = (V.T @ F).reshape(p * K)
    return G, np.ascontiguousarray(b)


def ssr_from_moments(S: np.ndarray, V: np.ndarray, yy: np.ndarray, Gamma: np.ndarray, F: np.ndarray) -> float:
    """Sum of squared residuals from the period moments.

    ``sum_t [ yy_t - 2 f_t' Gamma' V_t + f_t' Gamma' S_t Gamma f_t ]``, which
    equals ``sum_{i,t} (r_{i,t} - c_{i,t-1} Gamma f_t)^2`` expanded in the
    sufficient statistics, so no long-form pass is needed. Clipped at zero
    against rounding.
    """
    B = np.asarray(F, dtype=float) @ np.asarray(Gamma, dtype=float).T  # (T, p): Gamma f_t
    cross = float(np.einsum("tl,tl->", B, V))
    quad = float(np.einsum("tl,tlm,tm->", B, S, B))
    return max(float(np.sum(yy)) - 2.0 * cross + quad, 0.0)


def penalty_vector(lam: float, n_obs: int, sigma_c: np.ndarray, penalize_intercept: bool = True) -> np.ndarray:
    """Per-row group penalties ``lambda N_S sigma^c_l`` of Eq. 8 (``sigma^c_0 = 1``).

    With ``penalize_intercept=False`` the constant row gets penalty 0 (D19
    keeps the BKS default ``True``). These are the penalties on the rows of
    ``Gamma`` in the original units; :func:`standardized_penalties` gives
    the (uniform) penalties on the rows of ``Gamma~``.
    """
    pen = float(lam) * float(n_obs) * np.asarray(sigma_c, dtype=float).ravel().copy()
    if not penalize_intercept:
        pen[0] = 0.0
    return pen


def standardized_penalties(lam: float, n_obs: int, p: int, penalize_intercept: bool = True) -> np.ndarray:
    """Per-row penalties of the standardised problem: ``lambda N_S`` for every row.

    Because ``||Gamma~_l|| = sigma_l ||Gamma_l||`` the row weights
    ``sigma_l`` of Eq. 8 are absorbed into the coordinates and every row of
    ``Gamma~`` carries the same penalty (``0`` for the constant row when
    ``penalize_intercept`` is False). Equals ``penalty_vector(lam, n_obs,
    ones(p), penalize_intercept)``.
    """
    return penalty_vector(lam, n_obs, np.ones(int(p)), penalize_intercept)


def _check_sigma(sigma_c: np.ndarray, p: int) -> np.ndarray:
    sigma = np.asarray(sigma_c, dtype=float).ravel()
    if sigma.shape != (p,):
        raise ValueError(f"sigma_c must have shape ({p},), got {sigma.shape}")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("sigma_c must be finite and strictly positive")
    return sigma


def standardized_moments(mom: PanelMoments, sigma_c: np.ndarray) -> PanelMoments:
    """Moments of the standardised instruments ``X~ = X diag(1/sigma)`` (module docstring).

    ``S~_t = D S_t D`` and ``V~_t = D V_t`` with ``D = diag(1/sigma_c)``;
    ``yy`` and ``n`` are unchanged. ``O(T p^2)``, i.e. free next to a fit, so
    it is rebuilt per call rather than cached on the panel. Any positive
    ``sigma_c`` gives an exact reparametrisation; ``panel.sigma_c`` (the
    penalty weights of Eq. 8) is the one that makes the penalty uniform and
    the columns of ``X~`` unit-scale.
    """
    sigma = _check_sigma(sigma_c, mom.p)
    d = 1.0 / sigma
    S = np.ascontiguousarray(np.asarray(mom.S, dtype=float) * (d[None, :, None] * d[None, None, :]))
    V = np.ascontiguousarray(np.asarray(mom.V, dtype=float) * d[None, :])
    return PanelMoments(S=S, V=V, yy=np.asarray(mom.yy, dtype=float), n=np.asarray(mom.n))


def to_standardized(Gamma: np.ndarray, sigma_c: np.ndarray) -> np.ndarray:
    """``Gamma~ = diag(sigma_c) Gamma``: the coefficient map of the standardised instruments."""
    Gamma = np.asarray(Gamma, dtype=float)
    sigma = _check_sigma(sigma_c, Gamma.shape[0])
    return Gamma * sigma[:, None]


def from_standardized(Gamma_tilde: np.ndarray, sigma_c: np.ndarray) -> np.ndarray:
    """``Gamma = diag(1/sigma_c) Gamma~``: back to the original instrument units (exact zeros stay zero)."""
    Gamma_tilde = np.asarray(Gamma_tilde, dtype=float)
    sigma = _check_sigma(sigma_c, Gamma_tilde.shape[0])
    return Gamma_tilde / sigma[:, None]


def objective_value(
    panel: IPCAPanel,
    Gamma: np.ndarray,
    F: np.ndarray,
    lam: float,
    sigma_c: np.ndarray,
    penalize_intercept: bool = True,
) -> float:
    """BKS Eq. 8: ``0.5 SSR + lam N_S sum_l sigma_l ||Gamma_l||_2 + sum_t ||f_t||^2``.

    Evaluated in the original units of ``Gamma`` and the panel instruments.
    ``N_S = panel.n_obs``; the SSR comes from :func:`ssr_from_moments`. The
    ridge on ``f`` is not scaled by ``lam N_S`` (that is what makes the
    ``+2 I_K`` of Eq. 16 exact). Equals the ``objective`` reported by
    :func:`fit_sparse_ipca` on its own ``(Gamma, F)`` up to rounding (the fit
    evaluates the same expression in the standardised coordinates).
    """
    Gamma = np.asarray(Gamma, dtype=float)
    F = np.asarray(F, dtype=float)
    _check_shapes(panel.p, panel.T, Gamma, F)
    sigma_c = np.asarray(sigma_c, dtype=float).ravel()
    if sigma_c.shape != (panel.p,):
        raise ValueError(f"sigma_c must have shape ({panel.p},), got {sigma_c.shape}")
    mom = panel.moments()
    ssr = ssr_from_moments(mom.S, mom.V, mom.yy, Gamma, F)
    pen = penalty_vector(lam, panel.n_obs, sigma_c, penalize_intercept)
    return 0.5 * ssr + float(pen @ np.linalg.norm(Gamma, axis=1)) + float(np.sum(F * F))


# ---------------------------------------------------------------------------
# initialisation (D20 / KKPS Alg. 1 lines 1-5)
# ---------------------------------------------------------------------------
def _managed_portfolio_basis(V: np.ndarray, n: np.ndarray, K: int, rng: np.random.Generator, jitter: float) -> np.ndarray:
    """Top-K left singular vectors of ``Y = [V_t / n_t]_t`` (``p x T``), plus jitter.

    KKPS Alg. 1 initialises with the leading eigenvectors of ``Y Y'`` where
    ``y_t = Z_t' x_t / N``; the left singular vectors of ``Y`` are the same
    thing. Directions beyond the rank of ``Y`` are filled with seeded random
    vectors so that ``Gamma`` always has ``K`` independent columns.

    The SVD is not scale-free: the fits call this with the standardised
    ``V~_t = D V_t`` (module docstring), so that the leading directions are
    those of the unit-scale managed portfolios and the starting point does
    not depend on the units of the instruments.
    """
    T, p = V.shape
    pop = n > 0
    W = np.zeros((p, T))
    W[:, pop] = (V[pop] / n[pop, None]).T
    U = np.zeros((p, K))
    if np.any(W != 0.0):
        u, s, _ = np.linalg.svd(W, full_matrices=False)
        rank = int(np.sum(s > s[0] * 1e-12)) if s.size else 0
        take = min(K, rank)
        U[:, :take] = u[:, :take]
        if take < K:
            U[:, take:] = rng.standard_normal((p, K - take)) / np.sqrt(p)
    else:
        U = rng.standard_normal((p, K)) / np.sqrt(p)
    if jitter > 0.0:
        U = U + jitter * rng.standard_normal((p, K)) / np.sqrt(p)
    return U


def initial_gamma(S: np.ndarray, V: np.ndarray, n: np.ndarray, K: int, seed: int) -> np.ndarray:
    """Starting ``Gamma`` for the penalised problem (D20), in the coordinates of ``(S, V)``.

    SVD of the managed-portfolio matrix (columns ``V_t / n_t``), a seeded
    jitter of relative size ``1e-3``, then a rescaling so that
    ``trace(Gamma' S_bar Gamma) = 2K`` with ``S_bar`` the mean of ``S_t`` over
    populated periods, i.e. the signal term of Eq. 16 is on the scale of its
    ``2 I_K`` ridge. Scale is only a starting point; the penalty balance of
    Eq. 8 pins it at the optimum.

    :func:`fit_sparse_ipca` and :func:`lambda_max` pass the *standardised*
    moments (:func:`standardized_moments`), so the returned matrix is a
    ``Gamma~`` and the starting point is invariant to the units of the
    instruments (the trace rescaling is invariant either way). This changes
    where the ARLS starts, not the estimator.
    """
    rng = np.random.default_rng(int(seed))
    U = _managed_portfolio_basis(np.asarray(V, float), np.asarray(n), int(K), rng, _JITTER)
    pop = np.asarray(n) > 0
    if np.any(pop):
        S_bar = np.asarray(S, float)[pop].mean(axis=0)
        q = float(np.einsum("lk,lm,mk->", U, S_bar, U))
    else:
        q = 0.0
    scale = float(np.sqrt(2.0 * K / q)) if (q > 0.0 and np.isfinite(q)) else 1.0
    return U * scale


def _warm_start(mom_t: PanelMoments, K: int, cfg: EstimationConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Initialisation plus ``cfg.n_warmup`` unpenalised ARLS sweeps (D20).

    ``mom_t`` must be the *standardised* moments (:func:`standardized_moments`);
    the returned ``Gamma`` is then a ``Gamma~`` and ``F`` the (coordinate-free)
    factors of its f-step.
    """
    S, V, n = mom_t.S, mom_t.V, mom_t.n
    p = mom_t.p
    Gamma = initial_gamma(S, V, n, K, cfg.seed)
    F = f_step(S, V, Gamma)
    zero_pen = np.zeros(p)
    inner_iters = 0
    for _ in range(int(cfg.n_warmup)):
        G, b = gram_from_moments(S, V, F)
        x, info = group_lasso_gram(G, b, K, zero_pen, x0=Gamma.ravel(), max_iter=int(cfg.inner_max_iter), tol=float(cfg.inner_tol))
        inner_iters += info["n_iter"]
        Gamma = x.reshape(p, K)
        F = f_step(S, V, Gamma)
    return Gamma, F, {"n_warmup": int(cfg.n_warmup), "warmup_inner_iters": inner_iters}


# ---------------------------------------------------------------------------
# Sparse IPCA fit
# ---------------------------------------------------------------------------
def fit_sparse_ipca(
    panel: IPCAPanel,
    cfg: EstimationConfig,
    lam: float | None = None,
    K: int | None = None,
    Gamma_init: np.ndarray | None = None,
    penalize_intercept: bool | None = None,
) -> SparseIPCAResult:
    """Fit BKS Eq. 8 by alternating regularised least squares (App. B.2).

    ``lam None`` uses ``cfg.lam`` (which must then be set); ``lam == 0``
    delegates to :func:`fit_ipca` (D18). ``K`` and ``penalize_intercept``
    default to the config. With ``Gamma_init`` (original units) the SVD
    initialisation and the warm-up sweeps are skipped (regularisation path,
    D22).

    The whole loop runs in the standardised coordinates of the module
    docstring: the moments are rescaled once (:func:`standardized_moments`),
    ``Gamma_init`` is converted on entry (:func:`to_standardized`), the
    penalties are the uniform :func:`standardized_penalties`, and the
    solution is converted back on exit (:func:`from_standardized`). This is
    an exact reparametrisation of Eq. 8; ``Gamma``, ``gamma_norms``,
    ``selected``, ``objective``/``obj_path``, ``total_r2`` and ``F`` are all
    reported in the original units and ``objective`` equals
    :func:`objective_value` on the returned pair up to rounding.

    Loop: f-step (Eq. 16) -> Gamma-step (group lasso, warm-started from the
    previous ``Gamma~``) -> objective; stop when the relative objective change
    is below ``cfg.tol`` or after ``cfg.max_iter`` sweeps (D23).
    ``obj_path[0]`` is the objective at the starting point and every later
    entry follows one penalised sweep, so the path is non-increasing.

    Factor moments ``mu_f``, ``Sigma_ff`` use ``ddof=1`` over populated
    periods (D25). ``total_r2 = 1 - SSR/sum r^2``,
    ``pred_r2 = 1 - SSR(f_t := mu_f)/sum r^2`` (BKS Section 3.2).

    ``meta`` carries ``penalties`` (original units, ``lam N_S sigma_l``),
    ``penalties_tilde`` (uniform), ``sigma_c`` and ``coordinates =
    "standardized"`` next to the iteration counters.
    """
    lam_v = cfg.lam if lam is None else lam
    if lam_v is None:
        raise ValueError("lam is None: pass lam explicitly or set EstimationConfig.lam")
    lam_v = float(lam_v)
    if lam_v < 0.0 or not np.isfinite(lam_v):
        raise ValueError(f"lam must be finite and >= 0, got {lam_v}")
    K_v = int(cfg.K if K is None else K)
    if K_v < 1:
        raise ValueError(f"K must be >= 1, got {K_v}")
    pen_int = bool(cfg.penalize_intercept if penalize_intercept is None else penalize_intercept)
    if panel.n_obs == 0:
        raise ValueError("panel is empty")
    if lam_v == 0.0:
        return fit_ipca(panel, K_v, cfg)

    mom = panel.moments()
    p = panel.p
    sigma_c = _check_sigma(panel.sigma_c, p)
    mom_t = standardized_moments(mom, sigma_c)
    S, V, yy, n = mom_t.S, mom_t.V, mom_t.yy, mom_t.n
    penalties = penalty_vector(lam_v, panel.n_obs, sigma_c, pen_int)  # original units, reported
    pen_t = standardized_penalties(lam_v, panel.n_obs, p, pen_int)  # uniform, used by the solver
    populated = n > 0

    meta: dict[str, Any] = {
        "penalize_intercept": pen_int,
        "penalties": penalties,
        "penalties_tilde": pen_t,
        "sigma_c": sigma_c.copy(),
        "coordinates": "standardized",
    }
    if Gamma_init is None:
        Gamma_t, _, warm_info = _warm_start(mom_t, K_v, cfg)
        meta.update(warm_info)
        meta["init"] = "svd+warmup"
    else:
        Gamma_in = np.array(Gamma_init, dtype=float, copy=True)
        if Gamma_in.shape != (p, K_v):
            raise ValueError(f"Gamma_init must have shape ({p}, {K_v}), got {Gamma_in.shape}")
        if not np.all(np.isfinite(Gamma_in)):
            raise ValueError("Gamma_init contains non-finite values")
        Gamma_t = to_standardized(Gamma_in, sigma_c)
        meta["init"] = "warm"
    # The f-step of the starting Gamma~ (identical to the warm-up's last f-step) plus the
    # projections A_t = Gamma~' S~_t Gamma~, rhs_t = Gamma~' V~_t that the objective is read from.
    F, A, rhs = _f_step_parts(S, V, Gamma_t)
    sum_yy = float(np.sum(yy))
    debug = logger.isEnabledFor(logging.DEBUG)

    def obj(Gm: np.ndarray, Fm: np.ndarray, Am: np.ndarray, rm: np.ndarray) -> float:
        # Eq. 8 in standardised coordinates: the SSR from the f-step projections
        # (_ssr_from_parts == ssr_from_moments) and the uniform penalty on the rows of Gamma~
        # (pen_t @ ||Gamma~_l|| == penalties @ ||Gamma_l||).
        ssr = _ssr_from_parts(sum_yy, Am, rm, Fm)
        return 0.5 * ssr + float(pen_t @ np.linalg.norm(Gm, axis=1)) + float(np.sum(Fm * Fm))

    current = obj(Gamma_t, F, A, rhs)
    obj_path: list[float] = [current]
    converged = False
    n_iter = 0
    inner_iters = 0
    inner_all_converged = True
    for it in range(1, int(cfg.max_iter) + 1):
        G, b = gram_from_moments(S, V, F)
        x, info = group_lasso_gram(G, b, K_v, pen_t, x0=Gamma_t.ravel(), max_iter=int(cfg.inner_max_iter), tol=float(cfg.inner_tol))
        inner_iters += info["n_iter"]
        inner_all_converged = inner_all_converged and info["converged"]
        Gamma_t = x.reshape(p, K_v)
        F, A, rhs = _f_step_parts(S, V, Gamma_t)
        new = obj(Gamma_t, F, A, rhs)
        rel = abs(current - new) / max(abs(current), 1e-300)
        obj_path.append(new)
        current = new
        n_iter = it
        if debug:
            logger.debug("sparse ipca lam=%.4g sweep %d obj=%.10g rel=%.2e active=%d", lam_v, it, new, rel, int(np.count_nonzero(np.linalg.norm(Gamma_t[1:], axis=1))))
        if rel < float(cfg.tol):
            converged = True
            break
    if not converged:
        logger.warning("fit_sparse_ipca(lam=%.4g, K=%d) did not converge in %d sweeps", lam_v, K_v, n_iter)
    meta.update({"inner_iters": inner_iters, "inner_all_converged": inner_all_converged})

    Gamma = from_standardized(Gamma_t, sigma_c)
    res = _build_result(panel, mom, Gamma, F, lam_v, K_v, current, obj_path, n_iter, converged, populated, meta)
    logger.info("fit_sparse_ipca lam=%.4g K=%d: %d sweeps, converged=%s, selected=%d/%d, total_r2=%.4f", lam_v, K_v, n_iter, converged, res.n_selected, panel.L, res.total_r2)
    return res


def _build_result(
    panel: IPCAPanel,
    mom: PanelMoments,
    Gamma: np.ndarray,
    F: np.ndarray,
    lam: float,
    K: int,
    objective: float,
    obj_path: list[float],
    n_iter: int,
    converged: bool,
    populated: np.ndarray,
    meta: dict[str, Any],
    selected: np.ndarray | None = None,
) -> SparseIPCAResult:
    """Assemble a :class:`SparseIPCAResult` with the derived statistics (original units)."""
    mu_f, Sigma_ff = _factor_moments(F, populated)
    denom = float(np.sum(mom.yy))
    ssr = ssr_from_moments(mom.S, mom.V, mom.yy, Gamma, F)
    F_mu = np.tile(mu_f, (mom.T, 1)) * populated[:, None]
    ssr_pred = ssr_from_moments(mom.S, mom.V, mom.yy, Gamma, F_mu)
    total_r2 = 1.0 - ssr / denom if denom > 0.0 else 0.0
    pred_r2 = 1.0 - ssr_pred / denom if denom > 0.0 else 0.0
    gamma_norms = np.linalg.norm(Gamma, axis=1)
    if selected is None:
        selected = gamma_norms[1:] > 0.0
    return SparseIPCAResult(
        Gamma=np.ascontiguousarray(Gamma),
        F=np.ascontiguousarray(F),
        mu_f=mu_f,
        Sigma_ff=Sigma_ff,
        lam=float(lam),
        K=int(K),
        objective=float(objective),
        obj_path=[float(v) for v in obj_path],
        n_iter=int(n_iter),
        converged=bool(converged),
        total_r2=float(total_r2),
        pred_r2=float(pred_r2),
        gamma_norms=gamma_norms,
        selected=np.asarray(selected, dtype=bool),
        instrument_names=list(panel.instrument_names),
        periods=panel.periods,
        n_obs=int(panel.n_obs),
        populated=np.asarray(populated, dtype=bool),
        meta=meta,
    )


def _factor_moments(F: np.ndarray, populated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``mu_f`` and ``Sigma_ff`` (ddof=1) over populated periods (D25)."""
    K = int(F.shape[1])
    Fp = F[populated]
    if Fp.shape[0] == 0:
        return np.zeros(K), np.zeros((K, K))
    mu = Fp.mean(axis=0)
    if Fp.shape[0] < 2:
        return mu, np.zeros((K, K))
    Sigma = np.atleast_2d(np.cov(Fp, rowvar=False, ddof=1)).reshape(K, K)
    return mu, 0.5 * (Sigma + Sigma.T)


# ---------------------------------------------------------------------------
# plain IPCA (KKPS Algorithm 1), lambda = 0
# ---------------------------------------------------------------------------
def fit_ipca(panel: IPCAPanel, K: int, cfg: EstimationConfig) -> SparseIPCAResult:
    """Unregularised IPCA by ALS, KKPS Algorithm 1 with the ``Theta_Y`` normalisation (D18).

    Per iteration: F-step ``f_t = (Gamma' S_t Gamma)^+ Gamma' V_t`` (no ridge;
    pseudo-inverse), Gamma-step ``vect(Gamma) = G^+ b`` with the Gram pair of
    :func:`gram_from_moments` (pseudo-inverse because collinear instruments
    make ``G`` singular, D10a), then the normalisation of KKPS App. B.1:
    ``Gamma'Gamma = I_K``, ``(1/T) sum_t f_t f_t'`` diagonal descending, first
    non-zero entry of each ``Gamma`` column positive (App. G.2 [3'']).
    Convergence on the relative change of the SSR below ``cfg.tol``.

    The iterations run in the standardised coordinates of the module
    docstring (``S~``, ``V~``, ``Gamma~``): the SVD initialisation is taken
    there (scale-free starting point) and the pseudo-inverse acts on the
    well-conditioned ``G~`` instead of a Gram whose condition number is set
    by the instrument scales. The ``Theta_Y`` normalisation is applied to the
    iterates ``Gamma~`` (which keeps them unit-scale) and, on exit, to the
    reported pair in the **original units**, so ``Gamma'Gamma = I_K`` holds
    for the returned ``Gamma``. The two are related by an invertible
    rotation, which leaves fitted values and the SSR unchanged. When ``G`` is
    singular the minimum-norm solution is the one of the standardised
    coordinates (``Gamma`` is not unique in that case, DESIGN.md Part F
    point 4).

    After the loop one more F-step and the original-unit normalisation make
    the reported pair consistent (``F`` is the exact F-step of the reported
    ``Gamma``).

    ``objective`` and ``obj_path`` hold the SSR (Eq. 8 at ``lambda = 0`` is
    not scale-identified); ``lam = 0.0``; ``selected`` is all True.
    """
    K = int(K)
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if panel.n_obs == 0:
        raise ValueError("panel is empty")
    mom = panel.moments()
    p = panel.p
    sigma_c = _check_sigma(panel.sigma_c, p)
    mom_t = standardized_moments(mom, sigma_c)
    S, V, yy, n = mom_t.S, mom_t.V, mom_t.yy, mom_t.n
    populated = n > 0
    rng = np.random.default_rng(int(cfg.seed))
    Gamma_t = _managed_portfolio_basis(V, n, K, rng, jitter=0.0)

    obj_path: list[float] = []
    prev = np.inf
    converged = False
    n_iter = 0
    for it in range(1, int(cfg.max_iter) + 1):
        F = f_step(S, V, Gamma_t, ridge=0.0)
        G, b = gram_from_moments(S, V, F)
        x = np.linalg.pinv(G, rcond=_RCOND, hermitian=True) @ b
        Gamma_t = x.reshape(p, K)
        Gamma_t, F = _normalize_theta_y(Gamma_t, F, populated)
        ssr = ssr_from_moments(S, V, yy, Gamma_t, F)
        obj_path.append(ssr)
        n_iter = it
        rel = abs(prev - ssr) / max(abs(prev), 1e-300) if np.isfinite(prev) else np.inf
        logger.debug("ipca K=%d iter %d ssr=%.10g rel=%.2e", K, it, ssr, rel)
        if rel < float(cfg.tol):
            converged = True
            break
        prev = ssr
    # final consistent pair: exact F-step of the last Gamma~ (coordinate-free), then the
    # Theta_Y normalisation in the original units of Gamma.
    F = f_step(S, V, Gamma_t, ridge=0.0)
    ssr = ssr_from_moments(S, V, yy, Gamma_t, F)
    obj_path.append(ssr)
    Gamma = from_standardized(Gamma_t, sigma_c)
    Gamma, F = _normalize_theta_y(Gamma, F, populated)
    if not converged:
        logger.warning("fit_ipca(K=%d) did not converge in %d iterations", K, n_iter)
    meta = {
        "objective_kind": "ssr",
        "normalization": "Theta_Y",
        "init": "svd",
        "sigma_c": sigma_c.copy(),
        "coordinates": "standardized",
    }
    res = _build_result(panel, mom, Gamma, F, 0.0, K, ssr, obj_path, n_iter, converged, populated, meta, selected=np.ones(panel.L, dtype=bool))
    logger.info("fit_ipca K=%d: %d iterations, converged=%s, total_r2=%.4f", K, n_iter, converged, res.total_r2)
    return res


def _normalize_theta_y(Gamma: np.ndarray, F: np.ndarray, populated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate ``(Gamma, F)`` to the ``Theta_Y`` representative (KKPS App. B.1, G.2).

    1. ``Chol' Chol = Gamma'Gamma`` (Cholesky; eigen square root if not PD).
    2. ``OldV = (1/T) sum_t f_t f_t'`` over populated periods.
    3. ``Chol OldV Chol' = Orth Diag Orth'`` with ``Diag`` descending.
    4. ``Gamma' = Gamma Chol^-1 Orth``, ``f'_t = Orth' Chol f_t`` (so
       ``Gamma' f'_t = Gamma f_t``: fitted values are unchanged).
    5. Signs: first non-zero entry of each column of ``Gamma'`` positive.
    """
    K = int(Gamma.shape[1])
    GtG = Gamma.T @ Gamma
    GtG = 0.5 * (GtG + GtG.T)
    try:
        Lc = np.linalg.cholesky(GtG)  # lower, Lc Lc' = Gamma'Gamma -> Chol = Lc'
        Chol = Lc.T
        Gamma_r = np.linalg.solve(Lc, Gamma.T).T  # Gamma Chol^-1
    except np.linalg.LinAlgError:
        w, U = np.linalg.eigh(GtG)
        w = np.clip(w, 0.0, None)
        sq = np.sqrt(w)
        Chol = (U * sq).T  # diag(sq) U' ; Chol'Chol = U diag(w) U'
        inv_sq = np.where(sq > sq.max() * _RCOND if sq.size else False, 1.0 / np.where(sq > 0, sq, 1.0), 0.0)
        Gamma_r = Gamma @ (U * inv_sq)
    Fp = F[populated] if np.any(populated) else F
    OldV = Fp.T @ Fp / max(Fp.shape[0], 1)
    M = Chol @ OldV @ Chol.T
    M = 0.5 * (M + M.T)
    evals, Orth = np.linalg.eigh(M)
    order = np.argsort(evals)[::-1]
    Orth = Orth[:, order]
    Gamma_new = Gamma_r @ Orth
    F_new = (F @ Chol.T) @ Orth
    signs = _first_nonzero_signs(Gamma_new)
    return Gamma_new * signs, F_new * signs


def _first_nonzero_signs(Gamma: np.ndarray) -> np.ndarray:
    """``+1``/``-1`` per column so that its first non-zero entry becomes positive (KKPS [3''])."""
    K = Gamma.shape[1]
    signs = np.ones(K)
    for k in range(K):
        col = Gamma[:, k]
        scale = float(np.max(np.abs(col))) if col.size else 0.0
        if scale <= 0.0:
            continue
        idx = int(np.argmax(np.abs(col) > 1e-12 * scale))
        if col[idx] < 0.0:
            signs[k] = -1.0
    return signs


# ---------------------------------------------------------------------------
# regularisation path (D22)
# ---------------------------------------------------------------------------
def _intercept_only_solution(G00: np.ndarray, b0: np.ndarray, pen: float) -> np.ndarray:
    """Exact minimiser of the one-group problem ``0.5 x'G00 x - b0'x + pen ||x||_2``.

    The intercept block of the Gamma-step with every narrative row held at
    zero. With ``G00 = U diag(d) U'`` and ``c = U'b0`` the minimiser is
    ``x = (G00 + (pen/r) I)^-1 b0`` where ``r = ||x||`` solves
    ``sum_k c_k^2 / (d_k r + pen)^2 = 1`` (strictly decreasing in ``r``), so a
    scalar bisection is exact. ``x = 0`` when ``||b0|| <= pen``; ``pen = 0``
    is the least-squares solution ``G00^+ b0``.
    """
    G00 = 0.5 * (G00 + G00.T)
    K = int(G00.shape[0])
    if pen <= 0.0:
        return np.linalg.pinv(G00, rcond=_RCOND, hermitian=True) @ b0
    nb = float(np.linalg.norm(b0))
    if nb <= pen:
        return np.zeros(K)
    d, U = np.linalg.eigh(G00)
    d = np.clip(d, 0.0, None)
    c = U.T @ b0
    phi = lambda r: float(np.sum(c**2 / (d * r + pen) ** 2)) - 1.0  # phi(0+) > 0, decreasing
    lo = 0.0
    hi = max(float(np.linalg.norm(np.linalg.pinv(G00, rcond=_RCOND, hermitian=True) @ b0)), 1e-300)
    while phi(hi) > 0.0:  # only when b0 has mass outside range(G00): no finite minimiser, cap the norm
        hi *= 2.0
        if hi > 1e150:  # pragma: no cover - defensive
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if phi(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    r = 0.5 * (lo + hi)
    return U @ (c / (d + pen / r))


def lambda_max(panel: IPCAPanel, cfg: EstimationConfig, K: int | None = None) -> float:
    """Smallest ``lambda`` above which every narrative row is zero given the warm-up factors (D22).

    With ``(G, b)`` built from the warm-up ``F`` and ``x*_0(lambda)`` the
    intercept-only solution of the Gamma-step (every narrative row at zero),
    the KKT condition of the group lasso for an inactive row ``l >= 1`` is
    ``||b_l - G_{l0} x*_0|| <= lambda N_S sigma_l``, so the threshold is
    ``h(x*_0) := max_{l>=1} ||b_l - G_{l0} x*_0|| / (N_S sigma_l)``.

    The computation is done in the standardised coordinates of the module
    docstring, where ``b~_l = b_l / sigma_l``, ``G~_{l0} = G_{l0} / sigma_l``
    (``sigma_0 = 1``, so the intercept block and ``x*_0`` are the same in
    both) and the penalty weight is ``N_S`` for every row: the threshold
    ``max_{l>=1} ||b~_l - G~_{l0} x*_0|| / N_S`` is the same number, so
    ``lambda_max`` is invariant to the reparametrisation given the warm-up
    factors (the tests check this numerically). The warm-up itself starts
    from the standardised SVD (D20), which makes the value invariant to the
    units of the instruments as well.

    * ``penalize_intercept=False``: ``x*_0 = G_00^+ b_0`` (least squares) for
      every ``lambda`` and ``lam_max = h(x*_0)``.
    * ``penalize_intercept=True`` (D19 default): ``x*_0(lambda)`` is the
      group-soft-thresholded intercept (:func:`_intercept_only_solution` with
      penalty ``lambda N_S``), which is zero for ``lambda >= lam_dead :=
      ||b_0|| / N_S``. If ``h(0) >= lam_dead`` the intercept dies before the
      last narrative row and ``lam_max = h(0)`` (the whole ``Gamma`` is zero
      there). Otherwise the intercept is alive at the threshold and
      ``lam_max`` is the largest root of ``h(x*_0(lambda)) = lambda`` on
      ``(0, lam_dead)`` (geometric bracket, then bisection).

    Ignoring the intercept penalty (using the least-squares ``x*_0`` under the
    default) mis-places the threshold by a few percent in either direction
    and leaves narrative rows active at the reported ``lam_max``.

    Conditional on the warm-up factors: the first penalised ARLS sweep at
    ``lam_max`` starts from an all-zero narrative block, after which the
    factors move, so this is the usual path end point, not an exact
    guarantee for the joint problem.
    """
    K_v = int(cfg.K if K is None else K)
    if panel.n_obs == 0:
        raise ValueError("panel is empty")
    if panel.L == 0:
        return 0.0
    mom_t = standardized_moments(panel.moments(), _check_sigma(panel.sigma_c, panel.p))
    _, F, _ = _warm_start(mom_t, K_v, cfg)
    G, b = gram_from_moments(mom_t.S, mom_t.V, F)
    L = panel.L
    w = float(panel.n_obs) * np.ones(panel.p)  # pen~_l = lambda * N_S for every row (uniform)
    G00 = 0.5 * (G[:K_v, :K_v] + G[:K_v, :K_v].T)
    b0 = b[:K_v]
    G_n0 = G[K_v:, :K_v]  # narrative rows x intercept block
    b_n = b[K_v:]

    def threshold(x0: np.ndarray) -> float:
        # KKT bound of every narrative row at the intercept-only point x0
        resid = (b_n - G_n0 @ x0).reshape(L, K_v)
        return float(np.max(np.linalg.norm(resid, axis=1) / w[1:]))

    x_ls = _intercept_only_solution(G00, b0, 0.0)
    if not cfg.penalize_intercept:
        value = threshold(x_ls)
    else:
        lam_dead = float(np.linalg.norm(b0)) / w[0]
        lam_dead_threshold = threshold(np.zeros(K_v))
        if lam_dead_threshold >= lam_dead or threshold(x_ls) <= 0.0:
            value = lam_dead_threshold  # intercept already zero at the threshold (or no narrative signal at all)
        else:
            # intercept alive at the threshold: largest root of g(lam) = threshold(x0(lam)) - lam in (0, lam_dead).
            g = lambda lam: threshold(_intercept_only_solution(G00, b0, lam * w[0])) - lam
            hi = lam_dead  # g(hi) = lam_dead_threshold - lam_dead < 0
            lo = hi
            for _ in range(600):  # geometric scan downwards until the narrative block re-activates
                lo *= 0.9
                if g(lo) > 0.0:
                    break
                hi = lo
            else:  # pragma: no cover - g(0+) > 0 was checked above
                lo = 0.0
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if g(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= 1e-13 * hi:
                    break
            value = hi
    logger.info("lambda_max(K=%d) = %.6g", K_v, value)
    return float(value)


def lambda_grid(panel: IPCAPanel, cfg: EstimationConfig, K: int | None = None) -> np.ndarray:
    """Ascending grid: ``cfg.lam_grid.values`` if given, else ``n_lambdas`` log-spaced
    points from ``ratio * lam_max`` to ``lam_max`` (D22)."""
    g = cfg.lam_grid
    if g.values is not None:
        return np.array(sorted(float(v) for v in g.values), dtype=float)
    lmax = lambda_max(panel, cfg, K)
    if not np.isfinite(lmax) or lmax <= 0.0:
        raise ValueError(f"lambda_max is {lmax}; cannot build a log-spaced grid (no signal in the narrative rows?)")
    return np.logspace(np.log10(g.ratio * lmax), np.log10(lmax), int(g.n_lambdas))


def lambda_path(
    panel: IPCAPanel,
    cfg: EstimationConfig,
    lams: Sequence[float] | None = None,
    K: int | None = None,
    progress: ProgressFn | None = None,
) -> tuple[list[LambdaPathPoint], list[SparseIPCAResult]]:
    """Trace the regularisation path in ascending ``lambda`` with warm starts (D22).

    Each fit starts from the previous (denser) ``Gamma`` (original units;
    :func:`fit_sparse_ipca` converts it to the standardised coordinates on
    entry); the first fit (or a fit following a ``lambda = 0`` point, which
    is plain IPCA and lives on a different scale) uses the SVD initialisation
    and warm-up. Returns the path points (``mve_sharpe`` annualised with 12
    periods per year; tuning recomputes its criterion from the fits) and the
    fits themselves. ``progress(i, n, point)`` is called after every fit.
    """
    K_v = int(cfg.K if K is None else K)
    lam_list = lambda_grid(panel, cfg, K_v) if lams is None else np.array(sorted(float(v) for v in lams), dtype=float)
    if lam_list.size == 0:
        raise ValueError("lambda grid is empty")
    if np.any(lam_list < 0.0):
        raise ValueError("lambdas must be >= 0")
    points: list[LambdaPathPoint] = []
    fits: list[SparseIPCAResult] = []
    warm: np.ndarray | None = None
    total = int(lam_list.size)
    for i, lam in enumerate(lam_list):
        res = fit_sparse_ipca(panel, cfg, lam=float(lam), K=K_v, Gamma_init=warm)
        warm = res.Gamma.copy() if lam > 0.0 else None
        point = LambdaPathPoint(
            lam=float(lam),
            K=K_v,
            total_r2=float(res.total_r2),
            pred_r2=float(res.pred_r2),
            mve_sharpe=float(res.mve_sharpe()),
            n_selected=int(res.n_selected),
            gamma_norms=res.gamma_norms.copy(),
            objective=float(res.objective),
            converged=bool(res.converged),
            n_iter=int(res.n_iter),
        )
        points.append(point)
        fits.append(res)
        if progress is not None:
            progress(i, total, point)
    return points, fits


# ---------------------------------------------------------------------------
# canonical form (D24)
# ---------------------------------------------------------------------------
def canonicalize(res: SparseIPCAResult) -> SparseIPCAResult:
    """Orthogonal rotation to the canonical labelling of the factors (D24).

    ``R = Q diag(s)`` with ``Q`` the eigenvectors of ``Sigma_ff`` in descending
    eigenvalue order and ``s_k = +-1`` chosen so that the rotated factor mean
    ``(Q' mu_f)_k`` is non-negative (fallback when that mean is zero: first
    non-zero entry of column ``k`` of ``Gamma Q`` positive, KKPS [3'']).
    Applies ``Gamma -> Gamma R``, ``f_t -> R' f_t``, ``mu_f -> R' mu_f``,
    ``Sigma_ff -> R' Sigma_ff R`` (diagonal). Row norms of ``Gamma``,
    ``||f_t||`` and fitted values are invariant, so ``objective``, the R2s,
    ``selected`` and the MVE Sharpe are unchanged and copied through.
    """
    Gamma = np.asarray(res.Gamma, dtype=float)
    F = np.asarray(res.F, dtype=float)
    K = int(Gamma.shape[1])
    Sigma = np.atleast_2d(np.asarray(res.Sigma_ff, dtype=float))
    if Sigma.shape != (K, K):
        raise ValueError(f"Sigma_ff must have shape ({K}, {K}), got {Sigma.shape}")
    sym = 0.5 * (Sigma + Sigma.T)
    evals, Q = np.linalg.eigh(sym)
    order = np.argsort(evals)[::-1]
    Q = Q[:, order]
    mu = np.asarray(res.mu_f, dtype=float).ravel()
    mu_q = mu @ Q
    Gq = Gamma @ Q
    signs = np.ones(K)
    fallback = _first_nonzero_signs(Gq)
    tiny = 1e-14 * max(1.0, float(np.linalg.norm(mu)))
    for k in range(K):
        if abs(mu_q[k]) > tiny:
            signs[k] = 1.0 if mu_q[k] > 0.0 else -1.0
        else:
            signs[k] = fallback[k]
    R = Q * signs
    Gamma_new = Gamma @ R
    Sigma_new = R.T @ sym @ R
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
    return replace(
        res,
        Gamma=Gamma_new,
        F=F @ R,
        mu_f=mu @ R,
        Sigma_ff=Sigma_new,
        gamma_norms=np.linalg.norm(Gamma_new, axis=1),
        meta={**res.meta, "canonical": True},
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _check_shapes(p: int, T: int, Gamma: np.ndarray, F: np.ndarray) -> None:
    if Gamma.ndim != 2 or Gamma.shape[0] != p:
        raise ValueError(f"Gamma must have shape ({p}, K), got {Gamma.shape}")
    if F.ndim != 2 or F.shape != (T, Gamma.shape[1]):
        raise ValueError(f"F must have shape ({T}, {Gamma.shape[1]}), got {F.shape}")
