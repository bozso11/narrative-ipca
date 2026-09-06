"""Step 6: ``lambda`` (and ``K``) tuning (BKS Section 2, footnote 8, App. C.3; DESIGN.md D27-D29).

BKS choose the group-lasso penalty ``lambda`` of Eq. 8 by an economic
criterion rather than by the estimation loss: for a training sample ``S`` and
every ``lambda`` on the regularisation path they compute the in-sample Sharpe
ratio of the model-implied mean-variance-efficient (MVE) portfolio,

    SR(lambda; S) = sqrt( mu_f' Sigma_ff^-1 mu_f ),

with ``mu_f`` and ``Sigma_ff`` the mean and covariance of the fitted factors
``f_t`` over ``S``, and set ``lambda* = argmax_lambda SR(lambda; S)`` (BKS
Section 2). The Sharpe ratio is annualised here with
``EvaluationConfig.annualization`` periods per year (D34), which does not
change the argmax. Footnote 8 notes that tuning on the training sample may
bias in-sample performance upward; Appendix C.3 offers leave-one-period-out
cross-validation (LOOCV) as the alternative, implemented in
:func:`loocv_sharpe` (D28).

``K`` is fixed at ``EstimationConfig.K`` by default (BKS use 3 "to be
conservative"); ``TuningConfig.K_grid`` switches on the joint ``(lambda, K)``
search of Appendix C.3, which uses the same criterion (D29).

Conventions
-----------
* The criterion is recomputed from the returned fits, so the annualisation
  and the pseudo-inverse cut-off come from the evaluation config, not from the
  path tracer; ``LambdaPathPoint.mve_sharpe`` is overwritten accordingly.
* Ties (criteria equal up to a relative tolerance of :data:`TIE_REL_TOL`) are
  broken toward the sparser solution by default (D27): larger ``lambda``,
  then fewer selected narratives, then smaller ``K``; ``tie_break="denser"``
  reverses every preference.
* A non-finite criterion (for instance a degenerate LOOCV series) never wins;
  when every criterion is non-finite the tie-break alone decides and a
  warning is logged.
* The chosen fit is returned in canonical form (D24); all reported
  statistics are invariant to that relabelling.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from .config import EstimationConfig, EvaluationConfig, TuningConfig
from .evaluation import realized_sharpe
from .oos import oos_factor, sigma_ff_truncated
from .sparse_ipca import canonicalize, fit_sparse_ipca, lambda_grid, lambda_max, lambda_path
from .types import IPCAPanel, LambdaPathPoint, SparseIPCAResult, TuningResult, DEFAULT_RCOND

logger = logging.getLogger(__name__)

__all__ = [
    "tune",
    "is_sharpe_criterion",
    "loocv_sharpe",
    "loocv_folds",
    "select_best_point",
    "path_point_from_fit",
    "TIE_REL_TOL",
]

ProgressFn = Callable[[int, int, str], None]
"""``progress(done, total, message)`` callback of the tuning loops (D44)."""

TIE_REL_TOL: float = 1e-9
"""Two criterion values within this relative distance of the maximum count as tied (D27)."""


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def is_sharpe_criterion(res: SparseIPCAResult, annualization: float, rcond: float = DEFAULT_RCOND) -> float:
    """In-sample annualised MVE Sharpe ratio of a fit, the BKS tuning criterion (D27).

    ``SR = sqrt(annualization * mu_f' Sigma_ff^-1 mu_f)`` with the factor
    moments of the fit (``ddof = 1`` over populated periods, D25) and a
    pseudo-inverse of ``Sigma_ff`` with cut-off ``rcond`` (D43; ``Sigma_ff``
    is singular by construction when fewer than ``K`` narratives are
    selected). Invariant to any invertible rotation of the factors, so it
    does not depend on the canonical form.
    """
    return float(res.mve_sharpe(annualization=float(annualization), rcond=float(rcond)))


def loocv_folds(panel: IPCAPanel, max_folds: int | None = None, seed: int = 0) -> np.ndarray:
    """Indices of the periods left out by :func:`loocv_sharpe` (ascending).

    Candidates are the populated periods of ``panel`` (leaving out an empty
    period would change nothing). With ``max_folds`` smaller than their
    number, ``max_folds`` of them are drawn without replacement from
    ``numpy.random.default_rng(seed)`` (D28: ``loocv_max_folds`` subsamples
    the left-out periods for speed).
    """
    pop = np.flatnonzero(np.asarray(panel.moments().n) > 0)
    if max_folds is not None and int(max_folds) < pop.size:
        rng = np.random.default_rng(int(seed))
        pop = np.sort(rng.choice(pop, size=int(max_folds), replace=False))
    return pop.astype(np.int64)


def loocv_sharpe(
    panel: IPCAPanel,
    est_cfg: EstimationConfig,
    lam: float,
    K: int,
    annualization: float,
    max_folds: int | None = None,
    seed: int = 0,
    Gamma_init: np.ndarray | None = None,
    rcond: float = DEFAULT_RCOND,
    progress: ProgressFn | None = None,
) -> float:
    """Leave-one-period-out cross-validated MVE Sharpe ratio, BKS App. C.3 (D28).

    For every left-out period ``t`` (all populated periods, or a seeded
    subsample of ``max_folds`` of them, :func:`loocv_folds`):

    1. fit Eq. 8 at ``(lam, K)`` on ``S \\ {t}`` (``panel.subset_periods``,
       which recomputes ``sigma^c_l`` on the training sample, D26),
       warm-started from ``Gamma_init`` (the full-sample fit at the same
       ``(lam, K)``, fitted here when not supplied);
    2. form the left-out factor with the out-of-sample formula of BKS
       Section 4.2, ``f_t = (B'B + 2 I_K)^-1 B' r_t`` with ``B = C_{t-1} Gamma``
       from the left-out cross-section (:func:`narrative_ipca.oos.oos_factor`);
    3. ``f^MVE_t = b_MVE' f_t`` with ``b_MVE = mu_f' Sigma_ff^-1`` from the
       training fit's factor moments (D33).

    The stitched series ``{f^MVE_t}`` is scored with
    :func:`narrative_ipca.evaluation.realized_sharpe` (annualised, D34).
    Returns ``nan`` with fewer than two usable folds. Costs one fit per fold.

    ``lam = 0`` is plain IPCA (D18); the warm start is then ignored.
    """
    lam = float(lam)
    K = int(K)
    folds = loocv_folds(panel, max_folds, seed)
    if folds.size < 2:
        logger.warning("loocv_sharpe: only %d populated period(s); Sharpe undefined", folds.size)
        return float("nan")
    if Gamma_init is None and lam > 0.0:
        Gamma_init = fit_sparse_ipca(panel, est_cfg, lam=lam, K=K).Gamma
    slices = dict(panel.period_slices())
    T = panel.T
    values: list[float] = []
    n_unconverged = 0
    n_truncated = 0
    for j, t in enumerate(folds):
        keep = np.ones(T, dtype=bool)
        keep[t] = False
        train = panel.subset_periods(keep)
        if train.n_obs == 0:
            logger.warning("loocv_sharpe: training sample empty when leaving out period %d; fold skipped", t)
            continue
        fit = fit_sparse_ipca(train, est_cfg, lam=lam, K=K, Gamma_init=Gamma_init if lam > 0.0 else None)
        n_unconverged += int(not fit.converged)
        # App. C.3: b_MVE of the fold fit; a pseudo-inverse truncation of its
        # Sigma_ff makes this fold's f^MVE_t fragile (see oos.sigma_ff_truncated).
        n_truncated += int(sigma_ff_truncated(fit.Sigma_ff, rcond))
        sl = slices[int(t)]
        f_t = oos_factor(panel.X[sl], panel.y[sl], fit.Gamma)
        values.append(float(fit.b_mve(rcond=rcond) @ f_t))
        if progress is not None:
            progress(j + 1, int(folds.size), f"loocv lam={lam:.4g} K={K}: fold {j + 1}/{folds.size}")
    sharpe = realized_sharpe(np.asarray(values, dtype=float), annualization)
    logger.info(
        "loocv_sharpe(lam=%.4g, K=%d): %d folds, %d unconverged fits, Sharpe=%.4f",
        lam, K, len(values), n_unconverged, sharpe,
    )
    if n_truncated:
        logger.warning(
            "loocv_sharpe(lam=%.4g, K=%d): %d of %d fold fits have a singular Sigma_ff at rcond=%.1e; "
            "their b_MVE drop a factor direction and the stitched Sharpe is fragile",
            lam, K, n_truncated, len(values), rcond,
        )
    return sharpe


# ---------------------------------------------------------------------------
# path bookkeeping
# ---------------------------------------------------------------------------
def path_point_from_fit(fit: SparseIPCAResult, annualization: float, rcond: float, criterion: float | None = None) -> LambdaPathPoint:
    """Summarise a fit as a :class:`LambdaPathPoint` (``mve_sharpe`` at the given annualisation)."""
    return LambdaPathPoint(
        lam=float(fit.lam),
        K=int(fit.K),
        total_r2=float(fit.total_r2),
        pred_r2=float(fit.pred_r2),
        mve_sharpe=is_sharpe_criterion(fit, annualization, rcond),
        n_selected=int(fit.n_selected),
        gamma_norms=np.asarray(fit.gamma_norms, dtype=float).copy(),
        objective=float(fit.objective),
        converged=bool(fit.converged),
        n_iter=int(fit.n_iter),
        criterion=None if criterion is None else float(criterion),
    )


def select_best_point(points: list[LambdaPathPoint], tie_break: str = "sparser", rel_tol: float = TIE_REL_TOL) -> int:
    """Index of the path point with the largest criterion, ties broken per D27.

    Points whose criterion is within ``rel_tol * max(1, |best|)`` of the
    maximum are tied. ``"sparser"`` prefers, in order, larger ``lam``, fewer
    selected narratives, smaller ``K``; ``"denser"`` prefers the opposite.
    Non-finite criteria never win unless every criterion is non-finite, in
    which case the tie-break alone decides (with a warning).
    """
    if not points:
        raise ValueError("no path points to choose from")
    if tie_break not in ("sparser", "denser"):
        raise ValueError(f"tie_break must be 'sparser' or 'denser', got {tie_break!r}")
    crit = np.array([np.nan if p.criterion is None else float(p.criterion) for p in points], dtype=float)
    finite = np.isfinite(crit)
    if finite.any():
        best = float(np.max(crit[finite]))
        tol = float(rel_tol) * max(1.0, abs(best))
        candidates = np.flatnonzero(finite & (crit >= best - tol))
    else:
        logger.warning("select_best_point: every criterion is non-finite; the tie-break rule decides alone")
        candidates = np.arange(len(points))
    sign = 1.0 if tie_break == "sparser" else -1.0

    def key(i: int) -> tuple[float, float, float]:
        p = points[i]
        return (sign * float(p.lam), -sign * float(p.n_selected), -sign * float(p.K))

    return int(max((int(i) for i in candidates), key=key))


# ---------------------------------------------------------------------------
# the tuner
# ---------------------------------------------------------------------------
def tune(
    panel: IPCAPanel,
    est_cfg: EstimationConfig,
    tune_cfg: TuningConfig,
    eval_cfg: EvaluationConfig,
    progress: ProgressFn | None = None,
) -> TuningResult:
    """Choose ``lambda`` (and ``K``) by the configured criterion (D27-D29).

    Procedure, for every ``K`` in ``tune_cfg.K_grid`` (default ``(est_cfg.K,)``):

    * ``est_cfg.lam`` fixed: a single fit at that ``lambda`` (a path of one
      point); ``lam_max`` is still computed for reporting;
    * otherwise the regularisation path of :func:`sparse_ipca.lambda_path`
      over :func:`sparse_ipca.lambda_grid` (``n_lambdas`` log-spaced points in
      ``[ratio lam_max, lam_max]``, ascending with warm starts, D22), with the
      criterion of every point recomputed from its fit:
      ``"is_sharpe"`` -> :func:`is_sharpe_criterion` (BKS Section 2),
      ``"loocv_sharpe"`` -> :func:`loocv_sharpe` warm-started from that fit
      (App. C.3, ``loocv_max_folds`` folds, seeded with ``est_cfg.seed``).

    The point with the largest criterion wins (:func:`select_best_point`,
    ties toward the sparser solution unless ``tie_break="denser"``; with
    ``tune_cfg.tolerance > 0`` every point within that relative distance of
    the maximum counts as tied, so the sparsest such point wins); its fit
    is returned canonicalised (D24). ``path`` lists every point of every
    ``K`` (with ``criterion`` filled and ``mve_sharpe`` at
    ``eval_cfg.annualization``); ``lam_max`` is that of the chosen ``K``;
    ``meta`` records the grids, ``lam_max`` per ``K``, the chosen index and
    the criterion settings. ``progress(done, total, message)`` is called after
    every fitted point.

    Raises ``ValueError`` on an empty panel or an ill-formed ``K`` grid; a
    log-spaced grid that cannot be formed (``lam_max`` not positive) also
    raises, from :func:`sparse_ipca.lambda_grid`.
    """
    if panel.n_obs == 0:
        raise ValueError("panel is empty")
    Ks = tuple(int(k) for k in (tune_cfg.K_grid if tune_cfg.K_grid is not None else (est_cfg.K,)))
    if not Ks or any(k < 1 for k in Ks):
        raise ValueError(f"K candidates must be >= 1, got {Ks}")
    ann = float(eval_cfg.annualization)
    rcond = float(eval_cfg.rcond)
    criterion = str(tune_cfg.criterion)
    if criterion not in ("is_sharpe", "loocv_sharpe"):
        raise ValueError(f"unknown tuning criterion {criterion!r}")
    fixed = est_cfg.lam is not None

    grids: dict[int, list[float]] = {}
    lam_max_by_K: dict[int, float] = {}
    for K in Ks:
        lam_max_by_K[K] = float(lambda_max(panel, est_cfg, K))
        grids[K] = [float(est_cfg.lam)] if fixed else [float(v) for v in lambda_grid(panel, est_cfg, K)]
    total = int(sum(len(g) for g in grids.values()))
    logger.info(
        "tune: criterion=%s, K candidates=%s, %d path point(s)%s, tie_break=%s",
        criterion, Ks, total, " (fixed lambda)" if fixed else "", tune_cfg.tie_break,
    )

    def score(fit: SparseIPCAResult) -> float:
        if criterion == "is_sharpe":
            return is_sharpe_criterion(fit, ann, rcond)
        return loocv_sharpe(
            panel, est_cfg, fit.lam, fit.K, ann,
            max_folds=tune_cfg.loocv_max_folds, seed=int(est_cfg.seed),
            Gamma_init=fit.Gamma if fit.lam > 0.0 else None, rcond=rcond,
        )

    points: list[LambdaPathPoint] = []
    fits: list[SparseIPCAResult] = []
    done = 0
    for K in Ks:
        if fixed:
            k_fits = [fit_sparse_ipca(panel, est_cfg, lam=grids[K][0], K=K)]
        else:
            _, k_fits = lambda_path(panel, est_cfg, lams=grids[K], K=K)
        for fit in k_fits:
            value = score(fit)
            point = path_point_from_fit(fit, ann, rcond, criterion=value)
            points.append(point)
            fits.append(fit)
            done += 1
            logger.debug("tune: K=%d lam=%.4g selected=%d criterion=%.4f", K, fit.lam, fit.n_selected, value)
            if progress is not None:
                progress(done, total, f"K={K} lam={fit.lam:.4g} selected={fit.n_selected} {criterion}={value:.4f}")

    # TuningConfig.tolerance widens the tie band (0 = BKS exact argmax); the
    # numerical TIE_REL_TOL is the floor so that exact float ties still count.
    rel_tol = max(TIE_REL_TOL, float(getattr(tune_cfg, "tolerance", 0.0)))
    best = select_best_point(points, tune_cfg.tie_break, rel_tol=rel_tol)
    chosen = canonicalize(fits[best])
    n_tied = int(np.sum(_tied_mask(points, best)))
    # Section 2: SR(lambda; S) = sqrt(mu' Sigma^-1 mu). With the pseudo-inverse of
    # D43 the criterion silently drops factor directions whose variance is below
    # rcond * lambda_max(Sigma_ff), which happens where a factor dies along the
    # path (rank-deficient Gamma). Record where that cut was active; warn when
    # it shaped the winning point (see oos.sigma_ff_truncated).
    truncated = [sigma_ff_truncated(f.Sigma_ff, rcond) for f in fits]
    if truncated[best]:
        logger.warning(
            "tune: the chosen point (lam=%.4g, K=%d, %d selected) has a singular Sigma_ff at rcond=%.1e; "
            "its %s criterion and b_MVE drop at least one factor direction",
            chosen.lam, chosen.K, chosen.n_selected, rcond, criterion,
        )
    logger.info(
        "tune: chosen lam=%.4g K=%d (%s=%.4f, %d selected, %d tied point(s), lam_max=%.4g)",
        chosen.lam, chosen.K, criterion, float(points[best].criterion), chosen.n_selected, n_tied, lam_max_by_K[chosen.K],
    )
    meta: dict[str, Any] = {
        "criterion": criterion,
        "annualization": ann,
        "rcond": rcond,
        "tie_break": str(tune_cfg.tie_break),
        "tie_rel_tol": rel_tol,
        "fixed_lam": bool(fixed),
        "K_grid": list(Ks),
        "lam_grid": {int(K): list(g) for K, g in grids.items()},
        "lam_max": {int(K): v for K, v in lam_max_by_K.items()},
        "chosen_index": int(best),
        "best_criterion": float(points[best].criterion),
        "n_tied": n_tied,
        "n_points": total,
        "sigma_ff_truncated": truncated,
        "loocv_max_folds": tune_cfg.loocv_max_folds,
        "loocv_seed": int(est_cfg.seed),
    }
    return TuningResult(
        lam=float(chosen.lam),
        K=int(chosen.K),
        criterion=criterion,
        path=points,
        fit=chosen,
        lam_max=float(lam_max_by_K[chosen.K]),
        meta=meta,
    )


def _tied_mask(points: list[LambdaPathPoint], best: int, rel_tol: float = TIE_REL_TOL) -> np.ndarray:
    """Boolean mask of the points tied with ``points[best]`` on the criterion (for reporting)."""
    crit = np.array([np.nan if p.criterion is None else float(p.criterion) for p in points], dtype=float)
    b = crit[best]
    if not np.isfinite(b):
        return ~np.isfinite(crit)
    return np.isfinite(crit) & (crit >= b - rel_tol * max(1.0, abs(b)))
