"""Step 8: expanding-window out-of-sample factors and MVE portfolio (BKS Section 4.2; DESIGN.md D30-D33).

The in-sample fit of Eq. 8 chooses ``Gamma`` and the factors ``f_t`` jointly,
so the in-sample MVE Sharpe ratio is optimistic. BKS therefore re-estimate on
an expanding training window and, in every out-of-sample period, hold
``Gamma`` fixed and extract the factor from that period's cross-section alone:

    f^OOS_{t+1} = ( sum_i beta_{i,t} beta_{i,t}' + 2 I_K )^-1 sum_i beta_{i,t} r_{i,t+1},
    beta_{i,t} = c_{i,t} Gamma_hat,

with ``Gamma_hat`` estimated on a training sample that ends before period
``t + 1``, and the out-of-sample MVE return

    f^{MVE,OOS}_{t+1} = mu_f' Sigma_ff^-1 f^OOS_{t+1}

with ``mu_f``, ``Sigma_ff`` the moments of the *in-sample* factors of the
training window (D33). The ``2 I_K`` ridge is not a free parameter: it is the
``sum_t ||f_t||^2`` term of Eq. 8 that gives the closed-form f-step Eq. 16
(D32). BKS start the out-of-sample period in January 2000, retrain once each
December and apply the estimates to the following twelve months (D30); the
penalty is retuned at every refit (D31).

No look-ahead by construction: the panel pairs ``c_{i,t-1}`` with ``r_{i,t}``
(Eq. 7), the training panel of a refit at period index ``c`` holds the
periods ``< c`` only (and recomputes ``sigma^c_l`` on them, D26), and the
frozen ``(Gamma, mu_f, Sigma_ff, lambda)`` are applied to periods ``>= c``.

Conventions
-----------
* ``OOSResult.refit_periods[j]`` is the first period the ``j``-th refit is
  applied to (``panel.periods[c_j]``); the last training period is
  ``meta["train_end"][j]``.
* Out-of-sample periods without observations are skipped (no row in
  ``factors``); ``meta["n_skipped_empty"]`` counts them.
* When ``K`` is tuned per refit the factor frame has ``max K`` columns and
  rows of a smaller ``K`` are padded with ``NaN``; the MVE return is always
  defined.
* ``OOSConfig.enabled`` is a pipeline switch and is not consulted here.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .config import OOSConfig, PipelineConfig
from .evaluation import realized_sharpe
from .sparse_ipca import canonicalize, fit_sparse_ipca
from .types import IPCAPanel, OOSResult, SparseIPCAResult

logger = logging.getLogger(__name__)

__all__ = ["oos_factor", "run_oos", "oos_schedule", "sigma_ff_truncated", "RIDGE"]

ProgressFn = Callable[[int, int, str], None]
"""``progress(done, total, message)`` callback (D44)."""

RIDGE: float = 2.0
"""The ``2 I_K`` ridge of BKS Eq. 16 / Section 4.2 (D32)."""

#: Relative cut-off of the pseudo-inverse fallback (D43).
_RCOND = 1e-12


# ---------------------------------------------------------------------------
# one period
# ---------------------------------------------------------------------------
def oos_factor(C: np.ndarray, r: np.ndarray, Gamma: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """Out-of-sample factor of one period from its cross-section, BKS Section 4.2 (D32).

    ``f = (B'B + ridge I_K)^-1 B' r`` with ``B = C Gamma`` the model-implied
    loadings ``beta_i = c_i Gamma`` of the ``N`` assets observed in the
    period: a ridge cross-sectional regression of the period's returns on the
    frozen loadings. With ``ridge = 2`` this is exactly the in-sample f-step
    of Eq. 16 applied with a frozen ``Gamma`` (``S = C'C``, ``V = C'r``), which
    :func:`narrative_ipca.sparse_ipca.f_step` computes from the moments.

    Assumptions: ``C`` is ``(N, p)`` with the constant in column 0 (the rows
    of ``IPCAPanel.X`` of that period), ``r`` is ``(N,)``, ``Gamma`` is
    ``(p, K)``; all finite. ``ridge = 0`` gives the plain least-squares
    projection via a pseudo-inverse. An empty cross-section returns zeros.
    """
    C = np.atleast_2d(np.asarray(C, dtype=float))
    r = np.asarray(r, dtype=float).ravel()
    Gamma = np.asarray(Gamma, dtype=float)
    if Gamma.ndim != 2:
        raise ValueError(f"Gamma must be 2-D (p, K), got shape {Gamma.shape}")
    if C.ndim != 2 or C.shape[1] != Gamma.shape[0]:
        raise ValueError(f"C must be (N, {Gamma.shape[0]}) to match Gamma {Gamma.shape}, got {C.shape}")
    if r.shape[0] != C.shape[0]:
        raise ValueError(f"r must have {C.shape[0]} entries, got {r.shape[0]}")
    if ridge < 0.0:
        raise ValueError("ridge must be >= 0")
    K = int(Gamma.shape[1])
    if C.shape[0] == 0:
        return np.zeros(K)
    B = C @ Gamma
    lhs = B.T @ B + float(ridge) * np.eye(K)
    lhs = 0.5 * (lhs + lhs.T)
    rhs = B.T @ r
    f: np.ndarray | None = None
    if ridge > 0.0:
        try:
            f = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:  # pragma: no cover - PD by construction
            f = None
    if f is None or not np.all(np.isfinite(f)):
        f = np.linalg.pinv(lhs, rcond=_RCOND, hermitian=True) @ rhs
    return np.asarray(f, dtype=float).ravel()


def sigma_ff_truncated(Sigma_ff: np.ndarray, rcond: float) -> bool:
    """True when ``pinv(Sigma_ff, rcond)`` drops at least one factor direction.

    ``b_MVE = mu_f' Sigma_ff^-1`` (BKS Section 4.2, ``f^{MVE,OOS} = mu' Sigma^-1 f^OOS``)
    and the tuning criterion ``sqrt(mu' Sigma^-1 mu)`` (Section 2) are formed
    with a pseudo-inverse whose relative cut-off ``rcond`` discards the
    eigen-directions of ``Sigma_ff`` below ``rcond * lambda_max(Sigma_ff)``
    (D43). Near a rank-deficient ``Gamma`` (a dead factor direction at the
    sparse end of the path, or fewer than ``K - 1`` narratives selected) that
    cut makes both quantities discontinuous in the data: the dropped
    direction carries an MVE weight of order ``1 / lambda_min`` one side of
    the threshold and zero on the other. This flag reports when the cut was
    active so that the affected Sharpe ratios can be read with care; it does
    not change any number. A zero matrix counts as truncated.
    """
    S = np.atleast_2d(np.asarray(Sigma_ff, dtype=float))
    S = 0.5 * (S + S.T)
    if S.size == 0:
        return False
    ev = np.linalg.eigvalsh(S)
    top = float(np.max(ev))
    if not np.isfinite(top) or top <= 0.0:
        return True
    return bool(np.min(ev) <= float(rcond) * top)


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------
def oos_schedule(periods: pd.DatetimeIndex, cfg: OOSConfig) -> tuple[int, list[int], bool]:
    """First out-of-sample period index and the refit indices (D30).

    The first index is the first period with timestamp ``>= first_oos_period``
    when that ISO date is given, else ``T - round(oos_fraction * T)`` (at
    least one out-of-sample period). It is pushed forward to
    ``min_train_periods`` when fewer training periods would precede it (the
    third return value flags the move). Refits happen at ``first, first +
    refit_every, ...`` below ``T``. Raises ``ValueError`` when no period is
    left out of sample.
    """
    periods = pd.DatetimeIndex(periods)
    T = len(periods)
    if T == 0:
        raise ValueError("no periods")
    if cfg.first_oos_period is not None:
        ts = pd.Timestamp(cfg.first_oos_period)
        hits = np.flatnonzero(np.asarray(periods >= ts))
        if hits.size == 0:
            raise ValueError(f"first_oos_period {ts.date()} is after the last panel period {periods[-1].date()}")
        first = int(hits[0])
    else:
        n_oos = max(1, int(round(float(cfg.oos_fraction) * T)))
        first = T - n_oos
    moved = False
    if first < int(cfg.min_train_periods):
        logger.warning(
            "run_oos: first OOS period index %d leaves fewer than min_train_periods=%d training periods; "
            "moved to index %d", first, cfg.min_train_periods, cfg.min_train_periods,
        )
        first = int(cfg.min_train_periods)
        moved = True
    if first >= T:
        raise ValueError(
            f"no out-of-sample period: first OOS index {first} >= T={T} "
            f"(min_train_periods={cfg.min_train_periods}, first_oos_period={cfg.first_oos_period}, "
            f"oos_fraction={cfg.oos_fraction})"
        )
    refits = list(range(first, T, int(cfg.refit_every)))
    return first, refits, moved


# ---------------------------------------------------------------------------
# the expanding-window loop
# ---------------------------------------------------------------------------
def run_oos(panel: IPCAPanel, cfg: PipelineConfig, progress: ProgressFn | None = None) -> OOSResult:
    """Expanding-window out-of-sample factors and realised MVE Sharpe ratio (D30-D33).

    For every refit index ``c`` of :func:`oos_schedule`:

    1. training panel = ``panel.subset_periods(periods < c)`` (``sigma^c_l``
       recomputed on the training rows, D26/D31);
    2. model: :func:`narrative_ipca.tuning.tune` when ``cfg.oos.retune_lambda``
       is set, or at the first refit when ``cfg.estimation.lam`` is ``None``
       (there is no ``lambda`` yet); otherwise a fresh fit at the frozen
       ``(lambda, K)`` (:func:`narrative_ipca.sparse_ipca.fit_sparse_ipca`,
       cold start, canonicalised);
    3. the frozen ``Gamma`` and ``b_MVE = mu_f' Sigma_ff^-1`` of that fit are
       applied to the periods ``c, ..., c + refit_every - 1``:
       ``factors[t] = oos_factor(X_t, y_t, Gamma)`` from the rows of period
       ``t`` and ``mve[t] = b_MVE' factors[t]`` (D32, D33).

    ``sharpe`` is :func:`narrative_ipca.evaluation.realized_sharpe` of ``mve``
    with ``cfg.evaluation.annualization`` (D34). The histories are indexed by
    ``refit_periods`` (first period each refit is applied to):
    ``lam_history``, ``K_history``, ``n_selected_history``,
    ``is_sharpe_history`` (training MVE Sharpe), ``gamma_norm_history``
    (``||Gamma_l||`` per instrument, constant row included) and
    ``selected_history`` (boolean per narrative). ``fits`` keeps every
    training fit. ``meta`` records the schedule (``first_oos_index``,
    ``refit_indices``, ``train_end``, ``n_train_periods``), ``lam_max_history``,
    ``tuned_history`` and ``sigma_ff_truncated_history`` (whether the frozen
    ``b_MVE`` came from a pseudo-inverse that dropped a factor direction, see
    :func:`sigma_ff_truncated`) per refit, and ``refit_of_period`` (which refit
    served each out-of-sample period). ``progress(j, n_refits, message)`` is
    called before every refit.
    """
    from .tuning import tune  # lazy: tuning imports oos_factor from this module

    if panel.n_obs == 0 or panel.T == 0:
        raise ValueError("panel is empty")
    est_cfg, tune_cfg, eval_cfg, oos_cfg = cfg.estimation, cfg.tuning, cfg.evaluation, cfg.oos
    ann = float(eval_cfg.annualization)
    rcond = float(eval_cfg.rcond)
    periods = pd.DatetimeIndex(panel.periods)
    T = panel.T
    first, refits, moved = oos_schedule(periods, oos_cfg)
    step = int(oos_cfg.refit_every)
    n_refits = len(refits)
    slices = dict(panel.period_slices())
    all_t = np.arange(T)
    logger.info(
        "run_oos: %d periods, first OOS period %s (index %d, %d training periods), %d refit(s) every %d periods, retune_lambda=%s",
        T, periods[first].date(), first, first, n_refits, step, oos_cfg.retune_lambda,
    )

    lam_frozen: float | None = None if est_cfg.lam is None else float(est_cfg.lam)
    K_frozen: int = int(est_cfg.K)
    fits: list[SparseIPCAResult] = []
    refit_periods: list[pd.Timestamp] = []
    lam_hist: list[float] = []
    K_hist: list[int] = []
    nsel_hist: list[int] = []
    is_sharpe_hist: list[float] = []
    norm_rows: list[np.ndarray] = []
    sel_rows: list[np.ndarray] = []
    lam_max_hist: list[float] = []
    tuned_hist: list[bool] = []
    trunc_hist: list[bool] = []
    train_end: list[pd.Timestamp] = []
    n_train: list[int] = []
    oos_index: list[pd.Timestamp] = []
    oos_rows: list[np.ndarray] = []
    oos_mve: list[float] = []
    refit_of: list[int] = []
    n_skipped = 0

    for j, c in enumerate(refits):
        msg = f"refit {j + 1}/{n_refits}: training on {c} periods through {periods[c - 1].date()}"
        if progress is not None:
            progress(j, n_refits, msg)
        logger.info("run_oos: %s", msg)
        train = panel.subset_periods(all_t < c)
        if train.n_obs == 0:
            raise ValueError(f"training window [0, {c}) has no observations at refit {j + 1}; raise min_train_periods")
        use_tuner = bool(oos_cfg.retune_lambda) or lam_frozen is None
        if use_tuner:
            tr = tune(train, est_cfg, tune_cfg, eval_cfg)
            fit = tr.fit
            lam_frozen, K_frozen = float(tr.lam), int(tr.K)
            lam_max_hist.append(float(tr.lam_max))
        else:
            fit = canonicalize(fit_sparse_ipca(train, est_cfg, lam=lam_frozen, K=K_frozen))
            lam_max_hist.append(float("nan"))
        tuned_hist.append(use_tuner)
        Gamma = np.asarray(fit.Gamma, dtype=float)
        b = fit.b_mve(rcond=rcond)
        # Section 4.2: b_MVE = mu_f' Sigma_ff^-1 from the training fit. Flag a
        # pseudo-inverse truncation (near-singular Sigma_ff): the OOS MVE return
        # of this block is then discontinuous in the training data (see
        # sigma_ff_truncated); the numbers themselves are left as BKS define them.
        truncated = sigma_ff_truncated(fit.Sigma_ff, rcond)
        trunc_hist.append(truncated)
        if truncated:
            logger.warning(
                "run_oos: refit %d/%d (lam=%.4g, K=%d, %d selected): Sigma_ff is singular at rcond=%.1e, "
                "b_MVE drops at least one factor direction; the OOS MVE of this block is fragile",
                j + 1, n_refits, fit.lam, fit.K, fit.n_selected, rcond,
            )
        block = range(c, min(c + step, T))
        for t in block:
            sl = slices[t]
            if sl.stop <= sl.start:
                n_skipped += 1
                logger.debug("run_oos: period %s has no observations; skipped", periods[t].date())
                continue
            f = oos_factor(panel.X[sl], panel.y[sl], Gamma)
            oos_index.append(periods[t])
            oos_rows.append(f)
            oos_mve.append(float(b @ f))
            refit_of.append(j)
        fits.append(fit)
        refit_periods.append(pd.Timestamp(periods[c]))
        train_end.append(pd.Timestamp(periods[c - 1]))
        n_train.append(int(c))
        lam_hist.append(float(fit.lam))
        K_hist.append(int(fit.K))
        nsel_hist.append(int(fit.n_selected))
        is_sharpe_hist.append(float(fit.mve_sharpe(annualization=ann, rcond=rcond)))
        norm_rows.append(np.asarray(fit.gamma_norms, dtype=float).copy())
        sel_rows.append(np.asarray(fit.selected, dtype=bool).copy())
        logger.info(
            "run_oos: refit %d/%d lam=%.4g K=%d selected=%d/%d train Sharpe=%.3f applied to %d period(s)",
            j + 1, n_refits, fit.lam, fit.K, fit.n_selected, panel.L, is_sharpe_hist[-1], len(block),
        )

    idx = pd.DatetimeIndex(oos_index, name="period")
    K_max = max(K_hist) if K_hist else int(est_cfg.K)
    factors = pd.DataFrame(_pad_rows(oos_rows, K_max), index=idx, columns=[f"f{k + 1}" for k in range(K_max)])
    mve = pd.Series(np.asarray(oos_mve, dtype=float), index=idx, name="mve")
    sharpe = realized_sharpe(mve, ann)
    refit_idx = pd.DatetimeIndex(refit_periods, name="refit_period")
    logger.info("run_oos: %d out-of-sample periods (%d skipped as empty), realised MVE Sharpe %.3f", len(idx), n_skipped, sharpe)
    if progress is not None:
        progress(n_refits, n_refits, f"done: {len(idx)} out-of-sample periods, Sharpe {sharpe:.3f}")

    return OOSResult(
        factors=factors,
        mve=mve,
        sharpe=float(sharpe),
        refit_periods=refit_periods,
        lam_history=pd.Series(lam_hist, index=refit_idx, name="lam", dtype=float),
        K_history=pd.Series(K_hist, index=refit_idx, name="K", dtype=int),
        n_selected_history=pd.Series(nsel_hist, index=refit_idx, name="n_selected", dtype=int),
        gamma_norm_history=pd.DataFrame(
            np.vstack(norm_rows) if norm_rows else np.zeros((0, panel.p)), index=refit_idx, columns=list(panel.instrument_names)
        ),
        selected_history=pd.DataFrame(
            np.vstack(sel_rows) if sel_rows else np.zeros((0, panel.L), dtype=bool), index=refit_idx, columns=list(panel.topics)
        ),
        is_sharpe_history=pd.Series(is_sharpe_hist, index=refit_idx, name="is_sharpe", dtype=float),
        fits=fits,
        meta={
            "first_oos_index": int(first),
            "first_oos_period": pd.Timestamp(periods[first]),
            "first_oos_moved": bool(moved),
            "refit_every": step,
            "retune_lambda": bool(oos_cfg.retune_lambda),
            "min_train_periods": int(oos_cfg.min_train_periods),
            "n_refits": n_refits,
            "refit_indices": [int(c) for c in refits],
            "train_end": train_end,
            "n_train_periods": n_train,
            "lam_max_history": lam_max_hist,
            "tuned_history": tuned_hist,
            "sigma_ff_truncated_history": trunc_hist,
            "refit_of_period": pd.Series(refit_of, index=idx, name="refit", dtype=int),
            "n_oos_periods": int(len(idx)),
            "n_skipped_empty": int(n_skipped),
            "annualization": ann,
            "rcond": rcond,
            "K_max": int(K_max),
        },
    )


def _pad_rows(rows: list[np.ndarray], K_max: int) -> np.ndarray:
    """Stack factor vectors of possibly different lengths into ``(n, K_max)`` with ``NaN`` padding."""
    out = np.full((len(rows), int(K_max)), np.nan)
    for i, row in enumerate(rows):
        r = np.asarray(row, dtype=float).ravel()
        out[i, : r.size] = r
    return out
