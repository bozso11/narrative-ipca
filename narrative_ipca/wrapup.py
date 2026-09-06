"""Step 7 -- wrap-up: recover ``A``, the latent states and the impact vectors.

Implements the wrap-up step of Bybee, Kelly & Su (2023), "Narrative Asset
Pricing" (BKS): Section 2 step 3, Appendix B.1, the interpretation method of
Section 6.1 (Eq. 10-12) and the retrieval scores of Section 6.3.

Sparse IPCA (Eq. 8) delivers ``Gamma = [Gamma_0; Gamma_tilde]`` and the
factors ``f_t``. This module undoes the change of variables of Eq. 5 to return
to the structural objects of Eq. 1:

* ``A`` (L x K), the narrative-to-state loadings, from
  ``A = Gamma_tilde (Gamma_tilde' Gamma_tilde)^-1 Sigma_ff^-1``;
* ``x_tau`` (K,), the daily latent states, from ``x_tau = (A'A)^-1 A' z_tau``;
* the impact vectors ``I_{z->x} = A (A'A)^-1`` (Eq. 10),
  ``I_{z->MVE} = I_{z->x} b_MVE`` (Eq. 11) and, when a topic-term matrix is
  available, the term-level vector ``I_{w->MVE}`` (Eq. 12);
* projections of observable series (e.g. the market factor) on the factors,
  whose weights ``b_obs`` replace ``b_MVE`` in Eq. 11-12 (BKS footnote 17);
* retrieval scores ``I' z(m)`` of a day or an article (Section 6.3).

Numerical policy (DESIGN.md D43)
--------------------------------
Every inverse is a pseudo-inverse with the ``rcond`` cut-off of
``EvaluationConfig.rcond``. Matrices of the form ``M (M'M)^-1`` are computed
from one thin SVD of ``M`` (see :func:`_lsq_map`) with the *same* cut-off
semantics as ``numpy.linalg.pinv(M'M, rcond)`` -- singular values ``s_i`` of
``M`` with ``s_i^2 <= rcond * s_max^2`` are dropped -- but without squaring
the condition number by forming the Gram matrix. The group lasso routinely
zeroes rows of ``Gamma_tilde``; with fewer than ``K`` non-zero rows the map
``z -> x`` is rank deficient, ``A`` is a minimum-norm solution rather than the
structural loading matrix, and ``Sigma_ff`` is singular. Nothing raises in
that case: :func:`recover_A` returns a ``rank_deficient`` flag that
:func:`wrap_up` carries into ``WrapUpResult.rank_deficient``.

All functions are pure; no input is mutated.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .config import EvaluationConfig
from .types import ShockPanel, SparseIPCAResult, WrapUpResult

__all__ = [
    "recover_A",
    "state_variables",
    "impact_vectors",
    "project_observable",
    "term_impact",
    "wrap_up",
    "retrieval_scores",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private numerics
# ---------------------------------------------------------------------------
def _lsq_map(M: np.ndarray, rcond: float) -> tuple[np.ndarray, int]:
    """``M (M'M)^+`` and the numerical rank of ``M`` from one thin SVD.

    With ``M = U S V'`` (thin SVD, singular values ``s_1 >= s_2 >= ...``),
    ``M'M = V S^2 V'`` and ``(M'M)^+ = V_k S_k^-2 V_k'`` over the retained
    components ``k``. A component is retained when ``s_i^2 > rcond * s_1^2``,
    which is exactly the cut-off that ``numpy.linalg.pinv(M'M, rcond=rcond)``
    applies to the singular values of ``M'M``; the SVD of ``M`` itself is used
    so that the cut-off is decided without the rounding error of forming the
    Gram matrix. The product is returned as ``M @ (V_k S_k^-2 V_k')`` (rather
    than the algebraically equal ``U_k S_k^-1 V_k'``) so that a zero row of
    ``M`` -- a narrative the group lasso dropped -- gives an exactly zero row.

    ``M (M'M)^+`` is the transpose of the Moore-Penrose inverse of ``M``; for
    full column rank it equals ``M (M'M)^-1``, the matrix whose transpose maps
    an observation ``y`` to its least-squares coefficient ``(M'M)^-1 M' y``.

    Assumptions: ``M`` is finite and 2-D. An all-zero ``M`` returns zeros and
    rank 0.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {M.shape}")
    if M.size == 0 or not np.any(M):
        return np.zeros_like(M), 0
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s**2 > rcond * s[0] ** 2
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        return np.zeros_like(M), 0
    Vk = Vt[keep].T  # (n, rank)
    gram_pinv = (Vk / s[keep] ** 2) @ Vk.T  # (M'M)^+ = V_k S_k^-2 V_k'
    return M @ gram_pinv, rank


def _sym_pinv(S: np.ndarray, rcond: float) -> tuple[np.ndarray, int]:
    """Pseudo-inverse and numerical rank of a symmetric matrix (``Sigma_ff``).

    The input is symmetrised as ``(S + S') / 2`` first (a sample covariance is
    symmetric up to rounding). Eigenvalues ``w_i`` with
    ``|w_i| <= rcond * max|w|`` are treated as zero, matching
    :func:`narrative_ipca.types.mve_weights`, which uses ``pinv`` with the same
    ``rcond`` for ``b_MVE``.
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    wmax = float(np.max(np.abs(w))) if w.size else 0.0
    if wmax == 0.0:
        return np.zeros_like(S), 0
    keep = np.abs(w) > rcond * wmax
    rank = int(np.count_nonzero(keep))
    Sinv = (V[:, keep] / w[keep]) @ V[:, keep].T
    return Sinv, rank


def _check_finite(x: np.ndarray, name: str) -> None:
    """Raise ``ValueError`` if ``x`` holds NaN or infinity (input validation, not a solver failure)."""
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite")


def _align_columns(frame: pd.DataFrame, names: list[str], what: str) -> pd.DataFrame:
    """Reorder ``frame``'s columns to ``names`` (matched as strings); raise if any is missing."""
    lookup: dict[str, Any] = {}
    for c in frame.columns:
        lookup.setdefault(str(c), c)
    missing = [n for n in names if n not in lookup]
    if missing:
        raise ValueError(f"{what} is missing {len(missing)} column(s) required by the fit, e.g. {missing[:5]}")
    out = frame[[lookup[n] for n in names]].copy()
    out.columns = list(names)
    return out


def _align_observable(
    F: pd.DataFrame, target: pd.Series, period: str | None
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Rows of ``F`` and ``target`` on their common index with finite values.

    Returns ``(F_arr (n, K), y (n,), index)``. When ``period`` is given (a
    pandas offset alias such as ``"M"``) and both indexes are datetime-like,
    the match is made on the period label instead of the exact timestamp, so
    that a month-end-stamped observable pairs with a last-trading-day-stamped
    factor. Duplicate labels after conversion are an error.
    """
    if not isinstance(F, pd.DataFrame):
        raise TypeError("F must be a DataFrame (T, K) of factors")
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a Series")
    f_idx: pd.Index = F.index
    t_idx: pd.Index = target.index
    if period is not None and isinstance(f_idx, pd.DatetimeIndex) and isinstance(t_idx, pd.DatetimeIndex):
        f_idx = f_idx.to_period(period)
        t_idx = t_idx.to_period(period)
        if f_idx.has_duplicates or t_idx.has_duplicates:
            raise ValueError(f"index has duplicate {period!r} periods; cannot align F and target on periods")
    F_lab = F.set_axis(f_idx, axis=0)
    y_lab = target.set_axis(t_idx, axis=0)
    common = F_lab.index.intersection(y_lab.index)
    F_arr = F_lab.loc[common].to_numpy(dtype=float)
    y = y_lab.loc[common].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(F_arr), axis=1) if len(common) else np.zeros(0, dtype=bool)
    return F_arr[mask], y[mask], common[mask]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def recover_A(Gamma_tilde: np.ndarray, Sigma_ff: np.ndarray, rcond: float) -> tuple[np.ndarray, bool]:
    """Recover the narrative loading matrix ``A`` (BKS Section 2 step 3, App. B.1).

    Implements ``A = Gamma_tilde (Gamma_tilde' Gamma_tilde)^-1 Sigma_ff^-1``,
    the inverse of Eq. 5, ``Gamma_tilde = A (A'A)^-1 Sigma_ff^-1``. In words:
    ``A`` maps the K states to the L narratives (``z = A x + eta``, Eq. 1);
    ``Gamma_tilde`` maps the L narrative covariances to the K betas; the two
    are related through the factor covariance ``Sigma_ff`` because
    ``cov_{i,t} = beta_{i,t} Sigma_ff A'``.

    Both inverses are pseudo-inverses with cut-off ``rcond`` (D43):
    ``Gamma_tilde (Gamma_tilde' Gamma_tilde)^+`` via :func:`_lsq_map` and
    ``Sigma_ff^+`` via :func:`_sym_pinv`. Zero rows of ``Gamma_tilde`` (the
    narratives the group lasso dropped) give zero rows of ``A``.

    Parameters
    ----------
    Gamma_tilde:
        ``(L, K)`` narrative block of ``Gamma`` (rows ``1..L``).
    Sigma_ff:
        ``(K, K)`` sample covariance of the factors.
    rcond:
        Pseudo-inverse cut-off, ``EvaluationConfig.rcond``.

    Returns
    -------
    A:
        ``(L, K)`` loading matrix.
    rank_deficient:
        ``True`` when the map is not identified: fewer than ``K`` non-zero
        rows of ``Gamma_tilde``, numerical rank of ``Gamma_tilde`` below
        ``K``, or ``Sigma_ff`` singular (rank below ``K`` at ``rcond``). ``A``
        is then the minimum-norm solution (same row support, rank and column
        space as ``Gamma_tilde``); the states are identified only on the
        column space of ``A`` and the round trip
        ``A (A'A)^+ Sigma_ff^+ = Gamma_tilde`` no longer holds exactly
        (the pseudo-inverse does not distribute over the product).

    Assumptions: inputs are finite; ``Sigma_ff`` is symmetric positive
    semi-definite (it is symmetrised before inversion).
    """
    G = np.asarray(Gamma_tilde, dtype=float)
    if G.ndim != 2:
        raise ValueError(f"Gamma_tilde must be 2-D (L, K), got shape {G.shape}")
    L, K = G.shape
    S = np.atleast_2d(np.asarray(Sigma_ff, dtype=float))
    if S.shape != (K, K):
        raise ValueError(f"Sigma_ff must have shape ({K}, {K}) to match Gamma_tilde {G.shape}, got {S.shape}")
    _check_finite(G, "Gamma_tilde")
    _check_finite(S, "Sigma_ff")

    n_nonzero = int(np.count_nonzero(np.any(G != 0.0, axis=1)))
    G_map, rank_G = _lsq_map(G, rcond)
    S_inv, rank_S = _sym_pinv(S, rcond)
    A = G_map @ S_inv
    rank_deficient = bool(n_nonzero < K or rank_G < K or rank_S < K)
    if rank_deficient:
        logger.warning(
            "recover_A: rank deficient (non-zero rows %d, rank(Gamma_tilde) %d, rank(Sigma_ff) %d, K %d); "
            "A is a minimum-norm solution",
            n_nonzero,
            rank_G,
            rank_S,
            K,
        )
    else:
        logger.debug("recover_A: L=%d K=%d non-zero rows=%d", L, K, n_nonzero)
    return A, rank_deficient


def state_variables(A: np.ndarray, shocks: ShockPanel, rcond: float) -> pd.DataFrame:
    """Latent states ``x_tau = (A'A)^-1 A' z_tau`` for every day (BKS step 3, Eq. 10).

    In words: the state on day ``tau`` is the least-squares projection of the
    day's narrative shock vector on the columns of ``A`` (Eq. 1,
    ``z_tau = A x_tau + eta_tau``). Row-wise for the whole sample,
    ``X = Z A (A'A)^-1 = Z I_{z->x}`` using the symmetry of ``(A'A)^-1``.

    Days on which any shock is non-finite (the first ``window`` days, or gaps)
    get a ``NaN`` state row; all other days are computed. ``(A'A)^+`` uses
    cut-off ``rcond``; when ``A`` has rank below ``K`` the states are the
    minimum-norm solution (see :func:`recover_A`).

    Parameters
    ----------
    A:
        ``(L, K)`` loadings from :func:`recover_A`.
    shocks:
        Shock panel whose ``z`` columns are in the same order as the rows of
        ``A`` (positional match; :func:`wrap_up` aligns them by name).
    rcond:
        Pseudo-inverse cut-off.

    Returns
    -------
    ``(n_days, K)`` DataFrame on ``shocks.z.index`` with columns ``x1..xK``.
    """
    A_arr = np.asarray(A, dtype=float)
    if A_arr.ndim != 2:
        raise ValueError(f"A must be 2-D (L, K), got shape {A_arr.shape}")
    _check_finite(A_arr, "A")
    L, K = A_arr.shape
    z = shocks.z
    if not isinstance(z, pd.DataFrame):
        raise TypeError("shocks.z must be a DataFrame (n_days, L)")
    if z.shape[1] != L:
        raise ValueError(f"shocks.z has {z.shape[1]} columns but A has {L} rows")
    Z = z.to_numpy(dtype=float)
    I_zx, _ = _lsq_map(A_arr, rcond)
    X = np.full((Z.shape[0], K), np.nan)
    finite = np.all(np.isfinite(Z), axis=1)
    if finite.any():
        X[finite] = Z[finite] @ I_zx
    logger.debug("state_variables: %d of %d days finite", int(finite.sum()), Z.shape[0])
    return pd.DataFrame(X, index=z.index, columns=[f"x{k + 1}" for k in range(K)])


def impact_vectors(A: np.ndarray, b_mve: np.ndarray, rcond: float) -> tuple[np.ndarray, np.ndarray]:
    """Impact vectors of BKS Eq. 10-11.

    Eq. 10: ``I_{z->x} = A (A'A)^-1`` (``L x K``) -- column ``k`` gives the
    response of state ``k`` to a unit shock in each narrative, since
    ``x(s) = I_{z->x}' z(s)``.
    Eq. 11: ``I_{z->MVE} = I_{z->x} b_MVE`` (``L``,) with
    ``b_MVE = mu_f' Sigma_ff^-1``, so that ``x_MVE(s) = I_{z->MVE}' z(s)``.
    Passing the projection weights ``b_obs`` of an observable series instead of
    ``b_MVE`` gives that series' impact vector (BKS ``I_{z->Mkt}``,
    footnote 17).

    ``(A'A)^+`` uses cut-off ``rcond``. Assumptions: ``A`` and ``b`` finite.
    """
    A_arr = np.asarray(A, dtype=float)
    if A_arr.ndim != 2:
        raise ValueError(f"A must be 2-D (L, K), got shape {A_arr.shape}")
    b = np.asarray(b_mve, dtype=float).ravel()
    K = A_arr.shape[1]
    if b.shape != (K,):
        raise ValueError(f"b_mve must have shape ({K},) to match A {A_arr.shape}, got {b.shape}")
    _check_finite(A_arr, "A")
    _check_finite(b, "b_mve")
    I_zx, _ = _lsq_map(A_arr, rcond)
    return I_zx, I_zx @ b


def project_observable(
    F: pd.DataFrame, target: pd.Series, period: str | None = None
) -> tuple[np.ndarray, float]:
    """OLS projection of an observable period series on the factors (BKS Section 6.1).

    Fits ``target_t = a + b_obs' f_t + e_t`` by least squares on the common
    index of ``F`` and ``target`` (rows with non-finite values dropped) and
    returns the slope vector ``b_obs`` (``K``,) and the centred
    ``R2 = 1 - SSR / sum_t (target_t - mean)^2``. BKS project the market
    factor this way (``R2 = 97.6%`` in their sample) and use ``b_Mkt`` in
    place of ``b_MVE`` in Eq. 11-12.

    Parameters
    ----------
    F:
        ``(T, K)`` factors indexed by period; pass only populated periods.
    target:
        The observable period series (e.g. the market excess return).
    period:
        Optional pandas offset alias. When given and both indexes are
        ``DatetimeIndex``, rows are matched on the period label (so a
        month-end-stamped observable pairs with last-trading-day-stamped
        factors); otherwise the index must match exactly.

    Raises
    ------
    ValueError
        When fewer than ``K + 1`` finite common observations exist (an input
        alignment error, not a solver failure).

    Assumption: the factors are period returns on the same period grid as
    ``target`` (D4-D6); no lag is applied.
    """
    X, y, idx = _align_observable(F, target, period)
    K = F.shape[1]
    n = int(X.shape[0])
    if n < K + 1:
        raise ValueError(
            f"target overlaps the factor sample in only {n} finite period(s); need at least K + 1 = {K + 1} "
            f"(F index {F.index[:1].tolist()}.., target index {target.index[:1].tolist()}..)"
        )
    if n < len(F) // 2:
        logger.warning("project_observable: only %d of %d factor periods overlap the target", n, len(F))
    design = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    ss_res = float(resid @ resid)
    yc = y - y.mean()
    ss_tot = float(yc @ yc)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    logger.debug("project_observable: n=%d R2=%.4f", n, r2)
    return np.asarray(coef[1:], dtype=float), float(r2)


def term_impact(phi: pd.DataFrame, impact_z: np.ndarray, rcond: float) -> pd.Series:
    """Term-level impact vector ``I_{w->MVE}`` (BKS Eq. 12).

    BKS write the topic-term matrix ``Phi`` as ``V x L`` (column ``l`` is the
    term distribution ``phi_l`` of narrative ``l``) and map a term-frequency
    change ``dw`` (``V``,) to narrative shocks by least squares,
    ``z(s) = (Phi'Phi)^-1 Phi' dw(s)``; substituting into Eq. 11 gives
    ``x_MVE(s) = I_{z->MVE}' (Phi'Phi)^-1 Phi' dw(s)``, i.e.
    ``I_{w->MVE} = Phi (Phi'Phi)^-1 I_{z->MVE}`` (``V``,).

    Our ``phi`` is stored ``L x V`` with rows summing to one (FASTopic's
    topic-word distribution), i.e. ``P = Phi'``. Rewriting the same
    least-squares map in this orientation, ``z = (P P')^-1 P dw``, so
    ``x_MVE = I_z' (P P')^-1 P dw`` and therefore
    ``I_w = P' (P P')^-1 I_z``. The function computes exactly this with
    ``M = P'`` (``V x L``): ``I_w = M (M'M)^+ I_z`` via :func:`_lsq_map`
    (pseudo-inverse of ``P P'`` with cut-off ``rcond``, which matters because
    topics share vocabulary and ``P P'`` can be ill conditioned).

    The identity ``I_w' dw = I_z' z(dw)`` holds for every ``dw`` by
    construction (tested).

    Parameters
    ----------
    phi:
        ``(L, V)`` DataFrame, rows in the order of ``impact_z`` (positional
        match; :func:`wrap_up` aligns by topic name), columns = vocabulary.
    impact_z:
        ``(L,)`` narrative-level impact vector (``I_{z->MVE}`` or ``I_{z->obs}``).
    rcond:
        Pseudo-inverse cut-off.

    Returns
    -------
    ``(V,)`` Series indexed by the vocabulary, named ``"impact_w"``.
    """
    if not isinstance(phi, pd.DataFrame):
        raise TypeError("phi must be a DataFrame (L, V)")
    v = np.asarray(impact_z, dtype=float).ravel()
    L = v.shape[0]
    if phi.shape[0] != L:
        raise ValueError(f"phi must have L = {L} rows to match impact_z, got shape {phi.shape}")
    P = phi.to_numpy(dtype=float)
    _check_finite(P, "phi")
    _check_finite(v, "impact_z")
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        logger.warning("term_impact: phi rows do not sum to one (min %.4f, max %.4f)", row_sums.min(), row_sums.max())
    M_map, rank = _lsq_map(P.T, rcond)
    if rank < L:
        logger.warning("term_impact: phi has numerical rank %d < L = %d; using the minimum-norm solution", rank, L)
    I_w = M_map @ v
    return pd.Series(I_w, index=phi.columns, name="impact_w")


def wrap_up(
    fit: SparseIPCAResult,
    shocks: ShockPanel,
    eval_cfg: EvaluationConfig,
    observables: dict[str, pd.Series] | None = None,
    phi: pd.DataFrame | None = None,
) -> WrapUpResult:
    """Structural objects of a Sparse IPCA fit (BKS step 3 and Section 6.1).

    Runs, in order: :func:`recover_A` (``A``), :func:`state_variables`
    (``x_tau`` on the shock dates), ``b_MVE = mu_f' Sigma_ff^-1``
    (:meth:`SparseIPCAResult.b_mve`), ``x_MVE,tau = b_MVE' x_tau``,
    :func:`impact_vectors` (Eq. 10-11), :func:`project_observable` and the
    observable impact vectors ``I_{z->obs} = I_{z->x} b_obs`` for every entry
    of ``observables`` (footnote 17), and :func:`term_impact` (Eq. 12) when
    ``phi`` is given.

    Alignment: the columns of ``shocks.z`` and the rows of ``phi`` are
    matched to ``fit.instrument_names[1:]`` by name (a silent permutation
    would scramble every impact vector); observables are matched to the
    populated factor periods by index (or by period label when
    ``fit.meta["period"]`` is set). All pseudo-inverses use ``eval_cfg.rcond``.

    Returns
    -------
    :class:`WrapUpResult` with ``A`` and ``impact_z_to_x`` as ``(L, K)``
    DataFrames (index = topics, columns ``f1..fK``), ``states`` as
    ``(n_days, K)`` with columns ``x1..xK`` on the shock dates, ``x_mve`` and
    ``impact_z_to_mve`` as Series, ``impact_z_to_obs[name]`` and
    ``obs_projection[name] = {"b_obs", "r2", "n_obs"}`` per observable,
    ``impact_w_to_mve`` when ``phi`` is given (term-level vectors of the
    observables in ``meta["impact_w_to_obs"]``), and ``rank_deficient`` from
    :func:`recover_A`.
    """
    rcond = float(eval_cfg.rcond)
    K = int(fit.K)
    topics = [str(n) for n in fit.instrument_names[1:]]
    L = len(topics)
    Gt = np.asarray(fit.Gamma_tilde, dtype=float)
    if Gt.shape != (L, K):
        raise ValueError(f"fit.Gamma_tilde has shape {Gt.shape}, expected ({L}, {K}) from instrument_names/K")
    fcols = [f"f{k + 1}" for k in range(K)]

    z = _align_columns(shocks.z, topics, "shocks.z")
    A, rank_deficient = recover_A(Gt, fit.Sigma_ff, rcond)
    A_df = pd.DataFrame(A, index=topics, columns=fcols)

    aligned = ShockPanel(z=z, window=shocks.window, scale=shocks.scale)
    states = state_variables(A, aligned, rcond)

    b = np.asarray(fit.b_mve(rcond), dtype=float).ravel()
    x_mve = pd.Series(states.to_numpy() @ b, index=states.index, name="x_mve")
    I_zx, I_zmve = impact_vectors(A, b, rcond)
    impact_z_to_x = pd.DataFrame(I_zx, index=topics, columns=fcols)
    impact_z_to_mve = pd.Series(I_zmve, index=topics, name="impact_z_to_mve")

    # Observables: project on populated factor periods only (zero rows of F
    # for empty periods would otherwise enter the regression).
    impact_z_to_obs: dict[str, pd.Series] = {}
    obs_projection: dict[str, dict[str, Any]] = {}
    period = fit.meta.get("period") if isinstance(fit.meta, dict) else None
    if observables:
        F = fit.factors_frame()
        if fit.populated is not None:
            populated = np.asarray(fit.populated, dtype=bool)
        else:
            populated = np.any(F.to_numpy() != 0.0, axis=1)
        F = F.loc[populated]
        for name, series in observables.items():
            b_obs, r2 = project_observable(F, series, period=period)
            n_obs = int(_align_observable(F, series, period)[1].shape[0])
            impact_z_to_obs[name] = pd.Series(I_zx @ b_obs, index=topics, name=f"impact_z_to_{name}")
            obs_projection[name] = {"b_obs": b_obs, "r2": r2, "n_obs": n_obs}
            logger.info("wrap_up: observable %r projected on %d periods, R2 = %.3f", name, n_obs, r2)

    impact_w_to_mve: pd.Series | None = None
    meta: dict[str, Any] = {
        "K": K,
        "L": L,
        "lam": float(fit.lam),
        "n_selected": int(fit.n_selected),
        "rcond": rcond,
        "n_days": int(len(states)),
        "n_days_finite": int(np.isfinite(states.to_numpy()).all(axis=1).sum()),
    }
    if phi is not None:
        if not isinstance(phi, pd.DataFrame):
            raise TypeError("phi must be a DataFrame (L, V) indexed by topic")
        phi_lookup: dict[str, Any] = {}
        for r in phi.index:
            phi_lookup.setdefault(str(r), r)
        missing = [t for t in topics if t not in phi_lookup]
        if missing:
            raise ValueError(f"phi is missing rows for {len(missing)} topic(s), e.g. {missing[:5]}")
        phi_al = phi.loc[[phi_lookup[t] for t in topics]]
        impact_w_to_mve = term_impact(phi_al, I_zmve, rcond).rename("impact_w_to_mve")
        meta["impact_w_to_obs"] = {
            name: term_impact(phi_al, vec.to_numpy(), rcond).rename(f"impact_w_to_{name}")
            for name, vec in impact_z_to_obs.items()
        }

    logger.info(
        "wrap_up: L=%d K=%d selected=%d rank_deficient=%s days=%d (finite %d)",
        L,
        K,
        fit.n_selected,
        rank_deficient,
        meta["n_days"],
        meta["n_days_finite"],
    )
    return WrapUpResult(
        A=A_df,
        states=states,
        x_mve=x_mve,
        b_mve=b,
        impact_z_to_x=impact_z_to_x,
        impact_z_to_mve=impact_z_to_mve,
        impact_z_to_obs=impact_z_to_obs,
        obs_projection=obs_projection,
        impact_w_to_mve=impact_w_to_mve,
        rank_deficient=bool(rank_deficient),
        meta=meta,
    )


def retrieval_scores(impact_z: pd.Series, z: pd.DataFrame) -> pd.Series:
    """Model-implied impact ``I' z(m)`` of every row of ``z`` (BKS Section 6.3).

    For a day ``tau`` (row = ``z_tau``) or an article ``m`` (row = the
    article's own shock ``z(m) = theta_m - (1/w) sum_{j=1..w} theta_{tau_m - j}``,
    BKS footnote 18) the score is the inner product ``I_{z->target}' z``,
    the target's response implied by Eq. 11. BKS retrieve, on extreme market
    days, the article with the largest score.

    Columns of ``z`` are matched to ``impact_z.index`` by name (``z`` may hold
    extra columns; missing ones raise). Rows with any non-finite entry among
    the used columns score ``NaN`` rather than being silently treated as
    zero. If ``impact_z`` is a bare array it is matched positionally and
    ``z`` must then have exactly ``L`` columns.

    Returns a Series on ``z.index`` named after ``impact_z`` (or ``"score"``).
    """
    if not isinstance(z, pd.DataFrame):
        raise TypeError("z must be a DataFrame with one row per day or article")
    if isinstance(impact_z, pd.Series):
        Z = _align_columns(z, [str(c) for c in impact_z.index], "z").to_numpy(dtype=float)
        v = impact_z.to_numpy(dtype=float)
        name = impact_z.name if impact_z.name is not None else "score"
    else:
        v = np.asarray(impact_z, dtype=float).ravel()
        if z.shape[1] != v.shape[0]:
            raise ValueError(f"z has {z.shape[1]} columns but impact_z has {v.shape[0]} entries")
        Z = z.to_numpy(dtype=float)
        name = "score"
    _check_finite(v, "impact_z")
    scores = np.full(Z.shape[0], np.nan)
    finite = np.all(np.isfinite(Z), axis=1)
    if finite.any():
        scores[finite] = Z[finite] @ v
    return pd.Series(scores, index=z.index, name=str(name))
