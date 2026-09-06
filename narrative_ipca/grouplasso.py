"""Group lasso on the Gram form by groupwise majorisation descent (Yang & Zou 2015).

The Gamma-step of Sparse IPCA (BKS Eq. 8, App. B.2) is a group lasso whose
design matrix has one row ``kron(c_{i,t-1}, f_t)`` per asset-period. The loss
depends on the data only through the Gram pair

    G = sum_t (C_{t-1}' C_{t-1}) kron (f_t f_t'),    b = sum_t (C_{t-1}' r_t) kron f_t,

so this module solves the generic convex program

    min_x  0.5 x'Gx - b'x + sum_g pen_g ||x_g||_2

where the groups are contiguous blocks of ``group_size`` coordinates. With
``vect(Gamma)`` stacking the rows of ``Gamma`` (coordinate ``l*K + k`` holds
``Gamma[l, k]``) a group is one row ``Gamma_l`` and the program is Eq. 8 with
``{f_t}`` held fixed (up to the additive constants ``0.5 sum r^2`` and
``sum_t ||f_t||^2``).

Algorithm (groupwise majorisation descent, GMD)
-----------------------------------------------
For group ``g`` the smooth part restricted to ``x_g`` has Hessian ``G_gg``,
which is bounded above by ``gamma_g I`` with ``gamma_g = lambda_max(G_gg)``.
Minimising the resulting spherical majoriser plus the group penalty gives the
closed-form update

    u   = x_g + (b_g - (G x)_g) / gamma_g
    x_g <- max(0, 1 - pen_g / (gamma_g ||u||)) u          (group soft-threshold)

which is a descent step on the *exact* objective, so the objective is
non-increasing over sweeps. Convergence to the (convex) optimum follows from
Tseng (2001) for block coordinate descent with block-separable non-smooth
terms; the tests check the KKT residual rather than a reference solver.

Execution paths
---------------
The sweep loop is a sequential (Gauss-Seidel) pass over the groups and cannot
be vectorised, so it exists twice with the *same* control flow (update order,
active-set acceleration, refresh schedule, convergence rule):

* ``_gmd_loop_numba``: scalar loops compiled by numba (``nopython``,
  ``cache=True``); used when numba is importable.
* ``_gmd_loop_numpy``: the pure-numpy reference implementation; always
  available and used when numba is missing, when the environment variable
  ``NARRATIVE_IPCA_NO_NUMBA=1`` is set (the import is then skipped), or when
  the module flag ``USE_NUMBA`` is set to ``False`` at run time.

The two paths perform the same arithmetic in the same order; they differ
only in how the BLAS behind numpy associates the terms of the rank-``K``
updates of ``G x``, so results agree to rounding (the tests require
``~1e-12``). Everything around the loop (validation, block curvatures,
pinning of zero-curvature groups, the final ``G x`` refresh and the info
dict) is shared code.

The module is a self-contained convex solver and knows nothing about IPCA.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "group_soft_threshold",
    "group_lasso_gram",
    "group_lasso_objective",
    "kkt_violation",
    "active_backend",
    "HAVE_NUMBA",
    "USE_NUMBA",
]

# Full recomputation of ``G @ x`` every this many sweeps. The incremental
# rank-``group_size`` updates are exact in real arithmetic; the refresh only
# bounds accumulated rounding.
_REFRESH_EVERY = 20

#: ``NARRATIVE_IPCA_NO_NUMBA=1`` (any value other than ``""``, ``0``,
#: ``false``, ``no``) skips the numba import and forces the numpy path.
NUMBA_DISABLED_BY_ENV: bool = os.environ.get("NARRATIVE_IPCA_NO_NUMBA", "").strip().lower() not in ("", "0", "false", "no")

try:
    if NUMBA_DISABLED_BY_ENV:
        raise ImportError("numba disabled by NARRATIVE_IPCA_NO_NUMBA")
    import numba as _numba
except Exception:  # ImportError, or a broken numba installation
    _numba = None
    HAVE_NUMBA: bool = False
else:
    HAVE_NUMBA = True

#: Module flag: the jitted kernel is used only when this is ``True`` *and*
#: numba imported. Set it to ``False`` to force the numpy reference path at
#: run time (the tests parametrise over both values).
USE_NUMBA: bool = HAVE_NUMBA


def active_backend() -> str:
    """``"numba"`` or ``"numpy"``: the sweep implementation the next call will use."""
    return "numba" if (USE_NUMBA and HAVE_NUMBA) else "numpy"


# ---------------------------------------------------------------------------
# elementary operators
# ---------------------------------------------------------------------------
def group_soft_threshold(v: np.ndarray, t: float) -> np.ndarray:
    """Block soft-threshold ``max(0, 1 - t/||v||_2) v``, the prox of ``t ||.||_2``.

    Implements the per-group minimiser of ``0.5 ||x - v||^2 + t ||x||_2``
    (the update of the GMD algorithm once the majoriser is formed).

    Assumptions: ``t >= 0``; a non-positive ``t`` returns a copy of ``v``.
    Returns an exact zero vector when ``||v|| <= t`` so that group membership
    can be read off with ``!= 0``.
    """
    v = np.asarray(v, dtype=float)
    if t <= 0.0:
        return v.copy()
    nrm = float(np.linalg.norm(v))
    if nrm <= t or nrm == 0.0:
        return np.zeros_like(v)
    return (1.0 - t / nrm) * v


def group_lasso_objective(
    G: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    group_size: int,
    penalties: np.ndarray,
) -> float:
    """Evaluate ``0.5 x'Gx - b'x + sum_g pen_g ||x_g||_2``.

    This is BKS Eq. 8 with ``{f_t}`` fixed, minus the terms that do not depend
    on ``Gamma`` (``0.5 sum r^2`` and ``sum_t ||f_t||^2``). No constant is
    added; callers that need the full Eq. 8 value use
    :func:`narrative_ipca.sparse_ipca.objective_value`.
    """
    G, b, penalties, x, n_groups = _validate(G, b, group_size, penalties, x)
    assert x is not None
    Gx = G @ x
    quad = 0.5 * float(x @ Gx) - float(b @ x)
    blocks = x.reshape(n_groups, int(group_size))
    pen = float(penalties @ np.linalg.norm(blocks, axis=1))
    return quad + pen


def kkt_violation(
    G: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    group_size: int,
    penalties: np.ndarray,
) -> float:
    """Maximum KKT residual of the group lasso at ``x`` (zero at the optimum).

    With ``grad = Gx - b`` the optimality conditions of
    ``min 0.5 x'Gx - b'x + sum_g pen_g ||x_g||`` are

    * active group (``x_g != 0``):   ``grad_g + pen_g x_g / ||x_g|| = 0``,
    * inactive group (``x_g == 0``): ``||grad_g|| <= pen_g``.

    The residual of an active group is the norm of the left-hand side; that
    of an inactive group is ``max(0, ||grad_g|| - pen_g)``. Returns the
    maximum over groups. Assumes ``G`` symmetric PSD (the problem is convex).
    """
    G, b, penalties, x, n_groups = _validate(G, b, group_size, penalties, x)
    assert x is not None
    gs = int(group_size)
    grad = (G @ x - b).reshape(n_groups, gs)
    blocks = x.reshape(n_groups, gs)
    norms = np.linalg.norm(blocks, axis=1)
    worst = 0.0
    for g in range(n_groups):
        if norms[g] > 0.0:
            resid = float(np.linalg.norm(grad[g] + penalties[g] * blocks[g] / norms[g]))
        else:
            resid = max(0.0, float(np.linalg.norm(grad[g])) - float(penalties[g]))
        worst = max(worst, resid)
    return worst


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------
def group_lasso_gram(
    G: np.ndarray,
    b: np.ndarray,
    group_size: int,
    penalties: np.ndarray,
    x0: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimise ``0.5 x'Gx - b'x + sum_g pen_g ||x_g||_2`` by GMD (Yang & Zou 2015).

    This is the Gamma-step solver of BKS App. B.2 (DESIGN.md D21). Groups are
    the contiguous blocks ``x[g*group_size:(g+1)*group_size]``.

    Algorithm
    ---------
    1. Block curvatures ``gamma_g = lambda_max(sym(G_gg))`` once.
    2. Sweep over groups: ``u = x_g + (b_g - (Gx)_g)/gamma_g`` and
       ``x_g <- group_soft_threshold(u, pen_g/gamma_g)``; ``Gx`` is updated
       incrementally (``Gx += G[:, g] @ delta``) and fully refreshed every
       ``_REFRESH_EVERY`` sweeps and before returning.
    3. After a full sweep the solver cycles over the currently active groups
       only until they stop moving, then does another full sweep (standard
       active-set acceleration; every sweep of either kind is a descent step).
    4. Converged when a *full* sweep changes no coordinate by more than
       ``tol`` (max absolute change).

    Groups with ``gamma_g <= 0`` carry no curvature (for PSD ``G`` their rows
    and columns are zero) and are pinned to zero.

    The sweep loop runs in the numba kernel when available and in the numpy
    reference implementation otherwise (module docstring, "Execution paths");
    both give the same result to rounding.

    Parameters
    ----------
    G : (p, p) symmetric PSD Gram matrix, ``p = n_groups * group_size``.
    b : (p,) linear term.
    group_size : coordinates per group (``K`` for Sparse IPCA).
    penalties : (n_groups,) non-negative penalties, or a scalar broadcast.
    x0 : optional warm start (copied, never modified in place).
    max_iter : maximum number of sweeps (full and active-set sweeps both count).
    tol : convergence threshold on the largest absolute coordinate change.

    Returns
    -------
    x : (p,) solution with exact zeros in inactive groups.
    info : ``{"n_iter", "converged", "max_change", "objective"}`` where
        ``objective`` is :func:`group_lasso_objective` at ``x``.
    """
    G, b, penalties, x0v, n_groups = _validate(G, b, group_size, penalties, x0)
    gs = int(group_size)
    if max_iter < 0:
        raise ValueError(f"max_iter must be >= 0, got {max_iter}")
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    G = np.ascontiguousarray(G)
    b = np.ascontiguousarray(b)
    x = np.zeros(G.shape[0], dtype=float) if x0v is None else np.array(x0v, dtype=float, copy=True)
    if not np.all(np.isfinite(x)):
        raise ValueError("x0 contains non-finite values")

    curv = _block_curvatures(G, n_groups, gs)
    thr = np.where(curv > 0.0, penalties / np.where(curv > 0.0, curv, 1.0), 0.0)
    thr = np.ascontiguousarray(thr, dtype=float)

    # Pin zero-curvature groups before the first sweep.
    if n_groups > 0:
        dead = curv <= 0.0
        if np.any(dead):
            x.reshape(n_groups, gs)[dead] = 0.0
    Gx = np.ascontiguousarray(G @ x)

    if USE_NUMBA and HAVE_NUMBA:
        n_iter, converged, max_change = _gmd_loop_numba(G, b, x, Gx, curv, thr, gs, int(max_iter), float(tol), int(_REFRESH_EVERY))
    else:
        n_iter, converged, max_change = _gmd_loop_numpy(G, b, x, Gx, curv, thr, gs, int(max_iter), float(tol), int(_REFRESH_EVERY))

    Gx = G @ x
    blocks = x.reshape(n_groups, gs)
    objective = 0.5 * float(x @ Gx) - float(b @ x) + float(penalties @ np.linalg.norm(blocks, axis=1))
    if not converged:
        logger.debug("group_lasso_gram: not converged after %d sweeps (max_change=%.3e)", n_iter, max_change)
    info: dict[str, Any] = {
        "n_iter": int(n_iter),
        "converged": bool(converged),
        "max_change": float(max_change),
        "objective": float(objective),
    }
    return x, info


# ---------------------------------------------------------------------------
# the sweep loop, numpy reference implementation
# ---------------------------------------------------------------------------
def _gmd_loop_numpy(
    G: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    Gx: np.ndarray,
    curv: np.ndarray,
    thr: np.ndarray,
    gs: int,
    max_iter: int,
    tol: float,
    refresh_every: int,
) -> tuple[int, bool, float]:
    """GMD sweeps until convergence; the pure-numpy reference implementation.

    Updates ``x`` and ``Gx`` in place and returns ``(n_iter, converged,
    max_change)``. ``curv`` holds the block curvatures (``<= 0`` marks a
    pinned group), ``thr`` the per-group thresholds ``pen_g / curv_g``.
    :func:`_gmd_loop_numba` is the compiled twin with the same control flow.
    """
    n_groups = int(curv.shape[0])
    slices = [slice(g * gs, (g + 1) * gs) for g in range(n_groups)]

    def sweep(groups: list[int]) -> float:
        nonlocal Gx  # ``Gx += ...`` below is an in-place ufunc call on the caller's array
        max_change = 0.0
        for g in groups:
            gam = curv[g]
            if gam <= 0.0:
                continue
            sl = slices[g]
            xg = x[sl]
            u = xg + (b[sl] - Gx[sl]) / gam
            nrm = math.sqrt(float(u @ u))
            t = thr[g]
            if nrm <= t or nrm == 0.0:
                if not np.any(xg != 0.0):
                    continue
                new = np.zeros(gs)
            else:
                new = (1.0 - t / nrm) * u
            delta = new - xg
            change = float(np.abs(delta).max())
            if change > 0.0:
                x[sl] = new
                Gx += delta @ G[sl]  # G symmetric: G[:, sl] @ delta
                if change > max_change:
                    max_change = change
        return max_change

    all_groups = [g for g in range(n_groups) if curv[g] > 0.0]
    n_iter = 0
    converged = max_iter == 0 and n_groups == 0
    max_change = 0.0
    while n_iter < max_iter:
        max_change = sweep(all_groups)
        n_iter += 1
        if n_iter % refresh_every == 0:
            np.matmul(G, x, out=Gx)
        if max_change < tol:
            converged = True
            break
        nonzero = np.any(x.reshape(n_groups, gs) != 0.0, axis=1)
        active = [g for g in all_groups if nonzero[g]]
        if len(active) == len(all_groups):
            continue  # nothing to gain from an active-set cycle
        while n_iter < max_iter and active:
            mc = sweep(active)
            n_iter += 1
            if n_iter % refresh_every == 0:
                np.matmul(G, x, out=Gx)
            if mc < tol:
                break
    return int(n_iter), bool(converged), float(max_change)


# ---------------------------------------------------------------------------
# the sweep loop, numba kernel (same control flow as the numpy reference)
# ---------------------------------------------------------------------------
if HAVE_NUMBA:

    @_numba.njit(cache=True, nogil=True, error_model="numpy")
    def _matvec_into(G, x, Gx):  # pragma: no cover - compiled
        """``Gx[:] = G @ x`` (the periodic refresh)."""
        p = x.shape[0]
        for i in range(p):
            s = 0.0
            for j in range(p):
                s += G[i, j] * x[j]
            Gx[i] = s

    @_numba.njit(cache=True, nogil=True, error_model="numpy")
    def _gmd_sweep(G, b, x, Gx, curv, thr, gs, groups, n_list, u, new, delta, tmp):  # pragma: no cover - compiled
        """One GMD pass over ``groups[:n_list]``; returns the largest coordinate change.

        ``u``, ``new``, ``delta`` are length-``gs`` scratch buffers and ``tmp``
        a length-``p`` one, allocated once by the caller.
        """
        p = x.shape[0]
        max_change = 0.0
        for idx in range(n_list):
            g = groups[idx]
            gam = curv[g]
            if gam <= 0.0:
                continue
            off = g * gs
            nrm2 = 0.0
            for i in range(gs):
                ui = x[off + i] + (b[off + i] - Gx[off + i]) / gam
                u[i] = ui
                nrm2 += ui * ui
            nrm = math.sqrt(nrm2)
            t = thr[g]
            if nrm <= t or nrm == 0.0:
                nonzero = False
                for i in range(gs):
                    if x[off + i] != 0.0:
                        nonzero = True
                        break
                if not nonzero:
                    continue
                for i in range(gs):
                    new[i] = 0.0
            else:
                scale = 1.0 - t / nrm
                for i in range(gs):
                    new[i] = scale * u[i]
            change = 0.0
            for i in range(gs):
                d = new[i] - x[off + i]
                delta[i] = d
                ad = abs(d)
                if ad > change:
                    change = ad
            if change > 0.0:
                for i in range(gs):
                    x[off + i] = new[i]
                # Gx += delta @ G[off:off+gs]: the rank-gs product is accumulated
                # row by row into ``tmp`` (the association of a sequential dot
                # over i, as the BLAS product does) and then added to Gx. Each
                # pass is a plain loop over j, so it vectorises.
                d = delta[0]
                for j in range(p):
                    tmp[j] = d * G[off, j]
                for i in range(1, gs):
                    d = delta[i]
                    row = off + i
                    for j in range(p):
                        tmp[j] += d * G[row, j]
                for j in range(p):
                    Gx[j] += tmp[j]
                if change > max_change:
                    max_change = change
        return max_change

    @_numba.njit(cache=True, nogil=True, error_model="numpy")
    def _gmd_loop_numba(G, b, x, Gx, curv, thr, gs, max_iter, tol, refresh_every):  # pragma: no cover - compiled
        """Compiled twin of :func:`_gmd_loop_numpy` (same loop, scalar arithmetic)."""
        n_groups = curv.shape[0]
        u = np.empty(gs)
        new = np.empty(gs)
        delta = np.empty(gs)
        tmp = np.empty(x.shape[0])
        all_groups = np.empty(n_groups, dtype=np.int64)
        n_all = 0
        for g in range(n_groups):
            if curv[g] > 0.0:
                all_groups[n_all] = g
                n_all += 1
        active = np.empty(n_groups, dtype=np.int64)
        n_iter = 0
        converged = max_iter == 0 and n_groups == 0
        max_change = 0.0
        while n_iter < max_iter:
            max_change = _gmd_sweep(G, b, x, Gx, curv, thr, gs, all_groups, n_all, u, new, delta, tmp)
            n_iter += 1
            if n_iter % refresh_every == 0:
                _matvec_into(G, x, Gx)
            if max_change < tol:
                converged = True
                break
            n_active = 0
            for idx in range(n_all):
                g = all_groups[idx]
                off = g * gs
                nonzero = False
                for i in range(gs):
                    if x[off + i] != 0.0:
                        nonzero = True
                        break
                if nonzero:
                    active[n_active] = g
                    n_active += 1
            if n_active == n_all:
                continue  # nothing to gain from an active-set cycle
            while n_iter < max_iter and n_active > 0:
                mc = _gmd_sweep(G, b, x, Gx, curv, thr, gs, active, n_active, u, new, delta, tmp)
                n_iter += 1
                if n_iter % refresh_every == 0:
                    _matvec_into(G, x, Gx)
                if mc < tol:
                    break
        return n_iter, converged, max_change

else:

    def _gmd_loop_numba(*args, **kwargs):  # pragma: no cover - only reached when numba is missing
        raise RuntimeError("numba is not available; set USE_NUMBA = False or install narrative-ipca[accel]")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _block_curvatures(G: np.ndarray, n_groups: int, gs: int) -> np.ndarray:
    """``lambda_max`` of every symmetrised diagonal block; non-finite or negative -> 0.

    The blocks are gathered as one ``(n_groups, gs, gs)`` stack and passed to
    a single batched ``eigvalsh`` call (the same LAPACK routine per block as
    a per-block call, so the values are identical).
    """
    if n_groups == 0:
        return np.zeros(0, dtype=float)
    idx = np.arange(n_groups)
    blocks = G.reshape(n_groups, gs, n_groups, gs)[idx, :, idx, :]  # (n_groups, gs, gs)
    if gs == 1:
        c = blocks[:, 0, 0].astype(float)
    else:
        sym = 0.5 * (blocks + np.transpose(blocks, (0, 2, 1)))
        try:
            c = np.linalg.eigvalsh(sym)[:, -1]
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            c = np.empty(n_groups, dtype=float)
            for g in range(n_groups):
                try:
                    c[g] = np.linalg.eigvalsh(sym[g])[-1]
                except np.linalg.LinAlgError:
                    c[g] = np.linalg.norm(sym[g])  # Frobenius >= spectral: still a valid majoriser
    curv = np.where(np.isfinite(c) & (c > 0.0), c, 0.0)
    return np.ascontiguousarray(curv, dtype=float)


def _validate(
    G: np.ndarray,
    b: np.ndarray,
    group_size: int,
    penalties: np.ndarray,
    x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int]:
    """Shape-check the Gram inputs and return float arrays plus ``n_groups``."""
    G = np.asarray(G, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square 2-D, got shape {G.shape}")
    p = int(G.shape[0])
    if b.shape != (p,):
        raise ValueError(f"b must have shape ({p},) to match G {G.shape}, got {b.shape}")
    if not isinstance(group_size, (int, np.integer)) or group_size < 1:
        raise ValueError(f"group_size must be a positive int, got {group_size!r}")
    gs = int(group_size)
    if p % gs != 0:
        raise ValueError(f"group_size={gs} does not divide p={p}")
    n_groups = p // gs
    penalties = np.asarray(penalties, dtype=float)
    if penalties.ndim == 0:
        penalties = np.full(n_groups, float(penalties))
    penalties = penalties.ravel()
    if penalties.shape != (n_groups,):
        raise ValueError(f"penalties must have shape ({n_groups},), got {penalties.shape}")
    if np.any(penalties < 0) or not np.all(np.isfinite(penalties)):
        raise ValueError("penalties must be finite and non-negative")
    if x is not None:
        x = np.asarray(x, dtype=float).ravel()
        if x.shape != (p,):
            raise ValueError(f"x must have shape ({p},), got {x.shape}")
    return G, b, penalties, x, n_groups
