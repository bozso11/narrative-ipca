"""Mathematical tests for ``narrative_ipca.grouplasso``.

The solver is checked against optimality conditions (KKT residual), against
closed forms (unpenalised least squares, orthogonal design) and for the
descent property of every sweep, not against another solver. Every solver
test runs on both sweep implementations (the numba kernel and the numpy
reference, ``backend`` fixture), and the two are checked against each other
sweep by sweep and at convergence.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from narrative_ipca import grouplasso as gl
from narrative_ipca.grouplasso import (
    group_lasso_gram,
    group_lasso_objective,
    group_soft_threshold,
    kkt_violation,
)

BACKENDS = [
    pytest.param("numpy", id="numpy"),
    pytest.param("numba", id="numba", marks=pytest.mark.skipif(not gl.HAVE_NUMBA, reason="numba not available (not installed or disabled by NARRATIVE_IPCA_NO_NUMBA)")),
]


@pytest.fixture(params=BACKENDS)
def backend(request, monkeypatch):
    """Run the test on one sweep implementation; ``USE_NUMBA`` is restored afterwards."""
    monkeypatch.setattr(gl, "USE_NUMBA", request.param == "numba")
    assert gl.active_backend() == request.param
    return request.param


def _solve_with(backend_name: str, monkeypatch, *args, **kwargs):
    """``group_lasso_gram`` on a named backend (for the agreement tests)."""
    monkeypatch.setattr(gl, "USE_NUMBA", backend_name == "numba")
    return group_lasso_gram(*args, **kwargs)


# ---------------------------------------------------------------------------
# problem generators
# ---------------------------------------------------------------------------
def _gram_problem(seed: int, n_rows: int, n_groups: int, group_size: int, pen_scale: float = 0.3):
    """Least-squares Gram pair ``G = D'D``, ``b = D'y`` (singular when n_rows < p)."""
    rng = np.random.default_rng(seed)
    p = n_groups * group_size
    D = rng.standard_normal((n_rows, p))
    D[:, : p // 2] *= 3.0  # unequal column scales
    x_true = rng.standard_normal(p)
    x_true.reshape(n_groups, group_size)[rng.random(n_groups) < 0.5] = 0.0
    y = D @ x_true + 0.5 * rng.standard_normal(n_rows)
    G = D.T @ D
    b = D.T @ y
    pen = pen_scale * np.linalg.norm(b) / np.sqrt(n_groups) * rng.random(n_groups)
    pen[rng.random(n_groups) < 0.2] = 0.0  # some unpenalised groups
    return G, b, pen


def _ill_conditioned_problem(seed: int, n_groups: int, group_size: int, rho: float = 0.999):
    """Strongly correlated columns (condition number 1e4-1e5), like covariance instruments."""
    rng = np.random.default_rng(seed)
    p = n_groups * group_size
    common = rng.standard_normal((400, 1))
    D = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * rng.standard_normal((400, p))
    x_true = rng.standard_normal(p)
    x_true.reshape(n_groups, group_size)[rng.random(n_groups) < 0.6] = 0.0
    y = D @ x_true + 0.3 * rng.standard_normal(400)
    G = D.T @ D
    b = D.T @ y
    pen = 0.05 * np.linalg.norm(b) / np.sqrt(n_groups) * (0.5 + rng.random(n_groups))
    return G, b, pen


# ---------------------------------------------------------------------------
# group_soft_threshold
# ---------------------------------------------------------------------------
def test_group_soft_threshold_formula_and_hard_zero():
    v = np.array([3.0, 4.0])  # norm 5
    assert np.array_equal(group_soft_threshold(v, 5.0), np.zeros(2))
    assert np.array_equal(group_soft_threshold(v, 6.0), np.zeros(2))
    np.testing.assert_allclose(group_soft_threshold(v, 1.0), 0.8 * v)
    np.testing.assert_array_equal(group_soft_threshold(v, 0.0), v)
    np.testing.assert_array_equal(group_soft_threshold(v, -1.0), v)
    assert np.array_equal(group_soft_threshold(np.zeros(3), 0.1), np.zeros(3))


def test_group_soft_threshold_is_prox_of_group_norm():
    """``S(v, t)`` minimises ``0.5||x - v||^2 + t||x||`` over a random cloud of candidates."""
    rng = np.random.default_rng(1)
    v = rng.standard_normal(4)
    t = 0.7
    x_star = group_soft_threshold(v, t)
    f_star = 0.5 * np.sum((x_star - v) ** 2) + t * np.linalg.norm(x_star)
    for _ in range(200):
        x = x_star + 0.1 * rng.standard_normal(4)
        f = 0.5 * np.sum((x - v) ** 2) + t * np.linalg.norm(x)
        assert f >= f_star - 1e-12


# ---------------------------------------------------------------------------
# objective
# ---------------------------------------------------------------------------
def test_objective_matches_brute_force():
    G, b, pen = _gram_problem(0, 30, 5, 3)
    rng = np.random.default_rng(5)
    x = rng.standard_normal(15)
    ref = 0.5 * x @ G @ x - b @ x + sum(pen[g] * np.linalg.norm(x[3 * g : 3 * g + 3]) for g in range(5))
    assert group_lasso_objective(G, b, x, 3, pen) == pytest.approx(ref, rel=1e-13)
    # scalar penalty broadcast
    ref2 = 0.5 * x @ G @ x - b @ x + 0.4 * sum(np.linalg.norm(x[3 * g : 3 * g + 3]) for g in range(5))
    assert group_lasso_objective(G, b, x, 3, 0.4) == pytest.approx(ref2, rel=1e-13)


# ---------------------------------------------------------------------------
# (a) descent property sweep by sweep
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed,gs", [(0, 1), (1, 2), (2, 3), (3, 4)])
def test_objective_non_increasing_across_sweeps(backend, seed, gs):
    G, b, pen = _gram_problem(seed, 40, 8, gs)
    rng = np.random.default_rng(seed + 100)
    x = rng.standard_normal(G.shape[0])  # start away from the optimum
    values = [group_lasso_objective(G, b, x, gs, pen)]
    for _ in range(60):
        x, info = group_lasso_gram(G, b, gs, pen, x0=x, max_iter=1, tol=0.0)
        values.append(info["objective"])
        assert info["objective"] == pytest.approx(group_lasso_objective(G, b, x, gs, pen), rel=1e-12)
    values = np.asarray(values)
    tol = 1e-12 * np.maximum(1.0, np.abs(values[:-1]))
    assert np.all(np.diff(values) <= tol), "objective increased between sweeps"


# ---------------------------------------------------------------------------
# (b) KKT residual at convergence, PD and singular Gram matrices
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seed,n_rows,n_groups,gs",
    [(0, 60, 6, 3), (1, 80, 10, 2), (2, 50, 12, 1), (3, 100, 5, 4),
     (4, 8, 6, 3), (5, 10, 8, 2), (6, 5, 10, 1)],  # last three: n_rows < p -> singular G
)
def test_kkt_violation_small_at_convergence(backend, seed, n_rows, n_groups, gs):
    G, b, pen = _gram_problem(seed, n_rows, n_groups, gs)
    singular = n_rows < n_groups * gs
    if singular:
        assert np.linalg.matrix_rank(G) < G.shape[0]
    x, info = group_lasso_gram(G, b, gs, pen, max_iter=100_000, tol=1e-14)
    assert info["converged"]
    assert kkt_violation(G, b, x, gs, pen) < 1e-6
    # inactive groups are exact zeros
    blocks = x.reshape(n_groups, gs)
    norms = np.linalg.norm(blocks, axis=1)
    assert np.all((norms == 0.0) | (norms > 0.0))


def test_kkt_violation_is_zero_at_least_squares_solution_and_positive_elsewhere():
    G, b, _ = _gram_problem(7, 50, 4, 3)
    x_ls = np.linalg.solve(G, b)
    assert kkt_violation(G, b, x_ls, 3, np.zeros(4)) < 1e-9
    rng = np.random.default_rng(3)
    assert kkt_violation(G, b, x_ls + rng.standard_normal(12), 3, np.zeros(4)) > 1e-3
    # zero is optimal iff ||b_g|| <= pen_g for every group
    pen_big = np.linalg.norm(b.reshape(4, 3), axis=1) + 1.0
    assert kkt_violation(G, b, np.zeros(12), 3, pen_big) == 0.0
    assert kkt_violation(G, b, np.zeros(12), 3, pen_big - 2.0) > 0.0


# ---------------------------------------------------------------------------
# (c) unpenalised = least squares; orthogonal design closed form
# ---------------------------------------------------------------------------
def test_zero_penalty_pd_gram_equals_linear_solve(backend):
    rng = np.random.default_rng(11)
    D = rng.standard_normal((200, 12))
    G = D.T @ D + np.eye(12)
    b = D.T @ rng.standard_normal(200)
    x, info = group_lasso_gram(G, b, 3, np.zeros(4), max_iter=20_000, tol=1e-13)
    assert info["converged"]
    np.testing.assert_allclose(x, np.linalg.solve(G, b), atol=1e-8, rtol=1e-8)


def test_orthogonal_design_closed_form(backend):
    """Block-diagonal G: the solution is the group soft-threshold of each block's LS solution."""
    rng = np.random.default_rng(12)
    n_groups, gs = 6, 3
    G = np.zeros((18, 18))
    for g in range(n_groups):
        A = rng.standard_normal((gs, gs))
        G[3 * g : 3 * g + 3, 3 * g : 3 * g + 3] = A @ A.T + 0.5 * np.eye(gs)
    b = rng.standard_normal(18)
    pen = rng.random(n_groups) * 2.0
    x, info = group_lasso_gram(G, b, gs, pen, max_iter=5000, tol=1e-14)
    assert info["converged"]
    # Reference: per block minimise 0.5 x'Ax - b'x + pen ||x||  (Newton on the scalar ||x||)
    for g in range(n_groups):
        sl = slice(3 * g, 3 * g + 3)
        A = G[sl, sl]
        bg = b[sl]
        if np.linalg.norm(bg) <= pen[g]:
            assert np.all(x[sl] == 0.0)
            continue
        # x = (A + pen/||x|| I)^-1 b ; solve the scalar fixed point for r = ||x||
        r = np.linalg.norm(np.linalg.solve(A, bg))
        for _ in range(200):
            xr = np.linalg.solve(A + pen[g] / r * np.eye(gs), bg)
            r_new = np.linalg.norm(xr)
            if abs(r_new - r) < 1e-15:
                break
            r = r_new
        np.testing.assert_allclose(x[sl], np.linalg.solve(A + pen[g] / r * np.eye(gs), bg), atol=1e-9)


def test_group_size_one_is_lasso_soft_threshold_on_diagonal_gram(backend):
    rng = np.random.default_rng(13)
    d = rng.random(7) + 0.5
    G = np.diag(d)
    b = rng.standard_normal(7)
    pen = rng.random(7)
    x, info = group_lasso_gram(G, b, 1, pen, max_iter=100, tol=1e-15)
    ref = np.sign(b) * np.maximum(np.abs(b) - pen, 0.0) / d
    np.testing.assert_allclose(x, ref, atol=1e-14)
    assert info["converged"]


# ---------------------------------------------------------------------------
# (d) large penalty -> all zero
# ---------------------------------------------------------------------------
def test_large_penalty_gives_all_zero(backend):
    G, b, _ = _gram_problem(21, 40, 5, 3)
    pen = np.linalg.norm(b.reshape(5, 3), axis=1) * 1.01  # above the KKT bound of every group
    rng = np.random.default_rng(2)
    x, info = group_lasso_gram(G, b, 3, pen, x0=rng.standard_normal(15), max_iter=10_000, tol=1e-14)
    assert info["converged"]
    assert np.all(x == 0.0)
    assert info["objective"] == 0.0
    # just below the bound, at least one group must be active
    x2, _ = group_lasso_gram(G, b, 3, pen * 0.9, max_iter=10_000, tol=1e-14)
    assert np.any(x2 != 0.0)


# ---------------------------------------------------------------------------
# (e) warm starts reproduce the optimum
# ---------------------------------------------------------------------------
def test_warm_start_reproduces_optimum_and_is_idempotent(backend):
    G, b, pen = _gram_problem(31, 70, 8, 3)
    x_cold, info_cold = group_lasso_gram(G, b, 3, pen, max_iter=50_000, tol=1e-14)
    rng = np.random.default_rng(4)
    x_warm, info_warm = group_lasso_gram(G, b, 3, pen, x0=5.0 * rng.standard_normal(24), max_iter=50_000, tol=1e-14)
    assert info_cold["converged"] and info_warm["converged"]
    assert info_cold["objective"] == pytest.approx(info_warm["objective"], rel=1e-10)
    np.testing.assert_allclose(x_cold, x_warm, atol=1e-6)
    assert np.array_equal(x_cold != 0.0, x_warm != 0.0)
    # starting at the optimum: one sweep, no movement, input not modified
    x0 = x_cold.copy()
    x_again, info_again = group_lasso_gram(G, b, 3, pen, x0=x0, max_iter=100, tol=1e-10)
    assert info_again["converged"] and info_again["n_iter"] == 1
    np.testing.assert_allclose(x_again, x_cold, atol=1e-9)
    assert np.array_equal(x0, x_cold)


# ---------------------------------------------------------------------------
# zero-curvature groups, validation
# ---------------------------------------------------------------------------
def test_zero_curvature_group_is_pinned_to_zero(backend):
    rng = np.random.default_rng(41)
    D = rng.standard_normal((30, 9))
    D[:, 3:6] = 0.0  # second group has no curvature: G rows/cols are zero
    G = D.T @ D
    b = D.T @ rng.standard_normal(30)
    x0 = rng.standard_normal(9)
    x, info = group_lasso_gram(G, b, 3, np.array([0.1, 0.1, 0.1]), x0=x0, max_iter=5000, tol=1e-14)
    assert np.all(x[3:6] == 0.0)
    assert info["converged"]
    assert kkt_violation(G, b, x, 3, np.array([0.1, 0.1, 0.1])) < 1e-8


def test_validation_errors():
    G = np.eye(6)
    b = np.ones(6)
    with pytest.raises(ValueError):
        group_lasso_gram(G, b, 4, np.zeros(1))  # 4 does not divide 6
    with pytest.raises(ValueError):
        group_lasso_gram(G, b[:5], 2, np.zeros(3))
    with pytest.raises(ValueError):
        group_lasso_gram(G, b, 2, np.array([0.1, -0.1, 0.1]))
    with pytest.raises(ValueError):
        group_lasso_gram(G, b, 2, np.zeros(2))
    with pytest.raises(ValueError):
        group_lasso_gram(G, b, 2, np.zeros(3), x0=np.zeros(5))
    with pytest.raises(ValueError):
        group_lasso_gram(np.ones((2, 3)), np.ones(2), 1, np.zeros(2))


def test_max_iter_zero_returns_start_unconverged(backend):
    G, b, pen = _gram_problem(3, 30, 4, 2)
    x0 = np.arange(8, dtype=float)
    x, info = group_lasso_gram(G, b, 2, pen, x0=x0, max_iter=0)
    np.testing.assert_array_equal(x, x0)
    assert info["n_iter"] == 0 and not info["converged"]


# ---------------------------------------------------------------------------
# the two sweep implementations agree (same algorithm, rounding-level differences)
# ---------------------------------------------------------------------------
needs_numba = pytest.mark.skipif(not gl.HAVE_NUMBA, reason="numba not available (not installed or disabled by NARRATIVE_IPCA_NO_NUMBA)")


def _rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


@needs_numba
@pytest.mark.parametrize("seed,n_rows,n_groups,gs", [(0, 60, 6, 3), (1, 80, 10, 2), (2, 50, 12, 1), (3, 100, 5, 4), (4, 8, 6, 3)])
def test_backends_agree_sweep_by_sweep(monkeypatch, seed, n_rows, n_groups, gs):
    """One sweep at a time from a common start: identical update order and identical info."""
    G, b, pen = _gram_problem(seed, n_rows, n_groups, gs)
    rng = np.random.default_rng(seed + 7)
    x_np = rng.standard_normal(G.shape[0])
    x_nb = x_np.copy()
    for _ in range(25):
        x_np, info_np = _solve_with("numpy", monkeypatch, G, b, gs, pen, x0=x_np, max_iter=1, tol=0.0)
        x_nb, info_nb = _solve_with("numba", monkeypatch, G, b, gs, pen, x0=x_nb, max_iter=1, tol=0.0)
        assert info_np["n_iter"] == info_nb["n_iter"] == 1
        assert info_np["converged"] == info_nb["converged"]
        assert np.array_equal(x_np != 0.0, x_nb != 0.0)
        assert _rel_diff(x_nb, x_np) < 1e-12
        assert info_nb["max_change"] == pytest.approx(info_np["max_change"], rel=1e-10, abs=1e-14)
        assert info_nb["objective"] == pytest.approx(info_np["objective"], rel=1e-12)


@needs_numba
@pytest.mark.parametrize("seed,n_rows,n_groups,gs", [(0, 60, 6, 3), (1, 80, 10, 2), (2, 50, 12, 1), (3, 100, 5, 4), (5, 10, 8, 2)])
def test_backends_agree_at_convergence(monkeypatch, seed, n_rows, n_groups, gs):
    """Full solve with active-set cycles and refreshes: same sweep count, same support, same x."""
    G, b, pen = _gram_problem(seed, n_rows, n_groups, gs)
    rng = np.random.default_rng(seed + 11)
    x0 = rng.standard_normal(G.shape[0])
    x_np, info_np = _solve_with("numpy", monkeypatch, G, b, gs, pen, x0=x0, max_iter=100_000, tol=1e-13)
    x_nb, info_nb = _solve_with("numba", monkeypatch, G, b, gs, pen, x0=x0, max_iter=100_000, tol=1e-13)
    assert info_np["converged"] and info_nb["converged"]
    if n_rows >= n_groups * gs:
        assert info_np["n_iter"] == info_nb["n_iter"]
    else:
        # singular G: thousands of sweeps with per-sweep changes shrinking by a factor close to
        # one, so whether a change of ~1e-13 falls below tol is a rounding-level decision that
        # the two paths may take one sweep apart; the solutions still agree.
        assert abs(info_np["n_iter"] - info_nb["n_iter"]) <= 1
    assert np.array_equal(x_np != 0.0, x_nb != 0.0)
    assert _rel_diff(x_nb, x_np) < 1e-12
    assert info_nb["objective"] == pytest.approx(info_np["objective"], rel=1e-12)
    assert kkt_violation(G, b, x_nb, gs, pen) < 1e-6


@needs_numba
@pytest.mark.parametrize("max_iter", [19, 20, 21, 45, 200])
def test_backends_agree_across_refresh_and_active_set_cycles(monkeypatch, max_iter):
    """Budget-limited runs that cross the periodic ``G x`` refresh (every 20 sweeps) and mix
    full and active-set sweeps must stop after the same number of sweeps with the same x."""
    G, b, pen = _gram_problem(8, 90, 14, 3)
    x_np, info_np = _solve_with("numpy", monkeypatch, G, b, 3, pen, max_iter=max_iter, tol=1e-15)
    x_nb, info_nb = _solve_with("numba", monkeypatch, G, b, 3, pen, max_iter=max_iter, tol=1e-15)
    assert info_np["n_iter"] == info_nb["n_iter"]
    assert info_np["converged"] == info_nb["converged"]
    assert np.array_equal(x_np != 0.0, x_nb != 0.0)
    assert _rel_diff(x_nb, x_np) < 1e-12
    assert info_nb["max_change"] == pytest.approx(info_np["max_change"], rel=1e-8, abs=1e-15)


@needs_numba
@pytest.mark.parametrize("gs", [1, 2, 3])
def test_backends_agree_on_ill_conditioned_problem(monkeypatch, gs):
    """Correlated columns (like covariance instruments) with the solver stopped at its sweep budget,
    the regime of the Gamma-step inside the ARLS loop."""
    G, b, pen = _ill_conditioned_problem(21 + gs, 20, gs)
    assert np.linalg.cond(G) > 1e4
    rng = np.random.default_rng(3)
    x0 = 0.1 * rng.standard_normal(G.shape[0])
    for max_iter in (200, 5000):
        x_np, info_np = _solve_with("numpy", monkeypatch, G, b, gs, pen, x0=x0, max_iter=max_iter, tol=1e-12)
        x_nb, info_nb = _solve_with("numba", monkeypatch, G, b, gs, pen, x0=x0, max_iter=max_iter, tol=1e-12)
        assert info_np["converged"] == info_nb["converged"]
        if info_np["converged"]:
            assert abs(info_np["n_iter"] - info_nb["n_iter"]) <= 1  # rounding-level stopping decision, see above
        else:
            assert info_np["n_iter"] == info_nb["n_iter"] == max_iter
        assert np.array_equal(x_np != 0.0, x_nb != 0.0)
        assert _rel_diff(x_nb, x_np) < 1e-11
        assert info_nb["objective"] == pytest.approx(info_np["objective"], rel=1e-12)


@needs_numba
def test_backends_agree_with_pinned_groups_and_scalar_penalty(monkeypatch):
    rng = np.random.default_rng(41)
    D = rng.standard_normal((30, 12))
    D[:, 3:6] = 0.0  # a zero-curvature group
    G = D.T @ D
    b = D.T @ rng.standard_normal(30)
    x0 = rng.standard_normal(12)
    x_np, info_np = _solve_with("numpy", monkeypatch, G, b, 3, 0.2, x0=x0, max_iter=5000, tol=1e-14)
    x_nb, info_nb = _solve_with("numba", monkeypatch, G, b, 3, 0.2, x0=x0, max_iter=5000, tol=1e-14)
    assert np.all(x_np[3:6] == 0.0) and np.all(x_nb[3:6] == 0.0)
    assert info_np["n_iter"] == info_nb["n_iter"] and info_np["converged"] and info_nb["converged"]
    assert _rel_diff(x_nb, x_np) < 1e-12


# ---------------------------------------------------------------------------
# backend selection: module flag and environment variable
# ---------------------------------------------------------------------------
def test_active_backend_follows_module_flag(monkeypatch):
    monkeypatch.setattr(gl, "USE_NUMBA", False)
    assert gl.active_backend() == "numpy"
    monkeypatch.setattr(gl, "USE_NUMBA", True)
    assert gl.active_backend() == ("numba" if gl.HAVE_NUMBA else "numpy")
    assert isinstance(gl.NUMBA_DISABLED_BY_ENV, bool)


def test_env_var_forces_numpy_path_in_fresh_interpreter():
    """``NARRATIVE_IPCA_NO_NUMBA=1`` skips the numba import: HAVE_NUMBA is False and the solver still works."""
    code = "\n".join(
        [
            "import numpy as np, sys",
            "from narrative_ipca import grouplasso as gl",
            "assert gl.NUMBA_DISABLED_BY_ENV and not gl.HAVE_NUMBA and gl.active_backend() == 'numpy', (gl.HAVE_NUMBA, gl.active_backend())",
            "assert 'numba' not in sys.modules",
            "G = np.diag([1.0, 2.0, 3.0, 4.0]); b = np.array([1.0, -2.0, 0.1, 3.0])",
            "x, info = gl.group_lasso_gram(G, b, 2, np.array([0.5, 0.5]), max_iter=100, tol=1e-15)",
            "assert info['converged'] and gl.kkt_violation(G, b, x, 2, np.array([0.5, 0.5])) < 1e-12",
            "print('ok')",
        ]
    )
    env = {**os.environ, "NARRATIVE_IPCA_NO_NUMBA": "1"}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
